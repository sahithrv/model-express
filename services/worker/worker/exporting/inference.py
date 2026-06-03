from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from worker.datasets.storage import download_s3_uri
from worker.exporting.artifacts import load_export_manifest
from worker.exporting.metadata import build_demo_detection_payload, build_demo_prediction_payload
from worker.exporting.preprocessing import (
    image_size_from_metadata,
    image_to_chw_float32_array,
    normalization_values,
    prepare_image_for_inference,
)


def run_demo_inference_from_manifest(
    *,
    manifest_path: Path | None = None,
    manifest: dict | None = None,
    image_path: Path,
    top_k: int = 5,
    true_label: str | None = None,
    confidence_threshold: float | None = None,
    iou_threshold: float | None = None,
    max_detections: int | None = None,
) -> dict:
    """Run single-image demo inference from a worker-owned model export."""
    if manifest is None:
        if manifest_path is None:
            return _pending_payload("MANIFEST_NOT_CONFIGURED", "No export manifest was supplied.")
        manifest = load_export_manifest(manifest_path)
    if manifest.get("status") == "error":
        return _pending_payload(manifest.get("error_code", "MANIFEST_ERROR"), manifest.get("error", ""))

    artifact = _find_created_artifact(manifest, "torchscript")
    runtime = "torchscript"
    if artifact is None:
        artifact = _find_created_artifact(manifest, "framework_native_checkpoint")
        runtime = "framework_native_checkpoint"
    if artifact is None:
        return _pending_payload(
            "MODEL_ARTIFACT_UNAVAILABLE",
            "Demo inference requires a created TorchScript or framework-native checkpoint artifact in the export manifest.",
        )

    model_path = _resolve_artifact_path(manifest_path, artifact.get("path") or artifact.get("uri"))
    if model_path is None or not model_path.exists():
        return _pending_payload(
            "MODEL_ARTIFACT_NOT_FOUND",
            "The model artifact referenced by the manifest is not available.",
        )

    try:
        import torch
        from PIL import Image
    except Exception as exc:
        return _pending_payload("INFERENCE_DEPENDENCY_UNAVAILABLE", str(exc))

    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    class_labels = [str(label) for label in metadata.get("class_labels", [])]
    if not class_labels:
        return _pending_payload("CLASS_LABELS_UNAVAILABLE", "Export metadata has no class labels.")
    if _is_detection_metadata(metadata):
        return _run_detection_inference_from_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
            image_path=image_path,
            class_labels=class_labels,
            metadata=metadata,
            true_label=true_label,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

    try:
        load_started = time.perf_counter()
        if runtime == "torchscript":
            model = torch.jit.load(str(model_path), map_location="cpu")
        else:
            model = _load_framework_native_model(torch, model_path, metadata, class_labels)
        model.eval()
        model_load_ms = (time.perf_counter() - load_started) * 1000

        preprocess_started = time.perf_counter()
        tensor = _image_tensor(Image, torch, image_path, metadata).unsqueeze(0)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        inference_started = time.perf_counter()
        with torch.no_grad():
            logits = model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
        inference_ms = (time.perf_counter() - inference_started) * 1000

        postprocess_started = time.perf_counter()
        with torch.no_grad():
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            count = min(max(1, int(top_k)), len(class_labels), int(probabilities.numel()))
            values, indices = torch.topk(probabilities, k=count)
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
    except Exception as exc:
        return _pending_payload("INFERENCE_FAILED", str(exc))

    predictions = [
        {"label": class_labels[int(index)], "confidence": float(value)}
        for value, index in zip(values.tolist(), indices.tolist())
    ]
    payload = build_demo_prediction_payload(
        image_id=str(image_path),
        predictions=predictions,
        latency_ms=preprocess_ms + inference_ms + postprocess_ms,
        true_label=true_label,
    )
    payload["status"] = "ok"
    payload["runtime"] = runtime
    payload["latency_breakdown_ms"] = {
        "model_load": round(max(0.0, model_load_ms), 3),
        "preprocess": round(max(0.0, preprocess_ms), 3),
        "inference": round(max(0.0, inference_ms), 3),
        "postprocess": round(max(0.0, postprocess_ms), 3),
        "streaming_total": payload["latency_ms"],
        "single_request_total": round(
            max(0.0, model_load_ms + preprocess_ms + inference_ms + postprocess_ms),
            3,
        ),
    }
    return payload


def _run_detection_inference_from_manifest(
    *,
    manifest_path: Path | None,
    manifest: dict,
    image_path: Path,
    class_labels: list[str],
    metadata: dict,
    true_label: str | None,
    confidence_threshold: float | None,
    iou_threshold: float | None,
    max_detections: int | None,
) -> dict:
    artifact = _find_created_artifact(manifest, "onnx")
    if artifact is None:
        return _pending_payload(
            "DETECTION_ONNX_ARTIFACT_UNAVAILABLE",
            "YOLO demo inference requires a created ONNX detector artifact.",
        )

    model_path = _resolve_artifact_path(manifest_path, artifact.get("path") or artifact.get("uri"))
    if model_path is None or not model_path.exists():
        return _pending_payload(
            "MODEL_ARTIFACT_NOT_FOUND",
            "The ONNX detector artifact referenced by the manifest is not available.",
        )

    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except Exception as exc:
        return _pending_payload("INFERENCE_DEPENDENCY_UNAVAILABLE", str(exc))

    thresholds = _detection_thresholds(
        metadata,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )

    try:
        load_started = time.perf_counter()
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        model_load_ms = (time.perf_counter() - load_started) * 1000

        preprocess_started = time.perf_counter()
        with Image.open(image_path) as image:
            original_size = image.size
            prepared = prepare_image_for_inference(Image, image, metadata)
            input_size = prepared.size
            array = image_to_chw_float32_array(prepared, metadata)
        tensor = np.expand_dims(array, axis=0).astype("float32", copy=False)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        inference_started = time.perf_counter()
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        output_values = session.run(output_names, {input_name: tensor})
        inference_ms = (time.perf_counter() - inference_started) * 1000

        postprocess_started = time.perf_counter()
        outputs = dict(zip(output_names, output_values))
        detections = _postprocess_detection_outputs(
            outputs=outputs,
            class_labels=class_labels,
            input_size=input_size,
            original_size=original_size,
            thresholds=thresholds,
        )
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
    except Exception as exc:
        return _pending_payload("INFERENCE_FAILED", str(exc))

    payload = build_demo_detection_payload(
        image_id=str(image_path),
        detections=detections,
        latency_ms=preprocess_ms + inference_ms + postprocess_ms,
        true_label=true_label,
        postprocess_latency_ms=postprocess_ms,
    )
    payload["status"] = "ok"
    payload["runtime"] = "onnx"
    payload["latency_breakdown_ms"] = {
        "model_load": round(max(0.0, model_load_ms), 3),
        "preprocess": round(max(0.0, preprocess_ms), 3),
        "inference": round(max(0.0, inference_ms), 3),
        "postprocess": round(max(0.0, postprocess_ms), 3),
        "streaming_total": payload["latency_ms"],
        "single_request_total": round(
            max(0.0, model_load_ms + preprocess_ms + inference_ms + postprocess_ms),
            3,
        ),
    }
    image_metadata = payload.get("image_metadata") if isinstance(payload.get("image_metadata"), dict) else {}
    image_metadata.update(
        {
            "runtime": "onnx",
            "confidence_threshold": thresholds["confidence_threshold"],
            "iou_threshold": thresholds["iou_threshold"],
            "max_detections": thresholds["max_detections"],
        }
    )
    payload["image_metadata"] = image_metadata
    return payload


def _load_framework_native_model(torch, model_path: Path, metadata: dict, class_labels: list[str]):
    model_name = str(metadata.get("model") or metadata.get("model_name") or "")
    if not model_name:
        raise ValueError("framework-native checkpoint metadata is missing model name.")
    training_config = metadata.get("training_config") if isinstance(metadata.get("training_config"), dict) else {}
    from worker.training.modal_app import _build_model

    model = _build_model(
        model_name,
        len(class_labels),
        pretrained=False,
        freeze_backbone=False,
        fine_tune_strategy=str(training_config.get("fine_tune_strategy") or "full"),
        dropout=_float_value(training_config.get("dropout"), default=0.0),
    )
    checkpoint = torch.load(str(model_path), map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    return model


def _postprocess_detection_outputs(
    *,
    outputs: dict,
    class_labels: list[str],
    input_size: tuple[int, int],
    original_size: tuple[int, int],
    thresholds: dict,
) -> list[dict]:
    named = _detections_from_named_outputs(outputs, class_labels, input_size, original_size, thresholds)
    if named:
        return named
    rows = _yolo_rows_from_outputs(outputs, len(class_labels))
    detections = []
    for row in rows:
        detection = _detection_from_yolo_row(row, class_labels, input_size, original_size, thresholds)
        if detection is not None:
            detections.append(detection)
    return _class_aware_nms(
        detections,
        iou_threshold=thresholds["iou_threshold"],
        max_detections=thresholds["max_detections"],
    )


def _detections_from_named_outputs(
    outputs: dict,
    class_labels: list[str],
    input_size: tuple[int, int],
    original_size: tuple[int, int],
    thresholds: dict,
) -> list[dict]:
    lowered = {str(key).lower(): value for key, value in outputs.items()}
    boxes = _first_present(lowered, "boxes", "output_boxes")
    scores = _first_present(lowered, "scores", "output_scores")
    classes = _first_present(lowered, "classes", "class_ids", "labels")
    if boxes is None or scores is None or classes is None:
        return []
    try:
        import numpy as np
    except Exception:
        return []
    box_array = np.asarray(boxes).reshape(-1, 4)
    score_array = np.asarray(scores).reshape(-1)
    class_array = np.asarray(classes).reshape(-1)
    detections = []
    count = min(len(box_array), len(score_array), len(class_array))
    for index in range(count):
        score = _score_value(score_array[index])
        if score < thresholds["confidence_threshold"]:
            continue
        class_id = _safe_class_id(class_array[index], len(class_labels))
        box = _box_from_xyxy(
            [float(value) for value in box_array[index].tolist()],
            input_size=input_size,
            original_size=original_size,
        )
        if box is None:
            continue
        detections.append(_detection_record(class_id, score, box, class_labels))
    return _class_aware_nms(
        detections,
        iou_threshold=thresholds["iou_threshold"],
        max_detections=thresholds["max_detections"],
    )


def _first_present(record: dict, *keys: str):
    for key in keys:
        if key in record:
            return record[key]
    return None


def _yolo_rows_from_outputs(outputs: dict, class_count: int):
    try:
        import numpy as np
    except Exception:
        return []
    for value in outputs.values():
        array = np.asarray(value)
        if array.size == 0:
            continue
        while array.ndim > 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 2:
            continue
        rows, features = array.shape
        expected_features = max(6, class_count + 4)
        if (rows <= expected_features + 8 and features > rows) or (features < expected_features and rows >= expected_features):
            array = array.T
        return array
    return []


def _detection_from_yolo_row(
    row,
    class_labels: list[str],
    input_size: tuple[int, int],
    original_size: tuple[int, int],
    thresholds: dict,
) -> dict | None:
    values = [float(item) for item in row]
    if len(values) < 6:
        return None
    class_count = len(class_labels)
    objectness = 1.0
    if class_count > 0 and len(values) >= 4 + class_count:
        if len(values) == 5 + class_count:
            objectness = _score_value(values[4])
            class_scores = values[5 : 5 + class_count]
        else:
            class_scores = values[4 : 4 + class_count]
        scored = [_score_value(item) * objectness for item in class_scores]
        class_id = max(range(len(scored)), key=lambda index: scored[index])
        score = scored[class_id]
        box = _box_from_center_xywh(values[:4], input_size=input_size, original_size=original_size)
    else:
        score = _score_value(values[4])
        class_id = _safe_class_id(values[5], class_count)
        box = _box_from_xyxy(values[:4], input_size=input_size, original_size=original_size)
    if score < thresholds["confidence_threshold"] or box is None:
        return None
    return _detection_record(class_id, score, box, class_labels)


def _box_from_center_xywh(
    xywh: list[float],
    *,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> dict | None:
    input_width, input_height = input_size
    x_center, y_center, width, height = xywh
    if max(abs(x_center), abs(y_center), abs(width), abs(height)) <= 2.0:
        x_center *= input_width
        width *= input_width
        y_center *= input_height
        height *= input_height
    return _box_from_model_xyxy(
        x_center - width / 2,
        y_center - height / 2,
        x_center + width / 2,
        y_center + height / 2,
        input_size=input_size,
        original_size=original_size,
    )


def _box_from_xyxy(
    xyxy: list[float],
    *,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> dict | None:
    input_width, input_height = input_size
    original_width, original_height = original_size
    x1, y1, x2, y2 = xyxy
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    if max_coord <= 2.0:
        return _normalized_box_from_original_xyxy(
            x1 * original_width,
            y1 * original_height,
            x2 * original_width,
            y2 * original_height,
            original_size,
        )
    if max_coord <= max(input_width, input_height) * 1.25:
        return _box_from_model_xyxy(
            x1,
            y1,
            x2,
            y2,
            input_size=input_size,
            original_size=original_size,
        )
    return _normalized_box_from_original_xyxy(x1, y1, x2, y2, original_size)


def _box_from_model_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    input_size: tuple[int, int],
    original_size: tuple[int, int],
) -> dict | None:
    input_width, input_height = input_size
    original_width, original_height = original_size
    if original_width <= 0 or original_height <= 0:
        return None
    scale = min(input_width / original_width, input_height / original_height)
    if scale <= 0:
        return None
    pad_x = (input_width - original_width * scale) / 2
    pad_y = (input_height - original_height * scale) / 2
    return _normalized_box_from_original_xyxy(
        (x1 - pad_x) / scale,
        (y1 - pad_y) / scale,
        (x2 - pad_x) / scale,
        (y2 - pad_y) / scale,
        original_size,
    )


def _normalized_box_from_original_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    original_size: tuple[int, int],
) -> dict | None:
    original_width, original_height = original_size
    if original_width <= 0 or original_height <= 0:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(original_width), x1))
    y1 = max(0.0, min(float(original_height), y1))
    x2 = max(0.0, min(float(original_width), x2))
    y2 = max(0.0, min(float(original_height), y2))
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    return {
        "x": round(x1 / original_width, 6),
        "y": round(y1 / original_height, 6),
        "width": round(width / original_width, 6),
        "height": round(height / original_height, 6),
    }


def _class_aware_nms(detections: list[dict], *, iou_threshold: float, max_detections: int) -> list[dict]:
    selected = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if len(selected) >= max_detections:
            break
        if any(
            detection["class_id"] == kept["class_id"]
            and _box_iou(detection["box"], kept["box"]) > iou_threshold
            for kept in selected
        ):
            continue
        selected.append(detection)
    return selected


def _box_iou(left: dict, right: dict) -> float:
    left_x2 = left["x"] + left["width"]
    left_y2 = left["y"] + left["height"]
    right_x2 = right["x"] + right["width"]
    right_y2 = right["y"] + right["height"]
    inter_x1 = max(left["x"], right["x"])
    inter_y1 = max(left["y"], right["y"])
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    return intersection / union if union > 0 else 0.0


def _detection_record(class_id: int, score: float, box: dict, class_labels: list[str]) -> dict:
    label = class_labels[class_id] if 0 <= class_id < len(class_labels) else f"class_{class_id}"
    confidence = round(max(0.0, min(1.0, float(score))), 6)
    return {
        "label": label,
        "class_name": label,
        "class_id": class_id,
        "confidence": confidence,
        "score": confidence,
        "box": box,
        "x": box["x"],
        "y": box["y"],
        "width": box["width"],
        "height": box["height"],
    }


def _score_value(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if score < 0.0 or score > 1.0:
        try:
            import math

            return 1 / (1 + math.exp(-score))
        except OverflowError:
            return 1.0 if score > 0 else 0.0
    return score


def _safe_class_id(value: object, class_count: int) -> int:
    try:
        class_id = int(round(float(value)))
    except (TypeError, ValueError):
        class_id = 0
    if class_count <= 0:
        return class_id
    return max(0, min(class_count - 1, class_id))


def _detection_thresholds(
    metadata: dict,
    *,
    confidence_threshold: float | None,
    iou_threshold: float | None,
    max_detections: int | None,
) -> dict:
    defaults = metadata.get("confidence_threshold_defaults") if isinstance(metadata.get("confidence_threshold_defaults"), dict) else {}
    detection = defaults.get("detection") if isinstance(defaults.get("detection"), dict) else {}
    postprocessing = metadata.get("postprocessing_contract") if isinstance(metadata.get("postprocessing_contract"), dict) else {}
    nms = postprocessing.get("nms") if isinstance(postprocessing.get("nms"), dict) else {}
    confidence = _bounded_float(
        confidence_threshold,
        detection.get("confidence_threshold", nms.get("confidence_threshold", postprocessing.get("confidence_threshold", 0.25))),
        0.0,
        1.0,
    )
    iou = _bounded_float(
        iou_threshold,
        detection.get("iou_threshold", nms.get("iou_threshold", postprocessing.get("iou_threshold", 0.7))),
        0.0,
        1.0,
    )
    max_count = _positive_int(
        max_detections if max_detections is not None else detection.get("max_detections", nms.get("max_detections")),
        300,
    )
    return {
        "confidence_threshold": confidence,
        "iou_threshold": iou,
        "max_detections": min(max_count, 1000),
    }


def _is_detection_metadata(metadata: dict) -> bool:
    return str(metadata.get("model_kind", "")).lower() == "detection" or str(metadata.get("task_type", "")).lower() == "object_detection"


def _bounded_float(value: object, default: object, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        try:
            parsed = float(default)
        except (TypeError, ValueError):
            parsed = minimum
    return max(minimum, min(maximum, parsed))


def _checkpoint_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _strip_module_prefix(value)
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return _strip_module_prefix(checkpoint)
    raise ValueError("framework-native checkpoint does not contain a state_dict.")


def _strip_module_prefix(state_dict: dict) -> dict:
    if not any(str(key).startswith("module.") for key in state_dict):
        return state_dict
    return {
        (str(key)[7:] if str(key).startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _image_tensor(Image, torch, image_path: Path, metadata: dict):
    with Image.open(image_path) as image:
        prepared = prepare_image_for_inference(Image, image, metadata)
        array = image_to_chw_float32_array(prepared, metadata)
    return torch.from_numpy(array)


def _image_size(metadata: dict) -> int:
    return image_size_from_metadata(metadata)


def _normalization(metadata: dict) -> tuple[list[float], list[float]] | None:
    return normalization_values(metadata.get("preprocessing"))


def _three_floats(value: object, positive: bool = False) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        parsed = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if positive and any(item <= 0 for item in parsed):
        return None
    return parsed


def _find_created_artifact(manifest: dict, format_name: str) -> dict | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if (
            isinstance(artifact, dict)
            and artifact.get("format") == format_name
            and artifact.get("status") == "created"
        ):
            return artifact
    return None


def _resolve_artifact_path(manifest_path: Path | None, path_value: object) -> Path | None:
    if not path_value:
        return None
    value = str(path_value)
    if value.startswith("s3://"):
        parsed = urlparse(value)
        filename = Path(parsed.path).name or "model.torchscript.pt"
        destination = Path(".cache/artifacts") / _safe_path_part(parsed.netloc) / _safe_path_part(filename)
        try:
            return download_s3_uri(value, destination)
        except Exception:
            return None
    if value.startswith("file://"):
        parsed = urlparse(value)
        path_value = unquote(parsed.path)
        if len(path_value) > 2 and path_value[0] == "/" and path_value[2] == ":":
            path_value = path_value[1:]
        return Path(path_value)
    path = Path(value)
    if path.is_absolute():
        return path
    if manifest_path is None:
        return path
    return manifest_path.parent / path


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return safe or "artifact"


def _pending_payload(error_code: str, error: str) -> dict:
    return {
        "schema_version": "champion_demo_prediction_v1",
        "status": "pending",
        "error_code": str(error_code),
        "error": str(error),
        "top_k": [],
        "predicted_label": "",
        "confidence": 0.0,
        "latency_ms": 0.0,
    }
