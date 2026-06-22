from __future__ import annotations

import json
import os
import posixpath
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

from worker.datasets.storage import download_s3_uri
from worker.exporting.artifacts import (
    ArtifactPathValidationError,
    load_export_manifest,
    resolve_controlled_artifact_reference,
    safe_torch_load_checkpoint,
    validate_controlled_artifact_path,
)
from worker.exporting.metadata import build_demo_detection_payload, build_demo_prediction_payload
from worker.exporting.preprocessing import (
    image_size_from_metadata,
    image_to_chw_float32_array,
    normalization_values,
    prepare_image_for_inference,
    preprocessing_parity_diagnostics,
)

_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: OrderedDict[tuple, object] = OrderedDict()


def run_demo_inference_from_manifest(
    *,
    manifest_path: Path | None = None,
    manifest: dict | None = None,
    image_path: Path,
    top_k: int = 5,
    true_label: str | None = None,
    image_metadata: dict | None = None,
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

    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    class_labels = [str(label) for label in metadata.get("class_labels", [])]
    if not class_labels:
        return _pending_payload("CLASS_LABELS_UNAVAILABLE", "Export metadata has no class labels.")
    label_error = _class_label_error(class_labels)
    if label_error:
        return _pending_payload("CLASS_LABEL_ORDER_INVALID", label_error)
    image_metadata_payload = image_metadata if isinstance(image_metadata, dict) else {}
    parity = preprocessing_parity_diagnostics(metadata, image_metadata_payload)
    if parity.get("status") != "ok":
        issue = _first_parity_issue(parity)
        return _pending_payload(
            str(issue.get("code") or "DEMO_PREPROCESSING_PARITY_UNSAFE"),
            str(issue.get("message") or "Demo inference is not parity-safe for this image."),
            image_metadata=_inference_image_metadata(image_metadata_payload, parity, "unavailable"),
        )
    if _is_detection_metadata(metadata):
        return _run_detection_inference_from_manifest(
            manifest_path=manifest_path,
            manifest=manifest,
            image_path=image_path,
            class_labels=class_labels,
            metadata=metadata,
            true_label=true_label,
            image_metadata=image_metadata_payload,
            parity=parity,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

    artifact = _find_created_artifact(manifest, "onnx")
    runtime = "onnx"
    if artifact is None:
        artifact = _find_created_artifact(manifest, "torchscript")
        runtime = "torchscript"
    if artifact is None:
        artifact = _find_created_artifact(manifest, "framework_native_checkpoint")
        runtime = "framework_native_checkpoint"
    if artifact is None:
        return _pending_payload(
            "MODEL_ARTIFACT_UNAVAILABLE",
            "Demo inference requires a created ONNX, TorchScript, or framework-native checkpoint artifact in the export manifest.",
        )

    model_path, artifact_error = _resolve_artifact_path_with_error(manifest_path, artifact)
    if model_path is None or not model_path.exists():
        return _pending_payload(
            "MODEL_ARTIFACT_NOT_FOUND",
            artifact_error or "The model artifact referenced by the manifest is not available.",
        )

    if runtime == "onnx":
        return _run_classification_onnx_inference(
            model_path=model_path,
            image_path=image_path,
            metadata=metadata,
            image_metadata=image_metadata_payload,
            parity=parity,
            class_labels=class_labels,
            true_label=true_label,
            top_k=top_k,
        )

    try:
        import torch
        from PIL import Image
    except Exception as exc:
        return _pending_payload("INFERENCE_DEPENDENCY_UNAVAILABLE", _dependency_error_message(exc))

    try:
        load_started = time.perf_counter()
        model, model_cache_hit = _cached_runtime(
            _runtime_cache_key(model_path, runtime, metadata),
            lambda: _load_classification_runtime(torch, model_path, metadata, class_labels, runtime),
        )
        model_load_ms = 0.0 if model_cache_hit else (time.perf_counter() - load_started) * 1000

        preprocess_started = time.perf_counter()
        tensor = _image_tensor(Image, torch, image_path, metadata, image_metadata_payload).unsqueeze(0)
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
            if int(probabilities.numel()) != len(class_labels):
                return _pending_payload(
                    "LABEL_MAP_OUTPUT_MISMATCH",
                    f"Model output class count {int(probabilities.numel())} does not match export class_labels count {len(class_labels)}.",
                    image_metadata=_inference_image_metadata(image_metadata_payload, parity, runtime),
                )
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
    payload["image_metadata"] = _inference_image_metadata(image_metadata_payload, parity, runtime)
    payload["latency_breakdown_ms"] = {
        "model_load": round(max(0.0, model_load_ms), 3),
        "model_cache_hit": model_cache_hit,
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


def _run_classification_onnx_inference(
    *,
    model_path: Path,
    image_path: Path,
    metadata: dict,
    image_metadata: dict,
    parity: dict,
    class_labels: list[str],
    true_label: str | None,
    top_k: int,
) -> dict:
    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except Exception as exc:
        return _pending_payload("INFERENCE_DEPENDENCY_UNAVAILABLE", _dependency_error_message(exc))

    try:
        load_started = time.perf_counter()
        session, model_cache_hit = _cached_runtime(
            _runtime_cache_key(model_path, "onnx", metadata),
            lambda: ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]),
        )
        model_load_ms = 0.0 if model_cache_hit else (time.perf_counter() - load_started) * 1000

        preprocess_started = time.perf_counter()
        array = _image_array(Image, image_path, metadata, image_metadata)
        tensor = np.expand_dims(array, axis=0).astype("float32", copy=False)
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

        inference_started = time.perf_counter()
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        output_values = session.run(output_names, {input_name: tensor})
        inference_ms = (time.perf_counter() - inference_started) * 1000

        postprocess_started = time.perf_counter()
        logits = np.asarray(output_values[0], dtype=np.float32)
        if logits.ndim > 1:
            logits = logits.reshape(logits.shape[0], -1)[0]
        else:
            logits = logits.reshape(-1)
        if int(logits.size) != len(class_labels):
            return _pending_payload(
                "LABEL_MAP_OUTPUT_MISMATCH",
                f"ONNX output class count {int(logits.size)} does not match export class_labels count {len(class_labels)}.",
                image_metadata=_inference_image_metadata(image_metadata, parity, "onnx"),
            )
        probabilities = _softmax_numpy(logits)
        count = min(max(1, int(top_k)), len(class_labels), int(probabilities.size))
        indices = np.argsort(probabilities)[::-1][:count]
        postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
    except Exception as exc:
        return _pending_payload("INFERENCE_FAILED", str(exc))

    predictions = [
        {"label": class_labels[int(index)], "confidence": float(probabilities[int(index)])}
        for index in indices.tolist()
    ]
    payload = build_demo_prediction_payload(
        image_id=str(image_path),
        predictions=predictions,
        latency_ms=preprocess_ms + inference_ms + postprocess_ms,
        true_label=true_label,
    )
    payload["status"] = "ok"
    payload["runtime"] = "onnx"
    payload["image_metadata"] = _inference_image_metadata(image_metadata, parity, "onnx")
    payload["latency_breakdown_ms"] = {
        "model_load": round(max(0.0, model_load_ms), 3),
        "model_cache_hit": model_cache_hit,
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
    image_metadata: dict,
    parity: dict,
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

    model_path, artifact_error = _resolve_artifact_path_with_error(manifest_path, artifact)
    if model_path is None or not model_path.exists():
        return _pending_payload(
            "MODEL_ARTIFACT_NOT_FOUND",
            artifact_error or "The ONNX detector artifact referenced by the manifest is not available.",
        )

    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except Exception as exc:
        return _pending_payload("INFERENCE_DEPENDENCY_UNAVAILABLE", _dependency_error_message(exc))

    thresholds = _detection_thresholds(
        metadata,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )

    try:
        load_started = time.perf_counter()
        session, model_cache_hit = _cached_runtime(
            _runtime_cache_key(model_path, "onnx", metadata),
            lambda: ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]),
        )
        model_load_ms = 0.0 if model_cache_hit else (time.perf_counter() - load_started) * 1000

        preprocess_started = time.perf_counter()
        with Image.open(image_path) as image:
            original_size = image.size
            prepared = prepare_image_for_inference(
                Image,
                image,
                metadata,
                image_metadata=image_metadata,
                strict_metadata=True,
            )
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
        "model_cache_hit": model_cache_hit,
        "preprocess": round(max(0.0, preprocess_ms), 3),
        "inference": round(max(0.0, inference_ms), 3),
        "postprocess": round(max(0.0, postprocess_ms), 3),
        "streaming_total": payload["latency_ms"],
        "single_request_total": round(
            max(0.0, model_load_ms + preprocess_ms + inference_ms + postprocess_ms),
            3,
        ),
    }
    result_image_metadata = payload.get("image_metadata") if isinstance(payload.get("image_metadata"), dict) else {}
    result_image_metadata.update(
        {
            **_inference_image_metadata(image_metadata, parity, "onnx"),
            "runtime": "onnx",
            "confidence_threshold": thresholds["confidence_threshold"],
            "iou_threshold": thresholds["iou_threshold"],
            "max_detections": thresholds["max_detections"],
        }
    )
    payload["image_metadata"] = result_image_metadata
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
    checkpoint = safe_torch_load_checkpoint(torch, model_path)
    state_dict = _checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    return model


def _load_classification_runtime(torch, model_path: Path, metadata: dict, class_labels: list[str], runtime: str):
    if runtime == "torchscript":
        model = torch.jit.load(str(model_path), map_location="cpu")
    else:
        model = _load_framework_native_model(torch, model_path, metadata, class_labels)
    model.eval()
    return model


def _cached_runtime(key: tuple, loader):
    cache_size = _demo_model_cache_size()
    if cache_size <= 0:
        return loader(), False
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            _MODEL_CACHE.move_to_end(key)
            return cached, True
    runtime = loader()
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[key] = runtime
        _MODEL_CACHE.move_to_end(key)
        while len(_MODEL_CACHE) > cache_size:
            _MODEL_CACHE.popitem(last=False)
    return runtime, False


def _runtime_cache_key(model_path: Path, runtime: str, metadata: dict) -> tuple:
    try:
        stat = model_path.stat()
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
    except OSError:
        size = 0
        mtime_ns = 0
    return (
        str(model_path.resolve()),
        runtime,
        size,
        mtime_ns,
        _metadata_fingerprint(metadata),
    )


def _metadata_fingerprint(metadata: dict) -> str:
    try:
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        payload = str(metadata)
    return str(hash(payload))


def _demo_model_cache_size() -> int:
    value = os.getenv("MODEL_EXPRESS_DEMO_INFERENCE_MODEL_CACHE_SIZE", "").strip()
    if not value:
        return 2
    try:
        parsed = int(value)
    except ValueError:
        return 2
    return max(0, min(parsed, 8))


def clear_demo_inference_cache() -> None:
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def demo_prediction_result_from_inference(payload: dict) -> dict:
    if payload.get("status") == "ok":
        result = {
            "status": "SUCCEEDED",
            "predicted_label": payload.get("predicted_label", ""),
            "confidence": payload.get("confidence", 0.0),
            "top_k": payload.get("top_k", []),
            "latency_ms": payload.get("latency_ms", 0.0),
            "correct": payload.get("correct"),
            "runtime": payload.get("runtime", "torchscript"),
        }
        if isinstance(payload.get("image_metadata"), dict):
            result["image_metadata"] = payload["image_metadata"]
        if isinstance(payload.get("detections"), list):
            result.setdefault("image_metadata", {})
            result["image_metadata"]["detections"] = payload["detections"]
            result["image_metadata"]["detection_count"] = len(payload["detections"])
        if "postprocess_latency_ms" in payload:
            result.setdefault("image_metadata", {})
            result["image_metadata"]["postprocess_latency_ms"] = payload["postprocess_latency_ms"]
        return result

    error_code = str(payload.get("error_code", "INFERENCE_UNAVAILABLE"))
    status = "RUNTIME_UNAVAILABLE"
    if error_code in {"INFERENCE_FAILED", "LABEL_MAP_OUTPUT_MISMATCH", "CLASS_LABEL_ORDER_INVALID"}:
        status = "FAILED"
    result = {
        "status": status,
        "error_code": error_code,
        "error": payload.get("error", ""),
        "predicted_label": "",
        "confidence": 0.0,
        "top_k": [],
        "latency_ms": 0.0,
    }
    if isinstance(payload.get("image_metadata"), dict):
        result["image_metadata"] = payload["image_metadata"]
    return result


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


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _checkpoint_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _validated_state_dict(value)
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return _validated_state_dict(checkpoint)
    raise ValueError("framework-native checkpoint does not contain a state_dict.")


def _validated_state_dict(state_dict: dict) -> dict:
    if not all(isinstance(key, str) and _looks_like_tensor(value) for key, value in state_dict.items()):
        raise ValueError("framework-native checkpoint state_dict contains non-tensor entries.")
    return _strip_module_prefix(state_dict)


def _looks_like_tensor(value: object) -> bool:
    return hasattr(value, "detach") and hasattr(value, "shape")


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


def _image_tensor(Image, torch, image_path: Path, metadata: dict, image_metadata: dict | None = None):
    array = _image_array(Image, image_path, metadata, image_metadata)
    return torch.from_numpy(array)


def _image_array(Image, image_path: Path, metadata: dict, image_metadata: dict | None = None):
    with Image.open(image_path) as image:
        prepared = prepare_image_for_inference(
            Image,
            image,
            metadata,
            image_metadata=image_metadata,
            strict_metadata=True,
        )
        return image_to_chw_float32_array(prepared, metadata)


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


def _softmax_numpy(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    total = float(np.sum(exps))
    if total <= 0:
        return np.zeros_like(values)
    return exps / total


def _class_label_error(class_labels: list[str]) -> str:
    labels = [str(label).strip() for label in class_labels]
    if any(not label for label in labels):
        return "Export class_labels contains an empty label."
    if len(set(labels)) != len(labels):
        return "Export class_labels contains duplicates, so model output indices are ambiguous."
    return ""


def _first_parity_issue(parity: dict) -> dict:
    issues = parity.get("issues") if isinstance(parity, dict) else []
    if isinstance(issues, list):
        for issue in issues:
            if isinstance(issue, dict):
                return issue
    return {}


def _inference_image_metadata(image_metadata: dict | None, parity: dict, runtime: str) -> dict:
    source = image_metadata if isinstance(image_metadata, dict) else {}
    return {
        **source,
        "runtime": runtime,
        "preprocessing_parity": parity,
        "parity_status": parity.get("status", "unknown") if isinstance(parity, dict) else "unknown",
        "preprocessing_contract_applied": parity.get("status") == "ok" if isinstance(parity, dict) else False,
    }


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
    path, _error = _resolve_artifact_path_with_error(manifest_path, path_value)
    return path


def _resolve_artifact_path_with_error(manifest_path: Path | None, path_value: object) -> tuple[Path | None, str]:
    artifact = path_value if isinstance(path_value, dict) else {}
    value = _artifact_reference(path_value)
    if not value:
        return None, "READY export manifest is missing a usable model artifact path."
    export_dir = manifest_path.parent if manifest_path is not None else None

    def validate_resolved(resolved: Path, artifact_uri: str) -> tuple[Path | None, str]:
        if not resolved.exists() or not resolved.is_file():
            return None, f"Model artifact not found: {resolved}"
        external_error = _materialize_onnx_external_data_error(resolved, artifact, artifact_uri)
        if external_error:
            return None, external_error
        return resolved, ""

    if value.startswith("s3://"):
        destination = _s3_artifact_cache_path(value)
        try:
            downloaded = download_s3_uri(value, destination)
            resolved = validate_controlled_artifact_path(downloaded, export_dir=destination.parent)
            return validate_resolved(resolved, _artifact_s3_uri(artifact, value))
        except Exception as exc:
            return None, f"S3 artifact unavailable or credentials missing: {exc}"
    try:
        if value.startswith("file://"):
            resolved = resolve_controlled_artifact_reference(value, export_dir=export_dir)
            return validate_resolved(resolved, _artifact_s3_uri(artifact, value))
    except ArtifactPathValidationError as exc:
        return None, f"Local artifact is outside allowed roots: {exc}"
    path = Path(value)
    if manifest_path is None:
        try:
            resolved = resolve_controlled_artifact_reference(value)
            return validate_resolved(resolved, _artifact_s3_uri(artifact, value))
        except ArtifactPathValidationError as exc:
            return None, f"Local artifact is outside allowed roots: {exc}"
    try:
        if path.is_absolute():
            resolved = validate_controlled_artifact_path(path, export_dir=export_dir)
        else:
            resolved = validate_controlled_artifact_path(manifest_path.parent / path, export_dir=export_dir)
        return validate_resolved(resolved, _artifact_s3_uri(artifact, value))
    except ArtifactPathValidationError as exc:
        return None, f"Local artifact is outside allowed roots: {exc}"


def _artifact_reference(value: object) -> str:
    if isinstance(value, dict):
        for key in ("path", "uri", "artifact_uri", "model_uri", "download_url"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
        return ""
    return str(value).strip()


def _materialize_onnx_external_data(model_path: Path, artifact: dict, artifact_uri: str) -> bool:
    return _materialize_onnx_external_data_error(model_path, artifact, artifact_uri) == ""


def _materialize_onnx_external_data_error(model_path: Path, artifact: dict, artifact_uri: str) -> str:
    if not _is_onnx_artifact(model_path, artifact):
        return ""
    for candidate in _external_data_candidates(model_path, artifact):
        relative_path = _external_data_relative_path(candidate.get("path"))
        explicit = bool(candidate.get("explicit"))
        if not relative_path:
            if explicit:
                return "ONNX external data file missing: manifest entry does not include a safe relative path."
            continue
        destination = _safe_external_data_path(model_path.parent, relative_path)
        if destination is None:
            if explicit:
                return f"ONNX external data file missing or outside allowed roots: {relative_path}"
            continue
        sidecar_uri = _external_data_s3_uri(artifact_uri, candidate, relative_path)
        if destination.exists() and destination.is_file() and _external_data_file_matches(candidate, destination):
            continue
        if sidecar_uri:
            try:
                downloaded = download_s3_uri(sidecar_uri, destination)
                validate_controlled_artifact_path(downloaded, export_dir=model_path.parent)
                continue
            except Exception as exc:
                if explicit:
                    return f"ONNX external data file missing: {relative_path} ({exc})"
                continue
        source = _external_data_local_source(candidate, model_path.parent)
        if source is not None:
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
                continue
            except OSError as exc:
                if explicit:
                    return f"ONNX external data file missing: {relative_path} ({exc})"
                continue
        if explicit:
            return f"ONNX external data file missing: {relative_path}"
    return ""


def _s3_artifact_cache_path(artifact_uri: str) -> Path:
    parsed = urlparse(artifact_uri)
    key = _storage_relative_path(parsed.path.lstrip("/"))
    parts = [_safe_cache_path_part(part) for part in key.split("/") if part]
    if not parts:
        parts = ["artifact"]
    return Path(".cache/artifacts") / _safe_cache_path_part(parsed.netloc) / Path(*parts)


def _safe_cache_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    return safe if safe and safe not in {".", ".."} else "artifact"


def _external_data_file_matches(candidate: dict, destination: Path) -> bool:
    expected = _external_data_expected_bytes(candidate)
    if expected > 0:
        try:
            return destination.stat().st_size == expected
        except OSError:
            return False
    return bool(candidate.get("explicit"))


def _external_data_expected_bytes(candidate: dict) -> int:
    for key in ("bytes", "size_bytes"):
        try:
            value = int(candidate.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0


def _is_onnx_artifact(model_path: Path, artifact: dict) -> bool:
    artifact_format = str(artifact.get("format") or artifact.get("artifact_format") or "").strip().lower()
    return artifact_format == "onnx" or model_path.name.lower().endswith(".onnx")


def _external_data_candidates(model_path: Path, artifact: dict) -> list[dict]:
    candidates: list[dict] = []
    external_data = artifact.get("external_data") if isinstance(artifact.get("external_data"), list) else []
    for item in external_data:
        if isinstance(item, dict):
            candidates.append(
                {
                    **item,
                    "path": item.get("path") or item.get("relative_path") or item.get("file_name"),
                    "explicit": True,
                }
            )
    for location in _onnx_external_data_locations(model_path):
        candidates.append({"path": location, "explicit": True})
    inferred_name = _artifact_external_data_file_name(artifact, model_path)
    if inferred_name:
        candidates.append({"path": inferred_name, "explicit": False})

    unique: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        relative_path = _external_data_relative_path(candidate.get("path"))
        if not relative_path or relative_path in seen:
            continue
        seen.add(relative_path)
        unique.append({**candidate, "path": relative_path})
    return unique


def _artifact_external_data_file_name(artifact: dict, model_path: Path) -> str:
    for key in ("uri", "artifact_uri", "path", "model_uri", "download_url"):
        value = str(artifact.get(key) or "").strip()
        if not value:
            continue
        if value.startswith("s3://") or value.startswith("file://"):
            name = Path(urlparse(value).path).name
        else:
            name = Path(value).name
        if name.lower().endswith(".onnx"):
            return f"{name}.data"
    return f"{model_path.name}.data" if model_path.name.lower().endswith(".onnx") else ""


def _onnx_external_data_locations(model_path: Path) -> list[str]:
    try:
        import onnx
    except Exception:
        return []
    try:
        model = onnx.load(str(model_path), load_external_data=False)
    except Exception:
        return []
    locations: list[str] = []
    for tensor in getattr(model.graph, "initializer", []):
        for entry in getattr(tensor, "external_data", []):
            if getattr(entry, "key", "") != "location":
                continue
            location = str(getattr(entry, "value", "")).strip()
            if location and location not in locations:
                locations.append(location)
    return locations


def _external_data_relative_path(value: object) -> str:
    text = str(value or "").strip()
    if not text or "://" in text or Path(text).is_absolute() or (len(text) > 1 and text[1] == ":"):
        return ""
    parts = []
    for part in text.replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            return ""
        parts.append(part)
    return "/".join(parts)


def _safe_external_data_path(base_dir: Path, relative_path: str) -> Path | None:
    relative_path = _external_data_relative_path(relative_path)
    if not relative_path:
        return None
    base = base_dir.resolve()
    resolved = (base / relative_path).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return resolved


def _artifact_s3_uri(artifact: dict, fallback: str) -> str:
    if str(fallback).startswith("s3://"):
        return str(fallback)
    for key in ("uri", "artifact_uri", "path", "model_uri", "download_url"):
        value = str(artifact.get(key) or "").strip()
        if value.startswith("s3://"):
            return value
    return ""


def _external_data_s3_uri(artifact_uri: str, candidate: dict, relative_path: str) -> str:
    if not artifact_uri.startswith("s3://"):
        return ""
    expected_uri = _s3_sidecar_uri(artifact_uri, relative_path)
    if not expected_uri:
        return ""
    for key in ("uri", "artifact_uri", "artifactPath", "artifact_path"):
        value = str(candidate.get(key) or "").strip()
        if value.startswith("s3://"):
            return value if _normalize_s3_object_uri(value) == expected_uri else ""
    return expected_uri


def _s3_sidecar_uri(artifact_uri: str, relative_path: str) -> str:
    parsed = urlparse(artifact_uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        return ""
    artifact_key = _storage_relative_path(parsed.path.lstrip("/"))
    sidecar_path = _storage_relative_path(relative_path)
    if not artifact_key or not sidecar_path:
        return ""
    base_dir = posixpath.dirname(artifact_key)
    sidecar_key = posixpath.join("" if base_dir == "." else base_dir, sidecar_path)
    return f"s3://{parsed.netloc}/{sidecar_key}"


def _normalize_s3_object_uri(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc:
        return ""
    key = _storage_relative_path(parsed.path.lstrip("/"))
    return f"s3://{parsed.netloc}/{key}" if key else ""


def _storage_relative_path(value: str) -> str:
    parts = []
    for part in str(value).replace("\\", "/").split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            return ""
        parts.append(part)
    return "/".join(parts)


def _external_data_local_source(candidate: dict, artifact_dir: Path) -> Path | None:
    for key in ("artifact_path", "local_path", "path"):
        value = str(candidate.get(key) or "").strip()
        if not value or value.startswith("s3://"):
            continue
        try:
            source = resolve_controlled_artifact_reference(value, export_dir=artifact_dir)
        except ArtifactPathValidationError:
            continue
        if source.exists() and source.is_file():
            return source
    return None


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return safe or "artifact"


def _dependency_error_message(exc: Exception) -> str:
    missing = getattr(exc, "name", "") if isinstance(exc, ModuleNotFoundError) else ""
    if missing:
        package = "Pillow" if missing == "PIL" else missing
        return f"Install worker dependencies in services/worker/.venv. Missing Python package: {package}."
    return f"Install worker dependencies in services/worker/.venv. {exc}"


def _pending_payload(error_code: str, error: str, image_metadata: dict | None = None) -> dict:
    payload = {
        "schema_version": "champion_demo_prediction_v1",
        "status": "pending",
        "error_code": str(error_code),
        "error": str(error),
        "top_k": [],
        "predicted_label": "",
        "confidence": 0.0,
        "latency_ms": 0.0,
    }
    if isinstance(image_metadata, dict):
        payload["image_metadata"] = image_metadata
    return payload
