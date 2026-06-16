from __future__ import annotations

import hashlib
import time
from typing import Iterable

from worker.exporting.preprocessing import build_inference_contract

DEFAULT_RUNTIME = "onnx"


def build_champion_export_metadata(
    *,
    model_name: str,
    class_names: Iterable[str],
    image_size: int,
    preprocessing: dict | None = None,
    model_profile: dict | None = None,
    training_config: dict | None = None,
    artifact_uri: str | None = None,
    runtime: str | None = None,
    model_kind: str | None = None,
    task_type: str | None = None,
) -> dict:
    """Build additive export metadata; the backend remains responsible for export records."""
    profile = model_profile if isinstance(model_profile, dict) else {}
    config = training_config if isinstance(training_config, dict) else {}
    preprocessing_config = preprocessing if isinstance(preprocessing, dict) else {}
    labels = _class_labels(class_names, profile, config)
    resolved_model_kind = _resolve_model_kind(
        explicit=model_kind or task_type,
        model_name=model_name,
        profile=profile,
        config=config,
        preprocessing=preprocessing_config,
    )
    resolved_task_type = _resolve_task_type(
        explicit=task_type,
        model_kind=resolved_model_kind,
        profile=profile,
        config=config,
    )
    resolved_runtime = _resolve_runtime(runtime, profile, config)
    input_shape = _input_shape(image_size, profile, config)
    confidence_threshold_defaults = _confidence_threshold_defaults(
        resolved_model_kind,
        profile=profile,
        config=config,
    )
    latency_budget = _latency_budget(profile, config)
    inference_contract = build_inference_contract(
        image_size=int(image_size),
        preprocessing=preprocessing_config,
        model_kind=resolved_model_kind,
        task_type=resolved_task_type,
        runtime=resolved_runtime,
        input_shape=input_shape,
        class_labels=labels,
        confidence_threshold_defaults=confidence_threshold_defaults,
    )
    return {
        "schema_version": "champion_export_metadata_v1",
        "model": str(model_name),
        "model_kind": resolved_model_kind,
        "task_type": resolved_task_type,
        "runtime": resolved_runtime,
        "default_runtime": resolved_runtime,
        "supported_runtimes": _supported_runtimes(resolved_runtime, profile, config),
        "artifact_uri": artifact_uri or "",
        "format": "framework_native_checkpoint",
        "input_shape": input_shape,
        "class_labels": labels,
        "class_index_order": [{"index": index, "label": label} for index, label in enumerate(labels)],
        "class_label_order_hash": _class_label_order_hash(labels),
        "class_count": len(labels),
        "confidence_threshold_defaults": confidence_threshold_defaults,
        "latency_budget": latency_budget,
        "export_self_test": _export_self_test_placeholder(),
        "export_status": _export_status_placeholder(resolved_runtime),
        "preprocessing": preprocessing_config,
        "preprocessing_contract": inference_contract["preprocessing"],
        "postprocessing_contract": inference_contract["postprocessing"],
        "inference_contract": inference_contract,
        "training_config": config,
        "model_profile": profile,
        "limitations": [
            "Export metadata is worker-generated and must be accepted by backend validation before use.",
            "Live demo inference should use held-out or test images when available.",
            "Production runtimes must apply the inference_contract preprocessing exactly before model inference.",
            "Production runtimes must apply the postprocessing_contract and confidence_threshold_defaults before displaying results.",
        ],
    }


def build_demo_prediction_payload(
    *,
    image_id: str,
    predictions: Iterable[dict],
    latency_ms: float,
    true_label: str | None = None,
) -> dict:
    ranked_predictions = _rank_predictions(predictions)
    top_prediction = ranked_predictions[0] if ranked_predictions else None
    predicted_label = str(top_prediction.get("label")) if top_prediction else ""
    payload = {
        "schema_version": "champion_demo_prediction_v1",
        "image_id": str(image_id),
        "predicted_label": predicted_label,
        "confidence": float(top_prediction.get("confidence", 0.0)) if top_prediction else 0.0,
        "top_k": ranked_predictions,
        "latency_ms": round(max(0.0, float(latency_ms)), 3),
        "created_at_unix": int(time.time()),
    }
    if true_label is not None:
        payload["true_label"] = str(true_label)
        payload["correct"] = predicted_label == str(true_label)
    return payload


def build_demo_detection_payload(
    *,
    image_id: str,
    detections: Iterable[dict],
    latency_ms: float,
    true_label: str | None = None,
    postprocess_latency_ms: float | None = None,
) -> dict:
    ranked_detections = _rank_detections(detections)
    top_detection = ranked_detections[0] if ranked_detections else None
    predicted_label = str(top_detection.get("label")) if top_detection else ""
    top_k = [
        {
            "label": detection["label"],
            "confidence": detection["confidence"],
        }
        for detection in ranked_detections[:5]
    ]
    payload = {
        "schema_version": "champion_demo_prediction_v1",
        "image_id": str(image_id),
        "predicted_label": predicted_label,
        "confidence": float(top_detection.get("confidence", 0.0)) if top_detection else 0.0,
        "top_k": top_k,
        "detections": ranked_detections,
        "detection_count": len(ranked_detections),
        "latency_ms": round(max(0.0, float(latency_ms)), 3),
        "created_at_unix": int(time.time()),
        "image_metadata": {
            "task_type": "object_detection",
            "detections": ranked_detections,
            "detection_count": len(ranked_detections),
        },
    }
    if postprocess_latency_ms is not None:
        payload["postprocess_latency_ms"] = round(max(0.0, float(postprocess_latency_ms)), 3)
        payload["image_metadata"]["postprocess_latency_ms"] = payload["postprocess_latency_ms"]
    if true_label is not None:
        label = str(true_label)
        payload["true_label"] = label
        payload["correct"] = bool(label and any(detection["label"] == label for detection in ranked_detections))
    return payload


def _rank_predictions(predictions: Iterable[dict]) -> list[dict]:
    normalized: list[dict] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            continue
        label = prediction.get("label")
        if label is None:
            continue
        try:
            confidence = float(prediction.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        normalized.append(
            {
                "label": str(label),
                "confidence": round(max(0.0, min(1.0, confidence)), 6),
            }
        )
    return sorted(normalized, key=lambda item: item["confidence"], reverse=True)[:5]


def _rank_detections(detections: Iterable[dict]) -> list[dict]:
    normalized: list[dict] = []
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        label = detection.get("label") or detection.get("class_name")
        if label is None:
            continue
        box = _normalized_box(detection)
        if box is None:
            continue
        try:
            confidence = float(detection.get("confidence", detection.get("score", 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            class_id = int(detection.get("class_id", detection.get("class_index", -1)))
        except (TypeError, ValueError):
            class_id = -1
        item = {
            "label": str(label),
            "class_name": str(label),
            "class_id": class_id,
            "confidence": round(max(0.0, min(1.0, confidence)), 6),
            "score": round(max(0.0, min(1.0, confidence)), 6),
            "box": box,
            "x": box["x"],
            "y": box["y"],
            "width": box["width"],
            "height": box["height"],
        }
        normalized.append(item)
    return sorted(normalized, key=lambda item: item["confidence"], reverse=True)[:300]


def _normalized_box(detection: dict) -> dict | None:
    box_value = detection.get("box")
    if isinstance(box_value, dict):
        candidates = {
            "x": box_value.get("x", box_value.get("left")),
            "y": box_value.get("y", box_value.get("top")),
            "width": box_value.get("width", box_value.get("w")),
            "height": box_value.get("height", box_value.get("h")),
        }
        if all(value is not None for value in candidates.values()):
            return _bounded_box(candidates)
        xyxy = [
            box_value.get("x1", box_value.get("xmin")),
            box_value.get("y1", box_value.get("ymin")),
            box_value.get("x2", box_value.get("xmax")),
            box_value.get("y2", box_value.get("ymax")),
        ]
        if all(value is not None for value in xyxy):
            return _bounded_xyxy_box(xyxy)
    if isinstance(box_value, (list, tuple)) and len(box_value) >= 4:
        return _bounded_xyxy_box(box_value[:4])
    candidates = {
        "x": detection.get("x", detection.get("left")),
        "y": detection.get("y", detection.get("top")),
        "width": detection.get("width", detection.get("w")),
        "height": detection.get("height", detection.get("h")),
    }
    if all(value is not None for value in candidates.values()):
        return _bounded_box(candidates)
    xyxy = [
        detection.get("x1", detection.get("xmin")),
        detection.get("y1", detection.get("ymin")),
        detection.get("x2", detection.get("xmax")),
        detection.get("y2", detection.get("ymax")),
    ]
    if all(value is not None for value in xyxy):
        return _bounded_xyxy_box(xyxy)
    return None


def _bounded_box(value: dict) -> dict | None:
    try:
        x = float(value["x"])
        y = float(value["y"])
        width = float(value["width"])
        height = float(value["height"])
    except (TypeError, ValueError):
        return None
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    width = max(0.0, min(1.0 - x, width))
    height = max(0.0, min(1.0 - y, height))
    if width <= 0 or height <= 0:
        return None
    return {
        "x": round(x, 6),
        "y": round(y, 6),
        "width": round(width, 6),
        "height": round(height, 6),
    }


def _bounded_xyxy_box(value: Iterable[object]) -> dict | None:
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return _bounded_box({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1})


def _class_labels(class_names: Iterable[str], *records: dict) -> list[str]:
    labels = [str(class_name) for class_name in class_names if str(class_name).strip()]
    if labels:
        return labels
    for record in _candidate_records(*records):
        for key in ("class_labels", "class_names", "classes", "names"):
            value = record.get(key)
            if isinstance(value, list):
                labels = [str(item) for item in value if str(item).strip()]
                if labels:
                    return labels
            if isinstance(value, dict):
                labels = [str(value[item]) for item in _sorted_label_keys(value) if str(value[item]).strip()]
                if labels:
                    return labels
    return []


def _class_label_order_hash(labels: list[str]) -> str:
    payload = "\n".join(labels).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_model_kind(
    *,
    explicit: str | None,
    model_name: str,
    profile: dict,
    config: dict,
    preprocessing: dict,
) -> str:
    if _text_indicates_detection(explicit):
        return "detection"
    if _text_indicates_classification(explicit):
        return "classification"
    if _text_indicates_detection(model_name):
        return "detection"
    for record in _candidate_records(profile, config, preprocessing):
        for key in ("model_kind", "task_type", "model_type", "architecture", "arch", "task"):
            value = record.get(key)
            if _text_indicates_detection(value):
                return "detection"
            if _text_indicates_classification(value):
                return "classification"
        if bool(record.get("yolo_available")) or bool(record.get("object_detection_available")):
            return "detection"
        yolo_summary = record.get("yolo_summary")
        if isinstance(yolo_summary, dict) and (
            bool(yolo_summary.get("available")) or str(yolo_summary.get("format", "")).lower() == "yolo"
        ):
            return "detection"
    return "classification"


def _resolve_task_type(*, explicit: str | None, model_kind: str, profile: dict, config: dict) -> str:
    if _text_indicates_detection(explicit):
        return "object_detection"
    if _text_indicates_classification(explicit):
        return "image_classification"
    for record in _candidate_records(profile, config):
        value = record.get("task_type")
        if _text_indicates_detection(value):
            return "object_detection"
        if _text_indicates_classification(value):
            return "image_classification"
    return "object_detection" if model_kind == "detection" else "image_classification"


def _resolve_runtime(runtime: str | None, profile: dict, config: dict) -> str:
    for value in (runtime, *_values_for_keys(_candidate_records(profile, config), ("runtime", "default_runtime"))):
        normalized = _normalized_name(value)
        if not normalized:
            continue
        if normalized in {"pt", "pth", "pytorch", "checkpoint", "framework_native"}:
            return "framework_native_checkpoint"
        if normalized in {"onnxruntime", "ort"}:
            return "onnx"
        return normalized
    return DEFAULT_RUNTIME


def _supported_runtimes(runtime: str, profile: dict, config: dict) -> list[str]:
    runtimes = [runtime]
    for record in _candidate_records(profile, config):
        value = record.get("supported_runtimes") or record.get("runtimes")
        if isinstance(value, list):
            runtimes.extend(_normalized_name(item) for item in value)
    out = []
    for item in runtimes:
        normalized = _resolve_runtime(item, {}, {})
        if normalized and normalized not in out:
            out.append(normalized)
    return out or [DEFAULT_RUNTIME]


def _input_shape(image_size: int, profile: dict, config: dict) -> list[int]:
    for record in _candidate_records(profile, config):
        for key in ("input_shape", "sample_input_shape", "model_tensor_shape"):
            shape = _four_int_list(record.get(key))
            if shape is not None:
                return shape
    size = _positive_int(_first_value(_candidate_records(profile, config), ("image_size", "imgsz")), int(image_size))
    return [1, 3, size, size]


def _confidence_threshold_defaults(model_kind: str, *, profile: dict, config: dict) -> dict:
    records = _candidate_records(profile, config)
    if model_kind == "detection":
        confidence = _bounded_float(
            _first_value(
                records,
                (
                    "confidence_threshold",
                    "conf_threshold",
                    "score_threshold",
                    "default_confidence_threshold",
                    "nms_confidence_threshold",
                ),
            ),
            default=0.25,
            minimum=0.0,
            maximum=1.0,
        )
        iou = _bounded_float(
            _first_value(records, ("iou_threshold", "nms_iou_threshold", "nms_threshold")),
            default=0.7,
            minimum=0.0,
            maximum=1.0,
        )
        max_detections = _positive_int(
            _first_value(records, ("max_detections", "max_det", "nms_max_detections")),
            300,
        )
        return {
            "schema_version": "confidence_threshold_defaults_v1",
            "default_confidence_threshold": confidence,
            "detection": {
                "confidence_threshold": confidence,
                "score_threshold": confidence,
                "iou_threshold": iou,
                "nms_iou_threshold": iou,
                "max_detections": max_detections,
            },
        }
    threshold = _bounded_float(
        _first_value(
            records,
            ("confidence_threshold", "default_confidence_threshold", "top_prediction_min_confidence"),
        ),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )
    return {
        "schema_version": "confidence_threshold_defaults_v1",
        "default_confidence_threshold": threshold,
        "classification": {
            "top_prediction_min_confidence": threshold,
            "top_k": 5,
        },
    }


def _latency_budget(profile: dict, config: dict) -> dict:
    records = _candidate_records(profile, config)
    payload = {
        "schema_version": "latency_budget_v1",
        "status": "not_provided",
    }
    fields = {
        "target_latency_ms": ("target_latency_ms", "latency_budget_ms", "max_latency_ms", "latency_ms_budget"),
        "target_p95_latency_ms": (
            "target_p95_latency_ms",
            "latency_budget_p95_ms",
            "max_p95_latency_ms",
            "p95_latency_budget_ms",
        ),
        "estimated_latency_ms": ("estimated_pipeline_latency_ms", "estimated_latency_ms"),
        "estimated_p50_latency_ms": ("latency_p50_ms", "p50_latency_ms", "estimated_p50_latency_ms"),
        "estimated_p95_latency_ms": ("latency_p95_ms", "p95_latency_ms", "estimated_p95_latency_ms"),
        "estimated_throughput_images_per_second": (
            "estimated_throughput_images_per_second",
            "throughput_images_per_second",
            "fps",
        ),
    }
    for output_key, source_keys in fields.items():
        value = _maybe_float(_first_value(records, source_keys))
        if value is not None:
            payload[output_key] = round(value, 3)
    if len(payload) > 2:
        payload["status"] = "available"
    return payload


def _export_self_test_placeholder() -> dict:
    return {
        "schema_version": "champion_export_self_test_v1",
        "status": "not_run",
        "export_verified": False,
        "parity_status": "not_run",
        "latency_status": "not_run",
        "failed_inference_count": 0,
    }


def _export_status_placeholder(runtime: str) -> dict:
    return {
        "schema_version": "champion_export_status_v1",
        "status": "metadata_created",
        "runtime": runtime,
        "self_test_status": "not_run",
        "export_verified": False,
    }


def _candidate_records(*records: dict) -> list[dict]:
    out: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        out.append(record)
        for key in (
            "model_profile",
            "deployment_profile",
            "training_config",
            "dataset_profile",
            "metadata_summary",
            "profile",
            "yolo_summary",
        ):
            nested = record.get(key)
            if isinstance(nested, dict):
                out.append(nested)
                yolo_summary = nested.get("yolo_summary")
                if isinstance(yolo_summary, dict):
                    out.append(yolo_summary)
    return out


def _values_for_keys(records: list[dict], keys: tuple[str, ...]) -> list[object]:
    return [record[key] for record in records for key in keys if key in record]


def _sorted_label_keys(value: dict) -> list:
    def sort_key(item):
        try:
            return (0, int(item))
        except (TypeError, ValueError):
            return (1, str(item))

    return sorted(value, key=sort_key)


def _first_value(records: list[dict], keys: tuple[str, ...]) -> object:
    for record in records:
        for key in keys:
            value = record.get(key)
            if value not in (None, ""):
                return value
    return None


def _normalized_name(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_") if value is not None else ""


def _text_indicates_detection(value: object) -> bool:
    text = _normalized_name(value)
    return bool(
        text
        and (
            text in {"detection", "detector", "object_detection", "ultralytics_yolo_detector", "yolo"}
            or "object_detection" in text
            or "yolo" in text
            or text.endswith("_detector")
        )
    )


def _text_indicates_classification(value: object) -> bool:
    text = _normalized_name(value)
    return text in {"classification", "classifier", "image_classification"}


def _four_int_list(value: object) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    out = []
    for item in value:
        parsed = _positive_int(item, 0)
        if parsed <= 0:
            return None
        out.append(parsed)
    return out


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _maybe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    parsed = _maybe_float(value)
    if parsed is None:
        parsed = default
    return round(max(minimum, min(maximum, parsed)), 6)
