from __future__ import annotations

import statistics
import time
from typing import Iterable


def normalized_preprocessing_config(preprocessing: object, *, model_kind: str = "classification") -> dict:
    payload = dict(preprocessing) if isinstance(preprocessing, dict) else {}
    kind = _normalized_model_kind(model_kind)
    default_resize = "preserve_aspect_pad" if kind == "detection" else "squash"
    default_normalization = "none" if kind == "detection" else "imagenet"
    return {
        **payload,
        "resize_strategy": _normalized_resize_strategy(payload.get("resize_strategy"), default_resize),
        "crop_strategy": _normalized_name(payload.get("crop_strategy"), "none"),
        "normalization": _normalized_name(payload.get("normalization"), default_normalization),
        "bbox_mode": _normalized_name(payload.get("bbox_mode"), "ignore"),
    }


def build_inference_contract(
    *,
    image_size: int,
    preprocessing: object,
    model_kind: str = "classification",
    task_type: str | None = None,
    runtime: str = "onnx",
    input_shape: Iterable[int] | None = None,
    class_labels: Iterable[str] | None = None,
    confidence_threshold_defaults: dict | None = None,
) -> dict:
    resolved_model_kind = _normalized_model_kind(model_kind)
    resolved_task_type = _normalized_task_type(task_type, resolved_model_kind)
    tensor_shape = _model_tensor_shape(input_shape, image_size)
    image_size = int(tensor_shape[-1])
    labels = [str(label) for label in class_labels or []]
    thresholds = _confidence_threshold_defaults(confidence_threshold_defaults, resolved_model_kind)
    config = normalized_preprocessing_config(preprocessing, model_kind=resolved_model_kind)
    resize_strategy = config["resize_strategy"]
    crop_strategy = config["crop_strategy"]
    bbox_mode = config["bbox_mode"]
    normalization = config["normalization"]

    steps = [
        {"operation": "decode_image", "output_color_space": "RGB"},
    ]
    if _bbox_crop_requested(config):
        steps.append(
            {
                "operation": "bbox_crop",
                "mode": bbox_mode,
                "crop_strategy": crop_strategy,
                "requires_runtime_metadata": True,
                "required_metadata": ["bounding_boxes"],
                "accepted_coordinate_formats": ["xyxy_pixels", "xyxy_normalized", "xywh_pixels", "xywh_normalized"],
                "padding_fraction": 0.05,
            }
        )
    if resize_strategy == "preserve_aspect_pad":
        steps.append(
            {
                "operation": "resize_preserve_aspect_pad",
                "size": [int(image_size), int(image_size)],
                "resample": "bilinear",
                "pad_color_rgb": [0, 0, 0],
            }
        )
    elif crop_strategy == "center_crop" or resize_strategy == "center_crop":
        steps.append(
            {
                "operation": "resize_square_then_center_crop",
                "resize_size": [int(image_size * 1.15), int(image_size * 1.15)],
                "crop_size": [int(image_size), int(image_size)],
                "resample": "bilinear",
            }
        )
    else:
        steps.append(
            {
                "operation": "resize_exact",
                "size": [int(image_size), int(image_size)],
                "resample": "bilinear",
            }
        )

    steps.append({"operation": "to_float32", "scale": "divide_by_255", "layout": "NCHW"})
    if normalization == "none":
        steps.append({"operation": "normalize", "mode": "none"})
    elif normalization == "dataset":
        metadata = config.get("normalization_metadata") if isinstance(config.get("normalization_metadata"), dict) else {}
        steps.append(
            {
                "operation": "normalize",
                "mode": "dataset",
                "mean": _three_float_list(metadata.get("mean")) or [],
                "std": _three_float_list(metadata.get("std"), positive=True) or [],
                "fallback": "imagenet",
            }
        )
    else:
        steps.append(
            {
                "operation": "normalize",
                "mode": "imagenet",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            }
        )

    return {
        "schema_version": "model_express_inference_contract_v1",
        "model_kind": resolved_model_kind,
        "task_type": resolved_task_type,
        "runtime": _normalized_name(runtime, "onnx"),
        "default_runtime": _normalized_name(runtime, "onnx"),
        "input": {
            "accepted_source": "RGB image/frame",
            "model_tensor_shape": tensor_shape,
            "model_tensor_layout": "NCHW",
            "model_tensor_dtype": "float32",
        },
        "preprocessing": {
            "schema_version": "preprocessing_contract_v1",
            "config": config,
            "steps": steps,
            "train_time_augmentation_excluded": True,
        },
        "postprocessing": _postprocessing_contract(
            model_kind=resolved_model_kind,
            task_type=resolved_task_type,
            class_count=len(labels),
            confidence_threshold_defaults=thresholds,
        ),
        "confidence_threshold_defaults": thresholds,
        "deployment_notes": [
            "The runtime must apply these deterministic preprocessing steps before model inference.",
            "Training-only augmentation is intentionally excluded from live inference.",
        ],
    }


def prepare_image_for_inference(
    Image,
    image,
    metadata: dict,
    image_metadata: dict | None = None,
    *,
    strict_metadata: bool = False,
):
    image_size = image_size_from_metadata(metadata)
    config = _preprocessing_config_from_metadata(metadata)
    rgb = image.convert("RGB")
    if _bbox_crop_requested(config):
        bbox = _bbox_from_runtime_metadata(image_metadata, rgb.size)
        if bbox is None:
            if strict_metadata:
                raise ValueError(
                    "BBOX_METADATA_REQUIRED: preprocessing requires bounding_boxes metadata for parity-safe inference."
                )
        else:
            rgb = crop_image_to_bbox(rgb, bbox)
    resize_strategy = config["resize_strategy"]
    crop_strategy = config["crop_strategy"]
    if resize_strategy == "preserve_aspect_pad":
        return resize_with_padding(Image, rgb, image_size)
    if crop_strategy == "center_crop" or resize_strategy == "center_crop":
        resize_size = int(image_size * 1.15)
        resized = rgb.resize((resize_size, resize_size), Image.Resampling.BILINEAR)
        left = (resize_size - image_size) // 2
        top = (resize_size - image_size) // 2
        return resized.crop((left, top, left + image_size, top + image_size))
    return rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)


def resize_with_padding(Image, image, image_size: int):
    rgb = image.convert("RGB").copy()
    rgb.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    left = (image_size - rgb.width) // 2
    top = (image_size - rgb.height) // 2
    canvas.paste(rgb, (left, top))
    return canvas


def crop_image_to_bbox(image, bbox: tuple[int, int, int, int]):
    rgb = image.convert("RGB")
    width, height = rgb.size
    xmin, ymin, xmax, ymax = bbox
    pad_x = max(1, int((xmax - xmin) * 0.05))
    pad_y = max(1, int((ymax - ymin) * 0.05))
    crop_box = (
        max(0, xmin - pad_x),
        max(0, ymin - pad_y),
        min(width, xmax + pad_x),
        min(height, ymax + pad_y),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return rgb
    return rgb.crop(crop_box)


def preprocessing_parity_diagnostics(metadata: dict, image_metadata: dict | None = None) -> dict:
    config = _preprocessing_config_from_metadata(metadata)
    diagnostics = {
        "schema_version": "demo_preprocessing_parity_v1",
        "status": "ok",
        "issues": [],
        "preprocessing": {
            "resize_strategy": config["resize_strategy"],
            "crop_strategy": config["crop_strategy"],
            "normalization": config["normalization"],
            "bbox_mode": config["bbox_mode"],
        },
    }
    source_type = ""
    parity_safe = None
    if isinstance(image_metadata, dict):
        source_type = str(image_metadata.get("demo_source_type") or image_metadata.get("source") or "").lower()
        if isinstance(image_metadata.get("parity_safe"), bool):
            parity_safe = image_metadata["parity_safe"]
    if parity_safe is False or "thumbnail" in source_type:
        diagnostics["status"] = "unsafe"
        diagnostics["issues"].append(
            {
                "code": "DEMO_IMAGE_NOT_PARITY_SAFE",
                "message": "Demo image metadata marks the source as a thumbnail or non-parity-safe derivative.",
            }
        )
    elif source_type == "heldout_test" and parity_safe is not True:
        diagnostics["status"] = "unsafe"
        diagnostics["issues"].append(
            {
                "code": "HELDOUT_IMAGE_SOURCE_UNVERIFIED",
                "message": "Held-out demo image metadata does not prove that original image bytes are being used.",
            }
        )
    if _bbox_crop_requested(config) and not _has_runtime_bbox_metadata(image_metadata):
        diagnostics["status"] = "unsafe"
        diagnostics["issues"].append(
            {
                "code": "BBOX_METADATA_REQUIRED",
                "message": "Preprocessing requires bounding_boxes metadata before demo inference can match validation/test.",
            }
        )
    return diagnostics


def image_size_from_metadata(metadata: dict) -> int:
    shapes = [metadata.get("input_shape")]
    contract = metadata.get("inference_contract") if isinstance(metadata.get("inference_contract"), dict) else {}
    contract_input = contract.get("input") if isinstance(contract.get("input"), dict) else {}
    shapes.append(contract_input.get("model_tensor_shape"))
    for input_shape in shapes:
        if isinstance(input_shape, list) and len(input_shape) == 4:
            try:
                return max(1, int(input_shape[-1]))
            except (TypeError, ValueError):
                pass
    return 224


def image_to_chw_float32_array(image, metadata: dict):
    import numpy as np

    tensor = np.asarray(image, dtype=np.float32) / 255.0
    values = normalization_values(metadata)
    if values is not None:
        mean, std = values
        mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        tensor = (tensor - mean_array) / std_array
    return np.transpose(tensor, (2, 0, 1)).copy()


def normalization_values(preprocessing: object) -> tuple[list[float], list[float]] | None:
    config = _preprocessing_config_from_value(preprocessing)
    normalization = config["normalization"]
    if normalization == "none":
        return None
    if normalization == "dataset":
        metadata = config.get("normalization_metadata")
        if isinstance(metadata, dict):
            mean = _three_float_list(metadata.get("mean"))
            std = _three_float_list(metadata.get("std"), positive=True)
            if mean is not None and std is not None:
                return mean, std
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def benchmark_preprocessing_latency(
    metadata: dict,
    *,
    source_shapes: Iterable[tuple[int, int]] | None = None,
    iterations: int = 12,
    warmup: int = 3,
) -> dict:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        return {
            "schema_version": "preprocessing_latency_profile_v1",
            "status": "unavailable",
            "error": str(exc),
        }

    shapes = list(source_shapes or [(150, 150), (640, 480), (1280, 720)])
    profiles = []
    output_size = image_size_from_metadata(metadata)
    for width, height in shapes:
        image = Image.fromarray(np.zeros((int(height), int(width), 3), dtype=np.uint8), "RGB")
        for _ in range(max(0, warmup)):
            prepared = prepare_image_for_inference(Image, image, metadata)
            _ = image_to_chw_float32_array(prepared, metadata)
        samples = []
        for _ in range(max(1, iterations)):
            started = time.perf_counter()
            prepared = prepare_image_for_inference(Image, image, metadata)
            _ = image_to_chw_float32_array(prepared, metadata)
            samples.append((time.perf_counter() - started) * 1000.0)
        profiles.append(
            {
                "source_shape": [int(height), int(width), 3],
                "output_shape": [3, int(output_size), int(output_size)],
                "avg_ms": round(statistics.mean(samples), 3),
                "p50_ms": round(statistics.median(samples), 3),
                "p95_ms": round(_percentile(samples, 0.95), 3),
            }
        )
    return {
        "schema_version": "preprocessing_latency_profile_v1",
        "status": "measured",
        "measured_by": "worker_export_pil_numpy_preprocessing_benchmark_v1",
        "measured_at_unix": int(time.time()),
        "iterations": max(1, iterations),
        "warmup": max(0, warmup),
        "source_profiles": profiles,
    }


def preprocessing_latency_p95_ms(profile: dict) -> float:
    profiles = profile.get("source_profiles") if isinstance(profile, dict) else []
    values = []
    for item in profiles if isinstance(profiles, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            values.append(float(item.get("p95_ms") or 0))
        except (TypeError, ValueError):
            continue
    return max(values) if values else 0.0


def _normalized_name(value: object, default: str) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_") if value is not None else ""
    return text or default


def _normalized_resize_strategy(value: object, default: str) -> str:
    text = _normalized_name(value, default)
    if text in {"letterbox", "yolo_letterbox"}:
        return "preserve_aspect_pad"
    return text


def _normalized_model_kind(value: object) -> str:
    text = _normalized_name(value, "classification").replace(" ", "_")
    if text in {"detection", "detector", "object_detection", "ultralytics_yolo_detector"} or "yolo" in text:
        return "detection"
    return "classification"


def _normalized_task_type(value: object, model_kind: str) -> str:
    text = _normalized_name(value, "")
    if text in {"object_detection", "detection", "detector"} or "yolo" in text:
        return "object_detection"
    if text in {"image_classification", "classification", "classifier"}:
        return "image_classification"
    return "object_detection" if model_kind == "detection" else "image_classification"


def _model_tensor_shape(input_shape: Iterable[int] | None, image_size: int) -> list[int]:
    if input_shape is not None:
        parsed = []
        try:
            for item in input_shape:
                parsed.append(int(item))
        except (TypeError, ValueError):
            parsed = []
        if len(parsed) == 4 and all(item > 0 for item in parsed):
            return parsed
    return [1, 3, int(image_size), int(image_size)]


def _confidence_threshold_defaults(defaults: dict | None, model_kind: str) -> dict:
    payload = defaults if isinstance(defaults, dict) else {}
    if model_kind == "detection":
        detection = payload.get("detection") if isinstance(payload.get("detection"), dict) else {}
        confidence = _bounded_float(
            detection.get("confidence_threshold", payload.get("default_confidence_threshold")),
            default=0.25,
            minimum=0.0,
            maximum=1.0,
        )
        iou = _bounded_float(
            detection.get("iou_threshold", detection.get("nms_iou_threshold")),
            default=0.7,
            minimum=0.0,
            maximum=1.0,
        )
        max_detections = _positive_int(detection.get("max_detections"), 300)
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
    classification = payload.get("classification") if isinstance(payload.get("classification"), dict) else {}
    confidence = _bounded_float(
        classification.get("top_prediction_min_confidence", payload.get("default_confidence_threshold")),
        default=0.0,
        minimum=0.0,
        maximum=1.0,
    )
    return {
        "schema_version": "confidence_threshold_defaults_v1",
        "default_confidence_threshold": confidence,
        "classification": {
            "top_prediction_min_confidence": confidence,
            "top_k": _positive_int(classification.get("top_k"), 5),
        },
    }


def _postprocessing_contract(
    *,
    model_kind: str,
    task_type: str,
    class_count: int,
    confidence_threshold_defaults: dict,
) -> dict:
    if model_kind == "detection":
        detection = confidence_threshold_defaults["detection"]
        confidence = detection["confidence_threshold"]
        iou = detection["iou_threshold"]
        max_detections = detection["max_detections"]
        return {
            "schema_version": "postprocessing_contract_v1",
            "task_type": task_type,
            "model_output": "detections",
            "decoder": {
                "type": "yolo",
                "raw_output": "implementation_defined_yolo_head",
                "decoded_output": "boxes_scores_classes",
            },
            "outputs": {
                "boxes": {
                    "dtype": "float32",
                    "shape": ["num_detections", 4],
                    "coordinate_format": "xyxy",
                    "coordinate_space": "original_image_pixels",
                },
                "scores": {
                    "dtype": "float32",
                    "shape": ["num_detections"],
                    "range": [0.0, 1.0],
                },
                "classes": {
                    "dtype": "int64",
                    "shape": ["num_detections"],
                    "label_source": "metadata.class_labels",
                },
            },
            "output_tensors": [
                {"name": "boxes", "shape": ["num_detections", 4], "dtype": "float32"},
                {"name": "scores", "shape": ["num_detections"], "dtype": "float32"},
                {"name": "classes", "shape": ["num_detections"], "dtype": "int64"},
            ],
            "nms": {
                "enabled": True,
                "method": "class_aware_nms",
                "confidence_threshold": confidence,
                "iou_threshold": iou,
                "max_detections": max_detections,
            },
            "confidence_threshold": confidence,
            "iou_threshold": iou,
            "label_source": "metadata.class_labels",
            "class_count": int(class_count),
        }
    classification = confidence_threshold_defaults["classification"]
    return {
        "schema_version": "postprocessing_contract_v1",
        "task_type": task_type,
        "model_output": "logits",
        "output_tensors": [
            {"name": "logits", "shape": ["batch", int(class_count)], "dtype": "float32"},
        ],
        "activation": "softmax",
        "confidence_output": "probability",
        "confidence_threshold": classification["top_prediction_min_confidence"],
        "top_k": classification["top_k"],
        "label_source": "metadata.class_labels",
        "class_count": int(class_count),
    }


def _preprocessing_config_from_value(value: object) -> dict:
    if _looks_like_export_metadata(value):
        return _preprocessing_config_from_metadata(value)
    return normalized_preprocessing_config(value)


def _preprocessing_config_from_metadata(metadata: object) -> dict:
    if not isinstance(metadata, dict):
        return normalized_preprocessing_config({})
    model_kind = _normalized_model_kind(metadata.get("model_kind") or metadata.get("task_type"))
    contract = metadata.get("preprocessing_contract")
    if isinstance(contract, dict):
        config = contract.get("config") if isinstance(contract.get("config"), dict) else contract
        return normalized_preprocessing_config(config, model_kind=model_kind)
    inference_contract = metadata.get("inference_contract")
    if isinstance(inference_contract, dict):
        contract = inference_contract.get("preprocessing")
        if isinstance(contract, dict):
            config = contract.get("config") if isinstance(contract.get("config"), dict) else contract
            return normalized_preprocessing_config(config, model_kind=model_kind)
    return normalized_preprocessing_config(metadata.get("preprocessing"), model_kind=model_kind)


def _looks_like_export_metadata(value: object) -> bool:
    return isinstance(value, dict) and any(
        key in value
        for key in ("preprocessing_contract", "inference_contract", "input_shape", "class_labels", "preprocessing")
    )


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _bounded_float(value: object, *, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return round(max(minimum, min(maximum, parsed)), 6)


def _bbox_crop_requested(config: dict) -> bool:
    return (
        config.get("crop_strategy") in {"bbox_crop_if_available", "bbox_crop_ablation"}
        or config.get("resize_strategy") == "bbox_crop_if_available"
        or config.get("bbox_mode") in {"crop_if_available", "crop_and_compare_full_image"}
    )


def _has_runtime_bbox_metadata(image_metadata: dict | None) -> bool:
    if not isinstance(image_metadata, dict):
        return False
    for key in ("bounding_boxes", "bbox", "box", "annotation"):
        if key in image_metadata:
            return True
    metadata = image_metadata.get("metadata")
    return isinstance(metadata, dict) and _has_runtime_bbox_metadata(metadata)


def _bbox_from_runtime_metadata(
    image_metadata: dict | None,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(image_metadata, dict):
        return None
    for key in ("bounding_boxes", "bbox", "box"):
        value = image_metadata.get(key)
        bbox = _bbox_from_value(value, image_size)
        if bbox is not None:
            return bbox
    annotation = image_metadata.get("annotation")
    if isinstance(annotation, dict):
        bbox = _bbox_from_runtime_metadata(annotation, image_size)
        if bbox is not None:
            return bbox
    nested = image_metadata.get("metadata")
    if isinstance(nested, dict):
        return _bbox_from_runtime_metadata(nested, image_size)
    return None


def _bbox_from_value(value: object, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            return _bbox_from_value(value[0], image_size)
        if len(value) >= 4:
            return _bbox_from_xyxy(value[:4], image_size)
    if isinstance(value, tuple) and len(value) >= 4:
        return _bbox_from_xyxy(value[:4], image_size)
    if not isinstance(value, dict):
        return None
    candidates = [
        (value.get("xmin", value.get("x1")), value.get("ymin", value.get("y1")), value.get("xmax", value.get("x2")), value.get("ymax", value.get("y2"))),
        (
            value.get("left", value.get("x")),
            value.get("top", value.get("y")),
            None,
            None,
        ),
    ]
    for xmin, ymin, xmax, ymax in candidates:
        if xmin is None or ymin is None:
            continue
        if xmax is None or ymax is None:
            width = value.get("width", value.get("w"))
            height = value.get("height", value.get("h"))
            if width is None or height is None:
                continue
            try:
                xmax = float(xmin) + float(width)
                ymax = float(ymin) + float(height)
            except (TypeError, ValueError):
                continue
        bbox = _bbox_from_xyxy((xmin, ymin, xmax, ymax), image_size)
        if bbox is not None:
            return bbox
    return None


def _bbox_from_xyxy(values: object, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    try:
        xmin, ymin, xmax, ymax = [float(item) for item in values]
    except (TypeError, ValueError):
        return None
    width, height = image_size
    if max(abs(xmin), abs(ymin), abs(xmax), abs(ymax)) <= 2.0:
        xmin *= width
        xmax *= width
        ymin *= height
        ymax *= height
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    xmin = max(0, min(int(round(xmin)), width))
    xmax = max(0, min(int(round(xmax)), width))
    ymin = max(0, min(int(round(ymin)), height))
    ymax = max(0, min(int(round(ymax)), height))
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _three_float_list(value: object, positive: bool = False) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    out = []
    for item in value:
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            return None
        if positive and parsed <= 0:
            return None
        out.append(parsed)
    return out


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]
