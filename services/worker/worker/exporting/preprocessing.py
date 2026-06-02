from __future__ import annotations

import statistics
import time
from typing import Iterable


def normalized_preprocessing_config(preprocessing: object) -> dict:
    payload = dict(preprocessing) if isinstance(preprocessing, dict) else {}
    return {
        **payload,
        "resize_strategy": _normalized_name(payload.get("resize_strategy"), "squash"),
        "crop_strategy": _normalized_name(payload.get("crop_strategy"), "none"),
        "normalization": _normalized_name(payload.get("normalization"), "imagenet"),
        "bbox_mode": _normalized_name(payload.get("bbox_mode"), "ignore"),
    }


def build_inference_contract(
    *,
    image_size: int,
    preprocessing: object,
) -> dict:
    config = normalized_preprocessing_config(preprocessing)
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
        "input": {
            "accepted_source": "RGB image/frame",
            "model_tensor_shape": [1, 3, int(image_size), int(image_size)],
            "model_tensor_layout": "NCHW",
            "model_tensor_dtype": "float32",
        },
        "preprocessing": {
            "config": config,
            "steps": steps,
            "train_time_augmentation_excluded": True,
        },
        "postprocessing": {
            "model_output": "logits",
            "activation": "softmax",
            "label_source": "metadata.class_labels",
        },
        "deployment_notes": [
            "The runtime must apply these deterministic preprocessing steps before model inference.",
            "Training-only augmentation is intentionally excluded from live inference.",
        ],
    }


def prepare_image_for_inference(Image, image, metadata: dict):
    image_size = image_size_from_metadata(metadata)
    config = normalized_preprocessing_config(metadata.get("preprocessing"))
    rgb = image.convert("RGB")
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


def image_size_from_metadata(metadata: dict) -> int:
    input_shape = metadata.get("input_shape")
    if isinstance(input_shape, list) and len(input_shape) == 4:
        try:
            return max(1, int(input_shape[-1]))
        except (TypeError, ValueError):
            pass
    return 224


def image_to_chw_float32_array(image, metadata: dict):
    import numpy as np

    tensor = np.asarray(image, dtype=np.float32) / 255.0
    values = normalization_values(metadata.get("preprocessing"))
    if values is not None:
        mean, std = values
        mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
        tensor = (tensor - mean_array) / std_array
    return np.transpose(tensor, (2, 0, 1)).copy()


def normalization_values(preprocessing: object) -> tuple[list[float], list[float]] | None:
    config = normalized_preprocessing_config(preprocessing)
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
    text = str(value).strip().lower().replace("-", "_") if value is not None else ""
    return text or default


def _bbox_crop_requested(config: dict) -> bool:
    return (
        config.get("crop_strategy") in {"bbox_crop_if_available", "bbox_crop_ablation"}
        or config.get("resize_strategy") == "bbox_crop_if_available"
        or config.get("bbox_mode") in {"crop_if_available", "crop_and_compare_full_image"}
    )


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
