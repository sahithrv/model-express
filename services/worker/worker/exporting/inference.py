from __future__ import annotations

import time
from pathlib import Path

from worker.exporting.artifacts import load_export_manifest
from worker.exporting.metadata import build_demo_prediction_payload


def run_demo_inference_from_manifest(
    *,
    manifest_path: Path,
    image_path: Path,
    top_k: int = 5,
    true_label: str | None = None,
) -> dict:
    """Run single-image demo inference from a worker-owned TorchScript export."""
    manifest = load_export_manifest(manifest_path)
    if manifest.get("status") == "error":
        return _pending_payload(manifest.get("error_code", "MANIFEST_ERROR"), manifest.get("error", ""))

    artifact = _find_created_artifact(manifest, "torchscript")
    if artifact is None:
        return _pending_payload(
            "TORCHSCRIPT_ARTIFACT_UNAVAILABLE",
            "Demo inference requires a created TorchScript artifact in the export manifest.",
        )

    model_path = _resolve_artifact_path(manifest_path, artifact.get("path"))
    if model_path is None or not model_path.exists():
        return _pending_payload(
            "TORCHSCRIPT_ARTIFACT_NOT_FOUND",
            "The TorchScript artifact referenced by the manifest is not available.",
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

    try:
        started = time.perf_counter()
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        tensor = _image_tensor(Image, torch, image_path, metadata).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            count = min(max(1, int(top_k)), len(class_labels), int(probabilities.numel()))
            values, indices = torch.topk(probabilities, k=count)
        latency_ms = (time.perf_counter() - started) * 1000
    except Exception as exc:
        return _pending_payload("INFERENCE_FAILED", str(exc))

    predictions = [
        {"label": class_labels[int(index)], "confidence": float(value)}
        for value, index in zip(values.tolist(), indices.tolist())
    ]
    payload = build_demo_prediction_payload(
        image_id=str(image_path),
        predictions=predictions,
        latency_ms=latency_ms,
        true_label=true_label,
    )
    payload["status"] = "ok"
    payload["runtime"] = "torchscript"
    return payload


def _image_tensor(Image, torch, image_path: Path, metadata: dict):
    image_size = _image_size(metadata)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((image_size, image_size))
        pixels = list(rgb.getdata())

    tensor = torch.tensor(pixels, dtype=torch.float32).view(image_size, image_size, 3)
    tensor = tensor.permute(2, 0, 1) / 255.0
    normalization = _normalization(metadata)
    if normalization is not None:
        mean, std = normalization
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor
    return tensor


def _image_size(metadata: dict) -> int:
    input_shape = metadata.get("input_shape")
    if isinstance(input_shape, list) and len(input_shape) == 4:
        try:
            return max(1, int(input_shape[-1]))
        except (TypeError, ValueError):
            pass
    return 224


def _normalization(metadata: dict) -> tuple[list[float], list[float]] | None:
    preprocessing = metadata.get("preprocessing") if isinstance(metadata.get("preprocessing"), dict) else {}
    normalization = str(preprocessing.get("normalization", "imagenet")).lower()
    if normalization == "none":
        return None
    if normalization == "dataset":
        normalization_metadata = preprocessing.get("normalization_metadata")
        if isinstance(normalization_metadata, dict):
            mean = _three_floats(normalization_metadata.get("mean"))
            std = _three_floats(normalization_metadata.get("std"), positive=True)
            if mean is not None and std is not None:
                return mean, std
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


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


def _resolve_artifact_path(manifest_path: Path, path_value: object) -> Path | None:
    if not path_value:
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return manifest_path.parent / path


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
