from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from worker.datasets.storage import download_s3_uri
from worker.exporting.artifacts import load_export_manifest
from worker.exporting.metadata import build_demo_prediction_payload
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
