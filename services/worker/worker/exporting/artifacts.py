from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

from worker.exporting.metadata import build_champion_export_metadata


def produce_champion_export_artifacts(
    *,
    export_dir: Path,
    model_name: str,
    class_names: Iterable[str],
    image_size: int,
    model=None,
    preprocessing: dict | None = None,
    model_profile: dict | None = None,
    training_config: dict | None = None,
    formats: Iterable[str] = ("framework_native",),
    sample_input_shape: Iterable[int] | None = None,
) -> dict:
    """Create worker-owned export artifacts without touching backend records."""
    export_dir.mkdir(parents=True, exist_ok=True)
    requested_formats = [str(item).lower() for item in formats]
    input_shape = _input_shape(sample_input_shape, image_size)
    metadata = build_champion_export_metadata(
        model_name=model_name,
        class_names=class_names,
        image_size=image_size,
        preprocessing=preprocessing,
        model_profile=model_profile,
        training_config=training_config,
    )
    metadata["input_shape"] = input_shape

    artifacts: list[dict] = []
    if "framework_native" in requested_formats or "checkpoint" in requested_formats:
        artifacts.append(_write_framework_native_checkpoint(export_dir, model, metadata))
    if "torchscript" in requested_formats:
        artifacts.append(_write_torchscript(export_dir, model, input_shape))
    if "onnx" in requested_formats:
        artifacts.append(_write_onnx(export_dir, model, input_shape))

    manifest = {
        "schema_version": "champion_export_manifest_v1",
        "metadata": metadata,
        "artifacts": artifacts,
        "status": _overall_status(artifacts),
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_export_manifest(manifest_path: Path) -> dict:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "schema_version": "champion_export_manifest_v1",
            "status": "error",
            "error_code": "MANIFEST_NOT_FOUND",
            "error": f"Export manifest not found: {manifest_path}",
        }
    except json.JSONDecodeError as exc:
        return {
            "schema_version": "champion_export_manifest_v1",
            "status": "error",
            "error_code": "MANIFEST_INVALID_JSON",
            "error": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": "champion_export_manifest_v1",
            "status": "error",
            "error_code": "MANIFEST_INVALID_SHAPE",
            "error": "Export manifest must be a JSON object.",
        }
    return payload


def produce_existing_champion_export_manifest(
    *,
    export_dir: Path,
    source_artifact_path: Path,
    artifact_format: str,
    model_name: str,
    class_names: Iterable[str],
    image_size: int,
    preprocessing: dict | None = None,
    model_profile: dict | None = None,
    training_config: dict | None = None,
    sample_input_shape: Iterable[int] | None = None,
) -> dict:
    """Copy an existing worker-visible artifact into a controlled export directory."""
    export_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_artifact_path)
    if not source_path.exists() or not source_path.is_file():
        metadata = build_champion_export_metadata(
            model_name=model_name,
            class_names=class_names,
            image_size=image_size,
            preprocessing=preprocessing,
            model_profile=model_profile,
            training_config=training_config,
        )
        manifest = {
            "schema_version": "champion_export_manifest_v1",
            "metadata": metadata,
            "artifacts": [
                _skipped_artifact(
                    _manifest_artifact_format(artifact_format),
                    "ARTIFACT_NOT_FOUND",
                    f"Source artifact not found: {source_path}",
                )
            ],
            "status": "pending_dependencies",
        }
        manifest_path = export_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        return manifest

    input_shape = _input_shape(sample_input_shape, image_size)
    metadata = build_champion_export_metadata(
        model_name=model_name,
        class_names=class_names,
        image_size=image_size,
        preprocessing=preprocessing,
        model_profile=model_profile,
        training_config=training_config,
    )
    metadata["input_shape"] = input_shape

    artifact_name = _artifact_filename(artifact_format, source_path)
    destination = export_dir / artifact_name
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)

    artifact = _created_artifact(_manifest_artifact_format(artifact_format), destination)
    manifest = {
        "schema_version": "champion_export_manifest_v1",
        "metadata": metadata,
        "artifacts": [artifact],
        "status": "created",
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _write_framework_native_checkpoint(export_dir: Path, model, metadata: dict) -> dict:
    if model is None:
        return _skipped_artifact(
            "framework_native_checkpoint",
            "MODEL_UNAVAILABLE",
            "A model instance is required to create a framework-native checkpoint.",
        )
    if not hasattr(model, "state_dict"):
        return _skipped_artifact(
            "framework_native_checkpoint",
            "STATE_DICT_UNAVAILABLE",
            "The supplied model does not expose state_dict().",
        )
    try:
        import torch
    except Exception as exc:
        return _skipped_artifact("framework_native_checkpoint", "TORCH_UNAVAILABLE", str(exc))

    path = export_dir / "model.pt"
    try:
        torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    except Exception as exc:
        return _failed_artifact("framework_native_checkpoint", path, exc)
    return _created_artifact("framework_native_checkpoint", path)


def _write_torchscript(export_dir: Path, model, input_shape: list[int]) -> dict:
    if model is None:
        return _skipped_artifact(
            "torchscript",
            "MODEL_UNAVAILABLE",
            "A model instance is required to create a TorchScript artifact.",
        )
    try:
        import torch
    except Exception as exc:
        return _skipped_artifact("torchscript", "TORCH_UNAVAILABLE", str(exc))

    path = export_dir / "model.torchscript.pt"
    try:
        model.eval()
        dummy_input = torch.zeros(tuple(input_shape), dtype=torch.float32)
        traced = torch.jit.trace(model.cpu(), dummy_input)
        traced.save(str(path))
    except Exception as exc:
        return _failed_artifact("torchscript", path, exc)
    return _created_artifact("torchscript", path)


def _write_onnx(export_dir: Path, model, input_shape: list[int]) -> dict:
    if model is None:
        return _skipped_artifact(
            "onnx",
            "MODEL_UNAVAILABLE",
            "A model instance is required to create an ONNX artifact.",
        )
    try:
        import torch
    except Exception as exc:
        return _skipped_artifact("onnx", "TORCH_UNAVAILABLE", str(exc))

    path = export_dir / "model.onnx"
    try:
        model.eval()
        dummy_input = torch.zeros(tuple(input_shape), dtype=torch.float32)
        torch.onnx.export(
            model.cpu(),
            dummy_input,
            str(path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
    except Exception as exc:
        if "onnx" in str(exc).lower() and "install" in str(exc).lower():
            return _skipped_artifact("onnx", "ONNX_UNAVAILABLE", str(exc))
        return _failed_artifact("onnx", path, exc)
    return _created_artifact("onnx", path)


def _created_artifact(format_name: str, path: Path) -> dict:
    return {
        "format": format_name,
        "status": "created",
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() else 0,
    }


def _skipped_artifact(format_name: str, error_code: str, error: str) -> dict:
    return {"format": format_name, "status": "skipped", "error_code": error_code, "error": error}


def _failed_artifact(format_name: str, path: Path, exc: Exception) -> dict:
    return {
        "format": format_name,
        "status": "failed",
        "path": str(path),
        "error_code": "EXPORT_FAILED",
        "error": str(exc),
    }


def _overall_status(artifacts: list[dict]) -> str:
    if not artifacts:
        return "metadata_only"
    if any(artifact.get("status") == "created" for artifact in artifacts):
        return "created"
    if any(artifact.get("status") == "failed" for artifact in artifacts):
        return "failed"
    return "pending_dependencies"


def _input_shape(sample_input_shape: Iterable[int] | None, image_size: int) -> list[int]:
    if sample_input_shape is None:
        return [1, 3, int(image_size), int(image_size)]
    parsed = [int(item) for item in sample_input_shape]
    if len(parsed) != 4:
        raise ValueError("sample_input_shape must have four dimensions.")
    return parsed


def _manifest_artifact_format(artifact_format: str) -> str:
    normalized = str(artifact_format).lower()
    if normalized in {"pytorch", "checkpoint", "framework_native"}:
        return "framework_native_checkpoint"
    return normalized


def _artifact_filename(artifact_format: str, source_path: Path) -> str:
    normalized = str(artifact_format).lower()
    if normalized == "onnx":
        return "model.onnx"
    if normalized == "torchscript":
        return "model.torchscript.pt"
    if normalized in {"pytorch", "checkpoint", "framework_native"}:
        return "model.pt"
    if normalized == "safetensors":
        return "model.safetensors"
    suffix = source_path.suffix or ".bin"
    return f"model{suffix}"
