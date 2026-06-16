from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

from worker.exporting.metadata import build_champion_export_metadata
from worker.exporting.portable_bundle import (
    PORTABLE_ARTIFACT_FORMAT,
    portable_bundle_summary,
    write_portable_inference_bundle,
)
from worker.exporting.preprocessing import benchmark_preprocessing_latency
from worker.exporting.self_test import (
    export_self_test_failed,
    export_self_test_validation_errors,
    export_self_test_warning,
    run_onnx_pytorch_self_test,
)

MANIFEST_SCHEMA_VERSION = "champion_export_manifest_v1"
PROVENANCE_SCHEMA_VERSION = "worker_artifact_provenance_v1"


class ArtifactPathValidationError(ValueError):
    """Raised when an artifact path is outside worker-controlled storage."""


def controlled_artifact_roots(
    *,
    export_dir: Path | None = None,
    extra_roots: Iterable[Path] | None = None,
) -> list[Path]:
    roots: list[Path] = []

    def add_root(value: object) -> None:
        if not value:
            return
        try:
            resolved = Path(value).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return
        if resolved not in roots:
            roots.append(resolved)

    add_root(os.getenv("WORKER_EXPORT_ROOT", ".cache/exports"))
    add_root(os.getenv("WORKER_ARTIFACT_DOWNLOAD_ROOT", ".cache/artifacts"))
    add_root(".cache/artifacts")
    if export_dir is not None:
        add_root(export_dir)
    for root in extra_roots or ():
        add_root(root)
    return roots


def validate_controlled_artifact_path(
    path: Path,
    *,
    export_dir: Path | None = None,
    extra_roots: Iterable[Path] | None = None,
) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactPathValidationError(f"ARTIFACT_SOURCE_REJECTED: invalid artifact path: {exc}") from exc

    for root in controlled_artifact_roots(export_dir=export_dir, extra_roots=extra_roots):
        if _path_is_within(resolved, root):
            return resolved
    raise ArtifactPathValidationError("ARTIFACT_SOURCE_REJECTED: artifact path is outside worker-controlled roots")


def resolve_controlled_artifact_reference(
    value: str | Path,
    *,
    export_dir: Path | None = None,
    extra_roots: Iterable[Path] | None = None,
) -> Path:
    path = _path_from_artifact_reference(value)
    return validate_controlled_artifact_path(path, export_dir=export_dir, extra_roots=extra_roots)


def safe_torch_load_checkpoint(torch_module, checkpoint_path: Path):
    path = Path(checkpoint_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch_module.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        if "weights_only" in str(exc):
            raise ValueError(
                "CHECKPOINT_UNSAFE_PICKLE_REJECTED: installed torch does not support weights_only=True checkpoint loading"
            ) from exc
        raise ValueError(f"CHECKPOINT_LOAD_FAILED: {exc}") from exc
    except Exception as exc:
        raise ValueError(
            "CHECKPOINT_UNSAFE_PICKLE_REJECTED: checkpoint requires unsafe pickle or is not a weights-only checkpoint"
        ) from exc


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
    provenance: dict | None = None,
    validation_errors: Iterable[str] | None = None,
    export_self_test_samples: Iterable[dict] | None = None,
    export_self_test_tolerance: dict | None = None,
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
    metadata["preprocessing_latency_profile"] = benchmark_preprocessing_latency(metadata)
    metadata["provenance"] = _manifest_provenance(
        provenance,
        artifact_format=",".join(requested_formats) if requested_formats else "metadata_only",
        source="worker_generated",
        validation_errors=validation_errors,
    )

    artifacts: list[dict] = []
    if "framework_native" in requested_formats or "checkpoint" in requested_formats:
        artifacts.append(_write_framework_native_checkpoint(export_dir, model, metadata, provenance=provenance))
    if "torchscript" in requested_formats:
        artifacts.append(_write_torchscript(export_dir, model, input_shape, provenance=provenance))
    if "onnx" in requested_formats:
        artifacts.append(_write_onnx(export_dir, model, input_shape, provenance=provenance))

    onnx_artifact = _created_artifact_for_format(artifacts, "onnx")
    if onnx_artifact is not None:
        self_test = run_onnx_pytorch_self_test(
            model=model,
            onnx_path=Path(str(onnx_artifact.get("path") or "")),
            metadata=metadata,
            samples=export_self_test_samples,
            tolerance=export_self_test_tolerance,
        )
        metadata["export_self_test"] = self_test
        metadata["export_status"] = _export_status_from_self_test(metadata.get("runtime"), self_test)
        self_test_errors = export_self_test_validation_errors(self_test)
        if self_test_errors:
            metadata["provenance"]["validation_errors"] = list(
                dict.fromkeys([*metadata["provenance"].get("validation_errors", []), *self_test_errors])
            )

    status = _manifest_status(artifacts, metadata.get("export_self_test"))
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metadata": metadata,
        "artifacts": artifacts,
        "status": status,
    }
    _append_portable_bundle_artifact(
        export_dir,
        manifest,
        requested_format="onnx" if "onnx" in requested_formats else "",
        provenance=provenance,
    )
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_export_manifest(manifest_path: Path) -> dict:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "error",
            "error_code": "MANIFEST_NOT_FOUND",
            "error": f"Export manifest not found: {manifest_path}",
        }
    except json.JSONDecodeError as exc:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "error",
            "error_code": "MANIFEST_INVALID_JSON",
            "error": str(exc),
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
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
    provenance: dict | None = None,
    validation_errors: Iterable[str] | None = None,
) -> dict:
    """Copy an existing worker-visible artifact into a controlled export directory."""
    export_dir.mkdir(parents=True, exist_ok=True)
    validation_errors = list(validation_errors or [])
    manifest_format = _manifest_artifact_format(artifact_format)
    metadata = build_champion_export_metadata(
        model_name=model_name,
        class_names=class_names,
        image_size=image_size,
        preprocessing=preprocessing,
        model_profile=model_profile,
        training_config=training_config,
    )
    metadata["preprocessing_latency_profile"] = benchmark_preprocessing_latency(metadata)
    try:
        source_path = validate_controlled_artifact_path(Path(source_artifact_path), export_dir=export_dir)
    except ArtifactPathValidationError as exc:
        validation_errors.append(str(exc))
        metadata["provenance"] = _manifest_provenance(
            provenance,
            artifact_format=manifest_format,
            source="worker_controlled_copy",
            validation_errors=validation_errors,
        )
        return _write_manifest(
            export_dir,
            metadata,
            [
                _skipped_artifact(
                    manifest_format,
                    "ARTIFACT_SOURCE_REJECTED",
                    str(exc),
                    provenance=provenance,
                    validation_errors=validation_errors,
                )
            ],
            "failed",
            requested_format=manifest_format,
            provenance=provenance,
        )

    if not source_path.exists() or not source_path.is_file():
        validation_errors.append("ARTIFACT_NOT_FOUND: source artifact is not available to the worker")
        metadata["provenance"] = _manifest_provenance(
            provenance,
            artifact_format=manifest_format,
            source="worker_controlled_copy",
            validation_errors=validation_errors,
        )
        return _write_manifest(
            export_dir,
            metadata,
            [
                _skipped_artifact(
                    manifest_format,
                    "ARTIFACT_NOT_FOUND",
                    "Source artifact not found in worker-controlled storage.",
                    provenance=provenance,
                    validation_errors=validation_errors,
                )
            ],
            "pending_dependencies",
            requested_format=manifest_format,
            provenance=provenance,
        )

    input_shape = _input_shape(sample_input_shape, image_size)
    metadata["input_shape"] = input_shape
    metadata["provenance"] = _manifest_provenance(
        provenance,
        artifact_format=manifest_format,
        source="worker_controlled_copy",
        source_path=source_path,
        validation_errors=validation_errors,
    )

    artifact_name = _artifact_filename(artifact_format, source_path)
    destination = export_dir / artifact_name
    if source_path.resolve() != destination.resolve():
        shutil.copy2(source_path, destination)
    if _manifest_artifact_format(artifact_format) == "onnx":
        _copy_onnx_external_data(source_path, export_dir)

    if manifest_format == "onnx":
        artifact = _created_onnx_artifact(
            destination,
            provenance=provenance,
            source="worker_controlled_copy",
            source_path=source_path,
            validation_errors=validation_errors,
        )
    else:
        artifact = _created_artifact(
            manifest_format,
            destination,
            provenance=provenance,
            source="worker_controlled_copy",
            source_path=source_path,
            validation_errors=validation_errors,
        )
    return _write_manifest(
        export_dir,
        metadata,
        [artifact],
        "created",
        requested_format=manifest_format,
        provenance=provenance,
    )


def _write_framework_native_checkpoint(export_dir: Path, model, metadata: dict, *, provenance: dict | None = None) -> dict:
    if model is None:
        return _skipped_artifact(
            "framework_native_checkpoint",
            "MODEL_UNAVAILABLE",
            "A model instance is required to create a framework-native checkpoint.",
            provenance=provenance,
        )
    if not hasattr(model, "state_dict"):
        return _skipped_artifact(
            "framework_native_checkpoint",
            "STATE_DICT_UNAVAILABLE",
            "The supplied model does not expose state_dict().",
            provenance=provenance,
        )
    try:
        import torch
    except Exception as exc:
        return _skipped_artifact("framework_native_checkpoint", "TORCH_UNAVAILABLE", str(exc), provenance=provenance)

    path = export_dir / "model.pt"
    try:
        torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    except Exception as exc:
        return _failed_artifact("framework_native_checkpoint", path, exc, provenance=provenance)
    return _created_artifact("framework_native_checkpoint", path, provenance=provenance, source="worker_generated")


def _write_torchscript(export_dir: Path, model, input_shape: list[int], *, provenance: dict | None = None) -> dict:
    if model is None:
        return _skipped_artifact(
            "torchscript",
            "MODEL_UNAVAILABLE",
            "A model instance is required to create a TorchScript artifact.",
            provenance=provenance,
        )
    try:
        import torch
    except Exception as exc:
        return _skipped_artifact("torchscript", "TORCH_UNAVAILABLE", str(exc), provenance=provenance)

    path = export_dir / "model.torchscript.pt"
    try:
        model.eval()
        dummy_input = torch.zeros(tuple(input_shape), dtype=torch.float32)
        traced = torch.jit.trace(model.cpu(), dummy_input)
        traced.save(str(path))
    except Exception as exc:
        return _failed_artifact("torchscript", path, exc, provenance=provenance)
    return _created_artifact("torchscript", path, provenance=provenance, source="worker_generated")


def _write_onnx(export_dir: Path, model, input_shape: list[int], *, provenance: dict | None = None) -> dict:
    if model is None:
        return _skipped_artifact(
            "onnx",
            "MODEL_UNAVAILABLE",
            "A model instance is required to create an ONNX artifact.",
            provenance=provenance,
        )
    try:
        import torch
    except Exception as exc:
        return _skipped_artifact("onnx", "TORCH_UNAVAILABLE", str(exc), provenance=provenance)
    try:
        import onnxscript  # noqa: F401
    except Exception as exc:
        return _skipped_artifact("onnx", "ONNXSCRIPT_UNAVAILABLE", str(exc), provenance=provenance)

    path = export_dir / "model.onnx"
    try:
        model.eval()
        dummy_input = torch.zeros(tuple(input_shape), dtype=torch.float32)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            torch.onnx.export(
                model.cpu(),
                dummy_input,
                str(path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                opset_version=18,
            )
    except Exception as exc:
        if "onnx" in str(exc).lower() and "install" in str(exc).lower():
            return _skipped_artifact("onnx", "ONNX_UNAVAILABLE", str(exc), provenance=provenance)
        return _failed_artifact("onnx", path, exc, provenance=provenance)
    return _created_onnx_artifact(path, provenance=provenance, source="worker_generated")


def _created_artifact(
    format_name: str,
    path: Path,
    *,
    provenance: dict | None = None,
    source: str = "worker_generated",
    source_path: Path | None = None,
    validation_errors: Iterable[str] | None = None,
) -> dict:
    bytes_count = path.stat().st_size if path.exists() else 0
    return {
        "format": format_name,
        "status": "created",
        "path": str(path),
        "bytes": bytes_count,
        "provenance": _artifact_provenance(
            provenance,
            artifact_format=format_name,
            source=source,
            artifact_bytes=bytes_count,
            source_path=source_path,
            validation_errors=validation_errors,
        ),
    }


def _created_onnx_artifact(
    path: Path,
    *,
    provenance: dict | None = None,
    source: str = "worker_generated",
    source_path: Path | None = None,
    validation_errors: Iterable[str] | None = None,
) -> dict:
    artifact = _created_artifact(
        "onnx",
        path,
        provenance=provenance,
        source=source,
        source_path=source_path,
        validation_errors=validation_errors,
    )
    external_data = _onnx_external_data_records(path)
    if external_data:
        artifact["external_data"] = external_data
    return artifact


def _copy_onnx_external_data(source_model_path: Path, export_dir: Path) -> None:
    source_dir = source_model_path.resolve().parent
    for record in _onnx_external_data_records(source_model_path):
        source_path = Path(str(record.get("artifact_path") or "")).resolve()
        if not _path_is_within(source_path, source_dir):
            continue
        destination = _safe_external_data_path(export_dir, str(record.get("path") or ""))
        if destination is None or not source_path.exists() or not source_path.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)


def _onnx_external_data_records(model_path: Path) -> list[dict]:
    records: list[dict] = []
    seen: set[str] = set()
    locations = _onnx_external_data_locations(model_path)
    if not locations:
        fallback = model_path.with_name(f"{model_path.name}.data")
        if fallback.exists() and fallback.is_file():
            locations = [fallback.name]

    for location in locations:
        if location in seen:
            continue
        seen.add(location)
        data_path = _safe_external_data_path(model_path.parent, location)
        if data_path is None or not data_path.exists() or not data_path.is_file():
            continue
        records.append(
            {
                "path": location,
                "artifact_path": str(data_path),
                "bytes": data_path.stat().st_size,
            }
        )
    return records


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
        location = _tensor_external_data_location(tensor)
        if location and location not in locations:
            locations.append(location)
    return locations


def _tensor_external_data_location(tensor) -> str:
    for entry in getattr(tensor, "external_data", []):
        if getattr(entry, "key", "") == "location":
            value = str(getattr(entry, "value", "")).strip()
            if value:
                return value
    return ""


def _safe_external_data_path(base_dir: Path, relative_path: str) -> Path | None:
    if not relative_path:
        return None
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return None
    base = base_dir.resolve()
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return resolved


def _skipped_artifact(
    format_name: str,
    error_code: str,
    error: str,
    *,
    provenance: dict | None = None,
    validation_errors: Iterable[str] | None = None,
) -> dict:
    return {
        "format": format_name,
        "status": "skipped",
        "error_code": error_code,
        "error": error,
        "provenance": _artifact_provenance(
            provenance,
            artifact_format=format_name,
            source="unavailable",
            validation_errors=validation_errors,
        ),
    }


def _failed_artifact(format_name: str, path: Path, exc: Exception, *, provenance: dict | None = None) -> dict:
    return {
        "format": format_name,
        "status": "failed",
        "path": str(path),
        "error_code": "EXPORT_FAILED",
        "error": str(exc),
        "provenance": _artifact_provenance(
            provenance,
            artifact_format=format_name,
            source="worker_generated",
            validation_errors=[str(exc)],
        ),
    }


def _overall_status(artifacts: list[dict]) -> str:
    if not artifacts:
        return "metadata_only"
    if any(artifact.get("status") == "created" for artifact in artifacts):
        return "created"
    if any(artifact.get("status") == "failed" for artifact in artifacts):
        return "failed"
    return "pending_dependencies"


def _manifest_status(artifacts: list[dict], self_test: object) -> str:
    status = _overall_status(artifacts)
    if status != "created" or not isinstance(self_test, dict):
        return status
    if export_self_test_failed(self_test):
        return "failed_validation"
    if export_self_test_warning(self_test):
        return "created_with_warnings"
    return status


def _created_artifact_for_format(artifacts: list[dict], format_name: str) -> dict | None:
    for artifact in artifacts:
        if (
            isinstance(artifact, dict)
            and artifact.get("format") == format_name
            and artifact.get("status") == "created"
            and artifact.get("path")
        ):
            return artifact
    return None


def _export_status_from_self_test(runtime: object, self_test: dict) -> dict:
    status = str(self_test.get("status") or "not_run")
    if status == "passed":
        export_status = "self_test_passed"
    elif status == "failed":
        export_status = "self_test_failed"
    elif status == "warning":
        export_status = "self_test_warning"
    else:
        export_status = "self_test_not_run"
    return {
        "schema_version": "champion_export_status_v1",
        "status": export_status,
        "runtime": str(runtime or "onnx"),
        "self_test_status": status,
        "export_verified": bool(self_test.get("export_verified")),
        "diagnostic_reason": str(self_test.get("diagnostic_reason") or ""),
    }


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


def _write_manifest(
    export_dir: Path,
    metadata: dict,
    artifacts: list[dict],
    status: str,
    *,
    requested_format: str = "",
    provenance: dict | None = None,
) -> dict:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "metadata": metadata,
        "artifacts": artifacts,
        "status": status,
    }
    _append_portable_bundle_artifact(
        export_dir,
        manifest,
        requested_format=requested_format,
        provenance=provenance,
    )
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _append_portable_bundle_artifact(
    export_dir: Path,
    manifest: dict,
    *,
    requested_format: str,
    provenance: dict | None = None,
) -> None:
    if str(requested_format).lower() != "onnx":
        return
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    if any(isinstance(artifact, dict) and artifact.get("format") == PORTABLE_ARTIFACT_FORMAT for artifact in artifacts):
        return
    bundle_artifact = write_portable_inference_bundle(
        export_dir=export_dir,
        manifest=manifest,
        requested_format=requested_format,
        provenance=provenance,
    )
    artifacts.append(bundle_artifact)
    manifest["artifacts"] = artifacts
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    summary = portable_bundle_summary(bundle_artifact)
    metadata["portable_inference_bundle"] = summary
    if summary.get("artifact_uri"):
        metadata["portable_bundle_uri"] = summary["artifact_uri"]
    manifest["metadata"] = metadata


def _manifest_provenance(
    provenance: dict | None,
    *,
    artifact_format: str,
    source: str,
    source_path: Path | None = None,
    validation_errors: Iterable[str] | None = None,
) -> dict:
    record = _base_provenance(provenance, artifact_format=artifact_format, source=source)
    if source_path is not None and source_path.exists():
        record["source_artifact_bytes"] = source_path.stat().st_size
    record["validation_errors"] = [str(error) for error in (validation_errors or ()) if str(error)]
    return record


def _artifact_provenance(
    provenance: dict | None,
    *,
    artifact_format: str,
    source: str,
    artifact_bytes: int = 0,
    source_path: Path | None = None,
    validation_errors: Iterable[str] | None = None,
) -> dict:
    record = _manifest_provenance(
        provenance,
        artifact_format=artifact_format,
        source=source,
        source_path=source_path,
        validation_errors=validation_errors,
    )
    record["artifact_bytes"] = int(artifact_bytes)
    return record


def _base_provenance(provenance: dict | None, *, artifact_format: str, source: str) -> dict:
    source_data = provenance if isinstance(provenance, dict) else {}
    record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_by": "model-express-worker",
        "source": source,
        "artifact_format": artifact_format,
    }
    for key in ("source_job_id", "source_export_id", "export_job_id"):
        value = source_data.get(key)
        if value:
            record[key] = str(value)
    return record


def _path_from_artifact_reference(value: str | Path) -> Path:
    if isinstance(value, Path):
        return value
    text = str(value).strip()
    if not text:
        raise ArtifactPathValidationError("ARTIFACT_SOURCE_REJECTED: empty artifact path")
    if os.name == "nt" and len(text) > 2 and text[1] == ":":
        return Path(text)
    parsed = urlparse(text)
    if parsed.scheme == "file":
        path_value = unquote(parsed.path)
        if os.name == "nt" and path_value.startswith("/") and len(path_value) > 2 and path_value[2] == ":":
            path_value = path_value[1:]
        return Path(path_value)
    if parsed.scheme:
        raise ArtifactPathValidationError("UNSUPPORTED_ARTIFACT_URI: artifact URI must be s3:// or a worker-controlled local path")
    return Path(text)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
