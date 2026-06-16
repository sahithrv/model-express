from __future__ import annotations

import os
import base64
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from worker.datasets.cache import (
    cleanup_job_dataset_cache,
    dataset_archive_path,
    extract_dataset_archive,
    job_dataset_cache_root,
    should_persist_dataset_cache,
)
from worker.datasets.exemplars import generate_visual_exemplars
from worker.datasets.storage import download_s3_uri
from worker.exporting.artifacts import (
    ArtifactPathValidationError,
    PROVENANCE_SCHEMA_VERSION,
    produce_champion_export_artifacts,
    produce_existing_champion_export_manifest,
    resolve_controlled_artifact_reference,
    safe_torch_load_checkpoint,
    validate_controlled_artifact_path,
)
from worker.exporting.inference import run_demo_inference_from_manifest
from worker.orchestrator_client import OrchestratorClient

SUPPORTED_EXPORT_FORMATS = {"onnx", "torchscript", "pytorch", "safetensors"}
HELPER_EXPORT_FORMATS = {
    "onnx": "onnx",
    "torchscript": "torchscript",
    "pytorch": "framework_native",
}


def run_export_champion_job(client: OrchestratorClient, job: dict) -> None:
    config = _config(job)
    job_id = str(job["id"])
    requested_format = _export_format(config)
    export_dir = _export_dir(config, job_id, requested_format)
    dataset_profile = _dataset_profile(client, config)
    class_names = _class_names(config, dataset_profile)
    image_size = _positive_int(config.get("image_size"), 224)
    export_metadata = config.get("metadata") if isinstance(config.get("metadata"), dict) else {}
    deployment_profile = export_metadata.get("deployment_profile") if isinstance(export_metadata.get("deployment_profile"), dict) else {}
    model_profile = config.get("model_profile") if isinstance(config.get("model_profile"), dict) else {}
    if not model_profile and isinstance(deployment_profile.get("model_profile"), dict):
        model_profile = deployment_profile["model_profile"]
    model_name = (
        _first_string(config, "model", "model_name", "champion_model")
        or _first_string(deployment_profile, "model", "model_name", "champion_model")
        or _first_string(model_profile, "model", "model_name", "architecture")
        or "unknown_model"
    )
    preprocessing = config.get("preprocessing") if isinstance(config.get("preprocessing"), dict) else {}
    if not preprocessing and isinstance(deployment_profile.get("preprocessing"), dict):
        preprocessing = deployment_profile["preprocessing"]
    training_config = config.get("training_config") if isinstance(config.get("training_config"), dict) else {}
    if not training_config and isinstance(deployment_profile.get("training_config"), dict):
        training_config = deployment_profile["training_config"]
    if not training_config:
        training_config = config

    validation_errors: list[str] = []
    if requested_format not in SUPPORTED_EXPORT_FORMATS:
        validation_errors.append(
            f"unsupported export format {requested_format!r}; expected one of {sorted(SUPPORTED_EXPORT_FORMATS)}"
        )
        manifest = _error_manifest(export_dir, requested_format, validation_errors)
        result = _export_result_payload(
            status="FAILED",
            requested_format=requested_format,
            manifest=manifest,
            validation_errors=validation_errors,
        )
        client.report_champion_export_result(job_id, result)
        client.fail_job(job_id, "; ".join(validation_errors))
        return

    provenance = _export_provenance(config, job_id, requested_format)
    source_artifact, source_errors = _existing_artifact_source_path(config, requested_format, export_dir)
    validation_errors.extend(source_errors)
    if source_artifact is not None:
        manifest = produce_existing_champion_export_manifest(
            export_dir=export_dir,
            source_artifact_path=source_artifact,
            artifact_format=requested_format,
            model_name=model_name,
            class_names=class_names,
            image_size=image_size,
            preprocessing=preprocessing,
            model_profile=model_profile,
            training_config=training_config,
            sample_input_shape=config.get("sample_input_shape"),
            provenance=provenance,
            validation_errors=validation_errors,
        )
    elif source_errors:
        manifest = _error_manifest(export_dir, requested_format, validation_errors, provenance=provenance)
    elif requested_format in HELPER_EXPORT_FORMATS:
        model, checkpoint_metadata, checkpoint_errors = _load_model_from_convertible_checkpoint(
            config,
            requested_format,
            class_names,
            model_name,
            export_dir,
        )
        validation_errors.extend(checkpoint_errors)
        if checkpoint_errors:
            manifest = _error_manifest(export_dir, requested_format, validation_errors, provenance=provenance)
        else:
            if model is None:
                model, fallback_errors = _build_architecture_export_fallback(
                    training_config if isinstance(training_config, dict) else config,
                    requested_format,
                    class_names,
                    model_name,
                )
                validation_errors.extend(fallback_errors)
            label_order_errors = _class_label_order_errors(class_names, checkpoint_metadata)
            validation_errors.extend(label_order_errors)
            if label_order_errors:
                manifest = _error_manifest(export_dir, requested_format, validation_errors, provenance=provenance)
            else:
                if checkpoint_metadata:
                    if not class_names:
                        class_names = _metadata_class_names(checkpoint_metadata)
                    if not config.get("image_size"):
                        image_size = _metadata_image_size(checkpoint_metadata, image_size)
                    if not model_profile and isinstance(checkpoint_metadata.get("model_profile"), dict):
                        model_profile = checkpoint_metadata["model_profile"]
                    if not isinstance(config.get("training_config"), dict) and isinstance(
                        checkpoint_metadata.get("training_config"), dict
                    ):
                        training_config = checkpoint_metadata["training_config"]
                    model_name = _first_string(checkpoint_metadata, "model", "model_name") or model_name
                manifest = produce_champion_export_artifacts(
                    export_dir=export_dir,
                    model_name=model_name,
                    class_names=class_names,
                    image_size=image_size,
                    model=model,
                    preprocessing=preprocessing,
                    model_profile=model_profile,
                    training_config=training_config,
                    formats=(HELPER_EXPORT_FORMATS[requested_format],),
                    sample_input_shape=config.get("sample_input_shape"),
                    provenance=provenance,
                    validation_errors=validation_errors,
                )
    else:
        validation_errors.append(
            "safetensors export requires an existing worker-visible safetensors artifact; no source artifact was provided"
        )
        manifest = _error_manifest(export_dir, requested_format, validation_errors, status="pending_dependencies", provenance=provenance)

    created_artifacts = _created_artifacts(manifest)
    if created_artifacts:
        status = "READY"
    elif validation_errors and requested_format == "safetensors":
        status = "PENDING_ARTIFACT"
    elif any(artifact.get("status") == "failed" for artifact in manifest.get("artifacts", [])):
        status = "FAILED"
        validation_errors.extend(_artifact_errors(manifest))
    else:
        status = "PENDING_ARTIFACT"
        validation_errors.extend(_artifact_errors(manifest))

    result = _export_result_payload(
        status=status,
        requested_format=requested_format,
        manifest=manifest,
        validation_errors=validation_errors,
    )
    client.report_champion_export_result(job_id, result)
    if status == "FAILED":
        client.fail_job(job_id, result.get("error") or "champion export failed")
        return
    client.complete_job(job_id, mlflow_run_id="")


def run_champion_demo_prediction_job(client: OrchestratorClient, job: dict) -> None:
    config = _config(job)
    job_id = str(job["id"])
    manifest_path = _manifest_path(config)
    manifest_payload = _manifest_payload(config) if manifest_path is None else None
    image_path, image_error = _demo_image_path(config, job_id)
    image_uri = _first_string(config, "image_uri", "image_path", "local_image_path") or ""
    true_label = _first_string(config, "true_label", "label", "class_name")
    image_metadata = config.get("image_metadata") if isinstance(config.get("image_metadata"), dict) else {}

    if manifest_path is None and manifest_payload is None:
        payload = {
            "status": "RUNTIME_UNAVAILABLE",
            "error_code": "MANIFEST_NOT_CONFIGURED",
            "error": "No worker-owned export manifest path or manifest metadata was supplied or found.",
        }
    elif image_path is None:
        payload = {
            "status": "FAILED",
            "error_code": "IMAGE_UNAVAILABLE",
            "error": image_error or "Demo image is unavailable to the worker.",
        }
    else:
        inference = run_demo_inference_from_manifest(
            manifest_path=manifest_path,
            manifest=manifest_payload,
            image_path=image_path,
            top_k=_positive_int(config.get("top_k"), 5),
            true_label=true_label,
            image_metadata=image_metadata,
            confidence_threshold=_optional_float(config.get("confidence_threshold")),
            iou_threshold=_optional_float(config.get("iou_threshold")),
            max_detections=_optional_positive_int(config.get("max_detections")),
        )
        payload = _prediction_result_from_inference(inference)

    inference_image_metadata = payload.get("image_metadata") if isinstance(payload.get("image_metadata"), dict) else {}
    result_image_metadata = {**dict(image_metadata), **inference_image_metadata}
    latency_breakdown = payload.get("latency_breakdown_ms")
    if isinstance(latency_breakdown, dict):
        result_image_metadata["latency_breakdown_ms"] = latency_breakdown
        result_image_metadata["latency_measurement"] = {
            "latency_ms_semantics": "streaming_total_excludes_one_time_model_load",
            "runtime": payload.get("runtime", ""),
        }

    payload.update(
        {
            "image_uri": image_uri,
            "image_id": _first_string(config, "image_id") or payload.get("image_id", ""),
            "true_label": true_label or payload.get("true_label", ""),
            "image_metadata": result_image_metadata,
            "manifest_path": str(manifest_path) if manifest_path is not None else "",
        }
    )
    client.report_champion_demo_prediction_result(job_id, payload)
    if payload["status"] == "FAILED":
        client.fail_job(job_id, payload.get("error") or "champion demo prediction failed")
        return
    client.complete_job(job_id, mlflow_run_id="")


def run_generate_visual_exemplars_job(client: OrchestratorClient, job: dict) -> None:
    config = _config(job)
    job_id = str(job["id"])
    dataset_id = _first_string(config, "dataset_id")
    if not dataset_id:
        message = "dataset_id is required for generate_visual_exemplars jobs"
        client.fail_job(job_id, message)
        return

    dataset = client.get_dataset(dataset_id)
    cache_root = job_dataset_cache_root(job_id)
    try:
        archive_path = dataset_archive_path(dataset_id, cache_root)
        download_s3_uri(dataset["storage_uri"], archive_path)
        dataset_dir = extract_dataset_archive(archive_path, dataset_id, cache_root)
        caps = _exemplar_caps(config)
        output_dir = _exemplar_dir(dataset_id, job_id)
        pack = generate_visual_exemplars(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            images_per_class=caps["images_per_class"],
            max_total_images=caps["max_total_images"],
            max_image_bytes=caps["max_image_bytes"],
            max_total_bytes=caps["max_total_bytes"],
            image_size=caps["image_size"],
            quality=caps["quality"],
            seed=_positive_int(config.get("seed"), 0),
        )
        payload = _validated_exemplar_payload(pack, caps)
        client.report_dataset_visual_exemplars(dataset_id, payload)
        if payload.get("status") == "unavailable":
            client.fail_job(job_id, payload.get("error") or "visual exemplar generation unavailable")
            return
        client.complete_job(job_id, mlflow_run_id="")
    finally:
        if not should_persist_dataset_cache():
            cleanup_job_dataset_cache(job_id, cache_root)


def _config(job: dict) -> dict:
    config = job.get("config", {})
    return config if isinstance(config, dict) else {}


def _export_format(config: dict) -> str:
    return (_first_string(config, "format", "export_format", "requested_format") or "onnx").lower()


def _export_root() -> Path:
    return Path(os.getenv("WORKER_EXPORT_ROOT", ".cache/exports"))


def _export_dir(config: dict, job_id: str, requested_format: str) -> Path:
    champion_job_id = _first_string(config, "champion_job_id", "source_job_id", "training_job_id") or job_id
    return _export_root() / _safe_path_part(champion_job_id) / _safe_path_part(requested_format) / _safe_path_part(job_id)


def _export_provenance(config: dict, job_id: str, artifact_format: str) -> dict:
    return {
        "export_job_id": job_id,
        "source_job_id": _first_string(config, "champion_job_id", "source_job_id", "training_job_id"),
        "source_export_id": _first_string(config, "source_export_id", "export_id", "champion_export_id"),
        "artifact_format": artifact_format,
    }


def _exemplar_dir(dataset_id: str, job_id: str) -> Path:
    return Path(os.getenv("WORKER_EXEMPLAR_ROOT", ".cache/exemplars")) / _safe_path_part(dataset_id) / _safe_path_part(job_id)


def _dataset_profile(client: OrchestratorClient, config: dict) -> dict:
    profile = config.get("dataset_profile")
    if isinstance(profile, dict):
        return profile
    dataset_id = _first_string(config, "dataset_id")
    if not dataset_id:
        return {}
    try:
        dataset = client.get_dataset(dataset_id)
    except Exception:
        return {}
    profile = dataset.get("profile")
    return profile if isinstance(profile, dict) else {}


def _class_names(config: dict, dataset_profile: dict) -> list[str]:
    for key in ("class_names", "class_labels", "classes"):
        names = config.get(key)
        if isinstance(names, list):
            return [str(item) for item in names]
    profile_names = dataset_profile.get("class_names")
    if isinstance(profile_names, list):
        return [str(item) for item in profile_names]
    distribution = dataset_profile.get("class_distribution")
    if isinstance(distribution, dict):
        return sorted(str(key) for key in distribution)
    return []


def _load_model_from_convertible_checkpoint(
    config: dict,
    requested_format: str,
    class_names: list[str],
    fallback_model_name: str,
    export_dir: Path,
):
    if requested_format not in {"onnx", "torchscript"}:
        return None, {}, []
    checkpoint_path, path_errors = _existing_convertible_checkpoint_path(config, export_dir)
    if path_errors:
        return None, {}, path_errors
    if checkpoint_path is None:
        return None, {}, []

    try:
        import torch
    except Exception as exc:
        return None, {}, [f"TORCH_UNAVAILABLE: {exc}"]

    try:
        payload = safe_torch_load_checkpoint(torch, checkpoint_path)
    except FileNotFoundError as exc:
        return None, {}, [f"CHECKPOINT_LOAD_FAILED: {exc}"]
    except ValueError as exc:
        return None, {}, [str(exc)]

    metadata: dict = {}
    state_dict = None
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state_dict = payload["state_dict"]
        if isinstance(payload.get("metadata"), dict):
            metadata = payload["metadata"]
    elif isinstance(payload, dict):
        state_dict = payload
    if not isinstance(state_dict, dict):
        return None, metadata, ["CHECKPOINT_LOAD_FAILED: checkpoint does not contain a state_dict"]
    if not _state_dict_values_are_tensors(torch, state_dict):
        return None, metadata, ["CHECKPOINT_LOAD_FAILED: checkpoint state_dict contains non-tensor entries"]

    labels = class_names or _metadata_class_names(metadata)
    class_count = max(1, len(labels))
    training_config = metadata.get("training_config") if isinstance(metadata.get("training_config"), dict) else config
    model_name = (
        _first_string(config, "model", "model_name", "champion_model")
        or _first_string(metadata, "model", "model_name")
        or fallback_model_name
    )

    try:
        model = _build_torchvision_model(
            model_name=model_name,
            class_count=class_count,
            pretrained=False,
            freeze_backbone=_bool_value(training_config.get("freeze_backbone"), True),
            fine_tune_strategy=str(training_config.get("fine_tune_strategy") or "head_only"),
            dropout=_float_value(training_config.get("dropout"), 0.0),
        )
        model.load_state_dict(_normalized_state_dict(state_dict), strict=False)
        model.eval()
    except Exception as exc:
        return None, metadata, [f"CHECKPOINT_MODEL_LOAD_FAILED: {exc}"]

    return model, metadata, []


def _build_architecture_export_fallback(
    config: dict,
    requested_format: str,
    class_names: list[str],
    fallback_model_name: str,
):
    if requested_format not in {"onnx", "torchscript"}:
        return None, []
    if _is_detection_export_config(config, fallback_model_name):
        return None, ["SOURCE_ARTIFACT_REQUIRED: object detection exports require a trained detector artifact"]
    labels = class_names or _config_class_names(config)
    class_count = max(1, len(labels))
    try:
        model = _build_torchvision_model(
            model_name=fallback_model_name,
            class_count=class_count,
            pretrained=False,
            freeze_backbone=_bool_value(config.get("freeze_backbone"), True),
            fine_tune_strategy=str(config.get("fine_tune_strategy") or "head_only"),
            dropout=_float_value(config.get("dropout"), 0.0),
        )
        model.eval()
    except Exception as exc:
        return None, [f"ARCHITECTURE_EXPORT_FALLBACK_FAILED: {exc}"]
    return model, []


def _existing_convertible_checkpoint_path(config: dict, export_dir: Path) -> tuple[Path | None, list[str]]:
    errors: list[str] = []
    for key in (
        "checkpoint_path",
        "pytorch_path",
        "model_path",
        "checkpoint_uri",
        "pytorch_uri",
        "model_uri",
        "source_artifact_uri",
        "source_artifact_path",
        "artifact_path",
        "local_artifact_path",
        "artifact_uri",
        "export_artifact_uri",
    ):
        path, error = _artifact_source_path_for_key(config, key, "pytorch", export_dir)
        if error:
            errors.append(error)
            continue
        if path is not None:
            return path, errors
    return None, errors


def _metadata_class_names(metadata: dict) -> list[str]:
    for key in ("class_names", "class_labels", "classes"):
        values = metadata.get(key)
        if isinstance(values, list):
            return [str(item) for item in values]
    return []


def _class_label_order_errors(config_labels: list[str], metadata: dict) -> list[str]:
    metadata_labels = _metadata_class_names(metadata)
    if not config_labels or not metadata_labels:
        return []
    normalized_config = [str(label) for label in config_labels]
    normalized_metadata = [str(label) for label in metadata_labels]
    if normalized_config == normalized_metadata:
        return []
    return [
        "CLASS_LABEL_ORDER_MISMATCH: export class_labels do not match checkpoint metadata class_labels; "
        "refusing export because model output indices would map to the wrong labels"
    ]


def _config_class_names(config: dict) -> list[str]:
    for key in ("class_names", "class_labels", "classes"):
        values = config.get(key)
        if isinstance(values, list):
            return [str(item) for item in values]
    profile = config.get("dataset_profile") if isinstance(config.get("dataset_profile"), dict) else {}
    values = profile.get("class_names")
    if isinstance(values, list):
        return [str(item) for item in values]
    distribution = profile.get("class_distribution")
    if isinstance(distribution, dict):
        return sorted(str(key) for key in distribution)
    return []


def _is_detection_export_config(config: dict, model_name: str) -> bool:
    normalized = str(model_name or "").lower()
    return (
        str(config.get("task_type", "")).lower() == "object_detection"
        or str(config.get("model_kind", "")).lower() == "ultralytics_yolo_detector"
        or normalized.startswith("yolo")
    )


def _metadata_image_size(metadata: dict, default: int) -> int:
    input_shape = metadata.get("input_shape")
    if isinstance(input_shape, list) and len(input_shape) >= 4:
        return _positive_int(input_shape[-1], default)
    return _positive_int(metadata.get("image_size"), default)


def _normalized_state_dict(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        normalized = str(key)
        if normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        out[normalized] = value
    return out


def _state_dict_values_are_tensors(torch, state_dict: dict) -> bool:
    return all(isinstance(key, str) and torch.is_tensor(value) for key, value in state_dict.items())


def _existing_artifact_source_path(config: dict, requested_format: str, export_dir: Path) -> tuple[Path | None, list[str]]:
    format_keys = {
        "onnx": ("onnx_path", "onnx_artifact_path"),
        "torchscript": ("torchscript_path", "torchscript_artifact_path"),
        "pytorch": ("checkpoint_path", "pytorch_path", "model_path"),
        "safetensors": ("safetensors_path", "safetensors_artifact_path"),
    }
    uri_keys = {
        "onnx": ("onnx_artifact_uri", "onnx_uri"),
        "torchscript": ("torchscript_artifact_uri", "torchscript_uri"),
        "pytorch": ("checkpoint_uri", "pytorch_uri", "model_uri"),
        "safetensors": ("safetensors_artifact_uri", "safetensors_uri"),
    }
    keys = (*format_keys.get(requested_format, ()), *uri_keys.get(requested_format, ()))
    generic_keys = ("artifact_path", "local_artifact_path", "artifact_uri", "export_artifact_uri")
    errors: list[str] = []
    for key in (*keys, *generic_keys):
        path, error = _artifact_source_path_for_key(config, key, requested_format, export_dir)
        if error:
            errors.append(error)
            continue
        if path is not None:
            return path, errors
    return None, errors


def _artifact_source_path_for_key(
    config: dict,
    key: str,
    requested_format: str,
    export_dir: Path,
) -> tuple[Path | None, str | None]:
    value = _first_string(config, key)
    if not value:
        return None, None
    if not _artifact_value_matches_format(value, requested_format):
        return None, None
    if value.startswith("s3://"):
        destination = _downloaded_artifact_path(value, requested_format)
        try:
            downloaded = download_s3_uri(value, destination)
            return validate_controlled_artifact_path(downloaded, export_dir=destination.parent), None
        except ArtifactPathValidationError as exc:
            return None, str(exc)
        except Exception:
            return None, None
    try:
        return resolve_controlled_artifact_reference(value, export_dir=export_dir), None
    except ArtifactPathValidationError as exc:
        return None, str(exc)


def _artifact_value_matches_format(value: str, requested_format: str) -> bool:
    parsed = urlparse(value)
    path_value = unquote(parsed.path if parsed.scheme else value).lower()
    if requested_format == "onnx":
        return path_value.endswith(".onnx")
    if requested_format == "torchscript":
        return path_value.endswith(".torchscript.pt") or path_value.endswith(".torchscript")
    if requested_format == "pytorch":
        return path_value.endswith(".pt") or path_value.endswith(".pth")
    if requested_format == "safetensors":
        return path_value.endswith(".safetensors")
    return False


def _downloaded_artifact_path(uri: str, requested_format: str) -> Path:
    parsed = urlparse(uri)
    filename = Path(parsed.path).name or _artifact_filename_for_format(requested_format)
    return Path(os.getenv("WORKER_ARTIFACT_DOWNLOAD_ROOT", ".cache/artifacts")) / _safe_path_part(parsed.netloc) / _safe_path_part(filename)


def _artifact_filename_for_format(requested_format: str) -> str:
    if requested_format == "onnx":
        return "model.onnx"
    if requested_format == "torchscript":
        return "model.torchscript.pt"
    if requested_format == "safetensors":
        return "model.safetensors"
    return "model.pt"


def _manifest_path(config: dict) -> Path | None:
    for key in ("manifest_path", "export_manifest_path", "local_manifest_path", "manifest_uri", "export_manifest_uri"):
        value = _first_string(config, key)
        if value:
            if value.startswith("s3://"):
                destination = _downloaded_artifact_path(value, "manifest")
                try:
                    downloaded = download_s3_uri(value, destination)
                    return validate_controlled_artifact_path(downloaded, export_dir=destination.parent)
                except Exception:
                    continue
            try:
                return resolve_controlled_artifact_reference(value)
            except ArtifactPathValidationError:
                continue
    champion_job_id = _first_string(config, "champion_job_id", "source_job_id", "training_job_id")
    if not champion_job_id:
        return None
    base = _export_root() / _safe_path_part(champion_job_id)
    matches = sorted(base.glob("**/manifest.json")) if base.exists() else []
    return matches[0] if matches else None


def _manifest_payload(config: dict) -> dict | None:
    for record in _manifest_metadata_records(config):
        manifest = record.get("manifest")
        if isinstance(manifest, dict) and manifest:
            return manifest
        manifest = record.get("export_manifest")
        if isinstance(manifest, dict) and manifest:
            return manifest
    return _fallback_manifest_payload(config)


def _manifest_metadata_records(config: dict) -> list[dict]:
    records: list[dict] = []
    for key in ("export_metadata", "metadata"):
        value = config.get(key)
        if isinstance(value, dict):
            records.append(value)
            for nested_key in ("deployment_profile", "model_profile"):
                nested = value.get(nested_key)
                if isinstance(nested, dict):
                    records.append(nested)
    return records


def _fallback_manifest_payload(config: dict) -> dict | None:
    artifact_uri = _first_string(
        config,
        "export_artifact_uri",
        "artifact_uri",
        "checkpoint_uri",
        "model_uri",
        "onnx_artifact_uri",
        "torchscript_artifact_uri",
    )
    if not artifact_uri:
        return None
    artifact_format = _manifest_artifact_format_for_uri(artifact_uri)
    if not artifact_format:
        return None
    artifact_path: Path | None = None
    if artifact_uri.startswith("s3://"):
        destination = _downloaded_artifact_path(artifact_uri, artifact_format)
        try:
            artifact_path = validate_controlled_artifact_path(download_s3_uri(artifact_uri, destination), export_dir=destination.parent)
        except Exception:
            return None
    else:
        try:
            artifact_path = resolve_controlled_artifact_reference(artifact_uri)
        except ArtifactPathValidationError:
            return None
    metadata = _fallback_manifest_metadata(config)
    provenance = _export_provenance(config, _first_string(config, "export_job_id", "job_id"), artifact_format)
    metadata["provenance"] = _inline_manifest_provenance(provenance, artifact_format)
    artifact_bytes = artifact_path.stat().st_size if artifact_path.exists() else 0
    return {
        "schema_version": "champion_export_manifest_v1",
        "status": "created",
        "metadata": metadata,
        "artifacts": [
            {
                "format": artifact_format,
                "status": "created",
                "path": str(artifact_path),
                "uri": artifact_path.resolve().as_uri(),
                "bytes": artifact_bytes,
                "provenance": {
                    **_inline_manifest_provenance(provenance, artifact_format),
                    "artifact_bytes": artifact_bytes,
                },
            }
        ],
    }


def _fallback_manifest_metadata(config: dict) -> dict:
    for record in _manifest_metadata_records(config):
        metadata = record.get("metadata")
        if isinstance(metadata, dict) and metadata:
            return metadata
        model_profile = record.get("model_profile")
        if isinstance(model_profile, dict) and model_profile:
            return model_profile
    export_metadata = config.get("export_metadata") if isinstance(config.get("export_metadata"), dict) else {}
    return {
        "model": _first_string(config, "model", "model_name", "champion_model")
        or _first_string(export_metadata, "model", "model_name")
        or "mobilenet_v3_small",
        "class_labels": _class_names(config, export_metadata),
        "input_shape": config.get("sample_input_shape") or [1, 3, _positive_int(config.get("image_size"), 224), _positive_int(config.get("image_size"), 224)],
        "preprocessing": config.get("preprocessing") if isinstance(config.get("preprocessing"), dict) else {},
        "training_config": config.get("training_config") if isinstance(config.get("training_config"), dict) else config,
    }


def _inline_manifest_provenance(provenance: dict, artifact_format: str) -> dict:
    record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_by": "model-express-worker",
        "source": "controlled_legacy_manifest_fallback",
        "artifact_format": artifact_format,
        "validation_errors": [],
    }
    for key in ("source_job_id", "source_export_id", "export_job_id"):
        value = provenance.get(key)
        if value:
            record[key] = str(value)
    return record


def _manifest_artifact_format_for_uri(uri: str) -> str:
    normalized = unquote(urlparse(uri).path if urlparse(uri).scheme else uri).lower()
    if normalized.endswith(".onnx"):
        return "onnx"
    if normalized.endswith(".torchscript.pt") or normalized.endswith(".torchscript"):
        return "torchscript"
    if normalized.endswith(".pt") or normalized.endswith(".pth"):
        return "framework_native_checkpoint"
    if normalized.endswith(".safetensors"):
        return "safetensors"
    return ""


def _build_torchvision_model(
    *,
    model_name: str,
    class_count: int,
    pretrained: bool,
    freeze_backbone: bool,
    fine_tune_strategy: str,
    dropout: float,
):
    from torch import nn
    from torchvision import models

    normalized = model_name.lower()
    dropout = max(0.0, min(0.7, dropout))

    if "efficientnet_b2" in normalized:
        model = _torchvision_model(models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
        return model
    if "efficientnet_b1" in normalized:
        model = _torchvision_model(models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
        return model
    if "efficientnet" in normalized:
        model = _torchvision_model(models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
        return model
    if "resnet34" in normalized:
        model = _torchvision_model(models.resnet34, models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        _apply_transfer_strategy(model, "fc", freeze_backbone, fine_tune_strategy)
        model.fc = _classification_head(nn, in_features, class_count, dropout)
        return model
    if "resnet" in normalized:
        model = _torchvision_model(models.resnet18, models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        _apply_transfer_strategy(model, "fc", freeze_backbone, fine_tune_strategy)
        model.fc = _classification_head(nn, in_features, class_count, dropout)
        return model
    if "regnet_y_400mf" in normalized:
        model = _torchvision_model(models.regnet_y_400mf, models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        _apply_transfer_strategy(model, "fc", freeze_backbone, fine_tune_strategy)
        model.fc = _classification_head(nn, in_features, class_count, dropout)
        return model
    if "convnext_tiny" in normalized:
        model = _torchvision_model(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
        return model
    if "swin_t" in normalized:
        model = _torchvision_model(models.swin_t, models.Swin_T_Weights.DEFAULT if pretrained else None)
        in_features = model.head.in_features
        _apply_transfer_strategy(model, "head", freeze_backbone, fine_tune_strategy)
        model.head = _classification_head(nn, in_features, class_count, dropout)
        return model
    if "vit_b_16" in normalized:
        model = _torchvision_model(models.vit_b_16, models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        in_features = model.heads.head.in_features
        _apply_transfer_strategy(model, "heads", freeze_backbone, fine_tune_strategy)
        model.heads.head = _classification_head(nn, in_features, class_count, dropout)
        return model
    if "mobilenet_v3_large" in normalized:
        model = _torchvision_model(models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
        return model

    model = _torchvision_model(models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    in_features = model.classifier[-1].in_features
    _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
    _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
    return model


def _torchvision_model(factory, weights):
    try:
        return factory(weights=weights)
    except Exception:
        return factory(weights=None)


def _classification_head(nn, in_features: int, class_count: int, dropout: float = 0.0):
    head = nn.Linear(in_features, class_count)
    if dropout <= 0:
        return head
    return nn.Sequential(nn.Dropout(p=dropout), head)


def _replace_classifier_head(nn, classifier, in_features: int, class_count: int, dropout: float = 0.0) -> None:
    if len(classifier) >= 2 and isinstance(classifier[-2], nn.Dropout):
        classifier[-2].p = dropout
        classifier[-1] = nn.Linear(in_features, class_count)
        return
    classifier[-1] = _classification_head(nn, in_features, class_count, dropout)


def _apply_transfer_strategy(model, head_name: str, freeze_backbone: bool, fine_tune_strategy: str) -> None:
    if not freeze_backbone or fine_tune_strategy == "full":
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    if fine_tune_strategy == "last_block":
        children = list(model.children())
        if len(children) > 1:
            for parameter in children[-2].parameters():
                parameter.requires_grad = True
    head = getattr(model, head_name, None)
    if head is not None:
        for parameter in head.parameters():
            parameter.requires_grad = True


def _demo_image_path(config: dict, job_id: str) -> tuple[Path | None, str]:
    value = _first_string(config, "local_image_path", "image_path", "image_uri")
    if not value:
        return None, "image_uri or local_image_path is required"
    if value.startswith("data:image/"):
        return _inline_demo_image_path(value, job_id)
    if value.startswith("s3://"):
        destination = Path(os.getenv("WORKER_DEMO_IMAGE_ROOT", ".cache/demo_images")) / _safe_path_part(job_id) / Path(urlparse(value).path).name
        try:
            return download_s3_uri(value, destination), ""
        except Exception as exc:
            return None, str(exc)
    path = _path_from_uri_or_local(value)
    if path is None:
        return None, f"unsupported image URI: {value}"
    if not path.exists() or not path.is_file():
        return None, f"demo image not found: {path}"
    return path, ""


def _inline_demo_image_path(value: str, job_id: str) -> tuple[Path | None, str]:
    try:
        header, encoded = value.split(",", 1)
    except ValueError:
        return None, "inline image data URI is missing a payload"
    if ";base64" not in header.lower():
        return None, "inline image data URI must be base64 encoded"
    suffix = ".jpg"
    media_type = header.split(";", 1)[0].lower()
    if media_type.endswith("/png"):
        suffix = ".png"
    elif media_type.endswith("/webp"):
        suffix = ".webp"
    destination = Path(os.getenv("WORKER_DEMO_IMAGE_ROOT", ".cache/demo_images")) / _safe_path_part(job_id) / f"inline{suffix}"
    try:
        payload = base64.b64decode(encoded, validate=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    except Exception as exc:
        return None, f"inline image decode failed: {exc}"
    return destination, ""


def _path_from_uri_or_local(value: str) -> Path | None:
    if os.name == "nt" and len(value) > 2 and value[1] == ":":
        return Path(value)
    parsed = urlparse(value)
    if parsed.scheme == "file":
        path = unquote(parsed.path)
        if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
            path = path[1:]
        return Path(path)
    if parsed.scheme and parsed.scheme not in {"file"}:
        return None
    return Path(value)


def _export_result_payload(
    *,
    status: str,
    requested_format: str,
    manifest: dict,
    validation_errors: list[str],
) -> dict:
    created = _created_artifacts(manifest)
    artifact_uri = _file_uri(created[0]["path"]) if created else ""
    manifest_path = manifest.get("manifest_path")
    errors = list(dict.fromkeys(error for error in validation_errors if error))
    return {
        "status": status,
        "format": requested_format,
        "artifact_uri": artifact_uri,
        "manifest_uri": _file_uri(manifest_path) if manifest_path else "",
        "metadata": {
            "manifest": _public_manifest(manifest),
            "export_dir": str(Path(str(manifest_path)).parent) if manifest_path else "",
        },
        "validation_errors": errors,
        "error": "; ".join(errors),
    }


def _prediction_result_from_inference(payload: dict) -> dict:
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


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_positive_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    parsed = _positive_int(value, 0)
    return parsed if parsed > 0 else None


def _validated_exemplar_payload(pack: dict, caps: dict) -> dict:
    exemplars: list[dict] = []
    total_bytes = 0
    for item in pack.get("visual_exemplars", []):
        if not isinstance(item, dict):
            continue
        try:
            encoded_bytes = int(item.get("bytes", 0))
        except (TypeError, ValueError):
            continue
        if encoded_bytes < 0 or encoded_bytes > caps["max_image_bytes"]:
            continue
        if len(exemplars) >= caps["max_total_images"]:
            break
        if total_bytes + encoded_bytes > caps["max_total_bytes"]:
            break
        total_bytes += encoded_bytes
        exemplars.append(
            {
                "id": str(item.get("id", "")),
                "class_name": str(item.get("class_name", "")),
                "path": str(item.get("path", "")),
                "source_path": str(item.get("source_path", "")),
                "width": _positive_int(item.get("width"), 0),
                "height": _positive_int(item.get("height"), 0),
                "bytes": encoded_bytes,
                "mime_type": str(item.get("mime_type", "image/jpeg")),
            }
        )
    summary = pack.get("summary") if isinstance(pack.get("summary"), dict) else {}
    summary = {**summary, "image_count": len(exemplars), "total_bytes": total_bytes, **caps}
    return {
        "schema_version": "visual_exemplar_pack_v1",
        "status": str(pack.get("status", "created")),
        "visual_exemplars": exemplars,
        "demo_images": exemplars,
        "summary": summary,
        "profile_patch": {
            "visual_exemplars": exemplars,
            "demo_images": exemplars,
            "visual_exemplar_summary": summary,
        },
        "error_code": pack.get("error_code", ""),
        "error": pack.get("error", ""),
    }


def _exemplar_caps(config: dict) -> dict:
    return {
        "images_per_class": min(_positive_int(config.get("images_per_class"), 2), 4),
        "max_total_images": min(_positive_int(config.get("max_total_images"), 12), 24),
        "max_image_bytes": min(_positive_int(config.get("max_image_bytes"), 20_000), 100_000),
        "max_total_bytes": min(_positive_int(config.get("max_total_bytes"), 150_000), 1_500_000),
        "image_size": min(max(_positive_int(config.get("image_size"), 160), 32), 512),
        "quality": min(max(_positive_int(config.get("quality"), 75), 30), 95),
    }


def _created_artifacts(manifest: dict) -> list[dict]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return []
    return [
        artifact
        for artifact in artifacts
        if isinstance(artifact, dict) and artifact.get("status") == "created" and artifact.get("path")
    ]


def _artifact_errors(manifest: dict) -> list[str]:
    errors: list[str] = []
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        return errors
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get("status") == "created":
            continue
        error_code = str(artifact.get("error_code", "ARTIFACT_UNAVAILABLE"))
        error = str(artifact.get("error", "artifact was not created"))
        errors.append(f"{error_code}: {error}")
    return errors


def _error_manifest(
    export_dir: Path,
    requested_format: str,
    validation_errors: list[str],
    status: str = "failed",
    provenance: dict | None = None,
) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    artifact_format = _manifest_artifact_format_for_uri(f"model.{requested_format}") or requested_format
    provenance_record = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "generated_by": "model-express-worker",
        "source": "validation_error",
        "artifact_format": artifact_format,
        "validation_errors": [str(error) for error in validation_errors if error],
    }
    if provenance:
        for key in ("source_job_id", "source_export_id", "export_job_id"):
            value = provenance.get(key)
            if value:
                provenance_record[key] = str(value)
    manifest = {
        "schema_version": "champion_export_manifest_v1",
        "status": status,
        "metadata": {"format": requested_format, "provenance": provenance_record},
        "artifacts": [
            {
                "format": requested_format,
                "status": "failed" if status == "failed" else "skipped",
                "error_code": "VALIDATION_ERROR",
                "error": "; ".join(validation_errors),
                "provenance": {**provenance_record, "artifact_bytes": 0},
            }
        ],
    }
    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _public_manifest(manifest: dict) -> dict:
    public = dict(manifest)
    if "manifest_path" in public:
        public["manifest_path"] = str(public["manifest_path"])
    return public


def _file_uri(path_value: object) -> str:
    if not path_value:
        return ""
    return Path(str(path_value)).resolve().as_uri()


def _first_string(mapping: dict, *keys: str) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool_value(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _safe_path_part(value: str) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)[:120] or "unknown"
