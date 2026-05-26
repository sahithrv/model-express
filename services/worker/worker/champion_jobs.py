from __future__ import annotations

import os
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

from worker.datasets.cache import dataset_archive_path, extract_dataset_archive
from worker.datasets.exemplars import generate_visual_exemplars
from worker.datasets.storage import download_s3_uri
from worker.exporting.artifacts import (
    produce_champion_export_artifacts,
    produce_existing_champion_export_manifest,
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
    model_name = _first_string(config, "model", "model_name", "champion_model") or "unknown_model"
    preprocessing = config.get("preprocessing") if isinstance(config.get("preprocessing"), dict) else {}
    model_profile = config.get("model_profile") if isinstance(config.get("model_profile"), dict) else {}
    training_config = config.get("training_config") if isinstance(config.get("training_config"), dict) else config

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

    source_artifact = _existing_artifact_source_path(config, requested_format)
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
        )
    elif requested_format in HELPER_EXPORT_FORMATS:
        manifest = produce_champion_export_artifacts(
            export_dir=export_dir,
            model_name=model_name,
            class_names=class_names,
            image_size=image_size,
            model=None,
            preprocessing=preprocessing,
            model_profile=model_profile,
            training_config=training_config,
            formats=(HELPER_EXPORT_FORMATS[requested_format],),
            sample_input_shape=config.get("sample_input_shape"),
        )
    else:
        validation_errors.append(
            "safetensors export requires an existing worker-visible safetensors artifact; no source artifact was provided"
        )
        manifest = _error_manifest(export_dir, requested_format, validation_errors, status="pending_dependencies")

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
    image_path, image_error = _demo_image_path(config, job_id)
    image_uri = _first_string(config, "image_uri", "image_path", "local_image_path") or ""
    true_label = _first_string(config, "true_label", "label", "class_name")
    image_metadata = config.get("image_metadata") if isinstance(config.get("image_metadata"), dict) else {}

    if manifest_path is None:
        payload = {
            "status": "RUNTIME_UNAVAILABLE",
            "error_code": "MANIFEST_NOT_CONFIGURED",
            "error": "No worker-owned export manifest path was supplied or found.",
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
            image_path=image_path,
            top_k=_positive_int(config.get("top_k"), 5),
            true_label=true_label,
        )
        payload = _prediction_result_from_inference(inference)

    payload.update(
        {
            "image_uri": image_uri,
            "image_id": _first_string(config, "image_id") or payload.get("image_id", ""),
            "true_label": true_label or payload.get("true_label", ""),
            "image_metadata": image_metadata,
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
    archive_path = dataset_archive_path(dataset_id)
    download_s3_uri(dataset["storage_uri"], archive_path)
    dataset_dir = extract_dataset_archive(archive_path, dataset_id)
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


def _existing_artifact_source_path(config: dict, requested_format: str) -> Path | None:
    format_keys = {
        "onnx": ("onnx_path", "onnx_artifact_path"),
        "torchscript": ("torchscript_path", "torchscript_artifact_path"),
        "pytorch": ("checkpoint_path", "pytorch_path", "model_path"),
        "safetensors": ("safetensors_path", "safetensors_artifact_path"),
    }
    keys = (*format_keys.get(requested_format, ()), "artifact_path", "local_artifact_path")
    for key in keys:
        value = _first_string(config, key)
        if not value:
            continue
        path = _path_from_uri_or_local(value)
        if path is not None:
            return path
    return None


def _manifest_path(config: dict) -> Path | None:
    for key in ("manifest_path", "export_manifest_path", "local_manifest_path"):
        value = _first_string(config, key)
        if value:
            path = _path_from_uri_or_local(value)
            if path is not None:
                return path
    champion_job_id = _first_string(config, "champion_job_id", "source_job_id", "training_job_id")
    if not champion_job_id:
        return None
    base = _export_root() / _safe_path_part(champion_job_id)
    matches = sorted(base.glob("**/manifest.json")) if base.exists() else []
    return matches[0] if matches else None


def _demo_image_path(config: dict, job_id: str) -> tuple[Path | None, str]:
    value = _first_string(config, "local_image_path", "image_path", "image_uri")
    if not value:
        return None, "image_uri or local_image_path is required"
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
        return {
            "status": "SUCCEEDED",
            "predicted_label": payload.get("predicted_label", ""),
            "confidence": payload.get("confidence", 0.0),
            "top_k": payload.get("top_k", []),
            "latency_ms": payload.get("latency_ms", 0.0),
            "correct": payload.get("correct"),
            "runtime": payload.get("runtime", "torchscript"),
        }
    error_code = str(payload.get("error_code", "INFERENCE_UNAVAILABLE"))
    status = "RUNTIME_UNAVAILABLE"
    if error_code in {"INFERENCE_FAILED"}:
        status = "FAILED"
    return {
        "status": status,
        "error_code": error_code,
        "error": payload.get("error", ""),
        "predicted_label": "",
        "confidence": 0.0,
        "top_k": [],
        "latency_ms": 0.0,
    }


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
) -> dict:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "champion_export_manifest_v1",
        "status": status,
        "metadata": {"format": requested_format},
        "artifacts": [
            {
                "format": requested_format,
                "status": "failed" if status == "failed" else "skipped",
                "error_code": "VALIDATION_ERROR",
                "error": "; ".join(validation_errors),
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


def _safe_path_part(value: str) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)[:120] or "unknown"
