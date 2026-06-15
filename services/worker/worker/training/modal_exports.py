from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse


def _upload_manifest_artifacts(manifest: dict, remote_base: str, upload_file_to_s3_uri) -> tuple[dict, list[dict]]:
    public_manifest = json.loads(json.dumps(manifest))
    artifact_uris: list[dict] = []
    artifacts = public_manifest.get("artifacts") if isinstance(public_manifest.get("artifacts"), list) else []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("status") != "created":
            continue
        path = Path(str(artifact.get("path") or ""))
        if not path.exists() or not path.is_file():
            artifact["status"] = "failed"
            artifact["error_code"] = "ARTIFACT_NOT_FOUND"
            artifact["error"] = f"Created artifact disappeared before upload: {path}"
            continue
        remote_uri = f"{remote_base}/{path.name}"
        upload_file_to_s3_uri(path, remote_uri)
        if str(artifact.get("format") or "") == "onnx":
            artifact["external_data"] = _upload_onnx_external_data_files(
                artifact,
                remote_base,
                path.parent,
                upload_file_to_s3_uri,
            )
        artifact["path"] = remote_uri
        artifact["uri"] = remote_uri
        artifact_uris.append({"format": str(artifact.get("format") or ""), "uri": remote_uri})
    public_manifest.pop("manifest_path", None)
    if isinstance(public_manifest.get("metadata"), dict):
        onnx_artifact = next((item for item in artifact_uris if item["format"] == "onnx"), None)
        public_manifest["metadata"]["artifact_uri"] = onnx_artifact["uri"] if onnx_artifact else ""
    return public_manifest, artifact_uris


def _manifest_preprocessing_latency_ms(manifest: dict) -> float:
    try:
        from worker.exporting.preprocessing import preprocessing_latency_p95_ms
    except Exception:
        return 0.0
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    profile = metadata.get("preprocessing_latency_profile") if isinstance(metadata, dict) else {}
    return round(preprocessing_latency_p95_ms(profile), 3)


def _upload_onnx_external_data_files(
    artifact: dict,
    remote_base: str,
    artifact_dir: Path,
    upload_file_to_s3_uri,
) -> list[dict]:
    external_data = artifact.get("external_data") if isinstance(artifact.get("external_data"), list) else []
    uploaded: list[dict] = []
    for item in external_data:
        if not isinstance(item, dict):
            continue
        relative_path = _safe_manifest_relative_path(str(item.get("path") or ""))
        local_path = Path(str(item.get("artifact_path") or ""))
        if not local_path.is_absolute():
            local_path = artifact_dir / (relative_path or local_path)
        if not relative_path:
            relative_path = _safe_manifest_relative_path(local_path.name)
        if not local_path.exists() or not local_path.is_file() or not relative_path:
            continue
        remote_uri = f"{remote_base}/{relative_path}"
        upload_file_to_s3_uri(local_path, remote_uri)
        uploaded.append(
            {
                "path": relative_path,
                "uri": remote_uri,
                "artifact_path": remote_uri,
                "bytes": local_path.stat().st_size,
            }
        )
    return uploaded


def _safe_manifest_relative_path(value: str) -> str:
    parts = []
    for part in str(value).replace("\\", "/").split("/"):
        if part in {"", ".", ".."}:
            continue
        parts.append(part)
    return "/".join(parts)


def _artifact_remote_base_uri(dataset: dict, job_id: str) -> str:
    bucket = os.getenv("MODEL_EXPRESS_ARTIFACT_BUCKET", "").strip()
    dataset_uri = str(dataset.get("storage_uri") or "")
    if not bucket:
        parsed = urlparse(dataset_uri)
        bucket = parsed.netloc if parsed.scheme == "s3" else "model-express"
    prefix = os.getenv("MODEL_EXPRESS_ARTIFACT_PREFIX", "model-express/artifacts").strip("/")
    return f"s3://{bucket}/{prefix}/{_safe_path_part(job_id)}"


def _artifact_errors(manifest: dict) -> list[str]:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    errors = []
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("status") != "created":
            error_code = str(artifact.get("error_code") or artifact.get("status") or "EXPORT_UNAVAILABLE")
            error = str(artifact.get("error") or "")
            errors.append(f"{error_code}: {error}".strip())
    return errors


def _export_error(error_code: str, error: str) -> dict:
    return {
        "status": "FAILED",
        "format": "onnx",
        "artifact_uri": "",
        "onnx_artifact_uri": "",
        "manifest_uri": "",
        "manifest": {},
        "validation_errors": [f"{error_code}: {error}"],
    }


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return safe or "artifact"
