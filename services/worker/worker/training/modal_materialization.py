from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath

from worker.training.modal_resources import callback_token, training_attempt_id
from worker.training.modal_runtime import (
    DATASET_MATERIALIZATION_ROOT,
    DEFAULT_MODAL_TRAINING_DATASET_CACHE_ROOT,
    TORCH_CACHE_ROOT,
    _bool,
    dataset_volume,
    torch_cache_volume,
    training_volume_mounts,
)


def _configure_storage_env(payload: dict) -> None:
    os.environ["S3_ENDPOINT_URL"] = payload["s3_endpoint_url"]
    os.environ["AWS_ACCESS_KEY_ID"] = payload["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = payload["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = payload["aws_default_region"]
    _configure_storage_scope_env(payload)
    os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_ROOT))
    _configure_callback_env(payload)


def _configure_storage_scope_env(payload: dict) -> None:
    scope = payload.get("storage_scope") if isinstance(payload.get("storage_scope"), dict) else {}
    if scope:
        token = str(scope.get("token") or "").strip()
        os.environ["MODEL_EXPRESS_STORAGE_SCOPE"] = json.dumps(scope, sort_keys=True)
        os.environ["MODEL_EXPRESS_REQUIRE_STORAGE_SCOPE"] = "true"
        _set_or_clear_env("MODEL_EXPRESS_STORAGE_SCOPE_TOKEN", token)
        return
    os.environ.pop("MODEL_EXPRESS_STORAGE_SCOPE", None)
    os.environ.pop("MODEL_EXPRESS_STORAGE_SCOPE_TOKEN", None)
    os.environ.pop("MODEL_EXPRESS_REQUIRE_STORAGE_SCOPE", None)


def _configure_callback_env(payload: dict) -> None:
    job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
    token = str(payload.get("callback_token") or "").strip() or callback_token(job)
    attempt_id = str(payload.get("training_attempt_id") or "").strip() or training_attempt_id(job)
    _set_or_clear_env("MODEL_EXPRESS_CALLBACK_TOKEN", token)
    _set_or_clear_env("MODEL_EXPRESS_TRAINING_ATTEMPT_ID", attempt_id)


def _set_or_clear_env(name: str, value: str) -> None:
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def _dataset_checksum(dataset: dict, config: dict | None = None) -> str:
    for key in ("checksum_sha256", "sha256", "checksum"):
        value = dataset.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = dataset.get("metadata")
    if isinstance(metadata, dict):
        for key in ("checksum_sha256", "sha256", "checksum"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(config, dict):
        for key in ("dataset_checksum_sha256", "checksum_sha256", "sha256", "checksum"):
            value = config.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        materialization = config.get("dataset_materialization")
        if isinstance(materialization, dict):
            for key in ("dataset_checksum_sha256", "checksum_sha256", "sha256", "checksum"):
                value = materialization.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            cache_key = materialization.get("dataset_cache_key")
            if isinstance(cache_key, str) and cache_key.startswith("sha256-"):
                return cache_key.removeprefix("sha256-").strip()
    return ""


def _reload_modal_dataset_volume() -> None:
    reload = getattr(dataset_volume, "reload", None)
    if callable(reload):
        reload()


def _commit_modal_dataset_volume() -> None:
    commit = getattr(dataset_volume, "commit", None)
    if callable(commit):
        commit()


def _reload_modal_torch_cache_volume() -> None:
    reload = getattr(torch_cache_volume, "reload", None)
    if callable(reload):
        reload()


def _commit_modal_torch_cache_volume() -> None:
    commit = getattr(torch_cache_volume, "commit", None)
    if callable(commit):
        commit()


def _modal_sync_torch_cache_commit_enabled() -> bool:
    return _bool(os.getenv("MODEL_EXPRESS_MODAL_SYNC_TORCH_CACHE_COMMIT"), default=False)


def _modal_training_dataset_cache_root() -> Path:
    return Path(
        os.getenv(
            "MODEL_EXPRESS_MODAL_TRAINING_DATASET_CACHE_ROOT",
            DEFAULT_MODAL_TRAINING_DATASET_CACHE_ROOT,
        )
    )


def _modal_dataset_cache_relationship_fields(
    materialization_cache_root: Path | PurePosixPath | str,
    *,
    materialization_scope: str,
    training_cache_root: Path | PurePosixPath | str | None = None,
) -> dict:
    materialization_root = _modal_cache_path_text(materialization_cache_root)
    prewarm_root = _modal_cache_path_text(DATASET_MATERIALIZATION_ROOT)
    training_root = _modal_cache_path_text(training_cache_root or _modal_training_dataset_cache_root())
    training_mount_paths = sorted(_modal_cache_path_text(path) for path in training_volume_mounts)
    prewarm_root_matches_training_root = prewarm_root == training_root
    prewarm_root_mounted_for_training = prewarm_root in training_mount_paths
    prewarm_reusable_for_training = prewarm_root_matches_training_root and prewarm_root_mounted_for_training
    if prewarm_reusable_for_training:
        reuse_status = "reusable_for_training"
        training_scope = "modal_dataset_volume"
    elif not prewarm_root_matches_training_root:
        reuse_status = "staging_only_root_mismatch"
        training_scope = "modal_training_local"
    else:
        reuse_status = "staging_only_training_mount_missing"
        training_scope = "modal_training_local"
    return {
        "dataset_materialization_cache_root": materialization_root,
        "dataset_materialization_cache_scope": materialization_scope,
        "dataset_prewarm_cache_root": prewarm_root,
        "dataset_training_cache_root": training_root,
        "dataset_training_cache_scope": training_scope,
        "dataset_training_volume_mounts": training_mount_paths,
        "dataset_prewarm_root_matches_training_root": prewarm_root_matches_training_root,
        "dataset_prewarm_root_mounted_for_training": prewarm_root_mounted_for_training,
        "dataset_prewarm_reusable_for_training": prewarm_reusable_for_training,
        "dataset_prewarm_reuse_status": reuse_status,
    }


def _modal_cache_path_text(path: Path | PurePosixPath | str) -> str:
    return str(path).replace("\\", "/").rstrip("/") or "/"
