from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path


CACHE_ROOT = Path(".cache/datasets")
SCRATCH_ROOT = Path(tempfile.gettempdir()) / "model-express-worker-datasets"


def dataset_cache_root(cache_root: Path | str | None = None) -> Path:
    if cache_root is not None:
        return Path(cache_root)
    return Path(os.getenv("MODEL_EXPRESS_DATASET_CACHE_ROOT", str(CACHE_ROOT)))


def dataset_cache_dir(dataset_id: str, cache_root: Path | str | None = None) -> Path:
    return dataset_cache_root(cache_root) / _safe_dataset_id(dataset_id)


def dataset_archive_path(dataset_id: str, cache_root: Path | str | None = None) -> Path:
    return dataset_cache_dir(dataset_id, cache_root) / "archive.zip"


def dataset_extract_dir(dataset_id: str, cache_root: Path | str | None = None) -> Path:
    return dataset_cache_dir(dataset_id, cache_root) / "extracted"


def extract_dataset_archive(
    archive_path: Path,
    dataset_id: str,
    cache_root: Path | str | None = None,
) -> Path:
    extract_dir = dataset_extract_dir(dataset_id, cache_root)
    cache_dir = dataset_cache_dir(dataset_id, cache_root)

    if extract_dir.exists():
        return extract_dir

    temp_dir = cache_dir / "extracting"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(temp_dir)

    if extract_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
        return extract_dir
    try:
        temp_dir.replace(extract_dir)
    except PermissionError:
        if extract_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            return extract_dir
        raise
    return extract_dir


def cleanup_dataset_cache(dataset_id: str, cache_root: Path | str | None = None) -> None:
    shutil.rmtree(dataset_cache_dir(dataset_id, cache_root), ignore_errors=True)


def job_dataset_cache_root(job_id: str | None = None, cache_root: Path | str | None = None) -> Path:
    if cache_root is not None:
        return Path(cache_root)
    if should_persist_dataset_cache():
        return dataset_cache_root()
    configured_root = os.getenv("MODEL_EXPRESS_DATASET_CACHE_ROOT", "").strip()
    base_root = Path(configured_root) if configured_root else SCRATCH_ROOT
    safe_job_id = _safe_cache_component(job_id or str(os.getpid()))
    return base_root / "_jobs" / safe_job_id


def cleanup_job_dataset_cache(job_id: str | None = None, cache_root: Path | str | None = None) -> None:
    root = job_dataset_cache_root(job_id, cache_root)
    if should_persist_dataset_cache() and cache_root is None:
        return
    shutil.rmtree(root, ignore_errors=True)


def should_persist_dataset_cache() -> bool:
    value = os.getenv("MODEL_EXPRESS_PERSIST_DATASET_CACHE", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _safe_dataset_id(dataset_id: str) -> str:
    normalized = str(dataset_id).strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError("dataset_id is required for dataset cache paths")
    if "/" in normalized or "\\" in normalized:
        raise ValueError(f"dataset_id must not contain path separators: {dataset_id!r}")
    return normalized


def _safe_cache_component(value: str) -> str:
    normalized = str(value).strip().replace("\\", "_").replace("/", "_")
    normalized = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in normalized)
    return normalized.strip("._") or "worker"
