from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path


CACHE_ROOT = Path(".cache/datasets")


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

    if extract_dir.exists():
        return extract_dir
    
    temp_dir = dataset_cache_dir(dataset_id, cache_root) / "extracting"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(temp_dir)
    
    temp_dir.rename(extract_dir)
    return extract_dir


def cleanup_dataset_cache(dataset_id: str, cache_root: Path | str | None = None) -> None:
    shutil.rmtree(dataset_cache_dir(dataset_id, cache_root), ignore_errors=True)


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
