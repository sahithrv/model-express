from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


CACHE_ROOT = Path(".cache/datasets")

def dataset_cache_dir(dataset_id: str) -> Path:
    return CACHE_ROOT / dataset_id

def dataset_archive_path(dataset_id: str) -> Path:
    return dataset_cache_dir(dataset_id) / "archive.zip"

def dataset_extract_dir(dataset_id: str) -> Path:
    return dataset_cache_dir(dataset_id) / "extracted"

def extract_dataset_archive(archive_path: Path, dataset_id: str) -> Path:
    extract_dir = dataset_extract_dir(dataset_id)

    if extract_dir.exists():
        return extract_dir
    
    temp_dir = dataset_cache_dir(dataset_id) / "extracting"
    if temp_dir.exists():
        shutil.rmtree(temp_dir) #recursively delete the temp_dir
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(temp_dir)
    
    temp_dir.rename(extract_dir)
    return extract_dir