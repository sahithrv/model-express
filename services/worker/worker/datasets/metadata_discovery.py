from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path, PureWindowsPath


DEFAULT_MAX_INVENTORY_FILES = 20_000
DEFAULT_MAX_FILES_SEEN = 20_000
DEFAULT_MAX_SOURCES = 200
DEFAULT_MAX_SOURCE_BYTES = 1_000_000
DEFAULT_MAX_TOTAL_SOURCE_BYTES = 5_000_000

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_METADATA_DIR_NAMES = {
    "annotation",
    "annotations",
    "attribute",
    "attributes",
    "bbox",
    "bboxes",
    "boxes",
    "keypoint",
    "keypoints",
    "label",
    "labels",
    "landmark",
    "landmarks",
    "manifest",
    "manifests",
    "meta",
    "metadata",
    "part",
    "parts",
    "split",
    "splits",
}
_CSV_MANIFEST_NAMES = {
    "annotations.csv",
    "classes.csv",
    "labels.csv",
    "manifest.csv",
    "metadata.csv",
}
_SPLIT_FILE_NAMES = {
    "split.txt",
    "splits.txt",
    "test.txt",
    "train.txt",
    "val.txt",
    "valid.txt",
    "validation.txt",
}
_CUB_SIDECAR_NAMES = {
    "bounding_boxes.txt",
    "classes.txt",
    "image_class_labels.txt",
    "images.txt",
    "part_locs.txt",
    "parts.txt",
    "train_test_split.txt",
}
_UNSUPPORTED_METADATA_SUFFIXES = {
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".yaml",
    ".yml",
}


def build_metadata_import_payload(
    dataset_dir: Path,
    *,
    max_inventory_files: int | None = None,
    max_sources: int | None = None,
    max_source_bytes: int | None = None,
    max_total_source_bytes: int | None = None,
    max_files_seen: int | None = None,
) -> dict:
    """Build the bounded worker handoff for backend-owned metadata parsing."""
    root = Path(dataset_dir)
    caps = {
        "max_inventory_files": _positive_int(
            max_inventory_files,
            "MODEL_EXPRESS_METADATA_MAX_INVENTORY_FILES",
            DEFAULT_MAX_INVENTORY_FILES,
        ),
        "max_sources": _positive_int(
            max_sources,
            "MODEL_EXPRESS_METADATA_MAX_SOURCES",
            DEFAULT_MAX_SOURCES,
        ),
        "max_source_bytes": _positive_int(
            max_source_bytes,
            "MODEL_EXPRESS_METADATA_MAX_SOURCE_BYTES",
            DEFAULT_MAX_SOURCE_BYTES,
        ),
        "max_total_source_bytes": _positive_int(
            max_total_source_bytes,
            "MODEL_EXPRESS_METADATA_MAX_TOTAL_SOURCE_BYTES",
            DEFAULT_MAX_TOTAL_SOURCE_BYTES,
        ),
        "max_files_seen": _positive_int(
            max_files_seen,
            "MODEL_EXPRESS_PROFILE_MAX_METADATA_FILES",
            DEFAULT_MAX_FILES_SEEN,
        ),
    }
    inventory_files: list[dict] = []
    sources: list[dict] = []
    warnings: list[dict] = []
    seen_files = 0
    skipped_unsafe_paths = 0
    skipped_source_count = 0
    inlined_source_bytes = 0

    scan_truncated = False
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        seen_files += 1
        if seen_files > caps["max_files_seen"]:
            scan_truncated = True
            break
        try:
            relative_path = safe_relative_path(root, path)
        except ValueError:
            skipped_unsafe_paths += 1
            continue

        try:
            size_bytes = path.stat().st_size
        except OSError:
            continue

        declared_format = declared_metadata_format(relative_path)
        checksum = ""
        source: dict | None = None
        if declared_format:
            if len(sources) >= caps["max_sources"]:
                skipped_source_count += 1
            else:
                checksum = _sha256_file(path)
                source = {
                    "relative_path": relative_path,
                    "declared_format": declared_format,
                    "size_bytes": size_bytes,
                    "checksum_sha256": checksum,
                }
                source_warnings = _source_warnings(declared_format)
                if size_bytes > caps["max_source_bytes"]:
                    source_warnings.append("content_skipped_source_size_cap")
                elif inlined_source_bytes + size_bytes > caps["max_total_source_bytes"]:
                    source_warnings.append("content_skipped_total_size_cap")
                elif declared_format != "unsupported":
                    source["content_base64"] = base64.b64encode(path.read_bytes()).decode("ascii")
                    inlined_source_bytes += size_bytes
                if source_warnings:
                    source["warnings"] = source_warnings
                sources.append(source)

        if len(inventory_files) < caps["max_inventory_files"]:
            inventory_record = {
                "relative_path": relative_path,
                "size_bytes": size_bytes,
            }
            if checksum:
                inventory_record["checksum_sha256"] = checksum
            inventory_files.append(inventory_record)

    if skipped_unsafe_paths:
        warnings.append(
            {
                "code": "unsafe_relative_paths_skipped",
                "count": skipped_unsafe_paths,
                "message": "One or more files were skipped because their dataset-relative paths were unsafe.",
            }
        )
    if skipped_source_count:
        warnings.append(
            {
                "code": "metadata_source_count_cap",
                "count": skipped_source_count,
                "message": "One or more metadata candidates were not sent because the source count cap was reached.",
            }
        )
    if scan_truncated:
        warnings.append(
            {
                "code": "metadata_discovery_file_scan_cap",
                "count": caps["max_files_seen"],
                "message": "Metadata discovery stopped at the worker file scan cap.",
            }
        )
    if seen_files > len(inventory_files) or scan_truncated:
        warnings.append(
            {
                "code": "inventory_file_count_cap",
                "count": max(0, seen_files - len(inventory_files)),
                "message": "The dataset file inventory was truncated at the worker cap.",
            }
        )
    if any(source.get("declared_format") == "unsupported" for source in sources):
        warnings.append(
            {
                "code": "unsupported_metadata_candidates",
                "message": "Some metadata-like files were sent as unsupported candidates for backend warning/reporting.",
            }
        )

    return {
        "strict_mode": False,
        "sources": sources,
        "inventory": {
            "files": inventory_files,
            "file_count_seen": seen_files,
            "truncated": seen_files > len(inventory_files) or scan_truncated,
        },
        "worker_discovery": {
            "schema_version": "dataset_metadata_discovery_v1",
            "caps": caps,
            "inlined_source_bytes": inlined_source_bytes,
        },
        "warnings": warnings,
    }


def declared_metadata_format(relative_path: str) -> str | None:
    if not is_safe_relative_path(relative_path):
        return None
    parts = [part.lower() for part in relative_path.split("/")]
    name = parts[-1]
    suffix = Path(name).suffix.lower()
    metadataish = _metadataish_path(parts)

    if name in _CUB_SIDECAR_NAMES:
        return "cub_sidecars"
    if name in _SPLIT_FILE_NAMES:
        return "split_file"
    if name in _CSV_MANIFEST_NAMES or (suffix == ".csv" and metadataish):
        return "csv_manifest"
    if suffix == ".xml" and metadataish:
        return "pascal_voc_xml"
    if suffix == ".json" and metadataish:
        return "annotation_json" if any("annotation" in part for part in parts) else "unsupported"
    if suffix in _UNSUPPORTED_METADATA_SUFFIXES and metadataish:
        return "unsupported"
    if suffix == ".txt" and metadataish and suffix not in _IMAGE_SUFFIXES:
        return "unsupported"
    return None


def is_safe_relative_path(relative_path: str) -> bool:
    value = str(relative_path or "")
    if not value or "\\" in value or "\x00" in value:
        return False
    if any(ord(char) < 32 for char in value):
        return False
    if value.startswith("/"):
        return False
    windows_path = PureWindowsPath(value)
    if windows_path.drive or windows_path.is_absolute():
        return False
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    return True


def safe_relative_path(root: Path, path: Path) -> str:
    root_resolved = root.resolve(strict=True)
    path_resolved = path.resolve(strict=True)
    try:
        relative = path_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(f"path is outside dataset root: {path}") from exc
    relative_path = relative.as_posix()
    if not is_safe_relative_path(relative_path):
        raise ValueError(f"unsafe dataset-relative path: {relative_path!r}")
    return relative_path


def _metadataish_path(parts: list[str]) -> bool:
    if any(part in _METADATA_DIR_NAMES for part in parts[:-1]):
        return True
    name = parts[-1]
    return any(
        token in name
        for token in (
            "annotation",
            "attribute",
            "bbox",
            "box",
            "keypoint",
            "label",
            "landmark",
            "manifest",
            "metadata",
            "part",
            "split",
        )
    )


def _source_warnings(declared_format: str) -> list[str]:
    if declared_format == "unsupported":
        return ["unsupported_metadata_format"]
    return []


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: int | None, env_name: str, default: int) -> int:
    if value is None:
        raw_value = os.getenv(env_name, "").strip()
        if raw_value:
            try:
                value = int(raw_value)
            except ValueError:
                value = default
        else:
            value = default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
