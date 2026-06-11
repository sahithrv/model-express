from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml
except Exception:  # pragma: no cover - pyyaml is a worker dependency, but keep imports tolerant.
    yaml = None


SHARD_SCHEMA_VERSION = "dataset_shards.v1"
SHARD_FORMAT = "tar"
SHARD_MANIFEST_NAME = "manifest.json"
SHARD_ARTIFACT_DIR = "artifacts"
DEFAULT_MAX_SHARD_BYTES = 512 * 1024 * 1024
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_CONFIG_FILE_NAMES = {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml"}
YOLO_SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}
COMMON_IMAGE_ROOT_NAMES = {"images", "image", "imgs", "img", "jpegimages", "data"}


class ShardMaterializationError(RuntimeError):
    pass


@dataclass(frozen=True)
class _ShardEntry:
    source_path: Path
    relative_path: str
    kind: str
    split: str
    class_name: str | None = None


def dataset_shards_enabled() -> bool:
    value = os.getenv("MODEL_EXPRESS_DATASET_SHARDS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def shard_manifest_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "shards" / SHARD_MANIFEST_NAME


def create_shard_artifacts(
    *,
    dataset_dir: Path,
    artifact_root: Path,
    dataset_checksum: str | None,
    cache_key: str,
    storage_uri: str,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict:
    dataset_dir = Path(dataset_dir)
    artifact_root = Path(artifact_root)
    temp_root = artifact_root.with_name(f"{artifact_root.name}.building-{os.getpid()}-{os.urandom(4).hex()}")
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        yolo_config_path = _find_yolo_data_config(dataset_dir)
        if yolo_config_path is not None:
            entries, metadata = _yolo_shard_entries(dataset_dir, yolo_config_path, temp_root)
            task_type = "object_detection"
        else:
            entries, metadata = _classification_shard_entries(dataset_dir)
            task_type = "image_classification"

        shards = _write_shards(entries, temp_root / SHARD_ARTIFACT_DIR, max_shard_bytes=max_shard_bytes)
        manifest = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "dataset_checksum": _normalized_checksum(dataset_checksum),
            "task_type": task_type,
            "shard_format": SHARD_FORMAT,
            "cache_key": str(cache_key),
            "source_materialization": {
                "storage_uri": storage_uri,
                "source": "uploaded_zip",
                "source_dir": str(dataset_dir),
            },
            "file_counts": _file_counts(entries),
            "split_counts": _split_counts(entries),
            "class_counts": metadata.get("class_counts", {}),
            "split_class_counts": metadata.get("split_class_counts", {}),
            "object_counts": metadata.get("object_counts", {}),
            "yolo": metadata.get("yolo", {}),
            "shards": shards,
        }
        (temp_root / SHARD_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        temp_root.replace(artifact_root)
        return manifest
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def materialize_shard_artifacts(
    *,
    manifest_path: Path,
    output_dir: Path,
    dataset_checksum: str | None = None,
) -> dict:
    manifest_path = Path(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ShardMaterializationError(f"Missing dataset shard manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ShardMaterializationError(f"Invalid dataset shard manifest JSON: {manifest_path}") from exc

    _validate_manifest_header(manifest, dataset_checksum=dataset_checksum)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_paths = {
        str(file_record.get("relative_path") or "")
        for shard in manifest.get("shards", [])
        for file_record in shard.get("files", [])
    }
    expected_paths.discard("")
    extracted_paths: set[str] = set()
    for shard in manifest.get("shards", []):
        shard_path = _manifest_relative_path(manifest_path.parent, shard.get("path"))
        _validate_shard_file(shard_path, shard)
        extracted_paths.update(_extract_tar_shard(shard_path, output_dir, expected_paths))

    missing = expected_paths - extracted_paths
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ShardMaterializationError(f"Dataset shard extraction missed {len(missing)} files: {sample}")

    _validate_materialized_counts(output_dir, manifest)
    if manifest.get("task_type") == "object_detection":
        _validate_yolo_materialization(output_dir, manifest)
    return manifest


def _classification_shard_entries(dataset_dir: Path) -> tuple[list[_ShardEntry], dict]:
    entries: list[_ShardEntry] = []
    class_counts: dict[str, int] = defaultdict(int)
    split_class_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for path in _iter_dataset_files(dataset_dir):
        relative_path = _relative_path(dataset_dir, path)
        kind = "metadata"
        split = "unsplit"
        class_name: str | None = None
        if path.suffix.lower() in IMAGE_SUFFIXES:
            split, class_name = _classification_split_and_class(Path(relative_path))
            kind = "image"
            if class_name:
                class_counts[class_name] += 1
                split_class_counts[split][class_name] += 1
        entries.append(
            _ShardEntry(
                source_path=path,
                relative_path=relative_path,
                kind=kind,
                split=split,
                class_name=class_name,
            )
        )
    return entries, {
        "class_counts": dict(sorted(class_counts.items())),
        "split_class_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(split_class_counts.items())
        },
    }


def _yolo_shard_entries(
    dataset_dir: Path,
    config_path: Path,
    temp_root: Path,
) -> tuple[list[_ShardEntry], dict]:
    config = _load_yolo_config(config_path)
    names = _yolo_names(config)
    nc = int(config.get("nc") or len(names))
    split_images = {
        split: _resolve_yolo_split_images(config_path, config, split)
        for split in YOLO_SPLITS
        if _yolo_split_declared(config, split)
    }
    if "train" not in split_images or "val" not in split_images:
        raise ShardMaterializationError(
            f"YOLO shard artifacts require train and val splits in {config_path}"
        )

    entries: list[_ShardEntry] = []
    object_counts: dict[str, dict[str, int]] = {}
    label_counts: dict[str, int] = {}
    used_targets: set[str] = set()
    for split, image_paths in split_images.items():
        split_object_counts: dict[str, int] = defaultdict(int)
        label_counts[split] = 0
        image_root = _common_yolo_image_root(image_paths, split)
        for image_path in image_paths:
            image_rel = _yolo_target_relative_path(
                image_path,
                image_root=image_root,
                fallback_root=config_path.parent,
            )
            image_target = _unique_target_path(
                f"images/{split}/{image_rel.as_posix()}",
                used_targets,
            )
            entries.append(
                _ShardEntry(
                    source_path=image_path,
                    relative_path=image_target,
                    kind="image",
                    split=split,
                )
            )
            label_path = _yolo_label_path_for_image(image_path)
            label_target = str(Path(image_target.replace("images/", "labels/", 1)).with_suffix(".txt")).replace("\\", "/")
            if label_path is not None and label_path.is_file():
                label_counts[split] += 1
                entries.append(
                    _ShardEntry(
                        source_path=label_path,
                        relative_path=_unique_target_path(label_target, used_targets),
                        kind="label",
                        split=split,
                    )
                )
                for class_id, count in _yolo_label_class_counts(label_path).items():
                    split_object_counts[str(class_id)] += count
            else:
                empty_label = _empty_label_file(temp_root, split, Path(label_target).name)
                entries.append(
                    _ShardEntry(
                        source_path=empty_label,
                        relative_path=_unique_target_path(label_target, used_targets),
                        kind="label",
                        split=split,
                    )
                )
        object_counts[split] = dict(sorted(split_object_counts.items(), key=lambda item: int(item[0])))

    data_yaml = temp_root / "generated-data.yaml"
    _write_yolo_data_yaml(
        data_yaml,
        {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            **({"test": "images/test"} if "test" in split_images else {}),
            "nc": nc,
            "names": names,
        },
    )
    entries.append(
        _ShardEntry(
            source_path=data_yaml,
            relative_path="data.yaml",
            kind="metadata",
            split="metadata",
        )
    )
    return entries, {
        "object_counts": object_counts,
        "yolo": {
            "data_yaml": "data.yaml",
            "source_data_yaml": _relative_path(dataset_dir, config_path),
            "names": names,
            "nc": nc,
            "split_image_counts": {split: len(paths) for split, paths in split_images.items()},
            "split_label_file_counts": label_counts,
        },
    }


def _write_shards(
    entries: list[_ShardEntry],
    shard_dir: Path,
    *,
    max_shard_bytes: int,
) -> list[dict]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[_ShardEntry]] = defaultdict(list)
    for entry in entries:
        group = (entry.split or "unsplit", entry.class_name or entry.kind)
        groups[group].append(entry)

    shards: list[dict] = []
    for group_index, ((split, class_or_kind), group_entries) in enumerate(sorted(groups.items())):
        current: list[_ShardEntry] = []
        current_bytes = 0
        part = 0
        for entry in sorted(group_entries, key=lambda item: item.relative_path):
            size = _path_size(entry.source_path)
            if current and current_bytes + size > max_shard_bytes:
                shards.append(_write_one_shard(shard_dir, split, class_or_kind, group_index, part, current))
                current = []
                current_bytes = 0
                part += 1
            current.append(entry)
            current_bytes += size
        if current:
            shards.append(_write_one_shard(shard_dir, split, class_or_kind, group_index, part, current))
    return shards


def _write_one_shard(
    shard_dir: Path,
    split: str,
    class_or_kind: str,
    group_index: int,
    part: int,
    entries: list[_ShardEntry],
) -> dict:
    stem = _safe_shard_stem(f"{split}-{class_or_kind}-{group_index:03d}-{part:03d}")
    shard_path = shard_dir / f"{stem}.tar"
    with tarfile.open(shard_path, "w") as archive:
        for entry in entries:
            archive.add(entry.source_path, arcname=entry.relative_path, recursive=False)
    return {
        "name": shard_path.name,
        "path": f"{SHARD_ARTIFACT_DIR}/{shard_path.name}",
        "sha256": _sha256_file(shard_path),
        "bytes": _path_size(shard_path),
        "file_count": len(entries),
        "split": split,
        "class_or_kind": class_or_kind,
        "files": [
            {
                "relative_path": entry.relative_path,
                "kind": entry.kind,
                "split": entry.split,
                **({"class_name": entry.class_name} if entry.class_name else {}),
            }
            for entry in entries
        ],
    }


def _extract_tar_shard(shard_path: Path, output_dir: Path, expected_paths: set[str]) -> set[str]:
    extracted: set[str] = set()
    try:
        archive = tarfile.open(shard_path, "r")
    except tarfile.TarError as exc:
        raise ShardMaterializationError(f"Invalid dataset shard tar: {shard_path}") from exc
    with archive:
        for member in archive.getmembers():
            relative_name = _safe_member_name(member.name)
            if relative_name not in expected_paths:
                raise ShardMaterializationError(
                    f"Dataset shard {shard_path} contains unexpected file: {relative_name}"
                )
            target = output_dir / relative_name
            _ensure_child_path(output_dir, target)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ShardMaterializationError(
                    f"Dataset shard {shard_path} contains unsupported member: {relative_name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ShardMaterializationError(
                    f"Dataset shard {shard_path} could not read member: {relative_name}"
                )
            with source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.add(relative_name)
    return extracted


def _validate_manifest_header(manifest: dict, *, dataset_checksum: str | None) -> None:
    if manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ShardMaterializationError(
            f"Unsupported dataset shard schema version: {manifest.get('schema_version')!r}"
        )
    if manifest.get("shard_format") != SHARD_FORMAT:
        raise ShardMaterializationError(
            f"Unsupported dataset shard format: {manifest.get('shard_format')!r}"
        )
    expected_checksum = _normalized_checksum(dataset_checksum)
    manifest_checksum = _normalized_checksum(manifest.get("dataset_checksum"))
    if expected_checksum and manifest_checksum and expected_checksum != manifest_checksum:
        raise ShardMaterializationError(
            "Dataset shard checksum does not match requested dataset checksum."
        )
    if not isinstance(manifest.get("shards"), list) or not manifest["shards"]:
        raise ShardMaterializationError("Dataset shard manifest does not list any shards.")


def _validate_shard_file(path: Path, shard: dict) -> None:
    if not path.is_file():
        raise ShardMaterializationError(f"Missing dataset shard artifact: {path}")
    expected_sha = str(shard.get("sha256") or "").strip().lower()
    actual_sha = _sha256_file(path)
    if expected_sha and actual_sha != expected_sha:
        raise ShardMaterializationError(f"Dataset shard checksum mismatch: {path}")


def _validate_materialized_counts(output_dir: Path, manifest: dict) -> None:
    expected_total = int((manifest.get("file_counts") or {}).get("total") or 0)
    actual_total = sum(1 for path in output_dir.rglob("*") if path.is_file())
    if expected_total != actual_total:
        raise ShardMaterializationError(
            f"Dataset shard materialized {actual_total} files, expected {expected_total}."
        )


def _validate_yolo_materialization(output_dir: Path, manifest: dict) -> None:
    yolo = manifest.get("yolo") if isinstance(manifest.get("yolo"), dict) else {}
    data_yaml = output_dir / str(yolo.get("data_yaml") or "data.yaml")
    if not data_yaml.is_file():
        raise ShardMaterializationError(f"YOLO shard materialization is missing {data_yaml.name}.")
    config = _load_yolo_config(data_yaml)
    names = _yolo_names(config)
    if not names and not config.get("nc"):
        raise ShardMaterializationError("YOLO shard materialization produced data.yaml without names/nc.")
    for split, expected_count in (yolo.get("split_image_counts") or {}).items():
        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split
        images = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.rglob("*.txt")) if label_dir.is_dir() else []
        if len(images) != int(expected_count):
            raise ShardMaterializationError(
                f"YOLO shard materialized {len(images)} {split} images, expected {expected_count}."
            )
        missing_labels = [
            image
            for image in images
            if not (label_dir / image.relative_to(image_dir)).with_suffix(".txt").is_file()
        ]
        if missing_labels:
            sample = ", ".join(path.name for path in missing_labels[:5])
            raise ShardMaterializationError(f"YOLO shard materialization has unpaired labels: {sample}")
        expected_labels = int((yolo.get("split_label_file_counts") or {}).get(split) or len(images))
        if labels and len(labels) < expected_labels:
            raise ShardMaterializationError(
                f"YOLO shard materialized {len(labels)} {split} label files, expected {expected_labels}."
            )


def _file_counts(entries: Iterable[_ShardEntry]) -> dict:
    counts = {"total": 0, "images": 0, "labels": 0, "metadata": 0}
    for entry in entries:
        counts["total"] += 1
        if entry.kind == "image":
            counts["images"] += 1
        elif entry.kind == "label":
            counts["labels"] += 1
        else:
            counts["metadata"] += 1
    return counts


def _split_counts(entries: Iterable[_ShardEntry]) -> dict:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"files": 0, "images": 0, "labels": 0})
    for entry in entries:
        split = entry.split or "unsplit"
        counts[split]["files"] += 1
        if entry.kind == "image":
            counts[split]["images"] += 1
        elif entry.kind == "label":
            counts[split]["labels"] += 1
    return {split: dict(values) for split, values in sorted(counts.items())}


def _find_yolo_data_config(dataset_dir: Path) -> Path | None:
    for path in sorted(Path(dataset_dir).rglob("*")):
        if path.is_file() and path.name.lower() in YOLO_CONFIG_FILE_NAMES and _looks_like_yolo_data_config(path):
            return path
    return None


def _looks_like_yolo_data_config(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return ("train:" in text and ("val:" in text or "valid:" in text)) and ("names:" in text or re.search(r"(?m)^\s*nc\s*:", text))


def _load_yolo_config(path: Path) -> dict:
    if yaml is None:
        raise ShardMaterializationError("pyyaml is required for YOLO shard artifacts.")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ShardMaterializationError(f"YOLO data config {path} must be a mapping.")
    return loaded


def _write_yolo_data_yaml(path: Path, config: dict) -> None:
    if yaml is None:
        raise ShardMaterializationError("pyyaml is required for YOLO shard artifacts.")
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _resolve_yolo_split_images(config_path: Path, config: dict, split: str) -> list[Path]:
    raw_value = config.get("val" if split == "val" else split)
    if raw_value is None and split == "val":
        raw_value = config.get("valid")
    values = raw_value if isinstance(raw_value, list) else [raw_value]
    images: list[Path] = []
    for value in values:
        if value is None:
            continue
        target = _resolve_yolo_path(config_path, config, str(value))
        if target.is_dir():
            images.extend(path for path in target.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        elif target.is_file() and target.suffix.lower() in IMAGE_SUFFIXES:
            images.append(target)
        elif target.is_file():
            images.extend(_images_from_split_file(target, config_path.parent))
    return sorted(dict.fromkeys(path.resolve() for path in images), key=lambda path: str(path))


def _resolve_yolo_path(config_path: Path, config: dict, value: str) -> Path:
    path = Path(value)
    config_root = config_path.parent.resolve()
    if path.is_absolute():
        resolved = path.resolve()
        if not _path_is_within(resolved, config_root):
            raise ShardMaterializationError(f"YOLO data config path is outside the dataset root: {value!r}")
        return resolved
    root_value = str(config.get("path") or "").strip()
    if root_value:
        root = Path(root_value)
        if not root.is_absolute():
            root = config_path.parent / root
    else:
        root = config_path.parent
    resolved = (root / path).resolve()
    if not _path_is_within(resolved, config_root):
        raise ShardMaterializationError(f"YOLO data config path is outside the dataset root: {value!r}")
    return resolved


def _images_from_split_file(path: Path, config_base: Path) -> list[Path]:
    images: list[Path] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        image_path = Path(value)
        if not image_path.is_absolute():
            image_path = (config_base / image_path).resolve()
        if not _path_is_within(image_path, config_base.resolve()):
            raise ShardMaterializationError(f"YOLO split file path is outside the dataset root: {value!r}")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(image_path)
    return images


def _yolo_split_declared(config: dict, split: str) -> bool:
    if split == "val":
        return config.get("val") is not None or config.get("valid") is not None
    return config.get(split) is not None


def _yolo_names(config: dict) -> list[str]:
    names = config.get("names")
    if isinstance(names, dict):
        return [
            str(names[key])
            for key in sorted(
                names,
                key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
            )
        ]
    if isinstance(names, list):
        return [str(name) for name in names]
    nc = int(config.get("nc") or 0)
    return [str(index) for index in range(nc)]


def _yolo_label_path_for_image(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            return Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _yolo_label_class_counts(label_path: Path | None) -> dict[int, int]:
    if label_path is None or not label_path.is_file():
        return {}
    counts: dict[int, int] = defaultdict(int)
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            counts[int(float(parts[0]))] += 1
        except ValueError:
            continue
    return dict(counts)


def _common_yolo_image_root(image_paths: list[Path], split: str) -> Path | None:
    roots: list[Path] = []
    for image_path in image_paths:
        parts = list(image_path.parts)
        lowered = [part.lower() for part in parts]
        for index, part in enumerate(lowered):
            if part in COMMON_IMAGE_ROOT_NAMES and index + 1 < len(parts) and SPLIT_ALIASES.get(lowered[index + 1]) == split:
                roots.append(Path(*parts[: index + 2]))
                break
    if not roots:
        return None
    try:
        return Path(os.path.commonpath([str(path) for path in roots]))
    except ValueError:
        return None


def _yolo_target_relative_path(image_path: Path, *, image_root: Path | None, fallback_root: Path) -> Path:
    for root in (image_root, fallback_root):
        if root is None:
            continue
        try:
            return image_path.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            continue
    return Path(image_path.name)


def _classification_split_and_class(relative_path: Path) -> tuple[str, str]:
    parts = list(relative_path.parts)
    lowered = [part.lower() for part in parts]
    split = "unsplit"
    split_index: int | None = None
    for index, part in enumerate(lowered[:-1]):
        normalized = SPLIT_ALIASES.get(part)
        if normalized is not None:
            split = normalized
            split_index = index
            break
    if split_index is not None and split_index + 1 < len(parts) - 1:
        return split, parts[split_index + 1]
    if len(parts) >= 2:
        parent = parts[-2]
        if parent.lower() not in COMMON_IMAGE_ROOT_NAMES and SPLIT_ALIASES.get(parent.lower()) is None:
            return split, parent
    return split, "unknown"


def _iter_dataset_files(dataset_dir: Path) -> Iterable[Path]:
    for path in sorted(Path(dataset_dir).rglob("*")):
        if path.is_file():
            yield path


def _relative_path(root: Path, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ShardMaterializationError(f"Path is outside dataset root: {path}") from exc
    value = relative.as_posix()
    _safe_member_name(value)
    return value


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _safe_member_name(value: str) -> str:
    path = Path(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not str(value).strip():
        raise ShardMaterializationError(f"Unsafe dataset shard path: {value!r}")
    return path.as_posix()


def _ensure_child_path(root: Path, target: Path) -> None:
    try:
        target.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as exc:
        raise ShardMaterializationError(f"Dataset shard target escapes output dir: {target}") from exc


def _manifest_relative_path(root: Path, value: object) -> Path:
    if not value:
        raise ShardMaterializationError("Dataset shard manifest contains an empty shard path.")
    relative = _safe_member_name(str(value))
    return root / relative


def _unique_target_path(path: str, used: set[str]) -> str:
    safe = _safe_member_name(path)
    if safe not in used:
        used.add(safe)
        return safe
    candidate = Path(safe)
    digest = hashlib.sha256(safe.encode("utf-8")).hexdigest()[:10]
    renamed = candidate.with_name(f"{candidate.stem}-{digest}{candidate.suffix}").as_posix()
    used.add(renamed)
    return renamed


def _empty_label_file(temp_root: Path, split: str, name: str) -> Path:
    root = temp_root / "_empty_labels" / split
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text("", encoding="utf-8")
    return path


def _safe_shard_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned or "dataset-shard"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_size(path: Path) -> int:
    try:
        return int(Path(path).stat().st_size)
    except OSError:
        return 0


def _normalized_checksum(checksum: object) -> str:
    value = str(checksum or "").strip().lower()
    if value.startswith("sha256:"):
        value = value.split(":", 1)[1]
    if value.startswith("sha256-"):
        value = value.split("-", 1)[1]
    return value
