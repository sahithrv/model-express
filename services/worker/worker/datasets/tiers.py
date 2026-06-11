from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

try:
    import yaml
except Exception:  # pragma: no cover - pyyaml is a worker dependency, but keep imports tolerant.
    yaml = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_SPLITS = ("train", "val", "test")


def preview_subset_manifest_key(
    *,
    dataset_checksum: str,
    task_type: str,
    tier: str,
    fraction: float,
    seed: int,
    split_policy: str,
    image_size_family: str,
) -> str:
    payload = {
        "schema_version": "preview_subset_manifest.v1",
        "dataset_checksum": _stable_text(dataset_checksum),
        "task_type": _stable_text(task_type),
        "tier": _stable_text(tier),
        "fraction": round(_bounded_fraction(fraction), 6),
        "seed": int(seed),
        "split_policy": _stable_text(split_policy),
        "image_size_family": _stable_text(image_size_family),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"preview-subset-{digest[:32]}"


def build_classification_preview_subset(
    *,
    dataset_dir: Path,
    dataset_checksum: str,
    targets: list[int],
    split_indices: dict[str, list[int]],
    class_names: list[str],
    fraction: float,
    seed: int,
    split_policy: str,
    image_size_family: str,
) -> dict:
    fraction = _bounded_fraction(fraction)
    manifest_id = preview_subset_manifest_key(
        dataset_checksum=dataset_checksum,
        task_type="image_classification",
        tier="preview",
        fraction=fraction,
        seed=seed,
        split_policy=split_policy,
        image_size_family=image_size_family,
    )
    selected: dict[str, list[int]] = {}
    counts: dict[str, dict[str, int]] = {}
    for split, indices in split_indices.items():
        chosen = _stratified_indices(indices, targets, fraction=fraction, seed=seed, namespace=f"{manifest_id}:{split}")
        selected[split] = chosen
        counts[split] = _class_counts(chosen, targets, class_names)
    return {
        "schema_version": "preview_subset_manifest.v1",
        "manifest_id": manifest_id,
        "dataset_dir": str(dataset_dir),
        "dataset_checksum": _stable_text(dataset_checksum),
        "task_type": "image_classification",
        "tier": "preview",
        "fraction": fraction,
        "seed": int(seed),
        "split_policy": split_policy,
        "image_size_family": image_size_family,
        "indices": selected,
        "class_counts": counts,
        "split_counts": {split: len(indices) for split, indices in selected.items()},
        "class_names": list(class_names),
    }


def materialize_yolo_preview_subset(
    *,
    dataset_dir: Path,
    data_config_path: Path,
    output_root: Path,
    dataset_checksum: str,
    fraction: float,
    seed: int,
    split_policy: str,
    image_size_family: str,
) -> tuple[Path, dict]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to build YOLO preview subset manifests.")
    fraction = _bounded_fraction(fraction)
    config = _load_yolo_config(data_config_path)
    names = _yolo_names(config)
    manifest_id = preview_subset_manifest_key(
        dataset_checksum=dataset_checksum,
        task_type="object_detection",
        tier="preview",
        fraction=fraction,
        seed=seed,
        split_policy=split_policy,
        image_size_family=image_size_family,
    )
    subset_root = output_root / manifest_id
    if subset_root.exists():
        data_yaml = subset_root / "data.yaml"
        manifest_path = subset_root / "preview_subset_manifest.json"
        if data_yaml.is_file() and manifest_path.is_file():
            return data_yaml, json.loads(manifest_path.read_text(encoding="utf-8"))
        shutil.rmtree(subset_root)
    subset_root.mkdir(parents=True, exist_ok=True)

    source_splits = {
        split: _resolve_yolo_split_images(data_config_path, config, split)
        for split in YOLO_SPLITS
        if _split_declared(config, split)
    }
    if "train" not in source_splits or "val" not in source_splits:
        raise ValueError("YOLO preview subset requires train and val splits in data.yaml.")

    selected_splits: dict[str, list[Path]] = {}
    object_counts: dict[str, dict[str, int]] = {}
    image_counts: dict[str, int] = {}
    label_counts: dict[str, int] = {}
    for split, images in source_splits.items():
        labels = {image: _yolo_label_path_for_image(image) for image in images}
        chosen = _stratified_yolo_images(
            images,
            labels,
            fraction=fraction,
            seed=seed,
            namespace=f"{manifest_id}:{split}",
        )
        selected_splits[split] = chosen
        image_counts[split] = len(chosen)
        label_counts[split] = 0
        split_object_counts = defaultdict(int)
        for image_path in chosen:
            rel_name = _stable_yolo_subset_name(data_config_path.parent, image_path)
            target_image = subset_root / "images" / split / rel_name
            target_label = subset_root / "labels" / split / Path(rel_name).with_suffix(".txt")
            _link_or_copy(image_path, target_image)
            label_path = labels[image_path]
            if label_path is not None and label_path.is_file():
                _link_or_copy(label_path, target_label)
                label_counts[split] += 1
                for class_id, count in _yolo_label_class_counts(label_path).items():
                    split_object_counts[str(class_id)] += count
            else:
                target_label.parent.mkdir(parents=True, exist_ok=True)
                target_label.write_text("", encoding="utf-8")
        object_counts[split] = dict(sorted(split_object_counts.items(), key=lambda item: int(item[0])))

    data_yaml = subset_root / "data.yaml"
    subset_config = {
        "path": str(subset_root),
        "train": "images/train",
        "val": "images/val",
        "nc": int(config.get("nc") or len(names)),
        "names": names,
    }
    if "test" in selected_splits:
        subset_config["test"] = "images/test"
    data_yaml.write_text(yaml.safe_dump(subset_config, sort_keys=False), encoding="utf-8")

    manifest = {
        "schema_version": "preview_subset_manifest.v1",
        "manifest_id": manifest_id,
        "dataset_dir": str(dataset_dir),
        "source_data_yaml": str(data_config_path),
        "data_yaml": str(data_yaml),
        "dataset_checksum": _stable_text(dataset_checksum),
        "task_type": "object_detection",
        "tier": "preview",
        "fraction": fraction,
        "seed": int(seed),
        "split_policy": split_policy,
        "image_size_family": image_size_family,
        "split_image_counts": image_counts,
        "split_label_file_counts": label_counts,
        "split_object_counts": object_counts,
        "class_names": names,
        "nc": subset_config["nc"],
    }
    (subset_root / "preview_subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return data_yaml, manifest


def _stratified_indices(
    indices: list[int],
    targets: list[int],
    *,
    fraction: float,
    seed: int,
    namespace: str,
) -> list[int]:
    if not indices:
        return []
    target_count = _target_subset_count(len(indices), fraction)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index in indices:
        by_class[int(targets[index])].append(index)
    selected: list[int] = []
    rng = random.Random(_seed(seed, namespace))
    for class_indices in by_class.values():
        shuffled = list(class_indices)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:1])
    remaining = [index for index in indices if index not in set(selected)]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, target_count - len(selected))])
    return sorted(selected)


def _stratified_yolo_images(
    images: list[Path],
    labels: dict[Path, Path | None],
    *,
    fraction: float,
    seed: int,
    namespace: str,
) -> list[Path]:
    if not images:
        return []
    target_count = _target_subset_count(len(images), fraction)
    by_class: dict[int, list[Path]] = defaultdict(list)
    unlabeled: list[Path] = []
    for image in images:
        label_counts = _yolo_label_class_counts(labels.get(image))
        if not label_counts:
            unlabeled.append(image)
            continue
        for class_id in label_counts:
            by_class[class_id].append(image)
    rng = random.Random(_seed(seed, namespace))
    selected: list[Path] = []
    seen: set[Path] = set()
    for class_images in by_class.values():
        shuffled = list(dict.fromkeys(class_images))
        rng.shuffle(shuffled)
        for image in shuffled:
            if image not in seen:
                selected.append(image)
                seen.add(image)
                break
    remaining = [image for image in images if image not in seen]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, target_count - len(selected))])
    return sorted(selected, key=lambda path: str(path))


def _load_yolo_config(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) if yaml is not None else {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YOLO data config {path} must be a mapping.")
    return loaded


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
            raise ValueError(f"YOLO data config path is outside the dataset root: {value!r}")
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
        raise ValueError(f"YOLO data config path is outside the dataset root: {value!r}")
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
            raise ValueError(f"YOLO split file path is outside the dataset root: {value!r}")
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            images.append(image_path)
    return images


def _yolo_label_path_for_image(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "images":
            candidate = Path(*parts[:index], "labels", *parts[index + 1 :]).with_suffix(".txt")
            return candidate
    return image_path.with_suffix(".txt")


def _yolo_label_class_counts(label_path: Path | None) -> dict[int, int]:
    if label_path is None or not label_path.is_file():
        return {}
    counts: dict[int, int] = defaultdict(int)
    try:
        lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            counts[int(float(parts[0]))] += 1
        except ValueError:
            continue
    return dict(counts)


def _stable_yolo_subset_name(config_base: Path, image_path: Path) -> Path:
    del config_base
    return Path(image_path.name)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


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


def _split_declared(config: dict, split: str) -> bool:
    if split == "val":
        return config.get("val") is not None or config.get("valid") is not None
    return config.get(split) is not None


def _class_counts(indices: Iterable[int], targets: list[int], class_names: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in class_names}
    for index in indices:
        target = int(targets[index])
        if 0 <= target < len(class_names):
            counts[class_names[target]] += 1
    return counts


def _target_subset_count(total: int, fraction: float) -> int:
    if total <= 0:
        return 0
    return max(1, min(total, int(round(total * _bounded_fraction(fraction)))))


def _bounded_fraction(value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.25
    return max(0.01, min(1.0, parsed))


def _seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{namespace}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _stable_text(value: object) -> str:
    return str(value or "").strip().lower()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False
