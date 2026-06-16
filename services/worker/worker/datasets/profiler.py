from __future__ import annotations
import ast
import os
from pathlib import Path
import re
import statistics
from PIL import Image, ImageStat

PROFILE_SCHEMA_VERSION = "dataset_profile_v1"
PROFILE_CACHE_VERSION = "dataset_profile_v1_bounded_v1"
DEFAULT_PROFILE_MAX_IMAGES = 25_000
DEFAULT_PROFILE_MAX_FILES = 25_000
DEFAULT_PROFILE_MAX_CORRUPT_PATHS = 25
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NON_CLASS_DIR_NAMES = {
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
COMMON_IMAGE_ROOT_NAMES = {"images", "image", "imgs", "img", "jpegimages", "data"}
SPLIT_FILE_NAMES = {
    "train.txt",
    "val.txt",
    "valid.txt",
    "validation.txt",
    "test.txt",
    "split.txt",
    "splits.txt",
    "train_test_split.txt",
}
LABELS_FILE_NAMES = {"labels.csv", "classes.csv", "annotations.csv", "image_class_labels.txt"}
CLASS_HIERARCHY_NAMES = {"class_hierarchy.json", "class_hierarchy.csv", "hierarchy.json", "synsets.txt"}
CUB_METADATA_FILE_NAMES = {"classes.txt", "images.txt", "parts.txt", "part_locs.txt"}
CUB_BBOX_FILE_NAMES = {"bounding_boxes.txt"}
YOLO_CONFIG_FILE_NAMES = {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml"}
YOLO_LABEL_DIR_NAMES = {"label", "labels"}
YOLO_IMAGE_DIR_NAMES = {"images", "image", "imgs", "img", "jpegimages"}
YOLO_SPLIT_KEYS = {"train": "train", "val": "val", "valid": "val", "test": "test"}
CLASSIFICATION_SPLIT_DIR_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
    "holdout": "test",
    "heldout": "test",
}
METADATA_ARTIFACT_TYPES = {
    "annotation_xml",
    "annotation_json",
    "labels_csv",
    "labels_file",
    "metadata_file",
    "metadata_folder",
    "bounding_boxes",
    "class_hierarchy",
    "yolo_dataset_config",
    "yolo_label_file",
}
NORMALIZATION_THUMBNAIL_SIZE = 256


def _profile_caps(
    *,
    max_images: int | None = None,
    max_artifact_files: int | None = None,
    max_corrupt_paths: int | None = None,
) -> dict[str, int]:
    return {
        "max_images": _positive_int(max_images, "MODEL_EXPRESS_PROFILE_MAX_IMAGES", DEFAULT_PROFILE_MAX_IMAGES),
        "max_artifact_files": _positive_int(
            max_artifact_files,
            "MODEL_EXPRESS_PROFILE_MAX_FILES",
            DEFAULT_PROFILE_MAX_FILES,
        ),
        "max_corrupt_paths": _positive_int(
            max_corrupt_paths,
            "MODEL_EXPRESS_PROFILE_MAX_CORRUPT_PATHS",
            DEFAULT_PROFILE_MAX_CORRUPT_PATHS,
        ),
    }


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


def _add_profile_warning(
    warnings: list[dict],
    code: str,
    message: str,
    **fields,
) -> None:
    if any(warning.get("code") == code for warning in warnings):
        return
    warning = {"code": code, "message": message}
    warning.update({key: value for key, value in fields.items() if value is not None})
    warnings.append(warning)

def profile_image_folder(
    dataset_dir: Path,
    *,
    max_images: int | None = None,
    max_artifact_files: int | None = None,
    max_corrupt_paths: int | None = None,
) -> dict:
    dataset_dir = Path(dataset_dir)
    caps = _profile_caps(
        max_images=max_images,
        max_artifact_files=max_artifact_files,
        max_corrupt_paths=max_corrupt_paths,
    )
    warnings: list[dict] = []
    split_layout = split_folder_classification_layout(dataset_dir)
    class_root = Path(split_layout["root"]) if split_layout else resolve_classification_root(dataset_dir)
    class_counts: dict[str, int] = {}
    split_image_counts: dict[str, int] = {}
    split_class_counts: dict[str, dict[str, int]] = {}
    corrupt_images: list[str] = []
    corrupt_image_count = 0

    widths: list[int] = []
    heights: list[int] = []
    image_paths: list[Path] = []
    image_headers_opened = 0
    image_file_candidates_seen = 0
    image_header_cap_hit = False

    class_entries = (
        list(split_layout.get("class_dirs", []))
        if split_layout
        else [
            {"class_name": class_dir.name, "path": class_dir, "split": ""}
            for class_dir in classification_class_dirs(class_root)
        ]
    )
    for entry in class_entries:
        class_dir = Path(entry["path"])
        class_name = str(entry["class_name"])
        split_name = str(entry.get("split") or "")
        image_count = 0

        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXT:
                continue

            image_file_candidates_seen += 1
            if image_headers_opened >= caps["max_images"]:
                image_header_cap_hit = True
                break

            try:
                image_headers_opened += 1
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify() #image corruption check

                widths.append(width)
                heights.append(height)
                image_count += 1
                image_paths.append(image_path)
            except Exception:
                corrupt_image_count += 1
                if len(corrupt_images) < caps["max_corrupt_paths"]:
                    corrupt_images.append(str(image_path))
        if image_count > 0:
            class_counts[class_name] = class_counts.get(class_name, 0) + image_count
            if split_name:
                split_image_counts[split_name] = split_image_counts.get(split_name, 0) + image_count
                per_split = split_class_counts.setdefault(split_name, {})
                per_split[class_name] = per_split.get(class_name, 0) + image_count
        if image_header_cap_hit:
            break
    if image_header_cap_hit:
        _add_profile_warning(
            warnings,
            "profile_image_header_cap",
            f"Dataset profiling opened the first {caps['max_images']} image header(s) and skipped the remainder.",
            cap=caps["max_images"],
            seen=image_file_candidates_seen,
        )
    if corrupt_image_count > len(corrupt_images):
        _add_profile_warning(
            warnings,
            "profile_corrupt_paths_retained_cap",
            "Only a bounded sample of corrupt image paths was retained in the profile.",
            cap=caps["max_corrupt_paths"],
            count=corrupt_image_count,
        )
    
    total_images = sum(class_counts.values())
    non_empty_counts = [count for count in class_counts.values() if count > 0]

    if non_empty_counts:
        imbalance_ratio = max(non_empty_counts) / min(non_empty_counts)
    else:
        imbalance_ratio = 0.0
    dimension_stats = _dimension_stats(widths, heights)
    artifacts = detect_dataset_artifacts(dataset_dir, max_files=caps["max_artifact_files"], warnings=warnings)
    metadata_summary = _metadata_summary(artifacts)
    yolo_summary = _yolo_dataset_summary(
        dataset_dir,
        artifacts,
        max_files=caps["max_artifact_files"],
        max_images=caps["max_images"],
        warnings=warnings,
    )
    yolo_available = bool(yolo_summary.get("available"))
    bbox_count = int(yolo_summary.get("bbox_count") or 0)
    bbox_per_class = yolo_summary.get("bbox_per_class") or yolo_summary.get("box_distribution") or {}
    metadata_summary["bbox_count"] = bbox_count
    metadata_summary["bbox_per_class"] = bbox_per_class
    metadata_summary["yolo_summary"] = yolo_summary
    if yolo_available:
        metadata_summary["bbox_available"] = True
        metadata_summary["object_detection_available"] = True
        metadata_summary["yolo_available"] = True
    if yolo_summary.get("available"):
        yolo_images = list(_iter_all_image_files(dataset_dir, max_images=caps["max_images"], warnings=warnings))
        if yolo_images:
            image_paths = yolo_images
            total_images = len(yolo_images)
            dimension_stats = _dimension_stats_for_paths(yolo_images)
        yolo_class_names = list(yolo_summary.get("class_names") or [])
        if not yolo_class_names:
            yolo_class_names = [f"class_{class_id}" for class_id in yolo_summary.get("class_ids", [])]
        if yolo_class_names:
            box_distribution = yolo_summary.get("box_distribution") or {}
            class_counts = {
                str(class_name): int(box_distribution.get(str(class_name), 0))
                for class_name in yolo_class_names
            }
            non_empty_counts = [count for count in class_counts.values() if count > 0]
            imbalance_ratio = max(non_empty_counts) / min(non_empty_counts) if non_empty_counts else 0.0
    split_summary = split_summary_from_artifacts(artifacts)
    if split_layout:
        split_summary["has_explicit_split"] = True
        split_summary["split_folder_detected"] = True
        split_summary["split_folder_root"] = (
            "dataset_root" if class_root == dataset_dir else _safe_relative_to(dataset_dir, class_root)
        )
        split_summary["split_folder_image_counts"] = {
            split: split_image_counts[split]
            for split in sorted(split_image_counts)
        }
        split_summary["split_folder_class_counts"] = {
            split: {
                class_name: counts[class_name]
                for class_name in sorted(counts)
            }
            for split, counts in sorted(split_class_counts.items())
        }
        split_summary["split_folder_class_count_by_split"] = {
            split: len(counts)
            for split, counts in sorted(split_class_counts.items())
        }
    if yolo_summary.get("split_hints"):
        split_summary["has_explicit_split"] = True
        split_summary["yolo_split_hints"] = yolo_summary.get("split_hints")
        split_summary["yolo_split_image_counts"] = yolo_summary.get("split_image_counts") or {}
        split_summary["yolo_split_label_file_counts"] = yolo_summary.get("split_label_file_counts") or {}
    leakage_warnings = _leakage_warnings(image_paths)
    visual_traits = _visual_trait_summary(image_paths, artifacts, class_counts)
    traits = _dataset_traits(
        class_count=len(class_counts),
        image_count=total_images,
        imbalance_ratio=imbalance_ratio,
        dimension_stats=dimension_stats,
        artifacts=artifacts,
        corrupt_file_count=corrupt_image_count,
        visual_traits=visual_traits,
        metadata_summary=metadata_summary,
    )
    task_type = "object_detection" if yolo_available else "image_classification"
    class_names = list(yolo_summary.get("class_names") or []) if yolo_available else sorted(class_counts.keys())
    if not class_names:
        class_names = sorted(class_counts.keys())
    
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "dataset_path": str(dataset_dir),
        "image_folder_root": str(class_root),
        "layout_summary": {
            "image_folder_root": "dataset_root"
            if class_root == dataset_dir
            else _safe_relative_to(dataset_dir, class_root),
            "class_folder_count": len(class_counts),
            "single_wrapper_unwrapped": class_root != dataset_dir and _is_under_single_wrapper(dataset_dir, class_root),
            "split_folder_layout": bool(split_layout),
        },
        "task_type": task_type,
        "class_names": class_names,
        "class_count": len(class_names),
        "image_count": total_images,
        "class_distribution": class_counts,
        "images_per_class": class_counts,
        "total_images": total_images,
        "object_detection_available": bool(metadata_summary.get("object_detection_available")),
        "yolo_available": yolo_available,
        "object_count": bbox_count,
        "bbox_count": bbox_count,
        "bbox_per_class": bbox_per_class,
        "yolo_summary": yolo_summary,
        "corrupt_file_count": corrupt_image_count,
        "corrupt_image_count": corrupt_image_count,
        "corrupt_images": corrupt_images,
        "width_min": min(widths) if widths else None,
        "width_max": max(widths) if widths else None,
        "height_min": min(heights) if heights else None,
        "height_max": max(heights) if heights else None,
        "image_dimension_stats": dimension_stats,
        "imbalance_ratio": round(imbalance_ratio, 3),
        "split_summary": split_summary,
        "metadata_summary": metadata_summary,
        "visual_trait_summary": visual_traits,
        "leakage_warnings": leakage_warnings,
        "dataset_traits": traits,
        "artifacts": artifacts,
        "profile_caps": caps,
        "profile_scan": {
            "image_headers_opened": image_headers_opened,
            "image_file_candidates_seen": image_file_candidates_seen,
            "image_header_cap_hit": image_header_cap_hit,
            "truncated": bool(warnings),
        },
        "profile_warnings": warnings,
        "warnings": warnings,
    }


def resolve_classification_root(dataset_dir: Path) -> Path:
    """Find the folder whose immediate children are image-class folders."""
    root = Path(dataset_dir)
    candidates: list[Path] = []

    def add(candidate: Path) -> None:
        try:
            resolved = candidate.resolve()
        except OSError:
            return
        if not candidate.is_dir():
            return
        if all(existing.resolve() != resolved for existing in candidates):
            candidates.append(candidate)

    for image_root in _common_image_root_dirs(root):
        add(image_root)

    wrapper = _single_wrapper_dir(root)
    if wrapper is not None:
        for image_root in _common_image_root_dirs(wrapper):
            add(image_root)
        add(wrapper)

    add(root)

    scored: list[tuple[int, int, int, Path]] = []
    for index, candidate in enumerate(candidates):
        class_dirs = classification_class_dirs(candidate)
        image_count = sum(_image_count_under(path) for path in class_dirs)
        scored.append((len(class_dirs), image_count, -index, candidate))

    multi_class = [item for item in scored if item[0] >= 2 and item[1] > 0]
    if multi_class:
        return max(multi_class)[3]
    single_class = [item for item in scored if item[0] >= 1 and item[1] > 0]
    if single_class:
        return max(single_class)[3]
    return root


def classification_class_dirs(class_root: Path) -> list[Path]:
    try:
        children = sorted(Path(class_root).iterdir())
    except OSError:
        return []
    return [
        child
        for child in children
        if child.is_dir()
        and child.name.lower() not in NON_CLASS_DIR_NAMES
        and _contains_image_file(child)
    ]


def split_folder_classification_layout(dataset_dir: Path) -> dict | None:
    for candidate in _split_folder_candidate_roots(Path(dataset_dir)):
        class_dirs = _split_folder_class_dirs(candidate)
        if not class_dirs:
            continue
        return {
            "root": candidate,
            "class_dirs": class_dirs,
            "splits": sorted({entry["split"] for entry in class_dirs if entry.get("split")}),
        }
    return None


def _split_folder_candidate_roots(root: Path) -> list[Path]:
    candidates: list[Path] = []

    def add(candidate: Path) -> None:
        try:
            resolved = candidate.resolve()
        except OSError:
            return
        if not candidate.is_dir():
            return
        if all(existing.resolve() != resolved for existing in candidates):
            candidates.append(candidate)

    add(root)
    for image_root in _common_image_root_dirs(root):
        add(image_root)
    wrapper = _single_wrapper_dir(root)
    if wrapper is not None:
        add(wrapper)
        for image_root in _common_image_root_dirs(wrapper):
            add(image_root)
    return candidates


def _split_folder_class_dirs(root: Path) -> list[dict]:
    try:
        children = sorted(Path(root).iterdir())
    except OSError:
        return []

    split_dirs: list[tuple[str, Path]] = []
    non_split_image_dirs: list[Path] = []
    for child in children:
        if not child.is_dir() or child.name.lower() in NON_CLASS_DIR_NAMES:
            continue
        if not _contains_image_file(child):
            continue
        split_name = _normalize_classification_split_name(child.name)
        if split_name:
            split_dirs.append((split_name, child))
        else:
            non_split_image_dirs.append(child)
    if not split_dirs or non_split_image_dirs:
        return []

    class_dirs: list[dict] = []
    for split_name, split_dir in split_dirs:
        try:
            children = sorted(split_dir.iterdir())
        except OSError:
            continue
        for class_dir in children:
            if (
                class_dir.is_dir()
                and class_dir.name.lower() not in NON_CLASS_DIR_NAMES
                and _contains_image_file(class_dir)
            ):
                class_dirs.append(
                    {
                        "split": split_name,
                        "class_name": class_dir.name,
                        "path": class_dir,
                    }
                )
    return class_dirs


def _normalize_classification_split_name(value: str) -> str:
    return CLASSIFICATION_SPLIT_DIR_ALIASES.get(str(value or "").strip().lower(), "")


def _common_image_root_dirs(root: Path) -> list[Path]:
    try:
        children = {child.name.lower(): child for child in root.iterdir() if child.is_dir()}
    except OSError:
        return []
    return [
        children[name]
        for name in sorted(COMMON_IMAGE_ROOT_NAMES)
        if name in children and _contains_image_file(children[name])
    ]


def _single_wrapper_dir(root: Path) -> Path | None:
    try:
        candidates = [
            child
            for child in root.iterdir()
            if child.is_dir()
            and child.name.lower() not in NON_CLASS_DIR_NAMES
            and _contains_image_file(child)
        ]
    except OSError:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return None


def _contains_image_file(directory: Path) -> bool:
    try:
        return any(path.is_file() and path.suffix.lower() in IMAGE_EXT for path in directory.rglob("*"))
    except OSError:
        return False


def _image_count_under(directory: Path, limit: int = 1_000) -> int:
    count = 0
    try:
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXT:
                count += 1
                if count >= limit:
                    return count
    except OSError:
        return 0
    return count


def _safe_relative_to(root: Path, child: Path) -> str:
    try:
        return child.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return child.name


def _is_under_single_wrapper(root: Path, child: Path) -> bool:
    wrapper = _single_wrapper_dir(root)
    if wrapper is None:
        return False
    try:
        child.resolve().relative_to(wrapper.resolve())
        return True
    except (OSError, ValueError):
        return False


def _dimension_stats(widths: list[int], heights: list[int]) -> dict:
    if not widths or not heights:
        return {
            "width": {},
            "height": {},
            "aspect_ratio": {},
        }
    aspect_ratios = [width / height for width, height in zip(widths, heights) if height > 0]
    return {
        "width": _numeric_stats(widths),
        "height": _numeric_stats(heights),
        "aspect_ratio": _numeric_stats(aspect_ratios),
    }


def _dimension_stats_for_paths(image_paths: list[Path]) -> dict:
    widths: list[int] = []
    heights: list[int] = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception:
            continue
        widths.append(width)
        heights.append(height)
    return _dimension_stats(widths, heights)


def _numeric_stats(values: list[float]) -> dict:
    if not values:
        return {}
    sorted_values = sorted(values)
    return {
        "min": round(float(sorted_values[0]), 4),
        "max": round(float(sorted_values[-1]), 4),
        "mean": round(float(statistics.fmean(sorted_values)), 4),
        "median": round(float(statistics.median(sorted_values)), 4),
    }


def detect_dataset_artifacts(
    dataset_dir: Path,
    *,
    max_files: int | None = None,
    warnings: list[dict] | None = None,
) -> list[dict]:
    """Return bounded, profile-safe artifact records detected in an image dataset."""
    dataset_dir = Path(dataset_dir)
    file_cap = _positive_int(max_files, "MODEL_EXPRESS_PROFILE_MAX_FILES", DEFAULT_PROFILE_MAX_FILES)
    warning_list = warnings if warnings is not None else []
    class_root = resolve_classification_root(dataset_dir)
    artifacts = [
        {
            "artifact_type": "image_root",
            "path": str(class_root),
            "format": "folder",
            "description": "Root folder containing image-classification data.",
        }
    ]
    folder_artifacts: list[dict] = []
    for class_dir in classification_class_dirs(class_root)[:50]:
        folder_artifacts.append(
            {
                "artifact_type": "class_folder",
                "path": str(class_dir),
                "format": "folder",
                "description": "Image class folder.",
            }
        )

    for child in _bounded_dirs(dataset_dir, limit=200):
        name = child.name.lower()
        if name in NON_CLASS_DIR_NAMES:
            folder_artifacts.append(
                {
                    "artifact_type": "metadata_folder",
                    "path": str(child),
                    "format": "folder",
                    "description": "Detected metadata or annotation folder.",
                }
            )

    file_artifacts: list[dict] = []
    files_seen = 0
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        files_seen += 1
        if files_seen > file_cap:
            _add_profile_warning(
                warning_list,
                "profile_artifact_file_cap",
                f"Artifact detection inspected the first {file_cap} file(s) and skipped the remainder.",
                cap=file_cap,
                seen=files_seen,
            )
            break
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name in CUB_BBOX_FILE_NAMES:
            file_artifacts.append({"artifact_type": "bounding_boxes", "path": str(path), "format": "cub_txt"})
        elif name in YOLO_CONFIG_FILE_NAMES and _looks_like_yolo_dataset_config(path):
            file_artifacts.append({"artifact_type": "yolo_dataset_config", "path": str(path), "format": "yaml"})
        elif suffix == ".xml":
            artifact_type = "bounding_boxes" if _looks_like_bbox_xml(path) else "annotation_xml"
            file_artifacts.append({"artifact_type": artifact_type, "path": str(path), "format": "xml"})
        elif suffix == ".json" and "annotation" in name:
            file_artifacts.append({"artifact_type": "annotation_json", "path": str(path), "format": "json"})
        elif name in LABELS_FILE_NAMES:
            artifact_type = "labels_csv" if suffix == ".csv" else "labels_file"
            file_artifacts.append({"artifact_type": artifact_type, "path": str(path), "format": suffix.lstrip(".") or "txt"})
        elif name in SPLIT_FILE_NAMES:
            file_artifacts.append(
                {"artifact_type": "split_file", "path": str(path), "format": suffix.lstrip(".") or "txt"}
            )
        elif name in CLASS_HIERARCHY_NAMES:
            file_artifacts.append(
                {"artifact_type": "class_hierarchy", "path": str(path), "format": suffix.lstrip(".") or "txt"}
            )
        elif name in CUB_METADATA_FILE_NAMES:
            file_artifacts.append({"artifact_type": "metadata_file", "path": str(path), "format": suffix.lstrip(".") or "txt"})
        elif suffix == ".txt" and _is_yolo_label_file(path, dataset_dir):
            file_artifacts.append({"artifact_type": "yolo_label_file", "path": str(path), "format": "txt"})
    return (artifacts + file_artifacts + folder_artifacts)[:200]


def _bounded_dirs(root: Path, limit: int) -> list[Path]:
    out: list[Path] = []
    try:
        iterator = root.rglob("*")
        for path in iterator:
            if path.is_dir():
                out.append(path)
                if len(out) >= limit:
                    break
    except OSError:
        return out
    return out


def detect_split_files(dataset_dir: Path) -> list[dict]:
    return [
        artifact
        for artifact in detect_dataset_artifacts(dataset_dir)
        if artifact.get("artifact_type") == "split_file"
    ]


def compute_image_normalization_metadata(
    dataset_dir: Path,
    max_images: int = 500,
    max_dimension: int = NORMALIZATION_THUMBNAIL_SIZE,
) -> dict:
    """Compute channel mean/std metadata without mutating the dataset or job state."""
    pixel_count = 0
    channel_sums = [0.0, 0.0, 0.0]
    channel_squared_sums = [0.0, 0.0, 0.0]
    images_seen = 0
    target_dimension = max(32, int(max_dimension))

    for image_path in _iter_image_files(dataset_dir):
        if images_seen >= max(1, max_images):
            break
        try:
            with Image.open(image_path) as image:
                image.draft("RGB", (target_dimension, target_dimension))
                rgb_image = image.convert("RGB")
                rgb_image.thumbnail((target_dimension, target_dimension), Image.Resampling.BILINEAR)
                stat = ImageStat.Stat(rgb_image)
        except Exception:
            continue

        images_seen += 1
        sample_pixels = max(1, rgb_image.width * rgb_image.height)
        pixel_count += sample_pixels
        for index in range(3):
            channel_sums[index] += stat.sum[index] / 255.0
            channel_squared_sums[index] += stat.sum2[index] / (255.0 * 255.0)

    if pixel_count == 0:
        return {
            "normalization": "dataset",
            "status": "unavailable",
            "image_count": 0,
            "mean": None,
            "std": None,
        }

    mean = [channel_sum / pixel_count for channel_sum in channel_sums]
    variance = [
        max(1e-12, (channel_squared_sums[index] / pixel_count) - (mean[index] * mean[index]))
        for index in range(3)
    ]
    std = [variance_value**0.5 for variance_value in variance]
    return {
        "normalization": "dataset",
        "status": "computed",
        "image_count": images_seen,
        "pixel_count": pixel_count,
        "max_dimension": target_dimension,
        "mean": [round(value, 6) for value in mean],
        "std": [round(max(value, 1e-6), 6) for value in std],
    }


def _iter_image_files(dataset_dir: Path):
    image_root = resolve_classification_root(Path(dataset_dir))
    for image_path in sorted(image_root.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXT:
            yield image_path


def _iter_all_image_files(
    dataset_dir: Path,
    *,
    max_images: int | None = None,
    warnings: list[dict] | None = None,
):
    image_cap = _positive_int(max_images, "MODEL_EXPRESS_PROFILE_MAX_IMAGES", DEFAULT_PROFILE_MAX_IMAGES)
    yielded = 0
    for image_path in Path(dataset_dir).rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXT:
            if yielded >= image_cap:
                if warnings is not None:
                    _add_profile_warning(
                        warnings,
                        "profile_all_image_file_cap",
                        f"Profile image inventory used the first {image_cap} image file(s) and skipped the remainder.",
                        cap=image_cap,
                        seen=yielded + 1,
                    )
                break
            yielded += 1
            yield image_path


def _looks_like_bbox_xml(path: Path) -> bool:
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:4096].lower()
    except Exception:
        return False
    return "<bndbox>" in sample or all(token in sample for token in ("<xmin>", "<ymin>", "<xmax>", "<ymax>"))


def _looks_like_yolo_dataset_config(path: Path) -> bool:
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:8192].lower()
    except Exception:
        return False
    return ("train:" in sample or "val:" in sample or "valid:" in sample) and (
        "names:" in sample or re.search(r"(?m)^\s*nc\s*:", sample) is not None
    )


def _is_yolo_label_file(path: Path, dataset_dir: Path) -> bool:
    parts = {part.lower() for part in path.relative_to(dataset_dir).parts[:-1]}
    if not parts.intersection(YOLO_LABEL_DIR_NAMES):
        return False
    if path.name.lower() in SPLIT_FILE_NAMES or path.name.lower() in LABELS_FILE_NAMES:
        return False
    return _looks_like_yolo_label_txt(path)


def _looks_like_yolo_label_txt(path: Path) -> bool:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    except Exception:
        return False
    non_empty = [line for line in lines if line and not line.startswith("#")]
    if not non_empty:
        return True
    for line in non_empty[:20]:
        if _parse_yolo_label_row(line) is None:
            return False
    return True


def _parse_yolo_label_row(line: str) -> tuple[int, float, float, float, float] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields = stripped.split()
    if len(fields) < 5:
        return None
    try:
        class_value = float(fields[0])
        class_id = int(class_value)
        x_center, y_center, width, height = [float(value) for value in fields[1:5]]
    except ValueError:
        return None
    if class_id < 0 or abs(class_value - class_id) > 1e-6:
        return None
    coords = [x_center, y_center, width, height]
    if any(value < -0.001 or value > 1.001 for value in coords):
        return None
    if width <= 0 or height <= 0:
        return None
    return class_id, x_center, y_center, width, height


def _yolo_dataset_summary(
    dataset_dir: Path,
    artifacts: list[dict],
    *,
    max_files: int | None = None,
    max_images: int | None = None,
    warnings: list[dict] | None = None,
) -> dict:
    dataset_dir = Path(dataset_dir)
    file_cap = _positive_int(max_files, "MODEL_EXPRESS_PROFILE_MAX_FILES", DEFAULT_PROFILE_MAX_FILES)
    image_cap = _positive_int(max_images, "MODEL_EXPRESS_PROFILE_MAX_IMAGES", DEFAULT_PROFILE_MAX_IMAGES)
    config_paths = _unique_paths(
        [
            Path(str(artifact.get("path") or ""))
            for artifact in artifacts
            if artifact.get("artifact_type") == "yolo_dataset_config"
        ]
        + _find_yolo_config_paths(dataset_dir, max_files=file_cap, warnings=warnings)
    )
    label_paths = _unique_paths(
        [
            Path(str(artifact.get("path") or ""))
            for artifact in artifacts
            if artifact.get("artifact_type") == "yolo_label_file"
        ]
        + _find_yolo_label_paths(dataset_dir, max_files=file_cap, warnings=warnings)
    )[:file_cap]
    if len(label_paths) >= file_cap and warnings is not None:
        _add_profile_warning(
            warnings,
            "profile_yolo_label_file_cap",
            f"Yolo summary inspected at most {file_cap} label file(s).",
            cap=file_cap,
        )
    available = bool(config_paths or label_paths)
    class_names: list[str] = []
    split_hints: dict[str, str | list[str]] = {}
    split_config_path: Path | None = None
    split_path_hint: str | list[str] | None = None
    for config_path in config_paths[:3]:
        config = _parse_yolo_config_file(config_path)
        if not class_names and config["class_names"]:
            class_names = list(config["class_names"])
        if not split_hints and config["split_hints"]:
            split_hints = dict(config["split_hints"])
            split_config_path = config_path
            split_path_hint = config["path"]

    class_ids: set[int] = set()
    box_counts_by_id: dict[int, int] = {}
    parsed_label_counts_by_path: dict[Path, dict[int, int]] = {}
    bbox_count = 0
    for label_path in label_paths:
        parsed = _parse_yolo_label_file(label_path)
        parsed_label_counts_by_path[label_path] = parsed
        for class_id, count in parsed.items():
            class_ids.add(class_id)
            box_counts_by_id[class_id] = box_counts_by_id.get(class_id, 0) + count
            bbox_count += count

    class_name_by_id: dict[int, str] = {}
    if class_names:
        class_name_by_id = {class_id: name for class_id, name in enumerate(class_names)}
        for class_id in sorted(class_ids):
            if class_id not in class_name_by_id:
                class_name_by_id[class_id] = f"class_{class_id}"
                class_names.append(class_name_by_id[class_id])
    elif class_ids:
        class_name_by_id = {class_id: f"class_{class_id}" for class_id in sorted(class_ids)}
        class_names = [class_name_by_id[class_id] for class_id in sorted(class_ids)]

    distribution: dict[str, int] = {}
    for class_id, count in sorted(box_counts_by_id.items()):
        class_name = class_name_by_id.get(class_id, f"class_{class_id}")
        distribution[class_name] = distribution.get(class_name, 0) + count

    split_image_counts: dict[str, int] = {}
    split_label_file_counts: dict[str, int] = {}
    split_bbox_counts: dict[str, int] = {}
    if split_hints and split_config_path is not None:
        split_image_counts, split_label_file_counts, split_bbox_counts = _yolo_split_counts(
            dataset_dir=dataset_dir,
            config_path=split_config_path,
            path_hint=split_path_hint,
            split_hints=split_hints,
            label_paths=label_paths,
            parsed_label_counts_by_path=parsed_label_counts_by_path,
            max_images=image_cap,
            warnings=warnings,
        )

    image_count = sum(1 for _ in _iter_all_image_files(dataset_dir, max_images=image_cap, warnings=warnings)) if available else 0
    return {
        "available": available,
        "format": "yolo" if available else "",
        "config_count": len(config_paths),
        "config_paths": [str(path) for path in config_paths[:10]],
        "label_file_count": len(label_paths),
        "label_count": len(label_paths),
        "image_count": image_count,
        "bbox_count": bbox_count,
        "bbox_per_class": distribution,
        "class_names": class_names,
        "class_ids": sorted(class_ids),
        "box_distribution": distribution,
        "split_hints": split_hints,
        "split_paths": split_hints,
        "split_image_counts": split_image_counts,
        "split_label_file_counts": split_label_file_counts,
        "split_bbox_counts": split_bbox_counts,
    }


def _find_yolo_config_paths(
    dataset_dir: Path,
    *,
    max_files: int | None = None,
    warnings: list[dict] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    file_cap = _positive_int(max_files, "MODEL_EXPRESS_PROFILE_MAX_FILES", DEFAULT_PROFILE_MAX_FILES)
    seen = 0
    try:
        iterator = Path(dataset_dir).rglob("*")
    except OSError:
        return paths
    for path in iterator:
        if path.is_file():
            seen += 1
            if seen > file_cap:
                if warnings is not None:
                    _add_profile_warning(
                        warnings,
                        "profile_yolo_config_scan_cap",
                        f"Yolo config discovery scanned the first {file_cap} file(s) and skipped the remainder.",
                        cap=file_cap,
                    )
                break
        if path.is_file() and path.name.lower() in YOLO_CONFIG_FILE_NAMES and _looks_like_yolo_dataset_config(path):
            paths.append(path)
    return paths


def _find_yolo_label_paths(
    dataset_dir: Path,
    *,
    max_files: int | None = None,
    warnings: list[dict] | None = None,
) -> list[Path]:
    paths: list[Path] = []
    file_cap = _positive_int(max_files, "MODEL_EXPRESS_PROFILE_MAX_FILES", DEFAULT_PROFILE_MAX_FILES)
    seen = 0
    try:
        iterator = Path(dataset_dir).rglob("*.txt")
    except OSError:
        return paths
    for path in iterator:
        seen += 1
        if seen > file_cap:
            if warnings is not None:
                _add_profile_warning(
                    warnings,
                    "profile_yolo_label_scan_cap",
                    f"Yolo label discovery scanned the first {file_cap} text file(s) and skipped the remainder.",
                    cap=file_cap,
                )
            break
        if path.is_file() and _is_yolo_label_file(path, dataset_dir):
            paths.append(path)
            if len(paths) >= file_cap:
                break
    return paths


def _unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not str(path):
            continue
        key = _path_key(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve()).lower()
    except OSError:
        return str(path.absolute()).lower()


def _parse_yolo_config_file(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"class_names": [], "split_hints": {}, "path": None}
    return {
        "class_names": _parse_yolo_names(text),
        "split_hints": _parse_yolo_split_hints(text),
        "path": _parse_yolo_value_at_key(text, "path"),
    }


def _parse_yolo_split_hints(text: str) -> dict[str, str | list[str]]:
    hints: dict[str, str | list[str]] = {}
    for raw_key, normalized_key in YOLO_SPLIT_KEYS.items():
        value = _parse_yolo_value_at_key(text, raw_key)
        if value is not None and normalized_key not in hints:
            hints[normalized_key] = value
    return hints


def _parse_yolo_value_at_key(text: str, key: str) -> str | list[str] | None:
    lines = text.splitlines()
    pattern = re.compile(rf"^(\s*){re.escape(key)}\s*:\s*(.*?)\s*$", re.IGNORECASE)
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        base_indent = len(match.group(1))
        inline_value = _strip_yaml_comment(match.group(2)).strip()
        if inline_value:
            return _parse_yolo_scalar_or_list(inline_value)

        block_values: list[str] = []
        for block_line in lines[index + 1 :]:
            stripped = block_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(block_line) - len(block_line.lstrip())
            if indent <= base_indent:
                break
            listed = re.match(r"^\s*-\s*(.*?)\s*$", block_line)
            if listed:
                value = _parse_yolo_scalar_or_list(listed.group(1))
                if isinstance(value, str):
                    block_values.append(value)
        return block_values or None
    return None


def _parse_yolo_scalar_or_list(value: str) -> str | list[str] | None:
    cleaned = _strip_yaml_comment(value).strip()
    if not cleaned or cleaned.lower() in {"null", "none", "~"}:
        return None
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            parsed = ast.literal_eval(cleaned)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [_clean_yaml_scalar(item) for item in parsed if _clean_yaml_scalar(item)]
        return [
            _clean_yaml_scalar(item)
            for item in _split_inline_csv(cleaned[1:-1])
            if _clean_yaml_scalar(item)
        ]
    return _clean_yaml_scalar(cleaned)


def _strip_yaml_comment(value: str) -> str:
    quote: str | None = None
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        elif char == "#" and quote is None:
            return value[:index]
    return value


def _clean_yaml_scalar(value) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text.strip()


def _split_inline_csv(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    depth = 0
    for char in value:
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "[{(":
            depth += 1
        elif char in "]})" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def _yolo_split_counts(
    dataset_dir: Path,
    config_path: Path,
    path_hint: str | list[str] | None,
    split_hints: dict[str, str | list[str]],
    label_paths: list[Path],
    parsed_label_counts_by_path: dict[Path, dict[int, int]],
    max_images: int,
    warnings: list[dict] | None,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    image_counts: dict[str, int] = {}
    label_file_counts: dict[str, int] = {}
    bbox_counts: dict[str, int] = {}
    label_lookup = {_path_key(path): path for path in label_paths}

    for split, hint in split_hints.items():
        targets = _resolve_yolo_split_targets(config_path, path_hint, hint)
        image_counts[split] = sum(
            _count_yolo_images_for_target(target, config_path.parent, max_images=max_images, warnings=warnings)
            for target in targets
        )

        split_label_paths: set[Path] = set()
        for target in targets:
            split_label_paths.update(
                _yolo_label_paths_for_target(
                    target=target,
                    dataset_dir=dataset_dir,
                    split=split,
                    label_lookup=label_lookup,
                    config_base=config_path.parent,
                )
            )
        if not split_label_paths:
            split_label_paths.update(_known_label_paths_for_split_name(label_lookup, split))

        label_file_counts[split] = len(split_label_paths)
        bbox_counts[split] = sum(
            sum(parsed_label_counts_by_path.get(label_path, {}).values())
            for label_path in split_label_paths
        )
    return image_counts, label_file_counts, bbox_counts


def _resolve_yolo_split_targets(
    config_path: Path,
    path_hint: str | list[str] | None,
    split_hint: str | list[str],
) -> list[Path]:
    roots = _resolve_yolo_config_roots(config_path, path_hint)
    targets: list[Path] = []
    for value in _as_path_values(split_hint):
        if _is_remote_yolo_path(value):
            continue
        split_path = Path(value)
        if split_path.is_absolute():
            if _path_is_within(split_path, config_path.parent):
                targets.append(split_path)
        else:
            for root in roots:
                target = (root / split_path).resolve()
                if _path_is_within(target, config_path.parent):
                    targets.append(target)
    return _unique_paths(targets)


def _resolve_yolo_config_roots(config_path: Path, path_hint: str | list[str] | None) -> list[Path]:
    roots: list[Path] = []
    for value in _as_path_values(path_hint):
        if _is_remote_yolo_path(value):
            continue
        root = Path(value)
        resolved = root.resolve() if root.is_absolute() else (config_path.parent / root).resolve()
        if _path_is_within(resolved, config_path.parent):
            roots.append(resolved)
    return _unique_paths(roots) or [config_path.parent]


def _as_path_values(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _is_remote_yolo_path(value: str) -> bool:
    lowered = value.lower()
    return "://" in lowered or lowered.startswith(("s3:", "gs:"))


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (OSError, ValueError):
        return False


def _count_yolo_images_for_target(
    target: Path,
    config_base: Path,
    *,
    max_images: int,
    warnings: list[dict] | None,
) -> int:
    if target.is_dir():
        count = 0
        for path in target.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXT:
                if count >= max_images:
                    if warnings is not None:
                        _add_profile_warning(
                            warnings,
                            "profile_yolo_split_image_cap",
                            f"Yolo split image counting used at most {max_images} image file(s) per target.",
                            cap=max_images,
                        )
                    break
                count += 1
        return count
    if target.is_file() and target.suffix.lower() in IMAGE_EXT:
        return 1
    if target.is_file() and target.suffix.lower() == ".txt":
        return sum(1 for _ in _image_paths_from_yolo_split_file(target, config_base))
    return 0


def _yolo_label_paths_for_target(
    target: Path,
    dataset_dir: Path,
    split: str,
    label_lookup: dict[str, Path],
    config_base: Path,
) -> set[Path]:
    label_paths: set[Path] = set()
    if target.is_dir():
        for label_dir in _candidate_label_dirs_for_image_dir(target, dataset_dir, split):
            label_paths.update(_known_label_paths_under(label_dir, label_lookup))
    elif target.is_file() and target.suffix.lower() == ".txt":
        if any(part.lower() in YOLO_LABEL_DIR_NAMES for part in target.parts):
            known = label_lookup.get(_path_key(target))
            if known is not None:
                label_paths.add(known)
        else:
            for image_path in _image_paths_from_yolo_split_file(target, config_base):
                for label_path in _candidate_label_paths_for_image(image_path, dataset_dir):
                    known = label_lookup.get(_path_key(label_path))
                    if known is not None:
                        label_paths.add(known)
    elif target.is_file() and target.suffix.lower() in IMAGE_EXT:
        for label_path in _candidate_label_paths_for_image(target, dataset_dir):
            known = label_lookup.get(_path_key(label_path))
            if known is not None:
                label_paths.add(known)
    return label_paths


def _candidate_label_dirs_for_image_dir(image_dir: Path, dataset_dir: Path, split: str) -> list[Path]:
    candidates: list[Path] = []
    try:
        relative_parts = list(image_dir.resolve().relative_to(dataset_dir.resolve()).parts)
    except (OSError, ValueError):
        relative_parts = []
    for index, part in enumerate(relative_parts):
        if part.lower() in YOLO_IMAGE_DIR_NAMES:
            candidates.append(dataset_dir.joinpath(*(relative_parts[:index] + ["labels"] + relative_parts[index + 1 :])))

    for split_name in _yolo_split_aliases(split):
        candidates.append(dataset_dir / "labels" / split_name)
        parent = image_dir.parent
        if parent.name.lower() in YOLO_IMAGE_DIR_NAMES:
            candidates.append(parent.parent / "labels" / split_name)
    return _unique_paths(candidates)


def _candidate_label_paths_for_image(image_path: Path, dataset_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    try:
        relative_parts = list(image_path.resolve().relative_to(dataset_dir.resolve()).parts)
    except (OSError, ValueError):
        relative_parts = []
    for index, part in enumerate(relative_parts):
        if part.lower() in YOLO_IMAGE_DIR_NAMES:
            candidate = dataset_dir.joinpath(*(relative_parts[:index] + ["labels"] + relative_parts[index + 1 :]))
            candidates.append(candidate.with_suffix(".txt"))
    candidates.append(dataset_dir / "labels" / image_path.with_suffix(".txt").name)
    return _unique_paths(candidates)


def _known_label_paths_under(label_dir: Path, label_lookup: dict[str, Path]) -> set[Path]:
    out: set[Path] = set()
    if not label_dir.is_dir():
        return out
    for path in label_dir.rglob("*.txt"):
        known = label_lookup.get(_path_key(path))
        if known is not None:
            out.add(known)
            if len(out) >= len(label_lookup):
                break
    return out


def _known_label_paths_for_split_name(label_lookup: dict[str, Path], split: str) -> set[Path]:
    aliases = set(_yolo_split_aliases(split))
    return {
        path
        for path in label_lookup.values()
        if aliases.intersection({part.lower() for part in path.parts})
    }


def _yolo_split_aliases(split: str) -> list[str]:
    if split == "val":
        return ["val", "valid", "validation"]
    return [split]


def _image_paths_from_yolo_split_file(path: Path, config_base: Path):
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return
    for line in lines:
        value = _strip_yaml_comment(line).strip()
        if not value:
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            yield candidate
            continue
        from_list_dir = path.parent / candidate
        if from_list_dir.exists():
            yield from_list_dir
        else:
            yield config_base / candidate


def _yolo_class_names_from_configs(paths: list[Path]) -> list[str]:
    for path in paths[:3]:
        names = _parse_yolo_config_file(path)["class_names"]
        if names:
            return list(names)
    return []


def _parse_yolo_names(text: str) -> list[str]:
    lines = text.splitlines()
    pattern = re.compile(r"^(\s*)names\s*:\s*(.*?)\s*$", re.IGNORECASE)
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        base_indent = len(match.group(1))
        inline_value = _strip_yaml_comment(match.group(2)).strip()
        if inline_value:
            return _parse_yolo_inline_names(inline_value)

        block_items: list[tuple[object, object]] = []
        list_index = 0
        for block_line in lines[index + 1 :]:
            stripped = block_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(block_line) - len(block_line.lstrip())
            if indent <= base_indent:
                break
            cleaned = _strip_yaml_comment(stripped).strip()
            if not cleaned:
                continue
            keyed = re.match(r"^([^:]+)\s*:\s*(.+?)\s*$", cleaned)
            listed = re.match(r"^-\s*(.+?)\s*$", cleaned)
            if keyed:
                block_items.append((_clean_yaml_scalar(keyed.group(1)), keyed.group(2)))
            elif listed:
                block_items.append((list_index, listed.group(1)))
                list_index += 1
            elif cleaned.startswith("[") or cleaned.startswith("{"):
                return _parse_yolo_inline_names(cleaned)
        return _names_from_mapping_items(block_items)
    return []


def _parse_yolo_inline_names(value: str) -> list[str]:
    cleaned = _strip_yaml_comment(value).strip()
    if not cleaned:
        return []
    try:
        parsed = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        return [_clean_yaml_scalar(item) for item in parsed if _clean_yaml_scalar(item)]
    if isinstance(parsed, dict):
        return _names_from_mapping_items(list(parsed.items()))
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return [
            _clean_yaml_scalar(item)
            for item in _split_inline_csv(cleaned[1:-1])
            if _clean_yaml_scalar(item)
        ]
    if cleaned.startswith("{") and cleaned.endswith("}"):
        items: list[tuple[object, object]] = []
        for item in _split_inline_csv(cleaned[1:-1]):
            if ":" not in item:
                continue
            key, raw_name = item.split(":", 1)
            items.append((_clean_yaml_scalar(key), raw_name))
        return _names_from_mapping_items(items)
    return []


def _names_from_mapping_items(items: list[tuple[object, object]]) -> list[str]:
    ordered: list[tuple[tuple[int, int, int], str]] = []
    for order, (key, value) in enumerate(items):
        name = _clean_yaml_scalar(value)
        if not name:
            continue
        try:
            numeric_key = int(str(key).strip().strip("'\""))
            sort_key = (0, numeric_key, order)
        except ValueError:
            sort_key = (1, order, order)
        ordered.append((sort_key, name))
    return [name for _, name in sorted(ordered, key=lambda item: item[0])]


def _parse_yolo_label_file(path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return counts
    for line in lines:
        parsed = _parse_yolo_label_row(line)
        if parsed is None:
            continue
        class_id = parsed[0]
        counts[class_id] = counts.get(class_id, 0) + 1
    return counts


def _metadata_summary(artifacts: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        artifact_type = str(artifact.get("artifact_type", "unknown"))
        counts[artifact_type] = counts.get(artifact_type, 0) + 1
    yolo_available = any(
        artifact.get("artifact_type") in {"yolo_dataset_config", "yolo_label_file"}
        for artifact in artifacts
    )
    bbox_available = any(artifact.get("artifact_type") == "bounding_boxes" for artifact in artifacts) or yolo_available
    return {
        "artifact_counts": counts,
        "metadata_available": any(
            artifact.get("artifact_type") in METADATA_ARTIFACT_TYPES
            for artifact in artifacts
        ),
        "bbox_available": bbox_available,
        "object_detection_available": yolo_available or bbox_available,
        "yolo_available": yolo_available,
    }


def _visual_trait_summary(image_paths: list[Path], artifacts: list[dict], class_counts: dict[str, int]) -> dict:
    sample_paths = sorted(image_paths)[:100]
    brightness_values: list[float] = []
    edge_values: list[float] = []
    border_dominance_values: list[float] = []
    for image_path in sample_paths:
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB").resize((32, 32))
                grayscale = rgb_image.convert("L")
                pixels = list(grayscale.getdata())
                brightness_values.append(statistics.fmean(pixels) / 255.0)
                edge_values.append(_edge_strength(grayscale))
                border_dominance_values.append(_border_dominance(rgb_image))
        except Exception:
            continue

    bbox_summary = _bbox_trait_summary(artifacts)
    lighting_variation = _safe_stdev(brightness_values)
    edge_mean = statistics.fmean(edge_values) if edge_values else 0.0
    border_dominance = statistics.fmean(border_dominance_values) if border_dominance_values else 0.0
    fine_grained_possible = len(class_counts) >= 10 and sum(class_counts.values()) / max(len(class_counts), 1) < 120
    crop_plausibility_score = 0.0
    if bbox_summary["bbox_count"]:
        crop_plausibility_score = max(0.0, min(1.0, 1.0 - abs(bbox_summary["median_area_ratio"] - 0.45)))

    return {
        "schema_version": "visual_traits_v1",
        "sampled_image_count": len(brightness_values),
        "object_scale": bbox_summary["object_scale"],
        "object_area_ratio_median": bbox_summary["median_area_ratio"],
        "bbox_count": bbox_summary["bbox_count"],
        "background_dominance": _level(border_dominance, low=0.35, high=0.62),
        "background_dominance_score": round(border_dominance, 4),
        "blur_likelihood": _blur_level(edge_mean),
        "edge_strength_mean": round(edge_mean, 4),
        "lighting_variation": _level(lighting_variation, low=0.08, high=0.18),
        "lighting_variation_score": round(lighting_variation, 4),
        "fine_grained_possible": fine_grained_possible,
        "crop_plausibility": _level(crop_plausibility_score, low=0.25, high=0.55),
        "crop_plausibility_score": round(crop_plausibility_score, 4),
    }


def _edge_strength(image) -> float:
    width, height = image.size
    pixels = image.load()
    diffs: list[float] = []
    for y in range(height - 1):
        for x in range(width - 1):
            current = int(pixels[x, y])
            diffs.append(abs(current - int(pixels[x + 1, y])) / 255.0)
            diffs.append(abs(current - int(pixels[x, y + 1])) / 255.0)
    return statistics.fmean(diffs) if diffs else 0.0


def _border_dominance(image) -> float:
    width, height = image.size
    pixels = image.load()
    border_pixels = []
    for x in range(width):
        border_pixels.append(pixels[x, 0])
        border_pixels.append(pixels[x, height - 1])
    for y in range(height):
        border_pixels.append(pixels[0, y])
        border_pixels.append(pixels[width - 1, y])
    border_mean = tuple(statistics.fmean(pixel[channel] for pixel in border_pixels) for channel in range(3))
    all_pixels = list(image.getdata())
    close_to_border = 0
    for pixel in all_pixels:
        distance = sum(abs(float(pixel[channel]) - border_mean[channel]) for channel in range(3)) / (255.0 * 3)
        if distance <= 0.12:
            close_to_border += 1
    return close_to_border / max(len(all_pixels), 1)


def _bbox_trait_summary(artifacts: list[dict]) -> dict:
    from worker.datasets.annotations import parse_annotation_json_bboxes, parse_pascal_voc_bboxes

    bbox_count = 0
    area_ratios: list[float] = []
    for artifact in artifacts:
        if artifact.get("artifact_type") not in {"bounding_boxes", "annotation_json", "yolo_label_file"}:
            continue
        path = Path(str(artifact.get("path") or ""))
        try:
            if artifact.get("artifact_type") == "yolo_label_file":
                label_areas = _yolo_label_area_ratios(path)
                bbox_count += len(label_areas)
                area_ratios.extend(label_areas)
                continue
            if path.name.lower() in CUB_BBOX_FILE_NAMES:
                bbox_count += _count_cub_bbox_rows(path)
                continue
            elif path.suffix.lower() == ".xml":
                payload = parse_pascal_voc_bboxes(path)
            elif path.suffix.lower() == ".json":
                payload = parse_annotation_json_bboxes(path)
            else:
                continue
        except Exception:
            continue
        image_size = payload.get("image_size") if isinstance(payload.get("image_size"), dict) else {}
        image_area = (image_size.get("width") or 0) * (image_size.get("height") or 0)
        for item in payload.get("objects") or []:
            bbox = item.get("bbox") if isinstance(item, dict) else None
            if not isinstance(bbox, dict):
                continue
            try:
                area = max(0, int(bbox["xmax"]) - int(bbox["xmin"])) * max(0, int(bbox["ymax"]) - int(bbox["ymin"]))
            except (KeyError, TypeError, ValueError):
                continue
            if area > 0:
                bbox_count += 1
            if image_area > 0 and area > 0:
                area_ratios.append(max(0.0, min(1.0, area / image_area)))
    median_area = round(float(statistics.median(area_ratios)), 4) if area_ratios else 0.0
    if not area_ratios:
        object_scale = "unknown"
    elif median_area < 0.15:
        object_scale = "small"
    elif median_area > 0.65:
        object_scale = "large"
    else:
        object_scale = "medium"
    return {"bbox_count": bbox_count, "median_area_ratio": median_area, "object_scale": object_scale}


def _yolo_label_area_ratios(path: Path) -> list[float]:
    areas: list[float] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return areas
    for line in lines:
        parsed = _parse_yolo_label_row(line)
        if parsed is None:
            continue
        _, _, _, width, height = parsed
        area = width * height
        if area > 0:
            areas.append(max(0.0, min(1.0, area)))
    return areas


def _count_cub_bbox_rows(path: Path) -> int:
    count = 0
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return 0
    for line in lines:
        fields = line.strip().split()
        if len(fields) < 5:
            continue
        try:
            width = float(fields[3])
            height = float(fields[4])
        except ValueError:
            continue
        if width > 0 and height > 0:
            count += 1
    return count


def _safe_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _level(value: float, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value >= low:
        return "medium"
    return "low"


def _blur_level(edge_strength: float) -> str:
    if edge_strength <= 0.025:
        return "high"
    if edge_strength <= 0.06:
        return "medium"
    return "low"


def split_summary_from_artifacts(artifacts: list[dict]) -> dict:
    split_files = [artifact for artifact in artifacts if artifact.get("artifact_type") == "split_file"]
    return {
        "split_files_detected": len(split_files),
        "has_explicit_split": bool(split_files),
        "split_file_paths": [str(artifact.get("path")) for artifact in split_files[:10]],
    }


def _leakage_warnings(image_paths: list[Path]) -> list[str]:
    by_name: dict[str, int] = {}
    for path in image_paths:
        normalized = path.name.lower()
        by_name[normalized] = by_name.get(normalized, 0) + 1
    duplicate_names = [name for name, count in by_name.items() if count > 1]
    if duplicate_names:
        return [f"{len(duplicate_names)} duplicate image filename(s) detected across dataset folders."]
    return []


def _dataset_traits(
    class_count: int,
    image_count: int,
    imbalance_ratio: float,
    dimension_stats: dict,
    artifacts: list[dict],
    corrupt_file_count: int,
    visual_traits: dict | None = None,
    metadata_summary: dict | None = None,
) -> list[str]:
    traits: list[str] = []
    metadata_summary = metadata_summary or {}
    if metadata_summary.get("yolo_available"):
        traits.append("yolo_format")
        traits.append("object_detection")
    if image_count < 500:
        traits.append("small_dataset")
    elif image_count < 5000:
        traits.append("medium_dataset")
    if class_count >= 20:
        traits.append("many_classes")
    if class_count >= 10 and image_count and image_count / class_count < 120:
        traits.append("fine_grained_possible")
    if imbalance_ratio >= 1.5:
        traits.append("imbalanced")
    width = dimension_stats.get("width", {})
    height = dimension_stats.get("height", {})
    max_dimension = max(width.get("max", 0), height.get("max", 0))
    min_dimension = min(width.get("min", 0), height.get("min", 0))
    if max_dimension and max_dimension <= 160:
        traits.append("low_resolution")
    elif max_dimension >= 512:
        traits.append("high_resolution")
    if min_dimension and max_dimension > min_dimension * 2:
        traits.append("variable_image_dimensions")
    if metadata_summary.get("bbox_available") or any(artifact.get("artifact_type") == "bounding_boxes" for artifact in artifacts):
        traits.append("bbox_available")
    if any(
        artifact.get("artifact_type") in METADATA_ARTIFACT_TYPES
        for artifact in artifacts
    ):
        traits.append("metadata_available")
    if corrupt_file_count:
        traits.append("corrupt_files_detected")
    visual_traits = visual_traits or {}
    if visual_traits.get("object_scale") == "small":
        traits.append("small_objects_possible")
    if visual_traits.get("background_dominance") == "high":
        traits.append("background_dominant_possible")
    if visual_traits.get("blur_likelihood") == "high":
        traits.append("blur_possible")
    if visual_traits.get("lighting_variation") == "high":
        traits.append("lighting_variation")
    if visual_traits.get("fine_grained_possible"):
        traits.append("fine_grained_possible")
    if visual_traits.get("crop_plausibility") in {"medium", "high"}:
        traits.append("crop_plausible")
    return sorted(set(traits))
