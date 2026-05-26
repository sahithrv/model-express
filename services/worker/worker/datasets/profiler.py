from __future__ import annotations
from pathlib import Path
import statistics
from PIL import Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NON_CLASS_DIR_NAMES = {"annotation", "annotations", "metadata", "meta", "splits"}
SPLIT_FILE_NAMES = {"train.txt", "val.txt", "valid.txt", "validation.txt", "test.txt", "split.txt", "splits.txt"}
LABELS_FILE_NAMES = {"labels.csv", "classes.csv", "annotations.csv"}
CLASS_HIERARCHY_NAMES = {"class_hierarchy.json", "class_hierarchy.csv", "hierarchy.json", "synsets.txt"}

def profile_image_folder(dataset_dir: Path) -> dict:
    class_counts: dict[str, int] = {}
    corrupt_images: list[str] = []

    widths: list[int] = []
    heights: list[int] = []
    image_paths: list[Path] = []

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name.lower() in NON_CLASS_DIR_NAMES:
            continue

        class_name = class_dir.name
        class_counts[class_name] = 0

        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in IMAGE_EXT:
                continue

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify() #image corruption check

                widths.append(width)
                heights.append(height)
                class_counts[class_name]+=1
                image_paths.append(image_path)
            except Exception:
                corrupt_images.append(str(image_path))
    
    total_images = sum(class_counts.values())
    non_empty_counts = [count for count in class_counts.values() if count > 0]

    if non_empty_counts:
        imbalance_ratio = max(non_empty_counts) / min(non_empty_counts)
    else:
        imbalance_ratio = 0.0
    dimension_stats = _dimension_stats(widths, heights)
    artifacts = detect_dataset_artifacts(dataset_dir)
    metadata_summary = _metadata_summary(artifacts)
    split_summary = split_summary_from_artifacts(artifacts)
    leakage_warnings = _leakage_warnings(image_paths)
    visual_traits = _visual_trait_summary(image_paths, artifacts, class_counts)
    traits = _dataset_traits(
        class_count=len(class_counts),
        image_count=total_images,
        imbalance_ratio=imbalance_ratio,
        dimension_stats=dimension_stats,
        artifacts=artifacts,
        corrupt_file_count=len(corrupt_images),
        visual_traits=visual_traits,
    )
    
    return {
        "schema_version": "dataset_profile_v1",
        "dataset_path": str(dataset_dir),
        "task_type": "image_classification",
        "class_names": sorted(class_counts.keys()),
        "class_count": len(class_counts),
        "image_count": total_images,
        "class_distribution": class_counts,
        "images_per_class": class_counts,
        "total_images": total_images,
        "corrupt_file_count": len(corrupt_images),
        "corrupt_image_count": len(corrupt_images),
        "corrupt_images": corrupt_images[:25],
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
    }


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


def detect_dataset_artifacts(dataset_dir: Path) -> list[dict]:
    """Return bounded, profile-safe artifact records detected in an image dataset."""
    artifacts = [
        {
            "artifact_type": "image_root",
            "path": str(dataset_dir),
            "format": "folder",
            "description": "Root folder containing image-classification data.",
        }
    ]
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir():
            artifacts.append(
                {
                    "artifact_type": "class_folder",
                    "path": str(child),
                    "format": "folder",
                    "description": "Image class folder.",
                }
            )
            name = child.name.lower()
            if name in {"metadata", "meta", "annotations"}:
                artifacts.append(
                    {
                        "artifact_type": "metadata_folder",
                        "path": str(child),
                        "format": "folder",
                        "description": "Detected metadata or annotation folder.",
                    }
                )

    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        suffix = path.suffix.lower()
        if suffix == ".xml":
            artifact_type = "bounding_boxes" if _looks_like_bbox_xml(path) else "annotation_xml"
            artifacts.append({"artifact_type": artifact_type, "path": str(path), "format": "xml"})
        elif suffix == ".json" and "annotation" in name:
            artifacts.append({"artifact_type": "annotation_json", "path": str(path), "format": "json"})
        elif name in LABELS_FILE_NAMES:
            artifacts.append({"artifact_type": "labels_csv", "path": str(path), "format": "csv"})
        elif name in SPLIT_FILE_NAMES:
            artifacts.append(
                {"artifact_type": "split_file", "path": str(path), "format": suffix.lstrip(".") or "txt"}
            )
        elif name in CLASS_HIERARCHY_NAMES:
            artifacts.append(
                {"artifact_type": "class_hierarchy", "path": str(path), "format": suffix.lstrip(".") or "txt"}
            )
    return artifacts[:200]


def detect_split_files(dataset_dir: Path) -> list[dict]:
    return [
        artifact
        for artifact in detect_dataset_artifacts(dataset_dir)
        if artifact.get("artifact_type") == "split_file"
    ]


def compute_image_normalization_metadata(dataset_dir: Path, max_images: int = 500) -> dict:
    """Compute channel mean/std metadata without mutating the dataset or job state."""
    pixel_count = 0
    channel_sums = [0.0, 0.0, 0.0]
    channel_squared_sums = [0.0, 0.0, 0.0]
    images_seen = 0

    for image_path in _iter_image_files(dataset_dir):
        if images_seen >= max(1, max_images):
            break
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                pixels = list(rgb_image.getdata())
        except Exception:
            continue

        images_seen += 1
        pixel_count += len(pixels)
        for red, green, blue in pixels:
            values = (red / 255.0, green / 255.0, blue / 255.0)
            for index, value in enumerate(values):
                channel_sums[index] += value
                channel_squared_sums[index] += value * value

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
        "mean": [round(value, 6) for value in mean],
        "std": [round(max(value, 1e-6), 6) for value in std],
    }


def _iter_image_files(dataset_dir: Path):
    for image_path in sorted(dataset_dir.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXT:
            yield image_path


def _looks_like_bbox_xml(path: Path) -> bool:
    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:4096].lower()
    except Exception:
        return False
    return "<bndbox>" in sample or all(token in sample for token in ("<xmin>", "<ymin>", "<xmax>", "<ymax>"))


def _metadata_summary(artifacts: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for artifact in artifacts:
        artifact_type = str(artifact.get("artifact_type", "unknown"))
        counts[artifact_type] = counts.get(artifact_type, 0) + 1
    return {
        "artifact_counts": counts,
        "metadata_available": any(
            artifact.get("artifact_type")
            in {"annotation_xml", "annotation_json", "labels_csv", "metadata_folder", "bounding_boxes", "class_hierarchy"}
            for artifact in artifacts
        ),
        "bbox_available": any(artifact.get("artifact_type") == "bounding_boxes" for artifact in artifacts),
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

    area_ratios: list[float] = []
    for artifact in artifacts:
        if artifact.get("artifact_type") not in {"bounding_boxes", "annotation_json"}:
            continue
        path = Path(str(artifact.get("path") or ""))
        try:
            if path.suffix.lower() == ".xml":
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
    return {"bbox_count": len(area_ratios), "median_area_ratio": median_area, "object_scale": object_scale}


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
) -> list[str]:
    traits: list[str] = []
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
    if any(artifact.get("artifact_type") == "bounding_boxes" for artifact in artifacts):
        traits.append("bbox_available")
    if any(
        artifact.get("artifact_type") in {"annotation_xml", "annotation_json", "labels_csv", "metadata_folder", "class_hierarchy"}
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
