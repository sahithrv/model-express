from __future__ import annotations

import base64
import hashlib
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

from worker.datasets.profiler import IMAGE_EXT, NON_CLASS_DIR_NAMES


SELECTION_STRATEGY = "deterministic_risk_and_representative_sampling"


@dataclass(frozen=True)
class _ImageRecord:
    class_name: str
    path: Path
    relative_key: str


@dataclass
class _ImageCandidate:
    record: _ImageRecord
    image_id: str
    width: int
    height: int
    brightness: float
    contrast: float
    blur_score: float
    selection_basis: set[str] = field(default_factory=set)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            return 1.0
        return self.width / self.height

    @property
    def aspect_deviation(self) -> float:
        return abs(math.log(max(self.aspect_ratio, 0.001)))


def generate_visual_sample_pack(
    *,
    dataset_dir: Path,
    dataset_id: str = "",
    dataset_name: str = "",
    max_total_images: int = 48,
    max_high_detail_images: int = 6,
    max_image_bytes: int = 350_000,
    max_total_bytes: int = 8_000_000,
    image_size: int = 512,
    high_detail_image_size: int = 1024,
    quality: int = 82,
    seed: int = 0,
    max_metadata_images: int = 2_500,
) -> dict:
    """Build a deterministic, bounded visual-analysis sample pack.

    The returned public manifest intentionally identifies images only by stable
    image_id values and class/visual metadata. No local or dataset-relative file
    paths are included in the manifest or image inputs.
    """
    try:
        from PIL import Image, ImageFilter, ImageStat
    except Exception as exc:
        return {
            "schema_version": "visual_sample_pack_v1",
            "status": "unavailable",
            "error_code": "PIL_UNAVAILABLE",
            "error": str(exc),
            "sample_manifest": _empty_manifest(dataset_id, dataset_name),
            "image_inputs": [],
        }

    dataset_dir = Path(dataset_dir)
    total_limit = max(0, int(max_total_images))
    high_detail_limit = max(0, min(int(max_high_detail_images), total_limit))
    image_byte_limit = max(1, int(max_image_bytes))
    total_byte_limit = max(0, int(max_total_bytes))
    target_size = max(32, int(image_size))
    high_target_size = max(target_size, int(high_detail_image_size))
    quality = max(35, min(95, int(quality)))
    metadata_limit = max(total_limit, int(max_metadata_images))

    class_records = _class_image_records(dataset_dir)
    all_records = [record for records in class_records.values() for record in records]
    records_for_metadata = _metadata_records(class_records, all_records, metadata_limit, seed)
    candidates, skipped_corrupt = _load_candidates(
        records_for_metadata,
        dataset_id=dataset_id,
        seed=seed,
        Image=Image,
        ImageFilter=ImageFilter,
        ImageStat=ImageStat,
    )
    selected = _select_candidates(candidates, class_records, total_limit, seed)

    samples: list[dict] = []
    image_inputs: list[dict] = []
    total_bytes = 0
    encoded_high_detail_count = 0
    skipped_over_budget = 0

    for candidate in selected:
        detail_level = "standard"
        if encoded_high_detail_count < high_detail_limit and _prefers_high_detail(candidate):
            detail_level = "high"
        encode_size = high_target_size if detail_level == "high" else target_size
        try:
            prepared = _encode_jpeg_input(
                candidate.record.path,
                image_size=encode_size,
                max_image_bytes=image_byte_limit,
                quality=quality,
                Image=Image,
            )
        except Exception:
            skipped_corrupt += 1
            continue
        if total_bytes + prepared["bytes"] > total_byte_limit:
            skipped_over_budget += 1
            continue

        if detail_level == "high":
            encoded_high_detail_count += 1
        total_bytes += prepared["bytes"]
        basis = sorted(candidate.selection_basis) or ["class_representative"]
        samples.append(
            {
                "image_id": candidate.image_id,
                "class_name": candidate.record.class_name,
                "width": candidate.width,
                "height": candidate.height,
                "aspect_ratio": round(candidate.aspect_ratio, 4),
                "selection_basis": basis,
                "detail_level": detail_level,
                "prepared_width": prepared["width"],
                "prepared_height": prepared["height"],
                "prepared_bytes": prepared["bytes"],
                "mime_type": "image/jpeg",
            }
        )
        image_inputs.append(
            {
                "image_id": candidate.image_id,
                "detail_level": detail_level,
                "mime_type": "image/jpeg",
                "width": prepared["width"],
                "height": prepared["height"],
                "bytes": prepared["bytes"],
                "data_base64": prepared["data_base64"],
            }
        )

    per_class_counts: dict[str, int] = {}
    for sample in samples:
        class_name = str(sample["class_name"])
        per_class_counts[class_name] = per_class_counts.get(class_name, 0) + 1

    limitations = _coverage_limitations(
        total_images=len(all_records),
        images_analyzed=len(samples),
        classes_total=len(class_records),
        classes_covered=len(per_class_counts),
        skipped_corrupt=skipped_corrupt,
        skipped_over_budget=skipped_over_budget,
        metadata_limited=len(records_for_metadata) < len(all_records),
    )
    manifest = {
        "schema_version": "visual_sample_manifest_v1",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "selection_strategy": SELECTION_STRATEGY,
        "selection_basis": sorted({basis for sample in samples for basis in sample["selection_basis"]}),
        "images_available": len(all_records),
        "images_analyzed": len(samples),
        "classes_total": len(class_records),
        "classes_covered": len(per_class_counts),
        "class_coverage_ratio": _ratio(len(per_class_counts), len(class_records)),
        "per_class_counts": dict(sorted(per_class_counts.items())),
        "hard_example_count": 0,
        "edge_case_count": sum(1 for sample in samples if _is_edge_case_basis(sample["selection_basis"])),
        "high_detail_image_count": encoded_high_detail_count,
        "max_total_images": total_limit,
        "max_high_detail_images": high_detail_limit,
        "max_image_bytes": image_byte_limit,
        "max_total_bytes": total_byte_limit,
        "samples": samples,
        "limitations": limitations,
    }
    return {
        "schema_version": "visual_sample_pack_v1",
        "status": "created",
        "sample_manifest": manifest,
        "image_inputs": image_inputs,
        "summary": {
            "selection_strategy": SELECTION_STRATEGY,
            "image_count": len(samples),
            "total_bytes": total_bytes,
            "skipped_corrupt": skipped_corrupt,
            "skipped_over_budget": skipped_over_budget,
            "metadata_images_considered": len(records_for_metadata),
        },
    }


def _empty_manifest(dataset_id: str, dataset_name: str) -> dict:
    return {
        "schema_version": "visual_sample_manifest_v1",
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "selection_strategy": SELECTION_STRATEGY,
        "selection_basis": [],
        "images_available": 0,
        "images_analyzed": 0,
        "classes_total": 0,
        "classes_covered": 0,
        "class_coverage_ratio": 0.0,
        "per_class_counts": {},
        "hard_example_count": 0,
        "edge_case_count": 0,
        "high_detail_image_count": 0,
        "samples": [],
        "limitations": ["Visual sample pack could not be generated."],
    }


def _class_image_records(dataset_dir: Path) -> dict[str, list[_ImageRecord]]:
    records: dict[str, list[_ImageRecord]] = {}
    class_dirs = [
        path
        for path in sorted(dataset_dir.iterdir())
        if path.is_dir() and path.name.lower() not in NON_CLASS_DIR_NAMES
    ]
    for class_dir in class_dirs:
        class_records = []
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXT:
                continue
            relative_key = image_path.relative_to(dataset_dir).as_posix()
            class_records.append(_ImageRecord(class_dir.name, image_path, relative_key))
        if class_records:
            records[class_dir.name] = class_records
    return records


def _metadata_records(
    class_records: dict[str, list[_ImageRecord]],
    all_records: list[_ImageRecord],
    metadata_limit: int,
    seed: int,
) -> list[_ImageRecord]:
    if len(all_records) <= metadata_limit:
        return sorted(all_records, key=lambda record: record.relative_key)

    selected: dict[str, _ImageRecord] = {}
    for class_name, records in sorted(class_records.items()):
        ranked = sorted(records, key=lambda record: _stable_rank(seed, class_name, record.relative_key))
        if ranked:
            selected[ranked[0].relative_key] = ranked[0]

    remaining = [
        record
        for record in sorted(
            all_records,
            key=lambda record: _stable_rank(seed, record.class_name, record.relative_key),
        )
        if record.relative_key not in selected
    ]
    for record in remaining:
        if len(selected) >= metadata_limit:
            break
        selected[record.relative_key] = record
    return sorted(selected.values(), key=lambda record: record.relative_key)


def _load_candidates(
    records: list[_ImageRecord],
    *,
    dataset_id: str,
    seed: int,
    Image,
    ImageFilter,
    ImageStat,
) -> tuple[list[_ImageCandidate], int]:
    candidates: list[_ImageCandidate] = []
    skipped_corrupt = 0
    for record in records:
        try:
            with Image.open(record.path) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                stat = ImageStat.Stat(rgb.resize((1, 1)))
                brightness = float(sum(stat.mean) / 3.0)
                contrast = float(sum(ImageStat.Stat(rgb.convert("L")).stddev) / 1.0)
                edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES)
                blur_score = float(ImageStat.Stat(edges).mean[0])
        except Exception:
            skipped_corrupt += 1
            continue
        candidates.append(
            _ImageCandidate(
                record=record,
                image_id=_stable_image_id(dataset_id, record.class_name, record.relative_key),
                width=width,
                height=height,
                brightness=brightness,
                contrast=contrast,
                blur_score=blur_score,
            )
        )
    return candidates, skipped_corrupt


def _select_candidates(
    candidates: list[_ImageCandidate],
    class_records: dict[str, list[_ImageRecord]],
    total_limit: int,
    seed: int,
) -> list[_ImageCandidate]:
    if total_limit <= 0 or not candidates:
        return []

    by_class: dict[str, list[_ImageCandidate]] = {}
    for candidate in candidates:
        by_class.setdefault(candidate.record.class_name, []).append(candidate)
    for class_name, class_candidates in by_class.items():
        class_candidates.sort(
            key=lambda candidate: _stable_rank(seed, class_name, candidate.record.relative_key)
        )

    selected: dict[str, _ImageCandidate] = {}

    def add(candidate: _ImageCandidate, basis: str) -> None:
        existing = selected.get(candidate.image_id)
        if existing is not None:
            existing.selection_basis.add(basis)
            return
        if len(selected) >= total_limit:
            return
        candidate.selection_basis.add(basis)
        selected[candidate.image_id] = candidate

    if len(by_class) <= total_limit:
        for class_name in sorted(by_class):
            add(by_class[class_name][0], "class_representative")

    class_counts = {class_name: len(records) for class_name, records in class_records.items()}
    rare_classes = sorted(class_counts, key=lambda class_name: (class_counts[class_name], class_name))
    dominant_classes = sorted(
        class_counts,
        key=lambda class_name: (-class_counts[class_name], class_name),
    )
    for class_name in rare_classes[: max(1, min(4, total_limit))]:
        if class_name in by_class:
            add(by_class[class_name][0], "rare_class")
    for class_name in dominant_classes[: max(1, min(4, total_limit))]:
        if class_name in by_class:
            add(by_class[class_name][0], "dominant_class")

    _add_outliers(candidates, total_limit, add)

    class_names = sorted(by_class)
    class_offsets = {class_name: 0 for class_name in class_names}
    while len(selected) < total_limit:
        added_any = False
        for class_name in class_names:
            offset = class_offsets[class_name]
            class_candidates = by_class[class_name]
            if offset >= len(class_candidates):
                continue
            class_offsets[class_name] += 1
            add(class_candidates[offset], "class_representative")
            added_any = True
            if len(selected) >= total_limit:
                break
        if not added_any:
            break

    return sorted(
        selected.values(),
        key=lambda candidate: _stable_rank(seed, candidate.record.class_name, candidate.record.relative_key),
    )


def _add_outliers(candidates: list[_ImageCandidate], total_limit: int, add) -> None:
    outlier_specs = [
        ("aspect_ratio_outlier", sorted(candidates, key=lambda candidate: -candidate.aspect_deviation)),
        ("low_resolution_outlier", sorted(candidates, key=lambda candidate: candidate.area)),
        ("high_resolution_outlier", sorted(candidates, key=lambda candidate: -candidate.area)),
        ("brightness_outlier", sorted(candidates, key=lambda candidate: candidate.brightness)),
        ("contrast_outlier", sorted(candidates, key=lambda candidate: candidate.contrast)),
        ("blur_outlier", sorted(candidates, key=lambda candidate: candidate.blur_score)),
    ]
    per_basis = 2 if total_limit >= 16 else 1
    for basis, ranked_candidates in outlier_specs:
        for candidate in ranked_candidates[:per_basis]:
            add(candidate, basis)


def _prefers_high_detail(candidate: _ImageCandidate) -> bool:
    return bool(
        candidate.selection_basis
        & {
            "aspect_ratio_outlier",
            "low_resolution_outlier",
            "blur_outlier",
            "contrast_outlier",
        }
    )


def _is_edge_case_basis(selection_basis: list[str]) -> bool:
    return any(
        basis
        in {
            "aspect_ratio_outlier",
            "low_resolution_outlier",
            "high_resolution_outlier",
            "brightness_outlier",
            "contrast_outlier",
            "blur_outlier",
        }
        for basis in selection_basis
    )


def _encode_jpeg_input(
    source_path: Path,
    *,
    image_size: int,
    max_image_bytes: int,
    quality: int,
    Image,
) -> dict:
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        for candidate_quality in _quality_steps(quality):
            buffer = io.BytesIO()
            rgb.save(buffer, format="JPEG", quality=candidate_quality, optimize=True)
            payload = buffer.getvalue()
            if len(payload) <= max_image_bytes:
                return {
                    "width": rgb.width,
                    "height": rgb.height,
                    "bytes": len(payload),
                    "data_base64": base64.b64encode(payload).decode("ascii"),
                }
        raise ValueError("Compressed visual sample exceeded max_image_bytes.")


def _coverage_limitations(
    *,
    total_images: int,
    images_analyzed: int,
    classes_total: int,
    classes_covered: int,
    skipped_corrupt: int,
    skipped_over_budget: int,
    metadata_limited: bool,
) -> list[str]:
    limitations = ["Visual analysis is based on a bounded deterministic sample, not full inspection."]
    if images_analyzed < total_images:
        limitations.append("Sample is not image-complete.")
    if classes_covered < classes_total:
        limitations.append("Sample is not class-complete.")
    if skipped_corrupt:
        limitations.append(f"Skipped {skipped_corrupt} unreadable image(s) while building the sample.")
    if skipped_over_budget:
        limitations.append(f"Skipped {skipped_over_budget} encoded image(s) due to byte caps.")
    if metadata_limited:
        limitations.append("Risk scoring used a deterministic bounded metadata subset.")
    return limitations


def _quality_steps(quality: int) -> list[int]:
    start = max(35, min(95, int(quality)))
    return sorted({start, 78, 68, 55, 42, 35}, reverse=True)


def _stable_image_id(dataset_id: str, class_name: str, relative_key: str) -> str:
    payload = f"{dataset_id}:{class_name}:{relative_key}".encode("utf-8")
    return "img_" + hashlib.sha256(payload).hexdigest()[:12]


def _stable_rank(seed: int, class_name: str, relative_key: str) -> str:
    payload = f"{seed}:{class_name}:{relative_key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)
