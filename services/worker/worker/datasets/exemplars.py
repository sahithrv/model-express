from __future__ import annotations

import hashlib
from pathlib import Path

from worker.datasets.profiler import IMAGE_EXT, NON_CLASS_DIR_NAMES


def generate_visual_exemplars(
    *,
    dataset_dir: Path,
    output_dir: Path,
    images_per_class: int = 2,
    max_total_images: int = 12,
    max_image_bytes: int = 20_000,
    max_total_bytes: int = 150_000,
    image_size: int = 160,
    quality: int = 75,
    seed: int = 0,
) -> dict:
    """Create class-balanced, capped visual exemplars for profile/planner metadata."""
    try:
        from PIL import Image
    except Exception as exc:
        return {
            "schema_version": "visual_exemplar_pack_v1",
            "status": "unavailable",
            "error_code": "PIL_UNAVAILABLE",
            "error": str(exc),
            "visual_exemplars": [],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    per_class_limit = max(1, int(images_per_class))
    total_limit = max(0, int(max_total_images))
    total_byte_limit = max(0, int(max_total_bytes))
    image_byte_limit = max(1, int(max_image_bytes))
    target_size = max(16, int(image_size))

    exemplars: list[dict] = []
    skipped_corrupt = 0
    total_bytes = 0
    class_dirs = [
        path
        for path in sorted(dataset_dir.iterdir())
        if path.is_dir() and path.name.lower() not in NON_CLASS_DIR_NAMES
    ]

    for class_dir in class_dirs:
        if len(exemplars) >= total_limit:
            break
        selected = _deterministic_sample(_class_images(class_dir), per_class_limit, seed, class_dir.name)
        for source_path in selected:
            if len(exemplars) >= total_limit:
                break
            filename = _exemplar_filename(class_dir.name, source_path, len(exemplars))
            output_path = output_dir / filename
            try:
                width, height, encoded_bytes = _write_exemplar_image(
                    Image,
                    source_path,
                    output_path,
                    target_size,
                    image_byte_limit,
                    quality,
                )
            except Exception:
                skipped_corrupt += 1
                continue
            if total_bytes + encoded_bytes > total_byte_limit:
                output_path.unlink(missing_ok=True)
                break
            total_bytes += encoded_bytes
            exemplars.append(
                {
                    "id": f"{class_dir.name}:{source_path.stem}",
                    "class_name": class_dir.name,
                    "path": str(output_path),
                    "source_path": str(source_path),
                    "width": width,
                    "height": height,
                    "bytes": encoded_bytes,
                    "mime_type": "image/jpeg",
                }
            )

    return {
        "schema_version": "visual_exemplar_pack_v1",
        "status": "created",
        "visual_exemplars": exemplars,
        "summary": {
            "class_count": len(class_dirs),
            "image_count": len(exemplars),
            "total_bytes": total_bytes,
            "max_total_images": total_limit,
            "max_image_bytes": image_byte_limit,
            "max_total_bytes": total_byte_limit,
            "skipped_corrupt": skipped_corrupt,
            "sampling": "deterministic_class_balanced",
        },
    }


def _class_images(class_dir: Path) -> list[Path]:
    return [
        image_path
        for image_path in sorted(class_dir.rglob("*"))
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXT
    ]


def _deterministic_sample(paths: list[Path], limit: int, seed: int, class_name: str) -> list[Path]:
    ranked = sorted(paths, key=lambda path: _stable_rank(seed, class_name, path))
    return ranked[:limit]


def _stable_rank(seed: int, class_name: str, path: Path) -> str:
    payload = f"{seed}:{class_name}:{path.as_posix()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _exemplar_filename(class_name: str, source_path: Path, index: int) -> str:
    safe_class = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in class_name)
    digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:10]
    return f"{index:03d}_{safe_class}_{digest}.jpg"


def _write_exemplar_image(
    Image,
    source_path: Path,
    output_path: Path,
    image_size: int,
    max_image_bytes: int,
    quality: int,
) -> tuple[int, int, int]:
    with Image.open(source_path) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        for candidate_quality in _quality_steps(quality):
            rgb.save(output_path, format="JPEG", quality=candidate_quality, optimize=True)
            encoded_bytes = output_path.stat().st_size
            if encoded_bytes <= max_image_bytes:
                return rgb.width, rgb.height, encoded_bytes
        output_path.unlink(missing_ok=True)
        raise ValueError("Compressed exemplar exceeded max_image_bytes.")


def _quality_steps(quality: int) -> list[int]:
    start = max(35, min(95, int(quality)))
    return sorted({start, 70, 55, 40, 30}, reverse=True)
