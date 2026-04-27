from __future__ import annotations
from pathlib import Path
from PIL import Image

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def profile_image_folder(dataset_dir: Path) -> dict:
    class_counts: dict[str, int] = {}
    corrupt_images: list[str] = []

    widths: list[int] = []
    heights: list[int] = []

    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir():
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
            except Exception:
                corrupt_images.append(str(image_path))
    
    total_images = sum(class_counts.values())
    non_empty_counts = [count for count in class_counts.values() if count > 0]

    if non_empty_counts:
        imbalance_ratio = max(non_empty_counts) / min(non_empty_counts)
    else:
        imbalance_ratio = 0.0
    
    return {
        "dataset_path": str(dataset_dir),
        "task_type": "image_classification",
        "class_names": sorted(class_counts.keys()),
        "class_count": len(class_counts),
        "images_per_class": class_counts,
        "total_images": total_images,
        "corrupt_image_count": len(corrupt_images),
        "corrupt_images": corrupt_images[:25],
        "width_min": min(widths) if widths else None,
        "width_max": max(widths) if widths else None,
        "height_min": min(heights) if heights else None,
        "height_max": max(heights) if heights else None,
        "imbalance_ratio": round(imbalance_ratio, 3),
    }
