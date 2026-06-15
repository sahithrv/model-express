from __future__ import annotations

from pathlib import Path


class _BBoxCropDataset:
    def __init__(self, dataset, bbox_lookup: dict[str, tuple[int, int, int, int]], required: bool):
        self.dataset = dataset
        self.bbox_lookup = bbox_lookup
        self.required = required
        self.classes = getattr(dataset, "classes", [])
        self.targets = getattr(dataset, "targets", [])
        self.samples = getattr(dataset, "samples", [])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.dataset.loader(path)
        bbox = _bbox_for_image_path(path, self.bbox_lookup)
        if bbox is None:
            if self.required:
                raise ValueError(f"Missing bbox annotation for image '{path}'.")
        else:
            image = _crop_image_to_bbox(image, bbox)
        if self.dataset.transform is not None:
            image = self.dataset.transform(image)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return image, target


class _TransformedImageFolderView:
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
        self.target_transform = getattr(base_dataset, "target_transform", None)
        self.classes = getattr(base_dataset, "classes", [])
        self.class_to_idx = getattr(base_dataset, "class_to_idx", {})
        self.samples = getattr(base_dataset, "samples", [])
        self.imgs = self.samples
        self.targets = getattr(base_dataset, "targets", [])
        self.loader = getattr(base_dataset, "loader", None)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        if callable(self.loader):
            image = self.loader(path)
        else:
            image, target = self.base_dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def _bbox_for_image_path(path: str, lookup: dict[str, tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    image_path = Path(path)
    for key in (str(image_path.resolve()).lower(), image_path.name.lower(), image_path.stem.lower()):
        bbox = lookup.get(key)
        if bbox is not None:
            return bbox
    return None


def _crop_image_to_bbox(image, bbox: tuple[int, int, int, int]):
    image = image.convert("RGB")
    width, height = image.size
    xmin, ymin, xmax, ymax = bbox
    pad_x = max(1, int((xmax - xmin) * 0.05))
    pad_y = max(1, int((ymax - ymin) * 0.05))
    crop_box = (
        max(0, xmin - pad_x),
        max(0, ymin - pad_y),
        min(width, xmax + pad_x),
        min(height, ymax + pad_y),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        return image
    return image.crop(crop_box)
