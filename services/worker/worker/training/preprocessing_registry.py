from __future__ import annotations

from dataclasses import dataclass

from worker.training.augmentation import MIXED_SAMPLE_POLICY_TYPES, structured_policy_type


@dataclass(frozen=True)
class PreprocessingStrategySpec:
    kind: str
    name: str
    required_metadata: tuple[str, ...] = ()
    train_behavior: str = "same_as_eval"
    eval_behavior: str = "same_as_train"


RESIZE_STRATEGIES: dict[str, PreprocessingStrategySpec] = {
    "squash": PreprocessingStrategySpec("resize", "squash"),
    "preserve_aspect_pad": PreprocessingStrategySpec("resize", "preserve_aspect_pad"),
    "center_crop": PreprocessingStrategySpec(
        "resize",
        "center_crop",
        train_behavior="resize_then_center_crop",
        eval_behavior="resize_then_center_crop",
    ),
    "random_resized_crop": PreprocessingStrategySpec(
        "resize",
        "random_resized_crop",
        train_behavior="random_resized_crop",
        eval_behavior="squash",
    ),
    "bbox_crop_if_available": PreprocessingStrategySpec(
        "resize",
        "bbox_crop_if_available",
        required_metadata=("bounding_boxes",),
    ),
}

CROP_STRATEGIES: dict[str, PreprocessingStrategySpec] = {
    "none": PreprocessingStrategySpec("crop", "none"),
    "center_crop": PreprocessingStrategySpec(
        "crop",
        "center_crop",
        train_behavior="resize_then_center_crop",
        eval_behavior="resize_then_center_crop",
    ),
    "random_resized_crop": PreprocessingStrategySpec(
        "crop",
        "random_resized_crop",
        train_behavior="random_resized_crop",
        eval_behavior="squash",
    ),
    "bbox_crop_if_available": PreprocessingStrategySpec(
        "crop",
        "bbox_crop_if_available",
        required_metadata=("bounding_boxes",),
    ),
    "bbox_crop_ablation": PreprocessingStrategySpec(
        "crop",
        "bbox_crop_ablation",
        required_metadata=("bounding_boxes",),
        train_behavior="bbox_crop_and_compare_full_image",
        eval_behavior="bbox_crop_and_compare_full_image",
    ),
}

NORMALIZATION_STRATEGIES: dict[str, PreprocessingStrategySpec] = {
    "imagenet": PreprocessingStrategySpec("normalization", "imagenet"),
    "dataset": PreprocessingStrategySpec(
        "normalization",
        "dataset",
        required_metadata=("normalization_metadata",),
    ),
    "none": PreprocessingStrategySpec("normalization", "none"),
}

AUGMENTATION_STRATEGIES: dict[str, PreprocessingStrategySpec] = {
    "none": PreprocessingStrategySpec("augmentation", "none", train_behavior="no_op"),
    "custom": PreprocessingStrategySpec("augmentation", "custom", train_behavior="explicit_flags"),
    "light": PreprocessingStrategySpec("augmentation", "light", train_behavior="legacy_light"),
    "moderate": PreprocessingStrategySpec("augmentation", "moderate", train_behavior="legacy_moderate"),
    "strong": PreprocessingStrategySpec("augmentation", "strong", train_behavior="legacy_strong"),
    "basic": PreprocessingStrategySpec("augmentation", "basic", train_behavior="horizontal_flip"),
    "randaugment": PreprocessingStrategySpec(
        "augmentation",
        "randaugment",
        train_behavior="torchvision_randaugment",
        eval_behavior="no_op",
    ),
    "trivialaugment": PreprocessingStrategySpec(
        "augmentation",
        "trivialaugment",
        train_behavior="torchvision_trivialaugmentwide",
        eval_behavior="no_op",
    ),
    "autoaugment": PreprocessingStrategySpec(
        "augmentation",
        "autoaugment",
        train_behavior="torchvision_autoaugment",
        eval_behavior="no_op",
    ),
    "mixup": PreprocessingStrategySpec(
        "augmentation",
        "mixup",
        train_behavior="batch_mixed_sample",
        eval_behavior="no_op",
    ),
    "cutmix": PreprocessingStrategySpec(
        "augmentation",
        "cutmix",
        train_behavior="batch_mixed_sample",
        eval_behavior="no_op",
    ),
}

BBOX_MODES: dict[str, PreprocessingStrategySpec] = {
    "ignore": PreprocessingStrategySpec("bbox_mode", "ignore"),
    "crop_if_available": PreprocessingStrategySpec(
        "bbox_mode",
        "crop_if_available",
        required_metadata=("bounding_boxes",),
    ),
    "crop_and_compare_full_image": PreprocessingStrategySpec(
        "bbox_mode",
        "crop_and_compare_full_image",
        required_metadata=("bounding_boxes",),
        train_behavior="bbox_crop_and_compare_full_image",
        eval_behavior="bbox_crop_and_compare_full_image",
    ),
    "use_boxes_as_metadata": PreprocessingStrategySpec(
        "bbox_mode",
        "use_boxes_as_metadata",
        required_metadata=("bounding_boxes",),
        train_behavior="metadata_only",
        eval_behavior="metadata_only",
    ),
}

BBOX_CROP_STRATEGIES = {"bbox_crop_if_available", "bbox_crop_ablation"}
BBOX_CROP_MODES = {"crop_if_available", "crop_and_compare_full_image"}


def strategy_names(kind: str) -> set[str]:
    registry = _registry_for_kind(kind)
    return set(registry)


def all_strategy_names() -> dict[str, set[str]]:
    return {
        "resize": strategy_names("resize"),
        "crop": strategy_names("crop"),
        "normalization": strategy_names("normalization"),
        "augmentation": strategy_names("augmentation"),
        "bbox_mode": strategy_names("bbox_mode"),
    }


def normalize_preprocessing_config(preprocessing: object) -> dict:
    payload = dict(preprocessing) if isinstance(preprocessing, dict) else {}
    resize_strategy = _normalized_name(payload.get("resize_strategy"), default="squash")
    crop_strategy = _normalized_name(payload.get("crop_strategy"), default="none")
    normalization = _normalized_name(payload.get("normalization"), default="imagenet")
    bbox_mode = _normalized_name(payload.get("bbox_mode"), default="ignore")

    _require_supported("resize_strategy", resize_strategy, RESIZE_STRATEGIES)
    _require_supported("crop_strategy", crop_strategy, CROP_STRATEGIES)
    _require_supported("normalization", normalization, NORMALIZATION_STRATEGIES)
    _require_supported("bbox_mode", bbox_mode, BBOX_MODES)

    return {
        **payload,
        "resize_strategy": resize_strategy,
        "crop_strategy": crop_strategy,
        "normalization": normalization,
        "bbox_mode": bbox_mode,
    }


def validate_augmentation_strategy(augmentation: object) -> str:
    policy_type = structured_policy_type(augmentation if isinstance(augmentation, dict) else {})
    if not policy_type:
        return "custom" if isinstance(augmentation, dict) and augmentation else "none"
    _require_supported("augmentation policy_type", policy_type, AUGMENTATION_STRATEGIES)
    return policy_type


def bbox_crop_requested(preprocessing: object) -> bool:
    config = normalize_preprocessing_config(preprocessing)
    return (
        config["crop_strategy"] in BBOX_CROP_STRATEGIES
        or config["resize_strategy"] in BBOX_CROP_STRATEGIES
        or config["bbox_mode"] in BBOX_CROP_MODES
    )


def bbox_crop_required(preprocessing: object) -> bool:
    config = normalize_preprocessing_config(preprocessing)
    return (
        config["crop_strategy"] == "bbox_crop_ablation"
        or config["bbox_mode"] == "crop_and_compare_full_image"
        or config["resize_strategy"] == "bbox_crop_if_available"
        or config["crop_strategy"] == "bbox_crop_if_available"
        or config["bbox_mode"] == "crop_if_available"
    )


def bbox_compare_requested(preprocessing: object) -> bool:
    config = normalize_preprocessing_config(preprocessing)
    return (
        config["crop_strategy"] == "bbox_crop_ablation"
        or config["bbox_mode"] == "crop_and_compare_full_image"
    )


def build_image_transform(image_size: int, augmentation: dict, preprocessing: object, training: bool):
    from torchvision import transforms

    config = normalize_preprocessing_config(preprocessing)
    validate_augmentation_strategy(augmentation)

    steps = []
    resize_strategy = config["resize_strategy"]
    crop_strategy = config["crop_strategy"]
    random_crop = training and (
        bool(augmentation.get("random_crop"))
        or crop_strategy == "random_resized_crop"
        or resize_strategy == "random_resized_crop"
    )

    if random_crop:
        steps.append(transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))))
        steps.append(transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)))
    elif resize_strategy == "preserve_aspect_pad":
        steps.append(transforms.Lambda(lambda image: resize_with_padding(image, image_size)))
    elif crop_strategy == "center_crop" or resize_strategy == "center_crop":
        steps.append(transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))))
        steps.append(transforms.CenterCrop(image_size))
    else:
        steps.append(transforms.Resize((image_size, image_size)))

    if training and augmentation.get("horizontal_flip"):
        steps.append(transforms.RandomHorizontalFlip())
    if training and augmentation.get("vertical_flip"):
        steps.append(transforms.RandomVerticalFlip())
    if training and augmentation.get("color_jitter"):
        steps.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03))
    if training and augmentation.get("random_rotation"):
        steps.append(transforms.RandomRotation(10))
    if training:
        steps.extend(_advanced_augmentation_steps(transforms, augmentation))

    steps.append(transforms.ToTensor())
    if training and augmentation.get("random_erasing"):
        steps.append(transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)))
    values = normalization_values(config["normalization"], config)
    if values is not None:
        mean, std = values
        steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


def resize_with_padding(image, image_size: int):
    from PIL import Image

    image = image.convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def normalization_values(
    normalization: str,
    preprocessing: dict,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if normalization == "none":
        return None
    if normalization == "dataset":
        metadata = preprocessing.get("normalization_metadata")
        if isinstance(metadata, dict):
            mean = _three_float_tuple(metadata.get("mean"))
            std = _three_positive_float_tuple(metadata.get("std"))
            if mean is not None and std is not None:
                return mean, std
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _advanced_augmentation_steps(transforms, augmentation: dict) -> list:
    policy_type = structured_policy_type(augmentation)
    if policy_type in {"", "basic", "none", "custom", "light", "moderate", "strong"} | MIXED_SAMPLE_POLICY_TYPES:
        return []

    probability = float(augmentation.get("probability", 1.0))
    if policy_type == "randaugment":
        transform = _torchvision_transform(
            transforms,
            "RandAugment",
            num_ops=int(augmentation.get("num_ops", 2)),
            magnitude=int(augmentation.get("magnitude", 9)),
        )
    elif policy_type == "trivialaugment":
        transform = _torchvision_transform(
            transforms,
            "TrivialAugmentWide",
            num_magnitude_bins=int(augmentation.get("num_magnitude_bins", 31)),
        )
    elif policy_type == "autoaugment":
        transform = _autoaugment_transform(transforms, augmentation)
    else:
        raise ValueError(f"Unsupported image augmentation policy_type '{policy_type}'.")

    if probability >= 1.0:
        return [transform]
    return [transforms.RandomApply([transform], p=probability)]


def _torchvision_transform(transforms, name: str, **kwargs):
    transform_factory = getattr(transforms, name, None)
    if transform_factory is None:
        raise ValueError(f"torchvision.transforms.{name} is unavailable in this worker image.")
    return transform_factory(**kwargs)


def _autoaugment_transform(transforms, augmentation: dict):
    transform_factory = getattr(transforms, "AutoAugment", None)
    if transform_factory is None:
        raise ValueError("torchvision.transforms.AutoAugment is unavailable in this worker image.")

    policy_name = str(augmentation.get("autoaugment_policy") or "imagenet").strip().lower()
    policy_enum = getattr(transforms, "AutoAugmentPolicy", None)
    if policy_enum is None:
        return transform_factory()

    policies = {
        "imagenet": policy_enum.IMAGENET,
        "cifar10": policy_enum.CIFAR10,
        "svhn": policy_enum.SVHN,
    }
    if policy_name not in policies:
        raise ValueError("AutoAugment policy must be one of: imagenet, cifar10, svhn.")
    return transform_factory(policy=policies[policy_name])


def _registry_for_kind(kind: str) -> dict[str, PreprocessingStrategySpec]:
    registries = {
        "resize": RESIZE_STRATEGIES,
        "crop": CROP_STRATEGIES,
        "normalization": NORMALIZATION_STRATEGIES,
        "augmentation": AUGMENTATION_STRATEGIES,
        "bbox_mode": BBOX_MODES,
    }
    if kind not in registries:
        raise ValueError(f"Unknown preprocessing registry kind: {kind}")
    return registries[kind]


def _require_supported(
    field: str,
    value: str,
    registry: dict[str, PreprocessingStrategySpec],
) -> None:
    if value not in registry:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported {field} '{value}'. Supported values: {supported}.")


def _normalized_name(value: object, *, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip().lower().replace("-", "_")
    if not text:
        return default
    aliases = {
        "auto_augment": "autoaugment",
        "rand_augment": "randaugment",
        "trivial_augment": "trivialaugment",
        "trivialaugmentwide": "trivialaugment",
        "trivial_augment_wide": "trivialaugment",
    }
    return aliases.get(text, text)


def _three_float_tuple(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        parsed = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return parsed


def _three_positive_float_tuple(value: object) -> tuple[float, float, float] | None:
    parsed = _three_float_tuple(value)
    if parsed is None or any(item <= 0 for item in parsed):
        return None
    return parsed
