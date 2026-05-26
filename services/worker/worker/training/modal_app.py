from __future__ import annotations

import os
import random
from pathlib import Path

from worker.training.augmentation import (
    MIXED_SAMPLE_POLICY_TYPES,
    normalize_augmentation_config,
    structured_policy_type,
)

try:
    import modal
except Exception:  # pragma: no cover - local helper tests can run without Modal installed.
    modal = None


APP_NAME = "model-express-training"
DEFAULT_GPU = os.getenv("MODAL_GPU_TYPE", "T4")

EFFECTIVE_NUMBER_CLASS_BALANCING = {
    "effective_number",
    "effective_number_loss",
    "effective_number_class_balanced_loss",
    "class_balanced_loss",
    "class_balanced_effective_number",
}
BBOX_CROP_STRATEGIES = {"bbox_crop_if_available", "bbox_crop_ablation"}
BBOX_CROP_MODES = {"crop_if_available", "crop_and_compare_full_image"}

if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libglib2.0-0", "libgl1")
        .pip_install(
            "boto3",
            "numpy",
            "pillow",
            "requests",
            "scikit-learn",
            "torch",
            "torchvision",
        )
        .add_local_python_source("worker")
    )
    app = modal.App(APP_NAME)
else:
    image = None

    class _UnavailableModalApp:
        def function(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    app = _UnavailableModalApp()


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



@app.function(image=image, gpu=DEFAULT_GPU, timeout=60 * 60)
def train_image_classifier(payload: dict) -> dict:
    import requests
    import time
    import torch

    from worker.datasets.cache import dataset_archive_path, extract_dataset_archive
    from worker.datasets.storage import download_s3_uri

    started_at = time.time()
    job = payload["job"]
    config = job["config"]
    dataset = payload["dataset"]
    orchestrator_url = payload["orchestrator_url"].rstrip("/")

    _configure_storage_env(payload)

    job_id = job["id"]
    dataset_id = dataset["id"]
    epochs = _positive_int(config.get("epochs"), default=5)
    batch_size = _positive_int(config.get("batch_size"), default=16)
    learning_rate = _positive_float(config.get("learning_rate"), default=0.0003)
    image_size = _bounded_int(config.get("image_size"), default=224, minimum=96, maximum=384)
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    scheduler_name = str(config.get("scheduler", "none")).lower()
    weight_decay = _non_negative_float(config.get("weight_decay"), default=0.0)
    augmentation = normalize_augmentation_config(
        config.get("augmentation"),
        config.get("augmentation_policy", ""),
        config.get("augmentation_policy_config"),
    )
    class_balancing = str(config.get("class_balancing", "")).lower()
    sampling_strategy = str(config.get("sampling_strategy", "")).lower()
    preprocessing = config.get("preprocessing") if isinstance(config.get("preprocessing"), dict) else {}
    early_stopping_patience = _positive_int(config.get("early_stopping_patience"), default=0)
    model_name = str(config.get("model", "mobilenet_v3_small"))
    pretrained = _bool(config.get("pretrained"), default=True)
    freeze_backbone = _bool(config.get("freeze_backbone"), default=True)
    fine_tune_strategy = str(config.get("fine_tune_strategy", "head_only")).lower()
    gpu_type = str(config.get("gpu_type") or os.getenv("MODAL_GPU_TYPE", "T4"))
    modal_function_call_id, modal_input_id = _modal_identifiers()

    archive_path = dataset_archive_path(dataset_id)
    download_s3_uri(dataset["storage_uri"], archive_path)
    dataset_dir = extract_dataset_archive(archive_path, dataset_id)

    train_loader, val_loader, test_loader, class_names, class_weights, execution_metadata = _load_image_data(
        dataset_dir,
        batch_size,
        image_size,
        augmentation,
        class_balancing,
        sampling_strategy,
        preprocessing,
        config.get("class_balancing_config") if isinstance(config.get("class_balancing_config"), dict) else {},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_name, len(class_names), pretrained, freeze_backbone, fine_tune_strategy).to(device)

    criterion = _build_criterion(class_weights, class_balancing, device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = _build_optimizer(optimizer_name, trainable_parameters, learning_rate, weight_decay)
    scheduler = _build_scheduler(scheduler_name, optimizer, epochs)

    best_macro_f1 = 0.0
    best_accuracy = 0.0
    best_epoch = 0
    completed_epochs = 0
    final_eval_details = {"confusion_matrix": [], "per_class_metrics": {}}

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device, augmentation, len(class_names))
        val_loss, accuracy, macro_f1, final_eval_details = _evaluate(model, val_loader, criterion, device, class_names)
        improved = macro_f1 > best_macro_f1
        if improved:
            best_epoch = epoch
        best_macro_f1 = max(best_macro_f1, macro_f1)
        best_accuracy = max(best_accuracy, accuracy)
        completed_epochs = epoch
        if scheduler is not None:
            scheduler.step()
        runtime_seconds = time.time() - started_at
        estimated_cost_usd = runtime_seconds * _modal_gpu_price_per_second(gpu_type)

        _post_json(
            f"{orchestrator_url}/jobs/{job_id}/metrics",
            {
                "epoch": epoch,
                "metrics": {
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(val_loss, 6),
                    "accuracy": round(accuracy, 6),
                    "macro_f1": round(macro_f1, 6),
                    "best_accuracy": round(best_accuracy, 6),
                    "best_macro_f1": round(best_macro_f1, 6),
                    "learning_rate": learning_rate,
                    "image_size": image_size,
                    "weight_decay": weight_decay,
                },
            },
        )
        _post_training_run_summary(
            orchestrator_url,
            job_id,
            {
                "model": model_name,
                "provider": "modal",
                "gpu_type": gpu_type,
                "status": "RUNNING",
                "runtime_seconds": round(runtime_seconds, 3),
                "estimated_cost_usd": round(estimated_cost_usd, 6),
                "best_macro_f1": round(best_macro_f1, 6),
                "best_accuracy": round(best_accuracy, 6),
                "final_train_loss": round(train_loss, 6),
                "final_val_loss": round(val_loss, 6),
                "epochs_completed": epoch,
                "modal_function_call_id": modal_function_call_id,
                "modal_input_id": modal_input_id,
            },
        )
        if early_stopping_patience > 0 and epoch - best_epoch >= early_stopping_patience:
            break

    runtime_seconds = time.time() - started_at
    estimated_cost_usd = runtime_seconds * _modal_gpu_price_per_second(gpu_type)
    test_loss, test_accuracy, test_macro_f1, test_eval_details = _evaluate(
        model,
        test_loader,
        criterion,
        device,
        class_names,
        collect_examples=_label_quality_audit_requested(config),
    )
    if test_eval_details.get("confusion_matrix"):
        final_eval_details = test_eval_details
    bbox_ablation = _bbox_ablation_evaluation(
        model,
        execution_metadata,
        criterion,
        device,
        class_names,
        crop_metrics={"accuracy": test_accuracy, "macro_f1": test_macro_f1, "loss": test_loss},
    )
    model_profile = _model_profile(model, model_name, image_size, device, val_loader)
    _post_training_run_summary(
        orchestrator_url,
        job_id,
        {
            "model": model_name,
            "provider": "modal",
            "gpu_type": gpu_type,
            "status": "SUCCEEDED",
            "runtime_seconds": round(runtime_seconds, 3),
            "estimated_cost_usd": round(estimated_cost_usd, 6),
            "best_macro_f1": round(best_macro_f1, 6),
            "best_accuracy": round(best_accuracy, 6),
            "final_train_loss": round(train_loss, 6),
            "final_val_loss": round(val_loss, 6),
            "epochs_completed": completed_epochs,
            "modal_function_call_id": modal_function_call_id,
            "modal_input_id": modal_input_id,
        },
    )
    _post_training_run_evaluation(
        orchestrator_url,
        job_id,
        {
            "objective_profile": {
                "target_metric": str(config.get("target_metric", "macro_f1")),
                "metric_preferences": ["macro_f1", "accuracy", "per_class_f1", "latency"],
                "split_strategy": "train_validation_with_heldout_test_when_possible",
                "heldout_test_accuracy": round(test_accuracy, 6),
                "heldout_test_macro_f1": round(test_macro_f1, 6),
                "heldout_test_loss": round(test_loss, 6),
            },
            "per_class_metrics": final_eval_details.get("per_class_metrics", {}),
            "confusion_matrix": final_eval_details.get("confusion_matrix", []),
            "model_profile": {
                **model_profile,
                "pretrained": pretrained,
                "freeze_backbone": freeze_backbone,
                "fine_tune_strategy": fine_tune_strategy,
            },
            "holistic_scores": _holistic_scores(best_macro_f1, best_accuracy, estimated_cost_usd, runtime_seconds, model_profile),
            "preprocessing_summary": {
                "augmentation_policy": str(config.get("augmentation_policy", "")),
                "augmentation_policy_config": config.get("augmentation_policy_config")
                if isinstance(config.get("augmentation_policy_config"), dict)
                else {},
                "class_balancing": class_balancing,
                "sampling_strategy": sampling_strategy,
                "preprocessing": preprocessing,
                "worker_execution_metadata": _public_execution_metadata(execution_metadata),
                "bbox_crop_ablation": bbox_ablation,
            },
            "label_quality_audit": _label_quality_audit(config, test_eval_details, class_names),
            "recommendation_summary": (
                f"{model_name} finished with macro-F1 {best_macro_f1:.3f}, "
                f"accuracy {best_accuracy:.3f}, and estimated latency "
                f"{model_profile.get('estimated_latency_ms', 0):.1f}ms."
            ),
        },
    )

    _post_json(
        f"{orchestrator_url}/jobs/{job_id}/complete",
        {"mlflow_run_id": f"modal-{job_id}"},
    )

    return {
        "job_id": job_id,
        "model": model_name,
        "classes": class_names,
        "best_accuracy": best_accuracy,
        "best_macro_f1": best_macro_f1,
        "estimated_cost_usd": estimated_cost_usd,
        "runtime_seconds": runtime_seconds,
        "device": str(device),
    }


@app.function(image=image, timeout=60 * 20)
def profile_image_dataset(payload: dict) -> dict:
    import tempfile

    from worker.datasets.cache import dataset_archive_path, extract_dataset_archive
    from worker.datasets.profiler import profile_image_folder
    from worker.datasets.storage import download_s3_uri

    _configure_storage_env(payload)

    dataset = payload["dataset"]
    dataset_id = dataset["id"]
    with tempfile.TemporaryDirectory(prefix=f"model-express-profile-{dataset_id}-") as cache_root:
        archive_path = dataset_archive_path(dataset_id, cache_root)
        download_s3_uri(dataset["storage_uri"], archive_path)
        dataset_dir = extract_dataset_archive(archive_path, dataset_id, cache_root)
        return profile_image_folder(dataset_dir)


def _configure_storage_env(payload: dict) -> None:
    os.environ["S3_ENDPOINT_URL"] = payload["s3_endpoint_url"]
    os.environ["AWS_ACCESS_KEY_ID"] = payload["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = payload["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = payload["aws_default_region"]


def _load_image_data(
    dataset_dir: Path,
    batch_size: int,
    image_size: int,
    augmentation: dict,
    class_balancing: str,
    sampling_strategy: str,
    preprocessing: dict,
    class_balancing_config: dict | None = None,
):
    import torch
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from torchvision import datasets

    normalization_metadata = _dataset_normalization_metadata(dataset_dir, preprocessing)
    if normalization_metadata is not None:
        preprocessing = {**preprocessing, "normalization_metadata": normalization_metadata}
    train_transform = _image_transform(image_size, augmentation, preprocessing, training=True)
    val_transform = _image_transform(image_size, {}, preprocessing, training=False)
    crop_strategy = str(preprocessing.get("crop_strategy", "")).lower()
    bbox_mode = str(preprocessing.get("bbox_mode", "")).lower()
    bbox_crop_requested = crop_strategy in BBOX_CROP_STRATEGIES or bbox_mode in BBOX_CROP_MODES
    bbox_required = crop_strategy == "bbox_crop_ablation" or bbox_mode == "crop_and_compare_full_image"
    bbox_compare_requested = crop_strategy == "bbox_crop_ablation" or bbox_mode == "crop_and_compare_full_image"
    bbox_lookup: dict[str, tuple[int, int, int, int]] = {}
    execution_metadata: dict = {
        "bbox_crop": {
            "requested": bbox_crop_requested,
            "applied": False,
            "annotation_count": 0,
            "mode": bbox_mode or crop_strategy or "none",
            "compare_full_image": bbox_compare_requested,
        }
    }
    if bbox_crop_requested:
        bbox_lookup = _load_bbox_lookup(dataset_dir)
        execution_metadata["bbox_crop"]["annotation_count"] = len(bbox_lookup)
        if not bbox_lookup:
            message = (
                "BBox crop was requested, but no Pascal VOC XML or annotation JSON bounding boxes "
                "were found in the extracted dataset."
            )
            execution_metadata["bbox_crop"]["status"] = "missing_annotations"
            if bbox_required or crop_strategy == "bbox_crop_if_available" or bbox_mode == "crop_if_available":
                raise ValueError(message)
        else:
            execution_metadata["bbox_crop"]["applied"] = True
            execution_metadata["bbox_crop"]["status"] = "applied"

    uses_metadata_aware_dataset = _has_root_metadata_dirs(dataset_dir)
    base_dataset = _image_folder_dataset(datasets, dataset_dir)
    if len(base_dataset.classes) < 2:
        raise ValueError("Training requires at least two image classes.")
    if len(base_dataset) < 2:
        raise ValueError("Training requires at least two images.")

    validation_size = max(1, int(len(base_dataset) * 0.2))
    test_size = max(1, int(len(base_dataset) * 0.1)) if len(base_dataset) >= 6 else 0
    training_size = len(base_dataset) - validation_size - test_size
    if training_size < 1:
        test_size = 0
        validation_size = 1
        training_size = len(base_dataset) - validation_size

    generator = torch.Generator().manual_seed(42)
    shuffled_indices = torch.randperm(len(base_dataset), generator=generator).tolist()
    train_indices = shuffled_indices[:training_size]
    val_start = training_size
    val_end = val_start + validation_size
    val_indices = shuffled_indices[val_start:val_end]
    test_indices = shuffled_indices[val_end : val_end + test_size]
    if not test_indices:
        test_indices = val_indices

    train_dataset = _image_folder_dataset(datasets, dataset_dir, transform=train_transform)
    val_dataset = _image_folder_dataset(datasets, dataset_dir, transform=val_transform)
    full_image_val_dataset = _image_folder_dataset(datasets, dataset_dir, transform=val_transform)
    if bbox_lookup:
        train_dataset = _BBoxCropDataset(train_dataset, bbox_lookup, required=bbox_required)
        val_dataset = _BBoxCropDataset(val_dataset, bbox_lookup, required=bbox_required)
    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(val_dataset, val_indices)
    test_data = Subset(val_dataset, test_indices)
    full_image_val_data = Subset(full_image_val_dataset, val_indices)
    full_image_test_data = Subset(full_image_val_dataset, test_indices)

    sampler = None
    if _uses_weighted_sampler(class_balancing, sampling_strategy):
        counts = torch.zeros(len(base_dataset.classes), dtype=torch.float32)
        for index in train_indices:
            counts[int(base_dataset.targets[index])] += 1.0
        counts = torch.clamp(counts, min=1.0)
        weights = [float(1.0 / counts[int(base_dataset.targets[index])]) for index in train_indices]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    loader_workers = 0 if uses_metadata_aware_dataset else 2
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=loader_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=loader_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=loader_workers)
    if bbox_lookup and bbox_compare_requested:
        execution_metadata["_full_image_val_loader"] = DataLoader(
            full_image_val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=loader_workers,
        )
        execution_metadata["_full_image_test_loader"] = DataLoader(
            full_image_test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=loader_workers,
        )
    class_weights = _class_weights(
        base_dataset.targets,
        train_indices,
        len(base_dataset.classes),
        class_balancing,
        class_balancing_config or {},
    )
    return train_loader, val_loader, test_loader, base_dataset.classes, class_weights, execution_metadata


def _augmentation_from_policy(augmentation: dict, policy: str) -> dict:
    return normalize_augmentation_config(augmentation, policy)


def _image_folder_dataset(datasets, dataset_dir: Path, transform=None):
    metadata_names = {"annotation", "annotations", "metadata", "meta", "splits"}
    if not _has_root_metadata_dirs(dataset_dir):
        return datasets.ImageFolder(dataset_dir, transform=transform)

    class MetadataAwareImageFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            classes = sorted(
                entry.name
                for entry in os.scandir(directory)
                if entry.is_dir() and entry.name.lower() not in metadata_names
            )
            if not classes:
                raise FileNotFoundError(f"No class folders found in {directory}.")
            class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
            return classes, class_to_idx

    return MetadataAwareImageFolder(dataset_dir, transform=transform)


def _has_root_metadata_dirs(dataset_dir: Path) -> bool:
    metadata_names = {"annotation", "annotations", "metadata", "meta", "splits"}
    return any(child.is_dir() and child.name.lower() in metadata_names for child in dataset_dir.iterdir())


def _image_transform(image_size: int, augmentation: dict, preprocessing: dict, training: bool):
    from torchvision import transforms

    steps = []
    resize_strategy = str(preprocessing.get("resize_strategy", "")).lower()
    crop_strategy = str(preprocessing.get("crop_strategy", "")).lower()
    normalization = str(preprocessing.get("normalization", "imagenet")).lower()
    random_crop = training and (augmentation.get("random_crop") or crop_strategy == "random_resized_crop")

    if random_crop or (training and resize_strategy == "random_resized_crop"):
        steps.append(transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))))
        steps.append(transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)))
    elif resize_strategy == "preserve_aspect_pad":
        steps.append(transforms.Lambda(lambda image: _resize_with_padding(image, image_size)))
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
    normalization_values = _normalization_values(normalization, preprocessing)
    if normalization_values is not None:
        mean, std = normalization_values
        steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


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


def _resize_with_padding(image, image_size: int):
    from PIL import Image

    image = image.convert("RGB")
    image.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    left = (image_size - image.width) // 2
    top = (image_size - image.height) // 2
    canvas.paste(image, (left, top))
    return canvas


def _load_bbox_lookup(dataset_dir: Path) -> dict[str, tuple[int, int, int, int]]:
    from worker.datasets.annotations import parse_annotation_json_bboxes, parse_pascal_voc_bboxes

    lookup: dict[str, tuple[int, int, int, int]] = {}
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.suffix.lower() == ".xml":
                payload = parse_pascal_voc_bboxes(path)
            elif path.suffix.lower() == ".json" and "annotation" in path.name.lower():
                payload = parse_annotation_json_bboxes(path)
            else:
                continue
        except Exception:
            continue
        bbox = _union_annotation_bbox(payload.get("objects"))
        if bbox is None:
            continue
        filename = str(payload.get("filename") or "").strip()
        keys = {path.stem.lower(), path.name.lower()}
        if filename:
            keys.add(filename.lower())
            keys.add(Path(filename).stem.lower())
            matching_path = _resolve_annotation_image_path(dataset_dir, filename)
            if matching_path is not None:
                keys.add(str(matching_path.resolve()).lower())
        for key in keys:
            if key:
                lookup[key] = bbox
    return lookup


def _resolve_annotation_image_path(dataset_dir: Path, filename: str) -> Path | None:
    candidate = dataset_dir / filename
    if candidate.exists():
        return candidate
    filename_lower = Path(filename).name.lower()
    for image_path in dataset_dir.rglob("*"):
        if image_path.is_file() and image_path.name.lower() == filename_lower:
            return image_path
    return None


def _union_annotation_bbox(objects: object) -> tuple[int, int, int, int] | None:
    if not isinstance(objects, list):
        return None
    boxes = []
    for item in objects:
        if not isinstance(item, dict) or not isinstance(item.get("bbox"), dict):
            continue
        bbox = item["bbox"]
        try:
            xmin = int(bbox["xmin"])
            ymin = int(bbox["ymin"])
            xmax = int(bbox["xmax"])
            ymax = int(bbox["ymax"])
        except (KeyError, TypeError, ValueError):
            continue
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


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


def _class_weights(
    targets: list[int],
    train_indices: list[int],
    class_count: int,
    class_balancing: str,
    class_balancing_config: dict | None = None,
):
    weighted_modes = {"weighted_loss", "class_weighted_loss", "focal_loss"} | EFFECTIVE_NUMBER_CLASS_BALANCING
    if class_balancing not in weighted_modes:
        return None

    import torch

    counts = torch.zeros(class_count, dtype=torch.float32)
    for index in train_indices:
        counts[int(targets[index])] += 1.0
    counts = torch.clamp(counts, min=1.0)
    if class_balancing in EFFECTIVE_NUMBER_CLASS_BALANCING:
        beta = _bounded_float(
            (class_balancing_config or {}).get("effective_number_beta"),
            default=0.9999,
            minimum=0.9,
            maximum=0.99999,
        )
        effective_number = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
        weights = (1.0 - beta) / torch.clamp(effective_number, min=1e-8)
        return weights * (class_count / torch.clamp(weights.sum(), min=1e-8))
    weights = counts.sum() / (counts * class_count)
    return weights


def _uses_weighted_sampler(class_balancing: str, sampling_strategy: str) -> bool:
    return sampling_strategy in {"class_balanced_sampler", "weighted_random_sampler"} or class_balancing in {
        "class_balanced_sampler",
        "weighted_random_sampler",
    }


def _dataset_normalization_metadata(dataset_dir: Path, preprocessing: dict) -> dict | None:
    normalization = str(preprocessing.get("normalization", "imagenet")).lower()
    use_dataset_normalization = _bool(preprocessing.get("use_dataset_normalization"), default=False)
    if normalization != "dataset" and not use_dataset_normalization:
        return None

    from worker.datasets.profiler import compute_image_normalization_metadata

    metadata = compute_image_normalization_metadata(dataset_dir)
    return metadata if metadata.get("status") == "computed" else None


def _normalization_values(normalization: str, preprocessing: dict) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
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


def _build_criterion(class_weights, class_balancing: str, device):
    import torch
    from torch import nn
    import torch.nn.functional as F

    weight_tensor = class_weights.to(device) if class_weights is not None else None
    if class_balancing == "focal_loss":
        class FocalLoss(nn.Module):
            def __init__(self, weight=None, gamma: float = 2.0):
                super().__init__()
                self.weight = weight
                self.gamma = gamma

            def forward(self, logits, targets):
                if targets.dtype.is_floating_point:
                    log_probabilities = F.log_softmax(logits, dim=1)
                    weights = self.weight.view(1, -1) if self.weight is not None else 1.0
                    cross_entropy = -(targets * log_probabilities * weights).sum(dim=1)
                else:
                    cross_entropy = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
                probability = torch.exp(-cross_entropy)
                loss = ((1 - probability) ** self.gamma) * cross_entropy
                return loss.mean()

        return FocalLoss(weight=weight_tensor)
    return nn.CrossEntropyLoss(weight=weight_tensor)


def _apply_mixed_sample_augmentation(inputs, labels, augmentation: dict, class_count: int, device):
    import torch
    import torch.nn.functional as F

    policy_type = structured_policy_type(augmentation)
    if policy_type not in MIXED_SAMPLE_POLICY_TYPES or inputs.size(0) < 2:
        return inputs, labels
    probability = float(augmentation.get("probability", 1.0))
    if probability <= 0.0 or random.random() > probability:
        return inputs, labels
    alpha = float(augmentation.get("alpha", 0.2))
    if alpha <= 0.0:
        return inputs, labels

    beta_distribution = torch.distributions.Beta(alpha, alpha)
    lam = float(beta_distribution.sample().item())
    permutation = torch.randperm(inputs.size(0), device=device)
    labels_one_hot = F.one_hot(labels, num_classes=class_count).to(dtype=inputs.dtype)
    mixed_labels = (lam * labels_one_hot) + ((1.0 - lam) * labels_one_hot[permutation])

    if policy_type == "mixup":
        mixed_inputs = (lam * inputs) + ((1.0 - lam) * inputs[permutation])
        return mixed_inputs, mixed_labels

    mixed_inputs, adjusted_lam = _apply_cutmix(inputs, permutation, lam)
    mixed_labels = (adjusted_lam * labels_one_hot) + ((1.0 - adjusted_lam) * labels_one_hot[permutation])
    return mixed_inputs, mixed_labels


def _apply_cutmix(inputs, permutation, lam: float):
    _, _, height, width = inputs.shape
    cut_ratio = (1.0 - lam) ** 0.5
    cut_width = max(1, int(width * cut_ratio))
    cut_height = max(1, int(height * cut_ratio))
    center_x = random.randint(0, max(width - 1, 0))
    center_y = random.randint(0, max(height - 1, 0))
    x1 = max(center_x - cut_width // 2, 0)
    y1 = max(center_y - cut_height // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y2 = min(center_y + cut_height // 2, height)
    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[permutation, :, y1:y2, x1:x2]
    area = max(0, x2 - x1) * max(0, y2 - y1)
    adjusted_lam = 1.0 - (area / max(1, width * height))
    return mixed_inputs, float(adjusted_lam)


def _build_model(model_name: str, class_count: int, pretrained: bool = True, freeze_backbone: bool = True, fine_tune_strategy: str = "head_only"):
    from torch import nn
    from torchvision import models

    normalized = model_name.lower()

    if "efficientnet_b2" in normalized:
        model = _torchvision_model(models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
        head = nn.Linear(model.classifier[-1].in_features, class_count)
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        model.classifier[-1] = head
        return model
    if "efficientnet_b1" in normalized:
        model = _torchvision_model(models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)
        head = nn.Linear(model.classifier[-1].in_features, class_count)
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        model.classifier[-1] = head
        return model
    if "efficientnet" in normalized:
        model = _torchvision_model(models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        head = nn.Linear(model.classifier[-1].in_features, class_count)
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        model.classifier[-1] = head
        return model
    if "resnet34" in normalized:
        model = _torchvision_model(models.resnet34, models.ResNet34_Weights.DEFAULT if pretrained else None)
        head = nn.Linear(model.fc.in_features, class_count)
        _apply_transfer_strategy(model, "fc", freeze_backbone, fine_tune_strategy)
        model.fc = head
        return model
    if "resnet" in normalized:
        model = _torchvision_model(models.resnet18, models.ResNet18_Weights.DEFAULT if pretrained else None)
        head = nn.Linear(model.fc.in_features, class_count)
        _apply_transfer_strategy(model, "fc", freeze_backbone, fine_tune_strategy)
        model.fc = head
        return model
    if "regnet_y_400mf" in normalized:
        model = _torchvision_model(models.regnet_y_400mf, models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None)
        head = nn.Linear(model.fc.in_features, class_count)
        _apply_transfer_strategy(model, "fc", freeze_backbone, fine_tune_strategy)
        model.fc = head
        return model
    if "convnext_tiny" in normalized:
        model = _torchvision_model(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        head = nn.Linear(model.classifier[-1].in_features, class_count)
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        model.classifier[-1] = head
        return model
    if "swin_t" in normalized:
        model = _torchvision_model(models.swin_t, models.Swin_T_Weights.DEFAULT if pretrained else None)
        head = nn.Linear(model.head.in_features, class_count)
        _apply_transfer_strategy(model, "head", freeze_backbone, fine_tune_strategy)
        model.head = head
        return model
    if "vit_b_16" in normalized:
        model = _torchvision_model(models.vit_b_16, models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        head = nn.Linear(model.heads.head.in_features, class_count)
        _apply_transfer_strategy(model, "heads", freeze_backbone, fine_tune_strategy)
        model.heads.head = head
        return model
    if "mobilenet_v3_large" in normalized:
        model = _torchvision_model(models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        head = nn.Linear(model.classifier[-1].in_features, class_count)
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        model.classifier[-1] = head
        return model

    model = _torchvision_model(models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    head = nn.Linear(model.classifier[-1].in_features, class_count)
    _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
    model.classifier[-1] = head
    return model


def _torchvision_model(factory, weights):
    try:
        return factory(weights=weights)
    except Exception:
        return factory(weights=None)


def _apply_transfer_strategy(model, head_name: str, freeze_backbone: bool, fine_tune_strategy: str) -> None:
    if not freeze_backbone or fine_tune_strategy == "full":
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
    if fine_tune_strategy == "last_block":
        children = list(model.children())
        if len(children) > 1:
            for parameter in children[-2].parameters():
                parameter.requires_grad = True
    head = getattr(model, head_name, None)
    if head is not None:
        for parameter in head.parameters():
            parameter.requires_grad = True


def _build_optimizer(optimizer_name: str, parameters, learning_rate: float, weight_decay: float):
    import torch

    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def _build_scheduler(scheduler_name: str, optimizer, epochs: int):
    import torch

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    return None


def _train_one_epoch(model, loader, criterion, optimizer, device, augmentation: dict | None = None, class_count: int = 0) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    augmentation = augmentation or {}

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs, loss_labels = _apply_mixed_sample_augmentation(inputs, labels, augmentation, max(1, class_count), device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, loss_labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def _evaluate(
    model,
    loader,
    criterion,
    device,
    class_names: list[str],
    collect_examples: bool = False,
) -> tuple[float, float, float, dict]:
    import torch
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions = []
    targets = []
    confidences = []
    sample_paths = _sample_paths_from_loader(loader) if collect_examples else []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

            predicted = outputs.argmax(dim=1)
            probabilities = torch.softmax(outputs, dim=1)
            batch_confidences = probabilities.max(dim=1).values
            predictions.extend(predicted.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            confidences.extend(batch_confidences.cpu().tolist())

    correct = sum(1 for prediction, target in zip(predictions, targets) if prediction == target)
    accuracy = correct / max(len(targets), 1)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    labels = list(range(len(class_names)))
    details = {
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist() if targets else [],
        "per_class_metrics": classification_report(
            targets,
            predictions,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        if targets
        else {},
    }
    if collect_examples:
        details["example_predictions"] = _example_prediction_records(
            predictions,
            targets,
            confidences,
            sample_paths,
            class_names,
        )
    return total_loss / max(total_examples, 1), accuracy, float(macro_f1), details


def _sample_paths_from_loader(loader) -> list[str]:
    dataset = getattr(loader, "dataset", None)
    indices = getattr(dataset, "indices", None)
    source = getattr(dataset, "dataset", dataset)
    samples = getattr(source, "samples", None)
    if not isinstance(samples, list):
        return []
    if indices is None:
        return [str(sample[0]) for sample in samples]
    return [str(samples[int(index)][0]) for index in indices if int(index) < len(samples)]


def _example_prediction_records(
    predictions: list[int],
    targets: list[int],
    confidences: list[float],
    sample_paths: list[str],
    class_names: list[str],
) -> list[dict]:
    records = []
    for index, (prediction, target) in enumerate(zip(predictions, targets)):
        confidence = float(confidences[index]) if index < len(confidences) else 0.0
        records.append(
            {
                "path": sample_paths[index] if index < len(sample_paths) else "",
                "predicted_class": class_names[prediction] if prediction < len(class_names) else str(prediction),
                "true_class": class_names[target] if target < len(class_names) else str(target),
                "confidence": round(confidence, 6),
                "correct": prediction == target,
            }
        )
    return records


def _label_quality_audit_requested(config: dict) -> bool:
    mechanism = str(config.get("mechanism", "")).lower()
    audit_config = config.get("label_quality_audit") if isinstance(config.get("label_quality_audit"), dict) else {}
    if mechanism in {"label_noise_audit", "hard_example_audit"}:
        return True
    if audit_config:
        return _bool(audit_config.get("enabled"), default=False)
    return _bool(config.get("label_quality_audit"), default=False)


def _label_quality_audit(config: dict, eval_details: dict, class_names: list[str]) -> dict:
    if not _label_quality_audit_requested(config):
        return {
            "status": "not_requested",
            "report_only": True,
        }
    records = eval_details.get("example_predictions") if isinstance(eval_details, dict) else []
    if not isinstance(records, list):
        records = []
    high_confidence_wrong = [
        record
        for record in records
        if not record.get("correct") and float(record.get("confidence") or 0.0) >= 0.7
    ]
    low_confidence_correct = [
        record
        for record in records
        if record.get("correct") and float(record.get("confidence") or 0.0) <= 0.55
    ]
    hard_examples = sorted(
        records,
        key=lambda record: (bool(record.get("correct")), -float(record.get("confidence") or 0.0)),
    )
    return {
        "status": "completed",
        "report_only": True,
        "sample_count": len(records),
        "class_names": class_names,
        "high_confidence_wrong": high_confidence_wrong[:25],
        "low_confidence_correct": low_confidence_correct[:25],
        "hard_examples": hard_examples[:50],
        "notes": "Audit artifacts are report-only; worker does not mutate datasets or labels.",
    }


def _bbox_ablation_evaluation(model, execution_metadata: dict, criterion, device, class_names: list[str], crop_metrics: dict) -> dict:
    bbox_metadata = execution_metadata.get("bbox_crop") if isinstance(execution_metadata, dict) else {}
    if not isinstance(bbox_metadata, dict) or not bbox_metadata.get("compare_full_image"):
        return {"status": "not_requested"}
    full_image_loader = execution_metadata.get("_full_image_test_loader")
    if full_image_loader is None:
        return {
            "status": "unavailable",
            "reason": "full_image_loader_missing",
            "crop_metrics": crop_metrics,
        }
    full_loss, full_accuracy, full_macro_f1, _details = _evaluate(
        model,
        full_image_loader,
        criterion,
        device,
        class_names,
    )
    return {
        "status": "completed",
        "crop_metrics": {
            "loss": round(float(crop_metrics.get("loss") or 0.0), 6),
            "accuracy": round(float(crop_metrics.get("accuracy") or 0.0), 6),
            "macro_f1": round(float(crop_metrics.get("macro_f1") or 0.0), 6),
        },
        "full_image_metrics": {
            "loss": round(float(full_loss), 6),
            "accuracy": round(float(full_accuracy), 6),
            "macro_f1": round(float(full_macro_f1), 6),
        },
    }


def _public_execution_metadata(execution_metadata: dict) -> dict:
    return {key: value for key, value in execution_metadata.items() if not key.startswith("_")}


def _model_profile(model, model_name: str, image_size: int, device, loader) -> dict:
    import io
    import statistics
    import time
    import torch

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    model_size_mb = len(buffer.getvalue()) / (1024 * 1024)
    latency_samples = []
    model.eval()
    sample = None
    for inputs, _labels in loader:
        sample = inputs[:1].to(device)
        break
    if sample is not None:
        with torch.no_grad():
            for _ in range(2):
                _ = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            for _ in range(8):
                started = time.perf_counter()
                _ = model(sample)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latency_samples.append((time.perf_counter() - started) * 1000)
    estimated_latency_ms = statistics.median(latency_samples) if latency_samples else _fallback_latency_ms(model_name, image_size)
    return {
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "model_size_mb": round(model_size_mb, 3),
        "estimated_latency_ms": round(float(estimated_latency_ms), 3),
        "estimated_throughput_images_per_second": round(1000.0 / max(float(estimated_latency_ms), 1.0), 3),
        "image_size": image_size,
    }


def _holistic_scores(best_macro_f1: float, best_accuracy: float, estimated_cost_usd: float, runtime_seconds: float, model_profile: dict) -> dict:
    latency_ms = float(model_profile.get("estimated_latency_ms") or 0)
    quality_score = (best_macro_f1 * 0.65) + (best_accuracy * 0.35)
    latency_score = max(0.0, min(1.0, 1.0 - latency_ms / 160.0))
    cost_score = max(0.0, min(1.0, 1.0 - estimated_cost_usd / 10.0))
    runtime_score = max(0.0, min(1.0, 1.0 - runtime_seconds / 1800.0))
    overall_score = (quality_score * 0.62) + (latency_score * 0.18) + (cost_score * 0.12) + (runtime_score * 0.08)
    return {
        "quality_score": round(quality_score, 6),
        "latency_score": round(latency_score, 6),
        "cost_score": round(cost_score, 6),
        "runtime_score": round(runtime_score, 6),
        "overall_score": round(overall_score, 6),
    }


def _fallback_latency_ms(model_name: str, image_size: int) -> float:
    normalized = model_name.lower()
    base = 24.0
    if "mobilenet_v3_small" in normalized:
        base = 8.0
    elif "mobilenet_v3_large" in normalized:
        base = 12.0
    elif "efficientnet_b0" in normalized or "regnet" in normalized:
        base = 15.0
    elif "efficientnet_b1" in normalized:
        base = 22.0
    elif "efficientnet_b2" in normalized:
        base = 28.0
    elif "resnet34" in normalized:
        base = 38.0
    elif "convnext" in normalized:
        base = 55.0
    elif "swin" in normalized:
        base = 64.0
    elif "vit" in normalized:
        base = 95.0
    return base * max(0.5, (image_size / 224) ** 2)


def _post_json(url: str, payload: dict) -> None:
    import requests

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()


def _post_training_run_summary(orchestrator_url: str, job_id: str, payload: dict) -> None:
    _post_json(f"{orchestrator_url}/jobs/{job_id}/training-run-summary", payload)


def _post_training_run_evaluation(orchestrator_url: str, job_id: str, payload: dict) -> None:
    _post_json(f"{orchestrator_url}/jobs/{job_id}/training-run-evaluation", payload)


def _modal_gpu_price_per_second(gpu_type: str) -> float:
    base_type = gpu_type.split(":", 1)[0].upper()
    prices = {
        "T4": 0.000164,
        "L4": 0.000222,
        "A10": 0.000306,
        "L40S": 0.000542,
        "A100": 0.000583,
        "A100-40GB": 0.000583,
        "A100-80GB": 0.000694,
        "RTX-PRO-6000": 0.000842,
        "H100": 0.001097,
        "H200": 0.001261,
        "B200": 0.001736,
    }
    return prices.get(base_type, prices["T4"])


def _modal_identifiers() -> tuple[str, str]:
    try:
        import modal

        return modal.current_function_call_id() or "", modal.current_input_id() or ""
    except Exception:
        return "", ""


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _non_negative_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    parsed = _positive_int(value, default=default)
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))
