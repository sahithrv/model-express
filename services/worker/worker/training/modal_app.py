from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "model-express-training"
DEFAULT_GPU = os.getenv("MODAL_GPU_TYPE", "T4")

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


@app.function(image=image, gpu=DEFAULT_GPU, timeout=60 * 60)
def train_image_classifier(payload: dict) -> dict:
    import requests
    import time
    import torch
    from torch import nn

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
    augmentation = config.get("augmentation") if isinstance(config.get("augmentation"), dict) else {}
    class_balancing = str(config.get("class_balancing", "")).lower()
    early_stopping_patience = _positive_int(config.get("early_stopping_patience"), default=0)
    model_name = str(config.get("model", "mobilenet_v3_small"))
    gpu_type = str(config.get("gpu_type") or os.getenv("MODAL_GPU_TYPE", "T4"))
    modal_function_call_id, modal_input_id = _modal_identifiers()

    archive_path = dataset_archive_path(dataset_id)
    download_s3_uri(dataset["storage_uri"], archive_path)
    dataset_dir = extract_dataset_archive(archive_path, dataset_id)

    train_loader, val_loader, class_names, class_weights = _load_image_data(
        dataset_dir,
        batch_size,
        image_size,
        augmentation,
        class_balancing,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_name, len(class_names)).to(device)

    weight_tensor = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = _build_optimizer(optimizer_name, trainable_parameters, learning_rate, weight_decay)
    scheduler = _build_scheduler(scheduler_name, optimizer, epochs)

    best_macro_f1 = 0.0
    best_accuracy = 0.0
    best_epoch = 0
    completed_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, accuracy, macro_f1 = _evaluate(model, val_loader, criterion, device)
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


def _configure_storage_env(payload: dict) -> None:
    os.environ["S3_ENDPOINT_URL"] = payload["s3_endpoint_url"]
    os.environ["AWS_ACCESS_KEY_ID"] = payload["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = payload["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = payload["aws_default_region"]


def _load_image_data(dataset_dir: Path, batch_size: int, image_size: int, augmentation: dict, class_balancing: str):
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    train_transform = _image_transform(image_size, augmentation, training=True)
    val_transform = _image_transform(image_size, {}, training=False)

    base_dataset = datasets.ImageFolder(dataset_dir)
    if len(base_dataset.classes) < 2:
        raise ValueError("Training requires at least two image classes.")
    if len(base_dataset) < 2:
        raise ValueError("Training requires at least two images.")

    validation_size = max(1, int(len(base_dataset) * 0.2))
    training_size = len(base_dataset) - validation_size
    if training_size < 1:
        training_size = len(base_dataset) - 1
        validation_size = 1

    generator = torch.Generator().manual_seed(42)
    shuffled_indices = torch.randperm(len(base_dataset), generator=generator).tolist()
    train_indices = shuffled_indices[:training_size]
    val_indices = shuffled_indices[training_size : training_size + validation_size]

    train_dataset = datasets.ImageFolder(dataset_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(dataset_dir, transform=val_transform)
    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    class_weights = _class_weights(base_dataset.targets, train_indices, len(base_dataset.classes), class_balancing)
    return train_loader, val_loader, base_dataset.classes, class_weights


def _image_transform(image_size: int, augmentation: dict, training: bool):
    from torchvision import transforms

    steps = []
    if training and augmentation.get("random_crop"):
        steps.append(transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))))
        steps.append(transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)))
    else:
        steps.append(transforms.Resize((image_size, image_size)))

    if training and augmentation.get("horizontal_flip"):
        steps.append(transforms.RandomHorizontalFlip())
    if training and augmentation.get("color_jitter"):
        steps.append(transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03))
    if training and augmentation.get("random_rotation"):
        steps.append(transforms.RandomRotation(10))

    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)


def _class_weights(targets: list[int], train_indices: list[int], class_count: int, class_balancing: str):
    if class_balancing not in {"weighted_loss", "class_weighted_loss"}:
        return None

    import torch

    counts = torch.zeros(class_count, dtype=torch.float32)
    for index in train_indices:
        counts[int(targets[index])] += 1.0
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * class_count)
    return weights


def _build_model(model_name: str, class_count: int):
    from torch import nn
    from torchvision import models

    normalized = model_name.lower()

    if "efficientnet_b1" in normalized:
        try:
            model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        except Exception:
            model = models.efficientnet_b1(weights=None)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, class_count)
        return model

    if "efficientnet" in normalized:
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            model = models.efficientnet_b0(weights=None)
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, class_count)
        return model

    if "resnet34" in normalized:
        try:
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        except Exception:
            model = models.resnet34(weights=None)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, class_count)
        return model

    if "resnet" in normalized:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            model = models.resnet18(weights=None)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, class_count)
        return model

    try:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    except Exception:
        model = models.mobilenet_v3_small(weights=None)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, class_count)
    return model


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


def _train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def _evaluate(model, loader, criterion, device) -> tuple[float, float, float]:
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions = []
    targets = []

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
            predictions.extend(predicted.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    correct = sum(1 for prediction, target in zip(predictions, targets) if prediction == target)
    accuracy = correct / max(len(targets), 1)
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return total_loss / max(total_examples, 1), accuracy, float(macro_f1)


def _post_json(url: str, payload: dict) -> None:
    import requests

    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()


def _post_training_run_summary(orchestrator_url: str, job_id: str, payload: dict) -> None:
    _post_json(f"{orchestrator_url}/jobs/{job_id}/training-run-summary", payload)


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


def _non_negative_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    parsed = _positive_int(value, default=default)
    return max(minimum, min(maximum, parsed))
