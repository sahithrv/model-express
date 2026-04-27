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
    import torch
    from torch import nn

    from worker.datasets.cache import dataset_archive_path, extract_dataset_archive
    from worker.datasets.storage import download_s3_uri

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
    model_name = str(config.get("model", "mobilenet_v3_small"))

    archive_path = dataset_archive_path(dataset_id)
    download_s3_uri(dataset["storage_uri"], archive_path)
    dataset_dir = extract_dataset_archive(archive_path, dataset_id)

    train_loader, val_loader, class_names = _load_image_data(dataset_dir, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(model_name, len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )

    best_macro_f1 = 0.0
    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, accuracy, macro_f1 = _evaluate(model, val_loader, criterion, device)
        best_macro_f1 = max(best_macro_f1, macro_f1)
        best_accuracy = max(best_accuracy, accuracy)

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
                },
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
        "device": str(device),
    }


def _configure_storage_env(payload: dict) -> None:
    os.environ["S3_ENDPOINT_URL"] = payload["s3_endpoint_url"]
    os.environ["AWS_ACCESS_KEY_ID"] = payload["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = payload["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = payload["aws_default_region"]


def _load_image_data(dataset_dir: Path, batch_size: int):
    import torch
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    if len(dataset.classes) < 2:
        raise ValueError("Training requires at least two image classes.")
    if len(dataset) < 2:
        raise ValueError("Training requires at least two images.")

    validation_size = max(1, int(len(dataset) * 0.2))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        training_size = len(dataset) - 1
        validation_size = 1

    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(dataset, [training_size, validation_size], generator=generator)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader, dataset.classes


def _build_model(model_name: str, class_count: int):
    from torch import nn
    from torchvision import models

    normalized = model_name.lower()

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
