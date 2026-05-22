from __future__ import annotations

import os
import time

from worker.orchestrator_client import OrchestratorClient


def run_local_training(client: OrchestratorClient, job: dict) -> None:
    """Deterministic local training simulator for locking the job/metric contract."""
    config = job["config"]
    job_id = job["id"]

    model = str(config.get("model", "unknown_model"))
    epochs = _positive_int(config.get("epochs"), default=3)
    learning_rate = _positive_float(config.get("learning_rate"), default=0.0003)
    batch_size = _positive_int(config.get("batch_size"), default=16)
    image_size = _positive_int(config.get("image_size"), default=224)
    optimizer = str(config.get("optimizer", "adamw")).lower()
    scheduler = str(config.get("scheduler", "none")).lower()
    weight_decay = _positive_float(config.get("weight_decay"), default=0.0)
    augmentation = config.get("augmentation") if isinstance(config.get("augmentation"), dict) else {}
    augmentation_policy = str(config.get("augmentation_policy", "")).lower()
    class_balancing = str(config.get("class_balancing", "")).lower()
    sampling_strategy = str(config.get("sampling_strategy", "")).lower()
    preprocessing = config.get("preprocessing") if isinstance(config.get("preprocessing"), dict) else {}
    epoch_sleep = _positive_float(os.getenv("LOCAL_TRAINING_EPOCH_SECONDS"), default=0.5)
    started_at = time.time()

    model_score = _model_score(model)
    image_bonus = 0.015 if image_size >= 256 else 0.0
    optimizer_bonus = 0.012 if optimizer in {"adamw", "sgd"} else 0.0
    scheduler_bonus = 0.01 if scheduler in {"cosine", "step"} else 0.0
    regularization_bonus = 0.01 if 0 < weight_decay <= 0.05 else 0.0
    augmentation_bonus = 0.0
    if augmentation.get("horizontal_flip"):
        augmentation_bonus += 0.006
    if augmentation.get("color_jitter"):
        augmentation_bonus += 0.008
    if augmentation.get("random_crop"):
        augmentation_bonus += 0.006
    if augmentation_policy == "light":
        augmentation_bonus += 0.004
    elif augmentation_policy == "moderate":
        augmentation_bonus += 0.01
    elif augmentation_policy == "strong":
        augmentation_bonus += 0.012
    preprocessing_bonus = 0.0
    if str(preprocessing.get("resize_strategy", "")).lower() in {"preserve_aspect_pad", "center_crop"}:
        preprocessing_bonus += 0.004
    if str(preprocessing.get("crop_strategy", "")).lower() in {"random_resized_crop", "bbox_crop_if_available"}:
        preprocessing_bonus += 0.005
    if str(preprocessing.get("normalization", "")).lower() == "none":
        preprocessing_bonus -= 0.006
    balance_bonus = 0.0
    if class_balancing in {"weighted_loss", "class_weighted_loss", "focal_loss"}:
        balance_bonus += 0.012
    if class_balancing in {"class_balanced_sampler", "weighted_random_sampler"} or sampling_strategy in {
        "class_balanced_sampler",
        "weighted_random_sampler",
    }:
        balance_bonus += 0.01
    batch_penalty = 0.02 if batch_size < 8 else 0.0
    lr_penalty = 0.03 if learning_rate > 0.001 else 0.0
    final_macro_f1 = max(
        0.35,
        min(
            0.96,
            model_score
            + image_bonus
            + optimizer_bonus
            + scheduler_bonus
            + regularization_bonus
            + augmentation_bonus
            + preprocessing_bonus
            + balance_bonus
            - batch_penalty
            - lr_penalty,
        ),
    )
    best_macro_f1 = 0.0
    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        progress = epoch / epochs
        macro_f1 = round(0.24 + (final_macro_f1 - 0.24) * progress, 4)
        accuracy = round(min(0.97, macro_f1 + 0.035), 4)
        train_loss = round(max(0.04, 1.18 - 0.82 * progress - model_score * 0.08), 4)
        val_loss = round(max(0.05, 1.28 - 0.72 * progress - model_score * 0.05), 4)
        best_macro_f1 = max(best_macro_f1, macro_f1)
        best_accuracy = max(best_accuracy, accuracy)

        client.report_metric(
            job_id,
            epoch,
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "learning_rate": learning_rate,
            },
        )
        client.report_training_run_summary(
            job_id,
            {
                "model": model,
                "provider": "local",
                "gpu_type": str(config.get("gpu_type", "local")),
                "status": "RUNNING",
                "runtime_seconds": round(time.time() - started_at, 3),
                "estimated_cost_usd": 0,
                "best_macro_f1": best_macro_f1,
                "best_accuracy": best_accuracy,
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
                "epochs_completed": epoch,
            },
        )
        print(f"Reported training epoch {epoch}/{epochs} for {job_id} ({model})")
        time.sleep(epoch_sleep)

    client.report_training_run_summary(
        job_id,
        {
            "model": model,
            "provider": "local",
            "gpu_type": str(config.get("gpu_type", "local")),
            "status": "SUCCEEDED",
            "runtime_seconds": round(time.time() - started_at, 3),
            "estimated_cost_usd": 0,
            "best_macro_f1": best_macro_f1,
            "best_accuracy": best_accuracy,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "epochs_completed": epochs,
        },
    )
    client.report_training_run_evaluation(
        job_id,
        _local_evaluation_payload(
            config=config,
            model=model,
            best_macro_f1=best_macro_f1,
            best_accuracy=best_accuracy,
            runtime_seconds=round(time.time() - started_at, 3),
        ),
    )
    client.complete_job(job_id, mlflow_run_id=f"local-training-{job_id}")


def _model_score(model: str) -> float:
    normalized = model.lower()
    if "convnext" in normalized:
        return 0.86
    if "swin" in normalized:
        return 0.855
    if "vit" in normalized:
        return 0.85
    if "efficientnet_b2" in normalized:
        return 0.845
    if "efficientnet_b1" in normalized:
        return 0.835
    if "efficientnet" in normalized:
        return 0.82
    if "regnet" in normalized:
        return 0.805
    if "resnet" in normalized:
        return 0.78
    if "mobilenet_v3_large" in normalized:
        return 0.755
    if "mobilenet" in normalized:
        return 0.72

    checksum = sum(ord(char) for char in normalized)
    return 0.62 + (checksum % 20) / 100


def _local_evaluation_payload(config: dict, model: str, best_macro_f1: float, best_accuracy: float, runtime_seconds: float) -> dict:
    class_count = _positive_int(config.get("class_count"), default=2)
    model_profile = _local_model_profile(model, config)
    confusion_matrix = _synthetic_confusion_matrix(class_count, best_accuracy)
    per_class_metrics = {
        f"class_{index}": {
            "precision": round(max(0.0, min(1.0, best_macro_f1 - 0.02 + index * 0.005)), 4),
            "recall": round(max(0.0, min(1.0, best_macro_f1 - 0.015 + index * 0.004)), 4),
            "f1-score": round(max(0.0, min(1.0, best_macro_f1 - 0.01 + index * 0.003)), 4),
            "support": 20,
        }
        for index in range(class_count)
    }
    latency_ms = float(model_profile["estimated_latency_ms"])
    cost_score = 1.0
    latency_score = max(0.0, min(1.0, 1.0 - latency_ms / 120.0))
    quality_score = round((best_macro_f1 * 0.65) + (best_accuracy * 0.35), 4)
    overall_score = round((quality_score * 0.7) + (latency_score * 0.2) + (cost_score * 0.1), 4)
    return {
        "objective_profile": {
            "target_metric": str(config.get("target_metric", "macro_f1")),
            "metric_preferences": ["macro_f1", "accuracy", "per_class_f1", "latency"],
            "split_strategy": "simulated_train_validation_with_heldout_test_placeholder",
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "model_profile": model_profile,
        "holistic_scores": {
            "quality_score": quality_score,
            "latency_score": round(latency_score, 4),
            "cost_score": cost_score,
            "overall_score": overall_score,
            "runtime_seconds": runtime_seconds,
        },
        "recommendation_summary": (
            f"{model} simulated evaluation: macro-F1 {best_macro_f1:.3f}, "
            f"accuracy {best_accuracy:.3f}, estimated latency {latency_ms:.1f}ms."
        ),
        "preprocessing_summary": {
            "augmentation_policy": str(config.get("augmentation_policy", "")),
            "class_balancing": str(config.get("class_balancing", "")),
            "sampling_strategy": str(config.get("sampling_strategy", "")),
            "preprocessing": config.get("preprocessing") if isinstance(config.get("preprocessing"), dict) else {},
        },
    }


def _local_model_profile(model: str, config: dict) -> dict:
    normalized = model.lower()
    metadata = {
        "mobilenet_v3_small": (2_500_000, 9.5, 8.0),
        "mobilenet_v3_large": (5_400_000, 21.0, 12.0),
        "efficientnet_b0": (5_300_000, 20.0, 15.0),
        "efficientnet_b1": (7_800_000, 30.0, 22.0),
        "efficientnet_b2": (9_200_000, 35.0, 28.0),
        "regnet_y_400mf": (4_300_000, 17.0, 14.0),
        "resnet18": (11_700_000, 45.0, 25.0),
        "resnet34": (21_800_000, 84.0, 38.0),
        "convnext_tiny": (28_600_000, 109.0, 55.0),
        "swin_t": (28_300_000, 108.0, 64.0),
        "vit_b_16": (86_600_000, 330.0, 95.0),
    }
    parameter_count, size_mb, latency_ms = metadata.get(normalized, (8_000_000, 32.0, 30.0))
    image_size = _positive_int(config.get("image_size"), default=224)
    latency_scale = max(0.5, (image_size / 224) ** 2)
    return {
        "parameter_count": parameter_count,
        "estimated_model_size_mb": round(size_mb, 2),
        "estimated_latency_ms": round(latency_ms * latency_scale, 2),
        "estimated_throughput_images_per_second": round(1000.0 / max(latency_ms * latency_scale, 1.0), 2),
        "image_size": image_size,
        "fine_tune_strategy": str(config.get("fine_tune_strategy", "head_only")),
        "pretrained": bool(config.get("pretrained", True)),
    }


def _synthetic_confusion_matrix(class_count: int, accuracy: float) -> list[list[int]]:
    class_count = max(2, min(class_count, 12))
    diagonal = max(1, int(round(20 * accuracy)))
    off_diagonal = max(0, 20 - diagonal)
    matrix = []
    for row in range(class_count):
        values = [0 for _ in range(class_count)]
        values[row] = diagonal
        if class_count > 1 and off_diagonal > 0:
            values[(row + 1) % class_count] = off_diagonal
        matrix.append(values)
    return matrix


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
