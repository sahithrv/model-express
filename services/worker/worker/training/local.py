from __future__ import annotations

import os
import time

from worker.orchestrator_client import OrchestratorClient
from worker.training.augmentation import normalize_augmentation_config, structured_policy_type


def run_local_training(client: OrchestratorClient, job: dict) -> None:
    """Deterministic local training simulator for locking the job/metric contract."""
    config = job["config"]
    job_id = job["id"]

    if _is_detection_training_config(config):
        _run_local_yolo_detection_training(client, job)
        return

    model = str(config.get("model", "unknown_model"))
    provider = str(config.get("provider") or "local")
    epochs = _positive_int(config.get("epochs"), default=3)
    learning_rate = _positive_float(config.get("learning_rate"), default=0.0003)
    batch_size = _positive_int(config.get("batch_size"), default=16)
    image_size = _positive_int(config.get("image_size"), default=224)
    optimizer = str(config.get("optimizer", "adamw")).lower()
    scheduler = str(config.get("scheduler", "none")).lower()
    weight_decay = _positive_float(config.get("weight_decay"), default=0.0)
    dropout = _bounded_float(config.get("dropout"), default=0.0, minimum=0.0, maximum=0.7)
    optimizer_momentum = _bounded_float(config.get("optimizer_momentum"), default=0.9, minimum=0.0, maximum=0.99)
    scheduler_gamma = _bounded_float(config.get("scheduler_gamma"), default=0.5, minimum=0.05, maximum=0.95)
    scheduler_step_size = _positive_int(config.get("scheduler_step_size"), default=max(1, epochs // 3))
    label_smoothing = _bounded_float(config.get("label_smoothing"), default=0.0, minimum=0.0, maximum=0.3)
    gradient_clip_norm = _bounded_float(config.get("gradient_clip_norm"), default=0.0, minimum=0.0, maximum=10.0)
    augmentation = normalize_augmentation_config(
        config.get("augmentation"),
        config.get("augmentation_policy", ""),
        config.get("augmentation_policy_config"),
    )
    augmentation_policy = structured_policy_type(augmentation) or str(
        config.get("augmentation_policy", "")
    ).lower()
    class_balancing = str(config.get("class_balancing", "")).lower()
    sampling_strategy = str(config.get("sampling_strategy", "")).lower()
    preprocessing = config.get("preprocessing") if isinstance(config.get("preprocessing"), dict) else {}
    epoch_sleep = _positive_float(os.getenv("LOCAL_TRAINING_EPOCH_SECONDS"), default=0.5)
    started_at = time.time()

    model_score = _model_score(model)
    image_bonus = 0.015 if image_size >= 256 else 0.0
    optimizer_bonus = 0.012 if optimizer in {"adamw", "sgd"} else 0.0
    if optimizer == "sgd" and 0.75 <= optimizer_momentum <= 0.95:
        optimizer_bonus += 0.004
    scheduler_bonus = 0.01 if scheduler in {"cosine", "step"} else 0.0
    if scheduler == "step" and 0.2 <= scheduler_gamma <= 0.8 and scheduler_step_size >= 1:
        scheduler_bonus += 0.003
    regularization_bonus = 0.01 if 0 < weight_decay <= 0.05 else 0.0
    if 0.05 <= dropout <= 0.35:
        regularization_bonus += 0.006
    if 0.02 <= label_smoothing <= 0.15:
        regularization_bonus += 0.004
    if 0.5 <= gradient_clip_norm <= 5:
        regularization_bonus += 0.003
    augmentation_bonus = 0.0
    if augmentation.get("horizontal_flip"):
        augmentation_bonus += 0.006
    if augmentation.get("vertical_flip"):
        augmentation_bonus += 0.004
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
    elif augmentation_policy in {"basic", "randaugment", "trivialaugment", "autoaugment"}:
        augmentation_bonus += 0.011
    elif augmentation_policy in {"mixup", "cutmix"}:
        augmentation_bonus += 0.009
    preprocessing_bonus = 0.0
    if str(preprocessing.get("resize_strategy", "")).lower() in {"preserve_aspect_pad", "center_crop"}:
        preprocessing_bonus += 0.004
    if str(preprocessing.get("crop_strategy", "")).lower() in {"random_resized_crop", "bbox_crop_if_available"}:
        preprocessing_bonus += 0.005
    if str(preprocessing.get("normalization", "")).lower() == "none":
        preprocessing_bonus -= 0.006
    balance_bonus = 0.0
    if class_balancing in {
        "weighted_loss",
        "class_weighted_loss",
        "focal_loss",
        "effective_number",
        "effective_number_loss",
        "effective_number_class_balanced_loss",
        "class_balanced_loss",
        "class_balanced_effective_number",
    }:
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
                "provider": provider,
                "gpu_type": str(config.get("gpu_type", "local")),
                "status": "RUNNING",
                "runtime_seconds": round(time.time() - started_at, 3),
                "estimated_cost_usd": 0,
                "best_macro_f1": best_macro_f1,
                "best_accuracy": best_accuracy,
                "final_train_loss": train_loss,
                "final_val_loss": val_loss,
                "epochs_completed": epoch,
                **_summary_metadata(config),
            },
        )
        print(f"Reported training epoch {epoch}/{epochs} for {job_id} ({model})")
        time.sleep(epoch_sleep)

    client.report_training_run_summary(
        job_id,
        {
            "model": model,
            "provider": provider,
            "gpu_type": str(config.get("gpu_type", "local")),
            "status": "SUCCEEDED",
            "runtime_seconds": round(time.time() - started_at, 3),
            "estimated_cost_usd": 0,
            "best_macro_f1": best_macro_f1,
            "best_accuracy": best_accuracy,
            "final_train_loss": train_loss,
            "final_val_loss": val_loss,
            "epochs_completed": epochs,
            **_summary_metadata(config),
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


def _run_local_yolo_detection_training(client: OrchestratorClient, job: dict) -> None:
    config = job["config"]
    job_id = job["id"]

    model = str(config.get("model", "yolo11n.pt"))
    provider = str(config.get("provider") or "local")
    epochs = _positive_int(config.get("epochs"), default=5)
    batch_size = _positive_int(config.get("batch_size"), default=8)
    image_size = _positive_int(config.get("image_size"), default=640)
    learning_rate = _positive_float(config.get("learning_rate"), default=0.001)
    epoch_sleep = _positive_float(os.getenv("LOCAL_TRAINING_EPOCH_SECONDS"), default=0.5)
    started_at = time.time()

    model_quality = _detector_model_score(model)
    size_bonus = 0.015 if image_size >= 640 else 0.0
    batch_penalty = 0.015 if batch_size < 4 else 0.0
    lr_penalty = 0.02 if learning_rate > 0.005 else 0.0
    final_map50_95 = max(0.28, min(0.92, model_quality + size_bonus - batch_penalty - lr_penalty))
    final_map50 = max(final_map50_95, min(0.97, final_map50_95 + 0.16))
    best_map50_95 = 0.0
    best_map50 = 0.0
    box_loss = 0.0
    cls_loss = 0.0
    dfl_loss = 0.0

    for epoch in range(1, epochs + 1):
        progress = epoch / epochs
        map50_95 = round(0.16 + (final_map50_95 - 0.16) * progress, 4)
        map50 = round(0.28 + (final_map50 - 0.28) * progress, 4)
        precision = round(max(0.0, min(1.0, map50 - 0.045)), 4)
        recall = round(max(0.0, min(1.0, map50 - 0.065)), 4)
        box_loss = round(max(0.025, 1.12 - 0.74 * progress - model_quality * 0.08), 4)
        cls_loss = round(max(0.02, 0.82 - 0.50 * progress - model_quality * 0.05), 4)
        dfl_loss = round(max(0.018, 0.62 - 0.36 * progress - model_quality * 0.03), 4)
        val_loss = round(box_loss + cls_loss + dfl_loss, 4)
        best_map50_95 = max(best_map50_95, map50_95)
        best_map50 = max(best_map50, map50)

        client.report_metric(
            job_id,
            epoch,
            {
                "train_loss": round(val_loss * 1.04, 4),
                "val_loss": val_loss,
                "box_loss": box_loss,
                "cls_loss": cls_loss,
                "dfl_loss": dfl_loss,
                "mAP50_95": map50_95,
                "map50_95": map50_95,
                "mAP50": map50,
                "map50": map50,
                "precision": precision,
                "recall": recall,
                "learning_rate": learning_rate,
            },
        )
        client.report_training_run_summary(
            job_id,
            {
                "model": model,
                "provider": provider,
                "gpu_type": str(config.get("gpu_type", "local")),
                "status": "RUNNING",
                "runtime_seconds": round(time.time() - started_at, 3),
                "estimated_cost_usd": 0,
                "best_macro_f1": best_map50_95,
                "best_accuracy": best_map50,
                "final_train_loss": round(val_loss * 1.04, 4),
                "final_val_loss": val_loss,
                "epochs_completed": epoch,
                **_summary_metadata(config),
            },
        )
        print(f"Reported detector epoch {epoch}/{epochs} for {job_id} ({model})")
        time.sleep(epoch_sleep)

    runtime_seconds = round(time.time() - started_at, 3)
    client.report_training_run_summary(
        job_id,
        {
            "model": model,
            "provider": provider,
            "gpu_type": str(config.get("gpu_type", "local")),
            "status": "SUCCEEDED",
            "runtime_seconds": runtime_seconds,
            "estimated_cost_usd": 0,
            "best_macro_f1": best_map50_95,
            "best_accuracy": best_map50,
            "final_train_loss": round((box_loss + cls_loss + dfl_loss) * 1.04, 4),
            "final_val_loss": round(box_loss + cls_loss + dfl_loss, 4),
            "epochs_completed": epochs,
            **_summary_metadata(config),
        },
    )
    client.report_training_run_evaluation(
        job_id,
        _local_detection_evaluation_payload(
            config=config,
            model=model,
            best_map50_95=best_map50_95,
            best_map50=best_map50,
            box_loss=box_loss,
            cls_loss=cls_loss,
            dfl_loss=dfl_loss,
            runtime_seconds=runtime_seconds,
        ),
    )
    client.complete_job(job_id, mlflow_run_id=f"local-yolo-training-{job_id}")


def _is_detection_training_config(config: dict) -> bool:
    model = str(config.get("model", "")).lower()
    return (
        str(config.get("task_type", "")).lower() == "object_detection"
        or str(config.get("model_kind", "")).lower() == "ultralytics_yolo_detector"
        or model.startswith("yolo11")
        or model.startswith("yolo")
    )


def _summary_metadata(config: dict) -> dict:
    out = {}
    materialization = config.get("dataset_materialization")
    if isinstance(materialization, dict):
        out["dataset_materialization"] = materialization
    stage_telemetry = config.get("stage_telemetry")
    if isinstance(stage_telemetry, dict):
        out["stage_telemetry"] = stage_telemetry
    return out


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


def _detector_model_score(model: str) -> float:
    normalized = model.lower()
    if "yolo11x" in normalized:
        return 0.76
    if "yolo11l" in normalized:
        return 0.735
    if "yolo11m" in normalized:
        return 0.705
    if "yolo11s" in normalized:
        return 0.665
    if "yolo11n" in normalized:
        return 0.625
    return 0.58


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
            "augmentation_policy_config": config.get("augmentation_policy_config")
            if isinstance(config.get("augmentation_policy_config"), dict)
            else {},
            "class_balancing": str(config.get("class_balancing", "")),
            "sampling_strategy": str(config.get("sampling_strategy", "")),
            "preprocessing": config.get("preprocessing") if isinstance(config.get("preprocessing"), dict) else {},
            "training_hyperparameters": {
                "optimizer": str(config.get("optimizer", "adamw")),
                "scheduler": str(config.get("scheduler", "none")),
                "weight_decay": _positive_float(config.get("weight_decay"), default=0.0),
                "dropout": _bounded_float(config.get("dropout"), default=0.0, minimum=0.0, maximum=0.7),
                "optimizer_momentum": _bounded_float(config.get("optimizer_momentum"), default=0.9, minimum=0.0, maximum=0.99)
                if str(config.get("optimizer", "adamw")).lower() == "sgd"
                else 0,
                "scheduler_step_size": _positive_int(config.get("scheduler_step_size"), default=1)
                if str(config.get("scheduler", "none")).lower() == "step"
                else 0,
                "scheduler_gamma": _bounded_float(config.get("scheduler_gamma"), default=0.5, minimum=0.05, maximum=0.95)
                if str(config.get("scheduler", "none")).lower() == "step"
                else 0,
                "label_smoothing": _bounded_float(config.get("label_smoothing"), default=0.0, minimum=0.0, maximum=0.3),
                "gradient_clip_norm": _bounded_float(config.get("gradient_clip_norm"), default=0.0, minimum=0.0, maximum=10.0),
            },
        },
        "label_quality_audit": _local_label_quality_audit(config, per_class_metrics),
    }


def _local_detection_evaluation_payload(
    *,
    config: dict,
    model: str,
    best_map50_95: float,
    best_map50: float,
    box_loss: float,
    cls_loss: float,
    dfl_loss: float,
    runtime_seconds: float,
) -> dict:
    class_names = _detection_class_names(config)
    precision = round(max(0.0, min(1.0, best_map50 - 0.045)), 4)
    recall = round(max(0.0, min(1.0, best_map50 - 0.065)), 4)
    model_profile = _local_detection_model_profile(model, config)
    latency_ms = float(model_profile["estimated_latency_ms"])
    latency_score = max(0.0, min(1.0, 1.0 - latency_ms / 160.0))
    loss_score = max(0.0, min(1.0, 1.0 - (box_loss + cls_loss + dfl_loss) / 3.0))
    quality_score = round(best_map50_95 * 0.65 + best_map50 * 0.20 + recall * 0.15, 4)
    overall_score = round(quality_score * 0.78 + latency_score * 0.12 + loss_score * 0.10, 4)
    per_class_metrics = {
        class_name: {
            "precision": round(max(0.0, min(1.0, precision - 0.01 + index * 0.004)), 4),
            "recall": round(max(0.0, min(1.0, recall - 0.012 + index * 0.003)), 4),
            "AP50": round(max(0.0, min(1.0, best_map50 - 0.015 + index * 0.002)), 4),
            "AP50_95": round(max(0.0, min(1.0, best_map50_95 - 0.018 + index * 0.002)), 4),
            "support": 20,
        }
        for index, class_name in enumerate(class_names)
    }
    return {
        "objective_profile": {
            "target_metric": str(config.get("target_metric", "mAP50_95")),
            "metric_preferences": ["mAP50_95", "mAP50", "recall", "precision", "latency_p95_ms"],
            "task_type": "object_detection",
            "split_strategy": "yolo_train_validation_with_heldout_test_when_available",
            "heldout_test_map50_95": round(best_map50_95, 6),
            "heldout_test_map50": round(best_map50, 6),
            "heldout_test_precision": precision,
            "heldout_test_recall": recall,
            "heldout_test_box_loss": round(box_loss, 6),
            "heldout_test_cls_loss": round(cls_loss, 6),
            "heldout_test_dfl_loss": round(dfl_loss, 6),
        },
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": [],
        "model_profile": model_profile,
        "holistic_scores": {
            "quality_score": quality_score,
            "latency_score": round(latency_score, 4),
            "loss_health_score": round(loss_score, 4),
            "cost_score": 1.0,
            "overall_score": overall_score,
            "runtime_seconds": runtime_seconds,
            "detection_metrics": {
                "mAP50_95": round(best_map50_95, 6),
                "mAP50": round(best_map50, 6),
                "precision": precision,
                "recall": recall,
                "box_loss": round(box_loss, 6),
                "cls_loss": round(cls_loss, 6),
                "dfl_loss": round(dfl_loss, 6),
            },
        },
        "preprocessing_summary": {
            "task_type": "object_detection",
            "preserves_yolo_splits": True,
            "simulation": True,
            "exportable": False,
            "preprocessing": config.get("preprocessing") if isinstance(config.get("preprocessing"), dict) else {},
            "training_hyperparameters": {
                "learning_rate": _positive_float(config.get("learning_rate"), default=0.001),
                "batch_size": _positive_int(config.get("batch_size"), default=8),
                "epochs": _positive_int(config.get("epochs"), default=5),
                "image_size": _positive_int(config.get("image_size"), default=640),
            },
        },
        "recommendation_summary": (
            f"{model} simulated detector evaluation: mAP50-95 {best_map50_95:.3f}, "
            f"mAP50 {best_map50:.3f}, recall {recall:.3f}, estimated latency {latency_ms:.1f}ms."
        ),
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
        "dropout": _bounded_float(config.get("dropout"), default=0.0, minimum=0.0, maximum=0.7),
    }


def _local_detection_model_profile(model: str, config: dict) -> dict:
    normalized = model.lower()
    metadata = {
        "yolo11n.pt": (2_600_000, 5.4, 8.0),
        "yolo11s.pt": (9_400_000, 19.2, 14.0),
        "yolo11m.pt": (20_100_000, 41.0, 24.0),
        "yolo11l.pt": (25_300_000, 52.0, 38.0),
        "yolo11x.pt": (56_900_000, 114.0, 62.0),
    }
    parameter_count, size_mb, latency_ms = metadata.get(normalized, (9_400_000, 20.0, 18.0))
    image_size = _positive_int(config.get("image_size"), default=640)
    latency_scale = max(0.35, (image_size / 640) ** 2)
    class_names = _detection_class_names(config)
    return {
        "task_type": "object_detection",
        "model_kind": "ultralytics_yolo_detector",
        "runtime": "simulated_local_yolo",
        "simulation": True,
        "simulated_training": True,
        "exportable": False,
        "deployment_ready": False,
        "export_status": "SIMULATED_UNEXPORTABLE",
        "artifact_profile_status": "simulation_only",
        "parameter_count": parameter_count,
        "estimated_model_size_mb": round(size_mb, 2),
        "estimated_latency_ms": round(latency_ms * latency_scale, 2),
        "latency_p50_ms": round(latency_ms * latency_scale * 0.82, 2),
        "latency_p95_ms": round(latency_ms * latency_scale * 1.35, 2),
        "estimated_throughput_images_per_second": round(1000.0 / max(latency_ms * latency_scale, 1.0), 2),
        "image_size": image_size,
        "input_shape": [1, 3, image_size, image_size],
        "class_labels": class_names,
        "confidence_threshold": _bounded_float(config.get("confidence_threshold"), default=0.25, minimum=0.01, maximum=0.99),
        "iou_threshold": _bounded_float(config.get("iou_threshold"), default=0.7, minimum=0.1, maximum=0.99),
        "pretrained": True,
    }


def _detection_class_names(config: dict) -> list[str]:
    for key in ("class_names", "class_labels", "classes"):
        value = config.get(key)
        if isinstance(value, list):
            out = [str(item) for item in value if str(item).strip()]
            if out:
                return out[:200]
    metadata = config.get("metadata_summary") if isinstance(config.get("metadata_summary"), dict) else {}
    yolo_summary = metadata.get("yolo_summary") if isinstance(metadata.get("yolo_summary"), dict) else {}
    value = yolo_summary.get("class_names")
    if isinstance(value, list):
        out = [str(item) for item in value if str(item).strip()]
        if out:
            return out[:200]
    class_count = _positive_int(config.get("class_count"), default=2)
    return [f"class_{index}" for index in range(class_count)]


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


def _local_label_quality_audit(config: dict, per_class_metrics: dict) -> dict:
    mechanism = str(config.get("mechanism", "")).lower()
    requested = mechanism in {"label_noise_audit", "hard_example_audit"} or bool(
        config.get("label_quality_audit")
    )
    if not requested:
        return {"status": "not_requested", "report_only": True}
    hard_examples = [
        {
            "path": "",
            "true_class": class_name,
            "predicted_class": class_name,
            "confidence": round(max(0.0, min(1.0, float(metrics.get("f1-score", 0.0)))), 4),
            "correct": True,
        }
        for class_name, metrics in per_class_metrics.items()
    ]
    return {
        "status": "simulated",
        "report_only": True,
        "sample_count": len(hard_examples),
        "high_confidence_wrong": [],
        "low_confidence_correct": hard_examples[:25],
        "hard_examples": hard_examples[:50],
        "notes": "Local simulator emits report-only audit metadata and never mutates labels.",
    }


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


def _bounded_float(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))
