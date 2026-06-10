from __future__ import annotations

import base64
import csv
import contextvars
import json
import os
import random
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from worker.training.augmentation import (
    MIXED_SAMPLE_POLICY_TYPES,
    normalize_augmentation_config,
    structured_policy_type,
)
from worker.training.preprocessing_registry import (
    bbox_compare_requested,
    bbox_crop_required,
    bbox_crop_requested,
    build_image_transform,
    normalization_values,
    resize_with_padding,
)
from worker.training.modal_resources import (
    callback_identity,
    failure_callback_payload,
    modal_resources_from_payload,
    resource_telemetry,
)

try:
    import modal
except Exception:  # pragma: no cover - local helper tests can run without Modal installed.
    modal = None


def _modal_remote_path_env(name: str, default: str) -> PurePosixPath:
    value = os.getenv(name, default).strip() or default
    path = PurePosixPath(value.replace("\\", "/"))
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute POSIX path for Modal, got {value!r}.")
    if str(path) in {"/", "/root", "/tmp"}:
        raise ValueError(f"{name} cannot be mounted at {path}.")
    return path


APP_NAME = "model-express-training"
DEFAULT_GPU = os.getenv("MODAL_GPU_TYPE", "T4")
DATASET_MATERIALIZATION_ROOT = _modal_remote_path_env(
    "MODEL_EXPRESS_MODAL_DATASET_CACHE_ROOT",
    "/cache/model-express/datasets",
)
DATASET_VOLUME_NAME = os.getenv(
    "MODEL_EXPRESS_MODAL_DATASET_CACHE_VOLUME",
    "model-express-dataset-cache",
)
TORCH_CACHE_ROOT = _modal_remote_path_env(
    "MODEL_EXPRESS_MODAL_TORCH_CACHE_ROOT",
    "/cache/model-express/torch",
)
TORCH_CACHE_VOLUME_NAME = os.getenv(
    "MODEL_EXPRESS_MODAL_TORCH_CACHE_VOLUME",
    "model-express-torch-cache",
)
DEFAULT_ORCHESTRATOR_REPORT_TIMEOUT_SECONDS = 300
DEFAULT_MODAL_DATASET_MATERIALIZATION_TIMEOUT_SECONDS = 60 * 60
DEFAULT_MODAL_DATASET_PROFILE_TIMEOUT_SECONDS = 60 * 60
DEFAULT_MODAL_TRAINING_SCALEDOWN_WINDOW_SECONDS = 10 * 60
DEFAULT_COST_SENSITIVE_MODAL_TRAINING_SCALEDOWN_WINDOW_SECONDS = 120
DEFAULT_MODAL_METADATA_DATALOADER_WORKERS = 4
DEFAULT_MODAL_IMAGEFOLDER_DATALOADER_WORKERS = 2
DEFAULT_MODAL_TRAINING_DATASET_CACHE_ROOT = "/tmp/model-express/training-datasets"
DEFAULT_METADATA_BUNDLE_PAGE_SIZE = 5000
DEFAULT_METADATA_BUNDLE_MAX_RECORDS = 50_000
_MODAL_STAGE_EVENTS: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar(
    "model_express_modal_stage_events",
    default=None,
)
METADATA_ENDPOINT_UNAVAILABLE_STATUS_CODES = {404, 405, 501}
COMMON_IMAGE_ROOT_NAMES = ("images", "image", "imgs", "img", "JPEGImages", "jpegimages", "data")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT_METADATA_DIR_NAMES = {
    "annotation",
    "annotations",
    "attribute",
    "attributes",
    "bbox",
    "bboxes",
    "boxes",
    "keypoint",
    "keypoints",
    "label",
    "labels",
    "landmark",
    "landmarks",
    "manifest",
    "manifests",
    "meta",
    "metadata",
    "part",
    "parts",
    "split",
    "splits",
}

EFFECTIVE_NUMBER_CLASS_BALANCING = {
    "effective_number",
    "effective_number_loss",
    "effective_number_class_balanced_loss",
    "class_balanced_loss",
    "class_balanced_effective_number",
}
if modal is not None:
    try:
        dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=True)
        dataset_volume_mounts = {str(DATASET_MATERIALIZATION_ROOT): dataset_volume}
    except Exception:  # pragma: no cover - depends on Modal runtime/account setup.
        dataset_volume = None
        dataset_volume_mounts = {}
    try:
        torch_cache_volume = modal.Volume.from_name(TORCH_CACHE_VOLUME_NAME, create_if_missing=True)
        torch_cache_volume_mounts = {str(TORCH_CACHE_ROOT): torch_cache_volume}
    except Exception:  # pragma: no cover - depends on Modal runtime/account setup.
        torch_cache_volume = None
        torch_cache_volume_mounts = {}
    training_volume_mounts = dict(torch_cache_volume_mounts)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("libglib2.0-0", "libgl1")
        .pip_install(
            "boto3",
            "numpy",
            "pillow",
            "requests",
            "scikit-learn",
            "onnx",
            "onnxscript",
            "pyyaml",
            "torch",
            "torchvision",
            "ultralytics",
        )
        .env({"TORCH_HOME": str(TORCH_CACHE_ROOT)})
        .add_local_python_source("worker")
    )
    app = modal.App(APP_NAME)
else:
    image = None
    dataset_volume = None
    dataset_volume_mounts = {}
    torch_cache_volume = None
    torch_cache_volume_mounts = {}
    training_volume_mounts = {}

    class _UnavailableModalApp:
        def function(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    app = _UnavailableModalApp()


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
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


def _modal_dataset_materialization_timeout_seconds() -> int:
    return _positive_int_env(
        "MODEL_EXPRESS_MODAL_MATERIALIZATION_TIMEOUT_SECONDS",
        DEFAULT_MODAL_DATASET_MATERIALIZATION_TIMEOUT_SECONDS,
    )


def _modal_dataset_profile_timeout_seconds() -> int:
    return _positive_int_env(
        "MODEL_EXPRESS_MODAL_PROFILE_TIMEOUT_SECONDS",
        DEFAULT_MODAL_DATASET_PROFILE_TIMEOUT_SECONDS,
    )


def _optional_positive_int_env(name: str, default: int = 0) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return default if default > 0 else None
    try:
        parsed = int(value)
    except ValueError:
        return default if default > 0 else None
    return parsed if parsed > 0 else None


def _modal_training_min_containers() -> int | None:
    return _optional_positive_int_env("MODEL_EXPRESS_MODAL_TRAIN_MIN_CONTAINERS")


def _modal_training_buffer_containers() -> int | None:
    return _optional_positive_int_env("MODEL_EXPRESS_MODAL_TRAIN_BUFFER_CONTAINERS")


def _modal_training_scaledown_window_seconds() -> int | None:
    return _optional_positive_int_env(
        "MODEL_EXPRESS_MODAL_TRAIN_SCALEDOWN_WINDOW_SECONDS",
        DEFAULT_COST_SENSITIVE_MODAL_TRAINING_SCALEDOWN_WINDOW_SECONDS
        if _modal_cost_sensitive_defaults_enabled()
        else DEFAULT_MODAL_TRAINING_SCALEDOWN_WINDOW_SECONDS,
    )


def _modal_cost_sensitive_defaults_enabled() -> bool:
    return _bool(os.getenv("MODEL_EXPRESS_MODAL_COST_SENSITIVE_DEFAULTS"), default=False)


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


class _DatasetRelativePathResolver:
    def __init__(self, dataset_dir: Path):
        from worker.datasets.metadata_discovery import is_safe_relative_path

        self.dataset_dir = dataset_dir
        self.dataset_root = dataset_dir.resolve(strict=True)
        self.image_root_prefixes = _metadata_image_root_prefixes(dataset_dir)
        self.is_safe_relative_path = is_safe_relative_path

    def resolve(self, relative_path: str) -> Path | None:
        if not self.is_safe_relative_path(relative_path):
            return None
        candidate_paths = [relative_path]
        for image_root in self.image_root_prefixes:
            candidate_paths.append(f"{image_root}/{relative_path}")
        for candidate_path in candidate_paths:
            if not self.is_safe_relative_path(candidate_path):
                continue
            candidate = self.dataset_dir.joinpath(*candidate_path.split("/"))
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(self.dataset_root)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                return resolved
        return None



@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes=training_volume_mounts,
    min_containers=_modal_training_min_containers(),
    buffer_containers=_modal_training_buffer_containers(),
    scaledown_window=_modal_training_scaledown_window_seconds(),
)
def train_image_classifier(payload: dict) -> dict:
    try:
        return _train_image_classifier_impl(payload)
    except Exception as exc:
        if _report_modal_training_retryable_failure(payload, exc):
            return {
                "status": "retryable_failure_reported",
                "job_id": str((payload.get("job") or {}).get("id") or ""),
                "error": _modal_training_error_message(exc),
            }
        raise


def _train_image_classifier_impl(payload: dict) -> dict:
    import time
    import torch

    started_at = time.time()
    stage_events: list[dict] = []
    _MODAL_STAGE_EVENTS.set(stage_events if _modal_stage_telemetry_enabled() else None)
    job = payload["job"]
    config = job["config"]
    dataset = payload["dataset"]
    orchestrator_url = payload["orchestrator_url"].rstrip("/")

    _configure_storage_env(payload)

    job_id = job["id"]
    dataset_id = dataset["id"]
    modal_resources = modal_resources_from_payload(payload, config, detection_job=False)
    epochs = _positive_int(config.get("epochs"), default=5)
    batch_size = _positive_int(modal_resources.get("effective_batch_size"), default=16)
    learning_rate = _positive_float(config.get("learning_rate"), default=0.0003)
    image_size = _bounded_int(config.get("image_size"), default=224, minimum=96, maximum=384)
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    scheduler_name = str(config.get("scheduler", "none")).lower()
    weight_decay = _non_negative_float(config.get("weight_decay"), default=0.0)
    dropout = _bounded_float(config.get("dropout"), default=0.0, minimum=0.0, maximum=0.7)
    optimizer_momentum = _bounded_float(config.get("optimizer_momentum"), default=0.9, minimum=0.0, maximum=0.99)
    scheduler_step_size = _bounded_int(config.get("scheduler_step_size"), default=max(1, epochs // 3), minimum=1, maximum=max(1, epochs))
    scheduler_gamma = _bounded_float(config.get("scheduler_gamma"), default=0.5, minimum=0.05, maximum=0.95)
    label_smoothing = _bounded_float(config.get("label_smoothing"), default=0.0, minimum=0.0, maximum=0.3)
    gradient_clip_norm = _bounded_float(config.get("gradient_clip_norm"), default=0.0, minimum=0.0, maximum=10.0)
    augmentation = normalize_augmentation_config(
        config.get("augmentation"),
        config.get("augmentation_policy", ""),
        config.get("augmentation_policy_config"),
    )
    class_balancing = str(config.get("class_balancing", "")).lower()
    sampling_strategy = str(config.get("sampling_strategy", "")).lower()
    preprocessing = config.get("preprocessing") if isinstance(config.get("preprocessing"), dict) else {}
    class_balancing_config = config.get("class_balancing_config") if isinstance(config.get("class_balancing_config"), dict) else {}
    early_stopping_patience = _positive_int(config.get("early_stopping_patience"), default=0)
    model_name = str(config.get("model", "mobilenet_v3_small"))
    pretrained = _bool(config.get("pretrained"), default=True)
    freeze_backbone = _bool(config.get("freeze_backbone"), default=True)
    fine_tune_strategy = str(config.get("fine_tune_strategy", "head_only")).lower()
    gpu_type = str(modal_resources.get("effective_gpu_type") or config.get("gpu_type") or "T4")
    modal_function_call_id, modal_input_id = _modal_identifiers()
    modal_resources = {
        **modal_resources,
        "modal_function_call_id": modal_function_call_id,
        "modal_input_id": modal_input_id,
    }

    _modal_training_phase(job_id, "storage_configured", started_at)
    _modal_training_phase(job_id, "torch_cache_reload_start", started_at)
    _reload_modal_torch_cache_volume()
    _modal_training_phase(job_id, "torch_cache_reload_done", started_at)
    dataset_dir, dataset_materialization = _modal_training_dataset_for_job(
        payload,
        dataset=dataset,
        config=config,
        job_id=job_id,
        dataset_id=dataset_id,
        started_at=started_at,
    )
    _modal_training_phase(job_id, "metadata_fetch_start", started_at)
    metadata_bundle = _fetch_training_metadata_bundle(orchestrator_url, dataset_id, config)
    _modal_training_phase(
        job_id,
        "metadata_fetch_done",
        started_at,
        records=len(_metadata_manifest_records(metadata_bundle)),
    )

    _modal_training_phase(job_id, "data_load_start", started_at)
    dataset_tier_config = _dataset_tier_config(
        config,
        dataset=dataset,
        task_type="image_classification",
        dataset_dir=dataset_dir,
    )
    train_loader, val_loader, test_loader, class_names, class_weights, execution_metadata = _load_image_data(
        dataset_dir,
        batch_size,
        image_size,
        augmentation,
        class_balancing,
        sampling_strategy,
        preprocessing,
        class_balancing_config,
        metadata_bundle=metadata_bundle,
        dataset_tier_config=dataset_tier_config,
    )
    subset_manifest = execution_metadata.get("dataset_tier", {}).get("subset_manifest")
    if isinstance(subset_manifest, dict):
        dataset_materialization["subset_manifest"] = subset_manifest
    effective_batch_size = _positive_int(getattr(train_loader, "batch_size", None), default=batch_size)
    modal_resource_telemetry = resource_telemetry(
        modal_resources,
        effective_batch_size=effective_batch_size,
    )
    _modal_training_phase(
        job_id,
        "data_load_done",
        started_at,
        classes=len(class_names),
        train_examples=len(train_loader.dataset),
        val_examples=len(val_loader.dataset),
        test_examples=len(test_loader.dataset),
        metadata_status=execution_metadata.get("metadata_bundle", {}).get("status"),
        split_strategy=execution_metadata.get("metadata_bundle", {}).get("split_strategy"),
        loader_workers=execution_metadata.get("dataloader", {}).get("workers"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _modal_training_phase(job_id, "model_build_start", started_at, model=model_name, pretrained=pretrained, device=str(device))
    model = _build_model(model_name, len(class_names), pretrained, freeze_backbone, fine_tune_strategy, dropout).to(device)
    _modal_training_phase(job_id, "model_build_done", started_at, model=model_name, device=str(device))
    if pretrained and _modal_sync_torch_cache_commit_enabled():
        _modal_training_phase(job_id, "torch_cache_commit_start", started_at)
        _commit_modal_torch_cache_volume()
        _modal_training_phase(job_id, "torch_cache_commit_done", started_at)
    elif pretrained:
        _modal_training_phase(job_id, "torch_cache_commit_deferred", started_at)

    criterion = _build_criterion(class_weights, class_balancing, device, label_smoothing, class_balancing_config)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = _build_optimizer(optimizer_name, trainable_parameters, learning_rate, weight_decay, optimizer_momentum)
    scheduler = _build_scheduler(scheduler_name, optimizer, epochs, scheduler_step_size, scheduler_gamma)
    _modal_training_phase(job_id, "optimizer_ready", started_at, trainable_parameters=len(trainable_parameters))

    best_macro_f1 = 0.0
    best_accuracy = 0.0
    best_epoch = 0
    completed_epochs = 0
    final_eval_details = {"confusion_matrix": [], "per_class_metrics": {}}

    for epoch in range(1, epochs + 1):
        _modal_training_phase(job_id, "epoch_train_start", started_at, epoch=epoch, epochs=epochs)
        train_loss = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            augmentation,
            len(class_names),
            gradient_clip_norm,
            on_first_batch=lambda batch_size, current_epoch=epoch: _modal_training_phase(
                job_id,
                "epoch_train_first_batch",
                started_at,
                epoch=current_epoch,
                batch_size=batch_size,
            ),
        )
        _modal_training_phase(job_id, "epoch_train_done", started_at, epoch=epoch, train_loss=round(train_loss, 6))
        _modal_training_phase(job_id, "epoch_eval_start", started_at, epoch=epoch)
        val_loss, accuracy, macro_f1, final_eval_details = _evaluate(model, val_loader, criterion, device, class_names)
        _modal_training_phase(
            job_id,
            "epoch_eval_done",
            started_at,
            epoch=epoch,
            val_loss=round(val_loss, 6),
            accuracy=round(accuracy, 6),
            macro_f1=round(macro_f1, 6),
        )
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

        _post_job_json(
            orchestrator_url,
            job,
            "metrics",
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
                    "dropout": dropout,
                    "optimizer_momentum": optimizer_momentum,
                    "scheduler_step_size": scheduler_step_size if scheduler_name == "step" else 0,
                    "scheduler_gamma": scheduler_gamma if scheduler_name == "step" else 0,
                    "label_smoothing": label_smoothing,
                    "gradient_clip_norm": gradient_clip_norm,
                },
            },
            modal_resources=modal_resources,
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
                "dataset_materialization": dataset_materialization,
                "stage_telemetry": _modal_stage_telemetry_payload(
                    job,
                    runtime_seconds,
                    stage_events,
                    dataset_materialization,
                    gpu_type,
                    modal_resources=modal_resource_telemetry,
                ),
            },
            job=job,
            modal_resources=modal_resources,
        )
        if _should_stop_training_early(
            epoch=epoch,
            epochs=epochs,
            best_epoch=best_epoch,
            early_stopping_patience=early_stopping_patience,
            best_accuracy=best_accuracy,
            best_macro_f1=best_macro_f1,
            target_metric=str(config.get("target_metric", "macro_f1")),
        ):
            break

    runtime_seconds = time.time() - started_at
    estimated_cost_usd = runtime_seconds * _modal_gpu_price_per_second(gpu_type)
    _modal_training_phase(job_id, "final_eval_start", started_at)
    test_loss, test_accuracy, test_macro_f1, test_eval_details = _evaluate(
        model,
        test_loader,
        criterion,
        device,
        class_names,
        collect_examples=True,
    )
    _modal_training_phase(job_id, "final_eval_done", started_at)
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
    demo_images = _demo_images_from_test_examples(test_eval_details, class_names)
    _modal_training_phase(job_id, "export_start", started_at)
    export_bundle = _export_trained_champion_bundle(
        model=model,
        model_name=model_name,
        class_names=class_names,
        image_size=image_size,
        preprocessing=preprocessing,
        model_profile=model_profile,
        training_config=config,
        dataset=dataset,
        job_id=job_id,
    )
    _modal_training_phase(job_id, "export_done", started_at, status=export_bundle.get("status", ""))
    runtime_seconds = time.time() - started_at
    estimated_cost_usd = runtime_seconds * _modal_gpu_price_per_second(gpu_type)
    model_latency_ms = float(model_profile.get("estimated_latency_ms") or 0)
    preprocessing_latency_ms = float(export_bundle.get("preprocessing_latency_ms") or 0)
    pipeline_latency_ms = model_latency_ms + preprocessing_latency_ms if model_latency_ms or preprocessing_latency_ms else 0.0
    model_profile = {
        **model_profile,
        "estimated_model_latency_ms": round(model_latency_ms, 3),
        "estimated_preprocessing_latency_ms": round(preprocessing_latency_ms, 3),
        "estimated_pipeline_latency_ms": round(pipeline_latency_ms, 3),
        "estimated_latency_ms": round(pipeline_latency_ms or model_latency_ms, 3),
        "estimated_throughput_images_per_second": round(
            1000.0 / max(pipeline_latency_ms or model_latency_ms, 1.0),
            3,
        ),
        "class_labels": class_names,
        "input_shape": [1, 3, image_size, image_size],
        "preprocessing": preprocessing,
        "export_status": export_bundle.get("status", ""),
        "artifact_uri": export_bundle.get("artifact_uri", ""),
        "onnx_artifact_uri": export_bundle.get("onnx_artifact_uri", ""),
        "export_manifest_uri": export_bundle.get("manifest_uri", ""),
        "export_manifest": export_bundle.get("manifest", {}),
        "export_validation_errors": export_bundle.get("validation_errors", []),
        "modal_resources": modal_resource_telemetry,
    }
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
            "dataset_materialization": dataset_materialization,
            "stage_telemetry": _modal_stage_telemetry_payload(
                job,
                runtime_seconds,
                stage_events,
                dataset_materialization,
                gpu_type,
                modal_resources=modal_resource_telemetry,
            ),
        },
        job=job,
        modal_resources=modal_resources,
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
                "heldout_demo_images": demo_images,
                "modal_resources": modal_resource_telemetry,
            },
            "per_class_metrics": final_eval_details.get("per_class_metrics", {}),
            "confusion_matrix": final_eval_details.get("confusion_matrix", []),
            "model_profile": {
                **model_profile,
                "pretrained": pretrained,
                "freeze_backbone": freeze_backbone,
                "fine_tune_strategy": fine_tune_strategy,
                "dropout": dropout,
                "modal_resources": modal_resource_telemetry,
            },
            "holistic_scores": {
                **_holistic_scores(
                    best_macro_f1,
                    best_accuracy,
                    estimated_cost_usd,
                    runtime_seconds,
                    model_profile,
                ),
                "modal_resources": modal_resource_telemetry,
            },
            "preprocessing_summary": {
                "augmentation_policy": str(config.get("augmentation_policy", "")),
                "augmentation_policy_config": config.get("augmentation_policy_config")
                if isinstance(config.get("augmentation_policy_config"), dict)
                else {},
                "class_balancing": class_balancing,
                "sampling_strategy": sampling_strategy,
                "preprocessing": preprocessing,
                "worker_execution_metadata": _public_execution_metadata(execution_metadata),
                "dataset_materialization": dataset_materialization,
                "bbox_crop_ablation": bbox_ablation,
                "training_hyperparameters": {
                    "optimizer": optimizer_name,
                    "scheduler": scheduler_name,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "dropout": dropout,
                    "optimizer_momentum": optimizer_momentum if optimizer_name == "sgd" else 0,
                    "scheduler_step_size": scheduler_step_size if scheduler_name == "step" else 0,
                    "scheduler_gamma": scheduler_gamma if scheduler_name == "step" else 0,
                    "label_smoothing": label_smoothing,
                    "gradient_clip_norm": gradient_clip_norm,
                    "requested_batch_size": modal_resource_telemetry["requested_batch_size"],
                    "effective_batch_size": modal_resource_telemetry["effective_batch_size"],
                    "batch_size_policy": modal_resource_telemetry["batch_size_policy"],
                    "focal_loss_gamma": _bounded_float(
                        class_balancing_config.get("focal_loss_gamma"),
                        default=2.0,
                        minimum=0.5,
                        maximum=5.0,
                    )
                    if class_balancing == "focal_loss"
                    else 0,
                },
            },
            "label_quality_audit": _label_quality_audit(config, test_eval_details, class_names),
            "export_bundle": export_bundle,
            "recommendation_summary": (
                f"{model_name} finished with macro-F1 {best_macro_f1:.3f}, "
                f"accuracy {best_accuracy:.3f}, and estimated latency "
                f"{model_profile.get('estimated_latency_ms', 0):.1f}ms."
            ),
        },
        job=job,
        modal_resources=modal_resources,
    )

    _post_job_json(
        orchestrator_url,
        job,
        "complete",
        {"mlflow_run_id": f"modal-{job_id}"},
        modal_resources=modal_resources,
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
        "dataset_materialization": dataset_materialization,
    }


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes=training_volume_mounts,
    min_containers=_modal_training_min_containers(),
    buffer_containers=_modal_training_buffer_containers(),
    scaledown_window=_modal_training_scaledown_window_seconds(),
)
def train_yolo_detector(payload: dict) -> dict:
    try:
        return _train_yolo_detector_impl(payload)
    except Exception as exc:
        if _report_modal_training_retryable_failure(payload, exc):
            return {
                "status": "retryable_failure_reported",
                "job_id": str((payload.get("job") or {}).get("id") or ""),
                "error": _modal_training_error_message(exc),
            }
        raise


def _train_yolo_detector_impl(payload: dict) -> dict:
    import time

    from ultralytics import YOLO

    started_at = time.time()
    stage_events: list[dict] = []
    _MODAL_STAGE_EVENTS.set(stage_events if _modal_stage_telemetry_enabled() else None)
    job = payload["job"]
    config = job["config"]
    dataset = payload["dataset"]
    orchestrator_url = payload["orchestrator_url"].rstrip("/")

    _configure_storage_env(payload)

    job_id = job["id"]
    dataset_id = dataset["id"]
    model_name = str(config.get("model") or config.get("pretrained_weights") or "yolo11n.pt")
    modal_resources = modal_resources_from_payload(payload, config, detection_job=True)
    epochs = _positive_int(config.get("epochs"), default=8)
    batch_size = _positive_int(modal_resources.get("effective_batch_size"), default=8)
    image_size = _bounded_int(config.get("image_size"), default=640, minimum=160, maximum=1280)
    learning_rate = _positive_float(config.get("learning_rate"), default=0.001)
    confidence_threshold = _bounded_float(config.get("confidence_threshold"), default=0.25, minimum=0.01, maximum=0.99)
    iou_threshold = _bounded_float(config.get("iou_threshold"), default=0.7, minimum=0.1, maximum=0.99)
    gpu_type = str(modal_resources.get("effective_gpu_type") or config.get("gpu_type") or "T4")
    modal_function_call_id, modal_input_id = _modal_identifiers()
    modal_resources = {
        **modal_resources,
        "modal_function_call_id": modal_function_call_id,
        "modal_input_id": modal_input_id,
    }
    modal_resource_telemetry = resource_telemetry(modal_resources, effective_batch_size=batch_size)

    _modal_training_phase(job_id, "storage_configured", started_at)
    dataset_dir, dataset_materialization = _modal_training_dataset_for_job(
        payload,
        dataset=dataset,
        config=config,
        job_id=job_id,
        dataset_id=dataset_id,
        started_at=started_at,
    )
    pre_materialized = payload.get("_modal_pre_materialized_dataset")
    pre_materialized_data_config = (
        Path(str(pre_materialized.get("yolo_data_config")))
        if isinstance(pre_materialized, dict) and str(pre_materialized.get("yolo_data_config") or "").strip()
        else None
    )
    data_config_path = pre_materialized_data_config if pre_materialized_data_config else _find_yolo_data_config(dataset_dir, config)
    if data_config_path is None:
        raise ValueError(
            "YOLO detector training requires data.yaml/data.yml/dataset.yaml/dataset.yml "
            "with train/val image paths and names/nc."
        )
    if pre_materialized_data_config is None:
        data_config_path = _normalize_yolo_data_config_for_training(
            dataset_dir,
            data_config_path,
            output_root=Path(tempfile.gettempdir()) / "model-express-yolo-data-configs" / _safe_path_part(job_id),
        )
    if pre_materialized_data_config is None:
        data_config_path, subset_manifest = _prepare_yolo_dataset_tier(
            dataset_dir,
            data_config_path,
            config,
            dataset=dataset,
            output_root=Path(tempfile.gettempdir()) / "model-express-yolo-preview-subsets" / _safe_path_part(job_id),
        )
        if subset_manifest:
            dataset_materialization["subset_manifest"] = subset_manifest
    dataset_materialization["yolo_data_config"] = str(data_config_path)
    _modal_training_phase(job_id, "yolo_train_start", started_at, model=model_name, data=str(data_config_path))
    run_root = Path(tempfile.gettempdir()) / "model-express-yolo-runs" / _safe_path_part(job_id)
    detector = YOLO(model_name)
    posted_yolo_epochs: set[int] = set()
    _install_yolo_epoch_metrics_callback(
        detector,
        orchestrator_url=orchestrator_url,
        job_id=job_id,
        run_root=run_root,
        learning_rate=learning_rate,
        image_size=image_size,
        posted_epochs=posted_yolo_epochs,
        callback_identity=callback_identity(job, modal_resources),
    )
    detector.train(
        data=str(data_config_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        lr0=learning_rate,
        project=str(run_root),
        name="train",
        exist_ok=True,
        pretrained=True,
        plots=False,
        val=True,
    )
    _modal_training_phase(job_id, "yolo_train_done", started_at)
    final_yolo_epoch_rows_posted = _post_yolo_epoch_metrics(
        orchestrator_url,
        job_id,
        run_root,
        learning_rate=learning_rate,
        image_size=image_size,
        posted_epochs=posted_yolo_epochs,
        callback_identity=callback_identity(job, modal_resources),
    )
    yolo_epoch_rows_posted = len(posted_yolo_epochs) or final_yolo_epoch_rows_posted

    best_model_path = _yolo_best_model_path(run_root)
    trained_detector = YOLO(str(best_model_path)) if best_model_path is not None else detector
    class_names = _yolo_class_names(trained_detector, config)
    _modal_training_phase(job_id, "yolo_eval_start", started_at)
    validation_metrics = _collect_yolo_validation_metrics(
        trained_detector,
        data_config_path,
        split="val",
        image_size=image_size,
        batch_size=batch_size,
        class_names=class_names,
    )
    test_metrics = {}
    if _yolo_config_declares_split(data_config_path, "test"):
        test_metrics = _collect_yolo_validation_metrics(
            trained_detector,
            data_config_path,
            split="test",
            image_size=image_size,
            batch_size=batch_size,
            class_names=class_names,
        )
    _modal_training_phase(job_id, "yolo_eval_done", started_at)
    final_metrics = test_metrics or validation_metrics
    map50_95 = _metric_float(final_metrics, "mAP50_95", "map50_95", default=0.0)
    map50 = _metric_float(final_metrics, "mAP50", "map50", default=map50_95)
    precision = _metric_float(final_metrics, "precision", default=max(0.0, map50 - 0.05))
    recall = _metric_float(final_metrics, "recall", default=max(0.0, map50 - 0.07))
    box_loss = _metric_float(final_metrics, "box_loss", default=max(0.02, 1.0 - map50_95))
    cls_loss = _metric_float(final_metrics, "cls_loss", default=max(0.02, 0.75 - map50_95 * 0.5))
    dfl_loss = _metric_float(final_metrics, "dfl_loss", default=max(0.02, 0.55 - map50_95 * 0.35))
    val_loss = round(box_loss + cls_loss + dfl_loss, 6)
    runtime_seconds = time.time() - started_at
    estimated_cost_usd = runtime_seconds * _modal_gpu_price_per_second(gpu_type)
    model_profile = _yolo_model_profile(
        model_name=model_name,
        model_path=best_model_path,
        image_size=image_size,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        metrics=final_metrics,
    )
    _modal_training_phase(job_id, "export_start", started_at)
    export_bundle = _export_yolo_detector_bundle(
        model_path=best_model_path,
        model_name=model_name,
        class_names=class_names,
        image_size=image_size,
        model_profile=model_profile,
        training_config=config,
        dataset=dataset,
        job_id=job_id,
    )
    _modal_training_phase(job_id, "export_done", started_at, status=export_bundle.get("status", ""))
    runtime_seconds = time.time() - started_at
    estimated_cost_usd = runtime_seconds * _modal_gpu_price_per_second(gpu_type)
    if export_bundle.get("artifact_uri"):
        model_profile["artifact_uri"] = export_bundle.get("artifact_uri", "")
    if export_bundle.get("onnx_artifact_uri"):
        model_profile["onnx_artifact_uri"] = export_bundle.get("onnx_artifact_uri", "")
    if export_bundle.get("manifest_uri"):
        model_profile["export_manifest_uri"] = export_bundle.get("manifest_uri", "")
    model_profile["export_status"] = export_bundle.get("status", "")
    model_profile["export_manifest"] = export_bundle.get("manifest", {})
    model_profile["export_validation_errors"] = export_bundle.get("validation_errors", [])

    if yolo_epoch_rows_posted == 0:
        _post_job_json(
            orchestrator_url,
            job,
            "metrics",
            {
                "epoch": max(1, epochs),
                "metrics": {
                    "train_loss": val_loss,
                    "val_loss": val_loss,
                    "box_loss": round(box_loss, 6),
                    "cls_loss": round(cls_loss, 6),
                    "dfl_loss": round(dfl_loss, 6),
                    "mAP50_95": round(map50_95, 6),
                    "map50_95": round(map50_95, 6),
                    "mAP50": round(map50, 6),
                    "map50": round(map50, 6),
                    "precision": round(precision, 6),
                    "recall": round(recall, 6),
                    "learning_rate": learning_rate,
                    "image_size": image_size,
                },
            },
            modal_resources=modal_resources,
        )
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
            "best_macro_f1": round(map50_95, 6),
            "best_accuracy": round(map50, 6),
            "best_map50_95": round(map50_95, 6),
            "best_map50": round(map50, 6),
            "best_precision": round(precision, 6),
            "best_recall": round(recall, 6),
            "target_metric": "mAP50_95",
            "final_train_loss": val_loss,
            "final_val_loss": val_loss,
            "epochs_completed": epochs,
            "modal_function_call_id": modal_function_call_id,
            "modal_input_id": modal_input_id,
            "dataset_materialization": dataset_materialization,
            "stage_telemetry": _modal_stage_telemetry_payload(
                job,
                runtime_seconds,
                stage_events,
                dataset_materialization,
                gpu_type,
                modal_resources=modal_resource_telemetry,
            ),
        },
        job=job,
        modal_resources=modal_resources,
    )
    _post_training_run_evaluation(
        orchestrator_url,
        job_id,
        {
            "objective_profile": {
                "target_metric": str(config.get("target_metric", "mAP50_95")),
                "task_type": "object_detection",
                "metric_preferences": ["mAP50_95", "mAP50", "precision", "recall", "latency_p95_ms"],
                "split_strategy": "official_yolo_train_val_test_when_present",
                "heldout_split": "test" if test_metrics else "val",
                "heldout_test_map50_95": round(map50_95, 6),
                "heldout_test_map50": round(map50, 6),
                "heldout_test_precision": round(precision, 6),
                "heldout_test_recall": round(recall, 6),
                "heldout_test_box_loss": round(box_loss, 6),
                "heldout_test_cls_loss": round(cls_loss, 6),
                "heldout_test_dfl_loss": round(dfl_loss, 6),
                "modal_resources": modal_resource_telemetry,
            },
            "per_class_metrics": _yolo_per_class_metrics(class_names, final_metrics),
            "confusion_matrix": [],
            "model_profile": {
                **model_profile,
                "modal_resources": modal_resource_telemetry,
            },
            "holistic_scores": {
                **_detection_holistic_scores(
                    map50_95=map50_95,
                    map50=map50,
                    precision=precision,
                    recall=recall,
                    box_loss=box_loss,
                    cls_loss=cls_loss,
                    dfl_loss=dfl_loss,
                    estimated_cost_usd=estimated_cost_usd,
                    runtime_seconds=runtime_seconds,
                    model_profile=model_profile,
                ),
                "modal_resources": modal_resource_telemetry,
            },
            "preprocessing_summary": {
                "task_type": "object_detection",
                "preserved_yolo_config": str(data_config_path),
                "preserved_official_splits": True,
                "worker_execution_metadata": {"dataset_materialization": dataset_materialization},
                "training_hyperparameters": {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "requested_batch_size": modal_resource_telemetry["requested_batch_size"],
                    "effective_batch_size": modal_resource_telemetry["effective_batch_size"],
                    "batch_size_policy": modal_resource_telemetry["batch_size_policy"],
                    "epochs": epochs,
                    "image_size": image_size,
                },
            },
            "export_bundle": export_bundle,
            "recommendation_summary": (
                f"{model_name} detector finished with mAP50-95 {map50_95:.3f}, "
                f"mAP50 {map50:.3f}, recall {recall:.3f}, and estimated latency "
                f"{model_profile.get('estimated_latency_ms', 0):.1f}ms."
            ),
        },
        job=job,
        modal_resources=modal_resources,
    )
    _post_job_json(
        orchestrator_url,
        job,
        "complete",
        {"mlflow_run_id": f"modal-yolo-{job_id}"},
        modal_resources=modal_resources,
    )
    return {
        "job_id": job_id,
        "model": model_name,
        "classes": class_names,
        "mAP50_95": map50_95,
        "mAP50": map50,
        "estimated_cost_usd": estimated_cost_usd,
        "runtime_seconds": runtime_seconds,
        "dataset_materialization": dataset_materialization,
    }


def _modal_training_dataset_for_job(
    payload: dict,
    *,
    dataset: dict,
    config: dict,
    job_id: str,
    dataset_id: str,
    started_at: float,
) -> tuple[Path, dict]:
    pre_materialized = payload.get("_modal_pre_materialized_dataset")
    if isinstance(pre_materialized, dict):
        dataset_dir = Path(str(pre_materialized.get("dataset_dir") or ""))
        telemetry = pre_materialized.get("telemetry") if isinstance(pre_materialized.get("telemetry"), dict) else {}
        cache_root = Path(str(pre_materialized.get("cache_root") or _modal_training_dataset_cache_root()))
        materialization_scope = str(pre_materialized.get("materialization_scope") or "modal_batch_local")
        _modal_training_phase(
            job_id,
            "dataset_local_materialization_reused",
            started_at,
            dataset_id=dataset_id,
            cache_root=cache_root,
            dataset_dir=dataset_dir,
        )
        materialization = {
            **telemetry,
            "dataset_training_cache": "local",
            "dataset_materialization_reused_from_batch": True,
            **_modal_dataset_cache_relationship_fields(
                cache_root,
                materialization_scope=materialization_scope,
                training_cache_root=cache_root,
            ),
        }
        subset_manifest = pre_materialized.get("subset_manifest")
        if isinstance(subset_manifest, dict):
            materialization["subset_manifest"] = subset_manifest
        yolo_data_config = str(pre_materialized.get("yolo_data_config") or "").strip()
        if yolo_data_config:
            materialization["yolo_data_config"] = yolo_data_config
        return dataset_dir, materialization

    from worker.datasets.cache import ensure_dataset_materialized

    dataset_cache_root = _modal_training_dataset_cache_root()
    _modal_training_phase(
        job_id,
        "dataset_local_materialization_start",
        started_at,
        dataset_id=dataset_id,
        cache_root=dataset_cache_root,
    )
    materialized = ensure_dataset_materialized(
        dataset_id=dataset_id,
        storage_uri=dataset["storage_uri"],
        checksum_sha256=_dataset_checksum(dataset, config),
        cache_root=dataset_cache_root,
    )
    _modal_training_phase(
        job_id,
        "dataset_local_materialization_done",
        started_at,
        status=materialized.telemetry.get("dataset_materialization_status"),
        cache_hit=materialized.telemetry.get("dataset_materialization_cache_hit"),
        bytes_downloaded=materialized.telemetry.get("dataset_materialization_bytes_downloaded"),
    )
    return materialized.dataset_dir, {
        **materialized.telemetry,
        "dataset_training_cache": "local",
        **_modal_dataset_cache_relationship_fields(
            dataset_cache_root,
            materialization_scope="modal_training_local",
        ),
    }


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    timeout=60 * 60,
    volumes=training_volume_mounts,
    min_containers=_modal_training_min_containers(),
    buffer_containers=_modal_training_buffer_containers(),
    scaledown_window=_modal_training_scaledown_window_seconds(),
)
def train_modal_preview_batch(payload: dict) -> dict:
    return _train_modal_preview_batch_impl(payload)


def _train_modal_preview_batch_impl(payload: dict) -> dict:
    normalized = _validate_modal_preview_batch_payload(payload)
    batch = normalized["batch"]
    jobs = normalized["jobs"]
    if batch["task_type"] == "object_detection" and not _yolo_batch_preview_enabled():
        return _unsupported_modal_preview_batch_result(
            batch,
            jobs,
            reason="yolo_preview_batch_disabled",
        )
    if batch["task_type"] not in {"image_classification", "object_detection"}:
        return _unsupported_modal_preview_batch_result(
            batch,
            jobs,
            reason="task_type_not_supported_by_batch_runner",
        )

    from worker.datasets.cache import ensure_dataset_materialized

    _configure_storage_env(payload)

    dataset = normalized["dataset"]
    representative_config = jobs[0].get("config") if isinstance(jobs[0].get("config"), dict) else {}
    batch_cache_root = _modal_preview_batch_cache_root(batch["batch_id"])
    materialized = ensure_dataset_materialized(
        dataset_id=batch["dataset_id"],
        storage_uri=dataset["storage_uri"],
        checksum_sha256=_dataset_checksum(dataset, representative_config),
        cache_root=batch_cache_root,
    )
    pre_materialized = {
        "dataset_dir": str(materialized.dataset_dir),
        "telemetry": materialized.telemetry,
        "cache_root": str(batch_cache_root),
        "materialization_scope": "modal_batch_local",
    }
    runner_status = "classification_batch_completed"
    if batch["task_type"] == "object_detection":
        data_config_path = _find_yolo_data_config(materialized.dataset_dir, representative_config)
        if data_config_path is None:
            raise ValueError(
                "YOLO detector preview batch requires data.yaml/data.yml/dataset.yaml/dataset.yml "
                "with train/val image paths and names/nc."
            )
        yolo_output_root = batch_cache_root / "_preview_subsets" / _safe_path_part(batch["batch_id"])
        data_config_path = _normalize_yolo_data_config_for_training(
            materialized.dataset_dir,
            data_config_path,
            output_root=yolo_output_root / "_normalized",
        )
        data_config_path, subset_manifest = _prepare_yolo_dataset_tier(
            materialized.dataset_dir,
            data_config_path,
            representative_config,
            dataset=dataset,
            batch=batch,
            output_root=yolo_output_root,
            force_preview=True,
        )
        pre_materialized["yolo_data_config"] = str(data_config_path)
        if subset_manifest:
            pre_materialized["subset_manifest"] = subset_manifest
        runner_status = "yolo_batch_completed"
    job_results: list[dict] = []
    for index, job in enumerate(jobs):
        job_payload = {
            **payload,
            "job": job,
            "dataset": dataset,
            "_modal_pre_materialized_dataset": pre_materialized,
        }
        try:
            if batch["task_type"] == "object_detection":
                result = _train_yolo_detector_impl(job_payload)
            else:
                result = _train_image_classifier_impl(job_payload)
        except Exception as exc:
            reported = _report_modal_training_retryable_failure(job_payload, exc)
            job_results.append(
                {
                    "job_id": str(job.get("id") or ""),
                    "status": "retryable_failure_reported" if reported else "failed",
                    "error": _modal_training_error_message(exc),
                    "batch_index": index,
                    "batch_size": len(jobs),
                }
            )
            continue
        job_results.append(
            {
                "job_id": str(job.get("id") or ""),
                "status": "succeeded",
                "batch_index": index,
                "batch_size": len(jobs),
                "result": _modal_preview_batch_public_job_result(result),
            }
        )
    return {
        "schema_version": "modal_preview_batch_result.v1",
        "status": "completed",
        "runner_status": runner_status,
        "batch_id": batch["batch_id"],
        "batch_key": batch["batch_key"],
        "project_id": batch["project_id"],
        "plan_id": batch["plan_id"],
        "dataset_id": batch["dataset_id"],
        "dataset_cache_key": batch["dataset_cache_key"],
        "training_tier": batch["training_tier"],
        "task_type": batch["task_type"],
        "batch_size": len(jobs),
        "dataset_dir": str(materialized.dataset_dir),
        "dataset_materialization": {
            **materialized.telemetry,
            "dataset_training_cache": "local",
            "dataset_materialization_reused_by_jobs": len(jobs),
            **({"subset_manifest": pre_materialized["subset_manifest"]} if isinstance(pre_materialized.get("subset_manifest"), dict) else {}),
            **_modal_dataset_cache_relationship_fields(
                batch_cache_root,
                materialization_scope="modal_batch_local",
                training_cache_root=batch_cache_root,
            ),
        },
        "job_results": job_results,
    }


def _unsupported_modal_preview_batch_result(batch: dict, jobs: list[dict], *, reason: str) -> dict:
    return {
        "schema_version": "modal_preview_batch_result.v1",
        "status": "unsupported",
        "runner_status": "remote_batch_shell_unsupported",
        "batch_id": batch["batch_id"],
        "batch_key": batch["batch_key"],
        "project_id": batch["project_id"],
        "plan_id": batch["plan_id"],
        "dataset_id": batch["dataset_id"],
        "dataset_cache_key": batch["dataset_cache_key"],
        "training_tier": batch["training_tier"],
        "task_type": batch["task_type"],
        "batch_size": len(jobs),
        "job_results": [
            {
                "job_id": str(job.get("id") or ""),
                "status": "unsupported",
                "reason": reason,
                "batch_index": index,
                "batch_size": len(jobs),
            }
            for index, job in enumerate(jobs)
        ],
    }


def _modal_preview_batch_cache_root(batch_id: str) -> Path:
    return Path(tempfile.gettempdir()) / "model-express" / "batches" / _safe_path_part(batch_id)


def _modal_preview_batch_public_job_result(result: object) -> dict:
    if not isinstance(result, dict):
        return {}
    public_keys = (
        "job_id",
        "model",
        "classes",
        "best_accuracy",
        "best_macro_f1",
        "estimated_cost_usd",
        "runtime_seconds",
        "device",
    )
    return {key: result[key] for key in public_keys if key in result}


def _validate_modal_preview_batch_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Modal preview batch payload must be a dictionary.")
    batch = payload.get("batch")
    if not isinstance(batch, dict):
        raise ValueError("Modal preview batch payload requires a batch object.")
    if str(batch.get("schema_version") or "") != "modal_preview_batch.v1":
        raise ValueError("Modal preview batch schema_version must be modal_preview_batch.v1.")

    required_batch_fields = (
        "batch_id",
        "batch_key",
        "project_id",
        "plan_id",
        "dataset_id",
        "dataset_cache_key",
        "training_tier",
        "task_type",
    )
    normalized_batch = {field: str(batch.get(field) or "").strip() for field in required_batch_fields}
    for field in ("subset_fraction", "subset_seed", "split_policy", "image_size_family"):
        value = str(batch.get(field) or "").strip()
        if value:
            normalized_batch[field] = value
    missing = [field for field, value in normalized_batch.items() if not value]
    if missing:
        raise ValueError(f"Modal preview batch is missing required fields: {', '.join(missing)}.")
    if normalized_batch["training_tier"] != "preview":
        raise ValueError("Modal preview batch training_tier must be preview.")

    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) < 2:
        raise ValueError("Modal preview batch requires at least two jobs.")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Modal preview batch payload requires a dataset object.")
    if str(dataset.get("id") or "").strip() != normalized_batch["dataset_id"]:
        raise ValueError("Modal preview batch dataset id does not match the batch contract.")
    if not str(dataset.get("storage_uri") or "").strip():
        raise ValueError("Modal preview batch dataset requires a storage_uri.")
    if not str(payload.get("orchestrator_url") or "").strip():
        raise ValueError("Modal preview batch payload requires orchestrator_url.")

    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"Modal preview batch job at index {index} must be a dictionary.")
        job_id = str(job.get("id") or "").strip()
        if not job_id:
            raise ValueError(f"Modal preview batch job at index {index} is missing id.")
        if str(job.get("template") or "").strip() != "train_experiment":
            raise ValueError(f"Modal preview batch job {job_id} must use train_experiment.")
        if str(job.get("project_id") or "").strip() != normalized_batch["project_id"]:
            raise ValueError(f"Modal preview batch job {job_id} project_id does not match.")
        config = job.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"Modal preview batch job {job_id} requires config.")
        if str(config.get("provider") or "").strip().lower() != "modal":
            raise ValueError(f"Modal preview batch job {job_id} provider must be modal.")
        if str(config.get("plan_id") or "").strip() != normalized_batch["plan_id"]:
            raise ValueError(f"Modal preview batch job {job_id} plan_id does not match.")
        if str(config.get("dataset_id") or "").strip() != normalized_batch["dataset_id"]:
            raise ValueError(f"Modal preview batch job {job_id} dataset_id does not match.")
        if _modal_preview_batch_training_tier(config) != normalized_batch["training_tier"]:
            raise ValueError(f"Modal preview batch job {job_id} training_tier does not match.")
        if _modal_preview_batch_dataset_cache_key(config) != normalized_batch["dataset_cache_key"]:
            raise ValueError(f"Modal preview batch job {job_id} dataset_cache_key does not match.")
        if _modal_preview_batch_task_type(config) != normalized_batch["task_type"]:
            raise ValueError(f"Modal preview batch job {job_id} task_type does not match.")

    return {
        "batch": normalized_batch,
        "jobs": jobs,
        "dataset": dataset,
    }


def _modal_preview_batch_training_tier(config: dict) -> str:
    for key in ("training_tier", "dataset_tier"):
        value = str(config.get(key) or "").strip().lower()
        if value:
            return value
    modal_batch = config.get("modal_batch") if isinstance(config.get("modal_batch"), dict) else {}
    return str(modal_batch.get("training_tier") or "").strip().lower()


def _modal_preview_batch_dataset_cache_key(config: dict) -> str:
    materialization = config.get("dataset_materialization") if isinstance(config.get("dataset_materialization"), dict) else {}
    cache_key = materialization.get("dataset_cache_key")
    if isinstance(cache_key, str) and cache_key.strip():
        return cache_key.strip()
    checksum = config.get("dataset_checksum_sha256")
    if isinstance(checksum, str) and checksum.strip():
        return f"sha256-{checksum.strip().lower()}"
    return ""


def _modal_preview_batch_task_type(config: dict) -> str:
    task_type = str(config.get("task_type") or "").strip().lower()
    if task_type:
        return task_type
    model_kind = str(config.get("model_kind") or "").strip().lower()
    model = str(config.get("model") or "").strip().lower()
    if model_kind == "ultralytics_yolo_detector" or model.startswith("yolo"):
        return "object_detection"
    return "image_classification"


def _preview_dataset_tiers_enabled() -> bool:
    return _bool(os.getenv("MODEL_EXPRESS_PREVIEW_DATASET_TIERS"), default=False)


def _yolo_batch_preview_enabled() -> bool:
    return _bool(os.getenv("MODEL_EXPRESS_YOLO_BATCH_PREVIEW"), default=False)


def _dataset_tier_config(
    config: dict,
    *,
    dataset: dict,
    task_type: str,
    dataset_dir: Path | None = None,
    batch: dict | None = None,
    force_preview: bool = False,
) -> dict:
    tier = str(config.get("training_tier") or config.get("dataset_tier") or "").strip().lower()
    if not tier and isinstance(config.get("modal_batch"), dict):
        tier = str(config["modal_batch"].get("training_tier") or "").strip().lower()
    if force_preview:
        tier = "preview"
    if tier not in {"preview", "full", "champion_validation"}:
        tier = "full"
    fraction = _bounded_float(
        config.get("preview_fraction", config.get("dataset_fraction", config.get("preview_dataset_fraction", 0.25))),
        default=0.25,
        minimum=0.01,
        maximum=1.0,
    )
    seed = _bounded_int(config.get("preview_seed", config.get("seed", 42)), default=42, minimum=0, maximum=2**31 - 1)
    split_policy = str(config.get("split_policy") or "official_or_deterministic").strip().lower()
    image_size = _positive_int(config.get("image_size"), default=0)
    image_size_family = str(config.get("image_size_family") or "").strip().lower()
    if not image_size_family:
        image_size_family = _image_size_family(task_type, image_size)
    if isinstance(batch, dict):
        fraction = _bounded_float(batch.get("subset_fraction", fraction), default=fraction, minimum=0.01, maximum=1.0)
        seed = _bounded_int(batch.get("subset_seed", seed), default=seed, minimum=0, maximum=2**31 - 1)
        split_policy = str(batch.get("split_policy") or split_policy).strip().lower()
        image_size_family = str(batch.get("image_size_family") or image_size_family).strip().lower()
    return {
        "enabled": _preview_dataset_tiers_enabled(),
        "tier": tier,
        "task_type": task_type,
        "dataset_checksum": _dataset_checksum(dataset, config) or str(config.get("dataset_checksum_sha256") or ""),
        "fraction": fraction,
        "seed": seed,
        "split_policy": split_policy,
        "image_size_family": image_size_family,
        "dataset_dir": str(dataset_dir) if dataset_dir is not None else "",
    }


def _image_size_family(task_type: str, image_size: int) -> str:
    if image_size <= 0:
        return f"{task_type}:default"
    bucket = ((image_size + 31) // 32) * 32
    return f"{task_type}:{bucket}"


def _prepare_yolo_dataset_tier(
    dataset_dir: Path,
    data_config_path: Path,
    config: dict,
    *,
    dataset: dict,
    output_root: Path,
    batch: dict | None = None,
    force_preview: bool = False,
) -> tuple[Path, dict]:
    tier_config = _dataset_tier_config(
        config,
        dataset=dataset,
        task_type="object_detection",
        dataset_dir=dataset_dir,
        batch=batch,
        force_preview=force_preview,
    )
    if not tier_config["enabled"] or tier_config["tier"] != "preview":
        return data_config_path, {}
    from worker.datasets.tiers import materialize_yolo_preview_subset

    return materialize_yolo_preview_subset(
        dataset_dir=dataset_dir,
        data_config_path=data_config_path,
        output_root=output_root,
        dataset_checksum=str(tier_config["dataset_checksum"]),
        fraction=float(tier_config["fraction"]),
        seed=int(tier_config["seed"]),
        split_policy=str(tier_config["split_policy"]),
        image_size_family=str(tier_config["image_size_family"]),
    )


YOLO_CONFIG_FILE_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")


def _find_yolo_data_config(dataset_dir: Path, config: dict) -> Path | None:
    configured = str(config.get("yolo_data_config") or config.get("data_yaml") or "").strip()
    candidates: list[Path] = []
    if configured:
        configured_path = Path(configured)
        candidates.append(configured_path if configured_path.is_absolute() else dataset_dir / configured)
    for name in YOLO_CONFIG_FILE_NAMES:
        candidates.append(dataset_dir / name)
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.name.lower() in YOLO_CONFIG_FILE_NAMES:
            candidates.append(path)
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        if _looks_like_yolo_data_config(resolved):
            return resolved
    return None


def _looks_like_yolo_data_config(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return ("train:" in text and ("val:" in text or "valid:" in text)) and ("names:" in text or "nc:" in text)


def _normalize_yolo_data_config_for_training(
    dataset_dir: Path,
    data_config_path: Path,
    *,
    output_root: Path,
) -> Path:
    loaded = _load_yolo_training_config(data_config_path)

    normalized = dict(loaded)
    normalized["path"] = str(dataset_dir.resolve())
    for split in ("train", "val", "test"):
        source_key = "valid" if split == "val" and "val" not in normalized and "valid" in normalized else split
        if source_key not in normalized:
            continue
        normalized[split] = _normalize_yolo_split_value(
            dataset_dir,
            data_config_path,
            loaded,
            normalized[source_key],
        )
    normalized.pop("valid", None)

    missing_required = [
        split
        for split in ("train", "val")
        if split not in normalized or not _yolo_split_value_has_images(normalized[split])
    ]
    if missing_required:
        raise ValueError(
            "YOLO detector training could not resolve local image paths for "
            f"{', '.join(missing_required)} from {data_config_path}."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    normalized_path = output_root / "data.yaml"
    normalized_path.write_text(_dump_yolo_training_config(normalized), encoding="utf-8")
    return normalized_path


def _load_yolo_training_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except Exception:
        loaded = _load_simple_yolo_training_config(text)
    else:
        loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"YOLO data config {path} must be a mapping.")
    return loaded


def _dump_yolo_training_config(config: dict) -> str:
    try:
        import yaml
    except Exception:
        return _dump_simple_yolo_training_config(config)
    return yaml.safe_dump(config, sort_keys=False)


def _load_simple_yolo_training_config(text: str) -> dict:
    config: dict = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = _strip_yaml_comment(lines[index]).rstrip()
        index += 1
        if not raw.strip() or raw.startswith((" ", "\t")) or ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            config[key] = _parse_simple_yaml_value(value)
            continue
        block: list[str] = []
        while index < len(lines):
            child = _strip_yaml_comment(lines[index]).rstrip()
            if child and not child.startswith((" ", "\t")):
                break
            if child.strip():
                block.append(child.strip())
            index += 1
        config[key] = _parse_simple_yaml_block(block)
    return config


def _strip_yaml_comment(line: str) -> str:
    in_quote = ""
    for index, char in enumerate(line):
        if char in {"'", '"'}:
            in_quote = "" if in_quote == char else char if not in_quote else in_quote
            continue
        if char == "#" and not in_quote:
            return line[:index]
    return line


def _parse_simple_yaml_value(value: str) -> object:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_unquote_yaml_scalar(part.strip()) for part in inner.split(",")]
    unquoted = _unquote_yaml_scalar(value)
    try:
        return int(unquoted)
    except ValueError:
        return unquoted


def _parse_simple_yaml_block(block: list[str]) -> object:
    if not block:
        return ""
    if all(line.startswith("- ") for line in block):
        return [_unquote_yaml_scalar(line[2:].strip()) for line in block]
    mapped: dict[int | str, str] = {}
    for line in block:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        parsed_key: int | str
        try:
            parsed_key = int(key)
        except ValueError:
            parsed_key = key
        mapped[parsed_key] = _unquote_yaml_scalar(value.strip())
    if mapped and all(isinstance(key, int) for key in mapped):
        return [mapped[key] for key in sorted(mapped)]
    return mapped


def _unquote_yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        if value[0] == '"':
            try:
                return str(json.loads(value))
            except json.JSONDecodeError:
                pass
        return value[1:-1]
    return value


def _dump_simple_yolo_training_config(config: dict) -> str:
    lines: list[str] = []
    for key in ("path", "train", "val", "test", "nc"):
        if key in config:
            lines.append(f"{key}: {_format_simple_yaml_scalar(config[key])}")
    names = config.get("names")
    if isinstance(names, dict):
        lines.append("names:")
        for key in sorted(names, key=lambda item: int(item) if str(item).isdigit() else str(item)):
            lines.append(f"  {key}: {_format_simple_yaml_scalar(names[key])}")
    elif isinstance(names, list):
        lines.append("names:")
        for name in names:
            lines.append(f"  - {_format_simple_yaml_scalar(name)}")
    elif names is not None:
        lines.append(f"names: {_format_simple_yaml_scalar(names)}")
    for key, value in config.items():
        if key in {"path", "train", "val", "test", "nc", "names", "valid"}:
            continue
        if isinstance(value, (str, int, float, bool)):
            lines.append(f"{key}: {_format_simple_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _format_simple_yaml_scalar(value: object) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(_format_simple_yaml_scalar(item) for item in value) + "]"
    text = str(value)
    if not text or any(char in text for char in ":#[]{}") or text.strip() != text:
        return json.dumps(text)
    return text


def _normalize_yolo_split_value(
    dataset_dir: Path,
    data_config_path: Path,
    config: dict,
    value: object,
) -> object:
    if isinstance(value, list):
        return [
            str(_resolve_existing_yolo_path(dataset_dir, data_config_path, config, str(item)))
            for item in value
            if str(item or "").strip()
        ]
    if value is None:
        return value
    text = str(value).strip()
    if not text:
        return value
    return str(_resolve_existing_yolo_path(dataset_dir, data_config_path, config, text))


def _resolve_existing_yolo_path(
    dataset_dir: Path,
    data_config_path: Path,
    config: dict,
    value: str,
) -> Path:
    candidates = _yolo_local_path_candidates(dataset_dir, data_config_path, config, value)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    suffix = _normalized_yolo_suffix(value)
    if suffix:
        for candidate in sorted(dataset_dir.rglob(Path(suffix).name), key=lambda path: str(path)):
            if _path_suffix(candidate, suffix) and candidate.exists():
                return candidate.resolve()
    return candidates[0].resolve() if candidates else (data_config_path.parent / value).resolve()


def _yolo_local_path_candidates(
    dataset_dir: Path,
    data_config_path: Path,
    config: dict,
    value: str,
) -> list[Path]:
    path = Path(value)
    candidates: list[Path] = []
    config_parent = data_config_path.parent
    if path.is_absolute():
        candidates.append(path)
        for suffix in _absolute_path_suffixes(path):
            candidates.append(dataset_dir / suffix)
            candidates.append(config_parent / suffix)
        return _unique_paths(candidates)

    root_value = str(config.get("path") or "").strip()
    if root_value:
        root = Path(root_value)
        if root.is_absolute():
            if root.name:
                candidates.append(dataset_dir / root.name / path)
                candidates.append(config_parent / root.name / path)
            for suffix in _absolute_path_suffixes(root):
                candidates.append(dataset_dir / suffix / path)
                candidates.append(config_parent / suffix / path)
        else:
            candidates.append((config_parent / root / path).resolve())
    candidates.append(config_parent / path)
    candidates.append(dataset_dir / path)
    return _unique_paths(candidates)


def _absolute_path_suffixes(path: Path) -> list[Path]:
    parts = [part for part in path.parts if part and part != path.anchor]
    suffixes: list[Path] = []
    for index in range(len(parts)):
        suffixes.append(Path(*parts[index:]))
    return suffixes


def _normalized_yolo_suffix(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        suffixes = _absolute_path_suffixes(path)
        return suffixes[-1].as_posix() if suffixes else ""
    return value.replace("\\", "/").lstrip("./").strip()


def _path_suffix(path: Path, suffix: str) -> bool:
    normalized_path = path.resolve().as_posix().rstrip("/").lower()
    normalized_suffix = suffix.replace("\\", "/").strip("/").lower()
    return bool(normalized_suffix) and normalized_path.endswith(f"/{normalized_suffix}")


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _yolo_split_value_has_images(value: object) -> bool:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if item is None:
            continue
        path = Path(str(item))
        if path.is_dir() and any(child.suffix.lower() in IMAGE_SUFFIXES for child in path.rglob("*") if child.is_file()):
            return True
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            return True
        if path.is_file():
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                lines = []
            for line in lines:
                image_path = Path(line.strip())
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                    return True
    return False


def _yolo_config_declares_split(path: Path, split: str) -> bool:
    split = "val" if split == "valid" else split
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    keys = [split]
    if split == "val":
        keys.append("valid")
    return any(re.search(rf"(?m)^\s*{re.escape(key)}\s*:", text) is not None for key in keys)


def _collect_yolo_validation_metrics(
    detector,
    data_config_path: Path,
    *,
    split: str,
    image_size: int,
    batch_size: int,
    class_names: list[str] | None = None,
) -> dict:
    metrics = detector.val(
        data=str(data_config_path),
        split=split,
        imgsz=image_size,
        batch=batch_size,
        plots=False,
    )
    return _yolo_metrics_from_object(metrics, class_names=class_names)


def _yolo_metrics_from_object(metrics: object, *, class_names: list[str] | None = None) -> dict:
    results_dict = getattr(metrics, "results_dict", None)
    if not isinstance(results_dict, dict):
        results_dict = {}
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", None)
    out = {
        "mAP50_95": _first_metric_value(results_dict, "metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50_95", "map50_95"),
        "mAP50": _first_metric_value(results_dict, "metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50"),
        "precision": _first_metric_value(results_dict, "metrics/precision(B)", "metrics/precision", "precision"),
        "recall": _first_metric_value(results_dict, "metrics/recall(B)", "metrics/recall", "recall"),
        "box_loss": _first_metric_value(results_dict, "val/box_loss", "box_loss"),
        "cls_loss": _first_metric_value(results_dict, "val/cls_loss", "cls_loss"),
        "dfl_loss": _first_metric_value(results_dict, "val/dfl_loss", "dfl_loss"),
    }
    if box is not None:
        out["mAP50_95"] = out["mAP50_95"] or _object_float(box, "map")
        out["mAP50"] = out["mAP50"] or _object_float(box, "map50")
        out["precision"] = out["precision"] or _object_float(box, "mp")
        out["recall"] = out["recall"] or _object_float(box, "mr")
        per_class_metrics = _yolo_per_class_metrics_from_box(box, class_names or _yolo_class_names_from_metrics(metrics))
        if per_class_metrics:
            out["per_class_metrics"] = per_class_metrics
    if isinstance(speed, dict):
        inference_ms = _number(speed.get("inference"))
        preprocess_ms = _number(speed.get("preprocess"))
        postprocess_ms = _number(speed.get("postprocess"))
        if inference_ms > 0:
            out["latency_model_ms"] = inference_ms
        if preprocess_ms > 0:
            out["latency_preprocess_ms"] = preprocess_ms
        if postprocess_ms > 0:
            out["latency_postprocess_ms"] = postprocess_ms
    cleaned: dict[str, object] = {}
    for key, value in out.items():
        if key == "per_class_metrics":
            if isinstance(value, dict) and value:
                cleaned[key] = value
            continue
        if _number(value) > 0:
            cleaned[key] = round(float(value), 6)
    return cleaned


def _install_yolo_epoch_metrics_callback(
    detector,
    *,
    orchestrator_url: str,
    job_id: str,
    run_root: Path,
    learning_rate: float,
    image_size: int,
    posted_epochs: set[int],
    callback_identity: dict | None = None,
) -> None:
    def post_epoch_metrics(_trainer=None) -> None:
        _post_yolo_epoch_metrics(
            orchestrator_url,
            job_id,
            run_root,
            learning_rate=learning_rate,
            image_size=image_size,
            posted_epochs=posted_epochs,
            callback_identity=callback_identity,
        )

    add_callback = getattr(detector, "add_callback", None)
    if not callable(add_callback):
        return
    for event_name in ("on_train_epoch_end", "on_fit_epoch_end"):
        try:
            add_callback(event_name, post_epoch_metrics)
        except Exception as exc:
            print(f"[model-express] failed to register YOLO {event_name} metric callback: {exc}")


def _post_yolo_epoch_metrics(
    orchestrator_url: str,
    job_id: str,
    run_root: Path,
    *,
    learning_rate: float,
    image_size: int,
    posted_epochs: set[int] | None = None,
    callback_identity: dict | None = None,
) -> int:
    results_path = _find_yolo_results_csv(run_root)
    if results_path is None:
        return 0
    posted = 0
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                epoch = int(_first_metric_value(row, "epoch")) or posted + 1
                metrics = _yolo_epoch_metrics_from_row(row, learning_rate=learning_rate, image_size=image_size)
                if not metrics:
                    continue
                normalized_epoch = max(1, epoch)
                if posted_epochs is not None and normalized_epoch in posted_epochs:
                    continue
                identity = callback_identity if isinstance(callback_identity, dict) else {}
                _post_json(
                    f"{orchestrator_url}/jobs/{job_id}/metrics",
                    {
                        "epoch": normalized_epoch,
                        "metrics": metrics,
                        **identity,
                    },
                )
                if posted_epochs is not None:
                    posted_epochs.add(normalized_epoch)
                posted += 1
    except Exception as exc:
        print(f"[model-express] failed to post YOLO epoch metrics from {results_path}: {exc}")
        return posted
    return posted


def _find_yolo_results_csv(run_root: Path) -> Path | None:
    candidates = [
        run_root / "train" / "results.csv",
        run_root / "results.csv",
    ]
    candidates.extend(sorted(run_root.glob("**/results.csv"), key=lambda path: len(path.parts)))
    for path in candidates:
        if path.is_file():
            return path
    return None


def _yolo_epoch_metrics_from_row(row: dict, *, learning_rate: float, image_size: int) -> dict:
    metrics = {
        "mAP50_95": _first_metric_value(row, "metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50_95", "map50_95"),
        "mAP50": _first_metric_value(row, "metrics/mAP50(B)", "metrics/mAP50", "mAP50", "map50"),
        "precision": _first_metric_value(row, "metrics/precision(B)", "metrics/precision", "precision"),
        "recall": _first_metric_value(row, "metrics/recall(B)", "metrics/recall", "recall"),
        "box_loss": _first_metric_value(row, "val/box_loss", "box_loss"),
        "cls_loss": _first_metric_value(row, "val/cls_loss", "cls_loss"),
        "dfl_loss": _first_metric_value(row, "val/dfl_loss", "dfl_loss"),
    }
    train_box = _first_metric_value(row, "train/box_loss")
    train_cls = _first_metric_value(row, "train/cls_loss")
    train_dfl = _first_metric_value(row, "train/dfl_loss")
    val_box = _first_metric_value(row, "val/box_loss")
    val_cls = _first_metric_value(row, "val/cls_loss")
    val_dfl = _first_metric_value(row, "val/dfl_loss")
    if train_box > 0 or train_cls > 0 or train_dfl > 0:
        metrics["train_loss"] = train_box + train_cls + train_dfl
    if val_box > 0 or val_cls > 0 or val_dfl > 0:
        metrics["val_loss"] = val_box + val_cls + val_dfl
    metrics["learning_rate"] = _first_metric_value(row, "lr/pg0", "lr/pg1", "lr/pg2", "learning_rate") or learning_rate
    metrics["image_size"] = image_size
    if metrics["mAP50_95"] > 0:
        metrics["map50_95"] = metrics["mAP50_95"]
    if metrics["mAP50"] > 0:
        metrics["map50"] = metrics["mAP50"]
    return {key: round(float(value), 6) for key, value in metrics.items() if _number(value) > 0}


def _first_metric_value(values: dict, *keys: str) -> float:
    normalized_values = {}
    for raw_key, raw_value in values.items():
        normalized_values[str(raw_key).strip()] = raw_value
    for key in keys:
        number = _number(normalized_values.get(key))
        if number > 0:
            return number
    return 0.0


def _object_float(obj: object, attr: str) -> float:
    return _number(getattr(obj, attr, None))


def _numeric_sequence(value: object) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                continue
            number = _number(item)
            if number >= 0:
                out.append(number)
        return out
    number = _number(value)
    return [number] if number > 0 else []


def _numeric_matrix(value: object) -> list[list[float]]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        try:
            value = value.detach()
        except Exception:
            pass
    if hasattr(value, "cpu"):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if not isinstance(value, (list, tuple)):
        return []
    out: list[list[float]] = []
    for row in value:
        if not isinstance(row, (list, tuple)):
            continue
        numbers = [_number(item) for item in row]
        if any(number > 0 for number in numbers):
            out.append(numbers)
    return out


def _class_index_sequence(value: object, fallback_count: int) -> list[int]:
    values = _numeric_sequence(value)
    if values:
        return [int(item) for item in values]
    return list(range(max(0, fallback_count)))


def _yolo_class_names_from_metrics(metrics: object) -> list[str]:
    names = getattr(metrics, "names", None)
    if isinstance(names, dict):
        return [str(names[key]).strip() for key in sorted(names) if str(names[key]).strip()][:200]
    if isinstance(names, list):
        return [str(item).strip() for item in names if str(item).strip()][:200]
    return []


def _yolo_class_name(class_names: list[str], class_index: int, row_index: int) -> str:
    if 0 <= class_index < len(class_names) and class_names[class_index].strip():
        return class_names[class_index].strip()
    if 0 <= row_index < len(class_names) and class_names[row_index].strip():
        return class_names[row_index].strip()
    return f"class_{class_index if class_index >= 0 else row_index}"


def _yolo_metric_at(values: list[float], row_index: int, class_index: int) -> float:
    if 0 <= class_index < len(values):
        return values[class_index]
    if 0 <= row_index < len(values):
        return values[row_index]
    return 0.0


def _yolo_per_class_metrics_from_box(box: object, class_names: list[str]) -> dict:
    precision_values = _numeric_sequence(getattr(box, "p", None))
    recall_values = _numeric_sequence(getattr(box, "r", None))
    ap50_values = _numeric_sequence(getattr(box, "ap50", None))
    map_values = _numeric_sequence(getattr(box, "maps", None))
    all_ap = _numeric_matrix(getattr(box, "all_ap", None))
    row_count = max(len(precision_values), len(recall_values), len(ap50_values), len(map_values), len(all_ap))
    if row_count == 0:
        return {}
    class_indices = _class_index_sequence(getattr(box, "ap_class_index", None), row_count)
    out: dict[str, dict[str, float]] = {}
    for row_index in range(row_count):
        class_index = class_indices[row_index] if row_index < len(class_indices) else row_index
        ap_row = all_ap[row_index] if row_index < len(all_ap) else []
        ap50 = _yolo_metric_at(ap50_values, row_index, class_index)
        if ap50 <= 0 and ap_row:
            ap50 = ap_row[0]
        map50_95 = _yolo_metric_at(map_values, row_index, class_index)
        if map50_95 <= 0 and ap_row:
            positive_ap = [value for value in ap_row if value > 0]
            if positive_ap:
                map50_95 = sum(positive_ap) / len(positive_ap)
        precision = _yolo_metric_at(precision_values, row_index, class_index)
        recall = _yolo_metric_at(recall_values, row_index, class_index)
        if precision <= 0 and recall <= 0 and ap50 <= 0 and map50_95 <= 0:
            continue
        out[_yolo_class_name(class_names, class_index, row_index)] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "AP50": round(ap50, 6),
            "AP50_95": round(map50_95, 6),
        }
    return _yolo_per_class_metrics_with_macro_average(out)


def _metric_float(values: dict, *keys: str, default: float) -> float:
    for key in keys:
        number = _number(values.get(key))
        if number > 0:
            return float(number)
    return float(default)


def _number(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _yolo_best_model_path(run_root: Path) -> Path | None:
    candidates = []
    for name in ("best.pt", "last.pt"):
        candidates.extend(run_root.rglob(name))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def _yolo_class_names(detector, config: dict) -> list[str]:
    for key in ("class_names", "class_labels", "classes"):
        value = config.get(key)
        if isinstance(value, list):
            names = [str(item).strip() for item in value if str(item).strip()]
            if names:
                return names[:200]
    names = getattr(detector, "names", None)
    if isinstance(names, dict):
        return [str(names[key]).strip() for key in sorted(names) if str(names[key]).strip()][:200]
    if isinstance(names, list):
        return [str(item).strip() for item in names if str(item).strip()][:200]
    metadata = config.get("metadata_summary") if isinstance(config.get("metadata_summary"), dict) else {}
    yolo_summary = metadata.get("yolo_summary") if isinstance(metadata.get("yolo_summary"), dict) else {}
    value = yolo_summary.get("class_names")
    if isinstance(value, list):
        names = [str(item).strip() for item in value if str(item).strip()]
        if names:
            return names[:200]
    return ["class_0"]


def _yolo_model_profile(
    *,
    model_name: str,
    model_path: Path | None,
    image_size: int,
    class_names: list[str],
    confidence_threshold: float,
    iou_threshold: float,
    metrics: dict,
) -> dict:
    size_mb = 0.0
    if model_path is not None and model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
    model_latency_ms = _metric_float(
        metrics,
        "latency_model_ms",
        default=_fallback_yolo_latency_ms(model_name, image_size),
    )
    preprocess_ms = _metric_float(metrics, "latency_preprocess_ms", default=max(1.0, image_size / 512))
    postprocess_ms = _metric_float(metrics, "latency_postprocess_ms", default=max(1.0, len(class_names) * 0.08))
    pipeline_ms = model_latency_ms + preprocess_ms + postprocess_ms
    return {
        "task_type": "object_detection",
        "model_kind": "ultralytics_yolo_detector",
        "runtime": "onnx",
        "model_size_mb": round(size_mb, 3),
        "estimated_model_latency_ms": round(model_latency_ms, 3),
        "estimated_preprocessing_latency_ms": round(preprocess_ms, 3),
        "estimated_postprocessing_latency_ms": round(postprocess_ms, 3),
        "estimated_pipeline_latency_ms": round(pipeline_ms, 3),
        "estimated_latency_ms": round(pipeline_ms, 3),
        "latency_p50_ms": round(pipeline_ms, 3),
        "latency_p95_ms": round(pipeline_ms * 1.35, 3),
        "estimated_throughput_images_per_second": round(1000.0 / max(pipeline_ms, 1.0), 3),
        "image_size": image_size,
        "input_shape": [1, 3, image_size, image_size],
        "class_labels": class_names,
        "confidence_threshold": round(confidence_threshold, 4),
        "iou_threshold": round(iou_threshold, 4),
        "pretrained": True,
    }


def _fallback_yolo_latency_ms(model_name: str, image_size: int) -> float:
    normalized = model_name.lower()
    base = 14.0
    if "yolo11n" in normalized:
        base = 8.0
    elif "yolo11s" in normalized:
        base = 14.0
    elif "yolo11m" in normalized:
        base = 24.0
    elif "yolo11l" in normalized:
        base = 38.0
    elif "yolo11x" in normalized:
        base = 62.0
    return base * max(0.35, (image_size / 640) ** 2)


def _export_yolo_detector_bundle(
    *,
    model_path: Path | None,
    model_name: str,
    class_names: list[str],
    image_size: int,
    model_profile: dict,
    training_config: dict,
    dataset: dict,
    job_id: str,
) -> dict:
    if model_path is None or not model_path.exists():
        return _export_error("YOLO_CHECKPOINT_UNAVAILABLE", "Ultralytics did not produce best.pt or last.pt.")
    try:
        from ultralytics import YOLO
        from worker.datasets.storage import upload_file_to_s3_uri
        from worker.exporting.artifacts import produce_existing_champion_export_manifest
    except Exception as exc:
        return _export_error("EXPORT_DEPENDENCY_UNAVAILABLE", str(exc))

    export_dir = Path(os.getenv("WORKER_CHAMPION_EXPORT_ROOT", ".cache/champion_exports")) / _safe_path_part(job_id) / "yolo"
    validation_errors: list[str] = []
    onnx_path: Path | None = None
    try:
        exported = YOLO(str(model_path)).export(format="onnx", imgsz=image_size, opset=12)
        candidate = Path(str(exported))
        if candidate.exists():
            onnx_path = candidate
    except Exception as exc:
        validation_errors.append(f"YOLO_ONNX_EXPORT_FAILED: {exc}")

    source_path = onnx_path or model_path
    artifact_format = "onnx" if onnx_path is not None else "pytorch"
    try:
        manifest = produce_existing_champion_export_manifest(
            export_dir=export_dir,
            source_artifact_path=source_path,
            artifact_format=artifact_format,
            model_name=model_name,
            class_names=class_names,
            image_size=image_size,
            preprocessing={"resize_strategy": "letterbox", "normalization": "none"},
            model_profile=model_profile,
            training_config={**training_config, "task_type": "object_detection", "model_kind": "ultralytics_yolo_detector"},
            sample_input_shape=[1, 3, image_size, image_size],
        )
        remote_base = _artifact_remote_base_uri(dataset, job_id)
        public_manifest, artifact_uris = _upload_manifest_artifacts(manifest, remote_base, upload_file_to_s3_uri)
        validation_errors.extend(_artifact_errors(public_manifest))
        manifest_uri = f"{remote_base}/manifest.json"
        manifest_path = export_dir / "manifest.remote.json"
        manifest_path.write_text(json.dumps(public_manifest, indent=2, sort_keys=True), encoding="utf-8")
        upload_file_to_s3_uri(manifest_path, manifest_uri)
        onnx_artifact = next((item for item in artifact_uris if item["format"] == "onnx"), None)
        primary_artifact = onnx_artifact or (artifact_uris[0] if artifact_uris else None)
        status = "READY" if primary_artifact else "PENDING_ARTIFACT"
        return {
            "status": status,
            "format": primary_artifact["format"] if primary_artifact else artifact_format,
            "artifact_uri": primary_artifact["uri"] if primary_artifact else "",
            "onnx_artifact_uri": onnx_artifact["uri"] if onnx_artifact else "",
            "manifest_uri": manifest_uri,
            "manifest": public_manifest,
            "validation_errors": validation_errors,
        }
    except Exception as exc:
        validation_errors.append(f"EXPORT_FAILED: {exc}")
        return {
            "status": "FAILED",
            "format": artifact_format,
            "artifact_uri": "",
            "onnx_artifact_uri": "",
            "manifest_uri": "",
            "manifest": {},
            "validation_errors": validation_errors,
        }


def _yolo_per_class_metrics(class_names: list[str], metrics: dict) -> dict:
    per_class = metrics.get("per_class_metrics")
    if isinstance(per_class, dict) and per_class:
        return _yolo_per_class_metrics_with_macro_average(
            {
                str(class_name): {
                    "precision": _metric_float(metric_values if isinstance(metric_values, dict) else {}, "precision", default=0.0),
                    "recall": _metric_float(metric_values if isinstance(metric_values, dict) else {}, "recall", default=0.0),
                    "AP50": _metric_float(metric_values if isinstance(metric_values, dict) else {}, "AP50", "mAP50", "map50", default=0.0),
                    "AP50_95": _metric_float(metric_values if isinstance(metric_values, dict) else {}, "AP50_95", "mAP50_95", "map50_95", default=0.0),
                }
                for class_name, metric_values in per_class.items()
                if str(class_name).strip() and str(class_name).lower() != "macro avg"
            }
        )
    return {}


def _yolo_per_class_metrics_with_macro_average(per_class: dict) -> dict:
    cleaned: dict[str, dict[str, float]] = {}
    for class_name, metric_values in per_class.items():
        if not isinstance(metric_values, dict):
            continue
        normalized = {
            "precision": round(_number(metric_values.get("precision")), 6),
            "recall": round(_number(metric_values.get("recall")), 6),
            "AP50": round(_number(metric_values.get("AP50")), 6),
            "AP50_95": round(_number(metric_values.get("AP50_95")), 6),
        }
        if any(value > 0 for value in normalized.values()):
            cleaned[str(class_name).strip()] = normalized
    values = list(cleaned.values())
    if values:
        cleaned["macro avg"] = {
            "precision": round(sum(row["precision"] for row in values) / len(values), 6),
            "recall": round(sum(row["recall"] for row in values) / len(values), 6),
            "AP50": round(sum(row["AP50"] for row in values) / len(values), 6),
            "AP50_95": round(sum(row["AP50_95"] for row in values) / len(values), 6),
        }
    return cleaned


def _detection_holistic_scores(
    *,
    map50_95: float,
    map50: float,
    precision: float,
    recall: float,
    box_loss: float,
    cls_loss: float,
    dfl_loss: float,
    estimated_cost_usd: float,
    runtime_seconds: float,
    model_profile: dict,
) -> dict:
    latency_ms = float(model_profile.get("estimated_pipeline_latency_ms") or model_profile.get("estimated_latency_ms") or 0)
    quality_score = (map50_95 * 0.62) + (map50 * 0.18) + (recall * 0.14) + (precision * 0.06)
    loss_total = box_loss + cls_loss + dfl_loss
    loss_score = max(0.0, min(1.0, 1.0 - loss_total / 3.0))
    latency_score = max(0.0, min(1.0, 1.0 - latency_ms / 160.0))
    cost_score = max(0.0, min(1.0, 1.0 - estimated_cost_usd / 10.0))
    runtime_score = max(0.0, min(1.0, 1.0 - runtime_seconds / 1800.0))
    overall_score = quality_score * 0.70 + loss_score * 0.12 + latency_score * 0.10 + cost_score * 0.05 + runtime_score * 0.03
    return {
        "quality_score": round(quality_score, 6),
        "loss_health_score": round(loss_score, 6),
        "latency_score": round(latency_score, 6),
        "cost_score": round(cost_score, 6),
        "runtime_score": round(runtime_score, 6),
        "overall_score": round(overall_score, 6),
        "detection_metrics": {
            "mAP50_95": round(map50_95, 6),
            "mAP50": round(map50, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "box_loss": round(box_loss, 6),
            "cls_loss": round(cls_loss, 6),
            "dfl_loss": round(dfl_loss, 6),
        },
    }


@app.function(
    image=image,
    timeout=_modal_dataset_materialization_timeout_seconds(),
    volumes=dataset_volume_mounts,
)
def materialize_image_dataset(payload: dict) -> dict:
    from worker.datasets.cache import ensure_dataset_materialized

    _configure_storage_env(payload)

    dataset = payload["dataset"]
    dataset_id = dataset["id"]
    job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    _reload_modal_dataset_volume()
    materialized = ensure_dataset_materialized(
        dataset_id=dataset_id,
        storage_uri=dataset["storage_uri"],
        checksum_sha256=_dataset_checksum(dataset, config),
        cache_root=DATASET_MATERIALIZATION_ROOT,
    )
    _commit_modal_dataset_volume()
    dataset_materialization = {
        **materialized.telemetry,
        **_modal_dataset_cache_relationship_fields(
            DATASET_MATERIALIZATION_ROOT,
            materialization_scope="modal_dataset_volume",
        ),
    }
    return {
        "dataset_id": dataset_id,
        "dataset_dir": str(materialized.dataset_dir),
        "dataset_materialization": dataset_materialization,
    }


@app.function(
    image=image,
    timeout=_modal_dataset_profile_timeout_seconds(),
)
def profile_image_dataset(payload: dict) -> dict:
    from worker.datasets.cache import ensure_dataset_materialized
    from worker.datasets.metadata_discovery import build_metadata_import_payload
    from worker.datasets.profiler import profile_image_folder

    _configure_storage_env(payload)

    dataset = payload["dataset"]
    dataset_id = dataset["id"]
    job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    with tempfile.TemporaryDirectory(prefix=f"model-express-profile-{dataset_id}-") as cache_root:
        materialized = ensure_dataset_materialized(
            dataset_id=dataset_id,
            storage_uri=dataset["storage_uri"],
            checksum_sha256=_dataset_checksum(dataset, config),
            cache_root=cache_root,
        )
        dataset_dir = materialized.dataset_dir
        return {
            "profile": profile_image_folder(dataset_dir),
            "metadata_import": build_metadata_import_payload(dataset_dir),
            "dataset_materialization": materialized.telemetry,
        }


def _configure_storage_env(payload: dict) -> None:
    os.environ["S3_ENDPOINT_URL"] = payload["s3_endpoint_url"]
    os.environ["AWS_ACCESS_KEY_ID"] = payload["aws_access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = payload["aws_secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = payload["aws_default_region"]
    os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_ROOT))


def _dataset_checksum(dataset: dict, config: dict | None = None) -> str:
    for key in ("checksum_sha256", "sha256", "checksum"):
        value = dataset.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = dataset.get("metadata")
    if isinstance(metadata, dict):
        for key in ("checksum_sha256", "sha256", "checksum"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(config, dict):
        for key in ("dataset_checksum_sha256", "checksum_sha256", "sha256", "checksum"):
            value = config.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        materialization = config.get("dataset_materialization")
        if isinstance(materialization, dict):
            for key in ("dataset_checksum_sha256", "checksum_sha256", "sha256", "checksum"):
                value = materialization.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            cache_key = materialization.get("dataset_cache_key")
            if isinstance(cache_key, str) and cache_key.startswith("sha256-"):
                return cache_key.removeprefix("sha256-").strip()
    return ""


def _reload_modal_dataset_volume() -> None:
    reload = getattr(dataset_volume, "reload", None)
    if callable(reload):
        reload()


def _commit_modal_dataset_volume() -> None:
    commit = getattr(dataset_volume, "commit", None)
    if callable(commit):
        commit()


def _reload_modal_torch_cache_volume() -> None:
    reload = getattr(torch_cache_volume, "reload", None)
    if callable(reload):
        reload()


def _commit_modal_torch_cache_volume() -> None:
    commit = getattr(torch_cache_volume, "commit", None)
    if callable(commit):
        commit()


def _modal_sync_torch_cache_commit_enabled() -> bool:
    return _bool(os.getenv("MODEL_EXPRESS_MODAL_SYNC_TORCH_CACHE_COMMIT"), default=False)


def _modal_training_dataset_cache_root() -> Path:
    return Path(
        os.getenv(
            "MODEL_EXPRESS_MODAL_TRAINING_DATASET_CACHE_ROOT",
            DEFAULT_MODAL_TRAINING_DATASET_CACHE_ROOT,
        )
    )


def _modal_dataset_cache_relationship_fields(
    materialization_cache_root: Path | PurePosixPath | str,
    *,
    materialization_scope: str,
    training_cache_root: Path | PurePosixPath | str | None = None,
) -> dict:
    materialization_root = _modal_cache_path_text(materialization_cache_root)
    prewarm_root = _modal_cache_path_text(DATASET_MATERIALIZATION_ROOT)
    training_root = _modal_cache_path_text(training_cache_root or _modal_training_dataset_cache_root())
    training_mount_paths = sorted(_modal_cache_path_text(path) for path in training_volume_mounts)
    prewarm_root_matches_training_root = prewarm_root == training_root
    prewarm_root_mounted_for_training = prewarm_root in training_mount_paths
    prewarm_reusable_for_training = prewarm_root_matches_training_root and prewarm_root_mounted_for_training
    if prewarm_reusable_for_training:
        reuse_status = "reusable_for_training"
        training_scope = "modal_dataset_volume"
    elif not prewarm_root_matches_training_root:
        reuse_status = "staging_only_root_mismatch"
        training_scope = "modal_training_local"
    else:
        reuse_status = "staging_only_training_mount_missing"
        training_scope = "modal_training_local"
    return {
        "dataset_materialization_cache_root": materialization_root,
        "dataset_materialization_cache_scope": materialization_scope,
        "dataset_prewarm_cache_root": prewarm_root,
        "dataset_training_cache_root": training_root,
        "dataset_training_cache_scope": training_scope,
        "dataset_training_volume_mounts": training_mount_paths,
        "dataset_prewarm_root_matches_training_root": prewarm_root_matches_training_root,
        "dataset_prewarm_root_mounted_for_training": prewarm_root_mounted_for_training,
        "dataset_prewarm_reusable_for_training": prewarm_reusable_for_training,
        "dataset_prewarm_reuse_status": reuse_status,
    }


def _modal_cache_path_text(path: Path | PurePosixPath | str) -> str:
    return str(path).replace("\\", "/").rstrip("/") or "/"


def _modal_stage_telemetry_enabled() -> bool:
    return _bool(os.getenv("MODEL_EXPRESS_REMOTE_GPU_STAGE_TELEMETRY"), default=False)


def _modal_training_phase(job_id: str, phase: str, started_at: float, **fields: object) -> None:
    import time

    elapsed = max(0.0, time.time() - started_at)
    events = _MODAL_STAGE_EVENTS.get()
    if events is not None:
        events.append(
            {
                "phase": phase,
                "elapsed_seconds": round(elapsed, 6),
                "fields": {
                    key: _modal_stage_field_value(value)
                    for key, value in fields.items()
                    if value is not None
                },
            }
        )
    field_text = " ".join(
        f"{key}={_modal_training_phase_value(value)}"
        for key, value in fields.items()
        if value is not None
    )
    suffix = f" {field_text}" if field_text else ""
    print(f"Modal training {job_id} phase={phase} elapsed={elapsed:.2f}s{suffix}", flush=True)


def _modal_training_phase_value(value: object) -> str:
    text = str(value)
    return text.replace("\n", " ").replace("\r", " ")[:120]


def _modal_stage_field_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return round(float(value), 6)
    return _modal_training_phase_value(value)


def _modal_stage_telemetry_payload(
    job: dict,
    runtime_seconds: float,
    stage_events: list[dict],
    dataset_materialization: dict,
    gpu_type: str,
    *,
    modal_resources: dict | None = None,
) -> dict:
    resources = (
        modal_resources
        if isinstance(modal_resources, dict)
        else resource_telemetry({"effective_gpu_type": gpu_type})
    )
    if not _modal_stage_telemetry_enabled():
        return {
            "schema_version": "remote_gpu_stage_telemetry_v1",
            "gpu_type": gpu_type,
            "modal_resources": resources,
            "requested_gpu_type": resources.get("requested_gpu_type", ""),
            "effective_gpu_type": resources.get("effective_gpu_type", gpu_type),
            "memory_mb": resources.get("memory_mb", 0),
            "requested_batch_size": resources.get("requested_batch_size", 0),
            "effective_batch_size": resources.get("effective_batch_size", 0),
            "batch_size_policy": resources.get("batch_size_policy", ""),
            "resource_signature": resources.get("resource_signature", ""),
            "warm_container_policy": _modal_warm_container_policy(),
        }
    materialization_seconds = _stage_duration(
        stage_events,
        "dataset_local_materialization_start",
        "dataset_local_materialization_done",
    )
    active_training_seconds = _sum_stage_durations(
        stage_events,
        ("epoch_train_start", "epoch_train_done"),
    ) + _stage_duration(stage_events, "yolo_train_start", "yolo_train_done")
    evaluation_seconds = _sum_stage_durations(
        stage_events,
        ("epoch_eval_start", "epoch_eval_done"),
    ) + _stage_duration(stage_events, "final_eval_start", "final_eval_done") + _stage_duration(
        stage_events,
        "yolo_eval_start",
        "yolo_eval_done",
    )
    export_seconds = _stage_duration(stage_events, "export_start", "export_done")
    known_seconds = materialization_seconds + active_training_seconds + evaluation_seconds + export_seconds
    payload = {
        "schema_version": "remote_gpu_stage_telemetry_v1",
        "current_stage": stage_events[-1]["phase"] if stage_events else "",
        "events": stage_events[-80:],
        "queue_wait_seconds": _job_queue_wait_seconds(job),
        "dataset_materialization_seconds": round(materialization_seconds, 6),
        "dataset_download_seconds": _float_from_payload(
            dataset_materialization,
            "dataset_materialization_download_seconds",
        ),
        "dataset_extract_seconds": _float_from_payload(
            dataset_materialization,
            "dataset_materialization_extract_seconds",
        ),
        "dataset_materialization_wait_seconds": _float_from_payload(
            dataset_materialization,
            "dataset_materialization_wait_seconds",
        ),
        "active_training_seconds": round(active_training_seconds, 6),
        "evaluation_seconds": round(evaluation_seconds, 6),
        "export_seconds": round(export_seconds, 6),
        "idle_wait_seconds": round(max(0.0, runtime_seconds - known_seconds), 6),
        "runtime_seconds": round(max(0.0, runtime_seconds), 6),
        "gpu_type": gpu_type,
        "modal_resources": resources,
        "requested_gpu_type": resources.get("requested_gpu_type", ""),
        "effective_gpu_type": resources.get("effective_gpu_type", gpu_type),
        "memory_mb": resources.get("memory_mb", 0),
        "requested_batch_size": resources.get("requested_batch_size", 0),
        "effective_batch_size": resources.get("effective_batch_size", 0),
        "batch_size_policy": resources.get("batch_size_policy", ""),
        "resource_signature": resources.get("resource_signature", ""),
        "warm_container_policy": _modal_warm_container_policy(),
    }
    return payload


def _stage_duration(stage_events: list[dict], start_phase: str, done_phase: str) -> float:
    started = None
    for event in stage_events:
        phase = str(event.get("phase") or "")
        elapsed = _float_from_payload(event, "elapsed_seconds")
        if phase == start_phase:
            started = elapsed
        elif phase == done_phase and started is not None:
            return max(0.0, elapsed - started)
    return 0.0


def _sum_stage_durations(stage_events: list[dict], phase_pair: tuple[str, str]) -> float:
    start_phase, done_phase = phase_pair
    total = 0.0
    started = None
    for event in stage_events:
        phase = str(event.get("phase") or "")
        elapsed = _float_from_payload(event, "elapsed_seconds")
        if phase == start_phase:
            started = elapsed
        elif phase == done_phase and started is not None:
            total += max(0.0, elapsed - started)
            started = None
    return round(total, 6)


def _float_from_payload(payload: dict, key: str) -> float:
    try:
        value = float(payload.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, value), 6)


def _job_queue_wait_seconds(job: dict) -> float:
    created_at = _parse_datetime(job.get("created_at"))
    if created_at is None:
        return 0.0
    return round(max(0.0, (datetime.now(timezone.utc) - created_at).total_seconds()), 6)


def _parse_datetime(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _modal_warm_container_policy() -> dict:
    return {
        "min_containers": _modal_training_min_containers() or 0,
        "buffer_containers": _modal_training_buffer_containers() or 0,
        "scaledown_window_seconds": _modal_training_scaledown_window_seconds() or 0,
        "cost_sensitive_defaults_enabled": _modal_cost_sensitive_defaults_enabled(),
    }


class _MetadataBundleImageDataset:
    def __init__(self, spec: dict, transform=None):
        self.classes = list(spec["class_names"])
        self.class_to_idx = {class_name: index for index, class_name in enumerate(self.classes)}
        self.samples = list(spec["samples"])
        self.targets = list(spec["targets"])
        self.metadata_splits = list(spec["splits"])
        self.metadata_records = list(spec["records"])
        self.transform = transform
        self.target_transform = None
        self.loader = _load_rgb_image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def _load_rgb_image(path: str):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _fetch_training_metadata_bundle(orchestrator_url: str, dataset_id: str, config: dict) -> dict | None:
    import requests

    metadata_import_id = str(config.get("metadata_import_id") or "").strip()
    limit = _bounded_int(
        config.get("metadata_bundle_page_size") or config.get("metadata_bundle_limit"),
        default=DEFAULT_METADATA_BUNDLE_PAGE_SIZE,
        minimum=1,
        maximum=10_000,
    )
    max_records = _bounded_int(
        config.get("metadata_bundle_max_records"),
        default=DEFAULT_METADATA_BUNDLE_MAX_RECORDS,
        minimum=limit,
        maximum=200_000,
    )
    offset = 0
    merged_bundle: dict | None = None
    while offset < max_records:
        params: dict[str, object] = {
            "purpose": "training",
            "include": "bbox",
            "limit": min(limit, max_records - offset),
            "offset": offset,
        }
        if metadata_import_id:
            params["metadata_import_id"] = metadata_import_id
        try:
            response = requests.get(
                f"{orchestrator_url}/datasets/{dataset_id}/metadata/bundle",
                params=params,
                timeout=_orchestrator_report_timeout_seconds(),
            )
            if response.status_code in METADATA_ENDPOINT_UNAVAILABLE_STATUS_CODES or response.status_code == 204:
                return merged_bundle
            response.raise_for_status()
            page_payload = response.json()
        except Exception:
            return merged_bundle

        page_bundle = _coerce_metadata_bundle(page_payload)
        if not page_bundle:
            return merged_bundle
        merged_bundle = _merge_metadata_bundle_pages(merged_bundle, page_bundle)
        page_records = _metadata_manifest_records(page_bundle)
        if not _metadata_bundle_has_more(page_payload, page_bundle, len(page_records), limit):
            break
        offset += max(len(page_records), limit)
    return merged_bundle


def _load_image_data(
    dataset_dir: Path,
    batch_size: int,
    image_size: int,
    augmentation: dict,
    class_balancing: str,
    sampling_strategy: str,
    preprocessing: dict,
    class_balancing_config: dict | None = None,
    *,
    metadata_bundle: dict | None = None,
    dataset_tier_config: dict | None = None,
):
    import torch
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
    from torchvision import datasets

    normalization_metadata = _dataset_normalization_metadata(dataset_dir, preprocessing)
    if normalization_metadata is not None:
        preprocessing = {**preprocessing, "normalization_metadata": normalization_metadata}
    train_transform = _image_transform(image_size, augmentation, preprocessing, training=True)
    val_transform = _image_transform(image_size, {}, preprocessing, training=False)
    metadata_bundle_payload = _coerce_metadata_bundle(metadata_bundle)
    path_resolver = _DatasetRelativePathResolver(dataset_dir) if metadata_bundle_payload else None
    metadata_spec = _metadata_bundle_dataset_spec(dataset_dir, metadata_bundle, path_resolver=path_resolver)
    crop_strategy = str(preprocessing.get("crop_strategy", "")).lower()
    bbox_mode = str(preprocessing.get("bbox_mode", "")).lower()
    requested_bbox_crop = bbox_crop_requested(preprocessing)
    bbox_required = bbox_crop_required(preprocessing)
    compare_bbox_crop = bbox_compare_requested(preprocessing)
    bbox_lookup: dict[str, tuple[int, int, int, int]] = {}
    execution_metadata: dict = {
        "bbox_crop": {
            "requested": requested_bbox_crop,
            "applied": False,
            "annotation_count": 0,
            "mode": bbox_mode or crop_strategy or "none",
            "compare_full_image": compare_bbox_crop,
        },
        "metadata_bundle": _metadata_bundle_execution_metadata(metadata_bundle, metadata_spec),
    }
    if requested_bbox_crop:
        bbox_source = "legacy_sidecar"
        if metadata_spec is not None:
            bbox_lookup = _bbox_lookup_from_metadata_bundle(
                metadata_bundle or {},
                dataset_dir,
                path_resolver=path_resolver,
            )
            if bbox_lookup:
                bbox_source = "metadata_bundle"
        if not bbox_lookup:
            bbox_lookup = _load_bbox_lookup(dataset_dir)
        execution_metadata["bbox_crop"]["annotation_count"] = len(bbox_lookup)
        execution_metadata["bbox_crop"]["source"] = bbox_source if bbox_lookup else ""
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

    uses_metadata_aware_dataset = metadata_spec is not None or _has_root_metadata_dirs(dataset_dir)
    base_dataset = (
        _MetadataBundleImageDataset(metadata_spec)
        if metadata_spec is not None
        else _image_folder_dataset(datasets, dataset_dir)
    )
    if len(base_dataset.classes) < 2:
        raise ValueError("Training requires at least two image classes.")
    if len(base_dataset) < 2:
        raise ValueError("Training requires at least two images.")

    metadata_split_indices = _indices_from_metadata_splits(
        getattr(base_dataset, "metadata_splits", []),
        getattr(base_dataset, "targets", []),
    )
    if metadata_split_indices is None:
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
        execution_metadata["metadata_bundle"]["split_strategy"] = "deterministic_random"
    else:
        train_indices, val_indices, test_indices = metadata_split_indices
        execution_metadata["metadata_bundle"]["split_strategy"] = "metadata_official"

    tier_config = dataset_tier_config if isinstance(dataset_tier_config, dict) else {}
    if tier_config.get("enabled") and tier_config.get("tier") == "preview":
        from worker.datasets.tiers import build_classification_preview_subset

        subset_manifest = build_classification_preview_subset(
            dataset_dir=dataset_dir,
            dataset_checksum=str(tier_config.get("dataset_checksum") or ""),
            targets=list(getattr(base_dataset, "targets", [])),
            split_indices={
                "train": list(train_indices),
                "val": list(val_indices),
                "test": list(test_indices),
            },
            class_names=list(base_dataset.classes),
            fraction=float(tier_config.get("fraction") or 0.25),
            seed=int(tier_config.get("seed") or 42),
            split_policy=str(tier_config.get("split_policy") or execution_metadata["metadata_bundle"].get("split_strategy") or ""),
            image_size_family=str(tier_config.get("image_size_family") or _image_size_family("image_classification", image_size)),
        )
        train_indices = subset_manifest["indices"]["train"]
        val_indices = subset_manifest["indices"]["val"]
        test_indices = subset_manifest["indices"]["test"]
        execution_metadata["dataset_tier"] = {
            "tier": "preview",
            "subset_manifest": subset_manifest,
        }
    elif tier_config:
        execution_metadata["dataset_tier"] = {"tier": tier_config.get("tier") or "full"}

    if metadata_spec is not None:
        train_dataset = _MetadataBundleImageDataset(metadata_spec, transform=train_transform)
        val_dataset = _MetadataBundleImageDataset(metadata_spec, transform=val_transform)
        full_image_val_dataset = _MetadataBundleImageDataset(metadata_spec, transform=val_transform)
    else:
        train_dataset = _TransformedImageFolderView(base_dataset, transform=train_transform)
        val_dataset = _TransformedImageFolderView(base_dataset, transform=val_transform)
        full_image_val_dataset = _TransformedImageFolderView(base_dataset, transform=val_transform)
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

    loader_workers = _modal_dataloader_workers(uses_metadata_aware_dataset)
    loader_kwargs = _dataloader_kwargs(loader_workers)
    execution_metadata["dataloader"] = {
        "workers": loader_workers,
        "pin_memory": bool(loader_kwargs.get("pin_memory")),
        "prefetch_factor": loader_kwargs.get("prefetch_factor"),
    }
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, **loader_kwargs)
    if bbox_lookup and compare_bbox_crop:
        execution_metadata["_full_image_val_loader"] = DataLoader(
            full_image_val_data,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        execution_metadata["_full_image_test_loader"] = DataLoader(
            full_image_test_data,
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
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


def _modal_dataloader_workers(uses_metadata_aware_dataset: bool) -> int:
    default = (
        DEFAULT_MODAL_METADATA_DATALOADER_WORKERS
        if uses_metadata_aware_dataset
        else DEFAULT_MODAL_IMAGEFOLDER_DATALOADER_WORKERS
    )
    return _bounded_non_negative_int_env("MODEL_EXPRESS_MODAL_DATALOADER_WORKERS", default, maximum=16)


def _dataloader_kwargs(loader_workers: int) -> dict:
    kwargs = {
        "num_workers": loader_workers,
        "pin_memory": _bool(os.getenv("MODEL_EXPRESS_MODAL_DATALOADER_PIN_MEMORY"), default=True),
    }
    if loader_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = _bounded_int(
            os.getenv("MODEL_EXPRESS_MODAL_DATALOADER_PREFETCH_FACTOR"),
            default=2,
            minimum=1,
            maximum=8,
        )
    return kwargs


def _bounded_non_negative_int_env(name: str, default: int, maximum: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(0, min(maximum, parsed))


def _image_folder_dataset(datasets, dataset_dir: Path, transform=None):
    image_folder_root = _image_folder_root(dataset_dir)
    if not _has_root_metadata_dirs(image_folder_root):
        return datasets.ImageFolder(image_folder_root, transform=transform)

    class MetadataAwareImageFolder(datasets.ImageFolder):
        def find_classes(self, directory):
            classes = sorted(
                entry.name
                for entry in os.scandir(directory)
                if entry.is_dir() and entry.name.lower() not in ROOT_METADATA_DIR_NAMES
            )
            if not classes:
                raise FileNotFoundError(f"No class folders found in {directory}.")
            class_to_idx = {class_name: index for index, class_name in enumerate(classes)}
            return classes, class_to_idx

    return MetadataAwareImageFolder(image_folder_root, transform=transform)


def _has_root_metadata_dirs(dataset_dir: Path) -> bool:
    return any(child.is_dir() and child.name.lower() in ROOT_METADATA_DIR_NAMES for child in dataset_dir.iterdir())


def _image_folder_root(dataset_dir: Path) -> Path:
    if _has_class_directories(dataset_dir):
        return dataset_dir
    for image_root in _metadata_image_root_prefixes(dataset_dir):
        candidate = dataset_dir / image_root
        if _has_class_directories(candidate):
            return candidate
    wrapper = _single_wrapper_dir(dataset_dir)
    if wrapper is not None and _has_class_directories(wrapper):
        return wrapper
    return dataset_dir


def _has_class_directories(directory: Path) -> bool:
    try:
        class_dirs = [
            child
            for child in directory.iterdir()
            if child.is_dir() and child.name.lower() not in ROOT_METADATA_DIR_NAMES and _contains_image_file(child)
        ]
    except OSError:
        return False
    return len(class_dirs) >= 2


def _contains_image_file(directory: Path) -> bool:
    try:
        return any(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in directory.rglob("*"))
    except OSError:
        return False


def _single_wrapper_dir(dataset_dir: Path) -> Path | None:
    try:
        candidates = [
            child
            for child in dataset_dir.iterdir()
            if child.is_dir()
            and child.name.lower() not in ROOT_METADATA_DIR_NAMES
            and _contains_image_file(child)
        ]
    except OSError:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return None


def _metadata_bundle_dataset_spec(
    dataset_dir: Path,
    metadata_bundle: dict | None,
    *,
    path_resolver: _DatasetRelativePathResolver | None = None,
) -> dict | None:
    bundle = _coerce_metadata_bundle(metadata_bundle)
    if not bundle:
        return None
    records = _metadata_manifest_records(bundle)
    if not records:
        return None
    class_names, label_lookup = _metadata_class_mapping(bundle, records)
    if len(class_names) < 2:
        return None

    samples: list[tuple[str, int]] = []
    targets: list[int] = []
    splits: list[str] = []
    accepted_records: list[dict] = []
    skipped_records = 0
    resolver = path_resolver or _DatasetRelativePathResolver(dataset_dir)
    for record in records:
        if not isinstance(record, dict):
            skipped_records += 1
            continue
        relative_path = _metadata_record_relative_path(record)
        if not relative_path:
            skipped_records += 1
            continue
        image_path = resolver.resolve(relative_path)
        if image_path is None:
            skipped_records += 1
            continue
        target = _metadata_record_target(record, label_lookup)
        if target is None:
            skipped_records += 1
            continue
        samples.append((str(image_path), target))
        targets.append(target)
        splits.append(_normalize_metadata_split(record.get("split")))
        accepted_records.append(record)

    if len(samples) < 2 or len(set(targets)) < 2:
        return None
    return {
        "class_names": class_names,
        "samples": samples,
        "targets": targets,
        "splits": splits,
        "records": accepted_records,
        "record_count": len(records),
        "accepted_record_count": len(samples),
        "skipped_record_count": skipped_records,
    }


def _metadata_bundle_execution_metadata(metadata_bundle: dict | None, metadata_spec: dict | None) -> dict:
    bundle = _coerce_metadata_bundle(metadata_bundle)
    if not bundle:
        return {
            "status": "unavailable",
            "applied": False,
            "record_count": 0,
            "accepted_record_count": 0,
            "skipped_record_count": 0,
        }
    records = _metadata_manifest_records(bundle)
    if metadata_spec is None:
        return {
            "status": "not_usable",
            "applied": False,
            "record_count": len(records),
            "accepted_record_count": 0,
            "skipped_record_count": len(records),
        }
    return {
        "status": "applied",
        "applied": True,
        "record_count": int(metadata_spec.get("record_count") or len(records)),
        "accepted_record_count": int(metadata_spec.get("accepted_record_count") or 0),
        "skipped_record_count": int(metadata_spec.get("skipped_record_count") or 0),
        "class_count": len(metadata_spec.get("class_names") or []),
    }


def _coerce_metadata_bundle(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}
    for key in ("bundle", "metadata_bundle", "data"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return payload


def _metadata_manifest_records(bundle: dict) -> list:
    for key in ("manifest_records", "records", "samples", "files"):
        value = bundle.get(key)
        if isinstance(value, list):
            return value
    return []


def _metadata_annotations(bundle: dict) -> list:
    for key in ("annotations", "dataset_annotations", "bboxes"):
        value = bundle.get(key)
        if isinstance(value, list):
            return value
    return []


def _metadata_classes(bundle: dict) -> list:
    value = bundle.get("classes")
    return value if isinstance(value, list) else []


def _merge_metadata_bundle_pages(merged: dict | None, page: dict) -> dict:
    if merged is None:
        return {key: list(value) if isinstance(value, list) else value for key, value in page.items()}
    for key, value in page.items():
        if isinstance(value, list):
            merged.setdefault(key, [])
            if isinstance(merged[key], list):
                if key == "classes":
                    seen = {
                        identity
                        for item in merged[key]
                        for identity in [_metadata_class_identity(item)]
                        if identity
                    }
                    for item in value:
                        identity = _metadata_class_identity(item)
                        if identity and identity in seen:
                            continue
                        merged[key].append(item)
                        if identity:
                            seen.add(identity)
                    continue
                merged[key].extend(value)
        elif key not in merged:
            merged[key] = value
    return merged


def _metadata_bundle_has_more(page_payload: object, page_bundle: dict, record_count: int, limit: int) -> bool:
    for source in (page_payload, page_bundle):
        if not isinstance(source, dict):
            continue
        has_more = source.get("has_more")
        if isinstance(has_more, bool):
            return has_more
        pagination = source.get("pagination")
        if isinstance(pagination, dict):
            has_more = pagination.get("has_more")
            if isinstance(has_more, bool):
                return has_more
            if pagination.get("next_offset") is not None:
                return True
        if source.get("next_offset") is not None:
            return True
    return record_count >= limit and limit > 0


def _metadata_class_mapping(bundle: dict, records: list) -> tuple[list[str], dict[str, int]]:
    classes = []
    seen_classes = set()
    for item in _metadata_classes(bundle):
        if not isinstance(item, dict):
            continue
        name = _first_metadata_string(item, ("class_name", "name", "label_name", "class_key", "label_key"))
        if not name:
            continue
        class_index = _optional_int(item.get("class_index"))
        identity = _metadata_class_identity(item, name=name, class_index=class_index)
        if identity and identity in seen_classes:
            continue
        if identity:
            seen_classes.add(identity)
        classes.append(
            {
                "name": name,
                "index": class_index,
                "keys": [
                    _first_metadata_string(item, ("class_key",)),
                    _first_metadata_string(item, ("label_key",)),
                    _first_metadata_string(item, ("id",)),
                    str(class_index) if class_index is not None else "",
                ],
            }
        )
    if not classes:
        names = sorted(
            {
                label
                for record in records
                if isinstance(record, dict)
                for label in [_first_metadata_string(record, ("label_name", "class_name", "label", "label_key", "class_key"))]
                if label
            }
        )
        classes = [{"name": name, "index": index, "keys": [name]} for index, name in enumerate(names)]

    classes.sort(key=lambda item: (item["index"] is None, item["index"] if item["index"] is not None else item["name"], item["name"]))
    class_names = [str(item["name"]) for item in classes]
    lookup: dict[str, int] = {}
    for index, item in enumerate(classes):
        values = [item["name"], *item["keys"]]
        for value in values:
            normalized = _metadata_lookup_key(value)
            if normalized:
                lookup[normalized] = index
    return class_names, lookup


def _metadata_class_identity(item: object, *, name: str | None = None, class_index: int | None = None) -> str:
    if not isinstance(item, dict):
        return ""
    if class_index is None:
        class_index = _optional_int(item.get("class_index"))
    if class_index is not None:
        return f"index:{class_index}"
    for key in ("class_key", "label_key", "id"):
        normalized = _metadata_lookup_key(item.get(key))
        if normalized:
            return f"key:{normalized}"
    normalized_name = _metadata_lookup_key(
        name or _first_metadata_string(item, ("class_name", "name", "label_name", "class_key", "label_key"))
    )
    if normalized_name:
        return f"name:{normalized_name}"
    return ""


def _metadata_record_target(record: dict, label_lookup: dict[str, int]) -> int | None:
    for key in ("label_key", "class_key", "label_name", "class_name", "label", "category_name", "category_id", "class_index"):
        normalized = _metadata_lookup_key(record.get(key))
        if normalized and normalized in label_lookup:
            return label_lookup[normalized]
    return None


def _metadata_record_relative_path(record: dict) -> str:
    return _first_metadata_string(
        record,
        ("relative_path", "media_path", "image_path", "file_path", "path"),
    )


def _resolve_dataset_relative_file(dataset_dir: Path, relative_path: str) -> Path | None:
    return _DatasetRelativePathResolver(dataset_dir).resolve(relative_path)


def _metadata_image_root_prefixes(dataset_dir: Path) -> list[str]:
    try:
        actual_dirs = {child.name.lower(): child.name for child in dataset_dir.iterdir() if child.is_dir()}
    except OSError:
        return []
    prefixes: list[str] = []
    for prefix in COMMON_IMAGE_ROOT_NAMES:
        actual = actual_dirs.get(prefix.lower())
        if actual and actual not in prefixes:
            prefixes.append(actual)
    wrapper = _single_wrapper_dir(dataset_dir)
    if wrapper is not None:
        wrapper_prefix = wrapper.name
        if wrapper_prefix not in prefixes:
            prefixes.append(wrapper_prefix)
        try:
            wrapper_dirs = {child.name.lower(): child.name for child in wrapper.iterdir() if child.is_dir()}
        except OSError:
            wrapper_dirs = {}
        for prefix in COMMON_IMAGE_ROOT_NAMES:
            actual = wrapper_dirs.get(prefix.lower())
            if not actual:
                continue
            nested_prefix = f"{wrapper_prefix}/{actual}"
            if nested_prefix not in prefixes:
                prefixes.append(nested_prefix)
    return prefixes


def _indices_from_metadata_splits(
    splits: list[str],
    targets: list[int] | None = None,
) -> tuple[list[int], list[int], list[int]] | None:
    if not splits:
        return None
    train_indices = [index for index, split in enumerate(splits) if split == "train"]
    val_indices = [index for index, split in enumerate(splits) if split == "val"]
    test_indices = [index for index, split in enumerate(splits) if split == "test"]
    if not train_indices:
        return None
    if not val_indices and test_indices:
        derived = _derive_validation_indices_from_train(train_indices, targets or [])
        if derived is not None:
            train_indices, val_indices = derived
        else:
            val_indices = list(test_indices)
    if not val_indices:
        return None
    if not test_indices:
        test_indices = list(val_indices)
    return train_indices, val_indices, test_indices


def _derive_validation_indices_from_train(
    train_indices: list[int],
    targets: list[int],
) -> tuple[list[int], list[int]] | None:
    if len(train_indices) < 2:
        return None
    val_size = min(max(1, int(len(train_indices) * 0.2)), len(train_indices) - 1)
    shuffled = list(train_indices)
    random.Random(42).shuffle(shuffled)
    target_counts: dict[int, int] = {}
    for index in train_indices:
        if 0 <= index < len(targets):
            target = int(targets[index])
            target_counts[target] = target_counts.get(target, 0) + 1
    if target_counts:
        preferred = [
            index
            for index in shuffled
            if 0 <= index < len(targets) and target_counts.get(int(targets[index]), 0) > 1
        ]
        candidates = preferred if len(preferred) >= val_size else shuffled
    else:
        candidates = shuffled
    val_indices = sorted(candidates[:val_size])
    val_set = set(val_indices)
    remaining_train_indices = [index for index in train_indices if index not in val_set]
    if not remaining_train_indices or not val_indices:
        return None
    return remaining_train_indices, val_indices


def _normalize_metadata_split(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"training", "train"}:
        return "train"
    if normalized in {"val", "valid", "validation", "dev"}:
        return "val"
    if normalized in {"test", "testing", "holdout", "heldout"}:
        return "test"
    return ""


def _bbox_lookup_from_metadata_bundle(
    metadata_bundle: dict,
    dataset_dir: Path,
    *,
    path_resolver: _DatasetRelativePathResolver | None = None,
) -> dict[str, tuple[int, int, int, int]]:
    bundle = _coerce_metadata_bundle(metadata_bundle)
    if not bundle:
        return {}
    sample_paths: dict[str, str] = {}
    for record in _metadata_manifest_records(bundle):
        if not isinstance(record, dict):
            continue
        relative_path = _metadata_record_relative_path(record)
        if not relative_path:
            continue
        for key in (
            _first_metadata_string(record, ("sample_key",)),
            _first_metadata_string(record, ("id",)),
            relative_path,
            Path(relative_path).name,
            Path(relative_path).stem,
        ):
            normalized = _metadata_lookup_key(key)
            if normalized:
                sample_paths[normalized] = relative_path

    boxes_by_path: dict[str, list[tuple[int, int, int, int]]] = {}
    for annotation in _metadata_annotations(bundle):
        if not isinstance(annotation, dict):
            continue
        relative_path = _metadata_record_relative_path(annotation)
        if not relative_path:
            sample_key = _metadata_lookup_key(_first_metadata_string(annotation, ("sample_key", "record_key", "manifest_key")))
            relative_path = sample_paths.get(sample_key, "")
        if not relative_path:
            continue
        bbox = _metadata_bbox(annotation.get("bbox"))
        if bbox is None:
            continue
        boxes_by_path.setdefault(relative_path, []).append(bbox)

    lookup: dict[str, tuple[int, int, int, int]] = {}
    resolver = path_resolver or _DatasetRelativePathResolver(dataset_dir)
    for relative_path, boxes in boxes_by_path.items():
        image_path = resolver.resolve(relative_path)
        if image_path is None:
            continue
        union = _union_bbox_tuples(boxes)
        if union is None:
            continue
        for key in (
            str(image_path.resolve()).lower(),
            image_path.name.lower(),
            image_path.stem.lower(),
            relative_path.lower(),
        ):
            lookup[key] = union
    return lookup


def _metadata_bbox(value: object) -> tuple[int, int, int, int] | None:
    if not isinstance(value, dict):
        return None
    try:
        if all(key in value for key in ("xmin", "ymin", "xmax", "ymax")):
            xmin = int(value["xmin"])
            ymin = int(value["ymin"])
            xmax = int(value["xmax"])
            ymax = int(value["ymax"])
        elif all(key in value for key in ("x", "y", "width", "height")):
            xmin = int(value["x"])
            ymin = int(value["y"])
            xmax = xmin + int(value["width"])
            ymax = ymin + int(value["height"])
        else:
            return None
    except (TypeError, ValueError):
        return None
    if xmax <= xmin or ymax <= ymin:
        return None
    return xmin, ymin, xmax, ymax


def _union_bbox_tuples(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _first_metadata_string(record: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _metadata_lookup_key(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _image_transform(image_size: int, augmentation: dict, preprocessing: dict, training: bool):
    return build_image_transform(image_size, augmentation, preprocessing, training)


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
    return resize_with_padding(image, image_size)


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
    return normalization_values(normalization, preprocessing)


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


def _build_criterion(class_weights, class_balancing: str, device, label_smoothing: float = 0.0, class_balancing_config: dict | None = None):
    import torch
    from torch import nn
    import torch.nn.functional as F

    weight_tensor = class_weights.to(device) if class_weights is not None else None
    label_smoothing = _bounded_float(label_smoothing, default=0.0, minimum=0.0, maximum=0.3)
    focal_gamma = _bounded_float(
        (class_balancing_config or {}).get("focal_loss_gamma"),
        default=2.0,
        minimum=0.5,
        maximum=5.0,
    )
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

        return FocalLoss(weight=weight_tensor, gamma=focal_gamma)
    return nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)


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


def _build_model(
    model_name: str,
    class_count: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    fine_tune_strategy: str = "head_only",
    dropout: float = 0.0,
):
    from torch import nn
    from torchvision import models

    normalized = model_name.lower()
    dropout = _bounded_float(dropout, default=0.0, minimum=0.0, maximum=0.7)

    if "efficientnet_b2" in normalized:
        model = _torchvision_model(models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
        return model
    if "efficientnet_b1" in normalized:
        model = _torchvision_model(models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
        return model
    if "efficientnet" in normalized:
        model = _torchvision_model(models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
        return model
    if "resnet34" in normalized:
        model = _torchvision_model(models.resnet34, models.ResNet34_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        _apply_transfer_strategy(model, "fc", freeze_backbone, fine_tune_strategy)
        model.fc = _classification_head(nn, in_features, class_count, dropout)
        return model
    if "resnet" in normalized:
        model = _torchvision_model(models.resnet18, models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        _apply_transfer_strategy(model, "fc", freeze_backbone, fine_tune_strategy)
        model.fc = _classification_head(nn, in_features, class_count, dropout)
        return model
    if "regnet_y_400mf" in normalized:
        model = _torchvision_model(models.regnet_y_400mf, models.RegNet_Y_400MF_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        _apply_transfer_strategy(model, "fc", freeze_backbone, fine_tune_strategy)
        model.fc = _classification_head(nn, in_features, class_count, dropout)
        return model
    if "convnext_tiny" in normalized:
        model = _torchvision_model(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
        return model
    if "swin_t" in normalized:
        model = _torchvision_model(models.swin_t, models.Swin_T_Weights.DEFAULT if pretrained else None)
        in_features = model.head.in_features
        _apply_transfer_strategy(model, "head", freeze_backbone, fine_tune_strategy)
        model.head = _classification_head(nn, in_features, class_count, dropout)
        return model
    if "vit_b_16" in normalized:
        model = _torchvision_model(models.vit_b_16, models.ViT_B_16_Weights.DEFAULT if pretrained else None)
        in_features = model.heads.head.in_features
        _apply_transfer_strategy(model, "heads", freeze_backbone, fine_tune_strategy)
        model.heads.head = _classification_head(nn, in_features, class_count, dropout)
        return model
    if "mobilenet_v3_large" in normalized:
        model = _torchvision_model(models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        in_features = model.classifier[-1].in_features
        _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
        _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
        return model

    model = _torchvision_model(models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    in_features = model.classifier[-1].in_features
    _apply_transfer_strategy(model, "classifier", freeze_backbone, fine_tune_strategy)
    _replace_classifier_head(nn, model.classifier, in_features, class_count, dropout)
    return model


def _classification_head(nn, in_features: int, class_count: int, dropout: float = 0.0):
    head = nn.Linear(in_features, class_count)
    dropout = _bounded_float(dropout, default=0.0, minimum=0.0, maximum=0.7)
    if dropout <= 0:
        return head
    return nn.Sequential(nn.Dropout(p=dropout), head)


def _replace_classifier_head(nn, classifier, in_features: int, class_count: int, dropout: float = 0.0) -> None:
    if len(classifier) >= 2 and isinstance(classifier[-2], nn.Dropout):
        classifier[-2].p = _bounded_float(dropout, default=0.0, minimum=0.0, maximum=0.7)
        classifier[-1] = nn.Linear(in_features, class_count)
        return
    classifier[-1] = _classification_head(nn, in_features, class_count, dropout)


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


def _build_optimizer(optimizer_name: str, parameters, learning_rate: float, weight_decay: float, momentum: float = 0.9):
    import torch

    if optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def _build_scheduler(scheduler_name: str, optimizer, epochs: int, step_size: int | None = None, gamma: float = 0.5):
    import torch

    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, int(step_size or max(1, epochs // 3))),
            gamma=_bounded_float(gamma, default=0.5, minimum=0.05, maximum=0.95),
        )
    return None


def _batch_size(labels) -> int:
    size = getattr(labels, "size", None)
    if callable(size):
        try:
            return int(size(0))
        except Exception:
            pass
    try:
        return len(labels)
    except Exception:
        return 0


def _train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    augmentation: dict | None = None,
    class_count: int = 0,
    gradient_clip_norm: float = 0.0,
    on_first_batch=None,
) -> float:
    import torch

    model.train()
    total_loss = 0.0
    total_examples = 0
    augmentation = augmentation or {}
    gradient_clip_norm = _bounded_float(gradient_clip_norm, default=0.0, minimum=0.0, maximum=10.0)

    for batch_index, (inputs, labels) in enumerate(loader, start=1):
        if batch_index == 1 and callable(on_first_batch):
            on_first_batch(_batch_size(labels))
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs, loss_labels = _apply_mixed_sample_augmentation(inputs, labels, augmentation, max(1, class_count), device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, loss_labels)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
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


def _demo_images_from_test_examples(eval_details: dict, class_names: list[str], max_total: int = 32) -> list[dict]:
    records = eval_details.get("example_predictions") if isinstance(eval_details, dict) else []
    if not isinstance(records, list):
        return []
    wrong = [record for record in records if isinstance(record, dict) and record.get("correct") is False]
    correct = [record for record in records if isinstance(record, dict) and record.get("correct") is True]
    ranked = sorted(wrong, key=lambda record: -float(record.get("confidence") or 0.0)) + _class_balanced_records(correct, class_names)

    demo_images: list[dict] = []
    seen_paths: set[str] = set()
    for record in ranked:
        path = str(record.get("path") or "")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        thumbnail_uri, width, height, size_bytes = _thumbnail_data_uri(path)
        if not thumbnail_uri:
            continue
        true_label = str(record.get("true_class") or "")
        predicted_label = str(record.get("predicted_class") or "")
        demo_images.append(
            {
                "id": f"test:{Path(path).stem}",
                "image_id": Path(path).name,
                "uri": thumbnail_uri,
                "image_uri": thumbnail_uri,
                "thumbnail_uri": thumbnail_uri,
                "class_name": true_label,
                "label": true_label,
                "true_label": true_label,
                "split": "test",
                "width": width,
                "height": height,
                "size_bytes": size_bytes,
                "metadata": {
                    "source": "heldout_test",
                    "predicted_label_at_training": predicted_label,
                    "confidence_at_training": round(float(record.get("confidence") or 0.0), 6),
                    "correct_at_training": bool(record.get("correct")),
                },
            }
        )
        if len(demo_images) >= max_total:
            break
    return demo_images


def _class_balanced_records(records: list[dict], class_names: list[str]) -> list[dict]:
    by_class = {class_name: [] for class_name in class_names}
    for record in records:
        label = str(record.get("true_class") or "")
        by_class.setdefault(label, []).append(record)
    for class_records in by_class.values():
        class_records.sort(key=lambda record: -float(record.get("confidence") or 0.0))

    out: list[dict] = []
    while any(by_class.values()):
        for class_name in sorted(by_class):
            if by_class[class_name]:
                out.append(by_class[class_name].pop(0))
    return out


def _thumbnail_data_uri(path: str, image_size: int = 224, quality: int = 76) -> tuple[str, int, int, int]:
    try:
        from PIL import Image
    except Exception:
        return "", 0, 0, 0

    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
            output_path = Path(tempfile.gettempdir()) / f"model_express_demo_{os.getpid()}_{Path(path).stem}.jpg"
            rgb.save(output_path, format="JPEG", quality=quality, optimize=True)
            payload = base64.b64encode(output_path.read_bytes()).decode("ascii")
            size_bytes = output_path.stat().st_size
            output_path.unlink(missing_ok=True)
            return f"data:image/jpeg;base64,{payload}", rgb.width, rgb.height, size_bytes
    except Exception:
        return "", 0, 0, 0


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


def _export_trained_champion_bundle(
    *,
    model,
    model_name: str,
    class_names: list[str],
    image_size: int,
    preprocessing: dict,
    model_profile: dict,
    training_config: dict,
    dataset: dict,
    job_id: str,
) -> dict:
    try:
        from worker.exporting.artifacts import produce_champion_export_artifacts
        from worker.datasets.storage import upload_file_to_s3_uri
    except Exception as exc:
        return _export_error("EXPORT_DEPENDENCY_UNAVAILABLE", str(exc))

    export_dir = Path(os.getenv("WORKER_CHAMPION_EXPORT_ROOT", ".cache/champion_exports")) / _safe_path_part(job_id) / "training"
    try:
        manifest = produce_champion_export_artifacts(
            export_dir=export_dir,
            model_name=model_name,
            class_names=class_names,
            image_size=image_size,
            model=model,
            preprocessing=preprocessing,
            model_profile=model_profile,
            training_config=training_config,
            formats=("onnx", "torchscript", "framework_native"),
        )
        remote_base = _artifact_remote_base_uri(dataset, job_id)
        public_manifest, artifact_uris = _upload_manifest_artifacts(manifest, remote_base, upload_file_to_s3_uri)
        preprocessing_latency_ms = _manifest_preprocessing_latency_ms(public_manifest)
        onnx_artifact = next((item for item in artifact_uris if item["format"] == "onnx"), None)
        primary_artifact = onnx_artifact or (artifact_uris[0] if artifact_uris else None)
        validation_errors = _artifact_errors(public_manifest)
        status = "READY" if onnx_artifact else "PENDING_ARTIFACT"
        if not artifact_uris:
            status = "FAILED" if validation_errors else "PENDING_ARTIFACT"
        manifest_uri = f"{remote_base}/manifest.json"
        manifest_path = export_dir / "manifest.remote.json"
        manifest_path.write_text(json.dumps(public_manifest, indent=2, sort_keys=True), encoding="utf-8")
        upload_file_to_s3_uri(manifest_path, manifest_uri)
        return {
            "status": status,
            "format": primary_artifact["format"] if primary_artifact else "onnx",
            "artifact_uri": primary_artifact["uri"] if primary_artifact else "",
            "onnx_artifact_uri": onnx_artifact["uri"] if onnx_artifact else "",
            "manifest_uri": manifest_uri,
            "manifest": public_manifest,
            "preprocessing_latency_ms": preprocessing_latency_ms,
            "validation_errors": validation_errors,
        }
    except Exception as exc:
        return _export_error("EXPORT_FAILED", str(exc))


def _upload_manifest_artifacts(manifest: dict, remote_base: str, upload_file_to_s3_uri) -> tuple[dict, list[dict]]:
    public_manifest = json.loads(json.dumps(manifest))
    artifact_uris: list[dict] = []
    artifacts = public_manifest.get("artifacts") if isinstance(public_manifest.get("artifacts"), list) else []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("status") != "created":
            continue
        path = Path(str(artifact.get("path") or ""))
        if not path.exists() or not path.is_file():
            artifact["status"] = "failed"
            artifact["error_code"] = "ARTIFACT_NOT_FOUND"
            artifact["error"] = f"Created artifact disappeared before upload: {path}"
            continue
        remote_uri = f"{remote_base}/{path.name}"
        upload_file_to_s3_uri(path, remote_uri)
        if str(artifact.get("format") or "") == "onnx":
            artifact["external_data"] = _upload_onnx_external_data_files(
                artifact,
                remote_base,
                path.parent,
                upload_file_to_s3_uri,
            )
        artifact["path"] = remote_uri
        artifact["uri"] = remote_uri
        artifact_uris.append({"format": str(artifact.get("format") or ""), "uri": remote_uri})
    public_manifest.pop("manifest_path", None)
    if isinstance(public_manifest.get("metadata"), dict):
        onnx_artifact = next((item for item in artifact_uris if item["format"] == "onnx"), None)
        public_manifest["metadata"]["artifact_uri"] = onnx_artifact["uri"] if onnx_artifact else ""
    return public_manifest, artifact_uris


def _manifest_preprocessing_latency_ms(manifest: dict) -> float:
    try:
        from worker.exporting.preprocessing import preprocessing_latency_p95_ms
    except Exception:
        return 0.0
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    profile = metadata.get("preprocessing_latency_profile") if isinstance(metadata, dict) else {}
    return round(preprocessing_latency_p95_ms(profile), 3)


def _upload_onnx_external_data_files(
    artifact: dict,
    remote_base: str,
    artifact_dir: Path,
    upload_file_to_s3_uri,
) -> list[dict]:
    external_data = artifact.get("external_data") if isinstance(artifact.get("external_data"), list) else []
    uploaded: list[dict] = []
    for item in external_data:
        if not isinstance(item, dict):
            continue
        relative_path = _safe_manifest_relative_path(str(item.get("path") or ""))
        local_path = Path(str(item.get("artifact_path") or ""))
        if not local_path.is_absolute():
            local_path = artifact_dir / (relative_path or local_path)
        if not relative_path:
            relative_path = _safe_manifest_relative_path(local_path.name)
        if not local_path.exists() or not local_path.is_file() or not relative_path:
            continue
        remote_uri = f"{remote_base}/{relative_path}"
        upload_file_to_s3_uri(local_path, remote_uri)
        uploaded.append(
            {
                "path": relative_path,
                "uri": remote_uri,
                "artifact_path": remote_uri,
                "bytes": local_path.stat().st_size,
            }
        )
    return uploaded


def _safe_manifest_relative_path(value: str) -> str:
    parts = []
    for part in str(value).replace("\\", "/").split("/"):
        if part in {"", ".", ".."}:
            continue
        parts.append(part)
    return "/".join(parts)


def _artifact_remote_base_uri(dataset: dict, job_id: str) -> str:
    bucket = os.getenv("MODEL_EXPRESS_ARTIFACT_BUCKET", "").strip()
    dataset_uri = str(dataset.get("storage_uri") or "")
    if not bucket:
        parsed = urlparse(dataset_uri)
        bucket = parsed.netloc if parsed.scheme == "s3" else "model-express"
    prefix = os.getenv("MODEL_EXPRESS_ARTIFACT_PREFIX", "model-express/artifacts").strip("/")
    return f"s3://{bucket}/{prefix}/{_safe_path_part(job_id)}"


def _artifact_errors(manifest: dict) -> list[str]:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), list) else []
    errors = []
    for artifact in artifacts:
        if isinstance(artifact, dict) and artifact.get("status") != "created":
            error_code = str(artifact.get("error_code") or artifact.get("status") or "EXPORT_UNAVAILABLE")
            error = str(artifact.get("error") or "")
            errors.append(f"{error_code}: {error}".strip())
    return errors


def _export_error(error_code: str, error: str) -> dict:
    return {
        "status": "FAILED",
        "format": "onnx",
        "artifact_uri": "",
        "onnx_artifact_uri": "",
        "manifest_uri": "",
        "manifest": {},
        "validation_errors": [f"{error_code}: {error}"],
    }


def _safe_path_part(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return safe or "artifact"


def _holistic_scores(best_macro_f1: float, best_accuracy: float, estimated_cost_usd: float, runtime_seconds: float, model_profile: dict) -> dict:
    latency_ms = float(model_profile.get("estimated_pipeline_latency_ms") or model_profile.get("estimated_latency_ms") or 0)
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

    response = requests.post(url, json=payload, timeout=_orchestrator_report_timeout_seconds())
    response.raise_for_status()


def _report_modal_training_retryable_failure(payload: dict, exc: Exception) -> bool:
    job = payload.get("job") if isinstance(payload.get("job"), dict) else {}
    job_id = str(job.get("id") or "")
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    orchestrator_url = str(payload.get("orchestrator_url") or "").rstrip("/")
    if not job_id or not orchestrator_url:
        return False
    message = _modal_training_error_message(exc)
    modal_resources = modal_resources_from_payload(
        payload,
        config,
        detection_job=_modal_failure_detection_job(config),
    )
    try:
        _post_json(
            f"{orchestrator_url}/jobs/{job_id}/fail",
            failure_callback_payload(job, message, modal_resources),
        )
        return True
    except Exception:
        return False


def _modal_training_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"Modal training container failed before completion: {message}"[:2000]


def _orchestrator_report_timeout_seconds() -> int:
    return _positive_int_env(
        "MODEL_EXPRESS_WORKER_REPORT_TIMEOUT_SECONDS",
        DEFAULT_ORCHESTRATOR_REPORT_TIMEOUT_SECONDS,
    )


def _post_job_json(
    orchestrator_url: str,
    job: dict,
    endpoint: str,
    payload: dict,
    *,
    modal_resources: dict | None = None,
) -> None:
    job_id = str(job.get("id") or "")
    _post_json(
        f"{orchestrator_url}/jobs/{job_id}/{endpoint}",
        {
            **payload,
            **callback_identity(job, modal_resources),
        },
    )


def _post_training_run_summary(
    orchestrator_url: str,
    job_id: str,
    payload: dict,
    *,
    job: dict | None = None,
    modal_resources: dict | None = None,
) -> None:
    if isinstance(job, dict):
        payload = {**payload, **callback_identity(job, modal_resources)}
    _post_json(f"{orchestrator_url}/jobs/{job_id}/training-run-summary", payload)


def _post_training_run_evaluation(
    orchestrator_url: str,
    job_id: str,
    payload: dict,
    *,
    job: dict | None = None,
    modal_resources: dict | None = None,
) -> None:
    if isinstance(job, dict):
        payload = {**payload, **callback_identity(job, modal_resources)}
    _post_json(f"{orchestrator_url}/jobs/{job_id}/training-run-evaluation", payload)


def _modal_failure_detection_job(config: dict) -> bool:
    model = str(config.get("model", "")).lower()
    return (
        str(config.get("task_type", "")).lower() == "object_detection"
        or str(config.get("model_kind", "")).lower() == "ultralytics_yolo_detector"
        or model.startswith("yolo")
    )


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


def _should_stop_training_early(
    *,
    epoch: int,
    epochs: int,
    best_epoch: int,
    early_stopping_patience: int,
    best_accuracy: float,
    best_macro_f1: float,
    target_metric: str,
) -> bool:
    if early_stopping_patience <= 0 or epoch <= 0 or epochs <= 0:
        return False
    if epoch - max(best_epoch, 0) < early_stopping_patience:
        return False

    if epoch >= _early_stopping_min_epoch_for_plateau(epochs):
        return True

    return epoch >= _early_stopping_min_epoch_for_egregious_metrics(epochs) and _target_metric_is_egregiously_low(
        best_accuracy=best_accuracy,
        best_macro_f1=best_macro_f1,
        target_metric=target_metric,
    )


def _early_stopping_min_epoch_for_plateau(epochs: int) -> int:
    return max(1, epochs // 2 + 1)


def _early_stopping_min_epoch_for_egregious_metrics(epochs: int) -> int:
    quarter_epoch = max(1, (epochs + 3) // 4)
    return min(_early_stopping_min_epoch_for_plateau(epochs), max(3, quarter_epoch))


def _target_metric_is_egregiously_low(*, best_accuracy: float, best_macro_f1: float, target_metric: str) -> bool:
    threshold = 0.2
    normalized = str(target_metric or "macro_f1").strip().lower()
    if normalized == "accuracy":
        return best_accuracy < threshold
    return best_macro_f1 < threshold


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


def _bounded_float(value: object, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))
