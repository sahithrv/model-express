from __future__ import annotations

from copy import deepcopy

import pytest

from worker.training.classification_execution import (
    ClassificationExecutionError,
    load_torchvision_model,
    realization_matches_policy,
    resolve_classification_execution,
)
from worker.training.execution_capabilities import (
    capability_document_v1,
    resolve_accepted_config,
)


def _job_config(requested: dict, *, mode: str = "shadow") -> dict:
    document = capability_document_v1()
    accepted = resolve_accepted_config(
        "image_classification",
        "modal_torchvision",
        requested,
    )
    return {
        "execution_spec_v1": {
            "schema_version": "execution_spec_v1",
            "capability_version": document["capability_version"],
            "task": "image_classification",
            "runner": "modal_torchvision",
            "requested_config": deepcopy(requested),
            "accepted_config": accepted,
        },
        "execution_validation_v1": {"mode": mode},
    }


def _complete_request() -> dict:
    return {
        "model": "resnet18",
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 0.0003,
        "image_size": 224,
        "optimizer": "sgd",
        "optimizer_momentum": 0,
        "scheduler": "step",
        "scheduler_step_size": 1,
        "scheduler_gamma": 0.5,
        "weight_decay": 0,
        "dropout": 0,
        "label_smoothing": 0,
        "gradient_clip_norm": 0,
        "early_stopping_patience": 0,
        "pretrained": False,
        "freeze_backbone": False,
        "fine_tune_strategy": "full",
        "augmentation": {
            "horizontal_flip": False,
            "vertical_flip": False,
            "color_jitter": False,
            "random_crop": False,
            "random_rotation": False,
            "random_erasing": False,
        },
        "augmentation_policy": "mixup",
        "augmentation_policy_config": {
            "policy_type": "mixup",
            "alpha": 0,
            "probability": 0,
        },
        "class_balancing": "focal_loss",
        "class_balancing_config": {"focal_loss_gamma": 2.0},
        "sampling_strategy": "none",
        "preprocessing": {
            "resize_strategy": "preserve_aspect_pad",
            "crop_strategy": "none",
            "normalization": "none",
            "bbox_mode": "ignore",
            "use_dataset_normalization": False,
        },
    }


def test_every_supported_classifier_field_reaches_realized_execution() -> None:
    execution = resolve_classification_execution(_job_config(_complete_request()))

    assert execution.realized_config == execution.accepted_config
    assert execution.value("optimizer_momentum") == 0
    assert execution.value("pretrained") is False
    assert execution.value("freeze_backbone") is False
    assert execution.value("weight_decay") == 0
    assert execution.value("dropout") == 0
    assert execution.augmentation["alpha"] == 0
    assert execution.augmentation["probability"] == 0
    assert execution.preprocessing["normalization"] == "none"
    assert realization_matches_policy(execution)


def test_batch_recovery_is_the_only_approved_runtime_adjustment() -> None:
    execution = resolve_classification_execution(
        _job_config(_complete_request(), mode="enforce"),
        effective_batch_size=8,
    )

    assert execution.realized_config["batch_size"] == 8
    assert execution.adjustment_policy == "batch_size_recovery"
    assert execution.fidelity_mode == "enforce"
    assert realization_matches_policy(execution)


def test_explicit_unfrozen_backbone_realizes_full_fine_tuning() -> None:
    requested = _complete_request()
    requested["fine_tune_strategy"] = "head_only"
    execution = resolve_classification_execution(_job_config(requested))

    assert execution.realized_config["freeze_backbone"] is False
    assert execution.realized_config["fine_tune_strategy"] == "full"
    assert not realization_matches_policy(execution)


def test_dataset_normalization_flag_is_realized_as_dataset_normalization() -> None:
    requested = _complete_request()
    requested["preprocessing"]["normalization"] = "imagenet"
    requested["preprocessing"]["use_dataset_normalization"] = True
    execution = resolve_classification_execution(_job_config(requested))

    assert execution.preprocessing["normalization"] == "dataset"
    assert not realization_matches_policy(execution)


def test_worker_rejects_tampered_or_wrong_version_accepted_specs() -> None:
    config = _job_config(_complete_request())
    config["execution_spec_v1"]["accepted_config"]["epochs"] = 101
    with pytest.raises(ClassificationExecutionError, match="epochs"):
        resolve_classification_execution(config)

    config = _job_config(_complete_request())
    config["execution_spec_v1"]["capability_version"] = "stale"
    with pytest.raises(ClassificationExecutionError, match="capability version"):
        resolve_classification_execution(config)


def test_classifier_rejects_out_of_range_image_size_and_detection_preprocessing() -> None:
    requested = _complete_request()
    requested["image_size"] = 640
    with pytest.raises(ClassificationExecutionError, match="image_size"):
        resolve_classification_execution(_job_config(requested))

    requested = _complete_request()
    requested["preprocessing"]["resize_strategy"] = "yolo_letterbox"
    config = _job_config(requested)
    # Simulate a tampered accepted payload because the orchestrator normally
    # removes this unsupported classifier semantic before assignment.
    config["execution_spec_v1"]["accepted_config"]["preprocessing"]["resize_strategy"] = (
        "yolo_letterbox"
    )
    with pytest.raises(ValueError, match="Unsupported resize_strategy"):
        resolve_classification_execution(config)


def test_pretrained_loading_failure_never_retries_without_weights() -> None:
    calls = []

    def failing_factory(*, weights):
        calls.append(weights)
        raise RuntimeError("pretrained weights unavailable")

    with pytest.raises(RuntimeError, match="pretrained weights unavailable"):
        load_torchvision_model(failing_factory, "DEFAULT_WEIGHTS")

    assert calls == ["DEFAULT_WEIGHTS"]


def test_legacy_flat_jobs_remain_runnable_only_in_shadow_mode() -> None:
    legacy = _complete_request()
    execution = resolve_classification_execution(legacy)
    assert execution.versioned is False
    assert execution.fidelity_mode == "shadow"

    legacy["execution_validation_v1"] = {"mode": "enforce"}
    with pytest.raises(ClassificationExecutionError, match="requires config.execution_spec_v1"):
        resolve_classification_execution(legacy)
