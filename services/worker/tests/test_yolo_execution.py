from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from worker.training.execution_capabilities import (
    capability_document_v1,
    resolve_accepted_config,
)
from worker.training.yolo_execution import (
    PINNED_ULTRALYTICS_VERSION,
    YOLO_NATIVE_AUGMENTATION_DEFAULTS,
    YOLO_NATIVE_TRAINING_DEFAULTS,
    YoloExecutionError,
    capture_yolo_trainer_arguments,
    realize_yolo_trainer_arguments,
    resolve_yolo_execution,
    ultralytics_train_kwargs,
    yolo_framework_semantic_hash,
    yolo_realization_matches_policy,
)


WORKER_ROOT = Path(__file__).resolve().parents[1]


def _request(**overrides) -> dict:
    request = {
        "template": "train_yolo_detection",
        "model": "yolo11n.pt",
        "epochs": 8,
        "batch_size": 8,
        "learning_rate": 0.001,
        "image_size": 640,
    }
    request.update(overrides)
    return request


def _job_config(requested: dict | None = None, *, mode: str = "shadow") -> dict:
    requested = deepcopy(requested or _request())
    document = capability_document_v1()
    accepted = resolve_accepted_config("object_detection", "modal_ultralytics", requested)
    return {
        "execution_spec_v1": {
            "schema_version": "execution_spec_v1",
            "capability_version": document["capability_version"],
            "task": "object_detection",
            "runner": "modal_ultralytics",
            "requested_config": requested,
            "accepted_config": accepted,
        },
        "execution_validation_v1": {"mode": mode},
    }


def _trainer_arguments(execution, **overrides) -> dict:
    kwargs = ultralytics_train_kwargs(
        execution,
        data="/datasets/data.yaml",
        project="/runs/job-1",
        name="train",
        workers=2,
    )
    arguments = {
        **kwargs,
        "model": execution.realized_config["model"],
    }
    arguments.update(overrides)
    return arguments


def test_translator_maps_every_supported_field_and_native_semantic() -> None:
    execution = resolve_yolo_execution(_job_config())
    kwargs = ultralytics_train_kwargs(
        execution,
        data="/datasets/data.yaml",
        project="/runs/job-1",
        name="train",
        workers=3,
    )

    assert kwargs["epochs"] == 8
    assert kwargs["batch"] == 8
    assert kwargs["imgsz"] == 640
    assert kwargs["lr0"] == 0.001
    assert kwargs["pretrained"] is True
    assert kwargs["data"] == "/datasets/data.yaml"
    assert kwargs["project"] == "/runs/job-1"
    assert kwargs["workers"] == 3
    assert {key: kwargs[key] for key in YOLO_NATIVE_TRAINING_DEFAULTS} == (
        YOLO_NATIVE_TRAINING_DEFAULTS
    )
    assert {key: kwargs[key] for key in YOLO_NATIVE_AUGMENTATION_DEFAULTS} == (
        YOLO_NATIVE_AUGMENTATION_DEFAULTS
    )
    assert execution.realized_config["preprocessing"] == {"resize_strategy": "yolo_letterbox"}


def test_modal_image_and_worker_dependency_use_the_translator_version() -> None:
    dependency = f'"ultralytics=={PINNED_ULTRALYTICS_VERSION}"'
    assert dependency in (WORKER_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert dependency in (WORKER_ROOT / "worker" / "training" / "modal_runtime.py").read_text(
        encoding="utf-8"
    )


def test_mocked_yolo_train_receives_the_complete_translation() -> None:
    calls = []

    class FakeYOLO:
        def train(self, **kwargs):
            calls.append(kwargs)

    execution = resolve_yolo_execution(_job_config(_request(learning_rate=0.002)))
    kwargs = ultralytics_train_kwargs(
        execution,
        data="data.yaml",
        project="runs",
        name="train",
        workers=1,
    )
    FakeYOLO().train(**kwargs)

    assert calls == [kwargs]
    assert calls[0]["lr0"] == 0.002
    assert set(YOLO_NATIVE_AUGMENTATION_DEFAULTS).issubset(calls[0])
    assert set(YOLO_NATIVE_TRAINING_DEFAULTS).issubset(calls[0])


def test_unsupported_classifier_semantics_never_enter_yolo_kwargs() -> None:
    requested = _request(
        optimizer="sgd",
        scheduler="cosine",
        weight_decay=0.2,
        class_balancing="focal_loss",
        sampling_strategy="weighted_random_sampler",
        augmentation_policy="mixup",
        pretrained=False,
    )
    execution = resolve_yolo_execution(_job_config(requested))
    kwargs = ultralytics_train_kwargs(
        execution,
        data="data.yaml",
        project="runs",
        name="train",
        workers=1,
    )

    assert execution.accepted_config["pretrained"] is True
    assert "scheduler" not in execution.accepted_config
    assert "class_balancing" not in execution.accepted_config
    assert "sampling_strategy" not in execution.accepted_config
    assert "augmentation_policy" not in execution.accepted_config
    assert kwargs["optimizer"] == "auto"
    assert kwargs["weight_decay"] == 0.0005
    assert kwargs["mixup"] == 0.0


def test_batch_recovery_is_the_only_approved_adjustment() -> None:
    execution = resolve_yolo_execution(
        _job_config(mode="enforce"),
        effective_batch_size=4,
    )

    assert execution.realized_config["batch_size"] == 4
    assert execution.adjustment_policy == "batch_size_recovery"
    assert yolo_realization_matches_policy(execution)


def test_framework_batch_recovery_is_recorded_as_an_approved_adjustment() -> None:
    execution = resolve_yolo_execution(_job_config(mode="enforce"))
    captured = capture_yolo_trainer_arguments(
        SimpleNamespace(args=SimpleNamespace(**_trainer_arguments(execution, batch=4)))
    )
    realized = realize_yolo_trainer_arguments(execution, captured)

    assert realized.realized_config["batch_size"] == 4
    assert realized.adjustment_policy == "batch_size_recovery"
    assert not realized.framework_mismatches
    assert yolo_realization_matches_policy(realized)


def test_realized_trainer_arguments_match_the_observation_semantics() -> None:
    execution = resolve_yolo_execution(_job_config())
    trainer = SimpleNamespace(args=SimpleNamespace(**_trainer_arguments(execution)))
    captured = capture_yolo_trainer_arguments(trainer)
    realized = realize_yolo_trainer_arguments(execution, captured)

    assert realized.realized_config == execution.accepted_config
    assert not realized.framework_mismatches
    assert realized.trainer_arguments == captured
    assert yolo_realization_matches_policy(realized)


def test_changed_native_framework_default_is_a_mismatch() -> None:
    execution = resolve_yolo_execution(_job_config(mode="enforce"))
    captured = capture_yolo_trainer_arguments(
        SimpleNamespace(args=SimpleNamespace(**_trainer_arguments(execution, mosaic=0.5)))
    )
    realized = realize_yolo_trainer_arguments(execution, captured)

    assert realized.framework_mismatches == ("mosaic",)
    assert "preprocessing" not in realized.realized_config
    assert not yolo_realization_matches_policy(realized)


def test_semantic_hash_changes_with_execution_but_not_infrastructure() -> None:
    first = resolve_yolo_execution(_job_config(_request(learning_rate=0.001)))
    second = resolve_yolo_execution(_job_config(_request(learning_rate=0.002)))

    assert yolo_framework_semantic_hash(first) != yolo_framework_semantic_hash(second)
    first_kwargs = ultralytics_train_kwargs(
        first,
        data="first.yaml",
        project="first-runs",
        name="one",
        workers=1,
    )
    other_infrastructure_kwargs = ultralytics_train_kwargs(
        first,
        data="second.yaml",
        project="second-runs",
        name="two",
        workers=8,
    )
    assert (
        first_kwargs["lr0"]
        != ultralytics_train_kwargs(
            second,
            data="first.yaml",
            project="first-runs",
            name="one",
            workers=1,
        )["lr0"]
    )
    assert yolo_framework_semantic_hash(first) == yolo_framework_semantic_hash(first)
    for infrastructure_field in ("data", "project", "name", "workers"):
        assert (
            first_kwargs[infrastructure_field] != other_infrastructure_kwargs[infrastructure_field]
        )


def test_tampered_fixed_semantics_and_stale_versions_are_rejected() -> None:
    config = _job_config()
    config["execution_spec_v1"]["accepted_config"]["pretrained"] = False
    with pytest.raises(YoloExecutionError, match="packaged capability resolver"):
        resolve_yolo_execution(config)

    config = _job_config()
    config["execution_spec_v1"]["capability_version"] = "stale"
    with pytest.raises(YoloExecutionError, match="capability version"):
        resolve_yolo_execution(config)


def test_legacy_jobs_are_shadow_only() -> None:
    legacy = _request()
    execution = resolve_yolo_execution(legacy)
    assert not execution.versioned
    assert execution.fidelity_mode == "shadow"

    legacy["execution_validation_v1"] = {"mode": "enforce"}
    with pytest.raises(YoloExecutionError, match="execution_spec_v1"):
        resolve_yolo_execution(legacy)
