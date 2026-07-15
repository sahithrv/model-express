from __future__ import annotations

import base64
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from worker.training.execution_capabilities import (
    capability_document_v1,
    resolve_accepted_config,
)


TASK = "object_detection"
RUNNER = "modal_ultralytics"
EXECUTION_SPEC_SCHEMA = "execution_spec_v1"
OBSERVATION_SCHEMA = "execution_realization_v1"
PINNED_ULTRALYTICS_VERSION = "8.4.66"

# These values are the pinned Ultralytics detection defaults. Passing them
# explicitly prevents an image rebuild or upstream default change from silently
# changing the experiment while the accepted execution spec stays the same.
YOLO_NATIVE_TRAINING_DEFAULTS: dict[str, Any] = {
    "amp": True,
    "box": 7.5,
    "cls": 0.5,
    "cls_pw": 0.0,
    "close_mosaic": 10,
    "compile": False,
    "cos_lr": False,
    "deterministic": True,
    "dfl": 1.5,
    "fraction": 1.0,
    "freeze": None,
    "lrf": 0.01,
    "momentum": 0.937,
    "multi_scale": 0.0,
    "nbs": 64,
    "optimizer": "auto",
    "patience": 100,
    "rect": False,
    "resume": False,
    "seed": 0,
    "single_cls": False,
    "val": True,
    # Ultralytics resolves this to 0.0 whenever optimizer=auto before the
    # initialization callback, regardless of the YAML-level 0.1 default.
    "warmup_bias_lr": 0.0,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "weight_decay": 0.0005,
}

YOLO_NATIVE_AUGMENTATION_DEFAULTS: dict[str, Any] = {
    "bgr": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
    "cutmix": 0.0,
    "degrees": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "mixup": 0.0,
    "mosaic": 1.0,
    "perspective": 0.0,
    "scale": 0.5,
    "shear": 0.0,
    "translate": 0.1,
}

YOLO_TRAINER_ARGUMENT_ALLOWLIST = frozenset(
    {
        "model",
        "epochs",
        "batch",
        "imgsz",
        "lr0",
        "pretrained",
        "plots",
        "val",
        *YOLO_NATIVE_TRAINING_DEFAULTS,
        *YOLO_NATIVE_AUGMENTATION_DEFAULTS,
    }
)


class YoloExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class YoloExecution:
    accepted_config: dict[str, Any]
    realized_config: dict[str, Any]
    adjustment_policy: str
    fidelity_mode: str
    versioned: bool = True
    trainer_arguments: dict[str, Any] | None = None
    framework_mismatches: tuple[str, ...] = ()

    def value(self, field: str) -> Any:
        if field not in self.realized_config:
            raise YoloExecutionError(f"Accepted YOLO execution spec is missing {field!r}.")
        return self.realized_config[field]


def resolve_yolo_execution(
    job_config: dict,
    *,
    effective_batch_size: int | None = None,
) -> YoloExecution:
    spec = job_config.get(EXECUTION_SPEC_SCHEMA)
    if not isinstance(spec, dict):
        fidelity_mode = runner_fidelity_mode(job_config)
        if fidelity_mode == "enforce":
            raise YoloExecutionError(
                "Runner fidelity enforcement requires config.execution_spec_v1."
            )
        legacy_config = deepcopy(job_config)
        if not str(legacy_config.get("model") or "").strip():
            legacy_config["model"] = str(legacy_config.get("pretrained_weights") or "yolo11n.pt")
        accepted = resolve_accepted_config(TASK, RUNNER, legacy_config)
        return _build_yolo_execution(
            accepted,
            fidelity_mode=fidelity_mode,
            effective_batch_size=effective_batch_size,
            versioned=False,
        )
    if spec.get("schema_version") != EXECUTION_SPEC_SCHEMA:
        raise YoloExecutionError("Unsupported YOLO execution spec schema.")
    if spec.get("task") != TASK or spec.get("runner") != RUNNER:
        raise YoloExecutionError(f"YOLO worker requires task={TASK!r} and runner={RUNNER!r}.")
    document = capability_document_v1()
    if spec.get("capability_version") != document["capability_version"]:
        raise YoloExecutionError(
            "YOLO execution capability version does not match this worker image."
        )
    requested = spec.get("requested_config")
    accepted = spec.get("accepted_config")
    if not isinstance(requested, dict) or not isinstance(accepted, dict):
        raise YoloExecutionError(
            "YOLO execution spec requires requested_config and accepted_config objects."
        )
    worker_resolved = resolve_accepted_config(TASK, RUNNER, accepted)
    if worker_resolved != accepted:
        raise YoloExecutionError(
            "Accepted YOLO config does not match the packaged capability resolver."
        )
    return _build_yolo_execution(
        accepted,
        fidelity_mode=runner_fidelity_mode(job_config),
        effective_batch_size=effective_batch_size,
        versioned=True,
    )


def _build_yolo_execution(
    accepted: dict[str, Any],
    *,
    fidelity_mode: str,
    effective_batch_size: int | None,
    versioned: bool,
) -> YoloExecution:
    _validate_yolo_config(accepted)
    realized = deepcopy(accepted)
    adjustment_policy = ""
    accepted_batch_size = _strict_int(accepted, "batch_size", minimum=1, maximum=512)
    if effective_batch_size is not None and effective_batch_size != accepted_batch_size:
        if isinstance(effective_batch_size, bool) or not isinstance(effective_batch_size, int):
            raise YoloExecutionError("Effective YOLO batch size must be an integer.")
        if effective_batch_size <= 0:
            raise YoloExecutionError("Effective YOLO batch size must be positive.")
        realized["batch_size"] = effective_batch_size
        if effective_batch_size < accepted_batch_size:
            adjustment_policy = "batch_size_recovery"
    return YoloExecution(
        accepted_config=deepcopy(accepted),
        realized_config=realized,
        adjustment_policy=adjustment_policy,
        fidelity_mode=fidelity_mode,
        versioned=versioned,
    )


def runner_fidelity_mode(job_config: dict) -> str:
    override = os.getenv("MODEL_EXPRESS_RUNNER_FIDELITY_MODE", "").strip().lower()
    if override in {"shadow", "enforce"}:
        return override
    validation = job_config.get("execution_validation_v1")
    mode = str(validation.get("mode") or "").strip().lower() if isinstance(validation, dict) else ""
    return "enforce" if mode == "enforce" else "shadow"


def yolo_realization_matches_policy(execution: YoloExecution) -> bool:
    if execution.framework_mismatches:
        return False
    if execution.realized_config == execution.accepted_config:
        return True
    if execution.adjustment_policy != "batch_size_recovery":
        return False
    accepted = deepcopy(execution.accepted_config)
    realized = deepcopy(execution.realized_config)
    accepted_batch = accepted.pop("batch_size", None)
    realized_batch = realized.pop("batch_size", None)
    return (
        isinstance(accepted_batch, int)
        and not isinstance(accepted_batch, bool)
        and isinstance(realized_batch, int)
        and not isinstance(realized_batch, bool)
        and 0 < realized_batch < accepted_batch
        and accepted == realized
    )


def yolo_framework_semantic_arguments(execution: YoloExecution) -> dict[str, Any]:
    accepted_preprocessing = execution.accepted_config.get("preprocessing")
    if not isinstance(accepted_preprocessing, dict):
        raise YoloExecutionError("Accepted YOLO preprocessing must be an object.")
    return {
        "constructor": {"model": str(execution.value("model"))},
        "train": {
            "epochs": int(execution.value("epochs")),
            "batch": int(execution.value("batch_size")),
            "imgsz": int(execution.value("image_size")),
            "lr0": float(execution.value("learning_rate")),
            "pretrained": bool(execution.value("pretrained")),
            **deepcopy(YOLO_NATIVE_TRAINING_DEFAULTS),
            **deepcopy(YOLO_NATIVE_AUGMENTATION_DEFAULTS),
        },
        "preprocessing": {
            "resize_strategy": accepted_preprocessing["resize_strategy"],
            "training_implementation": "ultralytics.data.augment.v8_transforms",
            "evaluation_implementation": "ultralytics.data.augment.LetterBox",
            "center": True,
            "evaluation_scaleup": False,
            "padding_value": 114,
            "preserve_aspect_ratio": True,
            "rectangular_batches": False,
            "scale_fill": False,
            "stride": "model_stride",
        },
    }


def yolo_framework_semantic_hash(execution: YoloExecution) -> str:
    encoded = json.dumps(
        yolo_framework_semantic_arguments(execution),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return "sha256:" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def ultralytics_train_kwargs(
    execution: YoloExecution,
    *,
    data: str,
    project: str,
    name: str,
    workers: int,
) -> dict[str, Any]:
    semantic = yolo_framework_semantic_arguments(execution)["train"]
    return {
        "data": data,
        **semantic,
        "project": project,
        "name": name,
        "exist_ok": True,
        "plots": False,
        "workers": workers,
    }


def capture_yolo_trainer_arguments(trainer: Any) -> dict[str, Any]:
    args = getattr(trainer, "args", trainer)
    captured: dict[str, Any] = {}
    for name in sorted(YOLO_TRAINER_ARGUMENT_ALLOWLIST):
        if isinstance(args, dict):
            if name not in args:
                continue
            value = args[name]
        else:
            if not hasattr(args, name):
                continue
            value = getattr(args, name)
        captured[name] = _json_scalar(value)
    return captured


def realize_yolo_trainer_arguments(
    execution: YoloExecution,
    trainer_arguments: dict[str, Any],
) -> YoloExecution:
    required = {"model", "epochs", "batch", "imgsz", "lr0", "pretrained"}
    missing = sorted(required - trainer_arguments.keys())
    if missing:
        raise YoloExecutionError(
            "Ultralytics trainer did not expose required realized arguments: " + ", ".join(missing)
        )
    realized = deepcopy(execution.realized_config)
    realized["model"] = _trainer_string(trainer_arguments["model"], "model")
    realized["epochs"] = _trainer_int(trainer_arguments["epochs"], "epochs")
    realized["batch_size"] = _trainer_int(trainer_arguments["batch"], "batch")
    realized["image_size"] = _trainer_image_size(trainer_arguments["imgsz"])
    realized["learning_rate"] = _trainer_number(trainer_arguments["lr0"], "lr0")
    realized["pretrained"] = _trainer_bool(trainer_arguments["pretrained"], "pretrained")
    expected_train = yolo_framework_semantic_arguments(execution)["train"]
    accepted_field_arguments = {"epochs", "batch", "imgsz", "lr0", "pretrained"}
    framework_mismatches = tuple(
        sorted(
            name
            for name, expected in expected_train.items()
            if name not in accepted_field_arguments
            and (
                name not in trainer_arguments
                or not _framework_values_equal(trainer_arguments[name], expected)
            )
        )
    )
    if framework_mismatches:
        # The accepted schema has no planner-facing fields for Ultralytics'
        # native detection defaults. Omitting the fixed preprocessing leaf makes
        # the server return MISMATCH instead of falsely trusting the accepted
        # object; the exact divergent arguments remain in framework evidence.
        realized.pop("preprocessing", None)
    adjustment_policy = execution.adjustment_policy
    accepted_batch_size = execution.accepted_config.get("batch_size")
    if (
        isinstance(accepted_batch_size, int)
        and not isinstance(accepted_batch_size, bool)
        and 0 < realized["batch_size"] < accepted_batch_size
    ):
        adjustment_policy = "batch_size_recovery"
    return replace(
        execution,
        realized_config=realized,
        adjustment_policy=adjustment_policy,
        trainer_arguments=deepcopy(trainer_arguments),
        framework_mismatches=framework_mismatches,
    )


def _validate_yolo_config(config: dict[str, Any]) -> None:
    _strict_string(config, "model")
    _strict_int(config, "epochs", minimum=1, maximum=100)
    _strict_int(config, "batch_size", minimum=1, maximum=512)
    _strict_int(config, "image_size", minimum=160, maximum=1280)
    _strict_number(config, "learning_rate", minimum=0, maximum=1, exclusive_minimum=True)
    pretrained = config.get("pretrained")
    if pretrained is not True:
        raise YoloExecutionError("YOLO pretrained must be the fixed true runner semantic.")
    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, dict) or preprocessing != {
        "resize_strategy": "yolo_letterbox"
    }:
        raise YoloExecutionError(
            "YOLO preprocessing must be the fixed yolo_letterbox runner semantic."
        )


def _strict_int(
    config: dict[str, Any],
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise YoloExecutionError(f"{field} must be an integer.")
    if value < minimum or value > maximum:
        raise YoloExecutionError(f"{field} must be between {minimum} and {maximum}.")
    return value


def _strict_number(
    config: dict[str, Any],
    field: str,
    *,
    minimum: float,
    maximum: float,
    exclusive_minimum: bool = False,
) -> float:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise YoloExecutionError(f"{field} must be numeric.")
    numeric = float(value)
    if (exclusive_minimum and numeric <= minimum) or (not exclusive_minimum and numeric < minimum):
        raise YoloExecutionError(f"{field} is below its supported minimum.")
    if numeric > maximum:
        raise YoloExecutionError(f"{field} exceeds its supported maximum.")
    return numeric


def _strict_string(config: dict[str, Any], field: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value.strip():
        raise YoloExecutionError(f"{field} must be a non-empty string.")
    return value


def _trainer_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise YoloExecutionError(f"Ultralytics trainer argument {field} must be an integer.")
    return int(value)


def _trainer_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise YoloExecutionError(f"Ultralytics trainer argument {field} must be numeric.")
    return float(value)


def _trainer_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise YoloExecutionError(f"Ultralytics trainer argument {field} must be boolean.")
    return value


def _trainer_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise YoloExecutionError(
            f"Ultralytics trainer argument {field} must be a non-empty string."
        )
    return value


def _trainer_image_size(value: Any) -> int:
    if isinstance(value, (list, tuple)) and len(value) == 2 and value[0] == value[1]:
        value = value[0]
    return _trainer_int(value, "imgsz")


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    return str(value)


def _framework_values_equal(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (list, tuple)) and len(actual) == 2 and actual[0] == actual[1]:
        actual = actual[0]
    if isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            return float(actual) == float(expected)
    return actual == expected
