from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from worker.training.augmentation import normalize_augmentation_config
from worker.training.execution_capabilities import (
    capability_document_v1,
    resolve_accepted_config,
)
from worker.training.preprocessing_registry import normalize_preprocessing_config


TASK = "image_classification"
RUNNER = "modal_torchvision"
EXECUTION_SPEC_SCHEMA = "execution_spec_v1"
OBSERVATION_SCHEMA = "execution_realization_v1"


class ClassificationExecutionError(ValueError):
    pass


def load_torchvision_model(factory, weights):
    """Load exactly the requested initialization; never retry without weights."""
    return factory(weights=weights)


@dataclass(frozen=True)
class ClassificationExecution:
    accepted_config: dict[str, Any]
    realized_config: dict[str, Any]
    adjustment_policy: str
    fidelity_mode: str
    versioned: bool = True

    def value(self, field: str) -> Any:
        if field not in self.realized_config:
            raise ClassificationExecutionError(
                f"Accepted classification execution spec is missing {field!r}."
            )
        return self.realized_config[field]

    @property
    def preprocessing(self) -> dict[str, Any]:
        value = self.value("preprocessing")
        if not isinstance(value, dict):
            raise ClassificationExecutionError("preprocessing must be an object.")
        return deepcopy(value)

    @property
    def augmentation(self) -> dict[str, Any]:
        return normalize_augmentation_config(
            self.realized_config.get("augmentation"),
            self.realized_config.get("augmentation_policy", "none"),
            self.realized_config.get("augmentation_policy_config"),
        )


def resolve_classification_execution(
    job_config: dict,
    *,
    effective_batch_size: int | None = None,
) -> ClassificationExecution:
    spec = job_config.get(EXECUTION_SPEC_SCHEMA)
    if not isinstance(spec, dict):
        fidelity_mode = runner_fidelity_mode(job_config)
        if fidelity_mode == "enforce":
            raise ClassificationExecutionError(
                "Runner fidelity enforcement requires config.execution_spec_v1."
            )
        accepted = resolve_accepted_config(TASK, RUNNER, job_config)
        return _build_classification_execution(
            accepted,
            fidelity_mode=fidelity_mode,
            effective_batch_size=effective_batch_size,
            versioned=False,
        )
    if spec.get("schema_version") != EXECUTION_SPEC_SCHEMA:
        raise ClassificationExecutionError("Unsupported classification execution spec schema.")
    if spec.get("task") != TASK or spec.get("runner") != RUNNER:
        raise ClassificationExecutionError(
            f"Classification worker requires task={TASK!r} and runner={RUNNER!r}."
        )
    document = capability_document_v1()
    if spec.get("capability_version") != document["capability_version"]:
        raise ClassificationExecutionError(
            "Classification execution capability version does not match this worker image."
        )
    requested = spec.get("requested_config")
    accepted = spec.get("accepted_config")
    if not isinstance(requested, dict) or not isinstance(accepted, dict):
        raise ClassificationExecutionError(
            "Classification execution spec requires requested_config and accepted_config objects."
        )
    worker_resolved = resolve_accepted_config(TASK, RUNNER, accepted)
    if worker_resolved != accepted:
        raise ClassificationExecutionError(
            "Accepted classification config does not match the packaged capability resolver."
        )
    return _build_classification_execution(
        accepted,
        fidelity_mode=runner_fidelity_mode(job_config),
        effective_batch_size=effective_batch_size,
        versioned=True,
    )


def _build_classification_execution(
    accepted: dict[str, Any],
    *,
    fidelity_mode: str,
    effective_batch_size: int | None,
    versioned: bool,
) -> ClassificationExecution:
    _validate_classification_types(accepted)
    realized = deepcopy(accepted)
    adjustment_policy = ""
    requested_batch_size = _strict_int(accepted, "batch_size", minimum=1)
    if effective_batch_size is not None and effective_batch_size != requested_batch_size:
        if effective_batch_size <= 0:
            raise ClassificationExecutionError("Effective batch size must be positive.")
        realized["batch_size"] = effective_batch_size
        if effective_batch_size < requested_batch_size:
            adjustment_policy = "batch_size_recovery"
    realized = _effective_transfer_semantics(realized)
    realized = _effective_preprocessing_semantics(realized)
    return ClassificationExecution(
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


def realization_matches_policy(execution: ClassificationExecution) -> bool:
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


def _effective_transfer_semantics(config: dict[str, Any]) -> dict[str, Any]:
    realized = deepcopy(config)
    freeze_backbone = _strict_bool(realized, "freeze_backbone")
    strategy = _strict_string(realized, "fine_tune_strategy")
    if not freeze_backbone or strategy == "full":
        realized["freeze_backbone"] = False
        realized["fine_tune_strategy"] = "full"
    return realized


def _effective_preprocessing_semantics(config: dict[str, Any]) -> dict[str, Any]:
    realized = deepcopy(config)
    preprocessing = realized.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ClassificationExecutionError("preprocessing must be an object.")
    if preprocessing.get("use_dataset_normalization") is True:
        preprocessing["normalization"] = "dataset"
    return realized


def _validate_classification_types(config: dict[str, Any]) -> None:
    for field in ("epochs", "batch_size", "image_size"):
        _strict_int(config, field, minimum=1)
    if config["epochs"] > 100:
        raise ClassificationExecutionError("epochs must be <= 100.")
    if config["batch_size"] > 512:
        raise ClassificationExecutionError("batch_size must be <= 512.")
    if _strict_int(config, "early_stopping_patience", minimum=0) > 50:
        raise ClassificationExecutionError("early_stopping_patience must be <= 50.")
    _strict_number(
        config,
        "learning_rate",
        minimum=0,
        maximum=1,
        exclusive_minimum=True,
    )
    _strict_number(config, "weight_decay", minimum=0)
    _strict_number(config, "dropout", minimum=0, maximum=0.7)
    _strict_number(config, "gradient_clip_norm", minimum=0, maximum=10)
    _strict_number(config, "label_smoothing", minimum=0, maximum=0.3)
    for field in ("pretrained", "freeze_backbone"):
        _strict_bool(config, field)
    for field in ("model", "optimizer", "scheduler", "fine_tune_strategy"):
        _strict_string(config, field)
    if not 96 <= _strict_int(config, "image_size", minimum=1) <= 384:
        raise ClassificationExecutionError("image_size must be between 96 and 384.")
    normalize_preprocessing_config(config.get("preprocessing"))
    _require_choice(
        config,
        "model",
        {
            "mobilenet_v3_small",
            "mobilenet_v3_large",
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "resnet18",
            "resnet34",
            "regnet_y_400mf",
            "convnext_tiny",
            "swin_t",
            "vit_b_16",
        },
    )
    _require_choice(config, "optimizer", {"adamw", "adam", "sgd"})
    _require_choice(config, "scheduler", {"none", "cosine", "step"})
    _require_choice(config, "fine_tune_strategy", {"head_only", "last_block", "full"})
    _require_choice(
        config,
        "class_balancing",
        {
            "none",
            "weighted_loss",
            "effective_number_loss",
            "class_balanced_sampler",
            "weighted_random_sampler",
            "focal_loss",
        },
    )
    _require_choice(
        config,
        "sampling_strategy",
        {
            "none",
            "class_balanced_sampler",
            "weighted_random_sampler",
        },
    )
    if "optimizer_momentum" in config:
        _strict_number(config, "optimizer_momentum", minimum=0, maximum=0.99)
    if "scheduler_step_size" in config:
        step_size = _strict_int(config, "scheduler_step_size", minimum=1)
        if step_size > config["epochs"]:
            raise ClassificationExecutionError("scheduler_step_size may not exceed epochs.")
    if "scheduler_gamma" in config:
        _strict_number(config, "scheduler_gamma", minimum=0.05, maximum=0.95)
    preprocessing = config.get("preprocessing") or {}
    if "use_dataset_normalization" in preprocessing and not isinstance(
        preprocessing["use_dataset_normalization"], bool
    ):
        raise ClassificationExecutionError(
            "preprocessing.use_dataset_normalization must be boolean."
        )
    augmentation = config.get("augmentation") or {}
    if not isinstance(augmentation, dict):
        raise ClassificationExecutionError("augmentation must be an object.")
    for field, value in augmentation.items():
        if not isinstance(value, bool):
            raise ClassificationExecutionError(f"augmentation.{field} must be boolean.")
    policy_config = config.get("augmentation_policy_config") or {}
    if not isinstance(policy_config, dict):
        raise ClassificationExecutionError("augmentation_policy_config must be an object.")
    _optional_nested_number(policy_config, "alpha", minimum=0, maximum=1)
    _optional_nested_number(policy_config, "probability", minimum=0, maximum=1)
    _optional_nested_int(policy_config, "magnitude", minimum=0, maximum=15)
    _optional_nested_int(policy_config, "num_ops", minimum=1, maximum=3)
    _optional_nested_int(policy_config, "num_magnitude_bins", minimum=2, maximum=31)
    normalize_augmentation_config(
        augmentation,
        config.get("augmentation_policy", "none"),
        config.get("augmentation_policy_config"),
    )
    balancing_config = config.get("class_balancing_config") or {}
    if not isinstance(balancing_config, dict):
        raise ClassificationExecutionError("class_balancing_config must be an object.")
    _optional_nested_number(
        balancing_config,
        "effective_number_beta",
        minimum=0.9,
        maximum=0.99999,
    )
    _optional_nested_number(
        balancing_config,
        "focal_loss_gamma",
        minimum=0.5,
        maximum=5,
    )


def _strict_int(config: dict[str, Any], field: str, *, minimum: int) -> int:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ClassificationExecutionError(f"{field} must be an integer >= {minimum}.")
    return value


def _strict_number(
    config: dict[str, Any],
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    value = config.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClassificationExecutionError(f"{field} must be numeric.")
    parsed = float(value)
    if minimum is not None and (parsed < minimum or (exclusive_minimum and parsed == minimum)):
        operator = ">" if exclusive_minimum else ">="
        raise ClassificationExecutionError(f"{field} must be {operator} {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ClassificationExecutionError(f"{field} must be <= {maximum}.")
    return parsed


def _strict_bool(config: dict[str, Any], field: str) -> bool:
    value = config.get(field)
    if not isinstance(value, bool):
        raise ClassificationExecutionError(f"{field} must be a boolean.")
    return value


def _strict_string(config: dict[str, Any], field: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value:
        raise ClassificationExecutionError(f"{field} must be a non-empty string.")
    return value


def _require_choice(config: dict[str, Any], field: str, choices: set[str]) -> None:
    value = _strict_string(config, field)
    if value not in choices:
        supported = ", ".join(sorted(choices))
        raise ClassificationExecutionError(
            f"Unsupported {field} {value!r}; expected one of: {supported}."
        )


def _optional_nested_number(
    config: dict[str, Any],
    field: str,
    *,
    minimum: float,
    maximum: float,
) -> None:
    if field not in config:
        return
    value = config[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClassificationExecutionError(f"{field} must be numeric.")
    if not minimum <= float(value) <= maximum:
        raise ClassificationExecutionError(f"{field} must be between {minimum} and {maximum}.")


def _optional_nested_int(
    config: dict[str, Any],
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> None:
    if field not in config:
        return
    value = config[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClassificationExecutionError(f"{field} must be an integer.")
    if not minimum <= value <= maximum:
        raise ClassificationExecutionError(f"{field} must be between {minimum} and {maximum}.")
