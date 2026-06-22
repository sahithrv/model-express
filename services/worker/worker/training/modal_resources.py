from __future__ import annotations

import os
import re
from copy import deepcopy


MODAL_RESOURCE_SCHEMA_VERSION = "modal_resources.v1"
MODAL_GPU_LADDER = ("T4", "L4", "A10", "L40S", "A100-40GB", "A100-80GB")

_SYSTEM_OOM_PATTERNS = (
    "exit code 137",
    "exit status 137",
    "signal: killed",
    "sigkill",
    "oom killed",
    "out of memory killed",
    "memory limit",
    "container memory",
    "cannot allocate memory",
)
_CUDA_OOM_PATTERNS = (
    "cuda out of memory",
    "torch.cuda.outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "hip out of memory",
)
_GENERIC_OOM_PATTERNS = (
    "outofmemoryerror",
    "out of memory",
)
_OOM_TOKEN_PATTERN = re.compile(r"(^|[^a-z0-9])oom([^a-z0-9]|$)")


def resolve_modal_resources(config: dict, *, detection_job: bool = False) -> dict:
    modal_config = config.get("modal_resources") if isinstance(config.get("modal_resources"), dict) else {}
    requested_gpu = normalize_gpu_type(
        first_non_empty(
            modal_config.get("requested_gpu_type"),
            modal_config.get("gpu_type"),
            modal_config.get("gpu"),
            config.get("gpu_type"),
            os.getenv("MODEL_EXPRESS_MODAL_DEFAULT_GPU_TYPE"),
            os.getenv("MODAL_GPU_TYPE"),
            "T4",
        )
    )
    effective_gpu = normalize_gpu_type(
        first_non_empty(
            modal_config.get("effective_gpu_type"),
            modal_config.get("gpu"),
            requested_gpu,
        )
    )
    requested_batch = positive_int(
        first_non_empty(
            modal_config.get("requested_batch_size"),
            config.get("requested_batch_size"),
            config.get("batch_size"),
        ),
        8 if detection_job else 16,
    )
    effective_batch = positive_int(
        first_non_empty(
            modal_config.get("effective_batch_size"),
            config.get("effective_batch_size"),
            requested_batch,
        ),
        requested_batch,
    )
    memory_mb = positive_int(
        first_non_empty(
            modal_config.get("memory_mb"),
            modal_config.get("system_memory_mb"),
            config.get("modal_memory_mb"),
            config.get("memory_mb"),
        ),
        default_modal_memory_mb(effective_gpu, detection_job=detection_job),
    )
    batch_policy = str(modal_config.get("batch_size_policy") or "").strip()
    if not batch_policy:
        batch_policy = "preserved" if effective_batch == requested_batch else "auto_reduced"
    resource_attempt = positive_int(
        first_non_empty(
            modal_config.get("resource_attempt"),
            config.get("resource_attempt"),
            config.get("attempt"),
        ),
        1,
    )
    options = {
        "gpu": effective_gpu,
        "memory": memory_mb,
    }
    return {
        "schema_version": MODAL_RESOURCE_SCHEMA_VERSION,
        "requested_gpu_type": requested_gpu,
        "effective_gpu_type": effective_gpu,
        "memory_mb": memory_mb,
        "requested_batch_size": requested_batch,
        "effective_batch_size": effective_batch,
        "batch_size_policy": batch_policy,
        "resource_attempt": resource_attempt,
        "modal_function_options": options,
        "resource_signature": resource_signature(effective_gpu, effective_batch, memory_mb),
    }


def job_with_modal_resources(job: dict, modal_resources: dict) -> dict:
    enriched = deepcopy(job)
    config = enriched.get("config") if isinstance(enriched.get("config"), dict) else {}
    enriched["config"] = {
        **config,
        "modal_resources": {
            **(config.get("modal_resources") if isinstance(config.get("modal_resources"), dict) else {}),
            **modal_resources,
        },
        "requested_gpu_type": modal_resources.get("requested_gpu_type", ""),
        "effective_gpu_type": modal_resources.get("effective_gpu_type", ""),
        "requested_batch_size": modal_resources.get("requested_batch_size", 0),
        "effective_batch_size": modal_resources.get("effective_batch_size", 0),
        "batch_size_policy": modal_resources.get("batch_size_policy", ""),
    }
    return enriched


def modal_resources_from_payload(payload: dict, config: dict, *, detection_job: bool = False) -> dict:
    resources = payload.get("modal_resources") if isinstance(payload.get("modal_resources"), dict) else {}
    if resources:
        return {
            **resolve_modal_resources(
                {
                    **config,
                    "modal_resources": resources,
                },
                detection_job=detection_job,
            ),
            **resources,
        }
    return resolve_modal_resources(config, detection_job=detection_job)


def resource_telemetry(modal_resources: dict, *, effective_batch_size: int | None = None) -> dict:
    telemetry = {
        "schema_version": MODAL_RESOURCE_SCHEMA_VERSION,
        "requested_gpu_type": str(modal_resources.get("requested_gpu_type") or ""),
        "effective_gpu_type": str(modal_resources.get("effective_gpu_type") or ""),
        "memory_mb": positive_int(modal_resources.get("memory_mb"), 0),
        "requested_batch_size": positive_int(modal_resources.get("requested_batch_size"), 0),
        "effective_batch_size": positive_int(
            effective_batch_size if effective_batch_size is not None else modal_resources.get("effective_batch_size"),
            0,
        ),
        "batch_size_policy": str(modal_resources.get("batch_size_policy") or ""),
        "resource_attempt": positive_int(modal_resources.get("resource_attempt"), 0),
        "resource_signature": str(modal_resources.get("resource_signature") or ""),
        "modal_function_options": dict(
            modal_resources.get("modal_function_options")
            if isinstance(modal_resources.get("modal_function_options"), dict)
            else {}
        ),
    }
    if not telemetry["resource_signature"]:
        telemetry["resource_signature"] = resource_signature(
            telemetry["effective_gpu_type"],
            telemetry["effective_batch_size"],
            telemetry["memory_mb"],
        )
    return telemetry


def classify_oom_failure(exc_or_message: object) -> dict:
    message = str(exc_or_message or "").strip()
    normalized = re.sub(r"\s+", " ", message.lower())
    if any(pattern in normalized for pattern in _CUDA_OOM_PATTERNS):
        return {
            "is_oom": True,
            "failure_class": "oom",
            "oom_kind": "gpu_cuda",
            "retryable": True,
        }
    if any(pattern in normalized for pattern in _SYSTEM_OOM_PATTERNS):
        return {
            "is_oom": True,
            "failure_class": "oom",
            "oom_kind": "system_memory",
            "retryable": True,
        }
    if any(pattern in normalized for pattern in _GENERIC_OOM_PATTERNS) or _OOM_TOKEN_PATTERN.search(normalized):
        return {
            "is_oom": True,
            "failure_class": "oom",
            "oom_kind": "unknown_memory",
            "retryable": True,
        }
    return {
        "is_oom": False,
        "failure_class": "runtime",
        "oom_kind": "",
        "retryable": False,
    }


def failure_callback_payload(
    job: dict,
    error: str,
    modal_resources: dict | None = None,
    *,
    retryable: bool = True,
) -> dict:
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    resources = modal_resources if isinstance(modal_resources, dict) else modal_resources_from_payload({}, config)
    classification = classify_oom_failure(error)
    telemetry = resource_telemetry(resources)
    payload = {
        "error": error,
        "retryable": bool(retryable),
        "training_attempt_id": training_attempt_id(job),
        "failure_class": classification["failure_class"],
        "failure_type": classification["failure_class"],
        "oom": classification["is_oom"],
        "oom_kind": classification["oom_kind"],
        "requested_gpu_type": telemetry["requested_gpu_type"],
        "effective_gpu_type": telemetry["effective_gpu_type"],
        "memory_mb": telemetry["memory_mb"],
        "requested_batch_size": telemetry["requested_batch_size"],
        "effective_batch_size": telemetry["effective_batch_size"],
        "batch_size_policy": telemetry["batch_size_policy"],
        "modal_resource_signature": telemetry["resource_signature"],
        "modal_resources": telemetry,
    }
    for source in (resources, config.get("modal_resources") if isinstance(config.get("modal_resources"), dict) else {}):
        for key in ("modal_function_call_id", "modal_input_id"):
            if isinstance(source, dict) and source.get(key):
                payload[key] = str(source[key])
    return payload


def training_attempt_id(job: dict) -> str:
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    for value in (
        config.get("active_attempt_id"),
        job.get("training_attempt_id"),
        config.get("training_attempt_id"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    job_id = str(job.get("id") or "").strip()
    attempt = positive_int(job.get("attempt"), 0)
    return f"{job_id}:attempt-{attempt}" if job_id and attempt > 0 else ""


def callback_identity(job: dict, modal_resources: dict | None = None) -> dict:
    identity = {
        "training_attempt_id": training_attempt_id(job),
    }
    resources = modal_resources if isinstance(modal_resources, dict) else None
    if resources:
        telemetry = resource_telemetry(resources)
        identity.update(
            {
                "requested_gpu_type": telemetry["requested_gpu_type"],
                "effective_gpu_type": telemetry["effective_gpu_type"],
                "memory_mb": telemetry["memory_mb"],
                "requested_batch_size": telemetry["requested_batch_size"],
                "effective_batch_size": telemetry["effective_batch_size"],
                "batch_size_policy": telemetry["batch_size_policy"],
                "modal_resource_signature": telemetry["resource_signature"],
            }
        )
    return {key: value for key, value in identity.items() if value not in ("", None, 0)}


def callback_token(job: dict | None) -> str:
    if not isinstance(job, dict):
        return ""
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    session = remote_training_session_metadata(job)
    for value in (
        config.get("callback_token"),
        job.get("callback_token"),
        session.get("callback_token"),
        session.get("token"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def remote_training_session_metadata(job: dict | None) -> dict:
    if not isinstance(job, dict):
        return {}
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    for key in ("remote_training_session", "remoteTrainingSession", "RemoteTrainingSession"):
        value = config.get(key)
        if isinstance(value, dict):
            return deepcopy(value)
    return {}


def modal_callback_metadata(job: dict | None) -> dict:
    metadata: dict = {}
    if not isinstance(job, dict):
        return metadata
    attempt_id = training_attempt_id(job)
    if attempt_id:
        metadata["training_attempt_id"] = attempt_id
    token = callback_token(job)
    if token:
        metadata["callback_token"] = token
    session = remote_training_session_metadata(job)
    if session:
        metadata["remote_training_session"] = session
    return metadata


def normalize_gpu_type(value: object) -> str:
    normalized = str(value or "").strip().upper().replace("_", "-")
    if normalized in {"A100-40G", "A100-40GB"}:
        return "A100-40GB"
    if normalized in {"A100-80G", "A100-80GB"}:
        return "A100-80GB"
    return normalized or "T4"


def default_modal_memory_mb(gpu_type: str, *, detection_job: bool = False) -> int:
    gpu = normalize_gpu_type(gpu_type)
    defaults = {
        "T4": 16384,
        "L4": 24576,
        "A10": 32768,
        "L40S": 65536,
        "A100": 65536,
        "A100-40GB": 65536,
        "A100-80GB": 98304,
    }
    memory_mb = defaults.get(gpu, 24576)
    if detection_job:
        return max(memory_mb, 24576)
    return memory_mb


def resource_signature(gpu_type: str, batch_size: int, memory_mb: int) -> str:
    return (
        f"gpu={normalize_gpu_type(gpu_type)}"
        f"|batch={positive_int(batch_size, 0)}"
        f"|memory_mb={positive_int(memory_mb, 0)}"
    )


def first_non_empty(*values: object) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return ""


def positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
