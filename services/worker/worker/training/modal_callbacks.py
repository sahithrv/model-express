from __future__ import annotations

from worker.training.modal_resources import (
    callback_identity,
    callback_token,
    failure_callback_payload,
    modal_resources_from_payload,
)
from worker.training.modal_runtime import DEFAULT_ORCHESTRATOR_REPORT_TIMEOUT_SECONDS, _positive_int_env


def _post_json(url: str, payload: dict, *, callback_token: str = "") -> None:
    import requests

    kwargs = {"json": payload, "timeout": _orchestrator_report_timeout_seconds()}
    token = str(callback_token or "").strip()
    if token:
        kwargs["headers"] = {"Authorization": f"Bearer {token}"}
    response = requests.post(url, **kwargs)
    response.raise_for_status()


def _post_callback_json(url: str, payload: dict, callback_auth_token: str) -> None:
    token = str(callback_auth_token or "").strip()
    if token:
        _post_json(url, payload, callback_token=token)
        return
    _post_json(url, payload)


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
        _post_callback_json(
            f"{orchestrator_url}/jobs/{job_id}/fail",
            failure_callback_payload(job, message, modal_resources),
            callback_token(job),
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
    _post_callback_json(
        f"{orchestrator_url}/jobs/{job_id}/{endpoint}",
        {
            **payload,
            **callback_identity(job, modal_resources),
        },
        callback_token(job),
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
    _post_callback_json(
        f"{orchestrator_url}/jobs/{job_id}/training-run-summary",
        payload,
        callback_token(job),
    )


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
    _post_callback_json(
        f"{orchestrator_url}/jobs/{job_id}/training-run-evaluation",
        payload,
        callback_token(job),
    )


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
