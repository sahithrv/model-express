from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from urllib.parse import urlparse

from worker.diagnostics import log_event
from worker.orchestrator_client import OrchestratorClient


class ModalRetryableFailureReported(RuntimeError):
    pass


_MODAL_APP_SESSION_LOCK = threading.RLock()
_MODAL_APP_SESSION_DEPTH = 0


@contextmanager
def modal_app_session():
    """Keep the Modal app hydrated while dispatcher threads submit remote calls."""
    global _MODAL_APP_SESSION_DEPTH

    with _MODAL_APP_SESSION_LOCK:
        if _MODAL_APP_SESSION_DEPTH > 0:
            _MODAL_APP_SESSION_DEPTH += 1
            nested = True
        else:
            nested = False

    if nested:
        try:
            yield
        finally:
            with _MODAL_APP_SESSION_LOCK:
                _MODAL_APP_SESSION_DEPTH -= 1
        return

    try:
        from worker.training.modal_app import app
    except ModuleNotFoundError as exc:
        if exc.name == "modal":
            raise RuntimeError(
                "Modal is not installed. Install worker dependencies, then run `modal setup`."
            ) from exc
        raise

    with _modal_app_run(app):
        with _MODAL_APP_SESSION_LOCK:
            _MODAL_APP_SESSION_DEPTH = 1
        try:
            yield
        finally:
            with _MODAL_APP_SESSION_LOCK:
                _MODAL_APP_SESSION_DEPTH = 0


def run_modal_training(client: OrchestratorClient, job: dict) -> None:
    config = job["config"]
    if config.get("gpu_type"):
        os.environ["MODAL_GPU_TYPE"] = str(config["gpu_type"])

    detection_job = _is_detection_training_config(config)
    try:
        if detection_job:
            from worker.training.modal_app import app, train_yolo_detector as training_function
        else:
            from worker.training.modal_app import app, train_image_classifier as training_function
    except ModuleNotFoundError as exc:
        if exc.name == "modal":
            raise RuntimeError(
                "Modal is not installed. Install worker dependencies, then run `modal setup`."
            ) from exc
        raise

    dataset = client.get_dataset(str(config["dataset_id"]))

    orchestrator_url = os.getenv("MODAL_ORCHESTRATOR_URL", client.base_url)
    s3_endpoint_url = os.getenv("MODAL_S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))

    _require_remote_reachable_url("MODAL_ORCHESTRATOR_URL", orchestrator_url)
    _require_remote_reachable_url("MODAL_S3_ENDPOINT_URL", s3_endpoint_url)

    payload = {
        "job": job,
        "dataset": dataset,
        "orchestrator_url": orchestrator_url,
        "s3_endpoint_url": s3_endpoint_url,
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "model_express"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "model_express_password"),
        "aws_default_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }

    print(
        "Submitting Modal training job "
        f"{job['id']} model={config.get('model')} gpu={config.get('gpu_type') or os.getenv('MODAL_GPU_TYPE', 'T4')}"
    )

    try:
        with _modal_invocation_context(app):
            result = _remote_function(training_function)(payload)
    except Exception as exc:
        message = _modal_training_error_message(exc)
        client.fail_job(job["id"], message, retryable=True)
        log_event(
            "warn",
            "modal_training_retry_queued",
            job_id=job["id"],
            project_id=job.get("project_id", ""),
            error=message,
        )
        print(f"Modal training reported retryable failure for {job['id']}: {message}")
        return

    if isinstance(result, dict) and result.get("status") == "retryable_failure_reported":
        print(f"Modal training reported retryable failure for {job['id']}: {result.get('error', '')}")
        return

    _log_dataset_materialization(job.get("id", ""), job.get("project_id", ""), result)
    print(f"Modal training finished for {job['id']}: {result}")


def _is_detection_training_config(config: dict) -> bool:
    model = str(config.get("model", "")).lower()
    return (
        str(config.get("task_type", "")).lower() == "object_detection"
        or str(config.get("model_kind", "")).lower() == "ultralytics_yolo_detector"
        or model.startswith("yolo11")
        or model.startswith("yolo")
    )


def run_modal_dataset_profile(client: OrchestratorClient, job: dict) -> None:
    config = job["config"]
    dataset_id = str(config["dataset_id"])

    try:
        from worker.training.modal_app import app, profile_image_dataset
    except ModuleNotFoundError as exc:
        if exc.name == "modal":
            raise RuntimeError(
                "Modal is not installed. Install worker dependencies, then run `modal setup`."
            ) from exc
        raise

    dataset = client.get_dataset(dataset_id)
    s3_endpoint_url = os.getenv("MODAL_S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))
    _require_remote_reachable_url("MODAL_S3_ENDPOINT_URL", s3_endpoint_url)

    payload = {
        "job": job,
        "dataset": dataset,
        "s3_endpoint_url": s3_endpoint_url,
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "model_express"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "model_express_password"),
        "aws_default_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }

    print(f"Submitting Modal dataset profile job dataset={dataset_id}")
    try:
        with _modal_invocation_context(app):
            result = _remote_function(profile_image_dataset)(payload)
    except Exception as exc:
        message = _modal_dataset_profile_error_message(exc)
        client.fail_job(job["id"], message, retryable=True)
        log_event(
            "warn",
            "modal_dataset_profile_retry_queued",
            job_id=job["id"],
            project_id=job.get("project_id", ""),
            error=message,
        )
        print(f"Modal dataset profile reported retryable failure for {job['id']}: {message}")
        return

    profile = result.get("profile") if isinstance(result, dict) and isinstance(result.get("profile"), dict) else result
    if not isinstance(profile, dict):
        raise RuntimeError("Modal dataset profiler returned an invalid profile payload.")

    metadata_import = result.get("metadata_import") if isinstance(result, dict) else None
    if isinstance(metadata_import, dict):
        _send_dataset_metadata_import(client, dataset_id, metadata_import)
    _log_dataset_materialization(job.get("id", ""), job.get("project_id", ""), result)
    client.update_dataset_profile(dataset_id, profile)
    client.complete_job(job["id"], mlflow_run_id="")
    print(f"Modal dataset profile finished for {dataset_id}")


def run_modal_dataset_materialization(client: OrchestratorClient, job: dict) -> dict:
    config = job["config"]
    dataset_id = str(config["dataset_id"])

    try:
        from worker.training.modal_app import app, materialize_image_dataset
    except ModuleNotFoundError as exc:
        if exc.name == "modal":
            raise RuntimeError(
                "Modal is not installed. Install worker dependencies, then run `modal setup`."
            ) from exc
        raise

    dataset = client.get_dataset(dataset_id)
    s3_endpoint_url = os.getenv("MODAL_S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))
    _require_remote_reachable_url("MODAL_S3_ENDPOINT_URL", s3_endpoint_url)

    payload = {
        "job": job,
        "dataset": dataset,
        "s3_endpoint_url": s3_endpoint_url,
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "model_express"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "model_express_password"),
        "aws_default_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }

    print(f"Pre-warming Modal dataset materialization dataset={dataset_id}")
    try:
        with _modal_invocation_context(app):
            result = _remote_function(materialize_image_dataset)(payload)
    except Exception as exc:
        message = _modal_dataset_materialization_error_message(exc)
        client.fail_job(job["id"], message, retryable=True)
        log_event(
            "warn",
            "modal_dataset_materialization_retry_queued",
            job_id=job["id"],
            project_id=job.get("project_id", ""),
            error=message,
        )
        print(f"Modal dataset materialization reported retryable failure for {job['id']}: {message}")
        raise ModalRetryableFailureReported(message) from exc
    _log_dataset_materialization(job.get("id", ""), job.get("project_id", ""), result)
    return result


def _remote_function(function):
    remote = getattr(function, "remote", None)
    if remote is None:
        raise RuntimeError(
            "Modal is not installed or the Modal function was not registered. "
            "Install worker dependencies, then run `modal setup`."
        )
    return remote


@contextmanager
def _modal_invocation_context(app):
    if _modal_app_session_active():
        yield
        return
    with _modal_app_run(app):
        yield


@contextmanager
def _modal_app_run(app):
    run = getattr(app, "run", None)
    if run is None:
        raise RuntimeError(
            "Modal is not installed or the Modal app was not initialized. "
            "Install worker dependencies, then run `modal setup`."
        )
    with _modal_output_context():
        with run():
            yield


@contextmanager
def _modal_output_context():
    _make_output_streams_unicode_safe()

    try:
        import modal
    except Exception:
        yield
        return

    enable_output = getattr(modal, "enable_output", None)
    if not callable(enable_output):
        yield
        return

    with enable_output():
        yield


def _make_output_streams_unicode_safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue

        encoding = (getattr(stream, "encoding", "") or "").lower().replace("_", "-")
        errors = getattr(stream, "errors", None)
        if encoding in {"utf-8", "utf8", "utf-8-sig"} or errors == "replace":
            continue

        try:
            reconfigure(errors="replace")
        except (OSError, TypeError, ValueError):
            continue


def _modal_app_session_active() -> bool:
    with _MODAL_APP_SESSION_LOCK:
        return _MODAL_APP_SESSION_DEPTH > 0


def _require_remote_reachable_url(name: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be a full http(s) URL, got {value!r}")

    host = parsed.hostname or ""
    if host in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(
            f"{name} points at {value!r}, but Modal cannot reach your local localhost. "
            "Expose it with a tunnel and set the public URL before running Modal jobs."
        )


def _modal_training_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"Modal training invocation failed before completion: {message}"[:2000]


def _modal_dataset_profile_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"Modal dataset profile invocation failed before completion: {message}"[:2000]


def _modal_dataset_materialization_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"Modal dataset materialization invocation failed before completion: {message}"[:2000]


def _log_dataset_materialization(job_id: str, project_id: str, result: object) -> None:
    telemetry = result.get("dataset_materialization") if isinstance(result, dict) else None
    if not isinstance(telemetry, dict):
        return
    log_event(
        "info",
        "modal_dataset_materialization",
        job_id=job_id,
        project_id=project_id,
        status=telemetry.get("dataset_materialization_status"),
        cache_hit=telemetry.get("dataset_materialization_cache_hit"),
        cache_miss=telemetry.get("dataset_materialization_cache_miss"),
        bytes_downloaded=telemetry.get("dataset_materialization_bytes_downloaded"),
        extract_seconds=telemetry.get("dataset_materialization_extract_seconds"),
        wait_seconds=telemetry.get("dataset_materialization_wait_seconds"),
        dataset_checksum=telemetry.get("dataset_checksum"),
        storage_uri_fingerprint=telemetry.get("storage_uri_fingerprint"),
        dataset_cache_key=telemetry.get("dataset_materialization_cache_key"),
    )


def _send_dataset_metadata_import(client: OrchestratorClient, dataset_id: str, payload: dict) -> None:
    try:
        result = client.import_dataset_metadata(dataset_id, payload)
        if isinstance(result, dict) and result.get("status") == "unavailable":
            log_event(
                "warn",
                "dataset_metadata_import_unavailable",
                dataset_id=dataset_id,
                reason=result.get("reason"),
                status_code=result.get("status_code"),
            )
    except Exception as exc:
        log_event(
            "warn",
            "dataset_metadata_import_failed_nonfatal",
            dataset_id=dataset_id,
            error=str(exc),
        )
