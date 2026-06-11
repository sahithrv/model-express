from __future__ import annotations

import os
import secrets
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from worker.diagnostics import log_event
from worker.orchestrator_client import OrchestratorClient
from worker.training.modal_resources import (
    callback_identity,
    failure_callback_payload,
    job_with_modal_resources,
    modal_callback_metadata,
    resource_telemetry,
    resolve_modal_resources,
)


class ModalRetryableFailureReported(RuntimeError):
    pass


class ModalJobCancelled(RuntimeError):
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
    detection_job = _is_detection_training_config(config)
    modal_resources = resolve_modal_resources(config, detection_job=detection_job)
    job_payload = job_with_modal_resources(job, modal_resources)
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

    orchestrator_url = _modal_orchestrator_url(client)

    payload = {
        "job": job_payload,
        "dataset": dataset,
        "modal_resources": modal_resources,
        "orchestrator_url": orchestrator_url,
        **_modal_storage_payload(job_payload, dataset),
        **modal_callback_metadata(job_payload),
    }

    print(
        "Submitting Modal training job "
        f"{job['id']} model={config.get('model')} "
        f"requested_gpu={modal_resources['requested_gpu_type']} "
        f"effective_gpu={modal_resources['effective_gpu_type']} "
        f"memory_mb={modal_resources['memory_mb']} "
        f"batch={modal_resources['effective_batch_size']}"
    )

    try:
        with _modal_invocation_context(app):
            configured_function = _function_with_modal_options(training_function, modal_resources)
            result = _invoke_modal_function(configured_function, payload, client, job_payload, modal_resources)
    except ModalJobCancelled as exc:
        log_event(
            "info",
            "modal_training_cancelled",
            job_id=job["id"],
            project_id=job.get("project_id", ""),
            reason=str(exc),
            modal_resource_signature=modal_resources.get("resource_signature", ""),
        )
        print(f"Modal training cancelled for {job['id']}: {exc}")
        return
    except Exception as exc:
        message = _modal_training_error_message(exc)
        client.fail_job(
            job["id"],
            message,
            retryable=True,
            metadata=failure_callback_payload(job_payload, message, modal_resources),
            job=job_payload,
        )
        log_event(
            "warn",
            "modal_training_retry_queued",
            job_id=job["id"],
            project_id=job.get("project_id", ""),
            error=message,
            failure_class=failure_callback_payload(job_payload, message, modal_resources).get("failure_class"),
            modal_resource_signature=modal_resources.get("resource_signature", ""),
        )
        print(f"Modal training reported retryable failure for {job['id']}: {message}")
        return

    if isinstance(result, dict) and result.get("status") == "retryable_failure_reported":
        print(f"Modal training reported retryable failure for {job['id']}: {result.get('error', '')}")
        return

    _log_dataset_materialization(job.get("id", ""), job.get("project_id", ""), result)
    print(f"Modal training finished for {job['id']}: {result}")


def run_modal_training_batch(client: OrchestratorClient, jobs: list[dict], batch: dict) -> None:
    """Submit a Modal preview batch when enabled, otherwise preserve single-job behavior."""
    job_ids = [str(job.get("id") or "") for job in jobs]
    remote_status = "stubbed_single_job_fallback"
    log_event(
        "info",
        "modal_training_batch_submission_decision",
        project_id=batch.get("project_id", ""),
        plan_id=batch.get("plan_id", ""),
        batch_id=batch.get("batch_id", ""),
        batch_key=batch.get("batch_key", ""),
        dataset_id=batch.get("dataset_id", ""),
        dataset_cache_key=batch.get("dataset_cache_key", ""),
        training_tier=batch.get("training_tier", ""),
        task_type=batch.get("task_type", ""),
        job_ids=job_ids,
        status=remote_status,
    )

    if _modal_batch_runner_enabled() and _valid_modal_preview_batch(batch, jobs):
        result = _try_run_remote_modal_training_batch(client, jobs, batch)
        if _remote_modal_training_batch_completed(result):
            return
        remote_status = _remote_modal_training_batch_status(result)

    for job in _jobs_with_modal_batch_metadata(jobs, batch, remote_status):
        run_modal_training(client, job)


def _try_run_remote_modal_training_batch(client: OrchestratorClient, jobs: list[dict], batch: dict) -> dict:
    try:
        from worker.training.modal_app import app, train_modal_preview_batch
    except ModuleNotFoundError as exc:
        if exc.name == "modal":
            return {"status": "failed_before_invocation", "error": "Modal is not installed."}
        raise

    try:
        dataset = client.get_dataset(str(batch["dataset_id"]))
        orchestrator_url = _modal_orchestrator_url(client)
        tagged_jobs = _jobs_with_modal_batch_metadata(jobs, batch, "remote_batch_submitted")
        enriched_jobs = []
        resources_by_job = {}
        for tagged_job in tagged_jobs:
            tagged_config = tagged_job.get("config") if isinstance(tagged_job.get("config"), dict) else {}
            tagged_resources = resolve_modal_resources(
                tagged_config,
                detection_job=_is_detection_training_config(tagged_config),
            )
            resources_by_job[str(tagged_job.get("id") or "")] = tagged_resources
            enriched_jobs.append(job_with_modal_resources(tagged_job, tagged_resources))
        batch_resources = _modal_batch_invocation_resources(enriched_jobs, resources_by_job)
        payload = {
            "batch": batch,
            "jobs": enriched_jobs,
            "dataset": dataset,
            "modal_resources": batch_resources,
            "orchestrator_url": orchestrator_url,
            **_modal_storage_payload_for_jobs(enriched_jobs, dataset),
        }
        job_callback_metadata = {
            str(job.get("id") or ""): metadata
            for job in enriched_jobs
            for metadata in (modal_callback_metadata(job),)
            if str(job.get("id") or "") and metadata
        }
        if job_callback_metadata:
            payload["job_callback_metadata"] = job_callback_metadata
        with _modal_invocation_context(app):
            configured_function = _function_with_modal_options(train_modal_preview_batch, batch_resources)
            result = _remote_function(configured_function)(payload)
    except Exception as exc:
        message = _modal_training_batch_error_message(exc)
        log_event(
            "warn",
            "modal_training_batch_remote_fallback",
            project_id=batch.get("project_id", ""),
            plan_id=batch.get("plan_id", ""),
            batch_id=batch.get("batch_id", ""),
            batch_key=batch.get("batch_key", ""),
            job_ids=[str(job.get("id") or "") for job in jobs],
            error=message,
        )
        return {"status": "failed_before_completion", "error": message}

    log_event(
        "info",
        "modal_training_batch_remote_result",
        project_id=batch.get("project_id", ""),
        plan_id=batch.get("plan_id", ""),
        batch_id=batch.get("batch_id", ""),
        batch_key=batch.get("batch_key", ""),
        status=result.get("status") if isinstance(result, dict) else "",
        runner_status=result.get("runner_status") if isinstance(result, dict) else "",
        job_ids=[str(job.get("id") or "") for job in jobs],
    )
    return result if isinstance(result, dict) else {"status": "invalid_result"}


def _jobs_with_modal_batch_metadata(jobs: list[dict], batch: dict, remote_status: str) -> list[dict]:
    batch_size = len(jobs)
    tagged_jobs: list[dict] = []
    for index, job in enumerate(jobs):
        config = job.get("config") if isinstance(job.get("config"), dict) else {}
        tagged_jobs.append(
            {
                **job,
                "config": {
                    **config,
                    "modal_batch": {
                        **batch,
                        "batch_index": index,
                        "batch_size": batch_size,
                        "batch_remote_status": remote_status,
                    },
                },
            }
        )
    return tagged_jobs


def _modal_batch_invocation_resources(enriched_jobs: list[dict], resources_by_job: dict[str, dict]) -> dict:
    if not enriched_jobs:
        return resolve_modal_resources({}, detection_job=False)
    first_job = enriched_jobs[0]
    first_id = str(first_job.get("id") or "")
    selected = resources_by_job.get(first_id)
    if not selected:
        config = first_job.get("config") if isinstance(first_job.get("config"), dict) else {}
        selected = resolve_modal_resources(config, detection_job=_is_detection_training_config(config))
    return {
        **selected,
        "batch_job_count": len(enriched_jobs),
        "batch_job_ids": [str(job.get("id") or "") for job in enriched_jobs],
    }


def _remote_modal_training_batch_completed(result: dict) -> bool:
    status = str(result.get("status") or "").strip().lower() if isinstance(result, dict) else ""
    return status in {"succeeded", "success", "completed", "complete"}


def _remote_modal_training_batch_status(result: dict) -> str:
    if not isinstance(result, dict):
        return "remote_batch_invalid_result_single_job_fallback"
    runner_status = str(result.get("runner_status") or "").strip()
    if runner_status:
        return f"{runner_status}_single_job_fallback"
    status = str(result.get("status") or "").strip()
    if status:
        return f"remote_batch_{status}_single_job_fallback"
    return "remote_batch_unknown_single_job_fallback"


def _valid_modal_preview_batch(batch: dict, jobs: list[dict]) -> bool:
    if str(batch.get("schema_version") or "") != "modal_preview_batch.v1":
        return False
    if str(batch.get("training_tier") or "").strip().lower() != "preview":
        return False
    if len(jobs) < 2:
        return False
    if str(batch.get("task_type") or "").strip().lower() == "object_detection" and not _yolo_batch_preview_enabled():
        return False
    required = (
        "batch_id",
        "batch_key",
        "project_id",
        "plan_id",
        "dataset_id",
        "dataset_cache_key",
        "task_type",
    )
    return all(str(batch.get(field) or "").strip() for field in required)


def _modal_batch_runner_enabled() -> bool:
    return _optional_bool_value(os.getenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER")) is True


def _yolo_batch_preview_enabled() -> bool:
    return _optional_bool_value(os.getenv("MODEL_EXPRESS_YOLO_BATCH_PREVIEW")) is True


def _optional_bool_value(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


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

    payload = {
        "job": job,
        "dataset": dataset,
        **_modal_storage_payload(job, dataset),
        **modal_callback_metadata(job),
    }

    print(f"Submitting Modal dataset profile job dataset={dataset_id}")
    try:
        with _modal_invocation_context(app):
            result = _remote_function(profile_image_dataset)(payload)
    except Exception as exc:
        message = _modal_dataset_profile_error_message(exc)
        client.fail_job(job["id"], message, retryable=True, job=job)
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

    payload = {
        "job": job,
        "dataset": dataset,
        **_modal_storage_payload(job, dataset),
        **modal_callback_metadata(job),
    }

    print(f"Pre-warming Modal dataset materialization dataset={dataset_id}")
    try:
        with _modal_invocation_context(app):
            result = _remote_function(materialize_image_dataset)(payload)
    except Exception as exc:
        message = _modal_dataset_materialization_error_message(exc)
        client.fail_job(job["id"], message, retryable=True, job=job)
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


def _modal_orchestrator_url(client: OrchestratorClient) -> str:
    orchestrator_url = os.getenv("MODAL_ORCHESTRATOR_URL", client.base_url)
    _require_remote_reachable_url("MODAL_ORCHESTRATOR_URL", orchestrator_url)
    return orchestrator_url


def _modal_storage_payload(job: dict, dataset: dict) -> dict:
    return _modal_storage_payload_for_jobs([job], dataset)


def _modal_storage_payload_for_jobs(jobs: list[dict], dataset: dict) -> dict:
    s3_endpoint_url = os.getenv("MODAL_S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))
    _require_remote_reachable_url("MODAL_S3_ENDPOINT_URL", s3_endpoint_url)
    credentials = _modal_storage_credentials()
    scope = _modal_storage_scope(jobs, dataset)
    return {
        "s3_endpoint_url": s3_endpoint_url,
        **credentials,
        "storage_scope": scope,
    }


def _modal_storage_credentials() -> dict:
    access_key = (
        os.getenv("MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID", "").strip()
        or os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    )
    secret_key = (
        os.getenv("MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY", "").strip()
        or os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    )
    if not access_key or not secret_key:
        raise ValueError("Modal storage requires explicit scoped S3 credentials.")
    if (
        access_key == "model_express"
        and secret_key == "model_express_password"
        and not _env_flag("MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE", False)
    ):
        raise ValueError(
            "Modal storage refuses the default local MinIO root credentials; configure scoped credentials or presigned storage."
        )
    return {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "aws_default_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }


def _modal_storage_scope(jobs: list[dict], dataset: dict) -> dict:
    bucket, dataset_key = _dataset_storage_bucket_key(dataset)
    write_prefixes = []
    for job in jobs:
        prefix = _job_storage_prefix(job)
        if prefix:
            write_prefixes.append(prefix + "/")
    write_prefixes = sorted(set(write_prefixes))
    expires_at = _modal_storage_expires_at(jobs)
    return {
        "token": secrets.token_urlsafe(24),
        "buckets": [bucket],
        "read_keys": [dataset_key],
        "read_prefixes": write_prefixes,
        "write_prefixes": write_prefixes,
        "expires_at": expires_at,
    }


def _dataset_storage_bucket_key(dataset: dict) -> tuple[str, str]:
    storage_uri = str(dataset.get("storage_uri") or "").strip()
    parsed = urlparse(storage_uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("Modal storage scope requires an s3:// dataset storage_uri.")
    return parsed.netloc, parsed.path.lstrip("/")


def _job_storage_prefix(job: dict) -> str:
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    session = config.get("remote_training_session") if isinstance(config.get("remote_training_session"), dict) else {}
    prefix = str(session.get("storage_prefix") or "").strip().strip("/")
    if prefix:
        return prefix
    job_id = str(job.get("id") or "").strip()
    if not job_id:
        return ""
    return f"model-express/artifacts/{_safe_storage_part(job_id)}"


def _modal_storage_expires_at(jobs: list[dict]) -> str:
    for job in jobs:
        config = job.get("config") if isinstance(job.get("config"), dict) else {}
        session = config.get("remote_training_session") if isinstance(config.get("remote_training_session"), dict) else {}
        expires_at = str(session.get("expires_at") or "").strip()
        if expires_at:
            return expires_at
    return (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()


def _safe_storage_part(value: str) -> str:
    out = []
    last_dash = False
    for char in value.strip():
        if char.isalnum() or char in {"_", "-", "."}:
            out.append(char)
            last_dash = False
        elif not last_dash:
            out.append("-")
            last_dash = True
    text = "".join(out).strip("-.")
    return text or "unknown"


def _remote_function(function):
    remote = getattr(function, "remote", None)
    if remote is None:
        raise RuntimeError(
            "Modal is not installed or the Modal function was not registered. "
            "Install worker dependencies, then run `modal setup`."
        )
    return remote


def _invoke_modal_function(function, payload: dict, client: OrchestratorClient, job: dict, modal_resources: dict):
    spawn = getattr(function, "spawn", None)
    if not modal_spawn_invocation_enabled() or not callable(spawn):
        return _remote_function(function)(payload)
    call = spawn(payload)
    call_object_id = _modal_function_call_object_id(call)
    if call_object_id:
        _report_modal_call(client, job, modal_resources, call_object_id, "active")
    return _wait_for_modal_function_call(call, client, job, modal_resources, call_object_id)


def _wait_for_modal_function_call(call, client: OrchestratorClient, job: dict, modal_resources: dict, call_object_id: str):
    get = getattr(call, "get", None)
    if not callable(get):
        raise RuntimeError("Modal Function.spawn() did not return a FunctionCall with get().")
    poll_seconds = modal_call_poll_seconds()
    while True:
        try:
            return get(timeout=poll_seconds)
        except Exception as exc:
            if not _is_modal_get_timeout(exc):
                raise
            if not _job_attempt_no_longer_active(client, job):
                continue
            cancel_status = _cancel_modal_function_call(call)
            if call_object_id:
                _report_modal_call(client, job, modal_resources, call_object_id, cancel_status)
            raise ModalJobCancelled("backend marked this attempt stale or terminal") from exc


def _modal_function_call_object_id(call) -> str:
    for attr in ("object_id", "id"):
        value = str(getattr(call, attr, "") or "").strip()
        if value:
            return value
    hydrate = getattr(call, "hydrate", None)
    if callable(hydrate):
        try:
            hydrate()
        except Exception:
            return ""
        for attr in ("object_id", "id"):
            value = str(getattr(call, attr, "") or "").strip()
            if value:
                return value
    return str(getattr(call, "_object_id", "") or "").strip()


def _report_modal_call(
    client: OrchestratorClient,
    job: dict,
    modal_resources: dict,
    call_object_id: str,
    cancel_status: str,
) -> None:
    reporter = getattr(client, "report_modal_call", None)
    if not callable(reporter):
        return
    telemetry = resource_telemetry(modal_resources)
    payload = {
        **callback_identity(job, modal_resources),
        "modal_function_call_object_id": call_object_id,
        "cancel_status": cancel_status,
        "modal_resources": {
            **telemetry,
            "modal_function_call_object_id": call_object_id,
            "modal_call_cancel_status": cancel_status,
        },
    }
    try:
        try:
            reporter(str(job.get("id") or ""), payload, job=job)
        except TypeError:
            reporter(str(job.get("id") or ""), payload)
    except Exception as exc:
        log_event(
            "warn",
            "modal_function_call_report_failed",
            job_id=job.get("id", ""),
            project_id=job.get("project_id", ""),
            modal_function_call_object_id=call_object_id,
            error=str(exc),
        )


def _job_attempt_no_longer_active(client: OrchestratorClient, job: dict) -> bool:
    getter = getattr(client, "get_job", None)
    if not callable(getter):
        return False
    job_id = str(job.get("id") or "")
    if not job_id:
        return False
    try:
        latest = getter(job_id)
    except Exception as exc:
        log_event(
            "warn",
            "modal_attempt_status_check_failed",
            job_id=job_id,
            project_id=job.get("project_id", ""),
            error=str(exc),
        )
        return False
    latest_status = str(latest.get("status") or "").strip().upper()
    if latest_status in {"FAILED", "SUCCEEDED", "CANCELLED"}:
        return True
    latest_config = latest.get("config") if isinstance(latest.get("config"), dict) else {}
    expected_attempt = callback_identity(job, modal_resources=None).get("training_attempt_id", "")
    active_attempt = str(latest_config.get("active_attempt_id") or "").strip()
    return bool(expected_attempt and active_attempt and active_attempt != expected_attempt)


def _cancel_modal_function_call(call) -> str:
    cancel = getattr(call, "cancel", None)
    if not callable(cancel):
        return "cancel_unavailable"
    try:
        cancel(terminate_containers=modal_cancel_terminate_containers())
        return "cancel_requested"
    except TypeError:
        cancel()
        return "cancel_requested"
    except Exception as exc:
        log_event("warn", "modal_function_call_cancel_failed", error=str(exc))
        return "cancel_failed"


def _is_modal_get_timeout(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    if isinstance(exc, TimeoutError) or "timeout" in name:
        return True
    return "timed out" in str(exc).lower()


def _function_with_modal_options(function, modal_resources: dict):
    options = modal_resources.get("modal_function_options")
    if not isinstance(options, dict):
        options = {}
    options = {
        key: value
        for key, value in options.items()
        if key in {"gpu", "memory"} and value not in ("", None, 0)
    }
    if not options:
        return function
    with_options = getattr(function, "with_options", None)
    if not callable(with_options):
        return function
    return with_options(**options)


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


def _modal_training_batch_error_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = exc.__class__.__name__
    return f"Modal preview batch invocation failed before completion: {message}"[:2000]


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
        materialization_cache_root=telemetry.get("dataset_materialization_cache_root"),
        materialization_cache_scope=telemetry.get("dataset_materialization_cache_scope"),
        prewarm_cache_root=telemetry.get("dataset_prewarm_cache_root"),
        training_cache_root=telemetry.get("dataset_training_cache_root"),
        prewarm_reusable_for_training=telemetry.get("dataset_prewarm_reusable_for_training"),
        prewarm_reuse_status=telemetry.get("dataset_prewarm_reuse_status"),
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


def modal_spawn_invocation_enabled() -> bool:
    return _env_flag("MODEL_EXPRESS_MODAL_USE_SPAWN", True)


def modal_cancel_terminate_containers() -> bool:
    return _env_flag("MODEL_EXPRESS_MODAL_CANCEL_TERMINATE_CONTAINERS", True)


def modal_call_poll_seconds() -> float:
    value = os.getenv("MODEL_EXPRESS_MODAL_CALL_POLL_SECONDS", "").strip()
    if not value:
        return 5.0
    try:
        parsed = float(value)
    except ValueError:
        return 5.0
    return max(0.25, min(parsed, 60.0))


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}
