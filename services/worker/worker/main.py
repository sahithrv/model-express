from __future__ import annotations

import os
import random
import time
import traceback

import requests

for _key, _value in {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(_key, _value)

from worker.diagnostics import log_event
from worker.orchestrator_client import OrchestratorClient
from worker.jobs import run_job
from worker.training.modal_dispatcher import modal_dispatcher_enabled, run_modal_dispatcher

POLL_INTERVAL_SECONDS = 5
IDLE_LOG_SECONDS = 60
RETRY_INITIAL_SECONDS = 1.0
RETRY_MAX_SECONDS = 30.0
RETRY_JITTER_FRACTION = 0.2
FAIL_REPORT_MAX_ATTEMPTS = 3
GENERIC_PROVIDER_FALLBACK_TEMPLATES = [
    "profile_dataset",
    "analyze_dataset_visuals",
    "export_champion",
    "generate_visual_exemplars",
]


def main() -> None:
    oURL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8080")
    client = OrchestratorClient(oURL)
    project_id = os.getenv("PROJECT_ID")

    if modal_dispatcher_enabled():
        run_modal_dispatcher(client, project_id or "")
        return

    worker = _register_worker_with_retry(client, project_id or "")
    worker_id = worker["id"]
    log_event(
        "info",
        "worker_registered",
        worker_id=worker_id,
        project_id=os.getenv("PROJECT_ID", ""),
        gpu_type=os.getenv("GPU_TYPE", "local"),
    )

    last_idle_log_at = 0.0
    poll_backoff_attempt = 0
    while True:
        provider = _poll_provider()
        try:
            job = client.poll_job(
                worker_id,
                provider=provider,
                include_unspecified_provider_templates=GENERIC_PROVIDER_FALLBACK_TEMPLATES if provider else None,
            )
            poll_backoff_attempt = 0
        except requests.RequestException as exc:
            delay = _retry_delay_seconds(poll_backoff_attempt)
            poll_backoff_attempt += 1
            log_event(
                "warn",
                "worker_poll_failed_retrying",
                worker_id=worker_id,
                project_id=project_id or "",
                orchestrator_url=oURL,
                error=str(exc),
                retry_delay_seconds=round(delay, 3),
            )
            time.sleep(delay)
            continue

        if job is None:
            now = time.monotonic()
            idle_log_seconds = worker_idle_log_seconds()
            if idle_log_seconds > 0 and (last_idle_log_at == 0.0 or now - last_idle_log_at >= idle_log_seconds):
                print("There are no jobs.")
                log_event("debug", "worker_idle", worker_id=worker_id, project_id=project_id or "")
                last_idle_log_at = now
            time.sleep(worker_poll_interval_seconds())
            continue

        print(f"Running job {job['id']} ({job['template']})")
        log_event(
            "info",
            "job_started",
            worker_id=worker_id,
            job_id=job.get("id"),
            project_id=job.get("project_id"),
            template=job.get("template"),
        )

        try:
            with client.job_context(job):
                run_job(client, job)
            log_event(
                "info",
                "job_finished",
                worker_id=worker_id,
                job_id=job.get("id"),
                project_id=job.get("project_id"),
                template=job.get("template"),
            )
        except Exception as exc:
            retryable = _worker_exception_retryable(exc)
            try:
                _report_job_failure_with_retry(
                    client,
                    job,
                    str(exc),
                    retryable=retryable,
                    metadata={"failure_class": "worker_exception" if retryable else "validation"},
                )
            except Exception as report_exc:
                log_event(
                    "error",
                    "job_failure_report_failed",
                    worker_id=worker_id,
                    job_id=job.get("id"),
                    project_id=job.get("project_id"),
                    template=job.get("template"),
                    error=str(exc),
                    report_error=str(report_exc),
                    traceback=traceback.format_exc(),
                )
            log_event(
                "error",
                "job_failed",
                worker_id=worker_id,
                job_id=job.get("id"),
                project_id=job.get("project_id"),
                template=job.get("template"),
                error=str(exc),
                retryable=retryable,
                traceback=traceback.format_exc(),
            )
            print(f"Job {job['id']} failed: {exc}")


def _poll_provider() -> str | None:
    gpu_type = os.getenv("GPU_TYPE", "").strip().lower().replace("-", "_")
    if gpu_type.startswith("persistent_gpu") or gpu_type.startswith("persistent_disk"):
        return "persistent_gpu"
    return None


def _worker_exception_retryable(exc: Exception) -> bool:
    return not isinstance(exc, (KeyError, ValueError))


def _register_worker_with_retry(client: OrchestratorClient, project_id: str) -> dict:
    attempt = 0
    while True:
        try:
            worker = client.register_worker(project_id)
            if attempt:
                log_event("info", "worker_register_recovered", project_id=project_id, attempts=attempt + 1)
            return worker
        except requests.RequestException as exc:
            delay = _retry_delay_seconds(attempt)
            attempt += 1
            log_event(
                "warn",
                "worker_register_failed_retrying",
                project_id=project_id,
                orchestrator_url=getattr(client, "base_url", ""),
                error=str(exc),
                retry_delay_seconds=round(delay, 3),
            )
            time.sleep(delay)


def _report_job_failure_with_retry(
    client: OrchestratorClient,
    job: dict,
    error: str,
    *,
    retryable: bool,
    metadata: dict,
) -> None:
    attempt = 0
    while True:
        try:
            client.fail_job(
                job["id"],
                error,
                retryable=retryable,
                metadata=metadata,
                job=job,
            )
            return
        except requests.RequestException as exc:
            if attempt >= FAIL_REPORT_MAX_ATTEMPTS - 1 or not _transient_request_error(exc):
                raise
            delay = _retry_delay_seconds(attempt)
            attempt += 1
            log_event(
                "warn",
                "job_failure_report_retrying",
                job_id=job.get("id"),
                project_id=job.get("project_id"),
                error=str(exc),
                retry_delay_seconds=round(delay, 3),
                attempt=attempt,
                max_attempts=FAIL_REPORT_MAX_ATTEMPTS,
            )
            time.sleep(delay)


def _transient_request_error(exc: requests.RequestException) -> bool:
    if isinstance(exc, requests.HTTPError):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return status_code in {408, 429} or (isinstance(status_code, int) and status_code >= 500)
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


def _retry_delay_seconds(attempt: int) -> float:
    base = min(RETRY_MAX_SECONDS, RETRY_INITIAL_SECONDS * (2 ** max(0, attempt)))
    return base + random.uniform(0, base * RETRY_JITTER_FRACTION)


def worker_poll_interval_seconds() -> float:
    return _positive_float_env("MODEL_EXPRESS_WORKER_POLL_INTERVAL_SECONDS", POLL_INTERVAL_SECONDS)


def worker_idle_log_seconds() -> float:
    return _non_negative_float_env("MODEL_EXPRESS_WORKER_IDLE_LOG_SECONDS", IDLE_LOG_SECONDS)


def _positive_float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return float(default)
    try:
        parsed = float(value)
    except ValueError:
        return float(default)
    return parsed if parsed > 0 else float(default)


def _non_negative_float_env(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return float(default)
    try:
        parsed = float(value)
    except ValueError:
        return float(default)
    return parsed if parsed >= 0 else float(default)


if __name__ == "__main__":
    main()
