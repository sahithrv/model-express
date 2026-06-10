from __future__ import annotations

import os
import time
import traceback

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
GENERIC_PROVIDER_FALLBACK_TEMPLATES = [
    "profile_dataset",
    "analyze_dataset_visuals",
    "export_champion",
    "champion_demo_prediction",
    "generate_visual_exemplars",
]


def main() -> None:
    oURL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8080")
    client = OrchestratorClient(oURL)
    project_id = os.getenv("PROJECT_ID")

    if modal_dispatcher_enabled():
        run_modal_dispatcher(client, project_id or "")
        return

    worker = client.register_worker(project_id or "")
    worker_id = worker["id"]
    log_event(
        "info",
        "worker_registered",
        worker_id=worker_id,
        project_id=os.getenv("PROJECT_ID", ""),
        gpu_type=os.getenv("GPU_TYPE", "local"),
    )

    last_idle_log_at = 0.0
    while True:
        provider = _poll_provider()
        job = client.poll_job(
            worker_id,
            provider=provider,
            include_unspecified_provider_templates=GENERIC_PROVIDER_FALLBACK_TEMPLATES if provider else None,
        )

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
            client.fail_job(
                job["id"],
                str(exc),
                retryable=retryable,
                metadata={"failure_class": "worker_exception" if retryable else "validation"},
                job=job,
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
