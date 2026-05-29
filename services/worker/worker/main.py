from __future__ import annotations

import os
import time
import traceback

from worker.diagnostics import log_event
from worker.orchestrator_client import OrchestratorClient
from worker.jobs import run_job

POLL_INTERVAL_SECONDS = 5


def main() -> None:
    oURL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8080")
    client = OrchestratorClient(oURL)

    worker = client.register_worker(os.getenv("PROJECT_ID"))
    worker_id = worker["id"]
    log_event(
        "info",
        "worker_registered",
        worker_id=worker_id,
        project_id=os.getenv("PROJECT_ID", ""),
        gpu_type=os.getenv("GPU_TYPE", "local"),
    )

    while True:
        job = client.poll_job(worker_id)

        if job is None:
            print("There are no jobs.")
            time.sleep(POLL_INTERVAL_SECONDS)
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
            client.fail_job(job["id"], str(exc))
            log_event(
                "error",
                "job_failed",
                worker_id=worker_id,
                job_id=job.get("id"),
                project_id=job.get("project_id"),
                template=job.get("template"),
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            print(f"Job {job['id']} failed: {exc}")


if __name__ == "__main__":
    main()
