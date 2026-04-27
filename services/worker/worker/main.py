from __future__ import annotations

import os
import time
from worker.orchestrator_client import OrchestratorClient
from worker.jobs import run_job

REQUEST_TIMEOUT_SECONDS = 10
POLL_INTERVAL_SECONDS = 5


def main() -> None:
    oURL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8080")
    client = OrchestratorClient(oURL, timeout=REQUEST_TIMEOUT_SECONDS)

    worker = client.register_worker(os.getenv("PROJECT_ID"))
    worker_id = worker["id"]

    while True:
        job = client.poll_job(worker_id)

        if job is None:
            print("There are no jobs.")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        print(f"Running job {job['id']} ({job['template']})")

        try:
            run_job(client, job)
        except Exception as exc:
            client.fail_job(job["id"], str(exc))
            print(f"Job {job['id']} failed: {exc}")


if __name__ == "__main__":
    main()
