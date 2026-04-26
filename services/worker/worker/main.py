from __future__ import annotations

import os
import time

import requests

REQUEST_TIMEOUT_SECONDS = 10
POLL_INTERVAL_SECONDS = 5


def main() -> None:
    print("model-express-worker scaffold is ready")
    oURL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8080")
    project_id = os.getenv("PROJECT_ID")
    if not project_id:
        raise SystemExit("PROJECT_ID is required. Example: $env:PROJECT_ID='project_1'")

    worker = register_worker(oURL, project_id)
    worker_id = worker["id"]
    print(f"Registered worker {worker_id} ({worker['name']}) for project {worker['project_id']}")

    while True:
        job = poll_job(oURL, worker_id)
        if job is None:
            print("There are no jobs.")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        print(f"Running job {job['id']}")
        run_fake_job(oURL, job)


def register_worker(url: str, project_id: str) -> dict:
    response = requests.post(
        f"{url}/workers/register",
        json={
            "project_id": project_id,
            "name": os.getenv("WORKER_NAME", "local-worker-1"),
            "gpu_type": os.getenv("GPU_TYPE", "local"),
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def poll_job(url: str, worker_id: str) -> dict | None:
    response = requests.post(
        f"{url}/workers/{worker_id}/poll",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["job"]


def report_metric(url: str, job_id: str, epoch: int, metrics: dict[str, float]) -> dict:
    response = requests.post(
        f"{url}/jobs/{job_id}/metrics",
        json={
            "epoch": epoch,
            "metrics": metrics,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def complete_job(url: str, job_id: str, mlflow_run_id: str = "") -> dict:
    response = requests.post(
        f"{url}/jobs/{job_id}/complete",
        json={"mlflow_run_id": mlflow_run_id},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def fail_job(url: str, job_id: str, error: str) -> dict:
    response = requests.post(
        f"{url}/jobs/{job_id}/fail",
        json={"error": error},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def run_fake_job(url: str, job: dict) -> None:
    job_id = job["id"]

    try:
        for epoch in range(1, 4):
            metrics = {
                "train_loss": round(1.0 / epoch, 4),
                "val_loss": round(1.2 / epoch, 4),
                "macro_f1": round(0.2 * epoch, 4),
            }
            report_metric(url, job_id, epoch, metrics)
            print(f"Reported epoch {epoch} metrics for {job_id}: {metrics}")
            time.sleep(1)

        complete_job(url, job_id, mlflow_run_id=f"fake-mlflow-run-{job_id}")
        print(f"Completed job {job_id}")
    except Exception as exc:
        fail_job(url, job_id, str(exc))
        raise


if __name__ == "__main__":
    main()
