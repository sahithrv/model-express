import requests
import os
import time

DEFAULT_REQUEST_TIMEOUT_SECONDS = 10
DEFAULT_REPORT_TIMEOUT_SECONDS = 300
POLL_INTERVAL_SECONDS = 5

class OrchestratorClient:
    def __init__(self, base_url: str, timeout: int | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout if timeout is not None else request_timeout_seconds()
    
    def get_dataset(self, dataset_id: str) -> dict:
        response = requests.get(
            f"{self.base_url}/datasets/{dataset_id}", timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def update_dataset_profile(self, dataset_id: str, profile: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/datasets/{dataset_id}/profile",
            json={"profile": profile},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
    
    def register_worker(self, project_id: str) -> dict:
        response = requests.post(
            f"{self.base_url}/workers/register",
            json={
                "project_id": project_id,
                "name": os.getenv("WORKER_NAME", "local-worker-1"),
                "gpu_type": os.getenv("GPU_TYPE", "local"),
            },
            timeout=request_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()


    def poll_job(self, worker_id: str) -> dict | None:
        response = requests.post(
            f"{self.base_url}/workers/{worker_id}/poll",
            timeout=request_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()["job"]


    def report_metric(self, job_id: str, epoch: int, metrics: dict[str, float]) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/metrics",
            json={
                "epoch": epoch,
                "metrics": metrics,
            },
            timeout=report_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()


    def report_training_run_summary(self, job_id: str, summary: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/training-run-summary",
            json=summary,
            timeout=report_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()

    def report_training_run_evaluation(self, job_id: str, evaluation: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/training-run-evaluation",
            json=evaluation,
            timeout=report_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()

    def report_champion_export_result(self, job_id: str, result: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/champion-export-result",
            json=result,
            timeout=report_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()

    def report_champion_demo_prediction_result(self, job_id: str, result: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/champion-demo-prediction-result",
            json=result,
            timeout=report_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()

    def report_dataset_visual_exemplars(self, dataset_id: str, payload: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/datasets/{dataset_id}/visual-exemplars",
            json=payload,
            timeout=report_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()

    def report_dataset_visual_analysis_result(self, dataset_id: str, payload: dict) -> dict:
        response = requests.post(
            f"{self.base_url}/datasets/{dataset_id}/visual-analysis-result",
            json=payload,
            timeout=report_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()

    def complete_job(self, job_id: str, mlflow_run_id: str = "") -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/complete",
            json={"mlflow_run_id": mlflow_run_id},
            timeout=report_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()


    def fail_job(self, job_id: str, error: str) -> dict:
        response = requests.post(
            f"{self.base_url}/jobs/{job_id}/fail",
            json={"error": error},
            timeout=report_timeout_seconds(),
        )
        response.raise_for_status()
        return response.json()


    def run_fake_job(self, job: dict) -> None:
        job_id = job["id"]

        try:
            for epoch in range(1, 4):
                metrics = {
                    "train_loss": round(1.0 / epoch, 4),
                    "val_loss": round(1.2 / epoch, 4),
                    "macro_f1": round(0.2 * epoch, 4),
                }
                self.report_metric(job_id, epoch, metrics)
                print(f"Reported epoch {epoch} metrics for {job_id}: {metrics}")
                time.sleep(1)

            self.complete_job(job_id, mlflow_run_id=f"fake-mlflow-run-{job_id}")
            print(f"Completed job {job_id}")
        except Exception as exc:
            self.fail_job(job_id, str(exc))
            raise


def request_timeout_seconds() -> int:
    return _positive_int_env("MODEL_EXPRESS_WORKER_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)


def report_timeout_seconds() -> int:
    return _positive_int_env("MODEL_EXPRESS_WORKER_REPORT_TIMEOUT_SECONDS", DEFAULT_REPORT_TIMEOUT_SECONDS)


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default
