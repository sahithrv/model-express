from __future__ import annotations

from worker.datasets.cache import dataset_archive_path, extract_dataset_archive
from worker.datasets.profiler import profile_image_folder
from worker.datasets.storage import download_s3_uri
from worker.orchestrator_client import OrchestratorClient

def run_job(client: OrchestratorClient, job: dict) -> None:
    template = job["template"]
    if template == "profile_dataset":
        run_profile_dataset_job(client, job)
        return 
    client.run_fake_job(job)

def run_profile_dataset_job(client: OrchestratorClient, job: dict) -> None:
    dataset_id = job["config"]["dataset_id"]

    dataset = client.get_dataset(dataset_id)

    archive_path = dataset_archive_path(dataset_id)
    download_s3_uri(dataset["storage_uri"], archive_path)

    dataset_dir = extract_dataset_archive(archive_path, dataset_id)
    profile = profile_image_folder(dataset_dir)

    client.update_dataset_profile(dataset_id, profile)
    client.complete_job(job["id"], mlflow_run_id="")   