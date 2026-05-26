from __future__ import annotations

import os

from worker.datasets.cache import (
    cleanup_dataset_cache,
    dataset_archive_path,
    extract_dataset_archive,
    should_persist_dataset_cache,
)
from worker.datasets.profiler import profile_image_folder
from worker.datasets.storage import download_s3_uri
from worker.champion_jobs import (
    run_champion_demo_prediction_job,
    run_export_champion_job,
    run_generate_visual_exemplars_job,
)
from worker.orchestrator_client import OrchestratorClient
from worker.training.providers import run_training_job


def run_job(client: OrchestratorClient, job: dict) -> None:
    template = job["template"]
    if template == "profile_dataset":
        run_profile_dataset_job(client, job)
        return 
    if template == "train_experiment":
        run_training_job(client, job)
        return
    if template == "export_champion":
        run_export_champion_job(client, job)
        return
    if template == "champion_demo_prediction":
        run_champion_demo_prediction_job(client, job)
        return
    if template == "generate_visual_exemplars":
        run_generate_visual_exemplars_job(client, job)
        return
    raise ValueError(f"Unsupported job template: {template}")

def run_profile_dataset_job(client: OrchestratorClient, job: dict) -> None:
    dataset_id = job["config"]["dataset_id"]
    provider = _profile_provider(job)
    if provider == "modal":
        from worker.training.modal_provider import run_modal_dataset_profile

        run_modal_dataset_profile(client, job)
        return

    dataset = client.get_dataset(dataset_id)

    try:
        archive_path = dataset_archive_path(dataset_id)
        download_s3_uri(dataset["storage_uri"], archive_path)

        dataset_dir = extract_dataset_archive(archive_path, dataset_id)
        profile = profile_image_folder(dataset_dir)

        client.update_dataset_profile(dataset_id, profile)
        client.complete_job(job["id"], mlflow_run_id="")
    finally:
        if not should_persist_dataset_cache():
            cleanup_dataset_cache(dataset_id)


def _profile_provider(job: dict) -> str:
    config = job.get("config", {})
    if isinstance(config, dict) and config.get("provider"):
        return str(config["provider"]).lower()

    provider = os.getenv("MODEL_EXPRESS_DATASET_PROFILE_PROVIDER", "").strip().lower()
    if provider:
        return provider

    gpu_type = os.getenv("GPU_TYPE", "").strip().lower()
    return "modal" if gpu_type == "modal" else "local"
