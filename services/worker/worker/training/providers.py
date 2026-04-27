from __future__ import annotations

from worker.orchestrator_client import OrchestratorClient
from worker.training.local import run_local_training
from worker.training.modal_provider import run_modal_training


def run_training_job(client: OrchestratorClient, job: dict) -> None:
    config = job.get("config", {})
    provider = str(config.get("provider", "local")).lower()

    if provider == "local":
        run_local_training(client, job)
        return

    if provider == "modal":
        run_modal_training(client, job)
        return

    raise ValueError(f"Unsupported training provider: {provider}")
