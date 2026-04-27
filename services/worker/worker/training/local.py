from __future__ import annotations

import os
import time

from worker.orchestrator_client import OrchestratorClient


def run_local_training(client: OrchestratorClient, job: dict) -> None:
    """Deterministic local training simulator for locking the job/metric contract."""
    config = job["config"]
    job_id = job["id"]

    model = str(config.get("model", "unknown_model"))
    epochs = _positive_int(config.get("epochs"), default=3)
    learning_rate = _positive_float(config.get("learning_rate"), default=0.0003)
    batch_size = _positive_int(config.get("batch_size"), default=16)
    epoch_sleep = _positive_float(os.getenv("LOCAL_TRAINING_EPOCH_SECONDS"), default=0.5)

    model_score = _model_score(model)
    batch_penalty = 0.02 if batch_size < 8 else 0.0
    lr_penalty = 0.03 if learning_rate > 0.001 else 0.0
    final_macro_f1 = max(0.35, min(0.94, model_score - batch_penalty - lr_penalty))

    for epoch in range(1, epochs + 1):
        progress = epoch / epochs
        macro_f1 = round(0.24 + (final_macro_f1 - 0.24) * progress, 4)
        accuracy = round(min(0.97, macro_f1 + 0.035), 4)
        train_loss = round(max(0.04, 1.18 - 0.82 * progress - model_score * 0.08), 4)
        val_loss = round(max(0.05, 1.28 - 0.72 * progress - model_score * 0.05), 4)

        client.report_metric(
            job_id,
            epoch,
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "learning_rate": learning_rate,
            },
        )
        print(f"Reported training epoch {epoch}/{epochs} for {job_id} ({model})")
        time.sleep(epoch_sleep)

    client.complete_job(job_id, mlflow_run_id=f"local-training-{job_id}")


def _model_score(model: str) -> float:
    normalized = model.lower()
    if "efficientnet" in normalized:
        return 0.82
    if "resnet" in normalized:
        return 0.78
    if "mobilenet" in normalized:
        return 0.72

    checksum = sum(ord(char) for char in normalized)
    return 0.62 + (checksum % 20) / 100


def _positive_int(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
