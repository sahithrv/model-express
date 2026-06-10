from __future__ import annotations

import os
import time
from pathlib import Path

from worker.datasets.cache import ensure_dataset_materialized
from worker.orchestrator_client import OrchestratorClient
from worker.training.local import run_local_training


PROVIDER_NAME = "persistent_gpu"


def run_persistent_gpu_training(client: OrchestratorClient, job: dict) -> None:
    if not _enabled():
        raise ValueError("persistent_gpu provider requires MODEL_EXPRESS_PERSISTENT_GPU_PROVIDER=1")

    cache_root = _cache_root()
    if cache_root is None:
        raise ValueError("persistent_gpu provider requires MODEL_EXPRESS_PERSISTENT_GPU_CACHE_ROOT")

    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    dataset_id = str(config.get("dataset_id") or "")
    if not dataset_id:
        raise ValueError("persistent_gpu training requires config.dataset_id")

    dataset = client.get_dataset(dataset_id)
    storage_uri = str(dataset.get("storage_uri") or "")
    checksum = str(dataset.get("checksum_sha256") or config.get("dataset_checksum_sha256") or "")
    started = time.time()
    materialized = ensure_dataset_materialized(
        dataset_id=dataset_id,
        storage_uri=storage_uri,
        checksum_sha256=checksum,
        cache_root=cache_root,
    )
    telemetry = {
        **materialized.telemetry,
        "provider": PROVIDER_NAME,
        "dataset_materialization_cache_root": str(cache_root),
        "dataset_materialization_cache_scope": "persistent_disk",
        "dataset_materialization_path": str(materialized.dataset_dir),
    }
    enriched = {
        **job,
        "config": {
            **config,
            "provider": PROVIDER_NAME,
            "dataset_dir": str(materialized.dataset_dir),
            "dataset_materialization": telemetry,
            "stage_telemetry": {
                "schema_version": "remote_gpu_stage_telemetry_v1",
                "provider": PROVIDER_NAME,
                "dataset_materialization_seconds": telemetry.get("dataset_materialization_total_seconds", 0.0),
                "queue_wait_seconds": _queue_wait_seconds(job),
                "idle_wait_seconds": 0.0,
            },
        },
    }
    client.report_training_run_summary(
        str(job["id"]),
        {
            "provider": PROVIDER_NAME,
            "gpu_type": str(config.get("gpu_type") or os.getenv("GPU_TYPE", PROVIDER_NAME)),
            "status": "RUNNING",
            "runtime_seconds": round(time.time() - started, 3),
            "estimated_cost_usd": 0,
            "dataset_materialization": telemetry,
            "stage_telemetry": enriched["config"]["stage_telemetry"],
        },
    )
    run_local_training(client, enriched)


def _enabled() -> bool:
    return os.getenv("MODEL_EXPRESS_PERSISTENT_GPU_PROVIDER", "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_root() -> Path | None:
    value = os.getenv("MODEL_EXPRESS_PERSISTENT_GPU_CACHE_ROOT", "").strip()
    return Path(value) if value else None


def _queue_wait_seconds(job: dict) -> float:
    created_at = str(job.get("created_at") or "").strip()
    if not created_at:
        return 0.0
    try:
        from datetime import datetime, timezone

        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        return max(0.0, round((datetime.now(timezone.utc) - parsed).total_seconds(), 6))
    except Exception:
        return 0.0
