"""Shared best-effort progress helpers for training providers.

Provider integrations keep framework/runtime detail in bounded detail codes and
allowlisted metadata while reporting only the stable progress taxonomy.  These
helpers deliberately add another failure boundary around ``ProgressReporter``
so progress telemetry can never become part of the training success path.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from worker.orchestrator_client import OrchestratorClient
from worker.progress import DEFAULT_PROGRESS_INITIAL_REVISION, ProgressReporter


SIMULATOR_EXECUTION_MODE = "local_simulator"
SIMULATOR_FRAMEWORK = "deterministic_simulator"

_SIMULATOR_STAGE_DETAILS = {
    "worker_starting": "simulator_starting",
    "data_loading": "simulator_data_preparing",
    "model_initializing": "simulator_model_initializing",
    "evaluating": "simulator_evaluating",
    "finalizing": "simulator_finalizing",
}


def training_progress_reporter(
    client: OrchestratorClient,
    job: dict,
    *,
    initial_revision: int = DEFAULT_PROGRESS_INITIAL_REVISION,
) -> ProgressReporter | None:
    """Construct an attempt-scoped reporter without affecting training."""
    try:
        return ProgressReporter(client, job, initial_revision=initial_revision)
    except Exception:
        return None


def report_training_progress(
    progress_reporter: ProgressReporter | None,
    stage: str,
    **fields: object,
) -> bool:
    """Report one observation while keeping all telemetry failures non-fatal."""
    if progress_reporter is None:
        return False
    try:
        return bool(progress_reporter.report(stage, **fields))
    except Exception:
        return False


def simulator_progress_metadata(job: dict) -> dict[str, str]:
    """Return the bounded provider/runtime identity for simulator-backed work."""
    config = job.get("config") if isinstance(job.get("config"), dict) else {}
    provider = _simulator_provider(config.get("provider"))
    task_type = (
        "object_detection" if _is_detection_training_config(config) else "image_classification"
    )
    return {
        "provider": provider,
        "framework": SIMULATOR_FRAMEWORK,
        "execution_mode": SIMULATOR_EXECUTION_MODE,
        "task_type": task_type,
    }


def report_simulator_stage(
    progress_reporter: ProgressReporter | None,
    stage: str,
    *,
    metadata: Mapping[str, Any],
    message: str,
) -> bool:
    """Map one simulator operation into the stable stage taxonomy."""
    detail_code = _SIMULATOR_STAGE_DETAILS.get(stage)
    if not detail_code:
        return False
    return report_training_progress(
        progress_reporter,
        stage,
        detail_code=detail_code,
        message=message,
        metadata=metadata,
    )


def report_simulator_epoch(
    progress_reporter: ProgressReporter | None,
    *,
    current: int,
    total: int,
    metadata: Mapping[str, Any],
) -> bool:
    """Report monotonic simulator epoch progress, including the 0/M boundary."""
    if current <= 0:
        detail_code = "simulator_training"
        message = f"Simulator training is starting for {total} epochs."
    else:
        detail_code = "simulator_epoch_complete"
        message = f"Simulator training epoch {current} of {total} finished."
    return report_training_progress(
        progress_reporter,
        "training",
        current=current,
        total=total,
        unit="epoch",
        detail_code=detail_code,
        message=message,
        metadata=metadata,
    )


def _simulator_provider(value: object) -> str:
    normalized = str(value or "local").strip().lower().replace("-", "_")
    if normalized in {"persistent_gpu", "persistent_disk"}:
        return "persistent_gpu"
    return "local"


def _is_detection_training_config(config: dict) -> bool:
    model = str(config.get("model", "")).lower()
    return (
        str(config.get("task_type", "")).lower() == "object_detection"
        or str(config.get("model_kind", "")).lower() == "ultralytics_yolo_detector"
        or model.startswith("yolo11")
        or model.startswith("yolo")
    )
