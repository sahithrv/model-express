from __future__ import annotations

import os
from urllib.parse import urlparse

from worker.diagnostics import log_event
from worker.orchestrator_client import OrchestratorClient


def run_modal_training(client: OrchestratorClient, job: dict) -> None:
    config = job["config"]
    if config.get("gpu_type"):
        os.environ["MODAL_GPU_TYPE"] = str(config["gpu_type"])

    try:
        from worker.training.modal_app import app, train_image_classifier
    except ModuleNotFoundError as exc:
        if exc.name == "modal":
            raise RuntimeError(
                "Modal is not installed. Install worker dependencies, then run `modal setup`."
            ) from exc
        raise

    dataset = client.get_dataset(str(config["dataset_id"]))

    orchestrator_url = os.getenv("MODAL_ORCHESTRATOR_URL", client.base_url)
    s3_endpoint_url = os.getenv("MODAL_S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))

    _require_remote_reachable_url("MODAL_ORCHESTRATOR_URL", orchestrator_url)
    _require_remote_reachable_url("MODAL_S3_ENDPOINT_URL", s3_endpoint_url)

    payload = {
        "job": job,
        "dataset": dataset,
        "orchestrator_url": orchestrator_url,
        "s3_endpoint_url": s3_endpoint_url,
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "model_express"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "model_express_password"),
        "aws_default_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }

    print(
        "Submitting Modal training job "
        f"{job['id']} model={config.get('model')} gpu={config.get('gpu_type') or os.getenv('MODAL_GPU_TYPE', 'T4')}"
    )

    with app.run():
        result = _remote_function(train_image_classifier)(payload)

    print(f"Modal training finished for {job['id']}: {result}")


def run_modal_dataset_profile(client: OrchestratorClient, job: dict) -> None:
    config = job["config"]
    dataset_id = str(config["dataset_id"])

    try:
        from worker.training.modal_app import app, profile_image_dataset
    except ModuleNotFoundError as exc:
        if exc.name == "modal":
            raise RuntimeError(
                "Modal is not installed. Install worker dependencies, then run `modal setup`."
            ) from exc
        raise

    dataset = client.get_dataset(dataset_id)
    s3_endpoint_url = os.getenv("MODAL_S3_ENDPOINT_URL", os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"))
    _require_remote_reachable_url("MODAL_S3_ENDPOINT_URL", s3_endpoint_url)

    payload = {
        "dataset": dataset,
        "s3_endpoint_url": s3_endpoint_url,
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "model_express"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", "model_express_password"),
        "aws_default_region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }

    print(f"Submitting Modal dataset profile job dataset={dataset_id}")
    with app.run():
        result = _remote_function(profile_image_dataset)(payload)

    profile = result.get("profile") if isinstance(result, dict) and isinstance(result.get("profile"), dict) else result
    if not isinstance(profile, dict):
        raise RuntimeError("Modal dataset profiler returned an invalid profile payload.")

    metadata_import = result.get("metadata_import") if isinstance(result, dict) else None
    if isinstance(metadata_import, dict):
        _send_dataset_metadata_import(client, dataset_id, metadata_import)
    client.update_dataset_profile(dataset_id, profile)
    client.complete_job(job["id"], mlflow_run_id="")
    print(f"Modal dataset profile finished for {dataset_id}")


def _remote_function(function):
    remote = getattr(function, "remote", None)
    if remote is None:
        raise RuntimeError(
            "Modal is not installed or the Modal function was not registered. "
            "Install worker dependencies, then run `modal setup`."
        )
    return remote


def _require_remote_reachable_url(name: str, value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be a full http(s) URL, got {value!r}")

    host = parsed.hostname or ""
    if host in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(
            f"{name} points at {value!r}, but Modal cannot reach your local localhost. "
            "Expose it with a tunnel and set the public URL before running Modal jobs."
        )


def _send_dataset_metadata_import(client: OrchestratorClient, dataset_id: str, payload: dict) -> None:
    try:
        result = client.import_dataset_metadata(dataset_id, payload)
        if isinstance(result, dict) and result.get("status") == "unavailable":
            log_event(
                "warn",
                "dataset_metadata_import_unavailable",
                dataset_id=dataset_id,
                reason=result.get("reason"),
                status_code=result.get("status_code"),
            )
    except Exception as exc:
        log_event(
            "warn",
            "dataset_metadata_import_failed_nonfatal",
            dataset_id=dataset_id,
            error=str(exc),
        )
