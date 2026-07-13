from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from worker.progress import PROGRESS_TAXONOMY_VERSION, WORKER_PROGRESS_STAGES
from worker.training import persistent_gpu_provider
from worker.training.progress_reporting import (
    report_simulator_stage,
    simulator_progress_metadata,
)


class _FakeClient:
    def __init__(self, *, progress_error: BaseException | None = None) -> None:
        self.progress: list[dict] = []
        self.summaries: list[dict] = []
        self.progress_error = progress_error

    def get_dataset(self, dataset_id: str) -> dict:
        return {
            "id": dataset_id,
            "storage_uri": "s3://private-bucket/dataset.zip",
            "checksum_sha256": "a" * 64,
        }

    def report_progress(
        self,
        job_id: str,
        payload: dict,
        *,
        job: dict,
        timeout: float,
    ) -> dict:
        if self.progress_error is not None:
            raise self.progress_error
        self.progress.append(
            {
                "job_id": job_id,
                "payload": copy.deepcopy(payload),
                "job": copy.deepcopy(job),
                "timeout": timeout,
            }
        )
        return {"updated": True}

    def report_training_run_summary(self, job_id: str, summary: dict) -> dict:
        self.summaries.append({"job_id": job_id, "summary": copy.deepcopy(summary)})
        return {"ok": True}


def _job(provider: str = "persistent_gpu") -> dict:
    return {
        "id": "job_persistent",
        "project_id": "project_1",
        "config": {
            "provider": provider,
            "dataset_id": "dataset_1",
            "model": "resnet18",
            "epochs": 2,
            "active_attempt_id": "job_persistent:attempt-1",
            "callback_token": "callback-secret",
        },
    }


@pytest.mark.parametrize("provider", ["persistent_gpu", "persistent_disk"])
def test_persistent_provider_hands_one_reporter_into_simulator_with_stable_taxonomy(
    monkeypatch,
    tmp_path: Path,
    provider: str,
):
    cache_root = tmp_path / "private-persistent-cache"
    dataset_dir = cache_root / "dataset" / "extracted"
    monkeypatch.setenv("MODEL_EXPRESS_PERSISTENT_GPU_PROVIDER", "1")
    monkeypatch.setenv("MODEL_EXPRESS_PERSISTENT_GPU_CACHE_ROOT", str(cache_root))
    monkeypatch.setattr(
        persistent_gpu_provider,
        "ensure_dataset_materialized",
        lambda **_kwargs: SimpleNamespace(
            dataset_dir=dataset_dir,
            telemetry={
                "dataset_materialization_cache_hit": True,
                "dataset_materialization_status": "hit",
                "dataset_materialization_total_seconds": 0.01,
            },
        ),
    )
    delegated: dict = {}

    def fake_local_training(client, job, *, progress_reporter=None):
        delegated["client"] = client
        delegated["job"] = job
        delegated["progress_reporter"] = progress_reporter
        report_simulator_stage(
            progress_reporter,
            "model_initializing",
            metadata=simulator_progress_metadata(job),
            message="The deterministic classifier simulation is being initialized.",
        )

    monkeypatch.setattr(persistent_gpu_provider, "run_local_training", fake_local_training)
    client = _FakeClient()

    persistent_gpu_provider.run_persistent_gpu_training(client, _job(provider))

    payloads = [entry["payload"] for entry in client.progress]
    assert [payload["stage"] for payload in payloads] == [
        "worker_starting",
        "dataset_materializing",
        "dataset_materializing",
        "model_initializing",
    ]
    assert [payload["revision"] for payload in payloads] == [3, 4, 5, 6]
    assert all(payload["taxonomy_version"] == PROGRESS_TAXONOMY_VERSION for payload in payloads)
    assert all(payload["stage"] in WORKER_PROGRESS_STAGES for payload in payloads)
    assert all(payload["status"] == "running" for payload in payloads)
    assert all(payload["metadata"]["provider"] == "persistent_gpu" for payload in payloads)
    assert all(
        payload["metadata"]["execution_mode"] == "local_simulator" for payload in payloads
    )
    assert payloads[2]["metadata"]["cache_status"] == "hit"
    assert all(len(payload.get("detail_code", "").encode("utf-8")) <= 64 for payload in payloads)
    assert not ({"completed", "failed", "cancelled"} & {payload["stage"] for payload in payloads})
    assert str(cache_root) not in repr(client.progress)
    assert "s3://private-bucket" not in repr(client.progress)

    assert delegated["client"] is client
    assert delegated["progress_reporter"] is not None
    assert delegated["progress_reporter"].revision == 6
    assert delegated["job"]["config"]["provider"] == "persistent_gpu"
    assert delegated["job"]["config"]["dataset_dir"] == str(dataset_dir)
    assert delegated["job"]["config"]["stage_telemetry"]["schema_version"] == (
        "remote_gpu_stage_telemetry_v1"
    )
    assert client.summaries[0]["summary"]["dataset_materialization"][
        "dataset_materialization_cache_hit"
    ] is True


def test_persistent_provider_progress_outage_does_not_block_materialization_or_training(
    monkeypatch,
    tmp_path: Path,
):
    cache_root = tmp_path / "persistent-cache"
    monkeypatch.setenv("MODEL_EXPRESS_PERSISTENT_GPU_PROVIDER", "1")
    monkeypatch.setenv("MODEL_EXPRESS_PERSISTENT_GPU_CACHE_ROOT", str(cache_root))
    monkeypatch.setenv("MODEL_EXPRESS_PROGRESS_REPORT_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(
        persistent_gpu_provider,
        "ensure_dataset_materialized",
        lambda **_kwargs: SimpleNamespace(
            dataset_dir=cache_root / "extracted",
            telemetry={"dataset_materialization_cache_hit": False},
        ),
    )
    delegated = []

    def fake_local_training(_client, job, *, progress_reporter=None):
        delegated.append(job["id"])
        report_simulator_stage(
            progress_reporter,
            "model_initializing",
            metadata=simulator_progress_metadata(job),
            message="The deterministic classifier simulation is being initialized.",
        )

    monkeypatch.setattr(persistent_gpu_provider, "run_local_training", fake_local_training)
    client = _FakeClient(progress_error=requests.ConnectionError("progress unavailable"))

    persistent_gpu_provider.run_persistent_gpu_training(client, _job())

    assert client.progress == []
    assert delegated == ["job_persistent"]
    assert len(client.summaries) == 1
