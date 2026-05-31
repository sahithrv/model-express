from __future__ import annotations

import sys
import types

from worker.training.modal_provider import run_modal_training


class _FakeModalApp:
    def run(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeClient:
    base_url = "https://orchestrator.test"

    def __init__(self):
        self.failures = []

    def get_dataset(self, dataset_id: str) -> dict:
        return {"id": dataset_id, "storage_uri": "s3://bucket/dataset.zip"}

    def fail_job(self, job_id: str, error: str, retryable: bool = False) -> dict:
        self.failures.append({"job_id": job_id, "error": error, "retryable": retryable})
        return {"ok": True}


def test_modal_training_invocation_error_reports_retryable_failure(monkeypatch):
    def remote(_payload: dict):
        raise RuntimeError("container exited unexpectedly")

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_image_classifier = types.SimpleNamespace(remote=remote)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")

    client = _FakeClient()
    run_modal_training(
        client,
        {
            "id": "job_1",
            "project_id": "project_1",
            "config": {"dataset_id": "dataset_1", "provider": "modal"},
        },
    )

    assert len(client.failures) == 1
    assert client.failures[0]["job_id"] == "job_1"
    assert client.failures[0]["retryable"] is True
    assert "container exited unexpectedly" in client.failures[0]["error"]


def test_modal_training_remote_reported_failure_does_not_report_twice(monkeypatch):
    def remote(_payload: dict):
        return {"status": "retryable_failure_reported", "error": "already reported"}

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_image_classifier = types.SimpleNamespace(remote=remote)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")

    client = _FakeClient()
    run_modal_training(
        client,
        {
            "id": "job_1",
            "project_id": "project_1",
            "config": {"dataset_id": "dataset_1", "provider": "modal"},
        },
    )

    assert client.failures == []
