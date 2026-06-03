from __future__ import annotations

import sys
import types

import pytest

from worker.training import modal_provider
from worker.training.modal_provider import (
    ModalRetryableFailureReported,
    modal_app_session,
    run_modal_dataset_materialization,
    run_modal_dataset_profile,
    run_modal_training,
)


class _FakeModalApp:
    def __init__(self, events: list[str] | None = None):
        self.enter_count = 0
        self.exit_count = 0
        self.events = events

    def run(self):
        return self

    def __enter__(self):
        self.enter_count += 1
        if self.events is not None:
            self.events.append("app_enter")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.exit_count += 1
        if self.events is not None:
            self.events.append("app_exit")
        return False


class _FakeModalOutput:
    def __init__(self, events: list[str]):
        self.events = events

    def __enter__(self):
        self.events.append("output_enter")

    def __exit__(self, exc_type, exc, traceback):
        self.events.append("output_exit")
        return False


class _FakeStream:
    encoding = "cp1252"
    errors = "surrogateescape"

    def __init__(self):
        self.reconfigure_calls = []

    def reconfigure(self, **kwargs):
        self.reconfigure_calls.append(kwargs)
        if "errors" in kwargs:
            self.errors = kwargs["errors"]


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


def test_modal_training_reuses_existing_modal_app_session(monkeypatch):
    remote_calls = []

    def remote(payload: dict):
        remote_calls.append(payload["job"]["id"])
        return {}

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_image_classifier = types.SimpleNamespace(remote=remote)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")

    client = _FakeClient()
    with modal_app_session():
        for job_id in ["job_1", "job_2"]:
            run_modal_training(
                client,
                {
                    "id": job_id,
                    "project_id": "project_1",
                    "config": {"dataset_id": "dataset_1", "provider": "modal"},
                },
            )

    assert fake_modal_app.app.enter_count == 1
    assert fake_modal_app.app.exit_count == 1
    assert remote_calls == ["job_1", "job_2"]
    assert client.failures == []


def test_modal_training_dispatches_detection_jobs_to_yolo_remote(monkeypatch):
    remote_calls = []

    def classifier_remote(_payload: dict):
        raise AssertionError("classifier remote should not handle detection jobs")

    def yolo_remote(payload: dict):
        remote_calls.append(payload["job"]["config"]["model"])
        return {}

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_image_classifier = types.SimpleNamespace(remote=classifier_remote)
    fake_modal_app.train_yolo_detector = types.SimpleNamespace(remote=yolo_remote)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")

    client = _FakeClient()
    run_modal_training(
        client,
        {
            "id": "job_yolo",
            "project_id": "project_1",
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "model": "yolo11n.pt",
                "task_type": "object_detection",
                "model_kind": "ultralytics_yolo_detector",
            },
        },
    )

    assert remote_calls == ["yolo11n.pt"]
    assert client.failures == []


def test_modal_app_session_enables_modal_output(monkeypatch):
    events = []

    fake_modal = types.ModuleType("modal")
    fake_modal.enable_output = lambda: _FakeModalOutput(events)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp(events)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)

    with modal_app_session():
        pass

    assert events == ["output_enter", "app_enter", "app_exit", "output_exit"]


def test_modal_output_context_makes_cp1252_streams_tolerant(monkeypatch):
    stdout = _FakeStream()
    stderr = _FakeStream()
    events = []

    fake_modal = types.ModuleType("modal")
    fake_modal.enable_output = lambda: _FakeModalOutput(events)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(modal_provider.sys, "stdout", stdout)
    monkeypatch.setattr(modal_provider.sys, "stderr", stderr)

    with modal_provider._modal_output_context():
        pass

    assert stdout.reconfigure_calls == [{"errors": "replace"}]
    assert stderr.reconfigure_calls == [{"errors": "replace"}]
    assert events == ["output_enter", "output_exit"]


def test_modal_dataset_profile_invocation_error_reports_retryable_failure(monkeypatch):
    def remote(_payload: dict):
        raise TimeoutError("profile timed out")

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.profile_image_dataset = types.SimpleNamespace(remote=remote)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")

    client = _FakeClient()
    run_modal_dataset_profile(
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
    assert (
        "Modal dataset profile invocation failed before completion"
        in client.failures[0]["error"]
    )
    assert "profile timed out" in client.failures[0]["error"]


def test_modal_dataset_materialization_invocation_error_reports_retryable_failure(monkeypatch):
    def remote(_payload: dict):
        raise TimeoutError("materialization timed out")

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.materialize_image_dataset = types.SimpleNamespace(remote=remote)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")

    client = _FakeClient()
    with pytest.raises(ModalRetryableFailureReported):
        run_modal_dataset_materialization(
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
    assert (
        "Modal dataset materialization invocation failed before completion"
        in client.failures[0]["error"]
    )
    assert "materialization timed out" in client.failures[0]["error"]
