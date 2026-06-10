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
    run_modal_training_batch,
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


def test_modal_training_batch_stub_falls_back_to_single_training_with_metadata(monkeypatch):
    submitted = []

    def fake_run_modal_training(_client, job: dict):
        submitted.append(job)

    monkeypatch.setattr(modal_provider, "run_modal_training", fake_run_modal_training)

    client = _FakeClient()
    jobs = [
        {
            "id": "job_1",
            "project_id": "project_1",
            "config": {"dataset_id": "dataset_1", "provider": "modal"},
        },
        {
            "id": "job_2",
            "project_id": "project_1",
            "config": {"dataset_id": "dataset_1", "provider": "modal"},
        },
    ]
    batch = {
        "schema_version": "modal_preview_batch.v1",
        "batch_id": "modal-preview-batch-test",
        "batch_key": "project_1|plan_1|dataset_1|sha256-a|preview|image_classification",
        "project_id": "project_1",
        "plan_id": "plan_1",
        "dataset_id": "dataset_1",
        "dataset_cache_key": "sha256-a",
        "training_tier": "preview",
        "task_type": "image_classification",
    }

    run_modal_training_batch(client, jobs, batch)

    assert [job["id"] for job in submitted] == ["job_1", "job_2"]
    assert submitted[0]["config"]["modal_batch"]["batch_id"] == "modal-preview-batch-test"
    assert submitted[0]["config"]["modal_batch"]["batch_index"] == 0
    assert submitted[1]["config"]["modal_batch"]["batch_index"] == 1
    assert submitted[1]["config"]["modal_batch"]["batch_remote_status"] == "stubbed_single_job_fallback"


def test_modal_training_batch_flag_on_remote_unsupported_falls_back(monkeypatch):
    submitted = []
    remote_payloads = []

    def remote(payload: dict):
        remote_payloads.append(payload)
        return {
            "schema_version": "modal_preview_batch_result.v1",
            "status": "unsupported",
            "runner_status": "remote_batch_shell_unsupported",
            "job_results": [
                {"job_id": "job_1", "status": "unsupported"},
                {"job_id": "job_2", "status": "unsupported"},
            ],
        }

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_modal_preview_batch = types.SimpleNamespace(remote=remote)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER", "1")
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")
    monkeypatch.setattr(modal_provider, "run_modal_training", lambda _client, job: submitted.append(job))

    client = _FakeClient()
    jobs = [
        {
            "id": "job_1",
            "project_id": "project_1",
            "template": "train_experiment",
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "plan_id": "plan_1",
                "training_tier": "preview",
                "dataset_materialization": {"dataset_cache_key": "sha256-a"},
            },
        },
        {
            "id": "job_2",
            "project_id": "project_1",
            "template": "train_experiment",
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "plan_id": "plan_1",
                "training_tier": "preview",
                "dataset_materialization": {"dataset_cache_key": "sha256-a"},
            },
        },
    ]
    batch = {
        "schema_version": "modal_preview_batch.v1",
        "batch_id": "modal-preview-batch-test",
        "batch_key": "project_1|plan_1|dataset_1|sha256-a|preview|image_classification",
        "project_id": "project_1",
        "plan_id": "plan_1",
        "dataset_id": "dataset_1",
        "dataset_cache_key": "sha256-a",
        "training_tier": "preview",
        "task_type": "image_classification",
    }

    run_modal_training_batch(client, jobs, batch)

    assert len(remote_payloads) == 1
    assert remote_payloads[0]["batch"]["batch_id"] == "modal-preview-batch-test"
    assert [job["id"] for job in remote_payloads[0]["jobs"]] == ["job_1", "job_2"]
    assert remote_payloads[0]["dataset"]["id"] == "dataset_1"
    assert [job["id"] for job in submitted] == ["job_1", "job_2"]
    assert submitted[0]["config"]["modal_batch"]["batch_remote_status"] == (
        "remote_batch_shell_unsupported_single_job_fallback"
    )


def test_modal_training_batch_flag_on_remote_failure_falls_back(monkeypatch):
    submitted = []

    def remote(_payload: dict):
        raise RuntimeError("batch shell unavailable")

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_modal_preview_batch = types.SimpleNamespace(remote=remote)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER", "1")
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")
    monkeypatch.setattr(modal_provider, "run_modal_training", lambda _client, job: submitted.append(job))

    client = _FakeClient()
    jobs = [
        {
            "id": "job_1",
            "project_id": "project_1",
            "template": "train_experiment",
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "plan_id": "plan_1",
                "training_tier": "preview",
                "dataset_materialization": {"dataset_cache_key": "sha256-a"},
            },
        },
        {
            "id": "job_2",
            "project_id": "project_1",
            "template": "train_experiment",
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "plan_id": "plan_1",
                "training_tier": "preview",
                "dataset_materialization": {"dataset_cache_key": "sha256-a"},
            },
        },
    ]
    batch = {
        "schema_version": "modal_preview_batch.v1",
        "batch_id": "modal-preview-batch-test",
        "batch_key": "project_1|plan_1|dataset_1|sha256-a|preview|image_classification",
        "project_id": "project_1",
        "plan_id": "plan_1",
        "dataset_id": "dataset_1",
        "dataset_cache_key": "sha256-a",
        "training_tier": "preview",
        "task_type": "image_classification",
    }

    run_modal_training_batch(client, jobs, batch)

    assert [job["id"] for job in submitted] == ["job_1", "job_2"]
    assert submitted[0]["config"]["modal_batch"]["batch_remote_status"] == (
        "remote_batch_failed_before_completion_single_job_fallback"
    )


def test_modal_training_batch_flag_on_remote_completed_does_not_fallback(monkeypatch):
    submitted = []
    remote_payloads = []

    def remote(payload: dict):
        remote_payloads.append(payload)
        return {
            "schema_version": "modal_preview_batch_result.v1",
            "status": "completed",
            "runner_status": "classification_batch_completed",
            "job_results": [
                {"job_id": "job_1", "status": "succeeded"},
                {"job_id": "job_2", "status": "succeeded"},
            ],
        }

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_modal_preview_batch = types.SimpleNamespace(remote=remote)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER", "1")
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")
    monkeypatch.setattr(modal_provider, "run_modal_training", lambda _client, job: submitted.append(job))

    client = _FakeClient()
    jobs = [
        {
            "id": "job_1",
            "project_id": "project_1",
            "template": "train_experiment",
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "plan_id": "plan_1",
                "training_tier": "preview",
                "dataset_materialization": {"dataset_cache_key": "sha256-a"},
            },
        },
        {
            "id": "job_2",
            "project_id": "project_1",
            "template": "train_experiment",
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "plan_id": "plan_1",
                "training_tier": "preview",
                "dataset_materialization": {"dataset_cache_key": "sha256-a"},
            },
        },
    ]
    batch = {
        "schema_version": "modal_preview_batch.v1",
        "batch_id": "modal-preview-batch-test",
        "batch_key": "project_1|plan_1|dataset_1|sha256-a|preview|image_classification",
        "project_id": "project_1",
        "plan_id": "plan_1",
        "dataset_id": "dataset_1",
        "dataset_cache_key": "sha256-a",
        "training_tier": "preview",
        "task_type": "image_classification",
    }

    run_modal_training_batch(client, jobs, batch)

    assert len(remote_payloads) == 1
    assert submitted == []


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
