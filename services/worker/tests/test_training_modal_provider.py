from __future__ import annotations

import sys
import types
import os

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
from worker.training.modal_resources import classify_oom_failure


@pytest.fixture(autouse=True)
def _modal_storage_credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "scoped-test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "scoped-test-secret")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080",
        "https://localhost:8080",
        "https://0.0.0.0:8080",
        "https://10.0.0.5:8080",
        "https://169.254.10.1:8080",
        "http://orchestrator.example.test",
    ],
)
def test_modal_remote_urls_reject_local_private_or_insecure_targets(monkeypatch, url):
    monkeypatch.delenv("MODEL_EXPRESS_ALLOW_INSECURE_MODAL_URLS", raising=False)

    with pytest.raises(ValueError):
        modal_provider._require_remote_reachable_url("MODAL_ORCHESTRATOR_URL", url)


def test_modal_remote_url_allows_explicit_insecure_dev_override(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_ALLOW_INSECURE_MODAL_URLS", "true")

    modal_provider._require_remote_reachable_url("MODAL_ORCHESTRATOR_URL", "http://orchestrator.example.test")


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


class _FakeModalFunction:
    def __init__(self, remote):
        self.remote = remote
        self.options_calls = []

    def with_options(self, **kwargs):
        self.options_calls.append(kwargs)
        return self


class _FakeFunctionCall:
    def __init__(self, *, object_id: str = "fc-1", result: dict | None = None, errors: list[Exception] | None = None):
        self.object_id = object_id
        self.result = result or {}
        self.errors = list(errors or [])
        self.get_timeouts = []
        self.cancel_calls = []

    def get(self, *, timeout=None):
        self.get_timeouts.append(timeout)
        if self.errors:
            raise self.errors.pop(0)
        return self.result

    def cancel(self, *, terminate_containers: bool = False):
        self.cancel_calls.append({"terminate_containers": terminate_containers})


class _FakeSpawnModalFunction(_FakeModalFunction):
    def __init__(self, call: _FakeFunctionCall):
        def remote(_payload: dict):
            raise AssertionError("remote should not be used when spawn is available")

        super().__init__(remote)
        self.call = call
        self.spawn_payloads = []

    def spawn(self, payload: dict):
        self.spawn_payloads.append(payload)
        return self.call


class _FakeClient:
    base_url = "https://orchestrator.test"

    def __init__(self):
        self.failures = []
        self.modal_calls = []
        self.job_reads = []
        self.jobs_by_id = {}

    def get_dataset(self, dataset_id: str) -> dict:
        return {"id": dataset_id, "storage_uri": "s3://bucket/dataset.zip"}

    def get_job(self, job_id: str) -> dict:
        self.job_reads.append(job_id)
        return self.jobs_by_id.get(job_id, {"id": job_id, "status": "RUNNING", "config": {}})

    def report_modal_call(self, job_id: str, payload: dict) -> dict:
        self.modal_calls.append({"job_id": job_id, "payload": payload})
        return {"job_id": job_id}

    def fail_job(
        self,
        job_id: str,
        error: str,
        retryable: bool = False,
        metadata: dict | None = None,
        job: dict | None = None,
    ) -> dict:
        payload = {"job_id": job_id, "error": error, "retryable": retryable}
        if isinstance(metadata, dict):
            payload.update(metadata)
            payload["job_id"] = job_id
            payload["error"] = error
            payload["retryable"] = retryable
        self.failures.append(payload)
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


def test_modal_training_uses_with_options_for_per_job_gpu_and_memory(monkeypatch):
    remote_payloads = []

    def remote(payload: dict):
        remote_payloads.append(payload)
        return {}

    fake_function = _FakeModalFunction(remote)
    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_image_classifier = fake_function
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")
    monkeypatch.setenv("MODAL_GPU_TYPE", "T4")

    client = _FakeClient()
    run_modal_training(
        client,
        {
            "id": "job_1",
            "project_id": "project_1",
            "attempt": 2,
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "gpu_type": "A10",
                "batch_size": 12,
                "memory_mb": 49152,
                "active_attempt_id": "job_1:attempt-2",
                "callback_token": "callback-secret",
                "remote_training_session": {
                    "id": "session_1",
                    "orchestrator_public_url": "https://orchestrator.test",
                },
            },
        },
    )

    assert fake_function.options_calls == [{"gpu": "A10", "memory": 49152}]
    assert remote_payloads[0]["modal_resources"]["effective_gpu_type"] == "A10"
    assert remote_payloads[0]["modal_resources"]["effective_batch_size"] == 12
    assert remote_payloads[0]["job"]["config"]["modal_resources"]["memory_mb"] == 49152
    assert remote_payloads[0]["job"]["config"]["active_attempt_id"] == "job_1:attempt-2"
    assert remote_payloads[0]["training_attempt_id"] == "job_1:attempt-2"
    assert remote_payloads[0]["callback_token"] == "callback-secret"
    assert remote_payloads[0]["remote_training_session"]["id"] == "session_1"
    assert remote_payloads[0]["storage_scope"]["read_keys"] == ["dataset.zip"]
    assert remote_payloads[0]["storage_scope"]["write_prefixes"] == ["model-express/artifacts/job_1/"]
    assert remote_payloads[0]["aws_access_key_id"] == "scoped-test-key"
    assert client.failures == []
    assert sys.modules["worker.training.modal_app"].train_image_classifier.options_calls[0]["gpu"] == "A10"
    assert os.environ["MODAL_GPU_TYPE"] == "T4"


def test_modal_storage_rejects_default_root_credentials(monkeypatch):
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "model_express")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "model_express_password")

    with pytest.raises(ValueError, match="default local MinIO root credentials"):
        modal_provider._modal_storage_payload(
            {"id": "job_1", "config": {}},
            {"id": "dataset_1", "storage_uri": "s3://bucket/dataset.zip"},
        )


def test_modal_storage_payload_rejects_dataset_key_traversal(monkeypatch):
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")

    with pytest.raises(ValueError, match="dataset storage_uri"):
        modal_provider._modal_storage_payload(
            {"id": "job_1", "config": {}},
            {"id": "dataset_1", "storage_uri": "s3://bucket/datasets/%2e%2e/secrets.zip"},
        )


def test_modal_storage_payload_rejects_session_prefix_traversal(monkeypatch):
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")

    with pytest.raises(ValueError, match="path traversal"):
        modal_provider._modal_storage_payload(
            {"id": "job_1", "config": {"remote_training_session": {"storage_prefix": "model-express/artifacts/job_1/../job_2"}}},
            {"id": "dataset_1", "storage_uri": "s3://bucket/dataset.zip"},
        )


def test_modal_training_spawn_reports_function_call_object_id(monkeypatch):
    call = _FakeFunctionCall(object_id="fc-spawn", result={"status": "succeeded"})
    fake_function = _FakeSpawnModalFunction(call)
    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_image_classifier = fake_function
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_USE_SPAWN", "1")
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_CALL_POLL_SECONDS", "0.25")

    client = _FakeClient()
    run_modal_training(
        client,
        {
            "id": "job_spawn",
            "project_id": "project_1",
            "attempt": 1,
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "gpu_type": "L4",
                "batch_size": 8,
                "memory_mb": 32768,
                "active_attempt_id": "job_spawn:attempt-1",
            },
        },
    )

    assert fake_function.options_calls == [{"gpu": "L4", "memory": 32768}]
    assert len(fake_function.spawn_payloads) == 1
    assert call.get_timeouts == [0.25]
    assert call.cancel_calls == []
    assert client.failures == []
    assert len(client.modal_calls) == 1
    modal_call = client.modal_calls[0]
    assert modal_call["job_id"] == "job_spawn"
    assert modal_call["payload"]["training_attempt_id"] == "job_spawn:attempt-1"
    assert modal_call["payload"]["modal_function_call_object_id"] == "fc-spawn"
    assert modal_call["payload"]["cancel_status"] == "active"
    assert modal_call["payload"]["requested_gpu_type"] == "L4"
    assert modal_call["payload"]["effective_gpu_type"] == "L4"
    assert modal_call["payload"]["memory_mb"] == 32768
    assert modal_call["payload"]["modal_resource_signature"] == "gpu=L4|batch=8|memory_mb=32768"
    assert modal_call["payload"]["modal_resources"]["modal_function_options"] == {"gpu": "L4", "memory": 32768}
    assert modal_call["payload"]["modal_resources"]["modal_call_cancel_status"] == "active"


def test_modal_training_spawn_cancels_when_backend_requests_remote_cancel(monkeypatch):
    call = _FakeFunctionCall(object_id="fc-cancel", errors=[TimeoutError("timed out")])
    fake_function = _FakeSpawnModalFunction(call)
    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_image_classifier = fake_function
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_USE_SPAWN", "1")
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_CALL_POLL_SECONDS", "0.25")
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_CANCEL_TERMINATE_CONTAINERS", "1")

    client = _FakeClient()
    client.jobs_by_id["job_cancel"] = {
        "id": "job_cancel",
        "status": "FAILED",
        "config": {
            "active_attempt_id": "job_cancel:terminal-after-attempt-1",
            "cancel_requested": True,
            "terminate_remote_work": True,
        },
    }
    run_modal_training(
        client,
        {
            "id": "job_cancel",
            "project_id": "project_1",
            "attempt": 1,
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "gpu_type": "T4",
                "batch_size": 16,
                "memory_mb": 24576,
                "active_attempt_id": "job_cancel:attempt-1",
            },
        },
    )

    assert fake_function.spawn_payloads
    assert call.cancel_calls == [{"terminate_containers": True}]
    assert client.job_reads == ["job_cancel"]
    assert client.failures == []
    assert [entry["payload"]["cancel_status"] for entry in client.modal_calls] == ["active", "cancel_requested"]
    assert all(entry["payload"]["modal_function_call_object_id"] == "fc-cancel" for entry in client.modal_calls)


def test_modal_training_spawn_does_not_cancel_when_backend_already_succeeded(monkeypatch):
    call = _FakeFunctionCall(object_id="fc-complete", errors=[TimeoutError("timed out")])
    fake_function = _FakeSpawnModalFunction(call)
    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_image_classifier = fake_function
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_USE_SPAWN", "1")
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_CALL_POLL_SECONDS", "0.25")
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_CANCEL_TERMINATE_CONTAINERS", "1")

    client = _FakeClient()
    client.jobs_by_id["job_complete"] = {
        "id": "job_complete",
        "status": "SUCCEEDED",
        "config": {"active_attempt_id": "job_complete:terminal-after-attempt-1"},
    }
    run_modal_training(
        client,
        {
            "id": "job_complete",
            "project_id": "project_1",
            "attempt": 1,
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "gpu_type": "T4",
                "batch_size": 16,
                "memory_mb": 24576,
                "active_attempt_id": "job_complete:attempt-1",
            },
        },
    )

    assert fake_function.spawn_payloads
    assert call.cancel_calls == []
    assert client.job_reads == ["job_complete"]
    assert client.failures == []
    assert len(client.modal_calls) == 1
    assert client.modal_calls[0]["payload"]["cancel_status"] == "active"
    assert client.modal_calls[0]["payload"]["modal_function_call_object_id"] == "fc-complete"


def test_modal_training_invocation_oom_failure_reports_resource_metadata(monkeypatch):
    def remote(_payload: dict):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    fake_modal_app = types.ModuleType("worker.training.modal_app")
    fake_modal_app.app = _FakeModalApp()
    fake_modal_app.train_image_classifier = _FakeModalFunction(remote)
    monkeypatch.setitem(sys.modules, "worker.training.modal_app", fake_modal_app)
    monkeypatch.setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.test")
    monkeypatch.setenv("MODAL_S3_ENDPOINT_URL", "https://s3.test")

    client = _FakeClient()
    run_modal_training(
        client,
        {
            "id": "job_oom",
            "project_id": "project_1",
            "attempt": 1,
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "gpu_type": "T4",
                "batch_size": 16,
                "active_attempt_id": "job_oom:attempt-1",
            },
        },
    )

    assert len(client.failures) == 1
    failure = client.failures[0]
    assert failure["retryable"] is True
    assert failure["training_attempt_id"] == "job_oom:attempt-1"
    assert failure["failure_class"] == "oom"
    assert failure["oom"] is True
    assert failure["oom_kind"] == "gpu_cuda"
    assert failure["effective_gpu_type"] == "T4"
    assert failure["effective_batch_size"] == 16
    assert failure["modal_resource_signature"].startswith("gpu=T4|batch=16|memory_mb=")


def test_modal_oom_classifier_distinguishes_memory_failures():
    assert classify_oom_failure("CUDA out of memory") == {
        "is_oom": True,
        "failure_class": "oom",
        "oom_kind": "gpu_cuda",
        "retryable": True,
    }
    assert classify_oom_failure("container exited with exit code 137")["oom_kind"] == "system_memory"
    assert classify_oom_failure("training room validation failed")["is_oom"] is False


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
                "active_attempt_id": "job_1:attempt-1",
                "callback_token": "callback-secret-1",
                "remote_training_session": {"id": "session_1"},
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
                "active_attempt_id": "job_2:attempt-1",
                "callback_token": "callback-secret-2",
                "remote_training_session": {"id": "session_2"},
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
    assert remote_payloads[0]["job_callback_metadata"]["job_1"]["training_attempt_id"] == "job_1:attempt-1"
    assert remote_payloads[0]["job_callback_metadata"]["job_1"]["callback_token"] == "callback-secret-1"
    assert remote_payloads[0]["job_callback_metadata"]["job_1"]["remote_training_session"]["id"] == "session_1"
    assert remote_payloads[0]["job_callback_metadata"]["job_2"]["callback_token"] == "callback-secret-2"
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


def test_modal_training_batch_flag_on_remote_completed_with_failed_child_falls_back(monkeypatch):
    submitted = []

    def remote(_payload: dict):
        return {
            "schema_version": "modal_preview_batch_result.v1",
            "status": "completed",
            "runner_status": "classification_batch_completed",
            "job_results": [
                {"job_id": "job_1", "status": "failed", "error": "child failed before durable report"},
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

    assert [job["id"] for job in submitted] == ["job_1", "job_2"]
    assert submitted[0]["config"]["modal_batch"]["batch_remote_status"] == (
        "classification_batch_completed_single_job_fallback"
    )


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
    remote_payloads = []

    def remote(payload: dict):
        remote_payloads.append(payload)
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
            "config": {
                "dataset_id": "dataset_1",
                "provider": "modal",
                "active_attempt_id": "job_1:attempt-1",
                "callback_token": "callback-secret",
                "remote_training_session": {"id": "session_1"},
            },
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
    assert remote_payloads[0]["training_attempt_id"] == "job_1:attempt-1"
    assert remote_payloads[0]["callback_token"] == "callback-secret"
    assert remote_payloads[0]["remote_training_session"]["id"] == "session_1"


def test_modal_dataset_materialization_invocation_error_reports_retryable_failure(monkeypatch):
    remote_payloads = []

    def remote(payload: dict):
        remote_payloads.append(payload)
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
                "config": {
                    "dataset_id": "dataset_1",
                    "provider": "modal",
                    "active_attempt_id": "job_1:attempt-1",
                    "callback_token": "callback-secret",
                    "remote_training_session": {"id": "session_1"},
                },
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
    assert remote_payloads[0]["training_attempt_id"] == "job_1:attempt-1"
    assert remote_payloads[0]["callback_token"] == "callback-secret"
    assert remote_payloads[0]["remote_training_session"]["id"] == "session_1"
