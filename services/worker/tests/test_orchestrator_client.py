from __future__ import annotations

import requests
import pytest

from worker.orchestrator_client import OrchestratorClient


@pytest.fixture(autouse=True)
def _clear_api_token(monkeypatch):
    monkeypatch.delenv("MODEL_EXPRESS_API_TOKEN", raising=False)


class _FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"ok": True, "job": None}


def test_complete_job_uses_longer_report_timeout(monkeypatch):
    calls = []

    def fake_post(url: str, *, json: dict | None = None, timeout: int | None = None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setenv("MODEL_EXPRESS_WORKER_REPORT_TIMEOUT_SECONDS", "240")
    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    client.complete_job("job_1", mlflow_run_id="run_1")

    assert calls == [
        {
            "url": "http://orchestrator.test/jobs/job_1/complete",
            "json": {"mlflow_run_id": "run_1"},
            "timeout": 240,
        }
    ]


def test_fail_job_can_report_retryable_failure(monkeypatch):
    calls = []

    def fake_post(url: str, *, json: dict | None = None, timeout: int | None = None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setenv("MODEL_EXPRESS_WORKER_REPORT_TIMEOUT_SECONDS", "240")
    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    client.fail_job("job_1", "modal container exited", retryable=True)

    assert calls == [
        {
            "url": "http://orchestrator.test/jobs/job_1/fail",
            "json": {"error": "modal container exited", "retryable": True},
            "timeout": 240,
        }
    ]


def test_job_context_attaches_training_attempt_id(monkeypatch):
    calls = []

    def fake_post(url: str, *, json: dict | None = None, timeout: int | None = None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    job = {"id": "job_1", "config": {"active_attempt_id": "job_1_attempt_2"}}
    with client.job_context(job):
        client.complete_job("job_1", mlflow_run_id="run_1")
        client.fail_job("job_1", "worker lost connection", retryable=True)

    assert calls[0]["json"] == {
        "mlflow_run_id": "run_1",
        "training_attempt_id": "job_1_attempt_2",
    }
    assert calls[1]["json"] == {
        "error": "worker lost connection",
        "retryable": True,
        "training_attempt_id": "job_1_attempt_2",
    }


def test_job_context_attaches_callback_token_header(monkeypatch):
    calls = []

    def fake_post(
        url: str,
        *,
        json: dict | None = None,
        timeout: int | None = None,
        headers: dict | None = None,
    ):
        calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    job = {
        "id": "job_1",
        "config": {
            "active_attempt_id": "job_1_attempt_2",
            "callback_token": "callback-secret",
        },
    }
    with client.job_context(job):
        client.report_metric("job_1", 1, {"macro_f1": 0.42})
        client.complete_job("job_1", mlflow_run_id="run_1")

    assert calls[0]["headers"] == {"Authorization": "Bearer callback-secret"}
    assert calls[0]["json"]["training_attempt_id"] == "job_1_attempt_2"
    assert calls[1]["headers"] == {"Authorization": "Bearer callback-secret"}
    assert calls[1]["json"]["training_attempt_id"] == "job_1_attempt_2"


def test_callback_and_api_tokens_use_separate_headers(monkeypatch):
    calls = []

    def fake_post(
        url: str,
        *,
        json: dict | None = None,
        timeout: int | None = None,
        headers: dict | None = None,
    ):
        calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        return _FakeResponse()

    monkeypatch.setenv("MODEL_EXPRESS_API_TOKEN", "api-token")
    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    job = {
        "id": "job_1",
        "config": {
            "active_attempt_id": "job_1_attempt_2",
            "callback_token": "callback-secret",
        },
    }
    with client.job_context(job):
        client.complete_job("job_1", mlflow_run_id="run_1")

    assert calls[0]["headers"] == {
        "X-Model-Express-API-Token": "api-token",
        "Authorization": "Bearer callback-secret",
    }
    assert calls[0]["json"]["training_attempt_id"] == "job_1_attempt_2"


def test_missing_callback_token_is_not_invented(monkeypatch):
    calls = []

    def fake_post(url: str, **kwargs):
        calls.append({"url": url, **kwargs})
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    job = {"id": "job_1", "config": {"active_attempt_id": "job_1_attempt_2"}}
    with client.job_context(job):
        client.complete_job("job_1", mlflow_run_id="run_1")

    assert "headers" not in calls[0]
    assert calls[0]["json"]["training_attempt_id"] == "job_1_attempt_2"


def test_job_context_requires_training_attempt_id_for_job_callbacks(monkeypatch):
    calls = []

    def fake_post(url: str, **kwargs):
        calls.append({"url": url, **kwargs})
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    with client.job_context({"id": "job_1", "config": {"callback_token": "callback-secret"}}):
        try:
            client.complete_job("job_1", mlflow_run_id="run_1")
        except ValueError as exc:
            assert "requires training_attempt_id" in str(exc)
        else:
            raise AssertionError("complete_job should reject unbound job callbacks")

    assert calls == []


def test_poll_job_keeps_short_request_timeout(monkeypatch):
    calls = []

    def fake_post(url: str, *, json: dict | None = None, timeout: int | None = None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setenv("MODEL_EXPRESS_WORKER_REQUEST_TIMEOUT_SECONDS", "12")
    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    client.poll_job("worker_1")

    assert calls[0]["url"] == "http://orchestrator.test/workers/worker_1/poll"
    assert calls[0]["timeout"] == 12
    assert "json" not in calls[0] or calls[0]["json"] is None


def test_poll_job_sends_api_token_header_when_configured(monkeypatch):
    calls = []

    def fake_post(
        url: str,
        *,
        json: dict | None = None,
        timeout: int | None = None,
        headers: dict | None = None,
    ):
        calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        return _FakeResponse()

    monkeypatch.setenv("MODEL_EXPRESS_API_TOKEN", "api-token")
    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    client.poll_job("worker_1")

    assert calls[0]["url"] == "http://orchestrator.test/workers/worker_1/poll"
    assert calls[0]["headers"] == {"X-Model-Express-API-Token": "api-token"}


def test_poll_job_can_send_modal_filter(monkeypatch):
    calls = []

    def fake_post(url: str, *, json: dict | None = None, timeout: int | None = None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    client.poll_job(
        "worker_1",
        provider="modal",
        templates=["train_experiment", "profile_dataset"],
        include_unspecified_provider_templates=["profile_dataset"],
    )

    assert calls[0]["url"] == "http://orchestrator.test/workers/worker_1/poll"
    assert calls[0]["json"] == {
        "provider": "modal",
        "templates": ["train_experiment", "profile_dataset"],
        "include_unspecified_provider_templates": ["profile_dataset"],
    }


def test_heartbeat_worker_posts_to_worker_heartbeat(monkeypatch):
    calls = []

    def fake_post(url: str, *, json: dict | None = None, timeout: int | None = None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    client.heartbeat_worker("worker_1")

    assert calls[0]["url"] == "http://orchestrator.test/workers/worker_1/heartbeat"


def test_list_project_worker_requirements_gets_project_requirements(monkeypatch):
    calls = []

    class RequirementsResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"requirements": [{"id": "worker_requirement_1", "target_count": 4}]}

    def fake_get(url: str, *, params: dict | None = None, timeout: int | None = None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return RequirementsResponse()

    monkeypatch.setenv("MODEL_EXPRESS_WORKER_REQUEST_TIMEOUT_SECONDS", "12")
    monkeypatch.setattr(requests, "get", fake_get)

    client = OrchestratorClient("http://orchestrator.test")
    result = client.list_project_worker_requirements("project_1")

    assert result == [{"id": "worker_requirement_1", "target_count": 4}]
    assert calls == [
        {
            "url": "http://orchestrator.test/projects/project_1/worker-requirements",
            "params": None,
            "timeout": 12,
        }
    ]


def test_report_dispatcher_event_posts_project_dispatcher_event(monkeypatch):
    calls = []

    def fake_post(url: str, *, json: dict | None = None, timeout: int | None = None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setenv("MODEL_EXPRESS_WORKER_REPORT_TIMEOUT_SECONDS", "240")
    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    client.report_dispatcher_event(
        "project_1",
        "DISPATCHER_IDLE_EXIT",
        message="idle exit",
        payload={"slot_count": 0, "idle_seconds": 30.0},
    )

    assert calls == [
        {
            "url": "http://orchestrator.test/projects/project_1/dispatcher-events",
            "json": {
                "event_type": "DISPATCHER_IDLE_EXIT",
                "message": "idle exit",
                "plan_id": "",
                "payload": {"slot_count": 0, "idle_seconds": 30.0},
            },
            "timeout": 240,
        }
    ]


def test_import_dataset_metadata_old_backend_fallback(monkeypatch):
    calls = []

    class MissingResponse:
        status_code = 404

        def raise_for_status(self) -> None:
            raise AssertionError("raise_for_status should not be called for old-backend fallback")

        def json(self) -> dict:
            return {"error": "missing"}

    def fake_post(url: str, *, json: dict | None = None, timeout: int | None = None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return MissingResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    client = OrchestratorClient("http://orchestrator.test")
    result = client.import_dataset_metadata("ds_1", {"strict_mode": False, "sources": [], "inventory": {"files": []}})

    assert result["status"] == "unavailable"
    assert result["status_code"] == 404
    assert calls[0]["url"] == "http://orchestrator.test/datasets/ds_1/metadata/imports"


def test_get_dataset_metadata_bundle_uses_training_query(monkeypatch):
    calls = []

    def fake_get(url: str, *, params: dict | None = None, timeout: int | None = None):
        calls.append({"url": url, "params": params, "timeout": timeout})
        return _FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    client = OrchestratorClient("http://orchestrator.test", timeout=17)
    result = client.get_dataset_metadata_bundle("ds_1", metadata_import_id="imp_1", limit=50, offset=100)

    assert result == {"ok": True, "job": None}
    assert calls == [
        {
            "url": "http://orchestrator.test/datasets/ds_1/metadata/bundle",
            "params": {
                "purpose": "training",
                "include": "bbox",
                "limit": 50,
                "offset": 100,
                "metadata_import_id": "imp_1",
            },
            "timeout": 17,
        }
    ]
