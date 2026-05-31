from __future__ import annotations

import requests

from worker.orchestrator_client import OrchestratorClient


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
