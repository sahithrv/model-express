from __future__ import annotations

import requests

from worker.orchestrator_client import OrchestratorClient


class _FakeResponse:
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
