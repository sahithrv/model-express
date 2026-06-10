from __future__ import annotations

from contextlib import contextmanager

import pytest

from worker import main as worker_main


def test_provider_filtered_worker_polls_generic_providerless_templates(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def register_worker(self, project_id: str) -> dict:
            return {"id": "worker_1", "project_id": project_id}

        @contextmanager
        def job_context(self, job: dict):
            yield

        def poll_job(self, worker_id: str, *, provider=None, include_unspecified_provider_templates=None, **_kwargs):
            calls.append(
                {
                    "worker_id": worker_id,
                    "provider": provider,
                    "include_unspecified_provider_templates": include_unspecified_provider_templates,
                }
            )
            raise KeyboardInterrupt

    monkeypatch.setenv("GPU_TYPE", "persistent_gpu")
    monkeypatch.setenv("PROJECT_ID", "project_1")
    monkeypatch.setattr(worker_main, "OrchestratorClient", FakeClient)

    with pytest.raises(KeyboardInterrupt):
        worker_main.main()

    assert calls == [
        {
            "worker_id": "worker_1",
            "provider": "persistent_gpu",
            "include_unspecified_provider_templates": worker_main.GENERIC_PROVIDER_FALLBACK_TEMPLATES,
        }
    ]


def test_worker_reports_unexpected_job_failure_as_retryable(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, base_url: str):
            self.base_url = base_url
            self.polled = False

        def register_worker(self, project_id: str) -> dict:
            return {"id": "worker_1", "project_id": project_id}

        @contextmanager
        def job_context(self, job: dict):
            yield

        def poll_job(self, worker_id: str, **_kwargs):
            if self.polled:
                raise KeyboardInterrupt
            self.polled = True
            return {
                "id": "job_1",
                "project_id": "project_1",
                "template": "train_experiment",
                "config": {"active_attempt_id": "job_1_attempt_1"},
            }

        def fail_job(self, job_id: str, error: str, retryable: bool = False, metadata=None, job=None):
            calls.append(
                {
                    "job_id": job_id,
                    "error": error,
                    "retryable": retryable,
                    "metadata": metadata,
                    "job": job,
                }
            )

    def fail_run_job(_client, _job):
        raise RuntimeError("temporary worker failure")

    monkeypatch.setenv("PROJECT_ID", "project_1")
    monkeypatch.setattr(worker_main, "OrchestratorClient", FakeClient)
    monkeypatch.setattr(worker_main, "run_job", fail_run_job)

    with pytest.raises(KeyboardInterrupt):
        worker_main.main()

    assert calls == [
        {
            "job_id": "job_1",
            "error": "temporary worker failure",
            "retryable": True,
            "metadata": {"failure_class": "worker_exception"},
            "job": {
                "id": "job_1",
                "project_id": "project_1",
                "template": "train_experiment",
                "config": {"active_attempt_id": "job_1_attempt_1"},
            },
        }
    ]


def test_worker_idle_log_is_throttled(monkeypatch, capsys):
    poll_count = {"value": 0}
    sleep_count = {"value": 0}

    class FakeClient:
        def __init__(self, base_url: str):
            self.base_url = base_url

        def register_worker(self, project_id: str) -> dict:
            return {"id": "worker_1", "project_id": project_id}

        def poll_job(self, worker_id: str, **_kwargs):
            poll_count["value"] += 1
            return None

    def fake_sleep(seconds: float):
        sleep_count["value"] += 1
        if sleep_count["value"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setenv("PROJECT_ID", "project_1")
    monkeypatch.setenv("MODEL_EXPRESS_WORKER_POLL_INTERVAL_SECONDS", "0.01")
    monkeypatch.setenv("MODEL_EXPRESS_WORKER_IDLE_LOG_SECONDS", "60")
    monkeypatch.setattr(worker_main, "OrchestratorClient", FakeClient)
    monkeypatch.setattr(worker_main.time, "sleep", fake_sleep)

    with pytest.raises(KeyboardInterrupt):
        worker_main.main()

    assert poll_count["value"] == 2
    assert capsys.readouterr().out.count("There are no jobs.") == 1
