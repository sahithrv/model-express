from __future__ import annotations

import copy

import pytest
import requests

import worker.progress as progress_module
from worker.progress import MAX_DIAGNOSTICS
from worker.progress import MAX_PROGRESS_REPORT_ATTEMPTS
from worker.progress import MAX_PROGRESS_TIMEOUT_SECONDS
from worker.progress import ProgressReporter


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class _FakeClient:
    def __init__(self, effects: list[object] | None = None) -> None:
        self.effects = list(effects or [])
        self.calls: list[dict] = []

    def report_progress(
        self,
        job_id: str,
        payload: dict,
        *,
        job: dict,
        timeout: float,
    ) -> dict:
        self.calls.append(
            {
                "job_id": job_id,
                "payload": copy.deepcopy(payload),
                "job": copy.deepcopy(job),
                "timeout": timeout,
            }
        )
        effect = self.effects.pop(0) if self.effects else {"updated": True}
        if isinstance(effect, BaseException):
            raise effect
        return effect


@pytest.fixture(autouse=True)
def _isolate_progress_diagnostics(monkeypatch):
    monkeypatch.setattr(progress_module, "log_event", lambda *args, **kwargs: None)
    monkeypatch.delenv("MODEL_EXPRESS_PROGRESS_REPORTING_ENABLED", raising=False)
    monkeypatch.delenv("MODEL_EXPRESS_PROGRESS_HEARTBEAT_SECONDS", raising=False)
    monkeypatch.delenv("MODEL_EXPRESS_PROGRESS_REPORT_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("MODEL_EXPRESS_PROGRESS_REPORT_MAX_ATTEMPTS", raising=False)


def _job() -> dict:
    return {
        "id": "job_1",
        "project_id": "project_1",
        "config": {
            "active_attempt_id": "job_1:attempt-2",
            "callback_token": "callback-secret",
            "dataset_path": "/private/training-data",
            "prompt": "do not retain this",
        },
    }


def _http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(f"HTTP {status_code}", response=response)


def test_reporter_preserves_attempt_identity_and_monotonic_revisions():
    client = _FakeClient()
    reporter = ProgressReporter(client, _job(), initial_revision=2)

    assert reporter.report("remote_scheduled", message="Remote training was scheduled.")
    assert reporter.report(
        "training",
        current=1,
        total=3,
        unit="epochs",
        message="Epoch 1 of 3.",
        detail_code="epoch",
        metadata={"provider": "modal"},
    )

    assert [call["payload"]["revision"] for call in client.calls] == [3, 4]
    assert reporter.revision == 4
    assert client.calls[0]["job"] == {
        "id": "job_1",
        "config": {
            "active_attempt_id": "job_1:attempt-2",
            "callback_token": "callback-secret",
        },
    }
    assert all("training_attempt_id" not in call["payload"] for call in client.calls)
    assert "/private/training-data" not in repr(client.calls)
    assert "do not retain this" not in repr(client.calls)


def test_same_observation_is_throttled_but_boundaries_and_transitions_bypass():
    client = _FakeClient()
    clock = _Clock()
    reporter = ProgressReporter(
        client,
        _job(),
        heartbeat_interval_seconds=30,
        clock=clock,
    )

    assert reporter.report(
        "training", current=1, total=3, unit="epochs", detail_code="epoch"
    )
    assert not reporter.report(
        "training", current=1, total=3, unit="epochs", detail_code="epoch"
    )
    assert reporter.revision == 3

    # A meaningful same-stage progress boundary is not a heartbeat.
    assert reporter.report(
        "training", current=2, total=3, unit="epochs", detail_code="epoch"
    )
    # Stage transitions always bypass the heartbeat interval.
    assert reporter.report("evaluating", detail_code="validation")
    assert [call["payload"]["revision"] for call in client.calls] == [3, 4, 5]

    clock.now += 31
    assert reporter.report("evaluating", detail_code="validation")
    assert client.calls[-1]["payload"]["revision"] == 6


def test_timeout_retries_are_bounded_and_reuse_one_revision():
    client = _FakeClient([requests.Timeout(), requests.Timeout(), {"updated": True}])
    delays: list[float] = []
    reporter = ProgressReporter(client, _job(), sleep=delays.append, max_attempts=3)

    assert reporter.report("environment_starting")
    assert len(client.calls) == 3
    assert {call["payload"]["revision"] for call in client.calls} == {3}
    assert delays == [0.25, 0.5]
    assert reporter.diagnostics == ()


def test_timeout_exhaustion_is_nonfatal_and_records_only_bounded_reason_codes():
    client = _FakeClient([requests.Timeout(), requests.Timeout(), requests.Timeout()])
    reporter = ProgressReporter(client, _job(), sleep=lambda _: None, max_attempts=3)

    training_completed = False
    assert not reporter.report("environment_starting")
    training_completed = True

    assert training_completed
    assert len(client.calls) == 3
    assert reporter.diagnostics == (
        {
            "reason": "timeout_exhausted",
            "stage": "environment_starting",
            "revision": 3,
            "attempts": 3,
        },
    )
    assert "callback-secret" not in repr(reporter.diagnostics)
    assert "attempt-2" not in repr(reporter.diagnostics)


@pytest.mark.parametrize("status_code", [404, 405, 501])
def test_unsupported_endpoint_disables_instance_without_retry(status_code: int):
    client = _FakeClient(
        [
            {
                "status": "unavailable",
                "reason": "progress_endpoint_unavailable",
                "status_code": status_code,
            }
        ]
    )
    reporter = ProgressReporter(client, _job())

    assert not reporter.report("remote_scheduled")
    assert not reporter.report("environment_starting")
    assert len(client.calls) == 1
    assert not reporter.enabled
    assert reporter.diagnostics[0]["reason"] == "endpoint_unsupported"
    assert reporter.diagnostics[0]["status_code"] == status_code


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_rejection_is_not_retried_or_raised(status_code: int):
    client = _FakeClient([_http_error(status_code)])
    reporter = ProgressReporter(client, _job(), max_attempts=5)

    assert not reporter.report("remote_scheduled")
    assert not reporter.report("environment_starting")
    assert len(client.calls) == 1
    assert reporter.diagnostics[0]["reason"] == "authentication_rejected"


@pytest.mark.parametrize("status_code", [408, 425, 429, 500, 503])
def test_transient_http_failures_retry_only_to_the_bound(status_code: int):
    client = _FakeClient([_http_error(status_code) for _ in range(10)])
    delays: list[float] = []
    reporter = ProgressReporter(
        client,
        _job(),
        max_attempts=3,
        sleep=delays.append,
    )

    assert not reporter.report("remote_scheduled")
    assert len(client.calls) == 3
    assert delays == [0.25, 0.5]
    assert reporter.diagnostics[0]["reason"] == "transient_http_error_exhausted"
    assert reporter.diagnostics[0]["status_code"] == status_code


def test_unexpected_reporting_outage_never_fails_training():
    client = _FakeClient([RuntimeError("payload included /private/data and a token")])
    reporter = ProgressReporter(client, _job())

    result = ["training-started"]
    assert not reporter.report("data_loading")
    result.append("training-finished")

    assert result == ["training-started", "training-finished"]
    assert reporter.diagnostics[0]["reason"] == "reporter_error"
    assert "/private/data" not in repr(reporter.diagnostics)


def test_payload_builder_bounds_utf8_and_rejects_unsafe_or_invalid_fields():
    client = _FakeClient()
    reporter = ProgressReporter(client, _job())

    message = "é" * 400
    assert reporter.report("data_loading", message=message)
    sent_message = client.calls[-1]["payload"]["message"]
    assert len(sent_message.encode("utf-8")) <= 512
    assert sent_message.encode("utf-8").decode("utf-8") == sent_message

    assert reporter.report(
        "model_initializing",
        metadata={
            "provider": "Modal",
            "framework": "Torchvision",
            "execution_mode": "Remote_GPU",
            "cache_status": "Cache-Hit",
            "task_type": "Classification",
            "resource_class": "T4",
            "early_stopped": False,
        },
    )
    assert client.calls[-1]["payload"]["metadata"] == {
        "provider": "modal",
        "framework": "torchvision",
        "execution_mode": "remote_gpu",
        "cache_status": "cache-hit",
        "task_type": "classification",
        "resource_class": "t4",
        "early_stopped": False,
    }

    invalid_reports = (
        {"stage": "completed"},
        {"stage": "training", "current": 1, "total": None, "unit": "epochs"},
        {"stage": "training", "current": 2, "total": 1, "unit": "epochs"},
        {"stage": "training", "current": 1, "total": 2, "unit": "fortnights"},
        {"stage": "training", "message": "loading s3://private-bucket/data"},
        {"stage": "training", "metadata": {"prompt": "hidden"}},
        {"stage": "training", "metadata": {"provider": "/private/provider"}},
        {"stage": "training", "metadata": {"provider": "Modal GPU"}},
        {"stage": "training", "metadata": {"provider": 42}},
        {"stage": "training", "metadata": {"provider": 1.5}},
        {"stage": "training", "metadata": {"early_stopped": "false"}},
        {"stage": "training", "metadata": {"early_stopped": 0}},
    )
    call_count = len(client.calls)
    for values in invalid_reports:
        stage = values.pop("stage")
        assert not reporter.report(stage, **values)
    assert len(client.calls) == call_count
    assert all(item["reason"] == "invalid_payload" for item in reporter.diagnostics)


def test_disable_flag_is_a_noop_and_default_is_enabled(monkeypatch):
    disabled_client = _FakeClient()
    monkeypatch.setenv("MODEL_EXPRESS_PROGRESS_REPORTING_ENABLED", "false")
    disabled = ProgressReporter(disabled_client, _job())

    assert not disabled.enabled
    assert not disabled.report("remote_scheduled")
    assert disabled_client.calls == []

    monkeypatch.delenv("MODEL_EXPRESS_PROGRESS_REPORTING_ENABLED")
    enabled_client = _FakeClient()
    enabled = ProgressReporter(enabled_client, _job())
    assert enabled.enabled
    assert enabled.report("remote_scheduled")


def test_retry_configuration_is_hard_capped(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_PROGRESS_REPORT_MAX_ATTEMPTS", "100000")
    monkeypatch.setenv("MODEL_EXPRESS_PROGRESS_REPORT_TIMEOUT_SECONDS", "100000")
    client = _FakeClient([requests.Timeout() for _ in range(100)])
    reporter = ProgressReporter(client, _job(), sleep=lambda _: None)

    assert not reporter.report("environment_starting")
    assert len(client.calls) == MAX_PROGRESS_REPORT_ATTEMPTS
    assert all(call["timeout"] == MAX_PROGRESS_TIMEOUT_SECONDS for call in client.calls)


def test_local_diagnostics_are_bounded_and_exclude_arbitrary_payloads():
    client = _FakeClient()
    reporter = ProgressReporter(client, _job())

    for index in range(MAX_DIAGNOSTICS + 10):
        assert not reporter.report(
            "training",
            metadata={"prompt": f"secret prompt {index} /private/path"},
        )

    assert len(reporter.diagnostics) == MAX_DIAGNOSTICS
    rendered = repr(reporter.diagnostics)
    assert "secret prompt" not in rendered
    assert "/private/path" not in rendered
    assert "callback-secret" not in rendered
