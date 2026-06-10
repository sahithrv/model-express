from __future__ import annotations

import threading
import time

from worker.training import modal_dispatcher
from worker.training.modal_dispatcher import ModalDispatcher, MAX_DISPATCHER_SLOTS
from worker.training.modal_provider import ModalRetryableFailureReported


class _FakeClient:
    def __init__(
        self,
        jobs: list[dict],
        requirements: list[dict] | None = None,
        requirements_sequence: list[list[dict]] | None = None,
    ):
        self.jobs = list(jobs)
        self.requirements = list(requirements or [])
        self.requirements_sequence = list(requirements_sequence or [])
        self.registered = []
        self.polls = []
        self.heartbeats = []
        self.failures = []
        self.requirement_polls = []
        self.dispatcher_events = []

    def register_worker(self, project_id: str, *, name: str | None = None, gpu_type: str | None = None) -> dict:
        worker = {
            "id": f"worker_{len(self.registered) + 1}",
            "project_id": project_id,
            "name": name,
            "gpu_type": gpu_type,
        }
        self.registered.append(worker)
        return worker

    def heartbeat_worker(self, worker_id: str) -> dict:
        self.heartbeats.append(worker_id)
        return {"id": worker_id}

    def list_project_worker_requirements(self, project_id: str) -> list[dict]:
        self.requirement_polls.append(project_id)
        if self.requirements_sequence:
            self.requirements = list(self.requirements_sequence.pop(0))
        return list(self.requirements)

    def poll_job(
        self,
        worker_id: str,
        *,
        provider: str | None = None,
        templates: list[str] | None = None,
        include_unspecified_provider_templates: list[str] | None = None,
    ) -> dict | None:
        self.polls.append(
            {
                "worker_id": worker_id,
                "provider": provider,
                "templates": templates,
                "include_unspecified_provider_templates": include_unspecified_provider_templates,
            }
        )
        if not self.jobs:
            return None
        for index, job in enumerate(self.jobs):
            if templates and job.get("template") not in templates:
                continue
            job_provider = str((job.get("config") or {}).get("provider") or "").strip().lower()
            if provider and job_provider and job_provider != provider:
                continue
            if provider and not job_provider and job.get("template") not in (include_unspecified_provider_templates or []):
                continue
            return self.jobs.pop(index)
        return None

    def fail_job(self, job_id: str, error: str, retryable: bool = False) -> dict:
        self.failures.append({"job_id": job_id, "error": error, "retryable": retryable})
        return {"id": job_id}

    def report_dispatcher_event(
        self,
        project_id: str,
        event_type: str,
        *,
        message: str = "",
        plan_id: str = "",
        payload: dict | None = None,
    ) -> dict:
        event = {
            "project_id": project_id,
            "event_type": event_type,
            "message": message,
            "plan_id": plan_id,
            "payload": payload or {},
        }
        self.dispatcher_events.append(event)
        return {"event": event}


def _modal_job(
    job_id: str,
    cache_key: str,
    *,
    plan_id: str = "",
    dataset_id: str = "dataset_1",
    training_tier: str = "",
    task_type: str = "",
    model: str = "",
    model_kind: str = "",
) -> dict:
    config = {
        "dataset_id": dataset_id,
        "provider": "modal",
        "dataset_materialization": {
            "dataset_cache_key": cache_key,
            "cold_cache_policy": "single_materialization_per_checksum",
        },
    }
    if plan_id:
        config["plan_id"] = plan_id
    if training_tier:
        config["training_tier"] = training_tier
    if task_type:
        config["task_type"] = task_type
    if model:
        config["model"] = model
    if model_kind:
        config["model_kind"] = model_kind
    return {
        "id": job_id,
        "project_id": "project_1",
        "template": "train_experiment",
        "config": config,
    }


def _profile_job(job_id: str, *, provider: str = "") -> dict:
    config = {"dataset_id": "dataset_1"}
    if provider:
        config["provider"] = provider
    return {
        "id": job_id,
        "project_id": "project_1",
        "template": "profile_dataset",
        "config": config,
    }


def _visual_analysis_job(job_id: str, *, provider: str = "") -> dict:
    config = {"dataset_id": "dataset_1"}
    if provider:
        config["provider"] = provider
    return {
        "id": job_id,
        "project_id": "project_1",
        "template": "analyze_dataset_visuals",
        "config": config,
    }


def _modal_requirement(target_count: int, status: str = "ACTIVE") -> dict:
    return {
        "id": "worker_requirement_1",
        "project_id": "project_1",
        "provider": "modal",
        "target_count": target_count,
        "max_concurrent_jobs": target_count,
        "status": status,
    }


def test_modal_dispatcher_registers_logical_slots_and_filters_modal_polling():
    client = _FakeClient([], requirements=[_modal_requirement(2)])
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=2,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, _job: None,
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()

    assert [worker["gpu_type"] for worker in client.registered] == ["modal", "modal"]
    assert client.polls[0]["provider"] == "modal"
    assert client.polls[0]["templates"] == [
        "train_experiment",
        "profile_dataset",
        "analyze_dataset_visuals",
        "export_champion",
        "champion_demo_prediction",
        "generate_visual_exemplars",
    ]
    assert client.polls[0]["include_unspecified_provider_templates"] == [
        "profile_dataset",
        "analyze_dataset_visuals",
        "export_champion",
        "champion_demo_prediction",
        "generate_visual_exemplars",
    ]


def test_modal_dispatcher_zero_requirement_poll_slot_claims_profile_dataset():
    client = _FakeClient([_profile_job("job_1")])
    ran = []
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert ran == ["job_1"]
    assert client.polls[0]["templates"] == [
        "profile_dataset",
        "analyze_dataset_visuals",
        "export_champion",
        "champion_demo_prediction",
        "generate_visual_exemplars",
    ]


def test_modal_dispatcher_zero_requirement_poll_slot_claims_visual_analysis():
    client = _FakeClient([_visual_analysis_job("job_1")])
    ran = []
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert ran == ["job_1"]
    assert client.polls[0]["templates"] == [
        "profile_dataset",
        "analyze_dataset_visuals",
        "export_champion",
        "champion_demo_prediction",
        "generate_visual_exemplars",
    ]


def test_modal_dispatcher_zero_requirement_poll_slot_does_not_claim_training_jobs():
    client = _FakeClient([_modal_job("job_1", "sha256-a")])
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, _job: None,
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()

    assert client.jobs[0]["id"] == "job_1"
    assert client.polls[0]["templates"] == [
        "profile_dataset",
        "analyze_dataset_visuals",
        "export_champion",
        "champion_demo_prediction",
        "generate_visual_exemplars",
    ]


def test_modal_dispatcher_skips_dataset_prewarm_by_default():
    jobs = [_modal_job("job_1", "sha256-a")]
    client = _FakeClient(jobs, requirements=[_modal_requirement(1)])
    materialized = []
    ran = []

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        materializer=lambda _client, job: materialized.append(job["id"]),
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert materialized == []
    assert ran == ["job_1"]


def test_modal_dispatcher_batch_flag_off_keeps_single_job_runner():
    jobs = [
        _modal_job("job_1", "sha256-a", plan_id="plan_1", training_tier="preview"),
        _modal_job("job_2", "sha256-a", plan_id="plan_1", training_tier="preview"),
    ]
    client = _FakeClient(jobs, requirements=[_modal_requirement(2)])
    ran = []
    batches = []

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=2,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        batch_runner=lambda _client, batch_jobs, batch: batches.append((batch_jobs, batch)),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert sorted(ran) == ["job_1", "job_2"]
    assert batches == []


def test_modal_dispatcher_groups_preview_jobs_with_same_batch_contract(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER", "1")
    jobs = [
        _modal_job("job_1", "sha256-a", plan_id="plan_1", training_tier="preview"),
        _modal_job("job_2", "sha256-a", plan_id="plan_1", training_tier="preview"),
    ]
    client = _FakeClient(jobs, requirements=[_modal_requirement(2)])
    ran = []
    batches = []

    def batch_runner(_client, batch_jobs, batch):
        batches.append(
            {
                "job_ids": [job["id"] for job in batch_jobs],
                "batch": batch,
            }
        )

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=2,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        batch_runner=batch_runner,
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert ran == []
    assert len(batches) == 1
    assert batches[0]["job_ids"] == ["job_1", "job_2"]
    assert batches[0]["batch"]["schema_version"] == "modal_preview_batch.v1"
    assert batches[0]["batch"]["project_id"] == "project_1"
    assert batches[0]["batch"]["plan_id"] == "plan_1"
    assert batches[0]["batch"]["dataset_id"] == "dataset_1"
    assert batches[0]["batch"]["dataset_cache_key"] == "sha256-a"
    assert batches[0]["batch"]["training_tier"] == "preview"
    assert batches[0]["batch"]["task_type"] == "image_classification"
    assert [poll["worker_id"] for poll in client.polls[:2]] == ["worker_1", "worker_2"]


def test_modal_dispatcher_batch_respects_max_trials(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER", "1")
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_MAX_TRIALS", "2")
    jobs = [
        _modal_job(f"job_{index}", "sha256-a", plan_id="plan_1", training_tier="preview")
        for index in range(1, 6)
    ]
    client = _FakeClient(jobs, requirements=[_modal_requirement(5)])
    ran = []
    batches = []

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=5,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        batch_runner=lambda _client, batch_jobs, batch: batches.append(([job["id"] for job in batch_jobs], batch)),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert [batch_jobs for batch_jobs, _batch in batches] == [["job_1", "job_2"], ["job_3", "job_4"]]
    assert ran == ["job_5"]
    assert all(batch["batch_size"] == 2 for _batch_jobs, batch in batches)


def test_modal_dispatcher_batch_requires_same_cache_key(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER", "1")
    jobs = [
        _modal_job("job_1", "sha256-a", plan_id="plan_1", training_tier="preview"),
        _modal_job("job_2", "sha256-b", plan_id="plan_1", training_tier="preview"),
    ]
    client = _FakeClient(jobs, requirements=[_modal_requirement(2)])
    ran = []
    batches = []

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=2,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        batch_runner=lambda _client, batch_jobs, batch: batches.append((batch_jobs, batch)),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert sorted(ran) == ["job_1", "job_2"]
    assert batches == []


def test_modal_dispatcher_batch_requires_same_plan(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER", "1")
    jobs = [
        _modal_job("job_1", "sha256-a", plan_id="plan_1", training_tier="preview"),
        _modal_job("job_2", "sha256-a", plan_id="plan_2", training_tier="preview"),
    ]
    client = _FakeClient(jobs, requirements=[_modal_requirement(2)])
    ran = []
    batches = []

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=2,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        batch_runner=lambda _client, batch_jobs, batch: batches.append((batch_jobs, batch)),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert sorted(ran) == ["job_1", "job_2"]
    assert batches == []


def test_modal_dispatcher_groups_yolo_object_detection_jobs(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER", "1")
    monkeypatch.setenv("MODEL_EXPRESS_YOLO_BATCH_PREVIEW", "1")
    jobs = [
        _modal_job(
            "job_1",
            "sha256-a",
            plan_id="plan_1",
            training_tier="preview",
            task_type="object_detection",
            model="yolo11n.pt",
            model_kind="ultralytics_yolo_detector",
        ),
        _modal_job(
            "job_2",
            "sha256-a",
            plan_id="plan_1",
            training_tier="preview",
            task_type="object_detection",
            model="yolo11s.pt",
            model_kind="ultralytics_yolo_detector",
        ),
    ]
    client = _FakeClient(jobs, requirements=[_modal_requirement(2)])
    batches = []

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=2,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, _job: None,
        batch_runner=lambda _client, batch_jobs, batch: batches.append(([job["id"] for job in batch_jobs], batch)),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert batches[0][0] == ["job_1", "job_2"]
    assert batches[0][1]["task_type"] == "object_detection"


def test_modal_dispatcher_singleton_eligible_job_runs_as_single(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER", "1")
    client = _FakeClient(
        [_modal_job("job_1", "sha256-a", plan_id="plan_1", training_tier="preview")],
        requirements=[_modal_requirement(1)],
    )
    ran = []
    batches = []

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        batch_runner=lambda _client, batch_jobs, batch: batches.append((batch_jobs, batch)),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert ran == ["job_1"]
    assert batches == []


def test_modal_dispatcher_batch_flag_falls_back_for_ineligible_jobs(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_BATCH_RUNNER", "1")
    jobs = [
        _modal_job("job_1", "sha256-a", plan_id="plan_1"),
        _modal_job("job_2", "sha256-a", plan_id="plan_1", training_tier="preview"),
    ]
    client = _FakeClient(jobs, requirements=[_modal_requirement(2)])
    ran = []
    batches = []

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=2,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        batch_runner=lambda _client, batch_jobs, batch: batches.append((batch_jobs, batch)),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert sorted(ran) == ["job_1", "job_2"]
    assert batches == []


def test_modal_dispatcher_prewarms_same_cache_key_once_before_running_jobs(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DATASET_PREWARM_ENABLED", "true")
    jobs = [_modal_job("job_1", "sha256-a"), _modal_job("job_2", "sha256-a")]
    client = _FakeClient(jobs, requirements=[_modal_requirement(2)])
    materialized = []
    ran = []

    def materializer(_client, job):
        materialized.append(job["id"])
        return {
            "dataset_materialization": {
                "dataset_materialization_cache_miss": True,
                "dataset_prewarm_reusable_for_training": True,
                "dataset_prewarm_reuse_status": "reusable_for_training",
            }
        }

    def runner(_client, job):
        ran.append(job["id"])

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=2,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=runner,
        materializer=materializer,
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert len(materialized) == 1
    assert materialized[0] in {"job_1", "job_2"}
    assert sorted(ran) == ["job_1", "job_2"]
    assert client.failures == []
    assert dispatcher._warm_cache_keys == {"sha256-a"}


def test_modal_dispatcher_staging_only_prewarm_does_not_mark_training_cache_warm(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DATASET_PREWARM_ENABLED", "true")
    client = _FakeClient([_modal_job("job_1", "sha256-a")], requirements=[_modal_requirement(1)])
    ran = []

    def materializer(_client, _job):
        return {
            "dataset_materialization": {
                "dataset_materialization_cache_miss": True,
                "dataset_materialization_cache_root": "/cache/model-express/datasets",
                "dataset_training_cache_root": "/tmp/model-express/training-datasets",
                "dataset_prewarm_reusable_for_training": False,
                "dataset_prewarm_reuse_status": "staging_only_root_mismatch",
            }
        }

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        materializer=materializer,
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert ran == ["job_1"]
    assert dispatcher._warm_cache_keys == set()


def test_modal_dispatcher_does_not_double_fail_reported_retryable_materialization_errors(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DATASET_PREWARM_ENABLED", "true")
    client = _FakeClient([_modal_job("job_1", "sha256-a")], requirements=[_modal_requirement(1)])
    ran = []

    def materializer(fake_client, job):
        fake_client.fail_job(
            job["id"],
            "reported retryable materialization failure",
            retryable=True,
        )
        raise ModalRetryableFailureReported("reported retryable materialization failure")

    def runner(_client, job):
        ran.append(job["id"])

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=runner,
        materializer=materializer,
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert ran == []
    assert client.failures == [
        {
            "job_id": "job_1",
            "error": "reported retryable materialization failure",
            "retryable": True,
        }
    ]


def test_modal_dispatcher_retryable_prewarm_failure_marks_waiters_retryable(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DATASET_PREWARM_ENABLED", "true")
    client = _FakeClient(
        [_modal_job("job_1", "sha256-a"), _modal_job("job_2", "sha256-a")],
        requirements=[_modal_requirement(2)],
    )
    ran = []

    def materializer(fake_client, job):
        time.sleep(0.05)
        fake_client.fail_job(job["id"], "reported retryable materialization failure", retryable=True)
        raise ModalRetryableFailureReported("reported retryable materialization failure")

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=2,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        materializer=materializer,
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert ran == []
    assert sorted(failure["job_id"] for failure in client.failures) == ["job_1", "job_2"]
    assert all(failure["retryable"] is True for failure in client.failures)


def test_modal_dispatcher_waiter_preserves_retryable_failure_when_fail_report_fails(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DATASET_PREWARM_ENABLED", "true")
    client = _FakeClient(
        [_modal_job("job_1", "sha256-a"), _modal_job("job_2", "sha256-a")],
        requirements=[_modal_requirement(2)],
    )
    ran = []

    def materializer(fake_client, job):
        time.sleep(0.05)
        fake_client.fail_job(job["id"], "reported retryable materialization failure", retryable=True)
        raise ModalRetryableFailureReported("reported retryable materialization failure")

    def fail_job(job_id: str, error: str, retryable: bool = False) -> dict:
        client.failures.append({"job_id": job_id, "error": error, "retryable": retryable})
        if job_id == "job_2":
            raise RuntimeError("report failed")
        return {"id": job_id}

    client.fail_job = fail_job
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=2,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        materializer=materializer,
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert ran == []
    assert sorted(failure["job_id"] for failure in client.failures) == ["job_1", "job_2"]
    assert all(failure["retryable"] is True for failure in client.failures)


def test_modal_dispatcher_idle_exit_waits_for_zero_demand_refresh(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_EXIT_SECONDS", "0.01")
    client = _FakeClient([])
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        requirement_refresh_seconds=0.01,
        job_runner=lambda _client, _job: None,
        materializer=lambda _client, _job: {},
    )

    assert dispatcher._should_exit_for_idle() is False
    dispatcher.run_once()
    assert dispatcher._should_exit_for_idle() is False
    time.sleep(0.02)
    assert dispatcher._should_exit_for_idle() is False
    dispatcher.run_once()

    assert dispatcher._should_exit_for_idle() is True
    assert client.dispatcher_events[-1]["event_type"] == "DISPATCHER_IDLE_EXIT"


def test_modal_dispatcher_run_forever_zero_demand_does_not_open_modal_app(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_EXIT_SECONDS", "0.01")
    events = []

    class FakeModalSession:
        def __enter__(self):
            events.append("modal_enter")

        def __exit__(self, exc_type, exc, traceback):
            events.append("modal_exit")
            return False

    monkeypatch.setattr(modal_dispatcher, "modal_app_session", lambda: FakeModalSession())
    client = _FakeClient([])
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        requirement_refresh_seconds=0.01,
        job_runner=lambda _client, _job: None,
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_forever()

    assert events == []
    assert client.polls
    assert client.requirement_polls
    assert client.dispatcher_events[-1]["event_type"] == "DISPATCHER_IDLE_EXIT"


def test_modal_dispatcher_run_forever_zero_demand_claims_profile_dataset(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_EXIT_SECONDS", "0.01")
    monkeypatch.delenv("GPU_TYPE", raising=False)
    events = []

    class FakeModalSession:
        def __enter__(self):
            events.append("modal_enter")

        def __exit__(self, exc_type, exc, traceback):
            events.append("modal_exit")
            return False

    monkeypatch.setattr(modal_dispatcher, "modal_app_session", lambda: FakeModalSession())
    client = _FakeClient([_profile_job("profile_1")])
    ran = []
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        requirement_refresh_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_forever()

    assert ran == ["profile_1"]
    assert client.polls
    assert events == []
    assert client.dispatcher_events[-1]["event_type"] == "DISPATCHER_IDLE_EXIT"


def test_modal_dispatcher_zero_demand_modal_profile_runs_inside_modal_session(monkeypatch):
    monkeypatch.setenv("GPU_TYPE", "modal")
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_EXIT_SECONDS", "0.01")
    events = []
    in_session = {"active": False}

    class FakeModalSession:
        def __enter__(self):
            in_session["active"] = True
            events.append("modal_enter")

        def __exit__(self, exc_type, exc, traceback):
            events.append("modal_exit")
            in_session["active"] = False
            return False

    def runner(_client, job):
        assert in_session["active"] is True
        ran.append(job["id"])

    monkeypatch.setattr(modal_dispatcher, "modal_app_session", lambda: FakeModalSession())
    client = _FakeClient([_profile_job("profile_1")])
    ran = []
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        requirement_refresh_seconds=0.01,
        job_runner=runner,
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_forever()

    assert ran == ["profile_1"]
    assert events == ["modal_enter", "modal_exit"]
    assert client.dispatcher_events[-1]["event_type"] == "DISPATCHER_IDLE_EXIT"


def test_modal_dispatcher_failed_requirement_refresh_prevents_idle_exit(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_EXIT_SECONDS", "0.01")
    client = _FakeClient([])
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        requirement_refresh_seconds=0,
        job_runner=lambda _client, _job: None,
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    assert dispatcher._should_exit_for_idle() is False

    def fail_refresh(_project_id: str) -> list[dict]:
        raise RuntimeError("requirements unavailable")

    client.list_project_worker_requirements = fail_refresh
    time.sleep(0.02)
    dispatcher.run_once()

    assert dispatcher._should_exit_for_idle() is False


def test_modal_dispatcher_grows_slots_from_active_worker_requirement():
    jobs = [_modal_job(f"job_{index}", "sha256-a") for index in range(1, 5)]
    client = _FakeClient(
        jobs,
        requirements=[_modal_requirement(4)],
    )
    ran = []

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        requirement_refresh_seconds=0.01,
        job_runner=lambda _client, job: ran.append(job["id"]),
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert len(client.registered) == 4
    assert len(client.polls) >= 4
    assert sorted(ran) == ["job_1", "job_2", "job_3", "job_4"]


def test_modal_dispatcher_caps_requirement_slots():
    client = _FakeClient(
        [],
        requirements=[_modal_requirement(MAX_DISPATCHER_SLOTS + 10)],
    )
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=0.01,
        requirement_refresh_seconds=0.01,
        job_runner=lambda _client, _job: None,
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()

    assert len(client.registered) == MAX_DISPATCHER_SLOTS


def test_modal_dispatcher_stops_polling_and_heartbeating_idle_slots_after_shrink():
    client = _FakeClient([], requirements=[_modal_requirement(3)])
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=100,
        requirement_refresh_seconds=0,
        job_runner=lambda _client, _job: None,
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    assert len(client.registered) == 3

    client.requirements = [_modal_requirement(1)]
    client.polls.clear()
    client.heartbeats.clear()
    for slot in dispatcher.slots:
        slot.next_heartbeat_at = 0.0

    dispatcher.run_once()

    assert dispatcher.slot_count == 1
    assert [poll["worker_id"] for poll in client.polls] == ["worker_1"]
    assert client.heartbeats == ["worker_1"]
    assert client.dispatcher_events[-1]["event_type"] == "DISPATCHER_STATUS"
    assert client.dispatcher_events[-1]["payload"]["slot_count"] == 1


def test_modal_dispatcher_zero_demand_stops_idle_slot_polling_and_heartbeats(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DISPATCHER_IDLE_POLL_SLOTS", "0")
    client = _FakeClient([], requirements=[_modal_requirement(2)])
    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=100,
        requirement_refresh_seconds=0,
        job_runner=lambda _client, _job: None,
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    assert len(client.registered) == 2

    client.requirements = []
    client.polls.clear()
    client.heartbeats.clear()
    for slot in dispatcher.slots:
        slot.next_heartbeat_at = 0.0

    dispatcher.run_once()

    assert dispatcher.slot_count == 0
    assert client.polls == []
    assert client.heartbeats == []


def test_modal_dispatcher_collects_active_out_of_target_slots_without_repolling_them():
    jobs = [_modal_job("job_1", "sha256-a"), _modal_job("job_2", "sha256-a"), _modal_job("job_3", "sha256-a")]
    client = _FakeClient(jobs, requirements=[_modal_requirement(2)])
    started = {job_id: threading.Event() for job_id in ("job_1", "job_2")}
    release = threading.Event()
    ran = []

    def runner(_client, job):
        job_id = job["id"]
        if job_id in started:
            started[job_id].set()
            release.wait(timeout=2)
        ran.append(job_id)

    dispatcher = ModalDispatcher(
        client,
        "project_1",
        slot_count=1,
        poll_interval_seconds=0.01,
        heartbeat_interval_seconds=100,
        requirement_refresh_seconds=0,
        job_runner=runner,
        materializer=lambda _client, _job: {},
    )

    dispatcher.run_once()
    assert started["job_1"].wait(timeout=2)
    assert started["job_2"].wait(timeout=2)

    client.requirements = [_modal_requirement(1)]
    client.polls.clear()
    client.heartbeats.clear()
    for slot in dispatcher.slots:
        slot.next_heartbeat_at = 0.0

    dispatcher.run_once()

    assert sorted(client.heartbeats) == ["worker_1", "worker_2"]
    assert client.polls == []

    release.set()
    dispatcher.wait_for_active_jobs(timeout_seconds=2)

    assert sorted(ran) == ["job_1", "job_2", "job_3"]
    assert "worker_2" not in [poll["worker_id"] for poll in client.polls]
