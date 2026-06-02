from __future__ import annotations

from worker.training.modal_dispatcher import ModalDispatcher, MAX_DISPATCHER_SLOTS
from worker.training.modal_provider import ModalRetryableFailureReported


class _FakeClient:
    def __init__(self, jobs: list[dict], requirements: list[dict] | None = None):
        self.jobs = list(jobs)
        self.requirements = list(requirements or [])
        self.registered = []
        self.polls = []
        self.heartbeats = []
        self.failures = []
        self.requirement_polls = []

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
        return self.jobs.pop(0)

    def fail_job(self, job_id: str, error: str, retryable: bool = False) -> dict:
        self.failures.append({"job_id": job_id, "error": error, "retryable": retryable})
        return {"id": job_id}


def _modal_job(job_id: str, cache_key: str) -> dict:
    return {
        "id": job_id,
        "project_id": "project_1",
        "template": "train_experiment",
        "config": {
            "dataset_id": "dataset_1",
            "provider": "modal",
            "dataset_materialization": {
                "dataset_cache_key": cache_key,
                "cold_cache_policy": "single_materialization_per_checksum",
            },
        },
    }


def test_modal_dispatcher_registers_logical_slots_and_filters_modal_polling():
    client = _FakeClient([])
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
        "export_champion",
        "champion_demo_prediction",
        "generate_visual_exemplars",
    ]


def test_modal_dispatcher_skips_dataset_prewarm_by_default():
    jobs = [_modal_job("job_1", "sha256-a")]
    client = _FakeClient(jobs)
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


def test_modal_dispatcher_prewarms_same_cache_key_once_before_running_jobs(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DATASET_PREWARM_ENABLED", "true")
    jobs = [_modal_job("job_1", "sha256-a"), _modal_job("job_2", "sha256-a")]
    client = _FakeClient(jobs)
    materialized = []
    ran = []

    def materializer(_client, job):
        materialized.append(job["id"])
        return {"dataset_materialization": {"dataset_materialization_cache_miss": True}}

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


def test_modal_dispatcher_does_not_double_fail_reported_retryable_materialization_errors(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_MODAL_DATASET_PREWARM_ENABLED", "true")
    client = _FakeClient([_modal_job("job_1", "sha256-a")])
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


def test_modal_dispatcher_grows_slots_from_active_worker_requirement():
    jobs = [_modal_job(f"job_{index}", "sha256-a") for index in range(1, 5)]
    client = _FakeClient(
        jobs,
        requirements=[
            {
                "id": "worker_requirement_1",
                "project_id": "project_1",
                "provider": "modal",
                "target_count": 4,
                "max_concurrent_jobs": 4,
                "status": "ACTIVE",
            }
        ],
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
        requirements=[
            {
                "provider": "modal",
                "target_count": MAX_DISPATCHER_SLOTS + 10,
                "status": "ACTIVE",
            }
        ],
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
