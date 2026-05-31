from __future__ import annotations

from worker.training.modal_dispatcher import ModalDispatcher


class _FakeClient:
    def __init__(self, jobs: list[dict]):
        self.jobs = list(jobs)
        self.registered = []
        self.polls = []
        self.heartbeats = []
        self.failures = []

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
    assert client.polls[0]["templates"] == ["train_experiment", "profile_dataset"]
    assert client.polls[0]["include_unspecified_provider_templates"] == ["profile_dataset"]


def test_modal_dispatcher_prewarms_same_cache_key_once_before_running_jobs():
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
