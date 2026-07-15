from __future__ import annotations

import copy

import pytest
import requests

from worker.progress import PROGRESS_TAXONOMY_VERSION, WORKER_PROGRESS_STAGES
from worker.training.local import run_local_training
from worker.training.progress_reporting import simulator_progress_metadata


class _FakeClient:
    def __init__(self, *, progress_error: BaseException | None = None) -> None:
        self.metrics: list[dict] = []
        self.summaries: list[dict] = []
        self.evaluations: list[dict] = []
        self.execution_observations: list[dict] = []
        self.completed: list[dict] = []
        self.progress: list[dict] = []
        self.events: list[str] = []
        self.progress_error = progress_error

    def report_progress(
        self,
        job_id: str,
        payload: dict,
        *,
        job: dict,
        timeout: float,
    ) -> dict:
        self.events.append(f"progress:{payload.get('stage')}")
        if self.progress_error is not None:
            raise self.progress_error
        self.progress.append(
            {
                "job_id": job_id,
                "payload": copy.deepcopy(payload),
                "job": copy.deepcopy(job),
                "timeout": timeout,
            }
        )
        return {"updated": True}

    def report_metric(self, job_id: str, epoch: int, metrics: dict[str, float]) -> dict:
        self.events.append(f"metric:{epoch}")
        self.metrics.append({"job_id": job_id, "epoch": epoch, "metrics": metrics})
        return {"ok": True}

    def report_training_run_summary(self, job_id: str, summary: dict) -> dict:
        self.events.append(f"summary:{summary.get('status')}")
        self.summaries.append({"job_id": job_id, "summary": summary})
        return {"ok": True}

    def report_training_run_evaluation(self, job_id: str, evaluation: dict) -> dict:
        self.events.append("evaluation")
        self.evaluations.append({"job_id": job_id, "evaluation": evaluation})
        return {"ok": True}

    def report_execution_observation(self, job_id: str, observation: dict, *, job: dict | None = None) -> dict:
        self.events.append("execution_observation")
        self.execution_observations.append({"job_id": job_id, "observation": observation, "job": job})
        return {"ok": True}

    def complete_job(self, job_id: str, mlflow_run_id: str = "") -> dict:
        self.events.append("complete")
        self.completed.append({"job_id": job_id, "mlflow_run_id": mlflow_run_id})
        return {"ok": True}


def _progress_job(*, detection: bool = False, epochs: int = 2) -> dict:
    config = {
        "provider": "local",
        "model": "yolo11n.pt" if detection else "resnet18",
        "epochs": epochs,
        "active_attempt_id": "job_progress:attempt-1",
        "callback_token": "callback-secret",
        "execution_spec_v1": {
            "schema_version": "execution_spec_v1",
            "accepted_config": {"epochs": epochs},
        },
    }
    if detection:
        config.update(
            {
                "task_type": "object_detection",
                "model_kind": "ultralytics_yolo_detector",
            }
        )
    return {"id": "job_progress", "project_id": "project_1", "config": config}


@pytest.mark.parametrize(
    ("detection", "task_type"),
    [
        (False, "image_classification"),
        (True, "object_detection"),
    ],
)
def test_local_simulator_progress_uses_stable_taxonomy_before_epoch_one(
    monkeypatch,
    detection: bool,
    task_type: str,
):
    monkeypatch.setenv("LOCAL_TRAINING_EPOCH_SECONDS", "0")
    client = _FakeClient()

    run_local_training(client, _progress_job(detection=detection, epochs=2))

    payloads = [entry["payload"] for entry in client.progress]
    assert [payload["stage"] for payload in payloads] == [
        "worker_starting",
        "data_loading",
        "model_initializing",
        "training",
        "training",
        "training",
        "evaluating",
        "finalizing",
    ]
    epoch_payloads = [payload for payload in payloads if payload["stage"] == "training"]
    assert [payload["current"] for payload in epoch_payloads] == [0, 1, 2]
    assert all(payload["total"] == 2 and payload["unit"] == "epoch" for payload in epoch_payloads)
    assert client.events.index("progress:training") < client.events.index("metric:1")
    assert all(payload["taxonomy_version"] == PROGRESS_TAXONOMY_VERSION for payload in payloads)
    assert all(payload["stage"] in WORKER_PROGRESS_STAGES for payload in payloads)
    assert all(payload["status"] == "running" for payload in payloads)
    assert all(payload["detail_code"].startswith("simulator_") for payload in payloads)
    assert all(
        payload["metadata"]
        == {
            "provider": "local",
            "framework": "deterministic_simulator",
            "execution_mode": "local_simulator",
            "task_type": task_type,
        }
        for payload in payloads
    )
    assert [payload["revision"] for payload in payloads] == list(range(3, 11))
    assert not ({"completed", "failed", "cancelled"} & {payload["stage"] for payload in payloads})

    finalizing_index = client.events.index("progress:finalizing")
    assert finalizing_index < client.events.index("summary:SUCCEEDED")
    assert finalizing_index < client.events.index("evaluation")
    assert finalizing_index < client.events.index("execution_observation")
    assert finalizing_index < client.events.index("complete")


def test_local_simulator_progress_outage_is_nonfatal(monkeypatch):
    monkeypatch.setenv("LOCAL_TRAINING_EPOCH_SECONDS", "0")
    monkeypatch.setenv("MODEL_EXPRESS_PROGRESS_REPORT_MAX_ATTEMPTS", "1")
    client = _FakeClient(progress_error=requests.ConnectionError("progress unavailable"))

    run_local_training(client, _progress_job(epochs=1))

    assert client.progress == []
    assert len(client.metrics) == 1
    assert len(client.evaluations) == 1
    assert len(client.execution_observations) == 1
    assert len(client.completed) == 1
    assert client.events[-1] == "complete"


def test_local_simulator_does_not_complete_when_final_callback_fails(monkeypatch):
    monkeypatch.setenv("LOCAL_TRAINING_EPOCH_SECONDS", "0")
    client = _FakeClient()

    def reject_evaluation(_job_id: str, _evaluation: dict) -> dict:
        raise RuntimeError("evaluation rejected")

    client.report_training_run_evaluation = reject_evaluation

    with pytest.raises(RuntimeError, match="evaluation rejected"):
        run_local_training(client, _progress_job(epochs=1))

    assert client.progress[-1]["payload"]["stage"] == "finalizing"
    assert client.summaries[-1]["summary"]["status"] == "SUCCEEDED"
    assert client.execution_observations == []
    assert client.completed == []


def test_simulator_provider_metadata_is_canonical_and_does_not_leak_raw_provider_detail():
    metadata = simulator_progress_metadata(
        {
            "config": {
                "provider": "/private/provider with unsafe detail",
                "task_type": "object_detection",
            }
        }
    )

    assert metadata == {
        "provider": "local",
        "framework": "deterministic_simulator",
        "execution_mode": "local_simulator",
        "task_type": "object_detection",
    }
    assert "private" not in repr(metadata)


def test_local_training_reports_yolo_detection_metrics(monkeypatch):
    monkeypatch.setenv("LOCAL_TRAINING_EPOCH_SECONDS", "0")
    client = _FakeClient()
    run_local_training(
        client,
        {
            "id": "job_yolo",
            "config": {
                "model": "yolo11n.pt",
                "task_type": "object_detection",
                "model_kind": "ultralytics_yolo_detector",
                "epochs": 2,
                "batch_size": 4,
                "image_size": 640,
                "class_names": ["face", "spoof"],
                "execution_spec_v1": {
                    "schema_version": "execution_spec_v1",
                    "accepted_config": {"model": "yolo11n.pt", "epochs": 2},
                },
            },
        },
    )

    assert len(client.metrics) == 2
    assert "mAP50_95" in client.metrics[-1]["metrics"]
    assert "box_loss" in client.metrics[-1]["metrics"]
    assert client.summaries[-1]["summary"]["status"] == "SUCCEEDED"
    evaluation = client.evaluations[-1]["evaluation"]
    assert evaluation["objective_profile"]["task_type"] == "object_detection"
    assert evaluation["model_profile"]["model_kind"] == "ultralytics_yolo_detector"
    assert evaluation["model_profile"]["simulation"] is True
    assert evaluation["model_profile"]["exportable"] is False
    assert evaluation["model_profile"]["deployment_ready"] is False
    assert evaluation["model_profile"]["export_status"] == "SIMULATED_UNEXPORTABLE"
    assert evaluation["preprocessing_summary"]["simulation"] is True
    assert evaluation["holistic_scores"]["detection_metrics"]["mAP50_95"] > 0
    assert client.execution_observations == [
        {
            "job_id": "job_yolo",
            "observation": {
                "schema_version": "execution_realization_v1",
                "stage": "FINALIZED",
                "idempotency_key": "local-simulator-final-v1",
                "realized_config": {"model": "yolo11n.pt", "epochs": 2},
                "framework_arguments": {"runtime": "deterministic_local_simulator"},
                "evidence": {"simulation": True},
                "simulated": True,
            },
            "job": {
                "id": "job_yolo",
                "config": {
                    "model": "yolo11n.pt",
                    "task_type": "object_detection",
                    "model_kind": "ultralytics_yolo_detector",
                    "epochs": 2,
                    "batch_size": 4,
                    "image_size": 640,
                    "class_names": ["face", "spoof"],
                    "execution_spec_v1": {
                        "schema_version": "execution_spec_v1",
                        "accepted_config": {"model": "yolo11n.pt", "epochs": 2},
                    },
                },
            },
        }
    ]
    assert client.completed[-1]["mlflow_run_id"].startswith("local-yolo-training-")
