from __future__ import annotations

from worker.training.local import run_local_training


class _FakeClient:
    def __init__(self) -> None:
        self.metrics: list[dict] = []
        self.summaries: list[dict] = []
        self.evaluations: list[dict] = []
        self.completed: list[dict] = []

    def report_metric(self, job_id: str, epoch: int, metrics: dict[str, float]) -> dict:
        self.metrics.append({"job_id": job_id, "epoch": epoch, "metrics": metrics})
        return {"ok": True}

    def report_training_run_summary(self, job_id: str, summary: dict) -> dict:
        self.summaries.append({"job_id": job_id, "summary": summary})
        return {"ok": True}

    def report_training_run_evaluation(self, job_id: str, evaluation: dict) -> dict:
        self.evaluations.append({"job_id": job_id, "evaluation": evaluation})
        return {"ok": True}

    def complete_job(self, job_id: str, mlflow_run_id: str = "") -> dict:
        self.completed.append({"job_id": job_id, "mlflow_run_id": mlflow_run_id})
        return {"ok": True}


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
    assert evaluation["holistic_scores"]["detection_metrics"]["mAP50_95"] > 0
    assert client.completed[-1]["mlflow_run_id"].startswith("local-yolo-training-")
