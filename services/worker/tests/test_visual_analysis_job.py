from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from PIL import Image

import worker.jobs as jobs


def test_analyze_dataset_visuals_dispatch_posts_sample_pack_and_completes(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_DATASET_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.delenv("MODEL_EXPRESS_PERSIST_DATASET_CACHE", raising=False)

    def fake_download(storage_uri: str, destination: Path) -> Path:
        assert storage_uri == "s3://bucket/dataset.zip"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("cat/one.jpg", _image_bytes((80, 60), (220, 30, 30)))
            archive.writestr("dog/one.jpg", _image_bytes((24, 160), (30, 30, 220)))
        return destination

    monkeypatch.setattr(jobs, "download_s3_uri", fake_download)
    monkeypatch.setattr(jobs, "_run_visual_llm_analysis", _fake_visual_llm_analysis)
    client = _FakeClient()
    job = {
        "id": "job_visual_1",
        "template": "analyze_dataset_visuals",
        "config": {
            "dataset_id": "dataset_1",
            "max_total_images": 2,
            "max_high_detail_images": 1,
            "max_image_bytes": 20_000,
            "max_total_bytes": 50_000,
            "image_size": 64,
            "seed": 11,
        },
    }

    jobs.run_job(client, job)

    payload = client.visual_analysis_payload
    manifest = payload["input_context"]["sample_manifest_summary"]

    assert client.completed_job_id == "job_visual_1"
    assert client.failed == []
    assert payload["schema_version"] == "dataset_visual_analysis_v1"
    assert payload["dataset_id"] == "dataset_1"
    assert payload["provider"] == "fake_visual"
    assert payload["model"] == "fake-model"
    assert payload["raw_output"]
    assert payload["input_context"]["evidence_only"] is True
    assert payload["input_context"]["raw_images_included_for_planner"] is False
    assert manifest["images_analyzed"] <= 2
    assert len(payload["sample_manifest"]) == manifest["images_analyzed"]
    assert all(item["image_id"].startswith("img_") for item in payload["sample_manifest"])

    serialized = json.dumps(payload, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "source_path" not in serialized
    assert "local_path" not in serialized
    assert "relative_path" not in serialized


def test_analyze_dataset_visuals_completes_with_unavailable_record_on_llm_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_DATASET_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("MODEL_EXPRESS_VISUAL_LLM_PROVIDER", "openai")
    monkeypatch.setenv("MODEL_EXPRESS_VISUAL_LLM_MODEL", "vision-model")
    monkeypatch.delenv("MODEL_EXPRESS_PERSIST_DATASET_CACHE", raising=False)

    def fake_download(storage_uri: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w") as archive:
            archive.writestr("cat/one.jpg", _image_bytes((80, 60), (220, 30, 30)))
            archive.writestr("dog/one.jpg", _image_bytes((24, 160), (30, 30, 220)))
        return destination

    def fail_visual_llm(dataset: dict, config: dict, pack: dict) -> dict:
        raise TimeoutError("simulated visual timeout")

    monkeypatch.setattr(jobs, "download_s3_uri", fake_download)
    monkeypatch.setattr(jobs, "_run_visual_llm_analysis", fail_visual_llm)
    client = _FakeClient()
    job = {
        "id": "job_visual_timeout",
        "template": "analyze_dataset_visuals",
        "config": {
            "dataset_id": "dataset_1",
            "max_total_images": 2,
            "max_high_detail_images": 1,
            "max_image_bytes": 20_000,
            "max_total_bytes": 50_000,
            "image_size": 64,
            "seed": 11,
        },
    }

    jobs.run_job(client, job)

    payload = client.visual_analysis_payload
    assert client.completed_job_id == "job_visual_timeout"
    assert client.failed == []
    assert payload["provider"] == "openai"
    assert payload["model"] == "vision-model"
    assert payload["confidence"] == "low"
    assert payload["visual_traits"] == []
    assert payload["preprocessing_hypotheses"] == []
    assert "simulated visual timeout" not in payload["raw_output"]
    assert payload["input_context"]["repair"]["llm_unavailable"] is True
    assert any("Visual LLM unavailable" in item for item in payload["limitations"])


def test_visual_dataset_metadata_merges_active_safe_metadata_summary():
    dataset = {
        "id": "dataset_1",
        "name": "CUB",
        "profile": {
            "class_count": 1,
            "total_images": 11796,
            "class_distribution": {"CUB_200_2011": 11796},
            "metadata_summary": {"bbox_available": False},
        },
    }
    config = {
        "dataset_id": "dataset_1",
        "agent_safe_metadata_summary": {
            "class_count": 200,
            "sample_count": 11788,
            "bbox_annotation_count": 11788,
            "bbox_sample_count": 11788,
            "bbox_coverage_ratio": 1.0,
            "annotation_counts": {"bbox": 11788},
            "capabilities": {"bbox_annotations": True},
        },
    }
    manifest = {"images_available": 11796, "classes_total": 1}

    metadata = jobs._visual_dataset_metadata(dataset, config, manifest)

    assert metadata["class_count"] == 200
    assert metadata["total_images"] == 11796
    assert metadata["agent_safe_metadata_summary"]["bbox_annotation_count"] == 11788
    assert metadata["metadata_summary"]["bbox_sample_count"] == 11788
    assert metadata["profile"]["agent_safe_metadata_summary"]["capabilities"]["bbox_annotations"] is True


def _fake_visual_llm_analysis(dataset: dict, config: dict, pack: dict) -> dict:
    manifest = pack["sample_manifest"]
    analysis = {
        "schema_version": "dataset_visual_analysis_v1",
        "dataset_id": config["dataset_id"],
        "dataset_name": dataset["name"],
        "total_images": manifest["images_available"],
        "images_analyzed": manifest["images_analyzed"],
        "trigger_reason": "initial_profile",
        "confidence": "low",
        "coverage_report": {
            "selection_strategy": manifest["selection_strategy"],
            "selection_basis": manifest["selection_basis"],
            "images_available": manifest["images_available"],
            "images_analyzed": manifest["images_analyzed"],
            "classes_total": manifest["classes_total"],
            "classes_covered": manifest["classes_covered"],
            "class_coverage_ratio": manifest["class_coverage_ratio"],
            "per_class_counts": manifest["per_class_counts"],
            "hard_example_count": manifest["hard_example_count"],
            "edge_case_count": manifest["edge_case_count"],
            "high_detail_image_count": manifest["high_detail_image_count"],
            "limitations": manifest["limitations"],
        },
        "classes_to_watch": [],
        "visual_traits": [],
        "preprocessing_hypotheses": [],
        "cautions": [],
        "limitations": manifest["limitations"],
    }
    return {
        "analysis": analysis,
        "raw_output": json.dumps(analysis, sort_keys=True),
        "input_messages": [
            {"role": "system", "content": "evidence only"},
            {"role": "user", "content": "manifest only"},
        ],
        "provider": "fake_visual",
        "model": "fake-model",
        "agent_version": "fake-agent-v1",
        "prompt_version": "fake-prompt-v1",
    }


class _FakeClient:
    def __init__(self) -> None:
        self.visual_analysis_payload = {}
        self.completed_job_id = ""
        self.failed: list[tuple[str, str]] = []

    def get_dataset(self, dataset_id: str) -> dict:
        assert dataset_id == "dataset_1"
        return {
            "id": dataset_id,
            "name": "Pets",
            "storage_uri": "s3://bucket/dataset.zip",
        }

    def report_dataset_visual_analysis_result(self, dataset_id: str, payload: dict) -> dict:
        assert dataset_id == "dataset_1"
        self.visual_analysis_payload = payload
        return {"ok": True}

    def complete_job(self, job_id: str, mlflow_run_id: str = "") -> dict:
        self.completed_job_id = job_id
        return {"ok": True, "mlflow_run_id": mlflow_run_id}

    def fail_job(self, job_id: str, error: str) -> dict:
        self.failed.append((job_id, error))
        return {"ok": True}


def _image_bytes(size: tuple[int, int], color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()
