from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from worker.artifact_plan import (
    ArtifactPlanError,
    _artifact_plan_hash,
    load_artifact_plan,
    validate_artifact_plan,
)
from worker.exporting.artifacts import produce_champion_export_artifacts
from worker.training.modal_yolo import _export_yolo_detector_bundle


def artifact_plan(*formats: str, task: str = "image_classification", restricted: bool = True) -> dict:
    runtime = {"onnx": "onnxruntime", "torchscript": "torchscript", "pytorch": "pytorch"}
    plan = {
        "schema_version": "artifact_plan_v1",
        "artifacts": [
            {
                "format": artifact_format,
                "precision": "fp32",
                "runtime": runtime[artifact_format],
                "execution_provider": "cpu_execution_provider",
                "execution_requirements": ["cpu_compatible"],
            }
            for artifact_format in formats
        ],
        "fallback_formats": [],
        "automatic": True,
        "policy_restricted": restricted,
        "effective_policy_hash": "sha256:policy" if restricted else "",
        "required_worker_capabilities": {
            "policy_contract_versions": ["policy_contract_v1"] if restricted else [],
            "artifact_plan_versions": ["artifact_plan_v1"] if restricted else [],
        },
        "artifact_plan_hash": "",
    }
    plan["artifact_plan_hash"] = _artifact_plan_hash(plan)
    runner = "modal_ultralytics" if task == "object_detection" else "modal_torchvision"
    validate_artifact_plan(plan, task=task, runner=runner)
    return plan


class ArtifactPlanTests(unittest.TestCase):
    def test_server_plan_hash_is_cross_runtime_stable_and_tampering_fails_closed(self) -> None:
        plan = artifact_plan("onnx")
        self.assertEqual(
            plan["artifact_plan_hash"],
            "sha256:Psw64Nasw8unmzo7Tu37kJuWD7SkEoTu8BOpGkEpiD4",
        )
        config = {"execution_spec_v1": {"artifact_plan": plan}}
        loaded = load_artifact_plan(config, task="image_classification", runner="modal_torchvision")
        self.assertEqual(loaded["artifacts"][0]["format"], "onnx")
        loaded["artifacts"][0]["precision"] = "fp16"
        with self.assertRaisesRegex(ArtifactPlanError, "hash mismatch"):
            validate_artifact_plan(loaded, task="image_classification", runner="modal_torchvision")

    def test_onnx_only_plan_produces_no_torchscript_pytorch_or_safetensors(self) -> None:
        plan = artifact_plan("onnx")
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = produce_champion_export_artifacts(
                export_dir=Path(temp_dir),
                model_name="resnet18",
                class_names=["cat", "dog"],
                image_size=224,
                model=None,
                formats=("onnx",),
                artifact_plan=plan,
            )
        formats = {artifact["format"] for artifact in manifest["artifacts"]}
        self.assertIn("onnx", formats)
        self.assertTrue(formats.isdisjoint({"torchscript", "framework_native_checkpoint", "safetensors"}))
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "forbidden automatic artifacts"):
                produce_champion_export_artifacts(
                    export_dir=Path(temp_dir), model_name="resnet18", class_names=["cat"], image_size=224,
                    model=None, formats=("onnx", "torchscript", "framework_native"), artifact_plan=plan,
                )

    def test_no_policy_keeps_current_automatic_artifact_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = produce_champion_export_artifacts(
                export_dir=Path(temp_dir), model_name="resnet18", class_names=["cat"], image_size=224,
                model=None, formats=("onnx", "torchscript", "framework_native"),
            )
        formats = {artifact["format"] for artifact in manifest["artifacts"]}
        self.assertTrue({"onnx", "torchscript", "framework_native_checkpoint"}.issubset(formats))

    def test_yolo_onnx_failure_does_not_fall_back_to_pytorch_when_plan_forbids_it(self) -> None:
        plan = artifact_plan("onnx", task="object_detection")
        fake_ultralytics = types.ModuleType("ultralytics")

        class FakeYOLO:
            def __init__(self, _path: str) -> None:
                pass

            def export(self, **_kwargs):
                raise RuntimeError("forced ONNX failure")

        fake_ultralytics.YOLO = FakeYOLO
        fake_storage = types.ModuleType("worker.datasets.storage")
        fake_storage.upload_file_to_s3_uri = lambda *_args, **_kwargs: self.fail("forbidden fallback uploaded an artifact")
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics, "worker.datasets.storage": fake_storage}):
                result = _export_yolo_detector_bundle(
                    model_path=checkpoint, model_name="yolo11n.pt", class_names=["cat"], image_size=640,
                    model_profile={}, training_config={}, dataset={"id": "dataset_1"}, job_id="job_1",
                    artifact_plan=plan,
                )
        self.assertEqual(result["status"], "FAILED")
        self.assertTrue(any("YOLO_ONNX_EXPORT_FAILED" in value for value in result["validation_errors"]))
        self.assertNotEqual(result.get("format"), "pytorch")

    def test_unknown_required_worker_capability_fails_closed(self) -> None:
        plan = artifact_plan("onnx")
        plan["required_worker_capabilities"]["artifact_plan_versions"] = ["artifact_plan_v999"]
        plan["artifact_plan_hash"] = _artifact_plan_hash(plan)
        with self.assertRaisesRegex(ArtifactPlanError, "unknown required worker artifact capability"):
            validate_artifact_plan(plan, task="image_classification", runner="modal_torchvision")

    def test_manual_local_simulator_export_keeps_legacy_onnx_handoff(self) -> None:
        plan = artifact_plan("onnx", restricted=False)
        plan["automatic"] = False
        plan["artifact_plan_hash"] = _artifact_plan_hash(plan)
        validate_artifact_plan(plan, task="image_classification", runner="local_simulator")


if __name__ == "__main__":
    unittest.main()
