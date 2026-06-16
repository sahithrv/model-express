from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worker.champion_jobs import (
    _build_torchvision_model,
    _demo_image_path,
    _validated_exemplar_payload,
    run_champion_demo_prediction_job,
    run_export_champion_job,
)
from worker.jobs import run_job


class FakeClient:
    def __init__(self) -> None:
        self.export_results: list[tuple[str, dict]] = []
        self.prediction_results: list[tuple[str, dict]] = []
        self.completed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.datasets: dict[str, dict] = {}

    def get_dataset(self, dataset_id: str) -> dict:
        return self.datasets[dataset_id]

    def report_champion_export_result(self, job_id: str, result: dict) -> dict:
        self.export_results.append((job_id, result))
        return {"ok": True}

    def report_champion_demo_prediction_result(self, job_id: str, result: dict) -> dict:
        self.prediction_results.append((job_id, result))
        return {"ok": True}

    def complete_job(self, job_id: str, mlflow_run_id: str = "") -> dict:
        self.completed.append(job_id)
        return {"ok": True}

    def fail_job(self, job_id: str, error: str) -> dict:
        self.failed.append((job_id, error))
        return {"ok": True}


class RaisingExportResultClient(FakeClient):
    def report_champion_export_result(self, job_id: str, result: dict) -> dict:
        super().report_champion_export_result(job_id, result)
        raise RuntimeError("export result callback failed")


class ChampionJobTests(unittest.TestCase):
    def test_export_reports_ready_only_when_existing_artifact_is_copied(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            export_root = root / "exports"
            source = export_root / "controlled" / "source.onnx"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"real onnx bytes")
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {
                    "format": "onnx",
                    "champion_job_id": "train_1",
                    "artifact_path": str(source),
                    "model": "mobilenet_v3_small",
                    "class_names": ["cat", "dog"],
                    "image_size": 64,
                },
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(export_root)}):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["format"], "onnx")
            self.assertTrue(result["artifact_uri"].startswith("file://"))
            self.assertTrue(result["artifact_uri"].endswith("/model.onnx"))
            self.assertIn("portable_bundle_uri", result["metadata"])
            self.assertTrue(result["metadata"]["portable_bundle_uri"].endswith("/portable_inference_bundle.zip"))
            self.assertTrue((export_root / "train_1" / "onnx" / "job_export" / "model.onnx").exists())
            manifest = result["metadata"]["manifest"]
            self.assertEqual(manifest["metadata"]["provenance"]["schema_version"], "worker_artifact_provenance_v1")
            self.assertEqual(manifest["artifacts"][0]["provenance"]["source"], "worker_controlled_copy")
            self.assertTrue(
                any(
                    artifact.get("format") == "portable_inference_bundle" and artifact.get("status") == "created"
                    for artifact in manifest["artifacts"]
                )
            )
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_export_result_callback_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = RaisingExportResultClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {"format": "not_a_real_format"},
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(Path(temp_dir) / "exports")}):
                with self.assertRaisesRegex(RuntimeError, "export result callback failed"):
                    run_export_champion_job(client, job)

            self.assertEqual(len(client.export_results), 1)
            self.assertEqual(client.export_results[0][0], "job_export")
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_export_rejects_absolute_artifact_path_outside_worker_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "outside" / "source.onnx"
            source.parent.mkdir()
            source.write_bytes(b"not worker owned")
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {
                    "format": "onnx",
                    "champion_job_id": "train_1",
                    "artifact_path": str(source),
                    "model": "mobilenet_v3_small",
                    "class_names": ["cat", "dog"],
                    "image_size": 64,
                },
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(root / "exports")}):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["artifact_uri"], "")
            self.assertIn("ARTIFACT_SOURCE_REJECTED", result["error"])
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_export_rejects_file_checkpoint_uri_outside_worker_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint = root / "outside" / "source.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"checkpoint bytes")
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {
                    "format": "torchscript",
                    "checkpoint_uri": checkpoint.resolve().as_uri(),
                    "model": "mobilenet_v3_small",
                    "class_names": ["cat", "dog"],
                    "image_size": 64,
                },
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(root / "exports")}):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["artifact_uri"], "")
            self.assertIn("ARTIFACT_SOURCE_REJECTED", result["error"])
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_onnx_export_does_not_relabel_checkpoint_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.pt"
            source.write_bytes(b"checkpoint bytes")
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {
                    "format": "onnx",
                    "champion_job_id": "train_1",
                    "artifact_uri": str(source),
                    "model": "mobilenet_v3_small",
                    "class_names": ["cat", "dog"],
                    "image_size": 64,
                },
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(root / "exports")}):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["format"], "onnx")
            self.assertEqual(result["artifact_uri"], "")
            self.assertIn("ARTIFACT_SOURCE_REJECTED", result["error"])
            self.assertFalse((root / "exports" / "train_1" / "onnx" / "job_export" / "model.onnx").exists())
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_onnx_export_converts_framework_checkpoint_when_available(self) -> None:
        try:
            import onnxscript  # noqa: F401
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"ONNX export dependencies are unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            export_root = root / "exports"
            model = _build_torchvision_model(
                model_name="mobilenet_v3_small",
                class_count=2,
                pretrained=False,
                freeze_backbone=True,
                fine_tune_strategy="head_only",
                dropout=0.0,
            )
            checkpoint = export_root / "checkpoints" / "source.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "model": "mobilenet_v3_small",
                        "class_labels": ["cat", "dog"],
                        "training_config": {
                            "freeze_backbone": True,
                            "fine_tune_strategy": "head_only",
                            "dropout": 0.0,
                        },
                    },
                },
                checkpoint,
            )
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {
                    "format": "onnx",
                    "champion_job_id": "train_1",
                    "source_artifact_uri": str(checkpoint),
                    "model": "mobilenet_v3_small",
                    "class_names": ["cat", "dog"],
                    "image_size": 32,
                },
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(export_root)}):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["format"], "onnx")
            self.assertTrue((export_root / "train_1" / "onnx" / "job_export" / "model.onnx").exists())
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_export_rejects_checkpoint_class_label_order_mismatch(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"torch is unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            export_root = root / "exports"
            model = _build_torchvision_model(
                model_name="mobilenet_v3_small",
                class_count=2,
                pretrained=False,
                freeze_backbone=True,
                fine_tune_strategy="head_only",
                dropout=0.0,
            )
            checkpoint = export_root / "checkpoints" / "source.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metadata": {
                        "model": "mobilenet_v3_small",
                        "class_labels": ["dog", "cat"],
                    },
                },
                checkpoint,
            )
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {
                    "format": "torchscript",
                    "champion_job_id": "train_1",
                    "source_artifact_uri": str(checkpoint),
                    "model": "mobilenet_v3_small",
                    "class_names": ["cat", "dog"],
                    "image_size": 32,
                },
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(export_root)}):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "FAILED")
            self.assertIn("CLASS_LABEL_ORDER_MISMATCH", result["error"])
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_export_rejects_unsafe_pickle_checkpoint_without_execution(self) -> None:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"torch is unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            export_root = root / "exports"
            checkpoint = export_root / "checkpoints" / "unsafe.pt"
            marker = root / "unsafe_executed.txt"
            checkpoint.parent.mkdir(parents=True)

            class UnsafePayload:
                def __reduce__(self):
                    return (
                        eval,
                        (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",),
                    )

            torch.save({"state_dict": UnsafePayload()}, checkpoint)
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {
                    "format": "torchscript",
                    "checkpoint_path": str(checkpoint),
                    "model": "mobilenet_v3_small",
                    "class_names": ["cat", "dog"],
                    "image_size": 32,
                },
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(export_root)}):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "FAILED")
            self.assertIn("CHECKPOINT_UNSAFE_PICKLE_REJECTED", result["error"])
            self.assertFalse(marker.exists())
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_export_fails_when_declared_source_artifact_is_missing_after_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            export_root = root / "exports"
            missing_source = export_root / "controlled" / "missing.onnx"
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {
                    "format": "onnx",
                    "champion_job_id": "train_1",
                    "artifact_path": str(missing_source),
                    "model": "mobilenet_v3_small",
                    "class_names": ["cat", "dog"],
                    "image_size": 64,
                },
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(export_root)}):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "FAILED")
            self.assertIn("ARTIFACT_NOT_FOUND", result["error"])
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_export_fails_object_detection_without_trained_detector_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {
                    "format": "onnx",
                    "task_type": "object_detection",
                    "model": "yolov8n",
                    "class_names": ["person"],
                    "image_size": 64,
                },
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(Path(temp_dir) / "exports")}):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "FAILED")
            self.assertIn("SOURCE_ARTIFACT_REQUIRED", result["error"])
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_export_fails_conversion_dependency_blocker_when_no_source_artifact_exists(self) -> None:
        def fake_export_artifacts(**kwargs):
            return {
                "schema_version": "champion_export_manifest_v1",
                "status": "pending_dependencies",
                "metadata": {"format": "onnx"},
                "artifacts": [
                    {
                        "format": "onnx",
                        "status": "skipped",
                        "error_code": "ONNXSCRIPT_UNAVAILABLE",
                        "error": "onnxscript is not installed",
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {"format": "onnx", "class_names": ["cat"], "image_size": 32},
            }

            with (
                patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(Path(temp_dir) / "exports")}),
                patch("worker.champion_jobs._build_architecture_export_fallback", return_value=(object(), [])),
                patch("worker.champion_jobs.produce_champion_export_artifacts", fake_export_artifacts),
            ):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "FAILED")
            self.assertIn("ONNXSCRIPT_UNAVAILABLE", result["error"])
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_export_without_artifact_reports_pending_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            client = FakeClient()
            job = {
                "id": "job_export",
                "template": "export_champion",
                "config": {"format": "torchscript", "class_names": ["cat"], "image_size": 32},
            }

            with (
                patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(Path(temp_dir) / "exports")}),
                patch("worker.champion_jobs._build_architecture_export_fallback", return_value=(None, [])),
            ):
                run_export_champion_job(client, job)

            _, result = client.export_results[0]
            self.assertEqual(result["status"], "PENDING_ARTIFACT")
            self.assertEqual(result["artifact_uri"], "")
            self.assertIn("MODEL_UNAVAILABLE", result["error"])
            self.assertEqual(client.completed, [])
            self.assertEqual(client.failed, [])

    def test_demo_prediction_reports_runtime_unavailable_without_manifest(self) -> None:
        client = FakeClient()
        job = {
            "id": "job_predict",
            "template": "champion_demo_prediction",
            "config": {"image_uri": "missing.jpg", "true_label": "cat"},
        }

        run_champion_demo_prediction_job(client, job)

        _, result = client.prediction_results[0]
        self.assertEqual(result["status"], "RUNTIME_UNAVAILABLE")
        self.assertEqual(result["error_code"], "MANIFEST_NOT_CONFIGURED")
        self.assertEqual(client.completed, [])
        self.assertEqual(client.failed, [])

    def test_demo_image_path_materializes_inline_data_uri(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_uri = (
                "data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
            )

            with patch.dict("os.environ", {"WORKER_DEMO_IMAGE_ROOT": str(Path(temp_dir) / "demo_images")}):
                path, error = _demo_image_path({"image_uri": image_uri}, "job_inline")

            self.assertEqual(error, "")
            self.assertIsNotNone(path)
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".png")

    def test_demo_image_path_prefers_original_artifact_over_thumbnail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            downloaded = []
            original_uri = "s3://bucket/model-express/artifacts/job_1/heldout_demo_images/cat.png"
            thumbnail_uri = "data:image/jpeg;base64,BBBB"

            def fake_download(storage_uri: str, destination: Path) -> Path:
                downloaded.append((storage_uri, destination))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"original image bytes")
                return destination

            with (
                patch("worker.champion_jobs.download_s3_uri", fake_download),
                patch.dict("os.environ", {"WORKER_DEMO_IMAGE_ROOT": str(Path(temp_dir) / "demo_images")}),
            ):
                path, error = _demo_image_path(
                    {
                        "image_uri": thumbnail_uri,
                        "image_metadata": {"source_artifact_uri": original_uri},
                    },
                    "job_predict",
                )

            self.assertEqual(error, "")
            self.assertIsNotNone(path)
            self.assertEqual(downloaded[0][0], original_uri)
            self.assertTrue(path.exists())

    def test_dispatch_rejects_unknown_templates_instead_of_faking_success(self) -> None:
        client = FakeClient()
        with self.assertRaises(ValueError):
            run_job(client, {"id": "job_unknown", "template": "not_real", "config": {}})

    def test_exemplar_payload_enforces_image_and_byte_caps(self) -> None:
        pack = {
            "status": "created",
            "visual_exemplars": [
                {"id": "a", "class_name": "cat", "path": "a.jpg", "source_path": "a.png", "bytes": 10},
                {"id": "b", "class_name": "dog", "path": "b.jpg", "source_path": "b.png", "bytes": 50},
                {"id": "c", "class_name": "dog", "path": "c.jpg", "source_path": "c.png", "bytes": 10},
            ],
        }
        payload = _validated_exemplar_payload(
            pack,
            {
                "images_per_class": 2,
                "max_total_images": 2,
                "max_image_bytes": 20,
                "max_total_bytes": 25,
                "image_size": 160,
                "quality": 75,
            },
        )

        self.assertEqual(payload["status"], "created")
        self.assertEqual([item["id"] for item in payload["visual_exemplars"]], ["a", "c"])
        self.assertEqual(payload["summary"]["total_bytes"], 20)
        self.assertEqual(payload["profile_patch"]["demo_images"], payload["visual_exemplars"])


if __name__ == "__main__":
    unittest.main()
