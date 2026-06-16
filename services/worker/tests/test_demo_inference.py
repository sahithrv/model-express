from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from worker.exporting.artifacts import produce_champion_export_artifacts
from worker.exporting.inference import _postprocess_detection_outputs, _resolve_artifact_path, run_demo_inference_from_manifest
from worker.exporting.preprocessing import prepare_image_for_inference, preprocessing_parity_diagnostics


class DemoInferenceTests(unittest.TestCase):
    def test_preserve_aspect_pad_preprocessing_matches_manifest_contract(self) -> None:
        image = Image.new("RGB", (8, 4), (255, 0, 0))
        prepared = prepare_image_for_inference(
            Image,
            image,
            {
                "input_shape": [1, 3, 8, 8],
                "preprocessing": {
                    "resize_strategy": "preserve_aspect_pad",
                    "crop_strategy": "none",
                    "normalization": "none",
                },
            },
        )

        self.assertEqual(prepared.size, (8, 8))
        self.assertEqual(prepared.getpixel((0, 0)), (0, 0, 0))
        self.assertEqual(prepared.getpixel((0, 2)), (255, 0, 0))

    def test_bbox_preprocessing_requires_runtime_metadata_for_parity(self) -> None:
        image = Image.new("RGB", (10, 10), (255, 0, 0))
        for x in range(5, 10):
            for y in range(10):
                image.putpixel((x, y), (0, 0, 255))
        metadata = {
            "input_shape": [1, 3, 4, 4],
            "preprocessing": {
                "resize_strategy": "squash",
                "crop_strategy": "bbox_crop_if_available",
                "normalization": "none",
            },
        }

        diagnostics = preprocessing_parity_diagnostics(metadata, {})
        self.assertEqual(diagnostics["status"], "unsafe")
        self.assertEqual(diagnostics["issues"][0]["code"], "BBOX_METADATA_REQUIRED")
        with self.assertRaisesRegex(ValueError, "BBOX_METADATA_REQUIRED"):
            prepare_image_for_inference(Image, image, metadata, strict_metadata=True)

        prepared = prepare_image_for_inference(
            Image,
            image,
            metadata,
            image_metadata={"bbox": {"x1": 5, "y1": 0, "x2": 10, "y2": 10}},
            strict_metadata=True,
        )
        self.assertEqual(prepared.size, (4, 4))
        self.assertGreater(prepared.getpixel((3, 2))[2], prepared.getpixel((3, 2))[0])

    def test_legacy_heldout_thumbnail_metadata_is_not_parity_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "demo.jpg"
            Image.new("RGB", (16, 16), (255, 0, 0)).save(image_path)
            manifest = {
                "schema_version": "champion_export_manifest_v1",
                "status": "created",
                "metadata": {
                    "model": "tiny",
                    "class_labels": ["cat", "dog"],
                    "input_shape": [1, 3, 16, 16],
                    "preprocessing": {"normalization": "none"},
                },
                "artifacts": [],
            }

            payload = run_demo_inference_from_manifest(
                manifest=manifest,
                image_path=image_path,
                true_label="cat",
                image_metadata={"source": "heldout_test"},
            )

            self.assertEqual(payload["status"], "pending")
            self.assertEqual(payload["error_code"], "HELDOUT_IMAGE_SOURCE_UNVERIFIED")
            self.assertEqual(payload["image_metadata"]["parity_status"], "unsafe")

    def test_inference_returns_pending_when_supported_artifact_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            manifest = produce_champion_export_artifacts(
                export_dir=export_dir,
                model_name="mobilenet_v3_small",
                class_names=["cat", "dog"],
                image_size=16,
                model=None,
                formats=("framework_native",),
            )
            image_path = export_dir / "demo.jpg"
            Image.new("RGB", (16, 16), (255, 0, 0)).save(image_path)

            payload = run_demo_inference_from_manifest(
                manifest_path=Path(manifest["manifest_path"]),
                image_path=image_path,
                true_label="cat",
            )

            self.assertEqual(payload["schema_version"], "champion_demo_prediction_v1")
            self.assertEqual(payload["status"], "pending")
            self.assertEqual(payload["error_code"], "MODEL_ARTIFACT_UNAVAILABLE")
            self.assertEqual(payload["top_k"], [])

    def test_torchscript_inference_returns_ranked_payload_when_available(self) -> None:
        try:
            import torch
            from torch import nn
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"torch is unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
            with torch.no_grad():
                model[1].weight.zero_()
                model[1].bias[:] = torch.tensor([0.1, 1.1])
            manifest = produce_champion_export_artifacts(
                export_dir=export_dir,
                model_name="tiny_linear",
                class_names=["cat", "dog"],
                image_size=4,
                model=model,
                preprocessing={"normalization": "none"},
                formats=("torchscript",),
            )
            image_path = export_dir / "demo.jpg"
            Image.new("RGB", (4, 4), (20, 20, 20)).save(image_path)

            payload = run_demo_inference_from_manifest(
                manifest_path=Path(manifest["manifest_path"]),
                image_path=image_path,
                true_label="dog",
            )

            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["predicted_label"], "dog")
            self.assertTrue(payload["correct"])
            self.assertEqual([item["label"] for item in payload["top_k"]], ["dog", "cat"])
            self.assertGreater(payload["latency_ms"], 0)
            self.assertGreater(payload["latency_breakdown_ms"]["preprocess"], 0)
            self.assertGreater(payload["latency_breakdown_ms"]["streaming_total"], 0)

            cached_payload = run_demo_inference_from_manifest(
                manifest_path=Path(manifest["manifest_path"]),
                image_path=image_path,
                true_label="dog",
            )
            self.assertTrue(cached_payload["latency_breakdown_ms"]["model_cache_hit"])

    def test_onnx_classification_inference_returns_ranked_payload_when_available(self) -> None:
        try:
            import onnxruntime  # noqa: F401
            import onnxscript  # noqa: F401
            import torch
            from torch import nn
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"ONNX inference dependencies are unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
            with torch.no_grad():
                model[1].weight.zero_()
                model[1].bias[:] = torch.tensor([0.1, 1.1])
            manifest = produce_champion_export_artifacts(
                export_dir=export_dir,
                model_name="tiny_linear",
                class_names=["cat", "dog"],
                image_size=4,
                model=model,
                preprocessing={"normalization": "none"},
                formats=("onnx",),
            )
            image_path = export_dir / "demo.jpg"
            Image.new("RGB", (4, 4), (20, 20, 20)).save(image_path)

            payload = run_demo_inference_from_manifest(
                manifest_path=Path(manifest["manifest_path"]),
                image_path=image_path,
                true_label="dog",
                image_metadata={"parity_safe": True, "demo_source_type": "custom_original_bytes"},
            )

            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["runtime"], "onnx")
            self.assertEqual(payload["predicted_label"], "dog")
            self.assertEqual([item["label"] for item in payload["top_k"]], ["dog", "cat"])
            self.assertEqual(payload["image_metadata"]["parity_status"], "ok")

    def test_s3_onnx_resolution_downloads_inferred_external_data_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original_cwd = Path.cwd()
            calls: list[str] = []

            def fake_download(uri: str, destination: Path) -> Path:
                calls.append(uri)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"external tensor bytes" if uri.endswith(".data") else b"onnx bytes")
                return destination

            try:
                os.chdir(root)
                with patch("worker.exporting.inference.download_s3_uri", fake_download):
                    model_path = _resolve_artifact_path(
                        None,
                        {
                            "format": "onnx",
                            "status": "created",
                            "path": "s3://bucket/model-express/artifacts/job_1/model.onnx",
                        },
                    )
            finally:
                os.chdir(original_cwd)

            self.assertIsNotNone(model_path)
            assert model_path is not None
            self.assertTrue(model_path.exists())
            self.assertEqual((model_path.parent / "model.onnx.data").read_bytes(), b"external tensor bytes")
            self.assertEqual(
                calls,
                [
                    "s3://bucket/model-express/artifacts/job_1/model.onnx",
                    "s3://bucket/model-express/artifacts/job_1/model.onnx.data",
                ],
            )

    def test_framework_native_checkpoint_inference_returns_ranked_payload(self) -> None:
        try:
            import torch
            from worker.training.modal_app import _build_model
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"torchvision model builder is unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            export_dir = Path(temp_dir)
            model = _build_model("mobilenet_v3_small", 2, pretrained=False, freeze_backbone=False)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.classifier[-1].bias[:] = torch.tensor([0.1, 1.1])
            manifest = produce_champion_export_artifacts(
                export_dir=export_dir,
                model_name="mobilenet_v3_small",
                class_names=["cat", "dog"],
                image_size=32,
                model=model,
                preprocessing={"normalization": "none"},
                training_config={"fine_tune_strategy": "full"},
                formats=("framework_native",),
            )
            image_path = export_dir / "demo.jpg"
            Image.new("RGB", (32, 32), (20, 20, 20)).save(image_path)

            payload = run_demo_inference_from_manifest(
                manifest_path=Path(manifest["manifest_path"]),
                image_path=image_path,
                true_label="dog",
            )

            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["runtime"], "framework_native_checkpoint")
            self.assertEqual(payload["predicted_label"], "dog")
            self.assertTrue(payload["correct"])

    def test_inference_rejects_file_artifact_uri_outside_worker_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            outside = root / "outside" / "model.torchscript.pt"
            outside.parent.mkdir()
            outside.write_bytes(b"not worker owned")
            image_path = root / "demo.jpg"
            Image.new("RGB", (4, 4), (20, 20, 20)).save(image_path)
            manifest = {
                "schema_version": "champion_export_manifest_v1",
                "status": "created",
                "metadata": {"class_labels": ["cat", "dog"]},
                "artifacts": [
                    {
                        "format": "torchscript",
                        "status": "created",
                        "path": outside.resolve().as_uri(),
                    }
                ],
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(root / "exports")}):
                payload = run_demo_inference_from_manifest(
                    manifest=manifest,
                    image_path=image_path,
                    true_label="dog",
                )

            self.assertEqual(payload["status"], "pending")
            self.assertEqual(payload["error_code"], "MODEL_ARTIFACT_NOT_FOUND")

    def test_framework_native_checkpoint_rejects_unsafe_pickle_without_execution(self) -> None:
        try:
            import torch
            from worker.training.modal_app import _build_model  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"torchvision model builder is unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            export_root = root / "exports"
            checkpoint = export_root / "job" / "model.pt"
            marker = root / "unsafe_executed.txt"
            checkpoint.parent.mkdir(parents=True)

            class UnsafePayload:
                def __reduce__(self):
                    return (
                        eval,
                        (f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')",),
                    )

            torch.save({"state_dict": UnsafePayload()}, checkpoint)
            image_path = root / "demo.jpg"
            Image.new("RGB", (32, 32), (20, 20, 20)).save(image_path)
            manifest = {
                "schema_version": "champion_export_manifest_v1",
                "status": "created",
                "metadata": {
                    "model": "mobilenet_v3_small",
                    "class_labels": ["cat", "dog"],
                    "input_shape": [1, 3, 32, 32],
                    "training_config": {"fine_tune_strategy": "full"},
                    "preprocessing": {"normalization": "none"},
                },
                "artifacts": [
                    {
                        "format": "framework_native_checkpoint",
                        "status": "created",
                        "path": str(checkpoint),
                    }
                ],
            }

            with patch.dict("os.environ", {"WORKER_EXPORT_ROOT": str(export_root)}):
                payload = run_demo_inference_from_manifest(
                    manifest=manifest,
                    image_path=image_path,
                    true_label="dog",
                )

            self.assertEqual(payload["status"], "pending")
            self.assertEqual(payload["error_code"], "INFERENCE_FAILED")
            self.assertIn("CHECKPOINT_UNSAFE_PICKLE_REJECTED", payload["error"])
            self.assertFalse(marker.exists())

    def test_yolo_detection_postprocess_decodes_rows_and_applies_nms(self) -> None:
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"numpy is unavailable: {exc}") from exc

        outputs = {
            "output0": np.array(
                [
                    [
                        [4.0, 4.2, 10.0],
                        [4.0, 4.2, 10.0],
                        [4.0, 4.0, 3.0],
                        [4.0, 4.0, 3.0],
                        [0.92, 0.88, 0.05],
                        [0.05, 0.04, 0.91],
                    ]
                ],
                dtype=np.float32,
            )
        }

        detections = _postprocess_detection_outputs(
            outputs=outputs,
            class_labels=["creeper", "zombie"],
            input_size=(8, 8),
            original_size=(8, 8),
            thresholds={"confidence_threshold": 0.2, "iou_threshold": 0.5, "max_detections": 10},
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "creeper")
        self.assertGreater(detections[0]["confidence"], 0.9)
        self.assertAlmostEqual(detections[0]["box"]["x"], 0.25, places=2)
        self.assertAlmostEqual(detections[0]["box"]["width"], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()
