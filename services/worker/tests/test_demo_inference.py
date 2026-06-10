from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from worker.exporting.artifacts import produce_champion_export_artifacts
from worker.exporting.inference import _postprocess_detection_outputs, run_demo_inference_from_manifest
from worker.exporting.preprocessing import prepare_image_for_inference


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
