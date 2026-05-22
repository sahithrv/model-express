from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from worker.exporting.artifacts import produce_champion_export_artifacts
from worker.exporting.inference import run_demo_inference_from_manifest


class DemoInferenceTests(unittest.TestCase):
    def test_inference_returns_pending_when_torchscript_artifact_is_missing(self) -> None:
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
            self.assertEqual(payload["error_code"], "TORCHSCRIPT_ARTIFACT_UNAVAILABLE")
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


if __name__ == "__main__":
    unittest.main()
