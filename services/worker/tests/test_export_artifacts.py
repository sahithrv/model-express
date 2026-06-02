from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worker.exporting.artifacts import (
    load_export_manifest,
    produce_champion_export_artifacts,
    produce_existing_champion_export_manifest,
)


class ExportArtifactTests(unittest.TestCase):
    def test_manifest_is_written_and_artifacts_are_pending_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = produce_champion_export_artifacts(
                export_dir=Path(temp_dir),
                model_name="mobilenet_v3_small",
                class_names=["cat", "dog"],
                image_size=128,
                model=None,
                formats=("framework_native", "torchscript", "onnx"),
            )

            self.assertEqual(manifest["schema_version"], "champion_export_manifest_v1")
            self.assertEqual(manifest["status"], "pending_dependencies")
            self.assertTrue(Path(manifest["manifest_path"]).exists())
            self.assertEqual(
                {artifact["format"] for artifact in manifest["artifacts"]},
                {"framework_native_checkpoint", "torchscript", "onnx"},
            )
            self.assertTrue(all(artifact["status"] == "skipped" for artifact in manifest["artifacts"]))

            loaded = load_export_manifest(Path(manifest["manifest_path"]))
            self.assertEqual(loaded["metadata"]["class_labels"], ["cat", "dog"])

    def test_framework_native_checkpoint_is_created_when_torch_model_is_available(self) -> None:
        try:
            import torch
            from torch import nn
        except Exception as exc:  # pragma: no cover - depends on optional local deps
            raise unittest.SkipTest(f"torch is unavailable: {exc}") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 2))
            manifest = produce_champion_export_artifacts(
                export_dir=Path(temp_dir),
                model_name="tiny_linear",
                class_names=["cat", "dog"],
                image_size=8,
                model=model,
                formats=("framework_native",),
            )

            artifact = manifest["artifacts"][0]
            self.assertEqual(artifact["status"], "created")
            self.assertTrue(Path(artifact["path"]).exists())
            payload = torch.load(artifact["path"], map_location="cpu", weights_only=False)
            self.assertIn("state_dict", payload)
            self.assertEqual(payload["metadata"]["input_shape"], [1, 3, 8, 8])

    def test_existing_onnx_copy_preserves_external_data_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_dir = root / "source"
            source_dir.mkdir()
            source = source_dir / "model.onnx"
            sidecar = source_dir / "model.onnx.data"
            source.write_bytes(b"onnx bytes")
            sidecar.write_bytes(b"external tensor bytes")

            manifest = produce_existing_champion_export_manifest(
                export_dir=root / "export",
                source_artifact_path=source,
                artifact_format="onnx",
                model_name="tiny",
                class_names=["cat", "dog"],
                image_size=8,
            )

            artifact = manifest["artifacts"][0]
            copied_sidecar = root / "export" / "model.onnx.data"
            self.assertEqual(artifact["status"], "created")
            self.assertTrue((root / "export" / "model.onnx").exists())
            self.assertTrue(copied_sidecar.exists())
            self.assertEqual(artifact["external_data"][0]["path"], "model.onnx.data")
            self.assertEqual(Path(artifact["external_data"][0]["artifact_path"]), copied_sidecar)


if __name__ == "__main__":
    unittest.main()
