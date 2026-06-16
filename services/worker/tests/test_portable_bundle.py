from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from worker.exporting.portable_bundle import (
    build_portable_inference_manifest,
    write_portable_inference_bundle,
)


class PortableBundleTests(unittest.TestCase):
    def test_bundle_contains_model_contract_examples_and_parity_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.onnx"
            sidecar_path = root / "model.onnx.data"
            model_path.write_bytes(b"onnx bytes")
            sidecar_path.write_bytes(b"external tensor bytes")
            manifest = _manifest(model_path, sidecar_path)

            artifact = write_portable_inference_bundle(
                export_dir=root,
                manifest=manifest,
                requested_format="onnx",
                provenance={"export_job_id": "export_job_1", "source_export_id": "export_1"},
            )

            self.assertEqual(artifact["format"], "portable_inference_bundle")
            self.assertEqual(artifact["status"], "created")
            self.assertTrue(Path(artifact["path"]).exists())
            self.assertIn("model.onnx", artifact["contents"])
            self.assertIn("parity/expected_outputs.json", artifact["contents"])

            with zipfile.ZipFile(artifact["path"]) as archive:
                names = set(archive.namelist())
                self.assertIn("model.onnx", names)
                self.assertIn("model.onnx.data", names)
                self.assertIn("manifest.json", names)
                self.assertIn("requirements.txt", names)
                self.assertIn("README.md", names)
                self.assertIn("examples/python_onnxruntime.py", names)
                self.assertIn("examples/node_onnxruntime.mjs", names)
                self.assertIn("parity/expected_outputs.json", names)
                portable_manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                expected_outputs = json.loads(archive.read("parity/expected_outputs.json").decode("utf-8"))
                python_example = archive.read("examples/python_onnxruntime.py").decode("utf-8")
                node_example = archive.read("examples/node_onnxruntime.mjs").decode("utf-8")

            self.assertEqual(portable_manifest["schema_version"], "portable_inference_manifest_v1")
            self.assertEqual(portable_manifest["runtime"], "onnx")
            self.assertEqual(portable_manifest["model_artifact"], "model.onnx")
            self.assertEqual(portable_manifest["external_data"][0]["path"], "model.onnx.data")
            self.assertEqual(portable_manifest["class_labels"], ["cat", "dog"])
            self.assertEqual(expected_outputs["samples"][0]["onnx"]["predicted_label"], "dog")
            self.assertNotIn("model_express_runtime", python_example)
            self.assertNotIn("model_express_runtime", node_example)

    def test_bundle_manifest_is_runtime_neutral_contract(self) -> None:
        manifest = _manifest(Path("model.onnx"), Path("model.onnx.data"))
        portable_manifest = build_portable_inference_manifest(
            manifest["metadata"],
            manifest["artifacts"][0],
            manifest,
        )

        self.assertEqual(portable_manifest["schema_version"], "portable_inference_manifest_v1")
        self.assertEqual(portable_manifest["runtime"], "onnx")
        self.assertEqual(portable_manifest["task_type"], "image_classification")
        self.assertIn("preprocessing_contract", portable_manifest)
        self.assertIn("postprocessing_contract", portable_manifest)
        self.assertEqual(portable_manifest["parity"]["expected_outputs"], "parity/expected_outputs.json")

    def test_missing_onnx_artifact_returns_skipped_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = write_portable_inference_bundle(
                export_dir=Path(temp_dir),
                manifest={"schema_version": "champion_export_manifest_v1", "metadata": {}, "artifacts": []},
                requested_format="onnx",
            )

        self.assertEqual(artifact["status"], "skipped")
        self.assertEqual(artifact["error_code"], "PORTABLE_BUNDLE_MODEL_UNAVAILABLE")


def _manifest(model_path: Path, sidecar_path: Path) -> dict:
    return {
        "schema_version": "champion_export_manifest_v1",
        "status": "created",
        "metadata": {
            "model_kind": "classification",
            "task_type": "image_classification",
            "runtime": "onnx",
            "input_shape": [1, 3, 8, 8],
            "class_labels": ["cat", "dog"],
            "class_label_order_hash": "hash",
            "confidence_threshold_defaults": {"classification": {"top_k": 5}},
            "preprocessing_contract": {
                "config": {"resize_strategy": "squash", "normalization": "none", "crop_strategy": "none"}
            },
            "postprocessing_contract": {"model_output": "logits", "activation": "softmax"},
            "inference_contract": {
                "input": {"model_tensor_shape": [1, 3, 8, 8]},
                "preprocessing": {
                    "config": {"resize_strategy": "squash", "normalization": "none", "crop_strategy": "none"}
                },
                "postprocessing": {"model_output": "logits", "activation": "softmax"},
            },
            "export_self_test": {
                "status": "passed",
                "runtime": "onnx",
                "sample_count": 1,
                "class_label_order_hash": "hash",
                "preprocessing_contract_hash": "preprocess-hash",
                "samples": [
                    {
                        "sample_id": "heldout:dog",
                        "onnx": {"predicted_label": "dog", "top_k": [{"label": "dog", "confidence": 0.9}]},
                    }
                ],
            },
        },
        "artifacts": [
            {
                "format": "onnx",
                "status": "created",
                "path": str(model_path),
                "external_data": [{"path": "model.onnx.data", "artifact_path": str(sidecar_path), "bytes": 21}],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
