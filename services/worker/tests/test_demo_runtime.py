from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from worker.exporting.demo_runtime import handle_message, predict_from_request


class DemoRuntimeTests(unittest.TestCase):
    def test_classification_payload_maps_to_succeeded_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cat.jpg"
            image_path.write_bytes(b"image")
            inference_payload = {
                "status": "ok",
                "predicted_label": "cat",
                "confidence": 0.91,
                "top_k": [{"label": "cat", "confidence": 0.91}],
                "latency_ms": 12.5,
                "correct": True,
                "runtime": "onnx",
                "image_metadata": {"runtime": "onnx"},
            }

            with (
                patch("worker.exporting.demo_runtime._manifest_path", return_value=Path(temp_dir) / "manifest.json"),
                patch("worker.exporting.demo_runtime._demo_image_path", return_value=(image_path, "")),
                patch("worker.exporting.demo_runtime.run_demo_inference_from_manifest", return_value=inference_payload),
            ):
                result = predict_from_request(
                    {
                        "request_id": "prediction-1",
                        "image_uri": image_path.as_uri(),
                        "image_id": "heldout-cat",
                        "true_label": "cat",
                        "top_k": 3,
                        "image_metadata": {"split": "test"},
                    }
                )

            self.assertEqual(result["status"], "SUCCEEDED")
            self.assertEqual(result["predicted_label"], "cat")
            self.assertEqual(result["image_id"], "heldout-cat")
            self.assertEqual(result["true_label"], "cat")
            self.assertTrue(result["image_metadata"]["local_runtime"])
            self.assertEqual(result["image_metadata"]["runtime_host"], "mission_control_python")

    def test_detection_payload_preserves_yolo_detection_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "dog.jpg"
            image_path.write_bytes(b"image")
            detections = [
                {
                    "label": "dog",
                    "class_id": 1,
                    "confidence": 0.88,
                    "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                }
            ]
            inference_payload = {
                "status": "ok",
                "predicted_label": "dog",
                "confidence": 0.88,
                "top_k": [{"label": "dog", "confidence": 0.88}],
                "detections": detections,
                "detection_count": 1,
                "latency_ms": 18.0,
                "postprocess_latency_ms": 3.0,
                "runtime": "onnx",
                "image_metadata": {"task_type": "object_detection"},
            }

            with (
                patch("worker.exporting.demo_runtime._manifest_path", return_value=Path(temp_dir) / "manifest.json"),
                patch("worker.exporting.demo_runtime._demo_image_path", return_value=(image_path, "")),
                patch("worker.exporting.demo_runtime.run_demo_inference_from_manifest", return_value=inference_payload),
            ):
                result = predict_from_request(
                    {
                        "request_id": "prediction-2",
                        "image_uri": image_path.as_uri(),
                        "image_metadata": {"split": "test"},
                        "confidence_threshold": 0.25,
                        "iou_threshold": 0.7,
                        "max_detections": 100,
                    }
                )

            self.assertEqual(result["status"], "SUCCEEDED")
            self.assertEqual(result["image_metadata"]["detections"], detections)
            self.assertEqual(result["image_metadata"]["detection_count"], 1)
            self.assertEqual(result["image_metadata"]["postprocess_latency_ms"], 3.0)

    def test_runtime_unavailable_maps_to_terminal_failed_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cat.jpg"
            image_path.write_bytes(b"image")
            inference_payload = {
                "status": "pending",
                "error_code": "INFERENCE_DEPENDENCY_UNAVAILABLE",
                "error": "onnxruntime is not installed",
                "top_k": [],
            }

            with (
                patch("worker.exporting.demo_runtime._manifest_path", return_value=Path(temp_dir) / "manifest.json"),
                patch("worker.exporting.demo_runtime._demo_image_path", return_value=(image_path, "")),
                patch("worker.exporting.demo_runtime.run_demo_inference_from_manifest", return_value=inference_payload),
            ):
                result = predict_from_request({"request_id": "prediction-3", "image_uri": image_path.as_uri()})

            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["error_code"], "INFERENCE_DEPENDENCY_UNAVAILABLE")
            self.assertTrue(result["image_metadata"]["local_runtime_unavailable"])

    def test_dispose_clears_cached_runtime(self) -> None:
        with patch("worker.exporting.demo_runtime.clear_demo_inference_cache") as clear_cache:
            response = handle_message({"id": "dispose-1", "op": "dispose"})

        self.assertTrue(response["ok"])
        self.assertEqual(response["status"], "disposed")
        clear_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()