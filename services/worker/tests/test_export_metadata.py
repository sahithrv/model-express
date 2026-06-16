from __future__ import annotations

import unittest

from worker.exporting.metadata import (
    build_champion_export_metadata,
    build_demo_detection_payload,
    build_demo_prediction_payload,
)


class ExportMetadataTests(unittest.TestCase):
    def test_champion_export_metadata_includes_demo_ready_shape(self) -> None:
        metadata = build_champion_export_metadata(
            model_name="mobilenet_v3_small",
            class_names=["cat", "dog"],
            image_size=224,
            preprocessing={"normalization": "dataset"},
            model_profile={"estimated_latency_ms": 8.2},
            training_config={"epochs": 3},
            artifact_uri="s3://bucket/model.pt",
        )

        self.assertEqual(metadata["schema_version"], "champion_export_metadata_v1")
        self.assertEqual(metadata["model_kind"], "classification")
        self.assertEqual(metadata["task_type"], "image_classification")
        self.assertEqual(metadata["runtime"], "onnx")
        self.assertEqual(metadata["default_runtime"], "onnx")
        self.assertEqual(metadata["input_shape"], [1, 3, 224, 224])
        self.assertEqual(metadata["class_labels"], ["cat", "dog"])
        self.assertEqual(metadata["class_index_order"], [{"index": 0, "label": "cat"}, {"index": 1, "label": "dog"}])
        self.assertTrue(metadata["class_label_order_hash"])
        self.assertEqual(metadata["class_count"], 2)
        self.assertEqual(metadata["artifact_uri"], "s3://bucket/model.pt")
        self.assertEqual(metadata["confidence_threshold_defaults"]["classification"]["top_k"], 5)
        self.assertEqual(metadata["latency_budget"]["estimated_latency_ms"], 8.2)
        self.assertEqual(metadata["export_self_test"]["status"], "not_run")
        self.assertFalse(metadata["export_status"]["export_verified"])
        self.assertIn("preprocessing_contract", metadata)
        self.assertIn("postprocessing_contract", metadata)
        contract = metadata["inference_contract"]
        self.assertEqual(contract["schema_version"], "model_express_inference_contract_v1")
        self.assertEqual(contract["model_kind"], "classification")
        self.assertEqual(contract["task_type"], "image_classification")
        self.assertEqual(contract["input"]["model_tensor_shape"], [1, 3, 224, 224])
        self.assertIn("preprocessing", contract)
        self.assertIn("postprocessing", contract)
        self.assertEqual(contract["postprocessing"]["model_output"], "logits")
        self.assertEqual(contract["postprocessing"]["activation"], "softmax")

    def test_yolo_detector_metadata_uses_detection_contract(self) -> None:
        metadata = build_champion_export_metadata(
            model_name="yolo11n.pt",
            class_names=["person", "helmet"],
            image_size=640,
            preprocessing={},
            model_profile={
                "task_type": "object_detection",
                "model_kind": "ultralytics_yolo_detector",
                "runtime": "onnx",
                "input_shape": [1, 3, 640, 640],
                "estimated_latency_ms": 8.0,
                "latency_p50_ms": 6.5,
                "latency_p95_ms": 12.25,
                "estimated_throughput_images_per_second": 125.0,
                "confidence_threshold": 0.31,
                "iou_threshold": 0.62,
                "max_detections": 50,
            },
            training_config={"task_type": "object_detection"},
        )

        self.assertEqual(metadata["model_kind"], "detection")
        self.assertEqual(metadata["task_type"], "object_detection")
        self.assertEqual(metadata["runtime"], "onnx")
        self.assertEqual(metadata["input_shape"], [1, 3, 640, 640])
        self.assertEqual(metadata["class_labels"], ["person", "helmet"])
        self.assertEqual(metadata["confidence_threshold_defaults"]["detection"]["confidence_threshold"], 0.31)
        self.assertEqual(metadata["confidence_threshold_defaults"]["detection"]["iou_threshold"], 0.62)
        self.assertEqual(metadata["confidence_threshold_defaults"]["detection"]["max_detections"], 50)
        self.assertEqual(metadata["latency_budget"]["estimated_p95_latency_ms"], 12.25)
        self.assertEqual(metadata["export_self_test"]["status"], "not_run")

        preprocessing = metadata["preprocessing_contract"]
        self.assertEqual(preprocessing["config"]["resize_strategy"], "preserve_aspect_pad")
        self.assertEqual(preprocessing["config"]["normalization"], "none")

        postprocessing = metadata["postprocessing_contract"]
        self.assertEqual(postprocessing["model_output"], "detections")
        self.assertNotIn("activation", postprocessing)
        self.assertEqual(set(postprocessing["outputs"]), {"boxes", "scores", "classes"})
        self.assertEqual(postprocessing["outputs"]["boxes"]["coordinate_format"], "xyxy")
        self.assertEqual(postprocessing["nms"]["confidence_threshold"], 0.31)
        self.assertEqual(postprocessing["nms"]["iou_threshold"], 0.62)
        self.assertEqual(postprocessing["nms"]["max_detections"], 50)

    def test_demo_prediction_payload_is_ranked_and_bounded(self) -> None:
        payload = build_demo_prediction_payload(
            image_id="image-1",
            true_label="dog",
            latency_ms=12.3456,
            predictions=[
                {"label": "cat", "confidence": 0.25},
                {"label": "dog", "confidence": 1.7},
                {"label": "fox", "confidence": "bad"},
            ],
        )

        self.assertEqual(payload["schema_version"], "champion_demo_prediction_v1")
        self.assertEqual(payload["predicted_label"], "dog")
        self.assertEqual(payload["confidence"], 1.0)
        self.assertTrue(payload["correct"])
        self.assertEqual(payload["latency_ms"], 12.346)
        self.assertEqual([prediction["label"] for prediction in payload["top_k"]], ["dog", "cat", "fox"])

    def test_demo_detection_payload_carries_boxes_and_detection_metadata(self) -> None:
        payload = build_demo_detection_payload(
            image_id="image-1",
            true_label="creeper",
            latency_ms=18.2,
            postprocess_latency_ms=2.4,
            detections=[
                {
                    "label": "zombie",
                    "class_id": 3,
                    "confidence": 0.35,
                    "box": {"x": 0.2, "y": 0.25, "width": 0.3, "height": 0.4},
                },
                {
                    "label": "creeper",
                    "class_id": 0,
                    "confidence": 0.91,
                    "box": {"x": 0.1, "y": 0.15, "width": 0.2, "height": 0.3},
                },
            ],
        )

        self.assertEqual(payload["predicted_label"], "creeper")
        self.assertTrue(payload["correct"])
        self.assertEqual(payload["detection_count"], 2)
        self.assertEqual(payload["detections"][0]["label"], "creeper")
        self.assertEqual(payload["image_metadata"]["task_type"], "object_detection")
        self.assertEqual(payload["image_metadata"]["detection_count"], 2)
        self.assertEqual(payload["postprocess_latency_ms"], 2.4)


if __name__ == "__main__":
    unittest.main()
