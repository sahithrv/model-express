from __future__ import annotations

import unittest

from worker.exporting.metadata import build_champion_export_metadata, build_demo_prediction_payload


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
        self.assertEqual(metadata["input_shape"], [1, 3, 224, 224])
        self.assertEqual(metadata["class_labels"], ["cat", "dog"])
        self.assertEqual(metadata["class_count"], 2)
        self.assertEqual(metadata["artifact_uri"], "s3://bucket/model.pt")
        contract = metadata["inference_contract"]
        self.assertEqual(contract["schema_version"], "model_express_inference_contract_v1")
        self.assertEqual(contract["input"]["model_tensor_shape"], [1, 3, 224, 224])
        self.assertIn("preprocessing", contract)
        self.assertIn("postprocessing", contract)

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


if __name__ == "__main__":
    unittest.main()
