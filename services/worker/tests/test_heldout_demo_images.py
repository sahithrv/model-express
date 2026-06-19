from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from worker.training.modal_app import (
    _compact_training_run_evaluation_payload,
    _demo_images_from_test_examples,
    _training_evaluation_payload_size_bytes,
)


class HeldoutDemoImageTests(unittest.TestCase):
    def test_demo_images_upload_original_artifact_and_keep_thumbnail_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cat.png"
            Image.new("RGB", (12, 8), (20, 40, 60)).save(image_path)
            uploads = []

            def fake_upload(source: Path, destination: str) -> None:
                uploads.append((Path(source), destination))

            with patch.dict(
                "os.environ",
                {"MODEL_EXPRESS_ARTIFACT_BUCKET": "", "MODEL_EXPRESS_ARTIFACT_PREFIX": "model-express/artifacts"},
            ):
                records = _demo_images_from_test_examples(
                    {
                        "example_predictions": [
                            {
                                "path": str(image_path),
                                "predicted_class": "cat",
                                "true_class": "cat",
                                "predicted_class_index": 0,
                                "true_class_index": 0,
                                "class_label_order_hash": "training-hash",
                                "confidence": 0.91,
                                "correct": True,
                            }
                        ]
                    },
                    ["cat"],
                    dataset={"storage_uri": "s3://bucket/dataset.zip"},
                    job_id="job/1",
                    artifact_uploader=fake_upload,
                )

        self.assertEqual(len(records), 1)
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0][0], image_path)
        self.assertTrue(uploads[0][1].startswith("s3://bucket/model-express/artifacts/job_1/heldout_demo_images/cat-"))
        image = records[0]
        self.assertEqual(image["image_uri"], uploads[0][1])
        self.assertEqual(image["uri"], uploads[0][1])
        self.assertEqual(image["original_image_uri"], uploads[0][1])
        self.assertEqual(image["source_artifact_uri"], uploads[0][1])
        self.assertFalse(image["image_uri"].startswith("data:image/"))
        self.assertTrue(image["thumbnail_uri"].startswith("data:image/jpeg;base64,"))
        self.assertNotEqual(image["image_uri"], image["thumbnail_uri"])
        self.assertEqual(image["preview_uri"], image["thumbnail_uri"])
        self.assertTrue(image["metadata"]["parity_safe"])
        self.assertEqual(image["metadata"]["demo_source_type"], "heldout_test_original_artifact")
        self.assertEqual(image["metadata"]["original_image_uri"], uploads[0][1])
        self.assertEqual(image["metadata"]["predicted_label_at_training"], "cat")
        self.assertEqual(image["metadata"]["true_class_index_at_training"], 0)
        self.assertEqual(image["metadata"]["predicted_class_index_at_training"], 0)
        self.assertEqual(image["metadata"]["class_label_order_hash"], "training-hash")

    def test_missing_artifact_upload_falls_back_to_thumbnail_with_unsafe_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cat.png"
            Image.new("RGB", (12, 8), (20, 40, 60)).save(image_path)
            records = _demo_images_from_test_examples(
                {
                    "example_predictions": [
                        {
                            "path": str(image_path),
                            "predicted_class": "dog",
                            "true_class": "cat",
                            "confidence": 0.91,
                            "correct": False,
                        }
                    ]
                },
                ["cat", "dog"],
            )

        self.assertEqual(len(records), 1)
        image = records[0]
        self.assertEqual(image["image_uri"], image["thumbnail_uri"])
        self.assertFalse(image["metadata"]["parity_safe"])
        self.assertEqual(image["metadata"]["parity_status"], "unsafe")
        self.assertEqual(image["metadata"]["parity_failure_reason"], "original_image_artifact_upload_not_configured")

    def test_demo_images_start_with_correct_examples_without_dropping_hard_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = {
                "cat_correct": root / "cat_correct.png",
                "dog_correct": root / "dog_correct.png",
                "cat_wrong": root / "cat_wrong.png",
                "dog_wrong": root / "dog_wrong.png",
            }
            for path in paths.values():
                Image.new("RGB", (12, 8), (20, 40, 60)).save(path)
            records = _demo_images_from_test_examples(
                {
                    "example_predictions": [
                        {
                            "path": str(paths["cat_wrong"]),
                            "predicted_class": "dog",
                            "true_class": "cat",
                            "confidence": 0.99,
                            "correct": False,
                        },
                        {
                            "path": str(paths["dog_wrong"]),
                            "predicted_class": "cat",
                            "true_class": "dog",
                            "confidence": 0.98,
                            "correct": False,
                        },
                        {
                            "path": str(paths["dog_correct"]),
                            "predicted_class": "dog",
                            "true_class": "dog",
                            "confidence": 0.80,
                            "correct": True,
                        },
                        {
                            "path": str(paths["cat_correct"]),
                            "predicted_class": "cat",
                            "true_class": "cat",
                            "confidence": 0.70,
                            "correct": True,
                        },
                    ]
                },
                ["cat", "dog"],
                max_total=4,
            )

        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["metadata"]["correct_at_training"], True)
        self.assertEqual(records[1]["metadata"]["correct_at_training"], False)
        self.assertEqual(records[2]["metadata"]["correct_at_training"], True)
        self.assertEqual(records[3]["metadata"]["correct_at_training"], False)
        self.assertEqual(records[1]["metadata"]["confidence_at_training"], 0.99)

    def test_compacted_evaluation_payload_stays_below_safe_threshold_for_many_demo_images(self) -> None:
        thumbnail = "data:image/jpeg;base64," + ("B" * 8_000)
        original = "data:image/png;base64," + ("A" * 100_000)
        payload = {
            "objective_profile": {
                "heldout_demo_images": [
                    {
                        "id": f"test:{index}",
                        "uri": original,
                        "image_uri": original,
                        "preview_uri": thumbnail,
                        "thumbnail_uri": thumbnail,
                        "original_image_uri": f"s3://bucket/model-express/artifacts/job_1/heldout_demo_images/{index}.png",
                        "metadata": {
                            "source_artifact_uri": f"s3://bucket/model-express/artifacts/job_1/heldout_demo_images/{index}.png",
                            "demo_source_type": "heldout_test_original_artifact",
                            "parity_safe": True,
                        },
                    }
                    for index in range(48)
                ]
            },
            "per_class_metrics": {},
            "confusion_matrix": [],
            "model_profile": {},
            "holistic_scores": {},
        }

        compacted = _compact_training_run_evaluation_payload(
            payload,
            reason="test",
            payload_bytes_before=_training_evaluation_payload_size_bytes(payload),
            soft_limit_bytes=1_500_000,
        )

        self.assertLess(_training_evaluation_payload_size_bytes(compacted), 1_500_000)
        diagnostics = compacted["objective_profile"]["diagnostics"]
        self.assertEqual(diagnostics[-1]["code"], "evaluation_payload_compacted_due_to_size")
        for image in compacted["objective_profile"]["heldout_demo_images"]:
            self.assertTrue(image["image_uri"].startswith("s3://"))
            self.assertTrue(image["thumbnail_uri"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
