from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from worker.training.modal_app import _demo_images_from_test_examples


class HeldoutDemoImageTests(unittest.TestCase):
    def test_demo_images_use_original_bytes_for_inference_and_thumbnail_for_preview(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cat.png"
            Image.new("RGB", (12, 8), (20, 40, 60)).save(image_path)

            records = _demo_images_from_test_examples(
                {
                    "example_predictions": [
                        {
                            "path": str(image_path),
                            "predicted_class": "cat",
                            "true_class": "cat",
                            "confidence": 0.91,
                            "correct": True,
                        }
                    ]
                },
                ["cat"],
            )

        self.assertEqual(len(records), 1)
        image = records[0]
        self.assertTrue(image["image_uri"].startswith("data:image/png;base64,"))
        self.assertTrue(image["thumbnail_uri"].startswith("data:image/jpeg;base64,"))
        self.assertNotEqual(image["image_uri"], image["thumbnail_uri"])
        self.assertEqual(image["preview_uri"], image["thumbnail_uri"])
        self.assertTrue(image["metadata"]["parity_safe"])
        self.assertEqual(image["metadata"]["demo_source_type"], "heldout_test_original_bytes")
        self.assertEqual(image["metadata"]["predicted_label_at_training"], "cat")

    def test_large_original_falls_back_to_thumbnail_with_unsafe_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "cat.png"
            Image.new("RGB", (12, 8), (20, 40, 60)).save(image_path)
            with patch.dict("os.environ", {"MODEL_EXPRESS_HELDOUT_DEMO_MAX_ORIGINAL_BYTES": "1"}):
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
        self.assertEqual(image["metadata"]["parity_failure_reason"], "original_image_exceeds_demo_inline_cap")


if __name__ == "__main__":
    unittest.main()
