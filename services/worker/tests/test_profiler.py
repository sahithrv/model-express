from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from worker.datasets.profiler import (
    compute_image_normalization_metadata,
    detect_dataset_artifacts,
    detect_split_files,
    profile_image_folder,
)


class DatasetProfilerTests(unittest.TestCase):
    def test_profile_detects_artifacts_splits_and_bbox_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            cat_dir = dataset_dir / "cat"
            dog_dir = dataset_dir / "dog"
            annotations_dir = dataset_dir / "annotations"
            cat_dir.mkdir()
            dog_dir.mkdir()
            annotations_dir.mkdir()
            Image.new("RGB", (10, 8), (255, 0, 0)).save(cat_dir / "one.jpg")
            Image.new("RGB", (12, 12), (0, 255, 0)).save(dog_dir / "two.jpg")
            (dataset_dir / "train.txt").write_text("cat/one.jpg\n", encoding="utf-8")
            (dataset_dir / "labels.csv").write_text("path,label\ncat/one.jpg,cat\n", encoding="utf-8")
            (annotations_dir / "one.xml").write_text(
                "<annotation><object><bndbox><xmin>1</xmin><ymin>1</ymin>"
                "<xmax>8</xmax><ymax>7</ymax></bndbox></object></annotation>",
                encoding="utf-8",
            )

            artifacts = detect_dataset_artifacts(dataset_dir)
            artifact_types = {artifact["artifact_type"] for artifact in artifacts}
            profile = profile_image_folder(dataset_dir)

            self.assertIn("split_file", artifact_types)
            self.assertIn("labels_csv", artifact_types)
            self.assertIn("bounding_boxes", artifact_types)
            self.assertEqual(len(detect_split_files(dataset_dir)), 1)
            self.assertTrue(profile["split_summary"]["has_explicit_split"])
            self.assertTrue(profile["metadata_summary"]["bbox_available"])
            self.assertIn("bbox_available", profile["dataset_traits"])
            self.assertEqual(profile["visual_trait_summary"]["schema_version"], "visual_traits_v1")
            self.assertGreaterEqual(profile["visual_trait_summary"]["sampled_image_count"], 2)
            self.assertIn(profile["visual_trait_summary"]["object_scale"], {"small", "medium", "large", "unknown"})

    def test_compute_image_normalization_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            class_dir = dataset_dir / "class_a"
            class_dir.mkdir()
            Image.new("RGB", (2, 1), (255, 0, 0)).save(class_dir / "red.png")
            Image.new("RGB", (2, 1), (0, 0, 255)).save(class_dir / "blue.png")

            metadata = compute_image_normalization_metadata(dataset_dir)

            self.assertEqual(metadata["status"], "computed")
            self.assertEqual(metadata["image_count"], 2)
            self.assertEqual(metadata["mean"], [0.5, 0.0, 0.5])
            self.assertEqual(len(metadata["std"]), 3)
            self.assertGreater(metadata["std"][0], 0)


if __name__ == "__main__":
    unittest.main()
