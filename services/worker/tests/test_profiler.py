from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

from PIL import Image

from worker.datasets.profiler import (
    compute_image_normalization_metadata,
    detect_dataset_artifacts,
    detect_split_files,
    profile_image_folder,
)


class DatasetProfilerTests(unittest.TestCase):
    def test_profile_image_header_cap_adds_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            for index in range(3):
                _write_image(dataset_dir / "cat" / f"{index}.jpg", (10, 10), (index, index, index))

            previous = os.environ.get("MODEL_EXPRESS_PROFILE_MAX_IMAGES")
            os.environ["MODEL_EXPRESS_PROFILE_MAX_IMAGES"] = "2"
            try:
                profile = profile_image_folder(dataset_dir)
            finally:
                if previous is None:
                    os.environ.pop("MODEL_EXPRESS_PROFILE_MAX_IMAGES", None)
                else:
                    os.environ["MODEL_EXPRESS_PROFILE_MAX_IMAGES"] = previous

            self.assertEqual(profile["image_count"], 2)
            self.assertTrue(profile["profile_scan"]["image_header_cap_hit"])
            self.assertIn("profile_image_header_cap", {warning["code"] for warning in profile["profile_warnings"]})

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

    def test_profile_unwraps_single_wrapper_and_common_image_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            cub_root = dataset_dir / "CUB_200_2011"
            images_root = cub_root / "images"
            _write_image(images_root / "001.Black_footed_Albatross" / "one.jpg", (80, 60), (240, 20, 20))
            _write_image(images_root / "002.Laysan_Albatross" / "two.jpg", (90, 70), (20, 80, 220))
            (cub_root / "classes.txt").write_text(
                "1 001.Black_footed_Albatross\n2 002.Laysan_Albatross\n",
                encoding="utf-8",
            )
            (cub_root / "images.txt").write_text(
                "1 001.Black_footed_Albatross/one.jpg\n2 002.Laysan_Albatross/two.jpg\n",
                encoding="utf-8",
            )
            (cub_root / "image_class_labels.txt").write_text("1 1\n2 2\n", encoding="utf-8")
            (cub_root / "train_test_split.txt").write_text("1 1\n2 0\n", encoding="utf-8")
            (cub_root / "bounding_boxes.txt").write_text("1 2 3 30 40\n2 4 5 25 35\n", encoding="utf-8")
            (cub_root / "attributes").mkdir()
            (cub_root / "parts").mkdir()

            profile = profile_image_folder(dataset_dir)

            self.assertEqual(profile["class_count"], 2)
            self.assertEqual(profile["total_images"], 2)
            self.assertEqual(
                set(profile["class_distribution"]),
                {"001.Black_footed_Albatross", "002.Laysan_Albatross"},
            )
            self.assertNotIn("CUB_200_2011", profile["class_distribution"])
            self.assertEqual(profile["layout_summary"]["image_folder_root"], "CUB_200_2011/images")
            self.assertTrue(profile["metadata_summary"]["bbox_available"])
            self.assertEqual(profile["visual_trait_summary"]["bbox_count"], 2)
            self.assertTrue(profile["split_summary"]["has_explicit_split"])

    def test_profile_uses_top_level_common_image_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            _write_image(dataset_dir / "images" / "cat" / "one.jpg", (12, 10), (255, 0, 0))
            _write_image(dataset_dir / "images" / "dog" / "two.jpg", (14, 12), (0, 255, 0))
            (dataset_dir / "attributes").mkdir()
            (dataset_dir / "parts").mkdir()

            profile = profile_image_folder(dataset_dir)

            self.assertEqual(profile["class_count"], 2)
            self.assertEqual(profile["class_distribution"], {"cat": 1, "dog": 1})
            self.assertEqual(profile["layout_summary"]["image_folder_root"], "images")

    def test_profile_detects_yolo_object_detection_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            _write_image(dataset_dir / "images" / "train" / "one.jpg", (16, 12), (255, 0, 0))
            _write_image(dataset_dir / "images" / "val" / "two.jpg", (18, 10), (0, 255, 0))
            _write_image(dataset_dir / "images" / "test" / "three.jpg", (12, 12), (0, 0, 255))
            train_label_dir = dataset_dir / "labels" / "train"
            val_label_dir = dataset_dir / "labels" / "val"
            train_label_dir.mkdir(parents=True)
            val_label_dir.mkdir(parents=True)
            (train_label_dir / "one.txt").write_text(
                "0 0.50 0.50 0.25 0.30\n1 0.25 0.25 0.10 0.10\n",
                encoding="utf-8",
            )
            (val_label_dir / "two.txt").write_text("1 0.45 0.40 0.20 0.25\n", encoding="utf-8")
            (dataset_dir / "data.yaml").write_text(
                "train: images/train\nval: images/val\ntest: images/test\nnc: 2\nnames: [real_face, fake_face]\n",
                encoding="utf-8",
            )

            artifacts = detect_dataset_artifacts(dataset_dir)
            artifact_types = {artifact["artifact_type"] for artifact in artifacts}
            profile = profile_image_folder(dataset_dir)

            self.assertIn("yolo_dataset_config", artifact_types)
            self.assertIn("yolo_label_file", artifact_types)
            self.assertEqual(profile["task_type"], "object_detection")
            self.assertEqual(profile["class_names"], ["real_face", "fake_face"])
            self.assertEqual(profile["class_count"], 2)
            self.assertEqual(profile["total_images"], 3)
            self.assertEqual(profile["bbox_count"], 3)
            self.assertEqual(profile["bbox_per_class"], {"real_face": 1, "fake_face": 2})
            self.assertTrue(profile["yolo_available"])
            self.assertTrue(profile["object_detection_available"])
            self.assertTrue(profile["metadata_summary"]["yolo_available"])
            self.assertTrue(profile["metadata_summary"]["object_detection_available"])
            self.assertEqual(profile["metadata_summary"]["bbox_count"], 3)
            self.assertEqual(profile["metadata_summary"]["bbox_per_class"], {"real_face": 1, "fake_face": 2})
            self.assertEqual(profile["metadata_summary"]["yolo_summary"], profile["yolo_summary"])
            self.assertEqual(
                profile["yolo_summary"]["split_hints"],
                {"train": "images/train", "val": "images/val", "test": "images/test"},
            )
            self.assertEqual(profile["yolo_summary"]["image_count"], 3)
            self.assertEqual(profile["yolo_summary"]["label_file_count"], 2)
            self.assertEqual(profile["yolo_summary"]["split_image_counts"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(profile["yolo_summary"]["split_label_file_counts"], {"train": 1, "val": 1, "test": 0})
            self.assertEqual(profile["yolo_summary"]["split_bbox_counts"], {"train": 2, "val": 1, "test": 0})
            self.assertTrue(profile["split_summary"]["has_explicit_split"])
            self.assertEqual(
                profile["split_summary"]["yolo_split_hints"],
                {"train": "images/train", "val": "images/val", "test": "images/test"},
            )
            self.assertIn("yolo_format", profile["dataset_traits"])
            self.assertIn("object_detection", profile["dataset_traits"])

    def test_profile_accepts_yolo_config_variants_and_name_formats(self) -> None:
        cases = [
            ("data.yml", "names: {0: real_face, 1: fake_face}\n"),
            ("dataset.yaml", "names:\n  0: real_face\n  1: fake_face\n"),
            ("dataset.yml", "names:\n  - real_face\n  - fake_face\n"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (config_name, names_yaml) in enumerate(cases):
                with self.subTest(config_name=config_name):
                    dataset_dir = root / f"case_{index}"
                    _write_image(dataset_dir / "images" / "train" / "one.jpg", (14, 10), (255, 0, 0))
                    label_dir = dataset_dir / "labels" / "train"
                    label_dir.mkdir(parents=True)
                    (label_dir / "one.txt").write_text("1 0.50 0.50 0.20 0.30\n", encoding="utf-8")
                    (dataset_dir / config_name).write_text(
                        f"train: images/train\nnc: 2\n{names_yaml}",
                        encoding="utf-8",
                    )

                    artifacts = detect_dataset_artifacts(dataset_dir)
                    profile = profile_image_folder(dataset_dir)

                    self.assertIn("yolo_dataset_config", {artifact["artifact_type"] for artifact in artifacts})
                    self.assertEqual(profile["task_type"], "object_detection")
                    self.assertEqual(profile["class_names"], ["real_face", "fake_face"])
                    self.assertEqual(profile["bbox_per_class"], {"fake_face": 1})
                    self.assertEqual(profile["yolo_summary"]["split_hints"], {"train": "images/train"})

    def test_profile_detects_yolo_labels_without_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            _write_image(dataset_dir / "images" / "train" / "one.jpg", (12, 8), (255, 0, 0))
            label_dir = dataset_dir / "labels" / "train"
            label_dir.mkdir(parents=True)
            (label_dir / "one.txt").write_text("0 0.50 0.50 0.25 0.25\n", encoding="utf-8")

            profile = profile_image_folder(dataset_dir)

            self.assertEqual(profile["task_type"], "object_detection")
            self.assertEqual(profile["class_names"], ["class_0"])
            self.assertEqual(profile["bbox_count"], 1)
            self.assertTrue(profile["metadata_summary"]["yolo_available"])


def _write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


if __name__ == "__main__":
    unittest.main()
