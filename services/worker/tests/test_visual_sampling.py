from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from worker.datasets.visual_sampling import generate_visual_sample_pack


class VisualSamplingTests(unittest.TestCase):
    def test_sample_pack_is_deterministic_capped_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            _write_image(dataset_dir / "cat" / "one.png", (80, 60), (240, 20, 20))
            _write_image(dataset_dir / "cat" / "two.png", (82, 60), (210, 40, 40))
            _write_image(dataset_dir / "dog" / "wide.png", (180, 24), (20, 20, 220))
            _write_image(dataset_dir / "dog" / "tall.png", (24, 180), (40, 40, 200))

            pack_a = generate_visual_sample_pack(
                dataset_dir=dataset_dir,
                dataset_id="dataset_1",
                dataset_name="pets",
                max_total_images=4,
                max_high_detail_images=1,
                max_image_bytes=30_000,
                max_total_bytes=100_000,
                image_size=48,
                high_detail_image_size=96,
                seed=17,
            )
            pack_b = generate_visual_sample_pack(
                dataset_dir=dataset_dir,
                dataset_id="dataset_1",
                dataset_name="pets",
                max_total_images=4,
                max_high_detail_images=1,
                max_image_bytes=30_000,
                max_total_bytes=100_000,
                image_size=48,
                high_detail_image_size=96,
                seed=17,
            )

            manifest = pack_a["sample_manifest"]
            samples = manifest["samples"]
            image_inputs = pack_a["image_inputs"]

            self.assertEqual(pack_a["status"], "created")
            self.assertLessEqual(len(samples), 4)
            self.assertEqual(
                [sample["image_id"] for sample in samples],
                [sample["image_id"] for sample in pack_b["sample_manifest"]["samples"]],
            )
            self.assertEqual(len(image_inputs), len(samples))
            self.assertLessEqual(manifest["high_detail_image_count"], 1)
            self.assertLessEqual(sum(item["bytes"] for item in image_inputs), 100_000)
            self.assertTrue(
                any("aspect_ratio_outlier" in sample["selection_basis"] for sample in samples)
            )
            self.assertTrue(all(sample["image_id"].startswith("img_") for sample in samples))

            serialized = json.dumps(pack_a, sort_keys=True)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("source_path", serialized)
            self.assertNotIn("local_path", serialized)
            self.assertNotIn("relative_path", serialized)

    def test_large_class_count_respects_sample_caps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            for index in range(15):
                _write_image(
                    dataset_dir / f"class_{index:02d}" / "one.jpg",
                    (64 + index, 48),
                    (index * 10 % 255, 80, 160),
                )

            pack = generate_visual_sample_pack(
                dataset_dir=dataset_dir,
                dataset_id="many_classes",
                max_total_images=5,
                max_high_detail_images=2,
                max_image_bytes=20_000,
                max_total_bytes=70_000,
                image_size=40,
                seed=3,
            )

            manifest = pack["sample_manifest"]

            self.assertEqual(manifest["images_available"], 15)
            self.assertLessEqual(manifest["images_analyzed"], 5)
            self.assertLessEqual(manifest["classes_covered"], 5)
            self.assertIn("Sample is not class-complete.", manifest["limitations"])

    def test_sample_pack_unwraps_single_wrapper_common_image_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            images_root = dataset_dir / "CUB_200_2011" / "images"
            _write_image(images_root / "001.Black_footed_Albatross" / "one.jpg", (80, 60), (240, 20, 20))
            _write_image(images_root / "002.Laysan_Albatross" / "two.jpg", (82, 62), (20, 80, 220))

            pack = generate_visual_sample_pack(
                dataset_dir=dataset_dir,
                dataset_id="cub",
                dataset_name="CUB",
                max_total_images=4,
                max_high_detail_images=1,
                max_image_bytes=30_000,
                max_total_bytes=100_000,
                image_size=48,
                seed=11,
            )

            manifest = pack["sample_manifest"]
            class_names = {sample["class_name"] for sample in manifest["samples"]}

            self.assertEqual(pack["status"], "created")
            self.assertEqual(manifest["images_available"], 2)
            self.assertEqual(manifest["classes_total"], 2)
            self.assertEqual(class_names, {"001.Black_footed_Albatross", "002.Laysan_Albatross"})
            self.assertNotIn("CUB_200_2011", class_names)
            self.assertNotIn("images", class_names)


def _write_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


if __name__ == "__main__":
    unittest.main()
