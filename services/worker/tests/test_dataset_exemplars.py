from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from worker.datasets.exemplars import generate_visual_exemplars


class DatasetExemplarTests(unittest.TestCase):
    def test_visual_exemplars_are_class_balanced_capped_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            output_a = root / "out_a"
            output_b = root / "out_b"
            for class_name, color in {"cat": (255, 0, 0), "dog": (0, 0, 255)}.items():
                class_dir = dataset_dir / class_name
                class_dir.mkdir(parents=True)
                for index in range(3):
                    Image.new("RGB", (80 + index, 60), color).save(class_dir / f"{index}.png")
            (dataset_dir / "cat" / "bad.jpg").write_bytes(b"not an image")

            pack_a = generate_visual_exemplars(
                dataset_dir=dataset_dir,
                output_dir=output_a,
                images_per_class=2,
                max_total_images=3,
                max_image_bytes=4_000,
                max_total_bytes=9_000,
                image_size=32,
                seed=7,
            )
            pack_b = generate_visual_exemplars(
                dataset_dir=dataset_dir,
                output_dir=output_b,
                images_per_class=2,
                max_total_images=3,
                max_image_bytes=4_000,
                max_total_bytes=9_000,
                image_size=32,
                seed=7,
            )

            self.assertEqual(pack_a["status"], "created")
            self.assertLessEqual(len(pack_a["visual_exemplars"]), 3)
            self.assertLessEqual(pack_a["summary"]["total_bytes"], 9_000)
            self.assertEqual(
                [Path(item["source_path"]).name for item in pack_a["visual_exemplars"]],
                [Path(item["source_path"]).name for item in pack_b["visual_exemplars"]],
            )
            self.assertTrue(all(Path(item["path"]).exists() for item in pack_a["visual_exemplars"]))
            self.assertTrue(all(item["bytes"] <= 4_000 for item in pack_a["visual_exemplars"]))


if __name__ == "__main__":
    unittest.main()
