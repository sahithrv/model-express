from __future__ import annotations

import base64
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from worker.datasets.metadata_discovery import (
    build_metadata_import_payload,
    declared_metadata_format,
    is_safe_relative_path,
)


class DatasetMetadataDiscoveryTests(unittest.TestCase):
    def test_build_metadata_import_payload_caps_sources_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            class_dir = dataset_dir / "cat"
            metadata_dir = dataset_dir / "metadata"
            class_dir.mkdir()
            metadata_dir.mkdir()
            Image.new("RGB", (2, 2), (255, 0, 0)).save(class_dir / "one.jpg")
            labels = "path,label\ncat/one.jpg,cat\n"
            (dataset_dir / "labels.csv").write_bytes(labels.encode("utf-8"))
            (metadata_dir / "export.yaml").write_text("names: [cat]\n", encoding="utf-8")
            (metadata_dir / "large.csv").write_text("x" * 64, encoding="utf-8")

            payload = build_metadata_import_payload(
                dataset_dir,
                max_source_bytes=32,
                max_total_source_bytes=64,
            )

            sources = {source["relative_path"]: source for source in payload["sources"]}
            self.assertFalse(payload["strict_mode"])
            self.assertIn("cat/one.jpg", {item["relative_path"] for item in payload["inventory"]["files"]})
            self.assertEqual(sources["labels.csv"]["declared_format"], "csv_manifest")
            self.assertEqual(
                base64.b64decode(sources["labels.csv"]["content_base64"]).decode("utf-8"),
                labels,
            )
            self.assertEqual(
                sources["labels.csv"]["checksum_sha256"],
                hashlib.sha256(labels.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(sources["metadata/export.yaml"]["declared_format"], "unsupported")
            self.assertIn("unsupported_metadata_format", sources["metadata/export.yaml"]["warnings"])
            self.assertNotIn("content_base64", sources["metadata/export.yaml"])
            self.assertIn("content_skipped_source_size_cap", sources["metadata/large.csv"]["warnings"])
            self.assertNotIn("content_base64", sources["metadata/large.csv"])
            self.assertIn(
                "unsupported_metadata_candidates",
                {warning["code"] for warning in payload["warnings"]},
            )

    def test_root_level_manifest_like_csv_is_sent_as_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            image_path = dataset_dir / "train" / "air hockey" / "001.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (2, 2), (255, 0, 0)).save(image_path)
            content = "class_id,filepaths,labels,data_set\n0,train/air hockey/001.jpg,air hockey,train\n"
            sports_csv = dataset_dir / "sports.csv"
            sports_csv.write_bytes(content.encode("utf-8"))

            payload = build_metadata_import_payload(dataset_dir)

            sources = {source["relative_path"]: source for source in payload["sources"]}
            self.assertEqual(sources["sports.csv"]["declared_format"], "csv_manifest")
            self.assertEqual(
                base64.b64decode(sources["sports.csv"]["content_base64"]).decode("utf-8"),
                content,
            )

    def test_rejects_unsafe_relative_paths_before_transport(self) -> None:
        for path in ("../labels.csv", "/labels.csv", "C:/labels.csv", "metadata\\labels.csv", "a//b.csv"):
            with self.subTest(path=path):
                self.assertFalse(is_safe_relative_path(path))

        self.assertTrue(is_safe_relative_path("metadata/labels.csv"))
        self.assertEqual(declared_metadata_format("../labels.csv"), None)

    def test_discovers_parts_and_attributes_metadata_candidates(self) -> None:
        self.assertEqual(declared_metadata_format("parts/part_locs.txt"), "cub_sidecars")
        self.assertEqual(declared_metadata_format("attributes/attributes.txt"), "unsupported")
        self.assertEqual(declared_metadata_format("landmarks/points.json"), "unsupported")

    def test_discovers_nested_cub_sidecars(self) -> None:
        for path in (
            "CUB_200_2011/classes.txt",
            "CUB_200_2011/images.txt",
            "CUB_200_2011/image_class_labels.txt",
            "CUB_200_2011/train_test_split.txt",
            "CUB_200_2011/bounding_boxes.txt",
        ):
            with self.subTest(path=path):
                self.assertEqual(declared_metadata_format(path), "cub_sidecars")


if __name__ == "__main__":
    unittest.main()
