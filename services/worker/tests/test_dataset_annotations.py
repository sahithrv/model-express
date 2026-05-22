from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from worker.datasets.annotations import (
    parse_annotation_json_bboxes,
    parse_pascal_voc_bboxes,
    parse_split_file,
)


class DatasetAnnotationTests(unittest.TestCase):
    def test_parse_split_file_supports_filename_and_inline_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            train_path = dataset_dir / "train.txt"
            split_path = dataset_dir / "splits.txt"
            train_path.write_text("cat/one.jpg\ndog/two.jpg,dog\n", encoding="utf-8")
            split_path.write_text("val cat/three.jpg cat\ncat/four.jpg cat test\n", encoding="utf-8")

            train_records = parse_split_file(train_path, dataset_dir)
            split_records = parse_split_file(split_path, dataset_dir)

            self.assertEqual([record["split"] for record in train_records], ["train", "train"])
            self.assertEqual(train_records[0]["label"], "cat")
            self.assertEqual(split_records[0]["split"], "val")
            self.assertEqual(split_records[1]["split"], "test")

    def test_parse_pascal_voc_bboxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            xml_path = Path(temp_dir) / "one.xml"
            xml_path.write_text(
                "<annotation><filename>one.jpg</filename><size><width>10</width>"
                "<height>12</height><depth>3</depth></size><object><name>cat</name>"
                "<bndbox><xmin>1</xmin><ymin>2</ymin><xmax>8</xmax><ymax>9</ymax>"
                "</bndbox></object></annotation>",
                encoding="utf-8",
            )

            payload = parse_pascal_voc_bboxes(xml_path)

            self.assertEqual(payload["format"], "pascal_voc_xml")
            self.assertEqual(payload["filename"], "one.jpg")
            self.assertEqual(payload["image_size"]["width"], 10)
            self.assertEqual(payload["objects"][0]["label"], "cat")
            self.assertEqual(payload["objects"][0]["bbox"]["xmax"], 8)

    def test_parse_annotation_json_bboxes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "annotations.json"
            json_path.write_text(
                '{"filename":"one.jpg","objects":[{"label":"cat","bbox":[1,2,7,8]}]}',
                encoding="utf-8",
            )

            payload = parse_annotation_json_bboxes(json_path)

            self.assertEqual(payload["format"], "annotation_json")
            self.assertEqual(payload["objects"][0]["bbox"], {"xmin": 1, "ymin": 2, "xmax": 8, "ymax": 10})


if __name__ == "__main__":
    unittest.main()
