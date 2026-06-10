from __future__ import annotations

from pathlib import Path

import pytest

from worker.datasets.tiers import build_classification_preview_subset, materialize_yolo_preview_subset


def test_classification_preview_subset_is_deterministic_and_class_stratified(tmp_path: Path) -> None:
    targets = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    split_indices = {
        "train": list(range(0, 9)),
        "val": [9, 10],
        "test": [11],
    }

    first = build_classification_preview_subset(
        dataset_dir=tmp_path,
        dataset_checksum="a" * 64,
        targets=targets,
        split_indices=split_indices,
        class_names=["cat", "dog", "owl"],
        fraction=0.5,
        seed=7,
        split_policy="metadata_official",
        image_size_family="image_classification:224",
    )
    second = build_classification_preview_subset(
        dataset_dir=tmp_path,
        dataset_checksum="a" * 64,
        targets=targets,
        split_indices=split_indices,
        class_names=["cat", "dog", "owl"],
        fraction=0.5,
        seed=7,
        split_policy="metadata_official",
        image_size_family="image_classification:224",
    )

    assert first["manifest_id"] == second["manifest_id"]
    assert first["indices"] == second["indices"]
    assert first["split_counts"]["train"] == 4
    assert set(targets[index] for index in first["indices"]["train"]) == {0, 1, 2}


def test_yolo_preview_subset_preserves_pairs_and_valid_data_yaml(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
        for index in range(4):
            image = dataset / "images" / split / f"{split}_{index}.jpg"
            image.write_bytes(b"fake image bytes")
            class_id = index % 2
            (dataset / "labels" / split / f"{split}_{index}.txt").write_text(
                f"{class_id} 0.5 0.5 0.25 0.25\n",
                encoding="utf-8",
            )
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 2",
                "names: [cat, dog]",
            ]
        ),
        encoding="utf-8",
    )

    preview_yaml, manifest = materialize_yolo_preview_subset(
        dataset_dir=dataset,
        data_config_path=data_yaml,
        output_root=tmp_path / "subsets",
        dataset_checksum="b" * 64,
        fraction=0.5,
        seed=11,
        split_policy="official_yolo",
        image_size_family="object_detection:512",
    )
    preview_yaml_again, manifest_again = materialize_yolo_preview_subset(
        dataset_dir=dataset,
        data_config_path=data_yaml,
        output_root=tmp_path / "subsets",
        dataset_checksum="b" * 64,
        fraction=0.5,
        seed=11,
        split_policy="official_yolo",
        image_size_family="object_detection:512",
    )

    assert preview_yaml == preview_yaml_again
    assert manifest["manifest_id"] == manifest_again["manifest_id"]
    assert preview_yaml.is_file()
    text = preview_yaml.read_text(encoding="utf-8")
    assert "train: images/train" in text
    assert "val: images/val" in text
    assert "test: images/test" in text
    assert manifest["split_image_counts"] == {"train": 2, "val": 2, "test": 2}
    for split in ("train", "val", "test"):
        images = sorted((preview_yaml.parent / "images" / split).glob("*.jpg"))
        labels = sorted((preview_yaml.parent / "labels" / split).glob("*.txt"))
        assert len(images) == len(labels) == 2
        assert {path.stem for path in images} == {path.stem for path in labels}
