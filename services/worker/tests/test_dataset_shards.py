from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from worker.datasets.cache import dataset_materialization_cache_key, ensure_dataset_materialized
from worker.datasets.shards import (
    ShardMaterializationError,
    create_shard_artifacts,
    materialize_shard_artifacts,
)


def test_classification_shard_manifest_reproduces_file_counts_and_class_counts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        for class_name, count in {"cat": 2, "dog": 1}.items():
            root = dataset / split / class_name
            root.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                (root / f"{split}_{class_name}_{index}.jpg").write_bytes(b"image")
    (dataset / "labels.csv").write_text("filename,label\n", encoding="utf-8")

    manifest = create_shard_artifacts(
        dataset_dir=dataset,
        artifact_root=tmp_path / "shards",
        dataset_checksum="a" * 64,
        cache_key="sha256-" + "a" * 64,
        storage_uri="s3://bucket/dataset.zip",
        max_shard_bytes=16,
    )
    materialize_shard_artifacts(
        manifest_path=tmp_path / "shards" / "manifest.json",
        output_dir=tmp_path / "materialized",
        dataset_checksum="a" * 64,
    )

    assert manifest["task_type"] == "image_classification"
    assert manifest["file_counts"] == {"total": 7, "images": 6, "labels": 0, "metadata": 1}
    assert manifest["class_counts"] == {"cat": 4, "dog": 2}
    assert manifest["split_class_counts"]["train"] == {"cat": 2, "dog": 1}
    assert sum(1 for path in (tmp_path / "materialized").rglob("*") if path.is_file()) == 7
    assert (tmp_path / "materialized" / "train" / "cat" / "train_cat_0.jpg").is_file()


def test_yolo_shard_extraction_preserves_pairs_and_valid_data_yaml(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        for index in range(3):
            image = dataset / "images" / split / f"{split}_{index}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"image")
            label = dataset / "labels" / split / f"{split}_{index}.txt"
            label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(f"{index % 2} 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (dataset / "data.yaml").write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "nc: 2",
                "names: [cat, dog]",
            ]
        ),
        encoding="utf-8",
    )

    manifest = create_shard_artifacts(
        dataset_dir=dataset,
        artifact_root=tmp_path / "shards",
        dataset_checksum="b" * 64,
        cache_key="sha256-" + "b" * 64,
        storage_uri="s3://bucket/yolo.zip",
        max_shard_bytes=16,
    )
    materialized = tmp_path / "materialized"
    materialize_shard_artifacts(
        manifest_path=tmp_path / "shards" / "manifest.json",
        output_dir=materialized,
        dataset_checksum="b" * 64,
    )

    assert manifest["task_type"] == "object_detection"
    assert manifest["yolo"]["split_image_counts"] == {"train": 3, "val": 3}
    assert (materialized / "data.yaml").is_file()
    data_yaml = (materialized / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/train" in data_yaml
    assert "val: images/val" in data_yaml
    for split in ("train", "val"):
        images = sorted((materialized / "images" / split).glob("*.jpg"))
        labels = sorted((materialized / "labels" / split).glob("*.txt"))
        assert len(images) == len(labels) == 3
        assert {path.stem for path in images} == {path.stem for path in labels}


def test_materialization_zip_fallback_still_works_when_shards_flag_is_off(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MODEL_EXPRESS_DATASET_SHARDS", "0")
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("cats/one.jpg", "image")
    calls = []

    def fake_download(_storage_uri: str, destination: Path) -> Path:
        calls.append(destination)
        shutil.copyfile(source_zip, destination)
        return destination

    materialized = ensure_dataset_materialized(
        dataset_id="dataset_1",
        storage_uri="s3://bucket/dataset.zip",
        checksum_sha256="c" * 64,
        cache_root=tmp_path / "cache",
        download_fn=fake_download,
    )

    cache_key = dataset_materialization_cache_key("c" * 64, "s3://bucket/dataset.zip")
    assert len(calls) == 1
    assert (materialized.dataset_dir / "cats" / "one.jpg").read_text(encoding="utf-8") == "image"
    assert materialized.telemetry["dataset_materialization_shards_enabled"] is False
    assert not (tmp_path / "cache" / cache_key / "shards" / "manifest.json").exists()


def test_existing_missing_shard_artifact_fails_without_completing_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MODEL_EXPRESS_DATASET_SHARDS", "1")
    dataset = tmp_path / "dataset"
    (dataset / "cat").mkdir(parents=True)
    (dataset / "cat" / "one.jpg").write_bytes(b"image")
    cache_root = tmp_path / "cache"
    checksum = "d" * 64
    cache_key = dataset_materialization_cache_key(checksum, "s3://bucket/dataset.zip")
    cache_dir = cache_root / cache_key
    create_shard_artifacts(
        dataset_dir=dataset,
        artifact_root=cache_dir / "shards",
        dataset_checksum=checksum,
        cache_key=cache_key,
        storage_uri="s3://bucket/dataset.zip",
    )
    first_shard = next((cache_dir / "shards" / "artifacts").glob("*.tar"))
    first_shard.unlink()

    def fail_download(_storage_uri: str, _destination: Path) -> Path:
        raise AssertionError("zip fallback should not run when an existing shard manifest is corrupt")

    with pytest.raises(ShardMaterializationError, match="Missing dataset shard artifact"):
        ensure_dataset_materialized(
            dataset_id="dataset_1",
            storage_uri="s3://bucket/dataset.zip",
            checksum_sha256=checksum,
            cache_root=cache_root,
            download_fn=fail_download,
        )

    assert not (cache_dir / ".complete").exists()
    assert not (cache_dir / "manifest.json").exists()


def test_corrupt_shard_checksum_fails_clearly(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "cat").mkdir(parents=True)
    (dataset / "cat" / "one.jpg").write_bytes(b"image")
    create_shard_artifacts(
        dataset_dir=dataset,
        artifact_root=tmp_path / "shards",
        dataset_checksum="e" * 64,
        cache_key="sha256-" + "e" * 64,
        storage_uri="s3://bucket/dataset.zip",
    )
    manifest_path = tmp_path / "shards" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ShardMaterializationError, match="checksum mismatch"):
        materialize_shard_artifacts(
            manifest_path=manifest_path,
            output_dir=tmp_path / "materialized",
            dataset_checksum="e" * 64,
        )
