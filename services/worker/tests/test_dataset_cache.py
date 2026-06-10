from __future__ import annotations

import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest

from worker.datasets.cache import (
    cleanup_dataset_cache,
    cleanup_job_dataset_cache,
    dataset_archive_path,
    dataset_cache_dir,
    dataset_materialization_cache_key,
    ensure_dataset_materialized,
    extract_dataset_archive,
    job_dataset_cache_root,
    storage_uri_fingerprint,
)
from worker.jobs import run_profile_dataset_job


def test_dataset_cache_root_can_be_scoped_to_temp_directory(tmp_path):
    dataset_id = "dataset_123"
    archive_path = dataset_archive_path(dataset_id, tmp_path)
    archive_path.parent.mkdir(parents=True)

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("cats/one.txt", "meow")

    extracted = extract_dataset_archive(archive_path, dataset_id, tmp_path)

    assert extracted == tmp_path / dataset_id / "extracted"
    assert (extracted / "cats" / "one.txt").read_text() == "meow"

    cleanup_dataset_cache(dataset_id, tmp_path)
    assert not dataset_cache_dir(dataset_id, tmp_path).exists()


@pytest.mark.parametrize("dataset_id", ["", "../dataset", "nested/dataset", "nested\\dataset"])
def test_dataset_cache_rejects_unsafe_dataset_ids(dataset_id):
    with pytest.raises(ValueError):
        dataset_cache_dir(dataset_id)


def test_non_persistent_job_cache_is_job_scoped(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_EXPRESS_DATASET_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.delenv("MODEL_EXPRESS_PERSIST_DATASET_CACHE", raising=False)

    first = job_dataset_cache_root("job_1")
    second = job_dataset_cache_root("job_2")

    assert first == tmp_path / "cache" / "_jobs" / "job_1"
    assert second == tmp_path / "cache" / "_jobs" / "job_2"
    assert first != second


def test_persistent_job_cache_uses_shared_dataset_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_EXPRESS_DATASET_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("MODEL_EXPRESS_PERSIST_DATASET_CACHE", "true")

    assert job_dataset_cache_root("job_1") == tmp_path / "cache"


def test_cleanup_job_dataset_cache_removes_non_persistent_job_root(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_EXPRESS_DATASET_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.delenv("MODEL_EXPRESS_PERSIST_DATASET_CACHE", raising=False)
    root = job_dataset_cache_root("job_1")
    (root / "dataset_1").mkdir(parents=True)

    cleanup_job_dataset_cache("job_1")

    assert not root.exists()


def test_materialized_dataset_cache_key_uses_checksum_when_available():
    checksum = "a" * 64
    storage_uri = "s3://Bucket/path/to/dataset.zip"

    key = dataset_materialization_cache_key(checksum, storage_uri)

    assert key == f"sha256-{checksum}"


def test_materialized_dataset_cache_key_falls_back_to_storage_fingerprint_without_checksum():
    storage_uri = "s3://Bucket/path/to/dataset.zip"

    key = dataset_materialization_cache_key("", storage_uri)

    assert key == f"uri-{storage_uri_fingerprint(storage_uri)}"


def test_ensure_dataset_materialized_reuses_completed_cache(tmp_path):
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("cats/one.txt", "meow")
    calls = []

    def fake_download(_storage_uri, destination):
        calls.append(destination)
        shutil.copyfile(source_zip, destination)
        return destination

    first = ensure_dataset_materialized(
        dataset_id="dataset_1",
        storage_uri="s3://bucket/dataset.zip",
        checksum_sha256="b" * 64,
        cache_root=tmp_path / "materialized",
        download_fn=fake_download,
    )
    second = ensure_dataset_materialized(
        dataset_id="dataset_1",
        storage_uri="s3://bucket/dataset.zip",
        checksum_sha256="b" * 64,
        cache_root=tmp_path / "materialized",
        download_fn=fake_download,
    )

    assert len(calls) == 1
    assert first.dataset_dir == second.dataset_dir
    assert (second.dataset_dir / "cats" / "one.txt").read_text() == "meow"
    assert first.telemetry["dataset_materialization_cache_miss"] is True
    assert first.telemetry["dataset_materialization_status"] == "materialized"
    assert first.telemetry["dataset_materialization_download_seconds"] >= 0
    assert first.telemetry["dataset_materialization_total_seconds"] >= first.telemetry["dataset_materialization_extract_seconds"]
    assert second.telemetry["dataset_materialization_cache_hit"] is True
    assert second.telemetry["dataset_materialization_status"] == "hit"
    assert second.telemetry["dataset_materialization_bytes_downloaded"] == 0
    assert second.telemetry["dataset_materialization_total_seconds"] >= 0


def test_persistent_disk_materialization_reuses_checksum_cache_across_worker_restart(tmp_path):
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("cats/one.txt", "meow")
    calls = []

    def fake_download(_storage_uri, destination):
        calls.append(destination)
        shutil.copyfile(source_zip, destination)
        return destination

    cache_root = tmp_path / "persistent-disk-cache"
    checksum = "d" * 64
    first = ensure_dataset_materialized(
        dataset_id="dataset_1",
        storage_uri="s3://bucket/dataset.zip",
        checksum_sha256=checksum,
        cache_root=cache_root,
        download_fn=fake_download,
    )
    second = ensure_dataset_materialized(
        dataset_id="dataset_1",
        storage_uri="s3://bucket/dataset.zip",
        checksum_sha256=checksum,
        cache_root=cache_root,
        download_fn=fake_download,
    )

    assert len(calls) == 1
    assert first.dataset_dir == second.dataset_dir
    assert second.telemetry["dataset_materialization_cache_hit"] is True
    assert second.telemetry["dataset_materialization_cache_key"] == dataset_materialization_cache_key(
        checksum,
        "s3://bucket/dataset.zip",
    )


def test_concurrent_materialization_uses_single_download(monkeypatch, tmp_path):
    import threading
    import time

    monkeypatch.setattr("worker.datasets.cache.LOCK_POLL_SECONDS", 0.01)
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("dogs/one.txt", "woof")
    calls = []
    calls_lock = threading.Lock()

    def fake_download(_storage_uri, destination):
        with calls_lock:
            calls.append(destination)
        time.sleep(0.1)
        shutil.copyfile(source_zip, destination)
        return destination

    def materialize():
        return ensure_dataset_materialized(
            dataset_id="dataset_1",
            storage_uri="s3://bucket/dataset.zip",
            checksum_sha256="c" * 64,
            cache_root=tmp_path / "materialized",
            download_fn=fake_download,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = list(pool.map(lambda _index: materialize(), range(2)))

    assert len(calls) == 1
    assert first.dataset_dir == second.dataset_dir
    assert (first.dataset_dir / "dogs" / "one.txt").read_text() == "woof"
    materialization_misses = {
        first.telemetry["dataset_materialization_cache_miss"],
        second.telemetry["dataset_materialization_cache_miss"],
    }
    assert materialization_misses == {True, False}


def test_profile_dataset_uses_modal_provider_when_worker_is_modal(monkeypatch):
    called = {}

    def fake_modal_profile(client, job):
        called["client"] = client
        called["job"] = job

    import worker.training.modal_provider as modal_provider

    monkeypatch.setenv("GPU_TYPE", "modal")
    monkeypatch.delenv("MODEL_EXPRESS_DATASET_PROFILE_PROVIDER", raising=False)
    monkeypatch.setattr(modal_provider, "run_modal_dataset_profile", fake_modal_profile)

    client = object()
    job = {"id": "job_1", "config": {"dataset_id": "dataset_1"}}

    run_profile_dataset_job(client, job)

    assert called == {"client": client, "job": job}
