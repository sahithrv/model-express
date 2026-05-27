from __future__ import annotations

import zipfile

import pytest

from worker.datasets.cache import (
    cleanup_dataset_cache,
    cleanup_job_dataset_cache,
    dataset_archive_path,
    dataset_cache_dir,
    extract_dataset_archive,
    job_dataset_cache_root,
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
