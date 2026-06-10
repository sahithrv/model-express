from __future__ import annotations

import pytest

from worker.training import providers


def test_persistent_gpu_provider_dispatches_to_persistent_training(monkeypatch):
    called = {}

    def fake_persistent(client, job):
        called["client"] = client
        called["job"] = job

    monkeypatch.setattr(providers, "run_persistent_gpu_training", fake_persistent)
    client = object()
    job = {"id": "job_1", "config": {"provider": "persistent-gpu"}}

    providers.run_training_job(client, job)

    assert called == {"client": client, "job": job}


def test_unknown_training_provider_still_rejected():
    with pytest.raises(ValueError, match="Unsupported training provider"):
        providers.run_training_job(object(), {"id": "job_1", "config": {"provider": "unsupported"}})
