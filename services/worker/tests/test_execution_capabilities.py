from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from worker.training import execution_capabilities_generated
from worker.training.execution_capabilities import (
    capability_document_v1,
    capability_profile,
    normalize_execution_config,
    resolve_accepted_config,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = REPOSITORY_ROOT / "contracts" / "experiment_execution_capabilities.v1.json"
FIXTURES_PATH = REPOSITORY_ROOT / "contracts" / "experiment_execution_capabilities.v1.fixtures.json"


def test_generated_python_capabilities_match_canonical_contract():
    canonical = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert capability_document_v1() == canonical
    assert execution_capabilities_generated.SCHEMA_VERSION == canonical["schema_version"]
    assert execution_capabilities_generated.CAPABILITY_VERSION == canonical["capability_version"]


def test_python_normalizes_shared_capability_fixtures():
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    assert fixtures
    for fixture in fixtures:
        assert (
            normalize_execution_config(
                fixture["task"],
                fixture["runner"],
                fixture["input"],
            )
            == fixture["expected"]
        ), fixture["name"]


def test_python_accepted_resolver_drops_inactive_conditional_fields():
    accepted = resolve_accepted_config(
        "image_classification",
        "modal_torchvision",
        {"model": "resnet18", "optimizer": "adamw", "optimizer_momentum": 0},
    )
    assert accepted["optimizer"] == "adamw"
    assert "optimizer_momentum" not in accepted


def test_every_profile_classifies_every_catalog_field():
    document = capability_document_v1()
    catalog_fields = set(document["field_catalog"])
    for profile in document["profiles"].values():
        assert set(profile["fields"]) == catalog_fields


def test_detection_profiles_expose_current_narrow_execution_surface():
    for runner in ("local_simulator", "modal_ultralytics"):
        profile = capability_profile("object_detection", runner)
        for field in ("model", "epochs", "batch_size", "learning_rate", "image_size"):
            assert profile["fields"][field]["classification"] == "executed"
        for field in ("optimizer", "scheduler", "weight_decay", "pretrained"):
            assert profile["fields"][field]["classification"] == "unsupported"


def test_generated_capability_artifacts_are_current():
    result = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "generate_experiment_execution_capabilities.py"),
            "--check",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_modal_image_packages_generated_capability_module():
    generated_path = Path(execution_capabilities_generated.__file__).resolve()
    worker_package = REPOSITORY_ROOT / "services" / "worker" / "worker"
    assert generated_path.is_relative_to(worker_package)
    modal_runtime = (worker_package / "training" / "modal_runtime.py").read_text(encoding="utf-8")
    assert '.add_local_python_source("worker")' in modal_runtime
