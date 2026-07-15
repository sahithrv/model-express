#!/usr/bin/env python3
"""Generate embedded Go and packaged Python execution-capability data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPOSITORY_ROOT / "contracts" / "experiment_execution_capabilities.v1.json"
GO_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "services"
    / "orchestrator"
    / "internal"
    / "execution"
    / "capabilities_generated.go"
)
PYTHON_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "services"
    / "worker"
    / "worker"
    / "training"
    / "execution_capabilities_generated.py"
)


def _load_contract() -> dict[str, Any]:
    with CONTRACT_PATH.open(encoding="utf-8") as source:
        document = json.load(source)
    _validate_contract(document)
    return document


def _validate_contract(document: dict[str, Any]) -> None:
    if document.get("schema_version") != "experiment_execution_capabilities.v1":
        raise ValueError("capability contract must use schema experiment_execution_capabilities.v1")
    if not str(document.get("capability_version") or "").strip():
        raise ValueError("capability contract must declare capability_version")

    classifications = set(document.get("classifications") or [])
    expected_classifications = {"executed", "conditional", "metadata_only", "unsupported"}
    if classifications != expected_classifications:
        raise ValueError(
            f"capability classifications must be {sorted(expected_classifications)}, "
            f"got {sorted(classifications)}"
        )
    reason_codes = document.get("reason_codes") or {}
    field_catalog = document.get("field_catalog") or {}
    profiles = document.get("profiles") or {}
    tasks = document.get("tasks") or {}
    runners = document.get("runners") or {}
    if not reason_codes or not field_catalog or not profiles or not tasks or not runners:
        raise ValueError(
            "capability contract must define reason codes, fields, profiles, tasks, and runners"
        )

    catalog_fields = set(field_catalog)
    for path, definition in field_catalog.items():
        if not path or not isinstance(definition, dict) or not definition.get("type"):
            raise ValueError(f"field_catalog entry {path!r} must declare a type")
        aliases = definition.get("aliases") or {}
        values = set(definition.get("values") or [])
        unknown_alias_targets = set(aliases.values()) - values
        if values and unknown_alias_targets:
            raise ValueError(
                f"field_catalog entry {path!r} aliases unknown values "
                f"{sorted(unknown_alias_targets)}"
            )

    for profile_name, profile in profiles.items():
        if profile.get("task") not in tasks:
            raise ValueError(
                f"profile {profile_name!r} references unknown task {profile.get('task')!r}"
            )
        if profile.get("runner") not in runners:
            raise ValueError(
                f"profile {profile_name!r} references unknown runner {profile.get('runner')!r}"
            )
        profile_fields = set((profile.get("fields") or {}).keys())
        missing = catalog_fields - profile_fields
        extra = profile_fields - catalog_fields
        if missing or extra:
            raise ValueError(
                f"profile {profile_name!r} field coverage differs from field_catalog; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for path, capability in profile["fields"].items():
            classification = capability.get("classification")
            reason_code = capability.get("reason_code")
            if classification not in classifications:
                raise ValueError(
                    f"profile {profile_name!r} field {path!r} has unknown classification "
                    f"{classification!r}"
                )
            if reason_code not in reason_codes:
                raise ValueError(
                    f"profile {profile_name!r} field {path!r} has unknown reason code "
                    f"{reason_code!r}"
                )
        for section in ("defaults", "default_expressions", "constraints", "fixed_semantics"):
            unknown = set((profile.get(section) or {}).keys()) - catalog_fields
            if unknown:
                raise ValueError(
                    f"profile {profile_name!r} {section} references unknown fields "
                    f"{sorted(unknown)}"
                )

    referenced_profiles: set[str] = set()
    for task_name, task in tasks.items():
        for runner_name, selection in (task.get("runners") or {}).items():
            if runner_name not in runners:
                raise ValueError(f"task {task_name!r} references unknown runner {runner_name!r}")
            profile_name = selection.get("profile")
            if profile_name not in profiles:
                raise ValueError(f"task {task_name!r} references unknown profile {profile_name!r}")
            profile = profiles[profile_name]
            if profile.get("task") != task_name or profile.get("runner") != runner_name:
                raise ValueError(
                    f"task {task_name!r} runner {runner_name!r} selects mismatched profile "
                    f"{profile_name!r}"
                )
            referenced_profiles.add(profile_name)
    unreferenced = set(profiles) - referenced_profiles
    if unreferenced:
        raise ValueError(f"capability profiles are not selected by a task: {sorted(unreferenced)}")


def _canonical_json(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _source_hash(canonical_json: str) -> str:
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _render_go(canonical_json: str, source_hash: str) -> str:
    header = "// Code generated by the experiment execution capability generator; DO NOT EDIT."
    return f'''{header}
// Source SHA-256: {source_hash}

package execution

const generatedExecutionCapabilitiesJSON = `{canonical_json}`
'''


def _render_python(canonical_json: str, source_hash: str) -> str:
    header = "# Code generated by the experiment execution capability generator; DO NOT EDIT."
    return f'''{header}
# Source SHA-256: {source_hash}

from __future__ import annotations

import json


CAPABILITY_DOCUMENT = json.loads(r\'''{canonical_json}\''')
CAPABILITY_VERSION = str(CAPABILITY_DOCUMENT["capability_version"])
SCHEMA_VERSION = str(CAPABILITY_DOCUMENT["schema_version"])
'''


def _write_or_check(path: Path, expected: str, check: bool) -> bool:
    if check:
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"missing generated artifact: {path.relative_to(REPOSITORY_ROOT)}")
            return False
        if actual != expected:
            print(f"stale generated artifact: {path.relative_to(REPOSITORY_ROOT)}")
            return False
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")
    print(f"generated {path.relative_to(REPOSITORY_ROOT)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when generated artifacts do not match the canonical capability document",
    )
    args = parser.parse_args()

    document = _load_contract()
    canonical_json = _canonical_json(document)
    source_hash = _source_hash(canonical_json)
    results = [
        _write_or_check(GO_OUTPUT_PATH, _render_go(canonical_json, source_hash), args.check),
        _write_or_check(
            PYTHON_OUTPUT_PATH,
            _render_python(canonical_json, source_hash),
            args.check,
        ),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
