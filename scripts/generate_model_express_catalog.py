#!/usr/bin/env python3
"""Generate deterministic Go, Python, and TypeScript catalog bindings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPOSITORY_ROOT / "contracts" / "model_express_catalog.v1.json"
EXECUTION_CAPABILITIES_PATH = (
    REPOSITORY_ROOT / "contracts" / "experiment_execution_capabilities.v1.json"
)
GO_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "services"
    / "orchestrator"
    / "internal"
    / "catalog"
    / "catalog_generated.go"
)
PYTHON_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "services"
    / "worker"
    / "worker"
    / "model_express_catalog_generated.py"
)
TYPESCRIPT_OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "apps"
    / "mission-control"
    / "src"
    / "modelExpressCatalog.generated.ts"
)

REQUIRED_CATEGORIES = {
    "tasks",
    "runners",
    "models",
    "model_families",
    "deployment_tiers",
    "latency_tiers",
    "model_size_tiers",
    "fine_tuning_modes",
    "resize_strategies",
    "crop_strategies",
    "bounding_box_modes",
    "normalization_strategies",
    "augmentation_operations",
    "augmentation_policies",
    "optimizers",
    "schedulers",
    "resolution_strategies",
    "class_balancing_strategies",
    "sampling_strategies",
    "export_formats",
    "precisions",
    "runtimes",
    "execution_providers",
    "execution_requirements",
}
ENTRY_FIELDS = {
    "id",
    "aliases",
    "available",
    "tasks",
    "runners",
    "attributes",
    "implies",
    "equivalent_to",
}
EXECUTION_FIELD_CATEGORIES = {
    "augmentation.color_jitter": "augmentation_operations",
    "augmentation.horizontal_flip": "augmentation_operations",
    "augmentation.random_crop": "augmentation_operations",
    "augmentation.random_erasing": "augmentation_operations",
    "augmentation.random_rotation": "augmentation_operations",
    "augmentation.vertical_flip": "augmentation_operations",
    "augmentation_policy": "augmentation_policies",
    "augmentation_policy_config.policy_type": "augmentation_policies",
    "class_balancing": "class_balancing_strategies",
    "fine_tune_strategy": "fine_tuning_modes",
    "model": "models",
    "optimizer": "optimizers",
    "preprocessing.bbox_mode": "bounding_box_modes",
    "preprocessing.crop_strategy": "crop_strategies",
    "preprocessing.normalization": "normalization_strategies",
    "preprocessing.resize_strategy": "resize_strategies",
    "resolution_strategy": "resolution_strategies",
    "sampling_strategy": "sampling_strategies",
    "scheduler": "schedulers",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def _load_and_validate() -> dict[str, Any]:
    document = _load_json(CATALOG_PATH)
    _validate_catalog(document)
    _validate_execution_capability_references(
        document,
        _load_json(EXECUTION_CAPABILITIES_PATH),
    )
    return document


def _validate_catalog(document: dict[str, Any]) -> None:
    if document.get("schema_version") != "model_express_catalog.v1":
        raise ValueError("catalog must use schema model_express_catalog.v1")
    if not str(document.get("catalog_version") or "").strip():
        raise ValueError("catalog must declare catalog_version")
    categories = document.get("categories")
    if not isinstance(categories, dict):
        raise ValueError("catalog must declare categories")
    missing = REQUIRED_CATEGORIES - set(categories)
    if missing:
        raise ValueError(f"catalog is missing required categories {sorted(missing)}")

    canonical_ids: dict[str, set[str]] = {}
    for category, entries in categories.items():
        if not isinstance(entries, list):
            raise ValueError(f"catalog category {category!r} must be a list")
        ids: set[str] = set()
        normalized_identifiers: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != ENTRY_FIELDS:
                raise ValueError(
                    f"catalog entry in {category!r} must contain exactly {sorted(ENTRY_FIELDS)}"
                )
            identifier = entry["id"]
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"catalog category {category!r} contains an invalid ID")
            if identifier in ids:
                raise ValueError(f"catalog category {category!r} duplicates ID {identifier!r}")
            normalized_identifier = identifier.strip().lower()
            if normalized_identifier in normalized_identifiers:
                raise ValueError(
                    f"catalog category {category!r} duplicates normalized ID {identifier!r}"
                )
            ids.add(identifier)
            normalized_identifiers.add(normalized_identifier)
            if not isinstance(entry["aliases"], list) or not all(
                isinstance(alias, str) and alias for alias in entry["aliases"]
            ):
                raise ValueError(f"catalog entry {category}/{identifier} has invalid aliases")
            for alias in entry["aliases"]:
                normalized_alias = alias.strip().lower()
                if normalized_alias in normalized_identifiers:
                    raise ValueError(f"catalog category {category!r} duplicates alias {alias!r}")
                normalized_identifiers.add(normalized_alias)
            if not isinstance(entry["available"], bool):
                raise ValueError(f"catalog entry {category}/{identifier} has invalid availability")
            for key in ("tasks", "runners", "implies", "equivalent_to"):
                if not isinstance(entry[key], list) or not all(
                    isinstance(value, str) for value in entry[key]
                ):
                    raise ValueError(f"catalog entry {category}/{identifier} has invalid {key}")
            if not isinstance(entry["attributes"], dict):
                raise ValueError(f"catalog entry {category}/{identifier} has invalid attributes")
        canonical_ids[category] = ids

    task_ids = canonical_ids["tasks"]
    runner_ids = canonical_ids["runners"]
    for category, entries in categories.items():
        for entry in entries:
            identifier = entry["id"]
            unknown_tasks = set(entry["tasks"]) - task_ids
            unknown_runners = set(entry["runners"]) - runner_ids
            if unknown_tasks:
                raise ValueError(
                    f"catalog entry {category}/{identifier} references unknown tasks "
                    f"{sorted(unknown_tasks)}"
                )
            if unknown_runners:
                raise ValueError(
                    f"catalog entry {category}/{identifier} references unknown runners "
                    f"{sorted(unknown_runners)}"
                )
            for reference in entry["implies"] + entry["equivalent_to"]:
                target_category, separator, target_id = reference.partition("/")
                if not separator or target_id not in canonical_ids.get(target_category, set()):
                    raise ValueError(
                        f"catalog entry {category}/{identifier} has unknown reference {reference!r}"
                    )

    for model in categories["models"]:
        attributes = model["attributes"]
        references = {
            "family": "model_families",
            "deployment_tier": "deployment_tiers",
            "latency_tier": "latency_tiers",
        }
        for attribute, category in references.items():
            value = attributes.get(attribute)
            if value not in canonical_ids[category]:
                raise ValueError(
                    f"model {model['id']!r} has unknown {attribute} {value!r}"
                )
        unknown_modes = set(attributes.get("fine_tuning_modes") or []) - canonical_ids[
            "fine_tuning_modes"
        ]
        if unknown_modes:
            raise ValueError(
                f"model {model['id']!r} references unknown fine-tuning modes "
                f"{sorted(unknown_modes)}"
            )


def _validate_execution_capability_references(
    catalog: dict[str, Any], execution: dict[str, Any]
) -> None:
    categories = catalog["categories"]
    canonical = {
        category: {entry["id"] for entry in entries}
        for category, entries in categories.items()
    }
    aliases = {
        category: {alias: entry["id"] for entry in entries for alias in entry["aliases"]}
        for category, entries in categories.items()
    }
    unknown_tasks = set(execution.get("tasks") or {}) - canonical["tasks"]
    unknown_runners = set(execution.get("runners") or {}) - canonical["runners"]
    if unknown_tasks or unknown_runners:
        raise ValueError(
            "execution capabilities reference catalog-unknown tasks/runners: "
            f"tasks={sorted(unknown_tasks)}, runners={sorted(unknown_runners)}"
        )

    field_catalog = execution.get("field_catalog") or {}
    for field_path, category in EXECUTION_FIELD_CATEGORIES.items():
        definition = field_catalog.get(field_path)
        if not isinstance(definition, dict):
            raise ValueError(f"execution capabilities are missing field {field_path!r}")
        if definition.get("catalog_category") != category:
            raise ValueError(
                f"execution field {field_path!r} must reference catalog category {category!r}"
            )
        values = set(definition.get("values") or [])
        catalog_id = definition.get("catalog_id")
        if catalog_id is not None:
            values.add(catalog_id)
        unknown_values = values - canonical[category]
        if unknown_values:
            raise ValueError(
                f"execution field {field_path!r} references unknown catalog IDs "
                f"{sorted(unknown_values)}"
            )
        execution_aliases = definition.get("aliases") or {}
        unknown_aliases = {
            alias: target
            for alias, target in execution_aliases.items()
            if aliases[category].get(alias) != target
        }
        if unknown_aliases:
            raise ValueError(
                f"execution field {field_path!r} aliases diverge from the catalog: "
                f"{unknown_aliases}"
            )


def _canonical_json(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _source_hash(canonical_json: str) -> str:
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _render_go(canonical_json: str, source_hash: str) -> str:
    return f'''// Code generated by generate_model_express_catalog.py; DO NOT EDIT.
// Source SHA-256: {source_hash}

package catalog

const generatedCatalogJSONV1 = `{canonical_json}`
'''


def _render_python(canonical_json: str, source_hash: str) -> str:
    return f'''# Code generated by generate_model_express_catalog.py; DO NOT EDIT.
# Source SHA-256: {source_hash}

from __future__ import annotations

import json


CATALOG_DOCUMENT = json.loads(r\'''{canonical_json}\''')
CATALOG_VERSION = str(CATALOG_DOCUMENT["catalog_version"])
SCHEMA_VERSION = str(CATALOG_DOCUMENT["schema_version"])
CATALOG_IDS = {{
    category: frozenset(entry["id"] for entry in entries)
    for category, entries in CATALOG_DOCUMENT["categories"].items()
}}
CATALOG_ALIASES = {{
    category: {{alias: entry["id"] for entry in entries for alias in entry["aliases"]}}
    for category, entries in CATALOG_DOCUMENT["categories"].items()
}}
'''


def _render_typescript(canonical_json: str, source_hash: str) -> str:
    escaped = canonical_json.replace("`", "\\`").replace("${", "\\${")
    return f'''// Code generated by generate_model_express_catalog.py; DO NOT EDIT.
// Source SHA-256: {source_hash}

export const MODEL_EXPRESS_CATALOG = JSON.parse(String.raw`{escaped}`) as {{
  schema_version: string;
  catalog_version: string;
  categories: Record<string, readonly unknown[]>;
}};
export const MODEL_EXPRESS_CATALOG_VERSION = MODEL_EXPRESS_CATALOG.catalog_version;
export const MODEL_EXPRESS_CATALOG_SCHEMA_VERSION = MODEL_EXPRESS_CATALOG.schema_version;
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
        help="fail when generated artifacts differ from the canonical catalog",
    )
    args = parser.parse_args()

    document = _load_and_validate()
    canonical_json = _canonical_json(document)
    source_hash = _source_hash(canonical_json)
    results = [
        _write_or_check(GO_OUTPUT_PATH, _render_go(canonical_json, source_hash), args.check),
        _write_or_check(
            PYTHON_OUTPUT_PATH,
            _render_python(canonical_json, source_hash),
            args.check,
        ),
        _write_or_check(
            TYPESCRIPT_OUTPUT_PATH,
            _render_typescript(canonical_json, source_hash),
            args.check,
        ),
    ]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
