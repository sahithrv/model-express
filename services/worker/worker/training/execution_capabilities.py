from __future__ import annotations

from copy import deepcopy
from typing import Any

from worker.training.execution_capabilities_generated import CAPABILITY_DOCUMENT


CAPABILITY_SCHEMA_VERSION_V1 = "experiment_execution_capabilities.v1"
_CLASSIFICATIONS = {"executed", "conditional", "metadata_only", "unsupported"}
_SCHEDULER_STEP_DEFAULT = "max(1, floor(epochs / 3))"


def capability_document_v1() -> dict[str, Any]:
    validate_capability_document(CAPABILITY_DOCUMENT)
    return deepcopy(CAPABILITY_DOCUMENT)


def capability_profile(task: str, runner: str) -> dict[str, Any]:
    validate_capability_document(CAPABILITY_DOCUMENT)
    return deepcopy(_profile_from_document(CAPABILITY_DOCUMENT, task, runner))


def normalize_execution_config(task: str, runner: str, config: dict) -> dict:
    """Apply contract aliases/defaults without changing current worker dispatch behavior."""
    validate_capability_document(CAPABILITY_DOCUMENT)
    profile = _profile_from_document(CAPABILITY_DOCUMENT, task, runner)
    catalog = CAPABILITY_DOCUMENT["field_catalog"]
    root_fields = {path.split(".", 1)[0] for path in catalog}
    normalized = {
        key: deepcopy(value)
        for key, value in config.items()
        if key in root_fields
    }

    for path, value in profile.get("defaults", {}).items():
        if not _has_path(normalized, path):
            _set_path(normalized, path, deepcopy(value))
    for path, expression in profile.get("default_expressions", {}).items():
        if not _has_path(normalized, path):
            _set_path(normalized, path, _evaluate_default(expression, normalized))

    for path in sorted(catalog, key=lambda item: (item.count("."), item)):
        present, value = _value_at_path(normalized, path)
        if not present:
            continue
        definition = catalog[path]
        constraint = profile.get("constraints", {}).get(path, {})
        _set_path(normalized, path, _normalize_value(path, value, definition, constraint))
    return normalized


def validate_capability_document(document: dict[str, Any]) -> None:
    if document.get("schema_version") != CAPABILITY_SCHEMA_VERSION_V1:
        raise ValueError(f"Unsupported capability schema {document.get('schema_version')!r}.")
    if not str(document.get("capability_version") or "").strip():
        raise ValueError("Execution capability_version is required.")
    if set(document.get("classifications") or []) != _CLASSIFICATIONS:
        raise ValueError("Execution capability classifications are incomplete.")

    reason_codes = document.get("reason_codes") or {}
    catalog = document.get("field_catalog") or {}
    profiles = document.get("profiles") or {}
    tasks = document.get("tasks") or {}
    runners = document.get("runners") or {}
    if not reason_codes or not catalog or not profiles or not tasks or not runners:
        raise ValueError(
            "Execution capabilities require reason codes, fields, profiles, tasks, and runners."
        )

    catalog_fields = set(catalog)
    for path, definition in catalog.items():
        if not path or not isinstance(definition, dict) or not definition.get("type"):
            raise ValueError(f"Execution capability field {path!r} must declare a type.")
        values = set(definition.get("values") or [])
        alias_targets = set((definition.get("aliases") or {}).values())
        if values and not alias_targets.issubset(values):
            raise ValueError(f"Execution capability field {path!r} has an invalid alias target.")

    for name, profile in profiles.items():
        if profile.get("task") not in tasks:
            raise ValueError(f"Execution capability profile {name!r} has an unknown task.")
        if profile.get("runner") not in runners:
            raise ValueError(f"Execution capability profile {name!r} has an unknown runner.")
        fields = profile.get("fields") or {}
        if set(fields) != catalog_fields:
            raise ValueError(
                f"Execution capability profile {name!r} must classify every catalog field."
            )
        for path, capability in fields.items():
            if capability.get("classification") not in _CLASSIFICATIONS:
                raise ValueError(
                    f"Execution capability profile {name!r} field {path!r} has an "
                    "unknown classification."
                )
            if capability.get("reason_code") not in reason_codes:
                raise ValueError(
                    f"Execution capability profile {name!r} field {path!r} has an "
                    "unknown reason code."
                )
        for section in ("defaults", "default_expressions", "constraints", "fixed_semantics"):
            if not set((profile.get(section) or {})).issubset(catalog_fields):
                raise ValueError(
                    f"Execution capability profile {name!r} {section} references an unknown field."
                )
        for expression in (profile.get("default_expressions") or {}).values():
            if expression != _SCHEDULER_STEP_DEFAULT:
                raise ValueError(
                    f"Execution capability profile {name!r} has an unsupported default expression."
                )

    referenced_profiles: set[str] = set()
    for task_name, task in tasks.items():
        for runner_name, selection in (task.get("runners") or {}).items():
            if runner_name not in runners:
                raise ValueError(f"Execution capability task {task_name!r} has an unknown runner.")
            profile_name = selection.get("profile")
            if profile_name not in profiles:
                raise ValueError(f"Execution capability task {task_name!r} has an unknown profile.")
            profile = profiles[profile_name]
            if profile.get("task") != task_name or profile.get("runner") != runner_name:
                raise ValueError(
                    f"Execution capability task {task_name!r} runner {runner_name!r} has a "
                    "mismatched profile."
                )
            referenced_profiles.add(profile_name)
    if referenced_profiles != set(profiles):
        raise ValueError("Every execution capability profile must be selected by a task.")


def _profile_from_document(document: dict, task: str, runner: str) -> dict:
    task_capability = document["tasks"].get(task)
    if task_capability is None:
        raise ValueError(f"Unknown execution capability task {task!r}.")
    selection = task_capability.get("runners", {}).get(runner)
    if selection is None:
        raise ValueError(f"Unknown execution capability runner {runner!r} for task {task!r}.")
    profile_name = selection["profile"]
    return document["profiles"][profile_name]


def _normalize_value(path: str, value: object, definition: dict, constraint: dict) -> object:
    field_type = definition["type"]
    if field_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"Execution capability field {path!r} must be a string.")
        normalization = definition.get("normalization")
        if normalization == "trim":
            value = value.strip()
        elif normalization == "lowercase_trim":
            value = value.strip().lower()
        value = (definition.get("aliases") or {}).get(value, value)
        return (constraint.get("aliases") or {}).get(value, value)
    if field_type == "string_array":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"Execution capability field {path!r} must be a string array.")
        return [item.strip() for item in value if item.strip()]
    return deepcopy(value)


def _evaluate_default(expression: str, config: dict) -> int:
    if expression != _SCHEDULER_STEP_DEFAULT:
        raise ValueError(f"Unsupported execution capability default expression {expression!r}.")
    present, epochs = _value_at_path(config, "epochs")
    if not present or isinstance(epochs, bool) or not isinstance(epochs, (int, float)):
        raise ValueError(
            "Execution capability default scheduler_step_size requires integer epochs."
        )
    integer_epochs = int(epochs)
    if integer_epochs != epochs:
        raise ValueError(
            "Execution capability default scheduler_step_size requires integer epochs."
        )
    return max(1, integer_epochs // 3)


def _has_path(config: dict, path: str) -> bool:
    present, _ = _value_at_path(config, path)
    return present


def _value_at_path(config: dict, path: str) -> tuple[bool, object]:
    current: object = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _set_path(config: dict, path: str, value: object) -> None:
    parts = path.split(".")
    current = config
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            nested = {}
            current[part] = nested
        current = nested
    current[parts[-1]] = value
