from __future__ import annotations

from copy import deepcopy

from worker.model_express_catalog_generated import CATALOG_DOCUMENT


CATALOG_SCHEMA_VERSION_V1 = "model_express_catalog.v1"


def catalog_document_v1() -> dict:
    validate_catalog_document(CATALOG_DOCUMENT)
    return deepcopy(CATALOG_DOCUMENT)


def resolve_catalog_entry(
    category: str,
    value: str,
    *,
    available_only: bool = False,
    task: str = "",
    runner: str = "",
) -> dict | None:
    normalized = str(value or "").strip().lower()
    for entry in CATALOG_DOCUMENT["categories"].get(category, []):
        identifiers = [entry["id"], *entry["aliases"]]
        if normalized not in {str(identifier).strip().lower() for identifier in identifiers}:
            continue
        if available_only and not entry["available"]:
            return None
        if task and task not in entry["tasks"]:
            return None
        if runner and runner not in entry["runners"]:
            return None
        return deepcopy(entry)
    return None


def available_catalog_ids(category: str) -> frozenset[str]:
    return frozenset(
        str(entry["id"])
        for entry in CATALOG_DOCUMENT["categories"].get(category, [])
        if entry["available"]
    )


def require_catalog_id(
    category: str,
    value: str,
    *,
    task: str = "",
    runner: str = "",
) -> str:
    entry = resolve_catalog_entry(
        category,
        value,
        available_only=True,
        task=task,
        runner=runner,
    )
    if entry is None:
        context = ""
        if task:
            context += f" for task {task!r}"
        if runner:
            context += f" on runner {runner!r}"
        raise ValueError(f"Unknown or unavailable {category} ID {value!r}{context}.")
    return str(entry["id"])


def validate_catalog_document(document: dict) -> None:
    if document.get("schema_version") != CATALOG_SCHEMA_VERSION_V1:
        raise ValueError(f"Unsupported catalog schema {document.get('schema_version')!r}.")
    if not str(document.get("catalog_version") or "").strip():
        raise ValueError("Catalog version is required.")
    categories = document.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Catalog categories are required.")
    for category, entries in categories.items():
        if not isinstance(entries, list):
            raise ValueError(f"Catalog category {category!r} must be a list.")
        identifiers: set[str] = set()
        for entry in entries:
            identifier = str(entry.get("id") or "").strip().lower()
            if not identifier or identifier in identifiers:
                raise ValueError(f"Catalog category {category!r} has an invalid ID.")
            identifiers.add(identifier)
            for alias in entry.get("aliases") or []:
                normalized_alias = str(alias).strip().lower()
                if not normalized_alias or normalized_alias in identifiers:
                    raise ValueError(
                        f"Catalog category {category!r} has an invalid alias {alias!r}."
                    )
                identifiers.add(normalized_alias)
