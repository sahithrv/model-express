from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from worker.model_express_catalog import (
    available_catalog_ids,
    catalog_document_v1,
    require_catalog_id,
    resolve_catalog_entry,
)
from worker.model_express_catalog_generated import CATALOG_DOCUMENT


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class ModelExpressCatalogTests(unittest.TestCase):
    def test_generated_python_catalog_matches_canonical_json(self) -> None:
        canonical = json.loads(
            (REPOSITORY_ROOT / "contracts" / "model_express_catalog.v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(CATALOG_DOCUMENT, canonical)
        self.assertEqual(catalog_document_v1(), canonical)

    def test_all_aliases_resolve_to_the_same_canonical_ids(self) -> None:
        for category, entries in CATALOG_DOCUMENT["categories"].items():
            for expected in entries:
                resolved = resolve_catalog_entry(category, expected["id"])
                self.assertIsNotNone(resolved)
                self.assertEqual(resolved["id"], expected["id"])
                for alias in expected["aliases"]:
                    resolved = resolve_catalog_entry(category, alias)
                    self.assertIsNotNone(resolved)
                    self.assertEqual(resolved["id"], expected["id"])
        self.assertIsNone(resolve_catalog_entry("models", "unknown_model"))

    def test_worker_model_registries_and_export_formats_do_not_drift(self) -> None:
        classification_models = {
            entry["id"]
            for entry in CATALOG_DOCUMENT["categories"]["models"]
            if entry["available"]
            and "image_classification" in entry["tasks"]
            and "modal_torchvision" in entry["runners"]
        }
        detection_models = {
            entry["id"]
            for entry in CATALOG_DOCUMENT["categories"]["models"]
            if entry["available"]
            and "object_detection" in entry["tasks"]
            and "modal_ultralytics" in entry["runners"]
        }
        self.assertEqual(
            _normalized_dispatch_ids(
                REPOSITORY_ROOT
                / "services"
                / "worker"
                / "worker"
                / "training"
                / "modal_app.py",
                "_build_model",
            ),
            classification_models,
        )
        self.assertEqual(
            _normalized_dispatch_ids(
                REPOSITORY_ROOT / "services" / "worker" / "worker" / "champion_jobs.py",
                "_build_torchvision_model",
            ),
            classification_models,
        )
        local_path = (
            REPOSITORY_ROOT / "services" / "worker" / "worker" / "training" / "local.py"
        )
        self.assertEqual(
            _metadata_registry_ids(local_path, "_local_model_profile"),
            classification_models,
        )
        self.assertEqual(
            _metadata_registry_ids(local_path, "_local_detection_model_profile"),
            detection_models,
        )
        champion_source = (
            REPOSITORY_ROOT / "services" / "worker" / "worker" / "champion_jobs.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'SUPPORTED_EXPORT_FORMATS = available_catalog_ids("export_formats")',
            champion_source,
        )
        self.assertEqual(
            available_catalog_ids("export_formats"),
            frozenset({"onnx", "torchscript", "pytorch", "safetensors"}),
        )

    def test_unknown_models_fail_before_torchvision_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown or unavailable models ID"):
            require_catalog_id(
                "models",
                "unknown_model",
                task="image_classification",
                runner="modal_torchvision",
            )
        for relative_path, function_name in (
            (("training", "modal_app.py"), "_build_model"),
            (("champion_jobs.py",), "_build_torchvision_model"),
        ):
            path = REPOSITORY_ROOT / "services" / "worker" / "worker"
            for part in relative_path:
                path /= part
            self.assertTrue(_requires_catalog_before_framework_import(path, function_name))


def _normalized_dispatch_ids(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    identifiers: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
            continue
        if not isinstance(node.left, ast.Name) or node.left.id != "normalized":
            continue
        if len(node.comparators) == 1 and isinstance(node.comparators[0], ast.Constant):
            value = node.comparators[0].value
            if isinstance(value, str):
                identifiers.add(value)
    return identifiers


def _metadata_registry_ids(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    for node in function.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "metadata" for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Dict):
            return {
                str(key.value)
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
    raise AssertionError(f"metadata registry not found in {function_name}")


def _requires_catalog_before_framework_import(path: Path, function_name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    require_line = min(
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require_catalog_id"
    )
    import_lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom) and node.module in {"torch", "torchvision"}
    ]
    return bool(import_lines) and require_line < min(import_lines)


if __name__ == "__main__":
    unittest.main()
