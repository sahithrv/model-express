from __future__ import annotations

import json
import re
import unittest

from scripts import generate_model_express_catalog as generator


class ModelExpressCatalogGeneratorTests(unittest.TestCase):
    def test_all_language_outputs_embed_exact_canonical_document_deterministically(self) -> None:
        document = generator._load_and_validate()
        canonical = generator._canonical_json(document)
        source_hash = generator._source_hash(canonical)

        go_first = generator._render_go(canonical, source_hash)
        python_first = generator._render_python(canonical, source_hash)
        typescript_first = generator._render_typescript(canonical, source_hash)
        self.assertEqual(go_first, generator._render_go(canonical, source_hash))
        self.assertEqual(python_first, generator._render_python(canonical, source_hash))
        self.assertEqual(typescript_first, generator._render_typescript(canonical, source_hash))

        embedded = [
            re.search(r"generatedCatalogJSONV1 = `([\s\S]*)`\n$", go_first),
            re.search(r"CATALOG_DOCUMENT = json\.loads\(r'''([\s\S]*)'''\)", python_first),
            re.search(r"JSON\.parse\(String\.raw`([\s\S]*)`\) as", typescript_first),
        ]
        for match in embedded:
            self.assertIsNotNone(match)
            self.assertEqual(json.loads(match.group(1)), document)

    def test_execution_capability_ids_and_aliases_cross_validate(self) -> None:
        catalog = generator._load_json(generator.CATALOG_PATH)
        execution = generator._load_json(generator.EXECUTION_CAPABILITIES_PATH)
        generator._validate_execution_capability_references(catalog, execution)

        categories = catalog["categories"]
        aliases = {
            category: {alias: entry["id"] for entry in entries for alias in entry["aliases"]}
            for category, entries in categories.items()
        }
        for field, category in generator.EXECUTION_FIELD_CATEGORIES.items():
            definition = execution["field_catalog"][field]
            for alias, target in (definition.get("aliases") or {}).items():
                self.assertEqual(aliases[category][alias], target)


if __name__ == "__main__":
    unittest.main()
