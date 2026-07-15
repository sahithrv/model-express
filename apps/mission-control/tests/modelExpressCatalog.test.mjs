import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const appRoot = path.resolve(testDir, "..");
const repositoryRoot = path.resolve(appRoot, "..", "..");

function generatedCatalogDocument() {
  const source = fs.readFileSync(
    path.join(appRoot, "src", "modelExpressCatalog.generated.ts"),
    "utf8",
  );
  const match = source.match(/JSON\.parse\(String\.raw`([\s\S]*)`\) as \{/);
  assert.ok(match, "generated TypeScript catalog must embed canonical JSON");
  return JSON.parse(match[1]);
}

test("generated TypeScript catalog exactly matches canonical JSON", () => {
  const canonical = JSON.parse(
    fs.readFileSync(
      path.join(repositoryRoot, "contracts", "model_express_catalog.v1.json"),
      "utf8",
    ),
  );
  assert.deepEqual(generatedCatalogDocument(), canonical);
});

test("TypeScript catalog aliases resolve deterministically", () => {
  const catalog = generatedCatalogDocument();
  for (const [category, entries] of Object.entries(catalog.categories)) {
    const resolved = new Map();
    for (const entry of entries) {
      resolved.set(entry.id.toLowerCase(), entry.id);
      for (const alias of entry.aliases) {
        assert.equal(resolved.has(alias.toLowerCase()), false, `${category}/${alias} is unique`);
        resolved.set(alias.toLowerCase(), entry.id);
      }
    }
    for (const entry of entries) {
      assert.equal(resolved.get(entry.id.toLowerCase()), entry.id);
      for (const alias of entry.aliases) {
        assert.equal(resolved.get(alias.toLowerCase()), entry.id);
      }
    }
  }
});
