package catalog_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"

	"model-express/services/orchestrator/internal/catalog"
)

func TestGeneratedCatalogMatchesCanonicalJSON(t *testing.T) {
	canonical := readRepositoryFile(t, "contracts", "model_express_catalog.v1.json")
	generated := []byte(catalog.GeneratedCatalogJSONV1())

	var canonicalDocument any
	if err := json.Unmarshal(canonical, &canonicalDocument); err != nil {
		t.Fatalf("decode canonical catalog: %v", err)
	}
	var generatedDocument any
	if err := json.Unmarshal(generated, &generatedDocument); err != nil {
		t.Fatalf("decode generated catalog: %v", err)
	}
	if !reflect.DeepEqual(canonicalDocument, generatedDocument) {
		t.Fatal("generated Go catalog does not match canonical JSON")
	}

	document := catalog.CatalogV1()
	if document.SchemaVersion != catalog.SchemaVersionV1 {
		t.Fatalf("unexpected catalog schema %q", document.SchemaVersion)
	}
	if document.CatalogVersion != "1.0.0" {
		t.Fatalf("unexpected catalog version %q", document.CatalogVersion)
	}
}

func TestEveryCatalogAliasResolvesToItsCanonicalID(t *testing.T) {
	document := catalog.CatalogV1()
	for category, entries := range document.Categories {
		for _, expected := range entries {
			entry, ok := catalog.Resolve(category, expected.ID)
			if !ok || entry.ID != expected.ID {
				t.Fatalf("canonical ID %s/%s did not resolve", category, expected.ID)
			}
			for _, alias := range expected.Aliases {
				entry, ok := catalog.Resolve(category, alias)
				if !ok || entry.ID != expected.ID {
					t.Fatalf("alias %s/%s resolved to %q, want %q", category, alias, entry.ID, expected.ID)
				}
			}
		}
	}
	if _, ok := catalog.Resolve("models", "unknown_model"); ok {
		t.Fatal("unknown model unexpectedly resolved")
	}
}

func TestCatalogImplicationAndEquivalenceReferencesExist(t *testing.T) {
	document := catalog.CatalogV1()
	for category, entries := range document.Categories {
		for _, entry := range entries {
			for _, reference := range append(entry.Implies, entry.EquivalentTo...) {
				targetCategory, targetID, ok := splitReference(reference)
				if !ok {
					t.Fatalf("invalid reference %s from %s/%s", reference, category, entry.ID)
				}
				resolved, ok := catalog.Resolve(targetCategory, targetID)
				if !ok || resolved.ID != targetID {
					t.Fatalf("unknown reference %s from %s/%s", reference, category, entry.ID)
				}
			}
		}
	}
}

func splitReference(reference string) (string, string, bool) {
	for index, value := range reference {
		if value == '/' && index > 0 && index < len(reference)-1 {
			return reference[:index], reference[index+1:], true
		}
	}
	return "", "", false
}

func readRepositoryFile(t *testing.T, pathParts ...string) []byte {
	t.Helper()
	_, sourceFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test source path")
	}
	repositoryRoot := filepath.Clean(filepath.Join(filepath.Dir(sourceFile), "..", "..", "..", ".."))
	path := filepath.Join(append([]string{repositoryRoot}, pathParts...)...)
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	return contents
}
