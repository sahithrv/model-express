package api

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"testing"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/catalog"
)

func TestSupportedModelCatalogMatchesCanonicalCatalog(t *testing.T) {
	assertModelSpecsMatchCatalog(t, "image_classification", supportedModelCatalog())
	assertModelSpecsMatchCatalog(t, "object_detection", supportedYOLODetectorModelCatalog())
}

func TestHardCodedPlannerAndReviewerModelsExistInCatalog(t *testing.T) {
	root := repositoryRootForCatalogTest(t)
	for _, path := range []string{
		filepath.Join(root, "services", "orchestrator", "internal", "agents", "planner.go"),
		filepath.Join(root, "services", "orchestrator", "internal", "agents", "reviewer.go"),
	} {
		parsed, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", path, err)
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			field, ok := node.(*ast.KeyValueExpr)
			if !ok {
				return true
			}
			name, ok := field.Key.(*ast.Ident)
			if !ok || name.Name != "Model" {
				return true
			}
			literal, ok := field.Value.(*ast.BasicLit)
			if !ok || literal.Kind != token.STRING {
				return true
			}
			model, err := strconv.Unquote(literal.Value)
			if err != nil {
				t.Fatalf("decode model literal %s: %v", literal.Value, err)
			}
			entry, ok := catalog.Resolve("models", model)
			if !ok || !entry.Available {
				t.Errorf("%s references catalog-unknown model %q", path, model)
			}
			return true
		})
	}
}

func TestGeneratedAllowLookupsPreserveCurrentAliases(t *testing.T) {
	tests := []struct {
		value   string
		allowed map[string]bool
	}{
		{"letterbox", allowedResizeStrategies()},
		{"trivialaugmentwide", allowedAugmentationPolicies()},
		{"class_weighted_loss", allowedClassBalancingStrategies()},
	}
	for _, test := range tests {
		if !allowedExperimentValue(test.value, test.allowed) {
			t.Fatalf("catalog alias %q is not accepted", test.value)
		}
	}
	for _, format := range catalog.CanonicalIDs("export_formats", true) {
		if got := normalizeChampionExportFormat(format); got != format {
			t.Fatalf("export format %q normalized to %q", format, got)
		}
	}
}

func assertModelSpecsMatchCatalog(t *testing.T, task string, actual []agents.SupportedModelSpec) {
	t.Helper()
	expected := supportedModelCatalogForTask(task)
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("supported %s model specs drifted from catalog", task)
	}
	for _, spec := range actual {
		entry, ok := catalog.Resolve("models", spec.Name)
		if !ok || !entry.Available || !containsString(entry.Tasks, task) {
			t.Fatalf("supported model %q is not available for task %q", spec.Name, task)
		}
	}
}

func repositoryRootForCatalogTest(t *testing.T) string {
	t.Helper()
	_, sourceFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve catalog test source")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(sourceFile), "..", "..", "..", ".."))
}
