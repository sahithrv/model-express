package execution_test

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/plans"
)

type normalizationFixture struct {
	Name     string         `json:"name"`
	Task     string         `json:"task"`
	Runner   string         `json:"runner"`
	Input    map[string]any `json:"input"`
	Expected map[string]any `json:"expected"`
}

func TestGeneratedCapabilitiesMatchCanonicalContract(t *testing.T) {
	canonical := readRepositoryFile(t, "contracts", "experiment_execution_capabilities.v1.json")
	embedded := []byte(execution.GeneratedCapabilitiesJSONV1())

	var canonicalDocument any
	if err := json.Unmarshal(canonical, &canonicalDocument); err != nil {
		t.Fatalf("decode canonical capabilities: %v", err)
	}
	var embeddedDocument any
	if err := json.Unmarshal(embedded, &embeddedDocument); err != nil {
		t.Fatalf("decode embedded capabilities: %v", err)
	}
	if !reflect.DeepEqual(canonicalDocument, embeddedDocument) {
		t.Fatal("embedded Go capabilities do not match the canonical contract")
	}

	document := execution.CapabilitiesV1()
	if document.SchemaVersion != execution.CapabilitySchemaVersionV1 {
		t.Fatalf("unexpected capability schema %q", document.SchemaVersion)
	}
	if document.CapabilityVersion != "1.0.0" {
		t.Fatalf("unexpected capability version %q", document.CapabilityVersion)
	}
}

func TestEveryPlannedExperimentFieldHasEveryTaskRunnerClassification(t *testing.T) {
	document := execution.CapabilitiesV1()
	expectedPaths := plannedExperimentCapabilityPaths()
	catalogPaths := sortedMapKeys(document.FieldCatalog)
	if !reflect.DeepEqual(catalogPaths, expectedPaths) {
		t.Fatalf("field catalog does not match PlannedExperiment fields\nwant: %v\n got: %v", expectedPaths, catalogPaths)
	}

	for profileName, profile := range document.Profiles {
		profilePaths := sortedMapKeys(profile.Fields)
		if !reflect.DeepEqual(profilePaths, expectedPaths) {
			t.Fatalf("profile %s does not classify every plan field", profileName)
		}
	}
}

func TestGoNormalizesSharedCapabilityFixtures(t *testing.T) {
	fixtureBytes := readRepositoryFile(
		t,
		"contracts",
		"experiment_execution_capabilities.v1.fixtures.json",
	)
	var fixtures []normalizationFixture
	if err := json.Unmarshal(fixtureBytes, &fixtures); err != nil {
		t.Fatalf("decode normalization fixtures: %v", err)
	}
	if len(fixtures) == 0 {
		t.Fatal("normalization fixtures must not be empty")
	}
	for _, fixture := range fixtures {
		fixture := fixture
		t.Run(fixture.Name, func(t *testing.T) {
			actual, err := execution.NormalizeExecutionConfig(
				fixture.Task,
				fixture.Runner,
				fixture.Input,
			)
			if err != nil {
				t.Fatalf("normalize fixture: %v", err)
			}
			actualJSON, err := json.Marshal(actual)
			if err != nil {
				t.Fatalf("encode normalized fixture: %v", err)
			}
			expectedJSON, err := json.Marshal(fixture.Expected)
			if err != nil {
				t.Fatalf("encode expected fixture: %v", err)
			}
			if string(actualJSON) != string(expectedJSON) {
				t.Fatalf("normalized fixture mismatch\nwant: %s\n got: %s", expectedJSON, actualJSON)
			}
		})
	}
}

func TestDetectionProfilesExposeCurrentNarrowExecutionSurface(t *testing.T) {
	for _, runner := range []string{"local_simulator", "modal_ultralytics"} {
		profile, err := execution.CapabilityProfileFor("object_detection", runner)
		if err != nil {
			t.Fatalf("load detection profile for %s: %v", runner, err)
		}
		for _, field := range []string{"model", "epochs", "batch_size", "learning_rate", "image_size"} {
			if got := profile.Fields[field].Classification; got != "executed" {
				t.Fatalf("%s field %s classification = %q, want executed", runner, field, got)
			}
		}
		for _, field := range []string{"optimizer", "scheduler", "weight_decay", "pretrained"} {
			if got := profile.Fields[field].Classification; got != "unsupported" {
				t.Fatalf("%s field %s classification = %q, want unsupported", runner, field, got)
			}
		}
	}
}

func plannedExperimentCapabilityPaths() []string {
	paths := jsonFieldNames(reflect.TypeOf(plans.PlannedExperiment{}), "")
	paths = append(paths, jsonFieldNames(reflect.TypeOf(plans.Preprocessing{}), "preprocessing.")...)
	paths = append(
		paths,
		jsonFieldNames(reflect.TypeOf(plans.AugmentationPolicyConfig{}), "augmentation_policy_config.")...,
	)
	paths = append(paths,
		"augmentation.color_jitter",
		"augmentation.horizontal_flip",
		"augmentation.random_crop",
		"augmentation.random_erasing",
		"augmentation.random_rotation",
		"augmentation.vertical_flip",
		"class_balancing_config.effective_number_beta",
		"class_balancing_config.focal_loss_gamma",
	)
	sort.Strings(paths)
	return paths
}

func jsonFieldNames(valueType reflect.Type, prefix string) []string {
	paths := make([]string, 0, valueType.NumField())
	for index := 0; index < valueType.NumField(); index++ {
		tag := strings.Split(valueType.Field(index).Tag.Get("json"), ",")[0]
		if tag == "" || tag == "-" {
			continue
		}
		paths = append(paths, prefix+tag)
	}
	return paths
}

func sortedMapKeys[T any](values map[string]T) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
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
