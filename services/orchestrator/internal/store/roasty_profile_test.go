package store

import (
	"reflect"
	"testing"

	"model-express/services/orchestrator/internal/policies"
)

func TestRoastyV1IsSeededAndResolvesOnlyFP32ClassificationONNX(t *testing.T) {
	storage := NewMemoryStore()
	profile, err := storage.GetCompatibilityProfile("roasty_v1", "1.0.0")
	if err != nil {
		t.Fatal(err)
	}
	if profile.OwnerAccountID != "" || profile.DocumentHash == "" || profile.CreatedBy != "builtin_generator" {
		t.Fatalf("roasty profile identity = %#v", profile)
	}
	project, _ := storage.CreateProject("roasty", "")
	version, err := storage.CreateExperimentPolicyVersion(policies.PolicyVersion{Document: policies.PolicyDocument{
		SchemaVersion: policies.PolicySchemaVersionV1,
		ProfileRefs:   []policies.ProfileRef{{ID: "roasty_v1", Version: "1.0.0"}},
		Rules:         []policies.Rule{},
	}})
	if err != nil {
		t.Fatal(err)
	}
	zero := int64(0)
	if _, err := storage.SetExperimentPolicyBinding(policies.BindingWrite{Scope: policies.ScopeProject, SubjectID: project.ID, PolicyVersionID: version.ID, ExpectedRevision: &zero}); err != nil {
		t.Fatal(err)
	}
	effective, err := policies.NewResolver(storage).Resolve(policies.ScopeContext{
		AccountID: project.AccountID, ProjectID: project.ID, Task: "image_classification", Runner: "modal_torchvision",
	})
	if err != nil {
		t.Fatal(err)
	}
	for category, expected := range map[string][]string{
		"tasks": {"image_classification"}, "runners": {"modal_torchvision"}, "export_formats": {"onnx"},
		"precisions": {"fp32"}, "runtimes": {"onnxruntime"}, "execution_providers": {"cpu_execution_provider"},
		"execution_requirements": {"cpu_compatible"},
	} {
		if got := effective.Snapshot.PermittedCatalog[category]; !reflect.DeepEqual(got, expected) {
			t.Fatalf("roasty %s = %#v, want %#v", category, got, expected)
		}
	}
}
