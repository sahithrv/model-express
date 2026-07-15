package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/store"
)

func TestPolicyPreviewIsReadOnlyAndBackwardCompatible(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("preview", "")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "memory://dataset", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	router := NewRouter(memoryStore)
	request := httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/experiment-policy/preview?dataset_id="+dataset.ID, nil)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("preview status = %d: %s", response.Code, response.Body.String())
	}
	var payload policyPreviewResponse
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if !payload.ReadOnly || payload.AuditRecorded || payload.Decision != policies.DecisionAllowed {
		t.Fatalf("preview metadata = %#v", payload)
	}
	if payload.Snapshot.ImplicitProfile == nil || payload.Snapshot.ImplicitProfile.ID != policies.ImplicitAllowAllProfileKey {
		t.Fatalf("implicit compatibility profile = %#v", payload.Snapshot.ImplicitProfile)
	}
	if got, want := payload.PermittedCounts["models"], len(catalog.CanonicalIDs("models", true)); got != want {
		t.Fatalf("model count = %d, want %d", got, want)
	}
	evaluations, err := memoryStore.ListExperimentPolicyEvaluations(project.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(evaluations) != 0 {
		t.Fatalf("read-only preview wrote audit rows: %#v", evaluations)
	}

	writeAttempt := httptest.NewRecorder()
	router.ServeHTTP(writeAttempt, httptest.NewRequest(http.MethodPut, "/projects/"+project.ID+"/experiment-policy/preview", strings.NewReader(`{}`)))
	if writeAttempt.Code != http.StatusNotFound {
		t.Fatalf("preview unexpectedly exposed a write method: %d %s", writeAttempt.Code, writeAttempt.Body.String())
	}
}

func TestPolicyPreviewReturnsEffectiveDenialsAndStructuredNoSpaceError(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("preview denied", "")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "memory://dataset", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	version, err := memoryStore.CreateExperimentPolicyVersion(policies.PolicyVersion{
		Document: policies.PolicyDocument{
			SchemaVersion: policies.PolicySchemaVersionV1,
			ProfileRefs:   []policies.ProfileRef{},
			Rules: []policies.Rule{{
				ID: "deny_all_models", Effect: policies.EffectDeny,
				Selector: policies.Selector{
					Kind: policies.SelectorCatalogIDs, Catalog: "models",
					IDs: catalog.CanonicalIDs("models", true),
				},
			}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	expected := int64(0)
	if _, err := memoryStore.SetExperimentPolicyBinding(policies.BindingWrite{
		Scope: policies.ScopeProject, SubjectID: project.ID,
		PolicyVersionID: version.ID, ExpectedRevision: &expected,
	}); err != nil {
		t.Fatal(err)
	}
	router := NewRouter(memoryStore)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/experiment-policy/preview?dataset_id="+dataset.ID, nil))
	if response.Code != http.StatusUnprocessableEntity {
		t.Fatalf("preview status = %d: %s", response.Code, response.Body.String())
	}
	var payload struct {
		Code                policies.ReasonCode `json:"code"`
		Decision            string              `json:"decision"`
		EffectivePolicyHash string              `json:"effective_policy_hash"`
		BlockedDimensions   []string            `json:"blocked_dimensions"`
		AuditRecorded       bool                `json:"audit_recorded"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Code != policies.ReasonNoValidConfiguration || payload.Decision != policies.DecisionDenied || payload.EffectivePolicyHash == "" {
		t.Fatalf("structured policy error = %#v", payload)
	}
	if !reflect.DeepEqual(payload.BlockedDimensions, []string{"models"}) || payload.AuditRecorded {
		t.Fatalf("blocked dimensions = %#v", payload)
	}
}

func TestPolicyPreviewAuthorizationAndScopeOwnership(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_ALLOW_LAN", "")
	t.Setenv("MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE", "")
	t.Setenv("MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL", "")
	memoryStore := store.NewMemoryStore()
	first, err := memoryStore.CreateProject("first", "")
	if err != nil {
		t.Fatal(err)
	}
	firstDataset, err := memoryStore.CreateDataset(first.ID, "first dataset", "memory://first", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	second, err := memoryStore.CreateProject("second", "")
	if err != nil {
		t.Fatal(err)
	}
	secondDataset, err := memoryStore.CreateDataset(second.ID, "second dataset", "memory://second", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	secondJob, err := memoryStore.CreateJob(second.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": secondDataset.ID})
	if err != nil {
		t.Fatal(err)
	}
	router := NewRouter(memoryStore)
	for _, path := range []string{
		"/projects/" + first.ID + "/experiment-policy/preview?dataset_id=" + secondDataset.ID,
		"/projects/" + first.ID + "/experiment-policy/preview?job_id=" + secondJob.ID,
	} {
		response := httptest.NewRecorder()
		router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, path, nil))
		if response.Code != http.StatusNotFound {
			t.Fatalf("cross-project preview %s status = %d: %s", path, response.Code, response.Body.String())
		}
	}
	spoofed := httptest.NewRecorder()
	router.ServeHTTP(spoofed, httptest.NewRequest(http.MethodGet, "/projects/"+first.ID+"/experiment-policy/preview?dataset_id="+firstDataset.ID+"&account_id=attacker", nil))
	if spoofed.Code != http.StatusBadRequest {
		t.Fatalf("body-supplied account id status = %d: %s", spoofed.Code, spoofed.Body.String())
	}

	invalidTask := httptest.NewRecorder()
	router.ServeHTTP(invalidTask, httptest.NewRequest(http.MethodGet, "/projects/"+first.ID+"/experiment-policy/preview?task=not_a_task", nil))
	if invalidTask.Code != http.StatusBadRequest || !strings.Contains(invalidTask.Body.String(), string(policies.ReasonUnknownCatalogIdentifier)) {
		t.Fatalf("invalid task response = %d: %s", invalidTask.Code, invalidTask.Body.String())
	}
}

func TestPolicyPreviewUsesControlRouteAPITokenAuthorization(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_ALLOW_LAN", "true")
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "preview-token")
	t.Setenv("MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL", "")
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("authorized preview", "")
	if err != nil {
		t.Fatal(err)
	}
	router := NewRouter(memoryStore)
	path := "/projects/" + project.ID + "/experiment-policy/preview"
	unauthorized := httptest.NewRecorder()
	router.ServeHTTP(unauthorized, httptest.NewRequest(http.MethodGet, path, nil))
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized preview status = %d: %s", unauthorized.Code, unauthorized.Body.String())
	}
	authorizedRequest := httptest.NewRequest(http.MethodGet, path, nil)
	authorizedRequest.Header.Set("Authorization", "Bearer preview-token")
	authorized := httptest.NewRecorder()
	router.ServeHTTP(authorized, authorizedRequest)
	if authorized.Code != http.StatusOK {
		t.Fatalf("authorized preview status = %d: %s", authorized.Code, authorized.Body.String())
	}
}
