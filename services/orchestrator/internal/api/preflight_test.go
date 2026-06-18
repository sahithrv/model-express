package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/store"
)

func TestCloudPreflightReportsMissingOpenAIKeyBeforeUpload(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("MODEL_EXPRESS_LLM_API_KEY", "")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")

	resp := postCloudPreflight(t)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected failed preflight status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
	payload := decodeCloudPreflightResponse(t, resp)
	if payload.Status != "failed" {
		t.Fatalf("expected failed payload, got %#v", payload)
	}
	check := cloudPreflightCheckByID(payload.Checks, "openai_key")
	if check.Status != "failed" || !strings.Contains(check.Remediation, "MODEL_EXPRESS_LLM_API_KEY") {
		t.Fatalf("expected missing OpenAI key remediation, got %#v", check)
	}
	if strings.Contains(resp.Body.String(), "sk-") {
		t.Fatalf("preflight response leaked a key-like value: %s", resp.Body.String())
	}
}

func TestCloudPreflightReportsOpenAIAPIKeyFallbackSource(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("MODEL_EXPRESS_LLM_API_KEY", "")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "sk-test-fallback")

	resp := postCloudPreflight(t)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected preflight status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	payload := decodeCloudPreflightResponse(t, resp)
	if payload.Status != "ok" {
		t.Fatalf("expected ok payload, got %#v", payload)
	}
	check := cloudPreflightCheckByID(payload.Checks, "openai_key")
	if check.Status != "ok" {
		t.Fatalf("expected OpenAI key check ok, got %#v", check)
	}
	if source, _ := check.Metadata["source"].(string); source != "OPENAI_API_KEY" {
		t.Fatalf("expected OPENAI_API_KEY source, got %#v", check.Metadata)
	}
	if strings.Contains(resp.Body.String(), "sk-test-fallback") {
		t.Fatalf("preflight response leaked OpenAI key: %s", resp.Body.String())
	}
}

func TestCloudPreflightRouteRequiresAPITokenWhenExposed(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "api-token")
	router := NewRouter(store.NewMemoryStore())

	req := httptest.NewRequest(http.MethodPost, "/preflight/cloud", strings.NewReader(`{"stage":"worker_start"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusUnauthorized {
		t.Fatalf("expected exposed preflight to require API token, got %d: %s", resp.Code, resp.Body.String())
	}
}

func TestExecutePlanUsesAutomationDefaultsWhenProviderOmitted(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER", "modal")
	t.Setenv("MODEL_EXPRESS_DEFAULT_GPU_TYPE", "L4")
	t.Setenv("MODEL_EXPRESS_EXECUTION_PROFILE", "fast-remote")
	t.Setenv("MODEL_EXPRESS_V1_PROFILE", "")
	t.Setenv("MODEL_EXPRESS_ALLOW_LAN", "")
	t.Setenv("MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE", "")
	t.Setenv("MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "accuracy")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 1, 5, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 2)}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/plans/"+plan.ID+"/execute", bytes.NewReader([]byte(`{"max_concurrent_jobs":1}`)))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected execute status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}

	var payload executeExperimentPlanResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode execute response: %v", err)
	}
	if len(payload.Jobs) != 1 {
		t.Fatalf("expected one job, got %#v", payload.Jobs)
	}
	if got := jobConfigString(payload.Jobs[0].Config, "provider"); got != "modal" {
		t.Fatalf("expected default provider modal, got %q in %#v", got, payload.Jobs[0].Config)
	}
	if got := jobConfigString(payload.Jobs[0].Config, "gpu_type"); got != "L4" {
		t.Fatalf("expected default gpu L4, got %q in %#v", got, payload.Jobs[0].Config)
	}
	if payload.WorkerRequirement == nil || payload.WorkerRequirement.Provider != "modal" || payload.WorkerRequirement.GPUType != "L4" {
		t.Fatalf("expected modal/L4 worker requirement, got %#v", payload.WorkerRequirement)
	}
}

func setCloudPreflightEnv(t *testing.T) {
	t.Helper()
	values := map[string]string{
		"MODEL_EXPRESS_V1_PROFILE":                  "cloud",
		"MODEL_EXPRESS_EXECUTION_PROFILE":           "fast-remote",
		"MODEL_EXPRESS_ALLOW_LAN":                   "true",
		"MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE":    "true",
		"MODEL_EXPRESS_API_TOKEN":                   "api-token",
		"MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER":   "modal",
		"MODEL_EXPRESS_DEFAULT_GPU_TYPE":            "T4",
		"MODEL_EXPRESS_LLM_ENABLED":                 "true",
		"MODEL_EXPRESS_LLM_PROVIDER":                "openai",
		"MODEL_EXPRESS_LLM_MODEL":                   "gpt-test",
		"MODEL_EXPRESS_LLM_API_STYLE":               "responses",
		"MODEL_EXPRESS_LLM_STORED_RESPONSES":        "true",
		"MODEL_EXPRESS_LLM_API_KEY":                 "model-express-key",
		"MODEL_EXPRESS_VISUAL_LLM_API_KEY":          "",
		"OPENAI_API_KEY":                            "",
		"MODAL_ORCHESTRATOR_URL":                    "https://orchestrator.example.test",
		"MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL":      "",
		"S3_ENDPOINT_URL":                           "https://s3.example.test",
		"MODAL_S3_ENDPOINT_URL":                     "https://s3.example.test",
		"MODEL_EXPRESS_MODAL_S3_ENDPOINT_URL":       "",
		"S3_BUCKET":                                 "model-express",
		"MODEL_EXPRESS_ARTIFACT_BUCKET":             "model-express",
		"AWS_ACCESS_KEY_ID":                         "scoped-upload-key",
		"AWS_SECRET_ACCESS_KEY":                     "scoped-upload-secret",
		"MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID":     "scoped-modal-key",
		"MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY": "scoped-modal-secret",
		"MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE":    "false",
		"MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED":    "false",
		"MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED":   "false",
	}
	for key, value := range values {
		t.Setenv(key, value)
	}
}

func postCloudPreflight(t *testing.T) *httptest.ResponseRecorder {
	t.Helper()
	router := NewRouter(store.NewMemoryStore())
	req := httptest.NewRequest(http.MethodPost, "/preflight/cloud", strings.NewReader(`{"stage":"dataset_upload"}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer api-token")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	return resp
}

func decodeCloudPreflightResponse(t *testing.T, resp *httptest.ResponseRecorder) cloudPreflightResponse {
	t.Helper()
	var payload cloudPreflightResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode preflight response: %v\n%s", err, resp.Body.String())
	}
	return payload
}

func cloudPreflightCheckByID(checks []cloudPreflightCheck, id string) cloudPreflightCheck {
	for _, check := range checks {
		if check.ID == id {
			return check
		}
	}
	return cloudPreflightCheck{}
}
