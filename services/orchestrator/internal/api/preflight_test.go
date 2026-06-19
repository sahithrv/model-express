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

func TestCloudPreflightAutomaticTunnelsAllowMissingPublicURLs(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("S3_ENDPOINT_URL", "http://127.0.0.1:9000")
	t.Setenv("MODAL_S3_ENDPOINT_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_S3_ENDPOINT_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_TUNNEL_S3", "true")

	resp := postCloudPreflightBody(t, `{"stage":"worker_start","automatic_tunnels":true}`)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected automatic tunnel preflight status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	payload := decodeCloudPreflightResponse(t, resp)
	if payload.Status != "ok" {
		t.Fatalf("expected ok payload, got %#v", payload)
	}
	if check := cloudPreflightCheckByID(payload.Checks, "modal_orchestrator_url"); check.Status != "warning" {
		t.Fatalf("expected automatic orchestrator tunnel warning, got %#v", check)
	}
	if check := cloudPreflightCheckByID(payload.Checks, "modal_s3_endpoint"); check.Status != "warning" {
		t.Fatalf("expected automatic S3 tunnel warning, got %#v", check)
	}
}

func TestCloudPreflightAllowsGeneratedLocalMinIOWithAutomaticTunnel(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("S3_ENDPOINT_URL", "http://127.0.0.1:9000")
	t.Setenv("MODAL_S3_ENDPOINT_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_S3_ENDPOINT_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_TUNNEL_S3", "true")
	t.Setenv("MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE", "true")
	t.Setenv("AWS_ACCESS_KEY_ID", "mxgeneratedaccess")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "generated-secret")
	t.Setenv("MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID", "mxgeneratedaccess")
	t.Setenv("MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY", "generated-secret")

	resp := postCloudPreflightBody(t, `{"stage":"worker_start","automatic_tunnels":true}`)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected generated MinIO tunnel preflight status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	payload := decodeCloudPreflightResponse(t, resp)
	if payload.Status != "ok" {
		t.Fatalf("expected ok payload, got %#v", payload)
	}
	if check := cloudPreflightCheckByID(payload.Checks, "s3_endpoint"); check.Status != "warning" {
		t.Fatalf("expected local S3 endpoint tunnel warning, got %#v", check)
	}
	if check := cloudPreflightCheckByID(payload.Checks, "modal_s3_endpoint"); check.Status != "warning" {
		t.Fatalf("expected automatic S3 tunnel warning, got %#v", check)
	}
	if strings.Contains(resp.Body.String(), "generated-secret") {
		t.Fatalf("preflight response leaked generated secret: %s", resp.Body.String())
	}
}

func TestCloudPreflightRejectsUnsafeAutomaticS3TunnelTarget(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("S3_ENDPOINT_URL", "http://127.0.0.1:9001")
	t.Setenv("MODAL_S3_ENDPOINT_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_S3_ENDPOINT_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_TUNNEL_S3", "true")

	resp := postCloudPreflightBody(t, `{"stage":"worker_start","automatic_tunnels":true}`)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected unsafe S3 tunnel target status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
	payload := decodeCloudPreflightResponse(t, resp)
	check := cloudPreflightCheckByID(payload.Checks, "modal_s3_endpoint")
	if check.Status != "failed" || !strings.Contains(check.Remediation, "9001") {
		t.Fatalf("expected console-port S3 tunnel failure, got %#v", check)
	}
}

func TestCloudPreflightRejectsDefaultMinIORootCredentials(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("AWS_ACCESS_KEY_ID", "model_express")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "model_express_password")
	t.Setenv("MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID", "")
	t.Setenv("MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY", "")

	resp := postCloudPreflightBody(t, `{"stage":"worker_start","automatic_tunnels":true}`)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected default MinIO credentials status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
	payload := decodeCloudPreflightResponse(t, resp)
	failed := false
	for _, check := range payload.Checks {
		if check.Status == "failed" && strings.Contains(check.Message, "default local MinIO root credentials") {
			failed = true
		}
	}
	if !failed {
		t.Fatalf("expected default MinIO credential failure, got %#v", payload.Checks)
	}
}

func TestCloudPreflightRejectsUnsafeExplicitPublicURLs(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("MODAL_ORCHESTRATOR_URL", "https://10.0.0.2:8080")

	resp := postCloudPreflightBody(t, `{"stage":"worker_start","automatic_tunnels":true}`)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected unsafe URL preflight status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
	payload := decodeCloudPreflightResponse(t, resp)
	check := cloudPreflightCheckByID(payload.Checks, "modal_orchestrator_url")
	if check.Status != "failed" || !strings.Contains(check.Remediation, "public HTTPS origin") {
		t.Fatalf("expected unsafe orchestrator URL failure, got %#v", check)
	}
}

func TestCloudPreflightAutomaticTunnelsRequireAPIToken(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("MODEL_EXPRESS_ALLOW_LAN", "")
	t.Setenv("MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE", "")
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "")
	t.Setenv("MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL", "")

	response := newServer(store.NewMemoryStore()).runCloudPreflight(cloudPreflightRequest{
		Stage:            "worker_start",
		AutomaticTunnels: true,
	})
	if response.Status != "failed" {
		t.Fatalf("expected automatic tunnels without token to fail, got %#v", response)
	}
	check := cloudPreflightCheckByID(response.Checks, "api_token")
	if check.Status != "failed" || !strings.Contains(check.Message, "MODEL_EXPRESS_API_TOKEN") {
		t.Fatalf("expected api token failure, got %#v", check)
	}
}

func TestCloudPreflightAllowsConfiguredMemoryRetrieval(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER", "openai")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL", "text-embedding-3-small")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "sk-test-memory")

	resp := postCloudPreflight(t)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected preflight status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	payload := decodeCloudPreflightResponse(t, resp)
	check := cloudPreflightCheckByID(payload.Checks, "memory_retrieval")
	if check.Status != "ok" {
		t.Fatalf("expected memory retrieval check ok, got %#v", check)
	}
	if source, _ := check.Metadata["api_key_source"].(string); source != "OPENAI_API_KEY" {
		t.Fatalf("expected memory key fallback source, got %#v", check.Metadata)
	}
	if strings.Contains(resp.Body.String(), "sk-test-memory") {
		t.Fatalf("preflight response leaked memory embedding key: %s", resp.Body.String())
	}
}

func TestCloudPreflightRejectsIncompleteMemoryRetrieval(t *testing.T) {
	setCloudPreflightEnv(t)
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER", "openai")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL", "")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_API_KEY", "")
	t.Setenv("OPENAI_API_KEY", "")

	resp := postCloudPreflight(t)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected failed preflight status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
	payload := decodeCloudPreflightResponse(t, resp)
	check := cloudPreflightCheckByID(payload.Checks, "memory_retrieval")
	if check.Status != "failed" || !strings.Contains(check.Remediation, "MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL") {
		t.Fatalf("expected memory retrieval remediation, got %#v", check)
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
		"MODAL_TOKEN_ID":                            "modal-token-id",
		"MODAL_TOKEN_SECRET":                        "modal-token-secret",
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
	return postCloudPreflightBody(t, `{"stage":"dataset_upload","automatic_tunnels":true}`)
}

func postCloudPreflightBody(t *testing.T, body string) *httptest.ResponseRecorder {
	t.Helper()
	router := NewRouter(store.NewMemoryStore())
	req := httptest.NewRequest(http.MethodPost, "/preflight/cloud", strings.NewReader(body))
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
