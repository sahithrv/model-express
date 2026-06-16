package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func TestCreateJobRejectsUnsupportedTemplateAndUnsafeConfig(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	router := NewRouter(memoryStore)

	unsupportedReq := httptest.NewRequest(
		http.MethodPost,
		"/projects/"+project.ID+"/jobs",
		strings.NewReader(`{"template":"mobilenet_transfer","config":{"dataset_id":"dataset_1"}}`),
	)
	unsupportedReq.Header.Set("Content-Type", "application/json")
	unsupportedResp := httptest.NewRecorder()
	router.ServeHTTP(unsupportedResp, unsupportedReq)
	if unsupportedResp.Code != http.StatusBadRequest {
		t.Fatalf("expected unsupported template status %d, got %d: %s", http.StatusBadRequest, unsupportedResp.Code, unsupportedResp.Body.String())
	}

	unsafeReq := httptest.NewRequest(
		http.MethodPost,
		"/projects/"+project.ID+"/jobs",
		strings.NewReader(`{"template":"profile_dataset","config":{"dataset_id":"dataset_1","artifact_uri":"file:///etc/passwd"}}`),
	)
	unsafeReq.Header.Set("Content-Type", "application/json")
	unsafeResp := httptest.NewRecorder()
	router.ServeHTTP(unsafeResp, unsafeReq)
	if unsafeResp.Code != http.StatusBadRequest {
		t.Fatalf("expected unsafe config status %d, got %d: %s", http.StatusBadRequest, unsafeResp.Code, unsafeResp.Body.String())
	}
}

func TestAgentInvocationResponsesRedactRawTraceByDefault(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	if _, err := memoryStore.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:        project.ID,
		AgentName:        "planner",
		Provider:         "openai",
		Model:            "test-model",
		InputMessages:    []map[string]string{{"role": "user", "content": "secret-token-value"}},
		InputContext:     map[string]any{"secret": "secret-context-value"},
		RawOutput:        "secret-raw-output",
		ParsedOutput:     map[string]any{"secret": "secret-parsed-output"},
		ValidationStatus: memory.InvocationValidationValid,
	}); err != nil {
		t.Fatalf("create invocation: %v", err)
	}
	router := NewRouter(memoryStore)

	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/agent-invocations", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	body := resp.Body.String()
	for _, forbidden := range []string{"secret-token-value", "secret-context-value", "secret-raw-output", "secret-parsed-output", "input_messages", "input_context", "raw_output", "parsed_output"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("agent invocation response leaked %q: %s", forbidden, body)
		}
	}
}

func TestJSONBodyLimitRejectsOversizedRequest(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_JSON_BODY_MAX_BYTES", "65536")
	router := NewRouter(store.NewMemoryStore())
	body := `{"name":"` + strings.Repeat("a", 70_000) + `"}`
	req := httptest.NewRequest(http.MethodPost, "/projects", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("expected status %d, got %d: %s", http.StatusRequestEntityTooLarge, resp.Code, resp.Body.String())
	}
}

func TestPublicOrTunnelModeRequiresAPITokenForControlRoutes(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "")
	t.Setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.example.test")

	unauthenticatedRouter := NewRouter(store.NewMemoryStore())
	unauthenticatedResp := httptest.NewRecorder()
	unauthenticatedRouter.ServeHTTP(unauthenticatedResp, httptest.NewRequest(http.MethodGet, "/projects", nil))
	if unauthenticatedResp.Code != http.StatusUnauthorized {
		t.Fatalf("expected status %d without API token, got %d: %s", http.StatusUnauthorized, unauthenticatedResp.Code, unauthenticatedResp.Body.String())
	}

	t.Setenv("MODEL_EXPRESS_API_TOKEN", "api-token")
	authenticatedRouter := NewRouter(store.NewMemoryStore())
	req := httptest.NewRequest(http.MethodGet, "/projects", nil)
	req.Header.Set("Authorization", "Bearer api-token")
	authenticatedResp := httptest.NewRecorder()
	authenticatedRouter.ServeHTTP(authenticatedResp, req)
	if authenticatedResp.Code != http.StatusOK {
		t.Fatalf("expected status %d with API token, got %d: %s", http.StatusOK, authenticatedResp.Code, authenticatedResp.Body.String())
	}
}

func TestClosedRemoteTrainingSessionRejectsCallbackAndCloseIsIdempotent(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "api-token")
	t.Setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.example.test")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "modal worker", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	if _, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "modal",
	}); err != nil {
		t.Fatalf("create job: %v", err)
	}
	router := NewRouter(memoryStore)
	assigned := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
	sessionID := payloadString(payloadMap(assigned.Config, "remote_training_session"), "id")
	record, err := memoryStore.GetRemoteTrainingSession(sessionID)
	if err != nil {
		t.Fatalf("get remote session: %v", err)
	}
	if record.Status != runs.RemoteTrainingSessionStatusActive || record.CallbackTokenHash == "" {
		t.Fatalf("expected active stored session with callback token hash, got %#v", record)
	}
	if strings.Contains(record.CallbackTokenHash, jobConfigString(assigned.Config, "callback_token")) {
		t.Fatalf("callback token hash leaked raw callback token: %#v", record)
	}

	callbackBody := `{"epoch":1,"training_attempt_id":"` + callbackAttemptID(t, assigned) + `","metrics":{"loss":1.2}}`
	activeReq := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/metrics", strings.NewReader(callbackBody))
	activeReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, activeReq, assigned)
	activeResp := httptest.NewRecorder()
	router.ServeHTTP(activeResp, activeReq)
	if activeResp.Code != http.StatusCreated {
		t.Fatalf("expected active session callback without API token to use callback auth, got %d: %s", activeResp.Code, activeResp.Body.String())
	}

	closed, err := memoryStore.CloseRemoteTrainingSession(sessionID, runs.RemoteTrainingSessionStatusClosed)
	if err != nil {
		t.Fatalf("close remote session: %v", err)
	}
	closedAgain, err := memoryStore.CloseRemoteTrainingSession(sessionID, runs.RemoteTrainingSessionStatusFailed)
	if err != nil {
		t.Fatalf("close remote session again: %v", err)
	}
	if closedAgain.Status != closed.Status || closedAgain.ClosedAt == nil {
		t.Fatalf("expected idempotent terminal close, got first=%#v second=%#v", closed, closedAgain)
	}

	closedReq := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/metrics", strings.NewReader(callbackBody))
	closedReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, closedReq, assigned)
	closedResp := httptest.NewRecorder()
	router.ServeHTTP(closedResp, closedReq)
	if closedResp.Code != http.StatusUnauthorized {
		t.Fatalf("expected closed session callback status %d, got %d: %s", http.StatusUnauthorized, closedResp.Code, closedResp.Body.String())
	}
}

func TestChampionExportReadyResultRejectsBareFileURI(t *testing.T) {
	job := jobs.ExperimentJob{ID: "job_1"}
	export := runs.ChampionExport{ID: "export_1", Format: "onnx"}
	req := championExportResultRequest{
		Status:      runs.ChampionExportStatusReady,
		ArtifactURI: "file:///tmp/champion.onnx",
		Metadata:    map[string]any{"labels": []any{"cat"}},
	}

	err := validateChampionExportReadyResult(job, export, req)
	if err == nil {
		t.Fatal("expected bare file URI export result to be rejected")
	}
}

func TestChampionExportReadyResultAcceptsWorkerManifest(t *testing.T) {
	job := jobs.ExperimentJob{ID: "job_1"}
	export := runs.ChampionExport{ID: "export_1", Format: "onnx"}
	manifestJSON := `{
		"schema_version":"champion_export_manifest_v1",
		"status":"ok",
		"metadata":{
			"provenance":{
				"schema_version":"worker_artifact_provenance_v1",
				"generated_by":"model-express-worker",
				"source":"worker_generated",
				"export_job_id":"job_1",
				"source_export_id":"export_1"
			}
		},
		"artifacts":[{
			"format":"onnx",
			"status":"created",
			"path":"/tmp/champion.onnx",
			"provenance":{
				"schema_version":"worker_artifact_provenance_v1",
				"generated_by":"model-express-worker",
				"source":"worker_generated",
				"export_job_id":"job_1",
				"source_export_id":"export_1"
			}
		}]
	}`
	var manifest map[string]any
	if err := json.Unmarshal([]byte(manifestJSON), &manifest); err != nil {
		t.Fatalf("decode manifest: %v", err)
	}
	req := championExportResultRequest{
		Status:      runs.ChampionExportStatusReady,
		ArtifactURI: "file:///tmp/champion.onnx",
		Metadata:    map[string]any{"manifest": manifest},
	}

	if err := validateChampionExportReadyResult(job, export, req); err != nil {
		t.Fatalf("expected worker manifest to be accepted: %v", err)
	}
}

func TestChampionExportReadyResultAcceptsONNXWithAdditionalPortableInferenceBundle(t *testing.T) {
	job := jobs.ExperimentJob{ID: "job_1"}
	export := runs.ChampionExport{ID: "export_1", Format: "onnx"}
	manifest := validWorkerExportManifest("onnx", "job_1", "export_1")
	provenance := payloadMap(payloadMap(manifest, "metadata"), "provenance")
	manifest["artifacts"] = append(manifest["artifacts"].([]any), map[string]any{
		"format": "portable_inference_bundle",
		"status": "created",
		"path":   "/tmp/portable_inference_bundle.zip",
		"provenance": map[string]any{
			"schema_version":   "worker_artifact_provenance_v1",
			"generated_by":     "model-express-worker",
			"source":           "worker_generated",
			"export_job_id":    "job_1",
			"source_export_id": "export_1",
			"artifact_format":  "portable_inference_bundle",
		},
	})
	if len(provenance) == 0 {
		t.Fatal("expected test manifest provenance")
	}
	req := championExportResultRequest{
		Status:      runs.ChampionExportStatusReady,
		ArtifactURI: "file:///tmp/champion.onnx",
		Metadata:    map[string]any{"manifest": manifest},
	}

	if err := validateChampionExportReadyResult(job, export, req); err != nil {
		t.Fatalf("expected ONNX manifest with optional portable bundle to be accepted: %v", err)
	}
}

func TestChampionExportReadyResultRejectsZipOnlyONNXManifest(t *testing.T) {
	job := jobs.ExperimentJob{ID: "job_1"}
	export := runs.ChampionExport{ID: "export_1", Format: "onnx"}
	cases := []struct {
		name     string
		artifact map[string]any
	}{
		{
			name: "explicit portable bundle",
			artifact: map[string]any{
				"format": "portable_inference_bundle",
				"status": "created",
				"path":   "/tmp/portable_inference_bundle.zip",
			},
		},
		{
			name: "omitted format zip",
			artifact: map[string]any{
				"status": "created",
				"path":   "/tmp/portable_inference_bundle.zip",
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			manifest := validWorkerExportManifest("onnx", "job_1", "export_1")
			provenance := payloadMap(payloadMap(manifest, "metadata"), "provenance")
			tc.artifact["provenance"] = provenance
			manifest["artifacts"] = []any{tc.artifact}
			req := championExportResultRequest{
				Status:      runs.ChampionExportStatusReady,
				ArtifactURI: "file:///tmp/champion.onnx",
				Metadata:    map[string]any{"manifest": manifest},
			}

			err := validateChampionExportReadyResult(job, export, req)
			if err == nil || !strings.Contains(err.Error(), "created artifact for the requested format") {
				t.Fatalf("expected zip-only manifest to be rejected, got %v", err)
			}
		})
	}
}

func TestChampionExportReadyResultRejectsFailedSelfTest(t *testing.T) {
	job := jobs.ExperimentJob{ID: "job_1"}
	export := runs.ChampionExport{ID: "export_1", Format: "onnx"}
	manifest := validWorkerExportManifest("onnx", "job_1", "export_1")
	metadata := manifest["metadata"].(map[string]any)
	metadata["export_self_test"] = map[string]any{
		"schema_version":    "champion_export_self_test_v1",
		"status":            "failed",
		"diagnostic_reason": "ONNX_OUTPUT_MISMATCH",
	}
	req := championExportResultRequest{
		Status:      runs.ChampionExportStatusReady,
		ArtifactURI: "file:///tmp/champion.onnx",
		Metadata:    map[string]any{"manifest": manifest},
	}

	err := validateChampionExportReadyResult(job, export, req)
	if err == nil || !strings.Contains(err.Error(), "ONNX self-test failed") {
		t.Fatalf("expected failed self-test to reject READY export, got %v", err)
	}
}

func TestChampionExportReadyResultAcceptsFrameworkNativeCheckpointForPytorch(t *testing.T) {
	job := jobs.ExperimentJob{ID: "job_1"}
	export := runs.ChampionExport{ID: "export_1", Format: "pytorch"}
	manifest := validWorkerExportManifest("framework_native_checkpoint", "job_1", "export_1")
	req := championExportResultRequest{
		Status:      runs.ChampionExportStatusReady,
		ArtifactURI: "file:///tmp/champion.pt",
		Metadata:    map[string]any{"manifest": manifest},
	}

	if err := validateChampionExportReadyResult(job, export, req); err != nil {
		t.Fatalf("expected framework-native checkpoint manifest to satisfy pytorch export: %v", err)
	}
}

func TestChampionExportResultMetadataCarriesManifestURI(t *testing.T) {
	metadata := championExportResultMetadata(championExportResultRequest{
		ManifestURI: "file:///tmp/manifest.json",
		Metadata:    map[string]any{"existing": "value", "portable_bundle_uri": "file:///tmp/portable_inference_bundle.zip"},
	})

	if metadata["existing"] != "value" ||
		metadata["portable_bundle_uri"] != "file:///tmp/portable_inference_bundle.zip" ||
		metadata["manifest_uri"] != "file:///tmp/manifest.json" ||
		metadata["export_manifest_uri"] != "file:///tmp/manifest.json" {
		t.Fatalf("expected manifest URI to be preserved in metadata, got %#v", metadata)
	}
}

func validWorkerExportManifest(format string, jobID string, exportID string) map[string]any {
	provenance := map[string]any{
		"schema_version":   "worker_artifact_provenance_v1",
		"generated_by":     "model-express-worker",
		"source":           "worker_generated",
		"export_job_id":    jobID,
		"source_export_id": exportID,
	}
	return map[string]any{
		"schema_version": "champion_export_manifest_v1",
		"status":         "created",
		"metadata": map[string]any{
			"provenance": provenance,
		},
		"artifacts": []any{
			map[string]any{
				"format":     format,
				"status":     "created",
				"path":       "/tmp/champion",
				"provenance": provenance,
			},
		},
	}
}
