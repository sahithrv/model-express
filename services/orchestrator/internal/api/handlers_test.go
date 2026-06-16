package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
	"model-express/services/orchestrator/internal/strategies"
	"model-express/services/orchestrator/internal/workers"
)

func TestAutomationSettingsPersistAndReload(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)

	maxRounds := 3
	provider := "modal"
	gpuType := "T4"
	costMode := "cheap"
	budgetCap := 2.5
	enabled := true
	updated, err := applyAutomationSettingsUpdate(server.currentAutomationSettings(), settings.AutomationSettingsUpdate{
		AutoReviewExperiments:   &enabled,
		AutoScheduleFollowUps:   &enabled,
		AutoExecutePlans:        &enabled,
		MaxFollowUpRounds:       &maxRounds,
		DefaultTrainingProvider: &provider,
		DefaultGPUType:          &gpuType,
		CostMode:                &costMode,
		BudgetCapUSD:            &budgetCap,
	})
	if err != nil {
		t.Fatalf("apply automation settings update: %v", err)
	}

	saved, err := memoryStore.SaveAutomationSettings(updated)
	if err != nil {
		t.Fatalf("save automation settings: %v", err)
	}
	server.setAutomationSettings(saved)

	reloaded := newServer(memoryStore)
	current := reloaded.currentAutomationSettings()
	if !current.AutoReviewExperiments || !current.AutoScheduleFollowUps || !current.AutoExecutePlans {
		t.Fatalf("expected persisted automation toggles to be enabled, got %#v", current)
	}
	if current.MaxFollowUpRounds != maxRounds {
		t.Fatalf("expected max follow-up rounds %d, got %d", maxRounds, current.MaxFollowUpRounds)
	}
	if current.DefaultTrainingProvider != provider || current.DefaultGPUType != gpuType {
		t.Fatalf("expected provider %s/%s, got %s/%s", provider, gpuType, current.DefaultTrainingProvider, current.DefaultGPUType)
	}
	if current.CostMode != "prototype" || current.BudgetCapUSD != budgetCap {
		t.Fatalf("expected cost policy prototype/%.2f, got %#v", budgetCap, current)
	}
}

func TestResolveOrchestratorListenAddrDefaultsLoopback(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_ORCHESTRATOR_ADDR", "")
	t.Setenv("MODEL_EXPRESS_ALLOW_LAN", "")
	t.Setenv("MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE", "")
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "")
	t.Setenv("MODAL_ORCHESTRATOR_URL", "")
	t.Setenv("MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL", "")

	addr, err := ResolveOrchestratorListenAddr()
	if err != nil {
		t.Fatalf("resolve listen addr: %v", err)
	}
	if addr != defaultOrchestratorAddr {
		t.Fatalf("expected default addr %q, got %q", defaultOrchestratorAddr, addr)
	}
}

func TestResolveOrchestratorListenAddrGuardsLANBind(t *testing.T) {
	if err := validateOrchestratorListenAddr("0.0.0.0:8080", false, ""); err == nil {
		t.Fatal("expected non-loopback bind to require explicit LAN opt-in")
	}
	if err := validateOrchestratorListenAddr("0.0.0.0:8080", true, ""); err == nil {
		t.Fatal("expected non-loopback bind to require an API token")
	}
	if err := validateOrchestratorListenAddr("0.0.0.0:8080", true, "token"); err != nil {
		t.Fatalf("expected LAN bind with token to pass: %v", err)
	}
	if err := validateOrchestratorListenAddr("127.0.0.1:8080", false, ""); err != nil {
		t.Fatalf("expected loopback bind to pass without token: %v", err)
	}
}

func TestResolveOrchestratorListenAddrRequiresTokenForPublicTunnelConfig(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_ORCHESTRATOR_ADDR", "")
	t.Setenv("MODEL_EXPRESS_ALLOW_LAN", "")
	t.Setenv("MODEL_EXPRESS_ORCHESTRATOR_TUNNEL_MODE", "")
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "")
	t.Setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.example.test")
	t.Setenv("MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL", "")

	if _, err := ResolveOrchestratorListenAddr(); err == nil || !strings.Contains(err.Error(), "MODEL_EXPRESS_API_TOKEN") {
		t.Fatalf("expected public tunnel config to require MODEL_EXPRESS_API_TOKEN, got %v", err)
	}

	t.Setenv("MODEL_EXPRESS_API_TOKEN", "api-token")
	if addr, err := ResolveOrchestratorListenAddr(); err != nil || addr != defaultOrchestratorAddr {
		t.Fatalf("expected token-authenticated public tunnel config to pass, addr=%q err=%v", addr, err)
	}
}

func TestLANModeAPITokenMiddlewareKeepsHealthOpen(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_ALLOW_LAN", "true")
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "api-token")

	router := NewRouter(store.NewMemoryStore())
	healthResp := httptest.NewRecorder()
	router.ServeHTTP(healthResp, httptest.NewRequest(http.MethodGet, "/healthz", nil))
	if healthResp.Code != http.StatusOK {
		t.Fatalf("expected healthz to stay open, got %d: %s", healthResp.Code, healthResp.Body.String())
	}

	blockedResp := httptest.NewRecorder()
	router.ServeHTTP(blockedResp, httptest.NewRequest(http.MethodGet, "/projects", nil))
	if blockedResp.Code != http.StatusUnauthorized {
		t.Fatalf("expected missing API token to be rejected, got %d: %s", blockedResp.Code, blockedResp.Body.String())
	}

	req := httptest.NewRequest(http.MethodGet, "/projects", nil)
	req.Header.Set("X-Model-Express-Api-Token", "api-token")
	allowedResp := httptest.NewRecorder()
	router.ServeHTTP(allowedResp, req)
	if allowedResp.Code != http.StatusOK {
		t.Fatalf("expected valid API token to pass, got %d: %s", allowedResp.Code, allowedResp.Body.String())
	}
}

func TestWorkerCallbackAuthRequiresAttemptAndToken(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	if _, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID}); err != nil {
		t.Fatalf("create job: %v", err)
	}
	router := NewRouter(memoryStore)
	assigned := pollJobForCallback(t, router, worker.ID, `{}`)
	attemptID := callbackAttemptID(t, assigned)

	missingAttemptReq := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/metrics", strings.NewReader(`{"epoch":1,"metrics":{"loss":1.2}}`))
	missingAttemptReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, missingAttemptReq, assigned)
	missingAttemptResp := httptest.NewRecorder()
	router.ServeHTTP(missingAttemptResp, missingAttemptReq)
	if missingAttemptResp.Code != http.StatusBadRequest {
		t.Fatalf("expected missing attempt status %d, got %d: %s", http.StatusBadRequest, missingAttemptResp.Code, missingAttemptResp.Body.String())
	}

	missingTokenReq := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/metrics", strings.NewReader(`{"epoch":1,"training_attempt_id":"`+attemptID+`","metrics":{"loss":1.2}}`))
	missingTokenReq.Header.Set("Content-Type", "application/json")
	missingTokenResp := httptest.NewRecorder()
	router.ServeHTTP(missingTokenResp, missingTokenReq)
	if missingTokenResp.Code != http.StatusUnauthorized {
		t.Fatalf("expected missing token status %d, got %d: %s", http.StatusUnauthorized, missingTokenResp.Code, missingTokenResp.Body.String())
	}

	wrongTokenReq := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/metrics", strings.NewReader(`{"epoch":1,"training_attempt_id":"`+attemptID+`","metrics":{"loss":1.2}}`))
	wrongTokenReq.Header.Set("Content-Type", "application/json")
	wrongTokenReq.Header.Set(callbackTokenHeader, "wrong-token")
	wrongTokenResp := httptest.NewRecorder()
	router.ServeHTTP(wrongTokenResp, wrongTokenReq)
	if wrongTokenResp.Code != http.StatusUnauthorized {
		t.Fatalf("expected wrong token status %d, got %d: %s", http.StatusUnauthorized, wrongTokenResp.Code, wrongTokenResp.Body.String())
	}
}

func TestWorkerCallbackTokenReturnedByPollAllowsMetric(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	if _, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID}); err != nil {
		t.Fatalf("create job: %v", err)
	}
	router := NewRouter(memoryStore)
	assigned := pollJobForCallback(t, router, worker.ID, `{}`)
	body := `{"epoch":1,"training_attempt_id":"` + callbackAttemptID(t, assigned) + `","metrics":{"loss":1.2}}`
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/metrics", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusCreated {
		t.Fatalf("expected valid callback status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
}

func TestLANModeCallbackEndpointUsesAttemptTokenInsteadOfAPIToken(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_ALLOW_LAN", "true")
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "api-token")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	if _, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID}); err != nil {
		t.Fatalf("create job: %v", err)
	}
	router := NewRouter(memoryStore)

	blockedPollReq := httptest.NewRequest(http.MethodPost, "/workers/"+worker.ID+"/poll", strings.NewReader(`{}`))
	blockedPollReq.Header.Set("Content-Type", "application/json")
	blockedPollResp := httptest.NewRecorder()
	router.ServeHTTP(blockedPollResp, blockedPollReq)
	if blockedPollResp.Code != http.StatusUnauthorized {
		t.Fatalf("expected poll without API token to be rejected, got %d: %s", blockedPollResp.Code, blockedPollResp.Body.String())
	}

	pollReq := httptest.NewRequest(http.MethodPost, "/workers/"+worker.ID+"/poll", strings.NewReader(`{}`))
	pollReq.Header.Set("Content-Type", "application/json")
	pollReq.Header.Set("X-Model-Express-Api-Token", "api-token")
	pollResp := httptest.NewRecorder()
	router.ServeHTTP(pollResp, pollReq)
	if pollResp.Code != http.StatusOK {
		t.Fatalf("poll job status %d: %s", pollResp.Code, pollResp.Body.String())
	}
	var payload pollJobResponse
	if err := json.Unmarshal(pollResp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode poll response: %v", err)
	}
	if payload.Job == nil {
		t.Fatal("expected poll to return a job")
	}

	assigned := *payload.Job
	body := `{"epoch":1,"training_attempt_id":"` + callbackAttemptID(t, assigned) + `","metrics":{"loss":1.2}}`
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/metrics", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusCreated {
		t.Fatalf("expected callback token to authorize callback without API token, got %d: %s", resp.Code, resp.Body.String())
	}
}

func TestPollModalJobAddsCallbackTokenAndRemoteSessionWithoutPersistingToken(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "test-api-token")
	t.Setenv("MODAL_ORCHESTRATOR_URL", "https://orchestrator.example.test")
	t.Setenv("MODAL_S3_ENDPOINT_URL", "https://storage.example.test")

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
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "modal",
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	router := NewRouter(memoryStore)

	assigned := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)

	if assigned.ID != job.ID {
		t.Fatalf("expected modal job, got %#v", assigned)
	}
	if got := jobConfigString(assigned.Config, "callback_token_header"); got != callbackTokenHeader {
		t.Fatalf("expected callback token header %q, got %q", callbackTokenHeader, got)
	}
	session := payloadMap(assigned.Config, "remote_training_session")
	if payloadString(session, "id") == "" ||
		payloadString(session, "training_attempt_id") != callbackAttemptID(t, assigned) ||
		payloadString(session, "status") != "active" ||
		payloadString(session, "expires_at") == "" ||
		payloadString(session, "storage_prefix") == "" ||
		payloadString(session, "public_callback_url") != "https://orchestrator.example.test" ||
		payloadString(session, "public_storage_url") != "https://storage.example.test" {
		t.Fatalf("unexpected remote training session metadata: %#v", session)
	}
	storageScope := payloadMap(session, "storage_scope")
	if len(storageScope) == 0 {
		t.Fatalf("expected storage scope metadata, got %#v", session)
	}
	if redacted := payloadMap(session, "redacted_state"); redacted["public_callback_url_configured"] != true || redacted["public_storage_url_configured"] != true {
		t.Fatalf("expected redacted remote session state, got %#v", redacted)
	}
	persisted, err := memoryStore.GetJob(job.ID)
	if err != nil {
		t.Fatalf("get persisted job: %v", err)
	}
	if _, ok := persisted.Config["callback_token"]; ok {
		t.Fatalf("callback token was persisted in job config: %#v", persisted.Config)
	}
	if payloadString(payloadMap(persisted.Config, "remote_training_session"), "id") == "" {
		t.Fatalf("expected non-secret remote session metadata to persist: %#v", persisted.Config)
	}
}

func TestExpiredRemoteTrainingSessionRejectsCallback(t *testing.T) {
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
	session := payloadMap(assigned.Config, "remote_training_session")
	session["expires_at"] = time.Now().Add(-time.Minute).UTC().Format(time.RFC3339Nano)
	if _, err := memoryStore.UpdateJobConfig(assigned.ID, map[string]any{"remote_training_session": session}); err != nil {
		t.Fatalf("expire remote session: %v", err)
	}

	body := `{"epoch":1,"training_attempt_id":"` + callbackAttemptID(t, assigned) + `","metrics":{"loss":1.2}}`
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/metrics", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusUnauthorized {
		t.Fatalf("expected expired session status %d, got %d: %s", http.StatusUnauthorized, resp.Code, resp.Body.String())
	}
}

func TestTrainingRunSummaryPersistsRemoteGpuStageTelemetry(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/dataset.zip", "a", 12)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "modal",
		"gpu_type":   "T4",
		"model":      "mobilenet_v3_small",
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}

	router := NewRouter(memoryStore)
	assigned := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
	body := strings.NewReader(`{
		"training_attempt_id":"` + callbackAttemptID(t, assigned) + `",
		"model":"mobilenet_v3_small",
		"provider":"modal",
		"gpu_type":"T4",
		"status":"SUCCEEDED",
		"runtime_seconds":42.5,
		"estimated_cost_usd":0.21,
		"dataset_materialization":{
			"dataset_materialization_cache_key":"sha256-abc",
			"dataset_materialization_cache_miss":true,
			"dataset_materialization_bytes_downloaded":1234,
			"dataset_materialization_extract_seconds":1.25
		},
		"stage_telemetry":{
			"schema_version":"remote_gpu_stage_telemetry_v1",
			"dataset_materialization_seconds":2.5,
			"active_training_seconds":35.0,
			"evaluation_seconds":3.0,
			"export_seconds":1.0,
			"idle_wait_seconds":1.0
		}
	}`)
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/training-run-summary", body)
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}

	getReq := httptest.NewRequest(http.MethodGet, "/jobs/"+job.ID+"/training-run-summary", nil)
	getResp := httptest.NewRecorder()
	router.ServeHTTP(getResp, getReq)
	if getResp.Code != http.StatusOK {
		t.Fatalf("expected get status %d, got %d: %s", http.StatusOK, getResp.Code, getResp.Body.String())
	}
	var summary runs.TrainingRunSummary
	if err := json.Unmarshal(getResp.Body.Bytes(), &summary); err != nil {
		t.Fatalf("decode summary: %v", err)
	}
	if summary.DatasetMaterialization["dataset_materialization_cache_key"] != "sha256-abc" {
		t.Fatalf("expected materialization telemetry to round trip, got %#v", summary.DatasetMaterialization)
	}
	if summary.StageTelemetry["schema_version"] != "remote_gpu_stage_telemetry_v1" {
		t.Fatalf("expected stage telemetry to round trip, got %#v", summary.StageTelemetry)
	}
	if summary.RuntimeSeconds != 42.5 || summary.EstimatedCostUSD != 0.21 {
		t.Fatalf("legacy summary fields changed unexpectedly: %#v", summary)
	}
}

func TestValidationMetricScorePrefersDetectionMetricsForMapTarget(t *testing.T) {
	summary := runs.TrainingRunSummary{
		BestMacroF1:  0.94,
		BestAccuracy: 0.96,
	}
	evaluation := runs.TrainingRunEvaluation{
		ObjectiveProfile: map[string]any{
			"heldout_test_map50_95":  0.31,
			"heldout_test_map50":     0.52,
			"heldout_test_precision": 0.44,
			"heldout_test_recall":    0.39,
		},
	}

	score := validationMetricScore("mAP50_95", summary, evaluation)

	if score >= 0.60 {
		t.Fatalf("expected score to use explicit detection mAP instead of macro-F1 fallback, got %.3f", score)
	}
	if score <= 0.30 {
		t.Fatalf("expected detection metric blend to include mAP50/precision/recall, got %.3f", score)
	}
}

func TestPerClassMetricScoreUsesDetectionAPMetrics(t *testing.T) {
	score, ok := perClassMetricScore(map[string]any{
		"cat":       map[string]any{"AP50_95": 0.72, "AP50": 0.91, "precision": 0.82, "recall": 0.74},
		"dog":       map[string]any{"AP50_95": 0.21, "AP50": 0.45, "precision": 0.40, "recall": 0.30},
		"macro avg": map[string]any{"AP50_95": 0.465, "AP50": 0.68, "precision": 0.61, "recall": 0.52},
	})

	if !ok {
		t.Fatal("expected detection per-class metrics to produce a score")
	}
	if score <= 0.20 || score >= 0.60 {
		t.Fatalf("expected AP-based per-class score in detection range, got %.3f", score)
	}
}

func TestReportProjectDispatcherEventPersistsAllowlistedExecutionEvent(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	router := NewRouter(memoryStore)

	body := strings.NewReader(`{
		"event_type":"DISPATCHER_IDLE_EXIT",
		"message":"Dispatcher saw s3://private-bucket/raw.zip but should not expose it.",
		"payload":{
			"slot_count":0,
			"desired_slot_count":0,
			"registered_slot_count":2,
			"active_slot_count":0,
			"idle_seconds":30.5,
			"idle_exit_seconds":30,
			"storage_uri":"s3://private-bucket/raw.zip"
		}
	}`)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/dispatcher-events", body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}

	events, err := memoryStore.ListProjectExecutionEvents(project.ID, 10)
	if err != nil {
		t.Fatalf("list events: %v", err)
	}
	if len(events) != 1 || events[0].EventType != execution.EventDispatcherIdleExit {
		t.Fatalf("expected dispatcher idle event, got %#v", events)
	}
	if events[0].Payload["slot_count"] != float64(0) {
		t.Fatalf("expected slot count payload, got %#v", events[0].Payload)
	}
	if _, ok := events[0].Payload["storage_uri"]; ok {
		t.Fatalf("dispatcher event payload should be allowlisted, got %#v", events[0].Payload)
	}
}

func TestTrainingRunSummaryUpdatesWorkerRequirementMaterializationStatus(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/dataset.zip", strings.Repeat("d", 64), 12)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	planID := "plan_1"
	cacheKey := "sha256-" + strings.Repeat("d", 64)
	if _, _, err := memoryStore.UpsertWorkerRequirement(
		project.ID,
		planID,
		"modal",
		"modal",
		1,
		"test",
		execution.WorkerRequirementPolicy{
			DatasetID:                      dataset.ID,
			DatasetChecksum:                strings.Repeat("d", 64),
			DatasetCacheKey:                cacheKey,
			DatasetMaterializationStatus:   execution.DatasetMaterializationCold,
			ColdCachePolicy:                execution.ColdCachePolicySingleMaterialization,
			MaxConcurrentJobs:              1,
			MaxColdDatasetMaterializations: 1,
		},
	); err != nil {
		t.Fatalf("upsert worker requirement: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"plan_id":    planID,
		"dataset_id": dataset.ID,
		"provider":   "modal",
		"gpu_type":   "T4",
		"model":      "mobilenet_v3_small",
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}

	router := NewRouter(memoryStore)
	assigned := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
	body := strings.NewReader(`{
		"training_attempt_id":"` + callbackAttemptID(t, assigned) + `",
		"model":"mobilenet_v3_small",
		"provider":"modal",
		"gpu_type":"T4",
		"status":"SUCCEEDED",
		"runtime_seconds":42.5,
		"estimated_cost_usd":0.21,
		"dataset_materialization":{
			"dataset_materialization_cache_key":"` + cacheKey + `",
			"dataset_materialization_status":"materialized",
			"dataset_materialization_cache_miss":true,
			"dataset_prewarm_reusable_for_training":false,
			"dataset_prewarm_reuse_status":"staging_only_root_mismatch"
		}
	}`)
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/training-run-summary", body)
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}

	requirements, err := memoryStore.ListProjectWorkerRequirements(project.ID)
	if err != nil {
		t.Fatalf("list worker requirements: %v", err)
	}
	if len(requirements) != 1 {
		t.Fatalf("expected one worker requirement, got %d", len(requirements))
	}
	if requirements[0].DatasetMaterializationStatus != execution.DatasetMaterializationStagingOnly {
		t.Fatalf("expected staging-only requirement status, got %#v", requirements[0])
	}
}

func TestUpdateWorkerRequirementMaterializationStatus(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	requirement, _, err := memoryStore.UpsertWorkerRequirement(
		project.ID,
		"plan_1",
		"modal",
		"modal",
		1,
		"test",
		execution.WorkerRequirementPolicy{DatasetMaterializationStatus: execution.DatasetMaterializationCold},
	)
	if err != nil {
		t.Fatalf("upsert worker requirement: %v", err)
	}

	router := NewRouter(memoryStore)
	body := strings.NewReader(`{"dataset_materialization_status":"MATERIALIZING"}`)
	req := httptest.NewRequest(http.MethodPatch, "/worker-requirements/"+requirement.ID, body)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	var updated execution.WorkerRequirement
	if err := json.Unmarshal(resp.Body.Bytes(), &updated); err != nil {
		t.Fatalf("decode requirement: %v", err)
	}
	if updated.DatasetMaterializationStatus != execution.DatasetMaterializationMaterializing {
		t.Fatalf("expected materializing status, got %#v", updated)
	}
	if updated.Status != execution.WorkerRequirementPending {
		t.Fatalf("worker lifecycle status changed unexpectedly: %#v", updated)
	}

	invalidReq := httptest.NewRequest(
		http.MethodPatch,
		"/worker-requirements/"+requirement.ID,
		strings.NewReader(`{"dataset_materialization_status":"banana"}`),
	)
	invalidReq.Header.Set("Content-Type", "application/json")
	invalidResp := httptest.NewRecorder()
	router.ServeHTTP(invalidResp, invalidReq)
	if invalidResp.Code != http.StatusBadRequest {
		t.Fatalf("expected invalid status %d, got %d: %s", http.StatusBadRequest, invalidResp.Code, invalidResp.Body.String())
	}
}

func TestExperimentPlannerLLMConfigUsesHigherDefaultToolRounds(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_PLANNER_MAX_TOOL_ROUNDS", "")

	config := experimentPlannerLLMConfig(llm.Config{MaxToolRounds: llm.DefaultMaxToolRounds})
	if config.MaxToolRounds != plannerDefaultMaxToolRounds {
		t.Fatalf("expected planner tool rounds %d, got %d", plannerDefaultMaxToolRounds, config.MaxToolRounds)
	}

	config = experimentPlannerLLMConfig(llm.Config{MaxToolRounds: plannerDefaultMaxToolRounds + 2})
	if config.MaxToolRounds != plannerDefaultMaxToolRounds+2 {
		t.Fatalf("expected explicit higher generic tool rounds to win, got %d", config.MaxToolRounds)
	}
}

func TestExperimentPlannerLLMConfigAllowsPlannerToolRoundOverride(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_PLANNER_MAX_TOOL_ROUNDS", "6")

	config := experimentPlannerLLMConfig(llm.Config{MaxToolRounds: llm.DefaultMaxToolRounds})
	if config.MaxToolRounds != 6 {
		t.Fatalf("expected planner override to set tool rounds to 6, got %d", config.MaxToolRounds)
	}
}

func TestMemoryEmbeddingBackfillEndpointDisabledByDefault(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "false")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("memory demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	if _, err := memoryStore.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		ProjectID: project.ID,
		AgentName: agents.ExperimentPlannerAgentName,
		Kind:      memory.KindPlanningOutcome,
		Summary:   "weighted loss improved the champion",
		Payload: map[string]any{
			"outcome":      "improved_champion",
			"mechanism":    "class_imbalance",
			"intervention": "weighted_loss",
		},
	}); err != nil {
		t.Fatalf("create memory record: %v", err)
	}

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/memory-embeddings/backfill", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	var result memory.MemoryIndexResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("decode backfill response: %v", err)
	}
	if !result.Disabled || result.NoopReason == "" {
		t.Fatalf("expected disabled no-op backfill response, got %#v", result)
	}
}

func TestReportMetricRejectsNonPositiveEpoch(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}

	router := NewRouter(memoryStore)
	body := []byte(`{"epoch":0,"metrics":{"loss":1.2}}`)
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/metrics", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
	metrics, err := memoryStore.ListJobMetrics(job.ID)
	if err != nil {
		t.Fatalf("list metrics: %v", err)
	}
	if len(metrics) != 0 {
		t.Fatalf("expected no metric rows, got %d", len(metrics))
	}
}

func TestDatasetMetadataImportSummaryAndBundleEndpoints(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("metadata demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/dataset.zip", "dataset-sha", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	router := NewRouter(memoryStore)

	emptySummaryReq := httptest.NewRequest(http.MethodGet, "/datasets/"+dataset.ID+"/metadata/summary", nil)
	emptySummaryResp := httptest.NewRecorder()
	router.ServeHTTP(emptySummaryResp, emptySummaryReq)
	if emptySummaryResp.Code != http.StatusOK {
		t.Fatalf("expected empty summary status %d, got %d: %s", http.StatusOK, emptySummaryResp.Code, emptySummaryResp.Body.String())
	}
	var emptySummaryPayload struct {
		Summary          *datasets.AgentSafeDatasetMetadataSummary `json:"summary"`
		MetadataImportID string                                    `json:"metadata_import_id"`
	}
	if err := json.Unmarshal(emptySummaryResp.Body.Bytes(), &emptySummaryPayload); err != nil {
		t.Fatalf("decode empty summary response: %v", err)
	}
	if emptySummaryPayload.Summary != nil || emptySummaryPayload.MetadataImportID != "" {
		t.Fatalf("expected no active metadata import yet, got %#v", emptySummaryPayload)
	}

	csvContent := "path,label,split\ntrain/cat/one.jpg,cat,train\nval/dog/two.jpg,dog,valid\n"
	importPayload := map[string]any{
		"strict_mode": false,
		"source_kind": "worker_discovery",
		"sources": []map[string]any{{
			"relative_path":   "metadata/labels.csv",
			"declared_format": "csv_manifest",
			"content_base64":  base64.StdEncoding.EncodeToString([]byte(csvContent)),
			"size_bytes":      len(csvContent),
		}},
		"inventory": map[string]any{"files": []map[string]any{
			{"relative_path": "train/cat/one.jpg", "size_bytes": 10},
			{"relative_path": "val/dog/two.jpg", "size_bytes": 11},
		}},
	}
	body, _ := json.Marshal(importPayload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/metadata/imports", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected import status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	var importResponse struct {
		MetadataImport datasets.DatasetMetadataImport `json:"metadata_import"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &importResponse); err != nil {
		t.Fatalf("decode import response: %v", err)
	}
	if !importResponse.MetadataImport.Active || importResponse.MetadataImport.AgentSafeSummary.SampleCount != 2 {
		t.Fatalf("unexpected metadata import: %#v", importResponse.MetadataImport)
	}

	summaryReq := httptest.NewRequest(http.MethodGet, "/datasets/"+dataset.ID+"/metadata/summary", nil)
	summaryResp := httptest.NewRecorder()
	router.ServeHTTP(summaryResp, summaryReq)
	if summaryResp.Code != http.StatusOK {
		t.Fatalf("expected summary status %d, got %d: %s", http.StatusOK, summaryResp.Code, summaryResp.Body.String())
	}
	summaryBody := summaryResp.Body.String()
	if strings.Contains(summaryBody, "labels.csv") || strings.Contains(summaryBody, "s3://") || strings.Contains(summaryBody, "train/cat/one.jpg") {
		t.Fatalf("safe summary leaked raw path/source content: %s", summaryBody)
	}
	var summaryPayload struct {
		Summary datasets.AgentSafeDatasetMetadataSummary `json:"summary"`
	}
	if err := json.Unmarshal(summaryResp.Body.Bytes(), &summaryPayload); err != nil {
		t.Fatalf("decode summary response: %v", err)
	}
	if summaryPayload.Summary.SampleCount != 2 || !summaryPayload.Summary.OfficialSplitAvailable {
		t.Fatalf("unexpected safe summary: %#v", summaryPayload.Summary)
	}

	bundleReq := httptest.NewRequest(http.MethodGet, "/datasets/"+dataset.ID+"/metadata/bundle?purpose=training&limit=1&offset=0", nil)
	bundleResp := httptest.NewRecorder()
	router.ServeHTTP(bundleResp, bundleReq)
	if bundleResp.Code != http.StatusOK {
		t.Fatalf("expected bundle status %d, got %d: %s", http.StatusOK, bundleResp.Code, bundleResp.Body.String())
	}
	var bundlePayload struct {
		Bundle datasets.DatasetMetadataBundle `json:"bundle"`
	}
	if err := json.Unmarshal(bundleResp.Body.Bytes(), &bundlePayload); err != nil {
		t.Fatalf("decode bundle response: %v", err)
	}
	if bundlePayload.Bundle.ImportID != importResponse.MetadataImport.ID || len(bundlePayload.Bundle.ManifestRecords) != 1 || bundlePayload.Bundle.NextOffset == nil {
		t.Fatalf("unexpected bundle payload: %#v", bundlePayload.Bundle)
	}
}

func TestCompleteJobAcknowledgesBeforePostTrainingHooksFinish(t *testing.T) {
	llmStarted := make(chan struct{}, 1)
	releaseLLM := make(chan struct{})
	var releaseOnce sync.Once
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case llmStarted <- struct{}{}:
		default:
		}
		<-releaseLLM
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]string{
					"role": "assistant",
					"content": `{
						"summary": "Training run recorded.",
						"recommended_action": {
							"action_type": "RANK_MODELS",
							"confidence": 0.7,
							"rationale": "The run is terminal and should inform ranking.",
							"payload": {},
							"requires_approval": true
						},
						"quality_summary": "Usable validation signal.",
						"training_dynamics": "No blocking issue.",
						"cost_summary": "Cost was acceptable.",
						"risks": [],
						"findings": ["completed"],
						"rank_score": 0.6,
						"tags": ["completed"]
					}`,
				},
			}},
		})
	}))
	defer llmServer.Close()
	defer releaseOnce.Do(func() { close(releaseLLM) })

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "test-model")
	t.Setenv("MODEL_EXPRESS_TRAINING_MONITOR_LLM_ENABLED", "true")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 1, 5, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 3)}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"plan_id":       plan.ID,
		"dataset_id":    dataset.ID,
		"model":         "mobilenet_v3_small",
		"provider":      "local",
		"target_metric": "macro_f1",
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}

	router := NewRouter(memoryStore)
	assigned := pollJobForCallback(t, router, worker.ID, `{}`)
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/complete", strings.NewReader(`{"training_attempt_id":"`+callbackAttemptID(t, assigned)+`","mlflow_run_id":"run_1"}`))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(resp, req)
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(300 * time.Millisecond):
		releaseOnce.Do(func() { close(releaseLLM) })
		t.Fatal("complete endpoint waited for post-training LLM hooks")
	}
	if resp.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	completed, err := memoryStore.GetJob(job.ID)
	if err != nil {
		t.Fatalf("get completed job: %v", err)
	}
	if completed.Status != jobs.StatusSucceeded {
		t.Fatalf("expected completed job status, got %s", completed.Status)
	}

	select {
	case <-llmStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("expected post-training hook to start asynchronously")
	}
	releaseOnce.Do(func() { close(releaseLLM) })
}

func TestCreateChampionExportFallsBackToPendingArtifact(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(job.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	if _, err := memoryStore.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
		Model:    "mobilenet_v3_small",
		Provider: "local",
		Status:   jobs.StatusSucceeded,
	}); err != nil {
		t.Fatalf("upsert summary: %v", err)
	}
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:       project.ID,
		JobID:           job.ID,
		SelectionReason: "best validation score",
		Metrics:         map[string]any{"best_macro_f1": 0.9},
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}

	router := NewRouter(memoryStore)
	body := []byte(`{"format":"onnx"}`)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/exports", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusCreated {
		t.Fatalf("expected status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	var export runs.ChampionExport
	if err := json.Unmarshal(resp.Body.Bytes(), &export); err != nil {
		t.Fatalf("decode export: %v", err)
	}
	if export.ChampionID != champion.ID || export.JobID != job.ID {
		t.Fatalf("unexpected export target: %#v", export)
	}
	if export.Status != runs.ChampionExportStatusPendingArtifact {
		t.Fatalf("expected pending artifact status, got %s", export.Status)
	}
	if len(export.ValidationErrors) == 0 {
		t.Fatal("expected validation error explaining missing artifact")
	}
}

func TestCreateChampionExportIsIdempotentByChampionAndFormat(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(job.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	if _, err := memoryStore.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
		Model:    "mobilenet_v3_small",
		Provider: "local",
		Status:   jobs.StatusSucceeded,
	}); err != nil {
		t.Fatalf("upsert summary: %v", err)
	}
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:         project.ID,
		JobID:             job.ID,
		SelectionReason:   "best validation score",
		DeploymentProfile: map[string]any{"artifact_uri": "s3://bucket/model-express/artifacts/train_1/model.onnx"},
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}

	router := NewRouter(memoryStore)
	firstReq := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/exports", strings.NewReader(`{"format":"onnx","metadata":{"request":"first"}}`))
	firstReq.Header.Set("Content-Type", "application/json")
	firstResp := httptest.NewRecorder()
	router.ServeHTTP(firstResp, firstReq)
	if firstResp.Code != http.StatusCreated {
		t.Fatalf("expected first status %d, got %d: %s", http.StatusCreated, firstResp.Code, firstResp.Body.String())
	}
	var first runs.ChampionExport
	if err := json.Unmarshal(firstResp.Body.Bytes(), &first); err != nil {
		t.Fatalf("decode first export: %v", err)
	}

	rejectedReq := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/exports", strings.NewReader(`{"format":"onnx","artifact_uri":"file:///second.onnx","metadata":{"request":"rejected"}}`))
	rejectedReq.Header.Set("Content-Type", "application/json")
	rejectedResp := httptest.NewRecorder()
	router.ServeHTTP(rejectedResp, rejectedReq)
	if rejectedResp.Code != http.StatusBadRequest {
		t.Fatalf("expected request-supplied artifact uri status %d, got %d: %s", http.StatusBadRequest, rejectedResp.Code, rejectedResp.Body.String())
	}

	secondReq := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/exports", strings.NewReader(`{"format":"onnx","metadata":{"request":"second"}}`))
	secondReq.Header.Set("Content-Type", "application/json")
	secondResp := httptest.NewRecorder()
	router.ServeHTTP(secondResp, secondReq)
	if secondResp.Code != http.StatusCreated {
		t.Fatalf("expected second status %d, got %d: %s", http.StatusCreated, secondResp.Code, secondResp.Body.String())
	}
	var second runs.ChampionExport
	if err := json.Unmarshal(secondResp.Body.Bytes(), &second); err != nil {
		t.Fatalf("decode second export: %v", err)
	}
	if second.ID != first.ID {
		t.Fatalf("expected existing export %s to be updated, got new export %s", first.ID, second.ID)
	}
	if second.ChampionID != champion.ID || second.ArtifactURI != "s3://bucket/model-express/artifacts/train_1/model.onnx" {
		t.Fatalf("unexpected updated export: %#v", second)
	}
	exports, err := memoryStore.ListProjectChampionExports(project.ID)
	if err != nil {
		t.Fatalf("list exports: %v", err)
	}
	if len(exports) != 1 {
		t.Fatalf("expected one export after duplicate request, got %d", len(exports))
	}
	if got := exports[0].Metadata["request"]; got != "second" {
		t.Fatalf("expected updated metadata, got %#v", got)
	}
}

func TestCreateChampionDemoPredictionAuditsRuntimeUnavailable(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{
		"demo_images": []map[string]any{
			{
				"id":         "image_1",
				"uri":        "file:///dataset/test/cat/1.jpg",
				"class_name": "cat",
				"label":      "cat",
				"split":      "test",
				"width":      224,
				"height":     224,
			},
		},
	})
	if err != nil {
		t.Fatalf("update dataset profile: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(job.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	if _, err := memoryStore.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
		Model:    "mobilenet_v3_small",
		Provider: "local",
		Status:   jobs.StatusSucceeded,
	}); err != nil {
		t.Fatalf("upsert summary: %v", err)
	}
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:       project.ID,
		DatasetID:       dataset.ID,
		JobID:           job.ID,
		SelectionReason: "best validation score",
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}

	router := NewRouter(memoryStore)
	body := []byte(`{"image_uri":"file:///dataset/test/cat/1.jpg","top_k":3}`)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/demo-predictions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusAccepted {
		t.Fatalf("expected status %d, got %d: %s", http.StatusAccepted, resp.Code, resp.Body.String())
	}
	var payload struct {
		Prediction       runs.ChampionDemoPrediction `json:"prediction"`
		RuntimeAvailable bool                        `json:"runtime_available"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode prediction response: %v", err)
	}
	if payload.RuntimeAvailable {
		t.Fatal("expected runtime_available=false while inference runtime is unwired")
	}
	if payload.Prediction.ChampionID != champion.ID || payload.Prediction.JobID != job.ID {
		t.Fatalf("unexpected prediction target: %#v", payload.Prediction)
	}
	if payload.Prediction.Status != runs.ChampionDemoPredictionStatusRuntimeUnavailable {
		t.Fatalf("expected runtime unavailable status, got %s", payload.Prediction.Status)
	}
	if payload.Prediction.TrueLabel != "cat" || payload.Prediction.PredictedLabel != "" || payload.Prediction.Confidence != nil {
		t.Fatalf("prediction should audit metadata without pretending inference succeeded: %#v", payload.Prediction)
	}
	predictions, err := memoryStore.ListProjectChampionDemoPredictions(project.ID)
	if err != nil {
		t.Fatalf("list predictions: %v", err)
	}
	if len(predictions) != 1 {
		t.Fatalf("expected one prediction audit row, got %d", len(predictions))
	}
	events, err := memoryStore.ListProjectExecutionEvents(project.ID, 10)
	if err != nil {
		t.Fatalf("list events: %v", err)
	}
	if !hasExecutionEvent(events, execution.EventChampionDemoPrediction) {
		t.Fatal("expected champion demo prediction event")
	}
}

func TestChampionExportRequestQueuesWorkerJobAndAcceptsResult(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(job.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	if _, err := memoryStore.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
		Model:    "mobilenet_v3_small",
		Provider: "local",
		Status:   jobs.StatusSucceeded,
	}); err != nil {
		t.Fatalf("upsert summary: %v", err)
	}
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:         project.ID,
		DatasetID:         dataset.ID,
		JobID:             job.ID,
		SelectionReason:   "best validation score",
		DeploymentProfile: map[string]any{"artifact_uri": "file:///checkpoint.pt"},
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/exports", strings.NewReader(`{"format":"onnx"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	var export runs.ChampionExport
	if err := json.Unmarshal(resp.Body.Bytes(), &export); err != nil {
		t.Fatalf("decode export: %v", err)
	}
	if export.Status != runs.ChampionExportStatusPending {
		t.Fatalf("expected pending worker export, got %s", export.Status)
	}
	if export.ArtifactURI != "" {
		t.Fatalf("expected pending ONNX export to omit mismatched checkpoint artifact, got %q", export.ArtifactURI)
	}
	exportJob := findProjectJob(t, memoryStore, project.ID, jobs.TemplateExportChampion)
	if exportJob.Config["export_id"] != export.ID || exportJob.Config["champion_id"] != champion.ID {
		t.Fatalf("unexpected export job config: %#v", exportJob.Config)
	}
	if artifactURI := jobConfigString(exportJob.Config, "artifact_uri"); artifactURI != "" {
		t.Fatalf("expected export job artifact_uri to be empty until ONNX exists, got %q", artifactURI)
	}
	if sourceArtifactURI := jobConfigString(exportJob.Config, "source_artifact_uri"); sourceArtifactURI != "file:///checkpoint.pt" {
		t.Fatalf("expected source checkpoint to be preserved separately, got %q", sourceArtifactURI)
	}

	worker, err := memoryStore.RegisterWorker(project.ID, "export worker", "")
	if err != nil {
		t.Fatalf("register export worker: %v", err)
	}
	assignedExport := pollJobForCallback(t, router, worker.ID, `{"templates":["export_champion"]}`)
	exportArtifactURI := "file:///exports/champion.onnx"
	resultPayload, err := json.Marshal(map[string]any{
		"training_attempt_id": callbackAttemptID(t, assignedExport),
		"status":              "READY",
		"artifact_uri":        exportArtifactURI,
		"metadata": map[string]any{
			"labels": []string{"cat", "dog"},
			"manifest": map[string]any{
				"schema_version": "champion_export_manifest_v1",
				"status":         "ok",
				"metadata": map[string]any{
					"format": "onnx",
					"provenance": map[string]any{
						"schema_version":   "worker_artifact_provenance_v1",
						"generated_by":     "model-express-worker",
						"source":           "worker_generated",
						"export_job_id":    exportJob.ID,
						"source_export_id": export.ID,
						"artifact_format":  "onnx",
					},
				},
				"artifacts": []map[string]any{{
					"format": "onnx",
					"status": "created",
					"path":   "/exports/champion.onnx",
					"provenance": map[string]any{
						"schema_version":   "worker_artifact_provenance_v1",
						"generated_by":     "model-express-worker",
						"source":           "worker_generated",
						"export_job_id":    exportJob.ID,
						"source_export_id": export.ID,
						"artifact_format":  "onnx",
					},
				}},
			},
		},
	})
	if err != nil {
		t.Fatalf("marshal export result payload: %v", err)
	}
	resultReq := httptest.NewRequest(http.MethodPost, "/jobs/"+exportJob.ID+"/champion-export-result", bytes.NewReader(resultPayload))
	resultReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, resultReq, assignedExport)
	resultResp := httptest.NewRecorder()
	router.ServeHTTP(resultResp, resultReq)
	if resultResp.Code != http.StatusOK {
		t.Fatalf("expected result status %d, got %d: %s", http.StatusOK, resultResp.Code, resultResp.Body.String())
	}
	exports, err := memoryStore.ListProjectChampionExports(project.ID)
	if err != nil {
		t.Fatalf("list exports: %v", err)
	}
	if exports[0].Status != runs.ChampionExportStatusReady || exports[0].ArtifactURI != exportArtifactURI {
		t.Fatalf("expected ready export with artifact, got %#v", exports[0])
	}
	updatedJob, err := memoryStore.GetJob(exportJob.ID)
	if err != nil {
		t.Fatalf("get export job: %v", err)
	}
	if updatedJob.Status != jobs.StatusSucceeded {
		t.Fatalf("expected export job to complete, got %s", updatedJob.Status)
	}
}

func TestChampionArtifactURIForFormatPrefersMatchingArtifact(t *testing.T) {
	deploymentProfile := map[string]any{
		"artifact_uri": "file:///checkpoint.pt",
		"model_profile": map[string]any{
			"onnx_artifact_uri": "file:///model.onnx",
		},
	}

	if artifactURI := championArtifactURIForFormat(deploymentProfile, "onnx"); artifactURI != "file:///model.onnx" {
		t.Fatalf("expected ONNX artifact to be preferred, got %q", artifactURI)
	}
	if artifactURI := championArtifactURIForFormat(deploymentProfile, "pytorch"); artifactURI != "file:///checkpoint.pt" {
		t.Fatalf("expected checkpoint artifact to be preferred for pytorch, got %q", artifactURI)
	}
}

func TestChampionDemoPredictionQueuesWorkerJobAndAcceptsResult(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{
		"demo_images": []map[string]any{
			{"id": "image_1", "uri": "file:///dataset/test/cat/1.jpg", "class_name": "cat", "label": "cat"},
		},
	})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}
	trainingJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(trainingJob.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:       project.ID,
		DatasetID:       dataset.ID,
		JobID:           trainingJob.ID,
		SelectionReason: "best validation score",
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}
	if _, err := memoryStore.CreateChampionExport(runs.ChampionExportCreate{
		ProjectID:   project.ID,
		ChampionID:  champion.ID,
		JobID:       trainingJob.ID,
		Status:      runs.ChampionExportStatusReady,
		Format:      "onnx",
		ArtifactURI: "file:///exports/champion.onnx",
	}); err != nil {
		t.Fatalf("create ready export: %v", err)
	}

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/demo-predictions", strings.NewReader(`{"image_uri":"file:///dataset/test/cat/1.jpg","top_k":3}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusAccepted {
		t.Fatalf("expected status %d, got %d: %s", http.StatusAccepted, resp.Code, resp.Body.String())
	}
	var payload struct {
		Prediction       runs.ChampionDemoPrediction `json:"prediction"`
		RuntimeAvailable bool                        `json:"runtime_available"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !payload.RuntimeAvailable || payload.Prediction.Status != runs.ChampionDemoPredictionStatusPending {
		t.Fatalf("expected pending runtime-backed prediction, got %#v", payload)
	}
	predictionJob := findProjectJob(t, memoryStore, project.ID, jobs.TemplateChampionDemoPrediction)
	if predictionJob.Config["prediction_id"] != payload.Prediction.ID {
		t.Fatalf("unexpected prediction job config: %#v", predictionJob.Config)
	}

	worker, err := memoryStore.RegisterWorker(project.ID, "demo worker", "")
	if err != nil {
		t.Fatalf("register demo worker: %v", err)
	}
	assignedPrediction := pollJobForCallback(t, router, worker.ID, `{"templates":["champion_demo_prediction"]}`)
	resultReq := httptest.NewRequest(http.MethodPost, "/jobs/"+predictionJob.ID+"/champion-demo-prediction-result", strings.NewReader(`{"training_attempt_id":"`+callbackAttemptID(t, assignedPrediction)+`","status":"SUCCEEDED","predicted_label":"cat","confidence":0.97,"top_k":[{"label":"cat","confidence":0.97}],"latency_ms":12.5,"correct":true}`))
	resultReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, resultReq, assignedPrediction)
	resultResp := httptest.NewRecorder()
	router.ServeHTTP(resultResp, resultReq)
	if resultResp.Code != http.StatusOK {
		t.Fatalf("expected result status %d, got %d: %s", http.StatusOK, resultResp.Code, resultResp.Body.String())
	}
	predictions, err := memoryStore.ListProjectChampionDemoPredictions(project.ID)
	if err != nil {
		t.Fatalf("list predictions: %v", err)
	}
	if predictions[0].Status != runs.ChampionDemoPredictionStatusSucceeded || predictions[0].PredictedLabel != "cat" || predictions[0].Correct == nil || !*predictions[0].Correct {
		t.Fatalf("expected succeeded prediction result, got %#v", predictions[0])
	}
}

func TestChampionDemoPredictionUsesDeploymentArtifactWhenNoExportRecordExists(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	trainingJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(trainingJob.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	manifest := map[string]any{
		"schema_version": "champion_export_manifest_v1",
		"metadata": map[string]any{
			"model":        "mobilenet_v3_small",
			"class_labels": []string{"cat", "dog"},
			"input_shape":  []int{1, 3, 224, 224},
		},
		"artifacts": []map[string]any{
			{"format": "framework_native_checkpoint", "status": "created", "path": "s3://model-express/model-express/artifacts/job_1/model.pt"},
		},
	}
	if _, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:       project.ID,
		DatasetID:       dataset.ID,
		JobID:           trainingJob.ID,
		SelectionReason: "best validation score",
		DeploymentProfile: map[string]any{
			"artifact_uri": "s3://model-express/model-express/artifacts/job_1/model.pt",
			"model_profile": map[string]any{
				"export_manifest": manifest,
			},
		},
	}); err != nil {
		t.Fatalf("upsert champion: %v", err)
	}
	champion, err := memoryStore.GetProjectChampion(project.ID)
	if err != nil {
		t.Fatalf("get champion: %v", err)
	}
	if _, err := memoryStore.CreateChampionExport(runs.ChampionExportCreate{
		ProjectID:   project.ID,
		ChampionID:  champion.ID,
		JobID:       trainingJob.ID,
		Status:      runs.ChampionExportStatusReady,
		Format:      "onnx",
		ArtifactURI: "s3://model-express/model-express/artifacts/job_1/model.pt",
	}); err != nil {
		t.Fatalf("create stale mismatched export: %v", err)
	}

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/demo-predictions", strings.NewReader(`{"image_uri":"data:image/jpeg;base64,/9j/4AAQSkZJRg==","true_label":"cat"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusAccepted {
		t.Fatalf("expected status %d, got %d: %s", http.StatusAccepted, resp.Code, resp.Body.String())
	}
	predictionJob := findProjectJob(t, memoryStore, project.ID, jobs.TemplateChampionDemoPrediction)
	if artifactURI := jobConfigString(predictionJob.Config, "export_artifact_uri"); artifactURI != "s3://model-express/model-express/artifacts/job_1/model.pt" {
		t.Fatalf("expected deployment checkpoint artifact, got %q", artifactURI)
	}
	if manifest := payloadMap(payloadMap(predictionJob.Config, "export_metadata"), "manifest"); len(manifest) == 0 {
		t.Fatalf("expected deployment manifest metadata in prediction job config: %#v", predictionJob.Config)
	}
}

func TestChampionDemoPredictionAcceptsRuntimeUnavailableWorkerResult(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	trainingJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(trainingJob.ID, "mlflow-run"); err != nil {
		t.Fatalf("complete job: %v", err)
	}
	champion, err := memoryStore.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:       project.ID,
		DatasetID:       dataset.ID,
		JobID:           trainingJob.ID,
		SelectionReason: "best validation score",
	})
	if err != nil {
		t.Fatalf("upsert champion: %v", err)
	}
	if _, err := memoryStore.CreateChampionExport(runs.ChampionExportCreate{
		ProjectID:   project.ID,
		ChampionID:  champion.ID,
		JobID:       trainingJob.ID,
		Status:      runs.ChampionExportStatusReady,
		Format:      "onnx",
		ArtifactURI: "file:///exports/champion.onnx",
	}); err != nil {
		t.Fatalf("create ready export: %v", err)
	}

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/demo-predictions", strings.NewReader(`{"image_uri":"file:///dataset/test/cat/1.jpg","top_k":3}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusAccepted {
		t.Fatalf("expected status %d, got %d: %s", http.StatusAccepted, resp.Code, resp.Body.String())
	}
	var payload struct {
		Prediction runs.ChampionDemoPrediction `json:"prediction"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	predictionJob := findProjectJob(t, memoryStore, project.ID, jobs.TemplateChampionDemoPrediction)
	if predictionJob.Config["dataset_id"] != dataset.ID {
		t.Fatalf("expected dataset_id in prediction job config, got %#v", predictionJob.Config)
	}

	worker, err := memoryStore.RegisterWorker(project.ID, "demo worker", "")
	if err != nil {
		t.Fatalf("register demo worker: %v", err)
	}
	assignedPrediction := pollJobForCallback(t, router, worker.ID, `{"templates":["champion_demo_prediction"]}`)
	resultReq := httptest.NewRequest(http.MethodPost, "/jobs/"+predictionJob.ID+"/champion-demo-prediction-result", strings.NewReader(`{"training_attempt_id":"`+callbackAttemptID(t, assignedPrediction)+`","status":"RUNTIME_UNAVAILABLE","error":"No worker-owned export manifest path was supplied or found."}`))
	resultReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, resultReq, assignedPrediction)
	resultResp := httptest.NewRecorder()
	router.ServeHTTP(resultResp, resultReq)
	if resultResp.Code != http.StatusOK {
		t.Fatalf("expected result status %d, got %d: %s", http.StatusOK, resultResp.Code, resultResp.Body.String())
	}
	predictions, err := memoryStore.ListProjectChampionDemoPredictions(project.ID)
	if err != nil {
		t.Fatalf("list predictions: %v", err)
	}
	if predictions[0].Status != runs.ChampionDemoPredictionStatusRuntimeUnavailable || predictions[0].Error == "" {
		t.Fatalf("expected runtime-unavailable prediction result, got %#v", predictions[0])
	}
	updatedJob, err := memoryStore.GetJob(predictionJob.ID)
	if err != nil {
		t.Fatalf("get prediction job: %v", err)
	}
	if updatedJob.Status != jobs.StatusSucceeded {
		t.Fatalf("runtime-unavailable result should complete audit job, got %s", updatedJob.Status)
	}
}

func TestMergeDatasetVisualExemplarsEnforcesCaps(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification"})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}
	router := NewRouter(memoryStore)
	tooMany := make([]map[string]any, 49)
	for i := range tooMany {
		tooMany[i] = map[string]any{"uri": "file:///image-" + strconv.Itoa(i) + ".jpg", "class_name": "cat", "size_bytes": 1}
	}
	body, _ := json.Marshal(map[string]any{"visual_exemplars": tooMany})
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-exemplars", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected cap rejection %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}

	validBody := []byte(`{"visual_exemplars":[{"uri":"file:///cat/1.jpg","class_name":"cat","size_bytes":100}],"demo_images":[{"uri":"file:///cat/demo.jpg","class_name":"cat","size_bytes":100}]}`)
	validReq := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-exemplars", bytes.NewReader(validBody))
	validReq.Header.Set("Content-Type", "application/json")
	validResp := httptest.NewRecorder()
	router.ServeHTTP(validResp, validReq)
	if validResp.Code != http.StatusOK {
		t.Fatalf("expected merge status %d, got %d: %s", http.StatusOK, validResp.Code, validResp.Body.String())
	}
	updated, err := memoryStore.GetDataset(dataset.ID)
	if err != nil {
		t.Fatalf("get dataset: %v", err)
	}
	if len(cappedVisualExemplars(updated.Profile, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}, "visual_exemplars")) != 1 {
		t.Fatalf("expected one stored visual exemplar, got %#v", updated.Profile["visual_exemplars"])
	}
	if len(cappedVisualExemplars(updated.Profile, visualExemplarCaps{MaxTotalImages: 48, MaxPerClass: 6, MaxBytes: 3_000_000}, "demo_images")) != 1 {
		t.Fatalf("expected one stored demo image, got %#v", updated.Profile["demo_images"])
	}
}

func TestExpiredJobLeasesRequeueThenFailAtMaxAttempts(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	createdJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	for attempt := 1; attempt <= 2; attempt++ {
		assigned, err := memoryStore.PollJob(worker.ID, store.JobPollFilter{})
		if err != nil {
			t.Fatalf("poll attempt %d: %v", attempt, err)
		}
		if assigned.ID != createdJob.ID || assigned.Attempt != attempt || assigned.LeaseExpiresAt == nil {
			t.Fatalf("unexpected assignment at attempt %d: %#v", attempt, assigned)
		}
		recovered, err := memoryStore.RecoverExpiredJobLeases(time.Now().UTC().Add(3 * time.Hour))
		if err != nil {
			t.Fatalf("recover attempt %d: %v", attempt, err)
		}
		if len(recovered) != 1 || recovered[0].Status != jobs.StatusQueued {
			t.Fatalf("expected queued recovery at attempt %d, got %#v", attempt, recovered)
		}
	}
	assigned, err := memoryStore.PollJob(worker.ID, store.JobPollFilter{})
	if err != nil {
		t.Fatalf("poll final attempt: %v", err)
	}
	if assigned.Attempt != 3 {
		t.Fatalf("expected third attempt, got %#v", assigned)
	}
	recovered, err := memoryStore.RecoverExpiredJobLeases(time.Now().UTC().Add(3 * time.Hour))
	if err != nil {
		t.Fatalf("recover final attempt: %v", err)
	}
	if len(recovered) != 1 || recovered[0].Status != jobs.StatusFailed {
		t.Fatalf("expected failed recovery at max attempts, got %#v", recovered)
	}
}

func TestRetryableFailJobRequeuesUntilMaxAttempts(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "modal",
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	router := NewRouter(memoryStore)

	for attempt := 1; attempt <= 2; attempt++ {
		assigned := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
		if assigned.ID != job.ID || assigned.Attempt != attempt {
			t.Fatalf("unexpected assigned job at attempt %d: %#v", attempt, assigned)
		}
		body := strings.NewReader(`{"training_attempt_id":"` + callbackAttemptID(t, assigned) + `","error":"modal container exited before completion","retryable":true}`)
		req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/fail", body)
		req.Header.Set("Content-Type", "application/json")
		setCallbackToken(t, req, assigned)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		if resp.Code != http.StatusOK {
			t.Fatalf("expected retryable fail status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
		}
		requeued, err := memoryStore.GetJob(job.ID)
		if err != nil {
			t.Fatalf("get requeued job: %v", err)
		}
		if requeued.Status != jobs.StatusQueued || requeued.Attempt != attempt {
			t.Fatalf("expected queued retry at attempt %d, got %#v", attempt, requeued)
		}
		updatedWorker, err := memoryStore.GetWorker(worker.ID)
		if err != nil {
			t.Fatalf("get worker: %v", err)
		}
		if updatedWorker.CurrentJobID != "" || updatedWorker.Status != workers.StatusIdle {
			t.Fatalf("expected worker released after retryable failure, got %#v", updatedWorker)
		}
	}

	assigned := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
	if assigned.ID != job.ID || assigned.Attempt != 3 {
		t.Fatalf("unexpected final assignment: %#v", assigned)
	}
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/fail", strings.NewReader(`{"training_attempt_id":"`+callbackAttemptID(t, assigned)+`","error":"modal container exited before completion","retryable":true}`))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected final retryable fail status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	failed, err := memoryStore.GetJob(job.ID)
	if err != nil {
		t.Fatalf("get failed job: %v", err)
	}
	if failed.Status != jobs.StatusFailed || failed.Attempt != 3 {
		t.Fatalf("expected terminal failure after max attempts, got %#v", failed)
	}
}

func TestRetryableOOMFailureEscalatesGpuAndBlocksRepeatedResource(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "modal",
		"gpu_type":   "T4",
		"batch_size": 8,
		"task_type":  "object_detection",
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	router := NewRouter(memoryStore)

	assigned := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
	activeAttemptID := jobConfigString(assigned.Config, "active_attempt_id")
	body, _ := json.Marshal(map[string]any{
		"error":                    "CUDA out of memory",
		"retryable":                true,
		"training_attempt_id":      activeAttemptID,
		"oom":                      true,
		"failure_class":            "oom",
		"oom_kind":                 "gpu_cuda",
		"effective_gpu_type":       "T4",
		"effective_batch_size":     8,
		"memory_mb":                24576,
		"modal_resource_signature": "gpu=T4|batch=8|memory_mb=24576",
	})
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/fail", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected OOM retry status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	requeued, err := memoryStore.GetJob(job.ID)
	if err != nil {
		t.Fatalf("get requeued job: %v", err)
	}
	if requeued.Status != jobs.StatusQueued || jobConfigString(requeued.Config, "gpu_type") != "L4" {
		t.Fatalf("expected queued retry escalated to L4, got %#v", requeued)
	}
	history, ok := requeued.Config[modalOOMRetryHistoryKey].([]map[string]any)
	if !ok || len(history) != 1 {
		t.Fatalf("expected one OOM history entry, got %#v", requeued.Config[modalOOMRetryHistoryKey])
	}

	guardStore := store.NewMemoryStore()
	guardProject, err := guardStore.CreateProject("guard demo", "")
	if err != nil {
		t.Fatalf("create guard project: %v", err)
	}
	guardDataset, err := guardStore.CreateDataset(guardProject.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create guard dataset: %v", err)
	}
	guardWorker, err := guardStore.RegisterWorker(guardProject.ID, "worker", "modal")
	if err != nil {
		t.Fatalf("register guard worker: %v", err)
	}
	repeatedJob, err := guardStore.CreateJob(guardProject.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": guardDataset.ID,
		"provider":   "modal",
		"gpu_type":   "T4",
		"batch_size": 8,
		"task_type":  "object_detection",
		modalOOMRetryHistoryKey: []map[string]any{{
			"modal_resource_signature": "gpu=T4|batch=8|memory_mb=24576",
		}},
	})
	if err != nil {
		t.Fatalf("create repeated job: %v", err)
	}
	guardRouter := NewRouter(guardStore)
	assignedRepeated := pollJobForCallback(t, guardRouter, guardWorker.ID, `{"provider":"modal"}`)
	repeatedBody, _ := json.Marshal(map[string]any{
		"error":                    "CUDA out of memory",
		"retryable":                true,
		"training_attempt_id":      jobConfigString(assignedRepeated.Config, "active_attempt_id"),
		"oom":                      true,
		"failure_class":            "oom",
		"oom_kind":                 "gpu_cuda",
		"effective_gpu_type":       "T4",
		"effective_batch_size":     8,
		"memory_mb":                24576,
		"modal_resource_signature": "gpu=T4|batch=8|memory_mb=24576",
	})
	repeatedReq := httptest.NewRequest(http.MethodPost, "/jobs/"+repeatedJob.ID+"/fail", bytes.NewReader(repeatedBody))
	repeatedReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, repeatedReq, assignedRepeated)
	repeatedResp := httptest.NewRecorder()
	guardRouter.ServeHTTP(repeatedResp, repeatedReq)
	if repeatedResp.Code != http.StatusOK {
		t.Fatalf("expected repeated OOM status %d, got %d: %s", http.StatusOK, repeatedResp.Code, repeatedResp.Body.String())
	}
	blocked, err := guardStore.GetJob(repeatedJob.ID)
	if err != nil {
		t.Fatalf("get blocked job: %v", err)
	}
	if blocked.Status != jobs.StatusFailed {
		t.Fatalf("expected repeated OOM signature to fail without requeue, got %#v", blocked)
	}
}

func TestStaleModalAttemptCallbacksDoNotOverwriteActiveAttempt(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "modal",
		"gpu_type":   "T4",
		"batch_size": 8,
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	router := NewRouter(memoryStore)

	firstAttempt := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
	firstAttemptID := jobConfigString(firstAttempt.Config, "active_attempt_id")
	failBody, _ := json.Marshal(map[string]any{
		"error":                    "CUDA out of memory",
		"retryable":                true,
		"training_attempt_id":      firstAttemptID,
		"oom":                      true,
		"failure_class":            "oom",
		"effective_gpu_type":       "T4",
		"effective_batch_size":     8,
		"memory_mb":                16384,
		"modal_resource_signature": "gpu=T4|batch=8|memory_mb=16384",
	})
	failReq := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/fail", bytes.NewReader(failBody))
	failReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, failReq, firstAttempt)
	failResp := httptest.NewRecorder()
	router.ServeHTTP(failResp, failReq)
	if failResp.Code != http.StatusOK {
		t.Fatalf("expected fail status %d, got %d: %s", http.StatusOK, failResp.Code, failResp.Body.String())
	}

	secondAttempt := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
	secondAttemptID := jobConfigString(secondAttempt.Config, "active_attempt_id")
	goodSummary, _ := json.Marshal(map[string]any{
		"training_attempt_id": secondAttemptID,
		"status":              jobs.StatusRunning,
		"best_macro_f1":       0.9,
		"gpu_type":            "L4",
	})
	goodReq := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/training-run-summary", bytes.NewReader(goodSummary))
	goodReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, goodReq, secondAttempt)
	goodResp := httptest.NewRecorder()
	router.ServeHTTP(goodResp, goodReq)
	if goodResp.Code != http.StatusOK {
		t.Fatalf("expected active summary status %d, got %d: %s", http.StatusOK, goodResp.Code, goodResp.Body.String())
	}

	staleSummary, _ := json.Marshal(map[string]any{
		"training_attempt_id": firstAttemptID,
		"status":              jobs.StatusSucceeded,
		"best_macro_f1":       0.1,
		"gpu_type":            "T4",
	})
	staleReq := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/training-run-summary", bytes.NewReader(staleSummary))
	staleReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, staleReq, firstAttempt)
	staleResp := httptest.NewRecorder()
	router.ServeHTTP(staleResp, staleReq)
	if staleResp.Code != http.StatusConflict {
		t.Fatalf("expected stale summary status %d, got %d: %s", http.StatusConflict, staleResp.Code, staleResp.Body.String())
	}
	summary, err := memoryStore.GetTrainingRunSummary(job.ID)
	if err != nil {
		t.Fatalf("get summary: %v", err)
	}
	if summary.BestMacroF1 != 0.9 || summary.Status != jobs.StatusRunning || summary.GPUType != "L4" {
		t.Fatalf("stale summary overwrote active summary: %#v", summary)
	}

	completeReq := httptest.NewRequest(
		http.MethodPost,
		"/jobs/"+job.ID+"/complete",
		strings.NewReader(`{"training_attempt_id":"`+firstAttemptID+`","mlflow_run_id":"stale"}`),
	)
	completeReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, completeReq, firstAttempt)
	completeResp := httptest.NewRecorder()
	router.ServeHTTP(completeResp, completeReq)
	if completeResp.Code != http.StatusConflict {
		t.Fatalf("expected stale complete status %d, got %d: %s", http.StatusConflict, completeResp.Code, completeResp.Body.String())
	}
	activeJob, err := memoryStore.GetJob(job.ID)
	if err != nil {
		t.Fatalf("get active job: %v", err)
	}
	if activeJob.Status == jobs.StatusSucceeded || activeJob.MLflowRunID == "stale" {
		t.Fatalf("stale completion overwrote active job: %#v", activeJob)
	}
}

func TestFailJobDoesNotDowngradeSucceededJob(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "local",
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	router := NewRouter(memoryStore)
	assigned := pollJobForCallback(t, router, worker.ID, `{}`)
	if _, err := memoryStore.CompleteJob(job.ID, "run_1"); err != nil {
		t.Fatalf("complete job: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/fail", strings.NewReader(`{"training_attempt_id":"`+callbackAttemptID(t, assigned)+`","error":"late failure","retryable":true}`))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusConflict {
		t.Fatalf("expected late fail status %d, got %d: %s", http.StatusConflict, resp.Code, resp.Body.String())
	}
	updated, err := memoryStore.GetJob(job.ID)
	if err != nil {
		t.Fatalf("get job: %v", err)
	}
	if updated.Status != jobs.StatusSucceeded || updated.MLflowRunID != "run_1" {
		t.Fatalf("late fail downgraded terminal job: %#v", updated)
	}
}

func TestCancelPlanActiveExecutionStopsJobsRequirementsAndIgnoresStaleCallbacks(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	router := NewRouter(memoryStore)
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", strings.Repeat("c", 64), 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 2, 10, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
		testExperiment("efficientnet_b0", 6),
	}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	result, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{
		Provider:          "modal",
		GPUType:           "T4",
		MaxConcurrentJobs: 2,
	})
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	if len(result.Jobs) != 2 || result.WorkerRequirement == nil {
		t.Fatalf("expected two jobs and worker requirement, got %#v", result)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "modal slot", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	assigned := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
	activeAttemptID := jobConfigString(assigned.Config, "active_attempt_id")
	if activeAttemptID == "" {
		t.Fatalf("expected active attempt id on assigned job: %#v", assigned)
	}
	recordBody, _ := json.Marshal(map[string]any{
		"training_attempt_id":           activeAttemptID,
		"modal_function_call_object_id": "fc-active",
		"cancel_status":                 "active",
		"requested_gpu_type":            "T4",
		"effective_gpu_type":            "T4",
		"memory_mb":                     24576,
		"requested_batch_size":          16,
		"effective_batch_size":          16,
		"batch_size_policy":             "preserved",
		"modal_resource_signature":      "gpu=T4|batch=16|memory_mb=24576",
		"modal_resources": map[string]any{
			"effective_gpu_type":       "T4",
			"effective_batch_size":     16,
			"memory_mb":                24576,
			"modal_resource_signature": "gpu=T4|batch=16|memory_mb=24576",
		},
	})
	recordReq := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/modal-call", bytes.NewReader(recordBody))
	recordReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, recordReq, assigned)
	recordResp := httptest.NewRecorder()
	router.ServeHTTP(recordResp, recordReq)
	if recordResp.Code != http.StatusOK {
		t.Fatalf("expected modal call record status %d, got %d: %s", http.StatusOK, recordResp.Code, recordResp.Body.String())
	}

	cancelBody, _ := json.Marshal(map[string]any{
		"reason":                 "user_requested",
		"promote_best_available": true,
		"terminate_remote_work":  true,
	})
	cancelReq := httptest.NewRequest(http.MethodPost, "/plans/"+plan.ID+"/cancel-active-execution", bytes.NewReader(cancelBody))
	cancelReq.Header.Set("Content-Type", "application/json")
	cancelResp := httptest.NewRecorder()
	router.ServeHTTP(cancelResp, cancelReq)
	if cancelResp.Code != http.StatusOK {
		t.Fatalf("expected cancel status %d, got %d: %s", http.StatusOK, cancelResp.Code, cancelResp.Body.String())
	}
	var response cancelExecutionResponse
	if err := json.Unmarshal(cancelResp.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode cancel response: %v", err)
	}
	if response.Status != "CANCELLED_BY_USER" || response.QueuedJobsCancelled != 1 || response.ActiveJobsMarkedCancelling != 1 {
		t.Fatalf("unexpected cancel response: %#v", response)
	}
	if len(response.ModalCalls) != 2 {
		t.Fatalf("expected modal call rows for both open jobs, got %#v", response.ModalCalls)
	}
	var activeCall cancelModalCallResult
	for _, modalCall := range response.ModalCalls {
		if modalCall.JobID == assigned.ID {
			activeCall = modalCall
		}
	}
	if activeCall.ModalFunctionCallObjectID != "fc-active" || activeCall.CancelStatus != "cancel_requested" {
		t.Fatalf("expected active Modal call cancellation request, got %#v", activeCall)
	}
	if len(response.WorkerRequirements) != 1 || response.WorkerRequirements[0].Status != execution.WorkerRequirementCancelled {
		t.Fatalf("expected cancelled worker requirement, got %#v", response.WorkerRequirements)
	}
	if response.BestAvailableModel.Exportable || response.BestAvailableModel.Reason != "no_successful_completed_model" {
		t.Fatalf("expected no exportable model result, got %#v", response.BestAvailableModel)
	}

	latestAssigned, err := memoryStore.GetJob(assigned.ID)
	if err != nil {
		t.Fatalf("get assigned after cancel: %v", err)
	}
	cancelRequested, _ := latestAssigned.Config["cancel_requested"].(bool)
	if latestAssigned.Status != jobs.StatusFailed ||
		!cancelRequested ||
		jobConfigString(latestAssigned.Config, "failure_class") != "cancelled" ||
		jobConfigString(latestAssigned.Config, "modal_function_call_object_id") != "fc-active" {
		t.Fatalf("expected assigned job to be cancelled with Modal call metadata, got %#v", latestAssigned)
	}
	for _, job := range result.Jobs {
		updated, err := memoryStore.GetJob(job.ID)
		if err != nil {
			t.Fatalf("get job after cancel: %v", err)
		}
		if updated.Status != jobs.StatusFailed {
			t.Fatalf("expected cancelled job %s to be failed for compatibility, got %#v", updated.ID, updated)
		}
	}

	lateModalBody, _ := json.Marshal(map[string]any{
		"training_attempt_id":           activeAttemptID,
		"modal_function_call_object_id": "fc-late",
		"cancel_status":                 "active",
	})
	lateModalReq := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/modal-call", bytes.NewReader(lateModalBody))
	lateModalReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, lateModalReq, assigned)
	lateModalResp := httptest.NewRecorder()
	router.ServeHTTP(lateModalResp, lateModalReq)
	if lateModalResp.Code != http.StatusConflict {
		t.Fatalf("expected late modal call to be rejected with %d, got %d: %s", http.StatusConflict, lateModalResp.Code, lateModalResp.Body.String())
	}
	afterLateModal, err := memoryStore.GetJob(assigned.ID)
	if err != nil {
		t.Fatalf("get job after late modal callback: %v", err)
	}
	if jobConfigString(afterLateModal.Config, "modal_function_call_object_id") != "fc-active" {
		t.Fatalf("late modal callback overwrote active call id: %#v", afterLateModal.Config)
	}

	completeReq := httptest.NewRequest(
		http.MethodPost,
		"/jobs/"+assigned.ID+"/complete",
		strings.NewReader(`{"training_attempt_id":"`+activeAttemptID+`","mlflow_run_id":"stale"}`),
	)
	completeReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, completeReq, assigned)
	completeResp := httptest.NewRecorder()
	router.ServeHTTP(completeResp, completeReq)
	if completeResp.Code != http.StatusConflict {
		t.Fatalf("expected stale complete status %d, got %d: %s", http.StatusConflict, completeResp.Code, completeResp.Body.String())
	}
	afterComplete, err := memoryStore.GetJob(assigned.ID)
	if err != nil {
		t.Fatalf("get job after stale complete: %v", err)
	}
	if afterComplete.Status != jobs.StatusFailed || afterComplete.MLflowRunID == "stale" {
		t.Fatalf("stale completion overwrote cancelled job: %#v", afterComplete)
	}
	events, err := memoryStore.ListProjectExecutionEvents(project.ID, 20)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasExecutionEvent(events, execution.EventExecutionCancellationRequested) ||
		!hasExecutionEvent(events, execution.EventExecutionCancelled) ||
		!hasExecutionEvent(events, execution.EventJobStaleCallbackIgnored) {
		t.Fatalf("expected cancellation and stale callback events, got %#v", events)
	}
}

func TestPollJobFiltersModalProviderAndProfileFallback(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "modal dispatcher slot", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	if _, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "local",
	}); err != nil {
		t.Fatalf("create local job: %v", err)
	}
	modalJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "modal",
	})
	if err != nil {
		t.Fatalf("create modal job: %v", err)
	}

	assigned, err := memoryStore.PollJob(worker.ID, store.JobPollFilter{
		Provider:  "modal",
		Templates: []string{jobs.TemplateTrainExperiment},
	})
	if err != nil {
		t.Fatalf("poll modal job: %v", err)
	}
	if assigned.ID != modalJob.ID {
		t.Fatalf("expected modal job assignment, got %#v", assigned)
	}
	if _, err := memoryStore.CompleteJob(assigned.ID, ""); err != nil {
		t.Fatalf("complete modal job: %v", err)
	}

	profileJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateProfileDataset, map[string]any{
		"dataset_id": dataset.ID,
	})
	if err != nil {
		t.Fatalf("create profile job: %v", err)
	}
	assigned, err = memoryStore.PollJob(worker.ID, store.JobPollFilter{
		Provider:                            "modal",
		Templates:                           []string{jobs.TemplateProfileDataset},
		IncludeUnspecifiedProviderTemplates: []string{jobs.TemplateProfileDataset},
	})
	if err != nil {
		t.Fatalf("poll profile fallback job: %v", err)
	}
	if assigned.ID != profileJob.ID {
		t.Fatalf("expected profile fallback assignment, got %#v", assigned)
	}
}

func TestPollJobProviderFilterDefaultsToGenericProviderlessJobs(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "modal dispatcher slot", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	if _, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "local",
	}); err != nil {
		t.Fatalf("create local train job: %v", err)
	}
	profileJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateProfileDataset, map[string]any{
		"dataset_id": dataset.ID,
	})
	if err != nil {
		t.Fatalf("create profile job: %v", err)
	}

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodPost, "/workers/"+worker.ID+"/poll", strings.NewReader(`{"provider":"modal"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected poll status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	var payload pollJobResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode poll response: %v", err)
	}
	if payload.Job == nil || payload.Job.ID != profileJob.ID {
		t.Fatalf("expected provider-filtered poll to claim generic profile job, got %#v", payload.Job)
	}
}

func TestListDashboardEndpointsUsePaginationDefaults(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	var firstJob jobs.ExperimentJob
	for index := 0; index < 3; index++ {
		job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
			"dataset_id": dataset.ID,
			"provider":   "local",
			"model":      "model_" + strconv.Itoa(index),
		})
		if err != nil {
			t.Fatalf("create job: %v", err)
		}
		if index == 0 {
			firstJob = job
		}
		if _, err := memoryStore.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{Status: jobs.StatusQueued}); err != nil {
			t.Fatalf("upsert summary: %v", err)
		}
		if _, err := memoryStore.UpsertTrainingRunEvaluation(job.ID, runs.TrainingRunEvaluationUpdate{
			ObjectiveProfile: map[string]any{"score": index},
			PerClassMetrics:  map[string]any{"class": index},
			ConfusionMatrix:  [][]int{{index}},
		}); err != nil {
			t.Fatalf("upsert evaluation: %v", err)
		}
	}
	for epoch := 1; epoch <= 3; epoch++ {
		if _, err := memoryStore.ReportMetric(firstJob.ID, epoch, map[string]float64{"macro_f1": float64(epoch)}); err != nil {
			t.Fatalf("report metric: %v", err)
		}
	}

	router := NewRouter(memoryStore)
	assertPagedJobs(t, router, project.ID)
	assertPagedMetrics(t, router, firstJob.ID)
	assertPagedSummaries(t, router, project.ID)
	assertPagedEvaluations(t, router, project.ID)
}

func assertPagedJobs(t *testing.T, router http.Handler, projectID string) {
	t.Helper()
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/projects/"+projectID+"/jobs?limit=2", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("jobs status %d: %s", resp.Code, resp.Body.String())
	}
	var payload struct {
		Jobs       []jobs.ExperimentJob `json:"jobs"`
		Limit      int                  `json:"limit"`
		HasMore    bool                 `json:"has_more"`
		NextOffset int                  `json:"next_offset"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode jobs: %v", err)
	}
	if len(payload.Jobs) != 2 || payload.Limit != 2 || !payload.HasMore || payload.NextOffset != 2 {
		t.Fatalf("unexpected jobs page: %#v", payload)
	}
}

func assertPagedMetrics(t *testing.T, router http.Handler, jobID string) {
	t.Helper()
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/jobs/"+jobID+"/metrics?limit=2", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("metrics status %d: %s", resp.Code, resp.Body.String())
	}
	var payload struct {
		Metrics []jobs.EpochMetric `json:"metrics"`
		HasMore bool               `json:"has_more"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode metrics: %v", err)
	}
	if len(payload.Metrics) != 2 || payload.Metrics[0].Epoch != 2 || payload.Metrics[1].Epoch != 3 || !payload.HasMore {
		t.Fatalf("unexpected metrics page: %#v", payload)
	}
}

func assertPagedSummaries(t *testing.T, router http.Handler, projectID string) {
	t.Helper()
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/projects/"+projectID+"/training-run-summaries?limit=1", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("summaries status %d: %s", resp.Code, resp.Body.String())
	}
	var payload struct {
		Summaries []runs.TrainingRunSummary `json:"summaries"`
		HasMore   bool                      `json:"has_more"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode summaries: %v", err)
	}
	if len(payload.Summaries) != 1 || !payload.HasMore {
		t.Fatalf("unexpected summaries page: %#v", payload)
	}
}

func assertPagedEvaluations(t *testing.T, router http.Handler, projectID string) {
	t.Helper()
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/projects/"+projectID+"/training-run-evaluations?limit=1&compact=1", nil))
	if resp.Code != http.StatusOK {
		t.Fatalf("evaluations status %d: %s", resp.Code, resp.Body.String())
	}
	var payload struct {
		Evaluations []runs.TrainingRunEvaluation `json:"evaluations"`
		HasMore     bool                         `json:"has_more"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode evaluations: %v", err)
	}
	if len(payload.Evaluations) != 1 || !payload.HasMore {
		t.Fatalf("unexpected evaluations page: %#v", payload)
	}
}

func TestAutomaticReviewWaitDoesNotPersistDecision(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
		testExperiment("efficientnet_b0", 8),
	})
	recordTrainingSummary(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.55, 0.01)

	result, err := server.runAutomaticExperimentReview(projectID)
	if err != nil {
		t.Fatalf("run automatic review: %v", err)
	}
	if result.Decision != nil {
		t.Fatalf("expected no persisted WAIT decision, got %s", result.Decision.DecisionType)
	}
	if got := len(listAgentDecisions(t, server, projectID)); got != 0 {
		t.Fatalf("expected no decisions, got %d", got)
	}
}

func TestAutomaticReviewSchedulesFollowUpPlan(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")
	t.Setenv("MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS", "2")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	recordTrainingSummary(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.55, 0.01)

	result, err := server.runAutomaticExperimentReview(projectID)
	if err != nil {
		t.Fatalf("run automatic review: %v", err)
	}
	if result.Decision == nil || result.Decision.DecisionType != decisions.TypeAddExperiments {
		t.Fatalf("expected ADD_EXPERIMENTS decision, got %#v", result.Decision)
	}
	if result.FollowUpPlan == nil {
		t.Fatal("expected automatic follow-up plan")
	}
	if result.FollowUpPlan.SourceDecisionID != result.Decision.ID {
		t.Fatalf("expected follow-up source decision %s, got %s", result.Decision.ID, result.FollowUpPlan.SourceDecisionID)
	}

	_, err = server.runAutomaticExperimentReview(projectID)
	if err != nil {
		t.Fatalf("rerun automatic review: %v", err)
	}
	if got := len(listAgentDecisions(t, server, projectID)); got != 1 {
		t.Fatalf("expected one action decision after rerun, got %d", got)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 2 {
		t.Fatalf("expected initial plus one follow-up plan, got %d", got)
	}
}

func TestAutomaticReviewDeterministicFallbackAvoidsBestModelEpochLoop(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")
	t.Setenv("MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS", "3")

	best := testExperiment("efficientnet_b1", 12)
	best.Template = "efficientnet_transfer"
	control := testExperiment("resnet18", 10)
	control.Template = "resnet_transfer"
	control.LearningRate = 0.0002
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{best, control})
	createTerminalTrainingJob(t, server, plan, best, jobs.StatusSucceeded, 0.62)
	createTerminalTrainingJob(t, server, plan, control, jobs.StatusSucceeded, 0.58)

	result, err := server.runAutomaticExperimentReview(projectID)
	if err != nil {
		t.Fatalf("run automatic review: %v", err)
	}
	if result.FollowUpPlan == nil {
		t.Fatal("expected deterministic fallback to create a novel challenger plan")
	}
	for _, experiment := range result.FollowUpPlan.Experiments {
		model := strings.ToLower(experiment.Model)
		if model == "efficientnet_b1" || model == "resnet18" {
			t.Fatalf("deterministic fallback repeated low-value model %s in %#v", model, result.FollowUpPlan.Experiments)
		}
	}
	if len(result.Jobs) != 0 {
		t.Fatalf("expected no jobs with auto execution disabled, got %d", len(result.Jobs))
	}
}

func TestAutomaticReviewRespectsMaxFollowUpRounds(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")
	t.Setenv("MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS", "0")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	recordTrainingSummary(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.55, 0.01)

	result, err := server.runAutomaticExperimentReview(projectID)
	if err != nil {
		t.Fatalf("run automatic review: %v", err)
	}
	if result.Decision == nil || result.Decision.DecisionType != decisions.TypeAddExperiments {
		t.Fatalf("expected ADD_EXPERIMENTS decision, got %#v", result.Decision)
	}
	if result.FollowUpPlan != nil {
		t.Fatalf("expected max round guard to skip follow-up, got %s", result.FollowUpPlan.ID)
	}

	_, err = server.runAutomaticExperimentReview(projectID)
	if err != nil {
		t.Fatalf("rerun automatic review: %v", err)
	}
	if got := len(listAgentDecisions(t, server, projectID)); got != 1 {
		t.Fatalf("expected one deduplicated action decision, got %d", got)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected no automatic follow-up plans, got %d", got-1)
	}
}

func TestAutomaticReviewAutoExecutionCreatesWorkerRequirement(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "true")
	t.Setenv("MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER", "modal")
	t.Setenv("MODEL_EXPRESS_DEFAULT_GPU_TYPE", "T4")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	recordTrainingSummary(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.55, 0.01)

	result, err := server.runAutomaticExperimentReview(projectID)
	if err != nil {
		t.Fatalf("run automatic review: %v", err)
	}
	if result.FollowUpPlan == nil {
		t.Fatal("expected automatic follow-up plan")
	}
	if len(result.Jobs) != len(result.FollowUpPlan.Experiments) {
		t.Fatalf("expected one queued job per follow-up experiment, got %d jobs for %d experiments", len(result.Jobs), len(result.FollowUpPlan.Experiments))
	}

	requirements, err := server.store.ListProjectWorkerRequirements(projectID)
	if err != nil {
		t.Fatalf("list worker requirements: %v", err)
	}
	if len(requirements) != 1 {
		t.Fatalf("expected one worker requirement, got %d", len(requirements))
	}
	if requirements[0].PlanID != result.FollowUpPlan.ID {
		t.Fatalf("expected worker requirement for plan %s, got %s", result.FollowUpPlan.ID, requirements[0].PlanID)
	}
	if requirements[0].Status != execution.WorkerRequirementPending {
		t.Fatalf("expected pending worker requirement, got %s", requirements[0].Status)
	}

	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasExecutionEvent(events, execution.EventJobsQueued) {
		t.Fatalf("expected %s event, got %#v", execution.EventJobsQueued, events)
	}
	if !hasExecutionEvent(events, execution.EventWorkersRequired) {
		t.Fatalf("expected %s event, got %#v", execution.EventWorkersRequired, events)
	}
}

func TestAutomaticExecutionWorkerRequirementScalesToQueuedJobs(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_PROFILE", "fast-remote")
	t.Setenv("MODEL_EXPRESS_MAX_AUTO_WORKERS", "4")

	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("vision project", "fast live classifier")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	checksum := strings.Repeat("a", 64)
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", checksum, 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	experiments := []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
		testExperiment("efficientnet_b0", 6),
		testExperiment("regnet_y_400mf", 6),
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 3, 10, experiments, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	executionResult, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	if err := server.recordAutomaticExecutionQueued(plan, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"}, executionResult.Jobs); err != nil {
		t.Fatalf("record automatic execution queued: %v", err)
	}

	requirements, err := server.store.ListProjectWorkerRequirements(project.ID)
	if err != nil {
		t.Fatalf("list requirements: %v", err)
	}
	if len(requirements) != 1 {
		t.Fatalf("expected one worker requirement, got %d", len(requirements))
	}
	if requirements[0].TargetCount != 3 {
		t.Fatalf("expected target count to scale to 3 queued jobs, got %d", requirements[0].TargetCount)
	}
	if requirements[0].Status != execution.WorkerRequirementPending {
		t.Fatalf("expected pending requirement, got %s", requirements[0].Status)
	}
	if requirements[0].DatasetID != dataset.ID || requirements[0].DatasetChecksum != checksum {
		t.Fatalf("expected dataset materialization identity on requirement, got %#v", requirements[0])
	}
	expectedCacheKey := "sha256-" + checksum
	if requirements[0].DatasetCacheKey != expectedCacheKey {
		t.Fatalf("expected checksum cache key, got %q", requirements[0].DatasetCacheKey)
	}
	if requirements[0].ColdCachePolicy != execution.ColdCachePolicySingleMaterialization || requirements[0].MaxColdDatasetMaterializations != 1 {
		t.Fatalf("expected modal cold cache policy on requirement, got %#v", requirements[0])
	}
	materialization, ok := executionResult.Jobs[0].Config["dataset_materialization"].(map[string]any)
	if !ok {
		t.Fatalf("expected dataset materialization config, got %#v", executionResult.Jobs[0].Config)
	}
	if materialization["dataset_cache_key"] != expectedCacheKey || materialization["cold_cache_policy"] != execution.ColdCachePolicySingleMaterialization {
		t.Fatalf("unexpected dataset materialization config: %#v", materialization)
	}
}

func TestManualModalExecutionRequestedConcurrencyOverridesBalancedDefault(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_PROFILE", "fast-remote")
	t.Setenv("MODEL_EXPRESS_COST_MODES", "1")
	t.Setenv("MODEL_EXPRESS_MAX_AUTO_WORKERS", "8")

	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	current := server.currentAutomationSettings()
	current.CostMode = "balanced"
	server.setAutomationSettings(current)

	project, err := memoryStore.CreateProject("vision project", "fast live classifier")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", strings.Repeat("a", 64), 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 4, 10, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
		testExperiment("efficientnet_b0", 6),
		testExperiment("regnet_y_400mf", 6),
		testExperiment("convnext_tiny", 6),
	}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	result, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{
		Provider:          "modal",
		GPUType:           "T4",
		MaxConcurrentJobs: 4,
	})
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	if result.WorkerRequirement == nil {
		t.Fatal("expected worker requirement in execute response")
	}
	if result.WorkerRequirement.TargetCount != 4 || result.WorkerRequirement.MaxConcurrentJobs != 4 {
		t.Fatalf("expected explicit concurrency 4 to drive requirement, got %#v", result.WorkerRequirement)
	}
	requirements, err := memoryStore.ListProjectWorkerRequirements(project.ID)
	if err != nil {
		t.Fatalf("list requirements: %v", err)
	}
	if len(requirements) != 1 || requirements[0].TargetCount != 4 || requirements[0].MaxConcurrentJobs != 4 {
		t.Fatalf("expected persisted requirement concurrency 4, got %#v", requirements)
	}
}

func TestExecuteModalPlanAddsPreviewTrainingTierBehindFlag(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_MODAL_PREVIEW_TIER_METADATA", "1")

	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("vision project", "fast live classifier")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", strings.Repeat("a", 64), 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 2, 10, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
		testExperiment("efficientnet_b0", 6),
	}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	executionResult, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}

	if len(executionResult.Jobs) != 2 {
		t.Fatalf("expected two jobs, got %d", len(executionResult.Jobs))
	}
	for _, job := range executionResult.Jobs {
		if got := configString(job.Config, "training_tier"); got != "preview" {
			t.Fatalf("expected preview training tier for job %s, got %q in %#v", job.ID, got, job.Config)
		}
	}
}

func TestExecuteModalPlanLeavesTrainingTierUnsetWhenFlagOff(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_MODAL_PREVIEW_TIER_METADATA", "0")

	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("vision project", "fast live classifier")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", strings.Repeat("a", 64), 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 1, 10, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	executionResult, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}

	if got := configString(executionResult.Jobs[0].Config, "training_tier"); got != "" {
		t.Fatalf("expected no training tier when flag is off, got %q", got)
	}
}

func TestCostModePolicyConstrainsWorkerRequirementAndJobConfig(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_PROFILE", "fast-remote")
	t.Setenv("MODEL_EXPRESS_COST_MODES", "1")

	tests := []struct {
		mode                string
		expectedMode        string
		expectedConcurrency int
		expectedJobs        int
	}{
		{"prototype", "prototype", 1, 3},
		{"balanced", "balanced", 2, 4},
		{"quality", "quality", 4, 4},
	}
	for _, tc := range tests {
		t.Run(tc.mode, func(t *testing.T) {
			memoryStore := store.NewMemoryStore()
			server := newServer(memoryStore)
			current := server.currentAutomationSettings()
			current.CostMode = tc.mode
			server.setAutomationSettings(current)
			project, err := memoryStore.CreateProject("vision project", "fast live classifier")
			if err != nil {
				t.Fatalf("create project: %v", err)
			}
			dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", strings.Repeat("a", 64), 0)
			if err != nil {
				t.Fatalf("create dataset: %v", err)
			}
			plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 4, 10, []plans.PlannedExperiment{
				testExperiment("mobilenet_v3_small", 6),
				testExperiment("efficientnet_b0", 6),
				testExperiment("regnet_y_400mf", 6),
				testExperiment("convnext_tiny", 6),
			}, nil, "")
			if err != nil {
				t.Fatalf("create plan: %v", err)
			}
			result, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
			if err != nil {
				t.Fatalf("execute plan: %v", err)
			}
			if len(result.Jobs) != tc.expectedJobs {
				t.Fatalf("expected %d jobs, got %d cost policy %#v", tc.expectedJobs, len(result.Jobs), result.CostPolicy)
			}
			if result.CostPolicy["cost_mode"] != tc.expectedMode || int(result.CostPolicy["max_concurrent_jobs"].(int)) != tc.expectedConcurrency {
				t.Fatalf("unexpected cost policy: %#v", result.CostPolicy)
			}
			if err := server.recordAutomaticExecutionQueued(plan, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"}, result.Jobs); err != nil {
				t.Fatalf("record automatic execution queued: %v", err)
			}
			requirements, err := memoryStore.ListProjectWorkerRequirements(project.ID)
			if err != nil {
				t.Fatalf("list requirements: %v", err)
			}
			if len(requirements) != 1 || requirements[0].TargetCount != tc.expectedConcurrency || requirements[0].MaxConcurrentJobs != tc.expectedConcurrency {
				t.Fatalf("expected capped requirement concurrency %d, got %#v", tc.expectedConcurrency, requirements)
			}
			if got := configString(result.Jobs[0].Config, "cost_mode"); got != tc.expectedMode {
				t.Fatalf("expected job cost mode %s, got %q", tc.expectedMode, got)
			}
		})
	}
}

func TestBudgetCapBlocksQueuedFullTrainClearly(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_COST_MODES", "1")

	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	current := server.currentAutomationSettings()
	current.CostMode = "balanced"
	current.BudgetCapUSD = 1.0
	server.setAutomationSettings(current)
	project, err := memoryStore.CreateProject("vision project", "fast live classifier")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", strings.Repeat("b", 64), 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	priorJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID, "provider": "modal"})
	if err != nil {
		t.Fatalf("create prior job: %v", err)
	}
	cost := 1.0
	if _, err := memoryStore.UpsertTrainingRunSummary(priorJob.ID, runs.TrainingRunSummaryUpdate{EstimatedCostUSD: &cost, Status: jobs.StatusSucceeded}); err != nil {
		t.Fatalf("seed prior cost: %v", err)
	}
	full := testExperiment("efficientnet_b0", 8)
	full.Strategy = "full_train_promoted_candidate"
	preview := testExperiment("mobilenet_v3_small", 3)
	preview.Strategy = "preview"
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 2, 10, []plans.PlannedExperiment{preview, full}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	result, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	if len(result.Jobs) != 1 {
		t.Fatalf("expected only preview job under budget cap, got %d jobs", len(result.Jobs))
	}
	if got := configString(result.Jobs[0].Config, "training_tier"); got != "preview" {
		t.Fatalf("expected queued preview job, got tier %q", got)
	}
	skipped, ok := result.CostPolicy["skipped"].([]map[string]any)
	if !ok || len(skipped) != 1 || skipped[0]["reason"] != "budget_cap_reached" {
		t.Fatalf("expected budget skip payload, got %#v", result.CostPolicy)
	}
	events, err := memoryStore.ListProjectExecutionEvents(project.ID, 10)
	if err != nil {
		t.Fatalf("list events: %v", err)
	}
	if !executionEventsContainType(events, execution.EventCostBudgetBlocked) {
		t.Fatalf("expected budget blocked event, got %#v", events)
	}
}

func executionEventsContainType(events []execution.ExecutionEvent, eventType string) bool {
	for _, event := range events {
		if event.EventType == eventType {
			return true
		}
	}
	return false
}

func TestBudgetCapSelectsBestAvailableChampionWhenAllQueuedFullTrainIsBlocked(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_COST_MODES", "1")

	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	current := server.currentAutomationSettings()
	current.CostMode = "balanced"
	current.BudgetCapUSD = 1.0
	server.setAutomationSettings(current)
	project, err := memoryStore.CreateProject("vision project", "fast live classifier")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", strings.Repeat("d", 64), 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	priorJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"provider":   "modal",
		"model":      "mobilenet_v3_small",
	})
	if err != nil {
		t.Fatalf("create prior job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(priorJob.ID, ""); err != nil {
		t.Fatalf("complete prior job: %v", err)
	}
	cost := 1.0
	score := 0.74
	if _, err := memoryStore.UpsertTrainingRunSummary(priorJob.ID, runs.TrainingRunSummaryUpdate{
		Status:           jobs.StatusSucceeded,
		EstimatedCostUSD: &cost,
		BestMacroF1:      &score,
		BestAccuracy:     &score,
	}); err != nil {
		t.Fatalf("seed prior summary: %v", err)
	}
	if _, err := memoryStore.UpsertTrainingRunEvaluation(priorJob.ID, runs.TrainingRunEvaluationUpdate{
		ModelProfile: map[string]any{
			"artifact_uri":  "s3://model-express/model-express/artifacts/prior/model.onnx",
			"export_status": "ready",
		},
	}); err != nil {
		t.Fatalf("seed prior evaluation: %v", err)
	}
	full := testExperiment("efficientnet_b0", 8)
	full.Strategy = "full_train_promoted_candidate"
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 1, 10, []plans.PlannedExperiment{full}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	result, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	if len(result.Jobs) != 0 {
		t.Fatalf("expected no queued jobs under exhausted budget, got %d", len(result.Jobs))
	}
	champion, err := memoryStore.GetProjectChampion(project.ID)
	if err != nil {
		t.Fatalf("expected budget-stop champion: %v", err)
	}
	if champion.JobID != priorJob.ID {
		t.Fatalf("expected prior job champion %s, got %s", priorJob.ID, champion.JobID)
	}
	if champion.SourceDecisionID == "" {
		t.Fatalf("expected champion source decision")
	}
	exports, err := memoryStore.ListProjectChampionExports(project.ID)
	if err != nil {
		t.Fatalf("list champion exports: %v", err)
	}
	if len(exports) != 1 || exports[0].Status != runs.ChampionExportStatusReady || exports[0].ArtifactURI == "" {
		t.Fatalf("expected ready champion export, got %#v", exports)
	}
}

func TestCostPolicySkippedFullTrainSelectsChampionAfterAllowedJobsFinish(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_COST_MODES", "1")

	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	current := server.currentAutomationSettings()
	current.CostMode = "balanced"
	current.BudgetCapUSD = 1.0
	server.setAutomationSettings(current)
	project, err := memoryStore.CreateProject("vision project", "fast live classifier")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", strings.Repeat("e", 64), 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	priorJob, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID, "provider": "modal", "model": "mobilenet_v3_small"})
	if err != nil {
		t.Fatalf("create prior job: %v", err)
	}
	if _, err := memoryStore.CompleteJob(priorJob.ID, ""); err != nil {
		t.Fatalf("complete prior job: %v", err)
	}
	priorCost := 1.0
	priorScore := 0.60
	if _, err := memoryStore.UpsertTrainingRunSummary(priorJob.ID, runs.TrainingRunSummaryUpdate{
		Status:           jobs.StatusSucceeded,
		EstimatedCostUSD: &priorCost,
		BestMacroF1:      &priorScore,
		BestAccuracy:     &priorScore,
	}); err != nil {
		t.Fatalf("seed prior summary: %v", err)
	}
	preview := testExperiment("mobilenet_v3_small", 3)
	preview.Strategy = "preview"
	full := testExperiment("efficientnet_b0", 8)
	full.Strategy = "full_train_promoted_candidate"
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 2, 10, []plans.PlannedExperiment{preview, full}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	result, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	if len(result.Jobs) != 1 {
		t.Fatalf("expected only preview job queued, got %d", len(result.Jobs))
	}
	if _, err := memoryStore.GetProjectChampion(project.ID); !errors.Is(err, store.ErrNotFound) {
		t.Fatalf("expected no champion while preview job is still open, got err %v", err)
	}
	previewJob := result.Jobs[0]
	if _, err := memoryStore.CompleteJob(previewJob.ID, ""); err != nil {
		t.Fatalf("complete preview job: %v", err)
	}
	previewScore := 0.82
	previewCost := 0.20
	if _, err := memoryStore.UpsertTrainingRunSummary(previewJob.ID, runs.TrainingRunSummaryUpdate{
		Status:           jobs.StatusSucceeded,
		EstimatedCostUSD: &previewCost,
		BestMacroF1:      &previewScore,
		BestAccuracy:     &previewScore,
	}); err != nil {
		t.Fatalf("upsert preview summary: %v", err)
	}
	if _, err := memoryStore.UpsertTrainingRunEvaluation(previewJob.ID, runs.TrainingRunEvaluationUpdate{
		ModelProfile: map[string]any{
			"artifact_uri":  "s3://model-express/model-express/artifacts/preview/model.onnx",
			"export_status": "ready",
		},
	}); err != nil {
		t.Fatalf("upsert preview evaluation: %v", err)
	}

	selected, err := server.selectBestAvailableChampionIfCostStoppedAfterTrainingJob(previewJob)
	if err != nil {
		t.Fatalf("select budget-stop champion: %v", err)
	}
	if !selected {
		t.Fatalf("expected budget-stop champion selection after allowed jobs finished")
	}
	champion, err := memoryStore.GetProjectChampion(project.ID)
	if err != nil {
		t.Fatalf("expected champion: %v", err)
	}
	if champion.JobID != previewJob.ID {
		t.Fatalf("expected preview job champion %s, got %s", previewJob.ID, champion.JobID)
	}
	exports, err := memoryStore.ListProjectChampionExports(project.ID)
	if err != nil {
		t.Fatalf("list champion exports: %v", err)
	}
	if len(exports) != 1 || exports[0].Status != runs.ChampionExportStatusReady || exports[0].ArtifactURI != "s3://model-express/model-express/artifacts/preview/model.onnx" {
		t.Fatalf("expected ready preview export, got %#v", exports)
	}
}

func TestPersistentGPUProviderSchedulesOnlyWhenConfigured(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("vision project", "fast live classifier")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	checksum := strings.Repeat("c", 64)
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", checksum, 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 2, 10, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
		testExperiment("efficientnet_b0", 6),
	}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}
	if _, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "persistent_gpu", GPUType: "A10G"}); err == nil {
		t.Fatal("expected persistent provider to require configuration")
	}

	t.Setenv("MODEL_EXPRESS_PERSISTENT_GPU_PROVIDER", "1")
	t.Setenv("MODEL_EXPRESS_PERSISTENT_GPU_CACHE_ROOT", "/mnt/model-express-cache")
	result, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "persistent-disk", GPUType: "A10G"})
	if err != nil {
		t.Fatalf("execute persistent plan: %v", err)
	}
	if len(result.Jobs) != 2 {
		t.Fatalf("expected two persistent jobs, got %d", len(result.Jobs))
	}
	config := result.Jobs[0].Config
	if got := configString(config, "provider"); got != "persistent_gpu" {
		t.Fatalf("expected normalized persistent provider, got %q", got)
	}
	materialization := config["dataset_materialization"].(map[string]any)
	if materialization["dataset_cache_key"] != "sha256-"+checksum || materialization["cold_cache_policy"] != execution.ColdCachePolicySingleMaterialization {
		t.Fatalf("unexpected materialization policy: %#v", materialization)
	}
	persistentConfig := config["persistent_gpu"].(map[string]any)
	if persistentConfig["cache_root"] != "/mnt/model-express-cache" || persistentConfig["materialization_status"] != execution.DatasetMaterializationCold {
		t.Fatalf("unexpected persistent config: %#v", persistentConfig)
	}
	if err := server.recordAutomaticExecutionQueued(plan, executeExperimentPlanRequest{Provider: "persistent_gpu", GPUType: "A10G"}, result.Jobs); err != nil {
		t.Fatalf("record automatic execution queued: %v", err)
	}
	requirements, err := memoryStore.ListProjectWorkerRequirements(project.ID)
	if err != nil {
		t.Fatalf("list requirements: %v", err)
	}
	if len(requirements) != 1 || requirements[0].Provider != "persistent_gpu" || requirements[0].DatasetCacheKey != "sha256-"+checksum {
		t.Fatalf("expected persistent requirement metadata, got %#v", requirements)
	}
}

func TestFullPlanMockedIntegrationCostAwarePersistentProvider(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_PROFILE", "fast-remote")
	t.Setenv("MODEL_EXPRESS_COST_MODES", "1")
	t.Setenv("MODEL_EXPRESS_PERSISTENT_GPU_PROVIDER", "1")
	t.Setenv("MODEL_EXPRESS_PERSISTENT_GPU_CACHE_ROOT", "/mnt/model-express-cache")

	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	current := server.currentAutomationSettings()
	current.CostMode = "quality"
	current.BudgetCapUSD = 5
	current.DefaultTrainingProvider = "persistent_gpu"
	current.DefaultGPUType = "A10G"
	server.setAutomationSettings(current)

	project, err := memoryStore.CreateProject("vision project", "cost aware remote training")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	checksum := strings.Repeat("e", 64)
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "s3://bucket/dataset.zip", checksum, 1024)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	preview := testExperiment("mobilenet_v3_small", 3)
	preview.Strategy = "preview"
	full := testExperiment("efficientnet_b0", 8)
	full.Strategy = "full_train_promoted_candidate"
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 4, 12, []plans.PlannedExperiment{preview, full}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	result, err := server.executeStoredExperimentPlan(plan.ID, server.defaultExecuteExperimentPlanRequest())
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	if len(result.Jobs) != 2 {
		t.Fatalf("expected preview and full jobs, got %d", len(result.Jobs))
	}
	if result.CostPolicy["cost_mode"] != "quality" || result.CostPolicy["max_concurrent_jobs"] != 4 {
		t.Fatalf("unexpected cost policy payload: %#v", result.CostPolicy)
	}
	cacheKey := "sha256-" + checksum
	for _, job := range result.Jobs {
		if configString(job.Config, "provider") != "persistent_gpu" {
			t.Fatalf("expected persistent provider job, got %#v", job.Config)
		}
		materialization := job.Config["dataset_materialization"].(map[string]any)
		if materialization["dataset_cache_key"] != cacheKey {
			t.Fatalf("expected checksum cache key %s, got %#v", cacheKey, materialization)
		}
		persistentConfig := job.Config["persistent_gpu"].(map[string]any)
		if persistentConfig["cache_root"] != "/mnt/model-express-cache" || persistentConfig["dataset_cache_key"] != cacheKey {
			t.Fatalf("unexpected persistent config: %#v", persistentConfig)
		}
	}
	if err := server.recordAutomaticExecutionQueued(plan, server.defaultExecuteExperimentPlanRequest(), result.Jobs); err != nil {
		t.Fatalf("record automatic execution queued: %v", err)
	}
	requirements, err := memoryStore.ListProjectWorkerRequirements(project.ID)
	if err != nil {
		t.Fatalf("list requirements: %v", err)
	}
	if len(requirements) != 1 || requirements[0].Provider != "persistent_gpu" || requirements[0].MaxConcurrentJobs != 2 {
		t.Fatalf("expected persistent quality requirement, got %#v", requirements)
	}

	runtime := 12.0
	cost := 0.42
	epochs := 3
	reported, err := memoryStore.UpsertTrainingRunSummary(result.Jobs[0].ID, runs.TrainingRunSummaryUpdate{
		Provider:         "persistent_gpu",
		GPUType:          "A10G",
		Status:           jobs.StatusSucceeded,
		RuntimeSeconds:   &runtime,
		EstimatedCostUSD: &cost,
		EpochsCompleted:  &epochs,
		DatasetMaterialization: map[string]any{
			"dataset_materialization_cache_key":        cacheKey,
			"dataset_materialization_cache_hit":        false,
			"dataset_materialization_cache_miss":       true,
			"dataset_materialization_cache_root":       "/mnt/model-express-cache",
			"dataset_materialization_cache_scope":      "persistent_disk",
			"dataset_materialization_total_seconds":    2.5,
			"dataset_materialization_download_seconds": 1.1,
			"dataset_materialization_extract_seconds":  1.2,
		},
		StageTelemetry: map[string]any{
			"schema_version":                  "remote_gpu_stage_telemetry_v1",
			"dataset_materialization_seconds": 2.5,
			"active_training_seconds":         8.0,
			"evaluation_seconds":              1.0,
			"export_seconds":                  0.0,
			"queue_wait_seconds":              0.5,
			"idle_wait_seconds":               0.0,
		},
	})
	if err != nil {
		t.Fatalf("upsert training summary: %v", err)
	}
	if reported.Provider != "persistent_gpu" || reported.DatasetMaterialization["dataset_materialization_cache_key"] != cacheKey {
		t.Fatalf("expected provider/materialization summary, got %#v", reported)
	}
	if reported.StageTelemetry["active_training_seconds"] != 8.0 || reported.StageTelemetry["queue_wait_seconds"] != 0.5 {
		t.Fatalf("expected stage telemetry, got %#v", reported.StageTelemetry)
	}
}

func TestTerminalTrainingJobsSatisfyWorkerRequirementWhenPlanHasNoOpenJobs(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("vision project", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", "abc123", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 2, 10, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
		testExperiment("efficientnet_b0", 6),
	}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}
	executionResult, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	requirement, _, err := memoryStore.UpsertWorkerRequirement(project.ID, plan.ID, "modal", "T4", 2, "test", execution.WorkerRequirementPolicy{})
	if err != nil {
		t.Fatalf("upsert requirement: %v", err)
	}
	active := execution.WorkerRequirementActive
	if _, err := memoryStore.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{Status: &active}); err != nil {
		t.Fatalf("mark requirement active: %v", err)
	}

	if _, err := memoryStore.CompleteJob(executionResult.Jobs[0].ID, ""); err != nil {
		t.Fatalf("complete first job: %v", err)
	}
	server.updateWorkerRequirementDemandAfterTerminalJob(executionResult.Jobs[0])
	requirements, err := memoryStore.ListProjectWorkerRequirements(project.ID)
	if err != nil {
		t.Fatalf("list requirements after first completion: %v", err)
	}
	if requirements[0].Status != execution.WorkerRequirementActive {
		t.Fatalf("expected requirement to remain active while second job is open, got %s", requirements[0].Status)
	}

	if _, err := memoryStore.CompleteJob(executionResult.Jobs[1].ID, ""); err != nil {
		t.Fatalf("complete second job: %v", err)
	}
	server.updateWorkerRequirementDemandAfterTerminalJob(executionResult.Jobs[1])
	requirements, err = memoryStore.ListProjectWorkerRequirements(project.ID)
	if err != nil {
		t.Fatalf("list requirements after second completion: %v", err)
	}
	if requirements[0].Status != execution.WorkerRequirementSatisfied {
		t.Fatalf("expected satisfied requirement after all jobs terminal, got %s", requirements[0].Status)
	}
}

func TestModalWorkerRequirementIgnoresLocalWorkerCapacity(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("vision project", "fast live classifier")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", "abc123", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	if _, err := memoryStore.RegisterWorker(project.ID, "local worker", "local"); err != nil {
		t.Fatalf("register local worker: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", 1, 10, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	}, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	executionResult, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	if err := server.recordAutomaticExecutionQueued(plan, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"}, executionResult.Jobs); err != nil {
		t.Fatalf("record automatic execution queued: %v", err)
	}

	requirements, err := server.store.ListProjectWorkerRequirements(project.ID)
	if err != nil {
		t.Fatalf("list requirements: %v", err)
	}
	if len(requirements) != 1 {
		t.Fatalf("expected one worker requirement, got %d", len(requirements))
	}
	if requirements[0].Status != execution.WorkerRequirementPending {
		t.Fatalf("expected modal requirement to stay pending with only local workers active, got %s", requirements[0].Status)
	}
	if _, err := memoryStore.RegisterWorker(project.ID, "modal worker", "modal"); err != nil {
		t.Fatalf("register modal worker: %v", err)
	}
	if err := server.recordAutomaticExecutionQueued(plan, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"}, executionResult.Jobs); err != nil {
		t.Fatalf("record automatic execution queued after modal worker: %v", err)
	}
	requirements, err = server.store.ListProjectWorkerRequirements(project.ID)
	if err != nil {
		t.Fatalf("list requirements after modal worker: %v", err)
	}
	if requirements[0].Status != execution.WorkerRequirementActive {
		t.Fatalf("expected modal requirement to become active with modal worker, got %s", requirements[0].Status)
	}
}

func TestWorkerRequirementResetsPendingWhenTargetChanges(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	requirement, _, err := server.store.UpsertWorkerRequirement(projectID, plan.ID, "modal", "T4", 1, "test", execution.WorkerRequirementPolicy{})
	if err != nil {
		t.Fatalf("upsert worker requirement: %v", err)
	}
	active := execution.WorkerRequirementActive
	if _, err := server.store.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{Status: &active}); err != nil {
		t.Fatalf("mark worker requirement active: %v", err)
	}
	updated, _, err := server.store.UpsertWorkerRequirement(projectID, plan.ID, "modal", "T4", 2, "test", execution.WorkerRequirementPolicy{})
	if err != nil {
		t.Fatalf("upsert changed worker requirement: %v", err)
	}
	if updated.Status != execution.WorkerRequirementPending {
		t.Fatalf("expected target change to reopen requirement as pending, got %s", updated.Status)
	}
}

func TestAgentMemoryRecordsPersistAndFilter(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})

	invocation, err := server.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:         projectID,
		DatasetID:         plan.DatasetID,
		PlanID:            plan.ID,
		JobID:             "job_1",
		AgentName:         agents.TrainingMonitorAgentName,
		AgentVersion:      agents.TrainingMonitorAgentVersion,
		PromptVersion:     agents.TrainingMonitorPromptVersion,
		Provider:          "openai",
		Model:             "test-model",
		InputMessages:     []map[string]string{{"role": "user", "content": "evaluate this run"}},
		InputContext:      map[string]any{"job_id": "job_1"},
		RawOutput:         `{"summary":"stable"}`,
		ParsedOutput:      map[string]any{"summary": "stable"},
		ValidationStatus:  memory.InvocationValidationValid,
		AcceptedForMemory: true,
	})
	if err != nil {
		t.Fatalf("create invocation: %v", err)
	}

	record, err := server.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		InvocationID: invocation.ID,
		ProjectID:    projectID,
		DatasetID:    plan.DatasetID,
		PlanID:       plan.ID,
		JobID:        "job_1",
		AgentName:    agents.TrainingMonitorAgentName,
		Kind:         memory.KindTrainingEvaluation,
		Summary:      "The run is stable enough to rank.",
		Payload:      map[string]any{"rank_score": 0.72},
		Tags:         []string{"stable", "mobilenet"},
	})
	if err != nil {
		t.Fatalf("create memory record: %v", err)
	}

	records, err := server.store.ListProjectAgentMemoryRecords(projectID, memory.AgentMemoryFilter{
		DatasetID: plan.DatasetID,
		Kind:      memory.KindTrainingEvaluation,
		Limit:     5,
	})
	if err != nil {
		t.Fatalf("list memory records: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("expected one memory record, got %d", len(records))
	}
	if records[0].ID != record.ID {
		t.Fatalf("expected record %s, got %s", record.ID, records[0].ID)
	}
	if records[0].InvocationID != invocation.ID {
		t.Fatalf("expected memory record to link invocation %s, got %s", invocation.ID, records[0].InvocationID)
	}

	invocations, err := server.store.ListProjectAgentInvocations(projectID, memory.AgentInvocationFilter{
		DatasetID: plan.DatasetID,
		AgentName: agents.TrainingMonitorAgentName,
		Limit:     5,
	})
	if err != nil {
		t.Fatalf("list invocations: %v", err)
	}
	if len(invocations) != 1 {
		t.Fatalf("expected one invocation, got %d", len(invocations))
	}
	if invocations[0].ID != invocation.ID {
		t.Fatalf("expected invocation %s, got %s", invocation.ID, invocations[0].ID)
	}
}

func TestProjectObjectiveContextRecognizesLiveGoal(t *testing.T) {
	objective := projectObjectiveContext("Need accurate and fast live image classification on an edge device.")
	if objective.PrimaryObjective != "low_latency_live_service" {
		t.Fatalf("expected low latency live objective, got %s", objective.PrimaryObjective)
	}
	if objective.RankingWeights["latency"] > 0.1 {
		t.Fatalf("expected latency to be relaxed into a budget/tiebreaker, got %.3f", objective.RankingWeights["latency"])
	}
	if objective.RankingWeights["macro_f1"] <= objective.RankingWeights["latency"] {
		t.Fatalf("expected quality to dominate latency under live budget, got weights %#v", objective.RankingWeights)
	}
	if !strings.Contains(strings.Join(objective.DeploymentPriorities, " "), "25ms") {
		t.Fatalf("expected live latency budget guidance, got %#v", objective.DeploymentPriorities)
	}
	if len(objective.MetricPreferences) < 2 {
		t.Fatalf("expected multiple metric preferences, got %#v", objective.MetricPreferences)
	}
}

func TestValidatePlannedExperimentRejectsUnsupportedModel(t *testing.T) {
	experiment := testExperiment("unknown_mega_model", 6)
	err := validatePlannedExperiment(experiment, 0)
	if err == nil {
		t.Fatal("expected unsupported model validation error")
	}
}

func TestYOLODatasetAddsDetectorModelCatalogOptions(t *testing.T) {
	dataset := datasets.Dataset{
		Profile: map[string]any{
			"task_type": "object_detection",
			"metadata_summary": map[string]any{
				"yolo_available": true,
				"artifact_counts": map[string]any{
					"yolo_dataset_config": 1,
					"yolo_label_file":     12,
				},
			},
			"dataset_traits": []any{"yolo_format", "object_detection"},
		},
	}

	catalog := supportedModelCatalogForDataset(dataset, map[string]any{})

	yoloOptions := 0
	for _, spec := range catalog {
		if spec.Family == "yolo11" {
			yoloOptions++
			if spec.TaskType != "object_detection" {
				t.Fatalf("expected object_detection task type for %#v", spec)
			}
			if spec.ModelKind != "ultralytics_yolo_detector" {
				t.Fatalf("expected yolo detector model kind for %#v", spec)
			}
			if !spec.TrainingEnabled {
				t.Fatalf("yolo detector catalog options should be schedulable now that worker support is enabled: %#v", spec)
			}
		}
	}
	if yoloOptions < 2 {
		t.Fatalf("expected YOLO detector catalog options, got %d in %#v", yoloOptions, catalog)
	}
}

func TestBBoxMetadataAloneDoesNotAddYOLOCatalogOptions(t *testing.T) {
	dataset := datasets.Dataset{
		Profile: map[string]any{
			"task_type": "image_classification",
			"metadata_summary": map[string]any{
				"bbox_available":              true,
				"object_detection_available":  true,
				"bbox_annotation_count":       42,
				"yolo_available":              false,
				"yolo_dataset_config_present": false,
			},
			"dataset_traits": []any{"bbox_available"},
		},
	}

	catalog := supportedModelCatalogForDataset(dataset, map[string]any{})

	for _, spec := range catalog {
		if spec.Family == "yolo11" {
			t.Fatalf("did not expect YOLO detector option for bbox-only metadata: %#v", spec)
		}
	}
}

func TestDatasetPlanningInsightsUsesDetectionMetricsForYOLO(t *testing.T) {
	dataset := datasets.Dataset{
		Profile: map[string]any{
			"task_type":    "object_detection",
			"total_images": 120,
			"class_count":  2,
			"metadata_summary": map[string]any{
				"yolo_available": true,
				"yolo_summary": map[string]any{
					"available":        true,
					"label_file_count": 120,
					"bbox_count":       180,
				},
			},
			"dataset_traits": []any{"yolo_format", "object_detection"},
		},
	}

	insights := datasetPlanningInsights(dataset, map[string]any{})

	if insights.TaskType != "object_detection" {
		t.Fatalf("expected object_detection insights, got %s", insights.TaskType)
	}
	metrics := strings.Join(insights.RecommendedMetrics, " ")
	for _, expected := range []string{"mAP50_95", "box_loss", "cls_loss", "dfl_loss", "latency_p95_ms"} {
		if !strings.Contains(metrics, expected) {
			t.Fatalf("expected detection metric %s in %#v", expected, insights.RecommendedMetrics)
		}
	}
	if !strings.Contains(strings.Join(insights.Constraints, " "), "YOLO object-detection files detected") {
		t.Fatalf("expected YOLO detection constraint, got %#v", insights.Constraints)
	}
}

func TestValidatePlannedExperimentAcceptsYOLODetectorWorkerModels(t *testing.T) {
	experiment := testExperiment("yolo11n.pt", 8)
	experiment.Template = "yolo11_detection"

	err := validatePlannedExperiment(experiment, 0)

	if err != nil {
		t.Fatalf("expected detector-worker model to validate: %v", err)
	}
}

func TestValidatePlannedExperimentAcceptsStructuredPreprocessing(t *testing.T) {
	experiment := testExperiment("efficientnet_b0", 8)
	experiment.ImageSize = 224
	experiment.ResolutionStrategy = "fixed"
	experiment.Preprocessing = &plans.Preprocessing{
		ResizeStrategy: "preserve_aspect_pad",
		Normalization:  "imagenet",
		CropStrategy:   "bbox_crop_if_available",
		BBoxMode:       "crop_and_compare_full_image",
	}
	experiment.AugmentationPolicy = "moderate"
	experiment.Augmentation = map[string]any{"horizontal_flip": true, "color_jitter": true}
	experiment.ClassBalancing = "weighted_loss"
	experiment.SamplingStrategy = "class_balanced_sampler"

	if err := validatePlannedExperiment(experiment, 0); err != nil {
		t.Fatalf("expected structured preprocessing to validate: %v", err)
	}
}

func TestValidatePlannedExperimentRejectsUnsupportedPreprocessing(t *testing.T) {
	experiment := testExperiment("efficientnet_b0", 8)
	experiment.Preprocessing = &plans.Preprocessing{ResizeStrategy: "magic_resize"}

	err := validatePlannedExperiment(experiment, 0)
	if err == nil || !strings.Contains(err.Error(), "preprocessing.resize_strategy") {
		t.Fatalf("expected preprocessing validation error, got %v", err)
	}
}

func TestValidatePlannedExperimentRejectsUnsupportedAugmentationKey(t *testing.T) {
	experiment := testExperiment("efficientnet_b0", 8)
	experiment.Augmentation = map[string]any{"hallucinate_more_dogs": true}

	err := validatePlannedExperiment(experiment, 0)
	if err == nil || !strings.Contains(err.Error(), "unsupported augmentation key") {
		t.Fatalf("expected augmentation key validation error, got %v", err)
	}
}

func TestValidatePlannedExperimentAcceptsStructuredAutoAugmentationPolicy(t *testing.T) {
	experiment := testExperiment("efficientnet_b0", 8)
	experiment.AugmentationPolicy = "randaugment"
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{
		PolicyType:  "randaugment",
		Magnitude:   15,
		NumOps:      3,
		Probability: 0.5,
	}

	if err := validatePlannedExperiment(experiment, 0); err != nil {
		t.Fatalf("expected structured augmentation policy to validate: %v", err)
	}
}

func TestValidatePlannedExperimentRejectsUnsupportedStructuredAugmentationPolicy(t *testing.T) {
	experiment := testExperiment("efficientnet_b0", 8)
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{PolicyType: "mystery"}

	err := validatePlannedExperiment(experiment, 0)
	if err == nil || !strings.Contains(err.Error(), "augmentation_policy_config.policy_type") {
		t.Fatalf("expected structured policy validation error, got %v", err)
	}
}

func TestValidatePlannedExperimentAcceptsMixedSampleStructuredAugmentationPolicy(t *testing.T) {
	experiment := testExperiment("efficientnet_b0", 8)
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{
		PolicyType:  "mixup",
		Alpha:       0.2,
		Probability: 0.5,
	}

	if err := validatePlannedExperiment(experiment, 0); err != nil {
		t.Fatalf("expected mixed-sample policy to validate after worker support: %v", err)
	}
}

func TestPrepareAutoMLExperimentSamplesSupportedHyperparameters(t *testing.T) {
	experiment := testExperiment("efficientnet_b0", 8)
	experiment.AugmentationPolicy = "randaugment"
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{PolicyType: "randaugment"}
	minLR := 1e-5
	maxLR := 3e-4
	minWD := 0.01
	maxWD := 0.08
	minMag := 5.0
	maxMag := 11.0
	experiment.AutoML = &automl.ExperimentAutoML{
		Enabled: true,
		Seed:    7,
		Intent: automl.ExperimentIntent{
			Summary:           "Tune concrete hyperparameters inside a regularization strategy.",
			AllowedParameters: []string{"learning_rate", "weight_decay", "batch_size", "augmentation_policy_config.magnitude"},
		},
		SearchSpace: &automl.HyperparameterSearchSpace{Parameters: []automl.HyperparameterParameterSpec{
			{Name: "learning_rate", Type: automl.ParameterFloat, Min: &minLR, Max: &maxLR, Scale: automl.SearchScaleLog},
			{Name: "weight_decay", Type: automl.ParameterFloat, Min: &minWD, Max: &maxWD},
			{Name: "batch_size", Type: automl.ParameterInteger, IntChoices: []int{16, 32}},
			{Name: "augmentation_policy_config.magnitude", Type: automl.ParameterInteger, Min: &minMag, Max: &maxMag},
		}},
	}

	prepared, err := prepareAutoMLExperiment(experiment, 0, automl.SamplerSeededRandom)
	if err != nil {
		t.Fatalf("prepare automl experiment: %v", err)
	}
	if err := validatePlannedExperiment(prepared, 0); err != nil {
		t.Fatalf("prepared experiment should validate: %v", err)
	}
	if prepared.AutoML.Suggestion == nil || prepared.AutoML.Suggestion.Values["learning_rate"] == nil {
		t.Fatalf("expected concrete AutoML suggestion, got %#v", prepared.AutoML)
	}
	if prepared.AutoML.ValueProvenance["learning_rate"] != automl.ProvenanceRandomSearch {
		t.Fatalf("expected random_search provenance, got %#v", prepared.AutoML.ValueProvenance)
	}
	if prepared.AugmentationPolicyConfig.Magnitude < 5 || prepared.AugmentationPolicyConfig.Magnitude > 11 {
		t.Fatalf("expected sampled randaugment magnitude inside bounds, got %d", prepared.AugmentationPolicyConfig.Magnitude)
	}
}

func TestPrepareAutoMLExperimentsDisabledNoOps(t *testing.T) {
	server, _, _ := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	settings := server.currentAutomationSettings()
	settings.AutoMLEnabled = false
	if _, err := server.store.SaveAutomationSettings(settings); err != nil {
		t.Fatalf("save settings: %v", err)
	}
	server.setAutomationSettings(settings)
	experiment := testExperiment("efficientnet_b0", 8)
	experiment.AutoML = &automl.ExperimentAutoML{
		Enabled: true,
		SearchSpace: &automl.HyperparameterSearchSpace{Parameters: []automl.HyperparameterParameterSpec{
			{Name: "preprocessing", Type: automl.ParameterCategorical, Choices: []string{"none"}},
		}},
	}

	prepared, warnings, err := server.prepareAutoMLExperiments([]plans.PlannedExperiment{experiment})
	if err != nil {
		t.Fatalf("expected disabled AutoML to no-op: %v", err)
	}
	if len(warnings) == 0 {
		t.Fatal("expected disabled AutoML warning")
	}
	if prepared[0].AutoML == nil || prepared[0].AutoML.Enabled {
		t.Fatalf("expected AutoML to be marked disabled, got %#v", prepared[0].AutoML)
	}
	if err := validatePlannedExperiment(prepared[0], 0); err != nil {
		t.Fatalf("disabled AutoML block should not affect validation: %v", err)
	}
}

func TestPrepareAutoMLExperimentsAutoEnablesBackendDefaultWhenEnabled(t *testing.T) {
	server, _, _ := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	settings := server.currentAutomationSettings()
	settings.AutoMLEnabled = true
	settings.AutoMLSampler = automl.SamplerSeededRandom
	if _, err := server.store.SaveAutomationSettings(settings); err != nil {
		t.Fatalf("save settings: %v", err)
	}
	server.setAutomationSettings(settings)
	experiment := testExperiment("efficientnet_b0", 8)
	experiment.Template = "efficientnet_transfer"
	experiment.Optimizer = "sgd"
	experiment.Scheduler = "step"
	experiment.WeightDecay = 0.01
	experiment.Dropout = 0.1
	experiment.LabelSmoothing = 0.03
	experiment.GradientClipNorm = 1.0
	experiment.AugmentationPolicy = "randaugment"
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{PolicyType: "randaugment"}
	experiment.ClassBalancing = "focal_loss"
	experiment.FineTuneStrategy = "last_block"

	prepared, warnings, err := server.prepareAutoMLExperiments([]plans.PlannedExperiment{experiment})
	if err != nil {
		t.Fatalf("prepare AutoML experiments: %v", err)
	}
	if len(warnings) == 0 || !strings.Contains(warnings[0], "auto-enabled") {
		t.Fatalf("expected auto-enabled warning, got %#v", warnings)
	}
	got := prepared[0]
	if got.AutoML == nil || !got.AutoML.Enabled || got.AutoML.Suggestion == nil {
		t.Fatalf("expected backend AutoML suggestion, got %#v", got.AutoML)
	}
	for _, name := range []string{
		"learning_rate",
		"weight_decay",
		"batch_size",
		"epochs",
		"early_stopping_patience",
		"dropout",
		"label_smoothing",
		"gradient_clip_norm",
		"optimizer_momentum",
		"scheduler_step_size",
		"scheduler_gamma",
		"augmentation_policy_config.magnitude",
		"class_balancing_config.focal_loss_gamma",
	} {
		if !automl.CoversParameter(got.AutoML.SearchSpace, name) {
			t.Fatalf("expected backend default search space to cover %s: %#v", name, got.AutoML.SearchSpace)
		}
		if _, ok := got.AutoML.Suggestion.Values[name]; !ok {
			t.Fatalf("expected suggestion value for %s: %#v", name, got.AutoML.Suggestion.Values)
		}
	}
	if got.AutoML.SearchSpace.Parameters[0].Source != automl.ProvenanceBackendDefault {
		t.Fatalf("expected backend-default search-space provenance, got %#v", got.AutoML.SearchSpace.Parameters[0])
	}
	if got.AutoML.ValueProvenance["learning_rate"] != automl.ProvenanceRandomSearch {
		t.Fatalf("expected sampled learning_rate provenance, got %#v", got.AutoML.ValueProvenance)
	}
	findSpec := func(name string) automl.HyperparameterParameterSpec {
		t.Helper()
		for _, spec := range got.AutoML.SearchSpace.Parameters {
			if spec.Name == name {
				return spec
			}
		}
		t.Fatalf("expected backend default search space to include %s", name)
		return automl.HyperparameterParameterSpec{}
	}
	assertFloatBounds := func(name string, minValue float64, maxValue float64) {
		t.Helper()
		spec := findSpec(name)
		if spec.Min == nil || spec.Max == nil || *spec.Min != minValue || *spec.Max != maxValue {
			t.Fatalf("expected %s bounds [%g, %g], got %#v", name, minValue, maxValue, spec)
		}
	}
	assertIntRange := func(name string, minValue int, maxValue int) {
		t.Helper()
		spec := findSpec(name)
		if spec.Min == nil || spec.Max == nil || int(*spec.Min) != minValue || int(*spec.Max) != maxValue {
			t.Fatalf("expected %s range [%d, %d], got %#v", name, minValue, maxValue, spec)
		}
	}
	assertIntChoices := func(name string, choices []int) {
		t.Helper()
		spec := findSpec(name)
		if len(spec.IntChoices) != len(choices) {
			t.Fatalf("expected %s choices %#v, got %#v", name, choices, spec.IntChoices)
		}
		for index, choice := range choices {
			if spec.IntChoices[index] != choice {
				t.Fatalf("expected %s choices %#v, got %#v", name, choices, spec.IntChoices)
			}
		}
	}
	assertFloatBounds("learning_rate", 1e-5, 1e-1)
	assertFloatBounds("weight_decay", 0, 0.3)
	assertIntChoices("batch_size", []int{4, 8, 16, 32, 64, 128})
	assertIntRange("epochs", 3, 24)
	assertIntChoices("early_stopping_patience", []int{0, 2, 4, 6, 8, 10, 12})
	assertFloatBounds("dropout", 0, 0.7)
	assertFloatBounds("label_smoothing", 0, 0.3)
	assertFloatBounds("gradient_clip_norm", 0, 10)
	assertFloatBounds("optimizer_momentum", 0, 0.99)
	assertIntRange("scheduler_step_size", 1, 24)
	assertFloatBounds("scheduler_gamma", 0.05, 0.95)
	assertIntRange("augmentation_policy_config.magnitude", 0, 15)
	assertFloatBounds("augmentation_policy_config.probability", 0, 1)
	assertFloatBounds("class_balancing_config.focal_loss_gamma", 0.5, 5)
	if got.Model != experiment.Model || got.Template != experiment.Template || got.FineTuneStrategy != experiment.FineTuneStrategy {
		t.Fatalf("AutoML changed LLM-owned strategy fields: before %#v after %#v", experiment, got)
	}
	if got.AugmentationPolicy != "randaugment" || got.AugmentationPolicyConfig.PolicyType != "randaugment" || got.ClassBalancing != "focal_loss" {
		t.Fatalf("AutoML changed LLM-owned augmentation/class-balancing strategy: %#v", got)
	}
	if err := validatePlannedExperiment(got, 0); err != nil {
		t.Fatalf("prepared backend AutoML experiment should validate: %v", err)
	}
}

func TestValidatePlannedExperimentRejectsAutoMLStrategyField(t *testing.T) {
	experiment := testExperiment("efficientnet_b0", 8)
	experiment.AutoML = &automl.ExperimentAutoML{
		Enabled: true,
		SearchSpace: &automl.HyperparameterSearchSpace{Parameters: []automl.HyperparameterParameterSpec{
			{Name: "preprocessing", Type: automl.ParameterCategorical, Choices: []string{"none"}},
		}},
	}

	err := validatePlannedExperiment(experiment, 0)
	if err == nil || !strings.Contains(err.Error(), "AutoML search space") {
		t.Fatalf("expected AutoML strategy-field validation error, got %v", err)
	}
}

func TestPrepareAutoMLExperimentSamplesDeferredHyperparameters(t *testing.T) {
	experiment := testExperiment("resnet18", 10)
	experiment.Optimizer = "sgd"
	experiment.Scheduler = "step"
	experiment.ClassBalancing = "focal_loss"
	minDropout := 0.05
	maxDropout := 0.35
	minMomentum := 0.7
	maxMomentum := 0.95
	minStepSize := 1.0
	maxStepSize := 4.0
	minGamma := 0.2
	maxGamma := 0.8
	minSmooth := 0.0
	maxSmooth := 0.15
	minClip := 0.5
	maxClip := 3.0
	minFocal := 1.0
	maxFocal := 4.0
	experiment.AutoML = &automl.ExperimentAutoML{
		Enabled: true,
		Seed:    11,
		SearchSpace: &automl.HyperparameterSearchSpace{Parameters: []automl.HyperparameterParameterSpec{
			{Name: "dropout", Type: automl.ParameterFloat, Min: &minDropout, Max: &maxDropout},
			{Name: "optimizer_momentum", Type: automl.ParameterFloat, Min: &minMomentum, Max: &maxMomentum},
			{Name: "scheduler_step_size", Type: automl.ParameterInteger, Min: &minStepSize, Max: &maxStepSize},
			{Name: "scheduler_gamma", Type: automl.ParameterFloat, Min: &minGamma, Max: &maxGamma},
			{Name: "label_smoothing", Type: automl.ParameterFloat, Min: &minSmooth, Max: &maxSmooth},
			{Name: "gradient_clip_norm", Type: automl.ParameterFloat, Min: &minClip, Max: &maxClip},
			{Name: "class_balancing_config.focal_loss_gamma", Type: automl.ParameterFloat, Min: &minFocal, Max: &maxFocal},
		}},
	}

	prepared, err := prepareAutoMLExperiment(experiment, 0, automl.SamplerAdaptiveBayesian)
	if err != nil {
		t.Fatalf("prepare deferred automl experiment: %v", err)
	}
	if err := validatePlannedExperiment(prepared, 0); err != nil {
		t.Fatalf("prepared deferred experiment should validate: %v", err)
	}
	if prepared.Dropout < minDropout || prepared.Dropout > maxDropout {
		t.Fatalf("dropout out of range: %g", prepared.Dropout)
	}
	if prepared.OptimizerMomentum < minMomentum || prepared.OptimizerMomentum > maxMomentum {
		t.Fatalf("momentum out of range: %g", prepared.OptimizerMomentum)
	}
	if prepared.SchedulerStepSize < 1 || prepared.SchedulerStepSize > 4 {
		t.Fatalf("scheduler step size out of range: %d", prepared.SchedulerStepSize)
	}
	if prepared.SchedulerGamma < minGamma || prepared.SchedulerGamma > maxGamma {
		t.Fatalf("scheduler gamma out of range: %g", prepared.SchedulerGamma)
	}
	if prepared.LabelSmoothing < minSmooth || prepared.LabelSmoothing > maxSmooth {
		t.Fatalf("label smoothing out of range: %g", prepared.LabelSmoothing)
	}
	if prepared.GradientClipNorm < minClip || prepared.GradientClipNorm > maxClip {
		t.Fatalf("gradient clip norm out of range: %g", prepared.GradientClipNorm)
	}
	if prepared.ClassBalancingConfig["focal_loss_gamma"] == nil {
		t.Fatalf("expected focal loss gamma in class balancing config: %#v", prepared.ClassBalancingConfig)
	}
	if prepared.AutoML.ValueProvenance["dropout"] != automl.ProvenanceBayesianOptimizer {
		t.Fatalf("expected bayesian provenance, got %#v", prepared.AutoML.ValueProvenance)
	}
}

func TestExecuteExperimentPlanIncludesStructuredAugmentationPolicyConfig(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		func() plans.PlannedExperiment {
			experiment := testExperiment("efficientnet_b0", 8)
			experiment.AugmentationPolicy = "randaugment"
			experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{
				PolicyType:       "randaugment",
				Magnitude:        9,
				NumOps:           2,
				NumMagnitudeBins: 31,
				Probability:      0.75,
			}
			return experiment
		}(),
	})

	response, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "local"})
	if err != nil {
		t.Fatalf("execute experiment plan: %v", err)
	}
	if len(response.Jobs) != 1 {
		t.Fatalf("expected one job, got %d", len(response.Jobs))
	}
	if _, ok := response.Jobs[0].Config["augmentation_policy_config"]; !ok {
		t.Fatalf("expected structured augmentation policy in job config, got %#v", response.Jobs[0].Config)
	}
	if response.Jobs[0].ProjectID != projectID {
		t.Fatalf("expected job in project %s, got %s", projectID, response.Jobs[0].ProjectID)
	}
}

func TestExecuteExperimentPlanConfiguresYOLODetectorJob(t *testing.T) {
	experiment := testExperiment("yolo11n.pt", 6)
	experiment.Template = "yolo11_detection"
	experiment.ImageSize = 640
	server, _, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{experiment})
	if _, err := server.store.UpdateDatasetProfile(plan.DatasetID, map[string]any{
		"task_type":    "object_detection",
		"total_images": 2,
		"class_count":  2,
		"class_names":  []any{"real_face", "fake_face"},
		"metadata_summary": map[string]any{
			"yolo_available": true,
			"yolo_summary": map[string]any{
				"available":        true,
				"format":           "yolo",
				"label_file_count": 2,
				"bbox_count":       3,
				"class_names":      []any{"real_face", "fake_face"},
			},
		},
		"dataset_traits": []any{"yolo_format", "object_detection"},
	}); err != nil {
		t.Fatalf("profile yolo dataset: %v", err)
	}

	response, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "local"})
	if err != nil {
		t.Fatalf("execute yolo experiment plan: %v", err)
	}
	if len(response.Jobs) != 1 {
		t.Fatalf("expected one job, got %d", len(response.Jobs))
	}
	config := response.Jobs[0].Config
	if got := config["task_type"]; got != "object_detection" {
		t.Fatalf("expected object_detection task_type, got %#v in %#v", got, config)
	}
	if got := config["model_kind"]; got != "ultralytics_yolo_detector" {
		t.Fatalf("expected yolo detector model_kind, got %#v in %#v", got, config)
	}
	if got := config["pretrained_weights"]; got != "yolo11n.pt" {
		t.Fatalf("expected pretrained weights, got %#v in %#v", got, config)
	}
	if got := config["class_names"]; len(got.([]string)) != 2 {
		t.Fatalf("expected class names in yolo job config, got %#v", got)
	}
}

func TestExecuteExperimentPlanRejectsClassifierForYOLODataset(t *testing.T) {
	server, _, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 3)})
	if _, err := server.store.UpdateDatasetProfile(plan.DatasetID, map[string]any{
		"task_type":    "object_detection",
		"total_images": 2,
		"class_count":  1,
		"metadata_summary": map[string]any{
			"yolo_available": true,
			"artifact_counts": map[string]any{
				"yolo_dataset_config": 1,
				"yolo_label_file":     2,
			},
		},
	}); err != nil {
		t.Fatalf("profile yolo dataset: %v", err)
	}

	_, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "local"})
	if err == nil || !strings.Contains(err.Error(), "classifier model") {
		t.Fatalf("expected classifier/detection compatibility error, got %v", err)
	}
}

func TestDatasetPlannerCreatesYOLOInitialPlan(t *testing.T) {
	project := projects.Project{ID: "project_1", Goal: "detect objects live"}
	dataset := datasets.Dataset{
		ID:        "dataset_1",
		ProjectID: "project_1",
		Status:    datasets.StatusProfiled,
		Profile: map[string]any{
			"task_type":    "object_detection",
			"total_images": 20,
			"class_count":  1,
			"metadata_summary": map[string]any{
				"yolo_available": true,
				"yolo_summary": map[string]any{
					"available":        true,
					"format":           "yolo",
					"label_file_count": 20,
				},
			},
		},
	}

	recommendation, err := agents.NewDatasetPlanner().BuildExperimentPlan(project, dataset, agents.PlanPreferences{})
	if err != nil {
		t.Fatalf("build yolo initial plan: %v", err)
	}
	if recommendation.TargetMetric != "mAP50_95" {
		t.Fatalf("expected detector target metric, got %s", recommendation.TargetMetric)
	}
	if len(recommendation.Experiments) == 0 {
		t.Fatal("expected yolo experiments")
	}
	for _, experiment := range recommendation.Experiments {
		if !strings.HasPrefix(experiment.Model, "yolo11") {
			t.Fatalf("expected yolo model, got %#v", experiment)
		}
		if experiment.Template != "yolo11_detection" {
			t.Fatalf("expected yolo detection template, got %#v", experiment)
		}
	}
}

func TestHolisticRunScoreCanPreferFastModelForLiveGoal(t *testing.T) {
	slowSummary := runs.TrainingRunSummary{
		JobID:            "job_slow",
		Status:           jobs.StatusSucceeded,
		Model:            "vit_b_16",
		BestMacroF1:      0.97,
		BestAccuracy:     0.97,
		EstimatedCostUSD: 0.01,
		RuntimeSeconds:   60,
	}
	fastSummary := runs.TrainingRunSummary{
		JobID:            "job_fast",
		Status:           jobs.StatusSucceeded,
		Model:            "mobilenet_v3_small",
		BestMacroF1:      0.90,
		BestAccuracy:     0.90,
		EstimatedCostUSD: 0.01,
		RuntimeSeconds:   60,
	}
	evaluations := []runs.TrainingRunEvaluation{
		{JobID: "job_slow", ModelProfile: map[string]any{"estimated_latency_ms": 120.0}, HolisticScores: map[string]any{}},
		{JobID: "job_fast", ModelProfile: map[string]any{"estimated_latency_ms": 8.0}, HolisticScores: map[string]any{}},
	}

	liveBest, ok := bestSuccessfulTrainingSummaryForObjective("macro_f1", []runs.TrainingRunSummary{slowSummary, fastSummary}, evaluations, projectObjectiveContext("fast live service"))
	if !ok {
		t.Fatal("expected live best summary")
	}
	if liveBest.JobID != "job_fast" {
		t.Fatalf("expected live goal to prefer fast model, got %s", liveBest.JobID)
	}

	qualityBest, ok := bestSuccessfulTrainingSummaryForObjective("macro_f1", []runs.TrainingRunSummary{slowSummary, fastSummary}, evaluations, projectObjectiveContext("best quality model"))
	if !ok {
		t.Fatal("expected quality best summary")
	}
	if qualityBest.JobID != "job_slow" {
		t.Fatalf("expected quality goal to prefer higher metric model, got %s", qualityBest.JobID)
	}
}

func TestSelectChampionDecisionCreatesChampionRecord(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.91)
	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeSelectChampion, "Select the best live model.", map[string]any{
		"champion_job_id": job.ID,
	})
	if err != nil {
		t.Fatalf("create decision: %v", err)
	}

	if err := server.persistProjectChampionFromDecision(projectID, decision); err != nil {
		t.Fatalf("persist champion: %v", err)
	}
	champion, err := server.store.GetProjectChampion(projectID)
	if err != nil {
		t.Fatalf("get champion: %v", err)
	}
	if champion.JobID != job.ID {
		t.Fatalf("expected champion job %s, got %s", job.ID, champion.JobID)
	}
	if champion.SourceDecisionID != decision.ID {
		t.Fatalf("expected source decision %s, got %s", decision.ID, champion.SourceDecisionID)
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list events: %v", err)
	}
	if !hasExecutionEvent(events, execution.EventChampionSelected) {
		t.Fatal("expected champion selected event")
	}
}

func TestSelectChampionDecisionUsesHolisticValidationMetrics(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
		testExperiment("efficientnet_b0", 6),
	})
	requestedJob, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.96)
	healthyJob, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[1], jobs.StatusSucceeded, 0.91)

	badTrainLoss := 3.1
	badValLoss := 3.7
	if _, err := server.store.UpsertTrainingRunSummary(requestedJob.ID, runs.TrainingRunSummaryUpdate{
		FinalTrainLoss: &badTrainLoss,
		FinalValLoss:   &badValLoss,
	}); err != nil {
		t.Fatalf("upsert requested losses: %v", err)
	}
	goodTrainLoss := 0.18
	goodValLoss := 0.22
	if _, err := server.store.UpsertTrainingRunSummary(healthyJob.ID, runs.TrainingRunSummaryUpdate{
		FinalTrainLoss: &goodTrainLoss,
		FinalValLoss:   &goodValLoss,
	}); err != nil {
		t.Fatalf("upsert healthy losses: %v", err)
	}
	if _, err := server.store.UpsertTrainingRunEvaluation(requestedJob.ID, runs.TrainingRunEvaluationUpdate{
		ObjectiveProfile: map[string]any{
			"heldout_test_macro_f1": 0.62,
			"heldout_test_accuracy": 0.64,
			"heldout_test_loss":     3.6,
		},
		PerClassMetrics: map[string]any{
			"cat":       map[string]any{"recall": 0.42, "f1-score": 0.48},
			"dog":       map[string]any{"recall": 0.52, "f1-score": 0.54},
			"macro avg": map[string]any{"f1-score": 0.51},
		},
		ConfusionMatrix: [][]int{{10, 20}, {14, 16}},
		ModelProfile:    map[string]any{"class_labels": []string{"cat", "dog"}},
		HolisticScores: map[string]any{
			"overall_score": 0.55,
			"training_diagnostics": map[string]any{
				"severity":            0.92,
				"divergence_detected": true,
			},
		},
	}); err != nil {
		t.Fatalf("upsert requested evaluation: %v", err)
	}
	if _, err := server.store.UpsertTrainingRunEvaluation(healthyJob.ID, runs.TrainingRunEvaluationUpdate{
		ObjectiveProfile: map[string]any{
			"heldout_test_macro_f1": 0.90,
			"heldout_test_accuracy": 0.91,
			"heldout_test_loss":     0.24,
		},
		PerClassMetrics: map[string]any{
			"cat":       map[string]any{"recall": 0.90, "f1-score": 0.91},
			"dog":       map[string]any{"recall": 0.89, "f1-score": 0.90},
			"macro avg": map[string]any{"f1-score": 0.90},
		},
		ConfusionMatrix: [][]int{{28, 2}, {3, 27}},
		ModelProfile:    map[string]any{"class_labels": []string{"cat", "dog"}},
		HolisticScores:  map[string]any{"overall_score": 0.91},
	}); err != nil {
		t.Fatalf("upsert healthy evaluation: %v", err)
	}

	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeSelectChampion, "Select the highest macro_f1 model.", map[string]any{
		"champion_job_id": requestedJob.ID,
		"target_metric":   "macro_f1",
	})
	if err != nil {
		t.Fatalf("create decision: %v", err)
	}

	if err := server.persistProjectChampionFromDecision(projectID, decision); err != nil {
		t.Fatalf("persist champion: %v", err)
	}
	champion, err := server.store.GetProjectChampion(projectID)
	if err != nil {
		t.Fatalf("get champion: %v", err)
	}
	if champion.JobID != healthyJob.ID {
		t.Fatalf("expected holistic guard to choose healthy job %s, got %s", healthyJob.ID, champion.JobID)
	}
	if payloadString(champion.Metrics, "selection_source") != "backend_holistic_override" {
		t.Fatalf("expected backend override source, got %#v", champion.Metrics)
	}
	if payloadString(champion.Metrics, "requested_champion_job_id") != requestedJob.ID {
		t.Fatalf("expected requested job retained in metrics, got %#v", champion.Metrics)
	}
	if payloadFloat(champion.Metrics, "deployment_readiness_score") <= payloadFloat(champion.Metrics, "requested_deployment_readiness_score") {
		t.Fatalf("expected selected readiness score to beat requested score, got %#v", champion.Metrics)
	}
}

func TestSelectChampionQueuesONNXExportFromCheckpoint(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.91)
	if _, err := server.store.UpsertTrainingRunEvaluation(job.ID, runs.TrainingRunEvaluationUpdate{
		ModelProfile: map[string]any{
			"artifact_uri":   "file:///exports/model.pt",
			"export_status":  "PENDING_ARTIFACT",
			"class_labels":   []string{"cat", "dog"},
			"training_shape": []int{1, 3, 64, 64},
		},
	}); err != nil {
		t.Fatalf("upsert evaluation: %v", err)
	}
	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeSelectChampion, "Select the best live model.", map[string]any{
		"champion_job_id": job.ID,
	})
	if err != nil {
		t.Fatalf("create decision: %v", err)
	}

	if err := server.persistProjectChampionFromDecision(projectID, decision); err != nil {
		t.Fatalf("persist champion: %v", err)
	}
	exports, err := server.store.ListProjectChampionExports(projectID)
	if err != nil {
		t.Fatalf("list exports: %v", err)
	}
	if len(exports) != 1 {
		t.Fatalf("expected automatic ONNX export record, got %#v", exports)
	}
	export := exports[0]
	if export.Format != "onnx" || export.Status != runs.ChampionExportStatusPending || export.ArtifactURI != "" {
		t.Fatalf("expected pending ONNX conversion export, got %#v", export)
	}
	projectJobs, err := server.store.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	var exportJob jobs.ExperimentJob
	for _, projectJob := range projectJobs {
		if projectJob.Template == jobs.TemplateExportChampion {
			exportJob = projectJob
			break
		}
	}
	if exportJob.ID == "" {
		t.Fatalf("expected automatic export job, got %#v", projectJobs)
	}
	if exportJob.Config["export_id"] != export.ID {
		t.Fatalf("expected export job linked to automatic export, got %#v", exportJob.Config)
	}
	if sourceArtifactURI := jobConfigString(exportJob.Config, "source_artifact_uri"); sourceArtifactURI != "file:///exports/model.pt" {
		t.Fatalf("expected checkpoint source artifact, got %q", sourceArtifactURI)
	}
}

func TestStopProjectDecisionCreatesBestAvailableChampionRecord(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
		testExperiment("efficientnet_b0", 6),
	})
	createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.71)
	bestJob, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[1], jobs.StatusSucceeded, 0.78)
	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeStopProject, "Stop after budget review.", map[string]any{
		"target_metric": "macro_f1",
	})
	if err != nil {
		t.Fatalf("create decision: %v", err)
	}

	if err := server.persistProjectChampionFromDecision(projectID, decision); err != nil {
		t.Fatalf("persist fallback champion: %v", err)
	}
	champion, err := server.store.GetProjectChampion(projectID)
	if err != nil {
		t.Fatalf("get champion: %v", err)
	}
	if champion.JobID != bestJob.ID {
		t.Fatalf("expected fallback champion job %s, got %s", bestJob.ID, champion.JobID)
	}
	if champion.SourceDecisionID != decision.ID {
		t.Fatalf("expected source decision %s, got %s", decision.ID, champion.SourceDecisionID)
	}
	if payloadString(champion.Metrics, "selection_source") != "terminal_stop_best_available" {
		t.Fatalf("expected terminal stop selection source, got %#v", champion.Metrics["selection_source"])
	}
}

func TestTrainingRunEvaluationAddsBackendDiagnostics(t *testing.T) {
	server, _, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.74)
	if _, err := server.store.ReportMetric(job.ID, 1, map[string]float64{"train_loss": 1.0, "val_loss": 0.98}); err != nil {
		t.Fatalf("report metric: %v", err)
	}
	if _, err := server.store.ReportMetric(job.ID, 2, map[string]float64{"train_loss": 0.70, "val_loss": 1.12}); err != nil {
		t.Fatalf("report metric: %v", err)
	}

	router := NewRouter(server.store)
	worker, err := server.store.RegisterWorker(plan.ProjectID, "worker", "modal")
	if err != nil {
		t.Fatalf("register worker: %v", err)
	}
	assigned := pollJobForCallback(t, router, worker.ID, `{"provider":"modal"}`)
	body := []byte(`{"training_attempt_id":"` + callbackAttemptID(t, assigned) + `","holistic_scores":{"overall_score":0.72},"model_profile":{"estimated_latency_ms":10}}`)
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/training-run-evaluation", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, req, assigned)
	resp := httptest.NewRecorder()

	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("expected status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	var evaluation runs.TrainingRunEvaluation
	if err := json.NewDecoder(resp.Body).Decode(&evaluation); err != nil {
		t.Fatalf("decode evaluation: %v", err)
	}
	diagnostics := payloadMap(evaluation.HolisticScores, "training_diagnostics")
	if payloadString(diagnostics, "status") != "diverging" {
		t.Fatalf("expected diverging diagnostics, got %#v", diagnostics)
	}
	if detected, _ := evaluation.HolisticScores["divergence_detected"].(bool); !detected {
		t.Fatalf("expected divergence_detected in holistic scores, got %#v", evaluation.HolisticScores)
	}
}

func TestAutonomousExperimentPlannerDecisionSchedulesFollowUp(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_AGENT_MODE", "autonomous")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "true")
	t.Setenv("MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER", "modal")
	t.Setenv("MODEL_EXPRESS_DEFAULT_GPU_TYPE", "T4")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	payload, err := experimentPlannerDecisionPayload(experimentPlannerAddExperimentsRecommendation(), invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err != nil {
		t.Fatalf("build planner payload: %v", err)
	}
	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "try a stronger planner batch", payload)
	if err != nil {
		t.Fatalf("create planner decision: %v", err)
	}

	result := automaticExperimentReviewResult{Decision: &decision}
	if err := server.schedulePlannerDecision(projectID, plan, decision, result); err != nil {
		t.Fatalf("schedule planner decision: %v", err)
	}
	projectPlans := listExperimentPlans(t, server, projectID)
	if len(projectPlans) != 2 {
		t.Fatalf("expected original plus follow-up plan, got %d", len(projectPlans))
	}
	followUpPlan, ok := followUpPlanForDecision(projectPlans, decision.ID)
	if !ok {
		t.Fatalf("expected follow-up source decision %s, got plans %#v", decision.ID, projectPlans)
	}
	if followUpPlan.Experiments[0].ImageSize != 256 {
		t.Fatalf("expected aggressive planner config to be preserved, got image size %d", followUpPlan.Experiments[0].ImageSize)
	}

	requirements, err := server.store.ListProjectWorkerRequirements(projectID)
	if err != nil {
		t.Fatalf("list worker requirements: %v", err)
	}
	if len(requirements) != 1 {
		t.Fatalf("expected one worker requirement, got %d", len(requirements))
	}
	if requirements[0].PlanID != followUpPlan.ID {
		t.Fatalf("expected worker requirement for follow-up plan %s, got %s", followUpPlan.ID, requirements[0].PlanID)
	}
}

func TestProposedExperimentPlannerDecisionDoesNotAutoSchedule(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_AGENT_MODE", "propose")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	payload, err := experimentPlannerDecisionPayload(experimentPlannerAddExperimentsRecommendation(), invocation, "propose", plannerInputForPayload(t, server, projectID))
	if err != nil {
		t.Fatalf("build planner payload: %v", err)
	}
	if autoExecutable, _ := payload["auto_executable"].(bool); autoExecutable {
		t.Fatal("expected propose mode planner payload to be non-auto-executable")
	}
	if _, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "proposed planner batch", payload); err != nil {
		t.Fatalf("create planner decision: %v", err)
	}

	projectPlans, err := server.store.ListProjectExperimentPlans(projectID)
	if err != nil {
		t.Fatalf("list project plans: %v", err)
	}
	if len(projectPlans) != 1 {
		t.Fatalf("expected only the original plan in propose mode, got %d plans", len(projectPlans))
	}

	agentDecisions, err := server.store.ListProjectAgentDecisions(projectID)
	if err != nil {
		t.Fatalf("list agent decisions: %v", err)
	}
	if _, ok := actionDecisionForPlan(agentDecisions, plan.ID); ok {
		t.Fatal("expected proposed LLM decision to be ignored by automatic deterministic scheduling")
	}
}

func TestExperimentPlannerRetriesAfterBackendValidationRejection(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	responses := []string{
		`{
			"summary": "Try the same MobileNet mechanism for more epochs.",
			"decision_type": "ADD_EXPERIMENTS",
			"rationale": "The previous MobileNet result was the best prior run, so the family is worth refining.",
			"confidence": 0.72,
			"planning_mode": "exploit",
			"deterministic_diagnosis_used": ["plateau_score=0.40"],
			"evidence_used": ["previous MobileNet was best"],
			"hypothesis": "More epochs may improve the previous best MobileNet model.",
			"expected_failure_modes": ["plateau may continue"],
			"dataset_preprocessing_rationale": "Keep preprocessing stable while checking whether budgeted training helps.",
			"changed_variables": ["scheduler", "epochs"],
			"success_criteria": "Improve macro-F1 over the prior best run.",
			"stop_condition": "Select champion if the repeat does not improve.",
			"deployment_tradeoff": "Same deployment profile but more training cost.",
			"candidate_hypotheses": [{
				"hypothesis": "More epochs may improve the previous best MobileNet model.",
				"planning_mode": "exploit",
				"mechanism": "capacity_finetune",
				"intervention": "Extend the prior MobileNet training budget without changing preprocessing.",
				"proposed_changes": {"scheduler": "none", "epochs": 6},
				"expected_effect": "Determine whether the prior best model was simply undertrained.",
				"expected_metric_impact": 0.01,
				"expected_tradeoffs": ["more runtime"],
				"risk": "medium",
				"cost_level": "low",
				"novelty_score": 0.25,
				"evidence_used": ["previous MobileNet was best"],
				"similar_success_memory_ids": [],
				"similar_failure_memory_ids": [],
				"experiment_config": {
					"template": "mobilenet_transfer",
					"model": "mobilenet_v3_small",
					"epochs": 12,
					"batch_size": 16,
					"learning_rate": 0.0003,
					"reason": "Repeat MobileNet with more epochs."
				}
			}],
			"proposed_experiments": [{
				"template": "mobilenet_transfer",
				"model": "mobilenet_v3_small",
				"epochs": 12,
				"batch_size": 16,
				"learning_rate": 0.0003,
				"reason": "Repeat MobileNet with more epochs."
			}],
			"proposal_mechanisms": [{
				"experiment_index": 0,
				"mechanism": "capacity_finetune",
				"intervention": "Extend the prior MobileNet training budget without changing preprocessing.",
				"evidence_used": ["previous MobileNet was best"],
				"expected_effect": "Determine whether the prior best model was simply undertrained."
			}],
			"why_can_beat_champion": "The previous family was best, so longer training might help.",
			"expected_delta_vs_champion": 0.01,
			"risks": ["low novelty"],
			"expected_tradeoffs": ["more runtime"],
			"novelty_notes": ["longer training"],
			"rejected_options": [{"option":"different preprocessing","reason":"not tried in this proposal","evidence":"none","applies_when":["unknown"]}],
			"tags": ["retry-test"]
		}`,
		`{
			"summary": "Use backend feedback to switch to a meaningful EfficientNet preprocessing ablation.",
			"decision_type": "ADD_EXPERIMENTS",
			"rationale": "The rejected MobileNet repeat only changed epochs, so the corrected plan changes model family, resolution, augmentation, and class balancing.",
			"confidence": 0.83,
			"planning_mode": "preprocessing_ablation",
			"deterministic_diagnosis_used": ["plateau_score=0.40"],
			"evidence_used": ["backend rejected the minor-only repeat", "prior run plateaued"],
			"hypothesis": "EfficientNet with moderate augmentation and weighted loss can improve macro-F1 more than a shallow epoch repeat.",
			"expected_failure_modes": ["higher runtime"],
			"dataset_preprocessing_rationale": "Change preprocessing and balancing to test a real mechanism instead of only extending epochs.",
			"changed_variables": ["model_family", "resolution_strategy", "augmentation_policy", "class_balancing"],
			"success_criteria": "Improve macro-F1 by at least 0.01 without excessive latency.",
			"stop_condition": "Select champion if this mechanism does not improve.",
			"deployment_tradeoff": "More runtime for a stronger quality challenger.",
			"candidate_hypotheses": [{
				"hypothesis": "EfficientNet with moderate augmentation and weighted loss can improve macro-F1 more than a shallow epoch repeat.",
				"planning_mode": "preprocessing_ablation",
				"mechanism": "class_imbalance",
				"intervention": "Use weighted loss and weighted sampling with a stronger EfficientNet challenger.",
				"proposed_changes": {"model_family": "efficientnet", "resolution_strategy": "high_resolution_ablation", "augmentation_policy": "moderate", "class_balancing": "weighted_loss"},
				"expected_effect": "Improve macro-F1 by addressing class imbalance instead of repeating training budget.",
				"expected_metric_impact": 0.02,
				"expected_tradeoffs": ["higher runtime"],
				"risk": "medium",
				"cost_level": "medium",
				"novelty_score": 0.8,
				"evidence_used": ["backend rejected the minor-only repeat", "prior run plateaued"],
				"similar_success_memory_ids": [],
				"similar_failure_memory_ids": [],
				"experiment_config": {
					"template": "efficientnet_transfer",
					"model": "efficientnet_b0",
					"epochs": 10,
					"batch_size": 16,
					"learning_rate": 0.0003,
					"reason": "Corrected proposal changes family, augmentation, and balancing.",
					"image_size": 256,
					"resolution_strategy": "high_resolution_ablation",
					"augmentation_policy": "moderate",
					"class_balancing": "weighted_loss",
					"sampling_strategy": "weighted_random_sampler"
				}
			}],
			"proposed_experiments": [{
				"template": "efficientnet_transfer",
				"model": "efficientnet_b0",
				"epochs": 10,
				"batch_size": 16,
				"learning_rate": 0.0003,
				"reason": "Corrected proposal changes family, augmentation, and balancing.",
				"image_size": 256,
				"resolution_strategy": "high_resolution_ablation",
				"augmentation_policy": "moderate",
				"class_balancing": "weighted_loss",
				"sampling_strategy": "weighted_random_sampler"
			}],
			"proposal_mechanisms": [{
				"experiment_index": 0,
				"mechanism": "class_imbalance",
				"intervention": "Use weighted loss and weighted sampling with a stronger EfficientNet challenger.",
				"evidence_used": ["backend rejected the minor-only repeat", "prior run plateaued"],
				"expected_effect": "Improve macro-F1 by addressing class imbalance instead of repeating training budget."
			}],
			"why_can_beat_champion": "It changes the mechanism rather than repeating MobileNet with more epochs.",
			"expected_delta_vs_champion": 0.02,
			"risks": ["higher runtime"],
			"expected_tradeoffs": ["quality for cost"],
			"novelty_notes": ["new model family and class balancing"],
			"rejected_options": [{"option":"more MobileNet epochs","reason":"backend rejected minor-only repeat","evidence":"validation feedback","applies_when":["plateau"]}],
			"tags": ["retry-test", "corrected"]
		}`,
	}
	requestBodies := []string{}
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode llm request: %v", err)
		}
		combined := ""
		for _, message := range request.Messages {
			combined += message.Content + "\n"
		}
		requestBodies = append(requestBodies, combined)
		index := len(requestBodies) - 1
		if index >= len(responses) {
			index = len(responses) - 1
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]string{
					"role":    "assistant",
					"content": responses[index],
				},
			}},
		})
	}))
	defer llmServer.Close()

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "test-model")
	t.Setenv("MODEL_EXPRESS_AGENT_MODE", "autonomous")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)

	handled, err := server.runExperimentPlannerAfterTrainingJob(job)
	if err != nil {
		t.Fatalf("run planner: %v", err)
	}
	if !handled {
		t.Fatal("expected planner to handle the completed job")
	}
	if len(requestBodies) != 2 {
		t.Fatalf("expected initial request plus retry, got %d", len(requestBodies))
	}
	if !strings.Contains(requestBodies[1], "planner_validation_feedback") || !strings.Contains(requestBodies[1], "minor tuning knobs") {
		t.Fatalf("expected retry request to include backend validation feedback, got %s", requestBodies[1])
	}
	agentDecisions := listAgentDecisions(t, server, projectID)
	if len(agentDecisions) != 1 {
		t.Fatalf("expected one accepted decision after retry, got %d", len(agentDecisions))
	}
	if retryCount, _ := agentDecisions[0].Payload["validation_retry_count"].(int); retryCount != 1 {
		t.Fatalf("expected retry count 1 in decision payload, got %#v", agentDecisions[0].Payload["validation_retry_count"])
	}
	experiments := plannedExperimentsFromUnknown(t, agentDecisions[0].Payload["proposed_experiments"])
	if len(experiments) != 1 || experiments[0].Model != "efficientnet_b0" {
		t.Fatalf("expected corrected EfficientNet proposal, got %#v", experiments)
	}
	projectPlans := listExperimentPlans(t, server, projectID)
	if len(projectPlans) != 2 {
		t.Fatalf("expected corrected retry proposal to schedule one follow-up plan, got %d plans", len(projectPlans))
	}
	followUpPlan, ok := followUpPlanForDecision(projectPlans, agentDecisions[0].ID)
	if !ok {
		t.Fatalf("expected follow-up plan from accepted retry decision, got %#v", projectPlans)
	}
	if len(followUpPlan.Experiments) != 1 || followUpPlan.Experiments[0].Model != "efficientnet_b0" {
		t.Fatalf("expected scheduled follow-up to use corrected EfficientNet proposal, got %#v", followUpPlan.Experiments)
	}
	projectJobs, err := server.store.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	if len(projectJobs) != 1 {
		t.Fatalf("expected only the completed source job with auto execution disabled, got %d jobs", len(projectJobs))
	}
	invocations, err := server.store.ListProjectAgentInvocations(projectID, memory.AgentInvocationFilter{AgentName: agents.ExperimentPlannerAgentName})
	if err != nil {
		t.Fatalf("list invocations: %v", err)
	}
	if len(invocations) != 2 {
		t.Fatalf("expected two planner invocations, got %d", len(invocations))
	}
	foundRejected := false
	for _, invocation := range invocations {
		if invocation.DownstreamOutcome["backend_validation_status"] == "rejected" {
			foundRejected = true
			break
		}
	}
	if !foundRejected {
		t.Fatalf("expected one invocation to record backend rejection, got %#v", invocations)
	}
}

func TestExperimentPlannerRetriesAfterPlannerValidationRejection(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	response := func(fixChampionReason bool) string {
		secondReason := "Keep a compact latency control."
		secondStrategy := "calibration control"
		if fixChampionReason {
			secondReason = "Challenge the champion with a compact latency tradeoff control that can improve macro-F1 without losing deployability."
			secondStrategy = "champion challenge compact tradeoff"
		}
		recommendation := agents.ExperimentPlanningRecommendation{
			Summary:                       "Challenge the current champion with two targeted candidates.",
			DecisionType:                  decisions.TypeAddExperiments,
			Rationale:                     "The champion is strong, so the next fleet should test only candidates with a clear path to beating it.",
			Confidence:                    0.78,
			PlanningMode:                  "champion_challenge",
			DeterministicDiagnosisUsed:    []string{"champion still has measurable macro-F1 headroom"},
			EvidenceUsed:                  []string{"current champion is below target", "prior run plateaued"},
			Hypothesis:                    "A stronger challenger or compact tradeoff challenger can beat the current champion.",
			ExpectedFailureModes:          []string{"stronger model may overfit", "compact control may underperform"},
			DatasetPreprocessingRationale: "Keep preprocessing comparable while challenging the champion on architecture and regularization.",
			ChangedVariables:              []string{"model_family", "augmentation_policy", "regularization"},
			SuccessCriteria:               "Beat the current champion by at least 0.01 macro-F1.",
			StopCondition:                 "Select champion if neither challenger improves enough.",
			DeploymentTradeoff:            "Higher quality challengers must justify latency or preserve compact deployment.",
			CandidateHypotheses: []agents.CandidateHypothesis{
				{
					Hypothesis:           "EfficientNet with augmentation can beat the current champion on macro-F1.",
					PlanningMode:         "champion_challenge",
					Mechanism:            "architecture_challenge",
					Intervention:         "Challenge the champion with a stronger EfficientNet family and moderate augmentation.",
					ProposedChanges:      map[string]any{"model_family": "efficientnet", "augmentation_policy": "moderate"},
					ExpectedEffect:       "Improve macro-F1 through a stronger pretrained family.",
					ExpectedMetricImpact: 0.02,
					ExpectedTradeoffs:    []string{"higher runtime"},
					Risk:                 "medium",
					CostLevel:            "medium",
					NoveltyScore:         0.75,
					EvidenceUsed:         []string{"current champion is below target"},
					ExperimentConfig: plans.PlannedExperiment{
						Template:           "efficientnet_transfer",
						Model:              "efficientnet_b0",
						Epochs:             10,
						BatchSize:          16,
						LearningRate:       0.0003,
						Reason:             "Challenge the champion with a stronger EfficientNet candidate that can improve macro-F1.",
						AugmentationPolicy: "moderate",
						Strategy:           "champion challenge quality tradeoff",
					},
				},
				{
					Hypothesis:           "A compact regularized challenger can test whether the champion is beatable without a latency penalty.",
					PlanningMode:         "champion_challenge",
					Mechanism:            "regularization",
					Intervention:         "Use dropout and AdamW on a compact model as a deployable challenger.",
					ProposedChanges:      map[string]any{"dropout": 0.2, "optimizer": "adamw"},
					ExpectedEffect:       "Improve generalization while preserving compact deployment.",
					ExpectedMetricImpact: 0.012,
					ExpectedTradeoffs:    []string{"possible underfitting"},
					Risk:                 "low",
					CostLevel:            "low",
					NoveltyScore:         0.65,
					EvidenceUsed:         []string{"prior run plateaued"},
					ExperimentConfig: plans.PlannedExperiment{
						Template:     "mobilenet_transfer",
						Model:        "mobilenet_v3_small",
						Epochs:       10,
						BatchSize:    16,
						LearningRate: 0.00025,
						Reason:       secondReason,
						Optimizer:    "adamw",
						Dropout:      0.2,
						Strategy:     secondStrategy,
					},
				},
			},
			WhyCanBeatChampion:      "Each selected experiment changes a meaningful mechanism with a direct champion challenge rationale.",
			ExpectedDeltaVsChampion: 0.02,
			Risks:                   []string{"higher runtime", "regularized compact model may underfit"},
			ExpectedTradeoffs:       []string{"quality for runtime", "compact deployability for smaller upside"},
			NoveltyNotes:            []string{"architecture challenge", "regularization challenge"},
			RejectedOptions:         []agents.RejectedPlannerOption{{Option: "more epochs only", Reason: "not a meaningful champion challenge", Evidence: "prior plateau", AppliesWhen: []string{"plateau"}}},
			Tags:                    []string{"champion_challenge_retry"},
		}
		blob, err := json.Marshal(recommendation)
		if err != nil {
			t.Fatalf("marshal planner response: %v", err)
		}
		return string(blob)
	}
	responses := []string{response(false), response(true)}
	requestBodies := []string{}
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode llm request: %v", err)
		}
		combined := ""
		for _, message := range request.Messages {
			combined += message.Content + "\n"
		}
		requestBodies = append(requestBodies, combined)
		index := len(requestBodies) - 1
		if index >= len(responses) {
			index = len(responses) - 1
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]string{
					"role":    "assistant",
					"content": responses[index],
				},
			}},
		})
	}))
	defer llmServer.Close()

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "test-model")
	t.Setenv("MODEL_EXPRESS_AGENT_MODE", "autonomous")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)

	handled, err := server.runExperimentPlannerAfterTrainingJob(job)
	if err != nil {
		t.Fatalf("run planner: %v", err)
	}
	if !handled {
		t.Fatal("expected planner to handle the completed job")
	}
	if len(requestBodies) != 2 {
		t.Fatalf("expected planner validation retry, got %d requests", len(requestBodies))
	}
	if !strings.Contains(requestBodies[1], "planner_validation_feedback") || !strings.Contains(requestBodies[1], "champion_challenge experiment") {
		t.Fatalf("expected retry prompt to include champion challenge validation feedback, got %s", requestBodies[1])
	}

	agentDecisions := listAgentDecisions(t, server, projectID)
	if len(agentDecisions) != 1 {
		t.Fatalf("expected one accepted decision after retry, got %d", len(agentDecisions))
	}
	if retryCount, _ := agentDecisions[0].Payload["validation_retry_count"].(int); retryCount != 1 {
		t.Fatalf("expected retry count 1 in decision payload, got %#v", agentDecisions[0].Payload["validation_retry_count"])
	}
	experiments := plannedExperimentsFromUnknown(t, agentDecisions[0].Payload["proposed_experiments"])
	if len(experiments) != 2 {
		t.Fatalf("expected two corrected champion challenge experiments, got %#v", experiments)
	}
	if !strings.Contains(strings.ToLower(experiments[1].Reason+" "+experiments[1].Strategy), "champion") {
		t.Fatalf("expected corrected second experiment to explain champion challenge, got %#v", experiments[1])
	}
	projectPlans := listExperimentPlans(t, server, projectID)
	if len(projectPlans) != 2 {
		t.Fatalf("expected corrected retry proposal to schedule a follow-up plan, got %d plans", len(projectPlans))
	}
}

func TestExperimentPlannerResponsesToolCallsAreAuditOnly(t *testing.T) {
	requestCount := 0
	secondRequestBody := ""
	draftRecommendation := experimentPlannerAddExperimentsRecommendation()
	draftBlob, err := json.Marshal(draftRecommendation)
	if err != nil {
		t.Fatalf("marshal draft recommendation: %v", err)
	}
	toolArgs, err := json.Marshal(map[string]string{
		"recommendation_json": string(draftBlob),
	})
	if err != nil {
		t.Fatalf("marshal tool args: %v", err)
	}

	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Fatalf("expected responses endpoint, got %s", r.URL.Path)
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode responses request: %v", err)
		}
		requestCount++
		w.Header().Set("Content-Type", "application/json")
		switch requestCount {
		case 1:
			tools, _ := request["tools"].([]any)
			if len(tools) == 0 {
				t.Fatal("expected planner information tools to be declared")
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id": "resp_planner_tools",
				"output": []map[string]any{
					{
						"type":      "function_call",
						"id":        "fc_validate",
						"call_id":   "call_validate",
						"name":      agents.PlannerToolValidateCandidateExperiments,
						"arguments": string(toolArgs),
					},
					{
						"type":      "function_call",
						"id":        "fc_create_job",
						"call_id":   "call_create_job",
						"name":      "create_job",
						"arguments": `{}`,
					},
				},
			})
		case 2:
			if request["previous_response_id"] != "resp_planner_tools" {
				t.Fatalf("expected previous_response_id resp_planner_tools, got %#v", request["previous_response_id"])
			}
			encoded, _ := json.Marshal(request["input"])
			secondRequestBody = string(encoded)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":                   "resp_planner_final",
				"previous_response_id": "resp_planner_tools",
				"output": []map[string]any{{
					"type": "message",
					"content": []map[string]any{{
						"type": "output_text",
						"text": `{
							"summary": "Wait after bounded information requests.",
							"decision_type": "WAIT",
							"rationale": "The tool calls were informational only; no backend action should be created from them.",
							"confidence": 0.61,
							"risks": [],
							"expected_tradeoffs": [],
							"novelty_notes": [],
							"tags": ["responses_loop", "audit_only"]
						}`,
					}},
				}},
			})
		default:
			t.Fatalf("unexpected responses request %d", requestCount)
		}
	}))
	defer llmServer.Close()

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "openai")
	t.Setenv("MODEL_EXPRESS_LLM_API_STYLE", "responses")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_API_KEY", "test-key")
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "test-model")
	t.Setenv("MODEL_EXPRESS_AGENT_MODE", "autonomous")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)

	handled, err := server.runExperimentPlannerAfterTrainingJob(job)
	if err != nil {
		t.Fatalf("run responses planner: %v", err)
	}
	if !handled {
		t.Fatal("expected planner to handle completed job")
	}
	if requestCount != 2 {
		t.Fatalf("expected tool request plus final request, got %d", requestCount)
	}
	if !strings.Contains(secondRequestBody, "would_schedule_jobs") || !strings.Contains(secondRequestBody, "not allowlisted") {
		t.Fatalf("expected dry-run output and rejected unsupported tool in second request, got %s", secondRequestBody)
	}

	if decisions := listAgentDecisions(t, server, projectID); len(decisions) != 0 {
		t.Fatalf("tool calls must not create decisions, got %#v", decisions)
	}
	if projectPlans := listExperimentPlans(t, server, projectID); len(projectPlans) != 1 {
		t.Fatalf("tool calls must not schedule follow-up plans, got %d plans", len(projectPlans))
	}
	invocations, err := server.store.ListProjectAgentInvocations(projectID, memory.AgentInvocationFilter{AgentName: agents.ExperimentPlannerAgentName})
	if err != nil {
		t.Fatalf("list planner invocations: %v", err)
	}
	if len(invocations) != 1 {
		t.Fatalf("expected one planner invocation, got %d", len(invocations))
	}
	runtime, ok := invocations[0].InputContext["invocation_runtime"].(map[string]any)
	if !ok {
		t.Fatalf("expected invocation runtime audit metadata, got %#v", invocations[0].InputContext["invocation_runtime"])
	}
	if runtime["api_style"] != llm.APIStyleResponses {
		t.Fatalf("expected responses API style, got %#v", runtime["api_style"])
	}
	if runtime["tool_rounds"] != 1 {
		t.Fatalf("expected one tool round, got %#v", runtime["tool_rounds"])
	}
	rejected, _ := runtime["rejected_tool_calls"].([]agents.AgentToolResultTrace)
	if len(rejected) != 1 || rejected[0].Name != "create_job" {
		t.Fatalf("expected rejected create_job tool call in audit metadata, got %#v", runtime["rejected_tool_calls"])
	}
	dryRuns, _ := runtime["dry_run_validation_results"].([]map[string]any)
	if len(dryRuns) == 0 {
		t.Fatalf("expected dry-run validation results in audit metadata, got %#v", runtime["dry_run_validation_results"])
	}
}

func TestExperimentPlannerRejectsDuplicateProposedExperiment(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)

	duplicate := agents.ExperimentPlanningRecommendation{
		AgentName:           agents.ExperimentPlannerAgentName,
		Summary:             "Duplicate plan should not schedule.",
		DecisionType:        decisions.TypeAddExperiments,
		Rationale:           "This intentionally repeats the existing baseline.",
		Confidence:          0.9,
		ProposedExperiments: []plans.PlannedExperiment{plan.Experiments[0]},
	}
	duplicate.ProposedExperiments[0].Mechanism = "baseline_control"
	duplicate.ProposedExperiments[0].Intervention = "Repeat the existing baseline configuration."
	duplicate.ProposedExperiments[0].EvidenceUsed = []string{"existing baseline result"}
	duplicate.ProposedExperiments[0].ExpectedEffect = "Confirm whether the existing baseline is reproducible."
	duplicate.ProposalMechanisms = []agents.PlannerProposalMechanism{{
		ExperimentIndex: 0,
		Mechanism:       duplicate.ProposedExperiments[0].Mechanism,
		Intervention:    duplicate.ProposedExperiments[0].Intervention,
		EvidenceUsed:    duplicate.ProposedExperiments[0].EvidenceUsed,
		ExpectedEffect:  duplicate.ProposedExperiments[0].ExpectedEffect,
	}}

	_, err := experimentPlannerDecisionPayload(duplicate, invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err == nil {
		t.Fatal("expected duplicate proposed experiment to fail validation")
	}
}

func TestExperimentPlannerRejectsProposedExperimentWithoutMechanism(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	recommendation := experimentPlannerAddExperimentsRecommendation()
	recommendation.ProposedExperiments[0].Mechanism = ""
	recommendation.ProposalMechanisms = nil

	_, err := experimentPlannerDecisionPayload(recommendation, invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err == nil || !strings.Contains(err.Error(), "missing mechanism") {
		t.Fatalf("expected missing mechanism validation error, got %v", err)
	}
}

func TestExperimentPlannerRejectsMinorOnlyRepeat(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		func() plans.PlannedExperiment {
			experiment := testExperiment("efficientnet_b1", 12)
			experiment.Template = "efficientnet_transfer"
			return experiment
		}(),
	})
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	repeat := testExperiment("efficientnet_b1", 20)
	repeat.Template = "efficientnet_transfer"
	repeat.BatchSize = plan.Experiments[0].BatchSize
	repeat.LearningRate = plan.Experiments[0].LearningRate
	repeat.Mechanism = "capacity_finetune"
	repeat.Intervention = "Run the same EfficientNet-B1 setup for additional epochs."
	repeat.EvidenceUsed = []string{"prior EfficientNet-B1 run was inconclusive"}
	repeat.ExpectedEffect = "Check whether longer training alone improves macro-F1."
	repeat.Reason = "Try the same mechanism for more epochs."

	recommendation := agents.ExperimentPlanningRecommendation{
		AgentName:           agents.ExperimentPlannerAgentName,
		Summary:             "More epochs only should not schedule.",
		DecisionType:        decisions.TypeAddExperiments,
		Rationale:           "This intentionally repeats the existing mechanism with only more epochs.",
		Confidence:          0.7,
		ProposedExperiments: []plans.PlannedExperiment{repeat},
		ProposalMechanisms: []agents.PlannerProposalMechanism{{
			ExperimentIndex: 0,
			Mechanism:       repeat.Mechanism,
			Intervention:    repeat.Intervention,
			EvidenceUsed:    repeat.EvidenceUsed,
			ExpectedEffect:  repeat.ExpectedEffect,
		}},
	}

	_, err := experimentPlannerDecisionPayload(recommendation, invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err == nil || !strings.Contains(err.Error(), "minor tuning knobs") {
		t.Fatalf("expected minor-only repeat validation error, got %v", err)
	}
}

func TestEnsureFollowUpPlanRevalidatesStaleDecisionPayload(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		func() plans.PlannedExperiment {
			experiment := testExperiment("efficientnet_b1", 12)
			experiment.Template = "efficientnet_transfer"
			return experiment
		}(),
	})
	repeat := testExperiment("efficientnet_b1", 20)
	repeat.Template = "efficientnet_transfer"
	repeat.BatchSize = plan.Experiments[0].BatchSize
	repeat.LearningRate = plan.Experiments[0].LearningRate
	repeat.Reason = "Stale decision repeats EfficientNet-B1 with more epochs."
	novel := testExperiment("mobilenet_v3_large", 10)
	novel.ImageSize = 224
	novel.ResolutionStrategy = "low_latency"
	novel.Optimizer = "adamw"
	novel.Scheduler = "cosine"
	novel.AugmentationPolicy = "light"
	novel.ClassBalancing = "class_balanced_sampler"
	novel.SamplingStrategy = "class_balanced_sampler"
	novel.Pretrained = true
	novel.FreezeBackbone = true
	novel.FineTuneStrategy = "head_only"
	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "old mixed-quality ADD_EXPERIMENTS payload", map[string]any{
		"proposed_experiments": []plans.PlannedExperiment{repeat, novel},
	})
	if err != nil {
		t.Fatalf("create stale decision: %v", err)
	}

	followUp, created, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err != nil {
		t.Fatalf("ensure follow-up plan: %v", err)
	}
	if !created {
		t.Fatal("expected a new follow-up plan after filtering stale payload")
	}
	if len(followUp.Experiments) != 1 || followUp.Experiments[0].Model != "mobilenet_v3_large" {
		t.Fatalf("expected only novel experiment to survive stale payload filtering, got %#v", followUp.Experiments)
	}
	if len(followUp.Warnings) == 0 || !strings.Contains(strings.Join(followUp.Warnings, " "), "minor tuning knobs") {
		t.Fatalf("expected warning for filtered stale repeat, got %#v", followUp.Warnings)
	}
}

func TestEnsureFollowUpPlanBlocksWhenAllExperimentsFiltered(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	first := testExperiment("efficientnet_b1", 12)
	first.Template = "efficientnet_transfer"
	second := testExperiment("resnet18", 10)
	second.Template = "resnet_transfer"
	second.LearningRate = 0.0002
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{first, second})
	repeatFirst := first
	repeatFirst.Epochs = 24
	repeatFirst.Reason = "Stale payload only extends EfficientNet-B1 epochs."
	repeatSecond := second
	repeatSecond.Reason = "Stale payload repeats ResNet-18 exactly."
	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "old all-repeat ADD_EXPERIMENTS payload", map[string]any{
		"proposed_experiments": []plans.PlannedExperiment{repeatFirst, repeatSecond},
	})
	if err != nil {
		t.Fatalf("create stale decision: %v", err)
	}

	_, _, err = server.ensureFollowUpPlan(projectID, plan, decision)
	if err == nil || !errors.Is(err, errNoNovelFollowUpExperiments) {
		t.Fatalf("expected no-novel follow-up error, got %v", err)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected no follow-up plan to be created, got %d total plans", got)
	}
	projectJobs, err := server.store.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	if len(projectJobs) != 0 {
		t.Fatalf("expected no jobs to be created, got %d", len(projectJobs))
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasBlockedFollowUpEvent(events) {
		t.Fatalf("expected blocked follow-up event, got %#v", events)
	}
}

func TestEnsureFollowUpPlanAllowsRepeatExperimentsInRelaxedMode(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "false")

	first := testExperiment("efficientnet_b1", 12)
	first.Template = "efficientnet_transfer"
	second := testExperiment("resnet18", 10)
	second.Template = "resnet_transfer"
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{first, second})
	repeatFirst := first
	repeatFirst.Epochs = 24
	repeatFirst.Reason = "Stale payload extends EfficientNet-B1 epochs."
	repeatSecond := second
	repeatSecond.Reason = "Stale payload repeats ResNet-18 exactly."
	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "relaxed all-repeat ADD_EXPERIMENTS payload", map[string]any{
		"proposed_experiments": []plans.PlannedExperiment{repeatFirst, repeatSecond},
	})
	if err != nil {
		t.Fatalf("create repeat decision: %v", err)
	}

	followUp, created, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err != nil {
		t.Fatalf("expected relaxed follow-up plan, got %v", err)
	}
	if !created || len(followUp.Experiments) != 2 {
		t.Fatalf("expected both repeat experiments to be retained, created=%v experiments=%#v", created, followUp.Experiments)
	}
	warnings := strings.Join(followUp.Warnings, " ")
	if !strings.Contains(warnings, "Relaxed planner validation") || !strings.Contains(warnings, "Allowed relaxed follow-up experiment") {
		t.Fatalf("expected relaxed novelty warning, got %#v", followUp.Warnings)
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if hasBlockedFollowUpEvent(events) {
		t.Fatalf("did not expect blocked follow-up event in relaxed mode, got %#v", events)
	}
}

func TestExistingStaleFollowUpPlanIsRevalidatedBeforeExecution(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	first := testExperiment("efficientnet_b1", 12)
	first.Template = "efficientnet_transfer"
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{first})
	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "old accepted ADD_EXPERIMENTS payload", map[string]any{
		"proposed_experiments": []plans.PlannedExperiment{func() plans.PlannedExperiment {
			repeat := first
			repeat.Epochs = 18
			repeat.Reason = "Existing stale plan repeats EfficientNet-B1 with more epochs."
			return repeat
		}()},
	})
	if err != nil {
		t.Fatalf("create stale decision: %v", err)
	}
	stalePlan, err := server.store.CreateExperimentPlan(projectID, plan.DatasetID, "macro_f1", 1, 5, []plans.PlannedExperiment{func() plans.PlannedExperiment {
		repeat := first
		repeat.Epochs = 18
		return repeat
	}()}, nil, decision.ID)
	if err != nil {
		t.Fatalf("create stale follow-up plan: %v", err)
	}

	_, err = server.executeStoredExperimentPlan(stalePlan.ID, executeExperimentPlanRequest{Provider: "local"})
	if err == nil || !errors.Is(err, errNoNovelFollowUpExperiments) {
		t.Fatalf("expected stale follow-up execution to be blocked, got %v", err)
	}
	projectJobs, err := server.store.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	if len(projectJobs) != 0 {
		t.Fatalf("expected stale plan execution to create no jobs, got %d", len(projectJobs))
	}
}

func TestEnsureFollowUpPlanBlocksBBoxCropWithoutAnnotations(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "bbox_crop_ablation"
	experiment.Intervention = "Compare bbox crop against full-image training."
	experiment.EvidenceUsed = []string{"small objects may be background-dominated"}
	experiment.ExpectedEffect = "Improve object focus if annotated boxes are available."
	experiment.Preprocessing = &plans.Preprocessing{
		CropStrategy: "bbox_crop_ablation",
		BBoxMode:     "crop_and_compare_full_image",
	}
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"crop diagnosis"})

	_, _, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err == nil || !errors.Is(err, errNoNovelFollowUpExperiments) {
		t.Fatalf("expected bbox evidence validation block, got %v", err)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected no follow-up plan to be created, got %d total plans", got)
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasBlockedFollowUpEvent(events) {
		t.Fatalf("expected blocked follow-up event, got %#v", events)
	}
}

func TestEnsureFollowUpPlanAllowsMechanismEvidenceFailureInRelaxedMode(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "bbox_crop_ablation"
	experiment.Intervention = "Compare bbox crop against full-image training."
	experiment.EvidenceUsed = []string{"planner wants to test tighter crops"}
	experiment.ExpectedEffect = "Learn whether crop-focused training helps this dataset."
	experiment.Preprocessing = &plans.Preprocessing{
		CropStrategy: "bbox_crop_ablation",
		BBoxMode:     "crop_and_compare_full_image",
	}
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"planner-only crop hypothesis"})

	followUp, created, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err != nil {
		t.Fatalf("expected relaxed mechanism-evidence follow-up to pass, got %v", err)
	}
	if !created || len(followUp.Experiments) != 1 {
		t.Fatalf("expected one relaxed follow-up experiment, created=%v plan=%#v", created, followUp)
	}
	warnings := strings.Join(followUp.Warnings, " ")
	if !strings.Contains(warnings, "Relaxed planner validation") || !strings.Contains(warnings, "bbox/annotation") {
		t.Fatalf("expected relaxed mechanism-evidence warning, got %#v", followUp.Warnings)
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if hasBlockedFollowUpEvent(events) {
		t.Fatalf("did not expect blocked follow-up event in relaxed mode, got %#v", events)
	}
}

func TestEnsureFollowUpPlanAllowsBBoxCropWithBackendAnnotations(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	if _, err := server.store.UpdateDatasetProfile(plan.DatasetID, map[string]any{
		"metadata_summary": map[string]any{"bbox_available": true, "bbox_annotations_count": 12},
		"dataset_traits":   []string{"bbox_available", "small_objects"},
	}); err != nil {
		t.Fatalf("update dataset profile: %v", err)
	}
	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "bbox_crop_ablation"
	experiment.Intervention = "Compare bbox crop against full-image training."
	experiment.EvidenceUsed = []string{"backend profile reports bbox annotations", "visual traits show small objects"}
	experiment.ExpectedEffect = "Improve foreground focus without changing labels."
	experiment.Preprocessing = &plans.Preprocessing{
		CropStrategy: "bbox_crop_ablation",
		BBoxMode:     "crop_and_compare_full_image",
	}
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"bbox annotations available"})

	followUp, created, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err != nil {
		t.Fatalf("expected bbox-backed follow-up to pass, got %v", err)
	}
	if !created || len(followUp.Experiments) != 1 {
		t.Fatalf("expected one follow-up experiment, got created=%v plan=%#v", created, followUp)
	}
}

func TestEnsureFollowUpPlanBlocksAuditOnlyMechanismTrainingJob(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "label_noise_audit"
	experiment.Intervention = "Audit high-loss examples for likely label noise."
	experiment.EvidenceUsed = []string{"training dynamics show unstable per-class errors"}
	experiment.ExpectedEffect = "Produce an audit report before training changes."
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"label quality concerns"})

	_, _, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err == nil || !strings.Contains(err.Error(), "report-only") {
		t.Fatalf("expected report-only mechanism to be blocked, got %v", err)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected no follow-up plan to be created, got %d total plans", got)
	}
}

func TestLabelQualityAuditPlanCreatesAuditJobNotTrainingJob(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	audit := plans.PlannedExperiment{
		Template:       jobs.TemplateLabelQualityAudit,
		Mechanism:      "label_noise_audit",
		Intervention:   "Produce a report-only audit of high-confidence mistakes and high-loss examples.",
		EvidenceUsed:   []string{"label_quality_card reports asymmetric high-confidence errors"},
		ExpectedEffect: "Create a label-quality artifact before any training changes.",
		Reason:         "Label noise should be audited before training work is scheduled.",
	}
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{audit}, []string{"label quality concerns"})

	followUp, created, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err != nil {
		t.Fatalf("expected label-quality audit follow-up to pass, got %v", err)
	}
	if !created || len(followUp.Experiments) != 1 {
		t.Fatalf("expected one audit follow-up experiment, got created=%v plan=%#v", created, followUp)
	}

	result, err := server.executeStoredExperimentPlan(followUp.ID, executeExperimentPlanRequest{Provider: "local"})
	if err != nil {
		t.Fatalf("execute audit plan: %v", err)
	}
	if len(result.Jobs) != 1 {
		t.Fatalf("expected one audit job, got %d", len(result.Jobs))
	}
	job := result.Jobs[0]
	if job.Template != jobs.TemplateLabelQualityAudit {
		t.Fatalf("expected %s job, got %s", jobs.TemplateLabelQualityAudit, job.Template)
	}
	if configString(job.Config, "audit_type") != "label_noise_audit" {
		t.Fatalf("expected label_noise_audit config, got %#v", job.Config)
	}
	if _, ok := job.Config["model"]; ok {
		t.Fatalf("audit job should not carry a model training config, got %#v", job.Config)
	}
}

func TestMixedSampleAugmentationMechanismPassesWithStructuredPolicy(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "augmentation_mixed_sample"
	experiment.Intervention = "Use MixUp to smooth labels and reduce overconfident errors."
	experiment.EvidenceUsed = []string{"validation errors show overconfident confusion between visually similar classes"}
	experiment.ExpectedEffect = "Improve calibration and macro-F1 without switching model families."
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{
		PolicyType:  "mixup",
		Probability: 0.5,
		Alpha:       0.4,
	}
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"overconfident errors"})

	followUp, created, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err != nil {
		t.Fatalf("expected MixUp-backed mechanism to pass, got %v", err)
	}
	if !created || followUp.Experiments[0].AugmentationPolicyConfig.PolicyType != "mixup" {
		t.Fatalf("expected MixUp follow-up experiment, got %#v", followUp)
	}
}

func TestEffectiveNumberClassBalancingPassesWithBackendEvidence(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	if _, err := server.store.UpdateDatasetProfile(plan.DatasetID, map[string]any{
		"class_distribution": map[string]any{"common": 90, "rare": 12},
	}); err != nil {
		t.Fatalf("update dataset profile: %v", err)
	}
	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "class_imbalance"
	experiment.Intervention = "Use effective-number class-balanced loss for rare classes."
	experiment.EvidenceUsed = []string{"dataset profile shows common/rare class imbalance"}
	experiment.ExpectedEffect = "Improve minority recall while preserving macro-F1."
	experiment.ClassBalancing = "effective_number_loss"
	experiment.ClassBalancingConfig = map[string]any{"effective_number_beta": 0.99}
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"class imbalance evidence"})

	followUp, created, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err != nil {
		t.Fatalf("expected effective-number class-balanced follow-up to pass, got %v", err)
	}
	if !created || followUp.Experiments[0].ClassBalancing != "effective_number_loss" {
		t.Fatalf("expected effective-number follow-up experiment, got %#v", followUp)
	}
}

func TestEnsureFollowUpPlanBlocksHighResolutionWithoutObjectScaleEvidence(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "resolution_crop"
	experiment.Intervention = "Try a larger input size."
	experiment.EvidenceUsed = []string{"the current score is below target"}
	experiment.ExpectedEffect = "Check whether a larger input improves validation quality."
	experiment.ImageSize = 384
	experiment.ResolutionStrategy = "high_resolution_ablation"
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"quality is below target"})

	_, _, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err == nil || !strings.Contains(err.Error(), "object-scale") {
		t.Fatalf("expected high-resolution evidence validation block, got %v", err)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected no follow-up plan to be created, got %d total plans", got)
	}
}

func TestEnsureFollowUpPlanAllowsResolutionCropLinkedToAcceptedVisualHypothesis(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	analysis, err := server.store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      projectID,
		DatasetID:      plan.DatasetID,
		ImagesAnalyzed: 48,
		TriggerReason:  datasets.VisualTriggerInitialProfile,
		CoverageReport: datasets.VisualCoverageReport{
			ImagesAnalyzed:     48,
			ClassesTotal:       120,
			ClassesCovered:     39,
			ClassCoverageRatio: 0.325,
		},
		VisualTraits: []datasets.VisualTrait{
			{
				Trait:      "large_objects",
				Level:      "medium",
				Confidence: "medium",
				Evidence:   []string{"sampled subjects occupy large regions with visible background context"},
			},
		},
		PreprocessingHypotheses: []datasets.PreprocessingHypothesis{
			{
				ID:             "vh_001",
				Mechanism:      "resolution_crop",
				Summary:        "Use subject-centric crops to reduce background context around large dogs.",
				Evidence:       []string{"dogs are prominent but backgrounds remain visible in multiple sampled images"},
				ExpectedEffect: "Reduce background distraction while preserving breed morphology.",
				Confidence:     "medium",
				SupportStatus:  "needs_backend_validation",
			},
		},
		Limitations: []string{"bounded visual sample"},
	})
	if err != nil {
		t.Fatalf("create accepted visual analysis: %v", err)
	}

	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "resolution_crop"
	experiment.Intervention = "Use preserve-aspect resize with a moderate center crop linked to visual hypothesis vh_001."
	experiment.EvidenceUsed = []string{"visual hypothesis vh_001"}
	experiment.ExpectedEffect = "Improve breed-focused learning by reducing background context."
	experiment.ImageSize = 288
	experiment.ResolutionStrategy = "compare_224_256"
	experiment.Preprocessing = &plans.Preprocessing{
		ResizeStrategy: "preserve_aspect_pad",
		Normalization:  "imagenet",
		CropStrategy:   "center_crop",
		BBoxMode:       "ignore",
	}
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"accepted visual analysis " + analysis.ID})

	followUp, created, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err != nil {
		t.Fatalf("expected visual-hypothesis-linked resolution crop to pass, got %v", err)
	}
	if !created || len(followUp.Experiments) != 1 {
		t.Fatalf("expected one linked follow-up experiment, got created=%v plan=%#v", created, followUp)
	}
	evidenceBlob := strings.Join(followUp.Experiments[0].EvidenceUsed, " ")
	if !strings.Contains(evidenceBlob, analysis.ID) || !strings.Contains(evidenceBlob, "vh_001") || !strings.Contains(evidenceBlob, "large_objects") {
		t.Fatalf("expected stored experiment evidence to include accepted visual analysis details, got %#v", followUp.Experiments[0].EvidenceUsed)
	}
}

func TestEnsureFollowUpPlanBlocksUnsupportedVisualHypothesisEvidence(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	if _, err := server.store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      projectID,
		DatasetID:      plan.DatasetID,
		ImagesAnalyzed: 24,
		TriggerReason:  datasets.VisualTriggerInitialProfile,
		PreprocessingHypotheses: []datasets.PreprocessingHypothesis{
			{
				ID:             "vh_999",
				Mechanism:      "resolution_crop",
				Summary:        "Unsupported crop suggestion should remain evidence-only.",
				Evidence:       []string{"crop evidence should be ignored while unsupported"},
				ExpectedEffect: "Would change crop behavior if it were supported.",
				Confidence:     "medium",
				SupportStatus:  "unsupported",
			},
		},
	}); err != nil {
		t.Fatalf("create unsupported visual analysis: %v", err)
	}

	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "resolution_crop"
	experiment.Intervention = "Try high-resolution crop based only on visual hypothesis vh_999."
	experiment.EvidenceUsed = []string{"visual hypothesis vh_999"}
	experiment.ExpectedEffect = "Check whether higher-resolution cropping improves validation quality."
	experiment.ImageSize = 384
	experiment.ResolutionStrategy = "high_resolution_ablation"
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"visual hypothesis vh_999"})

	_, _, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err == nil || !strings.Contains(err.Error(), "object-scale") {
		t.Fatalf("expected unsupported visual hypothesis not to satisfy mechanism evidence, got %v", err)
	}
}

func TestEnsureFollowUpPlanAllowsMixedSampleLinkedToAcceptedVisualHypothesisWithPolicy(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	analysis, err := server.store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      projectID,
		DatasetID:      plan.DatasetID,
		ImagesAnalyzed: 48,
		TriggerReason:  datasets.VisualTriggerInitialProfile,
		VisualTraits: []datasets.VisualTrait{
			{
				Trait:      "fine_grained_similarity",
				Level:      "high",
				Confidence: "medium",
				Evidence:   []string{"sampled classes include visually similar breeds"},
			},
		},
		PreprocessingHypotheses: []datasets.PreprocessingHypothesis{
			{
				ID:             "vh_002",
				Mechanism:      "augmentation_mixed_sample",
				Summary:        "MixUp or CutMix may smooth decision boundaries among similar classes.",
				Evidence:       []string{"fine-grained similarity appears high in the visual sample"},
				ExpectedEffect: "Improve calibration across similar classes.",
				Confidence:     "medium",
				SupportStatus:  "needs_backend_validation",
			},
		},
	})
	if err != nil {
		t.Fatalf("create accepted visual analysis: %v", err)
	}

	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "augmentation_mixed_sample"
	experiment.Intervention = "Use MixUp linked to visual hypothesis vh_002 for visually similar classes."
	experiment.EvidenceUsed = []string{"visual hypothesis vh_002"}
	experiment.ExpectedEffect = "Improve calibration and reduce overconfident similar-class errors."
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{
		PolicyType:  "mixup",
		Probability: 0.5,
		Alpha:       0.3,
	}
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"accepted visual analysis " + analysis.ID})

	followUp, created, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err != nil {
		t.Fatalf("expected MixUp visual-hypothesis-linked follow-up to pass, got %v", err)
	}
	if !created || len(followUp.Experiments) != 1 {
		t.Fatalf("expected one MixUp follow-up experiment, got created=%v plan=%#v", created, followUp)
	}
	evidenceBlob := strings.Join(followUp.Experiments[0].EvidenceUsed, " ")
	if !strings.Contains(evidenceBlob, analysis.ID) || !strings.Contains(evidenceBlob, "vh_002") || !strings.Contains(evidenceBlob, "fine_grained_similarity") {
		t.Fatalf("expected MixUp experiment evidence to include accepted visual analysis details, got %#v", followUp.Experiments[0].EvidenceUsed)
	}
}

func TestEnsureFollowUpPlanBlocksMixedSampleVisualHypothesisWithoutPolicy(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	if _, err := server.store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      projectID,
		DatasetID:      plan.DatasetID,
		ImagesAnalyzed: 48,
		TriggerReason:  datasets.VisualTriggerInitialProfile,
		PreprocessingHypotheses: []datasets.PreprocessingHypothesis{
			{
				ID:             "vh_003",
				Mechanism:      "augmentation_mixed_sample",
				Summary:        "Mixed-sample augmentation may help similar classes.",
				Evidence:       []string{"fine-grained similarity appears high in the visual sample"},
				ExpectedEffect: "Improve calibration across similar classes.",
				Confidence:     "medium",
				SupportStatus:  "needs_backend_validation",
			},
		},
	}); err != nil {
		t.Fatalf("create accepted visual analysis: %v", err)
	}

	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "augmentation_mixed_sample"
	experiment.Intervention = "Use mixed-sample augmentation linked to visual hypothesis vh_003."
	experiment.EvidenceUsed = []string{"visual hypothesis vh_003"}
	experiment.ExpectedEffect = "Improve calibration and reduce similar-class confusion."
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"visual hypothesis vh_003"})

	_, _, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err == nil || !strings.Contains(err.Error(), "MixUp or CutMix") {
		t.Fatalf("expected visual hypothesis without structured mixed-sample policy to be blocked, got %v", err)
	}
}

func TestEnsureFollowUpPlanAllowsDiagnosisMatchedClassImbalanceMechanism(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	if _, err := server.store.UpdateDatasetProfile(plan.DatasetID, map[string]any{
		"class_distribution": map[string]any{"cat": 180, "dog": 24},
		"imbalance_ratio":    7.5,
	}); err != nil {
		t.Fatalf("update dataset profile: %v", err)
	}
	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "class_imbalance"
	experiment.Intervention = "Train with weighted loss to target rare-class recall."
	experiment.EvidenceUsed = []string{"backend profile imbalance_ratio=7.5", "minority recall is weak"}
	experiment.ExpectedEffect = "Improve macro-F1 by improving rare-class recall."
	experiment.ClassBalancing = "weighted_loss"
	experiment.SamplingStrategy = "none"
	experiment.AugmentationPolicy = "randaugment"
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{
		PolicyType: "randaugment",
		Magnitude:  9,
		NumOps:     2,
	}
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"class imbalance diagnosis"})

	followUp, created, err := server.ensureFollowUpPlan(projectID, plan, decision)
	if err != nil {
		t.Fatalf("expected diagnosis-matched class imbalance follow-up to pass, got %v", err)
	}
	if !created || len(followUp.Experiments) != 1 {
		t.Fatalf("expected one follow-up experiment, got created=%v plan=%#v", created, followUp)
	}
}

func TestExistingStaleFollowUpPlanRevalidatesMechanismDatasetEvidenceBeforeExecution(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	experiment := testExperiment("efficientnet_b1", 10)
	experiment.Template = "efficientnet_transfer"
	experiment.Mechanism = "bbox_crop_ablation"
	experiment.Intervention = "Stale bbox crop proposal without annotations."
	experiment.EvidenceUsed = []string{"old planner claimed boxes existed"}
	experiment.ExpectedEffect = "Should now be blocked by backend evidence validation."
	experiment.Preprocessing = &plans.Preprocessing{
		CropStrategy: "bbox_crop_ablation",
		BBoxMode:     "crop_and_compare_full_image",
	}
	decision := createLLMAddExperimentsDecision(t, server, projectID, plan.ID, []plans.PlannedExperiment{experiment}, []string{"stale bbox evidence"})
	stalePlan, err := server.store.CreateExperimentPlan(projectID, plan.DatasetID, "macro_f1", 1, 5, []plans.PlannedExperiment{experiment}, nil, decision.ID)
	if err != nil {
		t.Fatalf("create stale follow-up plan: %v", err)
	}

	_, err = server.executeStoredExperimentPlan(stalePlan.ID, executeExperimentPlanRequest{Provider: "local"})
	if err == nil || !errors.Is(err, errNoNovelFollowUpExperiments) {
		t.Fatalf("expected stale bbox follow-up execution to be blocked, got %v", err)
	}
	projectJobs, err := server.store.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	if len(projectJobs) != 0 {
		t.Fatalf("expected stale plan execution to create no jobs, got %d", len(projectJobs))
	}
}

func TestExperimentPlannerInputTracksChampionAndNoImprovementRounds(t *testing.T) {
	server, projectID, initialPlan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	initialJob, _ := createTerminalTrainingJob(t, server, initialPlan, initialPlan.Experiments[0], jobs.StatusSucceeded, 0.70)

	firstFollowUp, err := server.store.CreateExperimentPlan(projectID, initialPlan.DatasetID, "macro_f1", 1, 5, []plans.PlannedExperiment{
		testExperiment("efficientnet_b0", 10),
	}, nil, "decision_1")
	if err != nil {
		t.Fatalf("create first follow-up plan: %v", err)
	}
	createTerminalTrainingJob(t, server, firstFollowUp, firstFollowUp.Experiments[0], jobs.StatusSucceeded, 0.62)

	secondFollowUp, err := server.store.CreateExperimentPlan(projectID, initialPlan.DatasetID, "macro_f1", 1, 5, []plans.PlannedExperiment{
		testExperiment("resnet34", 10),
	}, nil, "decision_2")
	if err != nil {
		t.Fatalf("create second follow-up plan: %v", err)
	}
	createTerminalTrainingJob(t, server, secondFollowUp, secondFollowUp.Experiments[0], jobs.StatusSucceeded, 0.61)

	input, ready, err := server.buildExperimentPlannerInput(projectID, secondFollowUp.ID)
	if err != nil {
		t.Fatalf("build planner input: %v", err)
	}
	if !ready {
		t.Fatal("expected planner input to be ready")
	}
	if input.CurrentChampion == nil {
		t.Fatal("expected current champion context")
	}
	if input.CurrentChampion.JobID != initialJob.ID {
		t.Fatalf("expected initial job %s to remain champion, got %s", initialJob.ID, input.CurrentChampion.JobID)
	}
	if input.SourcePlanBaselineChampion == nil || input.SourcePlanBaselineChampion.JobID != initialJob.ID {
		t.Fatalf("expected latest plan baseline champion to be %s, got %#v", initialJob.ID, input.SourcePlanBaselineChampion)
	}
	if input.NoImprovementRounds != 2 {
		t.Fatalf("expected two no-improvement follow-up rounds, got %d", input.NoImprovementRounds)
	}
	if len(input.SourcePlanDeltas) != 1 {
		t.Fatalf("expected one source plan delta, got %d", len(input.SourcePlanDeltas))
	}
	if input.SourcePlanDeltas[0].DeltaScoreVsChampion >= 0 {
		t.Fatalf("expected latest follow-up to trail the champion, got delta %.3f", input.SourcePlanDeltas[0].DeltaScoreVsChampion)
	}
}

func TestExperimentPlannerInputUsesLossHeavyReadinessChampion(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
		testExperiment("efficientnet_b0", 6),
	})
	badJob, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.96)
	healthyJob, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[1], jobs.StatusSucceeded, 0.90)

	badTrainLoss := 3.0
	badValLoss := 3.4
	if _, err := server.store.UpsertTrainingRunSummary(badJob.ID, runs.TrainingRunSummaryUpdate{
		FinalTrainLoss: &badTrainLoss,
		FinalValLoss:   &badValLoss,
	}); err != nil {
		t.Fatalf("upsert bad losses: %v", err)
	}
	goodTrainLoss := 0.16
	goodValLoss := 0.21
	if _, err := server.store.UpsertTrainingRunSummary(healthyJob.ID, runs.TrainingRunSummaryUpdate{
		FinalTrainLoss: &goodTrainLoss,
		FinalValLoss:   &goodValLoss,
	}); err != nil {
		t.Fatalf("upsert healthy losses: %v", err)
	}
	if _, err := server.store.UpsertTrainingRunEvaluation(badJob.ID, runs.TrainingRunEvaluationUpdate{
		ObjectiveProfile: map[string]any{
			"heldout_test_macro_f1": 0.60,
			"heldout_test_accuracy": 0.62,
			"heldout_test_loss":     3.2,
		},
		PerClassMetrics: map[string]any{
			"cat":       map[string]any{"recall": 0.43, "f1-score": 0.48},
			"dog":       map[string]any{"recall": 0.50, "f1-score": 0.52},
			"macro avg": map[string]any{"f1-score": 0.50},
		},
		ConfusionMatrix: [][]int{{9, 21}, {15, 15}},
		ModelProfile:    map[string]any{"class_labels": []string{"cat", "dog"}},
		HolisticScores:  map[string]any{"overall_score": 0.52},
	}); err != nil {
		t.Fatalf("upsert bad evaluation: %v", err)
	}
	if _, err := server.store.UpsertTrainingRunEvaluation(healthyJob.ID, runs.TrainingRunEvaluationUpdate{
		ObjectiveProfile: map[string]any{
			"heldout_test_macro_f1": 0.89,
			"heldout_test_accuracy": 0.90,
			"heldout_test_loss":     0.23,
		},
		PerClassMetrics: map[string]any{
			"cat":       map[string]any{"recall": 0.90, "f1-score": 0.90},
			"dog":       map[string]any{"recall": 0.88, "f1-score": 0.89},
			"macro avg": map[string]any{"f1-score": 0.895},
		},
		ConfusionMatrix: [][]int{{27, 3}, {4, 26}},
		ModelProfile:    map[string]any{"class_labels": []string{"cat", "dog"}},
		HolisticScores:  map[string]any{"overall_score": 0.90},
	}); err != nil {
		t.Fatalf("upsert healthy evaluation: %v", err)
	}

	input, ready, err := server.buildExperimentPlannerInput(projectID, plan.ID)
	if err != nil {
		t.Fatalf("build planner input: %v", err)
	}
	if !ready {
		t.Fatal("expected planner input to be ready")
	}
	if input.CurrentChampion == nil || input.CurrentChampion.JobID != healthyJob.ID {
		t.Fatalf("expected healthy-loss job %s as planner champion, got %#v", healthyJob.ID, input.CurrentChampion)
	}
	if input.CurrentChampion.ScoreBasis != "loss_heavy_deployment_readiness" {
		t.Fatalf("expected readiness score basis, got %#v", input.CurrentChampion)
	}
	if input.CurrentChampion.FinalValLoss != goodValLoss {
		t.Fatalf("expected champion loss context, got %#v", input.CurrentChampion)
	}
}

func TestExperimentPlannerInputHandlesEmptyMemoryStore(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_LOG_ONLY", "false")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.70)

	input, ready, err := server.buildExperimentPlannerInput(projectID, plan.ID)
	if err != nil {
		t.Fatalf("build planner input with empty memory: %v", err)
	}
	if !ready {
		t.Fatal("expected planner input to be ready with empty memory")
	}
	if len(input.PriorMemory) != 0 {
		t.Fatalf("expected no prior memory, got %d", len(input.PriorMemory))
	}
	if len(input.SuccessfulStrategyMemory) != 0 || len(input.FailedStrategyMemory) != 0 || len(input.RejectedStrategyMemory) != 0 {
		t.Fatalf("expected empty strategy memory, got success=%d failed=%d rejected=%d", len(input.SuccessfulStrategyMemory), len(input.FailedStrategyMemory), len(input.RejectedStrategyMemory))
	}
	if len(input.RetrievedMemory) != 0 {
		t.Fatalf("expected no retrieved memory, got %d", len(input.RetrievedMemory))
	}
	if len(input.StrategyScorecards) != 0 {
		t.Fatalf("expected no strategy scorecards, got %d", len(input.StrategyScorecards))
	}
}

func TestExperimentPlannerMemoryRetrievalLogOnlyRecordsProbeWithoutPromptInjection(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_LOG_ONLY", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "false")
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_MIN_SCORE", "0.01")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.70)

	if _, err := server.store.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
		SourceTable:         memory.SourceAgentMemoryRecord,
		SourceID:            "memory_probe_1",
		ProjectID:           projectID,
		DatasetID:           plan.DatasetID,
		PlanID:              plan.ID,
		Kind:                memory.KindPlanningOutcome,
		Scope:               memory.ScopeDataset,
		EmbeddingModel:      "test-embedding",
		EmbeddingDimensions: 3,
		Embedding:           []float32{1, 0, 0},
		EmbeddingText:       "class imbalance weighted loss improved macro f1 for image classification",
		SummaryCard: map[string]any{
			"summary":      "Weighted loss helped minority classes.",
			"lesson":       "Use class balancing when macro F1 lags accuracy.",
			"storage_uri":  "s3://private-bucket/raw-card.json",
			"local_path":   `C:\Users\Sahith\secret-card.json`,
			"raw_prompt":   "hidden planner prompt",
			"base64_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg",
		},
		Metadata: map[string]any{
			"mechanism":                  "class_balancing",
			"intervention":               "weighted_cross_entropy",
			"outcome_status":             agents.ExperimentPlanningOutcomeImprovedChampion,
			"accepted_for_vector_memory": true,
			"token":                      "sk-secret-value",
		},
		QualityScore: 0.90,
		OutcomeScore: 0.80,
	}); err != nil {
		t.Fatalf("upsert memory embedding: %v", err)
	}

	input, ready, err := server.buildExperimentPlannerInput(projectID, plan.ID)
	if err != nil {
		t.Fatalf("build planner input: %v", err)
	}
	if !ready {
		t.Fatal("expected planner input to be ready")
	}
	if len(input.RetrievedMemory) != 0 {
		t.Fatalf("log-only retrieval should not inject prompt memory, got %d", len(input.RetrievedMemory))
	}

	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	var retrievalEvent *execution.ExecutionEvent
	for i := range events {
		if events[i].EventType == execution.EventMemoryRetrievalLogged {
			retrievalEvent = &events[i]
			break
		}
	}
	if retrievalEvent == nil {
		t.Fatalf("expected %s execution event, got %#v", execution.EventMemoryRetrievalLogged, events)
	}
	if retrievalEvent.Payload["log_only"] != true {
		t.Fatalf("expected log_only payload, got %#v", retrievalEvent.Payload)
	}
	if count := payloadFloat(retrievalEvent.Payload, "retrieved_count"); count < 1 {
		t.Fatalf("expected retrieved_count >= 1, got %#v", retrievalEvent.Payload)
	}
	cards, ok := retrievalEvent.Payload["retrieved_cards"].([]map[string]any)
	if !ok || len(cards) != 1 {
		t.Fatalf("expected one diagnostic card, got %#v", retrievalEvent.Payload["retrieved_cards"])
	}
	blob, err := json.Marshal(retrievalEvent.Payload)
	if err != nil {
		t.Fatalf("marshal retrieval payload: %v", err)
	}
	for _, leaked := range []string{"s3://private-bucket", `C:\Users\Sahith`, "hidden planner prompt", "/9j/4AAQSkZJRg", "sk-secret-value"} {
		if strings.Contains(string(blob), leaked) {
			t.Fatalf("expected retrieval diagnostics to avoid leaking %q, got %s", leaked, blob)
		}
	}
}

func TestMemoryRetrievalLogOnlySkipsEmbeddingProviderWhenLexicalFallbackIsAllowed(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_LOG_ONLY", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_LOG_ONLY_EMBEDDINGS", "false")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL", "test-embedding")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_BASE_URL", "")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_DIMENSIONS", "3")
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_MIN_SCORE", "0.01")

	embeddingCalls := 0
	embeddingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		embeddingCalls++
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "test-embedding",
			"data": []map[string]any{{
				"embedding": []float32{1, 0, 0},
			}},
		})
	}))
	defer embeddingServer.Close()
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_BASE_URL", embeddingServer.URL)

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("retrieval demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	for i := 0; i < 3; i++ {
		if _, err := memoryStore.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
			SourceTable:         memory.SourceAgentMemoryRecord,
			SourceID:            "memory_" + strconv.Itoa(i),
			ProjectID:           project.ID,
			DatasetID:           dataset.ID,
			Kind:                memory.KindPlanningOutcome,
			Scope:               memory.ScopeDataset,
			EmbeddingModel:      "test-embedding",
			EmbeddingDimensions: 3,
			Embedding:           []float32{1, 0, 0},
			EmbeddingText:       "class balancing improved minority recall and weighted loss helped",
			Metadata: map[string]any{
				"mechanism":                  "class_balancing",
				"accepted_for_vector_memory": true,
			},
		}); err != nil {
			t.Fatalf("seed embedding %d: %v", i, err)
		}
	}

	server := newServer(memoryStore)
	results, usage := server.searchRetrievedMemory(context.Background(), memory.MemoryRetrievalQuery{
		ProjectID: project.ID,
		DatasetID: dataset.ID,
		AgentName: agents.ExperimentPlannerAgentName,
		Purpose:   "experiment_planner",
		Text:      "class balancing improved minority recall",
		Kinds:     []string{memory.KindPlanningOutcome},
		Limit:     5,
	}, "plan_1", "")
	if embeddingCalls != 0 {
		t.Fatalf("expected no embedding provider calls, got %d", embeddingCalls)
	}
	if !usage.LogOnly || !usage.Skipped || usage.Injected {
		t.Fatalf("unexpected log-only usage: %#v", usage)
	}
	if usage.ProviderCallCount != 0 {
		t.Fatalf("expected zero provider calls, got %#v", usage)
	}
	if len(results) == 0 {
		t.Fatal("expected lexical fallback to return retrieved memory")
	}
}

func TestMemoryRetrievalQueryCacheAvoidsRepeatedProviderCalls(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_LOG_ONLY", "false")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL", "test-embedding")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_DIMENSIONS", "3")
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_MIN_SCORE", "0.01")

	embeddingCalls := 0
	embeddingServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		embeddingCalls++
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "test-embedding",
			"data": []map[string]any{{
				"embedding": []float32{1, 0, 0},
			}},
		})
	}))
	defer embeddingServer.Close()
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDING_BASE_URL", embeddingServer.URL)

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("retrieval cache demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "file:///dataset", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	for i := 0; i < 3; i++ {
		if _, err := memoryStore.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
			SourceTable:         memory.SourceAgentMemoryRecord,
			SourceID:            "cache_memory_" + strconv.Itoa(i),
			ProjectID:           project.ID,
			DatasetID:           dataset.ID,
			Kind:                memory.KindPlanningOutcome,
			Scope:               memory.ScopeDataset,
			EmbeddingModel:      "test-embedding",
			EmbeddingDimensions: 3,
			Embedding:           []float32{1, 0, 0},
			EmbeddingText:       "class balancing improved minority recall and weighted loss helped",
			Metadata: map[string]any{
				"mechanism":                  "class_balancing",
				"accepted_for_vector_memory": true,
			},
		}); err != nil {
			t.Fatalf("seed embedding %d: %v", i, err)
		}
	}

	server := newServer(memoryStore)
	query := memory.MemoryRetrievalQuery{
		ProjectID: project.ID,
		DatasetID: dataset.ID,
		AgentName: agents.ExperimentPlannerAgentName,
		Purpose:   "experiment_planner",
		Text:      "class balancing improved minority recall",
		Kinds:     []string{memory.KindPlanningOutcome},
		Limit:     5,
	}

	firstResults, firstUsage := server.searchRetrievedMemory(context.Background(), query, "plan_1", "")
	if len(firstResults) == 0 {
		t.Fatal("expected first retrieval to return results")
	}
	if firstUsage.Cached || firstUsage.ProviderCallCount != 1 {
		t.Fatalf("unexpected first retrieval usage: %#v", firstUsage)
	}

	secondResults, secondUsage := server.searchRetrievedMemory(context.Background(), query, "plan_1", "")
	if len(secondResults) == 0 {
		t.Fatal("expected second retrieval to return results")
	}
	if !secondUsage.Cached || secondUsage.ProviderCallCount != 0 {
		t.Fatalf("unexpected cached retrieval usage: %#v", secondUsage)
	}
	if embeddingCalls != 1 {
		t.Fatalf("expected one embedding provider call total, got %d", embeddingCalls)
	}
}

func TestProjectTelemetrySummaryIncludesInvocationsAndEmbeddingUsage(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})

	invocation, err := server.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:        projectID,
		AgentName:        agents.ExperimentPlannerAgentName,
		Provider:         llm.ProviderOpenAI,
		Model:            "gpt-5.4-mini",
		ValidationStatus: memory.InvocationValidationValid,
		InputMessages: []map[string]string{
			{"role": "system", "content": "You are a planner."},
			{"role": "user", "content": "Improve the project."},
		},
		InputContext: map[string]any{
			"invocation_runtime": map[string]any{
				"api_style": llm.APIStyleResponses,
				"provider":  llm.ProviderOpenAI,
				"model":     "gpt-5.4-mini",
				"llm_usage": map[string]any{
					"input_tokens":        120,
					"output_tokens":       24,
					"total_tokens":        144,
					"cached_input_tokens": 18,
					"reasoning_tokens":    6,
				},
			},
			"planner_context_snapshot": map[string]any{
				"context_version": "planner_context_snapshot_v1",
				"project": map[string]any{
					"id":   projectID,
					"name": "Telemetry",
					"goal": "Measure cost",
				},
				"dataset_card": map[string]any{
					"id":   plan.DatasetID,
					"name": "Dataset",
				},
				"objective_context": map[string]any{
					"primary_objective": "macro_f1",
					"goal_text":         "Improve macro F1",
				},
				"champion_card": map[string]any{
					"current": map[string]any{"job_id": "job_1"},
				},
			},
		},
		RawOutput:         `{"decision_type":"WAIT"}`,
		ParsedOutput:      map[string]any{"decision_type": "WAIT"},
		AcceptedForMemory: false,
	})
	if err != nil {
		t.Fatalf("create agent invocation: %v", err)
	}

	if _, err := server.store.CreateMemoryEmbeddingUsageEvent(memory.MemoryEmbeddingUsageEvent{
		ProjectID:           projectID,
		Purpose:             memory.EmbeddingUsagePurposeSourceIndex,
		SourceTable:         memory.SourceStrategyScorecard,
		SourceID:            "scorecard_1",
		EmbeddingModel:      "text-embedding-3-small",
		EmbeddingDimensions: 1536,
		InputBytes:          2048,
		ProviderCallCount:   2,
		Metadata: map[string]any{
			"mechanism": "class_balancing",
		},
	}); err != nil {
		t.Fatalf("create source-index usage event: %v", err)
	}

	if _, err := server.store.CreateMemoryEmbeddingUsageEvent(memory.MemoryEmbeddingUsageEvent{
		ProjectID:           projectID,
		Purpose:             memory.EmbeddingUsagePurposeRetrievalQuery,
		RetrievalPurpose:    "experiment_planner",
		EmbeddingModel:      "text-embedding-3-small",
		EmbeddingDimensions: 1536,
		InputBytes:          320,
		ProviderCallCount:   1,
		RetrievedCount:      3,
		Injected:            true,
		LogOnly:             false,
		Cached:              true,
		Metadata: map[string]any{
			"query_hash": "hash_1",
		},
	}); err != nil {
		t.Fatalf("create retrieval usage event: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/projects/"+projectID+"/telemetry-summary?limit=10", nil)
	resp := httptest.NewRecorder()
	router := NewRouter(server.store)
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusOK {
		t.Fatalf("telemetry summary status = %d body=%s", resp.Code, resp.Body.String())
	}

	var payload struct {
		AgentInvocations           []memory.AgentInvocation           `json:"agent_invocations"`
		MemoryEmbeddingUsageEvents []memory.MemoryEmbeddingUsageEvent `json:"memory_embedding_usage_events"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode telemetry summary: %v", err)
	}
	var foundInvocation bool
	for _, row := range payload.AgentInvocations {
		if row.ID == invocation.ID {
			foundInvocation = true
			break
		}
	}
	if !foundInvocation {
		t.Fatalf("expected telemetry summary to include invocation %s", invocation.ID)
	}
	var foundSourceIndex bool
	var foundRetrievalQuery bool
	for _, row := range payload.MemoryEmbeddingUsageEvents {
		switch row.Purpose {
		case memory.EmbeddingUsagePurposeSourceIndex:
			foundSourceIndex = true
		case memory.EmbeddingUsagePurposeRetrievalQuery:
			foundRetrievalQuery = true
		}
	}
	if !foundSourceIndex || !foundRetrievalQuery {
		t.Fatalf("expected telemetry summary to include source_index and retrieval_query usage events, got %#v", payload.MemoryEmbeddingUsageEvents)
	}
}

func TestExperimentPlannerInputIncludesDatasetAndStrategyContext(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	if _, err := server.store.UpdateDatasetProfile(plan.DatasetID, map[string]any{
		"task_type":           "image_classification",
		"class_count":         4,
		"total_images":        240,
		"imbalance_ratio":     3.2,
		"corrupt_image_count": 1,
		"width_min":           120,
		"width_max":           640,
		"height_min":          100,
		"height_max":          512,
		"visual_exemplars": []map[string]any{
			{"class_name": "cat", "uri": "s3://bucket/cat-1.jpg", "size_bytes": 1024, "width": 224, "height": 224, "split": "test"},
			{"class_name": "dog", "uri": "s3://bucket/dog-1.jpg", "size_bytes": 2048, "width": 224, "height": 224, "split": "test"},
		},
	}); err != nil {
		t.Fatalf("update dataset profile: %v", err)
	}
	createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.70)
	if _, err := server.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		ProjectID: projectID,
		DatasetID: plan.DatasetID,
		PlanID:    plan.ID,
		AgentName: agents.ExperimentPlannerAgentName,
		Kind:      memory.KindPlanningOutcome,
		Summary:   "Weighted loss improved macro-F1.",
		Payload: map[string]any{
			"outcome_status":             agents.ExperimentPlanningOutcomeImprovedChampion,
			"lesson":                     "Weighted loss improved macro-F1.",
			"actual_delta_vs_champion":   0.02,
			"expected_delta_vs_champion": 0.01,
			"total_cost_usd":             0.03,
			"total_runtime_seconds":      120.0,
			"actual_best_run": map[string]any{
				"job_id": "job_outcome",
				"model":  "efficientnet_b0",
			},
			"proposed_experiments": []plans.PlannedExperiment{
				testExperiment("efficientnet_b0", 10),
			},
			"rejected_options": []agents.RejectedPlannerOption{
				{
					Option:      "Only add more epochs to MobileNet",
					Reason:      "It does not address class imbalance.",
					Evidence:    "class_imbalance_score is high",
					AppliesWhen: []string{"class_imbalance"},
				},
			},
		},
		Tags: []string{"planner_outcome", "improved_champion"},
	}); err != nil {
		t.Fatalf("create planner outcome memory: %v", err)
	}

	input, ready, err := server.buildExperimentPlannerInput(projectID, plan.ID)
	if err != nil {
		t.Fatalf("build planner input: %v", err)
	}
	if !ready {
		t.Fatal("expected planner input to be ready")
	}
	if input.DatasetInsights.ImbalanceRatio != 3.2 {
		t.Fatalf("expected imbalance insight, got %.2f", input.DatasetInsights.ImbalanceRatio)
	}
	if len(input.DatasetInsights.RecommendedPreprocessing) == 0 {
		t.Fatal("expected preprocessing recommendations")
	}
	if len(input.SuccessfulStrategyMemory) != 1 {
		t.Fatalf("expected one successful strategy memory, got %d", len(input.SuccessfulStrategyMemory))
	}
	if input.SuccessfulStrategyMemory[0].BestModel != "efficientnet_b0" {
		t.Fatalf("expected best model from strategy memory, got %s", input.SuccessfulStrategyMemory[0].BestModel)
	}
	if input.DeterministicDiagnosis.ClassImbalanceScore == 0 {
		t.Fatal("expected deterministic diagnosis in planner input")
	}
	if len(input.RejectedStrategyMemory) != 1 {
		t.Fatalf("expected relevant rejected strategy memory, got %d", len(input.RejectedStrategyMemory))
	}
	if input.VisualExemplarContext == nil || !input.VisualExemplarContext.Enabled {
		t.Fatalf("expected enabled visual exemplar context, got %#v", input.VisualExemplarContext)
	}
	if input.VisualExemplarContext.ExemplarCount != 2 {
		t.Fatalf("expected two visual exemplars in planner context, got %d", input.VisualExemplarContext.ExemplarCount)
	}
	if !input.VisualExemplarContext.EvidenceOnly {
		t.Fatal("expected visual exemplars to be evidence-only")
	}
}

func TestExperimentPlannerInputPrefersLatestAcceptedVisualAnalysis(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	if _, err := server.store.UpdateDatasetProfile(plan.DatasetID, map[string]any{
		"task_type":    "image_classification",
		"class_count":  3,
		"total_images": 180,
		"visual_exemplars": []map[string]any{
			{"class_name": "cat", "uri": "file:///raw-legacy-exemplar.jpg", "size_bytes": 1024},
		},
	}); err != nil {
		t.Fatalf("update dataset profile: %v", err)
	}
	if _, err := server.store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      projectID,
		DatasetID:      plan.DatasetID,
		ImagesAnalyzed: 24,
		CoverageReport: datasets.VisualCoverageReport{ImagesAnalyzed: 24, ClassesTotal: 3, ClassesCovered: 2},
		VisualTraits:   []datasets.VisualTrait{{Trait: "blur", Level: "medium", Confidence: "low", Evidence: []string{"some sampled images are soft"}}},
		TriggerDetails: map[string]any{"raw_visual_agent_output": "SHOULD_NOT_LEAK"},
	}); err != nil {
		t.Fatalf("create accepted visual analysis: %v", err)
	}
	if _, err := server.store.RejectDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      projectID,
		DatasetID:      plan.DatasetID,
		ImagesAnalyzed: 32,
		VisualTraits:   []datasets.VisualTrait{{Trait: "rejected_trait", Evidence: []string{"should not appear"}}},
	}); err != nil {
		t.Fatalf("create rejected visual analysis: %v", err)
	}
	latest, err := server.store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      projectID,
		DatasetID:      plan.DatasetID,
		ImagesAnalyzed: 48,
		TriggerReason:  datasets.VisualTriggerDeficiencyReanalysis,
		CoverageReport: datasets.VisualCoverageReport{
			ImagesAnalyzed:     48,
			ClassesTotal:       3,
			ClassesCovered:     3,
			ClassCoverageRatio: 1,
			PerClassCounts:     map[string]int{"cat": 16, "dog": 16, "rabbit": 16},
		},
		VisualTraits: []datasets.VisualTrait{
			{Trait: "small_objects", Level: "high", Confidence: "medium", Evidence: []string{"subjects occupy small image regions"}},
		},
		ClassesToWatch: []datasets.ClassWatchItem{
			{ClassName: "cat", Reason: "similar silhouettes", Evidence: []string{"cat/dog profiles overlap"}, Confidence: "medium"},
		},
		PreprocessingHypotheses: []datasets.PreprocessingHypothesis{
			{ID: "vh_001", Mechanism: "resolution_crop", Summary: "Try preserve-aspect resize.", Evidence: []string{"small objects"}, ExpectedEffect: "Preserve detail.", Confidence: "medium", SupportStatus: "supported"},
		},
		Limitations: []string{"bounded visual sample"},
		TriggerDetails: map[string]any{
			"raw_visual_agent_output": "SHOULD_NOT_LEAK",
			"image_payload":           "file:///secret-image.jpg",
		},
	})
	if err != nil {
		t.Fatalf("create latest accepted visual analysis: %v", err)
	}
	createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.70)

	input, ready, err := server.buildExperimentPlannerInput(projectID, plan.ID)
	if err != nil {
		t.Fatalf("build planner input: %v", err)
	}
	if !ready {
		t.Fatal("expected planner input to be ready")
	}
	if input.VisualExemplarContext == nil || input.VisualExemplarContext.AnalysisID != latest.ID {
		t.Fatalf("expected latest accepted visual analysis in planner context, got %#v", input.VisualExemplarContext)
	}
	if input.VisualExemplarContext.Source != "dataset_visual_analysis" || input.VisualExemplarContext.ExemplarCount != 0 {
		t.Fatalf("expected durable visual analysis to replace legacy exemplar fallback, got %#v", input.VisualExemplarContext)
	}
	snapshot := agents.BuildPlannerContextSnapshot(input)
	evidence := snapshot.VisualEvidence
	if evidence["analysis_id"] != latest.ID || evidence["trigger_reason"] != string(datasets.VisualTriggerDeficiencyReanalysis) {
		t.Fatalf("expected latest accepted visual analysis metadata, got %#v", evidence)
	}
	if evidence["raw_images_included"] != false || evidence["raw_visual_output_included"] != false {
		t.Fatalf("expected raw visual payload exclusion flags, got %#v", evidence)
	}
	if _, ok := evidence["class_evidence"]; ok {
		t.Fatalf("expected legacy class_evidence to be omitted when durable analysis is available, got %#v", evidence)
	}
	blob, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal visual evidence: %v", err)
	}
	for _, forbidden := range []string{"SHOULD_NOT_LEAK", "file:///secret-image.jpg", "file:///raw-legacy-exemplar.jpg", "rejected_trait"} {
		if strings.Contains(string(blob), forbidden) {
			t.Fatalf("expected planner visual evidence to exclude %q: %s", forbidden, string(blob))
		}
	}
}

func TestExperimentPlannerPromptContextCompactsLongFollowUpHistory(t *testing.T) {
	server, projectID, initialPlan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	initialJob, _ := createTerminalTrainingJob(t, server, initialPlan, initialPlan.Experiments[0], jobs.StatusSucceeded, 0.74)

	var latestPlan plans.ExperimentPlan
	for i := 0; i < 8; i++ {
		experiment := testExperiment([]string{
			"efficientnet_b0",
			"resnet34",
			"convnext_tiny",
			"regnet_y_400mf",
		}[i%4], 8+i)
		experiment.Reason = "follow-up mechanism coverage"
		experiment.Strategy = "planner ledger coverage"
		followUp, err := server.store.CreateExperimentPlan(projectID, initialPlan.DatasetID, "macro_f1", 1, 5, []plans.PlannedExperiment{experiment}, nil, "decision_long_history_"+strconv.Itoa(i))
		if err != nil {
			t.Fatalf("create follow-up plan %d: %v", i, err)
		}
		score := 0.61 + float64(i)*0.01
		if i == 3 {
			score = 0.76
		}
		createTerminalTrainingJob(t, server, followUp, followUp.Experiments[0], jobs.StatusSucceeded, score)
		latestPlan = followUp
	}

	if _, err := server.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		ProjectID: projectID,
		DatasetID: initialPlan.DatasetID,
		PlanID:    latestPlan.ID,
		AgentName: agents.ExperimentPlannerAgentName,
		Kind:      memory.KindPlanningOutcome,
		Summary:   "High-resolution EfficientNet improved the champion; repeat only with clear coverage gaps.",
		Payload: map[string]any{
			"outcome_status":             agents.ExperimentPlanningOutcomeImprovedChampion,
			"lesson":                     "High-resolution EfficientNet improved the champion.",
			"actual_delta_vs_champion":   0.02,
			"expected_delta_vs_champion": 0.01,
			"actual_best_run": map[string]any{
				"job_id": initialJob.ID,
				"model":  "efficientnet_b0",
			},
			"proposed_experiments": []plans.PlannedExperiment{
				testExperiment("efficientnet_b0", 10),
			},
			"rejected_options": []agents.RejectedPlannerOption{
				{
					Option:      "Raw repeat of every prior follow-up",
					Reason:      "The prompt should use distilled coverage instead of raw prior lists.",
					Evidence:    "long follow-up history",
					AppliesWhen: []string{},
				},
			},
		},
		Tags: []string{"planner_outcome", "improved_champion"},
	}); err != nil {
		t.Fatalf("create planner outcome memory: %v", err)
	}
	if _, err := server.store.CreateStrategyScorecard(strategies.StrategyScorecardCreate{
		ProjectID:        projectID,
		DatasetID:        initialPlan.DatasetID,
		SourceDecisionID: "decision_long_history_3",
		SourcePlanID:     initialPlan.ID,
		FollowUpPlanID:   latestPlan.ID,
		StrategyType:     "architecture_coverage",
		PlanningMode:     "champion_challenge",
		DatasetTraits:    map[string]any{"class_count": 4},
		ObjectiveProfile: map[string]any{"target_metric": "macro_f1"},
		ProposedChanges:  map[string]any{"models": []string{"efficientnet_b0", "resnet34", "convnext_tiny"}},
		ExpectedDelta:    0.01,
		ConfidenceBefore: 0.73,
		Outcome:          strategies.OutcomeImprovedChampion,
		Lesson:           "Architecture coverage found the best challenger without carrying raw history.",
		Tags:             []string{"coverage", "improved_champion"},
	}); err != nil {
		t.Fatalf("create strategy scorecard: %v", err)
	}

	input, ready, err := server.buildExperimentPlannerInput(projectID, latestPlan.ID)
	if err != nil {
		t.Fatalf("build planner input: %v", err)
	}
	if !ready {
		t.Fatal("expected planner input to be ready")
	}

	agent := agents.NewExperimentPlannerAgent(capturingPlannerGenerator{
		response: `{
			"summary": "Select the distilled champion after long follow-up history.",
			"decision_type": "WAIT",
			"rationale": "The compact ledger already captures champion and coverage state.",
			"confidence": 0.7,
			"planning_mode": "stop_or_select",
			"deterministic_diagnosis_used": ["no_improvement_rounds"],
			"evidence_used": ["current champion", "experiment ledger", "coverage summary"],
			"hypothesis": "",
			"expected_failure_modes": [],
			"dataset_preprocessing_rationale": "",
			"changed_variables": [],
			"success_criteria": "",
			"stop_condition": "Wait for user review.",
			"deployment_tradeoff": "",
			"proposed_experiments": [],
			"champion_job_id": "",
			"why_can_beat_champion": "",
			"expected_delta_vs_champion": 0,
			"stop_reason": "",
			"risks": [],
			"expected_tradeoffs": [],
			"novelty_notes": [],
			"rejected_options": [],
			"tags": ["compact_context"]
		}`,
	}, "test-model")
	trace, err := agent.PlanWithTrace(context.Background(), input)
	if err != nil {
		t.Fatalf("plan with trace: %v", err)
	}

	contextMap := trace.PromptContext
	for _, rawKey := range []string{"prior_jobs", "prior_plans", "prior_memory", "plan_jobs", "plan_evaluations", "dataset"} {
		if _, ok := contextMap[rawKey]; ok {
			t.Fatalf("expected compact planner prompt to omit raw %s after long follow-up history", rawKey)
		}
	}
	snapshot, ok := contextMap["planner_context_snapshot"].(agents.PlannerContextSnapshot)
	if !ok {
		t.Fatalf("expected planner context snapshot, got %#v", contextMap["planner_context_snapshot"])
	}
	if snapshot.DatasetCard.RawProfileIncluded {
		t.Fatal("expected raw dataset profile to be excluded from compact snapshot")
	}
	if snapshot.ChampionCard.Current == nil {
		t.Fatal("expected distilled current champion context")
	}
	if got := len(snapshot.CompletedExperimentLog); got == 0 || got > 24 {
		t.Fatalf("expected capped experiment ledger, got %d", got)
	}
	if got := len(snapshot.SearchCoverage.TriedMechanisms); got == 0 || got > 24 {
		t.Fatalf("expected capped experiment coverage mechanisms, got %d", got)
	}
	if got := len(snapshot.StrategyLessons); got == 0 {
		t.Fatalf("expected distilled strategy lessons, got %d", got)
	}
	if got := len(snapshot.BlockedRepeats); got == 0 {
		t.Fatalf("expected distilled rejected repeat guidance, got %d", got)
	}
	if got := len(input.SuccessfulStrategyMemory); got == 0 {
		t.Fatalf("expected distilled successful strategy memory from prior outcomes, got %d", got)
	}
	if got := len(input.RejectedStrategyMemory); got == 0 {
		t.Fatalf("expected distilled rejected strategy memory from prior outcomes, got %d", got)
	}
	if got := len(input.StrategyScorecards); got == 0 {
		t.Fatalf("expected distilled strategy scorecards, got %d", got)
	}
}

func TestExperimentPlannerStopCriteriaSelectsChampionAfterStalledFollowUps(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "true")

	recommendation := experimentPlannerAddExperimentsRecommendation()
	input := agents.ExperimentPlannerInput{
		CurrentChampion: &agents.ExperimentChampion{
			JobID:        "job_champion",
			Model:        "mobilenet_v3_small",
			TargetMetric: "macro_f1",
			Score:        0.70,
		},
		NoImprovementRounds:          plannerNoImprovementRoundsToSelect,
		MinimumMeaningfulImprovement: plannerMinimumMeaningfulImprovement,
	}

	adjusted := applyExperimentPlannerStopCriteria(recommendation, input)
	if adjusted.DecisionType != decisions.TypeSelectChampion {
		t.Fatalf("expected SELECT_CHAMPION, got %s", adjusted.DecisionType)
	}
	if adjusted.ChampionJobID != "job_champion" {
		t.Fatalf("expected champion job id to be set, got %s", adjusted.ChampionJobID)
	}
	if len(adjusted.ProposedExperiments) != 0 {
		t.Fatal("expected converted champion selection to clear proposed experiments")
	}
	if adjusted.StopReason == "" {
		t.Fatal("expected stop reason")
	}
}

func TestExperimentPlannerStopCriteriaKeepsAddExperimentsByDefault(t *testing.T) {
	recommendation := experimentPlannerAddExperimentsRecommendation()
	input := agents.ExperimentPlannerInput{
		CurrentChampion: &agents.ExperimentChampion{
			JobID:        "job_champion",
			Model:        "mobilenet_v3_small",
			TargetMetric: "macro_f1",
			Score:        0.70,
		},
		NoImprovementRounds:          plannerNoImprovementRoundsToSelect,
		MinimumMeaningfulImprovement: plannerMinimumMeaningfulImprovement,
	}

	adjusted := applyExperimentPlannerStopCriteria(recommendation, input)
	if adjusted.DecisionType != decisions.TypeAddExperiments {
		t.Fatalf("expected ADD_EXPERIMENTS to remain schedulable by default, got %s", adjusted.DecisionType)
	}
	if len(adjusted.ProposedExperiments) == 0 {
		t.Fatal("expected proposed experiments to remain intact")
	}
	if !stringSliceContains(adjusted.Tags, "backend_stop_advisory") {
		t.Fatalf("expected backend stop advisory tag, got %#v", adjusted.Tags)
	}
	if !strings.Contains(strings.Join(adjusted.NoveltyNotes, " "), "advisory") {
		t.Fatalf("expected advisory novelty note, got %#v", adjusted.NoveltyNotes)
	}
}

func TestExperimentPlannerStopCriteriaSelectsNearCeilingChampion(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "true")

	recommendation := experimentPlannerAddExperimentsRecommendation()
	input := agents.ExperimentPlannerInput{
		CurrentChampion: &agents.ExperimentChampion{
			JobID:        "job_near_ceiling",
			Model:        "mobilenet_v3_small",
			TargetMetric: "accuracy",
			Score:        0.996448,
		},
		MinimumMeaningfulImprovement: plannerMinimumMeaningfulImprovement,
	}

	adjusted := applyExperimentPlannerStopCriteria(recommendation, input)
	if adjusted.DecisionType != decisions.TypeSelectChampion {
		t.Fatalf("expected SELECT_CHAMPION, got %s", adjusted.DecisionType)
	}
	if adjusted.ChampionJobID != "job_near_ceiling" {
		t.Fatalf("expected champion job id to be set, got %s", adjusted.ChampionJobID)
	}
	if len(adjusted.ProposedExperiments) != 0 {
		t.Fatal("expected near-ceiling champion selection to clear proposed experiments")
	}
	if !strings.Contains(adjusted.StopReason, "metric ceiling") {
		t.Fatalf("expected metric-ceiling stop reason, got %q", adjusted.StopReason)
	}
	if !stringSliceContains(adjusted.Tags, "near_metric_ceiling_guard") {
		t.Fatalf("expected near-ceiling guard tag, got %#v", adjusted.Tags)
	}
}

func TestNearCeilingChampionBlocksPlannerFollowUpScheduling(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "true")

	response := `{
		"summary": "Try four more compact challengers.",
		"decision_type": "ADD_EXPERIMENTS",
		"rationale": "The champion is strong, but the planner wants to test compact challenger models.",
		"confidence": 0.71,
		"planning_mode": "champion_challenge",
		"deterministic_diagnosis_used": ["no dominant failure mode"],
		"evidence_used": ["champion is near-perfect", "dataset has variable dimensions"],
		"hypothesis": "A compact challenger may close the tiny remaining gap.",
		"expected_failure_modes": ["the task may already be saturated"],
		"dataset_preprocessing_rationale": "Use moderate preprocessing changes for variable image dimensions.",
		"changed_variables": ["model_family", "fine_tune_strategy", "augmentation_policy"],
		"success_criteria": "Improve accuracy by at least 0.005.",
		"stop_condition": "Select champion if no challenger improves.",
		"deployment_tradeoff": "More training cost for a tiny possible gain.",
		"candidate_hypotheses": [
			{
				"hypothesis": "A higher-resolution EfficientNet challenger could capture remaining fine-grained errors.",
				"planning_mode": "champion_challenge",
				"mechanism": "resolution_crop",
				"intervention": "Challenge the near-ceiling champion with higher resolution and full fine-tuning.",
				"proposed_changes": {"model_family": "efficientnet", "resolution_strategy": "high_resolution_ablation", "fine_tune_strategy": "full"},
				"expected_effect": "Capture any remaining fine-grained or scale-dependent errors.",
				"expected_metric_impact": 0.005,
				"expected_tradeoffs": ["more training cost"],
				"risk": "high",
				"cost_level": "medium",
				"novelty_score": 0.55,
				"evidence_used": ["champion is near-perfect", "dataset has variable dimensions"],
				"similar_success_memory_ids": [],
				"similar_failure_memory_ids": [],
				"experiment_config": {
					"template": "efficientnet_transfer",
					"model": "efficientnet_b1",
					"epochs": 12,
					"batch_size": 16,
					"learning_rate": 0.0002,
					"reason": "Challenge the champion with a balanced EfficientNet.",
					"image_size": 240,
					"resolution_strategy": "high_resolution_ablation",
					"optimizer": "adamw",
					"scheduler": "cosine",
					"weight_decay": 0.01,
					"augmentation_policy": "moderate",
					"class_balancing": "weighted_loss",
					"sampling_strategy": "none",
					"strategy": "champion challenge",
					"pretrained": true,
					"freeze_backbone": false,
					"fine_tune_strategy": "full"
				}
			}
		],
		"proposed_experiments": [
			{
				"template": "efficientnet_transfer",
				"model": "efficientnet_b1",
				"epochs": 12,
				"batch_size": 16,
				"learning_rate": 0.0002,
				"reason": "Challenge the champion with a balanced EfficientNet.",
				"image_size": 240,
				"resolution_strategy": "high_resolution_ablation",
				"optimizer": "adamw",
				"scheduler": "cosine",
				"weight_decay": 0.01,
				"augmentation_policy": "moderate",
				"class_balancing": "weighted_loss",
				"sampling_strategy": "none",
				"strategy": "champion challenge",
				"pretrained": true,
				"freeze_backbone": false,
				"fine_tune_strategy": "full"
			}
		],
		"proposal_mechanisms": [
			{
				"experiment_index": 0,
				"mechanism": "resolution_crop",
				"intervention": "Challenge the near-ceiling champion with higher resolution and full fine-tuning.",
				"evidence_used": ["champion is near-perfect", "dataset has variable dimensions"],
				"expected_effect": "Capture any remaining fine-grained or scale-dependent errors."
			}
		],
		"champion_job_id": "",
		"why_can_beat_champion": "It changes model family and fine tuning.",
		"expected_delta_vs_champion": 0.005,
		"stop_reason": "",
		"risks": ["near-ceiling task leaves little headroom"],
		"expected_tradeoffs": ["more training cost"],
		"novelty_notes": ["larger compact challenger"],
		"rejected_options": [{"option":"more epochs only","reason":"too little upside","evidence":"near ceiling","applies_when":["near_ceiling"]}],
		"tags": ["near_ceiling_test"]
	}`
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]string{
					"role":    "assistant",
					"content": response,
				},
			}},
		})
	}))
	defer llmServer.Close()

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "test-model")
	t.Setenv("MODEL_EXPRESS_AGENT_MODE", "autonomous")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 1.0)
	setHealthyNearCeilingRunEvidence(t, server, job.ID)

	handled, err := server.runExperimentPlannerAfterTrainingJob(job)
	if err != nil {
		t.Fatalf("run planner: %v", err)
	}
	if !handled {
		t.Fatal("expected planner to handle the completed job")
	}

	agentDecisions := listAgentDecisions(t, server, projectID)
	if len(agentDecisions) != 1 {
		t.Fatalf("expected one planner decision, got %d", len(agentDecisions))
	}
	decision := agentDecisions[0]
	if decision.DecisionType != decisions.TypeSelectChampion {
		t.Fatalf("expected backend to convert ADD_EXPERIMENTS to SELECT_CHAMPION, got %s", decision.DecisionType)
	}
	if championJobID, _ := decision.Payload["champion_job_id"].(string); championJobID != job.ID {
		t.Fatalf("expected selected champion %s, got %#v", job.ID, decision.Payload["champion_job_id"])
	}
	if stopReason, _ := decision.Payload["stop_reason"].(string); !strings.Contains(stopReason, "metric ceiling") {
		t.Fatalf("expected metric-ceiling stop reason, got %q", stopReason)
	}
	if _, ok := decision.Payload["proposed_experiments"]; ok {
		t.Fatalf("expected converted champion decision to omit proposed experiments, got %#v", decision.Payload["proposed_experiments"])
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected no follow-up plan to be scheduled, got %d total plans", got)
	}
	projectJobs, err := server.store.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	if got := countJobsByTemplate(projectJobs, jobs.TemplateTrainExperiment); got != 1 {
		t.Fatalf("expected no follow-up training jobs with auto execution disabled, got %d training jobs", got)
	}
	if got := countJobsByTemplate(projectJobs, jobs.TemplateExportChampion); got != 1 {
		t.Fatalf("expected one queued champion export job, got %d", got)
	}
}

func TestPlannerWaitAfterCompletedTrainingSelectsChampion(t *testing.T) {
	response := `{
		"summary": "Pause new training until audit evidence is available.",
		"decision_type": "WAIT",
		"rationale": "The completed run is deployment-ready and another follow-up has low expected value.",
		"confidence": 0.78,
		"planning_mode": "stop_or_select",
		"deterministic_diagnosis_used": ["improvement_stagnation_score=1.000"],
		"evidence_used": ["current champion is deployment-ready", "more epochs are not justified"],
		"hypothesis": "Waiting avoids another low-value training batch.",
		"expected_failure_modes": ["audit may not change the next action"],
		"dataset_preprocessing_rationale": "Keep current preprocessing until new evidence arrives.",
		"changed_variables": ["decision_type"],
		"success_criteria": "Avoid additional low-value training.",
		"stop_condition": "Keep the current champion unless new audit evidence appears.",
		"deployment_tradeoff": "No deployment disruption.",
		"candidate_hypotheses": [],
		"proposed_experiments": [],
		"proposal_mechanisms": [],
		"champion_job_id": "",
		"why_can_beat_champion": "",
		"expected_delta_vs_champion": 0,
		"stop_reason": "",
		"risks": [],
		"expected_tradeoffs": ["less immediate experimentation"],
		"novelty_notes": [],
		"rejected_options": [],
		"tags": ["wait_for_audit"]
	}`
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]string{
					"role":    "assistant",
					"content": response,
				},
			}},
		})
	}))
	defer llmServer.Close()

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "test-model")
	t.Setenv("MODEL_EXPRESS_AGENT_MODE", "autonomous")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.82)

	handled, err := server.runExperimentPlannerAfterTrainingJob(job)
	if err != nil {
		t.Fatalf("run planner: %v", err)
	}
	if !handled {
		t.Fatal("expected planner to handle the completed job")
	}

	agentDecisions := listAgentDecisions(t, server, projectID)
	if len(agentDecisions) != 1 {
		t.Fatalf("expected one planner decision, got %d", len(agentDecisions))
	}
	decision := agentDecisions[0]
	if decision.DecisionType != decisions.TypeSelectChampion {
		t.Fatalf("expected WAIT to be converted to SELECT_CHAMPION, got %s", decision.DecisionType)
	}
	if championJobID, _ := decision.Payload["champion_job_id"].(string); championJobID != job.ID {
		t.Fatalf("expected selected champion %s, got %#v", job.ID, decision.Payload["champion_job_id"])
	}
	if stopReason, _ := decision.Payload["stop_reason"].(string); !strings.Contains(stopReason, "WAIT") {
		t.Fatalf("expected wait conversion stop reason, got %q", stopReason)
	}
	champion, err := server.store.GetProjectChampion(projectID)
	if err != nil {
		t.Fatalf("expected persisted project champion: %v", err)
	}
	if champion.JobID != job.ID || champion.SourceDecisionID != decision.ID {
		t.Fatalf("expected champion %s from decision %s, got %#v", job.ID, decision.ID, champion)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected no follow-up plan after WAIT conversion, got %d total plans", got)
	}
}

func TestNearCeilingStaleAddDecisionCannotScheduleFollowUp(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 1.0)
	setHealthyNearCeilingRunEvidence(t, server, job.ID)
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	recommendation := experimentPlannerAddExperimentsRecommendation()
	payload, err := experimentPlannerDecisionPayload(recommendation, invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err != nil {
		t.Fatalf("build stale planner payload: %v", err)
	}
	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "stale ADD_EXPERIMENTS decision", payload)
	if err != nil {
		t.Fatalf("create stale decision: %v", err)
	}
	stopReason, guardTag, ok, err := server.plannerFollowUpStopReason(projectID, plan, listExperimentPlans(t, server, projectID))
	if err != nil {
		t.Fatalf("compute planner stop reason: %v", err)
	}
	if !ok || guardTag != "near_metric_ceiling_guard" {
		summaries, err := server.store.ListProjectTrainingRunSummaries(projectID)
		if err != nil {
			t.Fatalf("list summaries for debug: %v", err)
		}
		evaluations, err := server.store.ListProjectTrainingRunEvaluations(projectID)
		if err != nil {
			t.Fatalf("list evaluations for debug: %v", err)
		}
		champion, _, _, rounds, _ := experimentPlannerPerformanceContext(plan.TargetMetric, listExperimentPlans(t, server, projectID), summaries, evaluations, agents.ProjectObjectiveContext{}, plan.ID)
		t.Fatalf("expected live near-ceiling guard before scheduling, ok=%t tag=%q reason=%q champion=%#v rounds=%d", ok, guardTag, stopReason, champion, rounds)
	}

	if err := server.schedulePlannerDecision(projectID, plan, decision, automaticExperimentReviewResult{Decision: &decision}); err != nil {
		t.Fatalf("schedule stale planner decision: %v", err)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected stale near-ceiling ADD decision to create no follow-up plan, got %d total plans", got)
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	foundBlocked := false
	for _, event := range events {
		if event.Payload["backend_stop_guard"] == "near_metric_ceiling_guard" {
			foundBlocked = true
			break
		}
	}
	if !foundBlocked {
		t.Fatalf("expected near-ceiling blocked scheduling event, got %#v", events)
	}
}

func TestPersistedChampionBlocksStaleAddDecisionBeforePlanCreation(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.82)
	selectDecision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeSelectChampion, "Select the completed champion.", map[string]any{
		"champion_job_id": job.ID,
	})
	if err != nil {
		t.Fatalf("create select champion decision: %v", err)
	}
	if err := server.persistProjectChampionFromDecision(projectID, selectDecision); err != nil {
		t.Fatalf("persist champion: %v", err)
	}
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	payload, err := experimentPlannerDecisionPayload(experimentPlannerAddExperimentsRecommendation(), invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err != nil {
		t.Fatalf("build stale planner payload: %v", err)
	}
	addDecision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "stale ADD_EXPERIMENTS decision", payload)
	if err != nil {
		t.Fatalf("create add decision: %v", err)
	}

	_, _, err = server.ensureFollowUpPlan(projectID, plan, addDecision)
	if err == nil || !errors.Is(err, errChampionSelectedFollowUpBlocked) {
		t.Fatalf("expected champion-selected follow-up block, got %v", err)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected no follow-up plan after champion selection, got %d total plans", got)
	}
	projectJobs, err := server.store.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	if got := countJobsByTemplate(projectJobs, jobs.TemplateTrainExperiment); got != 1 {
		t.Fatalf("expected only the completed champion training job, got %d training jobs", got)
	}
	if got := countJobsByTemplate(projectJobs, jobs.TemplateExportChampion); got != 1 {
		t.Fatalf("expected one queued champion export job, got %d", got)
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasBackendStopGuardEvent(events, "champion_selected_guard") {
		t.Fatalf("expected champion-selected blocked event, got %#v", events)
	}
}

func TestPersistedChampionBlocksFollowUpPlanByDefault(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "false")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.82)
	selectDecision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeSelectChampion, "Record the current champion without ending exploration.", map[string]any{
		"champion_job_id": job.ID,
	})
	if err != nil {
		t.Fatalf("create select champion decision: %v", err)
	}
	if err := server.persistProjectChampionFromDecision(projectID, selectDecision); err != nil {
		t.Fatalf("persist champion: %v", err)
	}
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	payload, err := experimentPlannerDecisionPayload(experimentPlannerAddExperimentsRecommendation(), invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err != nil {
		t.Fatalf("build planner payload: %v", err)
	}
	addDecision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "continue after selected champion", payload)
	if err != nil {
		t.Fatalf("create add decision: %v", err)
	}

	_, _, err = server.ensureFollowUpPlan(projectID, plan, addDecision)
	if err == nil || !errors.Is(err, errChampionSelectedFollowUpBlocked) {
		t.Fatalf("expected champion-selected follow-up block by default, got %v", err)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected no follow-up plan after champion selection, got %d total plans", got)
	}
	projectJobs, err := server.store.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	if got := countJobsByTemplate(projectJobs, jobs.TemplateTrainExperiment); got != 1 {
		t.Fatalf("expected only the completed champion training job, got %d training jobs", got)
	}
	if got := countJobsByTemplate(projectJobs, jobs.TemplateExportChampion); got != 1 {
		t.Fatalf("expected one queued champion export job, got %d", got)
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasBackendStopGuardEvent(events, "champion_selected_guard") {
		t.Fatalf("expected champion-selected blocked event, got %#v", events)
	}
}

func TestSelectChampionDecisionWithoutChampionRecordBlocksAutonomousScheduling(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.73)
	if _, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeSelectChampion, "Select champion but do not persist champion row yet.", map[string]any{
		"champion_job_id": job.ID,
	}); err != nil {
		t.Fatalf("create select champion decision: %v", err)
	}
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	payload, err := experimentPlannerDecisionPayload(experimentPlannerAddExperimentsRecommendation(), invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err != nil {
		t.Fatalf("build stale planner payload: %v", err)
	}
	addDecision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "stale ADD_EXPERIMENTS decision", payload)
	if err != nil {
		t.Fatalf("create add decision: %v", err)
	}

	if err := server.schedulePlannerDecision(projectID, plan, addDecision, automaticExperimentReviewResult{Decision: &addDecision}); err != nil {
		t.Fatalf("schedule stale add decision: %v", err)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 1 {
		t.Fatalf("expected SELECT_CHAMPION decision to block follow-up plan creation, got %d total plans", got)
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasBackendStopGuardEvent(events, "champion_selected_guard") {
		t.Fatalf("expected champion-selected guard event, got %#v", events)
	}
}

func TestReopenExperimentationAllowsFollowUpsAfterChampionSelection(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.79)
	selectDecision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeSelectChampion, "Select final champion.", map[string]any{
		"champion_job_id": job.ID,
	})
	if err != nil {
		t.Fatalf("create select champion decision: %v", err)
	}
	if err := server.persistProjectChampionFromDecision(projectID, selectDecision); err != nil {
		t.Fatalf("persist champion: %v", err)
	}

	router := NewRouter(server.store)
	body := []byte(`{"reason":"Open a new diagnosis round after reviewing the champion."}`)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+projectID+"/experimentation/reopen", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected reopen status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	var reopenResp reopenExperimentationResponse
	if err := json.NewDecoder(resp.Body).Decode(&reopenResp); err != nil {
		t.Fatalf("decode reopen response: %v", err)
	}
	if reopenResp.Decision.DecisionType != decisions.TypeReopenExperimentation {
		t.Fatalf("expected reopen decision, got %s", reopenResp.Decision.DecisionType)
	}
	if reopenResp.Event.EventType != execution.EventExperimentationReopened {
		t.Fatalf("expected reopen event, got %s", reopenResp.Event.EventType)
	}

	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	payload, err := experimentPlannerDecisionPayload(experimentPlannerAddExperimentsRecommendation(), invocation, "manual", plannerInputForPayload(t, server, projectID))
	if err != nil {
		t.Fatalf("build planner payload: %v", err)
	}
	addDecision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "new post-reopen ADD_EXPERIMENTS decision", payload)
	if err != nil {
		t.Fatalf("create add decision: %v", err)
	}

	followUpPlan, created, err := server.ensureFollowUpPlan(projectID, plan, addDecision)
	if err != nil {
		t.Fatalf("expected post-reopen follow-up plan creation, got %v", err)
	}
	if !created || followUpPlan.SourceDecisionID != addDecision.ID {
		t.Fatalf("expected created follow-up plan sourced from %s, got %#v created=%v", addDecision.ID, followUpPlan, created)
	}
}

func TestReopenExperimentationRequiresTerminalChampionState(t *testing.T) {
	server, projectID, _ := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})

	router := NewRouter(server.store)
	body := []byte(`{"reason":"try another round"}`)
	req := httptest.NewRequest(http.MethodPost, "/projects/"+projectID+"/experimentation/reopen", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
	if got := len(listAgentDecisions(t, server, projectID)); got != 0 {
		t.Fatalf("expected no reopen decision before champion selection, got %d", got)
	}
}

func TestPostChampionExistingFollowUpPlanCannotCreateJobs(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.76)
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	payload, err := experimentPlannerDecisionPayload(experimentPlannerAddExperimentsRecommendation(), invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err != nil {
		t.Fatalf("build planner payload: %v", err)
	}
	addDecision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, "accepted ADD_EXPERIMENTS decision", payload)
	if err != nil {
		t.Fatalf("create add decision: %v", err)
	}
	followUpPlan, _, err := server.ensureFollowUpPlan(projectID, plan, addDecision)
	if err != nil {
		t.Fatalf("create follow-up before champion selection: %v", err)
	}
	selectDecision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeSelectChampion, "Select champion before stale follow-up execution.", map[string]any{
		"champion_job_id": job.ID,
	})
	if err != nil {
		t.Fatalf("create select decision: %v", err)
	}
	if err := server.persistProjectChampionFromDecision(projectID, selectDecision); err != nil {
		t.Fatalf("persist champion: %v", err)
	}

	_, err = server.executeStoredExperimentPlan(followUpPlan.ID, executeExperimentPlanRequest{Provider: "local"})
	if err == nil || !errors.Is(err, errChampionSelectedFollowUpBlocked) {
		t.Fatalf("expected champion-selected execution block, got %v", err)
	}
	projectJobs, err := server.store.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	if got := countJobsByTemplate(projectJobs, jobs.TemplateTrainExperiment); got != 1 {
		t.Fatalf("expected no follow-up training jobs after champion selection, got %d training jobs", got)
	}
	if got := countJobsByTemplate(projectJobs, jobs.TemplateExportChampion); got != 1 {
		t.Fatalf("expected one queued champion export job, got %d", got)
	}
}

func TestExperimentPlannerSkipsNewPlanningAfterChampionSelected(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "true")

	llmCalls := 0
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		llmCalls++
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]string{
					"role":    "assistant",
					"content": `{"summary":"should not be called","decision_type":"WAIT","rationale":"skip","confidence":0.5}`,
				},
			}},
		})
	}))
	defer llmServer.Close()

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "test-model")
	t.Setenv("MODEL_EXPRESS_AGENT_MODE", "autonomous")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	championJob, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.80)
	selectDecision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeSelectChampion, "Select final champion.", map[string]any{
		"champion_job_id": championJob.ID,
	})
	if err != nil {
		t.Fatalf("create select decision: %v", err)
	}
	if err := server.persistProjectChampionFromDecision(projectID, selectDecision); err != nil {
		t.Fatalf("persist champion: %v", err)
	}
	followUpExperiment := testExperiment("efficientnet_b0", 10)
	followUpExperiment.Mechanism = "architecture_challenge"
	followUpExperiment.Intervention = "Stale post-champion architecture challenger."
	followUpExperiment.EvidenceUsed = []string{"stale decision predates champion selection"}
	followUpExperiment.ExpectedEffect = "Should be blocked before planning."
	followUpPlan, err := server.store.CreateExperimentPlan(projectID, plan.DatasetID, "macro_f1", 1, 5, []plans.PlannedExperiment{followUpExperiment}, nil, "stale_decision")
	if err != nil {
		t.Fatalf("create stale follow-up plan: %v", err)
	}
	followUpJob, _ := createTerminalTrainingJob(t, server, followUpPlan, followUpExperiment, jobs.StatusSucceeded, 0.79)

	handled, err := server.runExperimentPlannerAfterTrainingJob(followUpJob)
	if err != nil {
		t.Fatalf("run experiment planner: %v", err)
	}
	if !handled {
		t.Fatal("expected planner path to be handled by champion guard")
	}
	if llmCalls != 0 {
		t.Fatalf("expected champion guard to skip LLM call, got %d calls", llmCalls)
	}
	if _, ok := experimentPlannerDecisionForPlan(listAgentDecisions(t, server, projectID), followUpPlan.ID); ok {
		t.Fatal("expected no new planner decision after champion selection")
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasBackendStopGuardEvent(events, "champion_selected_guard") {
		t.Fatalf("expected champion-selected guard event, got %#v", events)
	}
}

func TestTrainingMonitorFailureRecordsAgentFailedEvent(t *testing.T) {
	llmCalls := 0
	requestModels := []string{}
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		llmCalls++
		var request struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode llm request: %v", err)
		}
		requestModels = append(requestModels, request.Model)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]string{
					"role":    "assistant",
					"content": "not-json",
				},
			}},
		})
	}))
	defer llmServer.Close()

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_TRAINING_MONITOR_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_MODEL", "visual-fallback-model")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)

	server.runTrainingMonitorAfterTrainingJob(job)
	if llmCalls != 1 {
		t.Fatalf("expected training monitor to call LLM once, got %d", llmCalls)
	}
	if len(requestModels) != 1 || requestModels[0] != "visual-fallback-model" {
		t.Fatalf("expected training monitor to use visual model fallback, got %#v", requestModels)
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasExecutionEvent(events, execution.EventAgentFailed) {
		t.Fatalf("expected agent failure event, got %#v", events)
	}
	if hasExecutionEvent(events, execution.EventExecutionFailed) {
		t.Fatalf("expected monitor failure not to be recorded as execution failure, got %#v", events)
	}
}

func TestTrainingMonitorDisabledByDefaultSkipsLLM(t *testing.T) {
	llmCalls := 0
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		llmCalls++
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]string{
					"role":    "assistant",
					"content": `{"summary":"ok","recommended_action":{"action_type":"RANK_MODELS","confidence":0.5,"rationale":"ok","payload":{},"requires_approval":true},"quality_summary":"ok","training_dynamics":"ok","cost_summary":"ok","risks":[],"findings":[],"rank_score":0.5,"tags":[]}`,
				},
			}},
		})
	}))
	defer llmServer.Close()

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "test-model")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)

	server.runTrainingMonitorAfterTrainingJob(job)
	if llmCalls != 0 {
		t.Fatalf("expected disabled training monitor to skip LLM, got %d calls", llmCalls)
	}
	records, err := server.store.ListProjectAgentMemoryRecords(projectID, memory.AgentMemoryFilter{
		Kind:  memory.KindTrainingEvaluation,
		Limit: 10,
	})
	if err != nil {
		t.Fatalf("list memory records: %v", err)
	}
	if len(records) != 0 {
		t.Fatalf("expected no training monitor memory records, got %#v", records)
	}
}

func TestExperimentPlannerFailureRecordsAgentFailedEvent(t *testing.T) {
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]string{
					"role":    "assistant",
					"content": "not-json",
				},
			}},
		})
	}))
	defer llmServer.Close()

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "test-model")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)

	handled, err := server.runExperimentPlannerAfterTrainingJob(job)
	if err == nil {
		t.Fatal("expected malformed planner output to fail")
	}
	if handled {
		t.Fatal("expected planner failure not to be handled")
	}
	events, eventErr := server.store.ListProjectExecutionEvents(projectID, 10)
	if eventErr != nil {
		t.Fatalf("list execution events: %v", eventErr)
	}
	if !hasExecutionEvent(events, execution.EventAgentFailed) {
		t.Fatalf("expected agent failure event, got %#v", events)
	}
	if hasExecutionEvent(events, execution.EventExecutionFailed) {
		t.Fatalf("expected planner failure not to be recorded as execution failure, got %#v", events)
	}
}

func TestPlanningLoopDoesNotRunDeterministicReviewerAfterLLMPlannerFailure(t *testing.T) {
	llmServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{
				"message": map[string]string{
					"role":    "assistant",
					"content": "not-json",
				},
			}},
		})
	}))
	defer llmServer.Close()

	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_LLM_PROVIDER", "local")
	t.Setenv("MODEL_EXPRESS_LLM_BASE_URL", llmServer.URL)
	t.Setenv("MODEL_EXPRESS_LLM_MODEL", "test-model")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)

	server.runPlanningLoopAfterTrainingJob(job)
	if decisions := listAgentDecisions(t, server, projectID); len(decisions) != 0 {
		t.Fatalf("expected no deterministic fallback decision after planner failure, got %#v", decisions)
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasExecutionEvent(events, execution.EventAgentFailed) {
		t.Fatalf("expected agent failure event, got %#v", events)
	}
	if hasExecutionEvent(events, execution.EventExecutionFailed) {
		t.Fatalf("expected planner failure not to become execution failure, got %#v", events)
	}
}

func TestExperimentPlannerOutcomeRecordedForCompletedFollowUp(t *testing.T) {
	server, projectID, initialPlan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	createTerminalTrainingJob(t, server, initialPlan, initialPlan.Experiments[0], jobs.StatusSucceeded, 0.70)

	input, ready, err := server.buildExperimentPlannerInput(projectID, initialPlan.ID)
	if err != nil {
		t.Fatalf("build planner input: %v", err)
	}
	if !ready {
		t.Fatal("expected planner input to be ready")
	}

	invocation := createExperimentPlannerInvocation(t, server, projectID, initialPlan)
	payload, err := experimentPlannerDecisionPayload(experimentPlannerAddExperimentsRecommendation(), invocation, "autonomous", input)
	if err != nil {
		t.Fatalf("build planner payload: %v", err)
	}
	decision, err := server.store.CreateAgentDecision(projectID, initialPlan.ID, decisions.TypeAddExperiments, "try a stronger planner batch", payload)
	if err != nil {
		t.Fatalf("create planner decision: %v", err)
	}

	followUpPlan, _, err := server.ensureFollowUpPlan(projectID, initialPlan, decision)
	if err != nil {
		t.Fatalf("ensure follow-up plan: %v", err)
	}
	scorecards, err := server.store.ListProjectStrategyScorecards(projectID, 10)
	if err != nil {
		t.Fatalf("list pending strategy scorecards: %v", err)
	}
	if len(scorecards) != 1 {
		t.Fatalf("expected one pending strategy scorecard, got %d", len(scorecards))
	}
	if scorecards[0].Outcome != strategies.OutcomePending {
		t.Fatalf("expected pending strategy scorecard, got %s", scorecards[0].Outcome)
	}
	followUpJob, _ := createTerminalTrainingJob(t, server, followUpPlan, followUpPlan.Experiments[0], jobs.StatusSucceeded, 0.72)

	if err := server.recordExperimentPlannerOutcomeAfterTrainingJob(followUpJob); err != nil {
		t.Fatalf("record planner outcome: %v", err)
	}
	updatedInvocation, err := server.store.GetAgentInvocation(invocation.ID)
	if err != nil {
		t.Fatalf("get updated invocation: %v", err)
	}
	if got := payloadString(updatedInvocation.DownstreamOutcome, "follow_up_plan_id"); got != followUpPlan.ID {
		t.Fatalf("expected downstream outcome for follow-up plan %s, got %s", followUpPlan.ID, got)
	}
	if got := payloadString(updatedInvocation.DownstreamOutcome, "outcome_status"); got != agents.ExperimentPlanningOutcomeImprovedChampion {
		t.Fatalf("expected improved outcome, got %s", got)
	}
	if delta := payloadFloat(updatedInvocation.DownstreamOutcome, "actual_delta_vs_champion"); delta <= 0 {
		t.Fatalf("expected positive champion delta, got %.3f", delta)
	}

	records, err := server.store.ListProjectAgentMemoryRecords(projectID, memory.AgentMemoryFilter{
		Kind: memory.KindPlanningOutcome,
	})
	if err != nil {
		t.Fatalf("list planning outcome memory: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("expected one planning outcome memory record, got %d", len(records))
	}
	if records[0].InvocationID != invocation.ID {
		t.Fatalf("expected outcome memory to link invocation %s, got %s", invocation.ID, records[0].InvocationID)
	}
	scorecards, err = server.store.ListProjectStrategyScorecards(projectID, 10)
	if err != nil {
		t.Fatalf("list updated strategy scorecards: %v", err)
	}
	if scorecards[0].Outcome != agents.ExperimentPlanningOutcomeImprovedChampion {
		t.Fatalf("expected improved scorecard outcome, got %s", scorecards[0].Outcome)
	}
	if scorecards[0].ActualDelta <= 0 {
		t.Fatalf("expected positive scorecard actual delta, got %.3f", scorecards[0].ActualDelta)
	}

	if err := server.recordExperimentPlannerOutcomeAfterTrainingJob(followUpJob); err != nil {
		t.Fatalf("record planner outcome again: %v", err)
	}
	records, err = server.store.ListProjectAgentMemoryRecords(projectID, memory.AgentMemoryFilter{
		Kind: memory.KindPlanningOutcome,
	})
	if err != nil {
		t.Fatalf("list planning outcome memory after rerun: %v", err)
	}
	if len(records) != 1 {
		t.Fatalf("expected outcome recording to be idempotent, got %d records", len(records))
	}

	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasExecutionEvent(events, execution.EventAgentOutcomeRecorded) {
		t.Fatal("expected planner outcome execution event")
	}
}

func newAutomaticReviewFixture(t *testing.T, experiments []plans.PlannedExperiment) (*Server, string, plans.ExperimentPlan) {
	t.Helper()

	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)

	project, err := memoryStore.CreateProject("vision project", "classify images")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	plan, err := memoryStore.CreateExperimentPlan(project.ID, dataset.ID, "macro_f1", len(experiments), 5, experiments, nil, "")
	if err != nil {
		t.Fatalf("create plan: %v", err)
	}

	return server, project.ID, plan
}

func recordTrainingSummary(
	t *testing.T,
	server *Server,
	plan plans.ExperimentPlan,
	experiment plans.PlannedExperiment,
	status string,
	score float64,
	cost float64,
) {
	t.Helper()

	job, err := server.store.CreateJob(plan.ProjectID, jobs.TemplateTrainExperiment, map[string]any{
		"plan_id":          plan.ID,
		"dataset_id":       plan.DatasetID,
		"model":            experiment.Model,
		"provider":         "modal",
		"gpu_type":         "T4",
		"target_metric":    plan.TargetMetric,
		"experiment_index": 0,
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}

	runtimeSeconds := 30.0
	epochsCompleted := experiment.Epochs
	if _, err := server.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
		Status:           status,
		RuntimeSeconds:   &runtimeSeconds,
		EstimatedCostUSD: &cost,
		BestMacroF1:      &score,
		BestAccuracy:     &score,
		EpochsCompleted:  &epochsCompleted,
	}); err != nil {
		t.Fatalf("upsert training summary: %v", err)
	}
}

func testExperiment(model string, epochs int) plans.PlannedExperiment {
	return plans.PlannedExperiment{
		Template:     "mobilenet_transfer",
		Model:        model,
		Epochs:       epochs,
		BatchSize:    16,
		LearningRate: 0.0003,
		Reason:       "test experiment",
	}
}

func listAgentDecisions(t *testing.T, server *Server, projectID string) []decisions.AgentDecision {
	t.Helper()

	agentDecisions, err := server.store.ListProjectAgentDecisions(projectID)
	if err != nil {
		t.Fatalf("list agent decisions: %v", err)
	}
	return agentDecisions
}

func listExperimentPlans(t *testing.T, server *Server, projectID string) []plans.ExperimentPlan {
	t.Helper()

	projectPlans, err := server.store.ListProjectExperimentPlans(projectID)
	if err != nil {
		t.Fatalf("list experiment plans: %v", err)
	}
	return projectPlans
}

func plannedExperimentsFromUnknown(t *testing.T, value any) []plans.PlannedExperiment {
	t.Helper()

	blob, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal experiments payload: %v", err)
	}
	var experiments []plans.PlannedExperiment
	if err := json.Unmarshal(blob, &experiments); err != nil {
		t.Fatalf("decode experiments payload: %v", err)
	}
	return experiments
}

func findProjectJob(t *testing.T, memoryStore *store.MemoryStore, projectID string, template string) jobs.ExperimentJob {
	t.Helper()

	projectJobs, err := memoryStore.ListProjectJobs(projectID)
	if err != nil {
		t.Fatalf("list project jobs: %v", err)
	}
	for _, job := range projectJobs {
		if job.Template == template {
			return job
		}
	}
	t.Fatalf("expected project job with template %s", template)
	return jobs.ExperimentJob{}
}

func pollJobForCallback(t *testing.T, router http.Handler, workerID string, body string) jobs.ExperimentJob {
	t.Helper()
	if body == "" {
		body = `{}`
	}
	req := httptest.NewRequest(http.MethodPost, "/workers/"+workerID+"/poll", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if apiToken := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_API_TOKEN")); apiToken != "" {
		req.Header.Set("X-Model-Express-Api-Token", apiToken)
	}
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("poll job status %d: %s", resp.Code, resp.Body.String())
	}
	var payload pollJobResponse
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode poll response: %v", err)
	}
	if payload.Job == nil {
		t.Fatal("expected poll to return a job")
	}
	if jobConfigString(payload.Job.Config, "active_attempt_id") == "" {
		t.Fatalf("expected active attempt id in poll response config: %#v", payload.Job.Config)
	}
	if jobConfigString(payload.Job.Config, "callback_token") == "" {
		t.Fatalf("expected callback token in poll response config: %#v", payload.Job.Config)
	}
	return *payload.Job
}

func setCallbackToken(t *testing.T, req *http.Request, job jobs.ExperimentJob) {
	t.Helper()
	token := jobConfigString(job.Config, "callback_token")
	if token == "" {
		t.Fatalf("poll response for job %s did not include callback token", job.ID)
	}
	req.Header.Set(callbackTokenHeader, token)
}

func callbackAttemptID(t *testing.T, job jobs.ExperimentJob) string {
	t.Helper()
	attemptID := jobConfigString(job.Config, "active_attempt_id")
	if attemptID == "" {
		t.Fatalf("poll response for job %s did not include active attempt id", job.ID)
	}
	return attemptID
}

func plannerInputForPayload(t *testing.T, server *Server, projectID string) agents.ExperimentPlannerInput {
	t.Helper()

	projectPlans := listExperimentPlans(t, server, projectID)
	summaries, err := server.store.ListProjectTrainingRunSummaries(projectID)
	if err != nil {
		t.Fatalf("list training summaries: %v", err)
	}
	return agents.ExperimentPlannerInput{
		PriorPlans:                   projectPlans,
		ExistingExperimentSignatures: experimentSignaturesForPlans(projectPlans),
		MinimumMeaningfulImprovement: plannerMinimumMeaningfulImprovement,
		SourcePlanDeltas:             []agents.ExperimentRunDelta{},
		StopSignals:                  []string{},
		PlanSummaries:                summaries,
	}
}

func hasExecutionEvent(events []execution.ExecutionEvent, eventType string) bool {
	for _, event := range events {
		if event.EventType == eventType {
			return true
		}
	}
	return false
}

func stringSliceContains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func hasBlockedFollowUpEvent(events []execution.ExecutionEvent) bool {
	for _, event := range events {
		if event.Payload["backend_validation_status"] == "blocked" {
			return true
		}
	}
	return false
}

func hasBackendStopGuardEvent(events []execution.ExecutionEvent, guard string) bool {
	for _, event := range events {
		if event.Payload["backend_stop_guard"] == guard {
			return true
		}
	}
	return false
}

func createTerminalTrainingJob(
	t *testing.T,
	server *Server,
	plan plans.ExperimentPlan,
	experiment plans.PlannedExperiment,
	status string,
	score float64,
) (jobs.ExperimentJob, runs.TrainingRunSummary) {
	t.Helper()

	job, err := server.store.CreateJob(plan.ProjectID, jobs.TemplateTrainExperiment, map[string]any{
		"plan_id":          plan.ID,
		"dataset_id":       plan.DatasetID,
		"model":            experiment.Model,
		"provider":         "modal",
		"gpu_type":         "T4",
		"target_metric":    plan.TargetMetric,
		"experiment_index": 0,
	})
	if err != nil {
		t.Fatalf("create job: %v", err)
	}

	runtimeSeconds := 30.0
	cost := 0.01
	epochsCompleted := experiment.Epochs
	summary, err := server.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
		Status:           status,
		RuntimeSeconds:   &runtimeSeconds,
		EstimatedCostUSD: &cost,
		BestMacroF1:      &score,
		BestAccuracy:     &score,
		EpochsCompleted:  &epochsCompleted,
	})
	if err != nil {
		t.Fatalf("upsert training summary: %v", err)
	}

	return job, summary
}

func setHealthyNearCeilingRunEvidence(t *testing.T, server *Server, jobID string) {
	t.Helper()
	trainLoss := 0.01
	valLoss := 0.012
	if _, err := server.store.UpsertTrainingRunSummary(jobID, runs.TrainingRunSummaryUpdate{
		FinalTrainLoss: &trainLoss,
		FinalValLoss:   &valLoss,
	}); err != nil {
		t.Fatalf("upsert near-ceiling losses: %v", err)
	}
	if _, err := server.store.UpsertTrainingRunEvaluation(jobID, runs.TrainingRunEvaluationUpdate{
		ObjectiveProfile: map[string]any{
			"heldout_test_macro_f1": 1.0,
			"heldout_test_accuracy": 1.0,
			"heldout_test_loss":     0.012,
		},
		PerClassMetrics: map[string]any{
			"cat":       map[string]any{"recall": 1.0, "f1-score": 1.0},
			"dog":       map[string]any{"recall": 1.0, "f1-score": 1.0},
			"macro avg": map[string]any{"f1-score": 1.0},
		},
		ConfusionMatrix: [][]int{{300, 0}, {0, 300}},
		ModelProfile: map[string]any{
			"class_labels":         []string{"cat", "dog"},
			"estimated_latency_ms": 1.0,
		},
		HolisticScores: map[string]any{"overall_score": 1.0},
	}); err != nil {
		t.Fatalf("upsert near-ceiling evaluation: %v", err)
	}
}

func createExperimentPlannerInvocation(t *testing.T, server *Server, projectID string, plan plans.ExperimentPlan) memory.AgentInvocation {
	t.Helper()

	invocation, err := server.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:         projectID,
		DatasetID:         plan.DatasetID,
		PlanID:            plan.ID,
		AgentName:         agents.ExperimentPlannerAgentName,
		AgentVersion:      agents.ExperimentPlannerAgentVersion,
		PromptVersion:     agents.ExperimentPlannerPromptVersion,
		Provider:          "openai",
		Model:             "test-model",
		InputMessages:     []map[string]string{{"role": "user", "content": "plan the next round"}},
		InputContext:      map[string]any{"plan_id": plan.ID},
		RawOutput:         `{"summary":"try a stronger follow-up"}`,
		ParsedOutput:      map[string]any{"summary": "try a stronger follow-up"},
		ValidationStatus:  memory.InvocationValidationValid,
		AcceptedForMemory: true,
	})
	if err != nil {
		t.Fatalf("create invocation: %v", err)
	}

	return invocation
}

func createLLMAddExperimentsDecision(
	t *testing.T,
	server *Server,
	projectID string,
	planID string,
	experiments []plans.PlannedExperiment,
	evidence []string,
) decisions.AgentDecision {
	t.Helper()

	mechanisms := make([]agents.PlannerProposalMechanism, 0, len(experiments))
	for index, experiment := range experiments {
		mechanisms = append(mechanisms, agents.PlannerProposalMechanism{
			ExperimentIndex: index,
			Mechanism:       experiment.Mechanism,
			Intervention:    experiment.Intervention,
			EvidenceUsed:    experiment.EvidenceUsed,
			ExpectedEffect:  experiment.ExpectedEffect,
		})
	}
	decision, err := server.store.CreateAgentDecision(projectID, planID, decisions.TypeAddExperiments, "LLM ADD_EXPERIMENTS test decision", map[string]any{
		"decision_source":      llmExperimentPlannerDecisionSource,
		"evidence_used":        evidence,
		"proposed_experiments": experiments,
		"proposal_mechanisms":  mechanisms,
	})
	if err != nil {
		t.Fatalf("create LLM add experiments decision: %v", err)
	}
	return decision
}

func countJobsByTemplate(projectJobs []jobs.ExperimentJob, template string) int {
	count := 0
	for _, job := range projectJobs {
		if job.Template == template {
			count++
		}
	}
	return count
}

type capturingPlannerGenerator struct {
	response string
}

func (g capturingPlannerGenerator) GenerateJSON(_ context.Context, _ llm.JSONRequest) ([]byte, error) {
	return []byte(g.response), nil
}

func experimentPlannerAddExperimentsRecommendation() agents.ExperimentPlanningRecommendation {
	followUp := plans.PlannedExperiment{
		Template:              "efficientnet_transfer",
		Model:                 "efficientnet_b1",
		Mechanism:             "class_imbalance",
		Intervention:          "Train EfficientNet-B1 with weighted loss, moderate augmentation, and cosine scheduling.",
		EvidenceUsed:          []string{"class_imbalance_score=0.55", "baseline macro-F1 remains below target"},
		ExpectedEffect:        "Improve macro-F1 by addressing minority-class errors while keeping a deployment-capable architecture.",
		Epochs:                12,
		BatchSize:             16,
		LearningRate:          0.0002,
		Reason:                "The baseline is promising but below target, so try a stronger EfficientNet with regularization.",
		ImageSize:             256,
		Optimizer:             "adamw",
		Scheduler:             "cosine",
		WeightDecay:           0.01,
		Augmentation:          map[string]any{"horizontal_flip": true, "color_jitter": true},
		ClassBalancing:        "weighted_loss",
		EarlyStoppingPatience: 3,
		Strategy:              "promising family exploitation",
	}

	return agents.ExperimentPlanningRecommendation{
		AgentName:                     agents.ExperimentPlannerAgentName,
		Summary:                       "The completed plan is promising but needs a stronger follow-up.",
		DecisionType:                  decisions.TypeAddExperiments,
		Rationale:                     "Validation quality improved enough to justify a more aggressive follow-up experiment.",
		Confidence:                    0.83,
		PlanningMode:                  "preprocessing_ablation",
		DeterministicDiagnosisUsed:    []string{"class_imbalance_score=0.55"},
		EvidenceUsed:                  []string{"dataset imbalance and validation gap support a preprocessing ablation"},
		Hypothesis:                    "A stronger efficient architecture with higher resolution, augmentation, and class balancing can beat the current champion.",
		ExpectedFailureModes:          []string{"higher resolution may increase runtime"},
		DatasetPreprocessingRationale: "Use weighted loss and augmentation to improve generalization on smaller or imbalanced image datasets.",
		ChangedVariables:              []string{"model_family", "image_size", "augmentation", "class_balancing"},
		SuccessCriteria:               "Beat the current champion by at least 0.01 macro-F1 without excessive runtime.",
		StopCondition:                 "Select the current champion if the follow-up does not improve macro-F1 enough to justify runtime.",
		DeploymentTradeoff:            "EfficientNet-B1 is slower than MobileNet, so it must show a meaningful quality gain to justify live deployment.",
		ProposedExperiments:           []plans.PlannedExperiment{followUp},
		ProposalMechanisms: []agents.PlannerProposalMechanism{{
			ExperimentIndex: 0,
			Mechanism:       followUp.Mechanism,
			Intervention:    followUp.Intervention,
			EvidenceUsed:    followUp.EvidenceUsed,
			ExpectedEffect:  followUp.ExpectedEffect,
		}},
		WhyCanBeatChampion:      "The proposed run changes architecture, image size, augmentation, scheduler, and regularization instead of only extending epochs.",
		ExpectedDeltaVsChampion: 0.01,
		Risks:                   []string{"higher runtime"},
		ExpectedTradeoffs:       []string{"more quality for more cost"},
		NoveltyNotes:            []string{"larger model, image size, augmentation, scheduler, weight decay"},
		RejectedOptions:         []agents.RejectedPlannerOption{{Option: "more epochs only", Reason: "does not address diagnosis", Evidence: "plateau/class imbalance", AppliesWhen: []string{"plateau", "class_imbalance"}}},
		Tags:                    []string{"follow_up"},
	}
}
