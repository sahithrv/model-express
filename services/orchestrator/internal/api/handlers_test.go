package api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
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
	enabled := true
	updated, err := applyAutomationSettingsUpdate(server.currentAutomationSettings(), settings.AutomationSettingsUpdate{
		AutoReviewExperiments:   &enabled,
		AutoScheduleFollowUps:   &enabled,
		AutoExecutePlans:        &enabled,
		MaxFollowUpRounds:       &maxRounds,
		DefaultTrainingProvider: &provider,
		DefaultGPUType:          &gpuType,
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

	router := NewRouter(memoryStore)
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/complete", strings.NewReader(`{"mlflow_run_id":"run_1"}`))
	req.Header.Set("Content-Type", "application/json")
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
		DeploymentProfile: map[string]any{"artifact_uri": "file:///first.onnx"},
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

	secondReq := httptest.NewRequest(http.MethodPost, "/projects/"+project.ID+"/champion/exports", strings.NewReader(`{"format":"onnx","artifact_uri":"file:///second.onnx","metadata":{"request":"second"}}`))
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
	if second.ChampionID != champion.ID || second.ArtifactURI != "file:///second.onnx" {
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
	exportJob := findProjectJob(t, memoryStore, project.ID, jobs.TemplateExportChampion)
	if exportJob.Config["export_id"] != export.ID || exportJob.Config["champion_id"] != champion.ID {
		t.Fatalf("unexpected export job config: %#v", exportJob.Config)
	}

	resultReq := httptest.NewRequest(http.MethodPost, "/jobs/"+exportJob.ID+"/champion-export-result", strings.NewReader(`{"status":"READY","artifact_uri":"file:///exports/champion.onnx","metadata":{"labels":["cat","dog"]}}`))
	resultReq.Header.Set("Content-Type", "application/json")
	resultResp := httptest.NewRecorder()
	router.ServeHTTP(resultResp, resultReq)
	if resultResp.Code != http.StatusOK {
		t.Fatalf("expected result status %d, got %d: %s", http.StatusOK, resultResp.Code, resultResp.Body.String())
	}
	exports, err := memoryStore.ListProjectChampionExports(project.ID)
	if err != nil {
		t.Fatalf("list exports: %v", err)
	}
	if exports[0].Status != runs.ChampionExportStatusReady || exports[0].ArtifactURI != "file:///exports/champion.onnx" {
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

	resultReq := httptest.NewRequest(http.MethodPost, "/jobs/"+predictionJob.ID+"/champion-demo-prediction-result", strings.NewReader(`{"status":"SUCCEEDED","predicted_label":"cat","confidence":0.97,"top_k":[{"label":"cat","confidence":0.97}],"latency_ms":12.5,"correct":true}`))
	resultReq.Header.Set("Content-Type", "application/json")
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

	resultReq := httptest.NewRequest(http.MethodPost, "/jobs/"+predictionJob.ID+"/champion-demo-prediction-result", strings.NewReader(`{"status":"RUNTIME_UNAVAILABLE","error":"No worker-owned export manifest path was supplied or found."}`))
	resultReq.Header.Set("Content-Type", "application/json")
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
		assigned, err := memoryStore.PollJob(worker.ID, store.JobPollFilter{})
		if err != nil {
			t.Fatalf("poll attempt %d: %v", attempt, err)
		}
		if assigned.ID != job.ID || assigned.Attempt != attempt {
			t.Fatalf("unexpected assigned job at attempt %d: %#v", attempt, assigned)
		}
		body := strings.NewReader(`{"error":"modal container exited before completion","retryable":true}`)
		req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/fail", body)
		req.Header.Set("Content-Type", "application/json")
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

	assigned, err := memoryStore.PollJob(worker.ID, store.JobPollFilter{})
	if err != nil {
		t.Fatalf("poll final attempt: %v", err)
	}
	if assigned.ID != job.ID || assigned.Attempt != 3 {
		t.Fatalf("unexpected final assignment: %#v", assigned)
	}
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/fail", strings.NewReader(`{"error":"modal container exited before completion","retryable":true}`))
	req.Header.Set("Content-Type", "application/json")
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
	body := []byte(`{"holistic_scores":{"overall_score":0.72},"model_profile":{"estimated_latency_ms":10}}`)
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/training-run-evaluation", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
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

func TestExistingStaleFollowUpPlanIsRevalidatedBeforeExecution(t *testing.T) {
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
	job, _ := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.996448)

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
	if len(projectJobs) != 1 {
		t.Fatalf("expected only the completed source job with auto execution disabled, got %d jobs", len(projectJobs))
	}
}

func TestNearCeilingStaleAddDecisionCannotScheduleFollowUp(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 8),
	})
	createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.996448)
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
	if len(projectJobs) != 1 {
		t.Fatalf("expected only the completed champion job, got %d jobs", len(projectJobs))
	}
	events, err := server.store.ListProjectExecutionEvents(projectID, 10)
	if err != nil {
		t.Fatalf("list execution events: %v", err)
	}
	if !hasBackendStopGuardEvent(events, "champion_selected_guard") {
		t.Fatalf("expected champion-selected blocked event, got %#v", events)
	}
}

func TestPersistedChampionAllowsFollowUpPlanByDefault(t *testing.T) {
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

	followUpPlan, created, err := server.ensureFollowUpPlan(projectID, plan, addDecision)
	if err != nil {
		t.Fatalf("expected follow-up after selected champion by default, got %v", err)
	}
	if !created || followUpPlan.SourceDecisionID != addDecision.ID {
		t.Fatalf("expected created follow-up from %s, got %#v created=%v", addDecision.ID, followUpPlan, created)
	}
	if got := len(listExperimentPlans(t, server, projectID)); got != 2 {
		t.Fatalf("expected initial plus follow-up plan, got %d", got)
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
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "true")

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
	if len(projectJobs) != 1 {
		t.Fatalf("expected no follow-up jobs after champion selection, got %d total jobs", len(projectJobs))
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
