package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/agents"
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
		assigned, err := memoryStore.PollJob(worker.ID)
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
	assigned, err := memoryStore.PollJob(worker.ID)
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
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "minio://datasets/example", "", 0)
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
}

func TestWorkerRequirementResetsPendingWhenTargetChanges(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	requirement, _, err := server.store.UpsertWorkerRequirement(projectID, plan.ID, "modal", "T4", 1, "test")
	if err != nil {
		t.Fatalf("upsert worker requirement: %v", err)
	}
	active := execution.WorkerRequirementActive
	if _, err := server.store.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{Status: &active}); err != nil {
		t.Fatalf("mark worker requirement active: %v", err)
	}
	updated, _, err := server.store.UpsertWorkerRequirement(projectID, plan.ID, "modal", "T4", 2, "test")
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
	if objective.RankingWeights["latency"] <= 0.1 {
		t.Fatalf("expected latency to be weighted strongly, got %.3f", objective.RankingWeights["latency"])
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
			"proposed_experiments": [{
				"template": "mobilenet_transfer",
				"model": "mobilenet_v3_small",
				"epochs": 12,
				"batch_size": 16,
				"learning_rate": 0.0003,
				"reason": "Repeat MobileNet with more epochs."
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

	_, err := experimentPlannerDecisionPayload(duplicate, invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err == nil {
		t.Fatal("expected duplicate proposed experiment to fail validation")
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
	repeat.Reason = "Try the same mechanism for more epochs."

	recommendation := agents.ExperimentPlanningRecommendation{
		AgentName:           agents.ExperimentPlannerAgentName,
		Summary:             "More epochs only should not schedule.",
		DecisionType:        decisions.TypeAddExperiments,
		Rationale:           "This intentionally repeats the existing mechanism with only more epochs.",
		Confidence:          0.7,
		ProposedExperiments: []plans.PlannedExperiment{repeat},
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

func TestExperimentPlannerStopCriteriaSelectsNearCeilingChampion(t *testing.T) {
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
		WhyCanBeatChampion:            "The proposed run changes architecture, image size, augmentation, scheduler, and regularization instead of only extending epochs.",
		ExpectedDeltaVsChampion:       0.01,
		Risks:                         []string{"higher runtime"},
		ExpectedTradeoffs:             []string{"more quality for more cost"},
		NoveltyNotes:                  []string{"larger model, image size, augmentation, scheduler, weight decay"},
		RejectedOptions:               []agents.RejectedPlannerOption{{Option: "more epochs only", Reason: "does not address diagnosis", Evidence: "plateau/class imbalance", AppliesWhen: []string{"plateau", "class_imbalance"}}},
		Tags:                          []string{"follow_up"},
	}
}
