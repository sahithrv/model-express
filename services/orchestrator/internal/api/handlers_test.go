package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
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
