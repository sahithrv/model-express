package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
	"model-express/services/orchestrator/internal/workers"
)

func TestFakeRunSmokeEndToEndSuccessVisibility(t *testing.T) {
	harness := newRunSmokeHarness(t)
	project, dataset, plan := harness.createProfiledProjectDatasetAndPlan()
	worker := harness.registerWorker(project.ID, "smoke modal worker", "modal")

	var executionResult executeExperimentPlanResponse
	harness.postJSON("/plans/"+plan.ID+"/execute", executeExperimentPlanRequest{
		Provider:          "modal",
		GPUType:           "T4",
		MaxConcurrentJobs: 1,
	}, http.StatusCreated, &executionResult)
	if len(executionResult.Jobs) != 1 {
		t.Fatalf("expected one queued training job, got %#v", executionResult.Jobs)
	}
	if executionResult.WorkerRequirement == nil ||
		executionResult.WorkerRequirement.Status != execution.WorkerRequirementActive ||
		executionResult.WorkerRequirement.TargetCount != 1 {
		t.Fatalf("expected visible active worker requirement, got %#v", executionResult.WorkerRequirement)
	}

	assigned := pollJobForCallback(t, harness.router, worker.ID, `{"provider":"modal"}`)
	if assigned.ID != executionResult.Jobs[0].ID ||
		assigned.Status != jobs.StatusAssigned ||
		assigned.WorkerID != worker.ID ||
		assigned.StartedAt == nil {
		t.Fatalf("expected queued training job to be assigned to worker, got %#v", assigned)
	}

	harness.postCallbackJSON("/jobs/"+assigned.ID+"/metrics", assigned, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"epoch":               1,
		"metrics": map[string]float64{
			"train_loss": 0.42,
			"val_loss":   0.31,
			"macro_f1":   0.99,
		},
	}, http.StatusCreated, nil)
	harness.reportSuccessfulTrainingSummary(assigned, plan.ID, dataset.ID, 0.99)
	harness.reportSuccessfulTrainingEvaluation(assigned, plan.ID, dataset.ID, 0.99)
	harness.postCallbackJSON("/jobs/"+assigned.ID+"/complete", assigned, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"mlflow_run_id":       "smoke-run-1",
	}, http.StatusOK, nil)

	var decision decisions.AgentDecision
	harness.postJSON("/projects/"+project.ID+"/review-experiments", nil, http.StatusCreated, &decision)
	if decision.ID == "" {
		t.Fatalf("expected reviewer decision to be persisted, got %#v", decision)
	}
	if decision.DecisionType != decisions.TypeSelectChampion {
		t.Fatalf("expected reviewer to select smoke champion, got %s with payload %#v", decision.DecisionType, decision.Payload)
	}

	var championPayload struct {
		Champion *runs.ProjectChampion `json:"champion"`
	}
	harness.getJSON("/projects/"+project.ID+"/champion", http.StatusOK, &championPayload)
	if championPayload.Champion == nil ||
		championPayload.Champion.JobID != assigned.ID ||
		championPayload.Champion.PlanID != plan.ID {
		t.Fatalf("expected smoke job champion for plan %s, got %#v", plan.ID, championPayload)
	}

	var jobsPayload struct {
		Jobs []jobs.ExperimentJob `json:"jobs"`
	}
	harness.getJSON("/projects/"+project.ID+"/jobs", http.StatusOK, &jobsPayload)
	if !smokeJobWithStatus(jobsPayload.Jobs, assigned.ID, jobs.StatusSucceeded) {
		t.Fatalf("expected succeeded job to be UI-visible, got %#v", jobsPayload.Jobs)
	}

	var evaluationsPayload struct {
		Evaluations []runs.TrainingRunEvaluation `json:"evaluations"`
	}
	harness.getJSON("/projects/"+project.ID+"/training-run-evaluations?compact=1", http.StatusOK, &evaluationsPayload)
	if len(evaluationsPayload.Evaluations) != 1 || evaluationsPayload.Evaluations[0].JobID != assigned.ID {
		t.Fatalf("expected compact evaluation to be UI-visible, got %#v", evaluationsPayload)
	}

	var decisionsPayload struct {
		Decisions []decisions.AgentDecision `json:"decisions"`
	}
	harness.getJSON("/projects/"+project.ID+"/agent-decisions", http.StatusOK, &decisionsPayload)
	if len(decisionsPayload.Decisions) == 0 || decisionsPayload.Decisions[0].ID != decision.ID {
		t.Fatalf("expected reviewer decision to be UI-visible, got %#v", decisionsPayload)
	}
}

func TestFakeRunSmokeMidRunFailureStaysVisibleAndRejectsStaleCompletion(t *testing.T) {
	harness := newRunSmokeHarness(t)
	project, _, plan := harness.createProfiledProjectDatasetAndPlan()
	worker := harness.registerWorker(project.ID, "smoke modal worker", "modal")

	var executionResult executeExperimentPlanResponse
	harness.postJSON("/plans/"+plan.ID+"/execute", executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"}, http.StatusCreated, &executionResult)
	assigned := pollJobForCallback(t, harness.router, worker.ID, `{"provider":"modal"}`)

	harness.postCallbackJSON("/jobs/"+assigned.ID+"/metrics", assigned, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"epoch":               1,
		"metrics":             map[string]float64{"val_loss": 9.9},
	}, http.StatusCreated, nil)
	harness.postCallbackJSON("/jobs/"+assigned.ID+"/fail", assigned, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"error":               "CUDA OOM during smoke test",
		"failure_class":       "worker_runtime",
		"retryable":           false,
	}, http.StatusOK, nil)

	harness.postCallbackJSON("/jobs/"+assigned.ID+"/complete", assigned, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"mlflow_run_id":       "late-success",
	}, http.StatusConflict, nil)

	var failedJob jobs.ExperimentJob
	harness.getJSON("/jobs/"+assigned.ID, http.StatusOK, &failedJob)
	if failedJob.Status != jobs.StatusFailed || !strings.Contains(failedJob.Error, "CUDA OOM") {
		t.Fatalf("expected failed job with worker error, got %#v", failedJob)
	}

	var summariesPayload struct {
		Summaries []runs.TrainingRunSummary `json:"summaries"`
	}
	harness.getJSON("/projects/"+project.ID+"/training-run-summaries", http.StatusOK, &summariesPayload)
	if len(summariesPayload.Summaries) != 1 || summariesPayload.Summaries[0].Status != jobs.StatusFailed {
		t.Fatalf("expected failed training summary to be visible, got %#v", summariesPayload)
	}

	var metricsPayload struct {
		Metrics []jobs.EpochMetric `json:"metrics"`
	}
	harness.getJSON("/jobs/"+assigned.ID+"/metrics", http.StatusOK, &metricsPayload)
	if len(metricsPayload.Metrics) != 1 || metricsPayload.Metrics[0].Metrics["val_loss"] != 9.9 {
		t.Fatalf("expected pre-failure metric to remain visible, got %#v", metricsPayload)
	}

	var championPayload struct {
		Champion *runs.ProjectChampion `json:"champion"`
	}
	harness.getJSON("/projects/"+project.ID+"/champion", http.StatusOK, &championPayload)
	if championPayload.Champion != nil {
		t.Fatalf("failed run should not expose a champion, got %#v", championPayload.Champion)
	}
}

func TestDatasetProfileFailureSmokeStaysVisibleAndDoesNotCreatePlan(t *testing.T) {
	harness := newRunSmokeHarness(t)
	project := harness.createProject()
	dataset := harness.createDataset(project.ID)
	worker := harness.registerWorker(project.ID, "profile worker", "modal")
	assignedProfile := pollJobForCallback(
		t,
		harness.router,
		worker.ID,
		`{"provider":"modal","templates":["profile_dataset"],"include_unspecified_provider_templates":["profile_dataset"]}`,
	)

	harness.postCallbackJSON("/jobs/"+assignedProfile.ID+"/fail", assignedProfile, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assignedProfile),
		"error":               "dataset archive could not be extracted",
		"failure_class":       "dataset_error",
	}, http.StatusOK, nil)

	var failedProfile jobs.ExperimentJob
	harness.getJSON("/jobs/"+assignedProfile.ID, http.StatusOK, &failedProfile)
	if failedProfile.Status != jobs.StatusFailed || !strings.Contains(failedProfile.Error, "dataset archive") {
		t.Fatalf("expected failed profile job with dataset error, got %#v", failedProfile)
	}

	var visibleDataset datasets.Dataset
	harness.getJSON("/datasets/"+dataset.ID, http.StatusOK, &visibleDataset)
	if visibleDataset.Status != datasets.StatusRegistered {
		t.Fatalf("expected dataset to remain unprofiled after profile failure, got %#v", visibleDataset)
	}

	var plansPayload struct {
		Plans []plans.ExperimentPlan `json:"plans"`
	}
	harness.getJSON("/projects/"+project.ID+"/plans", http.StatusOK, &plansPayload)
	if len(plansPayload.Plans) != 0 {
		t.Fatalf("profile failure should not create experiment plans, got %#v", plansPayload.Plans)
	}

	var jobsPayload struct {
		Jobs []jobs.ExperimentJob `json:"jobs"`
	}
	harness.getJSON("/projects/"+project.ID+"/jobs", http.StatusOK, &jobsPayload)
	if !smokeJobWithStatus(jobsPayload.Jobs, assignedProfile.ID, jobs.StatusFailed) {
		t.Fatalf("expected failed profile job to be UI-visible, got %#v", jobsPayload.Jobs)
	}
}

func TestFakeRunSmokeNoModalWorkerCreatesVisibleBlocker(t *testing.T) {
	harness := newRunSmokeHarness(t)
	project := harness.createProject()
	dataset := harness.createDataset(project.ID)
	localWorker := harness.registerWorker(project.ID, "local profile worker", "local")
	assignedProfile := pollJobForCallback(
		t,
		harness.router,
		localWorker.ID,
		`{"provider":"local","templates":["profile_dataset"],"include_unspecified_provider_templates":["profile_dataset"]}`,
	)
	harness.postJSON("/datasets/"+dataset.ID+"/profile", map[string]any{
		"profile": map[string]any{
			"task_type":          "image_classification",
			"image_count":        8,
			"total_images":       8,
			"class_count":        2,
			"class_distribution": map[string]any{"cat": 4, "dog": 4},
			"dataset_traits":     []string{"balanced_classes", "tiny_smoke_dataset"},
		},
	}, http.StatusOK, nil)
	harness.postCallbackJSON("/jobs/"+assignedProfile.ID+"/complete", assignedProfile, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assignedProfile),
		"mlflow_run_id":       "profile-local-smoke",
	}, http.StatusOK, nil)

	var plan plans.ExperimentPlan
	harness.postJSON("/projects/"+project.ID+"/plans", createExperimentPlanRequest{
		DatasetID:          dataset.ID,
		TargetMetric:       "macro_f1",
		RecommendedWorkers: 1,
		EstimatedMinutes:   2,
		Experiments:        []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 2)},
	}, http.StatusCreated, &plan)

	var executionResult executeExperimentPlanResponse
	harness.postJSON("/plans/"+plan.ID+"/execute", executeExperimentPlanRequest{
		Provider:          "modal",
		GPUType:           "T4",
		MaxConcurrentJobs: 1,
	}, http.StatusCreated, &executionResult)
	if len(executionResult.Jobs) != 1 {
		t.Fatalf("expected blocked run to still queue one training job, got %#v", executionResult.Jobs)
	}
	if executionResult.WorkerRequirement == nil {
		t.Fatalf("expected visible worker requirement, got %#v", executionResult)
	}
	if executionResult.WorkerRequirement.Status != execution.WorkerRequirementPending ||
		executionResult.WorkerRequirement.Provider != "modal" ||
		executionResult.WorkerRequirement.DatasetID != dataset.ID ||
		executionResult.WorkerRequirement.TargetCount != 1 {
		t.Fatalf("expected pending modal blocker scoped to dataset, got %#v", executionResult.WorkerRequirement)
	}

	var requirementsPayload struct {
		Requirements []execution.WorkerRequirement `json:"requirements"`
	}
	harness.getJSON("/projects/"+project.ID+"/worker-requirements", http.StatusOK, &requirementsPayload)
	if len(requirementsPayload.Requirements) != 1 ||
		requirementsPayload.Requirements[0].Status != execution.WorkerRequirementPending {
		t.Fatalf("expected pending blocker to be UI-visible, got %#v", requirementsPayload)
	}
}

type runSmokeHarness struct {
	t      *testing.T
	store  *store.MemoryStore
	router http.Handler
}

func newRunSmokeHarness(t *testing.T) runSmokeHarness {
	t.Helper()
	t.Setenv("MODEL_EXPRESS_AGENT_MODE", "")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "false")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "false")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "false")
	t.Setenv("MODEL_EXPRESS_COST_MODES", "false")
	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "false")
	t.Setenv("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", "false")
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", "false")
	t.Setenv("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", "false")
	t.Setenv("MODEL_EXPRESS_TRAINING_MONITOR_LLM_ENABLED", "false")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_ENABLED", "0")
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_ENABLED", "false")
	memoryStore := store.NewMemoryStore()
	return runSmokeHarness{
		t:      t,
		store:  memoryStore,
		router: NewRouter(memoryStore),
	}
}

func (h runSmokeHarness) createProfiledProjectDatasetAndPlan() (projects.Project, datasets.Dataset, plans.ExperimentPlan) {
	h.t.Helper()
	project := h.createProject()
	dataset := h.createDataset(project.ID)
	worker := h.registerWorker(project.ID, "profile worker", "modal")
	assignedProfile := pollJobForCallback(
		h.t,
		h.router,
		worker.ID,
		`{"provider":"modal","templates":["profile_dataset"],"include_unspecified_provider_templates":["profile_dataset"]}`,
	)
	h.postJSON("/datasets/"+dataset.ID+"/profile", map[string]any{
		"profile": map[string]any{
			"task_type":          "image_classification",
			"image_count":        8,
			"total_images":       8,
			"class_count":        2,
			"class_distribution": map[string]any{"cat": 4, "dog": 4},
			"dataset_traits":     []string{"balanced_classes", "tiny_smoke_dataset"},
		},
	}, http.StatusOK, nil)
	h.postCallbackJSON("/jobs/"+assignedProfile.ID+"/complete", assignedProfile, map[string]any{
		"training_attempt_id": callbackAttemptID(h.t, assignedProfile),
		"mlflow_run_id":       "profile-smoke",
	}, http.StatusOK, nil)

	time.Sleep(time.Millisecond)

	var plan plans.ExperimentPlan
	h.postJSON("/projects/"+project.ID+"/plans", createExperimentPlanRequest{
		DatasetID:          dataset.ID,
		TargetMetric:       "macro_f1",
		RecommendedWorkers: 1,
		EstimatedMinutes:   2,
		Experiments:        []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 2)},
	}, http.StatusCreated, &plan)
	return project, dataset, plan
}

func (h runSmokeHarness) createProject() projects.Project {
	h.t.Helper()
	var project projects.Project
	h.postJSON("/projects", map[string]any{
		"name": "smoke project",
		"goal": "prove run contract visibility",
	}, http.StatusCreated, &project)
	return project
}

func (h runSmokeHarness) createDataset(projectID string) datasets.Dataset {
	h.t.Helper()
	var dataset datasets.Dataset
	h.postJSON("/projects/"+projectID+"/datasets", map[string]any{
		"name":            "tiny smoke dataset",
		"storage_uri":     "s3://model-express/datasets/smoke-dataset.zip",
		"checksum_sha256": strings.Repeat("a", 64),
		"size_bytes":      128,
	}, http.StatusCreated, &dataset)
	return dataset
}

func (h runSmokeHarness) registerWorker(projectID string, name string, gpuType string) workers.Worker {
	h.t.Helper()
	var worker workers.Worker
	h.postJSON("/workers/register", map[string]any{
		"project_id": projectID,
		"name":       name,
		"gpu_type":   gpuType,
	}, http.StatusCreated, &worker)
	return worker
}

func (h runSmokeHarness) reportSuccessfulTrainingSummary(job jobs.ExperimentJob, planID string, datasetID string, score float64) {
	h.t.Helper()
	runtimeSeconds := 42.0
	cost := 0.12
	trainLoss := 0.03
	valLoss := 0.02
	epochsCompleted := 2
	h.postCallbackJSON("/jobs/"+job.ID+"/training-run-summary", job, map[string]any{
		"training_attempt_id": callbackAttemptID(h.t, job),
		"model":               "mobilenet_v3_small",
		"provider":            "modal",
		"gpu_type":            "T4",
		"status":              jobs.StatusSucceeded,
		"runtime_seconds":     runtimeSeconds,
		"estimated_cost_usd":  cost,
		"best_macro_f1":       score,
		"best_accuracy":       score,
		"final_train_loss":    trainLoss,
		"final_val_loss":      valLoss,
		"epochs_completed":    epochsCompleted,
		"dataset_materialization": map[string]any{
			"dataset_id":                              datasetID,
			"dataset_materialization_status":          "hit",
			"dataset_materialization_cache_key":       "sha256-" + strings.Repeat("a", 64),
			"dataset_materialization_elapsed_seconds": 0.01,
		},
		"stage_telemetry": map[string]any{
			"fake_trainer": true,
			"plan_id":      planID,
		},
	}, http.StatusOK, nil)
}

func (h runSmokeHarness) reportSuccessfulTrainingEvaluation(job jobs.ExperimentJob, planID string, datasetID string, score float64) {
	h.t.Helper()
	h.postCallbackJSON("/jobs/"+job.ID+"/training-run-evaluation", job, map[string]any{
		"training_attempt_id": callbackAttemptID(h.t, job),
		"objective_profile": map[string]any{
			"target_metric": "macro_f1",
			"task_type":     "image_classification",
		},
		"per_class_metrics": map[string]any{
			"cat": map[string]any{"precision": 1.0, "recall": 0.9, "f1": 0.95},
			"dog": map[string]any{"precision": 0.9, "recall": 1.0, "f1": 0.95},
		},
		"confusion_matrix": [][]int{{4, 0}, {1, 3}},
		"model_profile": map[string]any{
			"model_kind":    "image_classifier",
			"artifact_uri":  "s3://model-express/model-express/artifacts/" + job.ID + "/model.onnx",
			"dataset_id":    datasetID,
			"plan_id":       planID,
			"export_status": "ready",
		},
		"holistic_scores": map[string]any{
			"macro_f1": score,
			"accuracy": score,
		},
		"recommendation_summary": "Fake smoke trainer produced a deployable result.",
	}, http.StatusOK, nil)
}

func (h runSmokeHarness) postCallbackJSON(path string, job jobs.ExperimentJob, body any, wantStatus int, out any) {
	h.t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		h.t.Fatalf("marshal callback body: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	setCallbackToken(h.t, req, job)
	resp := httptest.NewRecorder()
	h.router.ServeHTTP(resp, req)
	if resp.Code != wantStatus {
		h.t.Fatalf("POST %s status %d, want %d: %s", path, resp.Code, wantStatus, resp.Body.String())
	}
	if out != nil {
		if err := json.Unmarshal(resp.Body.Bytes(), out); err != nil {
			h.t.Fatalf("decode POST %s response: %v", path, err)
		}
	}
}

func (h runSmokeHarness) postJSON(path string, body any, wantStatus int, out any) {
	h.t.Helper()
	var raw []byte
	if body != nil {
		var err error
		raw, err = json.Marshal(body)
		if err != nil {
			h.t.Fatalf("marshal POST %s body: %v", path, err)
		}
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	h.router.ServeHTTP(resp, req)
	if resp.Code != wantStatus {
		h.t.Fatalf("POST %s status %d, want %d: %s", path, resp.Code, wantStatus, resp.Body.String())
	}
	if out != nil {
		if err := json.Unmarshal(resp.Body.Bytes(), out); err != nil {
			h.t.Fatalf("decode POST %s response: %v", path, err)
		}
	}
}

func (h runSmokeHarness) getJSON(path string, wantStatus int, out any) {
	h.t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	resp := httptest.NewRecorder()
	h.router.ServeHTTP(resp, req)
	if resp.Code != wantStatus {
		h.t.Fatalf("GET %s status %d, want %d: %s", path, resp.Code, wantStatus, resp.Body.String())
	}
	if out != nil {
		if err := json.Unmarshal(resp.Body.Bytes(), out); err != nil {
			h.t.Fatalf("decode GET %s response: %v", path, err)
		}
	}
}

func smokeJobWithStatus(projectJobs []jobs.ExperimentJob, jobID string, status string) bool {
	for _, job := range projectJobs {
		if job.ID == jobID && job.Status == status {
			return true
		}
	}
	return false
}
