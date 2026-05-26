package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func TestValidatePlannedExperimentRejectsDirectVisualAnalysisTemplate(t *testing.T) {
	experiment := testExperiment("efficientnet_b0", 6)
	experiment.Template = "analyze_dataset_visuals"

	err := validatePlannedExperiment(experiment, 0)
	if err == nil || !strings.Contains(strings.ToLower(err.Error()), "template") {
		t.Fatalf("expected direct visual-analysis template to be rejected, got %v", err)
	}
}

func TestDatasetVisualAnalysisResultPersistsAcceptedEvidenceAndInvocation(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{
		"task_type":    "image_classification",
		"class_count":  2,
		"total_images": 4,
	})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	router := NewRouter(memoryStore)
	body, _ := json.Marshal(validVisualAnalysisPayload(dataset))
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected accepted result status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}

	latest, err := memoryStore.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		t.Fatalf("latest accepted visual analysis: %v", err)
	}
	if latest.ValidationStatus != datasets.VisualValidationStatusAccepted || latest.SourceInvocationID == "" {
		t.Fatalf("expected accepted persisted analysis with invocation, got %#v", latest)
	}
	invocation, err := memoryStore.GetAgentInvocation(latest.SourceInvocationID)
	if err != nil {
		t.Fatalf("get invocation: %v", err)
	}
	if invocation.AgentName != datasets.VisualAnalysisAgentName || invocation.ValidationStatus != memory.InvocationValidationValid {
		t.Fatalf("unexpected invocation: %#v", invocation)
	}
}

func TestDatasetVisualAnalysisResultRejectsExecutionAuthority(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	payload := validVisualAnalysisPayload(dataset)
	payload["raw_output"] = `{"proposed_experiments":[{"template":"train_experiment"}]}`
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected rejected result status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
	analyses, err := memoryStore.ListDatasetVisualAnalyses(dataset.ID)
	if err != nil {
		t.Fatalf("list visual analyses: %v", err)
	}
	if len(analyses) != 1 || analyses[0].ValidationStatus != datasets.VisualValidationStatusRejected {
		t.Fatalf("expected rejected visual analysis row, got %#v", analyses)
	}
}

func TestDatasetVisualAnalysisResultRejectsSampleManifestPathLeakage(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	payload := validVisualAnalysisPayload(dataset)
	payload["sample_manifest"] = []map[string]any{
		{"image_id": "img_1", "class_name": "daisy", "width": 100, "height": 100, "selection_basis": []string{"class_representative"}, "metadata": map[string]any{"path": "s3://bucket/daisy/1.jpg"}},
		{"image_id": "img_2", "class_name": "rose", "width": 120, "height": 100, "selection_basis": []string{"class_representative"}},
	}
	body, _ := json.Marshal(payload)
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusBadRequest {
		t.Fatalf("expected rejected path-leaking result status %d, got %d: %s", http.StatusBadRequest, resp.Code, resp.Body.String())
	}
}

func TestVisualAnalysisInvocationContextPreservesBoundaryFlags(t *testing.T) {
	context := visualAnalysisInvocationContext(map[string]any{
		"caps":                            map[string]any{"max_total_images": 48},
		"raw_images_included":             true,
		"raw_visual_output_included":      true,
		"visual_agent_prompt_for_planner": true,
		"unsafe":                          "file:///tmp/image.jpg",
	}, []datasets.VisualSampleManifestItem{{ImageID: "img_1"}})

	if context["raw_images_included"] != false || context["raw_visual_output_included"] != false || context["visual_agent_prompt_for_planner"] != false {
		t.Fatalf("expected reserved boundary flags to remain false, got %#v", context)
	}
	if _, ok := context["unsafe"]; ok {
		t.Fatalf("expected unsafe telemetry key to be dropped, got %#v", context)
	}
	if context["caps"] == nil || context["visual_agent_stream_is_separate"] != true {
		t.Fatalf("expected safe telemetry and separate-stream marker, got %#v", context)
	}
}

func TestRunDatasetVisualAnalysisQueuesBackendOwnedJob(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	if _, err := memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4}); err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analyses/run", strings.NewReader(`{"trigger_reason":"manual"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected run status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	found := false
	for _, job := range projectJobs {
		if job.Template == jobs.TemplateAnalyzeDatasetVisuals && job.Config["dataset_id"] == dataset.ID {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected analyze_dataset_visuals job, got %#v", projectJobs)
	}
}

func TestRunDatasetVisualAnalysisRespectsCooldownAndMaxRuns(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "60")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_MAX_RUNS_PER_PROFILE", "3")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	dataset, err = memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4})
	if err != nil {
		t.Fatalf("profile dataset: %v", err)
	}
	if _, err := memoryStore.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:          project.ID,
		DatasetID:          dataset.ID,
		DatasetName:        dataset.Name,
		ProfileFingerprint: datasetProfileFingerprint(dataset.Profile),
		SchemaVersion:      datasets.VisualAnalysisSchemaVersion,
		TriggerReason:      datasets.VisualTriggerInitialProfile,
		TotalImages:        4,
		ImagesAnalyzed:     2,
		Confidence:         "medium",
		CoverageReport: datasets.VisualCoverageReport{
			ImagesAvailable:    4,
			ImagesAnalyzed:     2,
			ClassesTotal:       2,
			ClassesCovered:     2,
			ClassCoverageRatio: 1,
		},
	}); err != nil {
		t.Fatalf("create accepted visual analysis: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analyses/run", strings.NewReader(`{"trigger_reason":"manual"}`))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusConflict {
		t.Fatalf("expected cooldown conflict, got %d: %s", resp.Code, resp.Body.String())
	}
	if !strings.Contains(resp.Body.String(), "cooling down") {
		t.Fatalf("expected cooldown policy in response, got %s", resp.Body.String())
	}

	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_MAX_RUNS_PER_PROFILE", "1")
	req = httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analyses/run", strings.NewReader(`{"trigger_reason":"manual"}`))
	req.Header.Set("Content-Type", "application/json")
	resp = httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusConflict {
		t.Fatalf("expected max-run conflict, got %d: %s", resp.Code, resp.Body.String())
	}
	if !strings.Contains(resp.Body.String(), "per-profile cap") {
		t.Fatalf("expected max-run policy in response, got %s", resp.Body.String())
	}
}

func TestTrainingRunEvaluationQueuesDeficiencyVisualReanalysis(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_MAX_RUNS_PER_PROFILE", "3")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	if _, err := memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 40, "class_count": 2}); err != nil {
		t.Fatalf("profile dataset: %v", err)
	}
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{
		"dataset_id": dataset.ID,
		"model":      "mobilenet_v3_small",
	})
	if err != nil {
		t.Fatalf("create training job: %v", err)
	}
	runtimeSeconds := 10.0
	costUSD := 0.1
	macroF1 := 0.42
	accuracy := 0.72
	epochs := 3
	if _, err := memoryStore.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
		Model:            "mobilenet_v3_small",
		Provider:         "local",
		Status:           jobs.StatusSucceeded,
		RuntimeSeconds:   &runtimeSeconds,
		EstimatedCostUSD: &costUSD,
		BestMacroF1:      &macroF1,
		BestAccuracy:     &accuracy,
		EpochsCompleted:  &epochs,
	}); err != nil {
		t.Fatalf("upsert summary: %v", err)
	}

	body := `{
		"objective_profile":{"target_metric":"macro_f1"},
		"per_class_metrics":{"rare":{"recall":0.18,"f1-score":0.16},"common":{"recall":0.91,"f1-score":0.88}},
		"confusion_matrix":[[12,18],[2,30]],
		"model_profile":{},
		"holistic_scores":{},
		"recommendation_summary":"low rare-class recall"
	}`
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/training-run-evaluation", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected evaluation accepted, got %d: %s", resp.Code, resp.Body.String())
	}

	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	found := false
	for _, candidate := range projectJobs {
		if candidate.Template != jobs.TemplateAnalyzeDatasetVisuals {
			continue
		}
		found = true
		if candidate.Config["trigger_reason"] != string(datasets.VisualTriggerDeficiencyReanalysis) {
			t.Fatalf("expected deficiency trigger, got %#v", candidate.Config)
		}
		triggerDetails, _ := candidate.Config["trigger_details"].(map[string]any)
		triggers, _ := triggerDetails["deficiency_triggers"].([]string)
		if len(triggers) == 0 {
			t.Fatalf("expected deficiency triggers in job config, got %#v", candidate.Config)
		}
	}
	if !found {
		t.Fatalf("expected deficiency visual-analysis job, got %#v", projectJobs)
	}
}

func TestListDatasetVisualAnalysesExposesRerunPolicy(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	if _, err := memoryStore.UpdateDatasetProfile(dataset.ID, map[string]any{"task_type": "image_classification", "total_images": 4}); err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/datasets/"+dataset.ID+"/visual-analyses", nil)
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected list status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	policy, ok := payload["rerun_policy"].(map[string]any)
	if !ok {
		t.Fatalf("expected rerun_policy object, got %#v", payload)
	}
	if allowed, _ := policy["manual_run_allowed"].(bool); !allowed {
		t.Fatalf("expected manual rerun to be allowed, got %#v", policy)
	}
	if policy["profile_fingerprint"] == "" {
		t.Fatalf("expected profile fingerprint in policy, got %#v", policy)
	}
}

func TestProfileUpdateQueuesVisualAnalysisBeforeInitialPlan(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "high accuracy")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}

	body, _ := json.Marshal(map[string]any{"profile": visualGateProfile()})
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/profile", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected profile status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}

	plans, err := memoryStore.ListProjectExperimentPlans(project.ID)
	if err != nil {
		t.Fatalf("list plans: %v", err)
	}
	if len(plans) != 0 {
		t.Fatalf("expected initial plan to wait for visual analysis, got %#v", plans)
	}
	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	if !hasVisualAnalysisJobForDataset(projectJobs, dataset.ID) {
		t.Fatalf("expected initial visual-analysis job before planning, got %#v", projectJobs)
	}
}

func TestInitialVisualAnalysisResultCreatesInitialPlan(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "high accuracy")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}
	if _, err := memoryStore.UpdateDatasetProfile(dataset.ID, visualGateProfile()); err != nil {
		t.Fatalf("profile dataset: %v", err)
	}

	body, _ := json.Marshal(validVisualAnalysisPayload(dataset))
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/visual-analysis-result", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusCreated {
		t.Fatalf("expected visual result status %d, got %d: %s", http.StatusCreated, resp.Code, resp.Body.String())
	}
	plans, err := memoryStore.ListProjectExperimentPlans(project.ID)
	if err != nil {
		t.Fatalf("list plans: %v", err)
	}
	if len(plans) != 1 || plans[0].DatasetID != dataset.ID {
		t.Fatalf("expected visual result to unblock initial plan, got %#v", plans)
	}
}

func TestInitialVisualAnalysisFailureFallsBackToInitialPlan(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_VISUAL_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_VISUAL_ANALYSIS_COOLDOWN_MINUTES", "0")

	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("demo", "high accuracy")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 0)
	if err != nil {
		t.Fatalf("create dataset: %v", err)
	}

	body, _ := json.Marshal(map[string]any{"profile": visualGateProfile()})
	req := httptest.NewRequest(http.MethodPost, "/datasets/"+dataset.ID+"/profile", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected profile status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	projectJobs, err := memoryStore.ListProjectJobs(project.ID)
	if err != nil {
		t.Fatalf("list jobs: %v", err)
	}
	var visualJob jobs.ExperimentJob
	for _, candidate := range projectJobs {
		if candidate.Template == jobs.TemplateAnalyzeDatasetVisuals {
			visualJob = candidate
			break
		}
	}
	if visualJob.ID == "" {
		t.Fatalf("expected visual job, got %#v", projectJobs)
	}

	req = httptest.NewRequest(http.MethodPost, "/jobs/"+visualJob.ID+"/fail", strings.NewReader(`{"error":"visual llm unavailable"}`))
	req.Header.Set("Content-Type", "application/json")
	resp = httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(resp, req)
	if resp.Code != http.StatusOK {
		t.Fatalf("expected visual fail fallback status %d, got %d: %s", http.StatusOK, resp.Code, resp.Body.String())
	}
	plans, err := memoryStore.ListProjectExperimentPlans(project.ID)
	if err != nil {
		t.Fatalf("list plans: %v", err)
	}
	if len(plans) != 1 || plans[0].DatasetID != dataset.ID {
		t.Fatalf("expected visual failure to fall back to initial plan, got %#v", plans)
	}
}

func visualGateProfile() map[string]any {
	return map[string]any{
		"schema_version":        "dataset_profile_v1",
		"task_type":             "image_classification",
		"class_count":           2,
		"total_images":          120,
		"imbalance_ratio":       1.2,
		"corrupt_image_count":   0,
		"class_names":           []string{"daisy", "rose"},
		"class_distribution":    map[string]int{"daisy": 60, "rose": 60},
		"image_dimension_stats": map[string]any{"width": map[string]any{"median": 224}, "height": map[string]any{"median": 224}},
	}
}

func hasVisualAnalysisJobForDataset(projectJobs []jobs.ExperimentJob, datasetID string) bool {
	for _, job := range projectJobs {
		if job.Template == jobs.TemplateAnalyzeDatasetVisuals && job.Config["dataset_id"] == datasetID {
			return true
		}
	}
	return false
}

func validVisualAnalysisPayload(dataset datasets.Dataset) map[string]any {
	return map[string]any{
		"schema_version":  datasets.VisualAnalysisSchemaVersion,
		"project_id":      dataset.ProjectID,
		"dataset_id":      dataset.ID,
		"dataset_name":    dataset.Name,
		"total_images":    4,
		"images_analyzed": 2,
		"trigger_reason":  "initial_profile",
		"confidence":      "medium",
		"provider":        "fake",
		"model":           "fake-vision",
		"source_job_id":   "job_visual",
		"raw_output":      `{"schema_version":"dataset_visual_analysis_v1"}`,
		"sample_manifest": []map[string]any{
			{"image_id": "img_1", "class_name": "daisy", "width": 100, "height": 100, "selection_basis": []string{"class_representative"}},
			{"image_id": "img_2", "class_name": "rose", "width": 120, "height": 100, "selection_basis": []string{"class_representative"}},
		},
		"coverage_report": map[string]any{
			"selection_strategy":      "deterministic_risk_and_representative_sampling",
			"selection_basis":         []string{"class_representative"},
			"images_available":        4,
			"images_analyzed":         2,
			"classes_total":           2,
			"classes_covered":         2,
			"class_coverage_ratio":    1,
			"per_class_counts":        map[string]int{"daisy": 1, "rose": 1},
			"high_detail_image_count": 0,
			"limitations":             []string{"Sample is bounded."},
		},
		"visual_traits": []map[string]any{
			{
				"trait":             "background_dominance",
				"level":             "medium",
				"confidence":        "medium",
				"evidence":          []string{"Several samples have broad background regions."},
				"example_image_ids": []string{"img_1"},
			},
		},
		"classes_to_watch": []map[string]any{
			{
				"class_name":        "daisy",
				"reason":            "Similar texture in sampled petals.",
				"evidence":          []string{"Petal textures overlap."},
				"example_image_ids": []string{"img_1"},
				"confidence":        "low",
			},
		},
		"preprocessing_hypotheses": []map[string]any{
			{
				"id":                    "vh_001",
				"mechanism":             "resolution_crop",
				"summary":               "Compare preserve_aspect_pad against squash resize.",
				"evidence":              []string{"Aspect ratios vary."},
				"expected_effect":       "Reduce shape distortion.",
				"confidence":            "medium",
				"support_status":        "supported",
				"suggested_image_sizes": []int{224},
				"suggested_preprocessing": map[string]any{
					"resize_strategy": "preserve_aspect_pad",
					"normalization":   "imagenet",
					"crop_strategy":   "none",
					"bbox_mode":       "ignore",
				},
			},
		},
		"cautions": []map[string]any{
			{"operation": "vertical_flip", "reason": "Orientation may matter.", "severity": "medium", "confidence": "medium", "example_image_ids": []string{"img_2"}},
		},
		"limitations": []string{"No experiment should be scheduled from this output without backend validation."},
	}
}
