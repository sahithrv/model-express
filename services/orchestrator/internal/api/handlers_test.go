package api

import (
	"testing"

	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
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
