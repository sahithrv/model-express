package api

import (
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

func TestAutonomousTrainingMonitorDecisionSchedulesFollowUp(t *testing.T) {
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
	job, summary := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)
	invocation, record := createTrainingMonitorArtifacts(t, server, job, summary)

	result, err := server.handleTrainingMonitorAgentDecision(job, summary, trainingMonitorAddExperimentsTrace(), invocation, record)
	if err != nil {
		t.Fatalf("handle training monitor decision: %v", err)
	}
	if result.Decision == nil || result.Decision.DecisionType != decisions.TypeAddExperiments {
		t.Fatalf("expected ADD_EXPERIMENTS decision, got %#v", result.Decision)
	}
	if result.FollowUpPlan == nil {
		t.Fatal("expected autonomous follow-up plan")
	}
	if result.FollowUpPlan.SourceDecisionID != result.Decision.ID {
		t.Fatalf("expected follow-up source decision %s, got %s", result.Decision.ID, result.FollowUpPlan.SourceDecisionID)
	}
	if len(result.Jobs) != len(result.FollowUpPlan.Experiments) {
		t.Fatalf("expected queued jobs for follow-up experiments, got %d", len(result.Jobs))
	}

	requirements, err := server.store.ListProjectWorkerRequirements(projectID)
	if err != nil {
		t.Fatalf("list worker requirements: %v", err)
	}
	if len(requirements) != 1 {
		t.Fatalf("expected one worker requirement, got %d", len(requirements))
	}
	if requirements[0].PlanID != result.FollowUpPlan.ID {
		t.Fatalf("expected worker requirement for follow-up plan %s, got %s", result.FollowUpPlan.ID, requirements[0].PlanID)
	}
}

func TestProposedTrainingMonitorDecisionDoesNotAutoSchedule(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_LLM_ENABLED", "true")
	t.Setenv("MODEL_EXPRESS_AGENT_MODE", "propose")
	t.Setenv("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", "true")
	t.Setenv("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", "true")

	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{
		testExperiment("mobilenet_v3_small", 6),
	})
	job, summary := createTerminalTrainingJob(t, server, plan, plan.Experiments[0], jobs.StatusSucceeded, 0.62)
	invocation, record := createTrainingMonitorArtifacts(t, server, job, summary)

	result, err := server.handleTrainingMonitorAgentDecision(job, summary, trainingMonitorAddExperimentsTrace(), invocation, record)
	if err != nil {
		t.Fatalf("handle training monitor decision: %v", err)
	}
	if result.Decision == nil || result.Decision.DecisionType != decisions.TypeAddExperiments {
		t.Fatalf("expected proposed ADD_EXPERIMENTS decision, got %#v", result.Decision)
	}
	if result.FollowUpPlan != nil {
		t.Fatalf("expected propose mode not to create a follow-up plan, got %s", result.FollowUpPlan.ID)
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

func createTrainingMonitorArtifacts(
	t *testing.T,
	server *Server,
	job jobs.ExperimentJob,
	summary runs.TrainingRunSummary,
) (memory.AgentInvocation, memory.AgentMemoryRecord) {
	t.Helper()

	invocation, err := server.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:         job.ProjectID,
		DatasetID:         summary.DatasetID,
		PlanID:            summary.PlanID,
		JobID:             job.ID,
		AgentName:         agents.TrainingMonitorAgentName,
		AgentVersion:      agents.TrainingMonitorAgentVersion,
		PromptVersion:     agents.TrainingMonitorPromptVersion,
		Provider:          "openai",
		Model:             "test-model",
		InputMessages:     []map[string]string{{"role": "user", "content": "evaluate this run"}},
		InputContext:      map[string]any{"job_id": job.ID},
		RawOutput:         `{"summary":"try a focused follow-up"}`,
		ParsedOutput:      map[string]any{"summary": "try a focused follow-up"},
		ValidationStatus:  memory.InvocationValidationValid,
		AcceptedForMemory: true,
	})
	if err != nil {
		t.Fatalf("create invocation: %v", err)
	}

	record, err := server.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		InvocationID: invocation.ID,
		ProjectID:    job.ProjectID,
		DatasetID:    summary.DatasetID,
		PlanID:       summary.PlanID,
		JobID:        job.ID,
		AgentName:    agents.TrainingMonitorAgentName,
		Kind:         memory.KindTrainingEvaluation,
		Summary:      "Try a focused follow-up.",
		Payload:      map[string]any{"rank_score": 0.62},
		Tags:         []string{"follow_up"},
	})
	if err != nil {
		t.Fatalf("create memory record: %v", err)
	}

	return invocation, record
}

func trainingMonitorAddExperimentsTrace() agents.TrainingMonitorEvaluationTrace {
	followUp := plans.PlannedExperiment{
		Template:     "efficientnet_transfer",
		Model:        "efficientnet_b0",
		Epochs:       8,
		BatchSize:    16,
		LearningRate: 0.0002,
		Reason:       "The baseline is promising but below target, so try a focused EfficientNet follow-up.",
	}

	return agents.TrainingMonitorEvaluationTrace{
		Recommendation: agents.TrainingEvaluationRecommendation{
			AgentName: agents.TrainingMonitorAgentName,
			Summary:   "The completed run is promising but needs a stronger follow-up.",
			RecommendedAction: decisions.AgentActionProposal{
				ActionType: decisions.TypeAddExperiments,
				Confidence: 0.81,
				Rationale:  "Validation quality improved enough to justify a focused follow-up experiment.",
				Payload: map[string]any{
					"proposed_experiments": []plans.PlannedExperiment{followUp},
				},
				RequiresApproval: false,
			},
			QualitySummary:   "Moderate validation quality.",
			TrainingDynamics: "No severe instability.",
			CostSummary:      "Low cost so far.",
			Findings:         []string{"baseline can be improved"},
			RankScore:        0.62,
			Tags:             []string{"follow_up"},
		},
		ValidationStatus: memory.InvocationValidationValid,
		AgentVersion:     agents.TrainingMonitorAgentVersion,
		PromptVersion:    agents.TrainingMonitorPromptVersion,
		PromptContext:    map[string]any{"test": true},
	}
}
