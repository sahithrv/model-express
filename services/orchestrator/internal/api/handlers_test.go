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
		AgentName:               agents.ExperimentPlannerAgentName,
		Summary:                 "The completed plan is promising but needs a stronger follow-up.",
		DecisionType:            decisions.TypeAddExperiments,
		Rationale:               "Validation quality improved enough to justify a more aggressive follow-up experiment.",
		Confidence:              0.83,
		ProposedExperiments:     []plans.PlannedExperiment{followUp},
		WhyCanBeatChampion:      "The proposed run changes architecture, image size, augmentation, scheduler, and regularization instead of only extending epochs.",
		ExpectedDeltaVsChampion: 0.01,
		Risks:                   []string{"higher runtime"},
		ExpectedTradeoffs:       []string{"more quality for more cost"},
		NoveltyNotes:            []string{"larger model, image size, augmentation, scheduler, weight decay"},
		Tags:                    []string{"follow_up"},
	}
}
