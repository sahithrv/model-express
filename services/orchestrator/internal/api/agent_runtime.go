package api

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/diagnostics"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
	"model-express/services/orchestrator/internal/strategies"
)

func (s *Server) recordExperimentPlannerOutcomeAfterTrainingJob(job jobs.ExperimentJob) error {
	summary, err := s.store.GetTrainingRunSummary(job.ID)
	if err != nil || summary.PlanID == "" {
		return nil
	}

	s.autoReviewMu.Lock()
	defer s.autoReviewMu.Unlock()

	plan, err := s.store.GetExperimentPlan(summary.PlanID)
	if err != nil {
		return err
	}
	if plan.SourceDecisionID == "" {
		return nil
	}

	summaries, err := s.store.ListProjectTrainingRunSummaries(job.ProjectID)
	if err != nil {
		return err
	}
	planSummaries := summariesForPlanID(summaries, plan.ID)
	if !planTrainingRunsComplete(plan, planSummaries) {
		return nil
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(job.ProjectID)
	if err != nil {
		return err
	}
	sourceDecision, ok := agentDecisionByID(agentDecisions, plan.SourceDecisionID)
	if !ok || sourceDecision.Payload["decision_source"] != llmExperimentPlannerDecisionSource {
		return nil
	}

	invocationID := payloadString(sourceDecision.Payload, "invocation_id")
	if invocationID == "" {
		return nil
	}
	invocation, err := s.store.GetAgentInvocation(invocationID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return nil
		}
		return err
	}
	if payloadString(invocation.DownstreamOutcome, "follow_up_plan_id") == plan.ID {
		return nil
	}

	projectPlans, err := s.store.ListProjectExperimentPlans(job.ProjectID)
	if err != nil {
		return err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(job.ProjectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return err
	}
	goalText := ""
	if project, err := s.store.GetProject(job.ProjectID); err == nil {
		goalText = project.Goal
	} else if !errors.Is(err, store.ErrNotFound) {
		return err
	}
	outcome, err := experimentPlanningOutcomeForPlan(sourceDecision, plan, projectPlans, summaries, evaluations, projectObjectiveContext(goalText))
	if err != nil {
		return err
	}
	outcomePayload, err := mapFromStruct(outcome)
	if err != nil {
		return err
	}

	updatedInvocation, err := s.store.UpdateAgentInvocationDownstreamOutcome(invocationID, outcomePayload)
	if err != nil {
		return err
	}

	tags := plannerOutcomeTags(outcome)
	record, err := s.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		InvocationID: updatedInvocation.ID,
		ProjectID:    job.ProjectID,
		DatasetID:    plan.DatasetID,
		PlanID:       plan.ID,
		AgentName:    agents.ExperimentPlannerAgentName,
		Kind:         memory.KindPlanningOutcome,
		Summary:      outcome.Lesson,
		Payload:      outcomePayload,
		Tags:         tags,
	})
	if err != nil {
		return err
	}
	s.indexMemoryCard(context.Background(), memory.NewAgentMemoryCard(record))

	if _, err := s.store.CreateExecutionEvent(job.ProjectID, plan.ID, execution.EventAgentOutcomeRecorded, fmt.Sprintf("Experiment Planner outcome recorded for follow-up plan %s.", plan.ID), map[string]any{
		"invocation_id":      updatedInvocation.ID,
		"memory_record_id":   record.ID,
		"source_decision_id": sourceDecision.ID,
		"outcome_status":     outcome.OutcomeStatus,
	}); err != nil {
		log.Printf("record experiment planner outcome event failed: %v", err)
	}
	if updatedScorecard, err := s.store.UpdateStrategyScorecardOutcomeByFollowUpPlan(plan.ID, strategies.StrategyScorecardOutcomeUpdate{
		ActualDelta:     outcome.ActualDeltaVsChampion,
		ConfidenceAfter: plannerOutcomeConfidence(outcome),
		CostUSD:         outcome.TotalCostUSD,
		RuntimeSeconds:  outcome.TotalRuntimeSeconds,
		Outcome:         outcome.OutcomeStatus,
		Lesson:          outcome.Lesson,
		Tags:            tags,
	}); err == nil {
		s.indexMemoryCard(context.Background(), memory.NewStrategyScorecardMemoryCard(updatedScorecard))
	} else if !errors.Is(err, store.ErrNotFound) {
		log.Printf("update strategy scorecard failed for follow-up plan %s: %v", plan.ID, err)
	}
	return nil
}

func (s *Server) runTrainingMonitorAfterTrainingJob(job jobs.ExperimentJob) {
	if !trainingMonitorLLMEnabled() {
		return
	}
	if !s.shouldRunLLMAgents() {
		return
	}

	summary, err := s.store.GetTrainingRunSummary(job.ID)
	if err != nil {
		log.Printf("training monitor skipped for job %s: summary unavailable: %v", job.ID, err)
		return
	}

	metrics, err := s.store.ListJobMetrics(job.ID)
	if err != nil {
		log.Printf("training monitor skipped for job %s: metrics unavailable: %v", job.ID, err)
		return
	}

	plan := plans.ExperimentPlan{}
	if summary.PlanID != "" {
		plan, err = s.store.GetExperimentPlan(summary.PlanID)
		if err != nil && !errors.Is(err, store.ErrNotFound) {
			log.Printf("training monitor skipped for job %s: plan unavailable: %v", job.ID, err)
			return
		}
	}
	goalText := ""
	project, err := s.store.GetProject(job.ProjectID)
	if err != nil {
		log.Printf("training monitor skipped project objective context for job %s: project unavailable: %v", job.ID, err)
	} else {
		goalText = project.Goal
	}
	var evaluation *runs.TrainingRunEvaluation
	if storedEvaluation, err := s.store.GetTrainingRunEvaluation(job.ID); err == nil {
		evaluation = &storedEvaluation
	} else if !errors.Is(err, store.ErrNotFound) {
		log.Printf("training monitor evaluation lookup failed for job %s: %v", job.ID, err)
	}

	priorMemory, err := s.store.ListProjectAgentMemoryRecords(job.ProjectID, memory.AgentMemoryFilter{
		DatasetID: summary.DatasetID,
		Limit:     12,
	})
	if err != nil {
		log.Printf("training monitor memory lookup failed for job %s: %v", job.ID, err)
		priorMemory = []memory.AgentMemoryRecord{}
	}

	automationSettings := s.currentAutomationSettings()
	config := llm.ConfigFromEnv(automationSettings.LLMEnabled, automationSettings.LLMProvider, automationSettings.LLMModel)
	client := llm.NewClient(config)
	agent := agents.NewTrainingMonitorAgentWithRuntime(client, config.Model, config)

	ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
	defer cancel()

	retrievedRunMemory := s.retrieveTrainingMonitorMemory(ctx, plan, job, summary, projectObjectiveContext(goalText))
	trace, err := agent.EvaluateWithTrace(ctx, agents.TrainingMonitorInput{
		Plan:               plan,
		Job:                job,
		Summary:            summary,
		Evaluation:         evaluation,
		Metrics:            metrics,
		ObjectiveContext:   projectObjectiveContext(goalText),
		MemoryRecords:      priorMemory,
		RetrievedRunMemory: retrievedRunMemory,
		OptimizerFeedback:  s.optimizerFeedbackSummary(jobConfigString(job.Config, "automl_study_id"), trainingMonitorTargetMetricFromJob(job, summary)),
	})
	acceptedForMemory := err == nil
	invocation, invocationErr := s.recordTrainingMonitorInvocation(job, summary, config, trace, acceptedForMemory)
	if invocationErr != nil {
		log.Printf("training monitor invocation write failed for job %s: %v", job.ID, invocationErr)
	}
	if err != nil {
		log.Printf("training monitor failed for job %s: %v", job.ID, err)
		diagnostics.Event("error", "training_monitor_failed", map[string]any{
			"job_id":        job.ID,
			"project_id":    job.ProjectID,
			"plan_id":       summary.PlanID,
			"invocation_id": invocation.ID,
			"provider":      config.Provider,
			"model":         config.Model,
			"api_style":     config.APIStyle,
			"error":         err.Error(),
		})
		if _, eventErr := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventAgentFailed, fmt.Sprintf("Training Monitor agent failed for job %s.", job.ID), map[string]any{
			"job_id":        job.ID,
			"invocation_id": invocation.ID,
			"error":         err.Error(),
			"agent_name":    agents.TrainingMonitorAgentName,
		}); eventErr != nil {
			log.Printf("record training monitor failure event failed: %v", eventErr)
		}
		return
	}

	recommendation := trace.Recommendation
	payload, err := mapFromStruct(recommendation)
	if err != nil {
		log.Printf("training monitor payload conversion failed for job %s: %v", job.ID, err)
		return
	}

	record, err := s.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		InvocationID: invocation.ID,
		ProjectID:    job.ProjectID,
		DatasetID:    summary.DatasetID,
		PlanID:       summary.PlanID,
		JobID:        job.ID,
		AgentName:    agents.TrainingMonitorAgentName,
		Kind:         memory.KindTrainingEvaluation,
		Summary:      recommendation.Summary,
		Payload:      payload,
		Tags:         recommendation.Tags,
	})
	if err != nil {
		log.Printf("training monitor memory write failed for job %s: %v", job.ID, err)
		return
	}
	s.indexMemoryCard(context.Background(), memory.NewAgentMemoryCard(record))

	if _, err := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventAgentRecommendationRecorded, fmt.Sprintf("Training Monitor recorded an evaluation for job %s.", job.ID), map[string]any{
		"job_id":           job.ID,
		"invocation_id":    invocation.ID,
		"memory_record_id": record.ID,
		"agent_name":       record.AgentName,
		"kind":             record.Kind,
	}); err != nil {
		log.Printf("record training monitor event failed: %v", err)
	}
}

func trainingMonitorLLMEnabled() bool {
	return envFlag("MODEL_EXPRESS_TRAINING_MONITOR_LLM_ENABLED", false)
}

func experimentPlannerLLMConfig(config llm.Config) llm.Config {
	defaultRounds := plannerDefaultMaxToolRounds
	if config.MaxToolRounds > defaultRounds {
		defaultRounds = config.MaxToolRounds
	}
	config.MaxToolRounds = envInt("MODEL_EXPRESS_PLANNER_MAX_TOOL_ROUNDS", defaultRounds, 1, 32)
	return config
}

func (s *Server) recordTrainingMonitorInvocation(
	job jobs.ExperimentJob,
	summary runs.TrainingRunSummary,
	config llm.Config,
	trace agents.TrainingMonitorEvaluationTrace,
	acceptedForMemory bool,
) (memory.AgentInvocation, error) {
	validationStatus := trace.ValidationStatus
	if validationStatus == "" {
		validationStatus = memory.InvocationValidationFailed
	}
	inputContext := agentInvocationInputContext(trace.PromptContext, config, trace.ToolRounds, trace.Usage, trace.ToolCalls, trace.ToolResults, trace.RejectedToolCalls, trace.DryRunValidationResults)

	return s.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:         job.ProjectID,
		DatasetID:         summary.DatasetID,
		PlanID:            summary.PlanID,
		JobID:             job.ID,
		AgentName:         agents.TrainingMonitorAgentName,
		AgentVersion:      trace.AgentVersion,
		PromptVersion:     trace.PromptVersion,
		Provider:          config.Provider,
		Model:             config.Model,
		InputMessages:     llmMessagesForMemory(trace.Request.Messages),
		InputContext:      inputContext,
		RawOutput:         string(trace.RawOutput),
		ParsedOutput:      trace.ParsedOutput,
		ValidationStatus:  validationStatus,
		ValidationError:   trace.ValidationError,
		AcceptedForMemory: acceptedForMemory,
		HumanFeedback:     map[string]any{},
		DownstreamOutcome: map[string]any{},
	})
}

func llmMessagesForMemory(messages []llm.Message) []map[string]string {
	out := make([]map[string]string, 0, len(messages))
	for _, message := range messages {
		out = append(out, map[string]string{
			"role":    message.Role,
			"content": message.Content,
		})
	}
	return out
}

func agentInvocationInputContext(
	promptContext map[string]any,
	config llm.Config,
	toolRounds int,
	usage *llm.Usage,
	toolCalls []agents.AgentToolCallTrace,
	toolResults []agents.AgentToolResultTrace,
	rejectedToolCalls []agents.AgentToolResultTrace,
	dryRunValidationResults []map[string]any,
) map[string]any {
	out := map[string]any{}
	for key, value := range promptContext {
		out[key] = value
	}
	out["invocation_runtime"] = map[string]any{
		"api_style":                  config.APIStyle,
		"provider":                   config.Provider,
		"model":                      config.Model,
		"reasoning_effort":           config.ReasoningEffort,
		"plateau_reasoning_effort":   config.PlateauReasoningEffort,
		"stored_responses":           config.StoredResponses,
		"max_tool_rounds":            config.MaxToolRounds,
		"tool_rounds":                toolRounds,
		"tool_calls":                 toolCalls,
		"tool_results":               toolResults,
		"tool_names":                 agentToolNames(toolCalls),
		"rejected_tool_calls":        rejectedToolCalls,
		"dry_run_validation_results": dryRunValidationResults,
		"tool_calls_are_questions":   true,
		"mutation_authority":         false,
	}
	if usage != nil {
		out["invocation_runtime"].(map[string]any)["llm_usage"] = usage
	}
	return out
}

func agentToolNames(calls []agents.AgentToolCallTrace) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, call := range calls {
		name := strings.TrimSpace(call.Name)
		if name == "" || seen[name] {
			continue
		}
		seen[name] = true
		out = append(out, name)
	}
	return out
}

func (s *Server) runExperimentPlannerAfterTrainingJob(job jobs.ExperimentJob) (bool, error) {
	if !s.shouldRunLLMAgents() || !s.shouldAutoReviewExperimentJobs() {
		return false, nil
	}

	summary, err := s.store.GetTrainingRunSummary(job.ID)
	if err != nil {
		return false, nil
	}
	if summary.PlanID == "" {
		return false, nil
	}

	s.autoReviewMu.Lock()
	defer s.autoReviewMu.Unlock()

	input, ready, err := s.buildExperimentPlannerInput(job.ProjectID, summary.PlanID)
	if err != nil || !ready {
		return false, err
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(job.ProjectID)
	if err != nil {
		return false, err
	}
	if decision, ok := experimentPlannerDecisionForPlan(agentDecisions, input.SourcePlan.ID); ok {
		if err := s.persistProjectChampionFromDecision(job.ProjectID, decision); err != nil {
			log.Printf("persist planner champion failed for project %s decision %s: %v", job.ProjectID, decision.ID, err)
		}
		result := automaticExperimentReviewResult{Decision: &decision}
		if decision.DecisionType == decisions.TypeAddExperiments &&
			s.shouldAutoScheduleFollowUps() &&
			s.currentAutomationSettings().AgentMode == llm.AgentModeAutonomous {
			return true, s.schedulePlannerDecision(job.ProjectID, input.SourcePlan, decision, result)
		}
		return true, nil
	}
	automationSettings := s.currentAutomationSettings()
	if terminalPlannerGuardsEnabledForMode(automationSettings.AgentMode) {
		if stopReason, stopDetails, selected, err := s.projectChampionSelectedFollowUpStopReason(job.ProjectID); err != nil {
			return false, err
		} else if selected {
			message := fmt.Sprintf("Experiment Planner skipped for plan %s because the project already has a selected champion.", input.SourcePlan.ID)
			s.recordChampionSelectedFollowUpBlocked(job.ProjectID, input.SourcePlan.ID, "", "", message, stopReason, stopDetails)
			return true, nil
		}
	}

	s.recordExperimentPlannerStarted(input)

	config := experimentPlannerLLMConfig(llm.ConfigFromEnv(automationSettings.LLMEnabled, automationSettings.LLMProvider, automationSettings.LLMModel))
	client := llm.NewClient(config)
	agent := agents.NewExperimentPlannerAgentWithRuntime(client, config.Model, config, agents.PlannerInformationToolOptions{
		ValidateCandidateExperiments: plannerCandidateDryRunValidator(input),
	})

	ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
	defer cancel()

	plannerAttempt, err := s.runExperimentPlannerWithBackendValidationRetry(ctx, agent, input, config, automationSettings.AgentMode)
	invocation := plannerAttempt.Invocation
	if err != nil {
		diagnostics.Event("error", "experiment_planner_failed", map[string]any{
			"job_id":        job.ID,
			"project_id":    job.ProjectID,
			"plan_id":       summary.PlanID,
			"invocation_id": invocation.ID,
			"provider":      config.Provider,
			"model":         config.Model,
			"api_style":     config.APIStyle,
			"error":         err.Error(),
		})
		if _, eventErr := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventAgentFailed, fmt.Sprintf("Experiment Planner agent failed for plan %s.", summary.PlanID), map[string]any{
			"invocation_id": invocation.ID,
			"error":         err.Error(),
			"agent_name":    agents.ExperimentPlannerAgentName,
		}); eventErr != nil {
			log.Printf("record experiment planner failure event failed: %v", eventErr)
		}
		return false, err
	}

	input = plannerAttempt.Input
	recommendation := plannerAttempt.Recommendation
	payload := plannerAttempt.Payload

	memoryPayload, err := mapFromStruct(recommendation)
	if err != nil {
		return false, err
	}
	memoryPayload["current_champion"] = input.CurrentChampion
	memoryPayload["source_plan_baseline_champion"] = input.SourcePlanBaselineChampion
	memoryPayload["source_plan_run_deltas"] = input.SourcePlanDeltas
	memoryPayload["dataset_planning_insights"] = input.DatasetInsights
	memoryPayload["objective_context"] = input.ObjectiveContext
	memoryPayload["deterministic_diagnosis"] = input.DeterministicDiagnosis
	memoryPayload["model_catalog"] = input.ModelCatalog
	memoryPayload["plan_evaluations"] = input.PlanEvaluations
	memoryPayload["successful_strategy_memory"] = input.SuccessfulStrategyMemory
	memoryPayload["failed_strategy_memory"] = input.FailedStrategyMemory
	memoryPayload["rejected_strategy_memory"] = input.RejectedStrategyMemory
	memoryPayload["strategy_scorecards"] = input.StrategyScorecards
	memoryPayload["optimizer_feedback_summary"] = input.OptimizerFeedback
	memoryPayload["no_improvement_rounds"] = input.NoImprovementRounds
	memoryPayload["stop_signals"] = input.StopSignals
	record, err := s.store.CreateAgentMemoryRecord(memory.AgentMemoryRecord{
		InvocationID: invocation.ID,
		ProjectID:    job.ProjectID,
		DatasetID:    input.SourcePlan.DatasetID,
		PlanID:       input.SourcePlan.ID,
		AgentName:    agents.ExperimentPlannerAgentName,
		Kind:         memory.KindPlanningFeedback,
		Summary:      recommendation.Summary,
		Payload:      memoryPayload,
		Tags:         recommendation.Tags,
	})
	if err != nil {
		return false, err
	}
	payload["memory_record_id"] = record.ID
	s.indexMemoryCard(context.Background(), memory.NewAgentMemoryCard(record))

	decisionType := strings.ToUpper(strings.TrimSpace(recommendation.DecisionType))
	if decisionType == decisions.TypeWait {
		return true, nil
	}

	decision, err := s.store.CreateAgentDecision(
		job.ProjectID,
		input.SourcePlan.ID,
		decisionType,
		recommendation.Rationale,
		payload,
	)
	if err != nil {
		return false, err
	}
	if err := s.persistProjectChampionFromDecision(job.ProjectID, decision); err != nil {
		log.Printf("persist planner champion failed for project %s decision %s: %v", job.ProjectID, decision.ID, err)
	}

	if _, err := s.store.CreateExecutionEvent(job.ProjectID, input.SourcePlan.ID, execution.EventAgentRecommendationRecorded, fmt.Sprintf("Experiment Planner recorded a plan-level decision for plan %s.", input.SourcePlan.ID), map[string]any{
		"invocation_id":    invocation.ID,
		"memory_record_id": record.ID,
		"decision_id":      decision.ID,
		"decision_type":    decision.DecisionType,
		"agent_name":       agents.ExperimentPlannerAgentName,
	}); err != nil {
		log.Printf("record experiment planner event failed: %v", err)
	}

	result := automaticExperimentReviewResult{Decision: &decision}
	if decision.DecisionType != decisions.TypeAddExperiments ||
		!s.shouldAutoScheduleFollowUps() ||
		automationSettings.AgentMode != llm.AgentModeAutonomous {
		return true, nil
	}

	return true, s.schedulePlannerDecision(job.ProjectID, input.SourcePlan, decision, result)
}

func (s *Server) recordExperimentPlannerStarted(input agents.ExperimentPlannerInput) {
	if input.Project.ID == "" || input.SourcePlan.ID == "" {
		return
	}
	if _, err := s.store.CreateExecutionEvent(input.Project.ID, input.SourcePlan.ID, execution.EventAgentStarted, "Experiment Planner started; reading completed runs, memories, and evaluations.", map[string]any{
		"agent_name":          agents.ExperimentPlannerAgentName,
		"completed_run_count": len(input.PlanSummaries),
		"memory_count":        len(input.PriorMemory),
		"evaluation_count":    len(input.PlanEvaluations),
	}); err != nil {
		log.Printf("record experiment planner start event failed for plan %s: %v", input.SourcePlan.ID, err)
	}
}

func (s *Server) buildExperimentPlannerInput(projectID string, planID string) (agents.ExperimentPlannerInput, bool, error) {
	project, err := s.store.GetProject(projectID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	plan, err := s.store.GetExperimentPlan(planID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	if plan.ProjectID != projectID {
		return agents.ExperimentPlannerInput{}, false, fmt.Errorf("%w: plan does not belong to project", store.ErrInvalidRequest)
	}

	dataset, err := s.store.GetDataset(plan.DatasetID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	projectPlans, err := s.store.ListProjectExperimentPlans(projectID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	projectJobs, err := s.store.ListProjectJobs(projectID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	summaries, err := s.store.ListProjectTrainingRunSummaries(projectID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(projectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return agents.ExperimentPlannerInput{}, false, err
	}

	planJobs := jobsForPlan(projectJobs, plan.ID)
	planSummaries := summariesForPlanID(summaries, plan.ID)
	planEvaluations := evaluationsForPlanID(evaluations, plan.ID)
	if !planTrainingRunsComplete(plan, planSummaries) {
		return agents.ExperimentPlannerInput{}, false, nil
	}
	automationSettings := s.currentAutomationSettings()
	minimumMeaningfulImprovement := plannerMinimumMeaningfulImprovementFromEnv(automationSettings.AgentMode)
	objectiveContext := projectObjectiveContext(project.Goal)
	currentChampion, baselineChampion, sourcePlanDeltas, noImprovementRounds, stopSignals := experimentPlannerPerformanceContext(
		plan.TargetMetric,
		projectPlans,
		summaries,
		evaluations,
		objectiveContext,
		plan.ID,
	)

	planMetrics := map[string][]jobs.EpochMetric{}
	for _, planJob := range planJobs {
		metrics, err := s.store.ListJobMetrics(planJob.ID)
		if err != nil && !errors.Is(err, store.ErrNotFound) {
			return agents.ExperimentPlannerInput{}, false, err
		}
		planMetrics[planJob.ID] = metrics
	}
	visualContext, err := s.plannerVisualEvidenceContext(dataset)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	metadataSummary, err := s.activeAgentSafeDatasetMetadataSummary(dataset)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}

	partialInput := agents.ExperimentPlannerInput{
		Project:                      project,
		Dataset:                      dataset,
		SourcePlan:                   plan,
		PlanJobs:                     planJobs,
		PlanSummaries:                planSummaries,
		PlanEvaluations:              planEvaluations,
		PlanMetrics:                  planMetrics,
		DatasetInsights:              datasetPlanningInsights(dataset, metadataSummary),
		VisualExemplarContext:        visualContext,
		ObjectiveContext:             objectiveContext,
		CurrentChampion:              currentChampion,
		SourcePlanBaselineChampion:   baselineChampion,
		SourcePlanDeltas:             sourcePlanDeltas,
		NoImprovementRounds:          noImprovementRounds,
		MinimumMeaningfulImprovement: minimumMeaningfulImprovement,
	}
	deterministicDiagnosis := agents.ComputePlannerDiagnosis(partialInput)

	priorMemory, err := s.store.ListProjectAgentMemoryRecords(projectID, memory.AgentMemoryFilter{
		DatasetID: plan.DatasetID,
		Limit:     25,
	})
	if err != nil {
		priorMemory = []memory.AgentMemoryRecord{}
	}
	successfulStrategyMemory, failedStrategyMemory := plannerStrategyMemory(priorMemory)
	rejectedStrategyMemory := plannerRejectedOptions(priorMemory, deterministicDiagnosis)
	scorecards, err := s.store.ListProjectStrategyScorecards(projectID, 12)
	if err != nil {
		scorecards = []strategies.StrategyScorecard{}
	}
	strategyScorecards := plannerStrategyScorecards(scorecards, plan.DatasetID)

	input := agents.ExperimentPlannerInput{
		Project:                      project,
		Dataset:                      dataset,
		SourcePlan:                   plan,
		PlanJobs:                     planJobs,
		PlanSummaries:                planSummaries,
		PlanEvaluations:              planEvaluations,
		PlanMetrics:                  planMetrics,
		DatasetInsights:              partialInput.DatasetInsights,
		VisualExemplarContext:        visualContext,
		ObjectiveContext:             objectiveContext,
		DeterministicDiagnosis:       deterministicDiagnosis,
		ModelCatalog:                 supportedModelCatalogForDataset(dataset, metadataSummary),
		CurrentChampion:              currentChampion,
		SourcePlanBaselineChampion:   baselineChampion,
		SourcePlanDeltas:             sourcePlanDeltas,
		NoImprovementRounds:          noImprovementRounds,
		StopSignals:                  stopSignals,
		MinimumMeaningfulImprovement: minimumMeaningfulImprovement,
		SuccessfulStrategyMemory:     successfulStrategyMemory,
		FailedStrategyMemory:         failedStrategyMemory,
		RejectedStrategyMemory:       rejectedStrategyMemory,
		StrategyScorecards:           strategyScorecards,
		OptimizerFeedback:            s.optimizerFeedbackSummariesForProject(projectID, plan.TargetMetric),
		PriorPlans:                   projectPlans,
		PriorJobs:                    projectJobs,
		PriorSummaries:               summaries,
		PriorEvaluations:             evaluations,
		PriorMemory:                  priorMemory,
		ExistingExperimentSignatures: experimentSignaturesForPlans(projectPlans),
		AgentMode:                    automationSettings.AgentMode,
		MaxExperiments:               maxLLMPlannerExperiments,
		MaxFollowUpRounds:            s.maxAutoFollowUpRounds(),
		FollowUpRound:                followUpRoundCount(projectPlans),
	}
	input.ProjectTrajectory = agents.ComputeProjectTrajectoryDiagnosis(input)
	input.RetrievedMemory = s.retrievePlannerMemory(context.Background(), input)
	return input, true, nil
}

func (s *Server) recordExperimentPlannerInvocation(
	input agents.ExperimentPlannerInput,
	config llm.Config,
	trace agents.ExperimentPlanningTrace,
	acceptedForMemory bool,
) (memory.AgentInvocation, error) {
	validationStatus := trace.ValidationStatus
	if validationStatus == "" {
		validationStatus = memory.InvocationValidationFailed
	}
	inputContext := agentInvocationInputContext(trace.PromptContext, config, trace.ToolRounds, trace.Usage, trace.ToolCalls, trace.ToolResults, trace.RejectedToolCalls, trace.DryRunValidationResults)

	return s.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:         input.Project.ID,
		DatasetID:         input.SourcePlan.DatasetID,
		PlanID:            input.SourcePlan.ID,
		AgentName:         agents.ExperimentPlannerAgentName,
		AgentVersion:      trace.AgentVersion,
		PromptVersion:     trace.PromptVersion,
		Provider:          config.Provider,
		Model:             config.Model,
		InputMessages:     llmMessagesForMemory(trace.Request.Messages),
		InputContext:      inputContext,
		RawOutput:         string(trace.RawOutput),
		ParsedOutput:      trace.ParsedOutput,
		ValidationStatus:  validationStatus,
		ValidationError:   trace.ValidationError,
		AcceptedForMemory: acceptedForMemory,
		HumanFeedback:     map[string]any{},
		DownstreamOutcome: map[string]any{},
	})
}

type experimentPlannerAttemptResult struct {
	Input          agents.ExperimentPlannerInput
	Trace          agents.ExperimentPlanningTrace
	Invocation     memory.AgentInvocation
	Recommendation agents.ExperimentPlanningRecommendation
	Payload        map[string]any
}

func (s *Server) runExperimentPlannerWithBackendValidationRetry(
	ctx context.Context,
	agent agents.ExperimentPlannerAgent,
	input agents.ExperimentPlannerInput,
	config llm.Config,
	agentMode string,
) (experimentPlannerAttemptResult, error) {
	attemptInput := input
	var result experimentPlannerAttemptResult
	var lastErr error
	for attempt := 0; attempt <= plannerBackendValidationRetryLimit; attempt++ {
		trace, err := agent.PlanWithTrace(ctx, attemptInput)
		acceptedForMemory := err == nil
		invocation, invocationErr := s.recordExperimentPlannerInvocation(attemptInput, config, trace, acceptedForMemory)
		if invocationErr != nil {
			log.Printf("experiment planner invocation write failed for plan %s: %v", input.SourcePlan.ID, invocationErr)
		}
		result = experimentPlannerAttemptResult{
			Input:      attemptInput,
			Trace:      trace,
			Invocation: invocation,
		}
		if err != nil {
			lastErr = err
			willRetry := attempt < plannerBackendValidationRetryLimit && shouldRetryExperimentPlannerTraceValidation(trace, err)
			s.recordPlannerValidationRejection(invocation, err, attempt, willRetry)
			if willRetry {
				attemptInput.ValidationFeedback = append(attemptInput.ValidationFeedback, plannerValidationFeedback(trace.Recommendation, err, attempt+1))
				continue
			}
			result.Recommendation = trace.Recommendation
			return result, err
		}

		recommendation := applyExperimentPlannerStopCriteria(trace.Recommendation, attemptInput)
		if strings.EqualFold(recommendation.DecisionType, decisions.TypeAddExperiments) {
			experiments, automlWarnings, prepareErr := s.prepareAutoMLExperimentsForProject(input.Project.ID, recommendation.ProposedExperiments)
			if prepareErr != nil {
				lastErr = prepareErr
				s.recordPlannerValidationRejection(invocation, prepareErr, attempt, attempt < plannerBackendValidationRetryLimit && shouldRetryExperimentPlannerValidation(recommendation))
				if attempt >= plannerBackendValidationRetryLimit || !shouldRetryExperimentPlannerValidation(recommendation) {
					result.Recommendation = recommendation
					return result, prepareErr
				}
				attemptInput.ValidationFeedback = append(attemptInput.ValidationFeedback, plannerValidationFeedback(recommendation, prepareErr, attempt+1))
				continue
			}
			recommendation.ProposedExperiments = experiments
			recommendation.NoveltyNotes = append(recommendation.NoveltyNotes, automlWarnings...)
		}
		payload, err := experimentPlannerDecisionPayload(recommendation, invocation, agentMode, attemptInput)
		if err == nil {
			if attempt > 0 {
				payload["validation_retry_count"] = attempt
				payload["validation_feedback_applied"] = attemptInput.ValidationFeedback
			}
			result.Recommendation = recommendation
			result.Payload = payload
			return result, nil
		}

		lastErr = err
		s.recordPlannerValidationRejection(invocation, err, attempt, attempt < plannerBackendValidationRetryLimit && shouldRetryExperimentPlannerValidation(recommendation))
		if attempt >= plannerBackendValidationRetryLimit || !shouldRetryExperimentPlannerValidation(recommendation) {
			result.Recommendation = recommendation
			return result, err
		}
		attemptInput.ValidationFeedback = append(attemptInput.ValidationFeedback, plannerValidationFeedback(recommendation, err, attempt+1))
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("%w: experiment planner validation retry failed", store.ErrInvalidRequest)
	}
	return result, lastErr
}

func (s *Server) recordPlannerValidationRejection(invocation memory.AgentInvocation, validationErr error, attempt int, willRetry bool) {
	if invocation.ID == "" || validationErr == nil {
		return
	}
	outcome := map[string]any{
		"backend_validation_status": "rejected",
		"backend_validation_error":  validationErr.Error(),
		"retry_attempt":             attempt,
		"will_retry":                willRetry,
	}
	if runtime, ok := invocation.InputContext["invocation_runtime"].(map[string]any); ok {
		if dryRuns, ok := runtime["dry_run_validation_results"]; ok {
			outcome["dry_run_validation_results"] = dryRuns
		}
		if rejected, ok := runtime["rejected_tool_calls"]; ok {
			outcome["rejected_tool_calls"] = rejected
		}
	}
	if _, err := s.store.UpdateAgentInvocationDownstreamOutcome(invocation.ID, outcome); err != nil {
		log.Printf("update planner invocation validation outcome failed for invocation %s: %v", invocation.ID, err)
	}
}

func shouldRetryExperimentPlannerValidation(recommendation agents.ExperimentPlanningRecommendation) bool {
	return strings.EqualFold(strings.TrimSpace(recommendation.DecisionType), decisions.TypeAddExperiments)
}

func shouldRetryExperimentPlannerTraceValidation(trace agents.ExperimentPlanningTrace, validationErr error) bool {
	if validationErr == nil {
		return false
	}
	if shouldRetryExperimentPlannerValidation(trace.Recommendation) {
		return true
	}
	return len(trace.RawOutput) > 0 && trace.ValidationStatus == memory.InvocationValidationInvalid
}

func plannerCandidateDryRunValidator(input agents.ExperimentPlannerInput) agents.PlannerCandidateDryRunValidator {
	return func(recommendation agents.ExperimentPlanningRecommendation) agents.PlannerCandidateDryRunResult {
		result := agents.PlannerCandidateDryRunResult{
			Valid:                   true,
			ValidationStatus:        "valid",
			ProposedExperimentCount: len(recommendation.ProposedExperiments),
			CandidateCount:          len(recommendation.CandidateHypotheses),
			SelectedCandidateCount:  len(recommendation.ProposedExperiments),
			WouldWriteRows:          false,
			WouldScheduleJobs:       false,
			Details: map[string]any{
				"dry_run_only": true,
			},
		}
		if strings.EqualFold(strings.TrimSpace(recommendation.DecisionType), decisions.TypeAddExperiments) {
			relaxedValidationWarnings := []string{}
			experiments, err := plannerExperimentsWithProposalMechanisms(recommendation)
			if err != nil {
				if plannerStrictValidationEnabled() {
					return invalidPlannerDryRunResult(result, err)
				}
				experiments, relaxedValidationWarnings = plannerExperimentsWithProposalMechanismsRelaxed(recommendation)
				relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
			}
			for index := range experiments {
				if experiments[index].AutoML == nil || !experiments[index].AutoML.Enabled {
					continue
				}
				prepared, err := prepareAutoMLExperiment(experiments[index], index, automl.SamplerSeededRandom)
				if err != nil {
					return invalidPlannerDryRunResult(result, err)
				}
				experiments[index] = prepared
			}
			for index, experiment := range experiments {
				if err := validatePlannedExperiment(experiment, index); err != nil {
					return invalidPlannerDryRunResult(result, err)
				}
				if err := validateExperimentDatasetCompatibility(experiment, input.Dataset, index); err != nil {
					return invalidPlannerDryRunResult(result, err)
				}
			}
			if err := validateLLMPlannerMechanismContract(experiments, recommendation.EvidenceUsed); err != nil {
				if plannerStrictValidationEnabled() {
					return invalidPlannerDryRunResult(result, err)
				}
				relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
			}
			if err := validateNovelProposedExperiments(experiments, input.PriorPlans); err != nil {
				if plannerStrictValidationEnabled() {
					return invalidPlannerDryRunResult(result, err)
				}
				relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
			}
			if err := validateMechanismDatasetEvidence(profileWithAgentSafeMetadataSummary(input.Dataset.Profile, input.DatasetInsights.AgentSafeMetadataSummary), experiments, recommendation.EvidenceUsed); err != nil {
				if plannerStrictValidationEnabled() {
					return invalidPlannerDryRunResult(result, err)
				}
				relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
			}
			result.Details["validated_experiment_count"] = len(experiments)
			if len(relaxedValidationWarnings) > 0 {
				result.Details["planner_validation_mode"] = "relaxed"
				result.Details["planner_validation_warnings"] = uniqueStrings(relaxedValidationWarnings)
			}
		}
		return result
	}
}

func invalidPlannerDryRunResult(result agents.PlannerCandidateDryRunResult, err error) agents.PlannerCandidateDryRunResult {
	result.Valid = false
	result.ValidationStatus = "invalid"
	if err != nil {
		result.ValidationError = err.Error()
	}
	result.WouldWriteRows = false
	result.WouldScheduleJobs = false
	return result
}

func plannerValidationFeedback(recommendation agents.ExperimentPlanningRecommendation, validationErr error, attempt int) agents.PlannerValidationFeedback {
	rejectedExperiments := make([]string, 0, len(recommendation.ProposedExperiments))
	rejectedModels := []string{}
	seenModels := map[string]bool{}
	for _, experiment := range recommendation.ProposedExperiments {
		rejectedExperiments = append(rejectedExperiments, experimentFeedbackSummary(experiment))
		model := strings.ToLower(strings.TrimSpace(experiment.Model))
		if model != "" && !seenModels[model] {
			seenModels[model] = true
			rejectedModels = append(rejectedModels, experiment.Model)
		}
	}
	feedback := agents.PlannerValidationFeedback{
		Attempt:             attempt,
		ValidationError:     validationErr.Error(),
		RejectedDecision:    recommendation.DecisionType,
		RejectedModels:      rejectedModels,
		RejectedExperiments: rejectedExperiments,
		Instructions: []string{
			"Return corrected JSON only.",
			"Do not repeat rejected experiment mechanisms.",
			"Change a meaningful mechanism such as model family, preprocessing, augmentation policy, sampling/class balancing, scheduler, optimizer, regularization, or resolution strategy.",
			"Only propose experiments that backend validation can schedule.",
		},
	}
	if validationErr != nil && strings.Contains(strings.ToLower(validationErr.Error()), "champion_challenge") {
		feedback.Instructions = append(
			feedback.Instructions,
			"For champion_challenge, every selected experiment_config reason or strategy must explicitly explain how that experiment can beat, improve on, or trade off against the current champion.",
		)
	}
	return feedback
}

func experimentFeedbackSummary(experiment plans.PlannedExperiment) string {
	return strings.TrimSpace(fmt.Sprintf(
		"template=%s model=%s epochs=%d batch_size=%d learning_rate=%g optimizer=%s scheduler=%s weight_decay=%g dropout=%g label_smoothing=%g gradient_clip_norm=%g image_size=%d resolution_strategy=%s augmentation_policy=%s class_balancing=%s sampling_strategy=%s reason=%s",
		experiment.Template,
		experiment.Model,
		experiment.Epochs,
		experiment.BatchSize,
		experiment.LearningRate,
		experiment.Optimizer,
		experiment.Scheduler,
		experiment.WeightDecay,
		experiment.Dropout,
		experiment.LabelSmoothing,
		experiment.GradientClipNorm,
		experiment.ImageSize,
		experiment.ResolutionStrategy,
		experiment.AugmentationPolicy,
		experiment.ClassBalancing,
		experiment.SamplingStrategy,
		experiment.Reason,
	))
}

func applyExperimentPlannerStopCriteria(
	recommendation agents.ExperimentPlanningRecommendation,
	input agents.ExperimentPlannerInput,
) agents.ExperimentPlanningRecommendation {
	decisionType := strings.ToUpper(strings.TrimSpace(recommendation.DecisionType))
	recommendation.DecisionType = decisionType
	if decisionType == decisions.TypeWait && plannerWaitShouldSelectChampion(recommendation) && input.CurrentChampion != nil && strings.TrimSpace(input.CurrentChampion.JobID) != "" {
		return selectChampionForPlannerWaitDecision(recommendation, input)
	}
	if decisionType != decisions.TypeAddExperiments {
		return recommendation
	}
	stopReason, guardTag, ok := experimentPlannerBackendStopReason(input)
	if !ok {
		return recommendation
	}
	if !terminalPlannerGuardsEnabledForMode(input.AgentMode) {
		recommendation.NoveltyNotes = append(recommendation.NoveltyNotes, "Backend stop advisory only; continuing is allowed because terminal planner guards are disabled: "+stopReason)
		recommendation.Tags = append(recommendation.Tags, "backend_stop_advisory", guardTag)
		return recommendation
	}

	recommendation.DecisionType = decisions.TypeSelectChampion
	recommendation.ChampionJobID = input.CurrentChampion.JobID
	recommendation.ProposedExperiments = nil
	recommendation.StopReason = stopReason
	recommendation.Summary = fmt.Sprintf("Select champion %s; backend stop criteria found no meaningful follow-up upside.", input.CurrentChampion.JobID)
	recommendation.Rationale = strings.TrimSpace(recommendation.Rationale + " Backend stop criteria applied: " + stopReason)
	recommendation.NoveltyNotes = append(recommendation.NoveltyNotes, "Backend guard converted ADD_EXPERIMENTS to SELECT_CHAMPION because additional training had insufficient meaningful upside.")
	recommendation.Tags = append(recommendation.Tags, "select_champion", guardTag)
	return recommendation
}

func plannerWaitShouldSelectChampion(recommendation agents.ExperimentPlanningRecommendation) bool {
	for _, tag := range recommendation.Tags {
		if strings.EqualFold(strings.TrimSpace(tag), "audit_only") {
			return false
		}
	}
	return len(recommendation.EvidenceUsed) > 0 ||
		len(recommendation.DeterministicDiagnosisUsed) > 0 ||
		len(recommendation.ChangedVariables) > 0 ||
		strings.TrimSpace(recommendation.StopCondition) != "" ||
		strings.TrimSpace(recommendation.StopReason) != "" ||
		strings.TrimSpace(recommendation.Hypothesis) != "" ||
		strings.TrimSpace(recommendation.DatasetPreprocessingRationale) != "" ||
		strings.TrimSpace(recommendation.SuccessCriteria) != "" ||
		strings.TrimSpace(recommendation.DeploymentTradeoff) != ""
}

func selectChampionForPlannerWaitDecision(
	recommendation agents.ExperimentPlanningRecommendation,
	input agents.ExperimentPlannerInput,
) agents.ExperimentPlanningRecommendation {
	championJobID := strings.TrimSpace(input.CurrentChampion.JobID)
	stopReason := "Planner returned WAIT after completed training; backend selected the current champion so the project has a deployable model before pausing."
	recommendation.DecisionType = decisions.TypeSelectChampion
	recommendation.ChampionJobID = championJobID
	recommendation.ProposedExperiments = nil
	recommendation.CandidateHypotheses = nil
	recommendation.CandidateRankings = nil
	recommendation.ProposalMechanisms = nil
	if strings.TrimSpace(recommendation.Summary) == "" || strings.EqualFold(strings.TrimSpace(recommendation.Summary), "wait") {
		recommendation.Summary = fmt.Sprintf("Select champion %s; planner pause converted to champion selection.", championJobID)
	}
	if strings.TrimSpace(recommendation.StopReason) == "" {
		recommendation.StopReason = stopReason
	} else if !strings.Contains(recommendation.StopReason, stopReason) {
		recommendation.StopReason = strings.TrimSpace(recommendation.StopReason + " " + stopReason)
	}
	recommendation.Rationale = strings.TrimSpace(recommendation.Rationale + " Backend guard applied: " + stopReason)
	recommendation.NoveltyNotes = append(recommendation.NoveltyNotes, "Backend guard converted WAIT to SELECT_CHAMPION because completed autonomous training must leave a persisted champion.")
	recommendation.Tags = uniqueStrings(append(recommendation.Tags, "select_champion", "wait_converted_to_champion"))
	return recommendation
}

func experimentPlannerBackendStopReason(input agents.ExperimentPlannerInput) (string, string, bool) {
	if reason, ok := nearMetricCeilingChampionStopReason(input); ok {
		return reason, "near_metric_ceiling_guard", true
	}
	if input.CurrentChampion == nil || input.NoImprovementRounds < plannerNoImprovementRoundsToSelect {
		return "", "", false
	}
	minimumMeaningfulImprovement := plannerMeaningfulImprovementThreshold(input.MinimumMeaningfulImprovement)
	return fmt.Sprintf(
		"Current champion %s remains unbeaten after %d consecutive follow-up plan(s) with less than %.3f target-metric improvement.",
		input.CurrentChampion.JobID,
		input.NoImprovementRounds,
		minimumMeaningfulImprovement,
	), "no_improvement_guard", true
}

func nearMetricCeilingChampionStopReason(input agents.ExperimentPlannerInput) (string, bool) {
	if input.CurrentChampion == nil {
		return "", false
	}
	ceiling, ok := boundedHigherIsBetterMetricCeiling(input.CurrentChampion.TargetMetric)
	if !ok {
		return "", false
	}
	minimumMeaningfulImprovement := plannerMeaningfulImprovementThreshold(input.MinimumMeaningfulImprovement)
	headroom := ceiling - input.CurrentChampion.Score
	if headroom < 0 {
		headroom = 0
	}
	if headroom > minimumMeaningfulImprovement {
		return "", false
	}
	return fmt.Sprintf(
		"Current champion %s already has %s %.3f, leaving %.3f possible headroom before the %.1f metric ceiling, which is below the minimum meaningful improvement %.3f.",
		input.CurrentChampion.JobID,
		input.CurrentChampion.TargetMetric,
		input.CurrentChampion.Score,
		headroom,
		ceiling,
		minimumMeaningfulImprovement,
	), true
}

func plannerMeaningfulImprovementThreshold(value float64) float64 {
	if value > 0 {
		return value
	}
	return plannerMinimumMeaningfulImprovement
}

func boundedHigherIsBetterMetricCeiling(metric string) (float64, bool) {
	switch normalizedPlannerTargetMetric(metric) {
	case "accuracy", "macro_f1", "deployment_readiness":
		return 1.0, true
	default:
		return 0, false
	}
}

func experimentPlannerDecisionPayload(
	recommendation agents.ExperimentPlanningRecommendation,
	invocation memory.AgentInvocation,
	agentMode string,
	input agents.ExperimentPlannerInput,
) (map[string]any, error) {
	payload := map[string]any{
		"decision_source":                 llmExperimentPlannerDecisionSource,
		"agent_name":                      agents.ExperimentPlannerAgentName,
		"invocation_id":                   invocation.ID,
		"confidence":                      recommendation.Confidence,
		"auto_executable":                 agentMode == llm.AgentModeAutonomous,
		"planning_mode":                   recommendation.PlanningMode,
		"hypothesis":                      recommendation.Hypothesis,
		"dataset_preprocessing_rationale": recommendation.DatasetPreprocessingRationale,
		"changed_variables":               recommendation.ChangedVariables,
		"success_criteria":                recommendation.SuccessCriteria,
		"deployment_tradeoff":             recommendation.DeploymentTradeoff,
		"candidate_hypotheses":            recommendation.CandidateHypotheses,
		"candidate_rankings":              recommendation.CandidateRankings,
		"proposal_mechanisms":             recommendation.ProposalMechanisms,
		"risks":                           recommendation.Risks,
		"expected_tradeoffs":              recommendation.ExpectedTradeoffs,
		"novelty_notes":                   recommendation.NoveltyNotes,
		"champion_job_id":                 recommendation.ChampionJobID,
		"why_can_beat_champion":           recommendation.WhyCanBeatChampion,
		"expected_delta_vs_champion":      recommendation.ExpectedDeltaVsChampion,
		"stop_reason":                     recommendation.StopReason,
		"current_champion":                input.CurrentChampion,
		"source_plan_baseline_champion":   input.SourcePlanBaselineChampion,
		"source_plan_run_deltas":          input.SourcePlanDeltas,
		"dataset_planning_insights":       input.DatasetInsights,
		"objective_context":               input.ObjectiveContext,
		"deterministic_diagnosis":         input.DeterministicDiagnosis,
		"deterministic_diagnosis_used":    recommendation.DeterministicDiagnosisUsed,
		"project_trajectory_card":         input.ProjectTrajectory,
		"evidence_used":                   recommendation.EvidenceUsed,
		"expected_failure_modes":          recommendation.ExpectedFailureModes,
		"stop_condition":                  recommendation.StopCondition,
		"rejected_options":                recommendation.RejectedOptions,
		"model_catalog":                   input.ModelCatalog,
		"plan_evaluations":                input.PlanEvaluations,
		"successful_strategy_memory":      input.SuccessfulStrategyMemory,
		"failed_strategy_memory":          input.FailedStrategyMemory,
		"rejected_strategy_memory":        input.RejectedStrategyMemory,
		"strategy_scorecards":             input.StrategyScorecards,
		"optimizer_feedback_summary":      input.OptimizerFeedback,
		"no_improvement_rounds":           input.NoImprovementRounds,
		"minimum_meaningful_improvement":  input.MinimumMeaningfulImprovement,
		"stop_signals":                    input.StopSignals,
	}

	if strings.EqualFold(recommendation.DecisionType, decisions.TypeAddExperiments) {
		relaxedValidationWarnings := []string{}
		var experiments []plans.PlannedExperiment
		if plannerStrictValidationEnabled() {
			var err error
			experiments, err = plannerExperimentsWithProposalMechanisms(recommendation)
			if err != nil {
				return nil, err
			}
			if err := validateLLMPlannerMechanismContract(experiments, recommendation.EvidenceUsed); err != nil {
				return nil, err
			}
			if err := validateNovelProposedExperiments(experiments, input.PriorPlans); err != nil {
				return nil, err
			}
		} else {
			experiments, relaxedValidationWarnings = plannerExperimentsWithProposalMechanismsRelaxed(recommendation)
			for index, experiment := range experiments {
				if err := validatePlannedExperiment(experiment, index); err != nil {
					return nil, err
				}
			}
			if err := validateLLMPlannerMechanismContract(experiments, recommendation.EvidenceUsed); err != nil {
				relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
			}
			if err := validateNovelProposedExperiments(experiments, input.PriorPlans); err != nil {
				relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
			}
		}
		for index, experiment := range experiments {
			if err := validateExperimentDatasetCompatibility(experiment, input.Dataset, index); err != nil {
				return nil, err
			}
		}
		payload["proposed_experiments"] = experiments
		if len(relaxedValidationWarnings) > 0 {
			payload["planner_validation_mode"] = "relaxed"
			payload["planner_validation_warnings"] = uniqueStrings(relaxedValidationWarnings)
		}
	}

	return payload, nil
}

func planTrainingRunsComplete(plan plans.ExperimentPlan, summaries []runs.TrainingRunSummary) bool {
	if plan.ID == "" || len(plan.Experiments) == 0 {
		return false
	}

	if len(summaries) < len(plan.Experiments) {
		return false
	}

	for _, summary := range summaries {
		if !isTerminalTrainingSummary(summary) {
			return false
		}
	}
	return true
}

func isTerminalTrainingSummary(summary runs.TrainingRunSummary) bool {
	switch strings.ToUpper(strings.TrimSpace(summary.Status)) {
	case jobs.StatusSucceeded, jobs.StatusFailed:
		return true
	default:
		return false
	}
}

func experimentPlannerDecisionForPlan(agentDecisions []decisions.AgentDecision, planID string) (decisions.AgentDecision, bool) {
	for _, decision := range agentDecisions {
		if decision.PlanID != planID {
			continue
		}
		if decision.Payload["decision_source"] != llmExperimentPlannerDecisionSource {
			continue
		}
		return decision, true
	}
	return decisions.AgentDecision{}, false
}
