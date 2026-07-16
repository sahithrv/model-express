package api

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/diagnostics"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plannervalidation"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
	"model-express/services/orchestrator/internal/strategies"
)

const (
	plannerRetryPolicyVersion               = "planner_backend_validation_retry_v1"
	plannerRetryReasonTraceValidation       = "trace_validation_rejected"
	plannerRetryReasonAutoMLPreparation     = "automl_preparation_rejected"
	plannerRetryReasonExecutionCapabilities = "execution_capability_rejected"
	plannerRetryReasonDecisionPayload       = "decision_payload_rejected"
	plannerDecisionPolicyVersion            = "planner_decision_postprocessor_v1"
)

func (s *Server) recordExperimentPlannerOutcomeAfterTrainingJob(job jobs.ExperimentJob) error {
	plan, ok, err := s.trainingJobExperimentPlan(job)
	if err != nil || !ok {
		return err
	}
	return s.recordExperimentPlannerOutcomeForPlan(plan)
}

func (s *Server) recordExperimentPlannerOutcomeForPlan(plan plans.ExperimentPlan) error {
	if plan.SourceDecisionID == "" {
		return nil
	}

	s.autoReviewMu.Lock()
	defer s.autoReviewMu.Unlock()
	return s.recordExperimentPlannerOutcomeForPlanLocked(plan)
}

func (s *Server) recordExperimentPlannerOutcomeForPlanLocked(plan plans.ExperimentPlan) error {
	summaries, err := s.store.ListProjectTrainingRunSummaries(plan.ProjectID)
	if err != nil {
		return err
	}
	planSummaries := summariesForPlanID(summaries, plan.ID)
	projectJobs, err := s.store.ListProjectJobs(plan.ProjectID)
	if err != nil {
		return err
	}
	candidateRows, candidateErr := s.store.ListDecisionCandidateProvenance(plan.SourceDecisionID)
	if candidateErr != nil && !errors.Is(candidateErr, store.ErrNotFound) {
		return candidateErr
	}
	if len(candidateRows) > 0 {
		if !candidatePlanExperimentsTerminal(plan, candidateJobsForPlan(projectJobs, plan.ID), candidateRows) {
			return nil
		}
	} else if !planTrainingRunsComplete(plan, planSummaries) {
		return nil
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(plan.ProjectID)
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

	projectPlans, err := s.store.ListProjectExperimentPlans(plan.ProjectID)
	if err != nil {
		return err
	}
	executionEvidenceByJob, err := s.executionEvidenceForJobs(projectJobs)
	if err != nil {
		return err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(plan.ProjectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return err
	}
	goalText := ""
	if project, err := s.store.GetProject(plan.ProjectID); err == nil {
		goalText = project.Goal
	} else if !errors.Is(err, store.ErrNotFound) {
		return err
	}
	outcome, err := experimentPlanningOutcomeForPlanWithTerminalCount(sourceDecision, plan, projectPlans, summaries, evaluations, projectObjectiveContext(goalText), executionEvidenceByJob, len(plan.Experiments))
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
		ProjectID:    plan.ProjectID,
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
	if outcome.EvidenceEligibleRunCount > 0 {
		s.indexMemoryCard(context.Background(), memory.NewAgentMemoryCard(record))
	}

	if _, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventAgentOutcomeRecorded, fmt.Sprintf("Experiment Planner outcome recorded for follow-up plan %s.", plan.ID), map[string]any{
		"invocation_id":      updatedInvocation.ID,
		"memory_record_id":   record.ID,
		"source_decision_id": sourceDecision.ID,
		"outcome_status":     outcome.OutcomeStatus,
	}); err != nil {
		log.Printf("record experiment planner outcome event failed: %v", err)
	}
	primaryEvidence := primaryOutcomeExecutionEvidence(outcome)
	scorecardOutcome := outcome.OutcomeStatus
	if scorecardOutcome == agents.ExperimentPlanningOutcomeExecutionIneligible {
		scorecardOutcome = strategies.OutcomeInvalidated
	}
	if updatedScorecard, err := s.store.UpdateStrategyScorecardOutcomeByFollowUpPlan(plan.ID, strategies.StrategyScorecardOutcomeUpdate{
		ActualDelta:               outcome.ActualDeltaVsChampion,
		ConfidenceAfter:           plannerOutcomeConfidence(outcome),
		CostUSD:                   outcome.TotalCostUSD,
		RuntimeSeconds:            outcome.TotalRuntimeSeconds,
		Outcome:                   scorecardOutcome,
		Lesson:                    outcome.Lesson,
		Tags:                      tags,
		FidelityVerdicts:          outcomeExecutionFidelityVerdicts(outcome),
		EvidenceEligible:          outcome.EvidenceEligibleRunCount > 0,
		RequestedMechanism:        primaryEvidence.RequestedMechanism,
		RealizedMechanismIdentity: primaryEvidence.RealizedMechanismIdentity,
		AcceptedSpecHash:          primaryEvidence.AcceptedSpecHash,
		RealizedEffectiveHash:     primaryEvidence.RealizedEffectiveHash,
		AdjustmentReasonCodes:     primaryEvidence.AdjustmentReasonCodes,
	}); err == nil {
		if updatedScorecard.EvidenceEligible {
			s.indexMemoryCard(context.Background(), memory.NewStrategyScorecardMemoryCard(updatedScorecard))
		}
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
		"api_style":                  llm.EffectiveAPIStyle(config.Provider, config.APIStyle),
		"configured_api_style":       config.APIStyle,
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
		if err := s.ensurePlannerCandidateProvenance(decision); err != nil {
			return true, err
		}
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
	if stopReason, stopDetails, selected, err := s.projectChampionSelectedFollowUpStopReason(job.ProjectID); err != nil {
		return false, err
	} else if selected {
		message := fmt.Sprintf("Experiment Planner skipped for plan %s because the project already has a selected champion.", input.SourcePlan.ID)
		s.recordChampionSelectedFollowUpBlocked(job.ProjectID, input.SourcePlan.ID, "", "", message, stopReason, stopDetails)
		return true, nil
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

	var decision decisions.AgentDecision
	policyReference := policies.PersistenceReference{
		EvaluationID:        plannerAttempt.PolicyEvaluation.ID,
		EffectivePolicyHash: plannerAttempt.PolicyEvaluation.EffectivePolicyHash,
		Status:              policyStatusAllowed,
	}
	if decisionType == decisions.TypeAddExperiments {
		candidateRows, provenanceErr := candidateProvenanceCreatesFromPayload(payload)
		if provenanceErr != nil {
			return false, provenanceErr
		}
		decision, _, err = s.store.CreateAgentDecisionWithCandidateProvenanceAndPolicy(
			job.ProjectID,
			input.SourcePlan.ID,
			decisionType,
			recommendation.Rationale,
			payload,
			candidateRows,
			policyReference,
		)
	} else {
		decision, err = s.store.CreateAgentDecisionWithPolicy(
			job.ProjectID,
			input.SourcePlan.ID,
			decisionType,
			recommendation.Rationale,
			payload,
			policyReference,
		)
	}
	if err != nil {
		return false, err
	}
	if err := s.persistProjectChampionFromDecision(job.ProjectID, decision); err != nil {
		log.Printf("persist planner champion failed for project %s decision %s: %v", job.ProjectID, decision.ID, err)
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
	executionEvents, err := s.store.ListProjectExecutionEvents(projectID, 100)
	if err != nil {
		executionEvents = []execution.ExecutionEvent{}
	}
	summaries, err := s.store.ListProjectTrainingRunSummaries(projectID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(projectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return agents.ExperimentPlannerInput{}, false, err
	}
	executionEvidenceByJob, err := s.executionEvidenceForJobs(projectJobs)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	learningSummaries := learningEligibleSummaries(summaries, executionEvidenceByJob)
	learningEvaluations := evaluationsForEligibleSummaries(evaluations, learningSummaries)

	planJobs := jobsForPlan(projectJobs, plan.ID)
	allPlanSummaries := summariesForPlanID(summaries, plan.ID)
	if !planTrainingRunsComplete(plan, allPlanSummaries) {
		return agents.ExperimentPlannerInput{}, false, nil
	}
	planSummaries := summariesForPlanID(learningSummaries, plan.ID)
	planEvaluations := evaluationsForPlanID(learningEvaluations, plan.ID)
	automationSettings := s.currentAutomationSettings()
	minimumMeaningfulImprovement := plannerMinimumMeaningfulImprovementFromEnv(automationSettings.AgentMode)
	objectiveContext := projectObjectiveContext(project.Goal)
	currentChampion, baselineChampion, sourcePlanDeltas, noImprovementRounds, stopSignals := experimentPlannerPerformanceContext(
		plan.TargetMetric,
		projectPlans,
		learningSummaries,
		learningEvaluations,
		objectiveContext,
		plan.ID,
	)
	attachPlannerExecutionEvidence(currentChampion, baselineChampion, sourcePlanDeltas, executionEvidenceByJob)

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
	executionCapabilityCard, executionFeedback, err := s.plannerExecutionCapabilityContext(dataset, metadataSummary, projectJobs, executionEvents)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	effectivePolicy, err := s.resolveProposalPolicy(project, dataset, policyOperationPropose)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	executionCapabilityCard = filterExecutionCapabilityCardByPolicy(executionCapabilityCard, effectivePolicy)

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
		ModelCatalog:                 effectiveSupportedModelCatalog(effectivePolicy),
		EffectiveCatalog:             effectivePolicy.PermittedCatalog,
		EffectivePolicyCard:          policies.PromptCardFromEffectivePolicy(effectivePolicy),
		EffectivePolicy:              &effectivePolicy,
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
		PriorSummaries:               learningSummaries,
		PriorEvaluations:             learningEvaluations,
		PriorMemory:                  priorMemory,
		ExistingExperimentSignatures: experimentSignaturesForPlans(projectPlans),
		ExecutionCapabilityCard:      executionCapabilityCard,
		ExecutionEnforcementFeedback: executionFeedback,
		ExecutionEvidence:            executionEvidenceList(executionEvidenceByJob),
		AgentMode:                    automationSettings.AgentMode,
		MaxExperiments:               maxLLMPlannerExperiments,
		MaxFollowUpRounds:            s.maxAutoFollowUpRounds(),
		FollowUpRound:                followUpRoundCount(projectPlans),
	}
	rolloutPolicy, err := calibration.PlannerRolloutPolicyFromEnvironment()
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, fmt.Errorf("planner rollout policy: %w", err)
	}
	rolloutAssignment, err := calibration.AssignPlannerRollout(rolloutPolicy, projectID)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, fmt.Errorf("planner rollout assignment: %w", err)
	}
	input.RolloutAssignment = &rolloutAssignment
	input.ProjectTrajectory = agents.ComputeProjectTrajectoryDiagnosis(input)
	input.RetrievalVariant, err = plannerRetrievalVariantForRollout(plannerRetrievalVariant(), input.RolloutAssignment)
	if err != nil {
		return agents.ExperimentPlannerInput{}, false, err
	}
	input.RetrievedMemory = s.retrievePlannerMemory(context.Background(), input)
	input.RankerV2PriorSnapshot = s.plannerRankerV2PriorSnapshot(projectID, time.Now().UTC(), minimumMeaningfulImprovement)
	return input, true, nil
}

func plannerRetrievalVariantForRollout(base memory.PlannerRetrievalVariant, assignment *calibration.PlannerRolloutAssignment) (memory.PlannerRetrievalVariant, error) {
	value := strings.ToLower(calibration.PlannerRolloutVariantValue(assignment, calibration.RolloutDimensionRetrieval))
	switch value {
	case "":
		return base, nil
	case "enabled":
		base.Enabled = true
		base.LogOnly = false
	case "log_only":
		base.Enabled = true
		base.LogOnly = true
	case "disabled":
		base.Enabled = false
		base.LogOnly = false
	default:
		return memory.PlannerRetrievalVariant{}, fmt.Errorf("planner rollout retrieval variant %q is invalid", value)
	}
	return base, nil
}

func (s *Server) plannerRankerV2PriorSnapshot(projectID string, evaluationStart time.Time, meaningfulImprovement float64) *calibration.RankerV2PriorSnapshot {
	evaluationStart = evaluationStart.UTC()
	if evaluationStart.IsZero() {
		evaluationStart = time.Now().UTC()
	}
	trainingWindow := calibration.TimeWindow{Start: evaluationStart.Add(-90 * 24 * time.Hour), End: evaluationStart}
	evaluationWindow := calibration.TimeWindow{Start: evaluationStart, End: evaluationStart.Add(30 * 24 * time.Hour)}
	observations, readErr := s.store.ReadCalibrationObservations(projectID, trainingWindow, 2000)
	candidates := observations.Candidates
	if readErr != nil {
		log.Printf("read ranker v2 calibration priors failed for project %s: %v", projectID, readErr)
		candidates = nil
	}
	snapshot, err := calibration.BuildRankerV2PriorSnapshot(
		candidates, trainingWindow, evaluationWindow, calibration.RankerV2PriorMinSampleSize,
		meaningfulImprovement, observations.CandidatesTruncated,
	)
	if err != nil {
		log.Printf("build ranker v2 calibration priors failed for project %s: %v", projectID, err)
		return nil
	}
	if readErr != nil {
		snapshot.SourceStatus = "read_failed_neutral_fallback"
	}
	return &snapshot
}

func (s *Server) plannerExecutionCapabilityContext(
	dataset datasets.Dataset,
	metadataSummary map[string]any,
	projectJobs []jobs.ExperimentJob,
	events []execution.ExecutionEvent,
) (execution.PlannerCapabilityCard, []execution.EnforcementFeedback, error) {
	task := "image_classification"
	if datasetHasYOLODetectionEvidence(dataset, metadataSummary) {
		task = "object_detection"
	}
	provider := s.defaultExecuteExperimentPlanRequest().Provider
	runner, err := executionRunnerFor(provider, task)
	if err != nil {
		return execution.PlannerCapabilityCard{}, nil, err
	}
	modelFamilies := []string{}
	for _, model := range supportedModelCatalogForDataset(dataset, metadataSummary) {
		if model.TaskType == task {
			modelFamilies = append(modelFamilies, model.Family)
		}
	}
	card, err := execution.BuildPlannerCapabilityCard(task, runner, executionValidationMode(), modelFamilies)
	if err != nil {
		return execution.PlannerCapabilityCard{}, nil, err
	}
	reports := executionValidationReports(projectJobs, events)
	return card, execution.SummarizeEnforcementFeedback(reports, task, runner, 12), nil
}

func executionValidationReports(projectJobs []jobs.ExperimentJob, events []execution.ExecutionEvent) []execution.ExecutionValidationReport {
	out := []execution.ExecutionValidationReport{}
	seen := map[string]bool{}
	appendReport := func(value any) {
		blob, err := json.Marshal(value)
		if err != nil {
			return
		}
		var report execution.ExecutionValidationReport
		if err := json.Unmarshal(blob, &report); err != nil || report.SchemaVersion != execution.ExecutionValidationSchemaVersionV1 {
			return
		}
		key := report.AcceptedSpecHash + "|" + report.ModelFamily + "|" + executionValidationSummary(report)
		if seen[key] {
			return
		}
		seen[key] = true
		out = append(out, report)
	}
	for _, job := range projectJobs {
		if value, ok := job.Config[execution.ExecutionValidationConfigKey]; ok {
			appendReport(value)
		}
	}
	for _, event := range events {
		if event.EventType == execution.EventExecutionValidationReported {
			appendReport(event.Payload["report"])
		}
	}
	return out
}

func (s *Server) recordExperimentPlannerInvocation(
	input agents.ExperimentPlannerInput,
	config llm.Config,
	trace agents.ExperimentPlanningTrace,
	acceptedForMemory bool,
	facts plannerInvocationFacts,
) (memory.AgentInvocation, error) {
	validationStatus := trace.ValidationStatus
	if validationStatus == "" {
		validationStatus = memory.InvocationValidationFailed
	}
	inputContext := agentInvocationInputContext(trace.PromptContext, config, trace.ToolRounds, trace.Usage, trace.ToolCalls, trace.ToolResults, trace.RejectedToolCalls, trace.DryRunValidationResults)
	variant, variantID, err := experimentPlannerVariant(input, config, trace)
	if err != nil {
		return memory.AgentInvocation{}, err
	}
	providerUsage, err := plannerProviderUsage(trace.Usage)
	if err != nil {
		return memory.AgentInvocation{}, err
	}
	derivedCost := plannerDerivedCost(config, trace.Usage)
	var strictVerdict *plannervalidation.Verdict
	if trace.StrictValidationVerdict.SchemaVersion != "" {
		verdict := trace.StrictValidationVerdict
		strictVerdict = &verdict
	}
	if runtime, ok := inputContext["invocation_runtime"].(map[string]any); ok {
		runtime["request_temperature"] = trace.Request.Temperature
		runtime["request_reasoning_effort"] = trace.Request.ReasoningEffort
		runtime["planner_variant_id"] = variantID
		runtime["attempt_group_id"] = facts.AttemptGroupID
		runtime["attempt_index"] = facts.AttemptIndex
		runtime["retry_reason"] = facts.RetryReason
		runtime["wall_latency_ms"] = facts.WallLatencyMS
		if input.RolloutAssignment != nil {
			runtime["planner_rollout_assignment"] = input.RolloutAssignment
		}
		if derivedCost != nil {
			runtime["derived_cost"] = derivedCost
		}
		if len(trace.OutputNormalizations) > 0 {
			runtime["output_normalizations"] = trace.OutputNormalizations
		}
	}

	return s.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:               input.Project.ID,
		DatasetID:               input.SourcePlan.DatasetID,
		PlanID:                  input.SourcePlan.ID,
		AgentName:               agents.ExperimentPlannerAgentName,
		AgentVersion:            trace.AgentVersion,
		PromptVersion:           trace.PromptVersion,
		PlannerVariantID:        variantID,
		PlannerVariant:          &variant,
		RolloutAssignment:       input.RolloutAssignment,
		ValidationMode:          variant.ValidationMode,
		AttemptGroupID:          facts.AttemptGroupID,
		AttemptIndex:            facts.AttemptIndex,
		RetryReason:             facts.RetryReason,
		WallLatencyMS:           facts.WallLatencyMS,
		ProviderUsage:           providerUsage,
		DerivedCost:             derivedCost,
		Provider:                config.Provider,
		Model:                   config.Model,
		InputMessages:           llmMessagesForMemory(trace.Request.Messages),
		InputContext:            inputContext,
		RawOutput:               string(trace.RawOutput),
		ParsedOutput:            trace.ParsedOutput,
		ValidationStatus:        validationStatus,
		ValidationError:         trace.ValidationError,
		StrictValidationVerdict: strictVerdict,
		AcceptedForMemory:       acceptedForMemory,
		HumanFeedback:           map[string]any{},
		DownstreamOutcome:       map[string]any{},
	})
}

type plannerInvocationFacts struct {
	AttemptGroupID string
	AttemptIndex   int
	RetryReason    string
	WallLatencyMS  float64
}

func experimentPlannerVariant(input agents.ExperimentPlannerInput, config llm.Config, trace agents.ExperimentPlanningTrace) (memory.PlannerVariant, string, error) {
	retrieval := input.RetrievalVariant
	if retrieval.MaxCards == 0 {
		retrieval = plannerRetrievalVariant()
	}
	executionValidatorMode := firstNonEmptyString(input.ExecutionCapabilityCard.Mode, executionValidationMode())
	variant := memory.PlannerVariant{
		IdentitySchemaVersion:        memory.PlannerVariantIdentitySchemaV1,
		AgentVersion:                 trace.AgentVersion,
		PromptVersion:                trace.PromptVersion,
		StaticPromptVersion:          trace.StaticPromptVersion,
		ContextBuilderVersion:        trace.ContextBuilderVersion,
		ToolPolicyVersion:            agents.ExperimentPlannerToolPolicyVersion,
		ValidatorVersion:             agents.ExperimentPlannerValidatorVersion,
		ValidationMode:               firstNonEmptyString(trace.ValidatorMode, plannerValidationMode()),
		ExecutionValidatorVersion:    execution.ExecutionValidationSchemaVersionV1,
		ExecutionValidationMode:      executionValidatorMode,
		RankerVersion:                agents.PlannerSchedulingRankerVersion(input),
		RankerMultiFidelityEnabled:   trace.RankerMultiFidelity,
		RetrievalPolicyVersion:       agents.ExperimentPlannerRetrievalPolicyVersion,
		Retrieval:                    retrieval,
		RetryPolicyVersion:           plannerRetryPolicyVersion,
		MaxBackendValidationRetries:  plannerBackendValidationRetryLimit(),
		DecisionPolicyVersion:        plannerDecisionPolicyVersion,
		AgentMode:                    llm.NormalizeAgentMode(input.AgentMode),
		TerminalPlannerGuards:        terminalPlannerGuardsEnabledForInput(input),
		MinimumMeaningfulImprovement: input.MinimumMeaningfulImprovement,
		MaxFollowUpRounds:            input.MaxFollowUpRounds,
		Provider:                     config.Provider,
		APIStyle:                     config.APIStyle,
		EffectiveAPIStyle:            llm.EffectiveAPIStyle(config.Provider, config.APIStyle),
		Model:                        firstNonEmptyString(trace.Request.Model, config.Model),
		EndpointFingerprint:          llm.EndpointFingerprint(config.BaseURL),
		RequestTemperature:           trace.Request.Temperature,
		TemperatureSent:              llm.EffectiveAPIStyle(config.Provider, config.APIStyle) == llm.APIStyleChatCompletions,
		RequestReasoningEffort:       trace.Request.ReasoningEffort,
		ReasoningEffortSent:          llm.EffectiveAPIStyle(config.Provider, config.APIStyle) == llm.APIStyleResponses && strings.TrimSpace(trace.Request.ReasoningEffort) != "",
		ConfiguredReasoningEffort:    config.ReasoningEffort,
		PlateauReasoningEffort:       config.PlateauReasoningEffort,
		StoredResponses:              config.StoredResponses,
		MaxToolRounds:                config.MaxToolRounds,
		MaxSelectedExperiments:       agents.EffectivePlannerMaxExperiments(input.MaxExperiments),
		MaxProviderRetries:           config.MaxRetries,
		RequestTimeoutMS:             config.Timeout.Milliseconds(),
	}
	variantID, err := memory.ComputePlannerVariantID(variant)
	if err != nil {
		return memory.PlannerVariant{}, "", fmt.Errorf("compute planner variant identity: %w", err)
	}
	return variant, variantID, nil
}

func plannerValidationMode() string {
	return plannervalidation.ModeFromEnvironment()
}

func plannerProviderUsage(usage *llm.Usage) (map[string]any, error) {
	if usage == nil {
		return map[string]any{}, nil
	}
	out, err := mapFromStruct(usage)
	if err != nil {
		return nil, fmt.Errorf("encode planner provider usage: %w", err)
	}
	return out, nil
}

func plannerDerivedCost(config llm.Config, usage *llm.Usage) *memory.PlannerInvocationCost {
	snapshot, err := llm.PricingSnapshotFromEnv()
	if err != nil {
		log.Printf("planner pricing snapshot ignored: %v", err)
		return nil
	}
	runtimeModel := config.Model
	if usage != nil {
		runtimeModel = firstNonEmptyString(usage.RequestModel, config.Model)
	}
	if snapshot != nil && !snapshot.MatchesRuntime(config.Provider, runtimeModel) {
		log.Printf("planner pricing snapshot %q ignored for unmatched runtime %s/%s", snapshot.PricingVersion, config.Provider, runtimeModel)
		return nil
	}
	derived, err := llm.DeriveCost(usage, snapshot)
	if err != nil {
		log.Printf("planner usage cost derivation failed: %v", err)
		return nil
	}
	if derived == nil || snapshot == nil {
		return nil
	}
	return &memory.PlannerInvocationCost{
		PricingVersion:                 derived.PricingVersion,
		Currency:                       "USD",
		Provider:                       config.Provider,
		Model:                          runtimeModel,
		InputTokens:                    usage.InputTokens,
		CachedInputTokens:              usage.CachedInputTokens,
		OutputTokens:                   usage.OutputTokens,
		InputUSDPerMillionTokens:       snapshot.InputUSDPerMillionTokens,
		CachedInputUSDPerMillionTokens: snapshot.CachedInputUSDPerMillionTokens,
		OutputUSDPerMillionTokens:      snapshot.OutputUSDPerMillionTokens,
		UncachedInputCostUSD:           derived.UncachedInputCostUSD,
		CachedInputCostUSD:             derived.CachedInputCostUSD,
		OutputCostUSD:                  derived.OutputCostUSD,
		TotalCostUSD:                   derived.TotalCostUSD,
	}
}

func newPlannerAttemptGroupID() string {
	value := make([]byte, 16)
	if _, err := rand.Read(value); err == nil {
		return "planner_attempt_" + hex.EncodeToString(value)
	}
	return fmt.Sprintf("planner_attempt_%d", time.Now().UTC().UnixNano())
}

type experimentPlannerAttemptResult struct {
	Input            agents.ExperimentPlannerInput
	Trace            agents.ExperimentPlanningTrace
	Invocation       memory.AgentInvocation
	Recommendation   agents.ExperimentPlanningRecommendation
	Payload          map[string]any
	PolicyEvaluation policies.Evaluation
}

func (s *Server) runExperimentPlannerWithBackendValidationRetry(
	ctx context.Context,
	agent agents.ExperimentPlannerAgent,
	input agents.ExperimentPlannerInput,
	config llm.Config,
	agentMode string,
) (experimentPlannerAttemptResult, error) {
	agentMode = llm.NormalizeAgentMode(firstNonEmptyString(agentMode, input.AgentMode))
	attemptInput := input
	attemptInput.AgentMode = agentMode
	if attemptInput.EffectivePolicy == nil {
		implicit := policies.ImplicitEffectivePolicy(attemptInput.ExecutionCapabilityCard.Task, attemptInput.ExecutionCapabilityCard.Runner)
		attemptInput.EffectivePolicy = &implicit
		attemptInput.EffectiveCatalog = implicit.PermittedCatalog
		attemptInput.EffectivePolicyCard = policies.PromptCardFromEffectivePolicy(implicit)
	}
	terminalGuards := terminalPlannerGuardsEnabledForMode(agentMode)
	attemptInput.TerminalPlannerGuardsEnabled = &terminalGuards
	var result experimentPlannerAttemptResult
	var lastErr error
	attemptGroupID := newPlannerAttemptGroupID()
	retryReason := ""
	retryLimit := plannerBackendValidationRetryLimit()
	for attempt := 0; attempt <= retryLimit; attempt++ {
		startedAt := time.Now()
		trace, err := agent.PlanWithTrace(ctx, attemptInput)
		wallLatencyMS := float64(time.Since(startedAt)) / float64(time.Millisecond)
		acceptedForMemory := err == nil
		invocation, invocationErr := s.recordExperimentPlannerInvocation(attemptInput, config, trace, acceptedForMemory, plannerInvocationFacts{
			AttemptGroupID: attemptGroupID,
			AttemptIndex:   attempt,
			RetryReason:    retryReason,
			WallLatencyMS:  wallLatencyMS,
		})
		if invocationErr != nil {
			result = experimentPlannerAttemptResult{
				Input: attemptInput,
				Trace: trace,
			}
			return result, fmt.Errorf("persist experiment planner invocation for plan %s: %w", input.SourcePlan.ID, invocationErr)
		}
		result = experimentPlannerAttemptResult{
			Input:      attemptInput,
			Trace:      trace,
			Invocation: invocation,
		}
		if err != nil {
			lastErr = err
			willRetry := attempt < retryLimit && shouldRetryExperimentPlannerTraceValidation(trace, err)
			if persistErr := s.persistPlannerValidationAttempt(invocation, trace.StrictValidationVerdict, attempt, false, willRetry); persistErr != nil {
				return result, persistErr
			}
			s.recordPlannerValidationRejection(invocation, err, attempt, willRetry)
			if willRetry {
				attemptInput.ValidationFeedback = append(attemptInput.ValidationFeedback, plannerValidationFeedback(trace.Recommendation, err, attempt+1))
				retryReason = plannerRetryReasonTraceValidation
				continue
			}
			result.Recommendation = trace.Recommendation
			return result, err
		}

		recommendation := applyExperimentPlannerStopCriteria(trace.Recommendation, attemptInput)
		if strings.EqualFold(recommendation.DecisionType, decisions.TypeAddExperiments) {
			experiments, automlWarnings, prepareErr := s.prepareAutoMLExperimentsForProjectWithPolicy(input.Project.ID, recommendation.ProposedExperiments, attemptInput.EffectivePolicy)
			if prepareErr != nil {
				lastErr = prepareErr
				willRetry := attempt < retryLimit && shouldRetryExperimentPlannerValidation(recommendation)
				if persistErr := s.persistPlannerValidationAttempt(invocation, trace.StrictValidationVerdict, attempt, false, willRetry); persistErr != nil {
					return result, persistErr
				}
				s.recordPlannerValidationRejection(invocation, prepareErr, attempt, willRetry)
				if !willRetry {
					result.Recommendation = recommendation
					return result, prepareErr
				}
				attemptInput.ValidationFeedback = append(attemptInput.ValidationFeedback, plannerValidationFeedback(recommendation, prepareErr, attempt+1))
				retryReason = plannerRetryReasonAutoMLPreparation
				continue
			}
			recommendation.ProposedExperiments = experiments
			recommendation.NoveltyNotes = append(recommendation.NoveltyNotes, automlWarnings...)
			experiments, executionNormalizationWarnings := normalizePlannerProposalExperimentsForExecution(recommendation.ProposedExperiments)
			recommendation.ProposedExperiments = experiments
			recommendation.NoveltyNotes = append(recommendation.NoveltyNotes, executionNormalizationWarnings...)
		}
		executionReports, capabilityErr := validatePlannerExecutionCapabilities(recommendation.ProposedExperiments, attemptInput)
		if capabilityErr != nil {
			lastErr = capabilityErr
			willRetry := attempt < retryLimit && shouldRetryExperimentPlannerValidation(recommendation)
			if persistErr := s.persistPlannerValidationAttempt(invocation, trace.StrictValidationVerdict, attempt, false, willRetry); persistErr != nil {
				return result, persistErr
			}
			s.recordPlannerValidationRejection(invocation, capabilityErr, attempt, willRetry)
			if !willRetry {
				result.Recommendation = recommendation
				return result, capabilityErr
			}
			attemptInput.ValidationFeedback = append(attemptInput.ValidationFeedback, plannerValidationFeedback(recommendation, capabilityErr, attempt+1, executionReports...))
			retryReason = plannerRetryReasonExecutionCapabilities
			continue
		}
		payload, err := experimentPlannerDecisionPayload(recommendation, invocation, agentMode, attemptInput)
		if err == nil && strings.EqualFold(recommendation.DecisionType, decisions.TypeAddExperiments) {
			_, err = candidateProvenanceCreatesFromPayload(payload)
		}
		if err == nil && strings.EqualFold(recommendation.DecisionType, decisions.TypeAddExperiments) {
			var evaluation policies.Evaluation
			var persistedExperiments []plans.PlannedExperiment
			persistedExperiments, err = plannedExperimentsFromPayload(payload)
			if err == nil {
				evaluation, err = s.recordProposalPolicyEvaluation(*attemptInput.EffectivePolicy, policyOperationPersistProposal, persistedExperiments, invocation.ID)
			}
			if err == nil {
				result.PolicyEvaluation = evaluation
				payload["proposal_policy_evaluation_id"] = evaluation.ID
				payload["effective_policy_hash"] = evaluation.EffectivePolicyHash
				payload["effective_policy_card"] = attemptInput.EffectivePolicyCard
			}
		}
		if err == nil {
			verdict := plannerStrictVerdictFromPayload(trace.StrictValidationVerdict, payload)
			if persistErr := s.persistPlannerValidationAttempt(invocation, verdict, attempt, true, false); persistErr != nil {
				return result, persistErr
			}
			if len(executionReports) > 0 {
				payload["execution_validation_reports"] = executionReports
			}
			if attempt > 0 {
				payload["validation_retry_count"] = attempt
				payload["validation_feedback_applied"] = attemptInput.ValidationFeedback
			}
			result.Recommendation = recommendation
			result.Payload = payload
			return result, nil
		}

		lastErr = err
		willRetry := attempt < retryLimit && shouldRetryExperimentPlannerValidation(recommendation)
		verdict := plannerStrictVerdictFromError(trace.StrictValidationVerdict, trace.ValidatorMode, err)
		if persistErr := s.persistPlannerValidationAttempt(invocation, verdict, attempt, false, willRetry); persistErr != nil {
			return result, persistErr
		}
		s.recordPlannerValidationRejection(invocation, err, attempt, willRetry)
		if !willRetry {
			result.Recommendation = recommendation
			return result, err
		}
		attemptInput.ValidationFeedback = append(attemptInput.ValidationFeedback, plannerValidationFeedback(recommendation, err, attempt+1))
		retryReason = plannerRetryReasonDecisionPayload
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

func (s *Server) persistPlannerValidationAttempt(invocation memory.AgentInvocation, verdict plannervalidation.Verdict, attempt int, accepted bool, willRetry bool) error {
	if invocation.ID == "" {
		return nil
	}
	mode := plannervalidation.NormalizeMode(firstNonEmptyString(verdict.Mode, invocation.ValidationMode))
	outcome := plannervalidation.Outcome{
		SchemaVersion:   plannervalidation.OutcomeSchemaVersionV1,
		Mode:            mode,
		FirstPassStatus: plannervalidation.FirstPassUnknown,
		EventualStatus:  plannervalidation.EventualRejected,
		RetryOutcome:    plannervalidation.RetryExhausted,
	}
	if attempt == 0 {
		if accepted {
			outcome.FirstPassStatus = plannervalidation.FirstPassAccepted
		} else {
			outcome.FirstPassStatus = plannervalidation.FirstPassRejected
		}
	} else {
		outcome.FirstPassStatus = plannervalidation.FirstPassRejected
	}
	switch {
	case accepted && attempt == 0:
		outcome.EventualStatus = plannervalidation.EventualAccepted
		outcome.RetryOutcome = plannervalidation.RetryNotNeeded
	case accepted:
		outcome.EventualStatus = plannervalidation.EventualAccepted
		outcome.RetryOutcome = plannervalidation.RetryAccepted
	case willRetry:
		outcome.EventualStatus = plannervalidation.EventualPending
		outcome.RetryOutcome = plannervalidation.RetryScheduled
	}
	if verdict.SchemaVersion == "" {
		verdict = plannervalidation.Verdict{
			SchemaVersion: plannervalidation.VerdictSchemaVersionV1,
			Mode:          mode,
			Status:        plannervalidation.VerdictNotEvaluated,
		}
	}
	if _, err := s.store.UpdateAgentInvocationValidation(invocation.ID, verdict, outcome); err != nil {
		return fmt.Errorf("persist planner validation outcome for invocation %s: %w", invocation.ID, err)
	}
	return nil
}

func plannerStrictVerdictFromPayload(base plannervalidation.Verdict, payload map[string]any) plannervalidation.Verdict {
	value, ok := payload["planner_strict_validation_verdict"]
	if !ok {
		return base
	}
	var verdict plannervalidation.Verdict
	if typed, ok := value.(plannervalidation.Verdict); ok {
		verdict = typed
	} else if blob, err := json.Marshal(value); err == nil {
		_ = json.Unmarshal(blob, &verdict)
	}
	return plannervalidation.Merge(base, verdict)
}

func plannerStrictVerdictFromError(base plannervalidation.Verdict, mode string, validationErr error) plannervalidation.Verdict {
	var evaluationErr plannervalidation.EvaluationError
	if !errors.As(validationErr, &evaluationErr) {
		return base
	}
	verdict := plannervalidation.Verdict{
		SchemaVersion: plannervalidation.VerdictSchemaVersionV1,
		Mode:          plannervalidation.NormalizeMode(mode),
		Status:        plannervalidation.VerdictWouldBlock,
		WouldBlock:    true,
		Findings:      append([]plannervalidation.Finding(nil), evaluationErr.Findings...),
	}
	return plannervalidation.Merge(base, verdict)
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
				if plannervalidation.IsStrict(plannerValidationMode()) {
					return invalidPlannerDryRunResult(result, err)
				}
				experiments, relaxedValidationWarnings = plannerExperimentsWithProposalMechanismsRelaxed(recommendation)
				relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
			}
			for index := range experiments {
				if experiments[index].AutoML == nil || !experiments[index].AutoML.Enabled {
					continue
				}
				scope := automl.ExecutionScope{
					CapabilityVersion: input.ExecutionCapabilityCard.CapabilityVersion,
					Task:              input.ExecutionCapabilityCard.Task,
					Runner:            input.ExecutionCapabilityCard.Runner,
				}
				if scope.Runner == "" {
					var scopeErr error
					scope, scopeErr = autoMLExecutionScopeForExperiment(experiments[index], "modal")
					if scopeErr != nil {
						return invalidPlannerDryRunResult(result, scopeErr)
					}
				}
				prepared, err := prepareAutoMLExperimentWithHistoryForExecution(experiments[index], index, automl.SamplerSeededRandom, nil, scope)
				if err != nil {
					return invalidPlannerDryRunResult(result, err)
				}
				experiments[index] = prepared
			}
			experiments, executionNormalizationWarnings := normalizePlannerProposalExperimentsForExecution(experiments)
			for index, experiment := range experiments {
				if err := validatePlannedExperiment(experiment, index); err != nil {
					return invalidPlannerDryRunResult(result, err)
				}
				if err := validateExperimentDatasetCompatibility(experiment, input.Dataset, index); err != nil {
					return invalidPlannerDryRunResult(result, err)
				}
			}
			if input.EffectivePolicy != nil {
				if _, err := policies.EvaluateProposal(*input.EffectivePolicy, policyOperationPersistProposal, experiments); err != nil {
					return invalidPlannerDryRunResult(result, err)
				}
			}
			if err := validateLLMPlannerMechanismContract(experiments, recommendation.EvidenceUsed); err != nil {
				if plannervalidation.IsStrict(plannerValidationMode()) {
					return invalidPlannerDryRunResult(result, err)
				}
				relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
			}
			if err := validateNovelProposedExperiments(experiments, input.PriorPlans); err != nil {
				if plannervalidation.IsStrict(plannerValidationMode()) {
					return invalidPlannerDryRunResult(result, err)
				}
				relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
			}
			if err := validateMechanismDatasetEvidence(profileWithAgentSafeMetadataSummary(input.Dataset.Profile, input.DatasetInsights.AgentSafeMetadataSummary), experiments, recommendation.EvidenceUsed); err != nil {
				if plannervalidation.IsStrict(plannerValidationMode()) {
					return invalidPlannerDryRunResult(result, err)
				}
				relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
			}
			executionReports, err := validatePlannerExecutionCapabilities(experiments, input)
			result.Details["execution_validation_mode"] = input.ExecutionCapabilityCard.Mode
			result.Details["execution_validation_reports"] = executionReports
			if err != nil {
				return invalidPlannerDryRunResult(result, err)
			}
			result.Details["validated_experiment_count"] = len(experiments)
			if len(executionNormalizationWarnings) > 0 {
				result.Details["execution_normalization_warnings"] = uniqueStrings(executionNormalizationWarnings)
			}
			if len(relaxedValidationWarnings) > 0 {
				result.Details["planner_validation_mode"] = "relaxed"
				result.Details["planner_validation_warnings"] = uniqueStrings(relaxedValidationWarnings)
			}
		}
		return result
	}
}

func validatePlannerExecutionCapabilities(
	experiments []plans.PlannedExperiment,
	input agents.ExperimentPlannerInput,
) ([]execution.ExecutionValidationReport, error) {
	if len(experiments) == 0 || input.ExecutionCapabilityCard.Runner == "" {
		return nil, nil
	}
	provider := providerForExecutionRunner(input.ExecutionCapabilityCard.Runner)
	reports := make([]execution.ExecutionValidationReport, 0, len(experiments))
	blocked := []string{}
	for index, experiment := range experiments {
		if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
			continue
		}
		spec, err := buildExecutionSpecV1(experiment, provider)
		if err != nil {
			return reports, err
		}
		modelSpec, _ := supportedModelSpecByName(experiment.Model)
		report, err := execution.ValidateExecutionSpecV1(spec, modelSpec.Family, input.ExecutionCapabilityCard.Mode)
		if err != nil {
			return reports, err
		}
		reports = append(reports, report)
		if report.Mode == execution.ValidationModeEnforce && report.WouldBlock {
			blocked = append(blocked, fmt.Sprintf("experiment %d: %s", index, executionValidationSummary(report)))
		}
	}
	if len(blocked) > 0 {
		return reports, fmt.Errorf("%w: execution fidelity enforcement rejected planner proposal: %s", store.ErrInvalidRequest, strings.Join(blocked, "; "))
	}
	return reports, nil
}

// validateProposalAcceptedSpecNovelty compares the executable semantic
// identity, not the raw requested experiment. This catches proposals whose
// unsupported or default-equivalent fields make them execution-time no-ops.
func validateProposalAcceptedSpecNovelty(experiments []plans.PlannedExperiment, input agents.ExperimentPlannerInput) error {
	if len(experiments) == 0 || strings.TrimSpace(input.ExecutionCapabilityCard.Runner) == "" {
		return nil
	}
	existing := map[string]string{}
	for _, evidence := range input.ExecutionEvidence {
		hash := strings.TrimSpace(evidence.AcceptedSpecHash)
		if hash != "" {
			existing[hash] = evidence.JobID
		}
	}
	proposed := map[string]int{}
	provider := providerForExecutionRunner(input.ExecutionCapabilityCard.Runner)
	for index, experiment := range experiments {
		if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
			continue
		}
		spec, err := buildExecutionSpecV1(experiment, provider)
		if err != nil {
			return err
		}
		if jobID, ok := existing[spec.AcceptedSpecHash]; ok {
			return fmt.Errorf("%w: proposed experiment %d is a proposal-time no-op matching accepted spec %s from job %s", store.ErrInvalidRequest, index, spec.AcceptedSpecHash, jobID)
		}
		if previous, ok := proposed[spec.AcceptedSpecHash]; ok {
			return fmt.Errorf("%w: proposed experiment %d is a proposal-time no-op matching accepted spec %s from proposed experiment %d", store.ErrInvalidRequest, index, spec.AcceptedSpecHash, previous)
		}
		proposed[spec.AcceptedSpecHash] = index
	}
	return nil
}

func providerForExecutionRunner(runner string) string {
	switch runner {
	case "modal_torchvision", "modal_ultralytics":
		return "modal"
	default:
		return "local"
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

func plannerValidationFeedback(recommendation agents.ExperimentPlanningRecommendation, validationErr error, attempt int, reports ...execution.ExecutionValidationReport) agents.PlannerValidationFeedback {
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
		FieldFindings:       plannerValidationFieldFindings(validationErr, reports),
		Instructions: []string{
			"Return corrected JSON only.",
			"Do not repeat the rejected experiment configuration unchanged.",
			"Change a meaningful mechanism such as model family, preprocessing, augmentation policy, sampling/class balancing, scheduler, optimizer, regularization, or resolution strategy.",
			"Only propose experiments that backend validation can schedule.",
		},
	}
	if validationErr != nil && strings.Contains(strings.ToLower(validationErr.Error()), "execution fidelity enforcement") {
		feedback.Instructions = append(feedback.Instructions,
			"Follow each execution-fidelity suggested alternative: remove the blocked field, activate its documented prerequisite, or pivot to an executed field from execution_capability_card.",
			"You may preserve the higher-level mechanism when it remains meaningful after removing the blocked no-op; otherwise propose a different supported mechanism.",
		)
	}
	if validationErr != nil && strings.Contains(strings.ToLower(validationErr.Error()), "champion_challenge") {
		feedback.Instructions = append(
			feedback.Instructions,
			"For champion_challenge, every selected experiment_config reason or strategy must explicitly explain how that experiment can beat, improve on, or trade off against the current champion.",
		)
	}
	return feedback
}

func plannerValidationFieldFindings(validationErr error, reports []execution.ExecutionValidationReport) []agents.PlannerValidationFieldFinding {
	const maxFindings = 24
	out := []agents.PlannerValidationFieldFinding{}
	appendFinding := func(finding agents.PlannerValidationFieldFinding) {
		if len(out) >= maxFindings || strings.TrimSpace(finding.Field) == "" {
			return
		}
		out = append(out, finding)
	}
	for _, report := range reports {
		if !report.WouldBlock {
			continue
		}
		for _, finding := range report.Findings {
			appendFinding(agents.PlannerValidationFieldFinding{
				Field:                finding.Field,
				RequestedValue:       finding.RequestedValue,
				AcceptedValue:        finding.AcceptedValue,
				ReasonCode:           finding.ReasonCode,
				SuggestedAlternative: finding.SuggestedAlternative,
			})
		}
	}
	if len(out) > 0 || validationErr == nil {
		return out
	}
	var evaluationErr plannervalidation.EvaluationError
	if errors.As(validationErr, &evaluationErr) {
		for _, finding := range evaluationErr.Findings {
			appendFinding(agents.PlannerValidationFieldFinding{
				Field:                finding.Stage,
				ReasonCode:           finding.Code,
				SuggestedAlternative: finding.Message,
			})
		}
	}
	return out
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
	if !terminalPlannerGuardsEnabledForInput(input) {
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

func terminalPlannerGuardsEnabledForInput(input agents.ExperimentPlannerInput) bool {
	if input.TerminalPlannerGuardsEnabled != nil {
		return *input.TerminalPlannerGuardsEnabled
	}
	return terminalPlannerGuardsEnabledForMode(input.AgentMode)
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
	recommendation.CandidateSelectionTrace = nil
	recommendation.CandidateRankingsV2 = nil
	recommendation.CandidateSelectionTraceV2 = nil
	recommendation.RankerShadowComparison = nil
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
		"planner_variant_id":              invocation.PlannerVariantID,
		"planner_rollout_cohort_id":       invocation.RolloutCohortID,
		"planner_rollout_policy_id":       invocation.RolloutPolicyID,
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
		"candidate_rankings_v1":           recommendation.CandidateRankingsV1,
		"candidate_selection_trace":       recommendation.CandidateSelectionTrace,
		"candidate_rankings_v2":           recommendation.CandidateRankingsV2,
		"candidate_selection_trace_v2":    recommendation.CandidateSelectionTraceV2,
		"ranker_shadow_comparison":        recommendation.RankerShadowComparison,
		"ranker_v2_prior_snapshot":        input.RankerV2PriorSnapshot,
		"scheduling_ranker_version":       agents.PlannerSchedulingRankerVersion(input),
		"planner_rollout_assignment":      input.RolloutAssignment,
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
		"execution_evidence":              input.ExecutionEvidence,
		"optimizer_feedback_summary":      input.OptimizerFeedback,
		"no_improvement_rounds":           input.NoImprovementRounds,
		"minimum_meaningful_improvement":  input.MinimumMeaningfulImprovement,
		"stop_signals":                    input.StopSignals,
	}

	if strings.EqualFold(recommendation.DecisionType, decisions.TypeAddExperiments) {
		if len(recommendation.CandidateHypotheses) > 0 && len(recommendation.CandidateRankings) == len(recommendation.CandidateHypotheses) {
			payload[candidateProvenanceSchemaPayloadKey] = calibration.CandidateProvenanceSchemaVersionV1
			payload["execution_capability_card"] = input.ExecutionCapabilityCard
		}
		mode := plannervalidation.ModeFromEnvironment()
		relaxedValidationWarnings := []string{}
		relaxedExperiments, relaxedWarnings := plannerExperimentsWithProposalMechanismsRelaxed(recommendation)
		relaxedValidationWarnings = append(relaxedValidationWarnings, relaxedWarnings...)
		for index, experiment := range relaxedExperiments {
			if err := validatePlannedExperiment(experiment, index); err != nil {
				return nil, typedPlannerValidationError(mode, "invalid_task_or_model", plannervalidation.CategoryInvalidTaskModel, "decision_payload", err)
			}
		}
		if err := validateLLMPlannerMechanismContract(relaxedExperiments, recommendation.EvidenceUsed); err != nil {
			relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
		}
		if err := validateNovelProposedExperiments(relaxedExperiments, input.PriorPlans); err != nil {
			relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
		}

		var strictExperiments []plans.PlannedExperiment
		var strictMappingErr error
		if mode != plannervalidation.ModeRelaxed {
			strictExperiments, strictMappingErr = plannerExperimentsWithProposalMechanisms(recommendation)
		}
		strictVerdict, strictErr := plannervalidation.Evaluate(mode, []plannervalidation.Check{
			{
				Code:     "proposal_mechanism_mapping",
				Category: plannervalidation.CategoryMechanismMismatch,
				Stage:    "decision_payload",
				Validate: func() error { return strictMappingErr },
			},
			{
				Code:     "mechanism_evidence_mismatch",
				Category: plannervalidation.CategoryMechanismMismatch,
				Stage:    "decision_payload",
				Validate: func() error {
					if strictMappingErr != nil {
						return nil
					}
					return validateLLMPlannerMechanismContract(strictExperiments, recommendation.EvidenceUsed)
				},
			},
			{
				Code:     "duplicate_or_minor_only_proposal",
				Category: plannervalidation.CategoryProposalNoOp,
				Stage:    "decision_payload",
				Validate: func() error {
					if strictMappingErr != nil {
						return nil
					}
					return validateNovelProposedExperiments(strictExperiments, input.PriorPlans)
				},
			},
			{
				Code:     "accepted_spec_no_op",
				Category: plannervalidation.CategoryProposalNoOp,
				Stage:    "decision_payload",
				Validate: func() error {
					if strictMappingErr != nil {
						return nil
					}
					return validateProposalAcceptedSpecNovelty(strictExperiments, input)
				},
			},
			{
				Code:     "invalid_task_or_model",
				Category: plannervalidation.CategoryInvalidTaskModel,
				Stage:    "decision_payload",
				Validate: func() error {
					if strictMappingErr != nil {
						return nil
					}
					for index, experiment := range strictExperiments {
						if err := validateExperimentDatasetCompatibility(experiment, input.Dataset, index); err != nil {
							return err
						}
					}
					return nil
				},
			},
		})
		if strictErr != nil {
			return nil, strictErr
		}

		experiments := relaxedExperiments
		if mode == plannervalidation.ModeStrict {
			experiments = strictExperiments
			relaxedValidationWarnings = nil
		}
		for index, experiment := range experiments {
			if err := validateExperimentDatasetCompatibility(experiment, input.Dataset, index); err != nil {
				return nil, typedPlannerValidationError(mode, "invalid_task_or_model", plannervalidation.CategoryInvalidTaskModel, "decision_payload", err)
			}
		}
		payload["proposed_experiments"] = experiments
		if strictVerdict.Status != plannervalidation.VerdictNotEvaluated {
			payload["planner_strict_validation_verdict"] = strictVerdict
		}
		if len(relaxedValidationWarnings) > 0 {
			payload["planner_validation_mode"] = mode
			payload["planner_validation_warnings"] = uniqueStrings(relaxedValidationWarnings)
		}
	}

	return payload, nil
}

func typedPlannerValidationError(mode, code, category, stage string, err error) error {
	if err == nil || plannervalidation.NormalizeMode(mode) == plannervalidation.ModeRelaxed {
		return err
	}
	return plannervalidation.EvaluationError{Findings: []plannervalidation.Finding{{
		Code: code, Category: category, Stage: stage, Message: err.Error(),
	}}}
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
