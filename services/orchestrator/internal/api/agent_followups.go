package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plannervalidation"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
	"model-express/services/orchestrator/internal/strategies"
)

type scheduleFollowUpExperimentsResponse struct {
	Decision     decisions.AgentDecision `json:"decision"`
	FollowUpPlan *plans.ExperimentPlan   `json:"follow_up_plan,omitempty"`
}

type reopenExperimentationRequest struct {
	Reason       string `json:"reason"`
	SourcePlanID string `json:"source_plan_id"`
}

type reopenExperimentationResponse struct {
	Decision decisions.AgentDecision  `json:"decision"`
	Event    execution.ExecutionEvent `json:"event"`
}

type automaticExperimentReviewResult struct {
	Decision     *decisions.AgentDecision
	FollowUpPlan *plans.ExperimentPlan
	Jobs         []jobs.ExperimentJob
}

func (s *Server) schedulePlannerDecision(projectID string, sourcePlan plans.ExperimentPlan, decision decisions.AgentDecision, result automaticExperimentReviewResult) error {
	projectPlans, err := s.store.ListProjectExperimentPlans(projectID)
	if err != nil {
		return err
	}
	if stopReason, guardTag, ok, err := s.plannerFollowUpStopReason(projectID, sourcePlan, projectPlans); err != nil {
		return err
	} else if ok {
		if _, eventErr := s.store.CreateExecutionEvent(projectID, sourcePlan.ID, execution.EventAgentOutcomeRecorded, fmt.Sprintf("Planner follow-up scheduling blocked for plan %s.", sourcePlan.ID), map[string]any{
			"source_decision_id":        decision.ID,
			"backend_validation_status": "blocked",
			"backend_stop_guard":        guardTag,
			"reason":                    stopReason,
		}); eventErr != nil {
			log.Printf("record planner follow-up stop event failed: %v", eventErr)
		}
		return nil
	}
	if _, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		followUpPlan, _, err := s.ensureFollowUpPlan(projectID, sourcePlan, decision)
		if err != nil {
			if errors.Is(err, errNoNovelFollowUpExperiments) {
				return nil
			}
			return err
		}
		result.FollowUpPlan = &followUpPlan
		_, err = s.executeAutomaticFollowUpPlan(result)
		return err
	}

	maxRounds := s.maxAutoFollowUpRounds()
	if followUpRoundCount(projectPlans) >= maxRounds {
		log.Printf(
			"llm planner follow-up scheduling skipped for project %s plan %s: max follow-up rounds reached (%d)",
			projectID,
			sourcePlan.ID,
			maxRounds,
		)
		return s.selectBestAvailableChampionAfterMaxFollowUpStop(sourcePlan, decision, projectPlans, maxRounds)
	}

	followUpPlan, _, err := s.ensureFollowUpPlan(projectID, sourcePlan, decision)
	if err != nil {
		if errors.Is(err, errNoNovelFollowUpExperiments) {
			return nil
		}
		s.recordFollowUpSchedulingFailed(projectID, sourcePlan.ID, decision.ID, err)
		return err
	}

	result.FollowUpPlan = &followUpPlan
	_, err = s.executeAutomaticFollowUpPlan(result)
	return err
}

func (s *Server) plannerFollowUpStopReason(projectID string, sourcePlan plans.ExperimentPlan, projectPlans []plans.ExperimentPlan) (string, string, bool, error) {
	if stopReason, _, ok, err := s.projectChampionSelectedFollowUpStopReason(projectID); err != nil {
		return "", "", false, err
	} else if ok {
		return stopReason, "champion_selected_guard", true, nil
	}
	if !terminalPlannerGuardsEnabledForMode(s.currentAutomationSettings().AgentMode) {
		return "", "", false, nil
	}
	project, err := s.store.GetProject(projectID)
	if err != nil {
		return "", "", false, err
	}
	summaries, err := s.store.ListProjectTrainingRunSummaries(projectID)
	if err != nil {
		return "", "", false, err
	}
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(projectID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return "", "", false, err
	}
	currentChampion, _, _, noImprovementRounds, _ := experimentPlannerPerformanceContext(
		sourcePlan.TargetMetric,
		projectPlans,
		summaries,
		evaluations,
		projectObjectiveContext(project.Goal),
		sourcePlan.ID,
	)
	stopReason, guardTag, ok := experimentPlannerBackendStopReason(agents.ExperimentPlannerInput{
		CurrentChampion:              currentChampion,
		NoImprovementRounds:          noImprovementRounds,
		MinimumMeaningfulImprovement: plannerMinimumMeaningfulImprovement,
	})
	return stopReason, guardTag, ok, nil
}

func (s *Server) selectBestAvailableChampionAfterMaxFollowUpStop(sourcePlan plans.ExperimentPlan, decision decisions.AgentDecision, projectPlans []plans.ExperimentPlan, maxRounds int) error {
	followUpRounds := followUpRoundCount(projectPlans)
	reason := fmt.Sprintf("max_followup_rounds reached (%d)", maxRounds)
	payload := map[string]any{
		"source_decision_id":        decision.ID,
		"blocked_decision_type":     decision.DecisionType,
		"backend_validation_status": "blocked",
		"backend_stop_guard":        "max_followup_rounds_guard",
		"reason":                    reason,
		"max_followup_rounds":       maxRounds,
		"followup_rounds":           followUpRounds,
	}
	selected, err := s.selectBestAvailableChampionForTerminalPlanStop(sourcePlan, terminalChampionSelectionOptions{
		DecisionSource: maxFollowUpRoundsChampionDecisionSource,
		Trigger:        "max_followup_rounds",
		EventType:      execution.EventAgentOutcomeRecorded,
		Rationale: fmt.Sprintf(
			"Max follow-up rounds were reached for plan %s after ADD_EXPERIMENTS decision %s, so the backend selected the best successful model available.",
			sourcePlan.ID,
			decision.ID,
		),
		EventMessage: fmt.Sprintf(
			"Max follow-up rounds were reached for plan %s; the best available champion was selected instead of scheduling another plan.",
			sourcePlan.ID,
		),
		NoChampionMessage: fmt.Sprintf(
			"Max follow-up rounds were reached for plan %s, but no successful training run is available to select as champion.",
			sourcePlan.ID,
		),
		Payload: payload,
	})
	if err != nil {
		return err
	}
	if !selected {
		log.Printf("max follow-up terminal selection found no successful training run for project %s plan %s", sourcePlan.ProjectID, sourcePlan.ID)
	}
	return nil
}

func (s *Server) reviewProjectExperiments(c *gin.Context) {
	_, decision, err := s.createReviewerDecision(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, decision)
}

func (s *Server) scheduleFollowUpExperiments(c *gin.Context) {
	projectID := c.Param("id")

	sourcePlan, decision, err := s.followUpSourceDecision(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if decision.DecisionType != decisions.TypeAddExperiments {
		c.JSON(http.StatusOK, scheduleFollowUpExperimentsResponse{
			Decision: decision,
		})
		return
	}

	plan, created, err := s.ensureFollowUpPlan(projectID, sourcePlan, decision)
	if err != nil {
		if errors.Is(err, errNoNovelFollowUpExperiments) {
			c.JSON(http.StatusOK, scheduleFollowUpExperimentsResponse{
				Decision: decision,
			})
			return
		}
		writeStoreError(c, err)
		return
	}

	status := http.StatusOK
	if created {
		status = http.StatusCreated
	}
	c.JSON(status, scheduleFollowUpExperimentsResponse{
		Decision:     decision,
		FollowUpPlan: &plan,
	})
}

func (s *Server) reopenProjectExperimentation(c *gin.Context) {
	projectID := c.Param("id")
	var req reopenExperimentationRequest
	if !bindJSON(c, &req) {
		return
	}

	project, err := s.store.GetProject(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	state, err := s.projectChampionSelectionState(project.ID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if !state.Terminal {
		writeStoreError(c, fmt.Errorf("%w: experimentation can only be reopened after a selected champion or SELECT_CHAMPION decision exists", store.ErrInvalidRequest))
		return
	}

	planID := strings.TrimSpace(req.SourcePlanID)
	if planID != "" {
		plan, err := s.store.GetExperimentPlan(planID)
		if err != nil {
			writeStoreError(c, err)
			return
		}
		if plan.ProjectID != project.ID {
			writeStoreError(c, fmt.Errorf("%w: source plan does not belong to project", store.ErrInvalidRequest))
			return
		}
	} else if state.PlanID != "" {
		planID = state.PlanID
	} else if projectPlans, err := s.store.ListProjectExperimentPlans(project.ID); err == nil {
		if latestPlan, ok := latestExperimentPlan(projectPlans); ok {
			planID = latestPlan.ID
		}
	} else {
		writeStoreError(c, err)
		return
	}

	reason := strings.TrimSpace(req.Reason)
	if reason == "" {
		reason = "User explicitly reopened experimentation after champion selection."
	}

	payload := map[string]any{
		"explicit_user_action":     true,
		"new_exploration_round":    true,
		"reason":                   reason,
		"reopened_at":              time.Now().UTC().Format(time.RFC3339),
		"terminal_reason":          state.Reason,
		"terminal_at":              state.TerminalAt.Format(time.RFC3339),
		"previous_source_plan_id":  state.PlanID,
		"previous_champion_job_id": state.ChampionJobID,
	}
	if state.DecisionID != "" {
		payload["previous_decision_id"] = state.DecisionID
	}
	if state.ChampionID != "" {
		payload["previous_champion_id"] = state.ChampionID
	}

	decision, err := s.store.CreateAgentDecision(project.ID, planID, decisions.TypeReopenExperimentation, reason, payload)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	event, err := s.store.CreateExecutionEvent(project.ID, planID, execution.EventExperimentationReopened, "Experimentation reopened by explicit user action after champion selection.", map[string]any{
		"decision_id":              decision.ID,
		"explicit_user_action":     true,
		"new_exploration_round":    true,
		"reason":                   reason,
		"previous_source_plan_id":  state.PlanID,
		"previous_champion_job_id": state.ChampionJobID,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, reopenExperimentationResponse{
		Decision: decision,
		Event:    event,
	})
}

func (s *Server) createReviewerDecision(projectID string) (plans.ExperimentPlan, decisions.AgentDecision, error) {
	project, err := s.store.GetProject(projectID)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}

	projectPlans, err := s.store.ListProjectExperimentPlans(project.ID)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}

	summaries, err := s.store.ListProjectTrainingRunSummaries(project.ID)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}

	latestPlan, ok := latestExperimentPlan(projectPlans)
	if !ok {
		recommendation := agents.NewExperimentReviewer().Review(project, plans.ExperimentPlan{}, summaries)
		decision, err := s.store.CreateAgentDecision(
			project.ID,
			recommendation.PlanID,
			recommendation.DecisionType,
			recommendation.Rationale,
			recommendation.Payload,
		)
		if err != nil {
			return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
		}
		if err := s.persistProjectChampionFromDecision(project.ID, decision); err != nil {
			log.Printf("persist reviewer champion failed for project %s decision %s: %v", project.ID, decision.ID, err)
		}
		return plans.ExperimentPlan{}, decision, nil
	}

	recommendation, policyReference, err := s.reviewerRecommendationWithPolicy(project, latestPlan, summaries)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}
	decision, err := s.store.CreateAgentDecisionWithPolicy(
		project.ID,
		recommendation.PlanID,
		recommendation.DecisionType,
		recommendation.Rationale,
		recommendation.Payload,
		policyReference,
	)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}
	if err := s.persistProjectChampionFromDecision(project.ID, decision); err != nil {
		log.Printf("persist reviewer champion failed for project %s decision %s: %v", project.ID, decision.ID, err)
	}

	return latestPlan, decision, nil
}

func (s *Server) reviewerRecommendationWithPolicy(project projects.Project, plan plans.ExperimentPlan, summaries []runs.TrainingRunSummary) (decisions.AgentDecisionRecommendation, policies.PersistenceReference, error) {
	dataset, err := s.store.GetDataset(plan.DatasetID)
	if err != nil {
		return decisions.AgentDecisionRecommendation{}, policies.PersistenceReference{}, err
	}
	effectivePolicy, err := s.resolveProposalPolicy(project, dataset, policyOperationPropose)
	if err != nil {
		return decisions.AgentDecisionRecommendation{}, policies.PersistenceReference{}, err
	}
	recommendation, err := agents.NewExperimentReviewer().ReviewWithPolicy(project, plan, summaries, &effectivePolicy)
	if err != nil {
		return decisions.AgentDecisionRecommendation{}, policies.PersistenceReference{}, err
	}
	if recommendation.DecisionType != decisions.TypeAddExperiments {
		return recommendation, policies.PersistenceReference{}, nil
	}
	experiments, err := plannedExperimentsFromPayload(recommendation.Payload)
	if err != nil {
		return decisions.AgentDecisionRecommendation{}, policies.PersistenceReference{}, err
	}
	experiments, warnings, err := s.prepareAutoMLExperimentsForProjectWithPolicy(project.ID, experiments, &effectivePolicy)
	if err != nil {
		return decisions.AgentDecisionRecommendation{}, policies.PersistenceReference{}, err
	}
	recommendation.Payload["proposed_experiments"] = experiments
	if len(warnings) > 0 {
		recommendation.Payload["automl_warnings"] = warnings
	}
	evaluation, err := s.recordProposalPolicyEvaluation(effectivePolicy, policyOperationPersistProposal, experiments, "")
	if err != nil {
		return decisions.AgentDecisionRecommendation{}, policies.PersistenceReference{}, err
	}
	recommendation.Payload["proposal_policy_evaluation_id"] = evaluation.ID
	recommendation.Payload["effective_policy_hash"] = evaluation.EffectivePolicyHash
	recommendation.Payload["effective_policy_card"] = policies.PromptCardFromEffectivePolicy(effectivePolicy)
	return recommendation, policies.PersistenceReference{EvaluationID: evaluation.ID, EffectivePolicyHash: evaluation.EffectivePolicyHash, Status: policyStatusAllowed}, nil
}

func (s *Server) followUpSourceDecision(projectID string) (plans.ExperimentPlan, decisions.AgentDecision, error) {
	agentDecisions, err := s.store.ListProjectAgentDecisions(projectID)
	if err != nil {
		return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
	}

	if len(agentDecisions) > 0 && agentDecisions[0].DecisionType == decisions.TypeAddExperiments && agentDecisions[0].PlanID != "" {
		plan, err := s.store.GetExperimentPlan(agentDecisions[0].PlanID)
		if err == nil {
			return plan, agentDecisions[0], nil
		}
		if !errors.Is(err, store.ErrNotFound) {
			return plans.ExperimentPlan{}, decisions.AgentDecision{}, err
		}
	}

	return s.createReviewerDecision(projectID)
}

func (s *Server) ensureFollowUpPlan(projectID string, sourcePlan plans.ExperimentPlan, decision decisions.AgentDecision) (plans.ExperimentPlan, bool, error) {
	if decision.DecisionType != decisions.TypeAddExperiments {
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: reviewer decision is not ADD_EXPERIMENTS", store.ErrInvalidRequest)
	}
	if sourcePlan.ID == "" {
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: follow-up experiments require a source plan", store.ErrInvalidRequest)
	}
	if err := s.ensurePlannerCandidateProvenance(decision); err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	project, err := s.store.GetProject(projectID)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	dataset, err := s.store.GetDataset(sourcePlan.DatasetID)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	effectivePolicy, err := s.resolveProposalPolicy(project, dataset, policyOperationPropose)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}

	projectPlans, err := s.store.ListProjectExperimentPlans(projectID)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	if stopReason, stopDetails, ok, err := s.projectChampionSelectedFollowUpStopReason(projectID); err != nil {
		return plans.ExperimentPlan{}, false, err
	} else if ok {
		message := "Follow-up scheduling blocked because the project already has a selected champion."
		s.recordChampionSelectedFollowUpBlocked(projectID, sourcePlan.ID, decision.ID, "", message, stopReason, stopDetails)
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: %s", errChampionSelectedFollowUpBlocked, stopReason)
	}
	if existingPlan, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		if _, err := s.recordProposalPolicyEvaluation(effectivePolicy, policyOperationReusePlan, existingPlan.Experiments, ""); err != nil {
			return plans.ExperimentPlan{}, false, err
		}
		if err := s.validateExistingFollowUpPlanStillNovel(projectID, decision.ID, existingPlan, projectPlans); err != nil {
			return plans.ExperimentPlan{}, false, err
		}
		if _, err := s.finalizeCandidateOutcomesForPlan(existingPlan.ID); err != nil {
			return plans.ExperimentPlan{}, false, err
		}
		return existingPlan, false, nil
	}

	experiments, err := plannedExperimentsFromPayload(decision.Payload)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	relaxedValidationWarnings := []string{}
	baseExperiments := append([]plans.PlannedExperiment(nil), experiments...)
	experiments, err = plannedExperimentsWithStoredProposalMechanisms(decision.Payload, experiments)
	if err != nil {
		if plannervalidation.IsStrict(plannerValidationMode()) {
			return plans.ExperimentPlan{}, false, err
		}
		experiments = baseExperiments
		relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
	}
	var automlWarnings []string
	experiments, automlWarnings, err = s.prepareAutoMLExperimentsForProjectWithPolicy(projectID, experiments, &effectivePolicy)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	for index, experiment := range experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			return plans.ExperimentPlan{}, false, err
		}
	}
	if err := s.validateExperimentsDatasetCompatibility(sourcePlan.DatasetID, experiments); err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	var executionWarnings []string
	experiments, executionWarnings, err = s.normalizeFollowUpExperimentsForExecution(projectID, sourcePlan.ID, decision.ID, experiments)
	if err != nil {
		message := "Follow-up scheduling blocked because the proposal contains fields the selected runner cannot execute."
		s.recordFollowUpValidationBlocked(projectID, sourcePlan.ID, decision.ID, "", message, []string{err.Error()})
		return plans.ExperimentPlan{}, false, err
	}
	relaxedValidationWarnings = append(relaxedValidationWarnings, executionWarnings...)
	acceptedSpecVerdict, acceptedSpecErr := plannervalidation.Evaluate(plannerValidationMode(), []plannervalidation.Check{{
		Code:     "accepted_spec_no_op",
		Category: plannervalidation.CategoryProposalNoOp,
		Stage:    "follow_up_proposal",
		Validate: func() error {
			return s.validateFollowUpAcceptedSpecNovelty(projectID, experiments)
		},
	}})
	if acceptedSpecErr != nil {
		message := "Follow-up scheduling blocked because the proposal duplicates an accepted executable spec."
		s.recordFollowUpValidationBlocked(projectID, sourcePlan.ID, decision.ID, "", message, []string{acceptedSpecErr.Error()})
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: %s", errNoNovelFollowUpExperiments, acceptedSpecErr.Error())
	}
	if acceptedSpecVerdict.WouldBlock {
		relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarningText(acceptedSpecVerdict.Findings[0].Message))
		if err := s.persistDecisionShadowStrictVerdict(decision, acceptedSpecVerdict); err != nil {
			return plans.ExperimentPlan{}, false, err
		}
	}
	if err := validateLLMPlannerStoredMechanismContract(decision, experiments); err != nil && plannervalidation.IsStrict(plannerValidationMode()) {
		message := "Follow-up scheduling blocked because the stored planner decision lacks a valid mechanism contract."
		s.recordFollowUpValidationBlocked(projectID, sourcePlan.ID, decision.ID, "", message, []string{err.Error()})
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: %s", errNoNovelFollowUpExperiments, err.Error())
	} else if err != nil {
		relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
	}
	experiments, err = s.validateFollowUpExperimentMechanismsAgainstDataset(projectID, sourcePlan.DatasetID, sourcePlan.ID, decision.ID, "", experiments, payloadStringSlice(decision.Payload, "evidence_used"))
	if err != nil {
		if plannervalidation.IsStrict(plannerValidationMode()) {
			return plans.ExperimentPlan{}, false, err
		}
		relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
	}
	skippedExperiments := []string{}
	if plannervalidation.IsStrict(plannerValidationMode()) {
		var filtered []plans.PlannedExperiment
		filtered, skippedExperiments = filterNovelPlannedExperiments(experiments, projectPlans)
		experiments = filtered
		if len(experiments) == 0 {
			message := "Follow-up scheduling blocked because every proposed experiment duplicated an existing experiment or only changed minor tuning knobs."
			s.recordFollowUpValidationBlocked(projectID, sourcePlan.ID, decision.ID, "", message, skippedExperiments)
			return plans.ExperimentPlan{}, false, fmt.Errorf("%w: follow-up decision has no novel experiments after filtering duplicate or minor-only repeats", errNoNovelFollowUpExperiments)
		}
	} else {
		_, skippedExperiments = filterNovelPlannedExperiments(experiments, projectPlans)
		for _, warning := range skippedExperiments {
			warning = strings.Replace(warning, "Skipped follow-up experiment", "Allowed relaxed follow-up experiment", 1)
			relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarningText(warning))
		}
	}

	warnings := []string{
		fmt.Sprintf("Follow-up plan generated from reviewer decision %s.", decision.ID),
		fmt.Sprintf("Previous plan: %s.", sourcePlan.ID),
	}
	if plannervalidation.IsStrict(plannerValidationMode()) {
		warnings = append(warnings, skippedExperiments...)
	}
	warnings = append(warnings, automlWarnings...)
	warnings = append(warnings, uniqueStrings(relaxedValidationWarnings)...)

	evaluation, err := s.recordProposalPolicyEvaluation(effectivePolicy, policyOperationPersistPlan, experiments, "")
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	plan, err := s.store.CreateExperimentPlanWithPolicy(
		projectID,
		sourcePlan.DatasetID,
		sourcePlan.TargetMetric,
		recommendedWorkersForExperiments(experiments),
		estimateFollowUpMinutes(experiments),
		experiments,
		warnings,
		decision.ID,
		policies.PersistenceReference{EvaluationID: evaluation.ID, EffectivePolicyHash: evaluation.EffectivePolicyHash, Status: policyStatusAllowed},
	)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	if err := s.persistAutoMLForPlan(plan); err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	if _, err := s.createPendingStrategyScorecard(projectID, sourcePlan, decision, plan); err != nil {
		log.Printf("create pending strategy scorecard failed for decision %s: %v", decision.ID, err)
	}
	if _, err := s.finalizeCandidateOutcomesForPlan(plan.ID); err != nil {
		return plans.ExperimentPlan{}, false, err
	}

	return plan, true, nil
}

func (s *Server) normalizeFollowUpExperimentsForExecution(projectID string, sourcePlanID string, decisionID string, experiments []plans.PlannedExperiment) ([]plans.PlannedExperiment, []string, error) {
	provider := s.defaultExecuteExperimentPlanRequest().Provider
	if provider == "" {
		provider = "local"
	}
	provider = normalizeTrainingProvider(provider)
	out := append([]plans.PlannedExperiment(nil), experiments...)
	warnings := []string{}
	for index, experiment := range out {
		if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
			continue
		}
		normalized, experimentWarnings, err := normalizeFollowUpExperimentForExecution(provider, experiment, index)
		if err != nil {
			return experiments, warnings, err
		}
		out[index] = normalized
		warnings = append(warnings, experimentWarnings...)
	}
	if len(warnings) > 0 {
		if _, err := s.store.CreateExecutionEvent(projectID, sourcePlanID, execution.EventExecutionValidationReported, "Follow-up proposal was normalized to remove runner no-op fields before scheduling.", map[string]any{
			"source_decision_id": decisionID,
			"warnings":           warnings,
		}); err != nil {
			log.Printf("record follow-up execution normalization event failed: %v", err)
		}
	}
	return out, warnings, nil
}

func normalizeFollowUpExperimentForExecution(provider string, experiment plans.PlannedExperiment, index int) (plans.PlannedExperiment, []string, error) {
	warnings := []string{}
	for attempt := 0; attempt < 4; attempt++ {
		spec, err := buildExecutionSpecV1(experiment, provider)
		if err != nil {
			return experiment, warnings, err
		}
		modelSpec, _ := supportedModelSpecByName(experiment.Model)
		report, err := execution.ValidateExecutionSpecV1(spec, modelSpec.Family, execution.ValidationModeEnforce)
		if err != nil {
			return experiment, warnings, err
		}
		if !report.WouldBlock {
			return experiment, warnings, nil
		}
		fields := followUpExecutionNoOpFields(report)
		if len(fields) == 0 {
			return experiment, warnings, fmt.Errorf(
				"%w: experiment %d would be blocked by execution fidelity enforcement: %s",
				store.ErrInvalidRequest, index, executionValidationSummary(report),
			)
		}
		changed := false
		for _, field := range fields {
			normalized, removed, err := removeFollowUpExperimentConfigPath(experiment, field)
			if err != nil {
				return experiment, warnings, err
			}
			if !removed {
				continue
			}
			experiment = normalized
			changed = true
			warnings = append(warnings, fmt.Sprintf("Removed unsupported %s from follow-up experiment %d because the active runner cannot execute that field.", field, index))
		}
		if !changed {
			return experiment, warnings, fmt.Errorf(
				"%w: experiment %d would be blocked by execution fidelity enforcement and no removable no-op fields were found: %s",
				store.ErrInvalidRequest, index, executionValidationSummary(report),
			)
		}
	}
	return experiment, warnings, fmt.Errorf("%w: experiment %d still failed execution fidelity enforcement after follow-up normalization", store.ErrInvalidRequest, index)
}

func followUpExecutionNoOpFields(report execution.ExecutionValidationReport) []string {
	fields := []string{}
	for _, finding := range report.Findings {
		if !finding.WouldBlock || strings.TrimSpace(finding.Field) == "" {
			continue
		}
		switch finding.ReasonCode {
		case "runner_does_not_consume", "simulator_does_not_model":
			fields = append(fields, finding.Field)
		}
	}
	return uniqueStrings(fields)
}

func removeFollowUpExperimentConfigPath(experiment plans.PlannedExperiment, path string) (plans.PlannedExperiment, bool, error) {
	config, err := experiment.RequestedConfig()
	if err != nil {
		return experiment, false, err
	}
	if !deleteNestedConfigPath(config, strings.Split(path, ".")) {
		return experiment, false, nil
	}
	data, err := json.Marshal(config)
	if err != nil {
		return experiment, false, err
	}
	var normalized plans.PlannedExperiment
	if err := json.Unmarshal(data, &normalized); err != nil {
		return experiment, false, err
	}
	return normalized, true, nil
}

func deleteNestedConfigPath(root map[string]any, parts []string) bool {
	if len(parts) == 0 || strings.TrimSpace(parts[0]) == "" {
		return false
	}
	key := strings.TrimSpace(parts[0])
	if len(parts) == 1 {
		if _, ok := root[key]; !ok {
			return false
		}
		delete(root, key)
		return true
	}
	child, ok := root[key].(map[string]any)
	if !ok {
		return false
	}
	removed := deleteNestedConfigPath(child, parts[1:])
	if removed && len(child) == 0 {
		delete(root, key)
	}
	return removed
}

func (s *Server) validateFollowUpAcceptedSpecNovelty(projectID string, experiments []plans.PlannedExperiment) error {
	projectJobs, err := s.store.ListProjectJobs(projectID)
	if err != nil {
		return err
	}
	existing := map[string]string{}
	for _, job := range projectJobs {
		if hash := acceptedSpecHashFromJob(job); hash != "" {
			existing[hash] = job.ID
		}
	}
	provider := s.defaultExecuteExperimentPlanRequest().Provider
	proposed := map[string]int{}
	for index, experiment := range experiments {
		if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
			continue
		}
		spec, err := buildExecutionSpecV1(experiment, provider)
		if err != nil {
			return err
		}
		if jobID, ok := existing[spec.AcceptedSpecHash]; ok {
			return fmt.Errorf("%w: follow-up experiment %d is a proposal-time no-op matching accepted spec %s from job %s", store.ErrInvalidRequest, index, spec.AcceptedSpecHash, jobID)
		}
		if previous, ok := proposed[spec.AcceptedSpecHash]; ok {
			return fmt.Errorf("%w: follow-up experiment %d is a proposal-time no-op matching proposed experiment %d by accepted spec %s", store.ErrInvalidRequest, index, previous, spec.AcceptedSpecHash)
		}
		proposed[spec.AcceptedSpecHash] = index
	}
	return nil
}

func (s *Server) persistDecisionShadowStrictVerdict(decision decisions.AgentDecision, verdict plannervalidation.Verdict) error {
	invocationID := payloadString(decision.Payload, "invocation_id")
	if invocationID == "" || !plannervalidation.IsShadow(verdict.Mode) {
		return nil
	}
	invocation, err := s.store.GetAgentInvocation(invocationID)
	if err != nil {
		return fmt.Errorf("load planner invocation %s for shadow strict verdict: %w", invocationID, err)
	}
	if invocation.StrictValidationVerdict != nil {
		verdict = plannervalidation.Merge(*invocation.StrictValidationVerdict, verdict)
	}
	outcome := plannervalidation.Outcome{
		SchemaVersion:   plannervalidation.OutcomeSchemaVersionV1,
		Mode:            plannervalidation.ModeShadowStrict,
		FirstPassStatus: plannervalidation.FirstPassAccepted,
		EventualStatus:  plannervalidation.EventualAccepted,
		RetryOutcome:    plannervalidation.RetryNotNeeded,
	}
	if invocation.ValidationOutcome != nil {
		outcome = *invocation.ValidationOutcome
	}
	if _, err := s.store.UpdateAgentInvocationValidation(invocationID, verdict, outcome); err != nil {
		return fmt.Errorf("persist follow-up shadow strict verdict for invocation %s: %w", invocationID, err)
	}
	return nil
}

func (s *Server) validateExistingFollowUpPlanStillNovel(projectID string, decisionID string, followUpPlan plans.ExperimentPlan, projectPlans []plans.ExperimentPlan) error {
	priorPlans := make([]plans.ExperimentPlan, 0, len(projectPlans))
	for _, plan := range projectPlans {
		if plan.ID == followUpPlan.ID {
			continue
		}
		priorPlans = append(priorPlans, plan)
	}
	for index, experiment := range followUpPlan.Experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			message := fmt.Sprintf("Existing follow-up plan %s is blocked because experiment %d is no longer valid.", followUpPlan.ID, index)
			s.recordFollowUpValidationBlocked(projectID, followUpPlan.ID, decisionID, followUpPlan.ID, message, []string{err.Error()})
			return fmt.Errorf("%w: %s", errNoNovelFollowUpExperiments, err.Error())
		}
	}
	if err := s.validateExperimentsDatasetCompatibility(followUpPlan.DatasetID, followUpPlan.Experiments); err != nil {
		message := fmt.Sprintf("Existing follow-up plan %s is blocked because its experiments no longer match the dataset task.", followUpPlan.ID)
		s.recordFollowUpValidationBlocked(projectID, followUpPlan.ID, decisionID, followUpPlan.ID, message, []string{err.Error()})
		return fmt.Errorf("%w: %s", errNoNovelFollowUpExperiments, err.Error())
	}
	if !plannervalidation.IsStrict(plannerValidationMode()) {
		return nil
	}
	if _, err := s.validateFollowUpExperimentMechanismsAgainstDataset(projectID, followUpPlan.DatasetID, followUpPlan.ID, decisionID, followUpPlan.ID, followUpPlan.Experiments, nil); err != nil {
		return err
	}
	filtered, skippedExperiments := filterNovelPlannedExperiments(followUpPlan.Experiments, priorPlans)
	if len(skippedExperiments) == 0 && len(filtered) == len(followUpPlan.Experiments) {
		return nil
	}
	message := fmt.Sprintf("Existing follow-up plan %s is blocked because it no longer passes backend novelty validation.", followUpPlan.ID)
	s.recordFollowUpValidationBlocked(projectID, followUpPlan.ID, decisionID, followUpPlan.ID, message, skippedExperiments)
	return fmt.Errorf("%w: existing follow-up plan %s is no longer schedulable", errNoNovelFollowUpExperiments, followUpPlan.ID)
}

func (s *Server) validateFollowUpExperimentMechanismsAgainstDataset(
	projectID string,
	datasetID string,
	planID string,
	decisionID string,
	followUpPlanID string,
	experiments []plans.PlannedExperiment,
	planEvidence []string,
) ([]plans.PlannedExperiment, error) {
	if len(experiments) == 0 {
		return experiments, nil
	}
	dataset, err := s.store.GetDataset(datasetID)
	if err != nil {
		return experiments, err
	}
	enrichedExperiments, err := s.experimentsWithAcceptedVisualEvidence(dataset, experiments)
	if err != nil {
		return experiments, err
	}
	metadataSummary, err := s.activeAgentSafeDatasetMetadataSummary(dataset)
	if err != nil {
		return experiments, err
	}
	if err := validateMechanismDatasetEvidence(profileWithAgentSafeMetadataSummary(dataset.Profile, metadataSummary), enrichedExperiments, planEvidence); err != nil {
		message := "Follow-up scheduling blocked because one or more proposed mechanisms lack backend-verifiable diagnosis or dataset support."
		if followUpPlanID != "" {
			message = fmt.Sprintf("Existing follow-up plan %s is blocked because one or more mechanisms lack backend-verifiable diagnosis or dataset support.", followUpPlanID)
		}
		if plannervalidation.IsStrict(plannerValidationMode()) {
			s.recordFollowUpValidationBlocked(projectID, planID, decisionID, followUpPlanID, message, []string{err.Error()})
		}
		return enrichedExperiments, fmt.Errorf("%w: %s", errNoNovelFollowUpExperiments, err.Error())
	}
	return enrichedExperiments, nil
}

func (s *Server) recordFollowUpValidationBlocked(projectID string, planID string, decisionID string, followUpPlanID string, message string, skippedExperiments []string) {
	payload := map[string]any{
		"decision_id":               decisionID,
		"backend_validation_status": "blocked",
		"backend_validation_error":  "no novel follow-up experiments after filtering duplicate or minor-only repeats",
		"skipped_experiments":       skippedExperiments,
	}
	if followUpPlanID != "" {
		payload["follow_up_plan_id"] = followUpPlanID
	}
	if _, err := s.store.CreateExecutionEvent(projectID, planID, execution.EventAgentOutcomeRecorded, message, payload); err != nil {
		log.Printf("record follow-up validation block failed for project %s decision %s: %v", projectID, decisionID, err)
	}
}

func (s *Server) recordFollowUpSchedulingFailed(projectID string, planID string, decisionID string, err error) {
	if err == nil {
		return
	}
	if _, eventErr := s.store.CreateExecutionEvent(projectID, planID, execution.EventAgentOutcomeRecorded, fmt.Sprintf("Planner follow-up scheduling failed for plan %s.", planID), map[string]any{
		"decision_id":               decisionID,
		"backend_validation_status": "failed",
		"backend_validation_error":  err.Error(),
	}); eventErr != nil {
		log.Printf("record follow-up scheduling failure failed for project %s decision %s: %v", projectID, decisionID, eventErr)
	}
}

func (s *Server) recordChampionSelectedFollowUpBlocked(projectID string, planID string, decisionID string, followUpPlanID string, message string, reason string, details map[string]any) {
	payload := map[string]any{
		"decision_id":               decisionID,
		"backend_validation_status": "blocked",
		"backend_validation_error":  "champion selected guard",
		"backend_stop_guard":        "champion_selected_guard",
		"reason":                    reason,
	}
	for key, value := range details {
		payload[key] = value
	}
	if followUpPlanID != "" {
		payload["follow_up_plan_id"] = followUpPlanID
	}
	if _, err := s.store.CreateExecutionEvent(projectID, planID, execution.EventAgentOutcomeRecorded, message, payload); err != nil {
		log.Printf("record champion-selected follow-up block failed for project %s decision %s: %v", projectID, decisionID, err)
	}
}

type projectChampionSelectionState struct {
	Terminal      bool
	TerminalAt    time.Time
	Reason        string
	PlanID        string
	DecisionID    string
	ChampionID    string
	ChampionJobID string
	Reopened      bool
	ReopenID      string
	ReopenAt      time.Time
}

func (s *Server) projectChampionSelectionState(projectID string) (projectChampionSelectionState, error) {
	state := projectChampionSelectionState{}

	champion, err := s.store.GetProjectChampion(projectID)
	if err == nil {
		state.Terminal = true
		state.TerminalAt = champion.UpdatedAt
		if state.TerminalAt.IsZero() {
			state.TerminalAt = champion.CreatedAt
		}
		state.Reason = fmt.Sprintf(
			"Project already has selected champion %s; follow-up scheduling requires an explicit reopen or new exploration round.",
			champion.JobID,
		)
		state.PlanID = champion.PlanID
		state.DecisionID = champion.SourceDecisionID
		state.ChampionID = champion.ID
		state.ChampionJobID = champion.JobID
	} else if !errors.Is(err, store.ErrNotFound) {
		return state, err
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(projectID)
	if err != nil {
		return state, err
	}
	for _, decision := range agentDecisions {
		decisionType := strings.ToUpper(strings.TrimSpace(decision.DecisionType))
		switch decisionType {
		case decisions.TypeSelectChampion:
			if !state.Terminal || decision.CreatedAt.After(state.TerminalAt) {
				state.Terminal = true
				state.TerminalAt = decision.CreatedAt
				state.Reason = fmt.Sprintf(
					"Project already has SELECT_CHAMPION decision %s; follow-up scheduling requires an explicit reopen or new exploration round.",
					decision.ID,
				)
				state.PlanID = decision.PlanID
				state.DecisionID = decision.ID
				state.ChampionID = ""
				state.ChampionJobID = payloadString(decision.Payload, "champion_job_id")
			}
		case decisions.TypeReopenExperimentation:
			if state.ReopenID == "" || decision.CreatedAt.After(state.ReopenAt) {
				state.ReopenID = decision.ID
				state.ReopenAt = decision.CreatedAt
			}
		}
	}
	if state.Terminal && state.ReopenID != "" && !state.ReopenAt.Before(state.TerminalAt) {
		state.Reopened = true
	}

	return state, nil
}

func (s *Server) projectChampionSelectedFollowUpStopReason(projectID string) (string, map[string]any, bool, error) {
	state, err := s.projectChampionSelectionState(projectID)
	if err != nil {
		return "", nil, false, err
	}
	if !state.Terminal || state.Reopened {
		return "", nil, false, nil
	}
	details := map[string]any{
		"source_decision_id": state.DecisionID,
		"champion_id":        state.ChampionID,
		"champion_job_id":    state.ChampionJobID,
		"source_plan_id":     state.PlanID,
		"terminal_at":        state.TerminalAt.Format(time.RFC3339),
		"reopen_required":    true,
	}
	if state.ReopenID != "" {
		details["latest_reopen_decision_id"] = state.ReopenID
		details["latest_reopen_at"] = state.ReopenAt.Format(time.RFC3339)
	}
	return state.Reason, details, true, nil
}

func (s *Server) createPendingStrategyScorecard(projectID string, sourcePlan plans.ExperimentPlan, decision decisions.AgentDecision, followUpPlan plans.ExperimentPlan) (strategies.StrategyScorecard, error) {
	datasetProfile := map[string]any{}
	computedTraits := []string{}
	if dataset, err := s.store.GetDataset(sourcePlan.DatasetID); err == nil {
		datasetProfile = dataset.Profile
		computedTraits = datasetProfileTraits(dataset.Profile)
	}
	datasetTraits := map[string]any{
		"dataset_id":                sourcePlan.DatasetID,
		"profile":                   datasetProfile,
		"computed_traits":           computedTraits,
		"dataset_planning_insights": decision.Payload["dataset_planning_insights"],
		"deterministic_diagnosis":   decision.Payload["deterministic_diagnosis"],
	}
	objectiveProfile := payloadMap(decision.Payload, "objective_context")
	proposedChanges := map[string]any{
		"hypothesis":            decision.Payload["hypothesis"],
		"changed_variables":     decision.Payload["changed_variables"],
		"proposed_experiments":  decision.Payload["proposed_experiments"],
		"proposal_mechanisms":   decision.Payload["proposal_mechanisms"],
		"candidate_hypotheses":  decision.Payload["candidate_hypotheses"],
		"candidate_rankings":    decision.Payload["candidate_rankings"],
		"rejected_options":      decision.Payload["rejected_options"],
		"success_criteria":      decision.Payload["success_criteria"],
		"deployment_tradeoff":   decision.Payload["deployment_tradeoff"],
		"why_can_beat_champion": decision.Payload["why_can_beat_champion"],
	}
	planningMode := payloadString(decision.Payload, "planning_mode")
	strategyType := planningMode
	if strategyType == "" {
		strategyType = "planner_followup"
	}
	mechanism, intervention, diagnosisTriggers, evidenceUsed, expectedEffect := strategyScorecardMechanismFields(decision.Payload)
	return s.store.CreateStrategyScorecard(strategies.StrategyScorecardCreate{
		ProjectID:         projectID,
		DatasetID:         sourcePlan.DatasetID,
		SourceDecisionID:  decision.ID,
		SourcePlanID:      sourcePlan.ID,
		FollowUpPlanID:    followUpPlan.ID,
		StrategyType:      strategyType,
		PlanningMode:      planningMode,
		Mechanism:         mechanism,
		Intervention:      intervention,
		DiagnosisTriggers: diagnosisTriggers,
		EvidenceUsed:      evidenceUsed,
		ExpectedEffect:    expectedEffect,
		DatasetTraits:     datasetTraits,
		ObjectiveProfile:  objectiveProfile,
		ProposedChanges:   proposedChanges,
		ExpectedDelta:     payloadFloat(decision.Payload, "expected_delta_vs_champion"),
		ConfidenceBefore:  payloadFloat(decision.Payload, "confidence"),
		Outcome:           strategies.OutcomePending,
		Lesson:            "Pending follow-up outcome.",
		Tags:              uniqueStrings([]string{"strategy_scorecard", planningMode, mechanism, strategies.OutcomePending}),
	})
}

func strategyScorecardMechanismFields(payload map[string]any) (string, string, []string, []string, string) {
	diagnosisTriggers := payloadStringSlice(payload, "deterministic_diagnosis_used")
	if len(diagnosisTriggers) == 0 {
		diagnosisTriggers = payloadStringSlice(payload, "diagnosis_triggers")
	}
	evidenceUsed := payloadStringSlice(payload, "evidence_used")
	mechanism := payloadString(payload, "mechanism")
	intervention := payloadString(payload, "intervention")
	expectedEffect := payloadString(payload, "expected_effect")

	if proposals, ok, err := plannerProposalMechanismsFromPayload(payload); err == nil && ok {
		for _, proposal := range proposals {
			if mechanism == "" {
				mechanism = proposal.Mechanism
			}
			if intervention == "" {
				intervention = proposal.Intervention
			}
			if len(evidenceUsed) == 0 {
				evidenceUsed = proposal.EvidenceUsed
			}
			if expectedEffect == "" {
				expectedEffect = proposal.ExpectedEffect
			}
			if mechanism != "" && intervention != "" && len(evidenceUsed) > 0 && expectedEffect != "" {
				break
			}
		}
	}
	if mechanism == "" || intervention == "" || len(evidenceUsed) == 0 || expectedEffect == "" {
		if experiments, err := plannedExperimentsFromPayloadLenient(payload); err == nil {
			for _, experiment := range experiments {
				if mechanism == "" {
					mechanism = experiment.Mechanism
				}
				if intervention == "" {
					intervention = experiment.Intervention
				}
				if len(evidenceUsed) == 0 {
					evidenceUsed = experiment.EvidenceUsed
				}
				if expectedEffect == "" {
					expectedEffect = experiment.ExpectedEffect
				}
				if mechanism != "" && intervention != "" && len(evidenceUsed) > 0 && expectedEffect != "" {
					break
				}
			}
		}
	}
	return strings.TrimSpace(mechanism), strings.TrimSpace(intervention), uniqueStrings(diagnosisTriggers), uniqueStrings(evidenceUsed), strings.TrimSpace(expectedEffect)
}

func (s *Server) runAutomaticExperimentReview(projectID string) (automaticExperimentReviewResult, error) {
	if !s.shouldAutoReviewExperimentJobs() {
		return automaticExperimentReviewResult{}, nil
	}

	s.autoReviewMu.Lock()
	defer s.autoReviewMu.Unlock()

	project, err := s.store.GetProject(projectID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	projectPlans, err := s.store.ListProjectExperimentPlans(project.ID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	latestPlan, ok := latestExperimentPlan(projectPlans)
	if !ok {
		return automaticExperimentReviewResult{}, nil
	}

	summaries, err := s.store.ListProjectTrainingRunSummaries(project.ID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	recommendation, policyReference, err := s.reviewerRecommendationWithPolicy(project, latestPlan, summaries)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}
	if recommendation.DecisionType == decisions.TypeWait {
		return automaticExperimentReviewResult{}, nil
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(project.ID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	decision, ok := actionDecisionForPlan(agentDecisions, latestPlan.ID)
	if !ok {
		if stopReason, stopDetails, selected, err := s.projectChampionSelectedFollowUpStopReason(project.ID); err != nil {
			return automaticExperimentReviewResult{}, err
		} else if selected {
			message := fmt.Sprintf("Automatic experiment review skipped for plan %s because the project already has a selected champion.", latestPlan.ID)
			s.recordChampionSelectedFollowUpBlocked(project.ID, latestPlan.ID, "", "", message, stopReason, stopDetails)
			return automaticExperimentReviewResult{}, nil
		}
		decision, err = s.store.CreateAgentDecisionWithPolicy(
			project.ID,
			recommendation.PlanID,
			recommendation.DecisionType,
			recommendation.Rationale,
			recommendation.Payload,
			policyReference,
		)
		if err != nil {
			return automaticExperimentReviewResult{}, err
		}
	}
	if err := s.persistProjectChampionFromDecision(project.ID, decision); err != nil {
		log.Printf("persist automatic reviewer champion failed for project %s decision %s: %v", project.ID, decision.ID, err)
	}

	result := automaticExperimentReviewResult{
		Decision: &decision,
	}

	if decision.DecisionType != decisions.TypeAddExperiments {
		return result, nil
	}
	if !s.shouldAutoScheduleFollowUps() {
		return result, nil
	}

	if _, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		followUpPlan, _, err := s.ensureFollowUpPlan(project.ID, latestPlan, decision)
		if err != nil {
			if errors.Is(err, errNoNovelFollowUpExperiments) {
				return result, nil
			}
			return automaticExperimentReviewResult{}, err
		}
		result.FollowUpPlan = &followUpPlan
		return s.executeAutomaticFollowUpPlan(result)
	}

	maxRounds := s.maxAutoFollowUpRounds()
	if followUpRoundCount(projectPlans) >= maxRounds {
		log.Printf(
			"automatic follow-up scheduling skipped for project %s plan %s: max follow-up rounds reached (%d)",
			project.ID,
			latestPlan.ID,
			maxRounds,
		)
		if err := s.selectBestAvailableChampionAfterMaxFollowUpStop(latestPlan, decision, projectPlans, maxRounds); err != nil {
			return automaticExperimentReviewResult{}, err
		}
		return result, nil
	}

	followUpPlan, _, err := s.ensureFollowUpPlan(project.ID, latestPlan, decision)
	if err != nil {
		if errors.Is(err, errNoNovelFollowUpExperiments) {
			return result, nil
		}
		return automaticExperimentReviewResult{}, err
	}

	result.FollowUpPlan = &followUpPlan
	return s.executeAutomaticFollowUpPlan(result)
}

func (s *Server) executeAutomaticFollowUpPlan(result automaticExperimentReviewResult) (automaticExperimentReviewResult, error) {
	if result.FollowUpPlan == nil || !s.shouldAutoExecuteExperimentPlans() {
		return result, nil
	}

	req := s.defaultExecuteExperimentPlanRequest()
	req.deferPlanAggregate = true
	planExecution, err := s.executeStoredExperimentPlan(result.FollowUpPlan.ID, req)
	if err != nil {
		if errors.Is(err, errNoNovelFollowUpExperiments) {
			return result, nil
		}
		if _, eventErr := s.store.CreateExecutionEvent(
			result.FollowUpPlan.ProjectID,
			result.FollowUpPlan.ID,
			execution.EventExecutionFailed,
			fmt.Sprintf("Automatic execution failed for plan %s.", result.FollowUpPlan.ID),
			map[string]any{"error": err.Error()},
		); eventErr != nil {
			log.Printf("record automatic execution failure event failed: %v", eventErr)
		}
		return automaticExperimentReviewResult{}, err
	}

	result.Jobs = planExecution.Jobs
	if complete, err := s.finalizeCandidateOutcomesForPlan(result.FollowUpPlan.ID); err != nil {
		return automaticExperimentReviewResult{}, err
	} else if complete {
		if err := s.recordExperimentPlannerOutcomeForPlanLocked(*result.FollowUpPlan); err != nil {
			return automaticExperimentReviewResult{}, err
		}
	}
	if err := s.recordAutomaticExecutionQueued(*result.FollowUpPlan, req, planExecution.Jobs); err != nil {
		return automaticExperimentReviewResult{}, err
	}
	return result, nil
}

func (s *Server) recordAutomaticExecutionQueued(plan plans.ExperimentPlan, req executeExperimentPlanRequest, queuedJobs []jobs.ExperimentJob) error {
	openJobCount := openTrainingJobCount(queuedJobs)
	if openJobCount == 0 {
		_, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventJobsQueued, fmt.Sprintf("Plan %s has no open automatic experiment jobs requiring workers.", plan.ID), map[string]any{
			"job_ids":        experimentJobIDs(queuedJobs),
			"open_job_count": 0,
		})
		return err
	}
	provider := req.Provider
	if provider == "" {
		provider = "local"
	}
	provider = normalizeTrainingProvider(provider)
	if err := validateTrainingProviderConfigured(provider); err != nil {
		return err
	}
	targetCount := s.targetWorkerCountForPlan(plan, openJobCount)
	activeWorkerCount := s.activeOrStartingWorkersForProject(plan.ProjectID, provider, req.GPUType)
	requirementPolicy, err := s.workerRequirementPolicyForPlan(plan, provider, targetCount)
	if err != nil {
		return err
	}
	if policy := costPolicyForSettings(s.currentAutomationSettings()); policy.Enabled && policy.MaxConcurrentJobs > 0 && requirementPolicy.MaxConcurrentJobs > policy.MaxConcurrentJobs {
		requirementPolicy.MaxConcurrentJobs = policy.MaxConcurrentJobs
	}

	requirement, created, err := s.store.UpsertWorkerRequirement(
		plan.ProjectID,
		plan.ID,
		provider,
		req.GPUType,
		targetCount,
		"auto_followup",
		requirementPolicy,
	)
	if err != nil {
		return err
	}

	requirementStatus := requirement.Status
	if activeWorkerCount >= requirement.TargetCount {
		active := execution.WorkerRequirementActive
		updated, updateErr := s.store.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{Status: &active})
		if updateErr != nil {
			return updateErr
		}
		requirement = updated
		requirementStatus = updated.Status
	} else if requirement.Status == execution.WorkerRequirementActive {
		pending := execution.WorkerRequirementPending
		updated, updateErr := s.store.UpdateWorkerRequirement(requirement.ID, execution.WorkerRequirementUpdate{Status: &pending})
		if updateErr != nil {
			return updateErr
		}
		requirement = updated
		requirementStatus = updated.Status
	}

	if _, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventJobsQueued, fmt.Sprintf("Queued %d automatic experiment job(s) for plan %s.", len(queuedJobs), plan.ID), map[string]any{
		"job_ids":                           experimentJobIDs(queuedJobs),
		"open_job_count":                    openJobCount,
		"active_worker_count":               activeWorkerCount,
		"worker_requirement_id":             requirement.ID,
		"target_count":                      requirement.TargetCount,
		"requirement_status":                requirementStatus,
		"provider":                          provider,
		"gpu_type":                          req.GPUType,
		"dataset_id":                        requirement.DatasetID,
		"dataset_checksum":                  requirement.DatasetChecksum,
		"dataset_cache_key":                 requirement.DatasetCacheKey,
		"materialization_status":            requirement.DatasetMaterializationStatus,
		"cold_cache_policy":                 requirement.ColdCachePolicy,
		"max_concurrent_jobs":               requirement.MaxConcurrentJobs,
		"max_cold_dataset_materializations": requirement.MaxColdDatasetMaterializations,
		"cost_policy":                       costPolicyForSettings(s.currentAutomationSettings()).Payload(),
	}); err != nil {
		return err
	}

	eventType := execution.EventWorkersRequired
	if !created {
		eventType = execution.EventWorkerScalingUpdated
	}
	message := fmt.Sprintf(
		"Automatic execution targets %d worker(s) for %d open job(s); %d worker(s) are already active or starting.",
		requirement.TargetCount,
		openJobCount,
		activeWorkerCount,
	)
	_, err = s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, eventType, message, map[string]any{
		"worker_requirement_id":             requirement.ID,
		"target_count":                      requirement.TargetCount,
		"open_job_count":                    openJobCount,
		"active_worker_count":               activeWorkerCount,
		"provider":                          requirement.Provider,
		"gpu_type":                          requirement.GPUType,
		"dataset_id":                        requirement.DatasetID,
		"dataset_checksum":                  requirement.DatasetChecksum,
		"dataset_cache_key":                 requirement.DatasetCacheKey,
		"materialization_status":            requirement.DatasetMaterializationStatus,
		"cold_cache_policy":                 requirement.ColdCachePolicy,
		"max_concurrent_jobs":               requirement.MaxConcurrentJobs,
		"max_cold_dataset_materializations": requirement.MaxColdDatasetMaterializations,
		"cost_policy":                       costPolicyForSettings(s.currentAutomationSettings()).Payload(),
	})
	return err
}

func actionDecisionForPlan(agentDecisions []decisions.AgentDecision, planID string) (decisions.AgentDecision, bool) {
	for _, decision := range agentDecisions {
		if decision.PlanID == planID && decision.DecisionType != decisions.TypeWait && decisionAutoExecutable(decision) {
			return decision, true
		}
	}

	return decisions.AgentDecision{}, false
}

func decisionAutoExecutable(decision decisions.AgentDecision) bool {
	value, ok := decision.Payload["auto_executable"].(bool)
	if ok && !value {
		return false
	}
	return true
}

func followUpRoundCount(projectPlans []plans.ExperimentPlan) int {
	count := 0
	for _, plan := range projectPlans {
		if plan.SourceDecisionID != "" {
			count++
		}
	}

	return count
}

func (s *Server) runAutomaticExperimentReviewAfterTrainingJob(job jobs.ExperimentJob) {
	if _, err := s.runAutomaticExperimentReview(job.ProjectID); err != nil {
		log.Printf("automatic experiment review failed after training job %s: %v", job.ID, err)
	}
}

func (s *Server) enqueueTrainingTerminalHooks(job jobs.ExperimentJob) {
	if job.ID == "" || job.Template != jobs.TemplateTrainExperiment {
		return
	}
	if !s.markTrainingTerminalHooksQueued(job.ID) {
		log.Printf("post-training hooks already queued for job %s", job.ID)
		return
	}
	go s.runTrainingTerminalHooks(job.ID)
}

func (s *Server) markTrainingTerminalHooksQueued(jobID string) bool {
	s.trainingTerminalHooksMu.Lock()
	defer s.trainingTerminalHooksMu.Unlock()

	if s.trainingTerminalHooksQueued == nil {
		s.trainingTerminalHooksQueued = make(map[string]bool)
	}
	if s.trainingTerminalHooksQueued[jobID] {
		return false
	}
	s.trainingTerminalHooksQueued[jobID] = true
	return true
}

func (s *Server) runTrainingTerminalHooks(jobID string) {
	job, err := s.store.GetJob(jobID)
	if err != nil {
		log.Printf("post-training hooks skipped for job %s: %v", jobID, err)
		return
	}
	if job.Template != jobs.TemplateTrainExperiment {
		return
	}
	if job.Status != jobs.StatusSucceeded && job.Status != jobs.StatusFailed {
		log.Printf("post-training hooks skipped for job %s: non-terminal status %s", job.ID, job.Status)
		return
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			message := fmt.Sprintf("post-training hooks panic for job %s: %v", job.ID, recovered)
			log.Print(message)
			s.recordTrainingTerminalHookEvent(job, "failed", message, fmt.Sprint(recovered))
		}
	}()

	s.recordTrainingTerminalHookEvent(job, "started", fmt.Sprintf("Post-training agent hooks started for job %s.", job.ID), "")
	if planID := jobConfigString(job.Config, "plan_id"); planID != "" {
		if _, err := s.finalizeCandidateOutcomesForPlan(planID); err != nil {
			log.Printf("candidate outcome finalization failed for plan %s after job %s: %v", planID, job.ID, err)
		}
	}
	if err := s.observeAutoMLTrialForJob(job); err != nil {
		log.Printf("AutoML trial observation failed for job %s: %v", job.ID, err)
	}
	s.runTrainingMonitorAfterTrainingJob(job)
	planningStatus, planningErr := s.runPlanningLoopAfterTrainingJob(job)
	if planningStatus == "" {
		planningStatus = "finished"
	}
	if planningErr != nil {
		log.Printf("post-training planner hook finished with %s status for job %s: %v", planningStatus, job.ID, planningErr)
		s.recordTrainingTerminalHookEvent(job, planningStatus, fmt.Sprintf("Post-training agent hooks finished with %s planner status for job %s.", planningStatus, job.ID), planningErr.Error())
		return
	}
	s.recordTrainingTerminalHookEvent(job, planningStatus, fmt.Sprintf("Post-training agent hooks finished for job %s.", job.ID), "")
}

func (s *Server) recordTrainingTerminalHookEvent(job jobs.ExperimentJob, status string, message string, errorText string) {
	planID := jobConfigString(job.Config, "plan_id")
	payload := map[string]any{
		"job_id":                       job.ID,
		"job_status":                   job.Status,
		"post_training_hooks_status":   status,
		"post_training_hooks_async":    true,
		"training_monitor_after_job":   true,
		"experiment_planner_after_job": true,
	}
	if errorText != "" {
		payload["error"] = errorText
	}
	if _, err := s.store.CreateExecutionEvent(job.ProjectID, planID, execution.EventAgentOutcomeRecorded, message, payload); err != nil {
		log.Printf("record post-training hook event failed for job %s: %v", job.ID, err)
	}
}

func (s *Server) runPlanningLoopAfterTrainingJob(job jobs.ExperimentJob) (string, error) {
	if err := s.recordExperimentPlannerOutcomeAfterTrainingJob(job); err != nil {
		log.Printf("record experiment planner outcome failed after training job %s: %v", job.ID, err)
	}
	if selected, err := s.selectBestAvailableChampionIfCostStoppedAfterTrainingJob(job); err != nil {
		log.Printf("budget-stop champion selection failed after training job %s: %v", job.ID, err)
	} else if selected {
		return "finished", nil
	}

	handled, err := s.runExperimentPlannerAfterTrainingJob(job)
	if err != nil {
		log.Printf("llm experiment planner failed after training job %s: %v", job.ID, err)
		outcome, fallbackErr := s.runDegradedPlannerFallbackAfterTrainingJob(job, err)
		if fallbackErr != nil {
			return "failed", fallbackErr
		}
		if outcome == degradedPlannerFallbackSelected || outcome == degradedPlannerFallbackDeferred {
			return "degraded", nil
		}
		return "failed", err
	}
	if handled {
		return "finished", nil
	}
	if result, err := s.runAutomaticExperimentReview(job.ProjectID); err != nil {
		log.Printf("automatic experiment review failed after training job %s: %v", job.ID, err)
		return "failed", err
	} else if result.Decision != nil || result.FollowUpPlan != nil || len(result.Jobs) > 0 {
		return "finished", nil
	}
	terminalOutcome, err := s.selectBestAvailableChampionAfterTerminalTrainingJob(job, "terminal_training_hook")
	if err != nil {
		return "failed", err
	}
	if terminalOutcome == terminalChampionFallbackSelected || terminalOutcome == terminalChampionFallbackDeferred {
		return "finished", nil
	}
	return "failed", fmt.Errorf("no successful training run is available to select as champion")
}

type degradedPlannerFallbackOutcome string

const (
	degradedPlannerFallbackNone     degradedPlannerFallbackOutcome = "none"
	degradedPlannerFallbackSelected degradedPlannerFallbackOutcome = "selected"
	degradedPlannerFallbackDeferred degradedPlannerFallbackOutcome = "deferred"
)

func (s *Server) runDegradedPlannerFallbackAfterTrainingJob(job jobs.ExperimentJob, plannerErr error) (degradedPlannerFallbackOutcome, error) {
	if _, err := s.store.GetProjectChampion(job.ProjectID); err == nil {
		return degradedPlannerFallbackSelected, nil
	} else if !errors.Is(err, store.ErrNotFound) {
		return degradedPlannerFallbackNone, err
	}

	plan, ok, err := s.trainingJobExperimentPlan(job)
	if err != nil || !ok {
		return degradedPlannerFallbackNone, err
	}
	openJobs, err := s.hasOpenTrainingJobForPlan(plan.ProjectID, plan.ID)
	if err != nil {
		return degradedPlannerFallbackNone, err
	}
	if openJobs {
		return degradedPlannerFallbackDeferred, nil
	}
	selected, err := s.selectBestAvailableChampionForTerminalPlanStop(plan, terminalChampionSelectionOptions{
		DecisionSource: llmPlannerDegradedChampionDecisionSource,
		Trigger:        "llm_planner_failure",
		EventType:      execution.EventAgentOutcomeRecorded,
		Rationale: fmt.Sprintf(
			"Experiment Planner failed after training job %s, so the backend selected the best successful model available for plan %s.",
			job.ID,
			plan.ID,
		),
		EventMessage: fmt.Sprintf(
			"Experiment Planner failed after training job %s; deterministic fallback selected the best available champion.",
			job.ID,
		),
		NoChampionMessage: fmt.Sprintf(
			"Experiment Planner failed after training job %s, but no successful model is available to select as champion.",
			job.ID,
		),
		Payload: map[string]any{
			"job_id":        job.ID,
			"planner_error": plannerErr.Error(),
		},
	})
	if err != nil {
		return degradedPlannerFallbackNone, err
	}
	if selected {
		return degradedPlannerFallbackSelected, nil
	}
	return degradedPlannerFallbackNone, nil
}

type terminalChampionFallbackOutcome string

const (
	terminalChampionFallbackNone     terminalChampionFallbackOutcome = "none"
	terminalChampionFallbackSelected terminalChampionFallbackOutcome = "selected"
	terminalChampionFallbackDeferred terminalChampionFallbackOutcome = "deferred"
)

func (s *Server) selectBestAvailableChampionAfterTerminalTrainingJob(job jobs.ExperimentJob, trigger string) (terminalChampionFallbackOutcome, error) {
	plan, ok, err := s.trainingJobExperimentPlan(job)
	if err != nil || !ok {
		if err != nil {
			return terminalChampionFallbackNone, err
		}
		return terminalChampionFallbackDeferred, nil
	}
	openJobs, err := s.hasOpenTrainingJobForPlan(plan.ProjectID, plan.ID)
	if err != nil {
		return terminalChampionFallbackNone, err
	}
	if openJobs {
		return terminalChampionFallbackDeferred, nil
	}
	selected, err := s.selectBestAvailableChampionForTerminalPlanStop(plan, terminalChampionSelectionOptions{
		DecisionSource: terminalTrainingChampionDecisionSource,
		Trigger:        trigger,
		EventType:      execution.EventAgentOutcomeRecorded,
		Rationale: fmt.Sprintf(
			"Training jobs for plan %s reached a terminal state, so the backend selected the best successful model available.",
			plan.ID,
		),
		EventMessage: fmt.Sprintf(
			"Terminal training hooks selected the best available champion for plan %s.",
			plan.ID,
		),
		NoChampionMessage: fmt.Sprintf(
			"Training jobs for plan %s reached a terminal state, but no successful model is available to select as champion.",
			plan.ID,
		),
		Payload: map[string]any{
			"job_id": job.ID,
		},
	})
	if err != nil {
		return terminalChampionFallbackNone, err
	}
	if selected {
		return terminalChampionFallbackSelected, nil
	}
	return terminalChampionFallbackNone, nil
}

func (s *Server) trainingJobExperimentPlan(job jobs.ExperimentJob) (plans.ExperimentPlan, bool, error) {
	planID := jobConfigString(job.Config, "plan_id")
	if summary, err := s.store.GetTrainingRunSummary(job.ID); err == nil && summary.PlanID != "" {
		planID = summary.PlanID
	} else if err != nil && !errors.Is(err, store.ErrNotFound) {
		return plans.ExperimentPlan{}, false, err
	}
	if planID == "" {
		return plans.ExperimentPlan{}, false, nil
	}
	plan, err := s.store.GetExperimentPlan(planID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return plans.ExperimentPlan{}, false, nil
		}
		return plans.ExperimentPlan{}, false, err
	}
	return plan, true, nil
}
