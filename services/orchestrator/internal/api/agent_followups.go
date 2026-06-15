package api

import (
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
	"model-express/services/orchestrator/internal/plans"
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
		return nil
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
	if !terminalPlannerGuardsEnabledForMode(s.currentAutomationSettings().AgentMode) {
		return "", "", false, nil
	}
	if stopReason, _, ok, err := s.projectChampionSelectedFollowUpStopReason(projectID); err != nil {
		return "", "", false, err
	} else if ok {
		return stopReason, "champion_selected_guard", true, nil
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

	recommendation := agents.NewExperimentReviewer().Review(project, latestPlan, summaries)
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

	return latestPlan, decision, nil
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

	projectPlans, err := s.store.ListProjectExperimentPlans(projectID)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	if terminalPlannerGuardsEnabledForMode(s.currentAutomationSettings().AgentMode) {
		if stopReason, stopDetails, ok, err := s.projectChampionSelectedFollowUpStopReason(projectID); err != nil {
			return plans.ExperimentPlan{}, false, err
		} else if ok {
			message := "Follow-up scheduling blocked because the project already has a selected champion."
			s.recordChampionSelectedFollowUpBlocked(projectID, sourcePlan.ID, decision.ID, "", message, stopReason, stopDetails)
			return plans.ExperimentPlan{}, false, fmt.Errorf("%w: %s", errChampionSelectedFollowUpBlocked, stopReason)
		}
	}
	if existingPlan, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		if err := s.validateExistingFollowUpPlanStillNovel(projectID, decision.ID, existingPlan, projectPlans); err != nil {
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
		if plannerStrictValidationEnabled() {
			return plans.ExperimentPlan{}, false, err
		}
		experiments = baseExperiments
		relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
	}
	var automlWarnings []string
	experiments, automlWarnings, err = s.prepareAutoMLExperimentsForProject(projectID, experiments)
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
	if err := validateLLMPlannerStoredMechanismContract(decision, experiments); err != nil && plannerStrictValidationEnabled() {
		message := "Follow-up scheduling blocked because the stored planner decision lacks a valid mechanism contract."
		s.recordFollowUpValidationBlocked(projectID, sourcePlan.ID, decision.ID, "", message, []string{err.Error()})
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: %s", errNoNovelFollowUpExperiments, err.Error())
	} else if err != nil {
		relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
	}
	experiments, err = s.validateFollowUpExperimentMechanismsAgainstDataset(projectID, sourcePlan.DatasetID, sourcePlan.ID, decision.ID, "", experiments, payloadStringSlice(decision.Payload, "evidence_used"))
	if err != nil {
		if plannerStrictValidationEnabled() {
			return plans.ExperimentPlan{}, false, err
		}
		relaxedValidationWarnings = append(relaxedValidationWarnings, plannerRelaxedValidationWarning(err))
	}
	skippedExperiments := []string{}
	if plannerStrictValidationEnabled() {
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
	if plannerStrictValidationEnabled() {
		warnings = append(warnings, skippedExperiments...)
	}
	warnings = append(warnings, automlWarnings...)
	warnings = append(warnings, uniqueStrings(relaxedValidationWarnings)...)

	plan, err := s.store.CreateExperimentPlan(
		projectID,
		sourcePlan.DatasetID,
		sourcePlan.TargetMetric,
		recommendedWorkersForExperiments(experiments),
		estimateFollowUpMinutes(experiments),
		experiments,
		warnings,
		decision.ID,
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

	return plan, true, nil
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
	if !plannerStrictValidationEnabled() {
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
		if plannerStrictValidationEnabled() {
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
			"Project already has selected champion %s; autonomous follow-up scheduling requires an explicit reopen or new exploration round.",
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
					"Project already has SELECT_CHAMPION decision %s; autonomous follow-up scheduling requires an explicit reopen or new exploration round.",
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

	recommendation := agents.NewExperimentReviewer().Review(project, latestPlan, summaries)
	if recommendation.DecisionType == decisions.TypeWait {
		return automaticExperimentReviewResult{}, nil
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(project.ID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	decision, ok := actionDecisionForPlan(agentDecisions, latestPlan.ID)
	if !ok {
		if terminalPlannerGuardsEnabledForMode(s.currentAutomationSettings().AgentMode) {
			if stopReason, stopDetails, selected, err := s.projectChampionSelectedFollowUpStopReason(project.ID); err != nil {
				return automaticExperimentReviewResult{}, err
			} else if selected {
				message := fmt.Sprintf("Automatic experiment review skipped for plan %s because the project already has a selected champion.", latestPlan.ID)
				s.recordChampionSelectedFollowUpBlocked(project.ID, latestPlan.ID, "", "", message, stopReason, stopDetails)
				return automaticExperimentReviewResult{}, nil
			}
		}
		decision, err = s.store.CreateAgentDecision(
			project.ID,
			recommendation.PlanID,
			recommendation.DecisionType,
			recommendation.Rationale,
			recommendation.Payload,
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
	if err := s.observeAutoMLTrialForJob(job); err != nil {
		log.Printf("AutoML trial observation failed for job %s: %v", job.ID, err)
	}
	s.runTrainingMonitorAfterTrainingJob(job)
	s.runPlanningLoopAfterTrainingJob(job)
	s.recordTrainingTerminalHookEvent(job, "finished", fmt.Sprintf("Post-training agent hooks finished for job %s.", job.ID), "")
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

func (s *Server) runPlanningLoopAfterTrainingJob(job jobs.ExperimentJob) {
	if err := s.recordExperimentPlannerOutcomeAfterTrainingJob(job); err != nil {
		log.Printf("record experiment planner outcome failed after training job %s: %v", job.ID, err)
	}
	if selected, err := s.selectBestAvailableChampionIfCostStoppedAfterTrainingJob(job); err != nil {
		log.Printf("budget-stop champion selection failed after training job %s: %v", job.ID, err)
	} else if selected {
		return
	}

	handled, err := s.runExperimentPlannerAfterTrainingJob(job)
	if err != nil {
		log.Printf("llm experiment planner failed after training job %s: %v", job.ID, err)
		return
	}
	if handled {
		return
	}
	s.runAutomaticExperimentReviewAfterTrainingJob(job)
}
