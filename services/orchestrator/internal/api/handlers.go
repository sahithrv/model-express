package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
)

type createProjectRequest struct {
	Name string `json:"name" binding:"required"`
	Goal string `json:"goal"`
}

type registerWorkerRequest struct {
	ProjectID string `json:"project_id" binding:"required"`
	Name      string `json:"name" binding:"required"`
	GPUType   string `json:"gpu_type"`
}

type createJobRequest struct {
	Template string         `json:"template" binding:"required"`
	Config   map[string]any `json:"config"`
}

type createDatasetRequest struct {
	Name           string `json:"name" binding:"required"`
	StorageURI     string `json:"storage_uri" binding:"required"`
	ChecksumSHA256 string `json:"checksum_sha256"`
	SizeBytes      int64  `json:"size_bytes"`
}

type createExperimentPlanRequest struct {
	DatasetID          string                    `json:"dataset_id" binding:"required"`
	TargetMetric       string                    `json:"target_metric"`
	Priority           string                    `json:"priority"`
	MaxWorkers         int                       `json:"max_workers"`
	TimeBudgetMinutes  int                       `json:"time_budget_minutes"`
	RecommendedWorkers int                       `json:"recommended_workers"`
	EstimatedMinutes   int                       `json:"estimated_minutes"`
	Experiments        []plans.PlannedExperiment `json:"experiments"`
	Warnings           []string                  `json:"warnings"`
}

type executeExperimentPlanRequest struct {
	Provider string `json:"provider"`
	GPUType  string `json:"gpu_type"`
}

type executeExperimentPlanResponse struct {
	Plan plans.ExperimentPlan `json:"plan"`
	Jobs []jobs.ExperimentJob `json:"jobs"`
}

type scheduleFollowUpExperimentsResponse struct {
	Decision     decisions.AgentDecision `json:"decision"`
	FollowUpPlan *plans.ExperimentPlan   `json:"follow_up_plan,omitempty"`
}

type automaticExperimentReviewResult struct {
	Decision     *decisions.AgentDecision
	FollowUpPlan *plans.ExperimentPlan
	Jobs         []jobs.ExperimentJob
}

const (
	llmTrainingMonitorDecisionSource = "llm_training_monitor"
	minLLMDecisionConfidence         = 0.50
)

type updateDatasetProfileRequest struct {
	Profile map[string]any `json:"profile" binding:"required"`
}

type reportMetricRequest struct {
	Epoch   int                `json:"epoch" binding:"required"`
	Metrics map[string]float64 `json:"metrics" binding:"required"`
}

type reportTrainingRunSummaryRequest = runs.TrainingRunSummaryUpdate

type completeJobRequest struct {
	MLflowRunID string `json:"mlflow_run_id"`
}

type failJobRequest struct {
	Error string `json:"error" binding:"required"`
}

type pollJobResponse struct {
	Job *jobs.ExperimentJob `json:"job"`
}

func (s *Server) health(c *gin.Context) {
	c.JSON(http.StatusOK, healthResponse{
		Status:    "ok",
		Service:   "orchestrator",
		Timestamp: time.Now().UTC(),
	})
}

func (s *Server) getAutomationSettings(c *gin.Context) {
	c.JSON(http.StatusOK, s.currentAutomationSettings())
}

func (s *Server) updateAutomationSettings(c *gin.Context) {
	var req settings.AutomationSettingsUpdate
	if !bindJSON(c, &req) {
		return
	}

	current := s.currentAutomationSettings()
	updated, err := applyAutomationSettingsUpdate(current, req)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	saved, err := s.store.SaveAutomationSettings(updated)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	s.setAutomationSettings(saved)
	c.JSON(http.StatusOK, saved)
}

func (s *Server) createProject(c *gin.Context) {
	var req createProjectRequest
	if !bindJSON(c, &req) {
		return
	}

	project, err := s.store.CreateProject(req.Name, req.Goal)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, project)
}

func (s *Server) listProjects(c *gin.Context) {
	projects, err := s.store.ListProjects()
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"projects": projects})
}

func (s *Server) getProject(c *gin.Context) {
	project, err := s.store.GetProject(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, project)
}

func (s *Server) createDataset(c *gin.Context) {
	var req createDatasetRequest
	if !bindJSON(c, &req) {
		return
	}

	dataset, err := s.store.CreateDataset(
		c.Param("id"),
		req.Name,
		req.StorageURI,
		req.ChecksumSHA256,
		req.SizeBytes,
	)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if _, err := s.store.CreateJob(dataset.ProjectID, jobs.TemplateProfileDataset, map[string]any{
		"dataset_id": dataset.ID,
	}); err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, dataset)
}

func (s *Server) listProjectDatasets(c *gin.Context) {
	datasets, err := s.store.ListProjectDatasets(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"datasets": datasets})
}

func (s *Server) getDataset(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, dataset)
}

func (s *Server) updateDatasetProfile(c *gin.Context) {
	var req updateDatasetProfileRequest
	if !bindJSON(c, &req) {
		return
	}

	dataset, err := s.store.UpdateDatasetProfile(c.Param("id"), req.Profile)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if err := s.createInitialPlanForDataset(dataset.ID); err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, dataset)
}

func (s *Server) createJob(c *gin.Context) {
	var req createJobRequest
	if !bindJSON(c, &req) {
		return
	}

	if req.Config == nil {
		req.Config = map[string]any{}
	}

	job, err := s.store.CreateJob(c.Param("id"), req.Template, req.Config)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, job)
}

func (s *Server) listProjectJobs(c *gin.Context) {
	jobs, err := s.store.ListProjectJobs(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"jobs": jobs})
}

func (s *Server) getJob(c *gin.Context) {
	job, err := s.store.GetJob(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) reportMetric(c *gin.Context) {
	var req reportMetricRequest
	if !bindJSON(c, &req) {
		return
	}

	metric, err := s.store.ReportMetric(c.Param("id"), req.Epoch, req.Metrics)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, metric)
}

func (s *Server) listJobMetrics(c *gin.Context) {
	metrics, err := s.store.ListJobMetrics(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"metrics": metrics})
}

func (s *Server) upsertTrainingRunSummary(c *gin.Context) {
	var req reportTrainingRunSummaryRequest
	if !bindJSON(c, &req) {
		return
	}

	summary, err := s.store.UpsertTrainingRunSummary(c.Param("id"), req)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, summary)
}

func (s *Server) getTrainingRunSummary(c *gin.Context) {
	summary, err := s.store.GetTrainingRunSummary(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, summary)
}

func (s *Server) listProjectTrainingRunSummaries(c *gin.Context) {
	summaries, err := s.store.ListProjectTrainingRunSummaries(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"summaries": summaries})
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
	if existingPlan, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		return existingPlan, false, nil
	}

	experiments, err := plannedExperimentsFromPayload(decision.Payload)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}

	warnings := []string{
		fmt.Sprintf("Follow-up plan generated from reviewer decision %s.", decision.ID),
		fmt.Sprintf("Previous plan: %s.", sourcePlan.ID),
	}

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

	return plan, true, nil
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

	result := automaticExperimentReviewResult{
		Decision: &decision,
	}

	if decision.DecisionType != decisions.TypeAddExperiments {
		return result, nil
	}
	if !s.shouldAutoScheduleFollowUps() {
		return result, nil
	}

	if existingPlan, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		result.FollowUpPlan = &existingPlan
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
	targetCount := plan.RecommendedWorkers
	if targetCount < 1 {
		targetCount = 1
	}
	if len(plan.Experiments) > 0 && targetCount > len(plan.Experiments) {
		targetCount = len(plan.Experiments)
	}

	provider := req.Provider
	if provider == "" {
		provider = "local"
	}

	requirement, created, err := s.store.UpsertWorkerRequirement(
		plan.ProjectID,
		plan.ID,
		provider,
		req.GPUType,
		targetCount,
		"auto_followup",
	)
	if err != nil {
		return err
	}

	if _, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventJobsQueued, fmt.Sprintf("Queued %d automatic experiment job(s) for plan %s.", len(queuedJobs), plan.ID), map[string]any{
		"job_ids":               experimentJobIDs(queuedJobs),
		"worker_requirement_id": requirement.ID,
		"provider":              provider,
		"gpu_type":              req.GPUType,
	}); err != nil {
		return err
	}

	if created {
		_, err = s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventWorkersRequired, fmt.Sprintf("Automatic execution requires %d worker(s) for plan %s.", requirement.TargetCount, plan.ID), map[string]any{
			"worker_requirement_id": requirement.ID,
			"target_count":          requirement.TargetCount,
			"provider":              requirement.Provider,
			"gpu_type":              requirement.GPUType,
		})
	}
	return err
}

func experimentJobIDs(experimentJobs []jobs.ExperimentJob) []string {
	out := make([]string, 0, len(experimentJobs))
	for _, job := range experimentJobs {
		out = append(out, job.ID)
	}
	return out
}

func mapFromStruct(value any) (map[string]any, error) {
	blob, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	out := map[string]any{}
	if err := json.Unmarshal(blob, &out); err != nil {
		return nil, err
	}
	return out, nil
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

func (s *Server) recordWorkerRequirementStatusEvent(requirement execution.WorkerRequirement) {
	eventType := ""
	message := ""
	switch requirement.Status {
	case execution.WorkerRequirementStarting:
		eventType = execution.EventWorkersStarting
		message = fmt.Sprintf("Starting %d worker(s) for plan %s.", requirement.TargetCount, requirement.PlanID)
	case execution.WorkerRequirementActive:
		eventType = execution.EventWorkersActive
		message = fmt.Sprintf("%d worker(s) are active for plan %s.", requirement.TargetCount, requirement.PlanID)
	case execution.WorkerRequirementFailed:
		eventType = execution.EventExecutionFailed
		message = fmt.Sprintf("Worker startup failed for plan %s.", requirement.PlanID)
	default:
		return
	}

	if _, err := s.store.CreateExecutionEvent(requirement.ProjectID, requirement.PlanID, eventType, message, map[string]any{
		"worker_requirement_id": requirement.ID,
		"status":                requirement.Status,
		"target_count":          requirement.TargetCount,
		"provider":              requirement.Provider,
		"gpu_type":              requirement.GPUType,
		"last_error":            requirement.LastError,
	}); err != nil {
		log.Printf("record worker requirement event failed: %v", err)
	}
}

func validWorkerRequirementStatus(status string) bool {
	switch strings.ToUpper(strings.TrimSpace(status)) {
	case execution.WorkerRequirementPending,
		execution.WorkerRequirementStarting,
		execution.WorkerRequirementActive,
		execution.WorkerRequirementFailed,
		execution.WorkerRequirementCancelled:
		return true
	default:
		return false
	}
}

func (s *Server) listProjectAgentDecisions(c *gin.Context) {
	agentDecisions, err := s.store.ListProjectAgentDecisions(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"decisions": agentDecisions})
}

func (s *Server) listProjectAgentMemoryRecords(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "25"))
	records, err := s.store.ListProjectAgentMemoryRecords(c.Param("id"), memory.AgentMemoryFilter{
		DatasetID: c.Query("dataset_id"),
		PlanID:    c.Query("plan_id"),
		JobID:     c.Query("job_id"),
		Kind:      c.Query("kind"),
		Limit:     limit,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"records": records})
}

func (s *Server) listProjectAgentInvocations(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "25"))
	invocations, err := s.store.ListProjectAgentInvocations(c.Param("id"), memory.AgentInvocationFilter{
		DatasetID: c.Query("dataset_id"),
		PlanID:    c.Query("plan_id"),
		JobID:     c.Query("job_id"),
		AgentName: c.Query("agent_name"),
		Limit:     limit,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"invocations": invocations})
}

func (s *Server) listProjectWorkerRequirements(c *gin.Context) {
	requirements, err := s.store.ListProjectWorkerRequirements(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"requirements": requirements})
}

func (s *Server) updateWorkerRequirement(c *gin.Context) {
	var req execution.WorkerRequirementUpdate
	if !bindJSON(c, &req) {
		return
	}
	if req.Status != nil {
		normalizedStatus := strings.ToUpper(strings.TrimSpace(*req.Status))
		if !validWorkerRequirementStatus(normalizedStatus) {
			writeStoreError(c, fmt.Errorf("%w: invalid worker requirement status", store.ErrInvalidRequest))
			return
		}
		req.Status = &normalizedStatus
	}

	requirement, err := s.store.UpdateWorkerRequirement(c.Param("id"), req)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if req.Status != nil {
		s.recordWorkerRequirementStatusEvent(requirement)
	}

	c.JSON(http.StatusOK, requirement)
}

func (s *Server) listProjectExecutionEvents(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "50"))
	events, err := s.store.ListProjectExecutionEvents(c.Param("id"), limit)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"events": events})
}

func (s *Server) completeJob(c *gin.Context) {
	var req completeJobRequest
	if !bindJSON(c, &req) {
		return
	}

	job, err := s.store.CompleteJob(c.Param("id"), req.MLflowRunID)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if job.Template == jobs.TemplateTrainExperiment {
		if _, err := s.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
			Status: jobs.StatusSucceeded,
		}); err != nil {
			writeStoreError(c, err)
			return
		}
		s.runTrainingMonitorAfterTrainingJob(job)
		s.runAutomaticExperimentReviewAfterTrainingJob(job)
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) failJob(c *gin.Context) {
	var req failJobRequest
	if !bindJSON(c, &req) {
		return
	}

	job, err := s.store.FailJob(c.Param("id"), req.Error)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if job.Template == jobs.TemplateTrainExperiment {
		if _, err := s.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
			Status: jobs.StatusFailed,
		}); err != nil {
			writeStoreError(c, err)
			return
		}
		s.runTrainingMonitorAfterTrainingJob(job)
		s.runAutomaticExperimentReviewAfterTrainingJob(job)
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) runAutomaticExperimentReviewAfterTrainingJob(job jobs.ExperimentJob) {
	if _, err := s.runAutomaticExperimentReview(job.ProjectID); err != nil {
		log.Printf("automatic experiment review failed after training job %s: %v", job.ID, err)
	}
}

func (s *Server) runTrainingMonitorAfterTrainingJob(job jobs.ExperimentJob) {
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
	client := llm.NewOpenAICompatibleClient(config)
	agent := agents.NewTrainingMonitorAgent(client, config.Model)

	ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
	defer cancel()

	trace, err := agent.EvaluateWithTrace(ctx, agents.TrainingMonitorInput{
		Plan:          plan,
		Job:           job,
		Summary:       summary,
		Metrics:       metrics,
		MemoryRecords: priorMemory,
	})
	acceptedForMemory := err == nil
	invocation, invocationErr := s.recordTrainingMonitorInvocation(job, summary, config, trace, acceptedForMemory)
	if invocationErr != nil {
		log.Printf("training monitor invocation write failed for job %s: %v", job.ID, invocationErr)
	}
	if err != nil {
		log.Printf("training monitor failed for job %s: %v", job.ID, err)
		if _, eventErr := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventExecutionFailed, fmt.Sprintf("Training Monitor failed for job %s.", job.ID), map[string]any{
			"job_id":        job.ID,
			"invocation_id": invocation.ID,
			"error":         err.Error(),
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

	if _, err := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventAgentRecommendationRecorded, fmt.Sprintf("Training Monitor recorded an evaluation for job %s.", job.ID), map[string]any{
		"job_id":           job.ID,
		"invocation_id":    invocation.ID,
		"memory_record_id": record.ID,
		"agent_name":       record.AgentName,
		"kind":             record.Kind,
	}); err != nil {
		log.Printf("record training monitor event failed: %v", err)
	}

	if _, err := s.handleTrainingMonitorAgentDecision(job, summary, trace, invocation, record); err != nil {
		log.Printf("training monitor decision handling failed for job %s: %v", job.ID, err)
		if _, eventErr := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventExecutionFailed, fmt.Sprintf("Training Monitor decision was not scheduled for job %s.", job.ID), map[string]any{
			"job_id":           job.ID,
			"invocation_id":    invocation.ID,
			"memory_record_id": record.ID,
			"error":            err.Error(),
		}); eventErr != nil {
			log.Printf("record training monitor decision failure event failed: %v", eventErr)
		}
	}
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
		InputContext:      trace.PromptContext,
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

func (s *Server) handleTrainingMonitorAgentDecision(
	job jobs.ExperimentJob,
	summary runs.TrainingRunSummary,
	trace agents.TrainingMonitorEvaluationTrace,
	invocation memory.AgentInvocation,
	record memory.AgentMemoryRecord,
) (automaticExperimentReviewResult, error) {
	if !s.shouldAutoReviewExperimentJobs() {
		return automaticExperimentReviewResult{}, nil
	}
	if trace.ValidationStatus != memory.InvocationValidationValid {
		return automaticExperimentReviewResult{}, nil
	}

	recommendation := trace.Recommendation
	action := recommendation.RecommendedAction
	decisionType, ok := llmActionDecisionType(action.ActionType)
	if !ok {
		return automaticExperimentReviewResult{}, nil
	}
	if action.Confidence < minLLMDecisionConfidence {
		log.Printf(
			"training monitor decision skipped for job %s: confidence %.2f below %.2f",
			job.ID,
			action.Confidence,
			minLLMDecisionConfidence,
		)
		return automaticExperimentReviewResult{}, nil
	}

	plan, ready, err := s.sourcePlanReadyForAgentDecision(job.ProjectID, summary.PlanID)
	if err != nil || !ready {
		return automaticExperimentReviewResult{}, err
	}

	payload, err := trainingMonitorDecisionPayload(action, invocation, record, s.currentAutomationSettings().AgentMode)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	agentDecisions, err := s.store.ListProjectAgentDecisions(job.ProjectID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}
	decision, ok := trainingMonitorDecisionForPlan(agentDecisions, plan.ID)
	if !ok {
		decision, err = s.store.CreateAgentDecision(
			job.ProjectID,
			plan.ID,
			decisionType,
			action.Rationale,
			payload,
		)
		if err != nil {
			return automaticExperimentReviewResult{}, err
		}
	}

	result := automaticExperimentReviewResult{Decision: &decision}
	if decision.DecisionType != decisions.TypeAddExperiments {
		return result, nil
	}
	if !s.shouldAutoScheduleFollowUps() || s.currentAutomationSettings().AgentMode != llm.AgentModeAutonomous {
		return result, nil
	}

	projectPlans, err := s.store.ListProjectExperimentPlans(job.ProjectID)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}
	if existingPlan, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		result.FollowUpPlan = &existingPlan
		return s.executeAutomaticFollowUpPlan(result)
	}

	maxRounds := s.maxAutoFollowUpRounds()
	if followUpRoundCount(projectPlans) >= maxRounds {
		log.Printf(
			"llm follow-up scheduling skipped for project %s plan %s: max follow-up rounds reached (%d)",
			job.ProjectID,
			plan.ID,
			maxRounds,
		)
		return result, nil
	}

	followUpPlan, _, err := s.ensureFollowUpPlan(job.ProjectID, plan, decision)
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	result.FollowUpPlan = &followUpPlan
	return s.executeAutomaticFollowUpPlan(result)
}

func llmActionDecisionType(actionType string) (string, bool) {
	switch strings.ToUpper(strings.TrimSpace(actionType)) {
	case decisions.TypeAddExperiments:
		return decisions.TypeAddExperiments, true
	case decisions.TypeStopProject:
		return decisions.TypeStopProject, true
	default:
		return "", false
	}
}

func trainingMonitorDecisionPayload(
	action decisions.AgentActionProposal,
	invocation memory.AgentInvocation,
	record memory.AgentMemoryRecord,
	agentMode string,
) (map[string]any, error) {
	payload := map[string]any{}
	for key, value := range action.Payload {
		payload[key] = value
	}

	if strings.EqualFold(action.ActionType, decisions.TypeAddExperiments) {
		experiments, err := plannedExperimentsFromPayload(payload)
		if err != nil {
			return nil, err
		}
		payload["proposed_experiments"] = experiments
	}

	payload["decision_source"] = llmTrainingMonitorDecisionSource
	payload["agent_name"] = agents.TrainingMonitorAgentName
	payload["invocation_id"] = invocation.ID
	payload["memory_record_id"] = record.ID
	payload["confidence"] = action.Confidence
	payload["llm_action_type"] = action.ActionType
	payload["llm_requires_approval"] = action.RequiresApproval
	payload["auto_executable"] = agentMode == llm.AgentModeAutonomous
	return payload, nil
}

func (s *Server) sourcePlanReadyForAgentDecision(projectID string, planID string) (plans.ExperimentPlan, bool, error) {
	if planID == "" {
		return plans.ExperimentPlan{}, false, nil
	}

	plan, err := s.store.GetExperimentPlan(planID)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}
	if plan.ProjectID != projectID {
		return plans.ExperimentPlan{}, false, fmt.Errorf("%w: plan does not belong to project", store.ErrInvalidRequest)
	}

	summaries, err := s.store.ListProjectTrainingRunSummaries(projectID)
	if err != nil {
		return plans.ExperimentPlan{}, false, err
	}

	return plan, planTrainingRunsComplete(plan, summaries), nil
}

func planTrainingRunsComplete(plan plans.ExperimentPlan, summaries []runs.TrainingRunSummary) bool {
	if plan.ID == "" || len(plan.Experiments) == 0 {
		return false
	}

	planSummaries := []runs.TrainingRunSummary{}
	for _, summary := range summaries {
		if summary.PlanID == plan.ID {
			planSummaries = append(planSummaries, summary)
		}
	}
	if len(planSummaries) < len(plan.Experiments) {
		return false
	}

	for _, summary := range planSummaries {
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

func trainingMonitorDecisionForPlan(agentDecisions []decisions.AgentDecision, planID string) (decisions.AgentDecision, bool) {
	for _, decision := range agentDecisions {
		if decision.PlanID != planID {
			continue
		}
		if decision.Payload["decision_source"] != llmTrainingMonitorDecisionSource {
			continue
		}
		return decision, true
	}
	return decisions.AgentDecision{}, false
}

func (s *Server) listWorkers(c *gin.Context) {
	workers, err := s.store.ListWorkers()
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"workers": workers})
}

func (s *Server) listProjectWorkers(c *gin.Context) {
	workers, err := s.store.ListProjectWorkers(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"workers": workers})
}

func (s *Server) getWorker(c *gin.Context) {
	worker, err := s.store.GetWorker(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, worker)
}

func (s *Server) registerWorker(c *gin.Context) {
	var req registerWorkerRequest
	if !bindJSON(c, &req) {
		return
	}

	worker, err := s.store.RegisterWorker(req.ProjectID, req.Name, req.GPUType)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, worker)
}

func (s *Server) heartbeatWorker(c *gin.Context) {
	worker, err := s.store.HeartbeatWorker(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, worker)
}

func (s *Server) pollJob(c *gin.Context) {
	job, err := s.store.PollJob(c.Param("id"))
	if err == nil {
		c.JSON(http.StatusOK, pollJobResponse{Job: job})
		return
	}

	if errors.Is(err, store.ErrNoJob) {
		c.JSON(http.StatusOK, pollJobResponse{Job: nil})
		return
	}

	writeStoreError(c, err)
}

func (s *Server) createExperimentPlan(c *gin.Context) {
	var req createExperimentPlanRequest
	if !bindJSON(c, &req) {
		return
	}

	targetMetric := req.TargetMetric
	recommendedWorkers := req.RecommendedWorkers
	estimatedMinutes := req.EstimatedMinutes
	experiments := req.Experiments
	warnings := req.Warnings

	if len(experiments) == 0 {
		project, err := s.store.GetProject(c.Param("id"))
		if err != nil {
			writeStoreError(c, err)
			return
		}

		dataset, err := s.store.GetDataset(req.DatasetID)
		if err != nil {
			writeStoreError(c, err)
			return
		}

		recommendation, err := agents.NewDatasetPlanner().BuildExperimentPlan(project, dataset, agents.PlanPreferences{
			Priority:          req.Priority,
			MaxWorkers:        req.MaxWorkers,
			TimeBudgetMinutes: req.TimeBudgetMinutes,
			TargetMetric:      req.TargetMetric,
		})
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		targetMetric = recommendation.TargetMetric
		recommendedWorkers = recommendation.RecommendedWorkers
		estimatedMinutes = recommendation.EstimatedMinutes
		experiments = recommendation.Experiments
		warnings = append(warnings, recommendation.Warnings...)
	}

	plan, err := s.store.CreateExperimentPlan(
		c.Param("id"),
		req.DatasetID,
		targetMetric,
		recommendedWorkers,
		estimatedMinutes,
		experiments,
		warnings,
		"",
	)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, plan)
}

func (s *Server) listProjectPlans(c *gin.Context) {
	plans, err := s.store.ListProjectExperimentPlans(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"plans": plans})
}

func (s *Server) listExperimentPlans(c *gin.Context) {
	plans, err := s.store.GetExperimentPlan(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, plans)
}

func (s *Server) executeExperimentPlan(c *gin.Context) {
	var req executeExperimentPlanRequest
	if !bindJSON(c, &req) {
		return
	}

	response, err := s.executeStoredExperimentPlan(c.Param("id"), req)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, response)
}

func (s *Server) createInitialPlanForDataset(datasetID string) error {
	dataset, err := s.store.GetDataset(datasetID)
	if err != nil {
		return err
	}

	existingPlans, err := s.store.ListProjectExperimentPlans(dataset.ProjectID)
	if err != nil {
		return err
	}
	for _, plan := range existingPlans {
		if plan.DatasetID == dataset.ID {
			return nil
		}
	}

	project, err := s.store.GetProject(dataset.ProjectID)
	if err != nil {
		return err
	}

	recommendation, err := agents.NewDatasetPlanner().BuildExperimentPlan(project, dataset, agents.PlanPreferences{
		Priority: agents.PriorityBalanced,
	})
	if err != nil {
		return fmt.Errorf("%w: %s", store.ErrInvalidRequest, err.Error())
	}

	plan, err := s.store.CreateExperimentPlan(
		project.ID,
		dataset.ID,
		recommendation.TargetMetric,
		recommendation.RecommendedWorkers,
		recommendation.EstimatedMinutes,
		recommendation.Experiments,
		recommendation.Warnings,
		"",
	)
	if err != nil {
		return err
	}

	if s.shouldAutoExecuteExperimentPlans() {
		_, err = s.executeStoredExperimentPlan(plan.ID, s.defaultExecuteExperimentPlanRequest())
		return err
	}

	return nil
}

func (s *Server) executeStoredExperimentPlan(planID string, req executeExperimentPlanRequest) (executeExperimentPlanResponse, error) {
	plan, err := s.store.GetExperimentPlan(planID)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}

	if len(plan.Experiments) == 0 {
		return executeExperimentPlanResponse{}, fmt.Errorf("%w: plan has no experiments to execute", store.ErrInvalidRequest)
	}

	provider := req.Provider
	if provider == "" {
		provider = "local"
	}

	existingJobs, err := s.store.ListProjectJobs(plan.ProjectID)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}

	jobsByExperiment := map[int]jobs.ExperimentJob{}
	for _, job := range existingJobs {
		if job.Template != jobs.TemplateTrainExperiment {
			continue
		}
		if job.Status == jobs.StatusFailed {
			continue
		}
		if configString(job.Config, "plan_id") != plan.ID {
			continue
		}
		jobProvider := configString(job.Config, "provider")
		if jobProvider == "" {
			jobProvider = "local"
		}
		if jobProvider != provider {
			continue
		}

		index, ok := configInt(job.Config, "experiment_index")
		if !ok {
			continue
		}
		jobsByExperiment[index] = job
	}

	out := make([]jobs.ExperimentJob, 0, len(plan.Experiments))
	for index, experiment := range plan.Experiments {
		if job, ok := jobsByExperiment[index]; ok {
			out = append(out, job)
			continue
		}

		config := map[string]any{
			"plan_id":             plan.ID,
			"dataset_id":          plan.DatasetID,
			"experiment_index":    index,
			"experiment_template": experiment.Template,
			"model":               experiment.Model,
			"epochs":              experiment.Epochs,
			"batch_size":          experiment.BatchSize,
			"learning_rate":       experiment.LearningRate,
			"target_metric":       plan.TargetMetric,
			"provider":            provider,
			"gpu_type":            req.GPUType,
		}

		job, err := s.store.CreateJob(plan.ProjectID, jobs.TemplateTrainExperiment, config)
		if err != nil {
			return executeExperimentPlanResponse{}, err
		}

		out = append(out, job)
	}

	return executeExperimentPlanResponse{
		Plan: plan,
		Jobs: out,
	}, nil
}

func configString(config map[string]any, key string) string {
	value, _ := config[key].(string)
	return value
}

func configInt(config map[string]any, key string) (int, bool) {
	switch value := config[key].(type) {
	case int:
		return value, true
	case int64:
		return int(value), true
	case float64:
		return int(value), true
	default:
		return 0, false
	}
}

func latestExperimentPlan(projectPlans []plans.ExperimentPlan) (plans.ExperimentPlan, bool) {
	if len(projectPlans) == 0 {
		return plans.ExperimentPlan{}, false
	}

	sort.Slice(projectPlans, func(i, j int) bool {
		return projectPlans[i].CreatedAt.After(projectPlans[j].CreatedAt)
	})

	return projectPlans[0], true
}

func followUpPlanForDecision(projectPlans []plans.ExperimentPlan, decisionID string) (plans.ExperimentPlan, bool) {
	for _, plan := range projectPlans {
		if plan.SourceDecisionID == decisionID {
			return plan, true
		}
	}

	return plans.ExperimentPlan{}, false
}

func plannedExperimentsFromPayload(payload map[string]any) ([]plans.PlannedExperiment, error) {
	value, ok := payload["proposed_experiments"]
	if !ok {
		return nil, fmt.Errorf("%w: reviewer decision does not include proposed_experiments", store.ErrInvalidRequest)
	}

	blob, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("%w: proposed_experiments could not be encoded", store.ErrInvalidRequest)
	}

	var experiments []plans.PlannedExperiment
	if err := json.Unmarshal(blob, &experiments); err != nil {
		return nil, fmt.Errorf("%w: proposed_experiments has an invalid shape", store.ErrInvalidRequest)
	}

	if len(experiments) == 0 {
		return nil, fmt.Errorf("%w: reviewer proposed no follow-up experiments", store.ErrInvalidRequest)
	}

	for index, experiment := range experiments {
		if strings.TrimSpace(experiment.Template) == "" {
			return nil, fmt.Errorf("%w: proposed experiment %d is missing template", store.ErrInvalidRequest, index)
		}
		if strings.TrimSpace(experiment.Model) == "" {
			return nil, fmt.Errorf("%w: proposed experiment %d is missing model", store.ErrInvalidRequest, index)
		}
		if experiment.Epochs < 1 {
			return nil, fmt.Errorf("%w: proposed experiment %d must have at least one epoch", store.ErrInvalidRequest, index)
		}
		if experiment.BatchSize < 1 {
			return nil, fmt.Errorf("%w: proposed experiment %d must have a positive batch size", store.ErrInvalidRequest, index)
		}
		if experiment.LearningRate <= 0 {
			return nil, fmt.Errorf("%w: proposed experiment %d must have a positive learning rate", store.ErrInvalidRequest, index)
		}
	}

	return experiments, nil
}

func recommendedWorkersForExperiments(experiments []plans.PlannedExperiment) int {
	if len(experiments) < 1 {
		return 1
	}
	return len(experiments)
}

func estimateFollowUpMinutes(experiments []plans.PlannedExperiment) int {
	maxEpochs := 1
	for _, experiment := range experiments {
		if experiment.Epochs > maxEpochs {
			maxEpochs = experiment.Epochs
		}
	}

	return max(5, maxEpochs*6)
}

func automationSettingsFromEnv() settings.AutomationSettings {
	defaultProvider := os.Getenv("MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER")
	if defaultProvider == "" {
		defaultProvider = "local"
	}

	return settings.AutomationSettings{
		AutoReviewExperiments:   envFlag("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", false),
		AutoScheduleFollowUps:   envFlag("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", false),
		AutoExecutePlans:        envFlag("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", os.Getenv("MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER") != ""),
		MaxFollowUpRounds:       maxAutoFollowUpRoundsFromEnv(),
		DefaultTrainingProvider: defaultProvider,
		DefaultGPUType:          os.Getenv("MODEL_EXPRESS_DEFAULT_GPU_TYPE"),
		LLMEnabled:              envFlag("MODEL_EXPRESS_LLM_ENABLED", false),
		AgentMode:               llm.NormalizeAgentMode(os.Getenv("MODEL_EXPRESS_AGENT_MODE")),
		LLMProvider:             defaultLLMProviderFromEnv(),
		LLMModel:                strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_MODEL")),
		UpdatedAt:               time.Now().UTC(),
	}
}

func (s *Server) currentAutomationSettings() settings.AutomationSettings {
	s.settingsMu.RLock()
	defer s.settingsMu.RUnlock()

	return s.automationSettings
}

func (s *Server) setAutomationSettings(automationSettings settings.AutomationSettings) {
	s.settingsMu.Lock()
	defer s.settingsMu.Unlock()

	s.automationSettings = automationSettings
}

func applyAutomationSettingsUpdate(current settings.AutomationSettings, update settings.AutomationSettingsUpdate) (settings.AutomationSettings, error) {
	if update.AutoReviewExperiments != nil {
		current.AutoReviewExperiments = *update.AutoReviewExperiments
	}
	if update.AutoScheduleFollowUps != nil {
		current.AutoScheduleFollowUps = *update.AutoScheduleFollowUps
	}
	if update.AutoExecutePlans != nil {
		current.AutoExecutePlans = *update.AutoExecutePlans
	}
	if update.MaxFollowUpRounds != nil {
		if *update.MaxFollowUpRounds < 0 {
			return settings.AutomationSettings{}, fmt.Errorf("%w: max_followup_rounds must be at least 0", store.ErrInvalidRequest)
		}
		current.MaxFollowUpRounds = *update.MaxFollowUpRounds
	}
	if update.DefaultTrainingProvider != nil {
		current.DefaultTrainingProvider = strings.ToLower(strings.TrimSpace(*update.DefaultTrainingProvider))
		if current.DefaultTrainingProvider == "" {
			current.DefaultTrainingProvider = "local"
		}
	}
	if update.DefaultGPUType != nil {
		current.DefaultGPUType = strings.TrimSpace(*update.DefaultGPUType)
	}
	if update.LLMEnabled != nil {
		current.LLMEnabled = *update.LLMEnabled
	}
	if update.AgentMode != nil {
		current.AgentMode = llm.NormalizeAgentMode(*update.AgentMode)
	}
	if update.LLMProvider != nil {
		current.LLMProvider = normalizeLLMProvider(*update.LLMProvider)
	}
	if update.LLMModel != nil {
		current.LLMModel = strings.TrimSpace(*update.LLMModel)
	}
	if current.AgentMode == "" {
		current.AgentMode = llm.AgentModePropose
	}
	if current.LLMProvider == "" {
		current.LLMProvider = llm.ProviderOpenAI
	}

	return current, nil
}

func (s *Server) defaultExecuteExperimentPlanRequest() executeExperimentPlanRequest {
	automationSettings := s.currentAutomationSettings()
	provider := automationSettings.DefaultTrainingProvider
	if provider == "" {
		provider = "local"
	}

	return executeExperimentPlanRequest{
		Provider: provider,
		GPUType:  automationSettings.DefaultGPUType,
	}
}

func (s *Server) shouldAutoReviewExperimentJobs() bool {
	return s.currentAutomationSettings().AutoReviewExperiments
}

func (s *Server) shouldAutoScheduleFollowUps() bool {
	return s.currentAutomationSettings().AutoScheduleFollowUps
}

func maxAutoFollowUpRoundsFromEnv() int {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS"))
	if value == "" {
		return 2
	}

	rounds, err := strconv.Atoi(value)
	if err != nil || rounds < 0 {
		return 2
	}

	return rounds
}

func (s *Server) maxAutoFollowUpRounds() int {
	return s.currentAutomationSettings().MaxFollowUpRounds
}

func (s *Server) shouldAutoExecuteExperimentPlans() bool {
	return s.currentAutomationSettings().AutoExecutePlans
}

func (s *Server) shouldRunLLMAgents() bool {
	return s.currentAutomationSettings().LLMEnabled
}

func defaultLLMProviderFromEnv() string {
	return normalizeLLMProvider(os.Getenv("MODEL_EXPRESS_LLM_PROVIDER"))
}

func normalizeLLMProvider(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case llm.ProviderOpenAICompatible:
		return llm.ProviderOpenAICompatible
	case llm.ProviderLocal:
		return llm.ProviderLocal
	default:
		return llm.ProviderOpenAI
	}
}

func envFlag(name string, defaultValue bool) bool {
	if value, ok := envFlagValue(name); ok {
		return value
	}

	return defaultValue
}

func envFlagValue(name string) (bool, bool) {
	value := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
	switch value {
	case "1", "true", "yes", "on":
		return true, true
	case "0", "false", "no", "off":
		return false, true
	}

	return false, false
}

func bindJSON(c *gin.Context, value any) bool {
	if err := c.ShouldBindJSON(value); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return false
	}

	return true
}

func writeStoreError(c *gin.Context, err error) {
	if errors.Is(err, store.ErrNotFound) {
		c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
		return
	}
	if errors.Is(err, store.ErrInvalidRequest) {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
}
