package api

import (
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
	"model-express/services/orchestrator/internal/jobs"
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

	execution, err := s.executeStoredExperimentPlan(result.FollowUpPlan.ID, s.defaultExecuteExperimentPlanRequest())
	if err != nil {
		return automaticExperimentReviewResult{}, err
	}

	result.Jobs = execution.Jobs
	return result, nil
}

func actionDecisionForPlan(agentDecisions []decisions.AgentDecision, planID string) (decisions.AgentDecision, bool) {
	for _, decision := range agentDecisions {
		if decision.PlanID == planID && decision.DecisionType != decisions.TypeWait {
			return decision, true
		}
	}

	return decisions.AgentDecision{}, false
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

func (s *Server) listProjectAgentDecisions(c *gin.Context) {
	agentDecisions, err := s.store.ListProjectAgentDecisions(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"decisions": agentDecisions})
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
		s.runAutomaticExperimentReviewAfterTrainingJob(job)
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) runAutomaticExperimentReviewAfterTrainingJob(job jobs.ExperimentJob) {
	if _, err := s.runAutomaticExperimentReview(job.ProjectID); err != nil {
		log.Printf("automatic experiment review failed after training job %s: %v", job.ID, err)
	}
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
