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
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
	"model-express/services/orchestrator/internal/strategies"
	"model-express/services/orchestrator/internal/workers"
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
	llmExperimentPlannerDecisionSource  = "llm_experiment_planner"
	minLLMDecisionConfidence            = 0.50
	maxLLMPlannerExperiments            = 5
	plannerMinimumMeaningfulImprovement = 0.005
	plannerNoImprovementRoundsToSelect  = 2
)

type updateDatasetProfileRequest struct {
	Profile map[string]any `json:"profile" binding:"required"`
}

type reportMetricRequest struct {
	Epoch   int                `json:"epoch" binding:"required"`
	Metrics map[string]float64 `json:"metrics" binding:"required"`
}

type createChampionExportRequest struct {
	Format      string         `json:"format"`
	ArtifactURI string         `json:"artifact_uri"`
	Metadata    map[string]any `json:"metadata"`
}

type createChampionDemoPredictionRequest struct {
	ImageURI      string         `json:"image_uri" binding:"required"`
	ImageID       string         `json:"image_id"`
	TrueLabel     string         `json:"true_label"`
	ImageMetadata map[string]any `json:"image_metadata"`
	TopK          int            `json:"top_k"`
}

type reportTrainingRunSummaryRequest = runs.TrainingRunSummaryUpdate
type reportTrainingRunEvaluationRequest = runs.TrainingRunEvaluationUpdate

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

func (s *Server) listDatasetVisualExemplars(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	caps := visualExemplarCapsFromQuery(c, 24, 4, 1_500_000)
	exemplars := cappedVisualExemplars(dataset.Profile, caps, "visual_exemplars", "exemplars")

	c.JSON(http.StatusOK, gin.H{
		"dataset_id":       dataset.ID,
		"source_of_truth":  "datasets.profile",
		"caps":             caps,
		"visual_exemplars": exemplars,
	})
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
	if req.Epoch < 1 {
		writeStoreError(c, fmt.Errorf("%w: epoch must be positive", store.ErrInvalidRequest))
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

func (s *Server) upsertTrainingRunEvaluation(c *gin.Context) {
	var req reportTrainingRunEvaluationRequest
	if !bindJSON(c, &req) {
		return
	}

	evaluation, err := s.store.UpsertTrainingRunEvaluation(c.Param("id"), req)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, evaluation)
}

func (s *Server) getTrainingRunEvaluation(c *gin.Context) {
	evaluation, err := s.store.GetTrainingRunEvaluation(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, evaluation)
}

func (s *Server) listProjectTrainingRunSummaries(c *gin.Context) {
	summaries, err := s.store.ListProjectTrainingRunSummaries(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"summaries": summaries})
}

func (s *Server) listProjectTrainingRunEvaluations(c *gin.Context) {
	evaluations, err := s.store.ListProjectTrainingRunEvaluations(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"evaluations": evaluations})
}

func (s *Server) getProjectChampion(c *gin.Context) {
	champion, err := s.store.GetProjectChampion(c.Param("id"))
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			c.JSON(http.StatusOK, gin.H{"champion": nil})
			return
		}
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"champion": champion})
}

func (s *Server) listProjectChampionExports(c *gin.Context) {
	exports, err := s.store.ListProjectChampionExports(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"exports": exports})
}

func (s *Server) createProjectChampionExport(c *gin.Context) {
	projectID := c.Param("id")
	var req createChampionExportRequest
	if !bindJSON(c, &req) {
		return
	}

	champion, err := s.store.GetProjectChampion(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	job, err := s.store.GetJob(champion.JobID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if job.Status != jobs.StatusSucceeded {
		writeStoreError(c, fmt.Errorf("%w: champion job must be succeeded before export", store.ErrInvalidRequest))
		return
	}
	if _, err := s.store.GetTrainingRunSummary(champion.JobID); err != nil {
		if !errors.Is(err, store.ErrNotFound) {
			writeStoreError(c, err)
			return
		}
		writeStoreError(c, fmt.Errorf("%w: champion job must have a training run summary before export", store.ErrInvalidRequest))
		return
	}

	format := normalizeChampionExportFormat(req.Format)
	if format == "" {
		writeStoreError(c, fmt.Errorf("%w: champion export format must be one of onnx, torchscript, pytorch, safetensors", store.ErrInvalidRequest))
		return
	}

	artifactURI := strings.TrimSpace(req.ArtifactURI)
	if artifactURI == "" {
		artifactURI = championArtifactURI(champion.DeploymentProfile)
	}
	status := runs.ChampionExportStatusReady
	validationErrors := []string{}
	if artifactURI == "" {
		status = runs.ChampionExportStatusPendingArtifact
		validationErrors = append(validationErrors, "selected champion has no exportable artifact URI yet")
	}

	metadata := championExportMetadata(champion, format, req.Metadata)
	export, err := s.store.CreateChampionExport(runs.ChampionExportCreate{
		ProjectID:        projectID,
		ChampionID:       champion.ID,
		JobID:            champion.JobID,
		Status:           status,
		Format:           format,
		ArtifactURI:      artifactURI,
		Metadata:         metadata,
		ValidationErrors: validationErrors,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if _, err := s.store.CreateExecutionEvent(projectID, champion.PlanID, execution.EventChampionExportRequested, fmt.Sprintf("Champion export requested for job %s.", champion.JobID), map[string]any{
		"champion_id": champion.ID,
		"export_id":   export.ID,
		"job_id":      champion.JobID,
		"status":      export.Status,
		"format":      export.Format,
	}); err != nil {
		log.Printf("record champion export event failed: %v", err)
	}

	c.JSON(http.StatusCreated, export)
}

func (s *Server) listProjectChampionDemoImages(c *gin.Context) {
	projectID := c.Param("id")
	champion, err := s.store.GetProjectChampion(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	dataset, err := s.store.GetDataset(champion.DatasetID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	caps := visualExemplarCapsFromQuery(c, 24, 4, 1_500_000)
	exemplars := cappedVisualExemplars(dataset.Profile, caps, "demo_images", "visual_exemplars")

	c.JSON(http.StatusOK, gin.H{
		"project_id":      projectID,
		"dataset_id":      dataset.ID,
		"champion_job_id": champion.JobID,
		"source_of_truth": "datasets.profile",
		"caps":            caps,
		"images":          exemplars,
	})
}

func (s *Server) listProjectChampionDemoPredictions(c *gin.Context) {
	predictions, err := s.store.ListProjectChampionDemoPredictions(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"predictions": predictions})
}

func (s *Server) createProjectChampionDemoPrediction(c *gin.Context) {
	var req createChampionDemoPredictionRequest
	if !bindJSON(c, &req) {
		return
	}
	if req.TopK < 1 {
		req.TopK = 5
	}
	if req.TopK > 10 {
		writeStoreError(c, fmt.Errorf("%w: top_k must be at most 10", store.ErrInvalidRequest))
		return
	}
	imageURI := strings.TrimSpace(req.ImageURI)
	if imageURI == "" {
		writeStoreError(c, fmt.Errorf("%w: image_uri is required", store.ErrInvalidRequest))
		return
	}

	champion, err := s.store.GetProjectChampion(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	dataset, err := s.store.GetDataset(champion.DatasetID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		writeStoreError(c, err)
		return
	}
	imageID := strings.TrimSpace(req.ImageID)
	trueLabel := strings.TrimSpace(req.TrueLabel)
	imageMetadata := map[string]any{}
	for key, value := range req.ImageMetadata {
		imageMetadata[key] = value
	}
	if err == nil {
		if matchedImageID, matchedTrueLabel, matchedMetadata, ok := championDemoImageMetadata(dataset.Profile, imageURI); ok {
			if imageID == "" {
				imageID = matchedImageID
			}
			if trueLabel == "" {
				trueLabel = matchedTrueLabel
			}
			for key, value := range matchedMetadata {
				if _, exists := imageMetadata[key]; !exists {
					imageMetadata[key] = value
				}
			}
		}
	}

	prediction, err := s.store.CreateChampionDemoPrediction(runs.ChampionDemoPredictionCreate{
		ProjectID:     champion.ProjectID,
		ChampionID:    champion.ID,
		JobID:         champion.JobID,
		DatasetID:     champion.DatasetID,
		ImageURI:      imageURI,
		ImageID:       imageID,
		ImageMetadata: imageMetadata,
		Status:        runs.ChampionDemoPredictionStatusRuntimeUnavailable,
		TrueLabel:     trueLabel,
		TopK:          []runs.DemoPredictionTopK{},
		Error:         "champion demo prediction runtime is not wired yet",
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if _, err := s.store.CreateExecutionEvent(champion.ProjectID, champion.PlanID, execution.EventChampionDemoPrediction, fmt.Sprintf("Champion demo prediction requested for job %s.", champion.JobID), map[string]any{
		"champion_id":   champion.ID,
		"prediction_id": prediction.ID,
		"job_id":        champion.JobID,
		"status":        prediction.Status,
		"image_uri":     prediction.ImageURI,
	}); err != nil {
		log.Printf("record champion demo prediction event failed: %v", err)
	}

	c.JSON(http.StatusAccepted, gin.H{
		"prediction":        prediction,
		"runtime_available": false,
		"contract": gin.H{
			"champion_job_id": champion.JobID,
			"image_uri":       imageURI,
			"top_k":           req.TopK,
			"returns":         []string{"predicted_label", "true_label", "confidence", "top_k", "latency_ms", "correct"},
		},
	})
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
	if _, err := s.createPendingStrategyScorecard(projectID, sourcePlan, decision, plan); err != nil {
		log.Printf("create pending strategy scorecard failed for decision %s: %v", decision.ID, err)
	}

	return plan, true, nil
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
	return s.store.CreateStrategyScorecard(strategies.StrategyScorecardCreate{
		ProjectID:        projectID,
		DatasetID:        sourcePlan.DatasetID,
		SourceDecisionID: decision.ID,
		SourcePlanID:     sourcePlan.ID,
		FollowUpPlanID:   followUpPlan.ID,
		StrategyType:     strategyType,
		PlanningMode:     planningMode,
		DatasetTraits:    datasetTraits,
		ObjectiveProfile: objectiveProfile,
		ProposedChanges:  proposedChanges,
		ExpectedDelta:    payloadFloat(decision.Payload, "expected_delta_vs_champion"),
		ConfidenceBefore: payloadFloat(decision.Payload, "confidence"),
		Outcome:          strategies.OutcomePending,
		Lesson:           "Pending follow-up outcome.",
		Tags:             uniqueStrings([]string{"strategy_scorecard", planningMode, strategies.OutcomePending}),
	})
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
	openJobCount := openTrainingJobCount(queuedJobs)
	if openJobCount == 0 {
		_, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventJobsQueued, fmt.Sprintf("Plan %s has no open automatic experiment jobs requiring workers.", plan.ID), map[string]any{
			"job_ids":        experimentJobIDs(queuedJobs),
			"open_job_count": 0,
		})
		return err
	}
	targetCount := s.targetWorkerCountForPlan(plan, openJobCount)
	activeWorkerCount := s.activeOrStartingWorkersForProject(plan.ProjectID)

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
		"job_ids":               experimentJobIDs(queuedJobs),
		"open_job_count":        openJobCount,
		"active_worker_count":   activeWorkerCount,
		"worker_requirement_id": requirement.ID,
		"target_count":          requirement.TargetCount,
		"requirement_status":    requirementStatus,
		"provider":              provider,
		"gpu_type":              req.GPUType,
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
		"worker_requirement_id": requirement.ID,
		"target_count":          requirement.TargetCount,
		"open_job_count":        openJobCount,
		"active_worker_count":   activeWorkerCount,
		"provider":              requirement.Provider,
		"gpu_type":              requirement.GPUType,
	})
	return err
}

func experimentJobIDs(experimentJobs []jobs.ExperimentJob) []string {
	out := make([]string, 0, len(experimentJobs))
	for _, job := range experimentJobs {
		out = append(out, job.ID)
	}
	return out
}

func openTrainingJobCount(experimentJobs []jobs.ExperimentJob) int {
	count := 0
	for _, job := range experimentJobs {
		if job.Template != jobs.TemplateTrainExperiment {
			continue
		}
		switch strings.ToUpper(strings.TrimSpace(job.Status)) {
		case jobs.StatusQueued, jobs.StatusAssigned, jobs.StatusRunning:
			count++
		}
	}
	return count
}

func (s *Server) targetWorkerCountForPlan(plan plans.ExperimentPlan, openJobCount int) int {
	if openJobCount < 1 {
		return 1
	}
	targetCount := plan.RecommendedWorkers
	if targetCount < 1 {
		targetCount = 1
	}
	if targetCount > openJobCount {
		targetCount = openJobCount
	}
	if maxWorkers := maxAutoWorkersFromEnv(); maxWorkers > 0 && targetCount > maxWorkers {
		targetCount = maxWorkers
	}
	return targetCount
}

func (s *Server) activeOrStartingWorkersForProject(projectID string) int {
	projectWorkers, err := s.store.ListProjectWorkers(projectID)
	if err != nil {
		return 0
	}
	count := 0
	for _, worker := range projectWorkers {
		if time.Since(worker.LastHeartbeat) > workers.HeartbeatLimit {
			continue
		}
		switch strings.ToUpper(strings.TrimSpace(worker.Status)) {
		case workers.StatusIdle, workers.StatusRunning:
			count++
		}
	}
	return count
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

func (s *Server) listProjectStrategyScorecards(c *gin.Context) {
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "25"))
	scorecards, err := s.store.ListProjectStrategyScorecards(c.Param("id"), limit)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"scorecards": scorecards})
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
		s.runPlanningLoopAfterTrainingJob(job)
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
		s.runPlanningLoopAfterTrainingJob(job)
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) runAutomaticExperimentReviewAfterTrainingJob(job jobs.ExperimentJob) {
	if _, err := s.runAutomaticExperimentReview(job.ProjectID); err != nil {
		log.Printf("automatic experiment review failed after training job %s: %v", job.ID, err)
	}
}

func (s *Server) runPlanningLoopAfterTrainingJob(job jobs.ExperimentJob) {
	if err := s.recordExperimentPlannerOutcomeAfterTrainingJob(job); err != nil {
		log.Printf("record experiment planner outcome failed after training job %s: %v", job.ID, err)
	}

	handled, err := s.runExperimentPlannerAfterTrainingJob(job)
	if err != nil {
		log.Printf("llm experiment planner failed after training job %s: %v", job.ID, err)
	}
	if handled {
		return
	}
	s.runAutomaticExperimentReviewAfterTrainingJob(job)
}

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
	outcome, err := experimentPlanningOutcomeForPlan(sourceDecision, plan, projectPlans, summaries)
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

	if _, err := s.store.CreateExecutionEvent(job.ProjectID, plan.ID, execution.EventAgentOutcomeRecorded, fmt.Sprintf("Experiment Planner outcome recorded for follow-up plan %s.", plan.ID), map[string]any{
		"invocation_id":      updatedInvocation.ID,
		"memory_record_id":   record.ID,
		"source_decision_id": sourceDecision.ID,
		"outcome_status":     outcome.OutcomeStatus,
	}); err != nil {
		log.Printf("record experiment planner outcome event failed: %v", err)
	}
	if _, err := s.store.UpdateStrategyScorecardOutcomeByFollowUpPlan(plan.ID, strategies.StrategyScorecardOutcomeUpdate{
		ActualDelta:     outcome.ActualDeltaVsChampion,
		ConfidenceAfter: plannerOutcomeConfidence(outcome),
		CostUSD:         outcome.TotalCostUSD,
		RuntimeSeconds:  outcome.TotalRuntimeSeconds,
		Outcome:         outcome.OutcomeStatus,
		Lesson:          outcome.Lesson,
		Tags:            tags,
	}); err != nil && !errors.Is(err, store.ErrNotFound) {
		log.Printf("update strategy scorecard failed for follow-up plan %s: %v", plan.ID, err)
	}
	return nil
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
	client := llm.NewOpenAICompatibleClient(config)
	agent := agents.NewTrainingMonitorAgent(client, config.Model)

	ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
	defer cancel()

	trace, err := agent.EvaluateWithTrace(ctx, agents.TrainingMonitorInput{
		Plan:             plan,
		Job:              job,
		Summary:          summary,
		Evaluation:       evaluation,
		Metrics:          metrics,
		ObjectiveContext: projectObjectiveContext(goalText),
		MemoryRecords:    priorMemory,
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
	config := llm.ConfigFromEnv(automationSettings.LLMEnabled, automationSettings.LLMProvider, automationSettings.LLMModel)
	client := llm.NewOpenAICompatibleClient(config)
	agent := agents.NewExperimentPlannerAgent(client, config.Model)

	ctx, cancel := context.WithTimeout(context.Background(), config.Timeout)
	defer cancel()

	trace, err := agent.PlanWithTrace(ctx, input)
	acceptedForMemory := err == nil
	invocation, invocationErr := s.recordExperimentPlannerInvocation(input, config, trace, acceptedForMemory)
	if invocationErr != nil {
		log.Printf("experiment planner invocation write failed for plan %s: %v", summary.PlanID, invocationErr)
	}
	if err != nil {
		if _, eventErr := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventExecutionFailed, fmt.Sprintf("Experiment Planner failed for plan %s.", summary.PlanID), map[string]any{
			"invocation_id": invocation.ID,
			"error":         err.Error(),
		}); eventErr != nil {
			log.Printf("record experiment planner failure event failed: %v", eventErr)
		}
		return false, err
	}

	recommendation := applyExperimentPlannerStopCriteria(trace.Recommendation, input)
	payload, err := experimentPlannerDecisionPayload(recommendation, invocation, automationSettings.AgentMode, input)
	if err != nil {
		if _, eventErr := s.store.CreateExecutionEvent(job.ProjectID, summary.PlanID, execution.EventExecutionFailed, fmt.Sprintf("Experiment Planner proposal failed validation for plan %s.", summary.PlanID), map[string]any{
			"invocation_id": invocation.ID,
			"error":         err.Error(),
		}); eventErr != nil {
			log.Printf("record experiment planner validation event failed: %v", eventErr)
		}
		return false, err
	}

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

	partialInput := agents.ExperimentPlannerInput{
		Project:                    project,
		Dataset:                    dataset,
		SourcePlan:                 plan,
		PlanJobs:                   planJobs,
		PlanSummaries:              planSummaries,
		PlanEvaluations:            planEvaluations,
		PlanMetrics:                planMetrics,
		DatasetInsights:            datasetPlanningInsights(dataset),
		VisualExemplarContext:      plannerVisualExemplarContext(dataset),
		ObjectiveContext:           objectiveContext,
		CurrentChampion:            currentChampion,
		SourcePlanBaselineChampion: baselineChampion,
		SourcePlanDeltas:           sourcePlanDeltas,
		NoImprovementRounds:        noImprovementRounds,
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

	return agents.ExperimentPlannerInput{
		Project:                      project,
		Dataset:                      dataset,
		SourcePlan:                   plan,
		PlanJobs:                     planJobs,
		PlanSummaries:                planSummaries,
		PlanEvaluations:              planEvaluations,
		PlanMetrics:                  planMetrics,
		DatasetInsights:              partialInput.DatasetInsights,
		VisualExemplarContext:        partialInput.VisualExemplarContext,
		ObjectiveContext:             objectiveContext,
		DeterministicDiagnosis:       deterministicDiagnosis,
		ModelCatalog:                 supportedModelCatalog(),
		CurrentChampion:              currentChampion,
		SourcePlanBaselineChampion:   baselineChampion,
		SourcePlanDeltas:             sourcePlanDeltas,
		NoImprovementRounds:          noImprovementRounds,
		StopSignals:                  stopSignals,
		MinimumMeaningfulImprovement: plannerMinimumMeaningfulImprovement,
		SuccessfulStrategyMemory:     successfulStrategyMemory,
		FailedStrategyMemory:         failedStrategyMemory,
		RejectedStrategyMemory:       rejectedStrategyMemory,
		StrategyScorecards:           strategyScorecards,
		PriorPlans:                   projectPlans,
		PriorJobs:                    projectJobs,
		PriorEvaluations:             evaluations,
		PriorMemory:                  priorMemory,
		ExistingExperimentSignatures: experimentSignaturesForPlans(projectPlans),
		MaxExperiments:               maxLLMPlannerExperiments,
		MaxFollowUpRounds:            s.maxAutoFollowUpRounds(),
		FollowUpRound:                followUpRoundCount(projectPlans),
	}, true, nil
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

func applyExperimentPlannerStopCriteria(
	recommendation agents.ExperimentPlanningRecommendation,
	input agents.ExperimentPlannerInput,
) agents.ExperimentPlanningRecommendation {
	decisionType := strings.ToUpper(strings.TrimSpace(recommendation.DecisionType))
	recommendation.DecisionType = decisionType
	if decisionType != decisions.TypeAddExperiments {
		return recommendation
	}
	if input.CurrentChampion == nil || input.NoImprovementRounds < plannerNoImprovementRoundsToSelect {
		return recommendation
	}

	stopReason := fmt.Sprintf(
		"Current champion %s remains unbeaten after %d consecutive follow-up plan(s) with less than %.3f target-metric improvement.",
		input.CurrentChampion.JobID,
		input.NoImprovementRounds,
		input.MinimumMeaningfulImprovement,
	)
	recommendation.DecisionType = decisions.TypeSelectChampion
	recommendation.ChampionJobID = input.CurrentChampion.JobID
	recommendation.ProposedExperiments = nil
	recommendation.StopReason = stopReason
	recommendation.Summary = fmt.Sprintf("Select champion %s; recent follow-ups did not meaningfully improve on it.", input.CurrentChampion.JobID)
	recommendation.Rationale = strings.TrimSpace(recommendation.Rationale + " Backend stop criteria applied: " + stopReason)
	recommendation.NoveltyNotes = append(recommendation.NoveltyNotes, "Backend guard converted ADD_EXPERIMENTS to SELECT_CHAMPION after repeated no-improvement rounds.")
	recommendation.Tags = append(recommendation.Tags, "select_champion", "no_improvement_guard")
	return recommendation
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
		"no_improvement_rounds":           input.NoImprovementRounds,
		"minimum_meaningful_improvement":  input.MinimumMeaningfulImprovement,
		"stop_signals":                    input.StopSignals,
	}

	if strings.EqualFold(recommendation.DecisionType, decisions.TypeAddExperiments) {
		if err := validateNovelProposedExperiments(recommendation.ProposedExperiments, input.PriorPlans); err != nil {
			return nil, err
		}
		payload["proposed_experiments"] = recommendation.ProposedExperiments
	}

	return payload, nil
}

func (s *Server) persistProjectChampionFromDecision(projectID string, decision decisions.AgentDecision) error {
	if strings.ToUpper(strings.TrimSpace(decision.DecisionType)) != decisions.TypeSelectChampion {
		return nil
	}

	championJobID := payloadString(decision.Payload, "champion_job_id")
	if championJobID == "" {
		if champion, ok := experimentChampionFromPayload(decision.Payload["current_champion"]); ok {
			championJobID = champion.JobID
		}
	}
	if championJobID == "" {
		return fmt.Errorf("%w: SELECT_CHAMPION decision is missing champion_job_id", store.ErrInvalidRequest)
	}

	summary, err := s.store.GetTrainingRunSummary(championJobID)
	if err != nil {
		return err
	}
	job, err := s.store.GetJob(championJobID)
	if err != nil && !errors.Is(err, store.ErrNotFound) {
		return err
	}
	goalText := ""
	if project, err := s.store.GetProject(projectID); err == nil {
		goalText = project.Goal
	}
	targetMetric := "macro_f1"
	if summary.PlanID != "" {
		if plan, err := s.store.GetExperimentPlan(summary.PlanID); err == nil && plan.TargetMetric != "" {
			targetMetric = plan.TargetMetric
		}
	}

	evaluationPayload := map[string]any{}
	deploymentProfile := map[string]any{
		"objective_context": projectObjectiveContext(goalText),
		"target_metric":     normalizedPlannerTargetMetric(targetMetric),
		"diagnostics":       "pending",
		"model_card": map[string]any{
			"intended_use":              goalText,
			"known_limitations":         []string{"Final export and production inference validation are still pending."},
			"recommended_preprocessing": []string{"Use the same image size, normalization, and augmentation assumptions from the winning experiment config."},
			"export_status":             "pending",
		},
	}
	if evaluation, err := s.store.GetTrainingRunEvaluation(championJobID); err == nil {
		if payload, payloadErr := mapFromStruct(evaluation); payloadErr == nil {
			evaluationPayload = payload
		}
		deploymentProfile["model_profile"] = evaluation.ModelProfile
		deploymentProfile["holistic_scores"] = evaluation.HolisticScores
		deploymentProfile["diagnostics"] = "available"
	} else if !errors.Is(err, store.ErrNotFound) {
		return err
	}

	selectionReason := strings.TrimSpace(decision.Rationale)
	if stopReason := payloadString(decision.Payload, "stop_reason"); stopReason != "" {
		selectionReason = strings.TrimSpace(selectionReason + " " + stopReason)
	}
	metrics := map[string]any{
		"model":                  summary.Model,
		"status":                 summary.Status,
		"best_macro_f1":          summary.BestMacroF1,
		"best_accuracy":          summary.BestAccuracy,
		"estimated_cost_usd":     summary.EstimatedCostUSD,
		"runtime_seconds":        summary.RuntimeSeconds,
		"epochs_completed":       summary.EpochsCompleted,
		"final_train_loss":       summary.FinalTrainLoss,
		"final_val_loss":         summary.FinalValLoss,
		"modal_function_call_id": summary.ModalFunctionCallID,
		"modal_input_id":         summary.ModalInputID,
	}
	if job.ID != "" {
		metrics["job_config"] = job.Config
		if model := configString(job.Config, "model"); model != "" && summary.Model == "" {
			metrics["model"] = model
		}
	}

	champion, err := s.store.UpsertProjectChampion(runs.ProjectChampionUpsert{
		ProjectID:         projectID,
		DatasetID:         summary.DatasetID,
		PlanID:            summary.PlanID,
		JobID:             championJobID,
		SourceDecisionID:  decision.ID,
		SelectionReason:   selectionReason,
		Metrics:           metrics,
		Evaluation:        evaluationPayload,
		DeploymentProfile: deploymentProfile,
	})
	if err != nil {
		return err
	}

	_, err = s.store.CreateExecutionEvent(projectID, summary.PlanID, execution.EventChampionSelected, fmt.Sprintf("Champion selected: %s for project %s.", championJobID, projectID), map[string]any{
		"champion_id":        champion.ID,
		"champion_job_id":    champion.JobID,
		"source_decision_id": decision.ID,
		"model":              metrics["model"],
	})
	return err
}

func (s *Server) schedulePlannerDecision(projectID string, sourcePlan plans.ExperimentPlan, decision decisions.AgentDecision, result automaticExperimentReviewResult) error {
	projectPlans, err := s.store.ListProjectExperimentPlans(projectID)
	if err != nil {
		return err
	}
	if existingPlan, ok := followUpPlanForDecision(projectPlans, decision.ID); ok {
		result.FollowUpPlan = &existingPlan
		_, err := s.executeAutomaticFollowUpPlan(result)
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
		return err
	}

	result.FollowUpPlan = &followUpPlan
	_, err = s.executeAutomaticFollowUpPlan(result)
	return err
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

func datasetPlanningInsights(dataset datasets.Dataset) agents.DatasetPlanningInsights {
	profile := dataset.Profile
	taskType := profileString(profile, "task_type")
	if taskType == "" {
		taskType = "image_classification"
	}
	classCount := profileInt(profile, "class_count")
	totalImages := profileInt(profile, "total_images")
	imageCount := profileInt(profile, "image_count")
	if totalImages == 0 {
		totalImages = imageCount
	}
	if imageCount == 0 {
		imageCount = totalImages
	}
	imbalanceRatio := profileFloat(profile, "imbalance_ratio")
	corruptImageCount := profileInt(profile, "corrupt_image_count")
	corruptFileCount := profileInt(profile, "corrupt_file_count")
	if corruptImageCount == 0 {
		corruptImageCount = corruptFileCount
	}
	if corruptFileCount == 0 {
		corruptFileCount = corruptImageCount
	}
	widthMin := profileInt(profile, "width_min")
	widthMax := profileInt(profile, "width_max")
	heightMin := profileInt(profile, "height_min")
	heightMax := profileInt(profile, "height_max")
	classDistribution := profileMap(profile, "class_distribution")
	if len(classDistribution) == 0 {
		classDistribution = profileMap(profile, "images_per_class")
	}
	imageDimensionStats := profileMap(profile, "image_dimension_stats")
	splitSummary := profileMap(profile, "split_summary")
	metadataSummary := profileMap(profile, "metadata_summary")
	leakageWarnings := profileStringSlice(profile, "leakage_warnings")
	datasetTraits := profileStringSlice(profile, "dataset_traits")
	artifacts := profileMapSlice(profile, "artifacts")

	constraints := []string{}
	recommendedPreprocessing := []string{"normalize with ImageNet statistics for transfer learning"}
	recommendedAugmentations := []string{}
	recommendedMetrics := []string{"accuracy", "macro_f1"}
	liveInferencePriorities := []string{
		"Prefer compact architectures when quality is close so the final model can classify live images with low latency.",
		"Only increase image_size when prior results show a meaningful quality gain over the deployment cost.",
	}

	if totalImages == 0 {
		constraints = append(constraints, "Dataset has not been profiled yet; use conservative transfer-learning defaults and prioritize profiling before aggressive search.")
	} else if totalImages < 500 {
		constraints = append(constraints, "Small dataset; avoid overfitting and prefer stronger augmentation, early stopping, and regularization.")
		recommendedAugmentations = append(recommendedAugmentations, "horizontal_flip", "color_jitter", "random_crop")
	} else if totalImages < 2000 {
		constraints = append(constraints, "Medium-small dataset; compare efficient transfer models with moderate augmentation.")
		recommendedAugmentations = append(recommendedAugmentations, "horizontal_flip", "color_jitter")
	}
	if imbalanceRatio >= 1.5 {
		constraints = append(constraints, fmt.Sprintf("Class imbalance detected (ratio %.2f); optimize macro-F1 and test class balancing.", imbalanceRatio))
		recommendedPreprocessing = append(recommendedPreprocessing, "class balancing with weighted_loss")
		recommendedMetrics = append(recommendedMetrics, "per_class_f1")
	}
	if corruptImageCount > 0 {
		constraints = append(constraints, fmt.Sprintf("%d corrupt image(s) were detected; clean or skip them before trusting final metrics.", corruptImageCount))
	}
	if len(leakageWarnings) > 0 {
		constraints = append(constraints, leakageWarnings...)
	}
	if metadataBool(metadataSummary, "bbox_available") || containsString(datasetTraits, "bbox_available") {
		recommendedPreprocessing = append(recommendedPreprocessing, "bbox_crop_if_available as an ablation against full-image training")
	}
	if metadataBool(metadataSummary, "metadata_available") || containsString(datasetTraits, "metadata_available") {
		recommendedPreprocessing = append(recommendedPreprocessing, "preserve metadata artifacts for controlled preprocessing ablations")
	}
	if widthMax > 0 && heightMax > 0 {
		maxDimension := widthMax
		if heightMax > maxDimension {
			maxDimension = heightMax
		}
		minDimension := widthMin
		if heightMin > 0 && (minDimension == 0 || heightMin < minDimension) {
			minDimension = heightMin
		}
		if maxDimension >= 512 {
			recommendedPreprocessing = append(recommendedPreprocessing, "compare 224 and 256 image_size before trying larger inputs")
		} else if maxDimension <= 160 {
			recommendedPreprocessing = append(recommendedPreprocessing, "avoid unnecessary upscaling beyond 224 unless validation gains justify latency")
		}
		if minDimension > 0 && maxDimension > minDimension*2 {
			constraints = append(constraints, "Large variation in image dimensions; prefer resize plus random crop to improve robustness.")
			recommendedAugmentations = append(recommendedAugmentations, "random_crop")
		}
	}
	if len(recommendedAugmentations) == 0 {
		recommendedAugmentations = append(recommendedAugmentations, "horizontal_flip if class semantics allow it")
	}

	summary := fmt.Sprintf(
		"%s dataset with %d classes, %d images, imbalance ratio %.2f, and %d corrupt image(s).",
		taskType,
		classCount,
		totalImages,
		imbalanceRatio,
		corruptImageCount,
	)

	return agents.DatasetPlanningInsights{
		Summary:                  summary,
		TaskType:                 taskType,
		ClassCount:               classCount,
		TotalImages:              totalImages,
		ImageCount:               imageCount,
		ClassDistribution:        classDistribution,
		ImbalanceRatio:           imbalanceRatio,
		CorruptImageCount:        corruptImageCount,
		CorruptFileCount:         corruptFileCount,
		WidthMin:                 widthMin,
		WidthMax:                 widthMax,
		HeightMin:                heightMin,
		HeightMax:                heightMax,
		ImageDimensionStats:      imageDimensionStats,
		SplitSummary:             splitSummary,
		MetadataSummary:          metadataSummary,
		LeakageWarnings:          leakageWarnings,
		DatasetTraits:            datasetTraits,
		Artifacts:                artifacts,
		Constraints:              uniqueStrings(constraints),
		RecommendedPreprocessing: uniqueStrings(recommendedPreprocessing),
		RecommendedAugmentations: uniqueStrings(recommendedAugmentations),
		RecommendedMetrics:       uniqueStrings(recommendedMetrics),
		LiveInferencePriorities:  liveInferencePriorities,
	}
}

func plannerVisualExemplarContext(dataset datasets.Dataset) *agents.PlannerVisualExemplarContext {
	caps := visualExemplarCaps{MaxTotalImages: 24, MaxPerClass: 2, MaxBytes: 512 * 1024}
	exemplars := cappedVisualExemplars(dataset.Profile, caps, "visual_exemplars")
	if len(exemplars) == 0 {
		return &agents.PlannerVisualExemplarContext{
			Enabled:      false,
			EvidenceOnly: true,
			ByteBudget:   int(caps.MaxBytes),
			PromptBudget: 4000,
			Summary:      "No backend-curated visual exemplars are available for this dataset.",
		}
	}

	byClass := map[string][]datasets.VisualExemplar{}
	var totalBytes int64
	for _, exemplar := range exemplars {
		className := exemplar.ClassName
		if className == "" {
			className = exemplar.Label
		}
		if className == "" {
			className = "unknown"
		}
		byClass[className] = append(byClass[className], exemplar)
		totalBytes += maxInt64(exemplar.SizeBytes, 0)
	}

	classNames := make([]string, 0, len(byClass))
	for className := range byClass {
		classNames = append(classNames, className)
	}
	sort.Strings(classNames)
	classEvidence := make([]agents.PlannerClassExemplar, 0, len(classNames))
	for _, className := range classNames {
		classExemplars := byClass[className]
		metadata := map[string]any{}
		first := classExemplars[0]
		if first.Split != "" {
			metadata["split"] = first.Split
		}
		if first.Width > 0 {
			metadata["width"] = first.Width
		}
		if first.Height > 0 {
			metadata["height"] = first.Height
		}
		classEvidence = append(classEvidence, agents.PlannerClassExemplar{
			ClassName:     className,
			ExemplarCount: len(classExemplars),
			Metadata:      metadata,
		})
	}

	warnings := []string{}
	if len(exemplars) >= caps.MaxTotalImages {
		warnings = append(warnings, "visual exemplar cap reached; planner saw a bounded class-balanced subset")
	}
	if totalBytes >= caps.MaxBytes {
		warnings = append(warnings, "visual exemplar byte budget reached")
	}

	return &agents.PlannerVisualExemplarContext{
		Enabled:        true,
		EvidenceOnly:   true,
		ExemplarCount:  len(exemplars),
		ClassCount:     len(classEvidence),
		ByteBudget:     int(caps.MaxBytes),
		PromptBudget:   4000,
		Summary:        fmt.Sprintf("%d backend-curated visual exemplar(s) across %d class(es) are available as evidence only.", len(exemplars), len(classEvidence)),
		ObservedTraits: datasetProfileTraits(dataset.Profile),
		ClassEvidence:  classEvidence,
		Warnings:       warnings,
	}
}

func datasetProfileTraits(profile map[string]any) []string {
	traits := []string{}
	classCount := profileInt(profile, "class_count")
	totalImages := profileInt(profile, "total_images")
	imbalanceRatio := profileFloat(profile, "imbalance_ratio")
	widthMax := profileInt(profile, "width_max")
	heightMax := profileInt(profile, "height_max")
	maxDimension := widthMax
	if heightMax > maxDimension {
		maxDimension = heightMax
	}

	if totalImages > 0 && totalImages < 500 {
		traits = append(traits, "small_dataset")
	} else if totalImages >= 500 && totalImages < 5000 {
		traits = append(traits, "medium_dataset")
	}
	if classCount >= 20 {
		traits = append(traits, "many_classes")
	}
	if classCount >= 10 && totalImages > 0 && totalImages/classCount < 120 {
		traits = append(traits, "fine_grained_possible")
	}
	if imbalanceRatio >= 1.5 {
		traits = append(traits, "imbalanced")
	}
	if maxDimension > 0 && maxDimension <= 160 {
		traits = append(traits, "low_resolution")
	} else if maxDimension >= 512 {
		traits = append(traits, "high_resolution")
	}
	if profileBool(profile, "metadata_available") || profile["metadata_summary"] != nil {
		traits = append(traits, "metadata_available")
	}
	if profileBool(profile, "bbox_available") || profileBool(profile, "bounding_boxes_available") {
		traits = append(traits, "bbox_available")
	}
	return uniqueStrings(traits)
}

func projectObjectiveContext(goal string) agents.ProjectObjectiveContext {
	goalText := strings.TrimSpace(goal)
	normalized := strings.ToLower(goalText)
	context := agents.ProjectObjectiveContext{
		GoalText:             goalText,
		PrimaryObjective:     "balanced_quality",
		MetricPreferences:    []string{"macro_f1", "accuracy"},
		DeploymentPriorities: []string{"explain quality, cost, and runtime tradeoffs"},
		Constraints:          []string{},
		RankingWeights: map[string]float64{
			"macro_f1":           0.35,
			"accuracy":           0.25,
			"per_class_behavior": 0.15,
			"latency":            0.10,
			"training_cost":      0.08,
			"runtime":            0.07,
		},
	}

	if containsAny(normalized, "live", "real-time", "realtime", "instant", "fast", "quick", "low latency") {
		context.PrimaryObjective = "low_latency_live_service"
		context.DeploymentPriorities = append(context.DeploymentPriorities, "prioritize fast inference when quality is close", "prefer compact models for live serving")
		context.Constraints = append(context.Constraints, "avoid heavy challenger models unless they clearly improve quality")
		context.RankingWeights["latency"] = 0.22
		context.RankingWeights["model_size"] = 0.10
		context.RankingWeights["macro_f1"] = 0.28
		context.RankingWeights["accuracy"] = 0.20
	}
	if containsAny(normalized, "cheap", "budget", "cost", "low cost", "inexpensive") {
		if context.PrimaryObjective == "balanced_quality" {
			context.PrimaryObjective = "budget_sensitive"
		}
		context.DeploymentPriorities = append(context.DeploymentPriorities, "prefer lower training and inference cost when quality is close")
		context.RankingWeights["training_cost"] = 0.18
		context.RankingWeights["runtime"] = 0.10
	}
	if containsAny(normalized, "accurate", "accuracy", "best", "quality", "high quality") {
		if context.PrimaryObjective == "balanced_quality" {
			context.PrimaryObjective = "quality_first"
		}
		context.DeploymentPriorities = append(context.DeploymentPriorities, "allow stronger models when they produce meaningful quality gains")
		context.RankingWeights["macro_f1"] = 0.38
		context.RankingWeights["accuracy"] = 0.30
	}
	if containsAny(normalized, "imbalanced", "minority", "rare class", "rare classes", "fair", "per-class", "per class") {
		context.MetricPreferences = append(context.MetricPreferences, "per_class_f1", "recall_by_class")
		context.DeploymentPriorities = append(context.DeploymentPriorities, "avoid selecting a model that hides weak minority-class behavior behind average accuracy")
		context.RankingWeights["per_class_behavior"] = 0.24
		context.RankingWeights["macro_f1"] = 0.40
	}
	if containsAny(normalized, "mobile", "edge", "browser", "desktop", "cpu") {
		context.DeploymentPriorities = append(context.DeploymentPriorities, "prefer compact CPU-friendly models and smaller image sizes")
		context.Constraints = append(context.Constraints, "consider model size and inference latency as first-class selection criteria")
		context.RankingWeights["model_size"] = 0.16
		context.RankingWeights["latency"] = 0.20
	}

	context.MetricPreferences = uniqueStrings(context.MetricPreferences)
	context.DeploymentPriorities = uniqueStrings(context.DeploymentPriorities)
	context.Constraints = uniqueStrings(context.Constraints)
	return context
}

func containsAny(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

func supportedModelCatalog() []agents.SupportedModelSpec {
	return []agents.SupportedModelSpec{
		{Name: "mobilenet_v3_small", Family: "mobilenet", DeploymentTier: "fast_live", DefaultImageSize: 224, MinRecommendedImages: 50, SupportsTransfer: true, ExpectedLatencyClass: "very_fast", RecommendedUse: "fast live baseline and compact champion refinement", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "mobilenet_v3_large", Family: "mobilenet", DeploymentTier: "fast_live", DefaultImageSize: 224, MinRecommendedImages: 80, SupportsTransfer: true, ExpectedLatencyClass: "fast", RecommendedUse: "higher-capacity MobileNet challenger for live use", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "efficientnet_b0", Family: "efficientnet", DeploymentTier: "fast_live", DefaultImageSize: 224, MinRecommendedImages: 80, SupportsTransfer: true, ExpectedLatencyClass: "fast", RecommendedUse: "strong quality/latency baseline", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "regnet_y_400mf", Family: "regnet", DeploymentTier: "fast_live", DefaultImageSize: 224, MinRecommendedImages: 100, SupportsTransfer: true, ExpectedLatencyClass: "fast", RecommendedUse: "compact architecture challenger", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "efficientnet_b1", Family: "efficientnet", DeploymentTier: "balanced", DefaultImageSize: 240, MinRecommendedImages: 150, SupportsTransfer: true, ExpectedLatencyClass: "medium", RecommendedUse: "balanced quality challenger", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "efficientnet_b2", Family: "efficientnet", DeploymentTier: "balanced", DefaultImageSize: 260, MinRecommendedImages: 250, SupportsTransfer: true, ExpectedLatencyClass: "medium", RecommendedUse: "stronger quality challenger when budget allows", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "resnet18", Family: "resnet", DeploymentTier: "balanced", DefaultImageSize: 224, MinRecommendedImages: 100, SupportsTransfer: true, ExpectedLatencyClass: "medium", RecommendedUse: "stable control architecture", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "resnet34", Family: "resnet", DeploymentTier: "balanced", DefaultImageSize: 224, MinRecommendedImages: 150, SupportsTransfer: true, ExpectedLatencyClass: "medium_slow", RecommendedUse: "larger ResNet comparison", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "convnext_tiny", Family: "convnext", DeploymentTier: "quality_challenger", DefaultImageSize: 224, MinRecommendedImages: 300, SupportsTransfer: true, ExpectedLatencyClass: "slow", RecommendedUse: "quality-first challenger", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "swin_t", Family: "swin", DeploymentTier: "quality_challenger", DefaultImageSize: 224, MinRecommendedImages: 500, SupportsTransfer: true, ExpectedLatencyClass: "slow", RecommendedUse: "transformer challenger for larger datasets", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
		{Name: "vit_b_16", Family: "vit", DeploymentTier: "quality_challenger", DefaultImageSize: 224, MinRecommendedImages: 800, SupportsTransfer: true, ExpectedLatencyClass: "slowest", RecommendedUse: "explicit quality-first experiments on larger datasets", SupportsFineTuneModes: []string{"head_only", "last_block", "full"}},
	}
}

func supportedModelNames() map[string]bool {
	out := map[string]bool{}
	for _, spec := range supportedModelCatalog() {
		out[strings.ToLower(spec.Name)] = true
	}
	return out
}

func plannerStrategyMemory(records []memory.AgentMemoryRecord) ([]agents.PlannerStrategyMemory, []agents.PlannerStrategyMemory) {
	successful := []agents.PlannerStrategyMemory{}
	failed := []agents.PlannerStrategyMemory{}
	for _, record := range records {
		if record.Kind != memory.KindPlanningOutcome {
			continue
		}
		entry := plannerStrategyMemoryFromRecord(record)
		switch entry.OutcomeStatus {
		case agents.ExperimentPlanningOutcomeImprovedChampion, agents.ExperimentPlanningOutcomeMinorImprovement:
			if len(successful) < 6 {
				successful = append(successful, entry)
			}
		case agents.ExperimentPlanningOutcomeNoImprovement, agents.ExperimentPlanningOutcomeFailed:
			if len(failed) < 6 {
				failed = append(failed, entry)
			}
		}
	}
	return successful, failed
}

func plannerRejectedOptions(records []memory.AgentMemoryRecord, diagnosis agents.PlannerDiagnosis) []agents.RejectedPlannerOption {
	out := []agents.RejectedPlannerOption{}
	failureModes := map[string]bool{}
	for _, mode := range diagnosis.RecommendedFailureModes {
		failureModes[strings.ToLower(strings.TrimSpace(mode))] = true
	}
	for _, record := range records {
		if record.Kind != memory.KindPlanningFeedback && record.Kind != memory.KindPlanningOutcome {
			continue
		}
		value, ok := record.Payload["rejected_options"]
		if !ok {
			continue
		}
		blob, err := json.Marshal(value)
		if err != nil {
			continue
		}
		var options []agents.RejectedPlannerOption
		if err := json.Unmarshal(blob, &options); err != nil {
			continue
		}
		for _, option := range options {
			if !rejectedOptionApplies(option, failureModes) {
				continue
			}
			out = append(out, option)
			if len(out) >= 8 {
				return out
			}
		}
	}
	return out
}

func rejectedOptionApplies(option agents.RejectedPlannerOption, failureModes map[string]bool) bool {
	if len(option.AppliesWhen) == 0 || len(failureModes) == 0 {
		return true
	}
	for _, condition := range option.AppliesWhen {
		normalized := strings.ToLower(strings.TrimSpace(condition))
		if failureModes[normalized] {
			return true
		}
	}
	return false
}

func plannerStrategyScorecards(scorecards []strategies.StrategyScorecard, datasetID string) []agents.PlannerStrategyScorecard {
	out := []agents.PlannerStrategyScorecard{}
	for _, scorecard := range scorecards {
		if scorecard.DatasetID != datasetID && len(out) >= 6 {
			continue
		}
		out = append(out, agents.PlannerStrategyScorecard{
			ID:               scorecard.ID,
			DatasetID:        scorecard.DatasetID,
			SourceDecisionID: scorecard.SourceDecisionID,
			SourcePlanID:     scorecard.SourcePlanID,
			FollowUpPlanID:   scorecard.FollowUpPlanID,
			StrategyType:     scorecard.StrategyType,
			PlanningMode:     scorecard.PlanningMode,
			DatasetTraits:    scorecard.DatasetTraits,
			ObjectiveProfile: scorecard.ObjectiveProfile,
			ProposedChanges:  scorecard.ProposedChanges,
			ExpectedDelta:    scorecard.ExpectedDelta,
			ActualDelta:      scorecard.ActualDelta,
			ConfidenceBefore: scorecard.ConfidenceBefore,
			ConfidenceAfter:  scorecard.ConfidenceAfter,
			CostUSD:          scorecard.CostUSD,
			RuntimeSeconds:   scorecard.RuntimeSeconds,
			Outcome:          scorecard.Outcome,
			Lesson:           scorecard.Lesson,
			Tags:             scorecard.Tags,
		})
		if len(out) >= 10 {
			break
		}
	}
	return out
}

func plannerStrategyMemoryFromRecord(record memory.AgentMemoryRecord) agents.PlannerStrategyMemory {
	bestModel := ""
	if champion, ok := experimentChampionFromPayload(record.Payload["actual_best_run"]); ok {
		bestModel = champion.Model
	}
	return agents.PlannerStrategyMemory{
		MemoryID:                record.ID,
		OutcomeStatus:           payloadString(record.Payload, "outcome_status"),
		Lesson:                  payloadString(record.Payload, "lesson"),
		BestModel:               bestModel,
		ActualDeltaVsChampion:   payloadFloat(record.Payload, "actual_delta_vs_champion"),
		ExpectedDeltaVsChampion: payloadFloat(record.Payload, "expected_delta_vs_champion"),
		TotalCostUSD:            payloadFloat(record.Payload, "total_cost_usd"),
		TotalRuntimeSeconds:     payloadFloat(record.Payload, "total_runtime_seconds"),
		ProposedModels:          proposedModelsFromPayload(record.Payload),
		Tags:                    record.Tags,
	}
}

func proposedModelsFromPayload(payload map[string]any) []string {
	value, ok := payload["proposed_experiments"]
	if !ok {
		return []string{}
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return []string{}
	}
	var experiments []plans.PlannedExperiment
	if err := json.Unmarshal(blob, &experiments); err != nil {
		return []string{}
	}
	models := []string{}
	for _, experiment := range experiments {
		if strings.TrimSpace(experiment.Model) != "" {
			models = append(models, experiment.Model)
		}
	}
	return uniqueStrings(models)
}

func agentDecisionByID(agentDecisions []decisions.AgentDecision, decisionID string) (decisions.AgentDecision, bool) {
	for _, decision := range agentDecisions {
		if decision.ID == decisionID {
			return decision, true
		}
	}
	return decisions.AgentDecision{}, false
}

func experimentPlanningOutcomeForPlan(
	sourceDecision decisions.AgentDecision,
	followUpPlan plans.ExperimentPlan,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
) (agents.ExperimentPlanningOutcome, error) {
	planSummaries := summariesForPlanID(summaries, followUpPlan.ID)
	proposedExperiments, err := plannedExperimentsFromPayload(sourceDecision.Payload)
	if err != nil {
		proposedExperiments = []plans.PlannedExperiment{}
	}

	baselineChampion := baselineChampionForPlannerOutcome(sourceDecision, followUpPlan, projectPlans, summaries)
	bestSummary, hasBest := bestSuccessfulTrainingSummary(followUpPlan.TargetMetric, planSummaries)

	var actualBest *agents.ExperimentChampion
	actualDelta := 0.0
	if hasBest {
		best := experimentChampionFromSummary(followUpPlan.TargetMetric, bestSummary)
		actualBest = &best
		if baselineChampion != nil {
			actualDelta = best.Score - baselineChampion.Score
		} else {
			actualDelta = best.Score
		}
	}

	expectedDelta := payloadFloat(sourceDecision.Payload, "expected_delta_vs_champion")
	metExpectedDelta := hasBest && actualDelta > plannerMinimumMeaningfulImprovement
	if expectedDelta > 0 {
		metExpectedDelta = hasBest && actualDelta >= expectedDelta
	}
	outcomeStatus := plannerOutcomeStatus(actualDelta, hasBest)
	outcome := agents.ExperimentPlanningOutcome{
		OutcomeType:             "planner_followup_result",
		OutcomeStatus:           outcomeStatus,
		SourceDecisionID:        sourceDecision.ID,
		SourcePlanID:            sourceDecision.PlanID,
		FollowUpPlanID:          followUpPlan.ID,
		BaselineChampion:        baselineChampion,
		ActualBestRun:           actualBest,
		ExpectedDeltaVsChampion: expectedDelta,
		ActualDeltaVsChampion:   actualDelta,
		MetExpectedDelta:        metExpectedDelta,
		TotalCostUSD:            totalSummaryCost(planSummaries),
		TotalRuntimeSeconds:     totalSummaryRuntime(planSummaries),
		TerminalRunCount:        len(planSummaries),
		SuccessfulRunCount:      successfulSummaryCount(planSummaries),
		FailedRunCount:          failedSummaryCount(planSummaries),
		ProposedExperiments:     proposedExperiments,
		CompletedAt:             time.Now().UTC(),
	}
	outcome.Lesson = plannerOutcomeLesson(followUpPlan.TargetMetric, outcome)
	return outcome, nil
}

func baselineChampionForPlannerOutcome(
	sourceDecision decisions.AgentDecision,
	followUpPlan plans.ExperimentPlan,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
) *agents.ExperimentChampion {
	if champion, ok := experimentChampionFromPayload(sourceDecision.Payload["current_champion"]); ok {
		return champion
	}
	if champion, ok := experimentChampionFromPayload(sourceDecision.Payload["source_plan_baseline_champion"]); ok {
		return champion
	}
	if summary, ok := bestSuccessfulTrainingSummaryBeforePlan(followUpPlan.TargetMetric, projectPlans, summaries, followUpPlan.ID); ok {
		champion := experimentChampionFromSummary(followUpPlan.TargetMetric, summary)
		return &champion
	}
	return nil
}

func experimentChampionFromPayload(value any) (*agents.ExperimentChampion, bool) {
	if value == nil {
		return nil, false
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return nil, false
	}
	var champion agents.ExperimentChampion
	if err := json.Unmarshal(blob, &champion); err != nil {
		return nil, false
	}
	if champion.JobID == "" {
		return nil, false
	}
	return &champion, true
}

func plannerOutcomeStatus(actualDelta float64, hasBest bool) string {
	if !hasBest {
		return agents.ExperimentPlanningOutcomeFailed
	}
	if actualDelta > plannerMinimumMeaningfulImprovement {
		return agents.ExperimentPlanningOutcomeImprovedChampion
	}
	if actualDelta > 0 {
		return agents.ExperimentPlanningOutcomeMinorImprovement
	}
	return agents.ExperimentPlanningOutcomeNoImprovement
}

func plannerOutcomeConfidence(outcome agents.ExperimentPlanningOutcome) float64 {
	switch outcome.OutcomeStatus {
	case agents.ExperimentPlanningOutcomeImprovedChampion:
		return minFloat(0.95, 0.70+maxFloat(0, outcome.ActualDeltaVsChampion)*4)
	case agents.ExperimentPlanningOutcomeMinorImprovement:
		return 0.55
	case agents.ExperimentPlanningOutcomeFailed:
		return 0.20
	default:
		return 0.35
	}
}

func plannerOutcomeLesson(targetMetric string, outcome agents.ExperimentPlanningOutcome) string {
	metric := normalizedPlannerTargetMetric(targetMetric)
	if outcome.OutcomeStatus == agents.ExperimentPlanningOutcomeFailed {
		return fmt.Sprintf("Planner follow-up plan %s produced no successful runs after %.3f total cost; avoid repeating this failed strategy without changing the setup.", outcome.FollowUpPlanID, outcome.TotalCostUSD)
	}
	bestModel := ""
	if outcome.ActualBestRun != nil {
		bestModel = outcome.ActualBestRun.Model
	}
	switch outcome.OutcomeStatus {
	case agents.ExperimentPlanningOutcomeImprovedChampion:
		return fmt.Sprintf("Planner follow-up plan %s improved the champion with %s by %.3f %s; similar strategy changes are worth reusing.", outcome.FollowUpPlanID, bestModel, outcome.ActualDeltaVsChampion, metric)
	case agents.ExperimentPlanningOutcomeMinorImprovement:
		return fmt.Sprintf("Planner follow-up plan %s only slightly improved the champion with %s by %.3f %s, below the meaningful threshold %.3f; treat this as weak evidence.", outcome.FollowUpPlanID, bestModel, outcome.ActualDeltaVsChampion, metric, plannerMinimumMeaningfulImprovement)
	default:
		return fmt.Sprintf("Planner follow-up plan %s failed to beat the prior champion; best run %s trailed by %.3f %s after %.3f total cost.", outcome.FollowUpPlanID, bestModel, outcome.ActualDeltaVsChampion, metric, outcome.TotalCostUSD)
	}
}

func plannerOutcomeTags(outcome agents.ExperimentPlanningOutcome) []string {
	tags := []string{"planner_outcome", outcome.OutcomeStatus}
	if outcome.MetExpectedDelta {
		tags = append(tags, "met_expected_delta")
	} else {
		tags = append(tags, "missed_expected_delta")
	}
	if outcome.ActualBestRun != nil && outcome.ActualBestRun.Model != "" {
		tags = append(tags, strings.ToLower(strings.TrimSpace(outcome.ActualBestRun.Model)))
	}
	return tags
}

func totalSummaryCost(summaries []runs.TrainingRunSummary) float64 {
	total := 0.0
	for _, summary := range summaries {
		total += summary.EstimatedCostUSD
	}
	return total
}

func totalSummaryRuntime(summaries []runs.TrainingRunSummary) float64 {
	total := 0.0
	for _, summary := range summaries {
		total += summary.RuntimeSeconds
	}
	return total
}

func successfulSummaryCount(summaries []runs.TrainingRunSummary) int {
	count := 0
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) == jobs.StatusSucceeded {
			count++
		}
	}
	return count
}

func failedSummaryCount(summaries []runs.TrainingRunSummary) int {
	count := 0
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) == jobs.StatusFailed {
			count++
		}
	}
	return count
}

func payloadString(payload map[string]any, key string) string {
	value, _ := payload[key].(string)
	return value
}

func payloadMap(payload map[string]any, key string) map[string]any {
	value, ok := payload[key]
	if !ok || value == nil {
		return map[string]any{}
	}
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return map[string]any{}
	}
	out := map[string]any{}
	if err := json.Unmarshal(blob, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func payloadFloat(payload map[string]any, key string) float64 {
	switch value := payload[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case json.Number:
		out, _ := value.Float64()
		return out
	default:
		return 0
	}
}

func maxFloat(left float64, right float64) float64 {
	if left > right {
		return left
	}
	return right
}

func minFloat(left float64, right float64) float64 {
	if left < right {
		return left
	}
	return right
}

func profileString(profile map[string]any, key string) string {
	value, _ := profile[key].(string)
	return value
}

func profileInt(profile map[string]any, key string) int {
	switch value := profile[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	case json.Number:
		out, _ := value.Int64()
		return int(out)
	default:
		return 0
	}
}

func profileFloat(profile map[string]any, key string) float64 {
	switch value := profile[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case json.Number:
		out, _ := value.Float64()
		return out
	default:
		return 0
	}
}

func profileBool(profile map[string]any, key string) bool {
	switch value := profile[key].(type) {
	case bool:
		return value
	case string:
		normalized := strings.ToLower(strings.TrimSpace(value))
		return normalized == "true" || normalized == "yes" || normalized == "1"
	default:
		return false
	}
}

func profileMap(profile map[string]any, key string) map[string]any {
	value, ok := profile[key].(map[string]any)
	if ok {
		return value
	}
	return map[string]any{}
}

func profileStringSlice(profile map[string]any, key string) []string {
	values, ok := profile[key].([]string)
	if ok {
		return values
	}
	rawValues, ok := profile[key].([]any)
	if !ok {
		return []string{}
	}
	out := []string{}
	for _, raw := range rawValues {
		if value, ok := raw.(string); ok {
			out = append(out, value)
		}
	}
	return out
}

func profileMapSlice(profile map[string]any, key string) []map[string]any {
	values, ok := profile[key].([]map[string]any)
	if ok {
		return values
	}
	rawValues, ok := profile[key].([]any)
	if !ok {
		return []map[string]any{}
	}
	out := []map[string]any{}
	for _, raw := range rawValues {
		if value, ok := raw.(map[string]any); ok {
			out = append(out, value)
		}
	}
	return out
}

func metadataBool(metadata map[string]any, key string) bool {
	switch value := metadata[key].(type) {
	case bool:
		return value
	case string:
		normalized := strings.ToLower(strings.TrimSpace(value))
		return normalized == "true" || normalized == "yes" || normalized == "1"
	default:
		return false
	}
}

func containsString(values []string, target string) bool {
	target = strings.ToLower(strings.TrimSpace(target))
	for _, value := range values {
		if strings.ToLower(strings.TrimSpace(value)) == target {
			return true
		}
	}
	return false
}

func uniqueStrings(values []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, trimmed)
	}
	return out
}

func jobsForPlan(projectJobs []jobs.ExperimentJob, planID string) []jobs.ExperimentJob {
	out := []jobs.ExperimentJob{}
	for _, job := range projectJobs {
		if configString(job.Config, "plan_id") == planID {
			out = append(out, job)
		}
	}
	return out
}

func summariesForPlanID(summaries []runs.TrainingRunSummary, planID string) []runs.TrainingRunSummary {
	out := []runs.TrainingRunSummary{}
	for _, summary := range summaries {
		if summary.PlanID == planID {
			out = append(out, summary)
		}
	}
	return out
}

func evaluationsForPlanID(evaluations []runs.TrainingRunEvaluation, planID string) []runs.TrainingRunEvaluation {
	out := []runs.TrainingRunEvaluation{}
	for _, evaluation := range evaluations {
		if evaluation.PlanID == planID {
			out = append(out, evaluation)
		}
	}
	return out
}

func experimentPlannerPerformanceContext(
	targetMetric string,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
	sourcePlanID string,
) (*agents.ExperimentChampion, *agents.ExperimentChampion, []agents.ExperimentRunDelta, int, []string) {
	championSummary, hasChampion := bestSuccessfulTrainingSummaryForObjective(targetMetric, summaries, evaluations, objectiveContext)
	if !hasChampion {
		return nil, nil, []agents.ExperimentRunDelta{}, 0, []string{"No successful champion run is available yet."}
	}

	champion := experimentChampionFromSummary(targetMetric, championSummary)
	baselineChampion := champion
	if baselineSummary, ok := bestSuccessfulTrainingSummaryBeforePlan(targetMetric, projectPlans, summaries, sourcePlanID); ok {
		baselineChampion = experimentChampionFromSummary(targetMetric, baselineSummary)
	}
	sourcePlanDeltas := experimentRunDeltasForPlan(targetMetric, summariesForPlanID(summaries, sourcePlanID), baselineChampion)
	noImprovementRounds := consecutiveNoImprovementFollowUpRounds(targetMetric, projectPlans, summaries)
	stopSignals := experimentPlannerStopSignals(champion, noImprovementRounds)
	return &champion, &baselineChampion, sourcePlanDeltas, noImprovementRounds, stopSignals
}

func experimentChampionFromSummary(targetMetric string, summary runs.TrainingRunSummary) agents.ExperimentChampion {
	return agents.ExperimentChampion{
		JobID:            summary.JobID,
		PlanID:           summary.PlanID,
		Model:            summary.Model,
		TargetMetric:     normalizedPlannerTargetMetric(targetMetric),
		Score:            plannerTargetMetricValue(targetMetric, summary),
		BestMacroF1:      summary.BestMacroF1,
		BestAccuracy:     summary.BestAccuracy,
		EstimatedCostUSD: summary.EstimatedCostUSD,
		RuntimeSeconds:   summary.RuntimeSeconds,
		EpochsCompleted:  summary.EpochsCompleted,
	}
}

func experimentRunDeltasForPlan(
	targetMetric string,
	summaries []runs.TrainingRunSummary,
	champion agents.ExperimentChampion,
) []agents.ExperimentRunDelta {
	out := make([]agents.ExperimentRunDelta, 0, len(summaries))
	for _, summary := range summaries {
		score := plannerTargetMetricValue(targetMetric, summary)
		out = append(out, agents.ExperimentRunDelta{
			JobID:                    summary.JobID,
			PlanID:                   summary.PlanID,
			Model:                    summary.Model,
			Status:                   summary.Status,
			TargetMetric:             normalizedPlannerTargetMetric(targetMetric),
			Score:                    score,
			BestMacroF1:              summary.BestMacroF1,
			BestAccuracy:             summary.BestAccuracy,
			EstimatedCostUSD:         summary.EstimatedCostUSD,
			RuntimeSeconds:           summary.RuntimeSeconds,
			EpochsCompleted:          summary.EpochsCompleted,
			ChampionJobID:            champion.JobID,
			DeltaScoreVsChampion:     score - champion.Score,
			DeltaCostVsChampion:      summary.EstimatedCostUSD - champion.EstimatedCostUSD,
			DeltaRuntimeVsChampion:   summary.RuntimeSeconds - champion.RuntimeSeconds,
			MeaningfullyImprovedOver: score > champion.Score+plannerMinimumMeaningfulImprovement,
		})
	}
	return out
}

func consecutiveNoImprovementFollowUpRounds(
	targetMetric string,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
) int {
	orderedPlans := append([]plans.ExperimentPlan(nil), projectPlans...)
	sort.Slice(orderedPlans, func(i, j int) bool {
		if orderedPlans[i].CreatedAt.Equal(orderedPlans[j].CreatedAt) {
			return orderedPlans[i].ID < orderedPlans[j].ID
		}
		return orderedPlans[i].CreatedAt.Before(orderedPlans[j].CreatedAt)
	})

	hasChampion := false
	championScore := 0.0
	noImprovementRounds := 0
	for _, plan := range orderedPlans {
		planSummaries := summariesForPlanID(summaries, plan.ID)
		if !planTrainingRunsComplete(plan, planSummaries) {
			continue
		}

		best, ok := bestSuccessfulTrainingSummary(targetMetric, planSummaries)
		if !ok {
			if plan.SourceDecisionID != "" && hasChampion {
				noImprovementRounds++
			}
			continue
		}

		score := plannerTargetMetricValue(targetMetric, best)
		if !hasChampion {
			hasChampion = true
			championScore = score
			continue
		}

		if plan.SourceDecisionID != "" {
			if score > championScore+plannerMinimumMeaningfulImprovement {
				noImprovementRounds = 0
			} else {
				noImprovementRounds++
			}
		}
		if score > championScore {
			championScore = score
		}
	}
	return noImprovementRounds
}

func experimentPlannerStopSignals(champion agents.ExperimentChampion, noImprovementRounds int) []string {
	signals := []string{
		fmt.Sprintf("Current champion is %s (%s) with %s %.3f.", champion.JobID, champion.Model, champion.TargetMetric, champion.Score),
	}
	if noImprovementRounds > 0 {
		signals = append(signals, fmt.Sprintf("%d consecutive completed follow-up plan(s) did not improve the champion by at least %.3f.", noImprovementRounds, plannerMinimumMeaningfulImprovement))
	}
	if noImprovementRounds >= plannerNoImprovementRoundsToSelect {
		signals = append(signals, "Backend policy will select the current champion instead of scheduling another follow-up unless a future run meaningfully improves it.")
	}
	return signals
}

func bestSuccessfulTrainingSummary(targetMetric string, summaries []runs.TrainingRunSummary) (runs.TrainingRunSummary, bool) {
	var best runs.TrainingRunSummary
	hasBest := false
	bestScore := 0.0
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) != jobs.StatusSucceeded {
			continue
		}
		score := plannerTargetMetricValue(targetMetric, summary)
		if !hasBest || score > bestScore || (score == bestScore && summary.EstimatedCostUSD < best.EstimatedCostUSD) {
			best = summary
			bestScore = score
			hasBest = true
		}
	}
	return best, hasBest
}

func bestSuccessfulTrainingSummaryForObjective(
	targetMetric string,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
) (runs.TrainingRunSummary, bool) {
	if len(evaluations) == 0 {
		return bestSuccessfulTrainingSummary(targetMetric, summaries)
	}
	evaluationsByJob := map[string]runs.TrainingRunEvaluation{}
	for _, evaluation := range evaluations {
		evaluationsByJob[evaluation.JobID] = evaluation
	}

	var best runs.TrainingRunSummary
	bestScore := 0.0
	hasBest := false
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) != jobs.StatusSucceeded {
			continue
		}
		score := holisticRunScore(targetMetric, summary, evaluationsByJob[summary.JobID], objectiveContext)
		if !hasBest || score > bestScore || (score == bestScore && summary.EstimatedCostUSD < best.EstimatedCostUSD) {
			best = summary
			bestScore = score
			hasBest = true
		}
	}
	return best, hasBest
}

func holisticRunScore(targetMetric string, summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation, objectiveContext agents.ProjectObjectiveContext) float64 {
	quality := plannerTargetMetricValue(targetMetric, summary)
	if overall := payloadFloat(evaluation.HolisticScores, "overall_score"); overall > 0 {
		quality = maxFloat(quality, overall)
	}
	latencyScore := 0.5
	if latencyMS := payloadFloat(evaluation.ModelProfile, "estimated_latency_ms"); latencyMS > 0 {
		latencyScore = maxFloat(0, minFloat(1, 1-latencyMS/160))
	}
	costScore := maxFloat(0, minFloat(1, 1-summary.EstimatedCostUSD/0.25))
	runtimeScore := maxFloat(0, minFloat(1, 1-summary.RuntimeSeconds/1800))

	if objectiveContext.PrimaryObjective == "low_latency_live_service" {
		return quality*0.62 + latencyScore*0.24 + costScore*0.08 + runtimeScore*0.06
	}
	if objectiveContext.PrimaryObjective == "budget_sensitive" {
		return quality*0.68 + costScore*0.18 + latencyScore*0.08 + runtimeScore*0.06
	}
	if objectiveContext.PrimaryObjective == "quality_first" {
		return quality*0.82 + latencyScore*0.07 + costScore*0.06 + runtimeScore*0.05
	}
	return quality*0.74 + latencyScore*0.12 + costScore*0.08 + runtimeScore*0.06
}

func bestSuccessfulTrainingSummaryBeforePlan(
	targetMetric string,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	sourcePlanID string,
) (runs.TrainingRunSummary, bool) {
	orderedPlans := append([]plans.ExperimentPlan(nil), projectPlans...)
	sort.Slice(orderedPlans, func(i, j int) bool {
		if orderedPlans[i].CreatedAt.Equal(orderedPlans[j].CreatedAt) {
			return orderedPlans[i].ID < orderedPlans[j].ID
		}
		return orderedPlans[i].CreatedAt.Before(orderedPlans[j].CreatedAt)
	})

	priorPlanIDs := map[string]bool{}
	for _, plan := range orderedPlans {
		if plan.ID == sourcePlanID {
			break
		}
		priorPlanIDs[plan.ID] = true
	}

	priorSummaries := []runs.TrainingRunSummary{}
	for _, summary := range summaries {
		if priorPlanIDs[summary.PlanID] {
			priorSummaries = append(priorSummaries, summary)
		}
	}
	return bestSuccessfulTrainingSummary(targetMetric, priorSummaries)
}

func plannerTargetMetricValue(targetMetric string, summary runs.TrainingRunSummary) float64 {
	switch normalizedPlannerTargetMetric(targetMetric) {
	case "accuracy":
		return summary.BestAccuracy
	default:
		return summary.BestMacroF1
	}
}

func normalizedPlannerTargetMetric(targetMetric string) string {
	normalized := strings.ToLower(strings.TrimSpace(targetMetric))
	if normalized == "" {
		return "macro_f1"
	}
	return normalized
}

func experimentSignaturesForPlans(projectPlans []plans.ExperimentPlan) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, plan := range projectPlans {
		for _, experiment := range plan.Experiments {
			signature := experimentSignature(experiment)
			if seen[signature] {
				continue
			}
			seen[signature] = true
			out = append(out, signature)
		}
	}
	sort.Strings(out)
	return out
}

func validateNovelProposedExperiments(experiments []plans.PlannedExperiment, projectPlans []plans.ExperimentPlan) error {
	existing := map[string]bool{}
	for _, plan := range projectPlans {
		for _, experiment := range plan.Experiments {
			existing[experimentSignature(experiment)] = true
		}
	}

	proposed := map[string]bool{}
	for index, experiment := range experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			return err
		}
		signature := experimentSignature(experiment)
		if existing[signature] {
			return fmt.Errorf("%w: proposed experiment %d duplicates an existing experiment signature %s", store.ErrInvalidRequest, index, signature)
		}
		if proposed[signature] {
			return fmt.Errorf("%w: proposed experiment %d duplicates another proposed experiment signature %s", store.ErrInvalidRequest, index, signature)
		}
		proposed[signature] = true
	}
	return nil
}

func experimentSignature(experiment plans.PlannedExperiment) string {
	augmentationBlob, _ := json.Marshal(experiment.Augmentation)
	preprocessingBlob, _ := json.Marshal(experiment.Preprocessing)
	return strings.Join([]string{
		strings.ToLower(strings.TrimSpace(experiment.Template)),
		strings.ToLower(strings.TrimSpace(experiment.Model)),
		strconv.Itoa(experiment.Epochs),
		strconv.Itoa(experiment.BatchSize),
		strconv.FormatFloat(experiment.LearningRate, 'g', -1, 64),
		strconv.Itoa(experiment.ImageSize),
		strings.ToLower(strings.TrimSpace(experiment.ResolutionStrategy)),
		string(preprocessingBlob),
		strings.ToLower(strings.TrimSpace(experiment.Optimizer)),
		strings.ToLower(strings.TrimSpace(experiment.Scheduler)),
		strconv.FormatFloat(experiment.WeightDecay, 'g', -1, 64),
		string(augmentationBlob),
		strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy)),
		strings.ToLower(strings.TrimSpace(experiment.ClassBalancing)),
		strings.ToLower(strings.TrimSpace(experiment.SamplingStrategy)),
		strconv.Itoa(experiment.EarlyStoppingPatience),
		strconv.FormatBool(experiment.Pretrained),
		strconv.FormatBool(experiment.FreezeBackbone),
		strings.ToLower(strings.TrimSpace(experiment.FineTuneStrategy)),
	}, ":")
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
	for index, experiment := range experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			writeStoreError(c, err)
			return
		}
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
		req := s.defaultExecuteExperimentPlanRequest()
		executionResult, err := s.executeStoredExperimentPlan(plan.ID, req)
		if err != nil {
			return err
		}
		return s.recordAutomaticExecutionQueued(plan, req, executionResult.Jobs)
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
		addOptionalExperimentConfig(config, experiment)

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

func addOptionalExperimentConfig(config map[string]any, experiment plans.PlannedExperiment) {
	if experiment.ImageSize > 0 {
		config["image_size"] = experiment.ImageSize
	}
	if experiment.ResolutionStrategy != "" {
		config["resolution_strategy"] = experiment.ResolutionStrategy
	}
	if experiment.Preprocessing != nil {
		config["preprocessing"] = experiment.Preprocessing
	}
	if experiment.Optimizer != "" {
		config["optimizer"] = experiment.Optimizer
	}
	if experiment.Scheduler != "" {
		config["scheduler"] = experiment.Scheduler
	}
	if experiment.WeightDecay > 0 {
		config["weight_decay"] = experiment.WeightDecay
	}
	if len(experiment.Augmentation) > 0 {
		config["augmentation"] = experiment.Augmentation
	}
	if experiment.AugmentationPolicy != "" {
		config["augmentation_policy"] = experiment.AugmentationPolicy
	}
	if experiment.ClassBalancing != "" {
		config["class_balancing"] = experiment.ClassBalancing
	}
	if experiment.SamplingStrategy != "" {
		config["sampling_strategy"] = experiment.SamplingStrategy
	}
	if experiment.EarlyStoppingPatience > 0 {
		config["early_stopping_patience"] = experiment.EarlyStoppingPatience
	}
	if experiment.Strategy != "" {
		config["strategy"] = experiment.Strategy
	}
	if experiment.Pretrained {
		config["pretrained"] = experiment.Pretrained
	}
	if experiment.FreezeBackbone {
		config["freeze_backbone"] = experiment.FreezeBackbone
	}
	if experiment.FineTuneStrategy != "" {
		config["fine_tune_strategy"] = experiment.FineTuneStrategy
	}
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
	if len(experiments) > maxLLMPlannerExperiments {
		return nil, fmt.Errorf("%w: proposed_experiments has %d experiments, max is %d", store.ErrInvalidRequest, len(experiments), maxLLMPlannerExperiments)
	}

	for index, experiment := range experiments {
		if err := validatePlannedExperiment(experiment, index); err != nil {
			return nil, err
		}
	}

	return experiments, nil
}

func validatePlannedExperiment(experiment plans.PlannedExperiment, index int) error {
	if strings.TrimSpace(experiment.Template) == "" {
		return fmt.Errorf("%w: proposed experiment %d is missing template", store.ErrInvalidRequest, index)
	}
	if strings.TrimSpace(experiment.Model) == "" {
		return fmt.Errorf("%w: proposed experiment %d is missing model", store.ErrInvalidRequest, index)
	}
	if !supportedModelNames()[strings.ToLower(strings.TrimSpace(experiment.Model))] {
		return fmt.Errorf("%w: proposed experiment %d uses unsupported model %q", store.ErrInvalidRequest, index, experiment.Model)
	}
	if experiment.Epochs < 1 || experiment.Epochs > 100 {
		return fmt.Errorf("%w: proposed experiment %d must have 1-100 epochs", store.ErrInvalidRequest, index)
	}
	if experiment.BatchSize < 1 || experiment.BatchSize > 512 {
		return fmt.Errorf("%w: proposed experiment %d must have batch_size 1-512", store.ErrInvalidRequest, index)
	}
	if experiment.LearningRate <= 0 || experiment.LearningRate > 1 {
		return fmt.Errorf("%w: proposed experiment %d must have learning_rate in (0, 1]", store.ErrInvalidRequest, index)
	}
	if experiment.ImageSize < 0 || experiment.ImageSize > 1024 {
		return fmt.Errorf("%w: proposed experiment %d image_size must be at most 1024", store.ErrInvalidRequest, index)
	}
	if experiment.ResolutionStrategy != "" && !allowedExperimentValue(experiment.ResolutionStrategy, allowedResolutionStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported resolution_strategy %q", store.ErrInvalidRequest, index, experiment.ResolutionStrategy)
	}
	if experiment.Preprocessing != nil {
		if err := validatePreprocessingConfig(*experiment.Preprocessing, index); err != nil {
			return err
		}
	}
	if experiment.Optimizer != "" && !allowedExperimentValue(experiment.Optimizer, allowedOptimizers()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported optimizer %q", store.ErrInvalidRequest, index, experiment.Optimizer)
	}
	if experiment.Scheduler != "" && !allowedExperimentValue(experiment.Scheduler, allowedSchedulers()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported scheduler %q", store.ErrInvalidRequest, index, experiment.Scheduler)
	}
	if experiment.WeightDecay < 0 || experiment.WeightDecay > 1 {
		return fmt.Errorf("%w: proposed experiment %d weight_decay must be between 0 and 1", store.ErrInvalidRequest, index)
	}
	if err := validateAugmentationConfig(experiment.Augmentation, index); err != nil {
		return err
	}
	if experiment.AugmentationPolicy != "" && !allowedExperimentValue(experiment.AugmentationPolicy, allowedAugmentationPolicies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported augmentation_policy %q", store.ErrInvalidRequest, index, experiment.AugmentationPolicy)
	}
	if experiment.ClassBalancing != "" && !allowedExperimentValue(experiment.ClassBalancing, allowedClassBalancingStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported class_balancing %q", store.ErrInvalidRequest, index, experiment.ClassBalancing)
	}
	if experiment.SamplingStrategy != "" && !allowedExperimentValue(experiment.SamplingStrategy, allowedSamplingStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported sampling_strategy %q", store.ErrInvalidRequest, index, experiment.SamplingStrategy)
	}
	if experiment.EarlyStoppingPatience < 0 || experiment.EarlyStoppingPatience > 50 {
		return fmt.Errorf("%w: proposed experiment %d early_stopping_patience must be between 0 and 50", store.ErrInvalidRequest, index)
	}
	if experiment.FineTuneStrategy != "" {
		switch strings.ToLower(strings.TrimSpace(experiment.FineTuneStrategy)) {
		case "head_only", "last_block", "full":
		default:
			return fmt.Errorf("%w: proposed experiment %d has unsupported fine_tune_strategy %q", store.ErrInvalidRequest, index, experiment.FineTuneStrategy)
		}
	}
	return nil
}

func validatePreprocessingConfig(preprocessing plans.Preprocessing, index int) error {
	if preprocessing.ResizeStrategy != "" && !allowedExperimentValue(preprocessing.ResizeStrategy, allowedResizeStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.resize_strategy %q", store.ErrInvalidRequest, index, preprocessing.ResizeStrategy)
	}
	if preprocessing.Normalization != "" && !allowedExperimentValue(preprocessing.Normalization, allowedNormalizations()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.normalization %q", store.ErrInvalidRequest, index, preprocessing.Normalization)
	}
	if preprocessing.CropStrategy != "" && !allowedExperimentValue(preprocessing.CropStrategy, allowedCropStrategies()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.crop_strategy %q", store.ErrInvalidRequest, index, preprocessing.CropStrategy)
	}
	if preprocessing.BBoxMode != "" && !allowedExperimentValue(preprocessing.BBoxMode, allowedBBoxModes()) {
		return fmt.Errorf("%w: proposed experiment %d has unsupported preprocessing.bbox_mode %q", store.ErrInvalidRequest, index, preprocessing.BBoxMode)
	}
	return nil
}

func validateAugmentationConfig(augmentation map[string]any, index int) error {
	for key, value := range augmentation {
		normalized := strings.ToLower(strings.TrimSpace(key))
		if !allowedExperimentValue(normalized, allowedAugmentationKeys()) {
			return fmt.Errorf("%w: proposed experiment %d has unsupported augmentation key %q", store.ErrInvalidRequest, index, key)
		}
		switch value.(type) {
		case bool, int, int64, float64, string:
		default:
			return fmt.Errorf("%w: proposed experiment %d augmentation.%s must be a bool, number, or string", store.ErrInvalidRequest, index, key)
		}
	}
	return nil
}

func allowedExperimentValue(value string, allowed map[string]bool) bool {
	return allowed[strings.ToLower(strings.TrimSpace(value))]
}

func allowedOptimizers() map[string]bool {
	return map[string]bool{"adamw": true, "adam": true, "sgd": true}
}

func allowedSchedulers() map[string]bool {
	return map[string]bool{"none": true, "cosine": true, "step": true}
}

func allowedResolutionStrategies() map[string]bool {
	return map[string]bool{
		"fixed":                    true,
		"low_latency":              true,
		"compare_224_256":          true,
		"high_resolution_ablation": true,
	}
}

func allowedResizeStrategies() map[string]bool {
	return map[string]bool{
		"squash":                 true,
		"preserve_aspect_pad":    true,
		"center_crop":            true,
		"random_resized_crop":    true,
		"bbox_crop_if_available": true,
	}
}

func allowedNormalizations() map[string]bool {
	return map[string]bool{"imagenet": true, "dataset": true, "none": true}
}

func allowedCropStrategies() map[string]bool {
	return map[string]bool{
		"none":                   true,
		"center_crop":            true,
		"random_resized_crop":    true,
		"bbox_crop_if_available": true,
		"bbox_crop_ablation":     true,
	}
}

func allowedBBoxModes() map[string]bool {
	return map[string]bool{
		"ignore":                      true,
		"crop_if_available":           true,
		"crop_and_compare_full_image": true,
		"use_boxes_as_metadata":       true,
	}
}

func allowedAugmentationPolicies() map[string]bool {
	return map[string]bool{"none": true, "light": true, "moderate": true, "strong": true, "custom": true}
}

func allowedAugmentationKeys() map[string]bool {
	return map[string]bool{
		"horizontal_flip": true,
		"vertical_flip":   true,
		"color_jitter":    true,
		"random_crop":     true,
		"random_rotation": true,
		"random_erasing":  true,
	}
}

func allowedClassBalancingStrategies() map[string]bool {
	return map[string]bool{
		"none":                    true,
		"weighted_loss":           true,
		"class_weighted_loss":     true,
		"class_balanced_sampler":  true,
		"weighted_random_sampler": true,
		"focal_loss":              true,
	}
}

func allowedSamplingStrategies() map[string]bool {
	return map[string]bool{
		"none":                    true,
		"class_balanced_sampler":  true,
		"weighted_random_sampler": true,
	}
}

type visualExemplarCaps struct {
	MaxTotalImages int   `json:"max_total_images"`
	MaxPerClass    int   `json:"max_per_class"`
	MaxBytes       int64 `json:"max_bytes"`
}

func visualExemplarCapsFromQuery(c *gin.Context, defaultTotal int, defaultPerClass int, defaultBytes int64) visualExemplarCaps {
	caps := visualExemplarCaps{
		MaxTotalImages: defaultTotal,
		MaxPerClass:    defaultPerClass,
		MaxBytes:       defaultBytes,
	}
	caps.MaxTotalImages = clampQueryInt(c, "max_total_images", caps.MaxTotalImages, 1, 50)
	caps.MaxPerClass = clampQueryInt(c, "max_per_class", caps.MaxPerClass, 1, 8)
	caps.MaxBytes = int64(clampQueryInt(c, "max_bytes", int(caps.MaxBytes), 1, 5_000_000))
	return caps
}

func clampQueryInt(c *gin.Context, key string, fallback int, minValue int, maxValue int) int {
	raw := strings.TrimSpace(c.Query(key))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func cappedVisualExemplars(profile map[string]any, caps visualExemplarCaps, keys ...string) []datasets.VisualExemplar {
	out := []datasets.VisualExemplar{}
	perClass := map[string]int{}
	var totalBytes int64

	for _, key := range keys {
		entries := profileEntries(profile[key])
		if len(entries) == 0 {
			continue
		}
		for _, entry := range entries {
			exemplar, ok := visualExemplarFromProfileEntry(entry)
			if !ok || exemplar.URI == "" {
				continue
			}
			className := exemplar.ClassName
			if className == "" {
				className = exemplar.Label
			}
			if className == "" {
				className = "unknown"
			}
			if perClass[className] >= caps.MaxPerClass {
				continue
			}
			nextBytes := totalBytes + maxInt64(exemplar.SizeBytes, 0)
			if exemplar.SizeBytes > 0 && nextBytes > caps.MaxBytes {
				continue
			}
			exemplar.ClassName = className
			out = append(out, exemplar)
			perClass[className]++
			if exemplar.SizeBytes > 0 {
				totalBytes = nextBytes
			}
			if len(out) >= caps.MaxTotalImages {
				return out
			}
		}
	}
	return out
}

func profileEntries(value any) []any {
	switch entries := value.(type) {
	case []any:
		return entries
	case []map[string]any:
		out := make([]any, 0, len(entries))
		for _, entry := range entries {
			out = append(out, entry)
		}
		return out
	default:
		return nil
	}
}

func visualExemplarFromProfileEntry(entry any) (datasets.VisualExemplar, bool) {
	blob, err := json.Marshal(entry)
	if err != nil {
		return datasets.VisualExemplar{}, false
	}
	var exemplar datasets.VisualExemplar
	if err := json.Unmarshal(blob, &exemplar); err != nil {
		return datasets.VisualExemplar{}, false
	}
	if exemplar.URI == "" {
		var raw map[string]any
		if err := json.Unmarshal(blob, &raw); err == nil {
			exemplar.URI = firstString(raw, "image_uri", "url", "path", "storage_uri")
			if exemplar.ClassName == "" {
				exemplar.ClassName = firstString(raw, "class", "class_name", "label")
			}
		}
	}
	return exemplar, true
}

func championDemoImageMetadata(profile map[string]any, imageURI string) (string, string, map[string]any, bool) {
	imageURI = strings.TrimSpace(imageURI)
	if imageURI == "" {
		return "", "", nil, false
	}
	for _, key := range []string{"demo_images", "visual_exemplars", "exemplars"} {
		for _, entry := range profileEntries(profile[key]) {
			exemplar, ok := visualExemplarFromProfileEntry(entry)
			if !ok || strings.TrimSpace(exemplar.URI) != imageURI {
				continue
			}
			trueLabel := exemplar.Label
			if trueLabel == "" {
				trueLabel = exemplar.ClassName
			}
			metadata := map[string]any{}
			for key, value := range exemplar.Metadata {
				metadata[key] = value
			}
			if exemplar.ClassName != "" {
				metadata["class_name"] = exemplar.ClassName
			}
			if exemplar.Label != "" {
				metadata["label"] = exemplar.Label
			}
			if exemplar.Split != "" {
				metadata["split"] = exemplar.Split
			}
			if exemplar.Width > 0 {
				metadata["width"] = exemplar.Width
			}
			if exemplar.Height > 0 {
				metadata["height"] = exemplar.Height
			}
			if exemplar.SizeBytes > 0 {
				metadata["size_bytes"] = exemplar.SizeBytes
			}
			if exemplar.MimeType != "" {
				metadata["mime_type"] = exemplar.MimeType
			}
			if exemplar.Description != "" {
				metadata["description"] = exemplar.Description
			}
			return exemplar.ID, trueLabel, metadata, true
		}
	}
	return "", "", nil, false
}

func normalizeChampionExportFormat(format string) string {
	format = strings.ToLower(strings.TrimSpace(format))
	if format == "" {
		format = "onnx"
	}
	switch format {
	case "onnx", "torchscript", "pytorch", "safetensors":
		return format
	default:
		return ""
	}
}

func championArtifactURI(deploymentProfile map[string]any) string {
	return firstString(deploymentProfile, "artifact_uri", "model_artifact_uri", "export_artifact_uri", "checkpoint_uri")
}

func championExportMetadata(champion runs.ProjectChampion, format string, requestMetadata map[string]any) map[string]any {
	metadata := map[string]any{
		"format":             format,
		"source_job_id":      champion.JobID,
		"selection_reason":   champion.SelectionReason,
		"metrics":            champion.Metrics,
		"evaluation":         champion.Evaluation,
		"deployment_profile": champion.DeploymentProfile,
	}
	for key, value := range requestMetadata {
		metadata[key] = value
	}
	return metadata
}

func firstString(values map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := values[key]; ok {
			if text, ok := value.(string); ok && strings.TrimSpace(text) != "" {
				return strings.TrimSpace(text)
			}
		}
	}
	return ""
}

func maxInt64(a int64, b int64) int64 {
	if a > b {
		return a
	}
	return b
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

func maxAutoWorkersFromEnv() int {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_MAX_AUTO_WORKERS"))
	if value == "" {
		return 4
	}
	count, err := strconv.Atoi(value)
	if err != nil || count < 1 {
		return 4
	}
	return count
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
