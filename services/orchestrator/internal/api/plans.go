package api

import (
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/store"
)

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
	Provider          string `json:"provider"`
	GPUType           string `json:"gpu_type"`
	MaxConcurrentJobs int    `json:"max_concurrent_jobs"`
}

type executeExperimentPlanResponse struct {
	Plan              plans.ExperimentPlan         `json:"plan"`
	Jobs              []jobs.ExperimentJob         `json:"jobs"`
	CostPolicy        map[string]any               `json:"cost_policy,omitempty"`
	WorkerRequirement *execution.WorkerRequirement `json:"worker_requirement,omitempty"`
}

type cancelExecutionRequest struct {
	Reason               string `json:"reason"`
	PromoteBestAvailable bool   `json:"promote_best_available"`
	TerminateRemoteWork  bool   `json:"terminate_remote_work"`
}

type cancelModalCallResult struct {
	JobID                     string `json:"job_id"`
	TrainingAttemptID         string `json:"training_attempt_id,omitempty"`
	ModalFunctionCallObjectID string `json:"modal_function_call_object_id,omitempty"`
	ModalFunctionCallID       string `json:"modal_function_call_id,omitempty"`
	ModalInputID              string `json:"modal_input_id,omitempty"`
	CancelStatus              string `json:"cancel_status"`
}

type cancelBestAvailableModel struct {
	JobID                   string  `json:"job_id,omitempty"`
	Exportable              bool    `json:"exportable"`
	ChampionSelectionSource string  `json:"champion_selection_source,omitempty"`
	Model                   string  `json:"model,omitempty"`
	Score                   float64 `json:"score,omitempty"`
	Reason                  string  `json:"reason,omitempty"`
}

type cancelExecutionResponse struct {
	ExecutionID                string                        `json:"execution_id"`
	Target                     map[string]string             `json:"target"`
	Status                     string                        `json:"status"`
	Message                    string                        `json:"message"`
	QueuedJobsCancelled        int                           `json:"queued_jobs_cancelled"`
	ActiveJobsMarkedCancelling int                           `json:"active_jobs_marked_cancelling"`
	AlreadyTerminalJobs        int                           `json:"already_terminal_jobs"`
	ModalCalls                 []cancelModalCallResult       `json:"modal_calls"`
	WorkerRequirements         []execution.WorkerRequirement `json:"worker_requirements"`
	BestAvailableModel         cancelBestAvailableModel      `json:"best_available_model"`
	Compatibility              map[string]bool               `json:"compatibility"`
	Jobs                       []jobs.ExperimentJob          `json:"jobs,omitempty"`
	Plans                      []cancelExecutionResponse     `json:"plans,omitempty"`
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

	metadataSummary, err := s.activeAgentSafeDatasetMetadataSummary(dataset)
	if err != nil {
		return err
	}
	if len(metadataSummary) > 0 {
		dataset.Profile = profileWithAgentSafeMetadataSummary(dataset.Profile, metadataSummary)
	}
	recommendation, err := agents.NewDatasetPlanner().BuildExperimentPlan(project, dataset, agents.PlanPreferences{
		Priority: agents.PriorityBalanced,
	})
	if err != nil {
		return fmt.Errorf("%w: %s", store.ErrInvalidRequest, err.Error())
	}
	experiments, automlWarnings, err := s.prepareAutoMLExperimentsForProject(project.ID, recommendation.Experiments)
	if err != nil {
		return err
	}
	warnings := append([]string(nil), recommendation.Warnings...)
	warnings = append(warnings, automlWarnings...)

	plan, err := s.store.CreateExperimentPlan(
		project.ID,
		dataset.ID,
		recommendation.TargetMetric,
		recommendation.RecommendedWorkers,
		recommendation.EstimatedMinutes,
		experiments,
		warnings,
		"",
	)
	if err != nil {
		return err
	}
	if err := s.persistAutoMLForPlan(plan); err != nil {
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
	if err := s.validateFollowUpPlanCanExecute(plan); err != nil {
		return executeExperimentPlanResponse{}, err
	}

	provider := req.Provider
	if provider == "" {
		provider = "local"
	}
	provider = normalizeTrainingProvider(provider)
	if err := validateTrainingProviderConfigured(provider); err != nil {
		return executeExperimentPlanResponse{}, err
	}
	req.MaxConcurrentJobs = effectiveExecutionMaxConcurrentJobs(provider, req.MaxConcurrentJobs)
	dataset, err := s.store.GetDataset(plan.DatasetID)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}
	costPolicy, err := s.costPolicyForPlan(plan)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}
	materializationPolicy := datasetMaterializationPolicy(dataset, provider, s.targetWorkerCountForPlanExecution(plan, len(plan.Experiments), req.MaxConcurrentJobs))
	if costPolicy.Enabled && costPolicy.MaxConcurrentJobs > 0 && materializationPolicy.MaxConcurrentJobs > costPolicy.MaxConcurrentJobs {
		materializationPolicy.MaxConcurrentJobs = costPolicy.MaxConcurrentJobs
	}

	existingJobs, err := s.store.ListProjectJobs(plan.ProjectID)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}
	automlSuggestions := s.automlSuggestionsByExperiment(plan.ID)

	jobsByExperiment := map[int]jobs.ExperimentJob{}
	for _, job := range existingJobs {
		if job.Template != jobs.TemplateTrainExperiment && job.Template != jobs.TemplateLabelQualityAudit {
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
		if err := validateExperimentDatasetCompatibility(experiment, dataset, index); err != nil {
			return executeExperimentPlanResponse{}, err
		}
		if job, ok := jobsByExperiment[index]; ok {
			out = append(out, job)
			continue
		}

		jobTemplate := experimentExecutionTemplate(experiment)
		tier := trainingTierForExperiment(experiment)
		if costPolicy.Enabled && jobTemplate == jobs.TemplateTrainExperiment {
			allowed, reason := costPolicy.AllowTrainingJob(tier)
			if !allowed {
				costPolicy.Skip(index, tier, reason)
				continue
			}
		}
		config := map[string]any{
			"plan_id":             plan.ID,
			"dataset_id":          plan.DatasetID,
			"experiment_index":    index,
			"experiment_template": experiment.Template,
			"target_metric":       plan.TargetMetric,
			"provider":            provider,
			"gpu_type":            req.GPUType,
		}
		addDatasetMaterializationConfig(config, materializationPolicy)
		if jobTemplate == jobs.TemplateTrainExperiment {
			if costPolicy.Enabled && tier != "" {
				config["training_tier"] = tier
			}
			config["model"] = experiment.Model
			config["epochs"] = experiment.Epochs
			config["batch_size"] = experiment.BatchSize
			config["learning_rate"] = experiment.LearningRate
			addCostPolicyConfig(config, costPolicy, tier)
			addPersistentGPUConfig(config, provider, dataset, materializationPolicy)
			addModalPreviewTrainingTierConfig(config, provider)
			if modelSpec, ok := supportedModelSpecByName(experiment.Model); ok {
				config["task_type"] = modelSpec.TaskType
				config["model_kind"] = modelSpec.ModelKind
				if modelSpec.PretrainedWeights != "" {
					config["pretrained_weights"] = modelSpec.PretrainedWeights
				}
				if modelSpec.DefaultImageSize > 0 && config["image_size"] == nil {
					config["image_size"] = modelSpec.DefaultImageSize
				}
				if modelSpec.TaskType == "object_detection" {
					if classNames := profileStringSlice(dataset.Profile, "class_names"); len(classNames) > 0 {
						config["class_names"] = classNames
					}
					if yoloSummary := profileMap(dataset.Profile, "yolo_summary"); len(yoloSummary) > 0 {
						config["yolo_summary"] = safeYOLOSummary(yoloSummary)
					}
				}
			}
		} else if jobTemplate == jobs.TemplateLabelQualityAudit {
			config["audit_type"] = strings.ToLower(strings.TrimSpace(experiment.Mechanism))
			config["report_only"] = true
		}
		addOptionalExperimentConfig(config, experiment)
		if metadataImport, err := s.store.GetActiveDatasetMetadataImport(plan.DatasetID); err == nil {
			config["metadata_import_id"] = metadataImport.ID
			config["metadata_summary"] = metadataImport.AgentSafeSummary
		} else if err != nil && !errors.Is(err, store.ErrNotFound) {
			return executeExperimentPlanResponse{}, err
		}
		if suggestion, ok := automlSuggestions[index]; ok {
			config["automl_study_id"] = suggestion.StudyID
			config["automl_suggestion_id"] = suggestion.ID
			config["automl_summary"] = automlJobSummary(experiment, suggestion)
		}

		job, err := s.store.CreateJob(plan.ProjectID, jobTemplate, config)
		if err != nil {
			return executeExperimentPlanResponse{}, err
		}
		if suggestion, ok := automlSuggestions[index]; ok && suggestion.ID != "" {
			if _, err := s.store.UpdateOptimizerSuggestionJob(suggestion.ID, job.ID); err != nil {
				log.Printf("link AutoML suggestion %s to job %s failed: %v", suggestion.ID, job.ID, err)
			}
		}

		out = append(out, job)
	}

	workerRequirement, err := s.ensureWorkerRequirementForPlanJobs(plan, provider, req.GPUType, out, "plan_execution", req.MaxConcurrentJobs)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}
	if err := s.recordCostPolicySkippedJobs(plan, costPolicy); err != nil {
		return executeExperimentPlanResponse{}, err
	}

	return executeExperimentPlanResponse{
		Plan:              plan,
		Jobs:              out,
		CostPolicy:        costPolicy.Payload(),
		WorkerRequirement: workerRequirement,
	}, nil
}

func (s *Server) validateFollowUpPlanCanExecute(plan plans.ExperimentPlan) error {
	if plan.SourceDecisionID == "" {
		return nil
	}
	if stopReason, stopDetails, ok, err := s.projectChampionSelectedFollowUpStopReason(plan.ProjectID); err != nil {
		return err
	} else if ok {
		message := fmt.Sprintf("Follow-up execution blocked for plan %s because the project already has a selected champion.", plan.ID)
		s.recordChampionSelectedFollowUpBlocked(plan.ProjectID, plan.ID, plan.SourceDecisionID, plan.ID, message, stopReason, stopDetails)
		return fmt.Errorf("%w: %s", errChampionSelectedFollowUpBlocked, stopReason)
	}
	projectPlans, err := s.store.ListProjectExperimentPlans(plan.ProjectID)
	if err != nil {
		return err
	}
	return s.validateExistingFollowUpPlanStillNovel(plan.ProjectID, plan.SourceDecisionID, plan, projectPlans)
}

func experimentExecutionTemplate(experiment plans.PlannedExperiment) string {
	if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		return jobs.TemplateLabelQualityAudit
	}
	return jobs.TemplateTrainExperiment
}

func addOptionalExperimentConfig(config map[string]any, experiment plans.PlannedExperiment) {
	if experiment.Mechanism != "" {
		config["mechanism"] = experiment.Mechanism
	}
	if experiment.Intervention != "" {
		config["intervention"] = experiment.Intervention
	}
	if len(experiment.EvidenceUsed) > 0 {
		config["evidence_used"] = experiment.EvidenceUsed
	}
	if experiment.ExpectedEffect != "" {
		config["expected_effect"] = experiment.ExpectedEffect
	}
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
	if experiment.Dropout > 0 {
		config["dropout"] = experiment.Dropout
	}
	if experiment.OptimizerMomentum > 0 {
		config["optimizer_momentum"] = experiment.OptimizerMomentum
	}
	if experiment.SchedulerStepSize > 0 {
		config["scheduler_step_size"] = experiment.SchedulerStepSize
	}
	if experiment.SchedulerGamma > 0 {
		config["scheduler_gamma"] = experiment.SchedulerGamma
	}
	if experiment.LabelSmoothing > 0 {
		config["label_smoothing"] = experiment.LabelSmoothing
	}
	if experiment.GradientClipNorm > 0 {
		config["gradient_clip_norm"] = experiment.GradientClipNorm
	}
	if len(experiment.Augmentation) > 0 {
		config["augmentation"] = experiment.Augmentation
	}
	if experiment.AugmentationPolicy != "" {
		config["augmentation_policy"] = experiment.AugmentationPolicy
	}
	if experiment.AugmentationPolicyConfig != nil {
		config["augmentation_policy_config"] = experiment.AugmentationPolicyConfig
	}
	if experiment.ClassBalancing != "" {
		config["class_balancing"] = experiment.ClassBalancing
	}
	if len(experiment.ClassBalancingConfig) > 0 {
		config["class_balancing_config"] = experiment.ClassBalancingConfig
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

func addDatasetMaterializationConfig(config map[string]any, policy execution.WorkerRequirementPolicy) {
	if policy.DatasetID == "" && policy.DatasetCacheKey == "" {
		return
	}
	if policy.DatasetChecksum != "" {
		config["dataset_checksum_sha256"] = policy.DatasetChecksum
	}
	materialization := map[string]any{
		"dataset_id":                policy.DatasetID,
		"dataset_checksum_sha256":   policy.DatasetChecksum,
		"dataset_cache_key":         policy.DatasetCacheKey,
		"status":                    policy.DatasetMaterializationStatus,
		"cold_cache_policy":         policy.ColdCachePolicy,
		"max_concurrent_jobs":       policy.MaxConcurrentJobs,
		"max_cold_materializations": policy.MaxColdDatasetMaterializations,
	}
	config["dataset_materialization"] = materialization
}

func addModalPreviewTrainingTierConfig(config map[string]any, provider string) {
	if !modalPreviewTierMetadataEnabled() {
		return
	}
	if strings.ToLower(strings.TrimSpace(provider)) != "modal" {
		return
	}
	if strings.TrimSpace(configString(config, "training_tier")) != "" {
		return
	}
	config["training_tier"] = "preview"
}

func modalPreviewTierMetadataEnabled() bool {
	return envFlag("MODEL_EXPRESS_MODAL_PREVIEW_TIER_METADATA", false)
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
	var automlWarnings []string
	var err error
	experiments, automlWarnings, err = s.prepareAutoMLExperimentsForProject(c.Param("id"), experiments)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	warnings = append(warnings, automlWarnings...)
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
	if err := s.persistAutoMLForPlan(plan); err != nil {
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

func (s *Server) cancelPlanActiveExecution(c *gin.Context) {
	var req cancelExecutionRequest
	if !bindOptionalJSON(c, &req) {
		return
	}
	response, err := s.cancelPlanActiveExecutionByID(c.Param("id"), req)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, response)
}

func (s *Server) cancelProjectActiveExecutions(c *gin.Context) {
	var req cancelExecutionRequest
	if !bindOptionalJSON(c, &req) {
		return
	}
	projectID := c.Param("id")
	if _, err := s.store.GetProject(projectID); err != nil {
		writeStoreError(c, err)
		return
	}
	planIDs, err := s.activePlanIDsForProject(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	response := cancelExecutionResponse{
		ExecutionID: "project:" + projectID,
		Target: map[string]string{
			"project_id": projectID,
		},
		Status:        "CANCELLED_BY_USER",
		Message:       "No active executions matched this project.",
		ModalCalls:    []cancelModalCallResult{},
		Compatibility: cancellationCompatibility(),
		Plans:         []cancelExecutionResponse{},
	}
	for _, planID := range planIDs {
		planResponse, cancelErr := s.cancelPlanActiveExecutionByID(planID, req)
		if cancelErr != nil {
			writeStoreError(c, cancelErr)
			return
		}
		response.Plans = append(response.Plans, planResponse)
		response.QueuedJobsCancelled += planResponse.QueuedJobsCancelled
		response.ActiveJobsMarkedCancelling += planResponse.ActiveJobsMarkedCancelling
		response.AlreadyTerminalJobs += planResponse.AlreadyTerminalJobs
		response.ModalCalls = append(response.ModalCalls, planResponse.ModalCalls...)
		response.WorkerRequirements = append(response.WorkerRequirements, planResponse.WorkerRequirements...)
		response.Jobs = append(response.Jobs, planResponse.Jobs...)
		if planResponse.BestAvailableModel.Exportable && !response.BestAvailableModel.Exportable {
			response.BestAvailableModel = planResponse.BestAvailableModel
		}
	}
	if len(response.Plans) > 0 {
		response.Message = fmt.Sprintf("Cancelled %d active execution(s) for project %s.", len(response.Plans), projectID)
	}
	c.JSON(http.StatusOK, response)
}
