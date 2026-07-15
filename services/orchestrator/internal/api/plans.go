package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/diagnostics"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/policies"
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
	Provider           string `json:"provider"`
	GPUType            string `json:"gpu_type"`
	MaxConcurrentJobs  int    `json:"max_concurrent_jobs"`
	deferPlanAggregate bool
}

type executeExperimentPlanResponse struct {
	Plan              plans.ExperimentPlan                  `json:"plan"`
	Jobs              []jobs.ExperimentJob                  `json:"jobs"`
	ValidationReports []execution.ExecutionValidationReport `json:"execution_validation_reports,omitempty"`
	CostPolicy        map[string]any                        `json:"cost_policy,omitempty"`
	WorkerRequirement *execution.WorkerRequirement          `json:"worker_requirement,omitempty"`
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
	effectivePolicy, err := s.resolveProposalPolicy(project, dataset, policyOperationPropose)
	if err != nil {
		return err
	}
	recommendation, err := agents.NewDatasetPlanner().BuildExperimentPlan(project, dataset, agents.PlanPreferences{
		Priority:        agents.PriorityBalanced,
		EffectivePolicy: &effectivePolicy,
	})
	if err != nil {
		var policyErr *policies.PolicyError
		if errors.As(err, &policyErr) {
			return err
		}
		return fmt.Errorf("%w: %s", store.ErrInvalidRequest, err.Error())
	}
	experiments, automlWarnings, err := s.prepareAutoMLExperimentsForProjectWithPolicy(project.ID, recommendation.Experiments, &effectivePolicy)
	if err != nil {
		return err
	}
	warnings := append([]string(nil), recommendation.Warnings...)
	warnings = append(warnings, automlWarnings...)

	evaluation, err := s.recordProposalPolicyEvaluation(effectivePolicy, policyOperationPersistPlan, experiments, "")
	if err != nil {
		return err
	}
	plan, err := s.store.CreateExperimentPlanWithPolicy(
		project.ID,
		dataset.ID,
		recommendation.TargetMetric,
		recommendation.RecommendedWorkers,
		recommendation.EstimatedMinutes,
		experiments,
		warnings,
		"",
		policies.PersistenceReference{EvaluationID: evaluation.ID, EffectivePolicyHash: evaluation.EffectivePolicyHash, Status: policyStatusAllowed},
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
	schedulePolicyEvaluation, err := s.recordPlanSchedulePolicy(plan, dataset, provider)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}
	schedulePolicyReference := policyReferenceForEvaluation(schedulePolicyEvaluation)
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
	if executionValidationMode() == execution.ValidationModeEnforce {
		preflightCostPolicy := costPolicy
		for index, experiment := range plan.Experiments {
			if _, ok := jobsByExperiment[index]; ok || experimentExecutionTemplate(experiment) != jobs.TemplateTrainExperiment {
				continue
			}
			if err := validateExperimentDatasetCompatibility(experiment, dataset, index); err != nil {
				return executeExperimentPlanResponse{}, err
			}
			if preflightCostPolicy.Enabled {
				allowed, _ := preflightCostPolicy.AllowTrainingJob(trainingTierForExperiment(experiment))
				if !allowed {
					continue
				}
			}
			spec, err := buildExecutionSpecV1WithPolicy(experiment, provider, schedulePolicyEvaluation)
			if err != nil {
				return executeExperimentPlanResponse{}, err
			}
			modelSpec, _ := supportedModelSpecByName(experiment.Model)
			report, err := execution.ValidateExecutionSpecV1(spec, modelSpec.Family, execution.ValidationModeEnforce)
			if err != nil {
				return executeExperimentPlanResponse{}, fmt.Errorf("validate execution spec: %w", err)
			}
			report.SetAcceptedDuplicate(matchingAcceptedSpecJobIDs(spec.AcceptedSpecHash, existingJobs))
			if report.WouldBlock {
				s.recordExecutionValidationReport(plan, index, report)
				return executeExperimentPlanResponse{}, fmt.Errorf(
					"%w: experiment %d would be blocked by execution fidelity enforcement: %s",
					store.ErrInvalidRequest, index, executionValidationSummary(report),
				)
			}
		}
	}

	out := make([]jobs.ExperimentJob, 0, len(plan.Experiments))
	validationReports := make([]execution.ExecutionValidationReport, 0, len(plan.Experiments))
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
		if jobTemplate == jobs.TemplateTrainExperiment {
			spec, err := addExecutionSpecV1WithPolicy(config, experiment, provider, schedulePolicyEvaluation)
			if err != nil {
				return executeExperimentPlanResponse{}, err
			}
			modelSpec, _ := supportedModelSpecByName(experiment.Model)
			report, err := execution.ValidateExecutionSpecV1(spec, modelSpec.Family, executionValidationMode())
			if err != nil {
				return executeExperimentPlanResponse{}, fmt.Errorf("validate execution spec: %w", err)
			}
			report.SetAcceptedDuplicate(matchingAcceptedSpecJobIDs(spec.AcceptedSpecHash, existingJobs, out))
			config[execution.ExecutionValidationConfigKey] = report
			validationReports = append(validationReports, report)
			s.recordExecutionValidationReport(plan, index, report)
			if report.Mode == execution.ValidationModeEnforce && report.WouldBlock {
				return executeExperimentPlanResponse{}, fmt.Errorf(
					"%w: experiment %d would be blocked by execution fidelity enforcement: %s",
					store.ErrInvalidRequest, index, executionValidationSummary(report),
				)
			}
			if report.AcceptedDuplicate.Skip {
				continue
			}
		}
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

		job, err := s.store.CreateJobWithOptions(plan.ProjectID, jobTemplate, config, store.CreateJobOptions{PolicyReference: schedulePolicyReference})
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
	complete, err := s.finalizeCandidateOutcomesForPlan(plan.ID)
	if err != nil {
		return executeExperimentPlanResponse{}, err
	}
	if complete && !req.deferPlanAggregate {
		if err := s.recordExperimentPlannerOutcomeForPlan(plan); err != nil {
			return executeExperimentPlanResponse{}, err
		}
	}

	return executeExperimentPlanResponse{
		Plan:              plan,
		Jobs:              out,
		ValidationReports: validationReports,
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
	if experiment.Mechanism != "" || experiment.IsFieldPresent("mechanism") {
		config["mechanism"] = experiment.Mechanism
	}
	if experiment.Intervention != "" || experiment.IsFieldPresent("intervention") {
		config["intervention"] = experiment.Intervention
	}
	if len(experiment.EvidenceUsed) > 0 || experiment.IsFieldPresent("evidence_used") {
		config["evidence_used"] = experiment.EvidenceUsed
	}
	if experiment.ExpectedEffect != "" || experiment.IsFieldPresent("expected_effect") {
		config["expected_effect"] = experiment.ExpectedEffect
	}
	if experiment.ImageSize > 0 {
		config["image_size"] = experiment.ImageSize
	}
	if experiment.ResolutionStrategy != "" || experiment.IsFieldPresent("resolution_strategy") {
		config["resolution_strategy"] = experiment.ResolutionStrategy
	}
	if experiment.Preprocessing != nil {
		config["preprocessing"] = experiment.Preprocessing
	}
	if experiment.Optimizer != "" || experiment.IsFieldPresent("optimizer") {
		config["optimizer"] = experiment.Optimizer
	}
	if experiment.Scheduler != "" || experiment.IsFieldPresent("scheduler") {
		config["scheduler"] = experiment.Scheduler
	}
	if experiment.WeightDecay > 0 || experiment.IsFieldPresent("weight_decay") {
		config["weight_decay"] = experiment.WeightDecay
	}
	if experiment.Dropout > 0 || experiment.IsFieldPresent("dropout") {
		config["dropout"] = experiment.Dropout
	}
	if experiment.OptimizerMomentum > 0 || experiment.IsFieldPresent("optimizer_momentum") {
		config["optimizer_momentum"] = experiment.OptimizerMomentum
	}
	if experiment.SchedulerStepSize > 0 || experiment.IsFieldPresent("scheduler_step_size") {
		config["scheduler_step_size"] = experiment.SchedulerStepSize
	}
	if experiment.SchedulerGamma > 0 || experiment.IsFieldPresent("scheduler_gamma") {
		config["scheduler_gamma"] = experiment.SchedulerGamma
	}
	if experiment.LabelSmoothing > 0 || experiment.IsFieldPresent("label_smoothing") {
		config["label_smoothing"] = experiment.LabelSmoothing
	}
	if experiment.GradientClipNorm > 0 || experiment.IsFieldPresent("gradient_clip_norm") {
		config["gradient_clip_norm"] = experiment.GradientClipNorm
	}
	if len(experiment.Augmentation) > 0 || experiment.IsFieldPresent("augmentation") {
		config["augmentation"] = experiment.Augmentation
	}
	if experiment.AugmentationPolicy != "" || experiment.IsFieldPresent("augmentation_policy") {
		config["augmentation_policy"] = experiment.AugmentationPolicy
	}
	if experiment.AugmentationPolicyConfig != nil {
		config["augmentation_policy_config"] = experiment.AugmentationPolicyConfig
	}
	if experiment.ClassBalancing != "" || experiment.IsFieldPresent("class_balancing") {
		config["class_balancing"] = experiment.ClassBalancing
	}
	if len(experiment.ClassBalancingConfig) > 0 || experiment.IsFieldPresent("class_balancing_config") {
		config["class_balancing_config"] = experiment.ClassBalancingConfig
	}
	if experiment.SamplingStrategy != "" || experiment.IsFieldPresent("sampling_strategy") {
		config["sampling_strategy"] = experiment.SamplingStrategy
	}
	if experiment.EarlyStoppingPatience > 0 || experiment.IsFieldPresent("early_stopping_patience") {
		config["early_stopping_patience"] = experiment.EarlyStoppingPatience
	}
	if experiment.Strategy != "" || experiment.IsFieldPresent("strategy") {
		config["strategy"] = experiment.Strategy
	}
	if experiment.Pretrained || experiment.IsFieldPresent("pretrained") {
		config["pretrained"] = experiment.Pretrained
	}
	if experiment.FreezeBackbone || experiment.IsFieldPresent("freeze_backbone") {
		config["freeze_backbone"] = experiment.FreezeBackbone
	}
	if experiment.FineTuneStrategy != "" || experiment.IsFieldPresent("fine_tune_strategy") {
		config["fine_tune_strategy"] = experiment.FineTuneStrategy
	}
}

func addExecutionSpecV1(
	config map[string]any,
	experiment plans.PlannedExperiment,
	provider string,
) (execution.ExecutionSpecV1, error) {
	spec, err := buildExecutionSpecV1(experiment, provider)
	if err != nil {
		return execution.ExecutionSpecV1{}, err
	}
	payload, err := spec.Payload()
	if err != nil {
		return execution.ExecutionSpecV1{}, err
	}
	config[execution.ExecutionSpecConfigKey] = payload
	return spec, nil
}

func addExecutionSpecV1WithPolicy(
	config map[string]any,
	experiment plans.PlannedExperiment,
	provider string,
	evaluation policies.Evaluation,
) (execution.ExecutionSpecV1, error) {
	spec, err := buildExecutionSpecV1WithPolicy(experiment, provider, evaluation)
	if err != nil {
		return execution.ExecutionSpecV1{}, err
	}
	payload, err := spec.Payload()
	if err != nil {
		return execution.ExecutionSpecV1{}, err
	}
	config[execution.ExecutionSpecConfigKey] = payload
	return spec, nil
}

func buildExecutionSpecV1(
	experiment plans.PlannedExperiment,
	provider string,
) (execution.ExecutionSpecV1, error) {
	modelSpec, ok := supportedModelSpecByName(experiment.Model)
	if !ok {
		return execution.ExecutionSpecV1{}, fmt.Errorf("%w: unsupported execution-spec model %q", store.ErrInvalidRequest, experiment.Model)
	}
	runner, err := executionRunnerFor(provider, modelSpec.TaskType)
	if err != nil {
		return execution.ExecutionSpecV1{}, err
	}
	requestedConfig, err := experiment.RequestedConfig()
	if err != nil {
		return execution.ExecutionSpecV1{}, err
	}
	resolutionInput := make(map[string]any, len(requestedConfig)+1)
	for key, value := range requestedConfig {
		resolutionInput[key] = value
	}
	if imageSize, ok := resolutionInput["image_size"].(float64); !ok || imageSize <= 0 {
		if modelSpec.DefaultImageSize > 0 {
			resolutionInput["image_size"] = modelSpec.DefaultImageSize
		}
	}
	spec, err := execution.BuildExecutionSpecV1(
		modelSpec.TaskType,
		runner,
		requestedConfig,
		resolutionInput,
	)
	if err != nil {
		return execution.ExecutionSpecV1{}, fmt.Errorf("resolve execution spec: %w", err)
	}
	return spec, nil
}

func buildExecutionSpecV1WithPolicy(
	experiment plans.PlannedExperiment,
	provider string,
	evaluation policies.Evaluation,
) (execution.ExecutionSpecV1, error) {
	modelSpec, ok := supportedModelSpecByName(experiment.Model)
	if !ok {
		return execution.ExecutionSpecV1{}, fmt.Errorf("%w: unsupported execution-spec model %q", store.ErrInvalidRequest, experiment.Model)
	}
	runner, err := executionRunnerFor(provider, modelSpec.TaskType)
	if err != nil {
		return execution.ExecutionSpecV1{}, err
	}
	requestedConfig, err := experiment.RequestedConfig()
	if err != nil {
		return execution.ExecutionSpecV1{}, err
	}
	resolutionInput := make(map[string]any, len(requestedConfig)+1)
	for key, value := range requestedConfig {
		resolutionInput[key] = value
	}
	if imageSize, ok := resolutionInput["image_size"].(float64); !ok || imageSize <= 0 {
		if modelSpec.DefaultImageSize > 0 {
			resolutionInput["image_size"] = modelSpec.DefaultImageSize
		}
	}
	artifactPlan, err := automaticArtifactPlanFromEvaluation(modelSpec.TaskType, runner, evaluation)
	if err != nil {
		return execution.ExecutionSpecV1{}, err
	}
	spec, err := execution.BuildExecutionSpecV1WithArtifactPlan(
		modelSpec.TaskType, runner, requestedConfig, resolutionInput, artifactPlan,
	)
	if err != nil {
		return execution.ExecutionSpecV1{}, fmt.Errorf("resolve execution spec: %w", err)
	}
	return spec, nil
}

func matchingAcceptedSpecJobIDs(hash string, groups ...[]jobs.ExperimentJob) []string {
	if strings.TrimSpace(hash) == "" {
		return nil
	}
	out := []string{}
	seen := map[string]bool{}
	for _, group := range groups {
		for _, job := range group {
			if job.ID == "" || seen[job.ID] || acceptedSpecHashFromJob(job) != hash {
				continue
			}
			seen[job.ID] = true
			out = append(out, job.ID)
		}
	}
	return out
}

func acceptedSpecHashFromJob(job jobs.ExperimentJob) string {
	value, ok := job.Config[execution.ExecutionSpecConfigKey]
	if !ok || value == nil {
		return ""
	}
	if payload, ok := value.(map[string]any); ok {
		return configString(payload, "accepted_spec_hash")
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	var spec execution.ExecutionSpecV1
	if err := json.Unmarshal(blob, &spec); err != nil {
		return ""
	}
	return spec.AcceptedSpecHash
}

func executionValidationSummary(report execution.ExecutionValidationReport) string {
	parts := []string{}
	for _, finding := range report.Findings {
		if !finding.WouldBlock {
			continue
		}
		parts = append(parts, fmt.Sprintf("%s (%s): %s", finding.Field, finding.ReasonCode, finding.SuggestedAlternative))
	}
	if len(parts) == 0 {
		return "no blocking findings"
	}
	return strings.Join(parts, "; ")
}

func (s *Server) recordExecutionValidationReport(plan plans.ExperimentPlan, experimentIndex int, report execution.ExecutionValidationReport) {
	for _, finding := range report.Findings {
		diagnostics.Event("info", "execution_validation_finding", map[string]any{
			"project_id": plan.ProjectID, "plan_id": plan.ID, "experiment_index": experimentIndex,
			"mode": report.Mode, "task": report.Task, "runner": report.Runner, "model_family": report.ModelFamily,
			"field": finding.Field, "classification": finding.Classification, "reason_code": finding.ReasonCode,
			"would_block": finding.WouldBlock,
		})
	}
	if len(report.Findings) == 0 && !report.AcceptedDuplicate.Skip {
		return
	}
	message := fmt.Sprintf("Execution fidelity validation reported %d finding(s) for experiment %d.", len(report.Findings), experimentIndex)
	if report.AcceptedDuplicate.Skip {
		message = fmt.Sprintf(
			"Experiment %d was skipped because its accepted execution semantics duplicate an existing job.",
			experimentIndex,
		)
	}
	if report.Mode == execution.ValidationModeEnforce && report.WouldBlock {
		message = fmt.Sprintf("Execution fidelity enforcement would block experiment %d.", experimentIndex)
	}
	if _, err := s.store.CreateExecutionEvent(plan.ProjectID, plan.ID, execution.EventExecutionValidationReported, message, map[string]any{
		"experiment_index": experimentIndex,
		"report":           report,
	}); err != nil {
		log.Printf("record execution validation event failed for plan %s experiment %d: %v", plan.ID, experimentIndex, err)
	}
}

func executionRunnerFor(provider, task string) (string, error) {
	switch normalizeTrainingProvider(provider) {
	case "local", "persistent_gpu", "persistent_disk":
		return "local_simulator", nil
	case "modal":
		if task == "object_detection" {
			return "modal_ultralytics", nil
		}
		if task == "image_classification" {
			return "modal_torchvision", nil
		}
	}
	return "", fmt.Errorf(
		"%w: no execution capability runner for provider %q and task %q",
		store.ErrInvalidRequest,
		provider,
		task,
	)
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
	if dataset.ProjectID != project.ID {
		writeStoreError(c, fmt.Errorf("%w: dataset does not belong to project", store.ErrInvalidRequest))
		return
	}
	effectivePolicy, err := s.resolveProposalPolicy(project, dataset, policyOperationPropose)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if len(experiments) == 0 {
		recommendation, err := agents.NewDatasetPlanner().BuildExperimentPlan(project, dataset, agents.PlanPreferences{
			Priority:          req.Priority,
			MaxWorkers:        req.MaxWorkers,
			TimeBudgetMinutes: req.TimeBudgetMinutes,
			TargetMetric:      req.TargetMetric,
			EffectivePolicy:   &effectivePolicy,
		})
		if err != nil {
			var policyErr *policies.PolicyError
			if errors.As(err, &policyErr) {
				writeStoreError(c, err)
			} else {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			}
			return
		}

		targetMetric = recommendation.TargetMetric
		recommendedWorkers = recommendation.RecommendedWorkers
		estimatedMinutes = recommendation.EstimatedMinutes
		experiments = recommendation.Experiments
		warnings = append(warnings, recommendation.Warnings...)
	}
	var automlWarnings []string
	experiments, automlWarnings, err = s.prepareAutoMLExperimentsForProjectWithPolicy(c.Param("id"), experiments, &effectivePolicy)
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

	evaluation, err := s.recordProposalPolicyEvaluation(effectivePolicy, policyOperationPersistPlan, experiments, "")
	if err != nil {
		writeStoreError(c, err)
		return
	}
	plan, err := s.store.CreateExperimentPlanWithPolicy(
		c.Param("id"),
		req.DatasetID,
		targetMetric,
		recommendedWorkers,
		estimatedMinutes,
		experiments,
		warnings,
		"",
		policies.PersistenceReference{EvaluationID: evaluation.ID, EffectivePolicyHash: evaluation.EffectivePolicyHash, Status: policyStatusAllowed},
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
	req := s.defaultExecuteExperimentPlanRequest()
	if !bindOptionalJSON(c, &req) {
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
