package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

const (
	TrainingMonitorAgentName     = "training_monitor"
	TrainingMonitorAgentVersion  = "v1"
	TrainingMonitorPromptVersion = "training_monitor_v1"

	trainingMonitorContextVersion    = "training_monitor_run_evaluation_cards_v1"
	trainingMonitorMaxRecentEpochs   = 5
	trainingMonitorMaxWorstClasses   = 5
	trainingMonitorMaxConfusionPairs = 5
	trainingMonitorMaxMemoryRecords  = 8
)

type TrainingMonitorAgent struct {
	generator       llm.JSONGenerator
	model           string
	reasoningEffort string
	maxToolRounds   int
}

type TrainingMonitorInput struct {
	Plan               plans.ExperimentPlan
	Job                jobs.ExperimentJob
	Summary            runs.TrainingRunSummary
	Evaluation         *runs.TrainingRunEvaluation
	Metrics            []jobs.EpochMetric
	ObjectiveContext   ProjectObjectiveContext
	MemoryRecords      []memory.AgentMemoryRecord
	RetrievedRunMemory []memory.MemoryRetrievalResult
	OptimizerFeedback  *automl.OptimizerFeedbackSummary
}

type TrainingEvaluationRecommendation struct {
	AgentName         string                        `json:"agent_name"`
	Summary           string                        `json:"summary"`
	RecommendedAction decisions.AgentActionProposal `json:"recommended_action"`
	QualitySummary    string                        `json:"quality_summary"`
	TrainingDynamics  string                        `json:"training_dynamics"`
	CostSummary       string                        `json:"cost_summary"`
	Risks             []string                      `json:"risks"`
	Findings          []string                      `json:"findings"`
	RankScore         float64                       `json:"rank_score"`
	Tags              []string                      `json:"tags"`
}

type TrainingMonitorEvaluationTrace struct {
	Recommendation          TrainingEvaluationRecommendation
	Request                 llm.JSONRequest
	PromptContext           map[string]any
	RawOutput               []byte
	ParsedOutput            map[string]any
	ValidationStatus        string
	ValidationError         string
	AgentVersion            string
	PromptVersion           string
	ResponseID              string
	PreviousResponseID      string
	ToolRounds              int
	Usage                   *llm.Usage
	ToolCalls               []AgentToolCallTrace
	ToolResults             []AgentToolResultTrace
	RejectedToolCalls       []AgentToolResultTrace
	DryRunValidationResults []map[string]any
}

func NewTrainingMonitorAgent(generator llm.JSONGenerator, model string) TrainingMonitorAgent {
	return TrainingMonitorAgent{
		generator: generator,
		model:     model,
	}
}

func NewTrainingMonitorAgentWithRuntime(generator llm.JSONGenerator, model string, config llm.Config) TrainingMonitorAgent {
	return TrainingMonitorAgent{
		generator:       generator,
		model:           model,
		reasoningEffort: config.ReasoningEffort,
		maxToolRounds:   config.MaxToolRounds,
	}
}

func (a TrainingMonitorAgent) Evaluate(ctx context.Context, input TrainingMonitorInput) (TrainingEvaluationRecommendation, error) {
	trace, err := a.EvaluateWithTrace(ctx, input)
	return trace.Recommendation, err
}

func (a TrainingMonitorAgent) EvaluateWithTrace(ctx context.Context, input TrainingMonitorInput) (TrainingMonitorEvaluationTrace, error) {
	trace := TrainingMonitorEvaluationTrace{
		PromptContext:    trainingMonitorPromptContext(input),
		ParsedOutput:     map[string]any{},
		ValidationStatus: memory.InvocationValidationFailed,
		AgentVersion:     TrainingMonitorAgentVersion,
		PromptVersion:    TrainingMonitorPromptVersion,
	}

	if a.generator == nil {
		err := fmt.Errorf("training monitor requires an llm generator")
		trace.ValidationError = err.Error()
		return trace, err
	}

	contextBlob, err := json.Marshal(trace.PromptContext)
	if err != nil {
		wrapped := fmt.Errorf("marshal training monitor context: %w", err)
		trace.ValidationError = wrapped.Error()
		return trace, wrapped
	}

	trace.Request = trainingMonitorJSONRequest(a.model, contextBlob)
	trace.Request.ReasoningEffort = a.reasoningEffort
	raw, err := a.generateTrainingMonitorJSON(ctx, &trace, input)
	if err != nil {
		trace.ValidationError = err.Error()
		return trace, err
	}
	trace.RawOutput = append([]byte(nil), raw...)
	trace.ParsedOutput = rawOutputObject(raw)

	var recommendation TrainingEvaluationRecommendation
	if err := json.Unmarshal(raw, &recommendation); err != nil {
		wrapped := fmt.Errorf("decode training monitor recommendation: %w", err)
		trace.ValidationStatus = memory.InvocationValidationInvalid
		trace.ValidationError = wrapped.Error()
		return trace, wrapped
	}

	recommendation.AgentName = TrainingMonitorAgentName
	if err := validateTrainingEvaluation(recommendation); err != nil {
		trace.ValidationStatus = memory.InvocationValidationInvalid
		trace.ValidationError = err.Error()
		trace.Recommendation = recommendation
		return trace, err
	}

	trace.Recommendation = recommendation
	trace.ValidationStatus = memory.InvocationValidationValid
	trace.ValidationError = ""
	return trace, nil
}

type trainingMonitorToolLoopGenerator interface {
	GenerateJSONWithTools(context.Context, llm.ToolLoopRequest) (llm.ToolLoopResult, error)
}

func (a TrainingMonitorAgent) generateTrainingMonitorJSON(ctx context.Context, trace *TrainingMonitorEvaluationTrace, input TrainingMonitorInput) ([]byte, error) {
	toolGenerator, ok := a.generator.(trainingMonitorToolLoopGenerator)
	if !ok {
		if usageGenerator, ok := a.generator.(jsonUsageGenerator); ok {
			result, err := usageGenerator.GenerateJSONWithUsage(ctx, trace.Request)
			if err != nil {
				return nil, err
			}
			trace.Usage = result.Usage
			return result.RawJSON, nil
		}
		return a.generator.GenerateJSON(ctx, trace.Request)
	}

	result, err := toolGenerator.GenerateJSONWithTools(ctx, llm.ToolLoopRequest{
		JSONRequest:   trace.Request,
		Tools:         llmToolDefinitions(TrainingMonitorInformationToolSpecs()),
		ToolAnswerer:  trainingMonitorInformationAnswerer{input: input},
		MaxToolRounds: a.maxToolRounds,
	})
	if err != nil {
		return nil, err
	}
	trace.ResponseID = result.ResponseID
	trace.PreviousResponseID = result.PreviousResponseID
	trace.ToolRounds = result.ToolRounds
	trace.Usage = result.Usage
	trace.ToolCalls = toolCallTraces(result.ToolCalls)
	trace.ToolResults, trace.RejectedToolCalls, trace.DryRunValidationResults = toolResultTraces(result.ToolResults)
	return result.FinalJSON, nil
}

type trainingMonitorInformationAnswerer struct {
	input TrainingMonitorInput
}

func (a trainingMonitorInformationAnswerer) AnswerInformationToolCall(_ context.Context, call llm.ToolCall) (llm.ToolResult, error) {
	args := map[string]any{}
	if len(call.Arguments) > 0 {
		if err := json.Unmarshal(call.Arguments, &args); err != nil {
			result := AgentInformationToolResult{
				ToolName: strings.TrimSpace(call.Name),
				Accepted: false,
				Error:    "tool arguments must be a JSON object",
			}
			encoded, _ := json.Marshal(result)
			return llm.ToolResult{CallID: call.CallID, Name: call.Name, Output: encoded}, nil
		}
	}
	result := ExecuteTrainingMonitorInformationTool(a.input, call.Name, args, TrainingMonitorInformationToolOptions{})
	encoded, err := json.Marshal(result)
	if err != nil {
		return llm.ToolResult{CallID: call.CallID, Name: call.Name}, fmt.Errorf("encode training monitor tool result: %w", err)
	}
	return llm.ToolResult{CallID: call.CallID, Name: call.Name, Output: encoded}, nil
}

func trainingMonitorJSONRequest(model string, contextBlob []byte) llm.JSONRequest {
	return llm.JSONRequest{
		Model:       model,
		Temperature: 0.2,
		Messages: []llm.Message{
			{
				Role: "system",
				Content: strings.TrimSpace(`You are the Model Express Training Monitor Agent.
Evaluate image-classification training runs holistically.
When approved information tools are available, you may ask bounded run-scoped backend questions before finalizing.
Tool calls are questions only: they cannot propose experiments, create plans, create jobs, create workers, export champions, run inference, or mutate datasets.
After any information requests, return only final valid JSON.
Consider validation quality, macro-F1, accuracy, per-class metrics, confusion matrix,
train/validation gap, metric stability, plateauing, cost, runtime, inference latency,
model size, compact optimizer feedback if present, and whether the run should inform future experiments.
Retrieved run memory, if present, is advisory prior-run context only; it is not scheduling authority.
This agent evaluates one run only. Do not propose new experiments or plan a follow-up batch.
Produce signals that the plan-level Experiment Planning Agent can use later.`),
			},
			{
				Role: "user",
				Content: fmt.Sprintf(`Evaluate this completed training run and return JSON with this exact shape:
{
  "summary": "short human-readable summary",
  "recommended_action": {
    "action_type": "RANK_MODELS|PRUNE_RUN|STOP_PROJECT|CHANGE_PREPROCESSING",
    "confidence": 0.0,
    "rationale": "why this action is useful",
    "payload": {},
    "requires_approval": true
  },
  "quality_summary": "quality assessment",
  "training_dynamics": "overfitting/underfitting/plateau/stability assessment",
  "cost_summary": "cost and runtime assessment",
  "risks": ["risk"],
  "findings": ["finding"],
  "rank_score": 0.0,
  "tags": ["short_tag"]
}

Do not include proposed_experiments. This is not the plan-level planner.
Payload may include supporting evidence such as overfitting indicators, plateau signals, or promising settings.
Use objective_context to judge whether the run fits the user's goal. For live/real-time goals, treat latency
under roughly 25ms as acceptable and penalize slow or oversized models only when quality is close or latency exceeds that budget.

Context:
%s`, string(contextBlob)),
			},
		},
	}
}

func rawOutputObject(raw []byte) map[string]any {
	out := map[string]any{}
	if len(raw) == 0 {
		return out
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func validateTrainingEvaluation(recommendation TrainingEvaluationRecommendation) error {
	if strings.TrimSpace(recommendation.Summary) == "" {
		return fmt.Errorf("training monitor recommendation missing summary")
	}
	action := recommendation.RecommendedAction
	if !allowedAgentAction(action.ActionType) {
		return fmt.Errorf("training monitor recommendation has invalid action_type %q", action.ActionType)
	}
	if action.Confidence < 0 || action.Confidence > 1 {
		return fmt.Errorf("training monitor recommendation confidence must be between 0 and 1")
	}
	if strings.TrimSpace(action.Rationale) == "" {
		return fmt.Errorf("training monitor recommendation missing action rationale")
	}
	if key, found := trainingMonitorPayloadHasForbiddenAuthority(action.Payload); found {
		return fmt.Errorf("training monitor recommendation payload contains forbidden scheduling or planning authority %q", key)
	}
	if recommendation.RankScore < 0 || recommendation.RankScore > 1 {
		return fmt.Errorf("training monitor recommendation rank_score must be between 0 and 1")
	}
	if recommendation.Tags == nil {
		recommendation.Tags = []string{}
	}
	return nil
}

func allowedAgentAction(actionType string) bool {
	switch strings.ToUpper(strings.TrimSpace(actionType)) {
	case "CHANGE_PREPROCESSING", "PRUNE_RUN", "RANK_MODELS", "STOP_PROJECT":
		return true
	default:
		return false
	}
}

func trainingMonitorPromptContext(input TrainingMonitorInput) map[string]any {
	context := map[string]any{
		"context_version":                       trainingMonitorContextVersion,
		"run_summary_card":                      trainingMonitorRunSummaryCard(input),
		"training_dynamics_card":                trainingMonitorDynamicsCard(input),
		"per_class_confusion_card":              trainingMonitorPerClassConfusionCard(input),
		"deployment_model_profile_card":         trainingMonitorDeploymentCard(input),
		"objective_fit_card":                    trainingMonitorObjectiveFitCard(input),
		"memory_summary":                        compactMemoryRecords(input.MemoryRecords),
		"backend_full_data_preservation_notice": "Full run summary, evaluation, epoch metrics, plan, and job config remain in backend storage; this prompt contains compact decision cards only.",
	}
	if input.OptimizerFeedback != nil {
		context["optimizer_feedback_summary"] = input.OptimizerFeedback
	}
	if retrievedRunMemory := compactRetrievedRunMemory(input.RetrievedRunMemory); len(retrievedRunMemory) > 0 {
		context["retrieved_run_memory"] = retrievedRunMemory
	}
	attachTrainingMonitorPromptBudget(context)
	return context
}

func trainingMonitorRunSummaryCard(input TrainingMonitorInput) map[string]any {
	targetMetric := trainingMonitorTargetMetric(input)
	card := map[string]any{
		"job_id":                input.Summary.JobID,
		"project_id":            input.Summary.ProjectID,
		"plan_id":               input.Summary.PlanID,
		"dataset_id":            input.Summary.DatasetID,
		"status":                input.Summary.Status,
		"model":                 input.Summary.Model,
		"provider":              input.Summary.Provider,
		"gpu_type":              input.Summary.GPUType,
		"target_metric":         targetMetric,
		"target_score":          trainingMonitorSummaryScore(input.Summary, targetMetric),
		"best_macro_f1":         input.Summary.BestMacroF1,
		"best_accuracy":         input.Summary.BestAccuracy,
		"final_train_loss":      input.Summary.FinalTrainLoss,
		"final_val_loss":        input.Summary.FinalValLoss,
		"final_loss_gap":        roundMonitorFloat(input.Summary.FinalValLoss - input.Summary.FinalTrainLoss),
		"epochs_completed":      input.Summary.EpochsCompleted,
		"runtime_seconds":       input.Summary.RuntimeSeconds,
		"estimated_cost_usd":    input.Summary.EstimatedCostUSD,
		"plan_target_metric":    input.Plan.TargetMetric,
		"plan_experiment_count": len(input.Plan.Experiments),
		"job_template":          input.Job.Template,
		"job_status":            input.Job.Status,
	}
	if profile := trainingMonitorExperimentProfile(input); len(profile) > 0 {
		card["experiment_profile"] = profile
	}
	return card
}

func trainingMonitorDynamicsCard(input TrainingMonitorInput) map[string]any {
	targetMetric := trainingMonitorTargetMetric(input)
	metrics := sortedTrainingMonitorMetrics(input.Metrics)
	recent := recentTrainingMonitorMetrics(metrics, targetMetric, trainingMonitorMaxRecentEpochs)
	card := map[string]any{
		"target_metric":         targetMetric,
		"best_target_score":     trainingMonitorSummaryScore(input.Summary, targetMetric),
		"final_train_loss":      input.Summary.FinalTrainLoss,
		"final_val_loss":        input.Summary.FinalValLoss,
		"final_loss_gap":        roundMonitorFloat(input.Summary.FinalValLoss - input.Summary.FinalTrainLoss),
		"epochs_completed":      input.Summary.EpochsCompleted,
		"epoch_count_total":     len(metrics),
		"recent_epoch_count":    len(recent),
		"recent_epochs":         recent,
		"more_epochs_justified": false,
		"plateau_signal":        "unknown",
		"instability_score":     0.0,
		"target_metric_delta":   0.0,
		"recent_metric_delta":   0.0,
		"train_loss_delta":      0.0,
		"validation_loss_delta": 0.0,
		"diagnostic_summary":    "No epoch metrics were available.",
	}
	if len(metrics) == 0 {
		return card
	}

	firstMetric, firstOK := firstMetricValue(metrics, targetMetric, "macro_f1", "accuracy")
	lastMetric, lastOK := lastMetricValue(metrics, targetMetric, "macro_f1", "accuracy")
	if firstOK && lastOK {
		metricDelta := roundMonitorFloat(lastMetric - firstMetric)
		card["target_metric_delta"] = metricDelta
		card["more_epochs_justified"] = metricDelta > 0.01
	}
	if len(recent) >= 2 {
		firstRecent, firstOK := mapFloatValue(recent[0], "target_metric_value")
		lastRecent, lastOK := mapFloatValue(recent[len(recent)-1], "target_metric_value")
		if firstOK && lastOK {
			recentDelta := roundMonitorFloat(lastRecent - firstRecent)
			card["recent_metric_delta"] = recentDelta
			card["plateau_signal"] = "active"
			if math.Abs(recentDelta) > 0.005 {
				card["plateau_signal"] = "not_active"
			}
			if recentDelta > 0.01 {
				card["more_epochs_justified"] = true
			}
		}
		card["instability_score"] = trainingMonitorInstabilityScore(recent)
	}
	if firstTrain, lastTrain, ok := firstLastMetricValues(metrics, "train_loss", "training_loss", "loss"); ok {
		card["train_loss_delta"] = roundMonitorFloat(lastTrain - firstTrain)
	}
	if firstVal, lastVal, ok := firstLastMetricValues(metrics, "val_loss", "validation_loss"); ok {
		card["validation_loss_delta"] = roundMonitorFloat(lastVal - firstVal)
	}
	card["diagnostic_summary"] = trainingMonitorDynamicsSummary(card)
	return card
}

func trainingMonitorPerClassConfusionCard(input TrainingMonitorInput) map[string]any {
	card := map[string]any{
		"has_per_class_metrics":            false,
		"has_confusion_matrix":             false,
		"class_count":                      0,
		"worst_classes":                    []map[string]any{},
		"top_confusion_pairs":              []map[string]any{},
		"accuracy_minus_macro_f1":          roundMonitorFloat(input.Summary.BestAccuracy - input.Summary.BestMacroF1),
		"minority_or_imbalance_signal":     false,
		"confusion_off_diagonal_total":     0,
		"per_class_summary_capped_at":      trainingMonitorMaxWorstClasses,
		"confusion_pair_summary_capped_at": trainingMonitorMaxConfusionPairs,
	}
	if input.Evaluation == nil {
		return card
	}
	classRows := trainingMonitorClassRows(input.Evaluation.PerClassMetrics)
	card["has_per_class_metrics"] = len(classRows) > 0
	card["class_count"] = len(classRows)
	card["worst_classes"] = cappedTrainingMonitorClassRows(classRows, trainingMonitorMaxWorstClasses)
	card["minority_or_imbalance_signal"] = trainingMonitorMinoritySignal(input.Summary, classRows)
	confusions, offDiagonal := trainingMonitorConfusionPairs(input.Evaluation.ConfusionMatrix, classRows)
	card["has_confusion_matrix"] = len(input.Evaluation.ConfusionMatrix) > 0
	card["top_confusion_pairs"] = confusions
	card["confusion_off_diagonal_total"] = offDiagonal
	return card
}

func trainingMonitorDeploymentCard(input TrainingMonitorInput) map[string]any {
	card := map[string]any{
		"model":              input.Summary.Model,
		"provider":           input.Summary.Provider,
		"gpu_type":           input.Summary.GPUType,
		"runtime_seconds":    input.Summary.RuntimeSeconds,
		"estimated_cost_usd": input.Summary.EstimatedCostUSD,
	}
	if input.Evaluation == nil {
		return card
	}
	card["model_profile_summary"] = compactSelectedAnyMap(input.Evaluation.ModelProfile, []string{
		"parameter_count",
		"trainable_parameter_count",
		"model_size_mb",
		"estimated_latency_ms",
		"estimated_throughput_images_per_second",
		"image_size",
		"pretrained",
		"freeze_backbone",
		"fine_tune_strategy",
	})
	card["holistic_score_summary"] = compactSelectedAnyMap(input.Evaluation.HolisticScores, []string{
		"quality_score",
		"latency_score",
		"cost_score",
		"runtime_score",
		"overall_score",
		"runtime_seconds",
	})
	card["recommendation_summary"] = input.Evaluation.RecommendationSummary
	return card
}

func trainingMonitorObjectiveFitCard(input TrainingMonitorInput) map[string]any {
	targetMetric := trainingMonitorTargetMetric(input)
	card := map[string]any{
		"target_metric":             targetMetric,
		"target_score":              trainingMonitorSummaryScore(input.Summary, targetMetric),
		"objective_context":         input.ObjectiveContext,
		"live_or_latency_sensitive": trainingMonitorLatencySensitive(input.ObjectiveContext),
	}
	if input.Evaluation != nil {
		card["evaluation_objective_profile"] = compactAnyMap(input.Evaluation.ObjectiveProfile, 10)
		if len(input.Evaluation.HolisticScores) > 0 {
			card["objective_score_components"] = compactSelectedAnyMap(input.Evaluation.HolisticScores, []string{
				"quality_score",
				"latency_score",
				"cost_score",
				"runtime_score",
				"overall_score",
			})
		}
	}
	return card
}

func trainingMonitorTargetMetric(input TrainingMonitorInput) string {
	if trimmed := strings.TrimSpace(input.Plan.TargetMetric); trimmed != "" {
		return trimmed
	}
	if input.Evaluation != nil {
		if target := stringFromAny(input.Evaluation.ObjectiveProfile["target_metric"]); target != "" {
			return target
		}
	}
	if len(input.ObjectiveContext.MetricPreferences) > 0 {
		if trimmed := strings.TrimSpace(input.ObjectiveContext.MetricPreferences[0]); trimmed != "" {
			return trimmed
		}
	}
	return "macro_f1"
}

func trainingMonitorSummaryScore(summary runs.TrainingRunSummary, targetMetric string) float64 {
	switch strings.ToLower(strings.TrimSpace(targetMetric)) {
	case "accuracy":
		return summary.BestAccuracy
	default:
		return summary.BestMacroF1
	}
}

func trainingMonitorExperimentProfile(input TrainingMonitorInput) map[string]any {
	for _, experiment := range input.Plan.Experiments {
		if input.Summary.Model != "" && experiment.Model != input.Summary.Model {
			continue
		}
		return compactPlannedExperimentProfile(experiment)
	}
	return compactJobConfigProfile(input.Job.Config)
}

func compactPlannedExperimentProfile(experiment plans.PlannedExperiment) map[string]any {
	out := map[string]any{
		"template":                experiment.Template,
		"model":                   experiment.Model,
		"mechanism":               experiment.Mechanism,
		"intervention":            experiment.Intervention,
		"evidence_used":           cappedStrings(experiment.EvidenceUsed, 6),
		"expected_effect":         experiment.ExpectedEffect,
		"epochs":                  experiment.Epochs,
		"batch_size":              experiment.BatchSize,
		"learning_rate":           experiment.LearningRate,
		"image_size":              experiment.ImageSize,
		"resolution_strategy":     experiment.ResolutionStrategy,
		"optimizer":               experiment.Optimizer,
		"scheduler":               experiment.Scheduler,
		"weight_decay":            experiment.WeightDecay,
		"dropout":                 experiment.Dropout,
		"optimizer_momentum":      experiment.OptimizerMomentum,
		"scheduler_step_size":     experiment.SchedulerStepSize,
		"scheduler_gamma":         experiment.SchedulerGamma,
		"label_smoothing":         experiment.LabelSmoothing,
		"gradient_clip_norm":      experiment.GradientClipNorm,
		"augmentation_policy":     experiment.AugmentationPolicy,
		"class_balancing":         experiment.ClassBalancing,
		"sampling_strategy":       experiment.SamplingStrategy,
		"early_stopping_patience": experiment.EarlyStoppingPatience,
		"pretrained":              experiment.Pretrained,
		"freeze_backbone":         experiment.FreezeBackbone,
		"fine_tune_strategy":      experiment.FineTuneStrategy,
	}
	if experiment.Preprocessing != nil {
		out["preprocessing"] = map[string]any{
			"resize_strategy":           experiment.Preprocessing.ResizeStrategy,
			"normalization":             experiment.Preprocessing.Normalization,
			"crop_strategy":             experiment.Preprocessing.CropStrategy,
			"bbox_mode":                 experiment.Preprocessing.BBoxMode,
			"use_dataset_normalization": experiment.Preprocessing.UseDatasetNormalization,
		}
	}
	if experiment.AugmentationPolicyConfig != nil {
		out["augmentation_policy_config"] = experiment.AugmentationPolicyConfig
	}
	if experiment.AutoML != nil && experiment.AutoML.Enabled {
		out["automl_summary"] = compactAutoMLPlanSummary(experiment.AutoML)
	}
	return compactNonZeroMap(out)
}

func compactJobConfigProfile(config map[string]any) map[string]any {
	if len(config) == 0 {
		return nil
	}
	out := map[string]any{}
	for _, key := range []string{
		"template",
		"model",
		"mechanism",
		"intervention",
		"expected_effect",
		"epochs",
		"batch_size",
		"learning_rate",
		"image_size",
		"resolution_strategy",
		"optimizer",
		"scheduler",
		"weight_decay",
		"dropout",
		"optimizer_momentum",
		"scheduler_step_size",
		"scheduler_gamma",
		"label_smoothing",
		"gradient_clip_norm",
		"augmentation_policy",
		"class_balancing",
		"sampling_strategy",
		"early_stopping_patience",
		"pretrained",
		"freeze_backbone",
		"fine_tune_strategy",
		"automl_study_id",
		"automl_suggestion_id",
		"automl_summary",
	} {
		if value, ok := config[key]; ok {
			out[key] = compactAnyValue(value)
		}
	}
	for _, key := range []string{"preprocessing", "augmentation_policy_config", "class_balancing_config"} {
		if value, ok := config[key]; ok {
			if nested, ok := value.(map[string]any); ok {
				out[key] = compactAnyMap(nested, 8)
			}
		}
	}
	if value, ok := config["evidence_used"]; ok {
		out["evidence_used"] = cappedStrings(stringsFromAny(value), 6)
	}
	return compactNonZeroMap(out)
}

func compactAutoMLPlanSummary(config *automl.ExperimentAutoML) map[string]any {
	if config == nil || !config.Enabled {
		return nil
	}
	out := map[string]any{
		"enabled":           config.Enabled,
		"sampler":           config.Sampler,
		"seed":              config.Seed,
		"final_values":      config.FinalValues,
		"value_provenance":  config.ValueProvenance,
		"validation_status": config.ValidationStatus,
		"validation_errors": config.ValidationErrors,
	}
	if config.SearchSpace != nil {
		names := []string{}
		for _, parameter := range config.SearchSpace.Parameters {
			names = append(names, parameter.Name)
		}
		out["search_parameters"] = names
	}
	return compactNonZeroMap(out)
}

func sortedTrainingMonitorMetrics(metrics []jobs.EpochMetric) []jobs.EpochMetric {
	out := append([]jobs.EpochMetric(nil), metrics...)
	sort.SliceStable(out, func(i, j int) bool {
		return out[i].Epoch < out[j].Epoch
	})
	return out
}

func recentTrainingMonitorMetrics(metrics []jobs.EpochMetric, targetMetric string, limit int) []map[string]any {
	if len(metrics) == 0 || limit <= 0 {
		return []map[string]any{}
	}
	start := len(metrics) - minInt(len(metrics), limit)
	out := make([]map[string]any, 0, len(metrics)-start)
	for _, metric := range metrics[start:] {
		row := map[string]any{"epoch": metric.Epoch}
		if value, ok := metricValue(metric.Metrics, targetMetric, "macro_f1", "accuracy"); ok {
			row["target_metric_value"] = value
		}
		for _, key := range []string{"macro_f1", "accuracy", "train_loss", "val_loss"} {
			if value, ok := metricValue(metric.Metrics, key); ok {
				row[key] = value
			}
		}
		out = append(out, row)
	}
	return out
}

func firstMetricValue(metrics []jobs.EpochMetric, keys ...string) (float64, bool) {
	for _, metric := range metrics {
		if value, ok := metricValue(metric.Metrics, keys...); ok {
			return value, true
		}
	}
	return 0, false
}

func lastMetricValue(metrics []jobs.EpochMetric, keys ...string) (float64, bool) {
	for index := len(metrics) - 1; index >= 0; index-- {
		if value, ok := metricValue(metrics[index].Metrics, keys...); ok {
			return value, true
		}
	}
	return 0, false
}

func firstLastMetricValues(metrics []jobs.EpochMetric, keys ...string) (float64, float64, bool) {
	first, firstOK := firstMetricValue(metrics, keys...)
	last, lastOK := lastMetricValue(metrics, keys...)
	return first, last, firstOK && lastOK
}

func metricValue(metrics map[string]float64, keys ...string) (float64, bool) {
	for _, key := range keys {
		if value, ok := metrics[key]; ok {
			return value, true
		}
	}
	return 0, false
}

func mapFloatValue(values map[string]any, key string) (float64, bool) {
	switch typed := values[key].(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	default:
		return 0, false
	}
}

func trainingMonitorInstabilityScore(recent []map[string]any) float64 {
	if len(recent) < 3 {
		return 0
	}
	values := []float64{}
	for _, row := range recent {
		if value, ok := mapFloatValue(row, "target_metric_value"); ok {
			values = append(values, value)
		}
	}
	if len(values) < 3 {
		return 0
	}
	directionChanges := 0
	previousDirection := 0
	for index := 1; index < len(values); index++ {
		delta := values[index] - values[index-1]
		direction := 0
		if delta > 0.001 {
			direction = 1
		} else if delta < -0.001 {
			direction = -1
		}
		if direction != 0 && previousDirection != 0 && direction != previousDirection {
			directionChanges++
		}
		if direction != 0 {
			previousDirection = direction
		}
	}
	return roundMonitorFloat(float64(directionChanges) / float64(len(values)-2))
}

func trainingMonitorDynamicsSummary(card map[string]any) string {
	recentDelta, _ := mapFloatValue(card, "recent_metric_delta")
	lossGap, _ := mapFloatValue(card, "final_loss_gap")
	plateau, _ := card["plateau_signal"].(string)
	switch {
	case lossGap > 0.15:
		return "Validation loss remains meaningfully above training loss; watch for overfitting."
	case plateau == "active" && math.Abs(recentDelta) <= 0.005:
		return "Recent target metric movement is flat; additional epochs may have limited value."
	case recentDelta > 0.01:
		return "Recent target metric is still improving; additional epochs may be justified if cost fits."
	default:
		return "Training dynamics do not show a strong continuation signal."
	}
}

type trainingMonitorClassRow struct {
	Label     string
	Precision float64
	Recall    float64
	F1        float64
	Support   float64
}

func trainingMonitorClassRows(perClass map[string]any) []trainingMonitorClassRow {
	rows := []trainingMonitorClassRow{}
	for label, value := range perClass {
		normalized := strings.ToLower(strings.TrimSpace(label))
		if normalized == "" || normalized == "accuracy" || strings.Contains(normalized, "avg") {
			continue
		}
		metrics := anyMap(value)
		if len(metrics) == 0 {
			continue
		}
		row := trainingMonitorClassRow{Label: label}
		row.Precision, _ = anyFloat(metrics["precision"])
		row.Recall, _ = anyFloat(metrics["recall"])
		row.F1, _ = anyFloat(firstPresent(metrics, "f1-score", "f1", "f1_score"))
		row.Support, _ = anyFloat(metrics["support"])
		rows = append(rows, row)
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].F1 == rows[j].F1 {
			return rows[i].Recall < rows[j].Recall
		}
		return rows[i].F1 < rows[j].F1
	})
	return rows
}

func cappedTrainingMonitorClassRows(rows []trainingMonitorClassRow, limit int) []map[string]any {
	if limit <= 0 {
		return []map[string]any{}
	}
	out := []map[string]any{}
	for _, row := range rows {
		out = append(out, map[string]any{
			"class":     row.Label,
			"precision": roundMonitorFloat(row.Precision),
			"recall":    roundMonitorFloat(row.Recall),
			"f1":        roundMonitorFloat(row.F1),
			"support":   row.Support,
		})
		if len(out) >= limit {
			break
		}
	}
	return out
}

func trainingMonitorMinoritySignal(summary runs.TrainingRunSummary, rows []trainingMonitorClassRow) bool {
	if summary.BestAccuracy-summary.BestMacroF1 > 0.05 {
		return true
	}
	for _, row := range rows {
		if row.Recall > 0 && row.Recall < 0.60 {
			return true
		}
	}
	return false
}

func trainingMonitorConfusionPairs(matrix [][]int, rows []trainingMonitorClassRow) ([]map[string]any, int) {
	type pair struct {
		From  string
		To    string
		Count int
	}
	labels := make([]string, 0, len(rows))
	for _, row := range rows {
		labels = append(labels, row.Label)
	}
	sort.Strings(labels)
	pairs := []pair{}
	offDiagonal := 0
	for fromIndex, row := range matrix {
		for toIndex, count := range row {
			if fromIndex == toIndex || count <= 0 {
				continue
			}
			offDiagonal += count
			pairs = append(pairs, pair{
				From:  classLabelForIndex(labels, fromIndex),
				To:    classLabelForIndex(labels, toIndex),
				Count: count,
			})
		}
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		return pairs[i].Count > pairs[j].Count
	})
	out := []map[string]any{}
	for _, pair := range pairs {
		out = append(out, map[string]any{
			"true_class":      pair.From,
			"predicted_class": pair.To,
			"count":           pair.Count,
		})
		if len(out) >= trainingMonitorMaxConfusionPairs {
			break
		}
	}
	return out, offDiagonal
}

func classLabelForIndex(labels []string, index int) string {
	if index >= 0 && index < len(labels) && strings.TrimSpace(labels[index]) != "" {
		return labels[index]
	}
	return fmt.Sprintf("class_%d", index)
}

func compactSelectedAnyMap(values map[string]any, keys []string) map[string]any {
	if len(values) == 0 {
		return nil
	}
	out := map[string]any{}
	for _, key := range keys {
		if value, ok := values[key]; ok {
			out[key] = compactAnyValue(value)
		}
	}
	return compactNonZeroMap(out)
}

func compactNonZeroMap(values map[string]any) map[string]any {
	out := map[string]any{}
	for key, value := range values {
		if isEmptyPromptValue(value) {
			continue
		}
		out[key] = value
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func isEmptyPromptValue(value any) bool {
	switch typed := value.(type) {
	case nil:
		return true
	case string:
		return strings.TrimSpace(typed) == ""
	case []string:
		return len(typed) == 0
	case []map[string]any:
		return len(typed) == 0
	case map[string]any:
		return len(typed) == 0
	case int:
		return typed == 0
	case float64:
		return typed == 0
	default:
		return false
	}
}

func anyMap(value any) map[string]any {
	switch typed := value.(type) {
	case map[string]any:
		return typed
	case map[string]float64:
		out := map[string]any{}
		for key, value := range typed {
			out[key] = value
		}
		return out
	default:
		return nil
	}
}

func firstPresent(values map[string]any, keys ...string) any {
	for _, key := range keys {
		if value, ok := values[key]; ok {
			return value
		}
	}
	return nil
}

func anyFloat(value any) (float64, bool) {
	switch typed := value.(type) {
	case float64:
		return typed, true
	case float32:
		return float64(typed), true
	case int:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case json.Number:
		parsed, err := typed.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

func trainingMonitorLatencySensitive(context ProjectObjectiveContext) bool {
	if strings.Contains(strings.ToLower(context.PrimaryObjective), "latency") || strings.Contains(strings.ToLower(context.PrimaryObjective), "live") {
		return true
	}
	for _, value := range append(context.DeploymentPriorities, context.Constraints...) {
		normalized := strings.ToLower(value)
		if strings.Contains(normalized, "latency") || strings.Contains(normalized, "real-time") || strings.Contains(normalized, "live") || strings.Contains(normalized, "fast") {
			return true
		}
	}
	return false
}

func attachTrainingMonitorPromptBudget(context map[string]any) {
	budget := map[string]any{
		"raw_sections_excluded": []string{
			"plan.experiments_full",
			"job.config_full",
			"training_run_evaluation_full",
			"epoch_metrics_full",
			"automl_trials_full",
			"automl_raw_search_history",
			"prior_memory_payloads",
		},
		"max_recent_epochs":   trainingMonitorMaxRecentEpochs,
		"max_worst_classes":   trainingMonitorMaxWorstClasses,
		"max_confusion_pairs": trainingMonitorMaxConfusionPairs,
		"max_memory_records":  trainingMonitorMaxMemoryRecords,
	}
	if retrievedRunMemory, ok := context["retrieved_run_memory"].([]map[string]any); ok {
		budget["retrieved_run_memory_count"] = len(retrievedRunMemory)
		if encoded, err := json.Marshal(retrievedRunMemory); err == nil {
			budget["retrieved_run_memory_approx_bytes"] = len(encoded)
		}
	}
	context["prompt_budget"] = budget
	encoded, err := json.Marshal(context)
	if err == nil {
		approxBytes := len(encoded)
		budget["approx_context_bytes"] = approxBytes
		budget["approx_input_bytes"] = approxBytes
		budget["approx_token_estimate"] = (approxBytes / 4) + 1
	}
}

func roundMonitorFloat(value float64) float64 {
	return math.Round(value*1000000) / 1000000
}

func compactEpochMetrics(metrics []jobs.EpochMetric) []map[string]any {
	out := make([]map[string]any, 0, len(metrics))
	for _, metric := range metrics {
		out = append(out, map[string]any{
			"epoch":   metric.Epoch,
			"metrics": metric.Metrics,
		})
	}
	return out
}

func compactRetrievedRunMemory(records []memory.MemoryRetrievalResult) []map[string]any {
	limit := minInt(len(records), trainingMonitorMaxMemoryRecords)
	out := make([]map[string]any, 0, limit)
	for _, record := range records {
		card := compactRetrievedRunMemoryRecord(record)
		if len(card) == 0 {
			continue
		}
		out = append(out, card)
		if len(out) >= trainingMonitorMaxMemoryRecords {
			break
		}
	}
	return out
}

func compactRetrievedRunMemoryRecord(record memory.MemoryRetrievalResult) map[string]any {
	sourceTable := firstTrainingMonitorMemoryText(record.SummaryCard, record.Metadata, "source_table")
	if sourceTable == "" {
		sourceTable = safeTrainingMonitorMemoryText(record.SourceTable)
	}
	sourceID := firstTrainingMonitorMemoryText(record.SummaryCard, record.Metadata, "source_id", "memory_id", "job_id")
	if sourceID == "" {
		sourceID = safeTrainingMonitorMemoryText(record.SourceID)
	}
	kind := firstTrainingMonitorMemoryText(record.SummaryCard, record.Metadata, "kind", "memory_kind")
	if kind == "" {
		kind = safeTrainingMonitorMemoryText(record.Kind)
	}

	out := map[string]any{
		"source_table": sourceTable,
		"source_id":    sourceID,
		"kind":         kind,
	}
	if record.Score != 0 {
		out["score"] = roundMonitorFloat(record.Score)
	}
	if reason := safeTrainingMonitorMemoryText(record.RetrievalReason); reason != "" {
		out["retrieval_reason"] = reason
	}

	addTrainingMonitorMemoryText(out, "model", record, "model", "best_model", "winner_model")
	addTrainingMonitorMemoryText(out, "model_family", record, "model_family", "family")
	addTrainingMonitorMemoryText(out, "dynamics", record, "dynamics", "training_dynamics", "training_dynamics_summary", "diagnostic_summary")
	addTrainingMonitorMemoryText(out, "outcome", record, "outcome", "outcome_status", "status")
	addTrainingMonitorMemoryText(out, "mechanism", record, "mechanism", "primary_mechanism", "mechanism_group")
	addTrainingMonitorMemoryText(out, "summary", record, "summary", "recommendation_summary", "quality_summary")
	addTrainingMonitorMemoryText(out, "lesson", record, "lesson")

	return compactNonZeroMap(out)
}

func addTrainingMonitorMemoryText(out map[string]any, outputKey string, record memory.MemoryRetrievalResult, keys ...string) {
	if text := firstTrainingMonitorMemoryText(record.SummaryCard, record.Metadata, keys...); text != "" {
		out[outputKey] = text
	}
}

func firstTrainingMonitorMemoryText(primary map[string]any, fallback map[string]any, keys ...string) string {
	for _, key := range keys {
		if text := safeTrainingMonitorMemoryText(primary[key]); text != "" {
			return text
		}
		if text := safeTrainingMonitorMemoryText(fallback[key]); text != "" {
			return text
		}
	}
	return ""
}

func safeTrainingMonitorMemoryText(value any) string {
	text, ok := value.(string)
	if !ok {
		return ""
	}
	text = strings.TrimSpace(text)
	if text == "" || unsafeTrainingMonitorMemoryText(text) {
		return ""
	}
	const maxLength = 320
	if len(text) > maxLength {
		text = strings.TrimSpace(text[:maxLength]) + "..."
	}
	return text
}

func unsafeTrainingMonitorMemoryText(text string) bool {
	lower := strings.ToLower(text)
	return strings.Contains(lower, "://") ||
		strings.Contains(lower, "embedding_text") ||
		strings.Contains(lower, "raw payload") ||
		strings.Contains(lower, "raw_payload") ||
		strings.Contains(lower, "full_run_evaluation") ||
		strings.Contains(lower, "epoch_history")
}

func compactMemoryRecords(records []memory.AgentMemoryRecord) []map[string]any {
	limit := minInt(len(records), trainingMonitorMaxMemoryRecords)
	out := make([]map[string]any, 0, limit)
	for _, record := range records {
		out = append(out, map[string]any{
			"agent_name": record.AgentName,
			"kind":       record.Kind,
			"summary":    record.Summary,
			"tags":       record.Tags,
		})
		if len(out) >= trainingMonitorMaxMemoryRecords {
			break
		}
	}
	return out
}
