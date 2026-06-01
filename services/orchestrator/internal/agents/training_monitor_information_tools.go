package agents

import (
	"encoding/json"
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
)

const (
	TrainingMonitorToolEpochMetrics                = "get_epoch_metrics"
	TrainingMonitorToolTrainingEvaluation          = "get_training_evaluation"
	TrainingMonitorToolMemoryRecords               = "get_memory_records"
	TrainingMonitorToolPlanConfig                  = "get_plan_config"
	TrainingMonitorToolJobConfig                   = "get_job_config"
	TrainingMonitorToolObjectiveContext            = "get_objective_context"
	TrainingMonitorToolValidateRecommendationDraft = "validate_recommendation_draft"
)

type TrainingMonitorInformationToolOptions struct{}

func TrainingMonitorInformationToolSpecs() []AgentInformationToolSpec {
	return []AgentInformationToolSpec{
		{
			Name:        TrainingMonitorToolEpochMetrics,
			Description: "Return recent epoch metrics for this run only.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        TrainingMonitorToolTrainingEvaluation,
			Description: "Return bounded training evaluation facts for this run only.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        TrainingMonitorToolMemoryRecords,
			Description: "Return capped prior memory summaries and advisory retrieved run memory when supplied for this project/dataset/job context.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        TrainingMonitorToolPlanConfig,
			Description: "Return compact plan context for this run. This is informational and cannot schedule work.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        TrainingMonitorToolJobConfig,
			Description: "Return compact job configuration for this run only.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        TrainingMonitorToolObjectiveContext,
			Description: "Return the project objective context used to evaluate this run.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        TrainingMonitorToolValidateRecommendationDraft,
			Description: "Dry-run validate a stringified TrainingEvaluationRecommendation draft JSON object. This never proposes experiments or schedules work.",
			Parameters: objectSchema(map[string]any{
				"recommendation_json": map[string]any{
					"type":        "string",
					"description": "Stringified draft TrainingEvaluationRecommendation JSON object.",
				},
			}, []string{"recommendation_json"}),
		},
	}
}

func ExecuteTrainingMonitorInformationTool(input TrainingMonitorInput, name string, args map[string]any, _ TrainingMonitorInformationToolOptions) AgentInformationToolResult {
	normalized := strings.ToLower(strings.TrimSpace(name))
	if args == nil {
		args = map[string]any{}
	}

	switch normalized {
	case TrainingMonitorToolEpochMetrics:
		return acceptedMonitorTool(normalized, map[string]any{
			"job_id":  input.Job.ID,
			"metrics": compactEpochMetrics(sortedTrainingMonitorMetrics(input.Metrics)),
		})
	case TrainingMonitorToolTrainingEvaluation:
		return acceptedMonitorTool(normalized, map[string]any{
			"job_id":     input.Job.ID,
			"summary":    trainingMonitorRunSummaryCard(input),
			"evaluation": compactTrainingMonitorEvaluation(input),
		})
	case TrainingMonitorToolMemoryRecords:
		payload := map[string]any{
			"memory_records": compactMemoryRecords(input.MemoryRecords),
		}
		if retrievedRunMemory := compactRetrievedRunMemory(input.RetrievedRunMemory); len(retrievedRunMemory) > 0 {
			payload["retrieved_run_memory"] = retrievedRunMemory
		}
		return acceptedMonitorTool(normalized, payload)
	case TrainingMonitorToolPlanConfig:
		return acceptedMonitorTool(normalized, map[string]any{
			"plan": compactTrainingMonitorPlan(input.Plan),
		})
	case TrainingMonitorToolJobConfig:
		return acceptedMonitorTool(normalized, map[string]any{
			"job": compactTrainingMonitorJob(input.Job),
		})
	case TrainingMonitorToolObjectiveContext:
		return acceptedMonitorTool(normalized, map[string]any{
			"objective_context": input.ObjectiveContext,
			"objective_fit":     trainingMonitorObjectiveFitCard(input),
		})
	case TrainingMonitorToolValidateRecommendationDraft:
		return trainingMonitorValidateRecommendationDraftTool(normalized, args)
	default:
		return AgentInformationToolResult{
			ToolName: normalized,
			Accepted: false,
			Error:    fmt.Sprintf("training monitor information tool %q is not allowlisted", name),
		}
	}
}

func acceptedMonitorTool(name string, payload map[string]any) AgentInformationToolResult {
	return AgentInformationToolResult{
		ToolName: name,
		Accepted: true,
		Payload:  payload,
	}
}

func trainingMonitorValidateRecommendationDraftTool(toolName string, args map[string]any) AgentInformationToolResult {
	for _, key := range []string{"recommendation_json", "recommendation"} {
		if text, ok := args[key].(string); ok && strings.TrimSpace(text) != "" {
			return trainingMonitorValidateRecommendationDraftJSON(toolName, []byte(text))
		}
	}

	value, ok := args["recommendation"]
	if !ok {
		value = args
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return rejectedMonitorTool(toolName, "recommendation draft could not be encoded")
	}
	return trainingMonitorValidateRecommendationDraftJSON(toolName, blob)
}

func trainingMonitorValidateRecommendationDraftJSON(toolName string, blob []byte) AgentInformationToolResult {
	var recommendation TrainingEvaluationRecommendation
	if err := json.Unmarshal(blob, &recommendation); err != nil {
		return rejectedMonitorTool(toolName, "recommendation draft has invalid shape")
	}
	recommendation.AgentName = TrainingMonitorAgentName
	status := map[string]any{
		"dry_run_only":        true,
		"would_write_rows":    false,
		"would_schedule_jobs": false,
	}
	if err := validateTrainingEvaluation(recommendation); err != nil {
		status["valid"] = false
		status["validation_status"] = "invalid"
		status["validation_error"] = err.Error()
		return acceptedMonitorTool(toolName, map[string]any{"dry_run_validation": status})
	}
	status["valid"] = true
	status["validation_status"] = "valid"
	return acceptedMonitorTool(toolName, map[string]any{"dry_run_validation": status})
}

func rejectedMonitorTool(name string, message string) AgentInformationToolResult {
	return AgentInformationToolResult{
		ToolName: name,
		Accepted: false,
		Error:    message,
	}
}

func compactTrainingMonitorEvaluation(input TrainingMonitorInput) map[string]any {
	if input.Evaluation == nil {
		return map[string]any{
			"available": false,
		}
	}
	return map[string]any{
		"available":              true,
		"recommendation_summary": input.Evaluation.RecommendationSummary,
		"per_class_metrics":      capPlannerPerClassMetrics(input.Evaluation.PerClassMetrics, trainingMonitorMaxWorstClasses),
		"confusion_matrix":       capPlannerConfusionMatrix(input.Evaluation.ConfusionMatrix, trainingMonitorMaxConfusionPairs),
		"model_profile":          compactAnyMap(input.Evaluation.ModelProfile, 10),
		"objective_profile":      compactAnyMap(input.Evaluation.ObjectiveProfile, 10),
		"holistic_scores":        compactAnyMap(input.Evaluation.HolisticScores, 10),
	}
}

func compactTrainingMonitorPlan(plan plans.ExperimentPlan) map[string]any {
	experiments := make([]map[string]any, 0, len(plan.Experiments))
	for _, experiment := range plan.Experiments {
		experiments = append(experiments, compactPlannedExperimentProfile(experiment))
		if len(experiments) >= 5 {
			break
		}
	}
	return map[string]any{
		"id":                    plan.ID,
		"dataset_id":            plan.DatasetID,
		"target_metric":         plan.TargetMetric,
		"experiment_count":      len(plan.Experiments),
		"experiments_capped":    experiments,
		"recommended_workers":   plan.RecommendedWorkers,
		"estimated_minutes":     plan.EstimatedMinutes,
		"source_decision_id":    plan.SourceDecisionID,
		"no_scheduling_handle":  true,
		"informational_only":    true,
		"raw_experiments_count": len(plan.Experiments),
	}
}

func compactTrainingMonitorJob(job jobs.ExperimentJob) map[string]any {
	return map[string]any{
		"id":                   job.ID,
		"project_id":           job.ProjectID,
		"template":             job.Template,
		"status":               job.Status,
		"config":               compactJobConfigProfile(job.Config),
		"created_at":           job.CreatedAt,
		"started_at":           job.StartedAt,
		"completed_at":         job.CompletedAt,
		"error":                job.Error,
		"no_scheduling_handle": true,
	}
}

func trainingMonitorPayloadHasForbiddenAuthority(payload map[string]any) (string, bool) {
	if len(payload) == 0 {
		return "", false
	}
	for key, value := range payload {
		normalized := strings.ToLower(strings.TrimSpace(key))
		if trainingMonitorForbiddenAuthorityKey(normalized) {
			return key, true
		}
		if nested, ok := value.(map[string]any); ok {
			if child, found := trainingMonitorPayloadHasForbiddenAuthority(nested); found {
				return key + "." + child, true
			}
		}
	}
	return "", false
}

func trainingMonitorForbiddenAuthorityKey(key string) bool {
	switch key {
	case "proposed_experiments",
		"candidate_hypotheses",
		"proposal_mechanisms",
		"plan",
		"plans",
		"job",
		"jobs",
		"workers",
		"commands",
		"schedule",
		"schedule_jobs",
		"create_plan",
		"create_job",
		"create_worker",
		"export_champion",
		"inference_run",
		"dataset_mutation",
		"dataset_mutations":
		return true
	default:
		return false
	}
}
