package agents

import (
	"encoding/json"
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

const (
	PlannerToolDatasetProfile               = "dataset_profile"
	PlannerToolDatasetMetadataSummary       = "dataset_metadata_summary"
	PlannerToolVisualSummary                = "visual_summary"
	PlannerToolMemory                       = "memory"
	PlannerToolScorecards                   = "scorecards"
	PlannerToolLedger                       = "ledger"
	PlannerToolRunDetails                   = "run_details"
	PlannerToolPerClassDetail               = "per_class_detail"
	PlannerToolMechanismCoverage            = "mechanism_coverage"
	PlannerToolModelCatalog                 = "model_catalog"
	PlannerToolRecentPlannerFailures        = "recent_planner_failures"
	PlannerToolValidateCandidateExperiments = "validate_candidate_experiments"

	plannerToolMaxRuns           = 8
	plannerToolMaxMemoryRecords  = 8
	plannerToolMaxScorecards     = 8
	plannerToolMaxPerClassRows   = 12
	plannerToolMaxConfusionRows  = 12
	plannerToolMaxValidationRows = 6
)

type AgentInformationToolSpec struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type AgentInformationToolResult struct {
	ToolName string         `json:"tool_name"`
	Accepted bool           `json:"accepted"`
	Payload  map[string]any `json:"payload,omitempty"`
	Error    string         `json:"error,omitempty"`
}

type PlannerCandidateDryRunResult struct {
	Valid                   bool           `json:"valid"`
	ValidationStatus        string         `json:"validation_status"`
	ValidationError         string         `json:"validation_error,omitempty"`
	ProposedExperimentCount int            `json:"proposed_experiment_count"`
	CandidateCount          int            `json:"candidate_count"`
	SelectedCandidateCount  int            `json:"selected_candidate_count"`
	WouldWriteRows          bool           `json:"would_write_rows"`
	WouldScheduleJobs       bool           `json:"would_schedule_jobs"`
	Details                 map[string]any `json:"details,omitempty"`
}

type PlannerCandidateDryRunValidator func(ExperimentPlanningRecommendation) PlannerCandidateDryRunResult

type PlannerInformationToolOptions struct {
	ValidateCandidateExperiments PlannerCandidateDryRunValidator
}

func ExperimentPlannerInformationToolSpecs() []AgentInformationToolSpec {
	return []AgentInformationToolSpec{
		{
			Name:        PlannerToolDatasetProfile,
			Description: "Return the active dataset and objective profile as compact backend-curated facts.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        PlannerToolDatasetMetadataSummary,
			Description: "Return the active normalized dataset metadata safe summary only; no source rows, paths, storage URIs, raw previews, or sidecar contents.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        PlannerToolVisualSummary,
			Description: "Return accepted visual-analysis evidence summaries only; no raw images, paths, prompts, or manifests.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        PlannerToolMemory,
			Description: "Return capped planning memory summaries and supplied retrieved compact memory cards for the active project and dataset.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        PlannerToolScorecards,
			Description: "Return capped prior strategy scorecards for the active project and dataset.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        PlannerToolLedger,
			Description: "Return a capped completed-run ledger for the active plan and project.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        PlannerToolRunDetails,
			Description: "Return bounded metrics and evaluation facts for an in-scope run.",
			Parameters: objectSchema(map[string]any{
				"job_id": map[string]any{"type": "string"},
			}, []string{"job_id"}),
		},
		{
			Name:        PlannerToolPerClassDetail,
			Description: "Return capped per-class metrics and confusion details for an in-scope run.",
			Parameters: objectSchema(map[string]any{
				"job_id": map[string]any{"type": "string"},
			}, []string{"job_id"}),
		},
		{
			Name:        PlannerToolMechanismCoverage,
			Description: "Return tried, blocked, failed, and eligible mechanism coverage for the active planning scope.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        PlannerToolModelCatalog,
			Description: "Return the backend-supported model catalog.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        PlannerToolRecentPlannerFailures,
			Description: "Return recent planner validation feedback and failed strategy summaries.",
			Parameters:  emptyObjectSchema(),
		},
		{
			Name:        PlannerToolValidateCandidateExperiments,
			Description: "Dry-run validate a stringified candidate ExperimentPlanningRecommendation JSON object. This never writes rows or schedules jobs.",
			Parameters: objectSchema(map[string]any{
				"recommendation_json": map[string]any{
					"type":        "string",
					"description": "Stringified draft ExperimentPlanningRecommendation JSON object.",
				},
			}, []string{"recommendation_json"}),
		},
	}
}

func ExecuteExperimentPlannerInformationTool(input ExperimentPlannerInput, name string, args map[string]any, options PlannerInformationToolOptions) AgentInformationToolResult {
	normalized := strings.ToLower(strings.TrimSpace(name))
	if args == nil {
		args = map[string]any{}
	}
	snapshot := BuildPlannerContextSnapshot(input)

	switch normalized {
	case PlannerToolDatasetProfile:
		return acceptedPlannerTool(normalized, map[string]any{
			"project":           snapshot.Project,
			"dataset_card":      snapshot.DatasetCard,
			"objective_context": snapshot.ObjectiveContext,
			"label_quality":     snapshot.LabelQualityCard,
			"prompt_budget":     snapshot.PromptBudget,
			"raw_profile":       "excluded",
		})
	case PlannerToolDatasetMetadataSummary:
		return acceptedPlannerTool(normalized, map[string]any{
			"dataset_id":                  snapshot.DatasetCard.ID,
			"metadata_summary":            snapshot.DatasetCard.MetadataSummary,
			"agent_safe_metadata_summary": snapshot.DatasetCard.AgentSafeMetadataSummary,
			"raw_metadata":                "excluded",
			"excluded": []string{
				"source rows",
				"source relative paths",
				"storage URIs",
				"raw previews",
				"sidecar contents",
				"manifest records",
			},
		})
	case PlannerToolVisualSummary:
		return acceptedPlannerTool(normalized, map[string]any{
			"visual_evidence": snapshot.VisualEvidence,
		})
	case PlannerToolMemory:
		payload := map[string]any{
			"successful_strategy_memory": capPlannerStrategyMemory(input.SuccessfulStrategyMemory, plannerToolMaxMemoryRecords),
			"failed_strategy_memory":     capPlannerStrategyMemory(input.FailedStrategyMemory, plannerToolMaxMemoryRecords),
			"rejected_strategy_memory":   capRejectedPlannerOptions(input.RejectedStrategyMemory, plannerToolMaxMemoryRecords),
		}
		if snapshot.RetrievedMemory != nil {
			payload["retrieved_memory"] = snapshot.RetrievedMemory
		}
		return acceptedPlannerTool(normalized, payload)
	case PlannerToolScorecards:
		return acceptedPlannerTool(normalized, map[string]any{
			"scorecards": capPlannerStrategyScorecards(input.StrategyScorecards, plannerToolMaxScorecards),
		})
	case PlannerToolLedger:
		return acceptedPlannerTool(normalized, map[string]any{
			"completed_experiment_ledger": capPlannerExperimentLog(snapshot.CompletedExperimentLog, plannerToolMaxRuns),
			"champion_card":               snapshot.ChampionCard,
			"training_dynamics_card":      snapshot.TrainingDynamicsCard,
			"source_plan_card":            snapshot.SourcePlanCard,
		})
	case PlannerToolRunDetails:
		jobID := strings.TrimSpace(stringFromAny(args["job_id"]))
		return plannerRunDetailsTool(input, normalized, jobID)
	case PlannerToolPerClassDetail:
		jobID := strings.TrimSpace(stringFromAny(args["job_id"]))
		return plannerPerClassDetailTool(input, normalized, jobID)
	case PlannerToolMechanismCoverage:
		return acceptedPlannerTool(normalized, map[string]any{
			"mechanism_coverage_card":             snapshot.MechanismCoverageCard,
			"backend_validation_gated_methods":    snapshot.BackendGatedMethods,
			"gated_methods_have_scheduling_power": false,
		})
	case PlannerToolModelCatalog:
		return acceptedPlannerTool(normalized, map[string]any{
			"model_catalog": input.ModelCatalog,
		})
	case PlannerToolRecentPlannerFailures:
		return acceptedPlannerTool(normalized, map[string]any{
			"planner_validation_feedback": capPlannerValidationFeedback(input.ValidationFeedback, plannerToolMaxValidationRows),
			"failed_strategy_memory":      capPlannerStrategyMemory(input.FailedStrategyMemory, plannerToolMaxMemoryRecords),
			"rejected_strategy_memory":    capRejectedPlannerOptions(input.RejectedStrategyMemory, plannerToolMaxMemoryRecords),
			"blocked_repeats":             snapshot.BlockedRepeats,
		})
	case PlannerToolValidateCandidateExperiments:
		return plannerValidateCandidateExperimentsTool(input, normalized, args, options)
	default:
		return AgentInformationToolResult{
			ToolName: normalized,
			Accepted: false,
			Error:    fmt.Sprintf("planner information tool %q is not allowlisted", name),
		}
	}
}

func acceptedPlannerTool(name string, payload map[string]any) AgentInformationToolResult {
	return AgentInformationToolResult{
		ToolName: name,
		Accepted: true,
		Payload:  payload,
	}
}

func plannerRunDetailsTool(input ExperimentPlannerInput, toolName string, jobID string) AgentInformationToolResult {
	if jobID == "" {
		return rejectedPlannerTool(toolName, "job_id is required")
	}
	job, ok := plannerScopedJob(input, jobID)
	if !ok {
		return rejectedPlannerTool(toolName, "job_id is outside the active planner scope")
	}
	summary, _ := plannerSummaryForJob(input.PriorSummaries, input.PlanSummaries, jobID)
	evaluation, _ := plannerEvaluationForJob(input.PriorEvaluations, input.PlanEvaluations, jobID)
	return acceptedPlannerTool(toolName, map[string]any{
		"job":        compactPlannerJob(job),
		"summary":    compactPlannerRunSummary(summary),
		"evaluation": compactPlannerRunEvaluation(evaluation),
		"metrics":    compactEpochMetrics(cappedPlannerMetrics(input.PlanMetrics[jobID], plannerToolMaxRuns)),
	})
}

func plannerPerClassDetailTool(input ExperimentPlannerInput, toolName string, jobID string) AgentInformationToolResult {
	if jobID == "" {
		return rejectedPlannerTool(toolName, "job_id is required")
	}
	if _, ok := plannerScopedJob(input, jobID); !ok {
		return rejectedPlannerTool(toolName, "job_id is outside the active planner scope")
	}
	evaluation, ok := plannerEvaluationForJob(input.PriorEvaluations, input.PlanEvaluations, jobID)
	if !ok {
		return acceptedPlannerTool(toolName, map[string]any{
			"job_id":  jobID,
			"message": "No training evaluation is available for this job.",
		})
	}
	return acceptedPlannerTool(toolName, map[string]any{
		"job_id":            jobID,
		"per_class_metrics": capPlannerPerClassMetrics(evaluation.PerClassMetrics, plannerToolMaxPerClassRows),
		"confusion_matrix":  capPlannerConfusionMatrix(evaluation.ConfusionMatrix, plannerToolMaxConfusionRows),
		"objective_profile": compactAnyMap(evaluation.ObjectiveProfile, 10),
		"holistic_scores":   compactAnyMap(evaluation.HolisticScores, 10),
	})
}

func plannerValidateCandidateExperimentsTool(input ExperimentPlannerInput, toolName string, args map[string]any, options PlannerInformationToolOptions) AgentInformationToolResult {
	recommendation, err := plannerRecommendationFromToolArgs(args)
	if err != nil {
		return rejectedPlannerTool(toolName, err.Error())
	}
	recommendation.AgentName = ExperimentPlannerAgentName
	if recommendation.DecisionType == "" {
		recommendation.DecisionType = decisions.TypeAddExperiments
	}
	recommendation, finalizeErr := FinalizePlannerRecommendation(input, recommendation)

	result := PlannerCandidateDryRunResult{
		ValidationStatus:        "valid",
		ProposedExperimentCount: len(recommendation.ProposedExperiments),
		CandidateCount:          len(recommendation.CandidateHypotheses),
		SelectedCandidateCount:  len(recommendation.ProposedExperiments),
		WouldWriteRows:          false,
		WouldScheduleJobs:       false,
		Details: map[string]any{
			"dry_run_only":       true,
			"candidate_rankings": recommendation.CandidateRankings,
		},
	}
	if finalizeErr != nil {
		result.Valid = false
		result.ValidationStatus = "invalid"
		result.ValidationError = finalizeErr.Error()
		return acceptedPlannerTool(toolName, map[string]any{"dry_run_validation": result})
	}
	if err := validateExperimentPlanningRecommendation(recommendation, maxPlannerExperiments(input.MaxExperiments)); err != nil {
		result.Valid = false
		result.ValidationStatus = "invalid"
		result.ValidationError = err.Error()
		return acceptedPlannerTool(toolName, map[string]any{"dry_run_validation": result})
	}
	if options.ValidateCandidateExperiments != nil {
		result = options.ValidateCandidateExperiments(recommendation)
		result.WouldWriteRows = false
		result.WouldScheduleJobs = false
	} else {
		result.Valid = true
		result.Details["backend_validation"] = "not_attached"
	}
	return acceptedPlannerTool(toolName, map[string]any{"dry_run_validation": result})
}

func plannerRecommendationFromToolArgs(args map[string]any) (ExperimentPlanningRecommendation, error) {
	for _, key := range []string{"recommendation_json", "recommendation"} {
		if text, ok := args[key].(string); ok && strings.TrimSpace(text) != "" {
			var recommendation ExperimentPlanningRecommendation
			if err := json.Unmarshal([]byte(text), &recommendation); err != nil {
				return ExperimentPlanningRecommendation{}, fmt.Errorf("recommendation draft has invalid JSON")
			}
			return recommendation, nil
		}
	}

	value, ok := args["recommendation"]
	if !ok {
		value = args
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return ExperimentPlanningRecommendation{}, fmt.Errorf("candidate recommendation could not be encoded")
	}
	var recommendation ExperimentPlanningRecommendation
	if err := json.Unmarshal(blob, &recommendation); err != nil {
		return ExperimentPlanningRecommendation{}, fmt.Errorf("candidate recommendation has invalid shape")
	}
	return recommendation, nil
}

func rejectedPlannerTool(name string, message string) AgentInformationToolResult {
	return AgentInformationToolResult{
		ToolName: name,
		Accepted: false,
		Error:    message,
	}
}

func plannerScopedJob(input ExperimentPlannerInput, jobID string) (jobs.ExperimentJob, bool) {
	for _, job := range append(append([]jobs.ExperimentJob{}, input.PlanJobs...), input.PriorJobs...) {
		if job.ID == jobID && job.ProjectID == input.Project.ID {
			return job, true
		}
	}
	return jobs.ExperimentJob{}, false
}

func plannerSummaryForJob(primary []runs.TrainingRunSummary, fallback []runs.TrainingRunSummary, jobID string) (runs.TrainingRunSummary, bool) {
	for _, summary := range append(append([]runs.TrainingRunSummary{}, primary...), fallback...) {
		if summary.JobID == jobID {
			return summary, true
		}
	}
	return runs.TrainingRunSummary{}, false
}

func plannerEvaluationForJob(primary []runs.TrainingRunEvaluation, fallback []runs.TrainingRunEvaluation, jobID string) (runs.TrainingRunEvaluation, bool) {
	for _, evaluation := range append(append([]runs.TrainingRunEvaluation{}, primary...), fallback...) {
		if evaluation.JobID == jobID {
			return evaluation, true
		}
	}
	return runs.TrainingRunEvaluation{}, false
}

func compactPlannerJob(job jobs.ExperimentJob) map[string]any {
	return map[string]any{
		"id":           job.ID,
		"template":     job.Template,
		"status":       job.Status,
		"config":       compactAnyMap(job.Config, 12),
		"created_at":   job.CreatedAt,
		"started_at":   job.StartedAt,
		"completed_at": job.CompletedAt,
		"error":        job.Error,
	}
}

func compactPlannerRunSummary(summary runs.TrainingRunSummary) map[string]any {
	if summary.JobID == "" {
		return nil
	}
	return map[string]any{
		"job_id":             summary.JobID,
		"plan_id":            summary.PlanID,
		"dataset_id":         summary.DatasetID,
		"status":             summary.Status,
		"model":              summary.Model,
		"provider":           summary.Provider,
		"best_accuracy":      summary.BestAccuracy,
		"best_macro_f1":      summary.BestMacroF1,
		"final_train_loss":   summary.FinalTrainLoss,
		"final_val_loss":     summary.FinalValLoss,
		"epochs_completed":   summary.EpochsCompleted,
		"runtime_seconds":    summary.RuntimeSeconds,
		"estimated_cost_usd": summary.EstimatedCostUSD,
	}
}

func compactPlannerRunEvaluation(evaluation runs.TrainingRunEvaluation) map[string]any {
	if evaluation.JobID == "" {
		return nil
	}
	return map[string]any{
		"job_id":                 evaluation.JobID,
		"summary":                evaluation.RecommendationSummary,
		"model_profile":          compactAnyMap(evaluation.ModelProfile, 10),
		"objective_profile":      compactAnyMap(evaluation.ObjectiveProfile, 10),
		"holistic_scores":        compactAnyMap(evaluation.HolisticScores, 10),
		"per_class_metric_count": len(evaluation.PerClassMetrics),
		"confusion_rows":         len(evaluation.ConfusionMatrix),
	}
}

func cappedPlannerMetrics(metrics []jobs.EpochMetric, limit int) []jobs.EpochMetric {
	if limit <= 0 || len(metrics) <= limit {
		return append([]jobs.EpochMetric(nil), metrics...)
	}
	return append([]jobs.EpochMetric(nil), metrics[len(metrics)-limit:]...)
}

func capPlannerPerClassMetrics(values map[string]any, limit int) map[string]any {
	if len(values) == 0 || limit <= 0 {
		return map[string]any{}
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		if strings.TrimSpace(key) != "" {
			keys = append(keys, key)
		}
	}
	out := map[string]any{}
	for _, key := range cappedStrings(keys, limit) {
		out[key] = compactAnyValue(values[key])
	}
	return out
}

func capPlannerConfusionMatrix(values [][]int, limit int) [][]int {
	if limit <= 0 || len(values) == 0 {
		return [][]int{}
	}
	rowLimit := minInt(len(values), limit)
	out := make([][]int, 0, rowLimit)
	for _, row := range values[:rowLimit] {
		columnLimit := minInt(len(row), limit)
		out = append(out, append([]int(nil), row[:columnLimit]...))
	}
	return out
}

func capPlannerStrategyMemory(values []PlannerStrategyMemory, limit int) []PlannerStrategyMemory {
	if limit <= 0 || len(values) == 0 {
		return []PlannerStrategyMemory{}
	}
	if len(values) > limit {
		return append([]PlannerStrategyMemory(nil), values[:limit]...)
	}
	return append([]PlannerStrategyMemory(nil), values...)
}

func capPlannerStrategyScorecards(values []PlannerStrategyScorecard, limit int) []PlannerStrategyScorecard {
	if limit <= 0 || len(values) == 0 {
		return []PlannerStrategyScorecard{}
	}
	if len(values) > limit {
		return append([]PlannerStrategyScorecard(nil), values[:limit]...)
	}
	return append([]PlannerStrategyScorecard(nil), values...)
}

func capPlannerExperimentLog(values []PlannerExperimentLog, limit int) []PlannerExperimentLog {
	if limit <= 0 || len(values) == 0 {
		return []PlannerExperimentLog{}
	}
	if len(values) > limit {
		return append([]PlannerExperimentLog(nil), values[:limit]...)
	}
	return append([]PlannerExperimentLog(nil), values...)
}

func capPlannerValidationFeedback(values []PlannerValidationFeedback, limit int) []PlannerValidationFeedback {
	if limit <= 0 || len(values) == 0 {
		return []PlannerValidationFeedback{}
	}
	if len(values) > limit {
		return append([]PlannerValidationFeedback(nil), values[:limit]...)
	}
	return append([]PlannerValidationFeedback(nil), values...)
}

func emptyObjectSchema() map[string]any {
	return objectSchema(map[string]any{}, []string{})
}

func objectSchema(properties map[string]any, required []string) map[string]any {
	return map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}
}

func plannedExperimentsFromRecommendationDraft(recommendation ExperimentPlanningRecommendation) []plans.PlannedExperiment {
	experiments := append([]plans.PlannedExperiment(nil), recommendation.ProposedExperiments...)
	for _, mechanism := range recommendation.ProposalMechanisms {
		if mechanism.ExperimentIndex < 0 || mechanism.ExperimentIndex >= len(experiments) {
			continue
		}
		experiments[mechanism.ExperimentIndex].Mechanism = mechanism.Mechanism
		experiments[mechanism.ExperimentIndex].Intervention = mechanism.Intervention
		experiments[mechanism.ExperimentIndex].EvidenceUsed = append([]string(nil), mechanism.EvidenceUsed...)
		experiments[mechanism.ExperimentIndex].ExpectedEffect = mechanism.ExpectedEffect
	}
	return experiments
}
