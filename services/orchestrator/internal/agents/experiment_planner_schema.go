package agents

import (
	"sort"

	"model-express/services/orchestrator/internal/llm"
)

func experimentPlannerStructuredOutputSchema() *llm.JSONSchemaFormat {
	return &llm.JSONSchemaFormat{
		Name:   "experiment_planning_recommendation",
		Strict: true,
		Schema: experimentPlannerRecommendationJSONSchema(),
	}
}

func experimentPlannerRecommendationJSONSchema() map[string]any {
	properties := map[string]any{
		"summary":                         plannerSchemaString(),
		"decision_type":                   plannerSchemaStringEnum("ADD_EXPERIMENTS", "SELECT_CHAMPION", "STOP_PROJECT", "WAIT"),
		"rationale":                       plannerSchemaString(),
		"confidence":                      plannerSchemaNumber(),
		"planning_mode":                   plannerSchemaStringEnum("explore", "exploit", "champion_challenge", "preprocessing_ablation", "class_imbalance_ablation", "stop_or_select"),
		"deterministic_diagnosis_used":    plannerSchemaStringArray(),
		"evidence_used":                   plannerSchemaStringArray(),
		"hypothesis":                      plannerSchemaString(),
		"primary_mechanism":               plannerSchemaString(),
		"governor_compliance":             plannerGovernorComplianceSchema(),
		"expected_failure_modes":          plannerSchemaStringArray(),
		"dataset_preprocessing_rationale": plannerSchemaString(),
		"changed_variables":               plannerSchemaStringArray(),
		"success_criteria":                plannerSchemaString(),
		"stop_condition":                  plannerSchemaString(),
		"deployment_tradeoff":             plannerSchemaString(),
		"candidate_hypotheses":            plannerSchemaArray(plannerCandidateHypothesisSchema()),
		"proposed_experiments":            plannerSchemaArray(plannerPlannedExperimentSchema()),
		"proposal_mechanisms":             plannerSchemaArray(plannerProposalMechanismSchema()),
		"champion_job_id":                 plannerSchemaString(),
		"why_can_beat_champion":           plannerSchemaString(),
		"expected_delta_vs_champion":      plannerSchemaNumber(),
		"stop_reason":                     plannerSchemaString(),
		"risks":                           plannerSchemaStringArray(),
		"expected_tradeoffs":              plannerSchemaStringArray(),
		"novelty_notes":                   plannerSchemaStringArray(),
		"rejected_options":                plannerSchemaArray(plannerRejectedOptionSchema()),
		"tags":                            plannerSchemaStringArray(),
	}
	return plannerSchemaObject(properties, []string{
		"summary",
		"decision_type",
		"rationale",
		"confidence",
		"planning_mode",
		"deterministic_diagnosis_used",
		"evidence_used",
		"hypothesis",
		"primary_mechanism",
		"governor_compliance",
		"expected_failure_modes",
		"dataset_preprocessing_rationale",
		"changed_variables",
		"success_criteria",
		"stop_condition",
		"deployment_tradeoff",
		"candidate_hypotheses",
		"proposed_experiments",
		"proposal_mechanisms",
		"champion_job_id",
		"why_can_beat_champion",
		"expected_delta_vs_champion",
		"stop_reason",
		"risks",
		"expected_tradeoffs",
		"novelty_notes",
		"rejected_options",
		"tags",
	})
}

func plannerGovernorComplianceSchema() map[string]any {
	properties := map[string]any{
		"blocked_mechanisms_seen":      plannerSchemaStringArray(),
		"avoided_blocked_mechanisms":   plannerSchemaBoolean(),
		"why_allowed_to_continue":      plannerSchemaString(),
		"expected_value_justification": plannerSchemaString(),
	}
	return plannerSchemaObject(properties, []string{
		"blocked_mechanisms_seen",
		"avoided_blocked_mechanisms",
		"why_allowed_to_continue",
		"expected_value_justification",
	})
}

func plannerRejectedOptionSchema() map[string]any {
	properties := map[string]any{
		"option":       plannerSchemaString(),
		"reason":       plannerSchemaString(),
		"evidence":     plannerSchemaString(),
		"applies_when": plannerSchemaStringArray(),
	}
	return plannerSchemaObject(properties, []string{"option", "reason", "evidence", "applies_when"})
}

func plannerProposalMechanismSchema() map[string]any {
	properties := map[string]any{
		"experiment_index": plannerSchemaInteger(),
		"mechanism":        plannerSchemaString(),
		"intervention":     plannerSchemaString(),
		"evidence_used":    plannerSchemaStringArray(),
		"expected_effect":  plannerSchemaString(),
	}
	return plannerSchemaObject(properties, []string{"experiment_index", "mechanism", "intervention", "evidence_used", "expected_effect"})
}

func plannerCandidateHypothesisSchema() map[string]any {
	properties := map[string]any{
		"hypothesis":                 plannerSchemaString(),
		"planning_mode":              plannerSchemaStringEnum("explore", "exploit", "champion_challenge", "preprocessing_ablation", "class_imbalance_ablation", "stop_or_select"),
		"mechanism":                  plannerSchemaString(),
		"intervention":               plannerSchemaString(),
		"proposed_changes":           plannerProposedChangesSchema(),
		"expected_effect":            plannerSchemaString(),
		"expected_metric_impact":     plannerSchemaNumber(),
		"forecast":                   plannerCandidateForecastSchema(),
		"expected_tradeoffs":         plannerSchemaStringArray(),
		"risk":                       plannerSchemaString(),
		"cost_level":                 plannerSchemaString(),
		"novelty_score":              plannerSchemaNumber(),
		"evidence_used":              plannerSchemaStringArray(),
		"similar_success_memory_ids": plannerSchemaStringArray(),
		"similar_failure_memory_ids": plannerSchemaStringArray(),
		"experiment_config":          plannerPlannedExperimentSchema(),
	}
	return plannerSchemaObject(properties, []string{
		"hypothesis",
		"planning_mode",
		"mechanism",
		"intervention",
		"proposed_changes",
		"expected_effect",
		"expected_metric_impact",
		"forecast",
		"expected_tradeoffs",
		"risk",
		"cost_level",
		"novelty_score",
		"evidence_used",
		"similar_success_memory_ids",
		"similar_failure_memory_ids",
		"experiment_config",
	})
}

func plannerCandidateForecastSchema() map[string]any {
	properties := map[string]any{
		"forecast_target":   plannerSchemaString(),
		"metric_direction":  plannerSchemaStringEnum("higher_is_better", "lower_is_better"),
		"score_basis":       plannerSchemaString(),
		"score_version":     plannerSchemaString(),
		"baseline_job_id":   plannerSchemaString(),
		"baseline_score":    plannerSchemaNumber(),
		"predicted_delta":   plannerSchemaNumber(),
		"prediction_source": plannerSchemaStringEnum("candidate.expected_metric_impact"),
		"units":             plannerSchemaStringEnum("fractional_score"),
		"valid_range": plannerSchemaObject(map[string]any{
			"min": plannerSchemaNumber(),
			"max": plannerSchemaNumber(),
		}, []string{"min", "max"}),
	}
	return plannerSchemaObject(properties, []string{
		"forecast_target",
		"metric_direction",
		"score_basis",
		"score_version",
		"baseline_job_id",
		"baseline_score",
		"predicted_delta",
		"prediction_source",
		"units",
		"valid_range",
	})
}

func plannerProposedChangesSchema() map[string]any {
	properties := map[string]any{}
	for _, field := range []string{
		"mechanism",
		"model_family",
		"model",
		"template",
		"image_size",
		"resolution_strategy",
		"resize_strategy",
		"crop_strategy",
		"normalization",
		"bbox_mode",
		"augmentation_policy",
		"class_balancing",
		"sampling_strategy",
		"fine_tune_strategy",
		"optimizer",
		"scheduler",
		"learning_rate",
		"epochs",
		"batch_size",
		"weight_decay",
		"label_smoothing",
		"dropout",
		"target_metric",
	} {
		properties[field] = plannerSchemaString()
	}
	return plannerSchemaObject(properties, plannerSchemaRequiredKeys(properties))
}

func plannerPlannedExperimentSchema() map[string]any {
	properties := map[string]any{
		"template":                   plannerSchemaString(),
		"model":                      plannerSchemaString(),
		"mechanism":                  plannerSchemaString(),
		"intervention":               plannerSchemaString(),
		"evidence_used":              plannerSchemaStringArray(),
		"expected_effect":            plannerSchemaString(),
		"epochs":                     plannerSchemaInteger(),
		"batch_size":                 plannerSchemaInteger(),
		"learning_rate":              plannerSchemaNumber(),
		"reason":                     plannerSchemaString(),
		"image_size":                 plannerSchemaInteger(),
		"resolution_strategy":        plannerSchemaString(),
		"preprocessing":              plannerPreprocessingSchema(),
		"optimizer":                  plannerSchemaString(),
		"scheduler":                  plannerSchemaString(),
		"weight_decay":               plannerSchemaNumber(),
		"dropout":                    plannerSchemaNumber(),
		"optimizer_momentum":         plannerSchemaNumber(),
		"scheduler_step_size":        plannerSchemaInteger(),
		"scheduler_gamma":            plannerSchemaNumber(),
		"label_smoothing":            plannerSchemaNumber(),
		"gradient_clip_norm":         plannerSchemaNumber(),
		"augmentation_policy":        plannerSchemaString(),
		"augmentation_policy_config": plannerAugmentationPolicyConfigSchema(),
		"class_balancing":            plannerSchemaString(),
		"sampling_strategy":          plannerSchemaString(),
		"early_stopping_patience":    plannerSchemaInteger(),
		"strategy":                   plannerSchemaString(),
		"pretrained":                 plannerSchemaBoolean(),
		"freeze_backbone":            plannerSchemaBoolean(),
		"fine_tune_strategy":         plannerSchemaString(),
	}
	return plannerSchemaObject(properties, plannerSchemaRequiredKeys(properties))
}

func plannerPreprocessingSchema() map[string]any {
	properties := map[string]any{
		"resize_strategy":           plannerSchemaString(),
		"normalization":             plannerSchemaString(),
		"crop_strategy":             plannerSchemaString(),
		"bbox_mode":                 plannerSchemaString(),
		"use_dataset_normalization": plannerSchemaBoolean(),
	}
	return plannerSchemaObject(properties, plannerSchemaRequiredKeys(properties))
}

func plannerAugmentationPolicyConfigSchema() map[string]any {
	properties := map[string]any{
		"policy_type":        plannerSchemaString(),
		"magnitude":          plannerSchemaInteger(),
		"num_ops":            plannerSchemaInteger(),
		"num_magnitude_bins": plannerSchemaInteger(),
		"probability":        plannerSchemaNumber(),
		"alpha":              plannerSchemaNumber(),
	}
	return plannerSchemaObject(properties, plannerSchemaRequiredKeys(properties))
}

func plannerSchemaObject(properties map[string]any, required []string) map[string]any {
	return map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}
}

func plannerSchemaArray(item map[string]any) map[string]any {
	return map[string]any{
		"type":  "array",
		"items": item,
	}
}

func plannerSchemaStringArray() map[string]any {
	return plannerSchemaArray(plannerSchemaString())
}

func plannerSchemaString() map[string]any {
	return map[string]any{"type": "string"}
}

func plannerSchemaStringEnum(values ...string) map[string]any {
	return map[string]any{
		"type": "string",
		"enum": values,
	}
}

func plannerSchemaNumber() map[string]any {
	return map[string]any{"type": "number"}
}

func plannerSchemaInteger() map[string]any {
	return map[string]any{"type": "integer"}
}

func plannerSchemaBoolean() map[string]any {
	return map[string]any{"type": "boolean"}
}

func plannerSchemaRequiredKeys(properties map[string]any) []string {
	keys := make([]string, 0, len(properties))
	for key := range properties {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
