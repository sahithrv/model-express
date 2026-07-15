package evals

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/llm"
)

type PlannerReplayVariant string

const (
	PlannerReplayVariantCurrentV1           PlannerReplayVariant = "current_v1"
	PlannerReplayVariantCompactStaticPrompt PlannerReplayVariant = "compact_static_prompt"
	PlannerReplayVariantContextV2           PlannerReplayVariant = "context_v2"
)

type PlannerReplayArtifact struct {
	FixtureName         string                       `json:"fixture_name"`
	TaskType            string                       `json:"task_type"`
	SchemaParseSuccess  bool                         `json:"schema_parse_success"`
	SchemaParseError    string                       `json:"schema_parse_error,omitempty"`
	RecommendationTitle string                       `json:"recommendation_title,omitempty"`
	CurrentPromptBytes  int                          `json:"current_prompt_bytes,omitempty"`
	CurrentPromptTokens int                          `json:"current_prompt_tokens,omitempty"`
	Variants            []PlannerReplayVariantResult `json:"variants"`
	BestVariant         PlannerReplayVariant         `json:"best_variant,omitempty"`
}

type PlannerReplayVariantResult struct {
	Variant                         PlannerReplayVariant `json:"variant"`
	PromptBytes                     int                  `json:"prompt_bytes"`
	ApproxPromptTokens              int                  `json:"approx_prompt_tokens"`
	SchemaParseSuccess              bool                 `json:"schema_parse_success"`
	BackendValidationPassed         bool                 `json:"backend_validation_passed"`
	FinalizerSucceeded              bool                 `json:"finalizer_succeeded"`
	FinalizerError                  string               `json:"finalizer_error,omitempty"`
	DuplicateSignatureRejectedCount int                  `json:"duplicate_signature_rejected_count,omitempty"`
	CandidateRankingScore           float64              `json:"candidate_ranking_score,omitempty"`
	CandidateMechanismDiversity     int                  `json:"candidate_mechanism_diversity,omitempty"`
	MechanismDiversity              int                  `json:"mechanism_diversity,omitempty"`
	TaskAligned                     bool                 `json:"task_aligned"`
	ValidModelSelection             bool                 `json:"valid_model_selection"`
	SelectedMechanisms              []string             `json:"selected_mechanisms,omitempty"`
	SelectedModels                  []string             `json:"selected_models,omitempty"`
	SelectedExperiments             int                  `json:"selected_experiments,omitempty"`
	Scores                          PlannerReplayScores  `json:"scores"`
	Rubric                          PlannerRubricScore   `json:"rubric"`
}

func ReplayPlannerResponse(fixture PlannerReplayFixture) ([]byte, error) {
	if len(fixture.Response) == 0 {
		return nil, errors.New("planner replay fixture has no response payload")
	}
	return json.Marshal(fixture.Response)
}

func ReplayPlannerResponseJSON(ctx context.Context, fixture PlannerReplayFixture) (PlannerReplayArtifact, error) {
	raw, err := ReplayPlannerResponse(fixture)
	if err != nil {
		return PlannerReplayArtifact{}, err
	}
	return ReplayPlannerResponseBytes(ctx, fixture, raw)
}

func ReplayPlannerResponseBytes(ctx context.Context, fixture PlannerReplayFixture, rawResponse []byte) (PlannerReplayArtifact, error) {
	_ = ctx // Retained for API compatibility; deterministic replay makes no calls.
	input := ExperimentPlannerInputFromReplayFixture(fixture)
	artifact := PlannerReplayArtifact{
		FixtureName: fixture.Name,
		TaskType:    strings.ToLower(strings.TrimSpace(input.DatasetInsights.TaskType)),
	}

	var recommendation agents.ExperimentPlanningRecommendation
	if err := json.Unmarshal(rawResponse, &recommendation); err != nil {
		artifact.SchemaParseError = err.Error()
		return artifact, err
	}
	artifact.SchemaParseSuccess = true
	artifact.RecommendationTitle = recommendation.Summary

	currentRequest, err := buildCurrentReplayRequest(input)
	if err != nil {
		return artifact, err
	}
	artifact.CurrentPromptBytes = replayApproximateJSONBytes(currentRequest)
	artifact.CurrentPromptTokens = replayApproximateTokens(artifact.CurrentPromptBytes)

	variants := []PlannerReplayVariant{
		PlannerReplayVariantCurrentV1,
		PlannerReplayVariantCompactStaticPrompt,
		PlannerReplayVariantContextV2,
	}
	results := make([]PlannerReplayVariantResult, 0, len(variants))
	for _, variant := range variants {
		// A request variant may change the model-facing projection, but the
		// backend finalizer always receives the same complete production input.
		scores, finalized, finalizeErr := scorePlannerRecommendationDetailed(input, recommendation, fixture.Expected)
		promptBytes, err := replayVariantPromptBytes(input, variant)
		if err != nil {
			return artifact, err
		}
		selectedMechanisms := selectedReplayMechanisms(finalized)
		selectedModels := replaySelectedModels(finalized)
		rubric := ScorePlannerRubric(input, rawResponse, PlannerRubricForFixture(fixture))
		result := PlannerReplayVariantResult{
			Variant:                         variant,
			PromptBytes:                     promptBytes,
			ApproxPromptTokens:              replayApproximateTokens(promptBytes),
			SchemaParseSuccess:              true,
			BackendValidationPassed:         scores.BackendValidationPassed,
			FinalizerSucceeded:              scores.FinalizerSucceeded,
			FinalizerError:                  scores.FinalizerError,
			DuplicateSignatureRejectedCount: scores.DuplicateSignatureRejectedCount,
			CandidateRankingScore:           scores.SelectedCandidateRankingScore,
			CandidateMechanismDiversity:     replayCandidateMechanismDiversity(recommendation.CandidateHypotheses),
			MechanismDiversity:              scores.MechanismDiversity,
			TaskAligned:                     scores.TaskAligned,
			ValidModelSelection:             scores.ValidModelSelection,
			SelectedMechanisms:              selectedMechanisms,
			SelectedModels:                  selectedModels,
			SelectedExperiments:             len(finalized.ProposedExperiments),
			Scores:                          scores,
			Rubric:                          rubric,
		}
		if finalizeErr != nil && result.FinalizerError == "" {
			result.FinalizerError = finalizeErr.Error()
		}
		results = append(results, result)
	}
	artifact.Variants = results
	artifact.BestVariant = replayBestPlannerVariant(results)
	return artifact, nil
}

func buildCurrentReplayRequest(input agents.ExperimentPlannerInput) (llm.JSONRequest, error) {
	agent := agents.NewExperimentPlannerAgent(nil, "replay-test-model")
	built, err := agent.BuildRequest(input, replayRequestVariant(PlannerReplayVariantCurrentV1))
	return built.Request, err
}

func replayVariantPromptBytes(input agents.ExperimentPlannerInput, variant PlannerReplayVariant) (int, error) {
	agent := agents.NewExperimentPlannerAgent(nil, "replay-model")
	built, err := agent.BuildRequest(input, replayRequestVariant(variant))
	if err != nil {
		return 0, err
	}
	return replayApproximateJSONBytes(built.Request), nil
}

func replayRequestVariant(variant PlannerReplayVariant) agents.ExperimentPlannerRequestVariant {
	switch variant {
	case PlannerReplayVariantCompactStaticPrompt:
		return agents.ExperimentPlannerRequestVariant{StaticPromptVersion: "compact_v1", ContextVersion: "v1"}
	case PlannerReplayVariantContextV2:
		return agents.ExperimentPlannerRequestVariant{StaticPromptVersion: "compact_v1", ContextVersion: "v2"}
	default:
		return agents.ExperimentPlannerRequestVariant{StaticPromptVersion: "v1", ContextVersion: "v1"}
	}
}

func replayBestPlannerVariant(results []PlannerReplayVariantResult) PlannerReplayVariant {
	if len(results) == 0 {
		return ""
	}
	ordered := append([]PlannerReplayVariantResult(nil), results...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].Rubric.QualityTier != ordered[j].Rubric.QualityTier {
			return ordered[i].Rubric.QualityTier > ordered[j].Rubric.QualityTier
		}
		if ordered[i].Rubric.SafetyPassed != ordered[j].Rubric.SafetyPassed {
			return ordered[i].Rubric.SafetyPassed && !ordered[j].Rubric.SafetyPassed
		}
		if ordered[i].Rubric.CorrectnessChecks != ordered[j].Rubric.CorrectnessChecks {
			return ordered[i].Rubric.CorrectnessChecks > ordered[j].Rubric.CorrectnessChecks
		}
		return ordered[i].PromptBytes < ordered[j].PromptBytes
	})
	return ordered[0].Variant
}

func replaySelectedModels(recommendation agents.ExperimentPlanningRecommendation) []string {
	models := []string{}
	seen := map[string]bool{}
	for _, experiment := range recommendation.ProposedExperiments {
		model := strings.ToLower(strings.TrimSpace(experiment.Model))
		if model == "" || seen[model] {
			continue
		}
		seen[model] = true
		models = append(models, experiment.Model)
	}
	return models
}

func replayCandidateMechanismDiversity(candidates []agents.CandidateHypothesis) int {
	if len(candidates) == 0 {
		return 0
	}
	unique := map[string]bool{}
	for _, candidate := range candidates {
		if mechanism := normalizeReplayValue(candidate.Mechanism); mechanism != "" {
			unique[mechanism] = true
		}
	}
	return len(unique)
}

func replayApproximateTokens(bytes int) int {
	if bytes <= 0 {
		return 0
	}
	return int(math.Ceil(float64(bytes) / 4.0))
}
