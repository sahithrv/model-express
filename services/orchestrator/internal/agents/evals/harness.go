package evals

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

type PlannerReplayVariant string

const (
	PlannerReplayVariantCurrentV1            PlannerReplayVariant = "current_v1"
	PlannerReplayVariantCompactStaticPrompt  PlannerReplayVariant = "compact_static_prompt"
	PlannerReplayVariantContextV2            PlannerReplayVariant = "context_v2"
	PlannerReplayVariantDistilledMemoryFirst PlannerReplayVariant = "distilled_memory_first"
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
}

type replayTraceGenerator struct {
	response []byte
	request  llm.JSONRequest
}

func (g *replayTraceGenerator) GenerateJSON(_ context.Context, req llm.JSONRequest) ([]byte, error) {
	g.request = req
	if len(g.response) == 0 {
		return nil, errors.New("replay generator has no response payload")
	}
	return g.response, nil
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

	_, currentRequest, err := captureCurrentReplayTrace(ctx, input, rawResponse)
	if err != nil {
		return artifact, err
	}
	artifact.CurrentPromptBytes = replayApproximateJSONBytes(currentRequest)
	artifact.CurrentPromptTokens = replayApproximateTokens(artifact.CurrentPromptBytes)

	variants := []PlannerReplayVariant{
		PlannerReplayVariantCurrentV1,
		PlannerReplayVariantCompactStaticPrompt,
		PlannerReplayVariantContextV2,
		PlannerReplayVariantDistilledMemoryFirst,
	}
	results := make([]PlannerReplayVariantResult, 0, len(variants))
	for _, variant := range variants {
		variantInput := replayVariantInput(input, variant)
		scores, finalized, finalizeErr := scorePlannerRecommendationDetailed(variantInput, recommendation, fixture.Expected)
		promptBytes := replayVariantPromptBytes(variantInput, variant)
		selectedMechanisms := selectedReplayMechanisms(finalized)
		selectedModels := replaySelectedModels(finalized)
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

func ReplayLiveMiniIfEnabled(ctx context.Context, agent agents.ExperimentPlannerAgent, fixture PlannerReplayFixture) (PlannerReplayArtifact, error) {
	if !plannerReplayLiveEnabled() {
		return PlannerReplayArtifact{}, errors.New("live planner replay disabled")
	}
	input := ExperimentPlannerInputFromReplayFixture(fixture)
	trace, err := agent.PlanWithTrace(ctx, input)
	if err != nil {
		return PlannerReplayArtifact{}, err
	}
	return ReplayPlannerResponseBytes(ctx, fixture, trace.RawOutput)
}

func captureCurrentReplayTrace(ctx context.Context, input agents.ExperimentPlannerInput, rawResponse []byte) (agents.ExperimentPlanningTrace, llm.JSONRequest, error) {
	gen := &replayTraceGenerator{response: rawResponse}
	agent := agents.NewExperimentPlannerAgent(gen, "replay-test-model")
	trace, err := agent.PlanWithTrace(ctx, input)
	return trace, gen.request, err
}

func replayVariantInput(input agents.ExperimentPlannerInput, variant PlannerReplayVariant) agents.ExperimentPlannerInput {
	switch variant {
	case PlannerReplayVariantContextV2:
		input = replayContextV2Input(input, false)
	case PlannerReplayVariantDistilledMemoryFirst:
		input = replayContextV2Input(input, true)
	}
	return input
}

func replayVariantPromptBytes(input agents.ExperimentPlannerInput, variant PlannerReplayVariant) int {
	switch variant {
	case PlannerReplayVariantCurrentV1:
		return replayPromptBytesForInput(input, currentPlannerSystemPrompt())
	case PlannerReplayVariantCompactStaticPrompt:
		return replayPromptBytesForInput(input, compactPlannerSystemPrompt())
	case PlannerReplayVariantContextV2:
		return replayPromptBytesForInput(replayContextV2Input(input, false), compactPlannerSystemPrompt())
	case PlannerReplayVariantDistilledMemoryFirst:
		return replayPromptBytesForInput(replayContextV2Input(input, true), compactPlannerSystemPrompt())
	default:
		return replayPromptBytesForInput(input, currentPlannerSystemPrompt())
	}
}

func replayPromptBytesForInput(input agents.ExperimentPlannerInput, systemPrompt string) int {
	contextBlob := replayContextBlob(input)
	request := llm.JSONRequest{
		Model:       "replay-model",
		Temperature: 0.35,
		Messages: []llm.Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: fmt.Sprintf("Context:\n%s", string(contextBlob))},
		},
	}
	return replayApproximateJSONBytes(request)
}

func replayContextBlob(input agents.ExperimentPlannerInput) []byte {
	snapshot := agents.BuildPlannerContextSnapshot(input)
	if blob, err := json.Marshal(snapshot); err == nil {
		return blob
	}
	return nil
}

func currentPlannerSystemPrompt() string {
	return strings.TrimSpace(`You are the Model Express Experiment Planning Agent.
Return only valid JSON.
Use backend validation, model catalog entries, and planner_context_snapshot evidence.
Do not bypass validation or invent unsupported fields.`)
}

func compactPlannerSystemPrompt() string {
	return strings.TrimSpace(`Return only valid JSON for the Experiment Planner.
Use planner_context_snapshot evidence, model catalog entries, and backend validation constraints.
Do not invent unsupported fields.`)
}

func replayContextV2Input(input agents.ExperimentPlannerInput, distilledMemoryFirst bool) agents.ExperimentPlannerInput {
	// Keep the decision-critical fields and trim the high-volume history for the V2-style replay
	// projection. This is only used in replay harness comparisons.
	input.PlanJobs = nil
	input.PlanMetrics = nil
	input.PlanEvaluations = nil
	input.PlanSummaries = trimTrainingSummaries(input.PlanSummaries, 6)
	input.PriorJobs = nil
	input.PriorEvaluations = nil
	input.PriorMemory = nil
	input.ExistingExperimentSignatures = trimStrings(input.ExistingExperimentSignatures, 12)
	input.PriorPlans = trimPlans(input.PriorPlans, 4)
	input.SuccessfulStrategyMemory = trimStrategyMemory(input.SuccessfulStrategyMemory, 4)
	input.FailedStrategyMemory = trimStrategyMemory(input.FailedStrategyMemory, 4)
	input.StrategyScorecards = trimStrategyScorecards(input.StrategyScorecards, 4)
	if distilledMemoryFirst {
		input.RetrievedMemory = distilledMemoryFirstResults(input)
	}
	return input
}

func distilledMemoryFirstResults(input agents.ExperimentPlannerInput) []memory.MemoryRetrievalResult {
	results := append([]memory.MemoryRetrievalResult(nil), input.RetrievedMemory...)
	if len(results) > 0 {
		return trimRetrievedMemory(results, 6)
	}
	out := []memory.MemoryRetrievalResult{}
	for _, memoryCard := range input.SuccessfulStrategyMemory {
		out = append(out, memory.MemoryRetrievalResult{
			SourceTable:     memory.SourceStrategyScorecard,
			SourceID:        memoryCard.MemoryID,
			ProjectID:       input.Project.ID,
			DatasetID:       input.Dataset.ID,
			Kind:            memory.KindPlanningOutcome,
			Score:           0.8,
			SemanticScore:   0.75,
			StructuredScore: 0.82,
			RetrievalReason: "distilled successful strategy lesson",
			SummaryCard: map[string]any{
				"outcome":      memoryCard.OutcomeStatus,
				"mechanism":    "distilled_strategy",
				"intervention": strings.Join(memoryCard.ProposedModels, ","),
				"lesson":       memoryCard.Lesson,
			},
			Metadata: map[string]any{
				"outcome": memoryCard.OutcomeStatus,
				"models":  memoryCard.ProposedModels,
			},
		})
	}
	for _, memoryCard := range input.FailedStrategyMemory {
		out = append(out, memory.MemoryRetrievalResult{
			SourceTable:     memory.SourceStrategyScorecard,
			SourceID:        memoryCard.MemoryID,
			ProjectID:       input.Project.ID,
			DatasetID:       input.Dataset.ID,
			Kind:            memory.KindPlanningFeedback,
			Score:           0.55,
			SemanticScore:   0.52,
			StructuredScore: 0.48,
			RetrievalReason: "distilled failed strategy lesson",
			SummaryCard: map[string]any{
				"outcome":   memoryCard.OutcomeStatus,
				"mechanism": "distilled_strategy",
				"lesson":    memoryCard.Lesson,
			},
			Metadata: map[string]any{
				"outcome": memoryCard.OutcomeStatus,
				"models":  memoryCard.ProposedModels,
			},
		})
	}
	return trimRetrievedMemory(out, 6)
}

func trimPlans(values []plans.ExperimentPlan, limit int) []plans.ExperimentPlan {
	if len(values) <= limit {
		return values
	}
	return append([]plans.ExperimentPlan(nil), values[:limit]...)
}

func trimTrainingSummaries(values []runs.TrainingRunSummary, limit int) []runs.TrainingRunSummary {
	if len(values) <= limit {
		return values
	}
	return append([]runs.TrainingRunSummary(nil), values[:limit]...)
}

func trimStrategyMemory(values []agents.PlannerStrategyMemory, limit int) []agents.PlannerStrategyMemory {
	if len(values) <= limit {
		return values
	}
	return append([]agents.PlannerStrategyMemory(nil), values[:limit]...)
}

func trimStrategyScorecards(values []agents.PlannerStrategyScorecard, limit int) []agents.PlannerStrategyScorecard {
	if len(values) <= limit {
		return values
	}
	return append([]agents.PlannerStrategyScorecard(nil), values[:limit]...)
}

func trimRetrievedMemory(values []memory.MemoryRetrievalResult, limit int) []memory.MemoryRetrievalResult {
	if len(values) <= limit {
		return values
	}
	return append([]memory.MemoryRetrievalResult(nil), values[:limit]...)
}

func trimStrings(values []string, limit int) []string {
	if len(values) <= limit {
		return values
	}
	return append([]string(nil), values[:limit]...)
}

func replayBestPlannerVariant(results []PlannerReplayVariantResult) PlannerReplayVariant {
	if len(results) == 0 {
		return ""
	}
	ordered := append([]PlannerReplayVariantResult(nil), results...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].BackendValidationPassed != ordered[j].BackendValidationPassed {
			return ordered[i].BackendValidationPassed && !ordered[j].BackendValidationPassed
		}
		if ordered[i].CandidateRankingScore == ordered[j].CandidateRankingScore {
			if ordered[i].MechanismDiversity == ordered[j].MechanismDiversity {
				return ordered[i].PromptBytes < ordered[j].PromptBytes
			}
			return ordered[i].MechanismDiversity > ordered[j].MechanismDiversity
		}
		return ordered[i].CandidateRankingScore > ordered[j].CandidateRankingScore
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

func plannerReplayLiveEnabled() bool {
	value := strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_REPLAY_LIVE_MINI")))
	if value == "" {
		value = strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_REPLAY_LIVE")))
	}
	switch value {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
