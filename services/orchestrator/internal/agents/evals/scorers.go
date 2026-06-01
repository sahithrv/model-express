package evals

import (
	"encoding/json"
	"math"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/plans"
)

type PlannerReplayScores struct {
	SchemaValid                     bool           `json:"schema_valid"`
	BackendValidationPassed         bool           `json:"backend_validation_passed"`
	CandidateRankingApplied         bool           `json:"candidate_ranking_applied"`
	AvoidedBlockedMechanisms        bool           `json:"avoided_blocked_mechanisms"`
	AvoidedDuplicateSignatures      bool           `json:"avoided_duplicate_signatures"`
	AvoidedArchitectureAfterPlateau bool           `json:"avoided_architecture_after_plateau"`
	ExpectedValueAboveFloor         bool           `json:"expected_value_above_floor"`
	EvidencePresent                 bool           `json:"evidence_present"`
	SelectedExperimentCountOK       bool           `json:"selected_experiment_count_ok"`
	RetrievedCardCount              int            `json:"retrieved_card_count,omitempty"`
	RetrievalPromptBytes            int            `json:"retrieval_prompt_bytes,omitempty"`
	SelectedCandidateMemoryScore    float64        `json:"selected_candidate_memory_score,omitempty"`
	RejectedCandidateMemoryPenalty  float64        `json:"rejected_candidate_memory_penalty,omitempty"`
	RetrievalHitSourceMix           map[string]int `json:"retrieval_hit_source_mix,omitempty"`
}

func ScorePlannerRecommendation(input agents.ExperimentPlannerInput, recommendation agents.ExperimentPlanningRecommendation, expected PlannerReplayExpected) PlannerReplayScores {
	finalized := recommendation
	decision := normalizedDecision(recommendation.DecisionType)
	maxExperiments := expected.MaxSelectedExperiments
	if maxExperiments <= 0 {
		maxExperiments = input.MaxExperiments
	}
	if maxExperiments <= 0 {
		maxExperiments = 5
	}
	input.MaxExperiments = maxExperiments

	rankingApplied := true
	finalizeOK := true
	if decision == decisions.TypeAddExperiments {
		rankingApplied = len(recommendation.CandidateHypotheses) > 0
		if rankingApplied {
			var err error
			finalized, err = agents.FinalizePlannerRecommendation(input, recommendation)
			finalizeOK = err == nil
			rankingApplied = len(finalized.CandidateRankings) == len(recommendation.CandidateHypotheses)
		}
	}

	selectedMechanisms := selectedReplayMechanisms(finalized)
	schemaValid := replaySchemaValid(recommendation)
	decisionAllowed := replayContainsNormalized(expected.AllowedDecisions, decision)
	if len(expected.AllowedDecisions) == 0 {
		decisionAllowed = true
	}
	avoidedBlocked := replayAvoidedMechanisms(selectedMechanisms, expected.ForbiddenMechanisms)
	if decision == decisions.TypeAddExperiments && len(expected.AllowedAddExperimentMechanisms) > 0 {
		avoidedBlocked = avoidedBlocked && replayMechanismsAllowed(selectedMechanisms, expected.AllowedAddExperimentMechanisms)
	}
	duplicatesAvoided := replayAvoidedDuplicateSignatures(finalized.ProposedExperiments)
	architectureAvoided := replayAvoidedArchitectureAfterPlateau(input, selectedMechanisms, expected)
	expectedValueOK := replayExpectedValueAboveFloor(input, finalized)
	evidencePresent := replayEvidencePresent(decision, finalized)
	countOK := decision != decisions.TypeAddExperiments || len(finalized.ProposedExperiments) <= maxExperiments
	addHasSelection := decision != decisions.TypeAddExperiments || len(finalized.ProposedExperiments) > 0
	retrievalMetrics := replayRetrievalMetrics(input, finalized)

	backendPassed := schemaValid &&
		decisionAllowed &&
		finalizeOK &&
		rankingApplied &&
		avoidedBlocked &&
		duplicatesAvoided &&
		architectureAvoided &&
		expectedValueOK &&
		evidencePresent &&
		countOK &&
		addHasSelection

	return PlannerReplayScores{
		SchemaValid:                     schemaValid,
		BackendValidationPassed:         backendPassed,
		CandidateRankingApplied:         rankingApplied,
		AvoidedBlockedMechanisms:        avoidedBlocked,
		AvoidedDuplicateSignatures:      duplicatesAvoided,
		AvoidedArchitectureAfterPlateau: architectureAvoided,
		ExpectedValueAboveFloor:         expectedValueOK,
		EvidencePresent:                 evidencePresent,
		SelectedExperimentCountOK:       countOK,
		RetrievedCardCount:              retrievalMetrics.RetrievedCardCount,
		RetrievalPromptBytes:            retrievalMetrics.RetrievalPromptBytes,
		SelectedCandidateMemoryScore:    retrievalMetrics.SelectedCandidateMemoryScore,
		RejectedCandidateMemoryPenalty:  retrievalMetrics.RejectedCandidateMemoryPenalty,
		RetrievalHitSourceMix:           retrievalMetrics.RetrievalHitSourceMix,
	}
}

type plannerReplayRetrievalMetrics struct {
	RetrievedCardCount             int
	RetrievalPromptBytes           int
	SelectedCandidateMemoryScore   float64
	RejectedCandidateMemoryPenalty float64
	RetrievalHitSourceMix          map[string]int
}

func replayRetrievalMetrics(input agents.ExperimentPlannerInput, recommendation agents.ExperimentPlanningRecommendation) plannerReplayRetrievalMetrics {
	metrics := plannerReplayRetrievalMetrics{
		RetrievedCardCount: len(input.RetrievedMemory),
	}
	if len(input.RetrievedMemory) > 0 {
		metrics.RetrievalPromptBytes = replayApproximateJSONBytes(input.RetrievedMemory)
	}

	sourceMix := map[string]int{}
	for _, ranking := range recommendation.CandidateRankings {
		retrievedScore := ranking.ScoreComponents["retrieved_memory"]
		if ranking.Selected {
			metrics.SelectedCandidateMemoryScore += retrievedScore
		}
		if ranking.Rejected && retrievedScore < 0 {
			metrics.RejectedCandidateMemoryPenalty += retrievedScore
		}
		for _, hit := range ranking.RetrievedMemoryHits {
			source := strings.TrimSpace(hit.SourceTable)
			if source == "" {
				source = strings.TrimSpace(hit.Kind)
			}
			if source != "" {
				sourceMix[source]++
			}
		}
	}
	if len(sourceMix) > 0 {
		metrics.RetrievalHitSourceMix = sourceMix
	}
	metrics.SelectedCandidateMemoryScore = replayRoundScore(metrics.SelectedCandidateMemoryScore)
	metrics.RejectedCandidateMemoryPenalty = replayRoundScore(metrics.RejectedCandidateMemoryPenalty)
	return metrics
}

func replaySchemaValid(recommendation agents.ExperimentPlanningRecommendation) bool {
	if strings.TrimSpace(recommendation.Summary) == "" ||
		strings.TrimSpace(recommendation.Rationale) == "" ||
		recommendation.Confidence < 0 ||
		recommendation.Confidence > 1 {
		return false
	}
	switch normalizedDecision(recommendation.DecisionType) {
	case decisions.TypeAddExperiments:
		return strings.TrimSpace(recommendation.PlanningMode) != "" &&
			len(recommendation.CandidateHypotheses) > 0 &&
			len(nonEmptyReplayStrings(recommendation.DeterministicDiagnosisUsed)) > 0 &&
			len(nonEmptyReplayStrings(recommendation.EvidenceUsed)) > 0 &&
			strings.TrimSpace(recommendation.Hypothesis) != "" &&
			strings.TrimSpace(recommendation.SuccessCriteria) != "" &&
			strings.TrimSpace(recommendation.StopCondition) != "" &&
			strings.TrimSpace(recommendation.WhyCanBeatChampion) != ""
	case decisions.TypeSelectChampion:
		return strings.TrimSpace(recommendation.ChampionJobID) != "" && strings.TrimSpace(recommendation.StopReason) != ""
	case decisions.TypeStopProject:
		return strings.TrimSpace(recommendation.StopReason) != ""
	case decisions.TypeWait:
		return strings.TrimSpace(recommendation.Rationale) != ""
	default:
		return false
	}
}

func selectedReplayMechanisms(recommendation agents.ExperimentPlanningRecommendation) []string {
	out := []string{}
	if len(recommendation.ProposalMechanisms) > 0 {
		for _, mechanism := range recommendation.ProposalMechanisms {
			out = append(out, normalizeReplayValue(mechanism.Mechanism))
		}
		return nonEmptyReplayStrings(out)
	}
	for _, experiment := range recommendation.ProposedExperiments {
		out = append(out, normalizeReplayValue(experiment.Mechanism))
	}
	for _, ranking := range recommendation.CandidateRankings {
		if ranking.Selected {
			out = append(out, normalizeReplayValue(ranking.Mechanism))
		}
	}
	return nonEmptyReplayStrings(out)
}

func replayAvoidedMechanisms(selected []string, forbidden []string) bool {
	forbiddenSet := replayStringSet(forbidden)
	for _, mechanism := range selected {
		if forbiddenSet[normalizeReplayValue(mechanism)] {
			return false
		}
	}
	return true
}

func replayMechanismsAllowed(selected []string, allowed []string) bool {
	allowedSet := replayStringSet(allowed)
	for _, mechanism := range selected {
		if !allowedSet[normalizeReplayValue(mechanism)] {
			return false
		}
	}
	return len(selected) > 0
}

func replayAvoidedDuplicateSignatures(experiments []plans.PlannedExperiment) bool {
	seen := map[string]bool{}
	for _, experiment := range experiments {
		signature := replayExperimentSignature(experiment)
		if seen[signature] {
			return false
		}
		seen[signature] = true
	}
	return true
}

func replayAvoidedArchitectureAfterPlateau(input agents.ExperimentPlannerInput, selected []string, expected PlannerReplayExpected) bool {
	plateauPressure := input.DeterministicDiagnosis.PlateauScore >= 0.60 ||
		input.DeterministicDiagnosis.ImprovementStagnationScore >= 0.60 ||
		input.NoImprovementRounds >= 2 ||
		replayContainsNormalized(expected.ForbiddenMechanisms, "architecture_challenge")
	if !plateauPressure {
		return true
	}
	for _, mechanism := range selected {
		if normalizeReplayValue(mechanism) == "architecture_challenge" {
			return false
		}
	}
	return true
}

func replayExpectedValueAboveFloor(input agents.ExperimentPlannerInput, recommendation agents.ExperimentPlanningRecommendation) bool {
	if normalizedDecision(recommendation.DecisionType) != decisions.TypeAddExperiments {
		return true
	}
	floor := math.Max(input.MinimumMeaningfulImprovement, 0.010)
	candidatesByIndex := map[int]agents.CandidateHypothesis{}
	for index, candidate := range recommendation.CandidateHypotheses {
		candidatesByIndex[index] = candidate
	}
	selectedCount := 0
	for _, ranking := range recommendation.CandidateRankings {
		if !ranking.Selected {
			continue
		}
		selectedCount++
		candidate := candidatesByIndex[ranking.CandidateIndex]
		if normalizeReplayValue(candidate.Mechanism) == "stop_select_champion" {
			continue
		}
		if candidate.ExpectedMetricImpact < floor {
			return false
		}
	}
	return selectedCount > 0
}

func replayEvidencePresent(decision string, recommendation agents.ExperimentPlanningRecommendation) bool {
	switch decision {
	case decisions.TypeAddExperiments:
		if len(nonEmptyReplayStrings(recommendation.EvidenceUsed)) == 0 {
			return false
		}
		for _, mechanism := range recommendation.ProposalMechanisms {
			if strings.TrimSpace(mechanism.Intervention) == "" ||
				len(nonEmptyReplayStrings(mechanism.EvidenceUsed)) == 0 ||
				strings.TrimSpace(mechanism.ExpectedEffect) == "" {
				return false
			}
		}
		return len(recommendation.ProposalMechanisms) > 0
	case decisions.TypeSelectChampion, decisions.TypeStopProject:
		return strings.TrimSpace(recommendation.StopReason) != "" || len(nonEmptyReplayStrings(recommendation.EvidenceUsed)) > 0
	case decisions.TypeWait:
		return strings.TrimSpace(recommendation.Rationale) != "" || len(nonEmptyReplayStrings(recommendation.EvidenceUsed)) > 0
	default:
		return false
	}
}

func replayExperimentSignature(experiment plans.PlannedExperiment) string {
	blob, _ := json.Marshal(struct {
		Template           string         `json:"template"`
		Model              string         `json:"model"`
		Epochs             int            `json:"epochs"`
		BatchSize          int            `json:"batch_size"`
		LearningRate       float64        `json:"learning_rate"`
		ImageSize          int            `json:"image_size"`
		ClassBalancing     string         `json:"class_balancing"`
		ClassBalancingConf map[string]any `json:"class_balancing_config"`
		SamplingStrategy   string         `json:"sampling_strategy"`
		AugmentationPolicy string         `json:"augmentation_policy"`
		ResolutionStrategy string         `json:"resolution_strategy"`
	}{
		Template:           normalizeReplayValue(experiment.Template),
		Model:              normalizeReplayValue(experiment.Model),
		Epochs:             experiment.Epochs,
		BatchSize:          experiment.BatchSize,
		LearningRate:       experiment.LearningRate,
		ImageSize:          experiment.ImageSize,
		ClassBalancing:     normalizeReplayValue(experiment.ClassBalancing),
		ClassBalancingConf: experiment.ClassBalancingConfig,
		SamplingStrategy:   normalizeReplayValue(experiment.SamplingStrategy),
		AugmentationPolicy: normalizeReplayValue(experiment.AugmentationPolicy),
		ResolutionStrategy: normalizeReplayValue(experiment.ResolutionStrategy),
	})
	return string(blob)
}

func normalizedDecision(value string) string {
	return strings.ToUpper(strings.TrimSpace(value))
}

func replayContainsNormalized(values []string, target string) bool {
	target = normalizeReplayValue(target)
	for _, value := range values {
		if normalizeReplayValue(value) == target {
			return true
		}
	}
	return false
}

func replayStringSet(values []string) map[string]bool {
	out := map[string]bool{}
	for _, value := range values {
		normalized := normalizeReplayValue(value)
		if normalized != "" {
			out[normalized] = true
		}
	}
	return out
}

func normalizeReplayValue(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func nonEmptyReplayStrings(values []string) []string {
	out := []string{}
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return out
}

func replayApproximateJSONBytes(value any) int {
	blob, err := json.Marshal(value)
	if err != nil {
		return 0
	}
	return len(blob)
}

func replayRoundScore(value float64) float64 {
	return math.Round(value*1000) / 1000
}
