package agents

import (
	"math"
	"sort"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/policies"
)

const (
	rankerV2EvidenceWeight      = 0.80
	rankerV2BaseComponent       = 0.05
	rankerV2ImpactCap           = 0.10
	rankerV2ImpactSaturation    = 0.05
	rankerV2NoveltyCap          = 0.05
	rankerV2PriorRateWeight     = 0.10
	rankerV2PriorMeanWeight     = 0.10
	rankerV2PriorMeanSaturation = 0.05
	rankerV2OutcomeDisclosure   = "V2 is shadow-only. Outcome comparisons are valid only for candidates executed by v1 (including v1/v2 overlap); v2-only selections are counterfactual and have no observed outcome label."
)

// rankPlannerCandidateHypothesesV2 scores and selects into a shadow artifact.
// The policy inherits v1's structural rejections, removes v1 impact/novelty
// components from deterministic evidence, reintroduces those self-reports only
// through hard-capped components, and blends the result with smoothed priors.
// Priors fall back from task+mechanism+family to task+mechanism, task, global,
// then neutral. Family diversity is applied after base scoring. The function
// cannot return experiments and therefore cannot affect scheduling.
func rankPlannerCandidateHypothesesV2(
	input ExperimentPlannerInput,
	candidates []CandidateHypothesis,
	v1Rankings []CandidateRanking,
	maxExperiments int,
) ([]CandidateRanking, []CandidateSelectionRound, *RankerShadowComparison) {
	if maxExperiments < 1 {
		maxExperiments = 5
	}
	if maxExperiments > 5 {
		maxExperiments = 5
	}
	snapshot := calibration.RankerV2PriorSnapshot{}
	if input.RankerV2PriorSnapshot != nil {
		snapshot = *input.RankerV2PriorSnapshot
	}
	v2 := make([]CandidateRanking, 0, len(v1Rankings))
	for _, v1 := range v1Rankings {
		candidate := candidates[v1.CandidateIndex]
		ranking := CandidateRanking{
			RankerVersion:  ExperimentPlannerShadowRankerVersion,
			CandidateIndex: v1.CandidateIndex, Hypothesis: v1.Hypothesis, PlanningMode: v1.PlanningMode,
			Mechanism: v1.Mechanism, Intervention: v1.Intervention, ExpectedEffect: v1.ExpectedEffect,
			RetrievedMemoryHits: append([]CandidateRetrievedMemoryHit(nil), v1.RetrievedMemoryHits...),
			PromotionDecision:   v1.PromotionDecision, StopReason: v1.StopReason,
			Rejected: v1.Rejected, Disposition: v1.Disposition, Reasons: append([]string(nil), v1.Reasons...), ExperimentSignature: v1.ExperimentSignature,
			PolicyFindings:     append([]policies.Finding(nil), v1.PolicyFindings...),
			ValidationFindings: append([]PlannerValidationFieldFinding(nil), v1.ValidationFindings...),
			ScoreComponents:    map[string]float64{},
		}
		if ranking.Rejected {
			ranking.ScoreComponents["rejected_by_v1_structural_gate"] = 1
			v2 = append(v2, ranking)
			continue
		}
		expectedGainComponent := v1.ScoreComponents["expected_gain"]
		noveltyComponent := v1.ScoreComponents["novelty"]
		deterministicEvidence := clampCandidate(v1.BaseScore-expectedGainComponent-noveltyComponent, 0, 1)
		impact := normalizedCandidateImpact(candidate)
		impactComponent := clampCandidate(impact/rankerV2ImpactSaturation, 0, 1) * rankerV2ImpactCap
		noveltyComponentV2 := clampCandidate(candidate.NoveltyScore, 0, 1) * rankerV2NoveltyCap
		prior := calibration.SelectRankerV2Prior(snapshot, candidateTask(input), ranking.Mechanism, inferExperimentFamily(candidate.ExperimentConfig.Model))
		priorRateComponent := (prior.SmoothedImprovementRate - 0.5) * rankerV2PriorRateWeight
		priorMeanComponent := clampCandidate(prior.SmoothedMeanImprovement/rankerV2PriorMeanSaturation, -1, 1) * rankerV2PriorMeanWeight
		evidenceComponent := deterministicEvidence * rankerV2EvidenceWeight
		score := rankerV2BaseComponent + evidenceComponent + impactComponent + noveltyComponentV2 + priorRateComponent + priorMeanComponent
		ranking.ScoreComponents = map[string]float64{
			"base":                   rankerV2BaseComponent,
			"deterministic_evidence": roundCandidateScore(evidenceComponent),
			"self_reported_impact":   roundCandidateScore(impactComponent),
			"self_reported_novelty":  roundCandidateScore(noveltyComponentV2),
			"prior_improvement_rate": roundCandidateScore(priorRateComponent),
			"prior_mean_improvement": roundCandidateScore(priorMeanComponent),
		}
		ranking.EmpiricalPrior = &prior
		ranking.Score = roundCandidateScore(clampCandidate(score, 0, 1))
		ranking.BaseScore = ranking.Score
		v2 = append(v2, ranking)
	}

	selectedIndexes, trace := selectPlannerCandidateIndexes(v2, candidates, maxExperiments)
	byCandidate := make(map[int]int, len(v2))
	for index := range v2 {
		byCandidate[v2[index].CandidateIndex] = index
	}
	for order, candidateIndex := range selectedIndexes {
		index, ok := byCandidate[candidateIndex]
		if !ok || order >= len(trace) {
			continue
		}
		entry, ok := selectedCandidateTraceEntry(trace[order])
		if !ok {
			continue
		}
		score := entry.AdjustedScore
		experimentIndex := order
		v2[index].Selected = true
		if v2[index].Disposition != PlannerCandidateAcceptedAfterNormalization {
			v2[index].Disposition = PlannerCandidateAccepted
		}
		v2[index].SelectionScore = &score
		v2[index].SelectionOrder = &order
		v2[index].SelectedExperimentIndex = &experimentIndex
		v2[index].SelectionAdjustments = append([]CandidateSelectionAdjustment(nil), entry.SelectionAdjustments...)
	}
	for index := range v2 {
		if !v2[index].Rejected && !v2[index].Selected {
			v2[index].Disposition = PlannerCandidateUnselectedByRank
		}
	}
	comparison := compareRankerSelections(v1Rankings, v2)
	return v2, trace, &comparison
}

func normalizedCandidateImpact(candidate CandidateHypothesis) float64 {
	value := candidate.ExpectedMetricImpact
	if candidate.Forecast != nil && candidate.Forecast.MetricDirection == calibration.MetricDirectionLowerIsBetter {
		value = -value
	}
	return math.Max(0, value)
}

func candidateTask(input ExperimentPlannerInput) string {
	if task := input.ExecutionCapabilityCard.Task; task != "" {
		return task
	}
	return "unknown"
}

func compareRankerSelections(v1, v2 []CandidateRanking) RankerShadowComparison {
	v1Ordering := rankerOrdering(v1)
	v2Ordering := rankerOrdering(v2)
	v1Positions := rankerPositions(v1Ordering)
	v2Positions := rankerPositions(v2Ordering)
	changes := []RankerOrderingChange{}
	for _, candidateIndex := range v1Ordering {
		if v2Position, ok := v2Positions[candidateIndex]; ok && v1Positions[candidateIndex] != v2Position {
			changes = append(changes, RankerOrderingChange{CandidateIndex: candidateIndex, V1Position: v1Positions[candidateIndex], V2Position: v2Position})
		}
	}
	sort.Slice(changes, func(i, j int) bool { return changes[i].CandidateIndex < changes[j].CandidateIndex })
	v1Selection := rankerSelection(v1)
	v2Selection := rankerSelection(v2)
	v1Set, v2Set := intSet(v1Selection), intSet(v2Selection)
	overlap, union := 0, map[int]bool{}
	for value := range v1Set {
		union[value] = true
		if v2Set[value] {
			overlap++
		}
	}
	for value := range v2Set {
		union[value] = true
	}
	v2Only, v1Only := setDifference(v2Set, v1Set), setDifference(v1Set, v2Set)
	overlapRate := 1.0
	if len(union) > 0 {
		overlapRate = float64(overlap) / float64(len(union))
	}
	return RankerShadowComparison{
		PolicyVersion: ExperimentPlannerShadowRankerVersion, SchedulingRankerVersion: ExperimentPlannerRankerVersion,
		ShadowOnly: true, V1Ordering: v1Ordering, V2Ordering: v2Ordering,
		V1Selection: v1Selection, V2Selection: v2Selection, OrderingChanges: changes,
		SelectionOverlapCount: overlap, SelectionOverlapRate: roundCandidateScore(overlapRate),
		SelectionSetChanged: len(v2Only) > 0 || len(v1Only) > 0,
		V2OnlySelections:    v2Only, V1OnlySelections: v1Only, OutcomeDisclosure: rankerV2OutcomeDisclosure,
	}
}

func rankerOrdering(rankings []CandidateRanking) []int {
	eligible := []CandidateRanking{}
	for _, ranking := range rankings {
		if !ranking.Rejected {
			eligible = append(eligible, ranking)
		}
	}
	sort.Slice(eligible, func(i, j int) bool {
		if eligible[i].Score == eligible[j].Score {
			return eligible[i].CandidateIndex < eligible[j].CandidateIndex
		}
		return eligible[i].Score > eligible[j].Score
	})
	out := make([]int, len(eligible))
	for index, ranking := range eligible {
		out[index] = ranking.CandidateIndex
	}
	return out
}

func rankerSelection(rankings []CandidateRanking) []int {
	selected := []CandidateRanking{}
	for _, ranking := range rankings {
		if ranking.Selected && ranking.SelectionOrder != nil {
			selected = append(selected, ranking)
		}
	}
	sort.Slice(selected, func(i, j int) bool { return *selected[i].SelectionOrder < *selected[j].SelectionOrder })
	out := make([]int, len(selected))
	for index, ranking := range selected {
		out[index] = ranking.CandidateIndex
	}
	return out
}

func rankerPositions(order []int) map[int]int {
	out := make(map[int]int, len(order))
	for position, candidateIndex := range order {
		out[candidateIndex] = position
	}
	return out
}

func intSet(values []int) map[int]bool {
	out := make(map[int]bool, len(values))
	for _, value := range values {
		out[value] = true
	}
	return out
}

func setDifference(left, right map[int]bool) []int {
	out := []int{}
	for value := range left {
		if !right[value] {
			out = append(out, value)
		}
	}
	sort.Ints(out)
	return out
}
