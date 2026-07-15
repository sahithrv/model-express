package agents

import (
	"encoding/json"
	"reflect"
	"testing"

	"model-express/services/orchestrator/internal/plans"
)

func TestSelectPlannerCandidateIndexesAppliesDynamicFamilyDiversity(t *testing.T) {
	candidates, rankings := selectorFixture(
		[]string{"mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"},
		[]float64{0.90, 0.89, 0.88, 0.80},
		nil,
	)

	selected, trace := selectPlannerCandidateIndexes(rankings, candidates, 4)
	if want := []int{0, 1, 3, 2}; !reflect.DeepEqual(selected, want) {
		t.Fatalf("expected diversity penalty to change selection order to %v, got %v", want, selected)
	}
	adjusted := traceEntryForCandidate(t, trace[2], 2)
	if adjusted.AdjustedScore != 0.76 || !hasSelectionAdjustment(adjusted.SelectionAdjustments, "family_diversity") {
		t.Fatalf("expected third same-family candidate to be adjusted to 0.76, got %#v", adjusted)
	}
	alternative := traceEntryForCandidate(t, trace[2], 3)
	if alternative.AdjustedScore != 0.80 || len(alternative.SelectionAdjustments) != 0 || !alternative.Selected {
		t.Fatalf("expected alternative family to win the adjusted round, got %#v", alternative)
	}
}

func TestSelectPlannerCandidateIndexesDiversityIsNotHardQuota(t *testing.T) {
	candidates, rankings := selectorFixture(
		[]string{"mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"},
		[]float64{0.95, 0.94, 0.93, 0.70},
		nil,
	)

	selected, trace := selectPlannerCandidateIndexes(rankings, candidates, 3)
	if want := []int{0, 1, 2}; !reflect.DeepEqual(selected, want) {
		t.Fatalf("expected stronger same-family candidate to remain selected, want %v got %v", want, selected)
	}
	entry := traceEntryForCandidate(t, trace[2], 2)
	if entry.AdjustedScore != 0.81 || !entry.Selected {
		t.Fatalf("expected adjusted same-family score to remain highest, got %#v", entry)
	}
}

func TestSelectPlannerCandidateIndexesUsesOriginalIndexForAdjustedTie(t *testing.T) {
	candidates, rankings := selectorFixture(
		[]string{"mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"},
		[]float64{0.95, 0.94, 0.92, 0.80},
		nil,
	)

	selected, _ := selectPlannerCandidateIndexes(rankings, candidates, 3)
	if want := []int{0, 1, 2}; !reflect.DeepEqual(selected, want) {
		t.Fatalf("expected candidate index 2 to win the exact adjusted tie, want %v got %v", want, selected)
	}
}

func TestSelectPlannerCandidateIndexesExcludesRejectedCandidatesFromRounds(t *testing.T) {
	candidates, rankings := selectorFixture(
		[]string{"mobilenet_v1", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"},
		[]float64{0.99, 0.90, 0.89, 0.88, 0.80},
		map[int]bool{0: true},
	)

	selected, trace := selectPlannerCandidateIndexes(rankings, candidates, 3)
	if want := []int{1, 2, 4}; !reflect.DeepEqual(selected, want) {
		t.Fatalf("expected rejected candidate not to count toward family adjustments, want %v got %v", want, selected)
	}
	for _, round := range trace {
		if round.TotalCandidateCount > 4 {
			t.Fatalf("expected only eligible candidates in trace, got %#v", round)
		}
		for _, entry := range round.Candidates {
			if entry.CandidateIndex == 0 {
				t.Fatalf("rejected candidate appeared in selection trace: %#v", round)
			}
		}
	}
}

func TestSelectPlannerCandidateIndexesHonorsMaxAndEligibility(t *testing.T) {
	candidates, rankings := selectorFixture(
		[]string{"mobilenet_v2", "efficientnet_b0", "resnet18", "convnext_tiny", "swin_tiny"},
		[]float64{0.90, 0.85, 0.80, 0.99, 0.98},
		map[int]bool{3: true, 4: true},
	)
	for maxExperiments := 1; maxExperiments <= 5; maxExperiments++ {
		selected, trace := selectPlannerCandidateIndexes(rankings, candidates, maxExperiments)
		wantCount := minInt(maxExperiments, 3)
		if len(selected) != wantCount || len(trace) != wantCount {
			t.Fatalf("max=%d: expected %d eligible selections, got selected=%v trace=%d", maxExperiments, wantCount, selected, len(trace))
		}
	}

	allRejected := append([]CandidateRanking(nil), rankings...)
	for index := range allRejected {
		allRejected[index].Rejected = true
	}
	selected, trace := selectPlannerCandidateIndexes(allRejected, candidates, 5)
	if len(selected) != 0 || len(trace) != 0 {
		t.Fatalf("expected no selection when every candidate is rejected, got %v %#v", selected, trace)
	}
}

func TestSelectPlannerCandidateIndexesTraceIsBoundedAndDeterministic(t *testing.T) {
	models := []string{"mobilenet_v1", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0", "resnet18", "convnext_tiny", "swin_tiny"}
	scores := []float64{0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91}
	candidates, rankings := selectorFixture(models, scores, nil)

	selectedA, traceA := selectPlannerCandidateIndexes(rankings, candidates, 3)
	selectedB, traceB := selectPlannerCandidateIndexes(rankings, candidates, 3)
	if !reflect.DeepEqual(selectedA, selectedB) || !reflect.DeepEqual(traceA, traceB) {
		t.Fatalf("expected repeatable selection and trace, got %v/%v and %#v/%#v", selectedA, selectedB, traceA, traceB)
	}
	for index, round := range traceA {
		if len(round.Candidates) > plannerSelectionTraceEntryLimit {
			t.Fatalf("round %d exceeded trace limit: %#v", index, round)
		}
		if round.TotalCandidateCount != len(rankings)-index {
			t.Fatalf("round %d has wrong total candidate count: %#v", index, round)
		}
		if !round.Truncated {
			t.Fatalf("round %d should disclose truncation: %#v", index, round)
		}
		entry := traceEntryForCandidate(t, round, round.SelectedCandidateIndex)
		if !entry.Selected {
			t.Fatalf("round %d omitted its selected candidate: %#v", index, round)
		}
	}
}

func TestRankPlannerCandidateHypothesesIsImmutableAndAuditable(t *testing.T) {
	input := ExperimentPlannerInput{MaxExperiments: 3}
	candidates := []CandidateHypothesis{
		rankingCandidate("mobilenet_v2", 0.030),
		rankingCandidate("mobilenet_v3_small", 0.028),
		rankingCandidate("mobilenet_v3_large", 0.026),
		rankingCandidate("efficientnet_b0", 0.020),
	}
	inputBefore, _ := json.Marshal(input)
	candidatesBefore, _ := json.Marshal(candidates)

	rankingsA, selectedA, mechanismsA, traceA := rankPlannerCandidateHypotheses(input, candidates, 3)
	rankingsB, selectedB, mechanismsB, traceB := rankPlannerCandidateHypotheses(input, candidates, 3)
	resultA, _ := json.Marshal([]any{rankingsA, selectedA, mechanismsA, traceA})
	resultB, _ := json.Marshal([]any{rankingsB, selectedB, mechanismsB, traceB})
	if string(resultA) != string(resultB) {
		t.Fatalf("expected byte-equivalent repeated rankings, first=%s second=%s", resultA, resultB)
	}
	inputAfter, _ := json.Marshal(input)
	candidatesAfter, _ := json.Marshal(candidates)
	if string(inputBefore) != string(inputAfter) || string(candidatesBefore) != string(candidatesAfter) {
		t.Fatalf("ranking mutated caller-owned input: input %s/%s candidates %s/%s", inputBefore, inputAfter, candidatesBefore, candidatesAfter)
	}

	selectedOrders := map[int]bool{}
	selectedExperimentIndexes := map[int]bool{}
	for _, ranking := range rankingsA {
		if ranking.Score != ranking.BaseScore {
			t.Fatalf("score must remain a base-score compatibility alias, got %#v", ranking)
		}
		if !ranking.Selected {
			if ranking.SelectionScore != nil || ranking.SelectionOrder != nil || ranking.SelectedExperimentIndex != nil {
				t.Fatalf("unselected candidate must not expose a fixed selection score/order: %#v", ranking)
			}
			continue
		}
		if ranking.SelectionScore == nil || ranking.SelectionOrder == nil || ranking.SelectedExperimentIndex == nil {
			t.Fatalf("selected candidate missing selection audit fields: %#v", ranking)
		}
		adjustmentTotal := 0.0
		for _, adjustment := range ranking.SelectionAdjustments {
			adjustmentTotal += adjustment.Value
		}
		if want := roundCandidateScore(ranking.BaseScore + adjustmentTotal); *ranking.SelectionScore != want {
			t.Fatalf("base score plus adjustments must equal selection score: want %.3f got %#v", want, ranking)
		}
		selectedOrders[*ranking.SelectionOrder] = true
		selectedExperimentIndexes[*ranking.SelectedExperimentIndex] = true
	}
	for index := range selectedA {
		if !selectedOrders[index] || !selectedExperimentIndexes[index] {
			t.Fatalf("selection order and experiment indexes must be contiguous, orders=%v experiment_indexes=%v", selectedOrders, selectedExperimentIndexes)
		}
	}
}

func TestCandidateRankingJSONRemainsBackwardCompatible(t *testing.T) {
	oldPayload := []byte(`{"candidate_index":1,"hypothesis":"legacy","score":0.74,"selected":true,"rejected":false,"reasons":["legacy reason"],"score_components":{"base":0.45},"experiment_signature":"legacy-signature"}`)
	var ranking CandidateRanking
	if err := json.Unmarshal(oldPayload, &ranking); err != nil {
		t.Fatalf("unmarshal old candidate ranking: %v", err)
	}
	if ranking.Score != 0.74 || !ranking.Selected || ranking.SelectionScore != nil {
		t.Fatalf("old ranking fields were not preserved: %#v", ranking)
	}
	encoded, err := json.Marshal(ranking)
	if err != nil {
		t.Fatalf("marshal old candidate ranking: %v", err)
	}
	var generic map[string]any
	if err := json.Unmarshal(encoded, &generic); err != nil {
		t.Fatalf("re-unmarshal candidate ranking: %v", err)
	}
	if generic["score"] != 0.74 || generic["selected"] != true {
		t.Fatalf("legacy compatibility fields missing after round trip: %s", encoded)
	}
}

func selectorFixture(models []string, scores []float64, rejected map[int]bool) ([]CandidateHypothesis, []CandidateRanking) {
	candidates := make([]CandidateHypothesis, 0, len(models))
	rankings := make([]CandidateRanking, 0, len(models))
	for index, model := range models {
		candidates = append(candidates, CandidateHypothesis{ExperimentConfig: rankingCandidate(model, 0.02).ExperimentConfig})
		rankings = append(rankings, CandidateRanking{
			CandidateIndex: index,
			Score:          scores[index],
			BaseScore:      scores[index],
			Rejected:       rejected[index],
		})
	}
	return candidates, rankings
}

func rankingCandidate(model string, expectedImpact float64) CandidateHypothesis {
	return CandidateHypothesis{
		Hypothesis:           "Weighted loss should improve minority recall.",
		PlanningMode:         "class_imbalance_ablation",
		Mechanism:            "class_imbalance",
		Intervention:         "weighted_cross_entropy",
		ProposedChanges:      map[string]any{"class_balancing": "weighted_loss"},
		ExpectedEffect:       "Improve minority recall and macro-F1.",
		ExpectedMetricImpact: expectedImpact,
		ExpectedTradeoffs:    []string{"possible precision tradeoff"},
		Risk:                 "low",
		CostLevel:            "low",
		NoveltyScore:         0.60,
		EvidenceUsed:         []string{"class imbalance evidence"},
		ExperimentConfig: plans.PlannedExperiment{
			Template:       "transfer_learning",
			Model:          model,
			Mechanism:      "class_imbalance",
			Intervention:   "weighted_cross_entropy",
			ExpectedEffect: "Improve minority recall and macro-F1.",
			EvidenceUsed:   []string{"class imbalance evidence"},
			Epochs:         10,
			BatchSize:      16,
			LearningRate:   0.0003,
			ClassBalancing: "weighted_loss",
			Reason:         "Test weighted loss on a valid candidate.",
			Strategy:       "Class imbalance ablation.",
		},
	}
}

func traceEntryForCandidate(t *testing.T, round CandidateSelectionRound, candidateIndex int) CandidateSelectionTraceEntry {
	t.Helper()
	for _, entry := range round.Candidates {
		if entry.CandidateIndex == candidateIndex {
			return entry
		}
	}
	t.Fatalf("candidate %d missing from trace round %#v", candidateIndex, round)
	return CandidateSelectionTraceEntry{}
}

func hasSelectionAdjustment(adjustments []CandidateSelectionAdjustment, code string) bool {
	for _, adjustment := range adjustments {
		if adjustment.Code == code {
			return true
		}
	}
	return false
}
