package agents

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/plans"
)

func TestRankerV2ShadowCannotChangeV1Scheduling(t *testing.T) {
	input := ExperimentPlannerInput{
		SourcePlan:              plannerSourcePlanForRankerV2(),
		ExecutionCapabilityCard: execution.PlannerCapabilityCard{Task: "image_classification"},
		MaxExperiments:          1,
	}
	input.RankerV2PriorSnapshot = rankerV2TestPriorSnapshot()
	first := rankingCandidate("resnet18", 0.04)
	second := rankingCandidate("efficientnet_b0", 0.02)
	recommendation := ExperimentPlanningRecommendation{
		DecisionType:        decisions.TypeAddExperiments,
		CandidateHypotheses: []CandidateHypothesis{first, second},
	}

	finalized, err := FinalizePlannerRecommendation(input, recommendation)
	if err != nil {
		t.Fatal(err)
	}
	if len(finalized.ProposedExperiments) != 1 || len(finalized.CandidateRankingsV2) != 2 || finalized.RankerShadowComparison == nil {
		t.Fatalf("finalized shadow artifact=%#v", finalized)
	}
	v1Selection := rankerSelection(finalized.CandidateRankings)
	v2Selection := rankerSelection(finalized.CandidateRankingsV2)
	if !reflect.DeepEqual(v1Selection, []int{0}) || !reflect.DeepEqual(v2Selection, []int{1}) {
		t.Fatalf("expected prior to change only shadow selection: v1=%v v2=%v", v1Selection, v2Selection)
	}
	if finalized.ProposedExperiments[0].Model != first.ExperimentConfig.Model {
		t.Fatalf("v2 changed scheduled experiment: proposed=%#v", finalized.ProposedExperiments)
	}
	comparison := finalized.RankerShadowComparison
	if comparison.SchedulingRankerVersion != ExperimentPlannerRankerVersion || !comparison.ShadowOnly || !comparison.SelectionSetChanged ||
		!reflect.DeepEqual(comparison.V2OnlySelections, []int{1}) || !strings.Contains(comparison.OutcomeDisclosure, "counterfactual") {
		t.Fatalf("comparison=%#v", comparison)
	}
	if finalized.CandidateRankings[0].RankerVersion != ExperimentPlannerRankerVersion || finalized.CandidateRankingsV2[0].RankerVersion != ExperimentPlannerShadowRankerVersion {
		t.Fatalf("ranker versions v1=%#v v2=%#v", finalized.CandidateRankings, finalized.CandidateRankingsV2)
	}
}

func TestGuardedRankerV2TreatmentSchedulesOnlyItsAssignedPolicy(t *testing.T) {
	policy := calibration.DefaultPlannerRolloutPolicy()
	policy.Enabled = true
	policy.State = calibration.RolloutStateActive
	policy.StagePercent = 100
	assignment, err := calibration.AssignPlannerRollout(policy, "project-ranker-v2-treatment")
	if err != nil {
		t.Fatal(err)
	}
	input := ExperimentPlannerInput{
		SourcePlan:              plannerSourcePlanForRankerV2(),
		ExecutionCapabilityCard: execution.PlannerCapabilityCard{Task: "image_classification"},
		MaxExperiments:          1,
		RolloutAssignment:       &assignment,
		RankerV2PriorSnapshot:   rankerV2TestPriorSnapshot(),
	}
	first := rankingCandidate("resnet18", 0.04)
	second := rankingCandidate("efficientnet_b0", 0.02)
	finalized, err := FinalizePlannerRecommendation(input, ExperimentPlanningRecommendation{
		DecisionType: decisions.TypeAddExperiments, CandidateHypotheses: []CandidateHypothesis{first, second},
	})
	if err != nil {
		t.Fatal(err)
	}
	if PlannerSchedulingRankerVersion(input) != ExperimentPlannerRankerV2Version || finalized.ProposedExperiments[0].Model != second.ExperimentConfig.Model {
		t.Fatalf("v2 treatment did not control its isolated ranker dimension: %#v", finalized)
	}
	if selected := rankerSelection(finalized.CandidateRankingsV1); !reflect.DeepEqual(selected, []int{0}) {
		t.Fatalf("v1 comparison was not preserved: %v", selected)
	}
	if selected := rankerSelection(finalized.CandidateRankings); !reflect.DeepEqual(selected, []int{1}) {
		t.Fatalf("active v2 provenance does not match scheduling: %v", selected)
	}
	if finalized.RankerShadowComparison == nil || finalized.RankerShadowComparison.ShadowOnly || !strings.Contains(finalized.RankerShadowComparison.OutcomeDisclosure, "unexecuted counterfactuals") {
		t.Fatalf("active comparison misstates outcome authority: %#v", finalized.RankerShadowComparison)
	}
}

func TestRankerV2CapsSelfReportedImpactAndNoveltyAndIsDeterministic(t *testing.T) {
	candidates := []CandidateHypothesis{
		rankingCandidate("resnet18", 0.05),
		rankingCandidate("efficientnet_b0", 5.0),
	}
	candidates[0].NoveltyScore = 1
	candidates[1].NoveltyScore = 100
	v1 := []CandidateRanking{
		{CandidateIndex: 0, BaseScore: 0.90, Score: 0.90, ScoreComponents: map[string]float64{"expected_gain": 0.30, "novelty": 0.16}},
		{CandidateIndex: 1, BaseScore: 0.90, Score: 0.90, ScoreComponents: map[string]float64{"expected_gain": 0.30, "novelty": 0.16}},
	}
	firstRankings, firstTrace, firstComparison := rankPlannerCandidateHypothesesV2(ExperimentPlannerInput{}, candidates, v1, 1)
	secondRankings, secondTrace, secondComparison := rankPlannerCandidateHypothesesV2(ExperimentPlannerInput{}, candidates, v1, 1)
	if !reflect.DeepEqual(firstRankings, secondRankings) || !reflect.DeepEqual(firstTrace, secondTrace) || !reflect.DeepEqual(firstComparison, secondComparison) {
		t.Fatalf("v2 ranker is not deterministic")
	}
	for _, ranking := range firstRankings {
		if ranking.ScoreComponents["self_reported_impact"] != rankerV2ImpactCap || ranking.ScoreComponents["self_reported_novelty"] != rankerV2NoveltyCap {
			t.Fatalf("self-reported components exceeded caps: %#v", ranking.ScoreComponents)
		}
		if ranking.EmpiricalPrior == nil || ranking.EmpiricalPrior.SourceLevel != "neutral" {
			t.Fatalf("empty cohorts did not use documented neutral fallback: %#v", ranking.EmpiricalPrior)
		}
	}
	if firstRankings[0].Score != firstRankings[1].Score {
		t.Fatalf("uncapped outlier changed score: %#v", firstRankings)
	}
}

func TestRankerV2AppliesDynamicFamilyDiversityAfterBaseScoring(t *testing.T) {
	candidates, v1 := selectorFixture(
		[]string{"mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"},
		[]float64{0.90, 0.89, 0.88, 0.80}, nil,
	)
	for index := range candidates {
		candidates[index].ExpectedMetricImpact = 0
		candidates[index].NoveltyScore = 0
		v1[index].ScoreComponents = map[string]float64{}
	}
	v2, trace, _ := rankPlannerCandidateHypothesesV2(ExperimentPlannerInput{}, candidates, v1, 3)
	if selected := rankerSelection(v2); !reflect.DeepEqual(selected, []int{0, 1, 3}) {
		t.Fatalf("v2 diversity selection=%v rankings=%#v", selected, v2)
	}
	entry := traceEntryForCandidate(t, trace[2], 2)
	if !hasSelectionAdjustment(entry.SelectionAdjustments, "family_diversity") {
		t.Fatalf("v2 did not apply diversity after base scoring: %#v", trace[2])
	}
}

func plannerSourcePlanForRankerV2() plans.ExperimentPlan {
	return plans.ExperimentPlan{TargetMetric: "macro_f1"}
}

func rankerV2TestPriorSnapshot() *calibration.RankerV2PriorSnapshot {
	start := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	evaluationStart := start.Add(90 * 24 * time.Hour)
	return &calibration.RankerV2PriorSnapshot{
		SnapshotVersion:  calibration.RankerV2PriorSnapshotVersionV1,
		TrainingWindow:   calibration.TimeWindow{Start: start, End: evaluationStart},
		EvaluationWindow: calibration.TimeWindow{Start: evaluationStart, End: evaluationStart.Add(30 * 24 * time.Hour)},
		MinSampleSize:    5, MeaningfulImprovement: 0.01,
		PseudoCount: calibration.RankerV2PriorPseudoCount, ImprovementWinsorLimit: calibration.RankerV2PriorWinsorLimit,
		SourceStatus: "available",
		Records: []calibration.RankerV2PriorRecord{
			{
				Key:        calibration.CohortKey{Grouping: "task_mechanism_model_family", Task: "image_classification", Mechanism: "class_imbalance", ModelFamily: "resnet"},
				SampleSize: 5, SmoothedMeanImprovement: -0.05, SmoothedImprovementRate: 0,
			},
			{
				Key:        calibration.CohortKey{Grouping: "task_mechanism_model_family", Task: "image_classification", Mechanism: "class_imbalance", ModelFamily: "efficientnet"},
				SampleSize: 5, SmoothedMeanImprovement: 0.05, SmoothedImprovementRate: 1,
			},
		},
	}
}
