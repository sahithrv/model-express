package evals

import (
	"path/filepath"
	"testing"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/plans"
)

func TestReplayPlateauBackboneLotteryRejectsArchitectureChallenge(t *testing.T) {
	fixture := loadPlateauFixture(t)
	input := ExperimentPlannerInputFromReplayFixture(fixture)
	recommendation := plateauAddRecommendation([]agents.CandidateHypothesis{
		plateauArchitectureCandidate("convnext_tiny", 0.02),
		plateauArchitectureCandidate("swin_tiny", 0.018),
	})

	scores := ScorePlannerRecommendation(input, recommendation, fixture.Expected)
	if scores.BackendValidationPassed {
		t.Fatalf("expected architecture-only plateau recommendation to fail replay scoring, got %#v", scores)
	}
	if scores.AvoidedBlockedMechanisms && scores.AvoidedArchitectureAfterPlateau {
		t.Fatalf("expected architecture_challenge to fail either forbidden-mechanism or plateau checks, got %#v", scores)
	}
}

func TestReplayPlateauBackboneLotteryAllowsClassImbalancePivot(t *testing.T) {
	fixture := loadPlateauFixture(t)
	input := ExperimentPlannerInputFromReplayFixture(fixture)
	recommendation := plateauAddRecommendation([]agents.CandidateHypothesis{
		plateauClassImbalanceCandidate(),
	})

	scores := ScorePlannerRecommendation(input, recommendation, fixture.Expected)
	if !scores.BackendValidationPassed {
		t.Fatalf("expected class imbalance pivot to pass replay scoring, got %#v", scores)
	}
	if !scores.CandidateRankingApplied {
		t.Fatalf("expected deterministic candidate ranking to be applied, got %#v", scores)
	}
}

func TestReplayPlateauBackboneLotteryAllowsSelectChampion(t *testing.T) {
	fixture := loadPlateauFixture(t)
	input := ExperimentPlannerInputFromReplayFixture(fixture)
	recommendation := agents.ExperimentPlanningRecommendation{
		Summary:       "Select the current champion instead of spending more runs.",
		DecisionType:  decisions.TypeSelectChampion,
		Rationale:     "Backbone exploration is low-yield and the current champion is the best validated result.",
		Confidence:    0.82,
		ChampionJobID: input.CurrentChampion.JobID,
		StopReason:    "Current champion is the best available result after a low-yield plateau.",
		EvidenceUsed:  []string{"22 completed runs", "recent macro-F1 uplift below useful delta"},
	}

	scores := ScorePlannerRecommendation(input, recommendation, fixture.Expected)
	if !scores.BackendValidationPassed {
		t.Fatalf("expected SELECT_CHAMPION to pass replay scoring, got %#v", scores)
	}
}

func TestReplayPlateauBackboneLotteryAllowsWait(t *testing.T) {
	fixture := loadPlateauFixture(t)
	input := ExperimentPlannerInputFromReplayFixture(fixture)
	recommendation := agents.ExperimentPlanningRecommendation{
		Summary:      "Wait for additional evaluation details before scheduling more work.",
		DecisionType: decisions.TypeWait,
		Rationale:    "The project is plateaued and should wait for per-class diagnostics before another experiment.",
		Confidence:   0.74,
		EvidenceUsed: []string{"plateau replay fixture", "missing per-class audit details"},
	}

	scores := ScorePlannerRecommendation(input, recommendation, fixture.Expected)
	if !scores.BackendValidationPassed {
		t.Fatalf("expected WAIT to pass replay scoring, got %#v", scores)
	}
}

func TestReplayFailsWhenCandidateRankingBypassed(t *testing.T) {
	fixture := loadPlateauFixture(t)
	input := ExperimentPlannerInputFromReplayFixture(fixture)
	recommendation := plateauAddRecommendation(nil)
	recommendation.ProposedExperiments = []plans.PlannedExperiment{
		{
			Template:       "transfer_learning",
			Model:          "mobilenet_v3_large",
			Mechanism:      "class_imbalance",
			Intervention:   "weighted_cross_entropy",
			EvidenceUsed:   []string{"class imbalance score"},
			ExpectedEffect: "Improve minority-class recall.",
			Epochs:         12,
			BatchSize:      16,
			LearningRate:   0.0003,
			ClassBalancing: "weighted_loss",
			Reason:         "Direct proposed_experiments should be treated as draft-only.",
		},
	}
	recommendation.ProposalMechanisms = []agents.PlannerProposalMechanism{
		{
			ExperimentIndex: 0,
			Mechanism:       "class_imbalance",
			Intervention:    "weighted_cross_entropy",
			EvidenceUsed:    []string{"class imbalance score"},
			ExpectedEffect:  "Improve minority-class recall.",
		},
	}

	scores := ScorePlannerRecommendation(input, recommendation, fixture.Expected)
	if scores.CandidateRankingApplied {
		t.Fatalf("expected candidate ranking bypass to be detected, got %#v", scores)
	}
	if scores.BackendValidationPassed {
		t.Fatalf("expected bypassed candidate ranking to fail replay scoring, got %#v", scores)
	}
}

func loadPlateauFixture(t *testing.T) PlannerReplayFixture {
	t.Helper()
	fixture, err := LoadPlannerReplayFixture(filepath.Join("testdata", "plateau_backbone_lottery.json"))
	if err != nil {
		t.Fatalf("load fixture: %v", err)
	}
	if fixture.Name != "plateau_backbone_lottery" {
		t.Fatalf("unexpected fixture name %q", fixture.Name)
	}
	return fixture
}

func plateauAddRecommendation(candidates []agents.CandidateHypothesis) agents.ExperimentPlanningRecommendation {
	return agents.ExperimentPlanningRecommendation{
		Summary:                       "Score a deterministic replay recommendation.",
		DecisionType:                  decisions.TypeAddExperiments,
		Rationale:                     "The backend should select only non-exhausted, evidence-backed candidates.",
		Confidence:                    0.81,
		PlanningMode:                  "class_imbalance_ablation",
		DeterministicDiagnosisUsed:    []string{"plateau_score=0.720", "class_imbalance_score=0.620"},
		EvidenceUsed:                  []string{"low-yield architecture plateau", "minority-class gap"},
		Hypothesis:                    "A non-architecture pivot can improve macro-F1 more efficiently than another backbone.",
		ExpectedFailureModes:          []string{"minority recall may remain weak"},
		DatasetPreprocessingRationale: "Keep preprocessing stable so the mechanism test isolates class imbalance.",
		ChangedVariables:              []string{"class_balancing", "sampling_strategy"},
		SuccessCriteria:               "Improve macro-F1 or minority recall by at least 0.01.",
		StopCondition:                 "Select champion if non-architecture pivots do not clear the useful delta.",
		DeploymentTradeoff:            "No meaningful deployment regression expected.",
		CandidateHypotheses:           candidates,
		WhyCanBeatChampion:            "It addresses minority-class evidence instead of repeating exhausted model-family exploration.",
		ExpectedDeltaVsChampion:       0.018,
		RejectedOptions: []agents.RejectedPlannerOption{
			{
				Option:      "another backbone sweep",
				Reason:      "architecture_challenge is exhausted by the replay trajectory",
				Evidence:    "22 runs improved macro-F1 by only 0.042",
				AppliesWhen: []string{"architecture_challenge", "plateau"},
			},
		},
	}
}

func plateauArchitectureCandidate(model string, expectedImpact float64) agents.CandidateHypothesis {
	return agents.CandidateHypothesis{
		Hypothesis:           "A new backbone might improve the plateau.",
		PlanningMode:         "champion_challenge",
		Mechanism:            "architecture_challenge",
		Intervention:         "backbone_family_challenge",
		ProposedChanges:      map[string]any{"model": model, "mechanism": "architecture_challenge"},
		ExpectedEffect:       "Improve macro-F1 through a different pretrained architecture.",
		ExpectedMetricImpact: expectedImpact,
		ExpectedTradeoffs:    []string{"higher runtime"},
		Risk:                 "medium",
		CostLevel:            "medium",
		NoveltyScore:         0.70,
		EvidenceUsed:         []string{"plateau_score=0.720"},
		ExperimentConfig: plans.PlannedExperiment{
			Template:       "transfer_learning",
			Model:          model,
			Mechanism:      "architecture_challenge",
			Intervention:   "backbone_family_challenge",
			EvidenceUsed:   []string{"plateau_score=0.720"},
			ExpectedEffect: "Improve macro-F1 through a different pretrained architecture.",
			Epochs:         14,
			BatchSize:      16,
			LearningRate:   0.0003,
			Reason:         "Challenge the current champion with another model family.",
			Strategy:       "Backbone-only challenger.",
		},
	}
}

func plateauClassImbalanceCandidate() agents.CandidateHypothesis {
	return agents.CandidateHypothesis{
		Hypothesis:              "Weighted loss should improve minority recall and macro-F1.",
		PlanningMode:            "class_imbalance_ablation",
		Mechanism:               "class_imbalance",
		Intervention:            "weighted_cross_entropy",
		ProposedChanges:         map[string]any{"class_balancing": "weighted_loss", "sampling_strategy": "class_balanced_sampler"},
		ExpectedEffect:          "Improve minority-class recall and macro-F1 without another backbone sweep.",
		ExpectedMetricImpact:    0.018,
		ExpectedTradeoffs:       []string{"possible majority-class precision drop"},
		Risk:                    "low",
		CostLevel:               "medium",
		NoveltyScore:            0.90,
		EvidenceUsed:            []string{"class_imbalance_score=0.620", "minority_class_failure_score=0.580"},
		SimilarFailureMemoryIDs: []string{"architecture_plateau"},
		ExperimentConfig: plans.PlannedExperiment{
			Template:       "transfer_learning",
			Model:          "mobilenet_v3_large",
			Mechanism:      "class_imbalance",
			Intervention:   "weighted_cross_entropy",
			EvidenceUsed:   []string{"class_imbalance_score=0.620", "minority_class_failure_score=0.580"},
			ExpectedEffect: "Improve minority-class recall and macro-F1 without another backbone sweep.",
			Epochs:         12,
			BatchSize:      16,
			LearningRate:   0.0003,
			ClassBalancing: "weighted_loss",
			ClassBalancingConfig: map[string]any{
				"weighting": "inverse_frequency",
			},
			SamplingStrategy: "class_balanced_sampler",
			Reason:           "Target minority-class and macro-F1 gaps with weighted loss.",
			Strategy:         "Class imbalance intervention.",
		},
	}
}
