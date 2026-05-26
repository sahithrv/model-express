package agents

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

func TestExperimentPlannerAgentValidatesAddExperiments(t *testing.T) {
	agent := NewExperimentPlannerAgent(fakeJSONGenerator{
		response: `{
			"summary": "EfficientNet is promising, but the next round should vary preprocessing and regularization.",
			"decision_type": "ADD_EXPERIMENTS",
			"rationale": "The source plan completed below target with stable but underpowered validation quality.",
			"confidence": 0.84,
			"planning_mode": "preprocessing_ablation",
			"deterministic_diagnosis_used": ["class_imbalance_score=0.55", "overfitting_score=0.41"],
			"evidence_used": ["dataset imbalance ratio is high", "validation/train loss gap is visible"],
			"hypothesis": "Higher resolution plus stronger augmentation and regularization will improve macro-F1 without making inference too slow.",
			"expected_failure_modes": ["larger input may increase latency"],
			"dataset_preprocessing_rationale": "The dataset is small and imbalanced, so weighted loss and moderate augmentation should help generalization.",
			"changed_variables": ["model_family", "image_size", "augmentation", "class_balancing"],
			"success_criteria": "Beat the current champion macro-F1 by at least 0.01 while staying within reasonable runtime.",
			"stop_condition": "Select the current champion if this ablation does not improve macro-F1 or minority recall.",
			"deployment_tradeoff": "EfficientNet-B1 may cost more than MobileNet, so it must show clear quality improvement to be worth live use.",
			"proposed_experiments": [
				{
					"template": "efficientnet_transfer",
					"model": "efficientnet_b1",
					"epochs": 12,
					"batch_size": 16,
					"learning_rate": 0.0002,
					"reason": "Try a larger EfficientNet with stronger augmentation.",
					"image_size": 256,
					"resolution_strategy": "high_resolution_ablation",
					"preprocessing": {
						"resize_strategy": "preserve_aspect_pad",
						"normalization": "imagenet",
						"crop_strategy": "center_crop",
						"bbox_mode": "ignore",
						"use_dataset_normalization": false
					},
					"optimizer": "adamw",
					"scheduler": "cosine",
					"weight_decay": 0.01,
					"augmentation": {"horizontal_flip": true, "color_jitter": true},
					"augmentation_policy": "moderate",
					"class_balancing": "weighted_loss",
					"sampling_strategy": "weighted_random_sampler",
					"early_stopping_patience": 3,
					"strategy": "promising family exploitation"
				}
			],
			"champion_job_id": "",
			"why_can_beat_champion": "It tests a stronger EfficientNet with higher resolution and regularization instead of merely extending the same baseline.",
			"expected_delta_vs_champion": 0.02,
			"stop_reason": "",
			"risks": ["higher runtime"],
			"expected_tradeoffs": ["better quality for more cost"],
			"novelty_notes": ["larger model and image size"],
			"rejected_options": [{"option": "more MobileNet epochs", "reason": "does not address imbalance", "evidence": "class imbalance signal", "applies_when": ["class_imbalance"]}],
			"tags": ["efficientnet", "augmentation"]
		}`,
	}, "test-model")

	trace, err := agent.PlanWithTrace(context.Background(), testExperimentPlannerInput())
	if err != nil {
		t.Fatalf("plan with trace: %v", err)
	}
	if trace.ValidationStatus != "valid" {
		t.Fatalf("expected valid trace, got %s", trace.ValidationStatus)
	}
	if trace.Recommendation.DecisionType != decisions.TypeAddExperiments {
		t.Fatalf("expected ADD_EXPERIMENTS, got %s", trace.Recommendation.DecisionType)
	}
	if trace.Recommendation.ProposedExperiments[0].ImageSize != 256 {
		t.Fatalf("expected image size to survive decode")
	}
	experiment := trace.Recommendation.ProposedExperiments[0]
	if experiment.ResolutionStrategy != "high_resolution_ablation" {
		t.Fatalf("expected resolution strategy to survive decode, got %q", experiment.ResolutionStrategy)
	}
	if experiment.Preprocessing == nil || experiment.Preprocessing.ResizeStrategy != "preserve_aspect_pad" {
		t.Fatalf("expected preprocessing config to survive decode, got %#v", experiment.Preprocessing)
	}
	if experiment.AugmentationPolicy != "moderate" {
		t.Fatalf("expected augmentation policy to survive decode, got %q", experiment.AugmentationPolicy)
	}
	if experiment.SamplingStrategy != "weighted_random_sampler" {
		t.Fatalf("expected sampling strategy to survive decode, got %q", experiment.SamplingStrategy)
	}
	if trace.Recommendation.PlanningMode != "preprocessing_ablation" {
		t.Fatalf("expected planning mode to survive decode")
	}
}

func TestExperimentPlannerPromptDocumentsPreprocessingContractAndVisualEvidence(t *testing.T) {
	request := experimentPlannerJSONRequest("test-model", []byte(`{"planner_context_snapshot":{"visual_evidence":{"enabled":true}}}`))
	if len(request.Messages) != 2 {
		t.Fatalf("expected system and user messages")
	}
	prompt := request.Messages[0].Content + "\n" + request.Messages[1].Content
	for _, expected := range []string{
		"resolution_strategy",
		"preprocessing.resize_strategy values",
		"augmentation_policy values",
		"class_balancing values",
		"sampling_strategy values",
		"focal_loss",
		"Return only valid JSON",
		"planner_context_snapshot",
		"visual_evidence, when present, only as backend-curated evidence",
		"Cite exemplar caps, warnings, or audit details",
		"Backend validation remains the gate",
		"planner_validation_feedback",
		"choose arbitrary files",
		"mutate datasets",
		"run export or inference",
		"create workers",
		"create jobs",
		"bypass backend validation",
	} {
		if !strings.Contains(prompt, expected) {
			t.Fatalf("expected prompt to contain %q", expected)
		}
	}
}

func TestExperimentPlannerAgentRejectsAddWithoutExperiments(t *testing.T) {
	agent := NewExperimentPlannerAgent(fakeJSONGenerator{
		response: `{
			"summary": "More work is useful.",
			"decision_type": "ADD_EXPERIMENTS",
			"rationale": "Need stronger models.",
			"confidence": 0.75,
			"proposed_experiments": [],
			"champion_job_id": "",
			"risks": [],
			"expected_tradeoffs": [],
			"novelty_notes": [],
			"tags": []
		}`,
	}, "test-model")

	_, err := agent.Plan(context.Background(), testExperimentPlannerInput())
	if err == nil || !strings.Contains(err.Error(), "missing proposed_experiments") {
		t.Fatalf("expected missing proposed_experiments error, got %v", err)
	}
}

func TestExperimentPlannerAgentRejectsAddWithoutHypothesis(t *testing.T) {
	agent := NewExperimentPlannerAgent(fakeJSONGenerator{
		response: `{
			"summary": "Try a small tweak.",
			"decision_type": "ADD_EXPERIMENTS",
			"rationale": "Need stronger models.",
			"confidence": 0.75,
			"planning_mode": "exploit",
			"deterministic_diagnosis_used": ["plateau_score=0.7"],
			"evidence_used": ["prior champion family is promising"],
			"hypothesis": "",
			"expected_failure_modes": ["may plateau again"],
			"dataset_preprocessing_rationale": "No dataset-specific change.",
			"changed_variables": ["epochs"],
			"success_criteria": "Improve macro-F1.",
			"stop_condition": "Stop if it plateaus.",
			"deployment_tradeoff": "Similar latency.",
			"proposed_experiments": [
				{
					"template": "mobilenet_transfer",
					"model": "mobilenet_v3_small",
					"epochs": 14,
					"batch_size": 16,
					"learning_rate": 0.0003,
					"reason": "More epochs."
				}
			],
			"champion_job_id": "",
			"why_can_beat_champion": "More epochs might help.",
			"risks": [],
			"expected_tradeoffs": [],
			"novelty_notes": [],
			"rejected_options": [{"option": "larger model", "reason": "latency", "evidence": "goal", "applies_when": ["latency"]}],
			"tags": []
		}`,
	}, "test-model")

	_, err := agent.Plan(context.Background(), testExperimentPlannerInput())
	if err == nil || !strings.Contains(err.Error(), "missing hypothesis") {
		t.Fatalf("expected missing hypothesis error, got %v", err)
	}
}

func TestExperimentPlannerAgentRejectsMinorOnlyTweaks(t *testing.T) {
	agent := NewExperimentPlannerAgent(fakeJSONGenerator{
		response: `{
			"summary": "Try a small tweak.",
			"decision_type": "ADD_EXPERIMENTS",
			"rationale": "Need a modest improvement.",
			"confidence": 0.75,
			"planning_mode": "exploit",
			"deterministic_diagnosis_used": ["plateau_score=0.7"],
			"evidence_used": ["prior champion family is strong"],
			"hypothesis": "More epochs and a lower learning rate may help.",
			"expected_failure_modes": ["may repeat plateau"],
			"dataset_preprocessing_rationale": "No dataset-specific preprocessing change is needed.",
			"changed_variables": ["epochs", "learning_rate"],
			"success_criteria": "Improve macro-F1.",
			"stop_condition": "Stop if no improvement.",
			"deployment_tradeoff": "Similar latency.",
			"proposed_experiments": [
				{
					"template": "mobilenet_transfer",
					"model": "mobilenet_v3_small",
					"epochs": 14,
					"batch_size": 16,
					"learning_rate": 0.0002,
					"reason": "More epochs and lower learning rate."
				}
			],
			"champion_job_id": "",
			"why_can_beat_champion": "More conservative optimization might help.",
			"risks": [],
			"expected_tradeoffs": [],
			"novelty_notes": [],
			"rejected_options": [{"option": "heavy model", "reason": "latency", "evidence": "goal", "applies_when": ["latency"]}],
			"tags": []
		}`,
	}, "test-model")

	_, err := agent.Plan(context.Background(), testExperimentPlannerInput())
	if err == nil || !strings.Contains(err.Error(), "only minor tuning knobs") {
		t.Fatalf("expected minor-only tuning error, got %v", err)
	}
}

func TestExperimentPlannerPromptContextIncludesDatasetAndStrategyMemory(t *testing.T) {
	input := testExperimentPlannerInput()
	input.Dataset = datasets.Dataset{
		ID:      "dataset_1",
		Name:    "raw-profile-guard",
		Profile: map[string]any{"raw_profile_marker": "do not leak this raw profile"},
	}
	input.DatasetInsights = DatasetPlanningInsights{
		Summary:                  "Small imbalanced image dataset.",
		RecommendedPreprocessing: []string{"weighted_loss"},
		RecommendedAugmentations: []string{"color_jitter"},
		LiveInferencePriorities:  []string{"prefer compact models"},
	}
	input.SuccessfulStrategyMemory = []PlannerStrategyMemory{
		{
			OutcomeStatus:         ExperimentPlanningOutcomeImprovedChampion,
			Lesson:                "Weighted loss improved macro-F1.",
			ActualDeltaVsChampion: 0.02,
			ProposedModels:        []string{"efficientnet_b0"},
		},
	}
	input.FailedStrategyMemory = []PlannerStrategyMemory{
		{
			OutcomeStatus:         ExperimentPlanningOutcomeNoImprovement,
			Lesson:                "More MobileNet epochs did not help.",
			ActualDeltaVsChampion: -0.04,
			ProposedModels:        []string{"mobilenet_v3_small"},
		},
	}

	context := experimentPlannerPromptContext(input)
	if _, ok := context["dataset_planning_insights"]; ok {
		t.Fatal("expected raw dataset planning insights to be omitted from compact prompt context")
	}
	snapshot, ok := context["planner_context_snapshot"].(PlannerContextSnapshot)
	if !ok {
		t.Fatalf("expected planner context snapshot, got %#v", context["planner_context_snapshot"])
	}
	if snapshot.DatasetCard.Summary != "Small imbalanced image dataset." {
		t.Fatalf("expected distilled dataset card summary, got %q", snapshot.DatasetCard.Summary)
	}
	if got := len(snapshot.StrategyLessons); got != 2 {
		t.Fatalf("expected successful and failed strategy lessons, got %d", got)
	}
	blob, err := json.Marshal(context)
	if err != nil {
		t.Fatalf("marshal compact prompt context: %v", err)
	}
	if strings.Contains(string(blob), "do not leak this raw profile") {
		t.Fatal("expected compact planner context to omit raw dataset.profile payload")
	}
}

func TestExperimentPlannerPromptContextMarksVisualExemplarsAsEvidenceOnly(t *testing.T) {
	input := testExperimentPlannerInput()
	input.VisualExemplarContext = &PlannerVisualExemplarContext{
		Enabled:       true,
		EvidenceOnly:  false,
		ExemplarCount: 4,
		ClassCount:    2,
		ByteBudget:    120000,
		PromptBudget:  1800,
		Summary:       "Examples show small objects against varied backgrounds.",
		ObservedTraits: []string{
			"small_object_scale",
			"background_dominance",
		},
		ClassEvidence: []PlannerClassExemplar{
			{ClassName: "rare", ExemplarCount: 2, ObservedTraits: []string{"blur", "low_light"}},
			{ClassName: "common", ExemplarCount: 2, ObservedTraits: []string{"centered_object"}},
		},
		Warnings: []string{"class examples are sampled evidence, not validation labels"},
		Audit: map[string]any{
			"source":           "datasets.profile.visual_exemplars",
			"max_total":        24,
			"max_per_class":    2,
			"selection_policy": "class_balanced",
		},
	}

	context := experimentPlannerPromptContext(input)
	snapshot, ok := context["planner_context_snapshot"].(PlannerContextSnapshot)
	if !ok {
		t.Fatalf("expected planner context snapshot, got %#v", context["planner_context_snapshot"])
	}
	exemplarContext := snapshot.VisualEvidence
	if exemplarContext == nil {
		t.Fatal("expected visual evidence in planner snapshot")
	}
	if exemplarContext["enabled"] != true {
		t.Fatalf("expected visual exemplar context to be enabled")
	}
	if exemplarContext["evidence_only"] != true {
		t.Fatalf("expected visual exemplar context to be forced evidence-only")
	}
	caps, ok := exemplarContext["caps"].(map[string]any)
	if !ok {
		t.Fatalf("expected visual exemplar caps, got %#v", exemplarContext["caps"])
	}
	if caps["byte_budget"] != 120000 || caps["prompt_budget"] != 1800 {
		t.Fatalf("expected visual exemplar caps to include budgets, got %#v", caps)
	}
	audit, ok := exemplarContext["audit"].(map[string]any)
	if !ok {
		t.Fatalf("expected visual exemplar audit details, got %#v", exemplarContext["audit"])
	}
	if audit["source"] != "datasets.profile.visual_exemplars" {
		t.Fatalf("expected audit source to survive prompt context, got %#v", audit)
	}
	instructions, _ := exemplarContext["instructions"].(string)
	if !strings.Contains(instructions, "return JSON only") || !strings.Contains(instructions, "backend validation") || !strings.Contains(instructions, "caps") || !strings.Contains(instructions, "audit") {
		t.Fatalf("expected evidence-only JSON/backend instructions, got %q", instructions)
	}
}

func TestComputePlannerDiagnosisFlagsEvidenceDrivenFailures(t *testing.T) {
	input := testExperimentPlannerInput()
	input.SourcePlan.TargetMetric = "macro_f1"
	input.DatasetInsights = DatasetPlanningInsights{ImbalanceRatio: 3.4}
	input.CurrentChampion = &ExperimentChampion{JobID: "champion_1", Score: 0.70}
	input.NoImprovementRounds = 2
	input.PlanSummaries = []runs.TrainingRunSummary{
		{
			JobID:            "job_1",
			Status:           jobs.StatusSucceeded,
			BestMacroF1:      0.68,
			BestAccuracy:     0.84,
			FinalTrainLoss:   0.18,
			FinalValLoss:     0.74,
			EstimatedCostUSD: 0.22,
			RuntimeSeconds:   1400,
		},
	}
	input.PlanMetrics = map[string][]jobs.EpochMetric{
		"job_1": {
			{JobID: "job_1", Epoch: 1, Metrics: map[string]float64{"macro_f1": 0.62}},
			{JobID: "job_1", Epoch: 2, Metrics: map[string]float64{"macro_f1": 0.67}},
			{JobID: "job_1", Epoch: 3, Metrics: map[string]float64{"macro_f1": 0.665}},
			{JobID: "job_1", Epoch: 4, Metrics: map[string]float64{"macro_f1": 0.668}},
			{JobID: "job_1", Epoch: 5, Metrics: map[string]float64{"macro_f1": 0.666}},
		},
	}
	input.PlanEvaluations = []runs.TrainingRunEvaluation{
		{
			JobID: "job_1",
			PerClassMetrics: map[string]any{
				"rare":   map[string]any{"recall": 0.32, "f1-score": 0.30},
				"common": map[string]any{"recall": 0.89, "f1-score": 0.88},
			},
			ModelProfile: map[string]any{"estimated_latency_ms": 120.0},
		},
	}
	input.ObjectiveContext = ProjectObjectiveContext{PrimaryObjective: "low_latency_live_service"}

	diagnosis := ComputePlannerDiagnosis(input)
	if diagnosis.OverfittingScore < 0.7 {
		t.Fatalf("expected overfitting signal, got %.3f", diagnosis.OverfittingScore)
	}
	if diagnosis.PlateauScore < 0.4 {
		t.Fatalf("expected plateau signal, got %.3f", diagnosis.PlateauScore)
	}
	if diagnosis.ClassImbalanceScore < 0.5 {
		t.Fatalf("expected class imbalance signal, got %.3f", diagnosis.ClassImbalanceScore)
	}
	if diagnosis.MinorityClassFailureScore < 0.5 {
		t.Fatalf("expected minority failure signal, got %.3f", diagnosis.MinorityClassFailureScore)
	}
	if diagnosis.BestMetricDeltaVsChampion >= 0 {
		t.Fatalf("expected negative champion delta, got %.3f", diagnosis.BestMetricDeltaVsChampion)
	}
	if !containsTestString(diagnosis.RecommendedFailureModes, "minority_class_failure") {
		t.Fatalf("expected minority class failure mode, got %#v", diagnosis.RecommendedFailureModes)
	}
}

func TestExperimentPlannerRejectsExploreWithoutTwoFamilies(t *testing.T) {
	recommendation := validExperimentPlannerRecommendationForMode("explore")
	recommendation.ProposedExperiments = []plans.PlannedExperiment{
		{
			Template:     "mobilenet_transfer",
			Model:        "mobilenet_v3_small",
			Epochs:       8,
			BatchSize:    16,
			LearningRate: 0.0003,
			Reason:       "test compact baseline",
		},
		{
			Template:     "mobilenet_transfer",
			Model:        "mobilenet_v3_large",
			Epochs:       8,
			BatchSize:    16,
			LearningRate: 0.0003,
			Reason:       "test larger MobileNet baseline",
		},
	}

	err := validateExperimentPlanningRecommendation(recommendation, 5)
	if err == nil || !strings.Contains(err.Error(), "at least two model families") {
		t.Fatalf("expected explore family validation error, got %v", err)
	}
}

func TestExperimentPlannerAcceptsClassImbalanceMode(t *testing.T) {
	recommendation := validExperimentPlannerRecommendationForMode("class_imbalance_ablation")
	recommendation.Hypothesis = "Weighted loss should improve minority recall and macro-F1."
	recommendation.SuccessCriteria = "Macro-F1 and minority recall improve without a large accuracy drop."
	recommendation.ChangedVariables = []string{"class_balancing", "loss"}
	recommendation.ProposedExperiments[0].ClassBalancing = "weighted_loss"
	recommendation.ProposedExperiments[0].Reason = "Test weighted loss for minority recall."
	recommendation.ProposedExperiments[1].ClassBalancing = "class_balanced_sampler"
	recommendation.ProposedExperiments[1].Reason = "Test class-balanced sampling for minority recall."

	if err := validateExperimentPlanningRecommendation(recommendation, 5); err != nil {
		t.Fatalf("expected class imbalance mode to validate: %v", err)
	}
}

func TestExperimentPlannerRanksCandidateHypotheses(t *testing.T) {
	input := testExperimentPlannerInput()
	input.MaxExperiments = 2
	input.DeterministicDiagnosis = PlannerDiagnosis{
		ClassImbalanceScore:       0.70,
		MinorityClassFailureScore: 0.80,
		RecommendedFailureModes:   []string{"class_imbalance", "minority_class_failure"},
	}
	duplicate := input.SourcePlan.Experiments[0]
	input.ExistingExperimentSignatures = []string{candidateExperimentSignature(duplicate)}

	agent := NewExperimentPlannerAgent(fakeJSONGenerator{
		response: `{
			"summary": "Rank several diagnosis-driven options before selecting a follow-up.",
			"decision_type": "ADD_EXPERIMENTS",
			"rationale": "Minority-class performance is the clearest weakness, so class balancing should be prioritized.",
			"confidence": 0.82,
			"planning_mode": "class_imbalance_ablation",
			"deterministic_diagnosis_used": ["minority_class_failure_score=0.80"],
			"evidence_used": ["per-class recall is weak"],
			"hypothesis": "Class balancing can improve minority recall and macro-F1.",
			"expected_failure_modes": ["majority-class precision may fall"],
			"dataset_preprocessing_rationale": "Use class balancing rather than a shallow epoch tweak.",
			"changed_variables": ["class_balancing", "augmentation"],
			"success_criteria": "Improve macro-F1 and minority recall.",
			"stop_condition": "Select champion if no candidate improves minority recall.",
			"deployment_tradeoff": "Keep latency close to the current fast model.",
			"candidate_hypotheses": [
				{
					"hypothesis": "More epochs might help.",
					"planning_mode": "exploit",
					"proposed_changes": {"epochs": 2, "learning_rate": 0.0002},
					"expected_metric_impact": 0.005,
					"expected_tradeoffs": ["small extra cost"],
					"risk": "low",
					"cost_level": "low",
					"novelty_score": 0.05,
					"evidence_used": [],
					"similar_success_memory_ids": [],
					"similar_failure_memory_ids": [],
					"experiment_config": {
						"template": "mobilenet_transfer",
						"model": "mobilenet_v3_small",
						"epochs": 8,
						"batch_size": 16,
						"learning_rate": 0.0002,
						"reason": "More epochs and a lower learning rate."
					}
				},
				{
					"hypothesis": "Weighted loss should improve minority recall.",
					"planning_mode": "class_imbalance_ablation",
					"proposed_changes": {"class_balancing": "weighted_loss", "augmentation": "moderate"},
					"expected_metric_impact": 0.03,
					"expected_tradeoffs": ["may reduce majority precision"],
					"risk": "medium",
					"cost_level": "low",
					"novelty_score": 0.8,
					"evidence_used": ["minority_class_failure_score is high"],
					"similar_success_memory_ids": [],
					"similar_failure_memory_ids": [],
					"experiment_config": {
						"template": "efficientnet_transfer",
						"model": "efficientnet_b0",
						"epochs": 12,
						"batch_size": 16,
						"learning_rate": 0.0003,
						"reason": "Test weighted loss for minority recall.",
						"image_size": 224,
						"optimizer": "adamw",
						"scheduler": "cosine",
						"weight_decay": 0.01,
						"augmentation": {"horizontal_flip": true},
						"class_balancing": "weighted_loss",
						"early_stopping_patience": 4,
						"strategy": "class imbalance ablation"
					}
				}
			],
			"proposed_experiments": [],
			"champion_job_id": "",
			"why_can_beat_champion": "It addresses minority recall directly instead of repeating shallow MobileNet tuning.",
			"expected_delta_vs_champion": 0.03,
			"stop_reason": "",
			"risks": ["weighted loss may alter precision"],
			"expected_tradeoffs": ["better minority recall for possible precision loss"],
			"novelty_notes": ["backend-ranked candidate hypotheses"],
			"rejected_options": [{"option": "more epochs only", "reason": "tiny change", "evidence": "plateau", "applies_when": ["plateau"]}],
			"tags": ["candidate_ranking"]
		}`,
	}, "test-model")

	trace, err := agent.PlanWithTrace(context.Background(), input)
	if err != nil {
		t.Fatalf("plan with candidate hypotheses: %v", err)
	}
	if len(trace.Recommendation.ProposedExperiments) != 1 {
		t.Fatalf("expected one selected experiment, got %d", len(trace.Recommendation.ProposedExperiments))
	}
	if trace.Recommendation.ProposedExperiments[0].ClassBalancing != "weighted_loss" {
		t.Fatalf("expected weighted-loss candidate to be selected")
	}
	if len(trace.Recommendation.CandidateRankings) != 2 {
		t.Fatalf("expected candidate rankings")
	}
	if !trace.Recommendation.CandidateRankings[1].Selected {
		t.Fatalf("expected second candidate to be selected, rankings: %#v", trace.Recommendation.CandidateRankings)
	}
}

func TestCandidateRankingTreatsPreprocessingFieldsAsMeaningfulMechanisms(t *testing.T) {
	input := testExperimentPlannerInput()
	input.SourcePlan.Experiments = []plans.PlannedExperiment{
		{
			Template:     "mobilenet_transfer",
			Model:        "mobilenet_v3_small",
			Epochs:       8,
			BatchSize:    16,
			LearningRate: 0.0003,
			Reason:       "baseline",
		},
	}

	candidate := CandidateHypothesis{
		Hypothesis:           "Preserve aspect ratio plus weighted sampling should improve small-object minority recall.",
		PlanningMode:         "preprocessing_ablation",
		ProposedChanges:      map[string]any{"resolution_strategy": "compare_224_256", "preprocessing": "preserve_aspect_pad", "sampling_strategy": "weighted_random_sampler"},
		ExpectedMetricImpact: 0.025,
		ExpectedTradeoffs:    []string{"slightly more preprocessing cost"},
		Risk:                 "medium",
		CostLevel:            "low",
		NoveltyScore:         0.20,
		EvidenceUsed:         []string{"visual exemplars show small objects", "minority recall is weak"},
		ExperimentConfig: plans.PlannedExperiment{
			Template:           "mobilenet_transfer",
			Model:              "mobilenet_v3_small",
			Epochs:             12,
			BatchSize:          16,
			LearningRate:       0.0003,
			Reason:             "Test preserve-aspect preprocessing and weighted sampling for minority recall.",
			ImageSize:          256,
			ResolutionStrategy: "compare_224_256",
			Preprocessing: &plans.Preprocessing{
				ResizeStrategy: "preserve_aspect_pad",
				Normalization:  "imagenet",
				CropStrategy:   "bbox_crop_if_available",
				BBoxMode:       "crop_if_available",
			},
			AugmentationPolicy: "light",
			ClassBalancing:     "weighted_loss",
			SamplingStrategy:   "weighted_random_sampler",
			Strategy:           "preprocessing ablation for small-object minority failure",
		},
	}

	ranking := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})
	if ranking.Rejected {
		t.Fatalf("expected preprocessing-aware candidate to remain eligible, got %#v", ranking.Reasons)
	}
	for _, reason := range ranking.Reasons {
		if strings.Contains(reason, "lacks a meaningful new mechanism") {
			t.Fatalf("expected preprocessing fields to count as meaningful mechanisms, got %#v", ranking.Reasons)
		}
	}
}

func validExperimentPlannerRecommendationForMode(mode string) ExperimentPlanningRecommendation {
	return ExperimentPlanningRecommendation{
		Summary:                       "Test a diagnosis-driven follow-up.",
		DecisionType:                  decisions.TypeAddExperiments,
		Rationale:                     "The prior family is promising and the diagnosis supports a targeted change.",
		Confidence:                    0.8,
		PlanningMode:                  mode,
		DeterministicDiagnosisUsed:    []string{"plateau_score=0.62"},
		EvidenceUsed:                  []string{"plan metric plateaued"},
		Hypothesis:                    "A targeted change can improve macro-F1.",
		ExpectedFailureModes:          []string{"may not improve validation quality"},
		DatasetPreprocessingRationale: "Use diagnosis-specific preprocessing choices.",
		ChangedVariables:              []string{"model_family", "augmentation"},
		SuccessCriteria:               "Improve macro-F1 by at least 0.01.",
		StopCondition:                 "Select champion if no candidate improves.",
		DeploymentTradeoff:            "Slightly higher cost must be justified by quality.",
		ProposedExperiments: []plans.PlannedExperiment{
			{
				Template:     "efficientnet_transfer",
				Model:        "efficientnet_b0",
				Epochs:       10,
				BatchSize:    16,
				LearningRate: 0.0003,
				Reason:       "Test a stronger family.",
			},
			{
				Template:     "mobilenet_transfer",
				Model:        "mobilenet_v3_small",
				Epochs:       10,
				BatchSize:    16,
				LearningRate: 0.0003,
				Reason:       "Keep a compact control.",
			},
		},
		WhyCanBeatChampion:      "It changes model family and augmentation rather than only tuning epochs.",
		ExpectedDeltaVsChampion: 0.02,
		Risks:                   []string{"higher runtime"},
		ExpectedTradeoffs:       []string{"quality for cost"},
		NoveltyNotes:            []string{"new model family"},
		RejectedOptions:         []RejectedPlannerOption{{Option: "more epochs", Reason: "plateaued", Evidence: "plateau_score", AppliesWhen: []string{"plateau"}}},
		Tags:                    []string{"diagnosis"},
	}
}

func containsTestString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func testExperimentPlannerInput() ExperimentPlannerInput {
	sourceExperiment := plans.PlannedExperiment{
		Template:     "mobilenet_transfer",
		Model:        "mobilenet_v3_small",
		Epochs:       6,
		BatchSize:    16,
		LearningRate: 0.0003,
		Reason:       "baseline",
	}
	return ExperimentPlannerInput{
		SourcePlan: plans.ExperimentPlan{
			ID:           "plan_1",
			TargetMetric: "macro_f1",
			Experiments:  []plans.PlannedExperiment{sourceExperiment},
		},
		PlanJobs: []jobs.ExperimentJob{
			{
				ID:       "job_1",
				Template: jobs.TemplateTrainExperiment,
				Status:   jobs.StatusSucceeded,
				Config: map[string]any{
					"model": "mobilenet_v3_small",
				},
			},
		},
		PlanSummaries: []runs.TrainingRunSummary{
			{
				JobID:        "job_1",
				PlanID:       "plan_1",
				Model:        "mobilenet_v3_small",
				Status:       jobs.StatusSucceeded,
				BestMacroF1:  0.62,
				BestAccuracy: 0.66,
			},
		},
		PlanMetrics: map[string][]jobs.EpochMetric{
			"job_1": {
				{JobID: "job_1", Epoch: 1, Metrics: map[string]float64{"macro_f1": 0.4}},
				{JobID: "job_1", Epoch: 6, Metrics: map[string]float64{"macro_f1": 0.62}},
			},
		},
		MaxExperiments: 3,
	}
}
