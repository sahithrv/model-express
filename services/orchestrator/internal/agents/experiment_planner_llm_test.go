package agents

import (
	"context"
	"encoding/json"
	"fmt"
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
			"proposal_mechanisms": [
				{
					"experiment_index": 0,
					"mechanism": "class_imbalance",
					"intervention": "weighted_loss and weighted_random_sampler with moderate augmentation",
					"evidence_used": ["dataset imbalance ratio is high", "validation/train loss gap is visible"],
					"expected_effect": "Improve macro-F1 and minority recall while keeping inference changes bounded."
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
	if got := trace.Recommendation.ProposalMechanisms[0].Mechanism; got != "class_imbalance" {
		t.Fatalf("expected proposal mechanism to survive decode, got %q", got)
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
		"proposal_mechanisms",
		"mechanism",
		"intervention",
		"expected_effect",
		"training_dynamics_card",
		"per_class_error_card",
		"deployment_card",
		"mechanism_coverage_card",
		"label_quality_card",
		"Mechanism values should come from this taxonomy",
		"Model family is a parameter inside a mechanism",
		"preprocessing.resize_strategy values",
		"augmentation_policy values",
		"class_balancing values",
		"sampling_strategy values",
		"focal_loss",
		"Return only valid JSON",
		"planner_context_snapshot",
		"visual_evidence, when present, only as backend-curated advisory evidence",
		"latest accepted visual-analysis IDs",
		"raw Visual Agent output",
		"raw images",
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

func TestExperimentPlannerRejectsProposalWithoutMechanismExpectations(t *testing.T) {
	recommendation := validExperimentPlannerRecommendationForMode("champion_challenge")
	recommendation.ProposalMechanisms = nil

	err := validateExperimentPlanningRecommendation(recommendation, 5)
	if err == nil || !strings.Contains(err.Error(), "missing proposal_mechanisms") {
		t.Fatalf("expected missing proposal_mechanisms error, got %v", err)
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
	if snapshot.TrainingDynamicsCard.TargetMetric != "macro_f1" {
		t.Fatalf("expected training dynamics target metric, got %#v", snapshot.TrainingDynamicsCard)
	}
	if snapshot.MechanismCoverageCard.TriedMechanisms == nil {
		t.Fatalf("expected mechanism coverage card to be present")
	}
	blob, err := json.Marshal(context)
	if err != nil {
		t.Fatalf("marshal compact prompt context: %v", err)
	}
	if strings.Contains(string(blob), "do not leak this raw profile") {
		t.Fatal("expected compact planner context to omit raw dataset.profile payload")
	}
}

func TestExperimentPlannerPromptContextUsesOnlySafeDatasetMetadataSummary(t *testing.T) {
	input := testExperimentPlannerInput()
	input.Dataset.Profile = map[string]any{
		"metadata_summary": map[string]any{
			"bbox_available": true,
			"relative_path":  "train/rare/img_001.jpg",
			"storage_uri":    "s3://secret-bucket/metadata.csv",
			"raw_preview":    map[string]any{"row": "raw source content"},
		},
	}
	input.DatasetInsights = DatasetPlanningInsights{
		Summary: "Dataset has normalized metadata.",
		MetadataSummary: map[string]any{
			"bbox_available": true,
			"relative_path":  "train/rare/img_001.jpg",
			"storage_uri":    "s3://secret-bucket/metadata.csv",
			"raw_preview":    "raw source content",
			"source_rows":    []any{map[string]any{"filename": "private/path.jpg"}},
		},
		AgentSafeMetadataSummary: map[string]any{
			"status":                "SUCCEEDED",
			"source_formats":        []string{"pascal_voc"},
			"bbox_annotation_count": 8,
			"bbox_sample_count":     4,
			"annotation_counts":     map[string]any{"bbox": 8},
			"split_counts":          map[string]any{"train": 20, "val": 5},
			"raw_preview":           "never show me",
			"storage_uri":           "s3://secret-bucket/metadata.csv",
			"warnings": []any{
				map[string]any{
					"code":      "missing_label",
					"message":   "missing labels were summarized",
					"source_id": "s3://secret-bucket/metadata.csv",
				},
				map[string]any{
					"code":    "raw_path",
					"message": "raw row at /tmp/private/labels.csv was skipped",
				},
			},
		},
	}

	context := experimentPlannerPromptContext(input)
	blob, err := json.Marshal(context)
	if err != nil {
		t.Fatalf("marshal compact prompt context: %v", err)
	}
	text := string(blob)
	for _, forbidden := range []string{
		"train/rare/img_001.jpg",
		"s3://secret-bucket",
		"metadata.csv",
		"raw source content",
		"private/path.jpg",
		"never show me",
		"/tmp/private",
		"source_id",
	} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("expected planner context to exclude raw metadata detail %q: %s", forbidden, text)
		}
	}
	for _, expected := range []string{"agent_safe_metadata_summary", "bbox_annotation_count", "pascal_voc", "missing labels were summarized"} {
		if !strings.Contains(text, expected) {
			t.Fatalf("expected safe metadata context to include %q: %s", expected, text)
		}
	}
}

func TestPlannerDatasetMetadataSummaryToolIsSafeOnly(t *testing.T) {
	input := testExperimentPlannerInput()
	input.DatasetInsights.AgentSafeMetadataSummary = map[string]any{
		"source_formats":        []string{"pascal_voc"},
		"bbox_annotation_count": 3,
		"raw_preview":           "raw xml preview",
		"relative_path":         "annotations/private.xml",
	}

	result := ExecuteExperimentPlannerInformationTool(input, PlannerToolDatasetMetadataSummary, nil, PlannerInformationToolOptions{})
	if !result.Accepted {
		t.Fatalf("expected dataset metadata summary tool to be accepted: %#v", result)
	}
	blob, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal tool result: %v", err)
	}
	text := string(blob)
	if strings.Contains(text, "raw xml preview") || strings.Contains(text, "annotations/private.xml") {
		t.Fatalf("expected dataset metadata tool to exclude raw details: %s", text)
	}
	if !strings.Contains(text, "bbox_annotation_count") || !strings.Contains(text, "raw_metadata") {
		t.Fatalf("expected dataset metadata tool to include safe summary and exclusions: %s", text)
	}
}

func TestPlannerContextCardsIncludeDecisionSignals(t *testing.T) {
	input := testExperimentPlannerInput()
	input.Dataset.Profile = map[string]any{"class_names": []any{"rare", "common"}}
	input.DatasetInsights = DatasetPlanningInsights{ImbalanceRatio: 4.2}
	input.CurrentChampion = &ExperimentChampion{JobID: "champion_1", Model: "mobilenet_v3_small", TargetMetric: "macro_f1", Score: 0.70}
	input.MinimumMeaningfulImprovement = 0.01
	input.SourcePlan.Experiments[0].Mechanism = "architecture_challenge"
	input.PriorPlans = []plans.ExperimentPlan{input.SourcePlan}
	input.PlanSummaries = []runs.TrainingRunSummary{
		{
			JobID:          "job_1",
			PlanID:         "plan_1",
			Model:          "mobilenet_v3_small",
			Status:         jobs.StatusSucceeded,
			BestMacroF1:    0.68,
			BestAccuracy:   0.84,
			FinalTrainLoss: 0.18,
			FinalValLoss:   0.74,
		},
	}
	input.PlanMetrics = map[string][]jobs.EpochMetric{
		"job_1": {
			{JobID: "job_1", Epoch: 1, Metrics: map[string]float64{"macro_f1": 0.61}},
			{JobID: "job_1", Epoch: 2, Metrics: map[string]float64{"macro_f1": 0.67}},
			{JobID: "job_1", Epoch: 3, Metrics: map[string]float64{"macro_f1": 0.668}},
			{JobID: "job_1", Epoch: 4, Metrics: map[string]float64{"macro_f1": 0.669}},
		},
	}
	input.PlanEvaluations = []runs.TrainingRunEvaluation{
		{
			JobID: "job_1",
			PerClassMetrics: map[string]any{
				"rare":   map[string]any{"recall": 0.31, "f1-score": 0.28, "support": 8},
				"common": map[string]any{"recall": 0.91, "f1-score": 0.89, "support": 80},
			},
			ConfusionMatrix: [][]int{
				{3, 7},
				{1, 64},
			},
			ModelProfile: map[string]any{
				"estimated_latency_ms":         125.0,
				"throughput_images_per_second": 42.0,
				"parameter_count":              2500000.0,
				"model_size_mb":                9.5,
			},
			HolisticScores: map[string]any{
				"label_quality": map[string]any{
					"high_confidence_error_count": 3,
					"label_noise_signal":          "rare/common asymmetric high-confidence mistakes",
				},
			},
		},
	}
	input.ObjectiveContext = ProjectObjectiveContext{
		PrimaryObjective:     "low_latency_live_service",
		DeploymentPriorities: []string{"low latency"},
		RankingWeights:       map[string]float64{"quality": 0.7, "latency": 0.3},
	}
	input.DeterministicDiagnosis = ComputePlannerDiagnosis(input)
	input.RejectedStrategyMemory = []RejectedPlannerOption{{
		Option:      "repeat architecture_challenge",
		Reason:      "same mechanism only changes epochs",
		Evidence:    "plateau",
		AppliesWhen: []string{"architecture_challenge"},
	}}
	input.StrategyScorecards = []PlannerStrategyScorecard{{
		ID:           "scorecard_1",
		StrategyType: "architecture_challenge",
		PlanningMode: "champion_challenge",
		ProposedChanges: map[string]any{
			"mechanism":          "architecture_challenge",
			"intervention":       "ResNet challenger",
			"diagnosis_triggers": []any{"plateau"},
			"models":             []any{"resnet18"},
		},
		ExpectedDelta: 0.02,
		ActualDelta:   -0.03,
		Outcome:       ExperimentPlanningOutcomeNoImprovement,
		Lesson:        "Architecture-only challenger did not beat the compact champion.",
		Tags:          []string{"architecture_challenge"},
	}}

	snapshot := BuildPlannerContextSnapshot(input)
	if snapshot.TrainingDynamicsCard.MoreEpochsJustified {
		t.Fatalf("expected plateaued metrics to block more-epochs justification, got %#v", snapshot.TrainingDynamicsCard)
	}
	if len(snapshot.PerClassErrorCard.WorstClasses) == 0 || snapshot.PerClassErrorCard.WorstClasses[0].ClassName != "rare" {
		t.Fatalf("expected rare class to be surfaced as worst class, got %#v", snapshot.PerClassErrorCard.WorstClasses)
	}
	if len(snapshot.PerClassErrorCard.TopConfusionPairs) == 0 || snapshot.PerClassErrorCard.TopConfusionPairs[0].ActualClass != "rare" || snapshot.PerClassErrorCard.TopConfusionPairs[0].PredictedClass != "common" {
		t.Fatalf("expected rare->common confusion pair, got %#v", snapshot.PerClassErrorCard.TopConfusionPairs)
	}
	if !snapshot.PerClassErrorCard.ClassBalancingUseful {
		t.Fatalf("expected class balancing signal, got %#v", snapshot.PerClassErrorCard)
	}
	if snapshot.DeploymentCard.BestLatencyMS != 125 {
		t.Fatalf("expected deployment latency to be compacted, got %#v", snapshot.DeploymentCard)
	}
	if !containsTestString(snapshot.MechanismCoverageCard.TriedMechanisms, "architecture_challenge") {
		t.Fatalf("expected tried mechanism coverage, got %#v", snapshot.MechanismCoverageCard)
	}
	if !containsTestString(snapshot.MechanismCoverageCard.BlockedMechanisms, "architecture_challenge") {
		t.Fatalf("expected blocked mechanism coverage, got %#v", snapshot.MechanismCoverageCard)
	}
	if len(snapshot.MechanismCoverageCard.FailedMechanismLessons) == 0 || snapshot.MechanismCoverageCard.FailedMechanismLessons[0].Mechanism != "architecture_challenge" {
		t.Fatalf("expected failed mechanism lesson, got %#v", snapshot.MechanismCoverageCard.FailedMechanismLessons)
	}
	if !snapshot.LabelQualityCard.AuditRecommended || !containsTestString(snapshot.LabelQualityCard.SuspectClasses, "rare") {
		t.Fatalf("expected label-quality audit signal for rare class, got %#v", snapshot.LabelQualityCard)
	}
	if len(snapshot.StrategyLessons) == 0 || snapshot.StrategyLessons[0].Mechanism != "architecture_challenge" || snapshot.StrategyLessons[0].ActualDelta != -0.03 {
		t.Fatalf("expected scorecard mechanism outcome fields in strategy lessons, got %#v", snapshot.StrategyLessons)
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

func TestPlannerVisualAnalysisContextCapsAndExcludesRawPayloads(t *testing.T) {
	analysis := datasets.DatasetVisualAnalysis{
		ID:              "analysis_1",
		DatasetID:       "dataset_1",
		SchemaVersion:   datasets.VisualAnalysisSchemaVersion,
		AnalysisVersion: 3,
		TriggerReason:   datasets.VisualTriggerInitialProfile,
		TriggerDetails: map[string]any{
			"raw_visual_agent_output": "SHOULD_NOT_LEAK",
			"images":                  []string{"file:///secret-image.jpg"},
		},
		ImagesAnalyzed: 64,
		Confidence:     "medium",
		CoverageReport: datasets.VisualCoverageReport{
			SelectionStrategy: "deterministic_risk_and_representative_sampling",
			SelectionBasis:    []string{"representative", "rare", "aspect_ratio", "blur", "brightness", "confusion", "extra"},
			ImagesAvailable:   1000,
			ImagesAnalyzed:    64,
			ClassesTotal:      20,
			ClassesCovered:    12,
			Limitations:       []string{"bounded sample", "not class complete", "one more", "two more", "three more"},
			PerClassCounts:    map[string]int{},
		},
		Limitations: []string{},
	}
	for i := 0; i < 18; i++ {
		className := fmt.Sprintf("class_%02d", i)
		analysis.CoverageReport.PerClassCounts[className] = i + 1
		analysis.VisualTraits = append(analysis.VisualTraits, datasets.VisualTrait{
			Trait:           fmt.Sprintf("trait_%02d", i),
			Level:           "high",
			Confidence:      "medium",
			Evidence:        []string{"visible bounded evidence", "extra evidence"},
			AffectedClasses: []string{className, "related_a", "related_b", "related_c"},
			ExampleImageIDs: []string{"img_1", "img_2"},
		})
		analysis.ClassesToWatch = append(analysis.ClassesToWatch, datasets.ClassWatchItem{
			ClassName:       className,
			Reason:          "visually similar sampled class",
			RelatedClasses:  []string{"related_a", "related_b", "related_c", "related_d", "related_e"},
			Evidence:        []string{"visual ambiguity", "shape overlap", "texture overlap", "extra evidence"},
			ExampleImageIDs: []string{"img_1", "img_2", "img_3", "img_4", "img_5"},
			Confidence:      "medium",
		})
		analysis.PreprocessingHypotheses = append(analysis.PreprocessingHypotheses, datasets.PreprocessingHypothesis{
			ID:                     fmt.Sprintf("vh_%02d", i),
			Mechanism:              "resolution_crop",
			Summary:                "Preserve aspect ratio for small objects.",
			Evidence:               []string{"small objects", "varied aspect ratios", "background dominance", "extra evidence"},
			SuggestedPreprocessing: &plans.Preprocessing{ResizeStrategy: "preserve_aspect_pad", Normalization: "imagenet", CropStrategy: "none", BBoxMode: "ignore"},
			SuggestedImageSizes:    []int{224, 256, 320, 384},
			ExpectedEffect:         "Reduce distortion.",
			Confidence:             "medium",
			SupportStatus:          "supported",
		})
		analysis.Cautions = append(analysis.Cautions, datasets.VisualCaution{
			Operation:       fmt.Sprintf("operation_%02d", i),
			Reason:          "orientation-sensitive evidence",
			Severity:        "medium",
			Confidence:      "medium",
			AffectedClasses: []string{className, "related_a", "related_b", "related_c", "related_d"},
			ExampleImageIDs: []string{"img_1", "img_2", "img_3", "img_4", "img_5"},
		})
		analysis.Limitations = append(analysis.Limitations, fmt.Sprintf("limitation_%02d", i))
	}
	analysis.PreprocessingHypotheses = append([]datasets.PreprocessingHypothesis{{
		ID:                "vh_unsupported_secret",
		Mechanism:         "delete_dataset",
		Summary:           "unsupported_secret_hypothesis",
		Evidence:          []string{"SHOULD_NOT_LEAK"},
		SupportStatus:     "unsupported",
		UnsupportedReason: "outside backend validation",
	}}, analysis.PreprocessingHypotheses...)

	input := testExperimentPlannerInput()
	input.VisualExemplarContext = NewPlannerVisualExemplarContextFromAnalysis(analysis)
	snapshot := BuildPlannerContextSnapshot(input)
	evidence := snapshot.VisualEvidence
	if evidence["source"] != "dataset_visual_analysis" || evidence["analysis_id"] != "analysis_1" {
		t.Fatalf("expected latest visual analysis metadata, got %#v", evidence)
	}
	if evidence["evidence_only"] != true || evidence["raw_images_included"] != false || evidence["raw_visual_output_included"] != false {
		t.Fatalf("expected evidence-only raw-exclusion flags, got %#v", evidence)
	}
	if got := len(evidence["observed_traits"].([]string)); got != plannerVisualMaxObservedTraits {
		t.Fatalf("expected capped observed traits, got %d", got)
	}
	if got := len(evidence["classes_to_watch"].([]datasets.ClassWatchItem)); got != plannerVisualMaxClassesToWatch {
		t.Fatalf("expected capped classes to watch, got %d", got)
	}
	if got := len(evidence["preprocessing_hypotheses"].([]datasets.PreprocessingHypothesis)); got != plannerVisualMaxHypotheses {
		t.Fatalf("expected capped preprocessing hypotheses, got %d", got)
	}
	if got := len(evidence["cautions"].([]datasets.VisualCaution)); got != plannerVisualMaxCautions {
		t.Fatalf("expected capped cautions, got %d", got)
	}
	if got := len(evidence["limitations"].([]string)); got != plannerVisualMaxLimitations {
		t.Fatalf("expected capped limitations, got %d", got)
	}
	coverage := evidence["class_coverage"].(datasets.VisualCoverageReport)
	if got := len(coverage.PerClassCounts); got != plannerVisualMaxPerClassCounts {
		t.Fatalf("expected capped per-class coverage counts, got %d", got)
	}
	blob, err := json.Marshal(evidence)
	if err != nil {
		t.Fatalf("marshal visual evidence: %v", err)
	}
	for _, forbidden := range []string{"SHOULD_NOT_LEAK", "file:///secret-image.jpg", "vh_unsupported_secret", "unsupported_secret_hypothesis"} {
		if strings.Contains(string(blob), forbidden) {
			t.Fatalf("expected compressed visual evidence to exclude %q: %s", forbidden, string(blob))
		}
	}
}

func TestPlannerContextIncludesBackendValidationGatedMethodProposals(t *testing.T) {
	input := testExperimentPlannerInput()
	input.DatasetInsights = DatasetPlanningInsights{
		ImbalanceRatio:    3.2,
		ClassDistribution: map[string]any{"rare": 10, "common": 80},
	}
	input.VisualExemplarContext = &PlannerVisualExemplarContext{
		Enabled:      true,
		EvidenceOnly: true,
		Source:       "dataset_visual_analysis",
		AnalysisID:   "analysis_1",
		PreprocessingHypotheses: []datasets.PreprocessingHypothesis{
			{
				ID:                     "vh_bbox_001",
				Mechanism:              "bbox_crop_ablation",
				Summary:                "Background dominates small subjects; bbox crop may focus the model.",
				Evidence:               []string{"background dominance around small subjects"},
				SuggestedPreprocessing: &plans.Preprocessing{ResizeStrategy: "bbox_crop_if_available", CropStrategy: "bbox_crop_ablation", BBoxMode: "crop_if_available", Normalization: "imagenet"},
				ExpectedEffect:         "Reduce background overfitting.",
				Confidence:             "medium",
				SupportStatus:          "needs_backend_validation",
			},
		},
	}

	snapshot := BuildPlannerContextSnapshot(input)
	if len(snapshot.BackendGatedMethods) == 0 {
		t.Fatal("expected backend-validation-gated method proposals")
	}
	for _, method := range snapshot.BackendGatedMethods {
		if method.SchedulingAuthority {
			t.Fatalf("gated method %s unexpectedly has scheduling authority", method.Mechanism)
		}
	}
	bbox, ok := findBackendGatedTestMethod(snapshot.BackendGatedMethods, "bbox_crop_ablation")
	if !ok {
		t.Fatalf("expected bbox gated proposal, got %#v", snapshot.BackendGatedMethods)
	}
	if bbox.SourceID != "vh_bbox_001" {
		t.Fatalf("expected visual hypothesis source id to be retained, got %#v", bbox)
	}
	if !containsTestSubstring(bbox.MissingRequirements, "bbox/annotation") {
		t.Fatalf("expected bbox proposal to require backend bbox evidence, got %#v", bbox.MissingRequirements)
	}
	if _, ok := bbox.SupportedConfigHints["preprocessing"]; !ok {
		t.Fatalf("expected bbox proposal to retain preprocessing hints, got %#v", bbox.SupportedConfigHints)
	}
	if bbox.ProposalStatus != "backend_validation_required" {
		t.Fatalf("expected bbox proposal to be gated, got %q", bbox.ProposalStatus)
	}

	classImbalance, ok := findBackendGatedTestMethod(snapshot.BackendGatedMethods, "class_imbalance")
	if !ok {
		t.Fatalf("expected class imbalance proposal, got %#v", snapshot.BackendGatedMethods)
	}
	if !containsTestSubstring(classImbalance.MissingRequirements, "class_balancing") {
		t.Fatalf("expected class imbalance proposal to require concrete balancing config, got %#v", classImbalance.MissingRequirements)
	}
	if containsTestSubstring(classImbalance.MissingRequirements, "evidence") {
		t.Fatalf("expected imbalance evidence to be satisfied by dataset profile, got %#v", classImbalance.MissingRequirements)
	}
}

func TestPlannerMechanismCoverageToolReturnsBackendGatedMethodProposals(t *testing.T) {
	input := testExperimentPlannerInput()
	input.Dataset.Profile = map[string]any{"bbox_available": true}

	result := ExecuteExperimentPlannerInformationTool(input, PlannerToolMechanismCoverage, nil, PlannerInformationToolOptions{})
	if !result.Accepted {
		t.Fatalf("expected mechanism coverage tool to be accepted: %#v", result)
	}
	if result.Payload["gated_methods_have_scheduling_power"] != false {
		t.Fatalf("expected gated methods to have no scheduling power, got %#v", result.Payload)
	}
	methods, ok := result.Payload["backend_validation_gated_methods"].([]PlannerBackendGatedMethod)
	if !ok {
		t.Fatalf("expected typed gated methods payload, got %#v", result.Payload["backend_validation_gated_methods"])
	}
	bbox, ok := findBackendGatedTestMethod(methods, "bbox_crop_ablation")
	if !ok {
		t.Fatalf("expected bbox gated proposal from profile evidence, got %#v", methods)
	}
	if bbox.SchedulingAuthority {
		t.Fatalf("bbox gated proposal should not schedule work: %#v", bbox)
	}
	if !containsTestSubstring(bbox.MissingRequirements, "bbox crop preprocessing") {
		t.Fatalf("expected bbox proposal to require concrete config, got %#v", bbox.MissingRequirements)
	}
}

func TestPlannerBackendGatedMethodsUseAgentSafeMetadataBBoxEvidence(t *testing.T) {
	input := testExperimentPlannerInput()
	input.Dataset.Profile = map[string]any{}
	input.DatasetInsights.AgentSafeMetadataSummary = map[string]any{
		"source_formats":        []string{"pascal_voc"},
		"bbox_annotation_count": 12,
		"bbox_sample_count":     6,
		"annotation_counts":     map[string]any{"bbox": 12},
	}

	snapshot := BuildPlannerContextSnapshot(input)
	bbox, ok := findBackendGatedTestMethod(snapshot.BackendGatedMethods, "bbox_crop_ablation")
	if !ok {
		t.Fatalf("expected normalized metadata bbox evidence to surface gated bbox method, got %#v", snapshot.BackendGatedMethods)
	}
	if bbox.Source != "dataset_metadata" {
		t.Fatalf("expected bbox source to cite dataset metadata, got %#v", bbox)
	}
	if containsTestSubstring(bbox.MissingRequirements, "bbox/annotation evidence") {
		t.Fatalf("expected normalized bbox summary to satisfy backend evidence requirement, got %#v", bbox.MissingRequirements)
	}
	if !containsTestSubstring(bbox.Evidence, "normalized metadata safe summary") {
		t.Fatalf("expected evidence to cite normalized safe summary, got %#v", bbox.Evidence)
	}
}

func TestInformationValidationToolsUseStringifiedDraftSchemas(t *testing.T) {
	plannerSpec, ok := findInformationToolSpec(ExperimentPlannerInformationToolSpecs(), PlannerToolValidateCandidateExperiments)
	if !ok {
		t.Fatal("expected planner validation tool spec")
	}
	assertStringifiedDraftSchema(t, plannerSpec.Parameters, "recommendation_json")

	monitorSpec, ok := findInformationToolSpec(TrainingMonitorInformationToolSpecs(), TrainingMonitorToolValidateRecommendationDraft)
	if !ok {
		t.Fatal("expected training monitor validation tool spec")
	}
	assertStringifiedDraftSchema(t, monitorSpec.Parameters, "recommendation_json")
}

func TestPlannerValidateCandidateExperimentsAcceptsRecommendationJSON(t *testing.T) {
	recommendation := validExperimentPlannerRecommendationForMode("explore")
	blob, err := json.Marshal(recommendation)
	if err != nil {
		t.Fatalf("marshal recommendation: %v", err)
	}
	called := false

	result := ExecuteExperimentPlannerInformationTool(
		testExperimentPlannerInput(),
		PlannerToolValidateCandidateExperiments,
		map[string]any{"recommendation_json": string(blob)},
		PlannerInformationToolOptions{
			ValidateCandidateExperiments: func(draft ExperimentPlanningRecommendation) PlannerCandidateDryRunResult {
				called = true
				if draft.Summary != recommendation.Summary {
					t.Fatalf("unexpected decoded recommendation summary %q", draft.Summary)
				}
				return PlannerCandidateDryRunResult{
					Valid:                   true,
					ValidationStatus:        "valid",
					ProposedExperimentCount: len(draft.ProposedExperiments),
					WouldWriteRows:          false,
					WouldScheduleJobs:       false,
				}
			},
		},
	)
	if !result.Accepted || !called {
		t.Fatalf("expected dry-run validation to accept stringified recommendation, got %#v called=%v", result, called)
	}
	validation, ok := result.Payload["dry_run_validation"].(PlannerCandidateDryRunResult)
	if !ok || !validation.Valid || validation.WouldScheduleJobs {
		t.Fatalf("expected valid dry-run-only result, got %#v", result.Payload["dry_run_validation"])
	}
}

func TestTrainingMonitorValidateRecommendationDraftAcceptsRecommendationJSON(t *testing.T) {
	recommendation := TrainingEvaluationRecommendation{
		Summary: "The run is stable enough to rank.",
		RecommendedAction: decisions.AgentActionProposal{
			ActionType: "RANK_MODELS",
			Confidence: 0.82,
			Rationale:  "Metrics are complete and no scheduling action is required.",
			Payload:    map[string]any{"rank_basis": "macro_f1"},
		},
		QualitySummary:   "Validation quality is usable.",
		TrainingDynamics: "Metrics are stable.",
		CostSummary:      "Cost is bounded.",
		Risks:            []string{"small evaluation sample"},
		Findings:         []string{"macro-F1 is available"},
		RankScore:        0.74,
		Tags:             []string{"dry_run"},
	}
	blob, err := json.Marshal(recommendation)
	if err != nil {
		t.Fatalf("marshal monitor recommendation: %v", err)
	}

	result := ExecuteTrainingMonitorInformationTool(
		TrainingMonitorInput{},
		TrainingMonitorToolValidateRecommendationDraft,
		map[string]any{"recommendation_json": string(blob)},
		TrainingMonitorInformationToolOptions{},
	)
	if !result.Accepted {
		t.Fatalf("expected monitor draft validation to accept stringified recommendation, got %#v", result)
	}
	validation, ok := result.Payload["dry_run_validation"].(map[string]any)
	if !ok || validation["valid"] != true || validation["would_schedule_jobs"] != false {
		t.Fatalf("expected valid dry-run-only monitor validation, got %#v", result.Payload["dry_run_validation"])
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
	recommendation.ProposalMechanisms[0].Mechanism = "class_imbalance"
	recommendation.ProposalMechanisms[0].Intervention = "weighted_loss for minority recall"
	recommendation.ProposalMechanisms[0].ExpectedEffect = "Improve minority recall and macro-F1."
	recommendation.ProposalMechanisms[1].Mechanism = "minority_targeting"
	recommendation.ProposalMechanisms[1].Intervention = "class_balanced_sampler for rare-class exposure"
	recommendation.ProposalMechanisms[1].ExpectedEffect = "Improve minority recall through balanced sampling."

	if err := validateExperimentPlanningRecommendation(recommendation, 5); err != nil {
		t.Fatalf("expected class imbalance mode to validate: %v", err)
	}
}

func TestExperimentPlannerClassImbalanceModeAllowsControlExperiment(t *testing.T) {
	recommendation := validExperimentPlannerRecommendationForMode("class_imbalance_ablation")
	recommendation.Hypothesis = "Weighted loss should improve minority recall and macro-F1."
	recommendation.SuccessCriteria = "Macro-F1 and minority recall improve without a large accuracy drop."
	recommendation.ChangedVariables = []string{"class_balancing", "model_family"}
	recommendation.ProposedExperiments[0].ClassBalancing = "weighted_loss"
	recommendation.ProposedExperiments[0].Reason = "Test weighted loss for minority recall."
	recommendation.ProposedExperiments[1].Reason = "Keep an unweighted architecture challenger as a comparison point."
	recommendation.ProposalMechanisms[0].Mechanism = "class_imbalance"
	recommendation.ProposalMechanisms[0].Intervention = "weighted_loss for minority recall"
	recommendation.ProposalMechanisms[0].ExpectedEffect = "Improve minority recall and macro-F1."
	recommendation.ProposalMechanisms[1].Mechanism = "architecture_challenge"
	recommendation.ProposalMechanisms[1].Intervention = "Compare the LLM-selected stronger architecture without class balancing."
	recommendation.ProposalMechanisms[1].ExpectedEffect = "Separate class-balancing gains from representation-capacity gains."

	if err := validateExperimentPlanningRecommendation(recommendation, 5); err != nil {
		t.Fatalf("expected class imbalance mode to allow non-imbalance controls: %v", err)
	}
}

func TestExperimentPlannerClassImbalanceMechanismRequiresBalancing(t *testing.T) {
	recommendation := validExperimentPlannerRecommendationForMode("class_imbalance_ablation")
	recommendation.Hypothesis = "Weighted loss should improve minority recall and macro-F1."
	recommendation.SuccessCriteria = "Macro-F1 and minority recall improve without a large accuracy drop."
	recommendation.ChangedVariables = []string{"class_balancing", "model_family"}
	recommendation.ProposalMechanisms[0].Mechanism = "class_imbalance"
	recommendation.ProposalMechanisms[0].Intervention = "Claims to test class imbalance but omits class balancing."
	recommendation.ProposalMechanisms[0].ExpectedEffect = "Improve minority recall and macro-F1."
	recommendation.ProposalMechanisms[1].Mechanism = "architecture_challenge"

	err := validateExperimentPlanningRecommendation(recommendation, 5)
	if err == nil || !strings.Contains(err.Error(), "must include a class balancing strategy") {
		t.Fatalf("expected class imbalance mechanism validation error, got %v", err)
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
					"mechanism": "optimizer_scheduler",
					"intervention": "A small epoch and learning-rate tweak on the same MobileNet baseline.",
					"proposed_changes": {"epochs": 2, "learning_rate": 0.0002},
					"expected_effect": "Marginally improve convergence if the prior run was undertrained.",
					"expected_metric_impact": 0.005,
					"expected_tradeoffs": ["small extra cost"],
					"risk": "low",
					"cost_level": "low",
					"novelty_score": 0.05,
					"evidence_used": ["prior curve had limited training budget"],
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
					"mechanism": "class_imbalance",
					"intervention": "Use weighted_loss with moderate augmentation on EfficientNet-B0.",
					"proposed_changes": {"class_balancing": "weighted_loss", "augmentation": "moderate"},
					"expected_effect": "Improve minority recall and macro-F1 by reweighting rare classes.",
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
	if trace.Recommendation.CandidateRankings[1].Mechanism != "class_imbalance" {
		t.Fatalf("expected selected ranking to expose mechanism, got %#v", trace.Recommendation.CandidateRankings[1])
	}
	if len(trace.Recommendation.ProposalMechanisms) != 1 || trace.Recommendation.ProposalMechanisms[0].Mechanism != "class_imbalance" {
		t.Fatalf("expected selected candidate mechanism sidecar, got %#v", trace.Recommendation.ProposalMechanisms)
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
		Mechanism:            "resolution_crop",
		Intervention:         "Preserve aspect ratio, use bbox crop if available, and keep weighted sampling for weak minority recall.",
		ProposedChanges:      map[string]any{"resolution_strategy": "compare_224_256", "preprocessing": "preserve_aspect_pad", "sampling_strategy": "weighted_random_sampler"},
		ExpectedEffect:       "Improve small-object recall by avoiding aspect distortion and improving rare-class exposure.",
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

func TestCandidateRankingRewardsDiagnosisMatchedNonModelMechanism(t *testing.T) {
	input := testExperimentPlannerInput()
	input.DeterministicDiagnosis = PlannerDiagnosis{
		ClassImbalanceScore:       0.72,
		MinorityClassFailureScore: 0.78,
		RecommendedFailureModes:   []string{"class_imbalance", "minority_class_failure"},
	}

	architectureOnly := CandidateHypothesis{
		Hypothesis:           "Try a different architecture.",
		PlanningMode:         "champion_challenge",
		Mechanism:            "architecture_challenge",
		Intervention:         "Swap MobileNet for ResNet without changing data or loss behavior.",
		ProposedChanges:      map[string]any{"model": "resnet18"},
		ExpectedEffect:       "Maybe improve representation quality.",
		ExpectedMetricImpact: 0.02,
		ExpectedTradeoffs:    []string{"higher latency"},
		Risk:                 "medium",
		CostLevel:            "medium",
		NoveltyScore:         0.60,
		EvidenceUsed:         []string{"baseline macro-F1 is below target"},
		ExperimentConfig: plans.PlannedExperiment{
			Template:     "resnet_transfer",
			Model:        "resnet18",
			Epochs:       12,
			BatchSize:    16,
			LearningRate: 0.0003,
			Reason:       "Architecture-only challenger.",
		},
	}
	classBalancing := CandidateHypothesis{
		Hypothesis:           "Weighted loss should improve minority recall.",
		PlanningMode:         "class_imbalance_ablation",
		Mechanism:            "class_imbalance",
		Intervention:         "Use weighted_loss and keep the existing compact model family.",
		ProposedChanges:      map[string]any{"class_balancing": "weighted_loss"},
		ExpectedEffect:       "Improve minority recall and macro-F1 by reweighting rare classes.",
		ExpectedMetricImpact: 0.02,
		ExpectedTradeoffs:    []string{"may reduce majority precision"},
		Risk:                 "medium",
		CostLevel:            "low",
		NoveltyScore:         0.55,
		EvidenceUsed:         []string{"minority_class_failure_score is high"},
		ExperimentConfig: plans.PlannedExperiment{
			Template:       "mobilenet_transfer",
			Model:          "mobilenet_v3_small",
			Epochs:         12,
			BatchSize:      16,
			LearningRate:   0.0003,
			Reason:         "Test weighted loss for minority recall.",
			ClassBalancing: "weighted_loss",
		},
	}

	architectureRanking := scorePlannerCandidate(input, architectureOnly, 0, map[string]bool{}, map[string]bool{})
	classRanking := scorePlannerCandidate(input, classBalancing, 1, map[string]bool{}, map[string]bool{})

	if classRanking.Score <= architectureRanking.Score {
		t.Fatalf("expected class-balancing mechanism to outrank architecture-only candidate, class=%#v architecture=%#v", classRanking, architectureRanking)
	}
	if !containsTestString(classRanking.Reasons, "diagnosis-matched non-model mechanism") {
		t.Fatalf("expected diagnosis-matched non-model mechanism reason, got %#v", classRanking.Reasons)
	}
	if !containsTestString(architectureRanking.Reasons, "architecture-only candidate lacks underfitting, plateau, or champion-challenge evidence") {
		t.Fatalf("expected architecture-only penalty, got %#v", architectureRanking.Reasons)
	}
}

func TestCandidateRankingPenalizesSameMechanismMinorVariant(t *testing.T) {
	input := testExperimentPlannerInput()
	input.SourcePlan.Experiments = []plans.PlannedExperiment{
		{
			Template:       "mobilenet_transfer",
			Model:          "mobilenet_v3_small",
			Epochs:         8,
			BatchSize:      16,
			LearningRate:   0.0003,
			Reason:         "weighted-loss baseline",
			ClassBalancing: "weighted_loss",
		},
	}
	candidate := CandidateHypothesis{
		Hypothesis:           "Repeat weighted loss with a slightly lower learning rate.",
		PlanningMode:         "class_imbalance_ablation",
		Mechanism:            "class_imbalance",
		Intervention:         "Keep weighted_loss and only adjust learning rate and epochs.",
		ProposedChanges:      map[string]any{"epochs": 2, "learning_rate": 0.0002},
		ExpectedEffect:       "Small optimization improvement.",
		ExpectedMetricImpact: 0.006,
		ExpectedTradeoffs:    []string{"small extra cost"},
		Risk:                 "low",
		CostLevel:            "low",
		NoveltyScore:         0.10,
		EvidenceUsed:         []string{"prior class imbalance experiment did not clearly win"},
		ExperimentConfig: plans.PlannedExperiment{
			Template:       "mobilenet_transfer",
			Model:          "mobilenet_v3_small",
			Epochs:         10,
			BatchSize:      16,
			LearningRate:   0.0002,
			Reason:         "Same weighted-loss mechanism with minor tuning.",
			ClassBalancing: "weighted_loss",
		},
	}

	ranking := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})
	if !ranking.Rejected {
		t.Fatalf("expected same-mechanism minor variant to be rejected, got %#v", ranking)
	}
	if !containsTestString(ranking.Reasons, "same mechanism only changes minor tuning knobs") {
		t.Fatalf("expected same-mechanism penalty reason, got %#v", ranking.Reasons)
	}
}

func TestCandidateRankingRejectsMissingMechanismExpectation(t *testing.T) {
	input := testExperimentPlannerInput()
	candidate := CandidateHypothesis{
		Hypothesis:           "Weighted loss should improve minority recall.",
		PlanningMode:         "class_imbalance_ablation",
		ProposedChanges:      map[string]any{"class_balancing": "weighted_loss"},
		ExpectedMetricImpact: 0.02,
		ExpectedTradeoffs:    []string{"may reduce majority precision"},
		Risk:                 "medium",
		CostLevel:            "low",
		NoveltyScore:         0.5,
		EvidenceUsed:         []string{"minority_class_failure_score is high"},
		ExperimentConfig: plans.PlannedExperiment{
			Template:       "mobilenet_transfer",
			Model:          "mobilenet_v3_small",
			Epochs:         12,
			BatchSize:      16,
			LearningRate:   0.0003,
			Reason:         "Test weighted loss.",
			ClassBalancing: "weighted_loss",
		},
	}

	ranking := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})
	if !ranking.Rejected || !containsTestString(ranking.Reasons, "candidate_hypotheses[0] missing mechanism") {
		t.Fatalf("expected missing mechanism rejection, got %#v", ranking)
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
		ProposalMechanisms: []PlannerProposalMechanism{
			{
				ExperimentIndex: 0,
				Mechanism:       "architecture_challenge",
				Intervention:    "Test EfficientNet-B0 as a controlled challenger.",
				EvidenceUsed:    []string{"plan metric plateaued"},
				ExpectedEffect:  "Improve macro-F1 through a stronger pretrained family.",
			},
			{
				ExperimentIndex: 1,
				Mechanism:       "baseline_control",
				Intervention:    "Keep MobileNet as a compact control.",
				EvidenceUsed:    []string{"plan metric plateaued"},
				ExpectedEffect:  "Provide a low-latency comparison point for the challenger.",
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

func containsTestSubstring(values []string, target string) bool {
	for _, value := range values {
		if strings.Contains(value, target) {
			return true
		}
	}
	return false
}

func findBackendGatedTestMethod(values []PlannerBackendGatedMethod, mechanism string) (PlannerBackendGatedMethod, bool) {
	for _, value := range values {
		if value.Mechanism == mechanism {
			return value, true
		}
	}
	return PlannerBackendGatedMethod{}, false
}

func findInformationToolSpec(values []AgentInformationToolSpec, name string) (AgentInformationToolSpec, bool) {
	for _, value := range values {
		if value.Name == name {
			return value, true
		}
	}
	return AgentInformationToolSpec{}, false
}

func assertStringifiedDraftSchema(t *testing.T, schema map[string]any, propertyName string) {
	t.Helper()
	if schema["type"] != "object" || schema["additionalProperties"] != false {
		t.Fatalf("expected strict top-level object schema, got %#v", schema)
	}
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected schema properties, got %#v", schema["properties"])
	}
	property, ok := properties[propertyName].(map[string]any)
	if !ok {
		t.Fatalf("expected %s property, got %#v", propertyName, properties)
	}
	if property["type"] != "string" {
		t.Fatalf("expected %s to be a string, got %#v", propertyName, property)
	}
	if _, exists := properties["recommendation"]; exists {
		t.Fatalf("strict tool schema must not expose open recommendation object: %#v", properties["recommendation"])
	}
	required, ok := schema["required"].([]string)
	if !ok || !containsTestString(required, propertyName) {
		t.Fatalf("expected %s to be required, got %#v", propertyName, schema["required"])
	}
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
