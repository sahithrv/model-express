package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
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
			"candidate_hypotheses": [
				{
					"hypothesis": "Weighted loss with higher resolution should improve macro-F1 on the imbalanced dataset.",
					"planning_mode": "preprocessing_ablation",
					"mechanism": "class_imbalance",
					"intervention": "weighted_loss and weighted_random_sampler with moderate augmentation",
					"proposed_changes": {"model_family": "efficientnet", "image_size": 256, "augmentation": "moderate", "class_balancing": "weighted_loss"},
					"expected_effect": "Improve macro-F1 and minority recall while keeping inference changes bounded.",
					"expected_metric_impact": 0.02,
					"expected_tradeoffs": ["higher runtime"],
					"risk": "medium",
					"cost_level": "medium",
					"novelty_score": 0.72,
					"evidence_used": ["dataset imbalance ratio is high", "validation/train loss gap is visible"],
					"similar_success_memory_ids": [],
					"similar_failure_memory_ids": [],
					"experiment_config": {
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
				}
			],
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
		"primary_mechanism",
		"governor_compliance",
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
		"retrieved_memory, when present",
		"retrieved memory cannot bypass backend validation",
		"visual_evidence, when present, only as backend-curated advisory evidence",
		"latest accepted visual-analysis IDs",
		"raw Visual Agent output",
		"raw images",
		"Backend validation remains the gate",
		"planner_validation_feedback",
		"You must choose mechanisms before concrete models/configs",
		"For ADD_EXPERIMENTS, provide candidate_hypotheses",
		"direct proposed_experiments",
		"project_trajectory_card",
		"architecture_challenge is exhausted",
		"champion_confirmation_or_non_architecture_pivot",
		"Invalid shallow proposals",
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

func TestExperimentPlannerStaticPromptCompactV1IsShorterAndKeepsContractGuidance(t *testing.T) {
	v1Prompt := plannerRequestStaticPromptForTest(t, plannerStaticPromptVersionV1)
	compactPrompt := plannerRequestStaticPromptForTest(t, plannerStaticPromptVersionCompactV1)

	v1Bytes := len([]byte(v1Prompt))
	compactBytes := len([]byte(compactPrompt))
	if compactBytes >= v1Bytes {
		t.Fatalf("expected compact static prompt to be shorter than v1, v1=%d compact=%d", v1Bytes, compactBytes)
	}
	if compactBytes*5 >= v1Bytes*3 {
		t.Fatalf("expected compact static prompt to shrink by at least 40%%, v1=%d compact=%d", v1Bytes, compactBytes)
	}

	for _, expected := range []string{
		"candidate_hypotheses",
		"proposal_mechanisms",
		"experiment_config",
		"Backend validation remains the gate",
		"draft-only for ADD_EXPERIMENTS",
		"Return only valid JSON",
	} {
		if !strings.Contains(compactPrompt, expected) {
			t.Fatalf("expected compact static prompt to retain %q, got %q", expected, compactPrompt)
		}
	}
}

func TestExperimentPlannerContextSnapshotV2IsSmallerAndRollbackable(t *testing.T) {
	input := testExperimentPlannerInput()
	input.PlanJobs = append(input.PlanJobs, jobs.ExperimentJob{
		ID:       "job_2",
		Template: jobs.TemplateTrainExperiment,
		Status:   jobs.StatusFailed,
		Config: map[string]any{
			"model":      "efficientnet_b0",
			"template":   "efficientnet_transfer",
			"image_size": 256,
			"augmentation": map[string]any{
				"horizontal_flip": true,
				"color_jitter":    true,
			},
		},
	})
	input.PlanSummaries = append(input.PlanSummaries, runs.TrainingRunSummary{
		JobID:            "job_2",
		PlanID:           "plan_1",
		Model:            "efficientnet_b0",
		Status:           jobs.StatusFailed,
		BestMacroF1:      0.58,
		BestAccuracy:     0.63,
		FinalTrainLoss:   0.24,
		FinalValLoss:     0.69,
		EstimatedCostUSD: 1.25,
		RuntimeSeconds:   842,
	})
	for i := 3; i <= 24; i++ {
		input.PlanSummaries = append(input.PlanSummaries, runs.TrainingRunSummary{
			JobID:            fmt.Sprintf("job_%d", i),
			PlanID:           "plan_1",
			Model:            fmt.Sprintf("efficientnet_b%d", i),
			Status:           jobs.StatusSucceeded,
			BestMacroF1:      0.6 + float64(i)*0.01,
			BestAccuracy:     0.65 + float64(i)*0.01,
			FinalTrainLoss:   0.30 - float64(i)*0.005,
			FinalValLoss:     0.60 - float64(i)*0.004,
			EstimatedCostUSD: 1.0 + float64(i)*0.1,
			RuntimeSeconds:   700 + float64(i)*10,
		})
	}
	for i := 3; i <= 24; i++ {
		jobID := fmt.Sprintf("job_%d", i)
		input.PlanEvaluations = append(input.PlanEvaluations, runs.TrainingRunEvaluation{
			JobID: jobID,
			PerClassMetrics: map[string]any{
				"rare":   map[string]any{"recall": 0.29 + float64(i)*0.005, "f1-score": 0.26 + float64(i)*0.005, "support": 8},
				"common": map[string]any{"recall": 0.92, "f1-score": 0.90, "support": 80},
			},
			ConfusionMatrix: [][]int{{2, 8}, {1, 65}},
			ModelProfile: map[string]any{
				"estimated_latency_ms":         118.0 + float64(i),
				"throughput_images_per_second": 44.0,
				"parameter_count":              3200000.0 + float64(i)*10000,
				"model_size_mb":                11.2 + float64(i)*0.1,
				"deployment_notes":             "this extra-long model profile exists so the compact snapshot can prove it strips v1-only verbose payloads from the ledger",
			},
			HolisticScores: map[string]any{
				"training_diagnostics": map[string]any{
					"trend":              "validation quality improved, then flattened, then improved again after targeted regularization",
					"gap_summary":        "train/validation gap remained visible across epochs but narrowed after the augmentation change",
					"loss_curve_notes":   "the run kept enough headroom to justify the mechanism pivot, not another shallow epoch-only repeat",
					"diagnostic_warning": "the repeated plateau signal should not be treated as a license for more of the same",
				},
			},
			RecommendationSummary: "long-lived diagnostic evidence for compact-snapshot shrinking",
		})
	}
	input.PlanEvaluations = append(input.PlanEvaluations, runs.TrainingRunEvaluation{
		JobID: "job_2",
		PerClassMetrics: map[string]any{
			"rare":   map[string]any{"recall": 0.29, "f1-score": 0.26, "support": 8},
			"common": map[string]any{"recall": 0.92, "f1-score": 0.90, "support": 80},
		},
		ConfusionMatrix: [][]int{{2, 8}, {1, 65}},
		ModelProfile: map[string]any{
			"estimated_latency_ms":         118.0,
			"throughput_images_per_second": 44.0,
			"parameter_count":              3200000.0,
			"model_size_mb":                11.2,
		},
		HolisticScores: map[string]any{
			"label_quality": map[string]any{
				"high_confidence_error_count": 2,
				"label_noise_signal":          "rare/common confusion remains",
			},
		},
	})
	input.SourcePlanDeltas = []ExperimentRunDelta{
		{
			JobID:                  "job_2",
			PlanID:                 "plan_1",
			Model:                  "efficientnet_b0",
			Status:                 jobs.StatusFailed,
			TargetMetric:           "macro_f1",
			Score:                  0.58,
			BestMacroF1:            0.58,
			BestAccuracy:           0.63,
			EstimatedCostUSD:       1.25,
			RuntimeSeconds:         842,
			EpochsCompleted:        10,
			ChampionJobID:          "champion_1",
			DeltaScoreVsChampion:   -0.04,
			DeltaCostVsChampion:    0.25,
			DeltaRuntimeVsChampion: 112,
		},
	}
	input.SuccessfulStrategyMemory = []PlannerStrategyMemory{
		{
			MemoryID:                "memory_success_1",
			OutcomeStatus:           ExperimentPlanningOutcomeImprovedChampion,
			Lesson:                  "Weighted loss improved minority recall on the same dataset shape.",
			BestModel:               "efficientnet_b0",
			ActualDeltaVsChampion:   0.018,
			ExpectedDeltaVsChampion: 0.02,
			TotalCostUSD:            1.2,
			TotalRuntimeSeconds:     810,
			ProposedModels:          []string{"efficientnet_b0", "mobilenet_v3_small"},
			Tags:                    []string{"class_imbalance", "success"},
		},
	}
	input.FailedStrategyMemory = []PlannerStrategyMemory{
		{
			MemoryID:                "memory_failed_1",
			OutcomeStatus:           ExperimentPlanningOutcomeNoImprovement,
			Lesson:                  "Architecture-only repeats plateaued.",
			BestModel:               "resnet18",
			ActualDeltaVsChampion:   -0.03,
			ExpectedDeltaVsChampion: 0.01,
			TotalCostUSD:            0.9,
			TotalRuntimeSeconds:     660,
			ProposedModels:          []string{"resnet18", "efficientnet_b0"},
			Tags:                    []string{"architecture_challenge", "failure"},
		},
	}
	input.RejectedStrategyMemory = []RejectedPlannerOption{
		{Option: "repeat architecture sweep", Reason: "same mechanism only changes epochs", Evidence: "architecture_challenge exhausted", AppliesWhen: []string{"architecture_challenge"}},
		{Option: "more epochs", Reason: "plateaued", Evidence: "training_dynamics_card says no more epochs", AppliesWhen: []string{"plateau"}},
	}
	input.RetrievedMemory = []memory.MemoryRetrievalResult{
		retrievedPlannerTestResult(memory.SourceStrategyScorecard, "scorecard_1", "strategy_scorecard", ExperimentPlanningOutcomeImprovedChampion, "similar success"),
		retrievedPlannerTestResult(memory.SourceAgentMemoryRecord, "memory_2", memory.KindPlanningFeedback, "rejected", "blocked repeat"),
		retrievedPlannerTestResult(memory.SourceDatasetPreprocessing, "dataset_1", memory.KindDatasetPreprocessingHypothesis, "", "similar preprocessing lesson"),
	}
	input.StrategyScorecards = []PlannerStrategyScorecard{
		{
			ID:               "scorecard_1",
			DatasetID:        "dataset_1",
			SourceDecisionID: "decision_1",
			SourcePlanID:     "plan_1",
			FollowUpPlanID:   "plan_2",
			StrategyType:     "class_imbalance",
			PlanningMode:     "class_imbalance_ablation",
			Mechanism:        "class_imbalance",
			Intervention:     "weighted_loss",
			EvidenceUsed:     []string{"minority class recall is weak", "validation/train gap is visible"},
			ExpectedEffect:   "Improve macro-F1 and minority recall.",
			DatasetTraits:    map[string]any{"imbalance_ratio": 4.2, "small_objects": true},
			ObjectiveProfile: map[string]any{"primary": "macro_f1", "live_budget": 25.0},
			ProposedChanges:  map[string]any{"class_balancing": "weighted_loss", "sampling_strategy": "weighted_random_sampler"},
			ExpectedDelta:    0.02,
			ActualDelta:      0.018,
			ConfidenceBefore: 0.72,
			ConfidenceAfter:  0.81,
			CostUSD:          1.15,
			RuntimeSeconds:   802,
			Outcome:          ExperimentPlanningOutcomeImprovedChampion,
			Lesson:           "Weighted loss helped more than another architecture sweep.",
			Tags:             []string{"class_imbalance", "improved"},
		},
	}
	input.OptimizerFeedback = []automl.OptimizerFeedbackSummary{
		{
			StudyID:                 "study_1",
			TrialCount:              6,
			SucceededTrialCount:     4,
			FailedTrialCount:        2,
			TargetMetric:            "macro_f1",
			BestScore:               0.74,
			BestJobID:               "trial_1",
			BestHyperparameters:     map[string]any{"learning_rate": 0.0002, "weight_decay": 0.01},
			TrainValidationGap:      0.11,
			Trend:                   "plateau",
			FailedParameterPatterns: []string{"more_epochs_only"},
			RecommendedNarrowing:    []string{"keep learning rate modest", "change mechanism before more epochs"},
		},
	}
	input.ValidationFeedback = []PlannerValidationFeedback{
		{
			Attempt:             2,
			ValidationError:     "experiment planner ADD_EXPERIMENTS missing candidate_hypotheses",
			RejectedDecision:    "ADD_EXPERIMENTS",
			RejectedModels:      []string{"mobilenet_v3_small"},
			RejectedExperiments: []string{"more epochs"},
			Instructions:        []string{"include candidate_hypotheses", "change mechanism materially"},
		},
	}
	input.DatasetInsights.AgentSafeMetadataSummary = map[string]any{
		"source_formats":        []string{"pascal_voc"},
		"bbox_annotation_count": 12,
		"bbox_sample_count":     6,
		"annotation_counts":     map[string]any{"bbox": 12},
	}
	input.ProjectTrajectory = PlannerProjectTrajectoryCard{
		CompletedTrainingRuns:  3,
		CompletedPlannerRounds: 2,
		FirstSuccessfulScore:   0.61,
		CurrentChampionScore:   0.7,
		AbsoluteChampionGain:   0.09,
		GainPerCompletedRun:    0.03,
		RecentBestDelta:        0.018,
		MinimumUsefulDelta:     0.01,
		NoImprovementRounds:    1,
		DecisionPressure:       "champion_confirmation_or_non_architecture_pivot",
		MechanismOutcomes: []PlannerMechanismOutcome{
			{
				Mechanism:           "architecture_challenge",
				AttemptCount:        3,
				PlanCount:           3,
				BestScore:           0.7,
				BestDeltaVsPrior:    0.0,
				RecentBestDelta:     0.0,
				Status:              "exhausted",
				ExhaustionReason:    "Repeated architecture-only sweeps failed to improve the champion.",
				AllowedNextOnlyWith: []string{"champion_confirmation", "non_architecture_pivot"},
			},
		},
		BlockedMechanisms: []string{"architecture_challenge"},
		Warnings:          []string{"avoid shallow repeats"},
	}
	addBulkyPlannerHistoryForV2Test(&input)

	v1Snapshot := plannerSnapshotForVersionTest(t, input, "v1")
	v2Snapshot := plannerSnapshotForVersionTest(t, input, "v2")

	if v1Snapshot.ContextVersion != "v1" {
		t.Fatalf("expected v1 snapshot version, got %q", v1Snapshot.ContextVersion)
	}
	if v2Snapshot.ContextVersion != "v2" {
		t.Fatalf("expected v2 snapshot version, got %q", v2Snapshot.ContextVersion)
	}

	v1Blob, err := json.Marshal(v1Snapshot)
	if err != nil {
		t.Fatalf("marshal v1 snapshot: %v", err)
	}
	v2Blob, err := json.Marshal(v2Snapshot)
	if err != nil {
		t.Fatalf("marshal v2 snapshot: %v", err)
	}
	if len(v2Blob) >= len(v1Blob) {
		t.Fatalf("expected v2 context to be smaller than v1, v1=%d v2=%d", len(v1Blob), len(v2Blob))
	}
	if len(v2Blob)*2 >= len(v1Blob) {
		t.Fatalf("expected v2 context to shrink by at least 50%%, v1=%d v2=%d", len(v1Blob), len(v2Blob))
	}
	if v2Snapshot.PromptBudget.SectionEstimates["planner_context_snapshot_total"].Bytes <= 0 {
		t.Fatalf("expected planner context section estimate to be populated, got %#v", v2Snapshot.PromptBudget.SectionEstimates["planner_context_snapshot_total"])
	}
}

func TestExperimentPlannerPromptBudgetSectionEstimatesCoverPlannerSections(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_PLANNER_CONTEXT_VERSION", "v2")
	t.Setenv("MODEL_EXPRESS_PLANNER_STATIC_PROMPT_VERSION", plannerStaticPromptVersionCompactV1)

	input := testExperimentPlannerInput()
	input.RetrievedMemory = []memory.MemoryRetrievalResult{
		retrievedPlannerTestResult(memory.SourceStrategyScorecard, "scorecard_1", "strategy_scorecard", ExperimentPlanningOutcomeImprovedChampion, "similar success"),
	}
	input.OptimizerFeedback = []automl.OptimizerFeedbackSummary{
		{
			StudyID:              "study_1",
			TrialCount:           4,
			SucceededTrialCount:  3,
			FailedTrialCount:     1,
			TargetMetric:         "macro_f1",
			BestScore:            0.74,
			BestJobID:            "trial_1",
			BestHyperparameters:  map[string]any{"learning_rate": 0.0002},
			Trend:                "improving",
			RecommendedNarrowing: []string{"keep the mechanism but narrow the hyperparameters"},
		},
	}
	input.ValidationFeedback = []PlannerValidationFeedback{
		{
			Attempt:          1,
			ValidationError:  "experiment planner ADD_EXPERIMENTS missing candidate_hypotheses",
			RejectedDecision: "ADD_EXPERIMENTS",
			Instructions:     []string{"include candidate_hypotheses", "keep proposed experiments backend-valid"},
		},
	}

	context := experimentPlannerPromptContext(input)
	snapshot, ok := context["planner_context_snapshot"].(PlannerContextSnapshot)
	if !ok {
		t.Fatalf("expected planner context snapshot, got %#v", context["planner_context_snapshot"])
	}
	budget := snapshot.PromptBudget
	required := []string{
		"static_instructions",
		"output_schema",
		"planner_context_snapshot_total",
		"dataset_card",
		"objective_context",
		"champion_card",
		"completed_experiment_log",
		"project_trajectory_card",
		"training_dynamics_card",
		"per_class_error_card",
		"deployment_card",
		"mechanism_coverage_card",
		"backend_gated_methods",
		"label_quality_card",
		"search_coverage",
		"strategy_lessons",
		"retrieved_memory",
		"blocked_repeats",
		"model_catalog",
		"optimizer_feedback",
		"validation_feedback",
	}
	for _, key := range required {
		estimate, ok := budget.SectionEstimates[key]
		if !ok {
			t.Fatalf("expected section estimate for %q, got %#v", key, budget.SectionEstimates)
		}
		if estimate.Bytes <= 0 || estimate.ApproximateTokens <= 0 {
			t.Fatalf("expected positive estimate for %q, got %#v", key, estimate)
		}
	}
}

func TestExperimentPlannerGovernorComplianceShapeAccepted(t *testing.T) {
	var recommendation ExperimentPlanningRecommendation
	if err := json.Unmarshal([]byte(`{
		"summary": "Use a governor-aware pivot.",
		"decision_type": "WAIT",
		"rationale": "No scheduling action is required for this shape test.",
		"confidence": 0.7,
		"primary_mechanism": "class_imbalance",
		"governor_compliance": {
			"blocked_mechanisms_seen": ["architecture_challenge"],
			"avoided_blocked_mechanisms": true,
			"why_allowed_to_continue": "The draft avoids exhausted architecture work.",
			"expected_value_justification": "The expected delta clears the project useful-delta floor."
		}
	}`), &recommendation); err != nil {
		t.Fatalf("unmarshal recommendation: %v", err)
	}

	if recommendation.PrimaryMechanism != "class_imbalance" {
		t.Fatalf("expected primary mechanism to decode, got %q", recommendation.PrimaryMechanism)
	}
	if !recommendation.GovernorCompliance.AvoidedBlockedMechanisms {
		t.Fatalf("expected governor compliance flag to decode")
	}
	if !containsTestString(recommendation.GovernorCompliance.BlockedMechanismsSeen, "architecture_challenge") {
		t.Fatalf("expected blocked mechanisms to decode, got %#v", recommendation.GovernorCompliance.BlockedMechanismsSeen)
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
	if err == nil || !strings.Contains(err.Error(), "requires candidate_hypotheses") {
		t.Fatalf("expected missing candidate_hypotheses error, got %v", err)
	}
}

func TestExperimentPlannerAgentRejectsAddWithoutHypothesis(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

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
			"candidate_hypotheses": [
				{
					"hypothesis": "Weighted loss should improve minority recall.",
					"planning_mode": "class_imbalance_ablation",
					"mechanism": "class_imbalance",
					"intervention": "Use weighted_loss on the compact model.",
					"proposed_changes": {"class_balancing": "weighted_loss"},
					"expected_effect": "Improve minority recall and macro-F1.",
					"expected_metric_impact": 0.02,
					"expected_tradeoffs": ["possible precision drop"],
					"risk": "medium",
					"cost_level": "low",
					"novelty_score": 0.6,
					"evidence_used": ["prior champion family is promising"],
					"similar_success_memory_ids": [],
					"similar_failure_memory_ids": [],
					"experiment_config": {
						"template": "mobilenet_transfer",
						"model": "mobilenet_v3_small",
						"epochs": 12,
						"batch_size": 16,
						"learning_rate": 0.0003,
						"reason": "Weighted-loss candidate.",
						"class_balancing": "weighted_loss"
					}
				}
			],
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
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

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
			"candidate_hypotheses": [
				{
					"hypothesis": "Weighted loss should improve minority recall.",
					"planning_mode": "class_imbalance_ablation",
					"mechanism": "class_imbalance",
					"intervention": "Use weighted_loss rather than another architecture sweep.",
					"proposed_changes": {"class_balancing": "weighted_loss"},
					"expected_effect": "Improve minority recall and macro-F1.",
					"expected_metric_impact": 0.02,
					"expected_tradeoffs": ["possible precision drop"],
					"risk": "medium",
					"cost_level": "low",
					"novelty_score": 0.6,
					"evidence_used": ["prior champion family is strong"],
					"similar_success_memory_ids": [],
					"similar_failure_memory_ids": [],
					"experiment_config": {
						"template": "mobilenet_transfer",
						"model": "mobilenet_v3_small",
						"epochs": 12,
						"batch_size": 16,
						"learning_rate": 0.0003,
						"reason": "Weighted-loss candidate.",
						"class_balancing": "weighted_loss"
					}
				}
			],
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

func TestExperimentPlannerAgentAllowsMinorOnlyTweaksInRelaxedMode(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "false")

	recommendation := validExperimentPlannerRecommendationForMode("exploit")
	recommendation.ChangedVariables = []string{"epochs", "learning_rate"}
	recommendation.ProposalMechanisms = nil

	if err := validateExperimentPlanningRecommendation(recommendation, 5); err != nil {
		t.Fatalf("expected relaxed planner validation to allow minor-only tweak: %v", err)
	}
}

func TestExperimentPlannerRejectsProposalWithoutMechanismExpectations(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

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

func TestPlannerTrainingDynamicsCardUsesYoloMAPEpochSlope(t *testing.T) {
	input := testExperimentPlannerInput()
	input.SourcePlan.TargetMetric = "mAP50_95"
	input.PlanSummaries = []runs.TrainingRunSummary{
		{
			JobID:        "job_yolo",
			PlanID:       "plan_1",
			Model:        "yolov8n.pt",
			Status:       jobs.StatusSucceeded,
			BestMacroF1:  0.44,
			BestAccuracy: 0.55,
		},
	}
	input.PlanMetrics = map[string][]jobs.EpochMetric{
		"job_yolo": {
			{JobID: "job_yolo", Epoch: 1, Metrics: map[string]float64{"mAP50_95": 0.20, "macro_f1": 0.90}},
			{JobID: "job_yolo", Epoch: 2, Metrics: map[string]float64{"mAP50_95": 0.30, "macro_f1": 0.91}},
			{JobID: "job_yolo", Epoch: 3, Metrics: map[string]float64{"mAP50_95": 0.36, "macro_f1": 0.92}},
			{JobID: "job_yolo", Epoch: 4, Metrics: map[string]float64{
				"mAP50_95":  0.44,
				"mAP50":     0.55,
				"precision": 0.61,
				"recall":    0.58,
				"box_loss":  0.31,
				"cls_loss":  0.22,
				"dfl_loss":  0.18,
				"macro_f1":  0.99,
			}},
		},
	}

	card := plannerTrainingDynamicsCard(input)
	if card.TargetMetric != "map50_95" {
		t.Fatalf("expected detector target metric to be preserved, got %#v", card)
	}
	if card.FinalMetric != 0.44 {
		t.Fatalf("expected final metric from mAP50_95 rows, got %#v", card)
	}
	if card.MetricSlopeLastNEpochs != 0.14 {
		t.Fatalf("expected last-3 slope from mAP50_95 rows, got %#v", card)
	}
	if card.DetectorMetrics["final_mAP50_95"] != 0.44 || card.DetectorMetrics["final_recall"] != 0.58 {
		t.Fatalf("expected compact detector evidence, got %#v", card.DetectorMetrics)
	}
	joinedEvidence := strings.Join(card.Evidence, " ")
	if !strings.Contains(joinedEvidence, "mAP50") || strings.Contains(joinedEvidence, "macro-F1") {
		t.Fatalf("expected detector evidence wording, got %#v", card.Evidence)
	}
}

func TestExperimentPlannerPromptContextIncludesRetrievedMemoryCards(t *testing.T) {
	input := testExperimentPlannerInput()
	input.RetrievedMemory = []memory.MemoryRetrievalResult{
		{
			SourceTable:     memory.SourceStrategyScorecard,
			SourceID:        "scorecard_1",
			ProjectID:       "project_1",
			DatasetID:       "dataset_1",
			Kind:            "strategy_scorecard",
			Score:           0.91,
			SemanticScore:   0.82,
			StructuredScore: 0.73,
			RetrievalReason: "similar class imbalance objective",
			SummaryCard: map[string]any{
				"outcome":        ExperimentPlanningOutcomeImprovedChampion,
				"mechanism":      "class_imbalance",
				"intervention":   "weighted_loss",
				"lesson":         "Weighted loss improved macro-F1 on a similar imbalance pattern.",
				"embedding_text": "do not leak embedding text",
			},
			Metadata: map[string]any{
				"source_plan_id": "plan_old",
				"raw_payload":    map[string]any{"secret": "do not leak raw payload"},
				"storage_uri":    "s3://private-bucket/memory.json",
				"mechanism":      "class_imbalance",
			},
		},
	}

	snapshot := BuildPlannerContextSnapshot(input)
	if snapshot.RetrievedMemory == nil {
		t.Fatal("expected retrieved memory snapshot")
	}
	if got := len(snapshot.RetrievedMemory.DistilledLessons); got != 1 {
		t.Fatalf("expected one distilled retrieved card, got %d", got)
	}
	card := snapshot.RetrievedMemory.DistilledLessons[0]
	if card.SourceTable != memory.SourceStrategyScorecard || card.SourceID != "scorecard_1" {
		t.Fatalf("expected source-linked retrieved card, got %#v", card)
	}
	if card.Outcome != ExperimentPlanningOutcomeImprovedChampion || card.Mechanism != "class_imbalance" {
		t.Fatalf("expected outcome and mechanism on retrieved card, got %#v", card)
	}
	if len(card.EvidenceUsed) == 0 || !containsStringFold(card.EvidenceUsed, "scorecard_1") {
		t.Fatalf("expected evidence IDs to preserve source IDs, got %#v", card.EvidenceUsed)
	}
	if snapshot.PromptBudget.RetrievedMemoryCount != 1 || snapshot.PromptBudget.RetrievedMemoryApproximateBytes <= 0 {
		t.Fatalf("expected retrieved memory prompt budget telemetry, got %#v", snapshot.PromptBudget)
	}
	rawBlob, err := json.Marshal(input.RetrievedMemory)
	if err != nil {
		t.Fatalf("marshal raw retrieval memory: %v", err)
	}
	blob, err := json.Marshal(snapshot.RetrievedMemory)
	if err != nil {
		t.Fatalf("marshal retrieved snapshot: %v", err)
	}
	if len(blob) >= len(rawBlob) {
		t.Fatalf("expected distilled retrieved memory to shrink, raw=%d compact=%d", len(rawBlob), len(blob))
	}
	text := string(blob)
	for _, forbidden := range []string{"embedding_text", "do not leak embedding text", "raw_payload", "do not leak raw payload", "private-bucket"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("expected retrieved memory snapshot to exclude %q: %s", forbidden, text)
		}
	}
}

func TestExperimentPlannerRetrievedMemorySnapshotRemainsBounded(t *testing.T) {
	input := testExperimentPlannerInput()
	for i := 0; i < 4; i++ {
		input.RetrievedMemory = append(input.RetrievedMemory, retrievedPlannerTestResult(memory.SourceDatasetPreprocessing, fmt.Sprintf("dataset_%d", i), memory.KindDatasetPreprocessingHypothesis, "", "dataset preprocessing analogy"))
	}
	for i := 0; i < 4; i++ {
		input.RetrievedMemory = append(input.RetrievedMemory, retrievedPlannerTestResult(memory.SourceStrategyScorecard, fmt.Sprintf("success_%d", i), "strategy_scorecard", ExperimentPlanningOutcomeImprovedChampion, "successful prior strategy"))
	}
	for i := 0; i < 4; i++ {
		input.RetrievedMemory = append(input.RetrievedMemory, retrievedPlannerTestResult(memory.SourceStrategyScorecard, fmt.Sprintf("failed_%d", i), "strategy_scorecard", ExperimentPlanningOutcomeNoImprovement, "failed prior strategy"))
	}
	for i := 0; i < 5; i++ {
		input.RetrievedMemory = append(input.RetrievedMemory, retrievedPlannerTestResult(memory.SourceAgentMemoryRecord, fmt.Sprintf("blocked_%d", i), "planning_feedback", "rejected", "blocked repeat"))
	}
	for i := 0; i < 3; i++ {
		input.RetrievedMemory = append(input.RetrievedMemory, retrievedPlannerTestResult(memory.SourceAgentMemoryRecord, fmt.Sprintf("other_%d", i), "training_evaluation", "", "related training dynamics"))
	}

	snapshot := BuildPlannerContextSnapshot(input)
	if snapshot.RetrievedMemory == nil {
		t.Fatal("expected retrieved memory snapshot")
	}
	retrieved := snapshot.RetrievedMemory
	if got := plannerRetrievedMemoryCardCount(retrieved); got > plannerRetrievedMemoryMaxTotal {
		t.Fatalf("expected total retrieved memory cap %d, got %d", plannerRetrievedMemoryMaxTotal, got)
	}
	if got := len(retrieved.DistilledLessons); got > plannerRetrievedMemoryMaxDistilled {
		t.Fatalf("expected distilled cap %d, got %d", plannerRetrievedMemoryMaxDistilled, got)
	}
	if got := len(retrieved.FailedOrBlockedLessons); got > plannerRetrievedMemoryMaxFailedOrBlocked {
		t.Fatalf("expected failed/blocked cap %d, got %d", plannerRetrievedMemoryMaxFailedOrBlocked, got)
	}
	if got := len(retrieved.RawFallbackCards); got != 0 {
		t.Fatalf("expected no raw fallback when distilled lessons exist, got %d", got)
	}
}

func TestExperimentPlannerRetrievedMemoryDedupesByMechanismAndAppliesWhenSignature(t *testing.T) {
	input := testExperimentPlannerInput()
	input.RetrievedMemory = []memory.MemoryRetrievalResult{
		{
			SourceTable:     memory.SourceStrategyScorecard,
			SourceID:        "scorecard_a",
			ProjectID:       "project_1",
			DatasetID:       "dataset_1",
			Kind:            "strategy_scorecard",
			Score:           0.91,
			SemanticScore:   0.83,
			StructuredScore: 0.79,
			RetrievalReason: "same mechanism and applies_when signature",
			SummaryCard: map[string]any{
				"outcome":        ExperimentPlanningOutcomeImprovedChampion,
				"mechanism":      "class_imbalance",
				"lesson":         "Weighted loss helped minority recall.",
				"applies_when":   []any{"low minority recall", "macro_f1 focus"},
				"evidence_used":  []any{"scorecard_a", "plan_10"},
				"source_plan_id": "plan_10",
			},
		},
		{
			SourceTable:     memory.SourceStrategyScorecard,
			SourceID:        "scorecard_b",
			ProjectID:       "project_1",
			DatasetID:       "dataset_1",
			Kind:            "strategy_scorecard",
			Score:           0.89,
			SemanticScore:   0.81,
			StructuredScore: 0.76,
			RetrievalReason: "same mechanism and applies_when signature",
			SummaryCard: map[string]any{
				"outcome":        ExperimentPlanningOutcomeImprovedChampion,
				"mechanism":      "class_imbalance",
				"lesson":         "Balanced sampler also helped minority recall.",
				"applies_when":   []any{"macro_f1 focus", "low minority recall"},
				"evidence_used":  []any{"scorecard_b", "plan_11"},
				"source_plan_id": "plan_11",
			},
		},
	}

	snapshot := BuildPlannerContextSnapshot(input)
	if snapshot.RetrievedMemory == nil {
		t.Fatal("expected retrieved memory snapshot")
	}
	if got := len(snapshot.RetrievedMemory.DistilledLessons); got != 1 {
		t.Fatalf("expected deduped distilled lessons, got %d", got)
	}
	card := snapshot.RetrievedMemory.DistilledLessons[0]
	if card.Mechanism != "class_imbalance" {
		t.Fatalf("unexpected deduped mechanism: %#v", card)
	}
	if !containsStringFold(card.EvidenceUsed, "scorecard_a") || !containsStringFold(card.EvidenceUsed, "scorecard_b") {
		t.Fatalf("expected merged evidence IDs after dedupe, got %#v", card.EvidenceUsed)
	}
	if containsStringFold(card.AppliesWhen, "scorecard_a") || containsStringFold(card.AppliesWhen, "scorecard_b") || containsStringFold(card.AppliesWhen, "plan_10") || containsStringFold(card.AppliesWhen, "plan_11") {
		t.Fatalf("expected applies_when signature to ignore evidence ids, got %#v", card.AppliesWhen)
	}
	if len(card.AppliesWhen) == 0 || !containsStringFold(card.AppliesWhen, "low minority recall") {
		t.Fatalf("expected deduped applies_when signature, got %#v", card.AppliesWhen)
	}
}

func TestExperimentPlannerModelCatalogIsCompactAndTaskSeparated(t *testing.T) {
	input := testExperimentPlannerInput()
	input.ModelCatalog = []SupportedModelSpec{
		{
			Name:                  "mobilenet_v3_small",
			Family:                "mobilenet",
			TaskType:              "image_classification",
			ModelKind:             "torchvision_classifier",
			DeploymentTier:        "fast_live",
			DefaultImageSize:      224,
			MinRecommendedImages:  50,
			SupportsTransfer:      true,
			TrainingEnabled:       true,
			ExpectedLatencyClass:  "very_fast",
			RecommendedUse:        "compact live baseline",
			SupportsFineTuneModes: []string{"head_only", "last_block", "full"},
		},
		{
			Name:                 "yolo11n.pt",
			Family:               "yolo11",
			TaskType:             "object_detection",
			ModelKind:            "ultralytics_yolo_detector",
			PretrainedWeights:    "yolo11n.pt",
			DeploymentTier:       "realtime_detector",
			DefaultImageSize:     640,
			MinRecommendedImages: 100,
			SupportsTransfer:     true,
			TrainingEnabled:      true,
			ExpectedLatencyClass: "very_fast",
			RecommendedUse:       "nano detector",
		},
		{
			Name:            "disabled_test_model",
			Family:          "resnet",
			TaskType:        "image_classification",
			ModelKind:       "torchvision_classifier",
			TrainingEnabled: false,
			RecommendedUse:  "should not be exposed",
		},
	}

	snapshot := BuildPlannerContextSnapshot(input)
	if len(snapshot.ModelCatalog) != 2 {
		t.Fatalf("expected only training-enabled model catalog entries, got %#v", snapshot.ModelCatalog)
	}
	blob, err := json.Marshal(snapshot.ModelCatalog)
	if err != nil {
		t.Fatalf("marshal compact model catalog: %v", err)
	}
	text := string(blob)
	for _, forbidden := range []string{"recommended_use", "supports_fine_tune_modes", "deployment_tier", "min_recommended_images", "expected_latency_class", "pretrained_weights", "training_enabled", "model_kind", "family", "quality_tier"} {
		if strings.Contains(text, forbidden) {
			t.Fatalf("expected compact catalog to omit %q: %s", forbidden, text)
		}
	}
	if !snapshot.ModelCatalog[0].Default || !snapshot.ModelCatalog[1].Default {
		t.Fatalf("expected task defaults to be marked in compact catalog, got %#v", snapshot.ModelCatalog)
	}
	taskTypes := []string{snapshot.ModelCatalog[0].TaskType, snapshot.ModelCatalog[1].TaskType}
	if !containsStringFold(taskTypes, "image_classification") || !containsStringFold(taskTypes, "object_detection") {
		t.Fatalf("expected explicit task separation in compact catalog, got %#v", snapshot.ModelCatalog)
	}
	if snapshot.ModelCatalog[0].TaskType == snapshot.ModelCatalog[1].TaskType && snapshot.ModelCatalog[0].ID == snapshot.ModelCatalog[1].ID {
		t.Fatalf("expected distinct compact catalog entries: %#v", snapshot.ModelCatalog)
	}
	for _, card := range snapshot.ModelCatalog {
		if card.ID == "disabled_test_model" {
			t.Fatalf("did not expect disabled model in compact catalog: %#v", snapshot.ModelCatalog)
		}
		if len(card.EligibilityNotes) == 0 {
			t.Fatalf("expected eligibility notes on compact catalog entry: %#v", card)
		}
		if strings.Contains(strings.Join(card.EligibilityNotes, ","), "transfer_enabled") || strings.Contains(strings.Join(card.EligibilityNotes, ","), "YOLO evidence required") {
			t.Fatalf("expected compact eligibility notes, got %#v", card.EligibilityNotes)
		}
	}
}

func TestExperimentPlannerPromptContextOmitsRetrievedMemoryWhenAbsent(t *testing.T) {
	snapshot := BuildPlannerContextSnapshot(testExperimentPlannerInput())
	if snapshot.RetrievedMemory != nil {
		t.Fatalf("expected no retrieved memory snapshot, got %#v", snapshot.RetrievedMemory)
	}
	blob, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatalf("marshal snapshot: %v", err)
	}
	if strings.Contains(string(blob), "retrieved_memory") {
		t.Fatalf("expected disabled/no-results snapshot to omit retrieved_memory: %s", string(blob))
	}
	if strings.Contains(string(blob), "retrieved_memory_count") {
		t.Fatalf("expected disabled/no-results prompt budget to omit retrieved memory telemetry: %s", string(blob))
	}
}

func TestPlannerMemoryToolIncludesRetrievedMemoryWhenPresent(t *testing.T) {
	input := testExperimentPlannerInput()
	input.RetrievedMemory = []memory.MemoryRetrievalResult{
		retrievedPlannerTestResult(memory.SourceStrategyScorecard, "scorecard_1", "strategy_scorecard", ExperimentPlanningOutcomeImprovedChampion, "similar success"),
	}

	result := ExecuteExperimentPlannerInformationTool(input, PlannerToolMemory, nil, PlannerInformationToolOptions{})
	if !result.Accepted {
		t.Fatalf("expected memory tool to be accepted: %#v", result)
	}
	retrieved, ok := result.Payload["retrieved_memory"].(*PlannerRetrievedMemorySnapshot)
	if !ok || retrieved == nil {
		t.Fatalf("expected retrieved memory payload, got %#v", result.Payload["retrieved_memory"])
	}
	if got := plannerRetrievedMemoryCardCount(retrieved); got != 1 {
		t.Fatalf("expected one retrieved card, got %d", got)
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
	recommendation := validExperimentPlannerRecommendationForMode("class_imbalance_ablation")
	recommendation.ChangedVariables = []string{"class_balancing", "sampling_strategy"}
	recommendation.ProposedExperiments[0].ClassBalancing = "weighted_loss"
	recommendation.ProposedExperiments[0].SamplingStrategy = "weighted_random_sampler"
	recommendation.ProposedExperiments[0].Reason = "Test weighted loss and sampling for minority recall."
	recommendation.CandidateHypotheses = []CandidateHypothesis{
		{
			Hypothesis:           "Weighted loss and sampling should improve minority recall.",
			PlanningMode:         "class_imbalance_ablation",
			Mechanism:            "class_imbalance",
			Intervention:         "Use weighted_loss plus weighted_random_sampler.",
			ProposedChanges:      map[string]any{"class_balancing": "weighted_loss", "sampling_strategy": "weighted_random_sampler"},
			ExpectedEffect:       "Improve minority recall and macro-F1.",
			ExpectedMetricImpact: 0.025,
			ExpectedTradeoffs:    []string{"may reduce majority precision"},
			Risk:                 "medium",
			CostLevel:            "low",
			NoveltyScore:         0.75,
			EvidenceUsed:         []string{"minority_class_failure_score is high"},
			ExperimentConfig:     recommendation.ProposedExperiments[0],
		},
	}
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

func TestPlannerValidateCandidateExperimentsReportsFinalizerRejection(t *testing.T) {
	recommendation := validExperimentPlannerRecommendationForMode("class_imbalance_ablation")
	recommendation.CandidateHypotheses = nil
	blob, err := json.Marshal(recommendation)
	if err != nil {
		t.Fatalf("marshal recommendation: %v", err)
	}

	result := ExecuteExperimentPlannerInformationTool(
		testExperimentPlannerInput(),
		PlannerToolValidateCandidateExperiments,
		map[string]any{"recommendation_json": string(blob)},
		PlannerInformationToolOptions{},
	)
	if !result.Accepted {
		t.Fatalf("expected dry-run validation payload, got %#v", result)
	}
	validation, ok := result.Payload["dry_run_validation"].(PlannerCandidateDryRunResult)
	if !ok || validation.Valid || validation.WouldScheduleJobs || validation.WouldWriteRows {
		t.Fatalf("expected invalid dry-run-only result, got %#v", result.Payload["dry_run_validation"])
	}
	if !strings.Contains(validation.ValidationError, "requires candidate_hypotheses") {
		t.Fatalf("expected finalizer rejection text, got %q", validation.ValidationError)
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
	input.SourcePlan = plans.ExperimentPlan{TargetMetric: "macro_f1"}
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

func TestComputeProjectTrajectoryDiagnosisExhaustsLowYieldArchitectureSweep(t *testing.T) {
	input := testExperimentPlannerInput()
	input.SourcePlan = plans.ExperimentPlan{TargetMetric: "macro_f1"}
	input.MinimumMeaningfulImprovement = 0
	input.CurrentChampion = &ExperimentChampion{JobID: "job_20", TargetMetric: "macro_f1", Score: 0.642}
	input.PriorPlans = []plans.ExperimentPlan{
		{
			ID:           "plan_arch",
			TargetMetric: "macro_f1",
			Experiments:  []plans.PlannedExperiment{{Model: "resnet18", Mechanism: "architecture_challenge"}},
		},
	}
	input.PriorJobs = nil
	input.PriorSummaries = nil
	input.PlanJobs = nil
	input.PlanSummaries = nil
	scores := []float64{0.600, 0.620, 0.635, 0.640, 0.641, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642, 0.642}
	for index, score := range scores {
		jobID := fmt.Sprintf("job_%02d", index+1)
		input.PriorJobs = append(input.PriorJobs, jobs.ExperimentJob{
			ID: jobID,
			Config: map[string]any{
				"plan_id":          "plan_arch",
				"experiment_index": 0,
				"model":            "resnet18",
				"mechanism":        "architecture_challenge",
			},
		})
		input.PriorSummaries = append(input.PriorSummaries, runs.TrainingRunSummary{
			JobID:       jobID,
			PlanID:      "plan_arch",
			Model:       "resnet18",
			Status:      jobs.StatusSucceeded,
			BestMacroF1: score,
		})
	}

	trajectory := ComputeProjectTrajectoryDiagnosis(input)
	if trajectory.CompletedTrainingRuns != 20 {
		t.Fatalf("expected all prior runs to count, got %#v", trajectory)
	}
	if trajectory.MinimumUsefulDelta != 0.010 {
		t.Fatalf("expected minimum useful delta floor, got %.3f", trajectory.MinimumUsefulDelta)
	}
	if trajectory.AbsoluteChampionGain != 0.042 || trajectory.GainPerCompletedRun != 0.002 {
		t.Fatalf("expected low-yield trajectory gain, got gain=%.3f per_run=%.3f", trajectory.AbsoluteChampionGain, trajectory.GainPerCompletedRun)
	}
	if trajectory.DecisionPressure != "champion_confirmation_or_non_architecture_pivot" {
		t.Fatalf("expected low-yield decision pressure, got %q", trajectory.DecisionPressure)
	}
	outcome, ok := findTrajectoryTestOutcome(trajectory.MechanismOutcomes, "architecture_challenge")
	if !ok {
		t.Fatalf("expected architecture outcome, got %#v", trajectory.MechanismOutcomes)
	}
	if outcome.AttemptCount != 20 || outcome.Status != "exhausted" {
		t.Fatalf("expected exhausted architecture outcome, got %#v", outcome)
	}
	if !containsTestString(trajectory.BlockedMechanisms, "architecture_challenge") {
		t.Fatalf("expected architecture to be blocked, got %#v", trajectory.BlockedMechanisms)
	}
}

func TestComputeProjectTrajectoryDiagnosisPrefersStoredMechanismMetadata(t *testing.T) {
	input := testExperimentPlannerInput()
	input.SourcePlan = plans.ExperimentPlan{TargetMetric: "macro_f1"}
	input.PriorPlans = []plans.ExperimentPlan{
		{
			ID:           "plan_imbalance",
			TargetMetric: "macro_f1",
			Experiments:  []plans.PlannedExperiment{{Model: "efficientnet_b2"}},
		},
	}
	input.PriorJobs = []jobs.ExperimentJob{
		{
			ID: "job_imbalance",
			Config: map[string]any{
				"plan_id":          "plan_imbalance",
				"experiment_index": 0,
				"model":            "efficientnet_b2",
				"mechanism":        "class_imbalance",
			},
		},
	}
	input.PriorSummaries = []runs.TrainingRunSummary{
		{
			JobID:       "job_imbalance",
			PlanID:      "plan_imbalance",
			Model:       "efficientnet_b2",
			Status:      jobs.StatusSucceeded,
			BestMacroF1: 0.71,
		},
	}
	input.PlanJobs = nil
	input.PlanSummaries = nil

	trajectory := ComputeProjectTrajectoryDiagnosis(input)
	outcome, ok := findTrajectoryTestOutcome(trajectory.MechanismOutcomes, "class_imbalance")
	if !ok || outcome.AttemptCount != 1 {
		t.Fatalf("expected stored class_imbalance mechanism to win over local model inference, got %#v", trajectory.MechanismOutcomes)
	}
	if architecture, ok := findTrajectoryTestOutcome(trajectory.MechanismOutcomes, "architecture_challenge"); ok && architecture.AttemptCount != 0 {
		t.Fatalf("did not expect architecture attempts when stored mechanism exists, got %#v", trajectory.MechanismOutcomes)
	}
}

func TestExperimentPlannerRejectsExploreWithoutTwoFamilies(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

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
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")

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

func TestCandidateRankingRetrievedSuccessfulMemoryAddsBonus(t *testing.T) {
	input := candidateRankingMemoryTestInput()
	candidate := candidateRankingClassImbalanceCandidate()
	baseline := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})

	input.RetrievedMemory = []memory.MemoryRetrievalResult{
		retrievedPlannerTestResult(memory.SourceStrategyScorecard, "scorecard_success", "strategy_scorecard", ExperimentPlanningOutcomeImprovedChampion, "weighted loss improved minority recall"),
	}
	ranking := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})

	if ranking.Score <= baseline.Score {
		t.Fatalf("expected retrieved success memory to increase score, baseline=%#v retrieved=%#v", baseline, ranking)
	}
	if ranking.ScoreComponents["retrieved_memory"] <= 0 || ranking.ScoreComponents["memory_similarity"] <= baseline.ScoreComponents["memory_similarity"] {
		t.Fatalf("expected positive retrieved memory components, got %#v", ranking.ScoreComponents)
	}
	if len(ranking.RetrievedMemoryHits) != 1 || ranking.RetrievedMemoryHits[0].Effect != "bonus" {
		t.Fatalf("expected bonus retrieved memory hit, got %#v", ranking.RetrievedMemoryHits)
	}
	if !containsTestString(ranking.Reasons, "similar retrieved successful strategy") {
		t.Fatalf("expected retrieved success reason, got %#v", ranking.Reasons)
	}
}

func TestCandidateRankingRetrievedFailedMemoryAppliesPenalty(t *testing.T) {
	input := candidateRankingMemoryTestInput()
	candidate := candidateRankingClassImbalanceCandidate()
	baseline := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})

	input.RetrievedMemory = []memory.MemoryRetrievalResult{
		retrievedPlannerTestResult(memory.SourceStrategyScorecard, "scorecard_failed", "strategy_scorecard", ExperimentPlanningOutcomeNoImprovement, "weighted loss did not improve macro-F1"),
	}
	ranking := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})

	if ranking.Score >= baseline.Score {
		t.Fatalf("expected retrieved failed memory to lower score, baseline=%#v retrieved=%#v", baseline, ranking)
	}
	if ranking.ScoreComponents["retrieved_memory"] >= 0 {
		t.Fatalf("expected negative retrieved memory component, got %#v", ranking.ScoreComponents)
	}
	if len(ranking.RetrievedMemoryHits) != 1 || ranking.RetrievedMemoryHits[0].Effect != "penalty" {
		t.Fatalf("expected penalty retrieved memory hit, got %#v", ranking.RetrievedMemoryHits)
	}
	if !containsTestString(ranking.Reasons, "similar retrieved failed mechanism") {
		t.Fatalf("expected retrieved failure reason, got %#v", ranking.Reasons)
	}
}

func TestCandidateRankingRetrievedRejectedOptionBlocksCandidate(t *testing.T) {
	input := candidateRankingMemoryTestInput()
	input.RetrievedMemory = []memory.MemoryRetrievalResult{
		retrievedPlannerTestResult(memory.SourceAgentMemoryRecord, "memory_rejected", memory.KindPlanningFeedback, "rejected", "rejected weighted loss repeat"),
	}

	ranking := scorePlannerCandidate(input, candidateRankingClassImbalanceCandidate(), 0, map[string]bool{}, map[string]bool{})

	if !ranking.Rejected || ranking.Score != 0 {
		t.Fatalf("expected rejected retrieved memory to block candidate, got %#v", ranking)
	}
	if ranking.ScoreComponents["retrieved_memory"] >= 0 {
		t.Fatalf("expected negative retrieved memory component, got %#v", ranking.ScoreComponents)
	}
	if len(ranking.RetrievedMemoryHits) != 1 || ranking.RetrievedMemoryHits[0].Effect != "blocked" {
		t.Fatalf("expected blocked retrieved memory hit, got %#v", ranking.RetrievedMemoryHits)
	}
	if !containsTestString(ranking.Reasons, "blocked by retrieved rejected option") {
		t.Fatalf("expected retrieved rejected option reason, got %#v", ranking.Reasons)
	}
}

func TestCandidateRankingUnrelatedRetrievedMemoryDoesNotOverrideStructuredMismatch(t *testing.T) {
	input := candidateRankingMemoryTestInput()
	candidate := candidateRankingClassImbalanceCandidate()
	baseline := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})
	input.RetrievedMemory = []memory.MemoryRetrievalResult{{
		SourceTable:     memory.SourceStrategyScorecard,
		SourceID:        "scorecard_unrelated",
		ProjectID:       "project_1",
		DatasetID:       "dataset_1",
		Kind:            "strategy_scorecard",
		Score:           0.99,
		SemanticScore:   0.99,
		StructuredScore: 0.1,
		RetrievalReason: "semantically mentions class balancing but structured mechanism is different",
		SummaryCard: map[string]any{
			"outcome":      ExperimentPlanningOutcomeImprovedChampion,
			"mechanism":    "architecture_challenge",
			"intervention": "resnet architecture swap",
			"lesson":       "Class balancing words appear here, but this memory is about architecture shopping.",
			"summary":      "weighted_loss minority recall class imbalance text should not be enough",
		},
		Metadata: map[string]any{
			"outcome":   ExperimentPlanningOutcomeImprovedChampion,
			"mechanism": "architecture_challenge",
			"models":    []string{"resnet18"},
		},
	}}

	ranking := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})

	if ranking.ScoreComponents["retrieved_memory"] != 0 || len(ranking.RetrievedMemoryHits) != 0 {
		t.Fatalf("expected unrelated retrieved memory to be ignored, got %#v hits=%#v", ranking.ScoreComponents, ranking.RetrievedMemoryHits)
	}
	if ranking.Score != baseline.Score {
		t.Fatalf("expected unrelated memory not to change score, baseline=%#v retrieved=%#v", baseline, ranking)
	}
}

func TestMultiFidelityPolicyDoesNotLetWeakYOLO11nBlockLargerYOLORescue(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_MULTI_FIDELITY_POLICY", "1")
	input := testExperimentPlannerInput()
	input.DeterministicDiagnosis.UnderfittingScore = 0.72
	input.RejectedStrategyMemory = []RejectedPlannerOption{{
		Option:      "YOLO11n champion",
		Reason:      "weak YOLO11n preview underperformed",
		Evidence:    "yolo11n recall was near zero on the preview subset",
		AppliesWhen: []string{"architecture_challenge"},
	}}
	candidate := CandidateHypothesis{
		Hypothesis:           "A larger YOLO detector may have enough capacity.",
		PlanningMode:         "champion_challenge",
		Mechanism:            "architecture_challenge",
		Intervention:         "Compare YOLO11s against the weak YOLO11n preview baseline.",
		ExpectedEffect:       "Improve recall and mAP while staying in the YOLO family.",
		ExpectedMetricImpact: 0.03,
		Risk:                 "medium",
		CostLevel:            "medium",
		NoveltyScore:         0.6,
		EvidenceUsed:         []string{"weak YOLO11n preview should not block larger YOLO variants"},
		ExperimentConfig: plans.PlannedExperiment{
			Template:     "yolo11_detection",
			Model:        "yolo11s.pt",
			Epochs:       8,
			BatchSize:    8,
			LearningRate: 0.001,
			ImageSize:    512,
			Reason:       "Challenge the weak nano detector with a small YOLO model.",
			Mechanism:    "architecture_challenge",
		},
	}

	ranking := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})

	if ranking.Rejected {
		t.Fatalf("expected YOLO11s rescue candidate not to be rejected, got %#v", ranking)
	}
	if ranking.PromotionDecision != "rescue" {
		t.Fatalf("expected rescue decision, got %#v", ranking)
	}
	if !strings.Contains(strings.Join(ranking.Reasons, " "), "rescued by multi-fidelity policy") {
		t.Fatalf("expected weak-baseline rescue reason, got %#v", ranking.Reasons)
	}
}

func TestMultiFidelityPolicyBudgetStopRejectsHighCostCandidate(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_MULTI_FIDELITY_POLICY", "1")
	input := testExperimentPlannerInput()
	input.DeterministicDiagnosis.UnderfittingScore = 0.8
	input.StopSignals = []string{"budget cap exceeded for new full training"}
	candidate := CandidateHypothesis{
		Hypothesis:           "An expensive model might improve quality.",
		PlanningMode:         "champion_challenge",
		Mechanism:            "architecture_challenge",
		Intervention:         "Try an expensive high-capacity challenger.",
		ExpectedEffect:       "Maybe improve the champion.",
		ExpectedMetricImpact: 0.04,
		Risk:                 "medium",
		CostLevel:            "high",
		NoveltyScore:         0.7,
		EvidenceUsed:         []string{"quality target remains unmet"},
		ExperimentConfig: plans.PlannedExperiment{
			Template:     "convnext_transfer",
			Model:        "convnext_tiny",
			Epochs:       30,
			BatchSize:    8,
			LearningRate: 0.0002,
			ImageSize:    384,
			Reason:       "Try a high-cost challenger.",
			Mechanism:    "architecture_challenge",
		},
	}

	ranking := scorePlannerCandidate(input, candidate, 0, map[string]bool{}, map[string]bool{})

	if !ranking.Rejected {
		t.Fatalf("expected budget policy to reject high-cost candidate, got %#v", ranking)
	}
	if ranking.PromotionDecision != "reject" || !strings.Contains(ranking.StopReason, "budget stop signal") {
		t.Fatalf("expected budget stop reason, got %#v", ranking)
	}
}

func candidateRankingMemoryTestInput() ExperimentPlannerInput {
	input := testExperimentPlannerInput()
	input.DeterministicDiagnosis = PlannerDiagnosis{
		ClassImbalanceScore:       0.72,
		MinorityClassFailureScore: 0.78,
		RecommendedFailureModes:   []string{"class_imbalance", "minority_class_failure"},
	}
	return input
}

func candidateRankingClassImbalanceCandidate() CandidateHypothesis {
	return CandidateHypothesis{
		Hypothesis:           "Weighted loss should improve minority recall.",
		PlanningMode:         "class_imbalance_ablation",
		Mechanism:            "class_imbalance",
		Intervention:         "Use weighted_loss for minority recall.",
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
			Mechanism:      "class_imbalance",
		},
	}
}

func retrievedPlannerTestResult(sourceTable string, sourceID string, kind string, outcome string, reason string) memory.MemoryRetrievalResult {
	summary := strings.TrimSpace(reason)
	if summary == "" {
		summary = "related compact memory"
	}
	return memory.MemoryRetrievalResult{
		SourceTable:     sourceTable,
		SourceID:        sourceID,
		ProjectID:       "project_1",
		DatasetID:       "dataset_1",
		Kind:            kind,
		Score:           0.8,
		SemanticScore:   0.7,
		StructuredScore: 0.6,
		RetrievalReason: reason,
		SummaryCard: map[string]any{
			"outcome":      outcome,
			"mechanism":    "class_imbalance",
			"intervention": "weighted_loss",
			"lesson":       summary,
			"summary":      summary,
		},
		Metadata: map[string]any{
			"outcome":   outcome,
			"mechanism": "class_imbalance",
		},
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
		CandidateHypotheses: []CandidateHypothesis{
			{
				Hypothesis:           "EfficientNet-B0 can challenge the current family with a stronger pretrained backbone.",
				PlanningMode:         mode,
				Mechanism:            "architecture_challenge",
				Intervention:         "Test EfficientNet-B0 as a controlled challenger.",
				ProposedChanges:      map[string]any{"model_family": "efficientnet", "augmentation": "moderate"},
				ExpectedEffect:       "Improve macro-F1 through a stronger pretrained family.",
				ExpectedMetricImpact: 0.02,
				ExpectedTradeoffs:    []string{"higher runtime"},
				Risk:                 "medium",
				CostLevel:            "medium",
				NoveltyScore:         0.7,
				EvidenceUsed:         []string{"plan metric plateaued"},
				ExperimentConfig: plans.PlannedExperiment{
					Template:     "efficientnet_transfer",
					Model:        "efficientnet_b0",
					Epochs:       10,
					BatchSize:    16,
					LearningRate: 0.0003,
					Reason:       "Test a stronger family.",
				},
			},
			{
				Hypothesis:           "A compact control keeps the comparison grounded.",
				PlanningMode:         mode,
				Mechanism:            "baseline_control",
				Intervention:         "Keep MobileNet as a compact control.",
				ProposedChanges:      map[string]any{"control": "compact_mobilenet"},
				ExpectedEffect:       "Provide a low-latency comparison point for the challenger.",
				ExpectedMetricImpact: 0.012,
				ExpectedTradeoffs:    []string{"lower upside"},
				Risk:                 "low",
				CostLevel:            "low",
				NoveltyScore:         0.4,
				EvidenceUsed:         []string{"plan metric plateaued"},
				ExperimentConfig: plans.PlannedExperiment{
					Template:     "mobilenet_transfer",
					Model:        "mobilenet_v3_small",
					Epochs:       10,
					BatchSize:    16,
					LearningRate: 0.0003,
					Reason:       "Keep a compact control.",
				},
			},
		},
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

func findTrajectoryTestOutcome(values []PlannerMechanismOutcome, mechanism string) (PlannerMechanismOutcome, bool) {
	for _, value := range values {
		if value.Mechanism == mechanism {
			return value, true
		}
	}
	return PlannerMechanismOutcome{}, false
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

func plannerRequestStaticPromptForTest(t *testing.T, version string) string {
	t.Helper()
	t.Setenv("MODEL_EXPRESS_PLANNER_STATIC_PROMPT_VERSION", version)
	request := experimentPlannerJSONRequest("test-model", []byte(`{"planner_context_snapshot":{}}`))
	if len(request.Messages) != 2 {
		t.Fatalf("expected system and user messages, got %d", len(request.Messages))
	}
	user := request.Messages[1].Content
	if index := strings.LastIndex(user, "Context:\n"); index >= 0 {
		user = user[:index]
	}
	return request.Messages[0].Content + "\n" + user
}

func plannerSnapshotForVersionTest(t *testing.T, input ExperimentPlannerInput, version string) PlannerContextSnapshot {
	t.Helper()
	t.Setenv("MODEL_EXPRESS_PLANNER_CONTEXT_VERSION", version)
	context := experimentPlannerPromptContext(input)
	snapshot, ok := context["planner_context_snapshot"].(PlannerContextSnapshot)
	if !ok {
		t.Fatalf("expected planner context snapshot, got %#v", context["planner_context_snapshot"])
	}
	return snapshot
}

func addBulkyPlannerHistoryForV2Test(input *ExperimentPlannerInput) {
	mechanisms := []string{"architecture_challenge", "class_imbalance", "resolution_crop", "regularization"}
	input.PriorJobs = nil
	input.PriorSummaries = nil
	input.PriorEvaluations = nil
	for index := 0; index < 30; index++ {
		jobID := fmt.Sprintf("history_job_%02d", index)
		planID := fmt.Sprintf("history_plan_%02d", index)
		model := fmt.Sprintf("efficientnet_b%d", index%4)
		mechanism := mechanisms[index%len(mechanisms)]
		input.PriorJobs = append(input.PriorJobs, jobs.ExperimentJob{
			ID:       jobID,
			Template: jobs.TemplateTrainExperiment,
			Status:   jobs.StatusSucceeded,
			Config: map[string]any{
				"model":     model,
				"mechanism": mechanism,
				"notes":     strings.Repeat("full historical config detail; ", 10),
			},
		})
		input.PriorSummaries = append(input.PriorSummaries, runs.TrainingRunSummary{
			JobID:            jobID,
			PlanID:           planID,
			Model:            model,
			Status:           jobs.StatusSucceeded,
			BestMacroF1:      0.60 + float64(index%8)*0.01,
			BestAccuracy:     0.64 + float64(index%6)*0.01,
			FinalTrainLoss:   0.34 + float64(index%4)*0.01,
			FinalValLoss:     0.45 + float64(index%5)*0.02,
			EpochsCompleted:  8 + index%5,
			EstimatedCostUSD: 0.8 + float64(index%5)*0.2,
			RuntimeSeconds:   500 + float64(index*25),
		})
		input.PriorEvaluations = append(input.PriorEvaluations, runs.TrainingRunEvaluation{
			JobID: jobID,
			ModelProfile: map[string]any{
				"estimated_latency_ms":         25 + index,
				"throughput_images_per_second": 30 + index,
				"parameter_count":              3000000 + index*10000,
				"model_size_mb":                11.5 + float64(index%5),
				"deployment_notes":             strings.Repeat("latency and package-size audit detail; ", 10),
				"runtime_profile":              strings.Repeat("batch timing histogram; ", 10),
			},
			HolisticScores: map[string]any{
				"training_diagnostics": map[string]any{
					"trend":             "plateau",
					"loss_gap":          0.12,
					"diagnostic_notes":  strings.Repeat("epoch-level dynamics and overfit notes; ", 12),
					"recommended_pivot": strings.Repeat("avoid shallow architecture repeats; ", 8),
				},
			},
		})
		input.ExistingExperimentSignatures = append(input.ExistingExperimentSignatures,
			fmt.Sprintf("%s|%s|%s|%s", planID, model, mechanism, strings.Repeat("augmentation=long_signature_component;", 8)))
	}
	for index := 0; index < 16; index++ {
		mechanism := mechanisms[index%len(mechanisms)]
		input.StrategyScorecards = append(input.StrategyScorecards, PlannerStrategyScorecard{
			ID:               fmt.Sprintf("scorecard_bulk_%02d", index),
			DatasetID:        "dataset_1",
			SourceDecisionID: fmt.Sprintf("decision_bulk_%02d", index),
			SourcePlanID:     fmt.Sprintf("history_plan_%02d", index),
			StrategyType:     mechanism,
			PlanningMode:     mechanism + "_mode",
			Mechanism:        mechanism,
			Intervention:     strings.Repeat("bounded intervention detail; ", 8),
			EvidenceUsed:     []string{strings.Repeat("historical evidence detail; ", 8)},
			ExpectedEffect:   strings.Repeat("expected effect detail; ", 8),
			DatasetTraits:    map[string]any{"trait_notes": strings.Repeat("dataset trait detail; ", 8)},
			ObjectiveProfile: map[string]any{"objective_notes": strings.Repeat("objective detail; ", 8)},
			ProposedChanges:  map[string]any{"change_notes": strings.Repeat("proposed change detail; ", 8)},
			Outcome:          ExperimentPlanningOutcomeNoImprovement,
			Lesson:           strings.Repeat("strategy lesson detail; ", 10),
			Tags:             []string{mechanism, "bulk_history"},
		})
	}
	for index := 0; index < 12; index++ {
		input.ModelCatalog = append(input.ModelCatalog, SupportedModelSpec{
			Name:                 fmt.Sprintf("catalog_model_%02d", index),
			Family:               fmt.Sprintf("family_%02d", index%6),
			TaskType:             "image_classification",
			ModelKind:            "torchvision_classifier",
			DeploymentTier:       "balanced",
			DefaultImageSize:     224 + (index%3)*32,
			TrainingEnabled:      true,
			ExpectedLatencyClass: "medium",
			RecommendedUse:       strings.Repeat("detailed model eligibility note; ", 8),
		})
	}
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

func containsStringFold(values []string, target string) bool {
	target = strings.ToLower(strings.TrimSpace(target))
	for _, value := range values {
		if strings.ToLower(strings.TrimSpace(value)) == target {
			return true
		}
	}
	return false
}
