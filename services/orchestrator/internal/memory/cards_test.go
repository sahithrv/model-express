package memory

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/strategies"
)

func TestNewStrategyScorecardMemoryCardIsCompactAndDeterministic(t *testing.T) {
	scorecard := strategies.StrategyScorecard{
		ID:                "scorecard_1",
		ProjectID:         "project_1",
		DatasetID:         "dataset_1",
		SourceDecisionID:  "decision_1",
		SourcePlanID:      "plan_source",
		FollowUpPlanID:    "plan_followup",
		StrategyType:      "preprocessing_plus_model_family",
		PlanningMode:      "class_imbalance_ablation",
		Mechanism:         "class_imbalance",
		Intervention:      "weighted loss and balanced sampler",
		DiagnosisTriggers: []string{"minority_class_failure", "class_imbalance"},
		EvidenceUsed:      []string{"worst recall class is 0.32"},
		ExpectedEffect:    "Improve minority recall.",
		DatasetTraits:     map[string]any{"class_count": 12, "imbalance": "high"},
		ObjectiveProfile:  map[string]any{"primary_objective": "macro_f1"},
		ProposedChanges:   map[string]any{"models": []any{"mobilenet_v3_large", "efficientnet_b0"}, "raw_prompt": strings.Repeat("x", 200)},
		ExpectedDelta:     0.03,
		ActualDelta:       0.041,
		ConfidenceBefore:  0.7,
		ConfidenceAfter:   0.82,
		CostUSD:           0.18,
		RuntimeSeconds:    1200,
		Outcome:           strategies.OutcomeImprovedChampion,
		Lesson:            "Class balancing improved the champion.",
		Tags:              []string{"class_imbalance", "minority_recall"},
		CreatedAt:         time.Date(2026, 5, 1, 0, 0, 0, 0, time.UTC),
	}

	card := NewStrategyScorecardMemoryCard(scorecard)
	again := NewStrategyScorecardMemoryCard(scorecard)

	if card.CardVersion != MemoryCardVersion {
		t.Fatalf("CardVersion = %q, want %q", card.CardVersion, MemoryCardVersion)
	}
	if !reflect.DeepEqual(card, again) {
		t.Fatalf("card builder should be deterministic\nfirst=%#v\nagain=%#v", card, again)
	}
	if card.SourceTable != SourceStrategyScorecard || card.SourceID != scorecard.ID {
		t.Fatalf("unexpected source: %#v", card)
	}
	if card.PlanID != "plan_source" {
		t.Fatalf("PlanID = %q, want plan_source", card.PlanID)
	}
	if card.Scope != ScopePlan {
		t.Fatalf("Scope = %q, want %q", card.Scope, ScopePlan)
	}
	if card.OutcomeScore <= 0 {
		t.Fatalf("OutcomeScore = %v, want positive", card.OutcomeScore)
	}
	if card.QualityScore != 0.82 {
		t.Fatalf("QualityScore = %v, want 0.82", card.QualityScore)
	}
	if card.SummaryCard["mechanism"] != "class_imbalance" || card.SummaryCard["outcome"] != strategies.OutcomeImprovedChampion {
		t.Fatalf("summary missing mechanism/outcome: %#v", card.SummaryCard)
	}
	if strings.Contains(card.Text, "raw_prompt") || strings.Contains(card.Text, strings.Repeat("x", 20)) {
		t.Fatalf("embedding text leaked raw proposed changes: %q", card.Text)
	}
	if _, ok := card.Metadata["raw_prompt"]; ok {
		t.Fatalf("metadata should not include raw proposed changes: %#v", card.Metadata)
	}
}

func TestNewAgentMemoryCardPlanningOutcome(t *testing.T) {
	record := AgentMemoryRecord{
		ID:        "memory_1",
		ProjectID: "project_1",
		DatasetID: "dataset_1",
		PlanID:    "plan_1",
		AgentName: "experiment_planner",
		Kind:      KindPlanningOutcome,
		Summary:   "Follow-up plan improved the champion.",
		Payload: map[string]any{
			"outcome_status":             strategies.OutcomeMinorImprovement,
			"lesson":                     "Resolution crop helped fine-grained classes but cost increased.",
			"expected_delta_vs_champion": 0.03,
			"actual_delta_vs_champion":   0.018,
			"actual_best_run":            map[string]any{"model": "efficientnet_b0"},
			"proposed_experiments": []any{
				map[string]any{"model": "efficientnet_b0"},
				map[string]any{"model": "mobilenet_v3_large"},
			},
		},
		Tags: []string{"resolution_crop", "fine_grained"},
	}

	card := NewAgentMemoryCard(record)

	if card.CardVersion != MemoryCardVersion {
		t.Fatalf("CardVersion = %q, want %q", card.CardVersion, MemoryCardVersion)
	}
	if card.Scope != ScopePlan {
		t.Fatalf("Scope = %q, want %q", card.Scope, ScopePlan)
	}
	if card.PlanID != "plan_1" {
		t.Fatalf("PlanID = %q, want plan_1", card.PlanID)
	}
	if card.SummaryCard["lesson"] == "" {
		t.Fatalf("summary card missing lesson: %#v", card.SummaryCard)
	}
	if card.SummaryCard["best_model"] != "efficientnet_b0" {
		t.Fatalf("best model = %#v", card.SummaryCard["best_model"])
	}
	models, ok := card.SummaryCard["models"].([]string)
	if !ok || !reflect.DeepEqual(models, []string{"efficientnet_b0", "mobilenet_v3_large"}) {
		t.Fatalf("models = %#v", card.SummaryCard["models"])
	}
	if card.OutcomeScore <= 0 {
		t.Fatalf("OutcomeScore = %v, want positive", card.OutcomeScore)
	}
	if !strings.Contains(card.Text, "Resolution crop helped") {
		t.Fatalf("Text missing lesson: %q", card.Text)
	}
}

func TestNewAgentMemoryCardTrainingEvaluationExcludesRawPayloadDump(t *testing.T) {
	record := AgentMemoryRecord{
		ID:        "memory_2",
		ProjectID: "project_1",
		DatasetID: "dataset_1",
		JobID:     "job_1",
		AgentName: "training_monitor",
		Kind:      KindTrainingEvaluation,
		Summary:   "Previous run plateaued.",
		Payload: map[string]any{
			"rank_score":        0.71,
			"quality_summary":   "Stable but low minority recall.",
			"training_dynamics": "Validation macro-F1 plateaued after epoch 6.",
			"prior_memory_payloads": []any{
				map[string]any{"raw": strings.Repeat("leak", 100)},
			},
			"recommended_action": map[string]any{"type": "RANK_MODELS", "confidence": 0.74},
		},
		Tags: []string{"plateau", "stable"},
	}

	card := NewAgentMemoryCard(record)

	if card.Scope != ScopeJob {
		t.Fatalf("Scope = %q, want %q", card.Scope, ScopeJob)
	}
	if card.QualityScore != 0.71 {
		t.Fatalf("QualityScore = %v, want 0.71", card.QualityScore)
	}
	if strings.Contains(card.Text, "prior_memory_payloads") || strings.Contains(card.Text, "leakleak") {
		t.Fatalf("embedding text leaked raw payload: %q", card.Text)
	}
	if _, ok := card.Metadata["prior_memory_payloads"]; ok {
		t.Fatalf("metadata leaked raw payload: %#v", card.Metadata)
	}
	if card.Metadata["recommended_action_type"] != "RANK_MODELS" {
		t.Fatalf("recommended action type = %#v", card.Metadata["recommended_action_type"])
	}
}

func TestNewAgentMemoryCardPlanningFeedbackAllowlistsRejectedOptions(t *testing.T) {
	record := AgentMemoryRecord{
		ID:        "memory_3",
		ProjectID: "project_1",
		DatasetID: "dataset_1",
		PlanID:    "plan_1",
		AgentName: "experiment_planner",
		Kind:      KindPlanningFeedback,
		Summary:   "Planner recommended a class imbalance ablation.",
		Payload: map[string]any{
			"decision_type": "ADD_EXPERIMENTS",
			"planning_mode": "class_imbalance_ablation",
			"hypothesis":    "Weighted loss may improve minority recall.",
			"proposal_mechanisms": []any{
				map[string]any{
					"mechanism":    "class_imbalance",
					"intervention": "weighted loss",
				},
			},
			"evidence_used":          []any{"minority recall is low"},
			"expected_failure_modes": []any{"majority precision may fall"},
			"changed_variables":      []any{"class_balancing", "sampling_strategy"},
			"rejected_options": []any{
				map[string]any{
					"option": "more epochs only",
					"reason": "prior runs plateaued",
				},
			},
			"strategy_scorecards":        []any{map[string]any{"raw": strings.Repeat("scorecard", 100)}},
			"successful_strategy_memory": []any{map[string]any{"raw": strings.Repeat("memory", 100)}},
			"plan_evaluations":           []any{map[string]any{"raw": strings.Repeat("evaluation", 100)}},
		},
		Tags: []string{"class_imbalance"},
	}

	card, ok := BuildAgentMemoryCard(record)
	if !ok {
		t.Fatalf("BuildAgentMemoryCard() ok = false")
	}
	if card.SummaryCard["mechanism"] != "class_imbalance" {
		t.Fatalf("mechanism = %#v", card.SummaryCard["mechanism"])
	}
	if !strings.Contains(card.Text, "more epochs only - prior runs plateaued") {
		t.Fatalf("Text missing rejected option summary: %q", card.Text)
	}
	for _, forbidden := range []string{"strategy_scorecards", "successful_strategy_memory", "plan_evaluations", "scorecardscorecard", "memorymemory", "evaluationevaluation"} {
		if strings.Contains(card.Text, forbidden) {
			t.Fatalf("embedding text leaked forbidden payload %q: %q", forbidden, card.Text)
		}
		if _, ok := card.Metadata[forbidden]; ok {
			t.Fatalf("metadata leaked forbidden payload key %q: %#v", forbidden, card.Metadata)
		}
	}
}

func TestBuildDatasetProfileCardIsSafeAndDeterministic(t *testing.T) {
	profiledAt := time.Date(2026, 5, 2, 0, 0, 0, 0, time.UTC)
	dataset := datasets.Dataset{
		ID:         "dataset_1",
		ProjectID:  "project_1",
		Name:       "fine-grained birds",
		Status:     datasets.StatusProfiled,
		ProfiledAt: &profiledAt,
		Profile: map[string]any{
			"schema_version": "dataset_profile_v2",
			"task_type":      "image_classification",
			"class_count":    12,
			"total_images":   480,
			"class_distribution": map[string]any{
				"rare":   10,
				"common": 100,
			},
			"image_dimension_stats": map[string]any{
				"width":  map[string]any{"min": 96, "max": 900},
				"height": map[string]any{"min": 128, "max": 700},
			},
			"metadata_summary": map[string]any{
				"bbox_available": true,
				"capabilities": map[string]any{
					"bbox":       true,
					"attributes": true,
				},
				"source_uri": "s3://bucket/raw-manifest.csv",
			},
			"dataset_traits": []any{"fine_grained", "s3://bucket/should-not-leak"},
			"artifacts": []any{
				map[string]any{
					"artifact_type": datasets.ArtifactBoundingBoxes,
					"path":          "annotations/bird001.xml",
				},
			},
			"visual_trait_summary": map[string]any{
				"object_scale":          "small",
				"fine_grained_possible": true,
				"raw_preview":           "https://example.test/preview.jpg",
			},
			"label_quality_audit": map[string]any{
				"audit_type":  "label_quality_audit",
				"status":      "completed",
				"report_only": true,
				"summary": map[string]any{
					"class_count":          12,
					"image_count":          480,
					"minority_class_count": 3,
					"majority_class_count": 1,
					"imbalance_ratio":      10,
				},
				"findings": []any{
					map[string]any{"kind": "asymmetric_confusion", "image_uri": "file:///tmp/leak.jpg"},
				},
			},
		},
	}

	card, ok := BuildDatasetProfileCard(dataset)
	again, againOK := BuildDatasetProfileCard(dataset)

	if !ok || !againOK {
		t.Fatalf("BuildDatasetProfileCard() ok = %t/%t", ok, againOK)
	}
	if !reflect.DeepEqual(card, again) {
		t.Fatalf("dataset profile card should be deterministic\nfirst=%#v\nagain=%#v", card, again)
	}
	if card.SourceTable != SourceDatasetProfile || card.Kind != KindDatasetProfile {
		t.Fatalf("unexpected source/kind: %#v", card)
	}
	if card.Metadata["bbox_available"] != true {
		t.Fatalf("bbox_available = %#v", card.Metadata["bbox_available"])
	}
	if card.SummaryCard["class_count_bucket"] != "medium_classes" || card.SummaryCard["imbalance_bucket"] != "extreme_imbalance" {
		t.Fatalf("unexpected buckets: %#v", card.SummaryCard)
	}
	for _, forbidden := range []string{"s3://", "file://", "https://", "annotations/bird001.xml", "raw-manifest.csv", "preview.jpg"} {
		if strings.Contains(card.Text, forbidden) {
			t.Fatalf("dataset profile card text leaked %q: %q", forbidden, card.Text)
		}
		if strings.Contains(stringFromTestAny(card.Metadata), forbidden) {
			t.Fatalf("dataset profile metadata leaked %q: %#v", forbidden, card.Metadata)
		}
	}
}

func TestBuildDatasetVisualAnalysisCardRequiresAcceptedAndExcludesImageReferences(t *testing.T) {
	analysis := datasets.DatasetVisualAnalysis{
		ID:               "analysis_1",
		ProjectID:        "project_1",
		DatasetID:        "dataset_1",
		DatasetName:      "birds",
		ValidationStatus: datasets.VisualValidationStatusAccepted,
		TriggerReason:    datasets.VisualTriggerInitialProfile,
		SourceJobID:      "job_visual",
		Confidence:       "high",
		TotalImages:      120,
		ImagesAnalyzed:   24,
		CoverageReport: datasets.VisualCoverageReport{
			ImagesAvailable:    120,
			ImagesAnalyzed:     24,
			ClassesTotal:       6,
			ClassesCovered:     6,
			ClassCoverageRatio: 1,
		},
		VisualTraits: []datasets.VisualTrait{{
			Trait:           "object_scale",
			Level:           "small",
			Confidence:      "high",
			Evidence:        []string{"small central object", "s3://bucket/sample.jpg"},
			ExampleImageIDs: []string{"file:///tmp/sample.jpg"},
		}},
		ClassesToWatch: []datasets.ClassWatchItem{{
			ClassName:       "sparrow",
			Reason:          "confused with finch",
			Evidence:        []string{"similar plumage"},
			ExampleImageIDs: []string{"https://example.test/sparrow.png"},
			Confidence:      "medium",
		}},
		Cautions: []datasets.VisualCaution{{
			Operation:       "bbox_crop",
			Reason:          "risk removing context",
			ExampleImageIDs: []string{"s3://bucket/caution.jpg"},
			Severity:        "medium",
			Confidence:      "medium",
		}},
	}

	card, ok := BuildDatasetVisualAnalysisCard(analysis)
	if !ok {
		t.Fatalf("BuildDatasetVisualAnalysisCard() ok = false")
	}
	if card.SourceTable != SourceDatasetVisualAnalysis || card.Kind != KindDatasetVisualAnalysis {
		t.Fatalf("unexpected visual source/kind: %#v", card)
	}
	if card.Metadata["visual_evidence_advisory_only"] != true || card.Metadata["backend_validation_still_required"] != true {
		t.Fatalf("visual safety flags missing: %#v", card.Metadata)
	}
	if card.QualityScore <= 0 {
		t.Fatalf("QualityScore = %v, want positive", card.QualityScore)
	}
	for _, forbidden := range []string{"s3://", "file://", "https://", "sample.jpg", "sparrow.png", "caution.jpg"} {
		if strings.Contains(card.Text, forbidden) {
			t.Fatalf("visual card text leaked %q: %q", forbidden, card.Text)
		}
		if strings.Contains(stringFromTestAny(card.Metadata), forbidden) {
			t.Fatalf("visual card metadata leaked %q: %#v", forbidden, card.Metadata)
		}
	}

	analysis.ValidationStatus = datasets.VisualValidationStatusRejected
	if _, ok := BuildDatasetVisualAnalysisCard(analysis); ok {
		t.Fatalf("rejected visual analysis should not produce an embedding card")
	}
}

func TestBuildDatasetPreprocessingHypothesisCards(t *testing.T) {
	analysis := datasets.DatasetVisualAnalysis{
		ID:               "analysis_1",
		ProjectID:        "project_1",
		DatasetID:        "dataset_1",
		ValidationStatus: datasets.VisualValidationStatusAccepted,
		TriggerReason:    datasets.VisualTriggerDeficiencyReanalysis,
		Confidence:       "medium",
		CoverageReport: datasets.VisualCoverageReport{
			ImagesAnalyzed:     32,
			ClassesTotal:       4,
			ClassesCovered:     4,
			ClassCoverageRatio: 1,
		},
		PreprocessingHypotheses: []datasets.PreprocessingHypothesis{{
			ID:        "bbox crop",
			Mechanism: "bbox_crop_ablation",
			Summary:   "Bounding-box crop may reduce background noise.",
			Evidence:  []string{"objects are small", "s3://bucket/raw-evidence.jpg"},
			SuggestedPreprocessing: &plans.Preprocessing{
				ResizeStrategy: "preserve_aspect",
				Normalization:  "imagenet",
				CropStrategy:   "bbox_crop",
				BBoxMode:       "use_if_available",
			},
			SuggestedImageSizes:         []int{224, 256, 256},
			SuggestedAugmentationPolicy: "randaugment",
			SuggestedAugmentationConfig: &plans.AugmentationPolicyConfig{PolicyType: "randaugment", Magnitude: 7, NumOps: 2},
			ExpectedEffect:              "Improve object focus.",
			Risk:                        "May remove contextual cues.",
			Confidence:                  "high",
			SupportStatus:               "supported",
		}},
	}

	cards := BuildDatasetPreprocessingHypothesisCards(analysis)
	if len(cards) != 1 {
		t.Fatalf("BuildDatasetPreprocessingHypothesisCards() returned %d cards, want 1", len(cards))
	}
	card := cards[0]
	if card.SourceTable != SourceDatasetPreprocessing || card.Kind != KindDatasetPreprocessingHypothesis {
		t.Fatalf("unexpected preprocessing source/kind: %#v", card)
	}
	if card.SourceID != "analysis_1#bbox_crop" {
		t.Fatalf("SourceID = %q", card.SourceID)
	}
	if card.Metadata["mechanism"] != "bbox_crop_ablation" {
		t.Fatalf("mechanism = %#v", card.Metadata["mechanism"])
	}
	if card.Metadata["visual_evidence_advisory_only"] != true || card.Metadata["backend_validation_still_required"] != true {
		t.Fatalf("preprocessing safety flags missing: %#v", card.Metadata)
	}
	if card.OutcomeScore <= 0 {
		t.Fatalf("OutcomeScore = %v, want positive support score", card.OutcomeScore)
	}
	for _, forbidden := range []string{"s3://", "raw-evidence.jpg"} {
		if strings.Contains(card.Text, forbidden) {
			t.Fatalf("preprocessing card text leaked %q: %q", forbidden, card.Text)
		}
		if strings.Contains(stringFromTestAny(card.Metadata), forbidden) {
			t.Fatalf("preprocessing card metadata leaked %q: %#v", forbidden, card.Metadata)
		}
	}
}

func stringFromTestAny(value any) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	return strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(strings.TrimSpace(strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(string(encoded), "\n", " "), "\t", " "), "\r", " ")), "\\", "/"), "%5c", "/"))
}
