package store

import (
	"reflect"
	"testing"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/strategies"
)

func TestMemoryStoreStrategyScorecardHydratesMechanismFields(t *testing.T) {
	store := NewMemoryStore()
	project, err := store.CreateProject("scorecard project", "maximize macro f1")
	if err != nil {
		t.Fatalf("CreateProject() error = %v", err)
	}

	scorecard, err := store.CreateStrategyScorecard(strategies.StrategyScorecardCreate{
		ProjectID:      project.ID,
		DatasetID:      "dataset_1",
		FollowUpPlanID: "plan_followup",
		ProposedChanges: map[string]any{
			"proposal_mechanisms": []any{
				map[string]any{
					"mechanism":          "resolution_crop",
					"intervention":       "preserve aspect ratio with 256px random resized crops",
					"diagnosis_triggers": []any{"variable_dimensions", "object_scale"},
					"evidence_used":      []any{"image_dimension_stats show high variance"},
					"expected_effect":    "Improve robustness to object scale and crop variation.",
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("CreateStrategyScorecard() error = %v", err)
	}
	if scorecard.Mechanism != "resolution_crop" {
		t.Fatalf("Mechanism = %q, want resolution_crop", scorecard.Mechanism)
	}
	if scorecard.Intervention != "preserve aspect ratio with 256px random resized crops" {
		t.Fatalf("Intervention = %q", scorecard.Intervention)
	}
	if !reflect.DeepEqual(scorecard.DiagnosisTriggers, []string{"variable_dimensions", "object_scale"}) {
		t.Fatalf("DiagnosisTriggers = %#v", scorecard.DiagnosisTriggers)
	}
	if !reflect.DeepEqual(scorecard.EvidenceUsed, []string{"image_dimension_stats show high variance"}) {
		t.Fatalf("EvidenceUsed = %#v", scorecard.EvidenceUsed)
	}
	if scorecard.ExpectedEffect != "Improve robustness to object scale and crop variation." {
		t.Fatalf("ExpectedEffect = %q", scorecard.ExpectedEffect)
	}

	updated, err := store.UpdateStrategyScorecardOutcomeByFollowUpPlan("plan_followup", strategies.StrategyScorecardOutcomeUpdate{
		ActualDelta:     0.04,
		ConfidenceAfter: 0.7,
		Outcome:         strategies.OutcomeMinorImprovement,
		Lesson:          "Resolution and crop changes helped.",
		Tags:            []string{"resolution_crop", "minor_improvement"},
	})
	if err != nil {
		t.Fatalf("UpdateStrategyScorecardOutcomeByFollowUpPlan() error = %v", err)
	}
	if updated.Mechanism != scorecard.Mechanism || updated.Intervention != scorecard.Intervention || updated.ExpectedEffect != scorecard.ExpectedEffect {
		t.Fatalf("update should preserve mechanism fields, got %#v", updated)
	}
}

func TestMemoryStoreDatasetVisualAnalysisCRUD(t *testing.T) {
	store := NewMemoryStore()
	project, err := store.CreateProject("visual project", "maximize macro f1")
	if err != nil {
		t.Fatalf("CreateProject() error = %v", err)
	}
	dataset, err := store.CreateDataset(project.ID, "flowers", "s3://bucket/flowers.zip", "", 1024)
	if err != nil {
		t.Fatalf("CreateDataset() error = %v", err)
	}

	first, err := store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      project.ID,
		DatasetID:      dataset.ID,
		SchemaVersion:  datasets.VisualAnalysisSchemaVersion,
		TriggerReason:  datasets.VisualTriggerInitialProfile,
		TotalImages:    100,
		ImagesAnalyzed: 32,
		CoverageReport: datasets.VisualCoverageReport{ImagesAnalyzed: 32, ClassesTotal: 4, ClassesCovered: 3},
		Confidence:     "medium",
		VisualTraits:   []datasets.VisualTrait{{Trait: "blur", Confidence: "low", Evidence: []string{"soft edges in samples"}}},
	})
	if err != nil {
		t.Fatalf("CreateDatasetVisualAnalysis() error = %v", err)
	}
	if first.AnalysisVersion != 1 || first.ValidationStatus != datasets.VisualValidationStatusAccepted {
		t.Fatalf("first analysis version/status = %d/%s", first.AnalysisVersion, first.ValidationStatus)
	}

	rejected, err := store.RejectDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:        project.ID,
		DatasetID:        dataset.ID,
		SchemaVersion:    datasets.VisualAnalysisSchemaVersion,
		TriggerReason:    datasets.VisualTriggerManual,
		ValidationErrors: []string{"bad image reference"},
	})
	if err != nil {
		t.Fatalf("RejectDatasetVisualAnalysis() error = %v", err)
	}
	if rejected.AnalysisVersion != 2 || rejected.ValidationStatus != datasets.VisualValidationStatusRejected {
		t.Fatalf("rejected analysis version/status = %d/%s", rejected.AnalysisVersion, rejected.ValidationStatus)
	}

	latest, err := store.CreateDatasetVisualAnalysis(datasets.DatasetVisualAnalysis{
		ProjectID:      project.ID,
		DatasetID:      dataset.ID,
		SchemaVersion:  datasets.VisualAnalysisSchemaVersion,
		TriggerReason:  datasets.VisualTriggerDeficiencyReanalysis,
		TotalImages:    100,
		ImagesAnalyzed: 48,
		CoverageReport: datasets.VisualCoverageReport{ImagesAnalyzed: 48, ClassesTotal: 4, ClassesCovered: 4},
		Confidence:     "high",
	})
	if err != nil {
		t.Fatalf("CreateDatasetVisualAnalysis() latest error = %v", err)
	}
	found, err := store.GetLatestAcceptedDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		t.Fatalf("GetLatestAcceptedDatasetVisualAnalysis() error = %v", err)
	}
	if found.ID != latest.ID {
		t.Fatalf("latest accepted ID = %s, want %s", found.ID, latest.ID)
	}
	analyses, err := store.ListDatasetVisualAnalyses(dataset.ID)
	if err != nil {
		t.Fatalf("ListDatasetVisualAnalyses() error = %v", err)
	}
	if len(analyses) != 3 || analyses[0].ID != latest.ID || analyses[2].ID != first.ID {
		t.Fatalf("unexpected visual analysis ordering: %#v", analyses)
	}
}
