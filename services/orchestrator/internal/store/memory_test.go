package store

import (
	"reflect"
	"testing"

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
