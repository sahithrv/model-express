package store

import (
	"database/sql"
	"errors"
	"reflect"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/memory"
)

func TestMemoryCandidateProvenanceIsAtomicIdempotentAndOutcomeNeutral(t *testing.T) {
	store := NewMemoryStore()
	project, err := store.CreateProject("candidate provenance", "calibrate planner forecasts")
	if err != nil {
		t.Fatal(err)
	}
	invocation, err := store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID: project.ID, AgentName: "experiment_planner", PlannerVariantID: memory.LegacyPlannerVariantID,
		ValidationStatus: memory.InvocationValidationValid,
	})
	if err != nil {
		t.Fatal(err)
	}
	candidates := testCandidateProvenanceCreates(invocation.ID, invocation.PlannerVariantID)
	decision, rows, err := store.CreateAgentDecisionWithCandidateProvenance(
		project.ID, "plan_1", decisions.TypeAddExperiments, "accepted planner batch", map[string]any{"preserved": true}, candidates,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0].CandidateIndex != 0 || rows[1].CandidateIndex != 1 {
		t.Fatalf("candidate indexes are not deterministic: %#v", rows)
	}
	if rows[0].SelectedExperimentIndex == nil || *rows[0].SelectedExperimentIndex != 0 {
		t.Fatalf("selected experiment index was not preserved: %#v", rows[0])
	}
	if rows[1].SelectedExperimentIndex != nil || rows[1].SelectionState != calibration.CandidateSelectionUnselected || rows[1].OutcomeStatus != calibration.CandidateOutcomeUnknown {
		t.Fatalf("unselected candidate was incorrectly treated as a negative outcome: %#v", rows[1])
	}
	for _, row := range rows {
		if row.RequestedConfigHash == "" || row.AcceptedSpecHash == "" || row.RealizedEffectiveHash != nil || row.FollowUpPlanID != nil || row.ExperimentID != nil || row.JobID != nil {
			t.Fatalf("decision-time execution lineage is incorrect: %#v", row)
		}
	}
	again, err := store.EnsureCandidateProvenance(decision, candidates)
	if err != nil {
		t.Fatal(err)
	}
	if len(again) != 2 || again[0].ID != rows[0].ID || again[1].ID != rows[1].ID {
		t.Fatalf("idempotent ensure duplicated candidate rows: before=%#v after=%#v", rows, again)
	}
	conflicting := testCandidateProvenanceCreates(invocation.ID, invocation.PlannerVariantID)
	conflicting[0].BaseScore = 0.12
	if _, err := store.EnsureCandidateProvenance(decision, conflicting); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("immutable provenance conflict was not rejected: %v", err)
	}
	invalid := testCandidateProvenanceCreates(invocation.ID, invocation.PlannerVariantID)
	invalid[0].AcceptedSpecHash = ""
	if _, _, err := store.CreateAgentDecisionWithCandidateProvenance(
		project.ID, "plan_2", decisions.TypeAddExperiments, "must roll back", nil, invalid,
	); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("invalid candidate provenance did not fail atomically: %v", err)
	}
	decisionsForProject, err := store.ListProjectAgentDecisions(project.ID)
	if err != nil || len(decisionsForProject) != 1 || decisionsForProject[0].Payload["preserved"] != true {
		t.Fatalf("existing decision payload changed: %#v err=%v", decisionsForProject, err)
	}
}

func TestMemoryCandidateProvenanceRepairsDecisionCreatedBeforeCandidateInsert(t *testing.T) {
	store := NewMemoryStore()
	project, _ := store.CreateProject("repair provenance", "repair partial persistence")
	invocation, err := store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID: project.ID, AgentName: "experiment_planner", PlannerVariantID: memory.LegacyPlannerVariantID,
		ValidationStatus: memory.InvocationValidationValid,
	})
	if err != nil {
		t.Fatal(err)
	}
	decision, err := store.CreateAgentDecision(project.ID, "plan_1", decisions.TypeAddExperiments, "decision committed first", map[string]any{"accepted": true})
	if err != nil {
		t.Fatal(err)
	}
	candidates := testCandidateProvenanceCreates(invocation.ID, invocation.PlannerVariantID)
	first, err := store.EnsureCandidateProvenance(decision, candidates)
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.EnsureCandidateProvenance(decision, candidates)
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 2 || !reflect.DeepEqual(first, second) {
		t.Fatalf("repair was not deterministic and idempotent: first=%#v second=%#v", first, second)
	}
}

func TestMemoryCandidateOutcomeFinalizationIsAtomicAndKeepsProposalProvenanceImmutable(t *testing.T) {
	store := NewMemoryStore()
	project, _ := store.CreateProject("candidate outcomes", "finalize")
	invocation, err := store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID: project.ID, AgentName: "experiment_planner", PlannerVariantID: memory.LegacyPlannerVariantID,
		ValidationStatus: memory.InvocationValidationValid,
	})
	if err != nil {
		t.Fatal(err)
	}
	creates := testCandidateProvenanceCreates(invocation.ID, invocation.PlannerVariantID)
	decision, before, err := store.CreateAgentDecisionWithCandidateProvenance(
		project.ID, "plan_1", decisions.TypeAddExperiments, "accepted", nil, creates,
	)
	if err != nil {
		t.Fatal(err)
	}
	planID, experimentID, jobID, attemptID, hash := "plan_2", "plan_2:experiment-0", "job_1", "job_1:attempt-1", "sha256:realized"
	actualScore, actualDelta, cost, runtimeSeconds, eligible, reason, terminal := 0.75, 0.05, 0.2, 10.0, true, "matched_finalized", calibration.CandidateTerminalSucceeded
	update := calibration.CandidateOutcomeUpdate{
		CandidateIndex: 0, FollowUpPlanID: planID, ExperimentID: experimentID, JobID: &jobID, AttemptID: &attemptID,
		RealizedEffectiveHash: &hash, OutcomeStatus: calibration.CandidateOutcomeObserved,
		ActualScore: &actualScore, ActualDelta: &actualDelta, TerminalState: &terminal,
		CostUSD: &cost, RuntimeSeconds: &runtimeSeconds, CalibrationEligible: &eligible, EligibilityReason: &reason,
	}
	first, err := store.FinalizeCandidateOutcomes(decision.ID, []calibration.CandidateOutcomeUpdate{update})
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.FinalizeCandidateOutcomes(decision.ID, []calibration.CandidateOutcomeUpdate{update})
	if err != nil || !reflect.DeepEqual(first, second) {
		t.Fatalf("idempotent finalization changed rows: first=%#v second=%#v err=%v", first, second, err)
	}
	if first[0].Forecast != before[0].Forecast || first[0].RequestedConfigHash != before[0].RequestedConfigHash || first[0].AcceptedSpecHash != before[0].AcceptedSpecHash {
		t.Fatalf("proposal-time provenance changed: before=%#v after=%#v", before[0], first[0])
	}
	if repaired, err := store.EnsureCandidateProvenance(decision, creates); err != nil || !reflect.DeepEqual(repaired, first) {
		t.Fatalf("post-finalization ensure was not idempotent: rows=%#v err=%v", repaired, err)
	}
	conflicting := update
	otherJob := "job_2"
	conflicting.JobID = &otherJob
	if _, err := store.FinalizeCandidateOutcomes(decision.ID, []calibration.CandidateOutcomeUpdate{conflicting}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("conflicting immutable lineage was accepted: %v", err)
	}
	afterConflict, _ := store.ListDecisionCandidateProvenance(decision.ID)
	if !reflect.DeepEqual(afterConflict, first) {
		t.Fatalf("failed finalization was not atomic: before=%#v after=%#v", first, afterConflict)
	}
	unselected := update
	unselected.CandidateIndex = 1
	if _, err := store.FinalizeCandidateOutcomes(decision.ID, []calibration.CandidateOutcomeUpdate{unselected}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("unselected candidate received an outcome: %v", err)
	}
}

func TestScanCandidateProvenancePreservesForecastSelectionAndNullableLineage(t *testing.T) {
	createdAt := time.Date(2026, time.July, 14, 12, 0, 0, 0, time.UTC)
	row := fakeAgentInvocationRow{values: []any{
		"candidate_provenance_1", "project_1", "agent_invocation_1", "decision_1", "planner_variant_1", 2,
		"sha256:requested", "sha256:accepted", "image_classification", "regularization",
		"macro_f1", calibration.MetricDirectionHigherIsBetter, "macro_f1_score", calibration.CandidateForecastScoreVersionV1, "job_champion",
		0.70, 0.02, calibration.CandidatePredictionSource, calibration.CandidateForecastUnits, 0.0, 1.0,
		0.77, "/payload/candidate_selection_trace/0/candidates/0", true, false, calibration.CandidateSelectionSelected,
		sql.NullInt64{Int64: 0, Valid: true}, calibration.CandidateOutcomeUnknown, []byte(`["selected"]`),
		sql.NullString{}, sql.NullString{}, sql.NullString{}, sql.NullString{}, sql.NullString{},
		sql.NullFloat64{}, sql.NullFloat64{}, sql.NullString{}, sql.NullFloat64{}, sql.NullFloat64{},
		sql.NullBool{}, sql.NullString{}, sql.NullTime{}, createdAt,
	}}
	candidate, err := scanCandidateProvenance(row)
	if err != nil {
		t.Fatal(err)
	}
	if candidate.CandidateIndex != 2 || candidate.SelectedExperimentIndex == nil || *candidate.SelectedExperimentIndex != 0 || candidate.Forecast.PredictedDelta != 0.02 {
		t.Fatalf("candidate provenance scan lost forecast or selection: %#v", candidate)
	}
	if candidate.FollowUpPlanID != nil || candidate.ExperimentID != nil || candidate.JobID != nil || candidate.AttemptID != nil || candidate.RealizedEffectiveHash != nil || candidate.FinalizedAt != nil {
		t.Fatalf("decision-time nullable lineage did not remain null: %#v", candidate)
	}
}

func testCandidateProvenanceCreates(invocationID string, variantID string) []calibration.CandidateProvenanceCreate {
	selectedIndex := 0
	forecast := calibration.CandidateForecastContract{
		ForecastTarget: "macro_f1", MetricDirection: calibration.MetricDirectionHigherIsBetter,
		ScoreBasis: "macro_f1_score", ScoreVersion: calibration.CandidateForecastScoreVersionV1,
		BaselineJobID: "job_champion", BaselineScore: 0.70, PredictedDelta: 0.02,
		PredictionSource: calibration.CandidatePredictionSource, Units: calibration.CandidateForecastUnits,
		ValidRange: calibration.CandidateForecastRange{Min: 0, Max: 1},
	}
	return []calibration.CandidateProvenanceCreate{
		{
			InvocationID: invocationID, PlannerVariantID: variantID, CandidateIndex: 0,
			RequestedConfigHash: "sha256:requested-0", AcceptedSpecHash: "sha256:accepted-0",
			Task: "image_classification", Mechanism: "class_imbalance", Forecast: forecast, BaseScore: 0.81,
			SelectionTraceReference: "/payload/candidate_selection_trace?candidate_index=0",
			Selected:                true, SelectionState: calibration.CandidateSelectionSelected, SelectedExperimentIndex: &selectedIndex,
			OutcomeStatus: calibration.CandidateOutcomeUnknown, Reasons: []string{"selected by deterministic backend ranking"},
		},
		{
			InvocationID: invocationID, PlannerVariantID: variantID, CandidateIndex: 1,
			RequestedConfigHash: "sha256:requested-1", AcceptedSpecHash: "sha256:accepted-1",
			Task: "image_classification", Mechanism: "regularization", Forecast: forecast, BaseScore: 0.61,
			SelectionTraceReference: "/payload/candidate_selection_trace?candidate_index=1",
			SelectionState:          calibration.CandidateSelectionUnselected,
			OutcomeStatus:           calibration.CandidateOutcomeUnknown, Reasons: []string{"eligible but not selected"},
		},
	}
}
