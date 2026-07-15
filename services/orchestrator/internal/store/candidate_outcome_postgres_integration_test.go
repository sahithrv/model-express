package store

import (
	"context"
	"os"
	"testing"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/memory"
)

func TestPostgresCandidateOutcomeFinalizationIntegration(t *testing.T) {
	databaseURL := os.Getenv("MODEL_EXPRESS_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("MODEL_EXPRESS_TEST_DATABASE_URL is not set")
	}
	postgres, err := NewPostgresStore(context.Background(), databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer postgres.Close()
	project, err := postgres.CreateProject("postgres candidate outcome", "verify outcome transaction")
	if err != nil {
		t.Fatal(err)
	}
	invocation, err := postgres.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID: project.ID, AgentName: "experiment_planner", PlannerVariantID: memory.LegacyPlannerVariantID,
		ValidationStatus: memory.InvocationValidationValid,
	})
	if err != nil {
		t.Fatal(err)
	}
	creates := testCandidateProvenanceCreates(invocation.ID, invocation.PlannerVariantID)
	decision, _, err := postgres.CreateAgentDecisionWithCandidateProvenance(
		project.ID, "", decisions.TypeAddExperiments, "postgres candidate outcome", nil, creates,
	)
	if err != nil {
		t.Fatal(err)
	}
	planID, experimentID, jobID, attemptID, hash := "plan_postgres", "plan_postgres:experiment-0", "job_postgres", "job_postgres:attempt-1", "sha256:realized"
	actualScore, actualDelta, cost, runtimeSeconds, eligible, reason, terminal := 0.75, 0.05, 0.2, 10.0, true, "matched_finalized", calibration.CandidateTerminalSucceeded
	update := calibration.CandidateOutcomeUpdate{
		CandidateIndex: 0, FollowUpPlanID: planID, ExperimentID: experimentID, JobID: &jobID, AttemptID: &attemptID,
		RealizedEffectiveHash: &hash, OutcomeStatus: calibration.CandidateOutcomeObserved,
		ActualScore: &actualScore, ActualDelta: &actualDelta, TerminalState: &terminal,
		CostUSD: &cost, RuntimeSeconds: &runtimeSeconds, CalibrationEligible: &eligible, EligibilityReason: &reason,
	}
	first, err := postgres.FinalizeCandidateOutcomes(decision.ID, []calibration.CandidateOutcomeUpdate{update})
	if err != nil {
		t.Fatal(err)
	}
	second, err := postgres.FinalizeCandidateOutcomes(decision.ID, []calibration.CandidateOutcomeUpdate{update})
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 2 || len(second) != 2 || first[0].FinalizedAt == nil || second[0].FinalizedAt == nil || !first[0].FinalizedAt.Equal(*second[0].FinalizedAt) {
		t.Fatalf("postgres finalization was not idempotent: first=%#v second=%#v", first, second)
	}
}
