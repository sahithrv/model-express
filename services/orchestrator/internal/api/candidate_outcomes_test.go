package api

import (
	"reflect"
	"testing"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

type candidateOutcomeFixture struct {
	server     *Server
	store      *store.MemoryStore
	projectID  string
	sourcePlan plans.ExperimentPlan
	plan       plans.ExperimentPlan
	decisionID string
	invocation memory.AgentInvocation
}

func TestCandidateOutcomeFinalizationMapsRankingOrderAndIsIdempotent(t *testing.T) {
	fixture := newCandidateOutcomeFixture(t)
	before := decisionCandidates(t, fixture)
	executionResult, err := fixture.server.executeStoredExperimentPlan(fixture.plan.ID, executeExperimentPlanRequest{Provider: "local"})
	if err != nil {
		t.Fatal(err)
	}
	if len(executionResult.Jobs) != 2 {
		t.Fatalf("queued jobs=%#v", executionResult.Jobs)
	}
	linked := decisionCandidates(t, fixture)
	for _, row := range linked {
		if !row.Selected {
			continue
		}
		if row.FollowUpPlanID == nil || *row.FollowUpPlanID != fixture.plan.ID || row.ExperimentID == nil || row.JobID == nil {
			t.Fatalf("selected candidate was not linked at scheduling: %#v", row)
		}
		job, _ := fixture.store.GetJob(*row.JobID)
		index, _ := configInt(job.Config, "experiment_index")
		if row.SelectedExperimentIndex == nil || index != *row.SelectedExperimentIndex {
			t.Fatalf("ranking order changed candidate/job mapping: candidate=%#v job=%#v", row, job)
		}
	}

	worker, err := fixture.store.RegisterWorker(fixture.projectID, "candidate worker", "local")
	if err != nil {
		t.Fatal(err)
	}
	for range executionResult.Jobs {
		assigned, err := fixture.store.PollJob(worker.ID, store.JobPollFilter{})
		if err != nil {
			t.Fatal(err)
		}
		index, _ := configInt(assigned.Config, "experiment_index")
		score := 0.70
		if index == 1 {
			score = 0.60
		}
		finalizeMatchedCandidateJob(t, fixture, *assigned, score, 0.20+0.10*float64(index), 10+float64(index))
	}
	complete, err := fixture.server.finalizeCandidateOutcomesForPlan(fixture.plan.ID)
	if err != nil || !complete {
		t.Fatalf("finalize complete=%v err=%v", complete, err)
	}
	after := decisionCandidates(t, fixture)
	byIndex := candidateRowsByIndex(after)
	higher := byIndex[2]
	lower := byIndex[0]
	if higher.SelectedExperimentIndex == nil || *higher.SelectedExperimentIndex != 0 || higher.ActualScore == nil || *higher.ActualScore != 0.70 || higher.ActualDelta == nil || !nearlyEqual(*higher.ActualDelta, 0.20) {
		t.Fatalf("higher-is-better outcome=%#v", higher)
	}
	if lower.SelectedExperimentIndex == nil || *lower.SelectedExperimentIndex != 1 || lower.ActualScore == nil || *lower.ActualScore != 0.60 || lower.ActualDelta == nil || !nearlyEqual(*lower.ActualDelta, -0.20) {
		t.Fatalf("lower-is-better outcome=%#v", lower)
	}
	for _, row := range []calibration.CandidateProvenance{higher, lower} {
		if row.OutcomeStatus != calibration.CandidateOutcomeObserved || row.TerminalState == nil || *row.TerminalState != calibration.CandidateTerminalSucceeded ||
			row.AttemptID == nil || row.RealizedEffectiveHash == nil || row.CalibrationEligible == nil || !*row.CalibrationEligible || row.FinalizedAt == nil {
			t.Fatalf("observed lineage incomplete: %#v", row)
		}
	}
	if unselected := byIndex[1]; unselected.OutcomeStatus != calibration.CandidateOutcomeUnknown || unselected.FollowUpPlanID != nil || unselected.TerminalState != nil {
		t.Fatalf("unselected candidate changed outcome: %#v", unselected)
	}
	for _, index := range []int{0, 2} {
		if after[index].Forecast != before[index].Forecast {
			t.Fatalf("proposal-time forecast mutated at candidate %d: before=%#v after=%#v", index, before[index].Forecast, after[index].Forecast)
		}
	}
	if err := fixture.server.recordExperimentPlannerOutcomeForPlan(fixture.plan); err != nil {
		t.Fatal(err)
	}
	if err := fixture.server.recordExperimentPlannerOutcomeForPlan(fixture.plan); err != nil {
		t.Fatal(err)
	}
	invocation, err := fixture.store.GetAgentInvocation(fixture.invocation.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !nearlyEqual(payloadFloat(invocation.DownstreamOutcome, "total_cost_usd"), 0.50) ||
		!nearlyEqual(payloadFloat(invocation.DownstreamOutcome, "total_runtime_seconds"), 21) ||
		payloadFloat(invocation.DownstreamOutcome, "terminal_run_count") != 2 {
		t.Fatalf("plan aggregates double-counted or omitted outcomes: %#v", invocation.DownstreamOutcome)
	}
	outcomeRecords, err := fixture.store.ListProjectAgentMemoryRecords(fixture.projectID, memory.AgentMemoryFilter{Kind: memory.KindPlanningOutcome})
	if err != nil || len(outcomeRecords) != 1 {
		t.Fatalf("repeated aggregate finalization duplicated outcomes: records=%#v err=%v", outcomeRecords, err)
	}
	complete, err = fixture.server.finalizeCandidateOutcomesForPlan(fixture.plan.ID)
	if err != nil || !complete {
		t.Fatalf("repeat finalize complete=%v err=%v", complete, err)
	}
	repeated := decisionCandidates(t, fixture)
	if !reflect.DeepEqual(after, repeated) {
		t.Fatalf("repeat finalization changed rows:\nfirst=%#v\nsecond=%#v", after, repeated)
	}
}

func TestCandidateOutcomeFinalizationExcludesMismatchedAndSimulatedExecution(t *testing.T) {
	for _, test := range []struct {
		name       string
		simulated  bool
		wantReason string
	}{
		{name: "mismatch", wantReason: execution.EvidenceReasonMismatched},
		{name: "simulated", simulated: true, wantReason: execution.EvidenceReasonSimulated},
	} {
		t.Run(test.name, func(t *testing.T) {
			fixture := newCandidateOutcomeFixture(t)
			result, err := fixture.server.executeStoredExperimentPlan(fixture.plan.ID, executeExperimentPlanRequest{Provider: "local"})
			if err != nil {
				t.Fatal(err)
			}
			worker, _ := fixture.store.RegisterWorker(fixture.projectID, "fidelity worker", "local")
			assigned, err := fixture.store.PollJob(worker.ID, store.JobPollFilter{})
			if err != nil {
				t.Fatal(err)
			}
			record, err := fixture.store.GetJobExecutionRecord(assigned.ID)
			if err != nil {
				t.Fatal(err)
			}
			realized := copyPayloadMap(record.AcceptedSpec.AcceptedSpec)
			if !test.simulated {
				realized["batch_size"] = 1.0
			}
			attemptID := jobConfigString(assigned.Config, "active_attempt_id")
			if _, _, err := fixture.store.AppendRealizationObservation(assigned.ID, execution.RealizationObservationCreate{
				AttemptID: attemptID, Stage: execution.ExecutionObservationFinalized, IdempotencyKey: "candidate-final",
				RealizedConfig: realized, Simulated: test.simulated,
			}); err != nil {
				t.Fatal(err)
			}
			score, runtimeSeconds, cost := 0.90, 12.0, 0.3
			if _, err := fixture.store.UpsertTrainingRunSummary(assigned.ID, runs.TrainingRunSummaryUpdate{
				Status: jobs.StatusSucceeded, BestMacroF1: &score, BestAccuracy: &score,
				RuntimeSeconds: &runtimeSeconds, EstimatedCostUSD: &cost,
			}); err != nil {
				t.Fatal(err)
			}
			if _, err := fixture.store.CompleteJob(assigned.ID, "run"); err != nil {
				t.Fatal(err)
			}
			if _, err := fixture.server.finalizeCandidateOutcomesForPlan(fixture.plan.ID); err != nil {
				t.Fatal(err)
			}
			experimentIndex, _ := configInt(assigned.Config, "experiment_index")
			row := selectedCandidateForExperiment(t, decisionCandidates(t, fixture), experimentIndex)
			if row.OutcomeStatus != calibration.CandidateOutcomeUnobserved || row.ActualScore != nil || row.ActualDelta != nil ||
				row.CalibrationEligible == nil || *row.CalibrationEligible || row.EligibilityReason == nil || *row.EligibilityReason != test.wantReason || row.RealizedEffectiveHash == nil {
				t.Fatalf("ineligible %s outcome=%#v jobs=%#v", test.name, row, result.Jobs)
			}
		})
	}
}

func TestCandidateOutcomeFinalizationDoesNotFabricateFailedOrCancelledMetrics(t *testing.T) {
	fixture := newCandidateOutcomeFixture(t)
	result, err := fixture.server.executeStoredExperimentPlan(fixture.plan.ID, executeExperimentPlanRequest{Provider: "local"})
	if err != nil {
		t.Fatal(err)
	}
	for _, job := range result.Jobs {
		index, _ := configInt(job.Config, "experiment_index")
		if index == 0 {
			if _, err := fixture.store.FailJob(job.ID, "training failed"); err != nil {
				t.Fatal(err)
			}
		} else if _, err := fixture.store.CancelJob(job.ID, "cancelled", map[string]any{"failure_class": "cancelled"}); err != nil {
			t.Fatal(err)
		}
	}
	complete, err := fixture.server.finalizeCandidateOutcomesForPlan(fixture.plan.ID)
	if err != nil || !complete {
		t.Fatalf("complete=%v err=%v", complete, err)
	}
	rows := decisionCandidates(t, fixture)
	for experimentIndex, wantState := range map[int]string{0: calibration.CandidateTerminalFailed, 1: calibration.CandidateTerminalCancelled} {
		row := selectedCandidateForExperiment(t, rows, experimentIndex)
		if row.OutcomeStatus != calibration.CandidateOutcomeUnobserved || row.TerminalState == nil || *row.TerminalState != wantState ||
			row.ActualScore != nil || row.ActualDelta != nil || row.CostUSD != nil || row.RuntimeSeconds != nil {
			t.Fatalf("terminal state %s fabricated outcome: %#v", wantState, row)
		}
	}
}

func TestBudgetSkippedCandidateAllowsPlanAggregateFinalization(t *testing.T) {
	fixture := newCandidateOutcomeFixture(t)
	job := createCandidateJob(t, fixture, 0)
	if _, err := fixture.store.FailJob(job.ID, "failed before summary"); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.CreateExecutionEvent(fixture.projectID, fixture.plan.ID, execution.EventCostBudgetBlocked, "budget skipped experiment", map[string]any{
		"skipped": []map[string]any{{"experiment_index": 1, "reason": "budget_cap_reached"}},
	}); err != nil {
		t.Fatal(err)
	}
	complete, err := fixture.server.finalizeCandidateOutcomesForPlan(fixture.plan.ID)
	if err != nil || !complete {
		t.Fatalf("budget terminal accounting complete=%v err=%v", complete, err)
	}
	skipped := selectedCandidateForExperiment(t, decisionCandidates(t, fixture), 1)
	if skipped.TerminalState == nil || *skipped.TerminalState != calibration.CandidateTerminalSkipped || skipped.JobID != nil || skipped.ActualScore != nil {
		t.Fatalf("budget-skipped candidate=%#v", skipped)
	}
	if err := fixture.server.recordExperimentPlannerOutcomeForPlan(fixture.plan); err != nil {
		t.Fatal(err)
	}
	invocation, err := fixture.store.GetAgentInvocation(fixture.invocation.ID)
	if err != nil {
		t.Fatal(err)
	}
	if payloadString(invocation.DownstreamOutcome, "follow_up_plan_id") != fixture.plan.ID || payloadFloat(invocation.DownstreamOutcome, "terminal_run_count") != 2 {
		t.Fatalf("plan aggregate did not include skipped experiment: %#v", invocation.DownstreamOutcome)
	}
	if err := fixture.server.recordExperimentPlannerOutcomeForPlan(fixture.plan); err != nil {
		t.Fatal(err)
	}
	records, err := fixture.store.ListProjectAgentMemoryRecords(fixture.projectID, memory.AgentMemoryFilter{Kind: memory.KindPlanningOutcome})
	if err != nil || len(records) != 1 {
		t.Fatalf("aggregate finalization was not idempotent: records=%#v err=%v", records, err)
	}
}

func TestAmbiguousHistoricalCandidateJobMappingIsNotBackfilled(t *testing.T) {
	fixture := newCandidateOutcomeFixture(t)
	first := createCandidateJob(t, fixture, 0)
	second := createCandidateJob(t, fixture, 0)
	if first.ID == second.ID {
		t.Fatal("fixture did not create distinct jobs")
	}
	if _, err := fixture.server.finalizeCandidateOutcomesForPlan(fixture.plan.ID); err != nil {
		t.Fatal(err)
	}
	row := selectedCandidateForExperiment(t, decisionCandidates(t, fixture), 0)
	if row.FollowUpPlanID == nil || row.ExperimentID == nil || row.JobID != nil || row.AttemptID != nil || row.OutcomeStatus != calibration.CandidateOutcomeUnknown {
		t.Fatalf("ambiguous mapping was inferred: %#v", row)
	}
}

func newCandidateOutcomeFixture(t *testing.T) candidateOutcomeFixture {
	t.Helper()
	server, projectID, sourcePlan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	memoryStore := server.store.(*store.MemoryStore)
	invocation := createExperimentPlannerInvocation(t, server, projectID, sourcePlan)
	experiments := []plans.PlannedExperiment{testExperiment("efficientnet_b0", 6), testExperiment("resnet18", 5)}
	specs := make([]execution.ExecutionSpecV1, len(experiments))
	for index, experiment := range experiments {
		spec, err := buildExecutionSpecV1(experiment, "local")
		if err != nil {
			t.Fatal(err)
		}
		specs[index] = spec
	}
	selectedExperimentOne := 1
	selectedExperimentZero := 0
	higherForecast := calibration.CandidateForecastContract{
		ForecastTarget: "macro_f1", MetricDirection: calibration.MetricDirectionHigherIsBetter,
		ScoreBasis: "macro_f1_score", ScoreVersion: calibration.CandidateForecastScoreVersionV1,
		BaselineScore: 0.50, PredictedDelta: 0.10, PredictionSource: calibration.CandidatePredictionSource,
		Units: calibration.CandidateForecastUnits, ValidRange: calibration.CandidateForecastRange{Min: 0, Max: 1},
	}
	lowerForecast := calibration.CandidateForecastContract{
		ForecastTarget: "macro_f1", MetricDirection: calibration.MetricDirectionLowerIsBetter,
		ScoreBasis: "macro_f1_score", ScoreVersion: calibration.CandidateForecastScoreVersionV1,
		BaselineScore: 0.80, PredictedDelta: -0.10, PredictionSource: calibration.CandidatePredictionSource,
		Units: calibration.CandidateForecastUnits, ValidRange: calibration.CandidateForecastRange{Min: -1, Max: 0},
	}
	candidates := []calibration.CandidateProvenanceCreate{
		{
			InvocationID: invocation.ID, PlannerVariantID: invocation.PlannerVariantID, CandidateIndex: 0,
			RolloutCohortID: invocation.RolloutCohortID, RolloutPolicyID: invocation.RolloutPolicyID,
			RequestedConfigHash: specs[1].RequestedConfigHash, AcceptedSpecHash: specs[1].AcceptedSpecHash,
			Task: specs[1].Task, Mechanism: "lower metric candidate", Forecast: lowerForecast, BaseScore: 0.9,
			SelectionTraceReference: "/trace/0", Selected: true, SelectionState: calibration.CandidateSelectionSelected,
			SelectedExperimentIndex: &selectedExperimentOne, OutcomeStatus: calibration.CandidateOutcomeUnknown,
		},
		{
			InvocationID: invocation.ID, PlannerVariantID: invocation.PlannerVariantID, CandidateIndex: 1,
			RolloutCohortID: invocation.RolloutCohortID, RolloutPolicyID: invocation.RolloutPolicyID,
			RequestedConfigHash: specs[0].RequestedConfigHash, AcceptedSpecHash: specs[0].AcceptedSpecHash,
			Task: specs[0].Task, Mechanism: "unselected candidate", Forecast: higherForecast, BaseScore: 0.2,
			SelectionTraceReference: "/trace/1", SelectionState: calibration.CandidateSelectionUnselected,
			OutcomeStatus: calibration.CandidateOutcomeUnknown,
		},
		{
			InvocationID: invocation.ID, PlannerVariantID: invocation.PlannerVariantID, CandidateIndex: 2,
			RolloutCohortID: invocation.RolloutCohortID, RolloutPolicyID: invocation.RolloutPolicyID,
			RequestedConfigHash: specs[0].RequestedConfigHash, AcceptedSpecHash: specs[0].AcceptedSpecHash,
			Task: specs[0].Task, Mechanism: "higher metric candidate", Forecast: higherForecast, BaseScore: 0.8,
			SelectionTraceReference: "/trace/2", Selected: true, SelectionState: calibration.CandidateSelectionSelected,
			SelectedExperimentIndex: &selectedExperimentZero, OutcomeStatus: calibration.CandidateOutcomeUnknown,
		},
	}
	payload := map[string]any{
		"decision_source":            llmExperimentPlannerDecisionSource,
		"invocation_id":              invocation.ID,
		"proposed_experiments":       experiments,
		"expected_delta_vs_champion": 0.10,
	}
	decision, _, err := memoryStore.CreateAgentDecisionWithCandidateProvenance(
		projectID, sourcePlan.ID, decisions.TypeAddExperiments, "candidate outcome fixture", payload, candidates,
	)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := memoryStore.CreateExperimentPlan(projectID, sourcePlan.DatasetID, sourcePlan.TargetMetric, 1, 10, experiments, nil, decision.ID)
	if err != nil {
		t.Fatal(err)
	}
	return candidateOutcomeFixture{
		server: server, store: memoryStore, projectID: projectID, sourcePlan: sourcePlan,
		plan: plan, decisionID: decision.ID, invocation: invocation,
	}
}

func finalizeMatchedCandidateJob(t *testing.T, fixture candidateOutcomeFixture, job jobs.ExperimentJob, score, cost, runtimeSeconds float64) {
	t.Helper()
	record, err := fixture.store.GetJobExecutionRecord(job.ID)
	if err != nil {
		t.Fatal(err)
	}
	attemptID := jobConfigString(job.Config, "active_attempt_id")
	if _, _, err := fixture.store.AppendRealizationObservation(job.ID, execution.RealizationObservationCreate{
		AttemptID: attemptID, Stage: execution.ExecutionObservationFinalized, IdempotencyKey: "matched-final",
		RealizedConfig: record.AcceptedSpec.AcceptedSpec,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
		Status: jobs.StatusSucceeded, BestMacroF1: &score, BestAccuracy: &score,
		RuntimeSeconds: &runtimeSeconds, EstimatedCostUSD: &cost,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.store.CompleteJob(job.ID, "run-"+job.ID); err != nil {
		t.Fatal(err)
	}
}

func createCandidateJob(t *testing.T, fixture candidateOutcomeFixture, experimentIndex int) jobs.ExperimentJob {
	t.Helper()
	spec, err := buildExecutionSpecV1(fixture.plan.Experiments[experimentIndex], "local")
	if err != nil {
		t.Fatal(err)
	}
	payload, err := spec.Payload()
	if err != nil {
		t.Fatal(err)
	}
	job, err := fixture.store.CreateJob(fixture.projectID, jobs.TemplateTrainExperiment, map[string]any{
		"plan_id": fixture.plan.ID, "dataset_id": fixture.plan.DatasetID, "experiment_index": experimentIndex,
		"target_metric": fixture.plan.TargetMetric, execution.ExecutionSpecConfigKey: payload,
	})
	if err != nil {
		t.Fatal(err)
	}
	return job
}

func decisionCandidates(t *testing.T, fixture candidateOutcomeFixture) []calibration.CandidateProvenance {
	t.Helper()
	rows, err := fixture.store.ListDecisionCandidateProvenance(fixture.decisionID)
	if err != nil {
		t.Fatal(err)
	}
	return rows
}

func candidateRowsByIndex(rows []calibration.CandidateProvenance) map[int]calibration.CandidateProvenance {
	out := map[int]calibration.CandidateProvenance{}
	for _, row := range rows {
		out[row.CandidateIndex] = row
	}
	return out
}

func selectedCandidateForExperiment(t *testing.T, rows []calibration.CandidateProvenance, experimentIndex int) calibration.CandidateProvenance {
	t.Helper()
	for _, row := range rows {
		if row.SelectedExperimentIndex != nil && *row.SelectedExperimentIndex == experimentIndex {
			return row
		}
	}
	t.Fatalf("selected candidate for experiment %d not found: %#v", experimentIndex, rows)
	return calibration.CandidateProvenance{}
}

func nearlyEqual(left, right float64) bool {
	delta := left - right
	return delta > -1e-9 && delta < 1e-9
}
