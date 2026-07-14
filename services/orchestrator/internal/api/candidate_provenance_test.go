package api

import (
	"reflect"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
)

func TestCandidateProvenanceIsIdenticalBeforeManualOrAutoScheduling(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	input, recommendation := candidateProvenancePlannerFixture(t, server, projectID, plan)

	manualPayload, err := experimentPlannerDecisionPayload(recommendation, invocation, llm.AgentModePropose, input)
	if err != nil {
		t.Fatal(err)
	}
	autoPayload, err := experimentPlannerDecisionPayload(recommendation, invocation, llm.AgentModeAutonomous, input)
	if err != nil {
		t.Fatal(err)
	}
	manualCandidates, err := candidateProvenanceCreatesFromPayload(manualPayload)
	if err != nil {
		t.Fatal(err)
	}
	autoCandidates, err := candidateProvenanceCreatesFromPayload(autoPayload)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(manualCandidates, autoCandidates) {
		t.Fatalf("scheduling mode changed decision-time provenance:\nmanual=%#v\nauto=%#v", manualCandidates, autoCandidates)
	}
	if len(manualCandidates) != 3 {
		t.Fatalf("expected every accepted recommendation candidate, got %d", len(manualCandidates))
	}
	if !manualCandidates[0].Selected || manualCandidates[0].SelectedExperimentIndex == nil || *manualCandidates[0].SelectedExperimentIndex != 0 {
		t.Fatalf("selected candidate did not map to zero-based experiment index: %#v", manualCandidates[0])
	}
	if manualCandidates[0].SelectionTraceReference != "/payload/candidate_selection_trace/0/candidates/0" {
		t.Fatalf("selected candidate does not reference its bounded selection trace entry: %#v", manualCandidates[0])
	}
	if manualCandidates[1].Selected || manualCandidates[1].Rejected || manualCandidates[1].OutcomeStatus != calibration.CandidateOutcomeUnknown {
		t.Fatalf("unselected candidate received a negative outcome: %#v", manualCandidates[1])
	}
	if !manualCandidates[2].Rejected || manualCandidates[2].OutcomeStatus != calibration.CandidateOutcomeUnknown {
		t.Fatalf("ranker-rejected candidate state is not auditable: %#v", manualCandidates[2])
	}
	for _, candidate := range manualCandidates {
		if candidate.RequestedConfigHash == "" || candidate.AcceptedSpecHash == "" {
			t.Fatalf("candidate execution identity is incomplete: %#v", candidate)
		}
	}
	if manualCandidates[0].Forecast.PredictedDelta == recommendation.ExpectedDeltaVsChampion ||
		manualCandidates[0].Forecast.PredictedDelta != recommendation.CandidateHypotheses[0].ExpectedMetricImpact {
		t.Fatalf("candidate forecast was conflated with recommendation delta: candidate=%v recommendation=%v", manualCandidates[0].Forecast.PredictedDelta, recommendation.ExpectedDeltaVsChampion)
	}

	manualDecision, manualRows, err := server.store.CreateAgentDecisionWithCandidateProvenance(
		projectID, plan.ID, decisions.TypeAddExperiments, recommendation.Rationale, manualPayload, manualCandidates,
	)
	if err != nil {
		t.Fatal(err)
	}
	autoDecision, autoRows, err := server.store.CreateAgentDecisionWithCandidateProvenance(
		projectID, plan.ID, decisions.TypeAddExperiments, recommendation.Rationale, autoPayload, autoCandidates,
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(manualRows) != 3 || manualRows[0].InvocationID != invocation.ID || manualRows[0].DecisionID != manualDecision.ID || manualRows[0].PlannerVariantID != invocation.PlannerVariantID {
		t.Fatalf("candidate is not traceable before scheduling: %#v", manualRows)
	}
	if len(autoRows) != len(manualRows) || autoRows[0].DecisionID != autoDecision.ID {
		t.Fatalf("autonomous decision did not persist candidates: %#v", autoRows)
	}
	for index := range manualRows {
		if !reflect.DeepEqual(manualRows[index].CandidateProvenanceCreate, autoRows[index].CandidateProvenanceCreate) {
			t.Fatalf("stored provenance differs by scheduling mode at index %d: manual=%#v auto=%#v", index, manualRows[index], autoRows[index])
		}
	}
	projectPlans, err := server.store.ListProjectExperimentPlans(projectID)
	if err != nil || len(projectPlans) != 1 {
		t.Fatalf("candidate provenance unexpectedly scheduled work: plans=%d err=%v", len(projectPlans), err)
	}
}

func TestExistingDecisionPathRepairsCandidateProvenanceWithoutDuplicates(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	input, recommendation := candidateProvenancePlannerFixture(t, server, projectID, plan)
	payload, err := experimentPlannerDecisionPayload(recommendation, invocation, llm.AgentModePropose, input)
	if err != nil {
		t.Fatal(err)
	}
	decision, err := server.store.CreateAgentDecision(projectID, plan.ID, decisions.TypeAddExperiments, recommendation.Rationale, payload)
	if err != nil {
		t.Fatal(err)
	}
	if err := server.ensurePlannerCandidateProvenance(decision); err != nil {
		t.Fatal(err)
	}
	if err := server.ensurePlannerCandidateProvenance(decision); err != nil {
		t.Fatal(err)
	}
	rows, err := server.store.ListDecisionCandidateProvenance(decision.ID)
	if err != nil || len(rows) != 3 {
		t.Fatalf("existing-decision repair duplicated or lost candidates: rows=%#v err=%v", rows, err)
	}
}

func TestInvalidPreAcceptanceForecastRemainsInvocationAuditOnly(t *testing.T) {
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	input, recommendation := candidateProvenancePlannerFixture(t, server, projectID, plan)
	recommendation.CandidateHypotheses[0].Forecast.Units = "percentage_points"
	if _, err := agents.FinalizePlannerRecommendation(input, recommendation); err == nil || !strings.Contains(err.Error(), "backend-frozen contract") {
		t.Fatalf("invalid frozen forecast was not rejected: %v", err)
	}
	if stored, err := server.store.GetAgentInvocation(invocation.ID); err != nil || stored.ID != invocation.ID {
		t.Fatalf("invocation audit was lost: %#v err=%v", stored, err)
	}
	if decisionsForProject, err := server.store.ListProjectAgentDecisions(projectID); err != nil || len(decisionsForProject) != 0 {
		t.Fatalf("invalid pre-acceptance attempt created a decision: %#v err=%v", decisionsForProject, err)
	}
	if candidates, err := server.store.ListProjectCandidateProvenance(projectID); err != nil || len(candidates) != 0 {
		t.Fatalf("invalid pre-acceptance attempt created candidate provenance: %#v err=%v", candidates, err)
	}
}

func candidateProvenancePlannerFixture(t *testing.T, server *Server, projectID string, plan plans.ExperimentPlan) (agents.ExperimentPlannerInput, agents.ExperimentPlanningRecommendation) {
	t.Helper()
	dataset, err := server.store.GetDataset(plan.DatasetID)
	if err != nil {
		t.Fatal(err)
	}
	recommendation := experimentPlannerAddExperimentsRecommendation()
	firstExperiment := recommendation.ProposedExperiments[0]
	first := agents.CandidateHypothesis{
		Hypothesis: recommendation.Hypothesis, PlanningMode: recommendation.PlanningMode,
		Mechanism: firstExperiment.Mechanism, Intervention: firstExperiment.Intervention,
		ProposedChanges: map[string]any{"class_balancing": firstExperiment.ClassBalancing, "image_size": firstExperiment.ImageSize},
		ExpectedEffect:  firstExperiment.ExpectedEffect, ExpectedMetricImpact: 0.03,
		ExpectedTradeoffs: []string{"higher runtime"}, Risk: "medium", CostLevel: "medium", NoveltyScore: 0.80,
		EvidenceUsed: append([]string(nil), firstExperiment.EvidenceUsed...), ExperimentConfig: firstExperiment,
	}
	secondExperiment := firstExperiment
	secondExperiment.Template = "mobilenet_transfer"
	secondExperiment.Model = "mobilenet_v3_large"
	secondExperiment.Mechanism = "regularization"
	secondExperiment.Intervention = "Increase weight decay and augmentation while keeping a compact family."
	secondExperiment.EvidenceUsed = []string{"validation gap supports stronger regularization"}
	secondExperiment.ExpectedEffect = "Improve generalization without changing the deployment class."
	secondExperiment.WeightDecay = 0.03
	second := agents.CandidateHypothesis{
		Hypothesis: "Stronger regularization should reduce the validation gap.", PlanningMode: "exploit",
		Mechanism: secondExperiment.Mechanism, Intervention: secondExperiment.Intervention,
		ProposedChanges: map[string]any{"weight_decay": 0.03, "augmentation_policy": secondExperiment.AugmentationPolicy},
		ExpectedEffect:  secondExperiment.ExpectedEffect, ExpectedMetricImpact: 0.01,
		ExpectedTradeoffs: []string{"slower convergence"}, Risk: "low", CostLevel: "low", NoveltyScore: 0.35,
		EvidenceUsed: append([]string(nil), secondExperiment.EvidenceUsed...), ExperimentConfig: secondExperiment,
	}
	duplicate := first
	duplicate.Hypothesis = "A duplicate candidate should be rejected by the ranker but retained for audit."
	duplicate.ExpectedMetricImpact = 0.02
	recommendation.CandidateHypotheses = []agents.CandidateHypothesis{first, second, duplicate}

	input := agents.ExperimentPlannerInput{
		Project: projectForPlannerInput(t, server, projectID), Dataset: dataset, SourcePlan: plan,
		CurrentChampion:         &agents.ExperimentChampion{JobID: "job_champion", TargetMetric: "macro_f1", Score: 0.70, ScoreBasis: "macro_f1_score"},
		DeterministicDiagnosis:  agents.PlannerDiagnosis{ClassImbalanceScore: 0.8, MinorityClassFailureScore: 0.7},
		ExecutionCapabilityCard: execution.PlannerCapabilityCard{Task: "image_classification", Runner: "local_simulator"},
		MaxExperiments:          1,
	}
	recommendation, err = agents.FinalizePlannerRecommendation(input, recommendation)
	if err != nil {
		t.Fatal(err)
	}
	return input, recommendation
}

func projectForPlannerInput(t *testing.T, server *Server, projectID string) projects.Project {
	t.Helper()
	value, err := server.store.GetProject(projectID)
	if err != nil {
		t.Fatal(err)
	}
	return value
}
