package api

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plannervalidation"
	"model-express/services/orchestrator/internal/plans"
)

func TestShadowStrictDecisionPayloadReturnsRelaxedResultAndTypedVerdict(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", plannervalidation.ModeShadowStrict)
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "false")
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	recommendation := experimentPlannerAddExperimentsRecommendation()
	recommendation.ProposalMechanisms = nil
	recommendation.ProposedExperiments[0].Mechanism = ""
	recommendation.ProposedExperiments[0].Intervention = ""
	recommendation.ProposedExperiments[0].EvidenceUsed = nil
	recommendation.ProposedExperiments[0].ExpectedEffect = ""

	payload, err := experimentPlannerDecisionPayload(recommendation, invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err != nil {
		t.Fatalf("shadow strict blocked relaxed payload: %v", err)
	}
	experiments, decodeErr := plannedExperimentsFromPayload(payload)
	if decodeErr != nil || len(experiments) != 1 {
		t.Fatalf("shadow strict did not return relaxed experiments: %#v", payload["proposed_experiments"])
	}
	value, ok := payload["planner_strict_validation_verdict"].(plannervalidation.Verdict)
	if !ok || !value.WouldBlock || value.Status != plannervalidation.VerdictWouldBlock {
		t.Fatalf("missing typed shadow verdict: %#v", payload["planner_strict_validation_verdict"])
	}
	foundMechanism := false
	for _, finding := range value.Findings {
		if finding.Category == plannervalidation.CategoryMechanismMismatch {
			foundMechanism = true
		}
	}
	if !foundMechanism {
		t.Fatalf("shadow verdict did not classify mechanism mismatch: %#v", value)
	}

	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", plannervalidation.ModeStrict)
	_, err = experimentPlannerDecisionPayload(recommendation, invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	if err == nil || !strings.Contains(err.Error(), "missing mechanism") {
		t.Fatalf("strict mode did not block the same check set: %v", err)
	}
}

func TestProposalNoOpUsesAcceptedSpecIdentity(t *testing.T) {
	recommendation := experimentPlannerAddExperimentsRecommendation()
	input := agents.ExperimentPlannerInput{
		ExecutionCapabilityCard: execution.PlannerCapabilityCard{
			Runner: "modal_torchvision",
		},
	}
	spec, err := buildExecutionSpecV1(recommendation.ProposedExperiments[0], "modal")
	if err != nil {
		t.Fatal(err)
	}
	input.ExecutionEvidence = []agents.ExperimentExecutionEvidence{{
		JobID:            "job_with_same_accepted_spec",
		AcceptedSpecHash: spec.AcceptedSpecHash,
	}}
	if err := validateProposalAcceptedSpecNovelty(recommendation.ProposedExperiments, input); err == nil || !strings.Contains(err.Error(), "proposal-time no-op") || !strings.Contains(err.Error(), spec.AcceptedSpecHash) {
		t.Fatalf("accepted-spec no-op was not detected: %v", err)
	}

	input.ExecutionEvidence = nil
	duplicate := append(recommendation.ProposedExperiments, recommendation.ProposedExperiments[0])
	if err := validateProposalAcceptedSpecNovelty(duplicate, input); err == nil || !strings.Contains(err.Error(), "proposed experiment 0") {
		t.Fatalf("in-batch accepted-spec no-op was not detected: %v", err)
	}
}

func TestShadowStrictTypesInvalidTaskOrModelFailure(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", plannervalidation.ModeShadowStrict)
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	invocation := createExperimentPlannerInvocation(t, server, projectID, plan)
	recommendation := experimentPlannerAddExperimentsRecommendation()
	recommendation.ProposedExperiments[0].Model = "not_a_supported_model"

	_, err := experimentPlannerDecisionPayload(recommendation, invocation, "autonomous", plannerInputForPayload(t, server, projectID))
	var evaluationErr plannervalidation.EvaluationError
	if !errors.As(err, &evaluationErr) || len(evaluationErr.Findings) != 1 || evaluationErr.Findings[0].Category != plannervalidation.CategoryInvalidTaskModel {
		t.Fatalf("invalid model was not typed for shadow metrics: %v %#v", err, evaluationErr)
	}
}

func TestPlannerValidationAttemptOutcomePersistsRetryAttribution(t *testing.T) {
	server, projectID, _ := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	groupID := "planner_attempt_shadow_test"
	first, err := server.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID: projectID, AgentName: agents.ExperimentPlannerAgentName,
		ValidationMode: plannervalidation.ModeStrict, AttemptGroupID: groupID, AttemptIndex: 0,
		ValidationStatus: memory.InvocationValidationInvalid,
	})
	if err != nil {
		t.Fatal(err)
	}
	verdict := plannervalidation.Verdict{
		SchemaVersion: plannervalidation.VerdictSchemaVersionV1,
		Mode:          plannervalidation.ModeStrict,
		Status:        plannervalidation.VerdictWouldBlock,
		WouldBlock:    true,
		Findings: []plannervalidation.Finding{{
			Code: "missing_evidence", Category: plannervalidation.CategoryMissingEvidence, Stage: "recommendation", Message: "missing evidence",
		}},
	}
	if err := server.persistPlannerValidationAttempt(first, verdict, 0, false, true); err != nil {
		t.Fatal(err)
	}

	second, err := server.store.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID: projectID, AgentName: agents.ExperimentPlannerAgentName,
		ValidationMode: plannervalidation.ModeStrict, AttemptGroupID: groupID, AttemptIndex: 1,
		RetryReason: plannerRetryReasonTraceValidation, ValidationStatus: memory.InvocationValidationValid,
	})
	if err != nil {
		t.Fatal(err)
	}
	passedVerdict := plannervalidation.Verdict{
		SchemaVersion: plannervalidation.VerdictSchemaVersionV1,
		Mode:          plannervalidation.ModeStrict,
		Status:        plannervalidation.VerdictPassed,
	}
	if err := server.persistPlannerValidationAttempt(second, passedVerdict, 1, true, false); err != nil {
		t.Fatal(err)
	}

	invocations, err := server.store.ListProjectAgentInvocations(projectID, memory.AgentInvocationFilter{AgentName: agents.ExperimentPlannerAgentName})
	if err != nil {
		t.Fatal(err)
	}
	byIndex := map[int]memory.AgentInvocation{}
	for _, invocation := range invocations {
		if invocation.AttemptGroupID == groupID {
			byIndex[invocation.AttemptIndex] = invocation
		}
	}
	if len(byIndex) != 2 {
		t.Fatalf("retry attempts lost group attribution: %#v", byIndex)
	}
	if outcome := byIndex[0].ValidationOutcome; outcome == nil || outcome.FirstPassStatus != plannervalidation.FirstPassRejected || outcome.EventualStatus != plannervalidation.EventualPending || outcome.RetryOutcome != plannervalidation.RetryScheduled {
		t.Fatalf("first attempt outcome = %#v", outcome)
	}
	if outcome := byIndex[1].ValidationOutcome; outcome == nil || outcome.FirstPassStatus != plannervalidation.FirstPassRejected || outcome.EventualStatus != plannervalidation.EventualAccepted || outcome.RetryOutcome != plannervalidation.RetryAccepted {
		t.Fatalf("accepted retry outcome = %#v", outcome)
	}
	if byIndex[1].RetryReason != plannerRetryReasonTraceValidation {
		t.Fatalf("retry reason lost: %#v", byIndex[1])
	}
}

func TestShadowStrictPlannerRunPersistsWouldBlockWithoutBlockingRelaxedDecision(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", plannervalidation.ModeShadowStrict)
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", execution.ValidationModeShadow)
	server, projectID, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{testExperiment("mobilenet_v3_small", 6)})
	createTerminalTrainingJob(t, server, plan, plan.Experiments[0], "SUCCEEDED", 0.62)
	input, ready, err := server.buildExperimentPlannerInput(projectID, plan.ID)
	if err != nil || !ready {
		t.Fatalf("build planner input ready=%v err=%v", ready, err)
	}

	recommendation := experimentPlannerAddExperimentsRecommendation()
	recommendation.EvidenceUsed = nil
	experiment := recommendation.ProposedExperiments[0]
	recommendation.CandidateHypotheses = []agents.CandidateHypothesis{{
		Hypothesis:           recommendation.Hypothesis,
		PlanningMode:         recommendation.PlanningMode,
		Mechanism:            experiment.Mechanism,
		Intervention:         experiment.Intervention,
		ProposedChanges:      map[string]any{"image_size": experiment.ImageSize, "class_balancing": experiment.ClassBalancing},
		ExpectedEffect:       experiment.ExpectedEffect,
		ExpectedMetricImpact: 0.02,
		ExpectedTradeoffs:    []string{"higher runtime"},
		Risk:                 "medium",
		CostLevel:            "medium",
		NoveltyScore:         0.7,
		EvidenceUsed:         append([]string(nil), experiment.EvidenceUsed...),
		ExperimentConfig:     experiment,
	}}
	raw, err := json.Marshal(recommendation)
	if err != nil {
		t.Fatal(err)
	}
	agent := agents.NewExperimentPlannerAgent(capturingPlannerGenerator{response: string(raw)}, "shadow-test-model")
	result, err := server.runExperimentPlannerWithBackendValidationRetry(context.Background(), agent, input, llm.Config{
		Provider: llm.ProviderOpenAI,
		Model:    "shadow-test-model",
	}, llm.AgentModePropose)
	if err != nil {
		t.Fatalf("shadow strict blocked relaxed planner run: %v", err)
	}
	if len(result.Payload) == 0 || result.Invocation.ID == "" {
		t.Fatalf("shadow strict did not return a decision: %#v", result)
	}
	stored, err := server.store.GetAgentInvocation(result.Invocation.ID)
	if err != nil {
		t.Fatal(err)
	}
	if stored.ValidationMode != plannervalidation.ModeShadowStrict || stored.PlannerVariant == nil || stored.PlannerVariant.ValidationMode != plannervalidation.ModeShadowStrict {
		t.Fatalf("shadow mode was not attributed to exact variant: %#v", stored)
	}
	if stored.StrictValidationVerdict == nil || !stored.StrictValidationVerdict.WouldBlock {
		t.Fatalf("shadow would-block verdict was not persisted: %#v", stored.StrictValidationVerdict)
	}
	foundMissingEvidence := false
	for _, finding := range stored.StrictValidationVerdict.Findings {
		if finding.Category == plannervalidation.CategoryMissingEvidence {
			foundMissingEvidence = true
		}
	}
	if !foundMissingEvidence {
		t.Fatalf("missing-evidence metric was not typed: %#v", stored.StrictValidationVerdict)
	}
	if stored.ValidationOutcome == nil || stored.ValidationOutcome.FirstPassStatus != plannervalidation.FirstPassAccepted || stored.ValidationOutcome.EventualStatus != plannervalidation.EventualAccepted || stored.ValidationOutcome.RetryOutcome != plannervalidation.RetryNotNeeded {
		t.Fatalf("shadow relaxed outcome was not persisted: %#v", stored.ValidationOutcome)
	}
}
