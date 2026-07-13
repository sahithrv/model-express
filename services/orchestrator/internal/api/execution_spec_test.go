package api

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/store"
)

func TestExecutePlanPreservesExplicitFalseAndZeroInLegacyAndCanonicalPayloads(t *testing.T) {
	var experiment plans.PlannedExperiment
	if err := json.Unmarshal([]byte(`{
		"template":"resnet_transfer",
		"model":"resnet18",
		"mechanism":"regularization",
		"epochs":8,
		"batch_size":16,
		"learning_rate":0.001,
		"reason":"presence-safe execution",
		"optimizer":"sgd",
		"optimizer_momentum":0,
		"pretrained":false,
		"freeze_backbone":false,
		"augmentation_policy":"mixup",
		"augmentation_policy_config":{
			"policy_type":"mixup",
			"probability":0,
			"alpha":0
		}
	}`), &experiment); err != nil {
		t.Fatalf("decode presence-safe experiment: %v", err)
	}
	server, _, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{experiment})
	if plan.ExecutionSpecStatus != execution.ExecutionSpecStatusVersioned {
		t.Fatalf("new plan status = %q", plan.ExecutionSpecStatus)
	}
	if plan.CapabilityVersion == "" {
		t.Fatal("new plan is missing capability version")
	}

	result, err := server.executeStoredExperimentPlan(
		plan.ID,
		executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"},
	)
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	if len(result.Jobs) != 1 {
		t.Fatalf("expected one job, got %d", len(result.Jobs))
	}
	job := result.Jobs[0]
	if job.ExecutionSpecStatus != execution.ExecutionSpecStatusVersioned {
		t.Fatalf("new job status = %q", job.ExecutionSpecStatus)
	}
	assertConfigValue(t, job.Config, "pretrained", false)
	assertConfigValue(t, job.Config, "freeze_backbone", false)
	assertConfigValue(t, job.Config, "optimizer_momentum", float64(0))

	encodedConfig, err := json.Marshal(job.Config)
	if err != nil {
		t.Fatalf("marshal job config: %v", err)
	}
	var workerPayload map[string]any
	if err := json.Unmarshal(encodedConfig, &workerPayload); err != nil {
		t.Fatalf("decode worker job config: %v", err)
	}
	assertConfigValue(t, workerPayload, "pretrained", false)
	assertConfigValue(t, workerPayload, "freeze_backbone", false)
	assertConfigValue(t, workerPayload, "optimizer_momentum", float64(0))
	policy := workerPayload["augmentation_policy_config"].(map[string]any)
	assertConfigValue(t, policy, "probability", float64(0))
	assertConfigValue(t, policy, "alpha", float64(0))

	spec := workerPayload[execution.ExecutionSpecConfigKey].(map[string]any)
	if spec["schema_version"] != execution.ExecutionSpecSchemaVersionV1 {
		t.Fatalf("unexpected execution spec: %#v", spec)
	}
	if spec["task"] != "image_classification" || spec["runner"] != "modal_torchvision" {
		t.Fatalf("unexpected execution target: %#v", spec)
	}
	if spec["requested_config_hash"] == "" || spec["accepted_spec_hash"] == "" {
		t.Fatalf("execution spec hashes are missing: %#v", spec)
	}
	requested := spec["requested_config"].(map[string]any)
	assertConfigValue(t, requested, "pretrained", false)
	assertConfigValue(t, requested, "freeze_backbone", false)
	assertConfigValue(t, requested, "optimizer_momentum", float64(0))
	accepted := spec["accepted_config"].(map[string]any)
	assertConfigValue(t, accepted, "pretrained", false)
	assertConfigValue(t, accepted, "freeze_backbone", false)
	assertConfigValue(t, accepted, "optimizer_momentum", float64(0))
	acceptedPolicy := accepted["augmentation_policy_config"].(map[string]any)
	assertConfigValue(t, acceptedPolicy, "probability", float64(0))
	assertConfigValue(t, acceptedPolicy, "alpha", float64(0))
}

func TestExecutionValidationShadowReportsWithoutBlocking(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", "shadow")
	experiment := testExperiment("resnet18", 8)
	experiment.ResolutionStrategy = "low_latency"
	server, _, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{experiment})

	result, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if err != nil {
		t.Fatalf("shadow execution was blocked: %v", err)
	}
	if len(result.Jobs) != 1 || len(result.ValidationReports) != 1 {
		t.Fatalf("unexpected shadow result: jobs=%d reports=%d", len(result.Jobs), len(result.ValidationReports))
	}
	report := result.ValidationReports[0]
	if report.Mode != execution.ValidationModeShadow || !report.WouldBlock {
		t.Fatalf("unexpected shadow report: %#v", report)
	}
	if _, ok := result.Jobs[0].Config[execution.ExecutionValidationConfigKey]; !ok {
		t.Fatalf("job config did not retain typed validation report: %#v", result.Jobs[0].Config)
	}
}

func TestExecutionValidationDefaultsToEnforceBeforeJobCreation(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", "")
	experiment := testExperiment("resnet18", 8)
	experiment.ResolutionStrategy = "low_latency"
	server, _, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{experiment})

	_, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if !errors.Is(err, store.ErrInvalidRequest) {
		t.Fatalf("default enforcement did not reject unsupported proposal: %v", err)
	}
	projectJobs, _ := server.store.ListProjectJobs(plan.ProjectID)
	for _, job := range projectJobs {
		if configString(job.Config, "plan_id") == plan.ID {
			t.Fatalf("default enforcement scheduled GPU work for blocked proposal: %#v", job)
		}
	}
}

func TestExecutionValidationEnforceBlocksBeforeJobCreation(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", "enforce")
	experiment := testExperiment("resnet18", 8)
	experiment.ResolutionStrategy = "low_latency"
	server, _, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{experiment})

	_, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if !errors.Is(err, store.ErrInvalidRequest) {
		t.Fatalf("expected enforcement error, got %v", err)
	}
	projectJobs, listErr := server.store.ListProjectJobs(plan.ProjectID)
	if listErr != nil {
		t.Fatalf("list project jobs: %v", listErr)
	}
	for _, job := range projectJobs {
		if configString(job.Config, "plan_id") == plan.ID {
			t.Fatalf("enforcement created a blocked plan job: %#v", job)
		}
	}
	events, listErr := server.store.ListProjectExecutionEvents(plan.ProjectID, 20)
	if listErr != nil {
		t.Fatalf("list validation events: %v", listErr)
	}
	found := false
	for _, event := range events {
		if event.EventType == execution.EventExecutionValidationReported && event.PlanID == plan.ID {
			found = true
		}
	}
	if !found {
		t.Fatal("enforcement did not persist the typed blocked report")
	}
}

func TestExecutionValidationEnforcePreflightsWholePlanBeforeCreatingJobs(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", "enforce")
	valid := testExperiment("resnet18", 8)
	blocked := testExperiment("efficientnet_b0", 8)
	blocked.ResolutionStrategy = "low_latency"
	server, _, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{valid, blocked})

	_, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if !errors.Is(err, store.ErrInvalidRequest) {
		t.Fatalf("expected whole-plan enforcement error, got %v", err)
	}
	projectJobs, listErr := server.store.ListProjectJobs(plan.ProjectID)
	if listErr != nil {
		t.Fatalf("list project jobs: %v", listErr)
	}
	for _, job := range projectJobs {
		if configString(job.Config, "plan_id") == plan.ID {
			t.Fatalf("whole-plan preflight created an earlier valid job before rejecting a later experiment: %#v", job)
		}
	}
}

func TestAcceptedHashDuplicateIsSkippedBeforeScheduling(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", "shadow")
	baseline := testExperiment("resnet18", 8)
	equivalent := testExperiment("resnet18", 8)
	equivalent.ResolutionStrategy = "low_latency"
	server, _, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{baseline, equivalent})

	result, err := server.executeStoredExperimentPlan(plan.ID, executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"})
	if err != nil {
		t.Fatalf("execute equivalent experiments: %v", err)
	}
	if len(result.Jobs) != 1 || len(result.ValidationReports) != 2 {
		t.Fatalf("accepted duplicate was not skipped: jobs=%d reports=%d", len(result.Jobs), len(result.ValidationReports))
	}
	if !result.ValidationReports[1].AcceptedDuplicate.Skip || len(result.ValidationReports[1].AcceptedDuplicate.MatchingJobIDs) != 1 {
		t.Fatalf("accepted-hash duplicate was not reported: %#v", result.ValidationReports[1].AcceptedDuplicate)
	}
}

func TestInfrastructureChangesPreserveAcceptedSemanticIdentity(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_EXECUTION_VALIDATION_MODE", "shadow")
	baseline := testExperiment("resnet18", 8)
	server, projectID, firstPlan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{baseline})
	first, err := server.executeStoredExperimentPlan(
		firstPlan.ID,
		executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"},
	)
	if err != nil || len(first.Jobs) != 1 {
		t.Fatalf("execute first plan: jobs=%d err=%v", len(first.Jobs), err)
	}
	equivalent := baseline
	equivalent.ResolutionStrategy = "low_latency"
	secondPlan, err := server.store.CreateExperimentPlan(
		projectID,
		firstPlan.DatasetID,
		"macro_f1",
		1,
		5,
		[]plans.PlannedExperiment{equivalent},
		nil,
		"",
	)
	if err != nil {
		t.Fatal(err)
	}
	second, err := server.executeStoredExperimentPlan(
		secondPlan.ID,
		executeExperimentPlanRequest{Provider: "modal", GPUType: "A100"},
	)
	if err != nil {
		t.Fatalf("execute equivalent plan: %v", err)
	}
	if len(second.Jobs) != 0 || len(second.ValidationReports) != 1 {
		t.Fatalf("infrastructure-only duplicate was scheduled: %#v", second)
	}
	decision := second.ValidationReports[0].AcceptedDuplicate
	if !decision.Skip || len(decision.MatchingJobIDs) != 1 || decision.MatchingJobIDs[0] != first.Jobs[0].ID {
		t.Fatalf("unexpected accepted duplicate decision: %#v", decision)
	}
}

func TestPlannerEnforcementErrorCarriesActionableAlternative(t *testing.T) {
	card, err := execution.BuildPlannerCapabilityCard(
		"image_classification", "modal_torchvision", execution.ValidationModeEnforce, []string{"resnet"},
	)
	if err != nil {
		t.Fatalf("build capability card: %v", err)
	}
	experiment := testExperiment("resnet18", 8)
	experiment.ResolutionStrategy = "low_latency"
	reports, err := validatePlannerExecutionCapabilities([]plans.PlannedExperiment{experiment}, agents.ExperimentPlannerInput{
		ExecutionCapabilityCard: card,
	})
	if err == nil || len(reports) != 1 {
		t.Fatalf("expected actionable planner enforcement result: reports=%#v err=%v", reports, err)
	}
	if !strings.Contains(err.Error(), "omit resolution_strategy") {
		t.Fatalf("planner retry feedback omitted the alternative: %v", err)
	}
}

func TestExecutionSpecUsesModelSpecificImageDefault(t *testing.T) {
	experiment := testExperiment("efficientnet_b1", 8)
	experiment.ImageSize = 0
	server, _, plan := newAutomaticReviewFixture(t, []plans.PlannedExperiment{experiment})

	result, err := server.executeStoredExperimentPlan(
		plan.ID,
		executeExperimentPlanRequest{Provider: "modal", GPUType: "T4"},
	)
	if err != nil {
		t.Fatalf("execute plan: %v", err)
	}
	job := result.Jobs[0]
	assertConfigValue(t, job.Config, "image_size", 240)
	spec := job.Config[execution.ExecutionSpecConfigKey].(map[string]any)
	accepted := spec["accepted_config"].(map[string]any)
	assertConfigValue(t, accepted, "image_size", float64(240))
}

func TestAutoMLZeroValuesBecomePresenceSafe(t *testing.T) {
	experiment := testExperiment("resnet18", 8)
	experiment.Optimizer = "sgd"
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{PolicyType: "mixup"}
	if err := applyAutoMLValueToExperiment(&experiment, "optimizer_momentum", 0.0); err != nil {
		t.Fatalf("apply zero momentum: %v", err)
	}
	if err := applyAutoMLValueToExperiment(
		&experiment,
		"augmentation_policy_config.probability",
		0.0,
	); err != nil {
		t.Fatalf("apply zero augmentation probability: %v", err)
	}
	if !experiment.IsFieldPresent("optimizer_momentum") {
		t.Fatal("AutoML zero momentum was not marked present")
	}
	if !experiment.IsFieldPresent("augmentation_policy_config.probability") {
		t.Fatal("AutoML zero augmentation probability was not marked present")
	}
	if value, ok := autoMLParameterValue(experiment, "optimizer_momentum"); !ok || value != float64(0) {
		t.Fatalf("AutoML zero momentum was not retained as a final value: value=%#v ok=%v", value, ok)
	}
	if value, ok := autoMLParameterValue(
		experiment,
		"augmentation_policy_config.probability",
	); !ok || value != float64(0) {
		t.Fatalf("AutoML zero probability was not retained as a final value: value=%#v ok=%v", value, ok)
	}
}

func assertConfigValue(t *testing.T, config map[string]any, key string, expected any) {
	t.Helper()
	actual, ok := config[key]
	if !ok {
		t.Fatalf("config is missing %s: %#v", key, config)
	}
	if actual != expected {
		t.Fatalf("config %s = %#v, want %#v", key, actual, expected)
	}
}
