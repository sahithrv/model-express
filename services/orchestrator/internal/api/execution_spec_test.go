package api

import (
	"encoding/json"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/plans"
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
