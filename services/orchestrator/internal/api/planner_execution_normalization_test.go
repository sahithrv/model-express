package api

import (
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/plans"
)

func TestNormalizePlannerProposalExperimentsForExecutionRepairsInactiveConditionals(t *testing.T) {
	card, err := execution.BuildPlannerCapabilityCard(
		"image_classification", "modal_torchvision", execution.ValidationModeEnforce, []string{"resnet"},
	)
	if err != nil {
		t.Fatalf("build capability card: %v", err)
	}
	input := agents.ExperimentPlannerInput{ExecutionCapabilityCard: card}
	experiment := testExperiment("resnet18", 8)
	experiment.Optimizer = "adamw"
	experiment.OptimizerMomentum = 0
	experiment.Scheduler = "cosine"
	experiment.SchedulerStepSize = 0
	experiment.SchedulerGamma = 0
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{
		PolicyType:       "rand_augment",
		Alpha:            0.3,
		Magnitude:        9,
		NumOps:           2,
		NumMagnitudeBins: 31,
		Probability:      0.8,
	}
	for _, field := range []string{
		"optimizer",
		"optimizer_momentum",
		"scheduler",
		"scheduler_step_size",
		"scheduler_gamma",
		"augmentation_policy_config.policy_type",
		"augmentation_policy_config.alpha",
		"augmentation_policy_config.magnitude",
		"augmentation_policy_config.num_ops",
		"augmentation_policy_config.num_magnitude_bins",
		"augmentation_policy_config.probability",
	} {
		experiment.MarkFieldPresent(field)
	}

	_, err = validatePlannerExecutionCapabilities([]plans.PlannedExperiment{experiment}, input)
	if err == nil {
		t.Fatal("raw proposal unexpectedly passed execution fidelity")
	}
	for _, field := range []string{"optimizer_momentum", "scheduler_step_size", "scheduler_gamma", "augmentation_policy_config.alpha", "augmentation_policy_config.num_magnitude_bins"} {
		if !strings.Contains(err.Error(), field) {
			t.Fatalf("raw proposal error did not mention %s: %v", field, err)
		}
	}

	normalized, warnings := normalizePlannerProposalExperimentsForExecution([]plans.PlannedExperiment{experiment})
	if len(warnings) == 0 {
		t.Fatal("expected conditional normalization warnings")
	}
	repaired := normalized[0]
	if repaired.AugmentationPolicy != "randaugment" || repaired.AugmentationPolicyConfig.PolicyType != "randaugment" {
		t.Fatalf("augmentation policy was not canonicalized and aligned: %#v", repaired)
	}
	for _, field := range []string{"optimizer_momentum", "scheduler_step_size", "scheduler_gamma", "augmentation_policy_config.alpha", "augmentation_policy_config.num_magnitude_bins"} {
		if repaired.IsFieldPresent(field) {
			t.Fatalf("%s presence was not cleared", field)
		}
	}
	for _, field := range []string{"augmentation_policy_config.magnitude", "augmentation_policy_config.num_ops", "augmentation_policy_config.probability"} {
		if !repaired.IsFieldPresent(field) {
			t.Fatalf("%s should remain present for randaugment", field)
		}
	}
	if err := validatePlannedExperiment(repaired, 0); err != nil {
		t.Fatalf("repaired proposal failed first validation: %v", err)
	}
	reports, err := validatePlannerExecutionCapabilities(normalized, input)
	if err != nil {
		t.Fatalf("repaired proposal failed execution fidelity: reports=%#v err=%v", reports, err)
	}
}

func TestNormalizePlannerProposalExperimentsPreservesActiveZeroConditionals(t *testing.T) {
	card, err := execution.BuildPlannerCapabilityCard(
		"image_classification", "modal_torchvision", execution.ValidationModeEnforce, []string{"resnet"},
	)
	if err != nil {
		t.Fatalf("build capability card: %v", err)
	}
	experiment := testExperiment("resnet18", 8)
	experiment.AugmentationPolicyConfig = &plans.AugmentationPolicyConfig{
		PolicyType:  "mixup",
		Alpha:       0,
		Probability: 0,
		Magnitude:   9,
		NumOps:      2,
	}
	for _, field := range []string{
		"augmentation_policy_config.policy_type",
		"augmentation_policy_config.alpha",
		"augmentation_policy_config.probability",
		"augmentation_policy_config.magnitude",
		"augmentation_policy_config.num_ops",
	} {
		experiment.MarkFieldPresent(field)
	}

	normalized, _ := normalizePlannerProposalExperimentsForExecution([]plans.PlannedExperiment{experiment})
	repaired := normalized[0]
	for _, field := range []string{"augmentation_policy_config.alpha", "augmentation_policy_config.probability"} {
		if !repaired.IsFieldPresent(field) {
			t.Fatalf("%s active zero value should remain present for mixup", field)
		}
	}
	for _, field := range []string{"augmentation_policy_config.magnitude", "augmentation_policy_config.num_ops"} {
		if repaired.IsFieldPresent(field) {
			t.Fatalf("%s should be removed for mixup", field)
		}
	}
	requested, err := repaired.RequestedConfig()
	if err != nil {
		t.Fatalf("requested config: %v", err)
	}
	policyConfig := requested["augmentation_policy_config"].(map[string]any)
	if policyConfig["alpha"] != float64(0) || policyConfig["probability"] != float64(0) {
		t.Fatalf("active zero-valued mixup fields were not preserved: %#v", policyConfig)
	}
	if reports, err := validatePlannerExecutionCapabilities(normalized, agents.ExperimentPlannerInput{ExecutionCapabilityCard: card}); err != nil {
		t.Fatalf("repaired mixup proposal failed execution fidelity: reports=%#v err=%v", reports, err)
	}
}

func TestPlannerAugmentationPolicyConfigFieldActiveDocumentsFidelityMatrix(t *testing.T) {
	cases := []struct {
		policy string
		field  string
		active bool
	}{
		{"mixup", "augmentation_policy_config.alpha", true},
		{"cutmix", "augmentation_policy_config.alpha", true},
		{"randaugment", "augmentation_policy_config.alpha", false},
		{"randaugment", "augmentation_policy_config.magnitude", true},
		{"randaugment", "augmentation_policy_config.num_ops", true},
		{"trivialaugment", "augmentation_policy_config.num_magnitude_bins", true},
		{"autoaugment", "augmentation_policy_config.num_magnitude_bins", false},
		{"autoaugment", "augmentation_policy_config.probability", true},
		{"basic", "augmentation_policy_config.probability", true},
		{"none", "augmentation_policy_config.probability", false},
	}
	for _, tc := range cases {
		if got := plannerAugmentationPolicyConfigFieldActive(tc.policy, tc.field); got != tc.active {
			t.Fatalf("%s active for %s = %v, want %v", tc.field, tc.policy, got, tc.active)
		}
	}
}
