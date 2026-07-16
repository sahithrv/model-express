package api

import (
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
)

func normalizePlannerProposalExperimentsForExecution(experiments []plans.PlannedExperiment) ([]plans.PlannedExperiment, []string) {
	normalized := append([]plans.PlannedExperiment(nil), experiments...)
	warnings := []string{}
	for index := range normalized {
		normalized[index] = clonePlannerExperimentForExecutionNormalization(normalized[index])
		experiment, experimentWarnings := normalizePlannerProposalExperimentForExecution(normalized[index], index)
		normalized[index] = experiment
		warnings = append(warnings, experimentWarnings...)
	}
	return normalized, warnings
}

func clonePlannerExperimentForExecutionNormalization(experiment plans.PlannedExperiment) plans.PlannedExperiment {
	if experiment.AugmentationPolicyConfig != nil {
		config := *experiment.AugmentationPolicyConfig
		experiment.AugmentationPolicyConfig = &config
	}
	return experiment
}

func normalizePlannerProposalExperimentForExecution(experiment plans.PlannedExperiment, index int) (plans.PlannedExperiment, []string) {
	if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		return experiment, nil
	}
	warnings := []string{}
	clearInactiveConditionalScalars(&experiment, index, &warnings)
	normalizePlannerAugmentationPolicyConfig(&experiment, index, &warnings)
	return experiment, warnings
}

func clearInactiveConditionalScalars(experiment *plans.PlannedExperiment, index int, warnings *[]string) {
	if experiment.IsFieldPresent("optimizer_momentum") && !strings.EqualFold(strings.TrimSpace(experiment.Optimizer), "sgd") {
		experiment.OptimizerMomentum = 0
		experiment.ClearFieldPresence("optimizer_momentum")
		*warnings = append(*warnings, plannerConditionalRepairWarning(index, "optimizer_momentum", "optimizer=sgd"))
	}
	if experiment.IsFieldPresent("scheduler_step_size") && !strings.EqualFold(strings.TrimSpace(experiment.Scheduler), "step") {
		experiment.SchedulerStepSize = 0
		experiment.ClearFieldPresence("scheduler_step_size")
		*warnings = append(*warnings, plannerConditionalRepairWarning(index, "scheduler_step_size", "scheduler=step"))
	}
	if experiment.IsFieldPresent("scheduler_gamma") && !strings.EqualFold(strings.TrimSpace(experiment.Scheduler), "step") {
		experiment.SchedulerGamma = 0
		experiment.ClearFieldPresence("scheduler_gamma")
		*warnings = append(*warnings, plannerConditionalRepairWarning(index, "scheduler_gamma", "scheduler=step"))
	}
}

func normalizePlannerAugmentationPolicyConfig(experiment *plans.PlannedExperiment, index int, warnings *[]string) {
	if experiment.AugmentationPolicyConfig == nil {
		return
	}
	policyType, ok := canonicalPlannerAugmentationPolicy(experiment.AugmentationPolicyConfig.PolicyType)
	if !ok || policyType == "" {
		return
	}
	if experiment.AugmentationPolicyConfig.PolicyType != policyType {
		experiment.AugmentationPolicyConfig.PolicyType = policyType
		experiment.MarkFieldPresent("augmentation_policy_config.policy_type")
	}
	if !strings.EqualFold(strings.TrimSpace(experiment.AugmentationPolicy), policyType) {
		experiment.AugmentationPolicy = policyType
		experiment.MarkFieldPresent("augmentation_policy")
		*warnings = append(*warnings, fmt.Sprintf("Set augmentation_policy=%s for planner experiment %d to match augmentation_policy_config.policy_type before execution fidelity validation.", policyType, index))
	}
	clearInactiveAugmentationPolicyField(experiment, index, warnings, policyType, "augmentation_policy_config.alpha")
	clearInactiveAugmentationPolicyField(experiment, index, warnings, policyType, "augmentation_policy_config.magnitude")
	clearInactiveAugmentationPolicyField(experiment, index, warnings, policyType, "augmentation_policy_config.num_magnitude_bins")
	clearInactiveAugmentationPolicyField(experiment, index, warnings, policyType, "augmentation_policy_config.num_ops")
	clearInactiveAugmentationPolicyField(experiment, index, warnings, policyType, "augmentation_policy_config.probability")
}

func clearInactiveAugmentationPolicyField(
	experiment *plans.PlannedExperiment,
	index int,
	warnings *[]string,
	policyType string,
	field string,
) {
	if !experiment.IsFieldPresent(field) || plannerAugmentationPolicyConfigFieldActive(policyType, field) {
		return
	}
	switch field {
	case "augmentation_policy_config.alpha":
		experiment.AugmentationPolicyConfig.Alpha = 0
	case "augmentation_policy_config.magnitude":
		experiment.AugmentationPolicyConfig.Magnitude = 0
	case "augmentation_policy_config.num_magnitude_bins":
		experiment.AugmentationPolicyConfig.NumMagnitudeBins = 0
	case "augmentation_policy_config.num_ops":
		experiment.AugmentationPolicyConfig.NumOps = 0
	case "augmentation_policy_config.probability":
		experiment.AugmentationPolicyConfig.Probability = 0
	}
	experiment.ClearFieldPresence(field)
	*warnings = append(*warnings, fmt.Sprintf(
		"Removed inactive %s from planner experiment %d because execution fidelity only accepts it when %s.",
		field,
		index,
		plannerAugmentationPolicyConfigActivation(field),
	))
}

func plannerAugmentationPolicyConfigFieldActive(policyType, field string) bool {
	policyType = strings.ToLower(strings.TrimSpace(policyType))
	switch field {
	case "augmentation_policy_config.alpha":
		return policyType == "mixup" || policyType == "cutmix"
	case "augmentation_policy_config.magnitude", "augmentation_policy_config.num_ops":
		return policyType == "randaugment"
	case "augmentation_policy_config.num_magnitude_bins":
		return policyType == "trivialaugment"
	case "augmentation_policy_config.probability":
		return policyType != "" && policyType != "none" && policyType != "custom"
	default:
		return true
	}
}

func plannerAugmentationPolicyConfigActivation(field string) string {
	switch field {
	case "augmentation_policy_config.alpha":
		return "augmentation_policy_config.policy_type is mixup or cutmix"
	case "augmentation_policy_config.magnitude", "augmentation_policy_config.num_ops":
		return "augmentation_policy_config.policy_type is randaugment"
	case "augmentation_policy_config.num_magnitude_bins":
		return "augmentation_policy_config.policy_type is trivialaugment"
	case "augmentation_policy_config.probability":
		return "augmentation_policy_config.policy_type is not none or custom"
	default:
		return "its conditional policy is active"
	}
}

func plannerConditionalRepairWarning(index int, field, activation string) string {
	return fmt.Sprintf("Removed inactive %s from planner experiment %d because execution fidelity only accepts it when %s.", field, index, activation)
}

func canonicalPlannerAugmentationPolicy(value string) (string, bool) {
	entry, ok := catalog.Resolve("augmentation_policies", value)
	if !ok {
		return strings.ToLower(strings.TrimSpace(value)), false
	}
	return entry.ID, true
}
