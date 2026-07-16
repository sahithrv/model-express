package api

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/policies"
)

func plannerCandidateExecutionValidator(input agents.ExperimentPlannerInput) agents.PlannerCandidateExecutionValidator {
	existingAccepted := map[string]string{}
	for _, evidence := range input.ExecutionEvidence {
		if hash := strings.TrimSpace(evidence.AcceptedSpecHash); hash != "" {
			existingAccepted[hash] = evidence.JobID
		}
	}
	proposedAccepted := map[string]int{}
	return func(candidate agents.CandidateHypothesis, index int) agents.PlannerCandidateValidationResult {
		result := agents.PlannerCandidateValidationResult{
			Experiment:  clonePlannerExperimentForExecutionNormalization(candidate.ExperimentConfig),
			Disposition: agents.PlannerCandidateAccepted,
			Stage:       "candidate_prevalidation",
		}
		originalConfig, _ := result.Experiment.RequestedConfig()
		result.Experiment, _ = normalizePlannerProposalExperimentForExecution(result.Experiment, index)
		normalizedConfig, _ := result.Experiment.RequestedConfig()
		for _, path := range changedPlannerConfigPaths(originalConfig, normalizedConfig) {
			requested, _ := nestedPlannerConfigValue(originalConfig, path)
			accepted, acceptedOK := nestedPlannerConfigValue(normalizedConfig, path)
			removed := !acceptedOK
			central := removed && plannerCandidateFieldCentral(candidate, path)
			finding := candidateValidationFinding(candidate, index, path)
			finding.Classification = "inactive_or_canonical_normalization"
			finding.RequestedValue = requested
			finding.AcceptedValue = accepted
			finding.Removed = removed
			finding.Normalized = true
			finding.Incidental = !central
			finding.CentralToMechanism = central
			finding.ReasonCode = "candidate_field_normalized"
			finding.Reason = "field was removed or canonicalized before execution fidelity validation"
			finding.Action = "leave_unchanged"
			if removed {
				finding.Action = "repair"
				finding.SuggestedPatch = map[string]any{"remove": path}
			}
			result.FieldFindings = append(result.FieldFindings, finding)
			result.Normalized = true
			if central {
				result.Disposition = agents.PlannerCandidateRejectedFidelity
				result.ReasonCode = "normalization_removed_core_mechanism"
				result.Reason = fmt.Sprintf("candidate %d claims %s as a core intervention, but the field is inactive and would be removed", index, path)
				return result
			}
		}

		if err := validatePlannedExperiment(result.Experiment, index); err != nil {
			result.Disposition = agents.PlannerCandidateRejectedInvalid
			result.Stage = "structure"
			result.ReasonCode = "invalid_experiment"
			result.Reason = err.Error()
			return result
		}
		if err := validateExperimentDatasetCompatibility(result.Experiment, input.Dataset, index); err != nil {
			result.Disposition = agents.PlannerCandidateRejectedInvalid
			result.Stage = "task_model_compatibility"
			result.ReasonCode = "invalid_task_model"
			result.Reason = err.Error()
			return result
		}
		if input.EffectivePolicy != nil {
			if _, err := policies.EvaluateProposal(*input.EffectivePolicy, policyOperationPersistProposal, []plans.PlannedExperiment{result.Experiment}); err != nil {
				result.Disposition = agents.PlannerCandidateRejectedPolicy
				result.Stage = "policy"
				result.ReasonCode = "policy_exclusion"
				result.Reason = err.Error()
				return result
			}
		}
		if strings.EqualFold(strings.TrimSpace(result.Experiment.Template), jobs.TemplateLabelQualityAudit) || input.ExecutionCapabilityCard.Runner == "" {
			return result
		}

		provider := providerForExecutionRunner(input.ExecutionCapabilityCard.Runner)
		for repairAttempt := 0; repairAttempt < 8; repairAttempt++ {
			spec, err := buildExecutionSpecV1(result.Experiment, provider)
			if err != nil {
				result.Disposition = agents.PlannerCandidateRejectedInvalid
				result.Stage = "execution_spec"
				result.ReasonCode = "execution_spec_invalid"
				result.Reason = err.Error()
				return result
			}
			modelSpec, _ := supportedModelSpecByName(result.Experiment.Model)
			report, err := execution.ValidateExecutionSpecV1(spec, modelSpec.Family, input.ExecutionCapabilityCard.Mode)
			if err != nil {
				result.Disposition = agents.PlannerCandidateRejectedFidelity
				result.Stage = "execution_fidelity"
				result.ReasonCode = "execution_capability_failure"
				result.Reason = err.Error()
				return result
			}
			repaired := false
			for _, finding := range report.Findings {
				detail := candidateValidationFinding(candidate, index, finding.Field)
				detail.Classification = finding.Classification
				detail.RequestedValue = finding.RequestedValue
				detail.AcceptedValue = finding.AcceptedValue
				detail.Normalized = finding.Classification == "normalized" || finding.Classification == "normalized_away"
				detail.ReasonCode = finding.ReasonCode
				detail.Reason = finding.Message
				detail.SuggestedAlternative = finding.SuggestedAlternative
				detail.Prerequisites = plannerFindingPrerequisites(finding)
				detail.Action = "leave_unchanged"
				if !finding.WouldBlock {
					result.FieldFindings = appendUniqueCandidateFinding(result.FieldFindings, detail)
					if detail.Normalized {
						result.Normalized = true
						result.Experiment = applyPlannerAcceptedField(result.Experiment, finding.Field, finding.AcceptedValue)
					}
					continue
				}
				central := plannerCandidateFieldCentral(candidate, finding.Field)
				detail.CentralToMechanism = central
				detail.Incidental = !central
				detail.Disposition = agents.PlannerCandidateRejectedFidelity
				detail.Action = "replace"
				if central {
					result.FieldFindings = appendUniqueCandidateFinding(result.FieldFindings, detail)
					result.Disposition = agents.PlannerCandidateRejectedFidelity
					result.Stage = "execution_fidelity"
					result.ReasonCode = "unsupported_core_mechanism"
					result.Reason = fmt.Sprintf("candidate %d cannot run faithfully because core field %s is unsupported", index, finding.Field)
					return result
				}
				normalized, removed, removeErr := removeFollowUpExperimentConfigPath(result.Experiment, finding.Field)
				if removeErr != nil || !removed {
					result.FieldFindings = appendUniqueCandidateFinding(result.FieldFindings, detail)
					result.Disposition = agents.PlannerCandidateRejectedFidelity
					result.Stage = "execution_fidelity"
					result.ReasonCode = "unsupported_field_not_repairable"
					result.Reason = finding.Message
					return result
				}
				result.Experiment = normalized
				result.Normalized = true
				repaired = true
				detail.Removed = true
				detail.Normalized = true
				detail.Disposition = agents.PlannerCandidateAcceptedAfterNormalization
				detail.Action = "repair"
				detail.SuggestedPatch = map[string]any{"remove": finding.Field}
				result.FieldFindings = appendUniqueCandidateFinding(result.FieldFindings, detail)
			}
			if repaired {
				continue
			}
			if report.WouldBlock {
				result.Disposition = agents.PlannerCandidateRejectedFidelity
				result.Stage = "execution_fidelity"
				result.ReasonCode = "execution_fidelity_blocked"
				result.Reason = executionValidationSummary(report)
				return result
			}
			if jobID, ok := existingAccepted[spec.AcceptedSpecHash]; ok {
				result.Disposition = agents.PlannerCandidateRejectedDuplicate
				result.Stage = "accepted_spec_duplicate"
				result.ReasonCode = "accepted_spec_duplicate"
				result.Reason = fmt.Sprintf("candidate %d matches accepted spec %s from job %s", index, spec.AcceptedSpecHash, jobID)
				finding := candidateValidationFinding(candidate, index, "accepted_spec_hash")
				finding.Disposition = result.Disposition
				finding.Classification = "duplicate"
				finding.ReasonCode = result.ReasonCode
				finding.Reason = result.Reason
				finding.MatchingAcceptedSpec = spec.AcceptedSpecHash
				finding.MatchingJobID = jobID
				finding.Action = "replace"
				result.FieldFindings = appendUniqueCandidateFinding(result.FieldFindings, finding)
				return result
			}
			if previous, ok := proposedAccepted[spec.AcceptedSpecHash]; ok {
				result.Disposition = agents.PlannerCandidateRejectedDuplicate
				result.Stage = "accepted_spec_duplicate"
				result.ReasonCode = "accepted_spec_duplicate_in_proposal"
				result.Reason = fmt.Sprintf("candidate %d matches accepted spec %s from candidate %d", index, spec.AcceptedSpecHash, previous)
				finding := candidateValidationFinding(candidate, index, "accepted_spec_hash")
				finding.Disposition = result.Disposition
				finding.Classification = "duplicate"
				finding.ReasonCode = result.ReasonCode
				finding.Reason = result.Reason
				finding.MatchingAcceptedSpec = spec.AcceptedSpecHash
				finding.Action = "replace"
				result.FieldFindings = appendUniqueCandidateFinding(result.FieldFindings, finding)
				return result
			}
			proposedAccepted[spec.AcceptedSpecHash] = index
			if result.Normalized {
				result.Disposition = agents.PlannerCandidateAcceptedAfterNormalization
			}
			return result
		}
		result.Disposition = agents.PlannerCandidateRejectedFidelity
		result.Stage = "execution_fidelity"
		result.ReasonCode = "normalization_did_not_converge"
		result.Reason = fmt.Sprintf("candidate %d still failed execution fidelity after bounded repair", index)
		return result
	}
}

func candidateValidationFinding(candidate agents.CandidateHypothesis, index int, field string) agents.PlannerValidationFieldFinding {
	return agents.PlannerValidationFieldFinding{
		CandidateIndex:     index,
		CandidateModel:     candidate.ExperimentConfig.Model,
		CandidateMechanism: candidate.Mechanism,
		ValidationStage:    "execution_fidelity",
		Field:              field,
	}
}

func plannerCandidateFieldCentral(candidate agents.CandidateHypothesis, field string) bool {
	normalizedField := strings.ToLower(strings.TrimSpace(field))
	root := strings.Split(normalizedField, ".")[0]
	for key := range candidate.ProposedChanges {
		normalizedKey := strings.ToLower(strings.TrimSpace(key))
		if normalizedKey == normalizedField || normalizedKey == root || strings.HasPrefix(normalizedField, normalizedKey+".") {
			return true
		}
	}
	needle := strings.NewReplacer("_", " ", ".", " ").Replace(normalizedField)
	text := strings.ToLower(strings.Join([]string{
		candidate.Hypothesis, candidate.Mechanism, candidate.Intervention, candidate.ExpectedEffect,
		candidate.ExperimentConfig.Reason, candidate.ExperimentConfig.Strategy,
	}, " "))
	rootNeedle := strings.NewReplacer("_", " ", ".", " ").Replace(root)
	return (needle != "" && strings.Contains(text, needle)) || (rootNeedle != "" && strings.Contains(text, rootNeedle))
}

func plannerFindingPrerequisites(finding execution.ExecutionValidationFinding) []string {
	if strings.TrimSpace(finding.SuggestedAlternative) == "" {
		return nil
	}
	return []string{finding.SuggestedAlternative}
}

func appendUniqueCandidateFinding(values []agents.PlannerValidationFieldFinding, finding agents.PlannerValidationFieldFinding) []agents.PlannerValidationFieldFinding {
	for _, existing := range values {
		if existing.CandidateIndex == finding.CandidateIndex && existing.Field == finding.Field && existing.ReasonCode == finding.ReasonCode {
			return values
		}
	}
	return append(values, finding)
}

func changedPlannerConfigPaths(left, right map[string]any) []string {
	paths := map[string]bool{}
	collectPlannerConfigPaths(left, "", paths)
	collectPlannerConfigPaths(right, "", paths)
	out := []string{}
	for path := range paths {
		leftValue, leftOK := nestedPlannerConfigValue(left, path)
		rightValue, rightOK := nestedPlannerConfigValue(right, path)
		if leftOK != rightOK || !reflect.DeepEqual(leftValue, rightValue) {
			out = append(out, path)
		}
	}
	sort.Strings(out)
	return out
}

func collectPlannerConfigPaths(root map[string]any, prefix string, out map[string]bool) {
	for key, value := range root {
		path := key
		if prefix != "" {
			path = prefix + "." + key
		}
		if child, ok := value.(map[string]any); ok {
			collectPlannerConfigPaths(child, path, out)
			continue
		}
		out[path] = true
	}
}

func nestedPlannerConfigValue(root map[string]any, path string) (any, bool) {
	parts := strings.Split(path, ".")
	var current any = root
	for _, part := range parts {
		object, ok := current.(map[string]any)
		if !ok {
			return nil, false
		}
		current, ok = object[part]
		if !ok {
			return nil, false
		}
	}
	return current, true
}

func applyPlannerAcceptedField(experiment plans.PlannedExperiment, path string, value any) plans.PlannedExperiment {
	if value == nil || strings.TrimSpace(path) == "" {
		return experiment
	}
	config, err := experiment.RequestedConfig()
	if err != nil {
		return experiment
	}
	parts := strings.Split(path, ".")
	current := config
	for _, part := range parts[:len(parts)-1] {
		child, ok := current[part].(map[string]any)
		if !ok {
			child = map[string]any{}
			current[part] = child
		}
		current = child
	}
	current[parts[len(parts)-1]] = value
	data, err := json.Marshal(config)
	if err != nil {
		return experiment
	}
	var normalized plans.PlannedExperiment
	if err := json.Unmarshal(data, &normalized); err != nil {
		return experiment
	}
	return normalized
}

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
