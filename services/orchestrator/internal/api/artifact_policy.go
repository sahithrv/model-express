package api

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/policies"
)

func artifactSelectionFromEvaluation(evaluation policies.Evaluation) (execution.ArtifactPolicySelection, error) {
	var snapshot policies.EffectiveSnapshot
	if err := json.Unmarshal(evaluation.EffectiveSnapshot, &snapshot); err != nil {
		return execution.ArtifactPolicySelection{}, fmt.Errorf("decode effective policy artifact selection: %w", err)
	}
	return execution.ArtifactPolicySelection{
		PermittedCatalog:    snapshot.PermittedCatalog,
		PolicyRestricted:    len(snapshot.PolicySources) > 0,
		EffectivePolicyHash: evaluation.EffectivePolicyHash,
	}, nil
}

func automaticArtifactPlanFromEvaluation(task, runner string, evaluation policies.Evaluation) (execution.ArtifactPlanV1, error) {
	selection, err := artifactSelectionFromEvaluation(evaluation)
	if err != nil {
		return execution.ArtifactPlanV1{}, err
	}
	return execution.BuildAutomaticArtifactPlanV1(task, runner, selection)
}

func manualArtifactPlanFromEvaluation(task, runner, format string, evaluation policies.Evaluation) (execution.ArtifactPlanV1, error) {
	selection, err := artifactSelectionFromEvaluation(evaluation)
	if err != nil {
		return execution.ArtifactPlanV1{}, err
	}
	return execution.BuildManualArtifactPlanV1(task, runner, format, selection)
}

func artifactPlanCapabilityUses(plan execution.ArtifactPlanV1, fieldPrefix string) []policies.CapabilityUse {
	if fieldPrefix == "" {
		fieldPrefix = "artifact_plan"
	}
	uses := []policies.CapabilityUse{}
	for index, artifact := range plan.Artifacts {
		prefix := fmt.Sprintf("%s.artifacts[%d]", fieldPrefix, index)
		uses = append(uses,
			policies.CapabilityUse{Catalog: "export_formats", ID: artifact.Format, FieldPath: prefix + ".format", Origin: "server_plan"},
			policies.CapabilityUse{Catalog: "precisions", ID: artifact.Precision, FieldPath: prefix + ".precision", Origin: "server_plan"},
			policies.CapabilityUse{Catalog: "runtimes", ID: artifact.Runtime, FieldPath: prefix + ".runtime", Origin: "server_plan"},
			policies.CapabilityUse{Catalog: "execution_providers", ID: artifact.ExecutionProvider, FieldPath: prefix + ".execution_provider", Origin: "server_plan"},
		)
		for requirementIndex, requirement := range artifact.ExecutionRequirements {
			uses = append(uses, policies.CapabilityUse{
				Catalog: "execution_requirements", ID: requirement,
				FieldPath: fmt.Sprintf("%s.execution_requirements[%d]", prefix, requirementIndex), Origin: "server_plan",
			})
		}
	}
	return uses
}

func evaluateArtifactPlan(
	effective policies.EffectivePolicy,
	base policies.Evaluation,
	operation string,
	subject any,
	plan execution.ArtifactPlanV1,
) (policies.Evaluation, error) {
	uses := artifactPlanCapabilityUses(plan, "artifact_plan")
	if config, ok := subject.(map[string]any); ok {
		uses = append(uses, requestedArtifactCapabilityUses(config)...)
	}
	artifactEvaluation, artifactErr := policies.EvaluateCapabilityUses(
		effective, operation, subject, uses,
	)
	merged := policies.MergeEvaluations(base, artifactEvaluation)
	if artifactErr != nil {
		return merged, policies.ErrorForEvaluation(merged, "artifact plan contains capabilities excluded by the effective experiment policy")
	}
	return merged, nil
}

func requestedArtifactCapabilityUses(config map[string]any) []policies.CapabilityUse {
	fields := []struct {
		Keys    []string
		Catalog string
	}{
		{Keys: []string{"precision", "artifact_precision"}, Catalog: "precisions"},
		{Keys: []string{"runtime", "deployment_runtime"}, Catalog: "runtimes"},
		{Keys: []string{"execution_provider"}, Catalog: "execution_providers"},
	}
	uses := []policies.CapabilityUse{}
	for _, field := range fields {
		for _, key := range field.Keys {
			if value := strings.TrimSpace(configString(config, key)); value != "" {
				uses = append(uses, policies.CapabilityUse{
					Catalog: field.Catalog, ID: value, FieldPath: "config." + key, Origin: "explicit",
				})
				break
			}
		}
	}
	if values, ok := config["execution_requirements"].([]any); ok {
		for index, value := range values {
			if text, ok := value.(string); ok && strings.TrimSpace(text) != "" {
				uses = append(uses, policies.CapabilityUse{
					Catalog: "execution_requirements", ID: text,
					FieldPath: fmt.Sprintf("config.execution_requirements[%d]", index), Origin: "explicit",
				})
			}
		}
	}
	return uses
}

func artifactPlanFromJobConfig(config map[string]any) (execution.ArtifactPlanV1, bool, error) {
	var value any
	if spec := payloadMap(config, execution.ExecutionSpecConfigKey); len(spec) > 0 {
		value = spec["artifact_plan"]
	}
	if value == nil {
		value = config[execution.ArtifactPlanConfigKey]
	}
	if value == nil {
		return execution.ArtifactPlanV1{}, false, nil
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return execution.ArtifactPlanV1{}, false, fmt.Errorf("marshal artifact plan: %w", err)
	}
	var plan execution.ArtifactPlanV1
	if err := json.Unmarshal(blob, &plan); err != nil {
		return execution.ArtifactPlanV1{}, false, fmt.Errorf("decode artifact plan: %w", err)
	}
	return plan, true, nil
}

func workerCapabilityFinding(workerPolicyVersions, workerArtifactVersions []string, plan execution.ArtifactPlanV1) policies.Finding {
	missing := []string{}
	for _, required := range plan.RequiredWorkerCapabilities.PolicyContractVersions {
		if !containsStringValue(workerPolicyVersions, required) {
			missing = append(missing, "policy:"+required)
		}
	}
	for _, required := range plan.RequiredWorkerCapabilities.ArtifactPlanVersions {
		if !containsStringValue(workerArtifactVersions, required) {
			missing = append(missing, "artifact:"+required)
		}
	}
	sort.Strings(missing)
	return policies.Finding{
		Code:      policies.ReasonWorkerCapabilityUnavailable,
		FieldPath: "worker.capability_versions", ID: strings.Join(missing, ","), Origin: "worker_advertisement",
		Remediation: "Upgrade or restart a worker that advertises policy_contract_v1 and artifact_plan_v1.",
	}
}

func containsStringValue(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}
