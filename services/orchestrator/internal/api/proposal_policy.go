package api

import (
	"errors"
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/projects"
)

const (
	policyOperationPropose         = "propose"
	policyOperationPersistProposal = "persist_proposal"
	policyOperationPersistPlan     = "persist_plan"
	policyOperationReusePlan       = "reuse_plan"
	policyStatusAllowed            = "allowed"
)

func (s *Server) resolveProposalPolicy(project projects.Project, dataset datasets.Dataset, operation string) (policies.EffectivePolicy, error) {
	metadataSummary, err := s.activeAgentSafeDatasetMetadataSummary(dataset)
	if err != nil {
		return policies.EffectivePolicy{}, err
	}
	task := "image_classification"
	if datasetHasYOLODetectionEvidence(dataset, metadataSummary) {
		task = "object_detection"
	}
	runner, err := executionRunnerFor(s.defaultExecuteExperimentPlanRequest().Provider, task)
	if err != nil {
		return policies.EffectivePolicy{}, err
	}
	result, resolveErr := policies.NewResolver(s.store).Resolve(policies.ScopeContext{
		AccountID: project.AccountID, ProjectID: project.ID, DatasetID: dataset.ID, Task: task, Runner: runner,
	})
	if resolveErr == nil {
		return result, nil
	}
	if result.EffectivePolicyHash == "" {
		return result, resolveErr
	}
	evaluation, evaluationErr := policies.EvaluationFromEffectivePolicy(result, operation, "system", "")
	if evaluationErr == nil {
		if created, createErr := s.store.CreateExperimentPolicyEvaluation(evaluation); createErr == nil {
			var policyErr *policies.PolicyError
			if errors.As(resolveErr, &policyErr) {
				policyErr.PolicyEvaluationID = created.ID
			}
		} else {
			return result, createErr
		}
	}
	return result, resolveErr
}

func (s *Server) recordProposalPolicyEvaluation(
	effective policies.EffectivePolicy,
	operation string,
	experiments []plans.PlannedExperiment,
	agentInvocationID string,
) (policies.Evaluation, error) {
	evaluation, evaluateErr := policies.EvaluateProposal(effective, operation, experiments)
	evaluation.AgentInvocationID = strings.TrimSpace(agentInvocationID)
	created, createErr := s.store.CreateExperimentPolicyEvaluation(evaluation)
	if createErr != nil {
		return policies.Evaluation{}, createErr
	}
	if evaluateErr != nil {
		var policyErr *policies.PolicyError
		if errors.As(evaluateErr, &policyErr) {
			policyErr.PolicyEvaluationID = created.ID
		}
		return created, evaluateErr
	}
	return created, nil
}

func effectiveSupportedModelCatalog(effective policies.EffectivePolicy) []agents.SupportedModelSpec {
	out := []agents.SupportedModelSpec{}
	for _, entry := range effective.PermittedCatalog["models"] {
		task := effective.Snapshot.Context.Task
		if task == "" && len(entry.Tasks) == 1 {
			task = entry.Tasks[0]
		}
		out = append(out, supportedModelSpecFromCatalog(entry, task))
	}
	return out
}

func filterExecutionCapabilityCardByPolicy(card execution.PlannerCapabilityCard, effective policies.EffectivePolicy) execution.PlannerCapabilityCard {
	document := execution.CapabilitiesV1()
	filtered := make([]execution.PlannerCapabilityRule, 0, len(card.Rules))
	for _, rule := range card.Rules {
		definition := document.FieldCatalog[rule.Field]
		if definition.CatalogCategory == "" || len(rule.Values) == 0 {
			filtered = append(filtered, rule)
			continue
		}
		values := make([]string, 0, len(rule.Values))
		for _, value := range rule.Values {
			if policies.IsPermitted(effective, definition.CatalogCategory, value) {
				values = append(values, value)
			}
		}
		rule.Values = values
		if len(rule.Values) > 0 || rule.Range != "" {
			filtered = append(filtered, rule)
		}
	}
	card.Rules = filtered
	card.ModelFamilies = nil
	for _, entry := range effective.PermittedCatalog["model_families"] {
		card.ModelFamilies = append(card.ModelFamilies, entry.ID)
	}
	return card
}

func policyNoProposalError(effective policies.EffectivePolicy, message string) error {
	blocked := append([]string(nil), effective.BlockedDimensions...)
	if len(blocked) == 0 {
		blocked = []string{"models"}
	}
	return &policies.PolicyError{
		Code: policies.ReasonNoValidConfiguration, Message: fmt.Sprintf("%s: %s", message, strings.Join(blocked, ", ")),
		EffectivePolicyHash: effective.EffectivePolicyHash, Findings: append([]policies.Finding(nil), effective.Findings...),
		BlockedDimensions: blocked, ContributingScopes: append([]policies.ScopeContribution(nil), effective.ContributingScopes...),
	}
}
