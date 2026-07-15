package store

import "model-express/services/orchestrator/internal/policies"

func normalizeEvaluationSlices(evaluation *policies.Evaluation) {
	if evaluation.CompatibilityProfiles == nil {
		evaluation.CompatibilityProfiles = []policies.ProfileSource{}
	}
	if evaluation.PolicySources == nil {
		evaluation.PolicySources = []policies.PolicySource{}
	}
	if evaluation.RequestedCapabilityUses == nil {
		evaluation.RequestedCapabilityUses = []policies.CapabilityUse{}
	}
	if evaluation.EffectiveCapabilityUses == nil {
		evaluation.EffectiveCapabilityUses = []policies.CapabilityUse{}
	}
	if evaluation.ReasonCodes == nil {
		evaluation.ReasonCodes = []policies.ReasonCode{}
	}
	if evaluation.Findings == nil {
		evaluation.Findings = []policies.Finding{}
	}
}
