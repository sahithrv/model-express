package execution

import (
	"sort"
	"strings"
)

const (
	LegacyEvidencePolicyAllow       = "allow"
	LegacyEvidencePolicyVisibleOnly = "visible_only"
)

const (
	EvidenceReasonMatchedFinalized           = "matched_finalized"
	EvidenceReasonApprovedAdjustment         = "approved_adjustment_allowlisted"
	EvidenceReasonLegacyCompatibility        = "legacy_compatibility_policy"
	EvidenceReasonLegacyVisibleOnly          = "legacy_visible_only"
	EvidenceReasonNoFinalRealization         = "no_final_realization"
	EvidenceReasonMismatched                 = "execution_mismatch"
	EvidenceReasonSimulated                  = "simulated_execution"
	EvidenceReasonAdjustmentNotAllowlisted   = "adjustment_not_allowlisted"
	EvidenceReasonVerdictNotEvidenceEligible = "verdict_not_evidence_eligible"
)

// EvidenceEligibility is the compact, server-derived view used by learning and
// automatic champion selection. It intentionally keeps request, acceptance,
// realization, and infrastructure identities separate.
type EvidenceEligibility struct {
	SchemaVersion             string   `json:"schema_version,omitempty"`
	CapabilityVersion         string   `json:"capability_version,omitempty"`
	LifecycleStatus           string   `json:"lifecycle_status,omitempty"`
	FidelityVerdict           string   `json:"fidelity_verdict"`
	RequestedConfigHash       string   `json:"requested_config_hash,omitempty"`
	AcceptedSpecHash          string   `json:"accepted_spec_hash,omitempty"`
	RealizedEffectiveHash     string   `json:"realized_effective_hash,omitempty"`
	AdjustmentReasonCodes     []string `json:"adjustment_reason_codes,omitempty"`
	LearningEligible          bool     `json:"learning_eligible"`
	AutomaticChampionEligible bool     `json:"automatic_champion_eligible"`
	EligibilityReason         string   `json:"eligibility_reason"`
}

func NormalizeLegacyEvidencePolicy(value string) string {
	if strings.EqualFold(strings.TrimSpace(value), LegacyEvidencePolicyVisibleOnly) {
		return LegacyEvidencePolicyVisibleOnly
	}
	return LegacyEvidencePolicyAllow
}

// DeriveEvidenceEligibility returns UNVERIFIED for jobs without a versioned
// execution record. The compatibility policy can keep those historical runs
// eligible while still marking them; visible_only makes them read-only history.
func DeriveEvidenceEligibility(record *ExecutionRecord, legacyPolicy string) EvidenceEligibility {
	if record == nil || strings.TrimSpace(record.AcceptedSpec.SchemaVersion) == "" {
		eligible := NormalizeLegacyEvidencePolicy(legacyPolicy) == LegacyEvidencePolicyAllow
		reason := EvidenceReasonLegacyVisibleOnly
		if eligible {
			reason = EvidenceReasonLegacyCompatibility
		}
		return EvidenceEligibility{
			FidelityVerdict:           ExecutionVerdictUnverified,
			LearningEligible:          eligible,
			AutomaticChampionEligible: eligible,
			EligibilityReason:         reason,
		}
	}

	result := EvidenceEligibility{
		SchemaVersion:         record.AcceptedSpec.SchemaVersion,
		CapabilityVersion:     record.AcceptedSpec.CapabilityVersion,
		RequestedConfigHash:   record.AcceptedSpec.RequestedConfigHash,
		AcceptedSpecHash:      record.AcceptedSpec.AcceptedSpecHash,
		AdjustmentReasonCodes: []string{},
	}
	attempt, ok := latestAttempt(record.Attempts)
	if !ok {
		result.EligibilityReason = EvidenceReasonNoFinalRealization
		return result
	}
	result.LifecycleStatus = attempt.LifecycleStatus
	result.RealizedEffectiveHash = attempt.RealizedEffectiveHash
	result.AdjustmentReasonCodes = append([]string(nil), attempt.AdjustmentReasonCodes...)
	if attempt.FidelityVerdict != nil {
		result.FidelityVerdict = strings.ToUpper(strings.TrimSpace(*attempt.FidelityVerdict))
	}
	if attempt.LifecycleStatus != ExecutionLifecycleFinalized {
		result.EligibilityReason = EvidenceReasonNoFinalRealization
		return result
	}

	switch result.FidelityVerdict {
	case ExecutionVerdictMatched:
		result.LearningEligible = true
		result.AutomaticChampionEligible = true
		result.EligibilityReason = EvidenceReasonMatchedFinalized
	case ExecutionVerdictApprovedAdjustment:
		if approvedEvidenceAdjustmentReasons(result.AdjustmentReasonCodes) {
			result.LearningEligible = true
			result.AutomaticChampionEligible = true
			result.EligibilityReason = EvidenceReasonApprovedAdjustment
		} else {
			result.EligibilityReason = EvidenceReasonAdjustmentNotAllowlisted
		}
	case ExecutionVerdictMismatch:
		result.EligibilityReason = EvidenceReasonMismatched
	case ExecutionVerdictSimulated:
		result.EligibilityReason = EvidenceReasonSimulated
	default:
		result.EligibilityReason = EvidenceReasonVerdictNotEvidenceEligible
	}
	return result
}

func latestAttempt(attempts []AttemptExecutionRecord) (AttemptExecutionRecord, bool) {
	if len(attempts) == 0 {
		return AttemptExecutionRecord{}, false
	}
	ordered := append([]AttemptExecutionRecord(nil), attempts...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].AttemptNumber != ordered[j].AttemptNumber {
			return ordered[i].AttemptNumber > ordered[j].AttemptNumber
		}
		return ordered[i].UpdatedAt.After(ordered[j].UpdatedAt)
	})
	return ordered[0], true
}

func approvedEvidenceAdjustmentReasons(reasons []string) bool {
	if len(reasons) == 0 {
		return false
	}
	for _, reason := range reasons {
		if strings.TrimSpace(reason) != ExecutionAdjustmentReasonBatchSizeReduced {
			return false
		}
	}
	return true
}
