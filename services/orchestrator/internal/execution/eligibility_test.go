package execution

import "testing"

func TestDeriveEvidenceEligibility(t *testing.T) {
	matched := ExecutionVerdictMatched
	adjusted := ExecutionVerdictApprovedAdjustment
	mismatch := ExecutionVerdictMismatch
	simulated := ExecutionVerdictSimulated
	tests := []struct {
		name       string
		record     *ExecutionRecord
		policy     string
		eligible   bool
		verdict    string
		reasonCode string
	}{
		{name: "legacy compatibility", policy: LegacyEvidencePolicyAllow, eligible: true, verdict: ExecutionVerdictUnverified, reasonCode: EvidenceReasonLegacyCompatibility},
		{name: "legacy visible only", policy: LegacyEvidencePolicyVisibleOnly, verdict: ExecutionVerdictUnverified, reasonCode: EvidenceReasonLegacyVisibleOnly},
		{name: "matched finalized", record: eligibilityTestRecord(ExecutionLifecycleFinalized, &matched, nil), eligible: true, verdict: matched, reasonCode: EvidenceReasonMatchedFinalized},
		{name: "matched initialized", record: eligibilityTestRecord(ExecutionLifecycleInitialized, &matched, nil), verdict: matched, reasonCode: EvidenceReasonNoFinalRealization},
		{name: "approved allowlisted adjustment", record: eligibilityTestRecord(ExecutionLifecycleFinalized, &adjusted, []string{ExecutionAdjustmentReasonBatchSizeReduced}), eligible: true, verdict: adjusted, reasonCode: EvidenceReasonApprovedAdjustment},
		{name: "approved adjustment without reason", record: eligibilityTestRecord(ExecutionLifecycleFinalized, &adjusted, nil), verdict: adjusted, reasonCode: EvidenceReasonAdjustmentNotAllowlisted},
		{name: "mismatch", record: eligibilityTestRecord(ExecutionLifecycleFinalized, &mismatch, nil), verdict: mismatch, reasonCode: EvidenceReasonMismatched},
		{name: "simulated", record: eligibilityTestRecord(ExecutionLifecycleFinalized, &simulated, nil), verdict: simulated, reasonCode: EvidenceReasonSimulated},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DeriveEvidenceEligibility(tt.record, tt.policy)
			if got.LearningEligible != tt.eligible || got.AutomaticChampionEligible != tt.eligible {
				t.Fatalf("eligibility = learning:%t champion:%t, want %t", got.LearningEligible, got.AutomaticChampionEligible, tt.eligible)
			}
			if got.FidelityVerdict != tt.verdict || got.EligibilityReason != tt.reasonCode {
				t.Fatalf("verdict/reason = %q/%q, want %q/%q", got.FidelityVerdict, got.EligibilityReason, tt.verdict, tt.reasonCode)
			}
		})
	}
}

func eligibilityTestRecord(lifecycle string, verdict *string, reasons []string) *ExecutionRecord {
	return &ExecutionRecord{
		AcceptedSpec: JobExecutionSpec{SchemaVersion: ExecutionSpecSchemaVersionV1, AcceptedSpecHash: "accepted"},
		Attempts: []AttemptExecutionRecord{{
			AttemptNumber: 1, LifecycleStatus: lifecycle, FidelityVerdict: verdict,
			RealizedEffectiveHash: "realized", AdjustmentReasonCodes: reasons,
		}},
	}
}
