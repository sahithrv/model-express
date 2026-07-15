package api

import (
	"errors"
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/store"
)

// validateTrainingCompletionFidelity deliberately uses the same eligibility
// derivation as planner learning and automatic champion selection. Shadow mode
// is the compatibility rollback: it changes behavior without mutating receipts.
func (s *Server) validateTrainingCompletionFidelity(job jobs.ExperimentJob) error {
	job = jobs.WithExecutionSpecStatus(job)
	if job.ExecutionSpecStatus != execution.ExecutionSpecStatusVersioned {
		return nil
	}
	if executionValidationMode() == execution.ValidationModeShadow {
		return nil
	}
	record, err := s.store.GetJobExecutionRecord(job.ID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			return fmt.Errorf("%w: versioned training completion requires a finalized execution realization", store.ErrInvalidRequest)
		}
		return err
	}
	if !isRealTrainingRunner(record.AcceptedSpec.Runner) {
		return nil
	}
	eligibility := execution.DeriveEvidenceEligibility(&record, execution.LegacyEvidencePolicyVisibleOnly)
	if eligibility.LifecycleStatus != execution.ExecutionLifecycleFinalized {
		return fmt.Errorf(
			"%w: versioned real-training completion requires a final realization (status=%s, reason=%s)",
			store.ErrInvalidRequest,
			firstNonEmptyString(eligibility.LifecycleStatus, execution.ExecutionLifecyclePending),
			eligibility.EligibilityReason,
		)
	}
	if !eligibility.LearningEligible {
		return fmt.Errorf(
			"%w: execution fidelity verdict %s is not eligible for successful completion (reason=%s)",
			store.ErrInvalidRequest,
			firstNonEmptyString(eligibility.FidelityVerdict, execution.ExecutionVerdictUnverified),
			eligibility.EligibilityReason,
		)
	}
	return nil
}

func isRealTrainingRunner(runner string) bool {
	return !strings.EqualFold(strings.TrimSpace(runner), "local_simulator")
}
