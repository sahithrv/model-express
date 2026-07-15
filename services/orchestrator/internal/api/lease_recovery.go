package api

import (
	"errors"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/diagnostics"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

const (
	defaultLeaseRecoveryInterval = 60 * time.Second
	leaseRecoveryIntervalEnv     = "MODEL_EXPRESS_LEASE_RECOVERY_INTERVAL_SECONDS"
)

func (s *Server) startLeaseRecovery() {
	interval := leaseRecoveryIntervalFromEnv()
	if interval == 0 {
		return
	}

	if _, err := s.recoverExpiredLeasesOnce(time.Now().UTC()); err != nil {
		log.Printf("startup lease recovery failed: %v", err)
	}

	go s.runLeaseRecoveryLoop(interval)
}

func (s *Server) runLeaseRecoveryLoop(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		if _, err := s.recoverExpiredLeasesOnce(time.Now().UTC()); err != nil {
			log.Printf("lease recovery failed: %v", err)
		}
	}
}

func leaseRecoveryIntervalFromEnv() time.Duration {
	value := strings.TrimSpace(os.Getenv(leaseRecoveryIntervalEnv))
	if value == "" {
		return defaultLeaseRecoveryInterval
	}
	seconds, err := strconv.Atoi(value)
	if err != nil || seconds < 0 {
		return defaultLeaseRecoveryInterval
	}
	if seconds == 0 {
		return 0
	}
	return time.Duration(seconds) * time.Second
}

func (s *Server) recoverExpiredLeasesOnce(now time.Time) ([]jobs.ExperimentJob, error) {
	recovered, err := s.store.RecoverExpiredJobLeases(now.UTC())
	if err != nil {
		diagnostics.Event("error", "job_lease_recovery_failed", map[string]any{
			"error": err.Error(),
		})
		return nil, err
	}
	if len(recovered) == 0 {
		return recovered, nil
	}

	requeuedCount := 0
	failedCount := 0
	for index, job := range recovered {
		if job.Status == jobs.StatusFailed {
			failedCount++
			s.handleRecoveredExpiredLeaseFailure(job)
			continue
		}
		if policyControlledJobTemplate(job.Template) {
			reconciled, reconcileErr := s.reconcileRecoveredJobPolicy(job)
			if reconcileErr != nil {
				return recovered, reconcileErr
			}
			job = reconciled
			recovered[index] = reconciled
		}
		requeuedCount++
	}

	diagnostics.Event("warn", "job_lease_recovery", map[string]any{
		"recovered_count": len(recovered),
		"requeued_count":  requeuedCount,
		"failed_count":    failedCount,
		"job_ids":         experimentJobIDs(recovered),
	})
	return recovered, nil
}

func (s *Server) reconcileRecoveredJobPolicy(job jobs.ExperimentJob) (jobs.ExperimentJob, error) {
	for attempt := 0; attempt < 3; attempt++ {
		evaluation, evaluateErr := s.evaluateJobPolicy(job.ProjectID, job.ID, job.Template, job.Config, policyOperationRequeueRun)
		if evaluation.EffectivePolicyHash == "" {
			return job, evaluateErr
		}
		updated, created, applied, err := s.store.ApplyQueuedJobPolicyEvaluation(job.ID, evaluation)
		if errors.Is(err, store.ErrPolicyChanged) {
			continue
		}
		if err != nil {
			return job, err
		}
		if !applied {
			return updated, nil
		}
		if evaluateErr != nil || created.Decision == policies.DecisionDenied {
			s.recordJobPolicyActivity(updated, created, execution.EventJobPolicyBlocked, "Lease-recovered job blocked by the current experiment policy.")
		} else {
			s.recordJobPolicyActivity(updated, created, execution.EventJobPolicyReconciled, "Lease-recovered job revalidated against the current experiment policy.")
		}
		return updated, nil
	}
	return job, store.ErrPolicyChanged
}

func (s *Server) handleRecoveredExpiredLeaseFailure(job jobs.ExperimentJob) {
	if job.Template == jobs.TemplateTrainExperiment {
		if _, err := s.store.UpsertTrainingRunSummary(job.ID, runs.TrainingRunSummaryUpdate{
			Status: jobs.StatusFailed,
		}); err != nil {
			log.Printf("lease recovery training summary update failed for job %s: %v", job.ID, err)
			diagnostics.Event("warn", "job_lease_recovery_summary_failed", map[string]any{
				"job_id":     job.ID,
				"project_id": job.ProjectID,
				"error":      err.Error(),
			})
		}
		s.updateWorkerRequirementDemandAfterTerminalJob(job)
	}

	if job.Template == jobs.TemplateTrainExperiment {
		s.enqueueTrainingTerminalHooks(job)
	} else {
		s.finalizeCandidateOutcomesAfterNonTrainingJob(job)
	}
}
