package api

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/diagnostics"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
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
	for _, job := range recovered {
		if job.Status == jobs.StatusFailed {
			failedCount++
			s.handleRecoveredExpiredLeaseFailure(job)
			continue
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

	planID := jobConfigString(job.Config, "plan_id")
	payload := map[string]any{
		"job_id":          job.ID,
		"worker_id":       job.WorkerID,
		"template":        job.Template,
		"attempt":         job.Attempt,
		"max_attempts":    job.MaxAttempts,
		"error":           job.Error,
		"recovery_reason": "expired_lease",
	}
	if _, err := s.store.CreateExecutionEvent(
		job.ProjectID,
		planID,
		execution.EventExecutionFailed,
		fmt.Sprintf("Job %s failed after its worker lease expired and attempts were exhausted.", job.ID),
		payload,
	); err != nil {
		log.Printf("record expired lease failure event failed for job %s: %v", job.ID, err)
		diagnostics.Event("warn", "job_lease_recovery_event_failed", map[string]any{
			"job_id":     job.ID,
			"project_id": job.ProjectID,
			"error":      err.Error(),
		})
	}
	if job.Template == jobs.TemplateTrainExperiment {
		s.enqueueTrainingTerminalHooks(job)
	}
}
