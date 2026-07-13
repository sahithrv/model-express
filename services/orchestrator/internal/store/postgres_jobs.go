package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/workers"
)

func (s *PostgresStore) CompleteJob(jobID string, mlflowRunID string) (jobs.ExperimentJob, error) {
	return s.finishJob(jobID, jobs.StatusSucceeded, mlflowRunID, "")
}

func (s *PostgresStore) RetryJob(jobID string, message string, options RetryJobOptions) (jobs.ExperimentJob, bool, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return jobs.ExperimentJob{}, false, err
	}
	defer tx.Rollback()

	job, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id")+" FOR UPDATE", jobID))
	if err != nil {
		return jobs.ExperimentJob{}, false, err
	}
	if isTerminalJobStatus(job.Status) {
		if err := clearWorkersForJobTx(ctx, tx, job.ID); err != nil {
			return jobs.ExperimentJob{}, false, err
		}
		return job, false, tx.Commit()
	}
	if job.MaxAttempts < 1 {
		job.MaxAttempts = defaultJobMaxAttempts
	}

	now := time.Now().UTC()
	requeued := job.Attempt < job.MaxAttempts && !options.ForceFail
	terminalAttempt := activeJobProgressAttempt(job)
	previousConfig := copyAnyMap(job.Config)
	if _, err := tx.ExecContext(ctx, `UPDATE attempt_execution_records SET lifecycle_status=$1, updated_at=now() WHERE job_id=$2 AND attempt_id=$3 AND lifecycle_status=$4`, execution.ExecutionLifecycleNotRealized, job.ID, jobAttemptID(job.ID, job.Attempt), execution.ExecutionLifecyclePending); err != nil {
		return jobs.ExperimentJob{}, false, err
	}
	nextConfig := copyAnyMap(job.Config)
	if options.Config != nil {
		nextConfig = copyAnyMap(options.Config)
	}
	nextConfig = jobConfigWithImmutableExecutionSpec(job.Config, nextConfig)
	if requeued {
		nextConfig = jobConfigWithPendingAttempt(nextConfig, job.ID, job.Attempt+1)
		configJSON, marshalErr := json.Marshal(nextConfig)
		if marshalErr != nil {
			return jobs.ExperimentJob{}, false, fmt.Errorf("marshal retry job config: %w", marshalErr)
		}
		job, err = scanJob(tx.QueryRowContext(ctx, `
			UPDATE experiment_jobs
			SET status = $1,
				error = $2,
				worker_id = '',
				mlflow_run_id = '',
				config = $4,
				started_at = NULL,
				completed_at = NULL,
				lease_owner_worker_id = '',
				lease_expires_at = NULL,
				lease_last_heartbeat_at = NULL
			WHERE id = $3
			RETURNING `+jobSelectColumns()+`
		`, jobs.StatusQueued, message, jobID, configJSON))
	} else {
		nextConfig = jobConfigWithTerminalAttempt(nextConfig, job.ID, terminalAttempt)
		configJSON, marshalErr := json.Marshal(nextConfig)
		if marshalErr != nil {
			return jobs.ExperimentJob{}, false, fmt.Errorf("marshal terminal retry job config: %w", marshalErr)
		}
		job, err = scanJob(tx.QueryRowContext(ctx, `
			UPDATE experiment_jobs
			SET status = $1,
				error = $2,
				config = $4,
				completed_at = now(),
				lease_owner_worker_id = '',
				lease_expires_at = NULL,
				lease_last_heartbeat_at = NULL
			WHERE id = $3
			RETURNING `+jobSelectColumns()+`
		`, jobs.StatusFailed, message, jobID, configJSON))
	}
	if err != nil {
		return jobs.ExperimentJob{}, false, err
	}

	if _, err := tx.ExecContext(ctx, `
		UPDATE workers
		SET status = CASE WHEN status = $3 THEN status ELSE $1 END,
			current_job_id = '',
			last_heartbeat = now()
		WHERE current_job_id = $2
	`, workers.StatusIdle, jobID, workers.StatusOffline); err != nil {
		return jobs.ExperimentJob{}, false, err
	}
	if requeued {
		err = closeRemoteTrainingSessionForJobConfigTx(ctx, tx, previousConfig, runs.RemoteTrainingSessionStatusExpired, now)
	} else {
		err = closeRemoteTrainingSessionForJobConfigTx(ctx, tx, previousConfig, runs.RemoteTrainingSessionStatusFailed, now)
	}
	if err != nil {
		return jobs.ExperimentJob{}, false, err
	}
	if requeued {
		transition, progress := newJobLifecycleTransition(
			job,
			execution.TransitionJobRetryQueued,
			terminalAttempt,
			jobs.ProgressStageQueued,
			jobs.ProgressStatusQueued,
			progressRevisionQueued,
			"retry_queued",
			"retryable_failure",
		)
		if _, _, _, err := commitJobLifecycleTx(ctx, tx, job, progress, transition, now); err != nil {
			return jobs.ExperimentJob{}, false, err
		}
	} else {
		transition, progress := newJobLifecycleTransition(
			job,
			execution.TransitionJobFailed,
			activeJobProgressAttempt(job),
			jobs.ProgressStageFailed,
			jobs.ProgressStatusFailed,
			progressRevisionTerminal,
			"attempts_exhausted",
			"attempts_exhausted",
		)
		if _, _, _, err := commitJobLifecycleTx(ctx, tx, job, progress, transition, now); err != nil {
			return jobs.ExperimentJob{}, false, err
		}
	}

	if err := tx.Commit(); err != nil {
		return jobs.ExperimentJob{}, false, err
	}

	return job, requeued, nil
}

func (s *PostgresStore) FailJob(jobID string, message string) (jobs.ExperimentJob, error) {
	return s.finishJob(jobID, jobs.StatusFailed, "", message)
}

func (s *PostgresStore) CancelJob(jobID string, message string, configPatch map[string]any) (jobs.ExperimentJob, error) {
	return s.finishJobWithTransition(jobID, jobs.StatusFailed, "", message, execution.TransitionJobCancelled, configPatch)
}

func (s *PostgresStore) finishJob(jobID string, status string, mlflowRunID string, message string) (jobs.ExperimentJob, error) {
	transition := execution.TransitionJobFailed
	if status == jobs.StatusSucceeded {
		transition = execution.TransitionJobCompleted
	}
	return s.finishJobWithTransition(jobID, status, mlflowRunID, message, transition, nil)
}

func (s *PostgresStore) finishJobWithTransition(jobID string, status string, mlflowRunID string, message string, transition execution.ExecutionTransition, configPatch map[string]any) (jobs.ExperimentJob, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	defer tx.Rollback()

	current, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id")+" FOR UPDATE", jobID))
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	if isTerminalJobStatus(current.Status) {
		if err := clearWorkersForJobTx(ctx, tx, current.ID); err != nil {
			return jobs.ExperimentJob{}, err
		}
		if err := tx.Commit(); err != nil {
			return jobs.ExperimentJob{}, err
		}
		return current, nil
	}
	terminalAttempt := activeJobProgressAttempt(current)
	nextConfig := copyAnyMap(current.Config)
	for key, value := range configPatch {
		nextConfig[key] = value
	}
	nextConfig = jobConfigWithTerminalAttempt(nextConfig, current.ID, terminalAttempt)
	previousConfig := copyAnyMap(current.Config)
	if _, err := tx.ExecContext(ctx, `UPDATE attempt_execution_records SET lifecycle_status=$1, updated_at=now() WHERE job_id=$2 AND attempt_id=$3 AND lifecycle_status=$4`, execution.ExecutionLifecycleNotRealized, current.ID, jobAttemptID(current.ID, current.Attempt), execution.ExecutionLifecyclePending); err != nil {
		return jobs.ExperimentJob{}, err
	}
	configJSON, err := json.Marshal(nextConfig)
	if err != nil {
		return jobs.ExperimentJob{}, fmt.Errorf("marshal terminal job config: %w", err)
	}

	job, err := scanJob(tx.QueryRowContext(ctx, `
		UPDATE experiment_jobs
		SET status = $1,
			mlflow_run_id = $2,
			error = $3,
			config = $5,
			completed_at = now(),
			lease_owner_worker_id = '',
			lease_expires_at = NULL,
			lease_last_heartbeat_at = NULL
		WHERE id = $4
		RETURNING `+jobSelectColumns()+`
	`, status, mlflowRunID, message, jobID, configJSON))
	if err != nil {
		return jobs.ExperimentJob{}, err
	}

	if job.WorkerID != "" {
		if _, err := tx.ExecContext(ctx, `
			UPDATE workers
			SET status = $1, current_job_id = '', last_heartbeat = now()
			WHERE id = $2
		`, workers.StatusIdle, job.WorkerID); err != nil {
			return jobs.ExperimentJob{}, err
		}
	}
	now := time.Now().UTC()
	if err := closeRemoteTrainingSessionForJobConfigTx(ctx, tx, previousConfig, status, now); err != nil {
		return jobs.ExperimentJob{}, err
	}
	stage := jobs.ProgressStageFailed
	progressStatus := jobs.ProgressStatusFailed
	detailCode := "backend_failure"
	reasonCode := "backend_failure"
	if transition == execution.TransitionJobCompleted {
		stage = jobs.ProgressStageCompleted
		progressStatus = jobs.ProgressStatusCompleted
		detailCode = "backend_completion"
		reasonCode = "backend_completion"
	} else if transition == execution.TransitionJobCancelled {
		stage = jobs.ProgressStageCancelled
		progressStatus = jobs.ProgressStatusCancelled
		detailCode = "user_cancelled"
		reasonCode = "user_cancelled"
	}
	transitionInput, progress := newJobLifecycleTransition(
		job,
		transition,
		terminalAttempt,
		stage,
		progressStatus,
		progressRevisionTerminal,
		detailCode,
		reasonCode,
	)
	if _, _, _, err := commitJobLifecycleTx(ctx, tx, job, progress, transitionInput, now); err != nil {
		return jobs.ExperimentJob{}, err
	}

	if err := tx.Commit(); err != nil {
		return jobs.ExperimentJob{}, err
	}

	return job, nil
}

func clearWorkersForJobTx(ctx context.Context, tx *sql.Tx, jobID string) error {
	_, err := tx.ExecContext(ctx, `
		UPDATE workers
		SET status = CASE WHEN status = $3 THEN status ELSE $1 END,
			current_job_id = '',
			last_heartbeat = now()
		WHERE current_job_id = $2
	`, workers.StatusIdle, jobID, workers.StatusOffline)
	return err
}
