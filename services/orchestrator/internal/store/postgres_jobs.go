package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

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

	requeued := job.Attempt < job.MaxAttempts && !options.ForceFail
	previousConfig := copyAnyMap(job.Config)
	nextConfig := copyAnyMap(job.Config)
	if options.Config != nil {
		nextConfig = copyAnyMap(options.Config)
	}
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
		nextConfig = jobConfigWithTerminalAttempt(nextConfig, job.ID, job.Attempt)
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
		err = closeRemoteTrainingSessionForJobConfigTx(ctx, tx, previousConfig, runs.RemoteTrainingSessionStatusExpired, time.Now().UTC())
	} else {
		err = closeRemoteTrainingSessionForJobConfigTx(ctx, tx, previousConfig, runs.RemoteTrainingSessionStatusFailed, time.Now().UTC())
	}
	if err != nil {
		return jobs.ExperimentJob{}, false, err
	}

	if err := tx.Commit(); err != nil {
		return jobs.ExperimentJob{}, false, err
	}

	return job, requeued, nil
}

func (s *PostgresStore) FailJob(jobID string, message string) (jobs.ExperimentJob, error) {
	return s.finishJob(jobID, jobs.StatusFailed, "", message)
}

func (s *PostgresStore) finishJob(jobID string, status string, mlflowRunID string, message string) (jobs.ExperimentJob, error) {
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
	nextConfig := jobConfigWithTerminalAttempt(current.Config, current.ID, current.Attempt)
	previousConfig := copyAnyMap(current.Config)
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
	if err := closeRemoteTrainingSessionForJobConfigTx(ctx, tx, previousConfig, status, time.Now().UTC()); err != nil {
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
