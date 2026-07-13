package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/workers"
)

func (s *PostgresStore) PollJob(workerID string, filter JobPollFilter) (*jobs.ExperimentJob, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	worker, err := scanWorker(tx.QueryRowContext(ctx, `
		SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
		FROM workers
		WHERE id = $1
	`, workerID))
	if err != nil {
		return nil, err
	}

	now := time.Now().UTC()
	if _, err := s.recoverExpiredJobLeasesTx(ctx, tx, now); err != nil {
		return nil, err
	}
	worker, err = scanWorker(tx.QueryRowContext(ctx, `
		SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
		FROM workers
		WHERE id = $1
	`, workerID))
	if err != nil {
		return nil, err
	}

	if worker.CurrentJobID != "" {
		job, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id")+" FOR UPDATE", worker.CurrentJobID))
		if err != nil {
			return nil, err
		}
		lockedWorker, err := scanWorker(tx.QueryRowContext(ctx, `
			SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
			FROM workers
			WHERE id = $1
			FOR UPDATE
		`, workerID))
		if err != nil {
			return nil, err
		}
		if lockedWorker.CurrentJobID != job.ID {
			if err := tx.Commit(); err != nil {
				return nil, err
			}
			return s.PollJob(workerID, filter)
		}
		worker = lockedWorker
		if _, err := tx.ExecContext(ctx, `
			UPDATE workers
			SET last_heartbeat = now()
			WHERE id = $1
		`, workerID); err != nil {
			return nil, err
		}

		if !isTerminalJobStatus(job.Status) {
			leaseExpiresAt := now.Add(defaultJobLeaseDuration)
			job, err = scanJob(tx.QueryRowContext(ctx, `
				UPDATE experiment_jobs
				SET lease_owner_worker_id = $1, lease_last_heartbeat_at = now(), lease_expires_at = $2
				WHERE id = $3
				RETURNING `+jobSelectColumns()+`
			`, workerID, leaseExpiresAt, job.ID))
			if err != nil {
				return nil, err
			}
		}

		if err := tx.Commit(); err != nil {
			return nil, err
		}

		return &job, nil
	}

	if worker.ProjectID == "" {
		if _, updateErr := tx.ExecContext(ctx, `
			UPDATE workers
			SET last_heartbeat = now()
			WHERE id = $1
		`, workerID); updateErr != nil {
			return nil, updateErr
		}
		if commitErr := tx.Commit(); commitErr != nil {
			return nil, commitErr
		}
		return nil, ErrNoJob
	}

	query, args := pollJobCandidateQuery(worker.ProjectID, filter)
	rows, err := tx.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}

	var job jobs.ExperimentJob
	foundJob := false
	for rows.Next() {
		candidate, scanErr := scanJob(rows)
		if scanErr != nil {
			rows.Close()
			return nil, scanErr
		}
		if !filter.Matches(candidate) {
			continue
		}
		job = candidate
		foundJob = true
		break
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return nil, err
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}
	if !foundJob {
		if _, updateErr := tx.ExecContext(ctx, `
			UPDATE workers
			SET last_heartbeat = now()
			WHERE id = $1
		`, workerID); updateErr != nil {
			return nil, updateErr
		}

		if commitErr := tx.Commit(); commitErr != nil {
			return nil, commitErr
		}

		return nil, ErrNoJob
	}
	if !filter.Matches(job) {
		if _, updateErr := tx.ExecContext(ctx, `
			UPDATE workers
			SET last_heartbeat = now()
			WHERE id = $1
		`, workerID); updateErr != nil {
			return nil, updateErr
		}
		if commitErr := tx.Commit(); commitErr != nil {
			return nil, commitErr
		}
		return nil, ErrNoJob
	}
	lockedWorker, err := scanWorker(tx.QueryRowContext(ctx, `
		SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
		FROM workers
		WHERE id = $1
		FOR UPDATE
	`, workerID))
	if err != nil {
		return nil, err
	}
	if lockedWorker.CurrentJobID != "" || lockedWorker.ProjectID != worker.ProjectID {
		if err := tx.Commit(); err != nil {
			return nil, err
		}
		return s.PollJob(workerID, filter)
	}
	worker = lockedWorker

	assignedConfig := jobConfigWithActiveAttempt(job.Config, job.ID, job.Attempt+1)
	assignedConfigJSON, err := json.Marshal(assignedConfig)
	if err != nil {
		return nil, fmt.Errorf("marshal active attempt job config: %w", err)
	}

	assignedJob, err := scanJob(tx.QueryRowContext(ctx, `
		UPDATE experiment_jobs
		SET worker_id = $1,
			status = $2,
			error = '',
			started_at = now(),
			attempt = attempt + 1,
			config = $5,
			lease_owner_worker_id = $1,
			lease_last_heartbeat_at = now(),
			lease_expires_at = $4
		WHERE id = $3
		RETURNING `+jobSelectColumns()+`
	`, workerID, jobs.StatusAssigned, job.ID, now.Add(defaultJobLeaseDuration), assignedConfigJSON))
	if err != nil {
		return nil, err
	}
	if _, err := createAttemptExecutionRecordTx(ctx, tx, assignedJob.ID, jobAttemptID(assignedJob.ID, assignedJob.Attempt), assignedJob.Attempt); err != nil && !errors.Is(normalizeSQLError(err), ErrNotFound) {
		return nil, err
	}
	transition, progress := newJobLifecycleTransition(
		assignedJob,
		execution.TransitionJobAssigned,
		assignedJob.Attempt,
		jobs.ProgressStageWorkerStarting,
		jobs.ProgressStatusRunning,
		progressRevisionAssigned,
		"worker_assigned",
		"worker_assignment",
	)
	if _, _, _, err := commitJobLifecycleTx(ctx, tx, assignedJob, progress, transition, now); err != nil {
		return nil, err
	}

	if _, err := tx.ExecContext(ctx, `
		UPDATE workers
		SET status = $1, current_job_id = $2, last_heartbeat = now()
		WHERE id = $3
	`, workers.StatusRunning, assignedJob.ID, workerID); err != nil {
		return nil, err
	}

	if err := tx.Commit(); err != nil {
		return nil, err
	}

	return &assignedJob, nil
}

func pollJobCandidateQuery(projectID string, filter JobPollFilter) (string, []any) {
	args := []any{jobs.StatusQueued, projectID}
	clauses := []string{"status = $1", "project_id = $2"}

	if templates := normalizedPollValues(filter.Templates); len(templates) > 0 {
		placeholders := make([]string, 0, len(templates))
		for _, template := range templates {
			args = append(args, template)
			placeholders = append(placeholders, fmt.Sprintf("$%d", len(args)))
		}
		clauses = append(clauses, "lower(template) IN ("+strings.Join(placeholders, ",")+")")
	}

	if provider := strings.ToLower(strings.TrimSpace(filter.Provider)); provider != "" {
		args = append(args, provider)
		providerPlaceholder := fmt.Sprintf("$%d", len(args))
		providerClause := "lower(coalesce(config->>'provider', '')) = " + providerPlaceholder
		if fallbackTemplates := normalizedPollValues(filter.IncludeUnspecifiedProviderTemplates); len(fallbackTemplates) > 0 {
			placeholders := make([]string, 0, len(fallbackTemplates))
			for _, template := range fallbackTemplates {
				args = append(args, template)
				placeholders = append(placeholders, fmt.Sprintf("$%d", len(args)))
			}
			providerClause = "(" + providerClause + " OR (coalesce(config->>'provider', '') = '' AND lower(template) IN (" + strings.Join(placeholders, ",") + ")))"
		}
		clauses = append(clauses, providerClause)
	}

	query := `
		SELECT id, project_id, worker_id, template, status, config, mlflow_run_id, error, attempt, max_attempts, lease_owner_worker_id, lease_expires_at, lease_last_heartbeat_at, created_at, started_at, completed_at
		FROM experiment_jobs
		WHERE ` + strings.Join(clauses, " AND ") + `
		ORDER BY created_at
		FOR UPDATE SKIP LOCKED
		LIMIT 1
	`
	return query, args
}

func normalizedPollValues(values []string) []string {
	set := normalizedSet(values)
	out := make([]string, 0, len(set))
	for value := range set {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func postgresPageLimitOffset(options PageOptions) (int, int) {
	limit := options.Limit
	if limit <= 0 {
		limit = 100
	}
	offset := options.Offset
	if offset < 0 {
		offset = 0
	}
	return limit, offset
}

func (s *PostgresStore) CreateJob(projectID string, template string, config map[string]any) (jobs.ExperimentJob, error) {
	if err := s.requireProject(projectID); err != nil {
		return jobs.ExperimentJob{}, err
	}
	if err := s.requireDatasetConfig(projectID, config); err != nil {
		return jobs.ExperimentJob{}, err
	}

	configJSON, err := json.Marshal(config)
	if err != nil {
		return jobs.ExperimentJob{}, fmt.Errorf("marshal job config: %w", err)
	}

	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	defer tx.Rollback()
	const query = `
		INSERT INTO experiment_jobs (project_id, template, status, config, max_attempts)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, project_id, worker_id, template, status, config, mlflow_run_id, error, attempt, max_attempts, lease_owner_worker_id, lease_expires_at, lease_last_heartbeat_at, created_at, started_at, completed_at
	`

	job, err := scanJob(tx.QueryRowContext(ctx, query, projectID, template, jobs.StatusQueued, configJSON, defaultJobMaxAttempts))
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	pendingConfig := jobConfigWithPendingAttempt(job.Config, job.ID, 1)
	pendingConfigJSON, err := json.Marshal(pendingConfig)
	if err != nil {
		return jobs.ExperimentJob{}, fmt.Errorf("marshal pending attempt job config: %w", err)
	}
	job, err = scanJob(tx.QueryRowContext(ctx, `
		UPDATE experiment_jobs
		SET config = $1
		WHERE id = $2
		RETURNING `+jobSelectColumns()+`
	`, pendingConfigJSON, job.ID))
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	if spec, ok := executionSpecFromConfig(job.ID, projectID, job.Config, job.CreatedAt); ok {
		if err := insertJobExecutionSpecTx(ctx, tx, spec); err != nil {
			return jobs.ExperimentJob{}, err
		}
	}
	transition, progress := newJobLifecycleTransition(
		job,
		execution.TransitionJobQueued,
		1,
		jobs.ProgressStageQueued,
		jobs.ProgressStatusQueued,
		progressRevisionQueued,
		"initial_queue",
		"job_created",
	)
	if _, _, _, err := commitJobLifecycleTx(ctx, tx, job, progress, transition, job.CreatedAt); err != nil {
		return jobs.ExperimentJob{}, err
	}
	if err := tx.Commit(); err != nil {
		return jobs.ExperimentJob{}, err
	}
	return job, nil
}

func (s *PostgresStore) GetJob(id string) (jobs.ExperimentJob, error) {
	return scanJob(s.db.QueryRowContext(context.Background(), selectJobSQL("id"), id))
}

func (s *PostgresStore) ListProjectJobs(projectID string) ([]jobs.ExperimentJob, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, worker_id, template, status, config, mlflow_run_id, error, attempt, max_attempts, lease_owner_worker_id, lease_expires_at, lease_last_heartbeat_at, created_at, started_at, completed_at
		FROM experiment_jobs
		WHERE project_id = $1
		ORDER BY created_at DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []jobs.ExperimentJob{}
	for rows.Next() {
		job, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, job)
	}

	return out, rows.Err()
}

func (s *PostgresStore) ListProjectJobsPage(projectID string, options PageOptions) ([]jobs.ExperimentJob, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	limit, offset := postgresPageLimitOffset(options)
	const query = `
		SELECT id, project_id, worker_id, template, status, config, mlflow_run_id, error, attempt, max_attempts, lease_owner_worker_id, lease_expires_at, lease_last_heartbeat_at, created_at, started_at, completed_at
		FROM experiment_jobs
		WHERE project_id = $1
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []jobs.ExperimentJob{}
	for rows.Next() {
		job, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, job)
	}
	return out, rows.Err()
}

func (s *PostgresStore) UpdateJobConfig(jobID string, patch map[string]any) (jobs.ExperimentJob, error) {
	if _, changesAcceptedSpec := patch[execution.ExecutionSpecConfigKey]; changesAcceptedSpec {
		return jobs.ExperimentJob{}, fmt.Errorf("%w: %s is immutable after scheduling", ErrInvalidRequest, execution.ExecutionSpecConfigKey)
	}
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	defer tx.Rollback()

	job, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id")+" FOR UPDATE", jobID))
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	next := copyAnyMap(job.Config)
	if next == nil {
		next = map[string]any{}
	}
	for key, value := range patch {
		next[key] = value
	}
	configJSON, err := json.Marshal(next)
	if err != nil {
		return jobs.ExperimentJob{}, fmt.Errorf("marshal patched job config: %w", err)
	}
	updated, err := scanJob(tx.QueryRowContext(ctx, `
		UPDATE experiment_jobs
		SET config = $1
		WHERE id = $2
		RETURNING `+jobSelectColumns()+`
	`, configJSON, jobID))
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	if err := tx.Commit(); err != nil {
		return jobs.ExperimentJob{}, err
	}
	return updated, nil
}

func (s *PostgresStore) RecoverExpiredJobLeases(now time.Time) ([]jobs.ExperimentJob, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()

	recovered, err := s.recoverExpiredJobLeasesTx(ctx, tx, now.UTC())
	if err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return recovered, nil
}

func (s *PostgresStore) recoverExpiredJobLeasesTx(ctx context.Context, tx *sql.Tx, now time.Time) ([]jobs.ExperimentJob, error) {
	rows, err := tx.QueryContext(ctx, `
		SELECT `+jobSelectColumns()+`
		FROM experiment_jobs
		WHERE status IN ($1, $2)
			AND lease_expires_at IS NOT NULL
			AND lease_expires_at <= $3
		ORDER BY lease_expires_at ASC
		FOR UPDATE
	`, jobs.StatusAssigned, jobs.StatusRunning, now)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	expired := []jobs.ExperimentJob{}
	for rows.Next() {
		job, err := scanJob(rows)
		if err != nil {
			return nil, err
		}
		expired = append(expired, job)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	recovered := []jobs.ExperimentJob{}
	for _, job := range expired {
		if job.MaxAttempts < 1 {
			job.MaxAttempts = defaultJobMaxAttempts
		}
		previousConfig := copyAnyMap(job.Config)
		if _, err := tx.ExecContext(ctx, `UPDATE attempt_execution_records SET lifecycle_status=$1, updated_at=now() WHERE job_id=$2 AND attempt_id=$3 AND lifecycle_status=$4`, execution.ExecutionLifecycleNotRealized, job.ID, jobAttemptID(job.ID, job.Attempt), execution.ExecutionLifecyclePending); err != nil {
			return nil, err
		}
		var updated jobs.ExperimentJob
		if job.Attempt >= job.MaxAttempts {
			nextConfig := jobConfigWithTerminalAttempt(job.Config, job.ID, job.Attempt)
			configJSON, marshalErr := json.Marshal(nextConfig)
			if marshalErr != nil {
				return nil, fmt.Errorf("marshal expired terminal job config: %w", marshalErr)
			}
			updated, err = scanJob(tx.QueryRowContext(ctx, `
				UPDATE experiment_jobs
				SET status = $1,
					error = $2,
					worker_id = '',
					config = $5,
					completed_at = $3,
					lease_owner_worker_id = '',
					lease_expires_at = NULL,
					lease_last_heartbeat_at = NULL
				WHERE id = $4
				RETURNING `+jobSelectColumns()+`
			`, jobs.StatusFailed, "job lease expired after maximum attempts", now, job.ID, configJSON))
		} else {
			nextConfig := jobConfigWithPendingAttempt(job.Config, job.ID, job.Attempt+1)
			configJSON, marshalErr := json.Marshal(nextConfig)
			if marshalErr != nil {
				return nil, fmt.Errorf("marshal expired retry job config: %w", marshalErr)
			}
			updated, err = scanJob(tx.QueryRowContext(ctx, `
				UPDATE experiment_jobs
				SET status = $1,
					error = '',
					worker_id = '',
					config = $3,
					started_at = NULL,
					lease_owner_worker_id = '',
					lease_expires_at = NULL,
					lease_last_heartbeat_at = NULL
				WHERE id = $2
				RETURNING `+jobSelectColumns()+`
			`, jobs.StatusQueued, job.ID, configJSON))
		}
		if err != nil {
			return nil, err
		}
		if _, err := tx.ExecContext(ctx, `
			UPDATE workers
			SET status = CASE WHEN status = $3 THEN status ELSE $1 END,
				current_job_id = ''
			WHERE current_job_id = $2
		`, workers.StatusIdle, job.ID, workers.StatusOffline); err != nil {
			return nil, err
		}
		if err := closeRemoteTrainingSessionForJobConfigTx(ctx, tx, previousConfig, runs.RemoteTrainingSessionStatusExpired, now); err != nil {
			return nil, err
		}
		if updated.Status == jobs.StatusFailed {
			transition, progress := newJobLifecycleTransition(
				updated,
				execution.TransitionJobFailed,
				activeJobProgressAttempt(updated),
				jobs.ProgressStageFailed,
				jobs.ProgressStatusFailed,
				progressRevisionTerminal,
				"lease_attempts_exhausted",
				"lease_attempts_exhausted",
			)
			if _, _, _, err := commitJobLifecycleTx(ctx, tx, updated, progress, transition, now); err != nil {
				return nil, err
			}
		} else {
			attempt := activeJobProgressAttempt(updated)
			transition, progress := newJobLifecycleTransition(
				updated,
				execution.TransitionJobLeaseRecovered,
				attempt,
				jobs.ProgressStageQueued,
				jobs.ProgressStatusQueued,
				progressRevisionQueued,
				"lease_recovered",
				"lease_expired",
			)
			if _, _, _, err := commitJobLifecycleTx(ctx, tx, updated, progress, transition, now); err != nil {
				return nil, err
			}
		}
		recovered = append(recovered, updated)
	}

	return recovered, nil
}

func (s *PostgresStore) ReportMetric(jobID string, epoch int, values map[string]float64) (jobs.EpochMetric, error) {
	if epoch < 1 {
		return jobs.EpochMetric{}, fmt.Errorf("%w: epoch must be positive", ErrInvalidRequest)
	}
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return jobs.EpochMetric{}, err
	}
	defer tx.Rollback()

	job, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id")+" FOR UPDATE", jobID))
	if err != nil {
		return jobs.EpochMetric{}, err
	}

	now := time.Now().UTC()
	becameRunning := job.Status == jobs.StatusAssigned
	if becameRunning {
		if _, err := tx.ExecContext(ctx, `
			UPDATE experiment_jobs
			SET status = $1
			WHERE id = $2
		`, jobs.StatusRunning, jobID); err != nil {
			return jobs.EpochMetric{}, err
		}
		job.Status = jobs.StatusRunning
	}

	if job.WorkerID != "" && (job.Status == jobs.StatusAssigned || job.Status == jobs.StatusRunning) {
		if _, err := tx.ExecContext(ctx, `
			UPDATE workers
			SET status = $1, last_heartbeat = now()
			WHERE id = $2
		`, workers.StatusRunning, job.WorkerID); err != nil {
			return jobs.EpochMetric{}, err
		}
		if _, err := tx.ExecContext(ctx, `
			UPDATE experiment_jobs
			SET lease_owner_worker_id = $1, lease_last_heartbeat_at = now(), lease_expires_at = $2
			WHERE id = $3
		`, job.WorkerID, now.Add(defaultJobLeaseDuration), jobID); err != nil {
			return jobs.EpochMetric{}, err
		}
	}
	if becameRunning {
		transition, progress := newJobLifecycleTransition(
			job,
			execution.TransitionJobRunning,
			activeJobProgressAttempt(job),
			jobs.ProgressStageTraining,
			jobs.ProgressStatusRunning,
			progressRevisionForEpoch(epoch),
			"first_metric",
			"first_metric",
		)
		current := int64(epoch)
		progress.Current = &current
		progress.Unit = "epoch"
		if _, _, _, err := commitJobLifecycleTx(ctx, tx, job, progress, transition, now); err != nil {
			return jobs.EpochMetric{}, err
		}
	} else if job.Status == jobs.StatusRunning {
		_, progress := newJobLifecycleTransition(
			job,
			execution.TransitionJobRunning,
			activeJobProgressAttempt(job),
			jobs.ProgressStageTraining,
			jobs.ProgressStatusRunning,
			progressRevisionForEpoch(epoch),
			"metric_reported",
			"metric_reported",
		)
		current := int64(epoch)
		progress.Current = &current
		progress.Unit = "epoch"
		if _, err := upsertJobProgressTx(ctx, tx, job, progress, now); err != nil {
			return jobs.EpochMetric{}, err
		}
	}

	metricsJSON, err := json.Marshal(values)
	if err != nil {
		return jobs.EpochMetric{}, fmt.Errorf("marshal metrics: %w", err)
	}

	metric, err := scanMetric(tx.QueryRowContext(ctx, `
		INSERT INTO epoch_metrics (job_id, epoch, metrics)
		VALUES ($1, $2, $3)
		ON CONFLICT (job_id, epoch)
		DO UPDATE SET metrics = EXCLUDED.metrics, created_at = now()
		RETURNING job_id, epoch, metrics, created_at
	`, jobID, epoch, metricsJSON))
	if err != nil {
		return jobs.EpochMetric{}, err
	}

	if err := tx.Commit(); err != nil {
		return jobs.EpochMetric{}, err
	}

	return metric, nil
}

func (s *PostgresStore) ListJobMetrics(jobID string) ([]jobs.EpochMetric, error) {
	if _, err := s.GetJob(jobID); err != nil {
		return nil, err
	}

	const query = `
		SELECT job_id, epoch, metrics, created_at
		FROM epoch_metrics
		WHERE job_id = $1
		ORDER BY epoch
	`

	rows, err := s.db.QueryContext(context.Background(), query, jobID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []jobs.EpochMetric{}
	for rows.Next() {
		metric, err := scanMetric(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, metric)
	}

	return out, rows.Err()
}

func (s *PostgresStore) ListJobMetricsPage(jobID string, options PageOptions) ([]jobs.EpochMetric, error) {
	if _, err := s.GetJob(jobID); err != nil {
		return nil, err
	}
	limit, offset := postgresPageLimitOffset(options)
	const query = `
		SELECT job_id, epoch, metrics, created_at
		FROM (
			SELECT job_id, epoch, metrics, created_at
			FROM epoch_metrics
			WHERE job_id = $1
			ORDER BY epoch DESC
			LIMIT $2 OFFSET $3
		) recent
		ORDER BY epoch
	`
	rows, err := s.db.QueryContext(context.Background(), query, jobID, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []jobs.EpochMetric{}
	for rows.Next() {
		metric, err := scanMetric(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, metric)
	}
	return out, rows.Err()
}
