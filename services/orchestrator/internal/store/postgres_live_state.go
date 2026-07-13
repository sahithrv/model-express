package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// GetProjectLiveState reads the global cursor first and every snapshot row
// from the same repeatable-read snapshot. A transition absent from this view
// necessarily commits with a cursor greater than SnapshotCursor, while a
// visible lifecycle/progress transition committed its event no later than the
// cursor read. This is the snapshot/stream no-gap boundary.
func (s *PostgresStore) GetProjectLiveState(ctx context.Context, projectID string) (ProjectLiveStateSnapshot, error) {
	tx, err := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	if err != nil {
		return ProjectLiveStateSnapshot{}, err
	}
	defer tx.Rollback()

	snapshot := ProjectLiveStateSnapshot{ProjectID: projectID}
	if err := tx.QueryRowContext(ctx, `
		SELECT last_sequence, transaction_timestamp()
		FROM execution_event_sequence_state
		WHERE id = 1
	`).Scan(&snapshot.SnapshotCursor, &snapshot.ObservedAt); err != nil {
		return ProjectLiveStateSnapshot{}, normalizeSQLError(err)
	}

	if err := scanProjectLiveStateJobCounts(ctx, tx, projectID, &snapshot.Jobs); err != nil {
		return ProjectLiveStateSnapshot{}, err
	}
	if err := scanProjectLiveStateWorkerCounts(ctx, tx, projectID, snapshot.ObservedAt, &snapshot.Workers, &snapshot.LastWorkerHeartbeat); err != nil {
		return ProjectLiveStateSnapshot{}, err
	}
	if err := scanProjectLiveStateRequirementCounts(ctx, tx, projectID, &snapshot.Requirements); err != nil {
		return ProjectLiveStateSnapshot{}, err
	}
	progress, total, err := listProjectLiveStateProgress(ctx, tx, projectID)
	if err != nil {
		return ProjectLiveStateSnapshot{}, err
	}
	snapshot.ActiveProgress = progress
	snapshot.ActiveProgressTotal = total

	event, err := scanExecutionEventV2(tx.QueryRowContext(ctx, `
		SELECT `+executionEventV2SelectColumns()+`
		FROM execution_events
		WHERE project_id = $1
		ORDER BY sequence DESC
		LIMIT 1
	`, projectID))
	if err == nil {
		snapshot.LatestEvent = &event
	} else if !errors.Is(err, ErrNotFound) {
		return ProjectLiveStateSnapshot{}, err
	}

	if err := tx.Commit(); err != nil {
		return ProjectLiveStateSnapshot{}, err
	}
	return snapshot, nil
}

func scanProjectLiveStateJobCounts(ctx context.Context, tx *sql.Tx, projectID string, counts *LiveStateJobCounts) error {
	attemptSQL := liveStateActiveAttemptSQL("job")
	query := `
		WITH project_jobs AS (
			SELECT job.id, job.status, job.attempt, ` + attemptSQL + ` AS progress_attempt
			FROM experiment_jobs AS job
			WHERE job.project_id = $1
		), aggregate AS (
			SELECT
				count(*)::integer AS total,
				count(*) FILTER (WHERE job.status = 'QUEUED' AND job.attempt = 0)::integer AS queued,
				count(*) FILTER (WHERE job.status = 'QUEUED' AND job.attempt > 0)::integer AS retrying,
				count(*) FILTER (WHERE job.status = 'ASSIGNED')::integer AS assigned,
				count(*) FILTER (WHERE job.status = 'RUNNING')::integer AS running,
				count(*) FILTER (WHERE job.status = 'SUCCEEDED')::integer AS succeeded,
				count(*) FILTER (WHERE job.status = 'FAILED' AND COALESCE(progress.stage, '') <> 'cancelled')::integer AS failed,
				count(*) FILTER (WHERE job.status = 'FAILED' AND progress.stage = 'cancelled')::integer AS cancelled
			FROM project_jobs AS job
			LEFT JOIN job_progress AS progress
				ON progress.job_id = job.id AND progress.attempt = job.progress_attempt
		)
		SELECT EXISTS(SELECT 1 FROM projects WHERE id = $1),
			total, queued, retrying, assigned, running, succeeded, failed, cancelled
		FROM aggregate
	`
	var exists bool
	if err := tx.QueryRowContext(ctx, query, projectID).Scan(
		&exists,
		&counts.Total,
		&counts.Queued,
		&counts.Retrying,
		&counts.Assigned,
		&counts.Running,
		&counts.Succeeded,
		&counts.Failed,
		&counts.Cancelled,
	); err != nil {
		return normalizeSQLError(err)
	}
	if !exists {
		return ErrNotFound
	}
	return nil
}

func scanProjectLiveStateWorkerCounts(
	ctx context.Context,
	tx *sql.Tx,
	projectID string,
	observedAt time.Time,
	counts *LiveStateWorkerCounts,
	lastHeartbeat **time.Time,
) error {
	var heartbeat sql.NullTime
	if err := tx.QueryRowContext(ctx, `
		SELECT
			count(*)::integer,
			count(*) FILTER (WHERE status = 'IDLE')::integer,
			count(*) FILTER (WHERE status = 'RUNNING')::integer,
			count(*) FILTER (WHERE status = 'OFFLINE')::integer,
			count(*) FILTER (
				WHERE status <> 'OFFLINE' AND last_heartbeat < $2 - interval '30 seconds'
			)::integer,
			max(last_heartbeat)
		FROM workers
		WHERE project_id = $1
	`, projectID, observedAt).Scan(
		&counts.Total,
		&counts.Idle,
		&counts.Running,
		&counts.Offline,
		&counts.Stale,
		&heartbeat,
	); err != nil {
		return normalizeSQLError(err)
	}
	if heartbeat.Valid {
		value := heartbeat.Time.UTC()
		*lastHeartbeat = &value
	}
	return nil
}

func scanProjectLiveStateRequirementCounts(ctx context.Context, tx *sql.Tx, projectID string, counts *LiveStateRequirementCounts) error {
	return normalizeSQLError(tx.QueryRowContext(ctx, `
		SELECT
			count(*) FILTER (WHERE status = 'PENDING')::integer,
			count(*) FILTER (WHERE status = 'STARTING')::integer,
			count(*) FILTER (WHERE status = 'ACTIVE')::integer,
			count(*) FILTER (WHERE status = 'SATISFIED')::integer,
			count(*) FILTER (WHERE status = 'FAILED')::integer,
			count(*) FILTER (WHERE status = 'CANCELLED')::integer
		FROM worker_requirements
		WHERE project_id = $1
	`, projectID).Scan(
		&counts.Pending,
		&counts.Starting,
		&counts.Active,
		&counts.Satisfied,
		&counts.Failed,
		&counts.Cancelled,
	))
}

func listProjectLiveStateProgress(ctx context.Context, tx *sql.Tx, projectID string) ([]LiveStateActiveProgress, int, error) {
	attemptSQL := liveStateActiveAttemptSQL("job")
	query := `
		SELECT
			count(*) OVER()::integer,
			progress.project_id, progress.job_id, progress.attempt,
			progress.taxonomy_version, progress.stage, progress.detail_code,
			progress.status, progress.current, progress.total, progress.unit,
			progress.safe_message, progress.revision, progress.heartbeat_at,
			progress.updated_at, progress.metadata,
			job.status, job.created_at, worker.last_heartbeat,
			job.lease_last_heartbeat_at
		FROM experiment_jobs AS job
		JOIN job_progress AS progress
			ON progress.job_id = job.id AND progress.attempt = ` + attemptSQL + `
		LEFT JOIN workers AS worker
			ON worker.id = CASE
				WHEN job.worker_id <> '' THEN job.worker_id
				ELSE job.lease_owner_worker_id
			END
		WHERE job.project_id = $1
			AND job.status IN ('QUEUED', 'ASSIGNED', 'RUNNING')
		ORDER BY
			CASE job.status
				WHEN 'RUNNING' THEN 0
				WHEN 'ASSIGNED' THEN 1
				WHEN 'QUEUED' THEN CASE WHEN job.attempt > 0 THEN 2 ELSE 3 END
				ELSE 4
			END,
			job.created_at ASC,
			job.id ASC
		LIMIT 8
	`
	rows, err := tx.QueryContext(ctx, query, projectID)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()

	out := make([]LiveStateActiveProgress, 0, ProjectLiveStateProgressLimit)
	total := 0
	for rows.Next() {
		item, count, err := scanProjectLiveStateProgress(rows)
		if err != nil {
			return nil, 0, err
		}
		total = count
		out = append(out, item)
	}
	if err := rows.Err(); err != nil {
		return nil, 0, err
	}
	return out, total, nil
}

func scanProjectLiveStateProgress(row rowScanner) (LiveStateActiveProgress, int, error) {
	var item LiveStateActiveProgress
	var total int
	var current sql.NullInt64
	var progressTotal sql.NullInt64
	var metadataJSON []byte
	var workerHeartbeat sql.NullTime
	var leaseHeartbeat sql.NullTime
	if err := row.Scan(
		&total,
		&item.Progress.ProjectID,
		&item.Progress.JobID,
		&item.Progress.Attempt,
		&item.Progress.TaxonomyVersion,
		&item.Progress.Stage,
		&item.Progress.DetailCode,
		&item.Progress.Status,
		&current,
		&progressTotal,
		&item.Progress.Unit,
		&item.Progress.Message,
		&item.Progress.Revision,
		&item.Progress.HeartbeatAt,
		&item.Progress.UpdatedAt,
		&metadataJSON,
		&item.JobStatus,
		&item.JobCreatedAt,
		&workerHeartbeat,
		&leaseHeartbeat,
	); err != nil {
		return LiveStateActiveProgress{}, 0, normalizeSQLError(err)
	}
	if current.Valid {
		value := current.Int64
		item.Progress.Current = &value
	}
	if progressTotal.Valid {
		value := progressTotal.Int64
		item.Progress.Total = &value
	}
	item.Progress.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &item.Progress.Metadata); err != nil {
			return LiveStateActiveProgress{}, 0, fmt.Errorf("unmarshal live-state progress metadata: %w", err)
		}
	}
	if workerHeartbeat.Valid {
		value := workerHeartbeat.Time.UTC()
		item.WorkerHeartbeatAt = &value
	}
	if leaseHeartbeat.Valid {
		value := leaseHeartbeat.Time.UTC()
		item.LeaseHeartbeatAt = &value
	}
	return item, total, nil
}

func liveStateActiveAttemptSQL(alias string) string {
	return `CASE
		WHEN length(` + alias + `.config->>'active_attempt_number') BETWEEN 1 AND 9
			AND (` + alias + `.config->>'active_attempt_number') ~ '^[1-9][0-9]*$'
			THEN (` + alias + `.config->>'active_attempt_number')::integer
		WHEN ` + alias + `.status = 'QUEUED' THEN GREATEST(` + alias + `.attempt + 1, 1)
		ELSE GREATEST(` + alias + `.attempt, 1)
	END`
}
