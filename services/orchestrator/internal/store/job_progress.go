package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"model-express/services/orchestrator/internal/jobs"
)

func (s *MemoryStore) GetJobProgress(jobID string, attempt int) (jobs.JobProgress, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if attempt < 0 {
		return jobs.JobProgress{}, fmt.Errorf("%w: progress attempt must be nonnegative", ErrInvalidRequest)
	}
	if _, ok := s.jobs[jobID]; !ok {
		return jobs.JobProgress{}, ErrNotFound
	}
	progress, ok := s.jobProgress[jobProgressKey(jobID, attempt)]
	if !ok {
		return jobs.JobProgress{}, ErrNotFound
	}
	return cloneJobProgress(progress), nil
}

func (s *MemoryStore) UpsertJobProgress(jobID string, update jobs.JobProgressUpsert) (jobs.JobProgress, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, ok := s.jobs[jobID]
	if !ok {
		return jobs.JobProgress{}, ErrNotFound
	}
	return s.upsertJobProgressLocked(job, update, time.Now().UTC())
}

// upsertJobProgressLocked is the memory lifecycle composition point. Callers
// must hold s.mu and can persist the job, progress, and event under that same
// critical section.
func (s *MemoryStore) upsertJobProgressLocked(job jobs.ExperimentJob, update jobs.JobProgressUpsert, now time.Time) (jobs.JobProgress, error) {
	normalized, err := jobs.NormalizeJobProgressUpsert(update)
	if err != nil {
		return jobs.JobProgress{}, fmt.Errorf("%w: invalid job progress: %v", ErrInvalidRequest, err)
	}

	key := jobProgressKey(job.ID, normalized.Attempt)
	if existing, ok := s.jobProgress[key]; ok {
		if jobs.IsTerminalJobProgressStage(existing.Stage) {
			return cloneJobProgress(existing), nil
		}
		if jobs.IsTerminalJobProgressStage(normalized.Stage) {
			if normalized.Revision <= existing.Revision {
				normalized.Revision = existing.Revision + 1
			}
		} else if normalized.Revision <= existing.Revision {
			return cloneJobProgress(existing), nil
		}
	}

	now = now.UTC()
	progress := jobs.JobProgress{
		ProjectID:       job.ProjectID,
		JobID:           job.ID,
		Attempt:         normalized.Attempt,
		TaxonomyVersion: normalized.TaxonomyVersion,
		Stage:           normalized.Stage,
		DetailCode:      normalized.DetailCode,
		Status:          normalized.Status,
		Current:         cloneInt64Pointer(normalized.Current),
		Total:           cloneInt64Pointer(normalized.Total),
		Unit:            normalized.Unit,
		Message:         normalized.Message,
		Revision:        normalized.Revision,
		HeartbeatAt:     now,
		UpdatedAt:       now,
		Metadata:        cloneProgressMetadata(normalized.Metadata),
	}
	s.jobProgress[key] = progress
	return cloneJobProgress(progress), nil
}

func (s *PostgresStore) GetJobProgress(jobID string, attempt int) (jobs.JobProgress, error) {
	if attempt < 0 {
		return jobs.JobProgress{}, fmt.Errorf("%w: progress attempt must be nonnegative", ErrInvalidRequest)
	}
	query := `
		SELECT ` + jobProgressSelectColumns() + `
		FROM job_progress
		WHERE job_id = $1 AND attempt = $2
	`
	return scanJobProgress(s.db.QueryRowContext(context.Background(), query, jobID, attempt))
}

func (s *PostgresStore) UpsertJobProgress(jobID string, update jobs.JobProgressUpsert) (jobs.JobProgress, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return jobs.JobProgress{}, err
	}
	defer tx.Rollback()

	job, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id")+" FOR UPDATE", jobID))
	if err != nil {
		return jobs.JobProgress{}, err
	}
	progress, err := upsertJobProgressTx(ctx, tx, job, update, time.Now().UTC())
	if err != nil {
		return jobs.JobProgress{}, err
	}
	if err := tx.Commit(); err != nil {
		return jobs.JobProgress{}, err
	}
	return progress, nil
}

// upsertJobProgressTx is the PostgreSQL lifecycle composition point. Its
// caller owns tx and can commit the job row, snapshot, and transition event as
// one unit.
func upsertJobProgressTx(ctx context.Context, tx *sql.Tx, job jobs.ExperimentJob, update jobs.JobProgressUpsert, now time.Time) (jobs.JobProgress, error) {
	normalized, err := jobs.NormalizeJobProgressUpsert(update)
	if err != nil {
		return jobs.JobProgress{}, fmt.Errorf("%w: invalid job progress: %v", ErrInvalidRequest, err)
	}
	metadataJSON, err := json.Marshal(normalized.Metadata)
	if err != nil {
		return jobs.JobProgress{}, fmt.Errorf("marshal job progress metadata: %w", err)
	}

	// A worker observation only advances a non-terminal snapshot with a strictly
	// newer revision. Backend-only terminal stages supersede any non-terminal
	// worker revision and remain monotonic. The fallback SELECT makes duplicate,
	// older, and post-terminal writes idempotently return the stored row.
	query := upsertJobProgressQuery()
	return scanJobProgress(tx.QueryRowContext(
		ctx,
		query,
		job.ID,
		job.ProjectID,
		normalized.Attempt,
		normalized.TaxonomyVersion,
		normalized.Stage,
		normalized.DetailCode,
		normalized.Status,
		normalized.Current,
		normalized.Total,
		normalized.Unit,
		normalized.Message,
		normalized.Revision,
		now.UTC(),
		metadataJSON,
	))
}

func upsertJobProgressQuery() string {
	return `
		WITH owned_job AS (
			SELECT id, project_id
			FROM experiment_jobs
			WHERE id = $1 AND project_id = $2
		), upserted AS (
			INSERT INTO job_progress (
				project_id, job_id, attempt, taxonomy_version, stage,
				detail_code, status, current, total, unit, safe_message,
				revision, heartbeat_at, updated_at, metadata
			)
			SELECT
				owned_job.project_id, owned_job.id, $3, $4, $5,
				$6, $7, $8, $9, $10, $11,
				$12, $13, $13, $14
			FROM owned_job
			WHERE true
			ON CONFLICT (job_id, attempt) DO UPDATE
			SET taxonomy_version = EXCLUDED.taxonomy_version,
				stage = EXCLUDED.stage,
				detail_code = EXCLUDED.detail_code,
				status = EXCLUDED.status,
				current = EXCLUDED.current,
				total = EXCLUDED.total,
				unit = EXCLUDED.unit,
				safe_message = EXCLUDED.safe_message,
				revision = CASE
					WHEN EXCLUDED.stage IN ('completed', 'failed', 'cancelled')
						THEN GREATEST(EXCLUDED.revision, job_progress.revision + 1)
					ELSE EXCLUDED.revision
				END,
				heartbeat_at = EXCLUDED.heartbeat_at,
				updated_at = EXCLUDED.updated_at,
				metadata = EXCLUDED.metadata
			WHERE job_progress.stage NOT IN ('completed', 'failed', 'cancelled')
				AND (
					EXCLUDED.revision > job_progress.revision OR
					EXCLUDED.stage IN ('completed', 'failed', 'cancelled')
				)
			RETURNING ` + jobProgressSelectColumns() + `
		)
		SELECT ` + jobProgressSelectColumns() + ` FROM upserted
		UNION ALL
		SELECT ` + jobProgressSelectColumns() + `
		FROM job_progress
		WHERE job_id = $1 AND project_id = $2 AND attempt = $3
			AND NOT EXISTS (SELECT 1 FROM upserted)
		LIMIT 1
	`
}

func scanJobProgress(row rowScanner) (jobs.JobProgress, error) {
	var progress jobs.JobProgress
	var current sql.NullInt64
	var total sql.NullInt64
	var metadataJSON []byte
	if err := row.Scan(
		&progress.ProjectID,
		&progress.JobID,
		&progress.Attempt,
		&progress.TaxonomyVersion,
		&progress.Stage,
		&progress.DetailCode,
		&progress.Status,
		&current,
		&total,
		&progress.Unit,
		&progress.Message,
		&progress.Revision,
		&progress.HeartbeatAt,
		&progress.UpdatedAt,
		&metadataJSON,
	); err != nil {
		return jobs.JobProgress{}, normalizeSQLError(err)
	}
	if current.Valid {
		progress.Current = cloneInt64Pointer(&current.Int64)
	}
	if total.Valid {
		progress.Total = cloneInt64Pointer(&total.Int64)
	}
	progress.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &progress.Metadata); err != nil {
			return jobs.JobProgress{}, fmt.Errorf("unmarshal job progress metadata: %w", err)
		}
	}
	return progress, nil
}

func jobProgressSelectColumns() string {
	return "project_id, job_id, attempt, taxonomy_version, stage, detail_code, status, current, total, unit, safe_message, revision, heartbeat_at, updated_at, metadata"
}

func jobProgressKey(jobID string, attempt int) string {
	return jobID + "\x00" + strconv.Itoa(attempt)
}

func cloneJobProgress(progress jobs.JobProgress) jobs.JobProgress {
	progress.Current = cloneInt64Pointer(progress.Current)
	progress.Total = cloneInt64Pointer(progress.Total)
	progress.Metadata = cloneProgressMetadata(progress.Metadata)
	return progress
}

func cloneInt64Pointer(value *int64) *int64 {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func cloneProgressMetadata(metadata map[string]any) map[string]any {
	if metadata == nil {
		return map[string]any{}
	}
	encoded, err := json.Marshal(metadata)
	if err != nil {
		return map[string]any{}
	}
	var out map[string]any
	if err := json.Unmarshal(encoded, &out); err != nil || out == nil {
		return map[string]any{}
	}
	return out
}
