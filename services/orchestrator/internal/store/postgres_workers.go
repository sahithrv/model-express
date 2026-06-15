package store

import (
	"context"
	"fmt"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/workers"
)

func (s *PostgresStore) RegisterWorker(projectID string, name string, gpuType string) (workers.Worker, error) {
	if projectID == "" {
		return workers.Worker{}, fmt.Errorf("%w: project_id is required", ErrInvalidRequest)
	}
	if err := s.requireProject(projectID); err != nil {
		return workers.Worker{}, err
	}
	if err := s.requireProjectDataset(projectID); err != nil {
		return workers.Worker{}, err
	}

	const query = `
		INSERT INTO workers (project_id, name, status, gpu_type)
		VALUES ($1, $2, $3, $4)
		RETURNING id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
	`

	return scanWorker(s.db.QueryRowContext(context.Background(), query, projectID, name, workers.StatusIdle, gpuType))
}

func (s *PostgresStore) ListWorkers() ([]workers.Worker, error) {
	const query = `
		SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
		FROM workers
		ORDER BY last_heartbeat DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []workers.Worker{}
	for rows.Next() {
		worker, err := scanWorker(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, worker)
	}

	return out, rows.Err()
}

func (s *PostgresStore) ListProjectWorkers(projectID string) ([]workers.Worker, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
		FROM workers
		WHERE project_id = $1
		ORDER BY last_heartbeat DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []workers.Worker{}
	for rows.Next() {
		worker, err := scanWorker(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, worker)
	}

	return out, rows.Err()
}

func (s *PostgresStore) GetWorker(workerID string) (workers.Worker, error) {
	const query = `
		SELECT id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
		FROM workers
		WHERE id=$1
	`

	return scanWorker(s.db.QueryRowContext(context.Background(), query, workerID))
}

func (s *PostgresStore) HeartbeatWorker(id string) (workers.Worker, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return workers.Worker{}, err
	}
	defer tx.Rollback()

	worker, err := scanWorker(tx.QueryRowContext(ctx, `
		UPDATE workers
		SET last_heartbeat = now()
		WHERE id = $1
		RETURNING id, project_id, name, status, gpu_type, last_heartbeat, current_job_id
	`, id))
	if err != nil {
		return workers.Worker{}, err
	}

	if worker.CurrentJobID != "" {
		now := worker.LastHeartbeat
		leaseExpiresAt := now.Add(defaultJobLeaseDuration)
		if _, err := tx.ExecContext(ctx, `
			UPDATE experiment_jobs
			SET lease_owner_worker_id = $1,
				lease_last_heartbeat_at = $2,
				lease_expires_at = $3
			WHERE id = $4
				AND status NOT IN ($5, $6)
		`, worker.ID, now, leaseExpiresAt, worker.CurrentJobID, jobs.StatusSucceeded, jobs.StatusFailed); err != nil {
			return workers.Worker{}, err
		}
	}

	if err := tx.Commit(); err != nil {
		return workers.Worker{}, err
	}
	return worker, nil
}
