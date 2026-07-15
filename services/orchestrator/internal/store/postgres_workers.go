package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/workers"
)

func (s *PostgresStore) RegisterWorker(projectID string, name string, gpuType string) (workers.Worker, error) {
	return s.RegisterWorkerWithCapabilities(projectID, name, gpuType, []string{execution.WorkerPolicyContractV1}, []string{execution.WorkerArtifactPlanV1})
}

func (s *PostgresStore) RegisterWorkerWithCapabilities(projectID string, name string, gpuType string, policyVersions []string, artifactVersions []string) (workers.Worker, error) {
	if projectID == "" {
		return workers.Worker{}, fmt.Errorf("%w: project_id is required", ErrInvalidRequest)
	}
	if err := s.requireProject(projectID); err != nil {
		return workers.Worker{}, err
	}
	if err := s.requireProjectDataset(projectID); err != nil {
		return workers.Worker{}, err
	}

	policyJSON, _ := json.Marshal(normalizeCapabilityVersions(policyVersions))
	artifactJSON, _ := json.Marshal(normalizeCapabilityVersions(artifactVersions))
	const query = `
		INSERT INTO workers (project_id, name, status, gpu_type, policy_capability_versions, artifact_capability_versions)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id, project_id, name, status, gpu_type, policy_capability_versions, artifact_capability_versions, last_heartbeat, current_job_id
	`

	return scanWorker(s.db.QueryRowContext(context.Background(), query, projectID, name, workers.StatusIdle, gpuType, policyJSON, artifactJSON))
}

func (s *PostgresStore) ListWorkers() ([]workers.Worker, error) {
	const query = `
		SELECT id, project_id, name, status, gpu_type, policy_capability_versions, artifact_capability_versions, last_heartbeat, current_job_id
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
		SELECT id, project_id, name, status, gpu_type, policy_capability_versions, artifact_capability_versions, last_heartbeat, current_job_id
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
		SELECT id, project_id, name, status, gpu_type, policy_capability_versions, artifact_capability_versions, last_heartbeat, current_job_id
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
		SELECT id, project_id, name, status, gpu_type, policy_capability_versions, artifact_capability_versions, last_heartbeat, current_job_id
		FROM workers
		WHERE id = $1
	`, id))
	if err != nil {
		return workers.Worker{}, err
	}

	if worker.CurrentJobID != "" {
		job, err := scanJob(tx.QueryRowContext(ctx, selectJobSQL("id")+" FOR UPDATE", worker.CurrentJobID))
		if err != nil {
			return workers.Worker{}, err
		}
		worker, err = scanWorker(tx.QueryRowContext(ctx, `
			UPDATE workers
			SET last_heartbeat = now()
			WHERE id = $1 AND current_job_id = $2
			RETURNING id, project_id, name, status, gpu_type, policy_capability_versions, artifact_capability_versions, last_heartbeat, current_job_id
		`, id, job.ID))
		if errors.Is(err, ErrNotFound) {
			if err := tx.Commit(); err != nil {
				return workers.Worker{}, err
			}
			return s.HeartbeatWorker(id)
		}
		if err != nil {
			return workers.Worker{}, err
		}
		now := worker.LastHeartbeat
		leaseExpiresAt := now.Add(defaultJobLeaseDuration)
		if !isTerminalJobStatus(job.Status) {
			if _, err := tx.ExecContext(ctx, `
			UPDATE experiment_jobs
			SET lease_owner_worker_id = $1,
				lease_last_heartbeat_at = $2,
				lease_expires_at = $3
			WHERE id = $4
		`, worker.ID, now, leaseExpiresAt, job.ID); err != nil {
				return workers.Worker{}, err
			}
		}
	} else {
		worker, err = scanWorker(tx.QueryRowContext(ctx, `
			UPDATE workers
			SET last_heartbeat = now()
			WHERE id = $1
			RETURNING id, project_id, name, status, gpu_type, policy_capability_versions, artifact_capability_versions, last_heartbeat, current_job_id
		`, id))
		if err != nil {
			return workers.Worker{}, err
		}
	}

	if err := tx.Commit(); err != nil {
		return workers.Worker{}, err
	}
	return worker, nil
}
