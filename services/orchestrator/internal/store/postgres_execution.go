package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"model-express/services/orchestrator/internal/execution"
)

func (s *PostgresStore) UpsertWorkerRequirement(projectID string, planID string, provider string, gpuType string, targetCount int, source string, policy execution.WorkerRequirementPolicy) (execution.WorkerRequirement, bool, error) {
	if err := s.requireProject(projectID); err != nil {
		return execution.WorkerRequirement{}, false, err
	}
	if targetCount < 1 {
		return execution.WorkerRequirement{}, false, fmt.Errorf("%w: target_count must be at least 1", ErrInvalidRequest)
	}

	existing, err := scanWorkerRequirement(s.db.QueryRowContext(context.Background(), `
		SELECT `+workerRequirementSelectColumns()+`
		FROM worker_requirements
		WHERE project_id = $1 AND plan_id = $2
	`, projectID, planID))
	if err == nil {
		status := existing.Status
		lastError := existing.LastError
		if existing.TargetCount != targetCount || status == execution.WorkerRequirementSatisfied || status == execution.WorkerRequirementFailed || status == execution.WorkerRequirementCancelled {
			status = execution.WorkerRequirementPending
			lastError = ""
		}
		updateQuery := `
			UPDATE worker_requirements
			SET provider = $1,
				gpu_type = $2,
				target_count = $3,
				source = $4,
				status = $5,
				last_error = $6,
				dataset_id = $7,
				dataset_checksum = $8,
				dataset_cache_key = $9,
				dataset_materialization_status = $10,
				cold_cache_policy = $11,
				max_concurrent_jobs = $12,
				max_cold_dataset_materializations = $13,
				updated_at = now()
			WHERE id = $14
			RETURNING ` + workerRequirementSelectColumns() + `
		`
		requirement, updateErr := scanWorkerRequirement(s.db.QueryRowContext(context.Background(), updateQuery, provider, gpuType, targetCount, source, status, lastError, policy.DatasetID, policy.DatasetChecksum, policy.DatasetCacheKey, policy.DatasetMaterializationStatus, policy.ColdCachePolicy, policy.MaxConcurrentJobs, policy.MaxColdDatasetMaterializations, existing.ID))
		return requirement, false, updateErr
	}
	if !errors.Is(err, ErrNotFound) {
		return execution.WorkerRequirement{}, false, err
	}

	insertQuery := `
		INSERT INTO worker_requirements (
			project_id, plan_id, provider, gpu_type, target_count, status, source,
			dataset_id, dataset_checksum, dataset_cache_key, dataset_materialization_status,
			cold_cache_policy, max_concurrent_jobs, max_cold_dataset_materializations
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
		RETURNING ` + workerRequirementSelectColumns() + `
	`
	requirement, err := scanWorkerRequirement(s.db.QueryRowContext(
		context.Background(),
		insertQuery,
		projectID,
		planID,
		provider,
		gpuType,
		targetCount,
		execution.WorkerRequirementPending,
		source,
		policy.DatasetID,
		policy.DatasetChecksum,
		policy.DatasetCacheKey,
		policy.DatasetMaterializationStatus,
		policy.ColdCachePolicy,
		policy.MaxConcurrentJobs,
		policy.MaxColdDatasetMaterializations,
	))
	return requirement, true, err
}

func (s *PostgresStore) ListProjectWorkerRequirements(projectID string) ([]execution.WorkerRequirement, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	query := `
		SELECT ` + workerRequirementSelectColumns() + `
		FROM worker_requirements
		WHERE project_id = $1
		ORDER BY updated_at DESC
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []execution.WorkerRequirement{}
	for rows.Next() {
		requirement, err := scanWorkerRequirement(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, requirement)
	}
	return out, rows.Err()
}

func (s *PostgresStore) UpdateWorkerRequirement(id string, update execution.WorkerRequirementUpdate) (execution.WorkerRequirement, error) {
	requirement, err := scanWorkerRequirement(s.db.QueryRowContext(context.Background(), `
		SELECT `+workerRequirementSelectColumns()+`
		FROM worker_requirements
		WHERE id = $1
	`, id))
	if err != nil {
		return execution.WorkerRequirement{}, err
	}
	if update.Status != nil {
		requirement.Status = *update.Status
	}
	if update.LastError != nil {
		requirement.LastError = *update.LastError
	}
	if update.DatasetMaterializationStatus != nil {
		requirement.DatasetMaterializationStatus = *update.DatasetMaterializationStatus
	}

	query := `
		UPDATE worker_requirements
		SET status = $1, last_error = $2, dataset_materialization_status = $3, updated_at = now()
		WHERE id = $4
		RETURNING ` + workerRequirementSelectColumns() + `
	`
	return scanWorkerRequirement(s.db.QueryRowContext(
		context.Background(),
		query,
		requirement.Status,
		requirement.LastError,
		requirement.DatasetMaterializationStatus,
		id,
	))
}

func (s *PostgresStore) CreateExecutionEvent(projectID string, planID string, eventType string, message string, payload map[string]any) (execution.ExecutionEvent, error) {
	if err := s.requireProject(projectID); err != nil {
		return execution.ExecutionEvent{}, err
	}
	if payload == nil {
		payload = map[string]any{}
	}
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return execution.ExecutionEvent{}, fmt.Errorf("marshal execution event payload: %w", err)
	}

	const query = `
		INSERT INTO execution_events (project_id, plan_id, event_type, message, payload)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, project_id, plan_id, event_type, message, payload, created_at
	`
	return scanExecutionEvent(s.db.QueryRowContext(context.Background(), query, projectID, planID, eventType, message, payloadJSON))
}

func (s *PostgresStore) ListProjectExecutionEvents(projectID string, limit int) ([]execution.ExecutionEvent, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 50
	}

	const query = `
		SELECT id, project_id, plan_id, event_type, message, payload, created_at
		FROM execution_events
		WHERE project_id = $1
		ORDER BY created_at DESC
		LIMIT $2
	`
	rows, err := s.db.QueryContext(context.Background(), query, projectID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []execution.ExecutionEvent{}
	for rows.Next() {
		event, err := scanExecutionEvent(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, event)
	}
	return out, rows.Err()
}
