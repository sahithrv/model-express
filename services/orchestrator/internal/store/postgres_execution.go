package store

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

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

	query := createExecutionEventQuery()
	return scanExecutionEvent(s.db.QueryRowContext(context.Background(), query, projectID, planID, eventType, message, payloadJSON))
}

func createExecutionEventQuery() string {
	return `
		WITH allocated_cursor AS (
			UPDATE execution_event_sequence_state
			SET last_sequence = last_sequence + 1
			WHERE id = 1
			RETURNING last_sequence
		), new_event AS (
			SELECT
				'execution_event_' || nextval('execution_event_id_seq') AS id,
				last_sequence AS sequence
			FROM allocated_cursor
		)
		INSERT INTO execution_events (
			id, project_id, plan_id, event_type, message, payload, sequence, idempotency_key
		)
		SELECT id, $1, $2, $3, $4, $5, sequence, 'event:' || id
		FROM new_event
		RETURNING ` + executionEventSelectColumns() + `
	`
}

func (s *PostgresStore) ListProjectExecutionEvents(projectID string, limit int) ([]execution.ExecutionEvent, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 50
	}

	query := `
		SELECT ` + executionEventSelectColumns() + `
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

func (s *PostgresStore) ListProjectExecutionEventsAfter(ctx context.Context, projectID string, cursor int64, limit int) ([]execution.ExecutionEvent, error) {
	if err := s.requireProjectContext(ctx, projectID); err != nil {
		return nil, err
	}
	if cursor < 0 {
		return nil, fmt.Errorf("%w: execution event cursor must be nonnegative", ErrInvalidRequest)
	}
	limit = boundedExecutionEventPageLimit(limit)
	query := listProjectExecutionEventsAfterQuery()
	rows, err := s.db.QueryContext(ctx, query, projectID, cursor, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []execution.ExecutionEvent{}
	for rows.Next() {
		event, err := scanExecutionEventV2(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, event)
	}
	return out, rows.Err()
}

func listProjectExecutionEventsAfterQuery() string {
	return `
		SELECT ` + executionEventV2SelectColumns() + `
		FROM execution_events
		WHERE project_id = $1 AND sequence > $2
		ORDER BY sequence ASC
		LIMIT $3
	`
}

func executionEventV2SelectColumns() string {
	return "id, project_id, plan_id, event_type, left(message, 512), " + executionEventV2PayloadProjectionSQL() + ", created_at, sequence"
}

func executionEventV2PayloadProjectionSQL() string {
	quotedKeys := make([]string, 0, len(execution.SafeExecutionEventMetadataKeys()))
	for _, key := range execution.SafeExecutionEventMetadataKeys() {
		quotedKeys = append(quotedKeys, "'"+key+"'")
	}
	return `COALESCE((
		SELECT jsonb_object_agg(projected.key, projected.value)
		FROM (
			SELECT entry.key,
				CASE
					WHEN jsonb_typeof(entry.value) = 'string'
						THEN to_jsonb(left(entry.value #>> '{}', 512))
					WHEN jsonb_typeof(entry.value) = 'boolean'
						THEN entry.value
					WHEN jsonb_typeof(entry.value) = 'number' AND length(entry.value #>> '{}') <= 64
						THEN entry.value
					WHEN jsonb_typeof(entry.value) = 'array' THEN (
						SELECT COALESCE(jsonb_agg(items.value ORDER BY items.ordinality), '[]'::jsonb)
						FROM (
							SELECT element.ordinality,
								CASE
									WHEN jsonb_typeof(element.value) = 'string'
										THEN to_jsonb(left(element.value #>> '{}', 80))
									WHEN jsonb_typeof(element.value) = 'boolean'
										THEN element.value
									WHEN jsonb_typeof(element.value) = 'number' AND length(element.value #>> '{}') <= 64
										THEN element.value
								END AS value
							FROM jsonb_array_elements(entry.value) WITH ORDINALITY AS element(value, ordinality)
							WHERE jsonb_typeof(element.value) IN ('string', 'boolean', 'number')
							ORDER BY element.ordinality
							LIMIT 8
						) AS items
						WHERE items.value IS NOT NULL
					)
				END AS value
			FROM jsonb_each(execution_events.payload) AS entry(key, value)
			WHERE entry.key IN (` + strings.Join(quotedKeys, ", ") + `)
		) AS projected
		WHERE projected.value IS NOT NULL
	), '{}'::jsonb) AS payload`
}

func scanExecutionEventV2(row rowScanner) (execution.ExecutionEvent, error) {
	var event execution.ExecutionEvent
	var payloadJSON []byte
	if err := row.Scan(
		&event.ID,
		&event.ProjectID,
		&event.PlanID,
		&event.EventType,
		&event.Message,
		&payloadJSON,
		&event.CreatedAt,
		&event.Sequence,
	); err != nil {
		return execution.ExecutionEvent{}, normalizeSQLError(err)
	}
	event.Payload = map[string]any{}
	if len(payloadJSON) > 0 {
		if err := json.Unmarshal(payloadJSON, &event.Payload); err != nil {
			return execution.ExecutionEvent{}, fmt.Errorf("unmarshal execution event stream payload: %w", err)
		}
	}
	return event, nil
}

func (s *PostgresStore) GetExecutionEventCursorState(ctx context.Context) (execution.ExecutionEventCursorState, error) {
	var state execution.ExecutionEventCursorState
	const query = `
		SELECT last_sequence, retained_sequence_floor
		FROM execution_event_sequence_state
		WHERE id = 1
	`
	if err := s.db.QueryRowContext(ctx, query).Scan(&state.LastSequence, &state.RetainedSequenceFloor); err != nil {
		return execution.ExecutionEventCursorState{}, normalizeSQLError(err)
	}
	return state, nil
}

func executionEventSelectColumns() string {
	return "id, project_id, plan_id, event_type, message, payload, created_at, sequence, idempotency_key"
}
