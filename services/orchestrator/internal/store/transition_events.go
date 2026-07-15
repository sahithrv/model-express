package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"model-express/services/orchestrator/internal/execution"
)

// CreateExecutionTransition appends a typed producer event exactly once. The
// legacy CreateExecutionEvent path intentionally remains available while v1
// activity synthesis is retained during rollout.
func (s *MemoryStore) CreateExecutionTransition(input execution.ExecutionTransitionEventInput) (execution.ExecutionEvent, bool, error) {
	create, err := execution.NewExecutionTransitionEvent(input)
	if err != nil {
		return execution.ExecutionEvent{}, false, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	return s.appendExecutionTransitionLocked(create, time.Now().UTC())
}

func (s *MemoryStore) appendExecutionTransitionLocked(create execution.ExecutionEventCreate, now time.Time) (execution.ExecutionEvent, bool, error) {
	if _, ok := s.projects[create.ProjectID]; !ok {
		return execution.ExecutionEvent{}, false, ErrNotFound
	}
	for _, existing := range s.executionEvents {
		if existing.IdempotencyKey == create.IdempotencyKey {
			return existing, false, nil
		}
	}

	event := execution.ExecutionEvent{
		ID:             s.newID("execution_event"),
		ProjectID:      create.ProjectID,
		PlanID:         create.PlanID,
		EventType:      create.EventType,
		Message:        create.Message,
		Payload:        cloneJSONMap(create.Payload),
		CreatedAt:      now,
		IdempotencyKey: create.IdempotencyKey,
	}
	s.nextExecutionEventSequence++
	event.Sequence = s.nextExecutionEventSequence
	s.executionEvents[event.ID] = event
	return event, true, nil
}

func (s *PostgresStore) CreateExecutionTransition(input execution.ExecutionTransitionEventInput) (execution.ExecutionEvent, bool, error) {
	create, err := execution.NewExecutionTransitionEvent(input)
	if err != nil {
		return execution.ExecutionEvent{}, false, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
	}

	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return execution.ExecutionEvent{}, false, err
	}
	defer tx.Rollback()

	event, created, err := appendExecutionTransitionTx(ctx, tx, create)
	if err != nil {
		return execution.ExecutionEvent{}, false, err
	}
	if err := tx.Commit(); err != nil {
		return execution.ExecutionEvent{}, false, err
	}
	return event, created, nil
}

func appendExecutionTransitionTx(ctx context.Context, tx *sql.Tx, create execution.ExecutionEventCreate) (execution.ExecutionEvent, bool, error) {
	payloadJSON, err := json.Marshal(create.Payload)
	if err != nil {
		return execution.ExecutionEvent{}, false, fmt.Errorf("marshal execution transition payload: %w", err)
	}

	// The cursor allocator row is also the producer idempotency lock. Taking it
	// before checking the key makes concurrent duplicate producers observe the
	// first committed row without consuming a second cursor.
	var lastSequence int64
	if err := tx.QueryRowContext(ctx, `
		SELECT last_sequence
		FROM execution_event_sequence_state
		WHERE id = 1
		FOR UPDATE
	`).Scan(&lastSequence); err != nil {
		return execution.ExecutionEvent{}, false, normalizeSQLError(err)
	}

	existing, err := scanExecutionEvent(tx.QueryRowContext(ctx, `
		SELECT `+executionEventSelectColumns()+`
		FROM execution_events
		WHERE idempotency_key = $1
	`, create.IdempotencyKey))
	if err == nil {
		return existing, false, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return execution.ExecutionEvent{}, false, err
	}

	nextSequence := lastSequence + 1
	if _, err := tx.ExecContext(ctx, `
		UPDATE execution_event_sequence_state
		SET last_sequence = $1
		WHERE id = 1
	`, nextSequence); err != nil {
		return execution.ExecutionEvent{}, false, err
	}

	event, err := scanExecutionEvent(tx.QueryRowContext(ctx, `
		INSERT INTO execution_events (
			id, project_id, plan_id, event_type, message, payload, sequence, idempotency_key
		)
		VALUES (
			'execution_event_' || nextval('execution_event_id_seq'), $1, $2, $3, $4, $5, $6, $7
		)
		RETURNING `+executionEventSelectColumns()+`
	`, create.ProjectID, create.PlanID, create.EventType, create.Message, payloadJSON, nextSequence, create.IdempotencyKey))
	if err != nil {
		return execution.ExecutionEvent{}, false, err
	}
	return event, true, nil
}

func executionEventIncludedInV1Activity(event execution.ExecutionEvent) bool {
	if !execution.IsDurableTransitionEventType(event.EventType) {
		return true
	}
	if event.EventType == execution.EventJobRetryQueuedTransition {
		return true
	}
	if event.EventType != execution.EventJobFailed {
		return false
	}
	reason, _ := event.Payload["reason_code"].(string)
	return reason == "attempts_exhausted" || reason == "lease_attempts_exhausted"
}
