package store

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func TestMemoryExecutionEventCursorIsGlobalAndPagesAscending(t *testing.T) {
	s := NewMemoryStore()
	projectA, _ := s.CreateProject("a", "")
	projectB, _ := s.CreateProject("b", "")
	first, _ := s.CreateExecutionEvent(projectA.ID, "", "ONE", "one", nil)
	other, _ := s.CreateExecutionEvent(projectB.ID, "", "OTHER", "other", nil)
	second, _ := s.CreateExecutionEvent(projectA.ID, "", "TWO", "two", nil)
	third, _ := s.CreateExecutionEvent(projectA.ID, "", "THREE", "three", nil)

	if first.Sequence != 1 || other.Sequence != 2 || second.Sequence != 3 || third.Sequence != 4 {
		t.Fatalf("event sequence is not globally monotonic: %#v %#v %#v %#v", first, other, second, third)
	}
	page, err := s.ListProjectExecutionEventsAfter(context.Background(), projectA.ID, first.Sequence, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(page) != 2 || page[0].Sequence != second.Sequence || page[1].Sequence != third.Sequence {
		t.Fatalf("cursor page has a gap or unstable order: %#v", page)
	}
	if page[0].IdempotencyKey != second.IdempotencyKey || page[1].IdempotencyKey != third.IdempotencyKey {
		t.Fatalf("cursor projection lost internal idempotency identity: %#v", page)
	}
	after, err := s.ListProjectExecutionEventsAfter(context.Background(), projectA.ID, third.Sequence, 2)
	if err != nil || len(after) != 0 {
		t.Fatalf("reconnect duplicated delivered events: rows=%#v err=%v", after, err)
	}
	state, err := s.GetExecutionEventCursorState(context.Background())
	if err != nil || state.LastSequence != 4 || state.RetainedSequenceFloor != 0 {
		t.Fatalf("cursor state = %#v, err=%v", state, err)
	}
}

func TestExecutionEventCursorFieldsDoNotChangeV1JSON(t *testing.T) {
	s := NewMemoryStore()
	project, _ := s.CreateProject("json", "")
	event, _ := s.CreateExecutionEvent(project.ID, "", "SAFE", "safe", map[string]any{"count": 1})
	blob, err := json.Marshal(event)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"sequence", "idempotency_key", event.IdempotencyKey} {
		if strings.Contains(string(blob), forbidden) {
			t.Fatalf("v1 execution event JSON changed: %s", blob)
		}
	}
}

func TestExecutionEventAfterQueryIsIndexedAscendingAndBounded(t *testing.T) {
	query := listProjectExecutionEventsAfterQuery()
	for _, required := range []string{
		"project_id = $1 AND sequence > $2",
		"ORDER BY sequence ASC",
		"LIMIT $3",
		"jsonb_each(execution_events.payload)",
		"left(message, 512)",
		"LIMIT 8",
		"idempotency_key",
	} {
		if !strings.Contains(query, required) {
			t.Fatalf("cursor query omitted %q: %s", required, query)
		}
	}
	for _, forbidden := range []string{"raw_output", "storage_uri"} {
		if strings.Contains(query, forbidden) {
			t.Fatalf("cursor query selected disallowed field %q: %s", forbidden, query)
		}
	}
}

func TestExecutionEventCreateAllocatesCursorAndInsertUnderOneLock(t *testing.T) {
	query := createExecutionEventQuery()
	for _, required := range []string{
		"UPDATE execution_event_sequence_state",
		"SET last_sequence = last_sequence + 1",
		"RETURNING last_sequence",
		"INSERT INTO execution_events",
		"sequence, idempotency_key",
		"'event:' || id",
	} {
		if !strings.Contains(query, required) {
			t.Fatalf("execution-event create query omitted %q: %s", required, query)
		}
	}
}

func TestV1ActivityExecutionEventQueryFiltersBeforeLimit(t *testing.T) {
	query := executionEventActivitySelectQuery()
	for _, required := range []string{
		"event_type NOT IN",
		"JOB_RETRY_QUEUED_TRANSITION",
		"payload->>'reason_code' IN ('attempts_exhausted', 'lease_attempts_exhausted')",
		"ORDER BY created_at DESC",
		"LIMIT $2",
	} {
		if !strings.Contains(query, required) {
			t.Fatalf("v1-compatible execution event query omitted %q: %s", required, query)
		}
	}
}
