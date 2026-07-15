package store

import (
	"strings"
	"testing"
)

func executionEventCursorMigrationSQL(t *testing.T) string {
	t.Helper()
	sqlBytes, err := migrationFiles.ReadFile("migrations/017_execution_event_cursor.sql")
	if err != nil {
		t.Fatal(err)
	}
	return string(sqlBytes)
}

func TestExecutionEventCursorMigrationBackfillsStableIdentity(t *testing.T) {
	sqlText := executionEventCursorMigrationSQL(t)
	for _, required := range []string{
		"ADD COLUMN IF NOT EXISTS sequence bigint",
		"ADD COLUMN IF NOT EXISTS idempotency_key text",
		"row_number() OVER (ORDER BY created_at ASC, id ASC)",
		"COALESCE((SELECT MAX(sequence) FROM execution_events), 0)",
		"WHERE sequence IS NULL",
		"AND event.sequence IS NULL",
		"SET idempotency_key = 'legacy:' || id",
		"ALTER COLUMN sequence SET NOT NULL",
		"ALTER COLUMN idempotency_key SET NOT NULL",
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_execution_events_sequence_unique",
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_execution_events_idempotency_key_unique",
		"ON execution_events(project_id, sequence)",
	} {
		if !strings.Contains(sqlText, required) {
			t.Fatalf("execution-event cursor migration omitted %q", required)
		}
	}
}

func TestExecutionEventCursorMigrationUsesTransactionalAllocatorState(t *testing.T) {
	sqlText := executionEventCursorMigrationSQL(t)
	for _, required := range []string{
		"CREATE TABLE IF NOT EXISTS execution_event_sequence_state",
		"last_sequence bigint NOT NULL DEFAULT 0",
		"retained_sequence_floor bigint NOT NULL DEFAULT 0",
		"CHECK (id = 1)",
		"COALESCE(MAX(sequence), 0)",
		"ON CONFLICT (id) DO UPDATE",
		"GREATEST(",
	} {
		if !strings.Contains(sqlText, required) {
			t.Fatalf("execution-event allocator migration omitted %q", required)
		}
	}
	if strings.Contains(sqlText, "nextval(") || strings.Contains(sqlText, "bigserial") {
		t.Fatalf("event cursor allocation must follow transaction commit order, not a PostgreSQL sequence: %s", sqlText)
	}
}
