ALTER TABLE execution_events
  ADD COLUMN IF NOT EXISTS sequence bigint,
  ADD COLUMN IF NOT EXISTS idempotency_key text;

-- Existing event IDs are the stable tie-breaker when two rows share a timestamp.
-- Only rows without a cursor are updated so rerunning migrations cannot renumber
-- events that a client may already have observed.
WITH ordered_execution_events AS (
  SELECT
    id,
    COALESCE((SELECT MAX(sequence) FROM execution_events), 0) +
      row_number() OVER (ORDER BY created_at ASC, id ASC)::bigint AS assigned_sequence
  FROM execution_events
  WHERE sequence IS NULL
)
UPDATE execution_events AS event
SET sequence = ordered.assigned_sequence
FROM ordered_execution_events AS ordered
WHERE event.id = ordered.id
  AND event.sequence IS NULL;

UPDATE execution_events
SET idempotency_key = 'legacy:' || id
WHERE idempotency_key IS NULL OR btrim(idempotency_key) = '';

-- Allocating a PostgreSQL sequence value does not follow transaction commit
-- order. This singleton row is updated in the same transaction as a future
-- event insert, so its row lock prevents a lower event cursor from committing
-- after a higher cursor has already become visible.
CREATE TABLE IF NOT EXISTS execution_event_sequence_state (
  id smallint PRIMARY KEY,
  last_sequence bigint NOT NULL DEFAULT 0,
  retained_sequence_floor bigint NOT NULL DEFAULT 0,
  CONSTRAINT execution_event_sequence_state_singleton CHECK (id = 1),
  CONSTRAINT execution_event_sequence_state_last_nonnegative CHECK (last_sequence >= 0),
  CONSTRAINT execution_event_sequence_state_floor_nonnegative CHECK (retained_sequence_floor >= 0),
  CONSTRAINT execution_event_sequence_state_floor_bounded CHECK (retained_sequence_floor <= last_sequence)
);

ALTER TABLE execution_event_sequence_state
  ADD COLUMN IF NOT EXISTS retained_sequence_floor bigint NOT NULL DEFAULT 0;

INSERT INTO execution_event_sequence_state (id, last_sequence, retained_sequence_floor)
SELECT 1, COALESCE(MAX(sequence), 0), 0
FROM execution_events
ON CONFLICT (id) DO UPDATE
SET last_sequence = GREATEST(
  execution_event_sequence_state.last_sequence,
  EXCLUDED.last_sequence
);

ALTER TABLE execution_events
  ALTER COLUMN sequence SET NOT NULL,
  ALTER COLUMN idempotency_key SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_execution_events_sequence_unique
  ON execution_events(sequence);

CREATE UNIQUE INDEX IF NOT EXISTS idx_execution_events_idempotency_key_unique
  ON execution_events(idempotency_key);

CREATE INDEX IF NOT EXISTS idx_execution_events_project_sequence
  ON execution_events(project_id, sequence);
