ALTER TABLE attempt_execution_records
  ALTER COLUMN realized_effective_hash DROP DEFAULT,
  ALTER COLUMN realized_effective_hash DROP NOT NULL,
  ADD COLUMN IF NOT EXISTS adjustment_reason_codes jsonb NOT NULL DEFAULT '[]'::jsonb;

UPDATE attempt_execution_records
SET realized_effective_hash = NULL
WHERE realized_effective_hash = '';

ALTER TABLE execution_realization_observations
  ADD COLUMN IF NOT EXISTS adjustment_reason_codes jsonb NOT NULL DEFAULT '[]'::jsonb;
