ALTER TABLE agent_invocations
  ADD COLUMN IF NOT EXISTS strict_validation_verdict jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS validation_outcome jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_agent_invocations_shadow_strict_would_block
  ON agent_invocations(project_id, planner_variant_id, created_at DESC)
  WHERE strict_validation_verdict->>'status' = 'would_block';

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_agent_invocations_strict_verdict_object' AND conrelid = 'agent_invocations'::regclass) THEN
    ALTER TABLE agent_invocations
      ADD CONSTRAINT chk_agent_invocations_strict_verdict_object CHECK (jsonb_typeof(strict_validation_verdict) = 'object');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_agent_invocations_validation_outcome_object' AND conrelid = 'agent_invocations'::regclass) THEN
    ALTER TABLE agent_invocations
      ADD CONSTRAINT chk_agent_invocations_validation_outcome_object CHECK (jsonb_typeof(validation_outcome) = 'object');
  END IF;
END $$;
