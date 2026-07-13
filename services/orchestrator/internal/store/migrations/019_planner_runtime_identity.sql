ALTER TABLE agent_invocations
  ADD COLUMN IF NOT EXISTS planner_variant_id text NOT NULL DEFAULT 'legacy_unknown',
  ADD COLUMN IF NOT EXISTS planner_variant jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS validation_mode text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS attempt_group_id text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS attempt_index integer NOT NULL DEFAULT -1,
  ADD COLUMN IF NOT EXISTS retry_reason text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS wall_latency_ms double precision NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS provider_usage jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS derived_cost jsonb NOT NULL DEFAULT '{}'::jsonb;

UPDATE agent_invocations
SET planner_variant_id = 'legacy_unknown'
WHERE planner_variant_id = '';

CREATE INDEX IF NOT EXISTS idx_agent_invocations_project_planner_variant_created
  ON agent_invocations(project_id, planner_variant_id, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_invocations_attempt_group_index
  ON agent_invocations(attempt_group_id, attempt_index)
  WHERE attempt_group_id <> '' AND attempt_index >= 0;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_agent_invocations_attempt_index' AND conrelid = 'agent_invocations'::regclass) THEN
    ALTER TABLE agent_invocations
      ADD CONSTRAINT chk_agent_invocations_attempt_index CHECK (attempt_index >= -1);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_agent_invocations_wall_latency' AND conrelid = 'agent_invocations'::regclass) THEN
    ALTER TABLE agent_invocations
      ADD CONSTRAINT chk_agent_invocations_wall_latency CHECK (wall_latency_ms >= 0);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'chk_agent_invocations_versioned_cost' AND conrelid = 'agent_invocations'::regclass) THEN
    ALTER TABLE agent_invocations
      ADD CONSTRAINT chk_agent_invocations_versioned_cost CHECK (
        derived_cost = '{}'::jsonb
        OR COALESCE((
          jsonb_typeof(derived_cost) = 'object'
          AND derived_cost ?& ARRAY[
            'pricing_version', 'currency', 'provider', 'model',
            'input_tokens', 'cached_input_tokens', 'output_tokens',
            'input_usd_per_million_tokens', 'cached_input_usd_per_million_tokens', 'output_usd_per_million_tokens',
            'uncached_input_cost_usd', 'cached_input_cost_usd', 'output_cost_usd', 'total_cost_usd'
          ]
          AND jsonb_typeof(derived_cost->'pricing_version') = 'string'
          AND jsonb_typeof(derived_cost->'total_cost_usd') = 'string'
          AND jsonb_typeof(derived_cost->'input_tokens') = 'number'
          AND jsonb_typeof(derived_cost->'cached_input_tokens') = 'number'
          AND jsonb_typeof(derived_cost->'output_tokens') = 'number'
          AND length(btrim(derived_cost->>'pricing_version')) > 0
          AND length(btrim(derived_cost->>'total_cost_usd')) > 0
        ), false)
      );
  END IF;
END $$;
