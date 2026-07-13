ALTER TABLE strategy_scorecards
  ADD COLUMN IF NOT EXISTS fidelity_verdicts jsonb NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS evidence_eligible boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS requested_mechanism text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS realized_mechanism_identity text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS accepted_spec_hash text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS realized_effective_hash text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS adjustment_reason_codes jsonb NOT NULL DEFAULT '[]'::jsonb;

CREATE INDEX IF NOT EXISTS idx_strategy_scorecards_fidelity_eligible
  ON strategy_scorecards(project_id, evidence_eligible, created_at DESC);
