ALTER TABLE planner_candidate_provenance
  ADD COLUMN IF NOT EXISTS attempt_id text,
  ADD COLUMN IF NOT EXISTS actual_score double precision,
  ADD COLUMN IF NOT EXISTS actual_delta double precision,
  ADD COLUMN IF NOT EXISTS terminal_state text,
  ADD COLUMN IF NOT EXISTS cost_usd double precision,
  ADD COLUMN IF NOT EXISTS runtime_seconds double precision,
  ADD COLUMN IF NOT EXISTS calibration_eligible boolean,
  ADD COLUMN IF NOT EXISTS eligibility_reason text,
  ADD COLUMN IF NOT EXISTS finalized_at timestamptz;

ALTER TABLE planner_candidate_provenance
  DROP CONSTRAINT IF EXISTS chk_planner_candidate_provenance_outcome;

ALTER TABLE planner_candidate_provenance
  ADD CONSTRAINT chk_planner_candidate_provenance_outcome CHECK (
    outcome_status IN ('unknown', 'observed', 'unobserved')
    AND (terminal_state IS NULL OR terminal_state IN ('SUCCEEDED', 'FAILED', 'CANCELLED', 'SKIPPED'))
  );

ALTER TABLE planner_candidate_provenance
  DROP CONSTRAINT IF EXISTS chk_planner_candidate_provenance_finalized_outcome;

ALTER TABLE planner_candidate_provenance
  ADD CONSTRAINT chk_planner_candidate_provenance_finalized_outcome CHECK (
    (
      outcome_status = 'unknown'
      AND actual_score IS NULL
      AND actual_delta IS NULL
      AND terminal_state IS NULL
      AND cost_usd IS NULL
      AND runtime_seconds IS NULL
      AND calibration_eligible IS NULL
      AND eligibility_reason IS NULL
      AND finalized_at IS NULL
    )
    OR (
      outcome_status = 'observed'
      AND selected
      AND terminal_state = 'SUCCEEDED'
      AND actual_score BETWEEN 0 AND 1
      AND actual_delta IS NOT NULL
      AND calibration_eligible = true
      AND length(btrim(eligibility_reason)) > 0
      AND finalized_at IS NOT NULL
    )
    OR (
      outcome_status = 'unobserved'
      AND selected
      AND terminal_state IS NOT NULL
      AND actual_score IS NULL
      AND actual_delta IS NULL
      AND calibration_eligible = false
      AND length(btrim(eligibility_reason)) > 0
      AND finalized_at IS NOT NULL
    )
  );

ALTER TABLE planner_candidate_provenance
  DROP CONSTRAINT IF EXISTS chk_planner_candidate_provenance_unselected_outcome;

ALTER TABLE planner_candidate_provenance
  ADD CONSTRAINT chk_planner_candidate_provenance_unselected_outcome CHECK (
    selected
    OR (
      outcome_status = 'unknown'
      AND followup_plan_id IS NULL
      AND experiment_id IS NULL
      AND job_id IS NULL
      AND attempt_id IS NULL
      AND realized_effective_hash IS NULL
      AND actual_score IS NULL
      AND actual_delta IS NULL
      AND terminal_state IS NULL
      AND cost_usd IS NULL
      AND runtime_seconds IS NULL
      AND calibration_eligible IS NULL
      AND eligibility_reason IS NULL
      AND finalized_at IS NULL
    )
  );

ALTER TABLE planner_candidate_provenance
  DROP CONSTRAINT IF EXISTS chk_planner_candidate_provenance_outcome_cost_runtime;

ALTER TABLE planner_candidate_provenance
  ADD CONSTRAINT chk_planner_candidate_provenance_outcome_cost_runtime CHECK (
    (cost_usd IS NULL OR cost_usd >= 0)
    AND (runtime_seconds IS NULL OR runtime_seconds >= 0)
  );

CREATE INDEX IF NOT EXISTS idx_planner_candidate_provenance_followup_plan
  ON planner_candidate_provenance(followup_plan_id, selected_experiment_index)
  WHERE selected;

-- Historical rows are intentionally not inferred here. Runtime finalization
-- attaches only source-decision/plan/index/job mappings that are unambiguous.
