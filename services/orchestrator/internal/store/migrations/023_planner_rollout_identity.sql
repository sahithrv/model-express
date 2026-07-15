ALTER TABLE agent_invocations
  ADD COLUMN IF NOT EXISTS rollout_cohort_id text NOT NULL DEFAULT 'legacy_unknown',
  ADD COLUMN IF NOT EXISTS rollout_policy_id text NOT NULL DEFAULT 'legacy_unknown',
  ADD COLUMN IF NOT EXISTS rollout_assignment jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE planner_candidate_provenance
  ADD COLUMN IF NOT EXISTS rollout_cohort_id text NOT NULL DEFAULT 'legacy_unknown',
  ADD COLUMN IF NOT EXISTS rollout_policy_id text NOT NULL DEFAULT 'legacy_unknown';

ALTER TABLE agent_invocations
  DROP CONSTRAINT IF EXISTS chk_agent_invocations_rollout_identity;

ALTER TABLE agent_invocations
  ADD CONSTRAINT chk_agent_invocations_rollout_identity CHECK (
    length(btrim(rollout_cohort_id)) > 0
    AND length(btrim(rollout_policy_id)) > 0
    AND jsonb_typeof(rollout_assignment) = 'object'
  );

ALTER TABLE planner_candidate_provenance
  DROP CONSTRAINT IF EXISTS chk_planner_candidate_provenance_rollout_identity;

ALTER TABLE planner_candidate_provenance
  ADD CONSTRAINT chk_planner_candidate_provenance_rollout_identity CHECK (
    length(btrim(rollout_cohort_id)) > 0
    AND length(btrim(rollout_policy_id)) > 0
  );

CREATE INDEX IF NOT EXISTS idx_agent_invocations_rollout_policy_cohort_created
  ON agent_invocations(project_id, rollout_policy_id, rollout_cohort_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_planner_candidate_provenance_rollout_policy_cohort_created
  ON planner_candidate_provenance(project_id, rollout_policy_id, rollout_cohort_id, created_at DESC);
