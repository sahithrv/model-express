ALTER TABLE workers
  ADD COLUMN IF NOT EXISTS policy_capability_versions jsonb NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS artifact_capability_versions jsonb NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE workers
  DROP CONSTRAINT IF EXISTS chk_worker_capability_versions;
ALTER TABLE workers
  ADD CONSTRAINT chk_worker_capability_versions CHECK (
    jsonb_typeof(policy_capability_versions) = 'array'
    AND jsonb_typeof(artifact_capability_versions) = 'array'
  );

ALTER TABLE job_execution_specs
  ADD COLUMN IF NOT EXISTS artifact_plan jsonb NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS artifact_plan_hash text NOT NULL DEFAULT '';

ALTER TABLE job_execution_specs
  DROP CONSTRAINT IF EXISTS chk_job_execution_artifact_plan;
ALTER TABLE job_execution_specs
  ADD CONSTRAINT chk_job_execution_artifact_plan CHECK (
    jsonb_typeof(artifact_plan) = 'object'
    AND (artifact_plan_hash = '' OR artifact_plan_hash ~ '^sha256:[A-Za-z0-9_-]{43}$')
  );

ALTER TABLE attempt_execution_records
  ADD COLUMN IF NOT EXISTS worker_policy_capability_version text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS worker_artifact_capability_version text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS artifact_plan_hash text NOT NULL DEFAULT '';

ALTER TABLE attempt_execution_records
  DROP CONSTRAINT IF EXISTS chk_attempt_execution_artifact_plan_hash;
ALTER TABLE attempt_execution_records
  ADD CONSTRAINT chk_attempt_execution_artifact_plan_hash CHECK (
    artifact_plan_hash = '' OR artifact_plan_hash ~ '^sha256:[A-Za-z0-9_-]{43}$'
  );
