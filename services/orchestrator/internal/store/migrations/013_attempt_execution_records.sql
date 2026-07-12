CREATE SEQUENCE IF NOT EXISTS attempt_execution_record_id_seq;
CREATE SEQUENCE IF NOT EXISTS execution_realization_observation_id_seq;

CREATE TABLE IF NOT EXISTS job_execution_specs (
  job_id text PRIMARY KEY REFERENCES experiment_jobs(id) ON DELETE CASCADE,
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  schema_version text NOT NULL,
  capability_version text NOT NULL,
  task text NOT NULL,
  runner text NOT NULL,
  requested_config_hash text NOT NULL,
  accepted_spec_hash text NOT NULL,
  accepted_spec jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_job_execution_specs_project_created
  ON job_execution_specs(project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS attempt_execution_records (
  id text PRIMARY KEY DEFAULT 'attempt_execution_' || nextval('attempt_execution_record_id_seq'),
  job_id text NOT NULL REFERENCES experiment_jobs(id) ON DELETE CASCADE,
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  attempt_id text NOT NULL,
  attempt_number integer NOT NULL CHECK (attempt_number > 0),
  lifecycle_status text NOT NULL DEFAULT 'PENDING',
  fidelity_verdict text,
  realized_effective_hash text NOT NULL DEFAULT '',
  latest_realized_config jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(job_id, attempt_id),
  UNIQUE(job_id, attempt_number)
);

CREATE INDEX IF NOT EXISTS idx_attempt_execution_records_job_created
  ON attempt_execution_records(job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_attempt_execution_records_project_created
  ON attempt_execution_records(project_id, created_at DESC);

CREATE TABLE IF NOT EXISTS execution_realization_observations (
  id text PRIMARY KEY DEFAULT 'realization_observation_' || nextval('execution_realization_observation_id_seq'),
  attempt_record_id text NOT NULL REFERENCES attempt_execution_records(id) ON DELETE CASCADE,
  attempt_id text NOT NULL,
  schema_version text NOT NULL,
  stage text NOT NULL,
  idempotency_key text NOT NULL,
  realized_config jsonb NOT NULL,
  framework_arguments jsonb NOT NULL DEFAULT '{}'::jsonb,
  evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
  adjustment_policy text NOT NULL DEFAULT '',
  simulated boolean NOT NULL DEFAULT false,
  realized_effective_hash text NOT NULL,
  fidelity_verdict text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(attempt_record_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS idx_execution_observations_attempt_created
  ON execution_realization_observations(attempt_record_id, created_at ASC);
