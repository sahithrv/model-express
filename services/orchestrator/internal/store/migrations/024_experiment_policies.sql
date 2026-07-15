CREATE SEQUENCE IF NOT EXISTS compatibility_profile_id_seq;
CREATE SEQUENCE IF NOT EXISTS experiment_policy_version_id_seq;
CREATE SEQUENCE IF NOT EXISTS experiment_policy_version_revision_seq;
CREATE SEQUENCE IF NOT EXISTS experiment_policy_binding_id_seq;
CREATE SEQUENCE IF NOT EXISTS experiment_policy_evaluation_id_seq;

CREATE TABLE IF NOT EXISTS accounts (
  id text PRIMARY KEY,
  name text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO accounts (id, name)
VALUES ('account_local_default', 'Local default')
ON CONFLICT (id) DO NOTHING;

ALTER TABLE projects
  ADD COLUMN IF NOT EXISTS account_id text REFERENCES accounts(id);

UPDATE projects
SET account_id = 'account_local_default'
WHERE account_id IS NULL;

ALTER TABLE projects
  ALTER COLUMN account_id SET DEFAULT 'account_local_default',
  ALTER COLUMN account_id SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_projects_account_created
  ON projects(account_id, created_at DESC);

CREATE TABLE IF NOT EXISTS compatibility_profiles (
  id text PRIMARY KEY DEFAULT 'compatibility_profile_' || nextval('compatibility_profile_id_seq'),
  profile_key text NOT NULL,
  semantic_version text NOT NULL,
  schema_version text NOT NULL,
  catalog_version text NOT NULL,
  document jsonb NOT NULL,
  document_hash text NOT NULL,
  owner_account_id text REFERENCES accounts(id) ON DELETE RESTRICT,
  created_at timestamptz NOT NULL DEFAULT now(),
  created_by text NOT NULL,
  UNIQUE(profile_key, semantic_version),
  CONSTRAINT chk_compatibility_profile_document_object CHECK (jsonb_typeof(document) = 'object'),
  CONSTRAINT chk_compatibility_profile_hash CHECK (document_hash ~ '^sha256:[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS experiment_policy_versions (
  id text PRIMARY KEY DEFAULT 'policy_version_' || nextval('experiment_policy_version_id_seq'),
  owner_account_id text NOT NULL REFERENCES accounts(id) ON DELETE RESTRICT,
  schema_version text NOT NULL,
  revision bigint NOT NULL UNIQUE DEFAULT nextval('experiment_policy_version_revision_seq'),
  document jsonb NOT NULL,
  document_hash text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  created_by text NOT NULL,
  CONSTRAINT chk_experiment_policy_version_document_object CHECK (jsonb_typeof(document) = 'object'),
  CONSTRAINT chk_experiment_policy_version_hash CHECK (document_hash ~ '^sha256:[0-9a-f]{64}$')
);

CREATE INDEX IF NOT EXISTS idx_experiment_policy_versions_owner_revision
  ON experiment_policy_versions(owner_account_id, revision DESC);

CREATE TABLE IF NOT EXISTS experiment_policy_bindings (
  id text PRIMARY KEY DEFAULT 'policy_binding_' || nextval('experiment_policy_binding_id_seq'),
  scope text NOT NULL CHECK (scope IN ('account', 'project', 'dataset', 'run')),
  policy_version_id text NOT NULL REFERENCES experiment_policy_versions(id) ON DELETE RESTRICT,
  account_id text REFERENCES accounts(id) ON DELETE RESTRICT,
  project_id text REFERENCES projects(id) ON DELETE RESTRICT,
  dataset_id text REFERENCES datasets(id) ON DELETE RESTRICT,
  experiment_job_id text REFERENCES experiment_jobs(id) ON DELETE RESTRICT,
  revision bigint NOT NULL CHECK (revision > 0),
  active boolean NOT NULL DEFAULT true,
  supersedes_id text REFERENCES experiment_policy_bindings(id) ON DELETE RESTRICT,
  superseded_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  created_by text NOT NULL,
  CONSTRAINT chk_experiment_policy_binding_subject CHECK (
    num_nonnulls(account_id, project_id, dataset_id, experiment_job_id) = 1
    AND (scope <> 'account' OR account_id IS NOT NULL)
    AND (scope <> 'project' OR project_id IS NOT NULL)
    AND (scope <> 'dataset' OR dataset_id IS NOT NULL)
    AND (scope <> 'run' OR experiment_job_id IS NOT NULL)
  ),
  CONSTRAINT chk_experiment_policy_binding_superseded CHECK (
    (active AND superseded_at IS NULL)
    OR (NOT active AND superseded_at IS NOT NULL)
  )
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_policy_binding_account_active
  ON experiment_policy_bindings(account_id) WHERE active AND scope = 'account';
CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_policy_binding_project_active
  ON experiment_policy_bindings(project_id) WHERE active AND scope = 'project';
CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_policy_binding_dataset_active
  ON experiment_policy_bindings(dataset_id) WHERE active AND scope = 'dataset';
CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_policy_binding_run_active
  ON experiment_policy_bindings(experiment_job_id) WHERE active AND scope = 'run';
CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_policy_binding_account_revision
  ON experiment_policy_bindings(account_id, revision) WHERE scope = 'account';
CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_policy_binding_project_revision
  ON experiment_policy_bindings(project_id, revision) WHERE scope = 'project';
CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_policy_binding_dataset_revision
  ON experiment_policy_bindings(dataset_id, revision) WHERE scope = 'dataset';
CREATE UNIQUE INDEX IF NOT EXISTS uq_experiment_policy_binding_run_revision
  ON experiment_policy_bindings(experiment_job_id, revision) WHERE scope = 'run';

CREATE TABLE IF NOT EXISTS experiment_policy_evaluations (
  id text PRIMARY KEY DEFAULT 'policy_evaluation_' || nextval('experiment_policy_evaluation_id_seq'),
  operation text NOT NULL,
  decision text NOT NULL CHECK (decision IN ('allowed', 'denied')),
  account_id text REFERENCES accounts(id) ON DELETE RESTRICT,
  project_id text REFERENCES projects(id) ON DELETE RESTRICT,
  dataset_id text REFERENCES datasets(id) ON DELETE RESTRICT,
  plan_id text REFERENCES experiment_plans(id) ON DELETE RESTRICT,
  job_id text REFERENCES experiment_jobs(id) ON DELETE RESTRICT,
  agent_invocation_id text REFERENCES agent_invocations(id) ON DELETE RESTRICT,
  champion_export_id text REFERENCES champion_exports(id) ON DELETE RESTRICT,
  candidate_config_hash text NOT NULL DEFAULT '',
  catalog_version text NOT NULL,
  compatibility_profile_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  policy_sources jsonb NOT NULL DEFAULT '[]'::jsonb,
  effective_snapshot jsonb NOT NULL,
  effective_policy_hash text NOT NULL,
  requested_capability_uses jsonb NOT NULL DEFAULT '[]'::jsonb,
  effective_capability_uses jsonb NOT NULL DEFAULT '[]'::jsonb,
  reason_codes jsonb NOT NULL DEFAULT '[]'::jsonb,
  findings jsonb NOT NULL DEFAULT '[]'::jsonb,
  actor_id text NOT NULL DEFAULT '',
  request_id text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT chk_experiment_policy_evaluation_snapshot CHECK (jsonb_typeof(effective_snapshot) = 'object'),
  CONSTRAINT chk_experiment_policy_evaluation_hash CHECK (effective_policy_hash ~ '^sha256:[0-9a-f]{64}$'),
  CONSTRAINT chk_experiment_policy_evaluation_arrays CHECK (
    jsonb_typeof(compatibility_profile_refs) = 'array'
    AND jsonb_typeof(policy_sources) = 'array'
    AND jsonb_typeof(requested_capability_uses) = 'array'
    AND jsonb_typeof(effective_capability_uses) = 'array'
    AND jsonb_typeof(reason_codes) = 'array'
    AND jsonb_typeof(findings) = 'array'
  )
);

CREATE INDEX IF NOT EXISTS idx_experiment_policy_evaluations_project_created
  ON experiment_policy_evaluations(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiment_policy_evaluations_job_created
  ON experiment_policy_evaluations(job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiment_policy_evaluations_hash
  ON experiment_policy_evaluations(effective_policy_hash);

ALTER TABLE experiment_plans
  ADD COLUMN IF NOT EXISTS proposal_policy_evaluation_id text REFERENCES experiment_policy_evaluations(id),
  ADD COLUMN IF NOT EXISTS effective_policy_hash text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS policy_status text NOT NULL DEFAULT '';

ALTER TABLE agent_decisions
  ADD COLUMN IF NOT EXISTS proposal_policy_evaluation_id text REFERENCES experiment_policy_evaluations(id),
  ADD COLUMN IF NOT EXISTS effective_policy_hash text NOT NULL DEFAULT '';

ALTER TABLE experiment_jobs
  ADD COLUMN IF NOT EXISTS dataset_id text REFERENCES datasets(id),
  ADD COLUMN IF NOT EXISTS plan_id text REFERENCES experiment_plans(id),
  ADD COLUMN IF NOT EXISTS schedule_policy_evaluation_id text REFERENCES experiment_policy_evaluations(id),
  ADD COLUMN IF NOT EXISTS effective_policy_hash text NOT NULL DEFAULT '',
  ADD COLUMN IF NOT EXISTS policy_eligibility_status text NOT NULL DEFAULT '';

ALTER TABLE job_execution_specs
  ADD COLUMN IF NOT EXISTS policy_evaluation_id text REFERENCES experiment_policy_evaluations(id),
  ADD COLUMN IF NOT EXISTS effective_policy_hash text NOT NULL DEFAULT '';

ALTER TABLE attempt_execution_records
  ADD COLUMN IF NOT EXISTS dispatch_policy_evaluation_id text REFERENCES experiment_policy_evaluations(id),
  ADD COLUMN IF NOT EXISTS effective_policy_hash text NOT NULL DEFAULT '';

ALTER TABLE champion_exports
  ADD COLUMN IF NOT EXISTS export_policy_evaluation_id text REFERENCES experiment_policy_evaluations(id),
  ADD COLUMN IF NOT EXISTS effective_policy_hash text NOT NULL DEFAULT '';

CREATE OR REPLACE FUNCTION reject_immutable_experiment_policy_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  RAISE EXCEPTION '% records are immutable', TG_TABLE_NAME;
END;
$$;

DROP TRIGGER IF EXISTS compatibility_profiles_immutable ON compatibility_profiles;
CREATE TRIGGER compatibility_profiles_immutable
  BEFORE UPDATE OR DELETE ON compatibility_profiles
  FOR EACH ROW EXECUTE FUNCTION reject_immutable_experiment_policy_mutation();

DROP TRIGGER IF EXISTS experiment_policy_versions_immutable ON experiment_policy_versions;
CREATE TRIGGER experiment_policy_versions_immutable
  BEFORE UPDATE OR DELETE ON experiment_policy_versions
  FOR EACH ROW EXECUTE FUNCTION reject_immutable_experiment_policy_mutation();

DROP TRIGGER IF EXISTS experiment_policy_evaluations_immutable ON experiment_policy_evaluations;
CREATE TRIGGER experiment_policy_evaluations_immutable
  BEFORE UPDATE OR DELETE ON experiment_policy_evaluations
  FOR EACH ROW EXECUTE FUNCTION reject_immutable_experiment_policy_mutation();

CREATE OR REPLACE FUNCTION restrict_experiment_policy_binding_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'experiment_policy_bindings records cannot be deleted';
  END IF;
  IF (to_jsonb(NEW) - ARRAY['active', 'superseded_at']::text[])
      IS DISTINCT FROM
     (to_jsonb(OLD) - ARRAY['active', 'superseded_at']::text[])
     OR OLD.active = false
     OR NEW.active = true
     OR OLD.superseded_at IS NOT NULL
     OR NEW.superseded_at IS NULL THEN
    RAISE EXCEPTION 'experiment_policy_bindings records are immutable except for one-way supersession';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS experiment_policy_bindings_restrict_mutation ON experiment_policy_bindings;
CREATE TRIGGER experiment_policy_bindings_restrict_mutation
  BEFORE UPDATE OR DELETE ON experiment_policy_bindings
  FOR EACH ROW EXECUTE FUNCTION restrict_experiment_policy_binding_mutation();
