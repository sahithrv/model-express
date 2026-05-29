CREATE SEQUENCE IF NOT EXISTS automl_study_id_seq;
CREATE SEQUENCE IF NOT EXISTS automl_suggestion_id_seq;
CREATE SEQUENCE IF NOT EXISTS automl_trial_id_seq;

ALTER TABLE automation_settings ADD COLUMN IF NOT EXISTS automl_enabled boolean NOT NULL DEFAULT false;
ALTER TABLE automation_settings ADD COLUMN IF NOT EXISTS automl_sampler text NOT NULL DEFAULT 'seeded_random';

CREATE TABLE IF NOT EXISTS automl_studies (
  id text PRIMARY KEY DEFAULT 'automl_study_' || nextval('automl_study_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  plan_id text NOT NULL DEFAULT '',
  dataset_id text NOT NULL DEFAULT '',
  source_decision_id text NOT NULL DEFAULT '',
  experiment_index integer NOT NULL DEFAULT 0,
  model text NOT NULL DEFAULT '',
  intent jsonb NOT NULL DEFAULT '{}'::jsonb,
  sampler text NOT NULL DEFAULT 'seeded_random',
  seed bigint NOT NULL DEFAULT 0,
  search_space jsonb NOT NULL DEFAULT '{}'::jsonb,
  strategy_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS automl_suggestions (
  id text PRIMARY KEY DEFAULT 'automl_suggestion_' || nextval('automl_suggestion_id_seq'),
  study_id text NOT NULL DEFAULT '',
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  plan_id text NOT NULL DEFAULT '',
  dataset_id text NOT NULL DEFAULT '',
  job_id text NOT NULL DEFAULT '',
  experiment_index integer NOT NULL DEFAULT 0,
  model text NOT NULL DEFAULT '',
  sampler text NOT NULL DEFAULT 'seeded_random',
  seed bigint NOT NULL DEFAULT 0,
  values jsonb NOT NULL DEFAULT '{}'::jsonb,
  final_values jsonb NOT NULL DEFAULT '{}'::jsonb,
  provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
  validation_status text NOT NULL DEFAULT '',
  validation_errors jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS automl_trials (
  id text PRIMARY KEY DEFAULT 'automl_trial_' || nextval('automl_trial_id_seq'),
  study_id text NOT NULL DEFAULT '',
  suggestion_id text NOT NULL DEFAULT '',
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  plan_id text NOT NULL DEFAULT '',
  dataset_id text NOT NULL DEFAULT '',
  job_id text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT '',
  target_metric text NOT NULL DEFAULT '',
  score double precision NOT NULL DEFAULT 0,
  metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
  error text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_automl_trials_job_id ON automl_trials(job_id) WHERE job_id <> '';
CREATE INDEX IF NOT EXISTS idx_automl_studies_project_created_at ON automl_studies(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_automl_studies_plan_experiment ON automl_studies(plan_id, experiment_index);
CREATE INDEX IF NOT EXISTS idx_automl_suggestions_project_created_at ON automl_suggestions(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_automl_suggestions_plan_experiment ON automl_suggestions(plan_id, experiment_index);
CREATE INDEX IF NOT EXISTS idx_automl_suggestions_job_id ON automl_suggestions(job_id) WHERE job_id <> '';
CREATE INDEX IF NOT EXISTS idx_automl_trials_project_updated_at ON automl_trials(project_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_automl_trials_study_id ON automl_trials(study_id, created_at);
