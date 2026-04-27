CREATE SEQUENCE IF NOT EXISTS project_id_seq;
CREATE SEQUENCE IF NOT EXISTS dataset_id_seq;
CREATE SEQUENCE IF NOT EXISTS worker_id_seq;
CREATE SEQUENCE IF NOT EXISTS job_id_seq;
CREATE SEQUENCE IF NOT EXISTS experiment_plan_id_seq;

CREATE TABLE IF NOT EXISTS projects (
  id text PRIMARY KEY DEFAULT 'project_' || nextval('project_id_seq'),
  name text NOT NULL,
  goal text NOT NULL,
  status text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workers (
  id text PRIMARY KEY DEFAULT 'worker_' || nextval('worker_id_seq'),
  project_id text NOT NULL DEFAULT '',
  name text NOT NULL,
  status text NOT NULL,
  gpu_type text NOT NULL DEFAULT '',
  last_heartbeat timestamptz NOT NULL DEFAULT now(),
  current_job_id text NOT NULL DEFAULT ''
);

ALTER TABLE workers ADD COLUMN IF NOT EXISTS project_id text NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS datasets (
  id text PRIMARY KEY DEFAULT 'dataset_' || nextval('dataset_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name text NOT NULL,
  storage_uri text NOT NULL,
  checksum_sha256 text NOT NULL DEFAULT '',
  size_bytes bigint NOT NULL DEFAULT 0,
  profile jsonb NOT NULL DEFAULT '{}'::jsonb,
  status text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  profiled_at timestamptz
);

CREATE TABLE IF NOT EXISTS experiment_plans (
  id text PRIMARY KEY DEFAULT 'plan_' || nextval('experiment_plan_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  status text NOT NULL,
  target_metric text NOT NULL,
  recommended_workers integer NOT NULL,
  estimated_minutes integer NOT NULL,
  experiments jsonb NOT NULL DEFAULT '[]'::jsonb,
  warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiment_jobs (
  id text PRIMARY KEY DEFAULT 'job_' || nextval('job_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  worker_id text NOT NULL DEFAULT '',
  template text NOT NULL,
  status text NOT NULL,
  config jsonb NOT NULL DEFAULT '{}'::jsonb,
  mlflow_run_id text NOT NULL DEFAULT '',
  error text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS epoch_metrics (
  job_id text NOT NULL REFERENCES experiment_jobs(id) ON DELETE CASCADE,
  epoch integer NOT NULL,
  metrics jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (job_id, epoch)
);

CREATE TABLE IF NOT EXISTS training_run_summaries (
  job_id text PRIMARY KEY REFERENCES experiment_jobs(id) ON DELETE CASCADE,
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  plan_id text NOT NULL DEFAULT '',
  dataset_id text NOT NULL DEFAULT '',
  model text NOT NULL DEFAULT '',
  provider text NOT NULL DEFAULT '',
  gpu_type text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT '',
  runtime_seconds double precision NOT NULL DEFAULT 0,
  estimated_cost_usd double precision NOT NULL DEFAULT 0,
  best_macro_f1 double precision NOT NULL DEFAULT 0,
  best_accuracy double precision NOT NULL DEFAULT 0,
  final_train_loss double precision NOT NULL DEFAULT 0,
  final_val_loss double precision NOT NULL DEFAULT 0,
  epochs_completed integer NOT NULL DEFAULT 0,
  modal_function_call_id text NOT NULL DEFAULT '',
  modal_input_id text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_experiment_jobs_project_id ON experiment_jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_experiment_jobs_status_created_at ON experiment_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_epoch_metrics_job_id ON epoch_metrics(job_id);
CREATE INDEX IF NOT EXISTS idx_workers_project_id ON workers(project_id);
CREATE INDEX IF NOT EXISTS idx_datasets_project_id ON datasets(project_id);
CREATE INDEX IF NOT EXISTS idx_experiment_plans_project_id ON experiment_plans(project_id);
CREATE INDEX IF NOT EXISTS idx_training_run_summaries_project_id ON training_run_summaries(project_id);
CREATE INDEX IF NOT EXISTS idx_training_run_summaries_plan_id ON training_run_summaries(plan_id);
