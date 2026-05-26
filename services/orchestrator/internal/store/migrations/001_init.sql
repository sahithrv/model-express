CREATE SEQUENCE IF NOT EXISTS project_id_seq;
CREATE SEQUENCE IF NOT EXISTS dataset_id_seq;
CREATE SEQUENCE IF NOT EXISTS dataset_profile_id_seq;
CREATE SEQUENCE IF NOT EXISTS worker_id_seq;
CREATE SEQUENCE IF NOT EXISTS job_id_seq;
CREATE SEQUENCE IF NOT EXISTS experiment_plan_id_seq;
CREATE SEQUENCE IF NOT EXISTS agent_decision_id_seq;
CREATE SEQUENCE IF NOT EXISTS worker_requirement_id_seq;
CREATE SEQUENCE IF NOT EXISTS execution_event_id_seq;
CREATE SEQUENCE IF NOT EXISTS agent_memory_id_seq;
CREATE SEQUENCE IF NOT EXISTS agent_invocation_id_seq;
CREATE SEQUENCE IF NOT EXISTS project_champion_id_seq;
CREATE SEQUENCE IF NOT EXISTS champion_export_id_seq;
CREATE SEQUENCE IF NOT EXISTS champion_demo_prediction_id_seq;
CREATE SEQUENCE IF NOT EXISTS strategy_scorecard_id_seq;

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

CREATE TABLE IF NOT EXISTS dataset_profiles (
  id text PRIMARY KEY DEFAULT 'dataset_profile_' || nextval('dataset_profile_id_seq'),
  dataset_id text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  profile_version text NOT NULL DEFAULT 'v1',
  class_count integer NOT NULL DEFAULT 0,
  image_count integer NOT NULL DEFAULT 0,
  class_distribution jsonb NOT NULL DEFAULT '{}'::jsonb,
  imbalance_ratio double precision NOT NULL DEFAULT 0,
  image_dimension_stats jsonb NOT NULL DEFAULT '{}'::jsonb,
  corrupt_file_count integer NOT NULL DEFAULT 0,
  split_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
  leakage_warnings jsonb NOT NULL DEFAULT '{}'::jsonb,
  recommended_metrics jsonb NOT NULL DEFAULT '[]'::jsonb,
  dataset_traits jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiment_plans (
  id text PRIMARY KEY DEFAULT 'plan_' || nextval('experiment_plan_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  status text NOT NULL,
  source_decision_id text NOT NULL DEFAULT '',
  target_metric text NOT NULL,
  recommended_workers integer NOT NULL,
  estimated_minutes integer NOT NULL,
  experiments jsonb NOT NULL DEFAULT '[]'::jsonb,
  warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE experiment_plans ADD COLUMN IF NOT EXISTS source_decision_id text NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS experiment_jobs (
  id text PRIMARY KEY DEFAULT 'job_' || nextval('job_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  worker_id text NOT NULL DEFAULT '',
  template text NOT NULL,
  status text NOT NULL,
  config jsonb NOT NULL DEFAULT '{}'::jsonb,
  mlflow_run_id text NOT NULL DEFAULT '',
  error text NOT NULL DEFAULT '',
  attempt integer NOT NULL DEFAULT 0,
  max_attempts integer NOT NULL DEFAULT 3,
  lease_owner_worker_id text NOT NULL DEFAULT '',
  lease_expires_at timestamptz,
  lease_last_heartbeat_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  completed_at timestamptz
);

ALTER TABLE experiment_jobs ADD COLUMN IF NOT EXISTS attempt integer NOT NULL DEFAULT 0;
ALTER TABLE experiment_jobs ADD COLUMN IF NOT EXISTS max_attempts integer NOT NULL DEFAULT 3;
ALTER TABLE experiment_jobs ADD COLUMN IF NOT EXISTS lease_owner_worker_id text NOT NULL DEFAULT '';
ALTER TABLE experiment_jobs ADD COLUMN IF NOT EXISTS lease_expires_at timestamptz;
ALTER TABLE experiment_jobs ADD COLUMN IF NOT EXISTS lease_last_heartbeat_at timestamptz;

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

CREATE TABLE IF NOT EXISTS training_run_evaluations (
  job_id text PRIMARY KEY REFERENCES experiment_jobs(id) ON DELETE CASCADE,
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  plan_id text NOT NULL DEFAULT '',
  dataset_id text NOT NULL DEFAULT '',
  objective_profile jsonb NOT NULL DEFAULT '{}'::jsonb,
  per_class_metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
  confusion_matrix jsonb NOT NULL DEFAULT '[]'::jsonb,
  model_profile jsonb NOT NULL DEFAULT '{}'::jsonb,
  holistic_scores jsonb NOT NULL DEFAULT '{}'::jsonb,
  recommendation_summary text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS project_champions (
  id text PRIMARY KEY DEFAULT 'champion_' || nextval('project_champion_id_seq'),
  project_id text NOT NULL UNIQUE REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL DEFAULT '',
  plan_id text NOT NULL DEFAULT '',
  job_id text NOT NULL DEFAULT '',
  source_decision_id text NOT NULL DEFAULT '',
  selection_reason text NOT NULL DEFAULT '',
  metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
  evaluation jsonb NOT NULL DEFAULT '{}'::jsonb,
  deployment_profile jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS champion_exports (
  id text PRIMARY KEY DEFAULT 'champion_export_' || nextval('champion_export_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  champion_id text NOT NULL REFERENCES project_champions(id) ON DELETE CASCADE,
  job_id text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT 'PENDING_ARTIFACT',
  format text NOT NULL DEFAULT 'onnx',
  artifact_uri text NOT NULL DEFAULT '',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  validation_errors jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS champion_demo_predictions (
  id text PRIMARY KEY DEFAULT 'champion_demo_prediction_' || nextval('champion_demo_prediction_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  champion_id text NOT NULL REFERENCES project_champions(id) ON DELETE CASCADE,
  job_id text NOT NULL DEFAULT '',
  dataset_id text NOT NULL DEFAULT '',
  image_uri text NOT NULL DEFAULT '',
  image_id text NOT NULL DEFAULT '',
  image_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  status text NOT NULL DEFAULT 'PENDING',
  predicted_label text NOT NULL DEFAULT '',
  true_label text NOT NULL DEFAULT '',
  confidence double precision,
  top_k jsonb NOT NULL DEFAULT '[]'::jsonb,
  latency_ms double precision,
  correct boolean,
  error text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_decisions (
  id text PRIMARY KEY DEFAULT 'decision_' || nextval('agent_decision_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  plan_id text NOT NULL DEFAULT '',
  decision_type text NOT NULL,
  rationale text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS automation_settings (
  singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
  auto_review_experiments boolean NOT NULL DEFAULT false,
  auto_schedule_followups boolean NOT NULL DEFAULT false,
  auto_execute_plans boolean NOT NULL DEFAULT false,
  max_followup_rounds integer NOT NULL DEFAULT 2,
  default_training_provider text NOT NULL DEFAULT 'local',
  default_gpu_type text NOT NULL DEFAULT '',
  llm_enabled boolean NOT NULL DEFAULT false,
  agent_mode text NOT NULL DEFAULT 'propose',
  llm_provider text NOT NULL DEFAULT 'openai',
  llm_model text NOT NULL DEFAULT '',
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE automation_settings ADD COLUMN IF NOT EXISTS llm_enabled boolean NOT NULL DEFAULT false;
ALTER TABLE automation_settings ADD COLUMN IF NOT EXISTS agent_mode text NOT NULL DEFAULT 'propose';
ALTER TABLE automation_settings ADD COLUMN IF NOT EXISTS llm_provider text NOT NULL DEFAULT 'openai';
ALTER TABLE automation_settings ADD COLUMN IF NOT EXISTS llm_model text NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS worker_requirements (
  id text PRIMARY KEY DEFAULT 'worker_requirement_' || nextval('worker_requirement_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  plan_id text NOT NULL DEFAULT '',
  provider text NOT NULL DEFAULT 'local',
  gpu_type text NOT NULL DEFAULT '',
  target_count integer NOT NULL DEFAULT 1,
  status text NOT NULL DEFAULT 'PENDING',
  source text NOT NULL DEFAULT '',
  last_error text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(project_id, plan_id)
);

CREATE TABLE IF NOT EXISTS execution_events (
  id text PRIMARY KEY DEFAULT 'execution_event_' || nextval('execution_event_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  plan_id text NOT NULL DEFAULT '',
  event_type text NOT NULL,
  message text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_invocations (
  id text PRIMARY KEY DEFAULT 'agent_invocation_' || nextval('agent_invocation_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL DEFAULT '',
  plan_id text NOT NULL DEFAULT '',
  job_id text NOT NULL DEFAULT '',
  agent_name text NOT NULL,
  agent_version text NOT NULL DEFAULT '',
  prompt_version text NOT NULL DEFAULT '',
  provider text NOT NULL DEFAULT '',
  model text NOT NULL DEFAULT '',
  input_messages jsonb NOT NULL DEFAULT '[]'::jsonb,
  input_context jsonb NOT NULL DEFAULT '{}'::jsonb,
  raw_output text NOT NULL DEFAULT '',
  parsed_output jsonb NOT NULL DEFAULT '{}'::jsonb,
  validation_status text NOT NULL DEFAULT '',
  validation_error text NOT NULL DEFAULT '',
  accepted_for_memory boolean NOT NULL DEFAULT false,
  human_feedback jsonb NOT NULL DEFAULT '{}'::jsonb,
  downstream_outcome jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_memory_records (
  id text PRIMARY KEY DEFAULT 'memory_' || nextval('agent_memory_id_seq'),
  invocation_id text NOT NULL DEFAULT '',
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL DEFAULT '',
  plan_id text NOT NULL DEFAULT '',
  job_id text NOT NULL DEFAULT '',
  agent_name text NOT NULL,
  kind text NOT NULL,
  summary text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  tags jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE agent_memory_records ADD COLUMN IF NOT EXISTS invocation_id text NOT NULL DEFAULT '';

CREATE TABLE IF NOT EXISTS strategy_scorecards (
  id text PRIMARY KEY DEFAULT 'strategy_scorecard_' || nextval('strategy_scorecard_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL DEFAULT '',
  source_decision_id text NOT NULL DEFAULT '',
  source_plan_id text NOT NULL DEFAULT '',
  followup_plan_id text NOT NULL DEFAULT '',
  strategy_type text NOT NULL DEFAULT '',
  planning_mode text NOT NULL DEFAULT '',
  dataset_traits jsonb NOT NULL DEFAULT '{}'::jsonb,
  objective_profile jsonb NOT NULL DEFAULT '{}'::jsonb,
  proposed_changes jsonb NOT NULL DEFAULT '{}'::jsonb,
  expected_delta double precision NOT NULL DEFAULT 0,
  actual_delta double precision NOT NULL DEFAULT 0,
  confidence_before double precision NOT NULL DEFAULT 0,
  confidence_after double precision NOT NULL DEFAULT 0,
  cost_usd double precision NOT NULL DEFAULT 0,
  runtime_seconds double precision NOT NULL DEFAULT 0,
  outcome text NOT NULL DEFAULT 'pending',
  lesson text NOT NULL DEFAULT '',
  tags jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_experiment_jobs_project_id ON experiment_jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_experiment_jobs_status_created_at ON experiment_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_experiment_jobs_project_status_created_at ON experiment_jobs(project_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_experiment_jobs_lease_expires_at ON experiment_jobs(status, lease_expires_at) WHERE lease_expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_epoch_metrics_job_id ON epoch_metrics(job_id);
CREATE INDEX IF NOT EXISTS idx_workers_project_id ON workers(project_id);
CREATE INDEX IF NOT EXISTS idx_datasets_project_id ON datasets(project_id);
CREATE INDEX IF NOT EXISTS idx_dataset_profiles_dataset_id ON dataset_profiles(dataset_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_experiment_plans_project_id ON experiment_plans(project_id);
CREATE INDEX IF NOT EXISTS idx_experiment_plans_source_decision_id ON experiment_plans(source_decision_id) WHERE source_decision_id <> '';
CREATE INDEX IF NOT EXISTS idx_training_run_summaries_project_id ON training_run_summaries(project_id);
CREATE INDEX IF NOT EXISTS idx_training_run_summaries_plan_id ON training_run_summaries(plan_id);
CREATE INDEX IF NOT EXISTS idx_training_run_evaluations_project_id ON training_run_evaluations(project_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_run_evaluations_plan_id ON training_run_evaluations(plan_id);
CREATE INDEX IF NOT EXISTS idx_project_champions_project_id ON project_champions(project_id);
CREATE INDEX IF NOT EXISTS idx_champion_exports_project_created_at ON champion_exports(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_champion_exports_champion_id ON champion_exports(champion_id);
CREATE INDEX IF NOT EXISTS idx_champion_demo_predictions_project_created_at ON champion_demo_predictions(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_champion_demo_predictions_champion_id ON champion_demo_predictions(champion_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_project_id ON agent_decisions(project_id);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_plan_id ON agent_decisions(plan_id);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_project_plan_created_at ON agent_decisions(project_id, plan_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_requirements_project_id ON worker_requirements(project_id);
CREATE INDEX IF NOT EXISTS idx_worker_requirements_status ON worker_requirements(status);
CREATE INDEX IF NOT EXISTS idx_worker_requirements_project_status ON worker_requirements(project_id, status);
CREATE INDEX IF NOT EXISTS idx_execution_events_project_id ON execution_events(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_invocations_project_id ON agent_invocations(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_invocations_dataset_id ON agent_invocations(dataset_id);
CREATE INDEX IF NOT EXISTS idx_agent_invocations_agent_name ON agent_invocations(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_invocations_job_id ON agent_invocations(job_id);
CREATE INDEX IF NOT EXISTS idx_agent_invocations_project_dataset_agent_created_at ON agent_invocations(project_id, dataset_id, agent_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_memory_project_id ON agent_memory_records(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_memory_dataset_id ON agent_memory_records(dataset_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_kind ON agent_memory_records(kind);
CREATE INDEX IF NOT EXISTS idx_agent_memory_invocation_id ON agent_memory_records(invocation_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_project_dataset_kind_created_at ON agent_memory_records(project_id, dataset_id, kind, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_scorecards_project_id ON strategy_scorecards(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_scorecards_dataset_id ON strategy_scorecards(dataset_id);
CREATE INDEX IF NOT EXISTS idx_strategy_scorecards_followup_plan_id ON strategy_scorecards(followup_plan_id);
CREATE INDEX IF NOT EXISTS idx_strategy_scorecards_outcome ON strategy_scorecards(outcome);
