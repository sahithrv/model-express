CREATE TABLE IF NOT EXISTS remote_training_sessions (
  id text PRIMARY KEY,
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id text NOT NULL REFERENCES experiment_jobs(id) ON DELETE CASCADE,
  training_attempt_id text NOT NULL,
  status text NOT NULL DEFAULT 'active',
  callback_token_hash text NOT NULL DEFAULT '',
  orchestrator_public_url text NOT NULL DEFAULT '',
  storage_public_url text NOT NULL DEFAULT '',
  storage_prefix text NOT NULL DEFAULT '',
  storage_scope jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL,
  closed_at timestamptz,
  UNIQUE(job_id, training_attempt_id)
);

CREATE INDEX IF NOT EXISTS idx_remote_training_sessions_project_status
  ON remote_training_sessions(project_id, status, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_remote_training_sessions_job_attempt
  ON remote_training_sessions(job_id, training_attempt_id);

CREATE INDEX IF NOT EXISTS idx_remote_training_sessions_expires
  ON remote_training_sessions(expires_at);
