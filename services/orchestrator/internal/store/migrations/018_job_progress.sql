CREATE TABLE IF NOT EXISTS job_progress (
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  job_id text NOT NULL REFERENCES experiment_jobs(id) ON DELETE CASCADE,
  attempt integer NOT NULL,
  taxonomy_version integer NOT NULL,
  stage text NOT NULL,
  detail_code text NOT NULL DEFAULT '',
  status text NOT NULL,
  current bigint,
  total bigint,
  unit text NOT NULL DEFAULT '',
  safe_message text NOT NULL DEFAULT '',
  revision bigint NOT NULL,
  heartbeat_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  PRIMARY KEY (job_id, attempt),
  CONSTRAINT job_progress_attempt_nonnegative CHECK (attempt >= 0),
  CONSTRAINT job_progress_taxonomy_supported CHECK (taxonomy_version = 1),
  CONSTRAINT job_progress_stage_valid CHECK (stage IN (
    'queued',
    'worker_starting',
    'remote_scheduled',
    'environment_starting',
    'dataset_materializing',
    'data_loading',
    'model_initializing',
    'training',
    'evaluating',
    'exporting',
    'finalizing',
    'completed',
    'failed',
    'cancelled'
  )),
  CONSTRAINT job_progress_status_valid CHECK (status IN (
    'queued', 'running', 'completed', 'failed', 'cancelled'
  )),
  CONSTRAINT job_progress_stage_status_consistent CHECK (
    (stage = 'queued' AND status = 'queued') OR
    (stage = 'completed' AND status = 'completed') OR
    (stage = 'failed' AND status = 'failed') OR
    (stage = 'cancelled' AND status = 'cancelled') OR
    (stage NOT IN ('queued', 'completed', 'failed', 'cancelled') AND status = 'running')
  ),
  CONSTRAINT job_progress_current_nonnegative CHECK (current IS NULL OR current >= 0),
  CONSTRAINT job_progress_total_nonnegative CHECK (total IS NULL OR total >= 0),
  CONSTRAINT job_progress_range_valid CHECK (current IS NULL OR total IS NULL OR current <= total),
  CONSTRAINT job_progress_detail_code_bounded CHECK (octet_length(detail_code) <= 64),
  CONSTRAINT job_progress_unit_bounded CHECK (octet_length(unit) <= 32),
  CONSTRAINT job_progress_safe_message_bounded CHECK (octet_length(safe_message) <= 512),
  CONSTRAINT job_progress_revision_bounded CHECK (
    revision >= 0 AND (
      stage IN ('completed', 'failed', 'cancelled') OR
      revision <= 9223372036854775806
    )
  ),
  CONSTRAINT job_progress_metadata_object CHECK (jsonb_typeof(metadata) = 'object'),
  CONSTRAINT job_progress_metadata_bounded CHECK (octet_length(metadata::text) <= 4096)
);

CREATE INDEX IF NOT EXISTS idx_job_progress_project_updated
  ON job_progress(project_id, updated_at DESC);
