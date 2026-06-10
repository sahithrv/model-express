CREATE INDEX IF NOT EXISTS idx_experiment_jobs_project_created_desc
  ON experiment_jobs (project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_training_run_summaries_project_updated_desc
  ON training_run_summaries (project_id, updated_at DESC);
