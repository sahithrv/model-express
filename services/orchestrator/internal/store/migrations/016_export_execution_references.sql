ALTER TABLE training_run_summaries
  ADD COLUMN IF NOT EXISTS execution_references jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE training_run_evaluations
  ADD COLUMN IF NOT EXISTS execution_references jsonb NOT NULL DEFAULT '{}'::jsonb;
