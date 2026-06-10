ALTER TABLE training_run_summaries ADD COLUMN IF NOT EXISTS dataset_materialization jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE training_run_summaries ADD COLUMN IF NOT EXISTS stage_telemetry jsonb NOT NULL DEFAULT '{}'::jsonb;
