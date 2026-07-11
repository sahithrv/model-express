ALTER TABLE automl_studies ADD COLUMN IF NOT EXISTS capability_version text NOT NULL DEFAULT '';
ALTER TABLE automl_studies ADD COLUMN IF NOT EXISTS task text NOT NULL DEFAULT '';
ALTER TABLE automl_studies ADD COLUMN IF NOT EXISTS runner text NOT NULL DEFAULT '';

ALTER TABLE automl_suggestions ADD COLUMN IF NOT EXISTS capability_version text NOT NULL DEFAULT '';
ALTER TABLE automl_suggestions ADD COLUMN IF NOT EXISTS task text NOT NULL DEFAULT '';
ALTER TABLE automl_suggestions ADD COLUMN IF NOT EXISTS runner text NOT NULL DEFAULT '';

ALTER TABLE automl_trials ADD COLUMN IF NOT EXISTS capability_version text NOT NULL DEFAULT '';
ALTER TABLE automl_trials ADD COLUMN IF NOT EXISTS task text NOT NULL DEFAULT '';
ALTER TABLE automl_trials ADD COLUMN IF NOT EXISTS runner text NOT NULL DEFAULT '';
