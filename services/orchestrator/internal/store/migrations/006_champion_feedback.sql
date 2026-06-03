CREATE SEQUENCE IF NOT EXISTS champion_feedback_id_seq;

CREATE TABLE IF NOT EXISTS champion_feedback (
  id text PRIMARY KEY DEFAULT 'champion_feedback_' || nextval('champion_feedback_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  champion_id text NOT NULL REFERENCES project_champions(id) ON DELETE CASCADE,
  prediction_id text NOT NULL DEFAULT '',
  job_id text NOT NULL DEFAULT '',
  dataset_id text NOT NULL DEFAULT '',
  image_uri text NOT NULL DEFAULT '',
  image_id text NOT NULL DEFAULT '',
  rating text NOT NULL,
  message text NOT NULL DEFAULT '',
  prediction_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
  metrics_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT champion_feedback_rating_check CHECK (rating IN ('good', 'mediocre', 'bad'))
);

CREATE INDEX IF NOT EXISTS idx_champion_feedback_project_created_at ON champion_feedback(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_champion_feedback_champion_created_at ON champion_feedback(champion_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_champion_feedback_prediction_id ON champion_feedback(prediction_id);
