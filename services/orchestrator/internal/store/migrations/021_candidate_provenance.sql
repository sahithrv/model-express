CREATE SEQUENCE IF NOT EXISTS candidate_provenance_id_seq;

CREATE TABLE IF NOT EXISTS planner_candidate_provenance (
  id text PRIMARY KEY DEFAULT 'candidate_provenance_' || nextval('candidate_provenance_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  invocation_id text NOT NULL REFERENCES agent_invocations(id) ON DELETE CASCADE,
  decision_id text NOT NULL REFERENCES agent_decisions(id) ON DELETE CASCADE,
  planner_variant_id text NOT NULL,
  candidate_index integer NOT NULL,
  requested_config_hash text NOT NULL,
  accepted_spec_hash text NOT NULL,
  task text NOT NULL,
  mechanism text NOT NULL,
  forecast_target text NOT NULL,
  metric_direction text NOT NULL,
  score_basis text NOT NULL,
  score_version text NOT NULL,
  baseline_job_id text NOT NULL DEFAULT '',
  baseline_score double precision NOT NULL,
  predicted_delta double precision NOT NULL,
  prediction_source text NOT NULL,
  forecast_units text NOT NULL,
  valid_range_min double precision NOT NULL,
  valid_range_max double precision NOT NULL,
  base_score double precision NOT NULL,
  selection_trace_reference text NOT NULL,
  selected boolean NOT NULL DEFAULT false,
  rejected boolean NOT NULL DEFAULT false,
  selection_state text NOT NULL,
  selected_experiment_index integer,
  outcome_status text NOT NULL DEFAULT 'unknown',
  reasons jsonb NOT NULL DEFAULT '[]'::jsonb,
  followup_plan_id text,
  experiment_id text,
  job_id text,
  realized_effective_hash text,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT uq_planner_candidate_provenance_decision_index UNIQUE (decision_id, candidate_index),
  CONSTRAINT chk_planner_candidate_provenance_index CHECK (candidate_index >= 0),
  CONSTRAINT chk_planner_candidate_provenance_hashes CHECK (
    length(btrim(requested_config_hash)) > 0 AND length(btrim(accepted_spec_hash)) > 0
  ),
  CONSTRAINT chk_planner_candidate_provenance_direction CHECK (
    metric_direction IN ('higher_is_better', 'lower_is_better')
  ),
  CONSTRAINT chk_planner_candidate_provenance_prediction_source CHECK (
    prediction_source = 'candidate.expected_metric_impact'
  ),
  CONSTRAINT chk_planner_candidate_provenance_forecast_version_units CHECK (
    score_version = 'planner_candidate_score_v1' AND forecast_units = 'fractional_score'
  ),
  CONSTRAINT chk_planner_candidate_provenance_forecast_range CHECK (
    valid_range_min < valid_range_max
    AND baseline_score BETWEEN 0 AND 1
    AND predicted_delta BETWEEN valid_range_min AND valid_range_max
    AND (
      (metric_direction = 'higher_is_better' AND predicted_delta >= 0)
      OR (metric_direction = 'lower_is_better' AND predicted_delta <= 0)
    )
  ),
  CONSTRAINT chk_planner_candidate_provenance_score CHECK (base_score BETWEEN 0 AND 1),
  CONSTRAINT chk_planner_candidate_provenance_state CHECK (
    NOT (selected AND rejected)
    AND (
      (selected AND selection_state = 'selected' AND selected_experiment_index IS NOT NULL AND selected_experiment_index >= 0)
      OR (rejected AND selection_state = 'rejected' AND selected_experiment_index IS NULL)
      OR (NOT selected AND NOT rejected AND selection_state = 'unselected' AND selected_experiment_index IS NULL)
    )
  ),
  CONSTRAINT chk_planner_candidate_provenance_outcome CHECK (length(btrim(outcome_status)) > 0),
  CONSTRAINT chk_planner_candidate_provenance_reasons CHECK (jsonb_typeof(reasons) = 'array')
);

CREATE INDEX IF NOT EXISTS idx_planner_candidate_provenance_project_variant_created
  ON planner_candidate_provenance(project_id, planner_variant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_planner_candidate_provenance_decision
  ON planner_candidate_provenance(decision_id, candidate_index);
