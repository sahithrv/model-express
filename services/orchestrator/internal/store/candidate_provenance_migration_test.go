package store

import (
	"strings"
	"testing"
)

func TestCandidateProvenanceMigrationDefinesImmutableDecisionTimeLineage(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/021_candidate_provenance.sql")
	if err != nil {
		t.Fatal(err)
	}
	sql := string(sqlBytes)
	for _, fragment := range []string{
		"CREATE TABLE IF NOT EXISTS planner_candidate_provenance",
		"invocation_id text NOT NULL REFERENCES agent_invocations(id)",
		"decision_id text NOT NULL REFERENCES agent_decisions(id)",
		"planner_variant_id text NOT NULL",
		"requested_config_hash text NOT NULL",
		"accepted_spec_hash text NOT NULL",
		"prediction_source = 'candidate.expected_metric_impact'",
		"score_version = 'planner_candidate_score_v1' AND forecast_units = 'fractional_score'",
		"selected_experiment_index integer",
		"outcome_status text NOT NULL DEFAULT 'unknown'",
		"followup_plan_id text",
		"realized_effective_hash text",
		"UNIQUE (decision_id, candidate_index)",
		"idx_planner_candidate_provenance_project_variant_created",
	} {
		if !strings.Contains(sql, fragment) {
			t.Errorf("candidate provenance migration missing %q", fragment)
		}
	}
}
