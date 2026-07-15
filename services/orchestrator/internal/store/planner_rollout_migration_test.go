package store

import (
	"strings"
	"testing"
)

func TestPlannerRolloutMigrationPersistsInvocationAndCandidateIdentity(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/023_planner_rollout_identity.sql")
	if err != nil {
		t.Fatal(err)
	}
	sqlText := string(sqlBytes)
	for _, required := range []string{
		"ALTER TABLE agent_invocations",
		"rollout_cohort_id text NOT NULL DEFAULT 'legacy_unknown'",
		"rollout_policy_id text NOT NULL DEFAULT 'legacy_unknown'",
		"rollout_assignment jsonb NOT NULL DEFAULT '{}'::jsonb",
		"ALTER TABLE planner_candidate_provenance",
		"chk_agent_invocations_rollout_identity",
		"chk_planner_candidate_provenance_rollout_identity",
		"idx_agent_invocations_rollout_policy_cohort_created",
		"idx_planner_candidate_provenance_rollout_policy_cohort_created",
	} {
		if !strings.Contains(sqlText, required) {
			t.Fatalf("planner rollout migration omitted %q", required)
		}
	}
}
