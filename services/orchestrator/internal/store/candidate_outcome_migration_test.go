package store

import (
	"strings"
	"testing"
)

func TestCandidateOutcomeMigrationAddsNullableIdempotentFinalizationFields(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/022_candidate_outcome_finalization.sql")
	if err != nil {
		t.Fatal(err)
	}
	sql := string(sqlBytes)
	for _, fragment := range []string{
		"ADD COLUMN IF NOT EXISTS attempt_id text",
		"ADD COLUMN IF NOT EXISTS actual_score double precision",
		"ADD COLUMN IF NOT EXISTS actual_delta double precision",
		"ADD COLUMN IF NOT EXISTS terminal_state text",
		"ADD COLUMN IF NOT EXISTS cost_usd double precision",
		"ADD COLUMN IF NOT EXISTS runtime_seconds double precision",
		"ADD COLUMN IF NOT EXISTS calibration_eligible boolean",
		"ADD COLUMN IF NOT EXISTS eligibility_reason text",
		"ADD COLUMN IF NOT EXISTS finalized_at timestamptz",
		"outcome_status IN ('unknown', 'observed', 'unobserved')",
		"terminal_state IN ('SUCCEEDED', 'FAILED', 'CANCELLED', 'SKIPPED')",
		"chk_planner_candidate_provenance_unselected_outcome",
		"idx_planner_candidate_provenance_followup_plan",
		"Historical rows are intentionally not inferred",
	} {
		if !strings.Contains(sql, fragment) {
			t.Errorf("candidate outcome migration missing %q", fragment)
		}
	}
}
