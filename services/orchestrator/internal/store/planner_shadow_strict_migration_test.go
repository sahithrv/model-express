package store

import (
	"strings"
	"testing"
)

func TestPlannerShadowStrictMigrationPersistsTypedVerdictAndOutcome(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/020_planner_shadow_strict_validation.sql")
	if err != nil {
		t.Fatal(err)
	}
	sql := string(sqlBytes)
	for _, fragment := range []string{
		"strict_validation_verdict jsonb NOT NULL DEFAULT '{}'::jsonb",
		"validation_outcome jsonb NOT NULL DEFAULT '{}'::jsonb",
		"strict_validation_verdict->>'status' = 'would_block'",
		"chk_agent_invocations_strict_verdict_object",
		"chk_agent_invocations_validation_outcome_object",
	} {
		if !strings.Contains(sql, fragment) {
			t.Errorf("migration missing %q", fragment)
		}
	}
}
