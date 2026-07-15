package store

import (
	"strings"
	"testing"
)

func TestPlannerRuntimeIdentityMigrationAddsQueryableVersionedAuditFields(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/019_planner_runtime_identity.sql")
	if err != nil {
		t.Fatal(err)
	}
	sqlText := string(sqlBytes)
	for _, required := range []string{
		"planner_variant_id text NOT NULL DEFAULT 'legacy_unknown'",
		"planner_variant jsonb NOT NULL DEFAULT '{}'::jsonb",
		"validation_mode text NOT NULL DEFAULT ''",
		"attempt_group_id text NOT NULL DEFAULT ''",
		"attempt_index integer NOT NULL DEFAULT -1",
		"retry_reason text NOT NULL DEFAULT ''",
		"wall_latency_ms double precision NOT NULL DEFAULT 0",
		"provider_usage jsonb NOT NULL DEFAULT '{}'::jsonb",
		"derived_cost jsonb NOT NULL DEFAULT '{}'::jsonb",
		"SET planner_variant_id = 'legacy_unknown'",
		"ON agent_invocations(project_id, planner_variant_id, created_at DESC)",
		"ON agent_invocations(attempt_group_id, attempt_index)",
		"DO $$",
		"FROM pg_constraint",
		"conrelid = 'agent_invocations'::regclass",
		"derived_cost ?& ARRAY[",
		"OR COALESCE((",
		"derived_cost->'pricing_version'",
		"derived_cost->'total_cost_usd'",
		"chk_agent_invocations_versioned_cost",
	} {
		if !strings.Contains(sqlText, required) {
			t.Fatalf("planner runtime identity migration omitted %q", required)
		}
	}
}
