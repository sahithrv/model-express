package store

import (
	"strings"
	"testing"
)

func TestExperimentPolicyMigrationCreatesImmutableScopedAuditSchema(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/024_experiment_policies.sql")
	if err != nil {
		t.Fatal(err)
	}
	sqlText := string(sqlBytes)
	for _, required := range []string{
		"CREATE TABLE IF NOT EXISTS accounts",
		"'account_local_default'",
		"ALTER TABLE projects",
		"account_id text REFERENCES accounts(id)",
		"CREATE TABLE IF NOT EXISTS compatibility_profiles",
		"UNIQUE(profile_key, semantic_version)",
		"CREATE TABLE IF NOT EXISTS experiment_policy_versions",
		"revision bigint NOT NULL UNIQUE",
		"CREATE TABLE IF NOT EXISTS experiment_policy_bindings",
		"num_nonnulls(account_id, project_id, dataset_id, experiment_job_id) = 1",
		"uq_experiment_policy_binding_account_active",
		"uq_experiment_policy_binding_project_active",
		"uq_experiment_policy_binding_dataset_active",
		"uq_experiment_policy_binding_run_active",
		"CREATE TABLE IF NOT EXISTS experiment_policy_evaluations",
		"effective_snapshot jsonb NOT NULL",
		"effective_policy_hash text NOT NULL",
		"requested_capability_uses jsonb",
		"effective_capability_uses jsonb",
		"reason_codes jsonb",
		"findings jsonb",
		"proposal_policy_evaluation_id",
		"schedule_policy_evaluation_id",
		"dispatch_policy_evaluation_id",
		"export_policy_evaluation_id",
		"reject_immutable_experiment_policy_mutation",
		"restrict_experiment_policy_binding_mutation",
	} {
		if !strings.Contains(sqlText, required) {
			t.Fatalf("experiment-policy migration omitted %q", required)
		}
	}
	for _, partialUnique := range []string{
		"WHERE active AND scope = 'account'",
		"WHERE active AND scope = 'project'",
		"WHERE active AND scope = 'dataset'",
		"WHERE active AND scope = 'run'",
	} {
		if !strings.Contains(sqlText, partialUnique) {
			t.Fatalf("experiment-policy migration omitted partial uniqueness %q", partialUnique)
		}
	}
}
