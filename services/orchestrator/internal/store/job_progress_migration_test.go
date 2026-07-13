package store

import (
	"strings"
	"testing"
)

func TestJobProgressMigrationCreatesAttemptScopedBoundedSnapshot(t *testing.T) {
	sqlBytes, err := migrationFiles.ReadFile("migrations/018_job_progress.sql")
	if err != nil {
		t.Fatal(err)
	}
	sqlText := string(sqlBytes)
	for _, required := range []string{
		"CREATE TABLE IF NOT EXISTS job_progress",
		"PRIMARY KEY (job_id, attempt)",
		"taxonomy_version integer NOT NULL",
		"heartbeat_at timestamptz NOT NULL",
		"updated_at timestamptz NOT NULL",
		"revision bigint NOT NULL",
		"jsonb_typeof(metadata) = 'object'",
		"octet_length(metadata::text) <= 4096",
		"job_progress_stage_status_consistent",
		"ON job_progress(project_id, updated_at DESC)",
	} {
		if !strings.Contains(sqlText, required) {
			t.Fatalf("job-progress migration omitted %q", required)
		}
	}
}
