package store

import (
	"strings"
	"testing"
)

func TestArtifactPlanAndRoastyMigrations(t *testing.T) {
	artifactBytes, err := migrationFiles.ReadFile("migrations/025_artifact_plans_and_worker_capabilities.sql")
	if err != nil {
		t.Fatal(err)
	}
	artifactSQL := string(artifactBytes)
	for _, required := range []string{
		"policy_capability_versions", "artifact_capability_versions", "artifact_plan jsonb",
		"artifact_plan_hash", "worker_policy_capability_version", "worker_artifact_capability_version",
	} {
		if !strings.Contains(artifactSQL, required) {
			t.Fatalf("artifact-plan migration omitted %q", required)
		}
	}
	roastyBytes, err := migrationFiles.ReadFile("migrations/026_roasty_v1.sql")
	if err != nil {
		t.Fatal(err)
	}
	roastySQL := string(roastyBytes)
	for _, required := range []string{"roasty_v1", "1.0.0", "\"fp32\"", "\"onnx\"", "\"onnxruntime\"", "ON CONFLICT (profile_key, semantic_version) DO NOTHING"} {
		if !strings.Contains(roastySQL, required) {
			t.Fatalf("roasty migration omitted %q", required)
		}
	}
}
