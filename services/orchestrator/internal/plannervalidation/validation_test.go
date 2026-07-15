package plannervalidation

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestModesUseOneStrictCheckImplementation(t *testing.T) {
	checksRun := 0
	checks := []Check{{
		Code:     "missing_evidence",
		Category: CategoryMissingEvidence,
		Stage:    "recommendation",
		Validate: func() error {
			checksRun++
			return errors.New("missing evidence")
		},
	}}

	relaxed, err := Evaluate(ModeRelaxed, checks)
	if err != nil || relaxed.Status != VerdictNotEvaluated || checksRun != 0 {
		t.Fatalf("relaxed evaluation = %#v, err=%v checks=%d", relaxed, err, checksRun)
	}
	shadow, err := Evaluate(ModeShadowStrict, checks)
	if err != nil || !shadow.WouldBlock || checksRun != 1 {
		t.Fatalf("shadow evaluation = %#v, err=%v checks=%d", shadow, err, checksRun)
	}
	strict, err := Evaluate(ModeStrict, checks)
	if err == nil || !strict.WouldBlock || checksRun != 2 {
		t.Fatalf("strict evaluation = %#v, err=%v checks=%d", strict, err, checksRun)
	}
	if shadow.Findings[0] != strict.Findings[0] {
		t.Fatalf("shadow and strict used different findings: shadow=%#v strict=%#v", shadow, strict)
	}
}

func TestModeFromEnvironmentDefaultsStrictWithObservableOneSwitchRollback(t *testing.T) {
	logDir := t.TempDir()
	t.Setenv("MODEL_EXPRESS_LOG_DIR", logDir)
	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", "")
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "")
	if resolution := ResolveModeFromEnvironment(); resolution.Mode != ModeStrict || resolution.Source != "strict_default" || resolution.Rollback {
		t.Fatalf("default resolution = %#v", resolution)
	}
	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", ModeShadowStrict)
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "false")
	if got := ModeFromEnvironment(); got != ModeShadowStrict {
		t.Fatalf("explicit mode = %q", got)
	}
	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", "invalid")
	if got := ModeFromEnvironment(); got != ModeStrict {
		t.Fatalf("invalid mode = %q", got)
	}

	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", ModeRelaxed)
	before := RelaxedRollbackDiagnosticCount()
	if got := ModeFromEnvironment(); got != ModeRelaxed {
		t.Fatalf("rollback mode = %q", got)
	}
	if got := RelaxedRollbackDiagnosticCount(); got != before+1 {
		t.Fatalf("rollback diagnostic counter = %d, want %d", got, before+1)
	}
	body, err := os.ReadFile(filepath.Join(logDir, "orchestrator.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var event map[string]any
	if err := json.Unmarshal(body, &event); err != nil {
		t.Fatal(err)
	}
	if event["event"] != RelaxedRollbackDiagnosticEvent || event["policy_version"] != RelaxedRollbackPolicyVersion || event["count"] != float64(before+1) {
		t.Fatalf("rollback diagnostic = %#v", event)
	}

	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", "")
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "false")
	if resolution := ResolveModeFromEnvironment(); resolution.Mode != ModeRelaxed || !resolution.Rollback || resolution.Source != "legacy_boolean_alias" {
		t.Fatalf("legacy rollback resolution = %#v", resolution)
	}
}

func TestStrictRolloutThresholdsAllowBoundedRetriesButRejectEscapes(t *testing.T) {
	thresholds := DefaultStrictRolloutThresholds()
	if thresholds.MaximumRetriesPerAttemptGroup != 1 || thresholds.MaximumUnsafeSchedules != 0 || thresholds.MaximumPostValidationEscapes != 0 {
		t.Fatalf("unsafe strict thresholds: %#v", thresholds)
	}
	bounded := StrictRolloutObservation{
		SampleSize: 100, EventualValidityRate: 0.99, RetryRate: 0.10, UnsupportedProposalRate: 0.02,
	}
	if assessment := AssessStrictRollout(bounded); !assessment.Passed {
		t.Fatalf("bounded first-pass failures should be allowed: %#v", assessment)
	}
	bounded.UnsafeSchedules = 1
	bounded.PostValidationEscapes = 1
	if assessment := AssessStrictRollout(bounded); assessment.Passed || len(assessment.Violations) != 2 {
		t.Fatalf("unsafe schedules and escapes must block rollout: %#v", assessment)
	}
	bounded.UnsafeSchedules = 0
	bounded.PostValidationEscapes = 0
	bounded.EventualValidityRate = math.NaN()
	if assessment := AssessStrictRollout(bounded); assessment.Passed {
		t.Fatalf("invalid strict rollout metrics must fail closed: %#v", assessment)
	}
}
