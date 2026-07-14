package plannervalidation

import (
	"errors"
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

func TestModeFromEnvironmentUsesModeWithLegacyBooleanFallback(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", "")
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "true")
	if got := ModeFromEnvironment(); got != ModeStrict {
		t.Fatalf("legacy true mode = %q", got)
	}
	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", ModeShadowStrict)
	t.Setenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", "false")
	if got := ModeFromEnvironment(); got != ModeShadowStrict {
		t.Fatalf("explicit mode = %q", got)
	}
	t.Setenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE", "invalid")
	if got := ModeFromEnvironment(); got != ModeRelaxed {
		t.Fatalf("invalid mode = %q", got)
	}
}
