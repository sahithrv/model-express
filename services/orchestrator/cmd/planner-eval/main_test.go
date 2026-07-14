package main

import "testing"

func TestLiveEvalRequiresExplicitEnvironmentOptIn(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_PLANNER_EVAL_LIVE", "")
	if liveEvalEnabled() {
		t.Fatal("live evaluation must be disabled by default")
	}
	t.Setenv("MODEL_EXPRESS_PLANNER_EVAL_LIVE", "true")
	if !liveEvalEnabled() {
		t.Fatal("explicit live evaluation opt-in was not recognized")
	}
}
