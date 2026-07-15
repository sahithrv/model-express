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

func TestDefaultFixtureScopeKeepsLiveCallsBounded(t *testing.T) {
	offline, err := loadFixtures("", false)
	if err != nil {
		t.Fatal(err)
	}
	live, err := loadFixtures("", true)
	if err != nil {
		t.Fatal(err)
	}
	if len(offline) != 15 {
		t.Fatalf("offline corpus has %d scenarios, want 15", len(offline))
	}
	if len(live) != 4 {
		t.Fatalf("live default has %d scenarios, want bounded starter set of 4", len(live))
	}
}
