package calibration

import (
	"math"
	"reflect"
	"testing"
)

func TestPlannerRolloutCohortsAreRestartStableAndNestedAcrossStages(t *testing.T) {
	policy := DefaultPlannerRolloutPolicy()
	policy.Enabled = true
	policy.State = RolloutStateActive
	policy.StagePercent = 5
	first, err := AssignPlannerRollout(policy, "project-stable")
	if err != nil {
		t.Fatal(err)
	}
	second, err := AssignPlannerRollout(policy, "project-stable")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("restart changed assignment: first=%#v second=%#v", first, second)
	}
	policy.StagePercent = 25
	expanded, err := AssignPlannerRollout(policy, "project-stable")
	if err != nil {
		t.Fatal(err)
	}
	if first.CohortID != expanded.CohortID || first.CohortBucket != expanded.CohortBucket || (first.Treatment && !expanded.Treatment) {
		t.Fatalf("guarded stages are not nested: 5%%=%#v 25%%=%#v", first, expanded)
	}
}

func TestPlannerRolloutStagesAdvanceOnlyAfterAdequateNonInferiorEvidence(t *testing.T) {
	for current, want := range map[int]int{5: 25, 25: 50, 50: 100} {
		thresholds, _ := DefaultPlannerRolloutPromotionThresholds(current)
		decision, err := EvaluatePlannerRolloutPromotion(current, PlannerRolloutPromotionObservation{
			SampleSize: thresholds.MinimumSampleSize, ObservedOutcomeSampleSize: thresholds.MinimumObservedOutcomeSamples,
			FirstPassValidityDelta: 0, EventualValidityDelta: 0, ObservedOutcomeDelta: 0,
		})
		if err != nil || decision.Action != RolloutActionAdvance || decision.NextStagePercent != want {
			t.Fatalf("stage %d decision=%#v err=%v", current, decision, err)
		}
	}
	thresholds, _ := DefaultPlannerRolloutPromotionThresholds(5)
	hold, err := EvaluatePlannerRolloutPromotion(5, PlannerRolloutPromotionObservation{SampleSize: thresholds.MinimumSampleSize - 1})
	if err != nil || hold.Action != RolloutActionHold || hold.NextStagePercent != 5 {
		t.Fatalf("undersized cohort advanced: %#v err=%v", hold, err)
	}
}

func TestPlannerRolloutPausesForRegressionAndNeverClaimsCounterfactualOutcomes(t *testing.T) {
	thresholds, _ := DefaultPlannerRolloutPromotionThresholds(25)
	observation := PlannerRolloutPromotionObservation{
		SampleSize: thresholds.MinimumSampleSize, ObservedOutcomeSampleSize: thresholds.MinimumObservedOutcomeSamples,
		FirstPassValidityDelta: 0, EventualValidityDelta: 0, ObservedOutcomeDelta: -0.02,
	}
	paused, err := EvaluatePlannerRolloutPromotion(25, observation)
	if err != nil || paused.Action != RolloutActionPause {
		t.Fatalf("inferior rollout did not pause: %#v err=%v", paused, err)
	}
	observation.ObservedOutcomeDelta = 0
	observation.UnexecutedCounterfactualClaims = 1
	manual, err := EvaluatePlannerRolloutPromotion(25, observation)
	if err != nil || manual.Action != RolloutActionManualReview {
		t.Fatalf("counterfactual outcome claim did not require review: %#v err=%v", manual, err)
	}
}

func TestPlannerRolloutPolicyIsolationPauseAndOneSwitchRollback(t *testing.T) {
	policy := DefaultPlannerRolloutPolicy()
	policy.Enabled = true
	policy.State = RolloutStateActive
	policy.Dimensions = []string{RolloutDimensionRanker, RolloutDimensionPrompt}
	policy.VariantValues[RolloutDimensionPrompt] = "compact_v1"
	if err := ValidatePlannerRolloutPolicy(policy); err == nil {
		t.Fatal("simultaneous ranker and prompt experiment was not rejected")
	}
	policy.Factorial = true
	if err := ValidatePlannerRolloutPolicy(policy); err != nil {
		t.Fatalf("explicit factorial policy was rejected: %v", err)
	}

	policy = DefaultPlannerRolloutPolicy()
	policy.Enabled = true
	policy.State = RolloutStatePaused
	paused, err := AssignPlannerRollout(policy, "project-paused")
	if err != nil || paused.Treatment || !paused.ManualReviewRequired || paused.PolicyID != policy.LastApprovedPolicyID {
		t.Fatalf("paused policy assignment=%#v err=%v", paused, err)
	}
	policy.State = RolloutStateActive
	policy.Rollback = true
	rolledBack, err := AssignPlannerRollout(policy, "project-paused")
	if err != nil || rolledBack.Treatment || rolledBack.State != RolloutStateRolledBack || rolledBack.PolicyID != policy.LastApprovedPolicyID {
		t.Fatalf("one-switch rollback assignment=%#v err=%v", rolledBack, err)
	}
}

func TestPlannerRolloutRejectsUnknownDimensionInsteadOfSilentlyIgnoringIt(t *testing.T) {
	policy := DefaultPlannerRolloutPolicy()
	policy.Dimensions = []string{"ranker", "surprise_dimension"}
	policy.VariantValues["surprise_dimension"] = "candidate"
	if err := ValidatePlannerRolloutPolicy(policy); err == nil {
		t.Fatal("unsupported rollout dimension was silently ignored")
	}
}

func TestPlannerRolloutRejectsUnknownRankerVariantInsteadOfMislabelingPolicy(t *testing.T) {
	policy := DefaultPlannerRolloutPolicy()
	policy.VariantValues[RolloutDimensionRanker] = "candidate_ranker_unknown"
	if err := ValidatePlannerRolloutPolicy(policy); err == nil {
		t.Fatal("unsupported ranker variant could be persisted as a treatment policy")
	}
}

func TestPlannerRolloutPromotionRejectsInvalidMetricsFailClosed(t *testing.T) {
	if _, err := EvaluatePlannerRolloutPromotion(5, PlannerRolloutPromotionObservation{SampleSize: 50, ObservedOutcomeSampleSize: 50, EventualValidityDelta: math.NaN()}); err == nil {
		t.Fatal("NaN promotion evidence could advance a rollout")
	}
}
