package evals

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
)

func TestStarterRubricScenariosPass(t *testing.T) {
	fixtures, err := LoadStarterPlannerRubricFixtures()
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := EvaluatePlannerRubricFixtures(fixtures)
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Summary.ScenarioCount != 4 || artifact.Summary.PassedCount != 4 {
		t.Fatalf("starter rubric summary = %#v", artifact.Summary)
	}
	for _, scenario := range artifact.Scenarios {
		if !scenario.Score.BackendSchedulable || !scenario.Score.Passed {
			t.Fatalf("starter scenario %q failed: %#v", scenario.FixtureName, scenario.Score)
		}
	}
}

func TestRubricFieldsHavePassingAndFailingMutations(t *testing.T) {
	classification := loadClassificationFixture(t)
	classificationRaw, _ := ReplayPlannerResponse(classification)
	classificationInput := ExperimentPlannerInputFromReplayFixture(classification)
	classificationRubric := PlannerRubricForFixture(classification)
	base := ScorePlannerRubric(classificationInput, classificationRaw, classificationRubric)
	if !base.Passed {
		t.Fatalf("classification base fixture must pass: %#v", base)
	}

	decisionMutation := decodeRecommendation(t, classificationRaw)
	decisionMutation.DecisionType = decisions.TypeWait
	decision := ScorePlannerRubric(classificationInput, encodeRecommendation(t, decisionMutation), classificationRubric)
	assertPassingAndFailingCheck(t, "decision", base.Decision, decision.Decision)

	mechanismMutation := decodeRecommendation(t, classificationRaw)
	for index := range mechanismMutation.CandidateHypotheses {
		mechanismMutation.CandidateHypotheses[index].Mechanism = "architecture_challenge"
		mechanismMutation.CandidateHypotheses[index].ExperimentConfig.Mechanism = "architecture_challenge"
	}
	for index := range mechanismMutation.ProposalMechanisms {
		mechanismMutation.ProposalMechanisms[index].Mechanism = "architecture_challenge"
	}
	mechanisms := ScorePlannerRubric(classificationInput, encodeRecommendation(t, mechanismMutation), classificationRubric)
	assertPassingAndFailingCheck(t, "mechanisms", base.Mechanisms, mechanisms.Mechanisms)

	evidenceMutation := decodeRecommendation(t, classificationRaw)
	evidenceMutation.EvidenceUsed = nil
	for index := range evidenceMutation.CandidateHypotheses {
		evidenceMutation.CandidateHypotheses[index].EvidenceUsed = nil
	}
	for index := range evidenceMutation.ProposalMechanisms {
		evidenceMutation.ProposalMechanisms[index].EvidenceUsed = nil
	}
	evidence := ScorePlannerRubric(classificationInput, encodeRecommendation(t, evidenceMutation), classificationRubric)
	assertPassingAndFailingCheck(t, "evidence", base.Evidence, evidence.Evidence)

	taskMutationInput := classificationInput
	taskMutationInput.DatasetInsights.TaskType = "object_detection"
	task := ScorePlannerRubric(taskMutationInput, classificationRaw, classificationRubric)
	assertPassingAndFailingCheck(t, "task compatibility", base.TaskCompatibility, task.TaskCompatibility)

	stopFixture := starterFixtureNamed(t, "select_champion_stop")
	stopRaw, _ := ReplayPlannerResponse(stopFixture)
	stopInput := ExperimentPlannerInputFromReplayFixture(stopFixture)
	stopRubric := PlannerRubricForFixture(stopFixture)
	stopBase := ScorePlannerRubric(stopInput, stopRaw, stopRubric)
	stopMutation := decodeRecommendation(t, stopRaw)
	stopMutation.StopReason = ""
	stop := ScorePlannerRubric(stopInput, encodeRecommendation(t, stopMutation), stopRubric)
	assertPassingAndFailingCheck(t, "stop behavior", stopBase.StopBehavior, stop.StopBehavior)

	safetyMutation := decodeRecommendation(t, classificationRaw)
	safetyMutation.Confidence = 2
	safety := ScorePlannerRubric(classificationInput, encodeRecommendation(t, safetyMutation), classificationRubric)
	assertPassingAndFailingCheck(t, "safety", base.Safety, safety.Safety)
}

func TestRubricAggregateIsIndependentOfFixtureOrder(t *testing.T) {
	fixtures, err := LoadStarterPlannerRubricFixtures()
	if err != nil {
		t.Fatal(err)
	}
	forward, err := EvaluatePlannerRubricFixtures(fixtures)
	if err != nil {
		t.Fatal(err)
	}
	reversed := append([]PlannerReplayFixture(nil), fixtures...)
	slices.Reverse(reversed)
	backward, err := EvaluatePlannerRubricFixtures(reversed)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(forward, backward) {
		t.Fatalf("fixture order changed aggregate:\nforward=%#v\nbackward=%#v", forward, backward)
	}
}

func TestFullCorpusCoversPR4ScenariosDecisionsAndMutations(t *testing.T) {
	fixtures, err := LoadPlannerRubricFixtures()
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := EvaluatePlannerRubricFixtures(fixtures)
	if err != nil {
		t.Fatal(err)
	}
	if artifact.Summary.ScenarioCount != 15 || artifact.Summary.PassedCount != 15 {
		t.Fatalf("full corpus summary = %#v", artifact.Summary)
	}
	if artifact.Summary.MutationCount != 15 || artifact.Summary.DetectedMutationCount != 15 {
		t.Fatalf("targeted mutations were not all detected: %#v", artifact.Summary)
	}
	coverage := map[string]bool{}
	decisionsSeen := map[string]bool{}
	for _, scenario := range artifact.Scenarios {
		if strings.TrimSpace(scenario.Description) == "" {
			t.Fatalf("scenario %q has no review explanation", scenario.FixtureName)
		}
		for _, tag := range scenario.Coverage {
			coverage[tag] = true
		}
		decisionsSeen[scenario.Score.DecisionType] = true
		for _, mutation := range scenario.Mutations {
			if !mutation.Detected {
				t.Fatalf("scenario %q mutation %q passed unexpectedly: %#v", scenario.FixtureName, mutation.Name, mutation)
			}
		}
	}
	for _, tag := range []string{
		"imbalance", "minority_failure", "overfit", "underfit", "plateau",
		"repeated_architecture_shopping", "latency", "cost", "duplicates", "no_ops",
		"label_audit", "wait", "memory_success", "memory_failure", "memory_rejection",
		"classification", "detection", "task_sensitive", "accepted_spec_identity",
	} {
		if !coverage[tag] {
			t.Errorf("full corpus is missing coverage tag %q", tag)
		}
	}
	for _, decisionType := range []string{
		decisions.TypeAddExperiments,
		decisions.TypeSelectChampion,
		decisions.TypeStopProject,
		decisions.TypeWait,
	} {
		if !decisionsSeen[decisionType] {
			t.Errorf("full corpus is missing decision type %q", decisionType)
		}
	}
}

func TestCheckedBaselineMatchesAndExplainsPerScenarioChanges(t *testing.T) {
	fixtures, err := LoadPlannerRubricFixtures()
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := EvaluatePlannerRubricFixtures(fixtures)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := LoadCheckedPlannerRubricBaseline()
	if err != nil {
		t.Fatal(err)
	}
	comparison := ComparePlannerRubricBaseline(baseline, artifact)
	if !comparison.Passed || len(comparison.Changes) != 0 {
		t.Fatalf("checked baseline mismatch: %#v", comparison)
	}

	changed := artifact
	changed.Scenarios = append([]PlannerRubricScenarioResult(nil), artifact.Scenarios...)
	changed.Scenarios[0].Score.Passed = false
	changed.Scenarios[0].Score.SafetyPassed = false
	comparison = ComparePlannerRubricBaseline(baseline, changed)
	if comparison.Passed || comparison.CriticalRegressions == 0 || len(comparison.Changes) == 0 {
		t.Fatalf("critical scenario regression was not rejected: %#v", comparison)
	}
	for _, change := range comparison.Changes {
		if change.FixtureName == changed.Scenarios[0].FixtureName && strings.Contains(change.Explanation, changed.Scenarios[0].Description) {
			return
		}
	}
	t.Fatalf("baseline changes did not include the per-scenario explanation: %#v", comparison.Changes)
}

func TestBaselineTolerancesSeparateCriticalQualityAndEfficiency(t *testing.T) {
	fixtures, err := LoadPlannerRubricFixtures()
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := EvaluatePlannerRubricFixtures(fixtures)
	if err != nil {
		t.Fatal(err)
	}
	baseline := BuildPlannerRubricBaseline(artifact)
	changed := artifact
	changed.Scenarios = append([]PlannerRubricScenarioResult(nil), artifact.Scenarios...)
	changed.Scenarios[0].ResponseBytes += baseline.Tolerances.Efficiency.MaxResponseByteIncreasePerScenario
	comparison := ComparePlannerRubricBaseline(baseline, changed)
	if !comparison.Passed || comparison.CriticalRegressions != 0 || comparison.QualityRegressions != 0 || len(comparison.Changes) != 1 || comparison.Changes[0].Dimension != "efficiency" {
		t.Fatalf("efficiency tolerance leaked into critical/quality gates: %#v", comparison)
	}
}

func TestBestVariantUsesRubricBeforeEfficiencyAndIgnoresRankerScore(t *testing.T) {
	valid := PlannerReplayVariantResult{
		Variant:               PlannerReplayVariantCurrentV1,
		PromptBytes:           10_000,
		CandidateRankingScore: 0,
		Rubric: PlannerRubricScore{
			QualityTier:       3,
			SafetyPassed:      true,
			CorrectnessChecks: 5,
		},
	}
	invalidConcise := PlannerReplayVariantResult{
		Variant:               PlannerReplayVariantCompactStaticPrompt,
		PromptBytes:           1,
		CandidateRankingScore: 999,
		Rubric: PlannerRubricScore{
			QualityTier:       1,
			SafetyPassed:      false,
			CorrectnessChecks: 0,
		},
	}
	if got := replayBestPlannerVariant([]PlannerReplayVariantResult{invalidConcise, valid}); got != valid.Variant {
		t.Fatalf("invalid concise/ranker-high output won quality selection: %q", got)
	}
}

func assertPassingAndFailingCheck(t *testing.T, name string, passing PlannerRubricCheck, failing PlannerRubricCheck) {
	t.Helper()
	if !passing.Passed {
		t.Fatalf("%s passing fixture failed: %#v", name, passing)
	}
	if failing.Passed || len(failing.Reasons) == 0 {
		t.Fatalf("%s failing mutation passed: %#v", name, failing)
	}
}

func decodeRecommendation(t *testing.T, raw []byte) agents.ExperimentPlanningRecommendation {
	t.Helper()
	var recommendation agents.ExperimentPlanningRecommendation
	if err := json.Unmarshal(raw, &recommendation); err != nil {
		t.Fatal(err)
	}
	return recommendation
}

func encodeRecommendation(t *testing.T, recommendation agents.ExperimentPlanningRecommendation) []byte {
	t.Helper()
	raw, err := json.Marshal(recommendation)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func starterFixtureNamed(t *testing.T, name string) PlannerReplayFixture {
	t.Helper()
	fixtures, err := LoadStarterPlannerRubricFixtures()
	if err != nil {
		t.Fatal(err)
	}
	for _, fixture := range fixtures {
		if fixture.Name == name {
			return fixture
		}
	}
	t.Fatalf("starter fixture %q not found", name)
	return PlannerReplayFixture{}
}
