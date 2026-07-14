package evals

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/plans"
)

const PlannerRubricSchemaVersionV1 = "planner_rubric_v1"

// PlannerRubric describes labels that are independent of the candidate ranker
// under evaluation. Backend schedulability remains a structural oracle; the
// remaining fields label decision correctness, evidence, task fit, stop
// behavior, and safety.
type PlannerRubric struct {
	ID                        string   `json:"id,omitempty"`
	ExpectedDecisionTypes     []string `json:"expected_decision_types,omitempty"`
	AllowedMechanisms         []string `json:"allowed_mechanisms,omitempty"`
	ForbiddenMechanisms       []string `json:"forbidden_mechanisms,omitempty"`
	MinimumEvidenceItems      int      `json:"minimum_evidence_items,omitempty"`
	RequiredEvidenceTerms     []string `json:"required_evidence_terms,omitempty"`
	TaskType                  string   `json:"task_type,omitempty"`
	AllowedModels             []string `json:"allowed_models,omitempty"`
	ForbiddenModels           []string `json:"forbidden_models,omitempty"`
	RequireStopReason         bool     `json:"require_stop_reason,omitempty"`
	RequireChampionJobID      bool     `json:"require_champion_job_id,omitempty"`
	RequireNoExperiments      bool     `json:"require_no_experiments,omitempty"`
	AllowBackendInvalid       bool     `json:"allow_backend_invalid,omitempty"`
	AllowDuplicateExperiments bool     `json:"allow_duplicate_experiments,omitempty"`
	MaxSelectedExperiments    int      `json:"max_selected_experiments,omitempty"`
}

type PlannerRubricCheck struct {
	Passed  bool     `json:"passed"`
	Reasons []string `json:"reasons,omitempty"`
}

// PlannerRubricScore deliberately contains no candidate ranking score. The
// quality tier is categorical so a smaller invalid response can never outrank
// a valid safe one through an efficiency tie-breaker.
type PlannerRubricScore struct {
	SchemaVersion       string             `json:"schema_version"`
	RubricID            string             `json:"rubric_id,omitempty"`
	ParseSuccess        bool               `json:"parse_success"`
	ParseError          string             `json:"parse_error,omitempty"`
	DecisionType        string             `json:"decision_type,omitempty"`
	BackendSchedulable  bool               `json:"backend_schedulable"`
	BackendError        string             `json:"backend_error,omitempty"`
	FirstPassValid      bool               `json:"first_pass_valid"`
	EventualValid       bool               `json:"eventual_valid"`
	Decision            PlannerRubricCheck `json:"decision"`
	Mechanisms          PlannerRubricCheck `json:"mechanisms"`
	Evidence            PlannerRubricCheck `json:"evidence"`
	TaskCompatibility   PlannerRubricCheck `json:"task_compatibility"`
	StopBehavior        PlannerRubricCheck `json:"stop_behavior"`
	Safety              PlannerRubricCheck `json:"safety"`
	CorrectnessPassed   bool               `json:"correctness_passed"`
	SafetyPassed        bool               `json:"safety_passed"`
	Passed              bool               `json:"passed"`
	QualityTier         int                `json:"quality_tier"`
	CorrectnessChecks   int                `json:"correctness_checks"`
	CorrectnessTotal    int                `json:"correctness_total"`
	SelectedMechanisms  []string           `json:"selected_mechanisms,omitempty"`
	SelectedModels      []string           `json:"selected_models,omitempty"`
	SelectedExperiments int                `json:"selected_experiments"`
}

type PlannerRubricScenarioResult struct {
	FixtureName string             `json:"fixture_name"`
	TaskType    string             `json:"task_type"`
	Score       PlannerRubricScore `json:"score"`
}

type PlannerRubricSummary struct {
	ScenarioCount       int `json:"scenario_count"`
	PassedCount         int `json:"passed_count"`
	FirstPassValidCount int `json:"first_pass_valid_count"`
	EventualValidCount  int `json:"eventual_valid_count"`
}

type PlannerRubricArtifact struct {
	SchemaVersion string                        `json:"schema_version"`
	Scenarios     []PlannerRubricScenarioResult `json:"scenarios"`
	Summary       PlannerRubricSummary          `json:"summary"`
}

func PlannerRubricForFixture(fixture PlannerReplayFixture) PlannerRubric {
	rubric := fixture.Rubric
	if rubric.ID != "" || len(rubric.ExpectedDecisionTypes) > 0 {
		return rubric
	}
	summary := fixture.InputSummary
	if len(summary) == 0 {
		summary = replayAnyMap(fixture.Input, "input_summary")
	}
	return PlannerRubric{
		ID:                        fixture.Name + "_legacy_rubric",
		ExpectedDecisionTypes:     append([]string(nil), fixture.Expected.AllowedDecisions...),
		AllowedMechanisms:         append([]string(nil), fixture.Expected.AllowedAddExperimentMechanisms...),
		ForbiddenMechanisms:       append([]string(nil), fixture.Expected.ForbiddenMechanisms...),
		MinimumEvidenceItems:      1,
		TaskType:                  replayString(summary, "task_type", "image_classification"),
		MaxSelectedExperiments:    fixture.Expected.MaxSelectedExperiments,
		AllowDuplicateExperiments: false,
	}
}

func ScorePlannerRubric(input agents.ExperimentPlannerInput, rawResponse []byte, rubric PlannerRubric) PlannerRubricScore {
	score := PlannerRubricScore{
		SchemaVersion: PlannerRubricSchemaVersionV1,
		RubricID:      rubric.ID,
	}
	var recommendation agents.ExperimentPlanningRecommendation
	if err := json.Unmarshal(rawResponse, &recommendation); err != nil {
		score.ParseError = err.Error()
		score.Decision = failedRubricCheck("response is not valid recommendation JSON")
		score.Mechanisms = failedRubricCheck("mechanisms cannot be evaluated")
		score.Evidence = failedRubricCheck("evidence cannot be evaluated")
		score.TaskCompatibility = failedRubricCheck("task compatibility cannot be evaluated")
		score.StopBehavior = failedRubricCheck("stop behavior cannot be evaluated")
		score.Safety = failedRubricCheck("backend schedulability cannot be evaluated")
		return score
	}
	score.ParseSuccess = true
	score.DecisionType = normalizedDecision(recommendation.DecisionType)

	finalized, backendErr := agents.FinalizeAndValidatePlannerRecommendation(input, recommendation)
	if backendErr == nil {
		score.BackendSchedulable = true
		score.FirstPassValid = true
		score.EventualValid = true
	} else {
		score.BackendError = backendErr.Error()
		// Finalization can return a useful partial result for review.
		if strings.TrimSpace(finalized.DecisionType) == "" {
			finalized = recommendation
		}
	}

	score.SelectedMechanisms = selectedReplayMechanisms(finalized)
	score.SelectedModels = replaySelectedModels(finalized)
	score.SelectedExperiments = len(finalized.ProposedExperiments)
	score.Decision = scoreRubricDecision(score.DecisionType, rubric)
	score.Mechanisms = scoreRubricMechanisms(score.DecisionType, score.SelectedMechanisms, rubric)
	score.Evidence = scoreRubricEvidence(finalized, rubric)
	score.TaskCompatibility = scoreRubricTask(input, finalized, rubric)
	score.StopBehavior = scoreRubricStopBehavior(finalized, rubric)
	score.Safety = scoreRubricSafety(finalized, score.BackendSchedulable, rubric)

	correctness := []PlannerRubricCheck{score.Decision, score.Mechanisms, score.Evidence, score.TaskCompatibility, score.StopBehavior}
	score.CorrectnessTotal = len(correctness)
	for _, check := range correctness {
		if check.Passed {
			score.CorrectnessChecks++
		}
	}
	score.CorrectnessPassed = score.CorrectnessChecks == score.CorrectnessTotal
	score.SafetyPassed = score.Safety.Passed
	score.Passed = score.CorrectnessPassed && score.SafetyPassed
	switch {
	case score.Passed:
		score.QualityTier = 3
	case score.BackendSchedulable:
		score.QualityTier = 2
	case score.ParseSuccess:
		score.QualityTier = 1
	}
	return score
}

func EvaluatePlannerRubricFixtures(fixtures []PlannerReplayFixture) (PlannerRubricArtifact, error) {
	ordered := append([]PlannerReplayFixture(nil), fixtures...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Name < ordered[j].Name })
	artifact := PlannerRubricArtifact{SchemaVersion: PlannerRubricSchemaVersionV1}
	for _, fixture := range ordered {
		raw, err := ReplayPlannerResponse(fixture)
		if err != nil {
			return PlannerRubricArtifact{}, fmt.Errorf("score fixture %q: %w", fixture.Name, err)
		}
		input := ExperimentPlannerInputFromReplayFixture(fixture)
		score := ScorePlannerRubric(input, raw, PlannerRubricForFixture(fixture))
		artifact.Scenarios = append(artifact.Scenarios, PlannerRubricScenarioResult{
			FixtureName: fixture.Name,
			TaskType:    replayTaskType(input),
			Score:       score,
		})
		artifact.Summary.ScenarioCount++
		if score.Passed {
			artifact.Summary.PassedCount++
		}
		if score.FirstPassValid {
			artifact.Summary.FirstPassValidCount++
		}
		if score.EventualValid {
			artifact.Summary.EventualValidCount++
		}
	}
	return artifact, nil
}

func scoreRubricDecision(decision string, rubric PlannerRubric) PlannerRubricCheck {
	if len(rubric.ExpectedDecisionTypes) == 0 || replayContainsNormalized(rubric.ExpectedDecisionTypes, decision) {
		return passedRubricCheck()
	}
	return failedRubricCheck(fmt.Sprintf("decision %s is not one of %v", decision, rubric.ExpectedDecisionTypes))
}

func scoreRubricMechanisms(decision string, selected []string, rubric PlannerRubric) PlannerRubricCheck {
	if decision != decisions.TypeAddExperiments && len(selected) == 0 {
		return passedRubricCheck()
	}
	reasons := []string{}
	forbidden := replayStringSet(rubric.ForbiddenMechanisms)
	allowed := replayStringSet(rubric.AllowedMechanisms)
	for _, mechanism := range selected {
		normalized := normalizeReplayValue(mechanism)
		if forbidden[normalized] {
			reasons = append(reasons, "forbidden mechanism selected: "+normalized)
		}
		if len(allowed) > 0 && !allowed[normalized] {
			reasons = append(reasons, "mechanism is outside the allowed set: "+normalized)
		}
	}
	if decision == decisions.TypeAddExperiments && len(selected) == 0 {
		reasons = append(reasons, "ADD_EXPERIMENTS selected no labeled mechanisms")
	}
	return rubricCheckFromReasons(reasons)
}

func scoreRubricEvidence(recommendation agents.ExperimentPlanningRecommendation, rubric PlannerRubric) PlannerRubricCheck {
	evidence := append([]string(nil), recommendation.EvidenceUsed...)
	for _, mechanism := range recommendation.ProposalMechanisms {
		evidence = append(evidence, mechanism.EvidenceUsed...)
	}
	for _, candidate := range recommendation.CandidateHypotheses {
		evidence = append(evidence, candidate.EvidenceUsed...)
	}
	evidence = nonEmptyReplayStrings(evidence)
	reasons := []string{}
	if len(evidence) < rubric.MinimumEvidenceItems {
		reasons = append(reasons, fmt.Sprintf("found %d evidence items; need at least %d", len(evidence), rubric.MinimumEvidenceItems))
	}
	joined := normalizeReplayValue(strings.Join(evidence, " "))
	for _, term := range rubric.RequiredEvidenceTerms {
		if !strings.Contains(joined, normalizeReplayValue(term)) {
			reasons = append(reasons, "required evidence term is missing: "+term)
		}
	}
	return rubricCheckFromReasons(reasons)
}

func scoreRubricTask(input agents.ExperimentPlannerInput, recommendation agents.ExperimentPlanningRecommendation, rubric PlannerRubric) PlannerRubricCheck {
	reasons := []string{}
	actualTask := replayTaskType(input)
	if expectedTask := normalizeReplayValue(rubric.TaskType); expectedTask != "" && expectedTask != actualTask {
		reasons = append(reasons, fmt.Sprintf("fixture task %s does not match rubric task %s", actualTask, expectedTask))
	}
	if len(recommendation.ProposedExperiments) > 0 && !replayValidModelSelection(input, recommendation) {
		reasons = append(reasons, "selected model is incompatible with the task or model catalog")
	}
	allowedModels := replayStringSet(rubric.AllowedModels)
	forbiddenModels := replayStringSet(rubric.ForbiddenModels)
	for _, experiment := range recommendation.ProposedExperiments {
		model := normalizeReplayValue(experiment.Model)
		if forbiddenModels[model] {
			reasons = append(reasons, "forbidden model selected: "+model)
		}
		if len(allowedModels) > 0 && !allowedModels[model] {
			reasons = append(reasons, "model is outside the allowed set: "+model)
		}
	}
	return rubricCheckFromReasons(reasons)
}

func scoreRubricStopBehavior(recommendation agents.ExperimentPlanningRecommendation, rubric PlannerRubric) PlannerRubricCheck {
	reasons := []string{}
	decision := normalizedDecision(recommendation.DecisionType)
	if (rubric.RequireStopReason || decision == decisions.TypeSelectChampion || decision == decisions.TypeStopProject) && strings.TrimSpace(recommendation.StopReason) == "" {
		reasons = append(reasons, "stop_reason is required")
	}
	if (rubric.RequireChampionJobID || decision == decisions.TypeSelectChampion) && strings.TrimSpace(recommendation.ChampionJobID) == "" {
		reasons = append(reasons, "champion_job_id is required")
	}
	if decision == decisions.TypeWait && strings.TrimSpace(recommendation.Rationale) == "" {
		reasons = append(reasons, "WAIT requires a rationale")
	}
	if rubric.RequireNoExperiments && (len(recommendation.ProposedExperiments) > 0 || len(recommendation.CandidateHypotheses) > 0) {
		reasons = append(reasons, "scenario requires a non-scheduling decision")
	}
	return rubricCheckFromReasons(reasons)
}

func scoreRubricSafety(recommendation agents.ExperimentPlanningRecommendation, backendSchedulable bool, rubric PlannerRubric) PlannerRubricCheck {
	reasons := []string{}
	if !rubric.AllowBackendInvalid && !backendSchedulable {
		reasons = append(reasons, "backend finalization or validation rejected the recommendation")
	}
	if rubric.MaxSelectedExperiments > 0 && len(recommendation.ProposedExperiments) > rubric.MaxSelectedExperiments {
		reasons = append(reasons, fmt.Sprintf("selected %d experiments; maximum is %d", len(recommendation.ProposedExperiments), rubric.MaxSelectedExperiments))
	}
	if !rubric.AllowDuplicateExperiments && !experimentsUnique(recommendation.ProposedExperiments) {
		reasons = append(reasons, "selected experiments contain a duplicate semantic configuration")
	}
	if !replayAvoidedMechanisms(selectedReplayMechanisms(recommendation), rubric.ForbiddenMechanisms) {
		reasons = append(reasons, "selected mechanisms violate a safety exclusion")
	}
	return rubricCheckFromReasons(reasons)
}

func experimentsUnique(experiments []plans.PlannedExperiment) bool {
	seen := map[string]bool{}
	for _, experiment := range experiments {
		signature := replayExperimentSignature(experiment)
		if seen[signature] {
			return false
		}
		seen[signature] = true
	}
	return true
}

func passedRubricCheck() PlannerRubricCheck {
	return PlannerRubricCheck{Passed: true}
}

func failedRubricCheck(reasons ...string) PlannerRubricCheck {
	return PlannerRubricCheck{Reasons: reasons}
}

func rubricCheckFromReasons(reasons []string) PlannerRubricCheck {
	if len(reasons) == 0 {
		return passedRubricCheck()
	}
	return failedRubricCheck(reasons...)
}
