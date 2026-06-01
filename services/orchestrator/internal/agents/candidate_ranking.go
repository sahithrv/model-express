package agents

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
)

func FinalizePlannerRecommendation(input ExperimentPlannerInput, recommendation ExperimentPlanningRecommendation) (ExperimentPlanningRecommendation, error) {
	if strings.ToUpper(strings.TrimSpace(recommendation.DecisionType)) != decisions.TypeAddExperiments {
		return recommendation, nil
	}
	if len(recommendation.CandidateHypotheses) == 0 {
		return recommendation, fmt.Errorf("experiment planner ADD_EXPERIMENTS requires candidate_hypotheses for backend ranking")
	}

	rankings, selected, mechanisms := RankPlannerCandidateHypotheses(input, recommendation.CandidateHypotheses, effectiveMaxPlannerExperiments(input))
	recommendation.CandidateRankings = rankings
	recommendation.ProposedExperiments = selected
	recommendation.ProposalMechanisms = mechanisms
	if len(selected) == 0 {
		return recommendation, fmt.Errorf("experiment planner ADD_EXPERIMENTS has no backend-ranked candidate_hypotheses that survived validation")
	}
	if strings.TrimSpace(recommendation.PlanningMode) == "" {
		for _, ranking := range rankings {
			if ranking.Selected && strings.TrimSpace(ranking.PlanningMode) != "" {
				recommendation.PlanningMode = ranking.PlanningMode
				break
			}
		}
	}
	return recommendation, nil
}

func mechanismExhausted(input ExperimentPlannerInput, mechanism string) (bool, string) {
	normalized := normalizeMechanism(mechanism)
	if normalized == "" {
		return false, ""
	}
	group := mechanismGroup(normalized)

	if blockedByRejectedStrategyMemory(input, normalized, group) {
		return true, fmt.Sprintf("mechanism %s is blocked by rejected strategy memory", normalized)
	}
	if blocked, reason := projectTrajectoryBlocksMechanism(input, normalized, group); blocked {
		return true, reason
	}
	if normalized == "architecture_challenge" && repeatedArchitectureAttemptsExhausted(input) {
		return true, "architecture_challenge exhausted after repeated architecture attempts without meaningful improvement"
	}
	return false, ""
}

func projectDecisionPressure(input ExperimentPlannerInput) string {
	if pressure := strings.ToLower(strings.TrimSpace(projectTrajectoryDecisionPressure(input))); pressure != "" {
		return pressure
	}
	switch {
	case containsAnyText(strings.ToLower(strings.Join(input.StopSignals, " ")), "stop", "select", "final", "budget", "exhaust"):
		return "critical"
	case input.MaxFollowUpRounds > 0 && input.FollowUpRound >= input.MaxFollowUpRounds:
		return "critical"
	case input.NoImprovementRounds >= 2:
		return "high"
	case input.NoImprovementRounds == 1:
		return "moderate"
	default:
		return "normal"
	}
}

func effectiveMaxPlannerExperiments(input ExperimentPlannerInput) int {
	maxExperiments := maxPlannerExperiments(input.MaxExperiments)
	switch projectDecisionPressure(input) {
	case "critical", "final", "select_champion", "stop_project":
		return minInt(maxExperiments, 1)
	case "high", "non_exhausted_mechanism_or_stop", "champion_confirmation_or_non_architecture_pivot":
		return minInt(maxExperiments, 2)
	case "moderate":
		return minInt(maxExperiments, 3)
	default:
		return maxExperiments
	}
}

func RankPlannerCandidateHypotheses(input ExperimentPlannerInput, candidates []CandidateHypothesis, maxExperiments int) ([]CandidateRanking, []plans.PlannedExperiment, []PlannerProposalMechanism) {
	if maxExperiments < 1 {
		maxExperiments = 5
	}
	if maxExperiments > 5 {
		maxExperiments = 5
	}

	existing := map[string]bool{}
	for _, signature := range input.ExistingExperimentSignatures {
		existing[signature] = true
	}
	seenProposed := map[string]bool{}
	rankings := make([]CandidateRanking, 0, len(candidates))
	for index, candidate := range candidates {
		ranking := scorePlannerCandidate(input, candidate, index, existing, seenProposed)
		rankings = append(rankings, ranking)
		if !ranking.Rejected {
			seenProposed[ranking.ExperimentSignature] = true
		}
	}

	ordered := append([]CandidateRanking(nil), rankings...)
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].Rejected != ordered[j].Rejected {
			return !ordered[i].Rejected
		}
		if ordered[i].Score == ordered[j].Score {
			return ordered[i].CandidateIndex < ordered[j].CandidateIndex
		}
		return ordered[i].Score > ordered[j].Score
	})

	selectedIndexes := map[int]bool{}
	selected := []plans.PlannedExperiment{}
	selectedMechanisms := []PlannerProposalMechanism{}
	selectedFamilyCounts := map[string]int{}
	for _, ranking := range ordered {
		if ranking.Rejected || len(selected) >= maxExperiments {
			continue
		}
		experiment := candidates[ranking.CandidateIndex].ExperimentConfig
		family := inferExperimentFamily(experiment.Model)
		if len(selected) >= 2 && selectedFamilyCounts[family] >= 2 {
			ranking.Score = roundCandidateScore(ranking.Score - 0.12)
		}
		selectedIndexes[ranking.CandidateIndex] = true
		selectedMechanisms = append(selectedMechanisms, plannerProposalMechanismFromCandidate(candidates[ranking.CandidateIndex], len(selected)))
		selected = append(selected, experiment)
		selectedFamilyCounts[family]++
	}

	for index := range rankings {
		if selectedIndexes[rankings[index].CandidateIndex] {
			rankings[index].Selected = true
			rankings[index].Reasons = append(rankings[index].Reasons, "selected by deterministic backend ranking")
		}
	}
	return rankings, selected, selectedMechanisms
}

func scorePlannerCandidate(input ExperimentPlannerInput, candidate CandidateHypothesis, index int, existing map[string]bool, seenProposed map[string]bool) CandidateRanking {
	experiment := candidate.ExperimentConfig
	signature := candidateExperimentSignature(experiment)
	ranking := CandidateRanking{
		CandidateIndex:      index,
		Hypothesis:          candidate.Hypothesis,
		PlanningMode:        candidate.PlanningMode,
		Mechanism:           normalizeMechanism(candidate.Mechanism),
		Intervention:        strings.TrimSpace(candidate.Intervention),
		ExpectedEffect:      strings.TrimSpace(candidate.ExpectedEffect),
		Score:               0.45,
		ScoreComponents:     map[string]float64{"base": 0.45},
		Reasons:             []string{},
		ExperimentSignature: signature,
	}

	if err := validatePlannedExperimentShape(experiment, index); err != nil {
		ranking.Rejected = true
		ranking.Score = 0
		ranking.Reasons = append(ranking.Reasons, err.Error())
		return ranking
	}
	if err := validateCandidateMechanismExpectation(candidate, index); err != nil {
		ranking.Rejected = true
		ranking.Score = 0
		ranking.Reasons = append(ranking.Reasons, err.Error())
		return ranking
	}
	if exhausted, reason := mechanismExhausted(input, candidate.Mechanism); exhausted {
		ranking.Rejected = true
		ranking.Score = 0
		ranking.Reasons = append(ranking.Reasons, reason)
		return ranking
	}
	if existing[signature] {
		ranking.Rejected = true
		ranking.Score = 0
		ranking.Reasons = append(ranking.Reasons, "duplicate experiment signature already exists")
		return ranking
	}
	if seenProposed[signature] {
		ranking.Rejected = true
		ranking.Score = 0
		ranking.Reasons = append(ranking.Reasons, "duplicate candidate signature in this planner output")
		return ranking
	}

	expectedGain := candidate.ExpectedMetricImpact
	if expectedGain < 0 {
		expectedGain = 0
	}
	expectedGainBonus := clampCandidate(expectedGain*6.0, 0, 0.30)
	noveltyBonus := clampCandidate(candidate.NoveltyScore, 0, 1) * 0.16
	ranking.Score += expectedGainBonus
	ranking.Score += noveltyBonus
	ranking.ScoreComponents["expected_gain"] = roundCandidateScore(expectedGainBonus)
	ranking.ScoreComponents["novelty"] = roundCandidateScore(noveltyBonus)

	costPenalty := candidateCostPenalty(candidate.CostLevel, experiment)
	if costPenalty > 0 {
		ranking.Score -= costPenalty
		ranking.Reasons = append(ranking.Reasons, fmt.Sprintf("cost penalty %.2f", costPenalty))
	}
	ranking.ScoreComponents["cost"] = roundCandidateScore(-costPenalty)
	riskPenalty := candidateRiskPenalty(candidate.Risk)
	if riskPenalty > 0 {
		ranking.Score -= riskPenalty
		ranking.Reasons = append(ranking.Reasons, fmt.Sprintf("risk penalty %.2f", riskPenalty))
	}
	ranking.ScoreComponents["risk"] = roundCandidateScore(-riskPenalty)

	deploymentFitScore, deploymentFitReason := candidateDeploymentFitScore(input.ObjectiveContext, experiment, expectedGain)
	ranking.Score += deploymentFitScore
	if deploymentFitReason != "" {
		ranking.Reasons = append(ranking.Reasons, deploymentFitReason)
	}
	ranking.ScoreComponents["deployment_fit"] = roundCandidateScore(deploymentFitScore)

	redundancyPenalty, redundancyReason := candidateRedundancyPenalty(input, candidate, experiment)
	if redundancyPenalty > 0 {
		ranking.Score -= redundancyPenalty
		ranking.Reasons = append(ranking.Reasons, redundancyReason)
	}
	ranking.ScoreComponents["redundancy"] = roundCandidateScore(-redundancyPenalty)

	mechanismScore, mechanismReasons := candidateMechanismScore(input, candidate, experiment)
	ranking.Score += mechanismScore
	ranking.Reasons = append(ranking.Reasons, mechanismReasons...)
	ranking.ScoreComponents["mechanism"] = roundCandidateScore(mechanismScore)

	if tinyOnlyCandidate(candidate) {
		ranking.Score -= 0.45
		ranking.Reasons = append(ranking.Reasons, "tiny-only candidate: only epochs, learning rate, or batch size changed")
	}
	if highCostWithoutEvidence(candidate, expectedGain) {
		ranking.Score -= 0.28
		ranking.Reasons = append(ranking.Reasons, "high-cost candidate lacks strong expected gain or evidence")
	}
	alignmentBonus, alignmentReasons := candidateDiagnosisAlignment(input.DeterministicDiagnosis, candidate, experiment)
	ranking.Score += alignmentBonus
	ranking.Reasons = append(ranking.Reasons, alignmentReasons...)
	ranking.ScoreComponents["diagnosis_alignment"] = roundCandidateScore(alignmentBonus)

	memoryBonus, memoryReasons, memoryHits, blockedByRetrievedMemory := candidateMemoryScore(input, candidate, experiment)
	ranking.Score += memoryBonus
	ranking.Reasons = append(ranking.Reasons, memoryReasons...)
	ranking.RetrievedMemoryHits = memoryHits
	ranking.ScoreComponents["memory_similarity"] = roundCandidateScore(memoryBonus)
	ranking.ScoreComponents["retrieved_memory"] = roundCandidateScore(retrievedMemoryComponent(memoryHits))
	if blockedByRetrievedMemory {
		ranking.Rejected = true
		ranking.Score = 0
		ranking.Reasons = append(ranking.Reasons, "blocked by retrieved rejected option")
		return ranking
	}

	if ranking.Score < 0.20 {
		ranking.Rejected = true
		ranking.Reasons = append(ranking.Reasons, "score below backend acceptance threshold")
	}
	ranking.Score = roundCandidateScore(clampCandidate(ranking.Score, 0, 1))
	if len(ranking.Reasons) == 0 {
		ranking.Reasons = append(ranking.Reasons, "balanced expected gain, novelty, cost, risk, diagnosis alignment, and memory fit")
	}
	return ranking
}

func candidateDiagnosisAlignment(diagnosis PlannerDiagnosis, candidate CandidateHypothesis, experiment plans.PlannedExperiment) (float64, []string) {
	bonus := 0.0
	reasons := []string{}
	text := strings.ToLower(strings.Join([]string{
		candidate.Hypothesis,
		candidate.Mechanism,
		candidate.Intervention,
		candidate.ExpectedEffect,
		experiment.Reason,
		experiment.Strategy,
		experiment.ClassBalancing,
		experiment.SamplingStrategy,
		experiment.ResolutionStrategy,
		experiment.AugmentationPolicy,
		compactJSON(experiment.ClassBalancingConfig),
		plannerPreprocessingText(experiment),
		strings.Join(candidate.EvidenceUsed, " "),
	}, " "))
	if (diagnosis.ClassImbalanceScore >= 0.45 || diagnosis.MinorityClassFailureScore >= 0.45) &&
		containsAnyText(text, "weight", "weighted", "focal", "balance", "sampler", "minority", "macro") {
		bonus += 0.18
		reasons = append(reasons, "aligned with class imbalance/minority failure diagnosis")
	}
	if diagnosis.OverfittingScore >= 0.55 && containsAnyText(text, "augment", "regular", "dropout", "weight_decay", "smaller", "freeze") {
		bonus += 0.14
		reasons = append(reasons, "aligned with overfitting diagnosis")
	}
	if diagnosis.UnderfittingScore >= 0.55 && (containsAnyText(text, "larger", "full", "fine", "capacity") || isHigherCapacityFamily(experiment.Model)) {
		bonus += 0.14
		reasons = append(reasons, "aligned with underfitting diagnosis")
	}
	if diagnosis.PlateauScore >= 0.55 && containsAnyText(text, "model", "family", "scheduler", "augmentation", "preprocess", "image") {
		bonus += 0.10
		reasons = append(reasons, "aligned with plateau diagnosis")
	}
	if diagnosis.LatencyPenalty >= 0.45 && isFastLiveModel(experiment.Model) {
		bonus += 0.10
		reasons = append(reasons, "uses latency-friendly model under latency penalty")
	}
	return bonus, reasons
}

func candidateMechanismScore(input ExperimentPlannerInput, candidate CandidateHypothesis, experiment plans.PlannedExperiment) (float64, []string) {
	score := 0.0
	reasons := []string{}
	mechanism := normalizeMechanism(candidate.Mechanism)
	if mechanism == "" {
		return score, reasons
	}

	if diagnosisMatchesMechanism(input.DeterministicDiagnosis, mechanism, candidate, experiment) {
		if isModelShoppingMechanism(mechanism) {
			score += 0.04
			reasons = append(reasons, "mechanism has some diagnosis support")
		} else {
			score += 0.12
			reasons = append(reasons, "diagnosis-matched non-model mechanism")
		}
	} else if mechanismUsuallyNeedsDiagnosis(mechanism) {
		score -= 0.10
		reasons = append(reasons, "mechanism is weakly supported by deterministic diagnosis")
	}

	if !mechanismReflectedInExperimentConfig(mechanism, experiment) {
		score -= 0.18
		reasons = append(reasons, "mechanism is not reflected in executable experiment config")
	}
	if architectureOnlyCandidate(candidate) && !architectureMechanismSupported(input.DeterministicDiagnosis) {
		score -= 0.22
		reasons = append(reasons, "architecture-only candidate lacks underfitting, plateau, or champion-challenge evidence")
	}
	if sameMechanismMinorVariant(input, candidate, experiment) {
		score -= 0.24
		reasons = append(reasons, "same mechanism only changes minor tuning knobs")
	}
	return score, reasons
}

func blockedByRejectedStrategyMemory(input ExperimentPlannerInput, mechanism string, group string) bool {
	blocked := mechanismsFromRejectedOptions(input.RejectedStrategyMemory)
	for blockedMechanism := range blocked {
		if mechanismMatches(blockedMechanism, mechanism, group) {
			return true
		}
	}
	return false
}

func projectTrajectoryBlocksMechanism(input ExperimentPlannerInput, mechanism string, group string) (bool, string) {
	trajectory, ok := projectTrajectoryValue(input)
	if !ok {
		return false, ""
	}
	for _, blocked := range stringsFromReflectField(trajectory, "BlockedMechanisms") {
		if mechanismMatches(blocked, mechanism, group) {
			return true, fmt.Sprintf("mechanism %s is blocked by project trajectory", mechanism)
		}
	}
	outcomes := reflectField(trajectory, "MechanismOutcomes")
	if !outcomes.IsValid() || (outcomes.Kind() != reflect.Slice && outcomes.Kind() != reflect.Array) {
		return false, ""
	}
	failureCount := 0
	for index := 0; index < outcomes.Len(); index++ {
		outcome := reflectIndirect(outcomes.Index(index))
		if !outcome.IsValid() {
			continue
		}
		outcomeMechanism := reflectedMechanismOutcomeMechanism(outcome)
		if !mechanismMatches(outcomeMechanism, mechanism, group) {
			continue
		}
		if reflectedBoolField(outcome, "Exhausted") || reflectedBoolField(outcome, "Blocked") {
			return true, fmt.Sprintf("mechanism %s is exhausted by project trajectory", mechanism)
		}
		status := strings.ToLower(strings.Join([]string{
			reflectedStringField(outcome, "Outcome"),
			reflectedStringField(outcome, "OutcomeStatus"),
			reflectedStringField(outcome, "Status"),
			reflectedStringField(outcome, "State"),
		}, " "))
		if containsAnyText(status, "exhausted", "blocked") {
			return true, fmt.Sprintf("mechanism %s is exhausted by project trajectory", mechanism)
		}
		if containsAnyText(status, ExperimentPlanningOutcomeNoImprovement, ExperimentPlanningOutcomeFailed, "no improvement", "failed") {
			failureCount += candidateMaxInt(1, reflectedIntField(outcome, "Attempts"), reflectedIntField(outcome, "AttemptCount"), reflectedIntField(outcome, "NoImprovementAttempts"), reflectedIntField(outcome, "FailedAttempts"))
		}
	}
	if mechanism == "architecture_challenge" && failureCount >= 2 {
		return true, "architecture_challenge exhausted by project trajectory after repeated no-improvement outcomes"
	}
	return false, ""
}

func projectTrajectoryDecisionPressure(input ExperimentPlannerInput) string {
	trajectory, ok := projectTrajectoryValue(input)
	if !ok {
		return ""
	}
	return reflectedStringField(trajectory, "DecisionPressure")
}

func projectTrajectoryValue(input ExperimentPlannerInput) (reflect.Value, bool) {
	inputValue := reflect.ValueOf(input)
	if inputValue.Kind() == reflect.Pointer {
		inputValue = reflectIndirect(inputValue)
	}
	if !inputValue.IsValid() || inputValue.Kind() != reflect.Struct {
		return reflect.Value{}, false
	}
	return validReflectField(inputValue, "ProjectTrajectory")
}

func reflectedMechanismOutcomeMechanism(outcome reflect.Value) string {
	for _, fieldName := range []string{"Mechanism", "MechanismName", "Name", "StrategyType"} {
		if value := reflectedStringField(outcome, fieldName); value != "" {
			return value
		}
	}
	return ""
}

func mechanismMatches(candidate string, mechanism string, group string) bool {
	normalized := normalizeMechanism(candidate)
	if normalized == "" {
		return false
	}
	return normalized == mechanism || mechanismGroup(normalized) == group
}

func repeatedArchitectureAttemptsExhausted(input ExperimentPlannerInput) bool {
	failures := 0
	for _, scorecard := range input.StrategyScorecards {
		if !mechanismMatches(mechanismFromScorecard(scorecard), "architecture_challenge", "architecture_challenge") &&
			!mechanismMatches(scorecard.StrategyType, "architecture_challenge", "architecture_challenge") {
			continue
		}
		switch scorecard.Outcome {
		case ExperimentPlanningOutcomeFailed, ExperimentPlanningOutcomeNoImprovement:
			failures++
		case ExperimentPlanningOutcomeMinorImprovement:
			if input.MinimumMeaningfulImprovement > 0 && scorecard.ActualDelta < input.MinimumMeaningfulImprovement {
				failures++
			}
		}
	}
	for _, memory := range input.FailedStrategyMemory {
		text := strings.ToLower(strings.Join(append([]string{memory.OutcomeStatus, memory.Lesson, memory.BestModel}, memory.Tags...), " "))
		if containsAnyText(text, "architecture_challenge", "architecture challenge", "model family", "model shopping") {
			failures++
		}
	}
	if failures >= 2 {
		return true
	}
	architectureExperiments := 0
	for _, experiment := range input.SourcePlan.Experiments {
		if mechanismMatches(inferExperimentMechanismTaxonomy(experiment), "architecture_challenge", "architecture_challenge") {
			architectureExperiments++
		}
	}
	for _, plan := range input.PriorPlans {
		for _, experiment := range plan.Experiments {
			if mechanismMatches(inferExperimentMechanismTaxonomy(experiment), "architecture_challenge", "architecture_challenge") {
				architectureExperiments++
			}
		}
	}
	return architectureExperiments >= 2 && input.NoImprovementRounds >= 2
}

func diagnosisMatchesMechanism(diagnosis PlannerDiagnosis, mechanism string, candidate CandidateHypothesis, experiment plans.PlannedExperiment) bool {
	text := candidateMechanismEvidenceText(candidate, experiment)
	switch mechanism {
	case "class_imbalance", "minority_targeting":
		return diagnosis.ClassImbalanceScore >= 0.45 || diagnosis.MinorityClassFailureScore >= 0.45 || containsAnyText(text, "minority", "rare class", "macro", "imbalance", "per-class", "per class")
	case "regularization", "augmentation_basic", "augmentation_auto", "augmentation_mixed_sample":
		return diagnosis.OverfittingScore >= 0.55 || diagnosis.InstabilityScore >= 0.45 || containsAnyText(text, "overfit", "small dataset", "blur", "lighting", "viewpoint", "background")
	case "optimizer_scheduler":
		return diagnosis.PlateauScore >= 0.55 || diagnosis.InstabilityScore >= 0.45 || containsAnyText(text, "plateau", "unstable", "scheduler", "optimization")
	case "capacity_finetune":
		return diagnosis.UnderfittingScore >= 0.55 || containsAnyText(text, "underfit", "capacity", "fine-tune", "fine tune", "full")
	case "resolution_crop", "bbox_crop_ablation":
		return containsAnyText(text, "small object", "object scale", "crop", "bbox", "aspect", "background", "resolution", "fine-grained")
	case "deployment_latency":
		return diagnosis.LatencyPenalty >= 0.45 || containsAnyText(text, "latency", "runtime", "cost", "live")
	case "architecture_challenge":
		return architectureMechanismSupported(diagnosis)
	case "baseline_control", "stop_select_champion":
		return true
	default:
		return false
	}
}

func candidateMechanismEvidenceText(candidate CandidateHypothesis, experiment plans.PlannedExperiment) string {
	return strings.ToLower(strings.Join([]string{
		candidate.Hypothesis,
		candidate.Mechanism,
		candidate.Intervention,
		candidate.ExpectedEffect,
		strings.Join(candidate.EvidenceUsed, " "),
		experiment.Reason,
		experiment.Strategy,
		experiment.ResolutionStrategy,
		experiment.AugmentationPolicy,
		compactJSON(experiment.AugmentationPolicyConfig),
		experiment.ClassBalancing,
		compactJSON(experiment.ClassBalancingConfig),
		experiment.SamplingStrategy,
		plannerPreprocessingText(experiment),
	}, " "))
}

func architectureMechanismSupported(diagnosis PlannerDiagnosis) bool {
	return diagnosis.UnderfittingScore >= 0.55 || diagnosis.PlateauScore >= 0.60 || diagnosis.ImprovementStagnationScore >= 0.60
}

func mechanismUsuallyNeedsDiagnosis(mechanism string) bool {
	switch mechanism {
	case "baseline_control", "stop_select_champion":
		return false
	default:
		return true
	}
}

func isModelShoppingMechanism(mechanism string) bool {
	return mechanism == "architecture_challenge"
}

func mechanismReflectedInExperimentConfig(mechanism string, experiment plans.PlannedExperiment) bool {
	switch mechanismGroup(mechanism) {
	case "class_imbalance":
		return nonDefaultText(experiment.ClassBalancing, "none") || nonDefaultText(experiment.SamplingStrategy, "none")
	case "augmentation":
		return nonDefaultText(experiment.AugmentationPolicy, "none") || len(experiment.Augmentation) > 0 || experiment.AugmentationPolicyConfig != nil
	case "resolution_crop":
		return (experiment.ImageSize > 0 && experiment.ImageSize != 224) || nonDefaultText(experiment.ResolutionStrategy, "fixed") || experiment.Preprocessing != nil
	case "regularization":
		return experiment.WeightDecay > 0 || experiment.Dropout > 0 || experiment.LabelSmoothing > 0 || experiment.GradientClipNorm > 0 || nonDefaultText(experiment.AugmentationPolicy, "none") || len(experiment.Augmentation) > 0 || experiment.AugmentationPolicyConfig != nil
	case "optimizer_scheduler":
		return nonDefaultText(experiment.Optimizer, "adamw") || nonDefaultText(experiment.Scheduler, "none") || experiment.WeightDecay > 0 || experiment.OptimizerMomentum > 0 || experiment.SchedulerStepSize > 0 || experiment.SchedulerGamma > 0
	case "capacity_finetune":
		return nonDefaultText(experiment.FineTuneStrategy, "head_only") || (experiment.Pretrained && !experiment.FreezeBackbone) || isHigherCapacityFamily(experiment.Model)
	case "deployment_latency":
		return strings.TrimSpace(experiment.Model) != ""
	case "architecture_challenge":
		return strings.TrimSpace(experiment.Model) != "" || strings.TrimSpace(experiment.Template) != ""
	case "label_noise_audit", "hard_example_audit":
		return strings.EqualFold(strings.TrimSpace(experiment.Template), "label_quality_audit")
	case "baseline_control", "stop_select_champion":
		return true
	default:
		return false
	}
}

func candidateMemoryScore(input ExperimentPlannerInput, candidate CandidateHypothesis, experiment plans.PlannedExperiment) (float64, []string, []CandidateRetrievedMemoryHit, bool) {
	bonus := 0.0
	reasons := []string{}
	hits := []CandidateRetrievedMemoryHit{}
	blocked := false
	model := strings.ToLower(strings.TrimSpace(experiment.Model))
	for _, memory := range input.SuccessfulStrategyMemory {
		if stringSliceContainsFold(candidate.SimilarSuccessMemoryIDs, memory.MemoryID) || stringSliceContainsFold(memory.ProposedModels, model) {
			bonus += 0.06
			reasons = append(reasons, "similar successful strategy memory")
			break
		}
	}
	for _, memory := range input.FailedStrategyMemory {
		if stringSliceContainsFold(candidate.SimilarFailureMemoryIDs, memory.MemoryID) || stringSliceContainsFold(memory.ProposedModels, model) {
			bonus -= 0.10
			reasons = append(reasons, "similar failed strategy memory")
			break
		}
	}
	for _, scorecard := range input.StrategyScorecards {
		if !scorecardSimilarToCandidate(scorecard, candidate, experiment) {
			continue
		}
		switch scorecard.Outcome {
		case ExperimentPlanningOutcomeImprovedChampion, ExperimentPlanningOutcomeMinorImprovement:
			bonus += 0.05
			reasons = append(reasons, "similar successful strategy scorecard")
		case ExperimentPlanningOutcomeNoImprovement, ExperimentPlanningOutcomeFailed:
			bonus -= 0.08
			reasons = append(reasons, "similar failed strategy scorecard")
		}
		break
	}
	retrievedScore, retrievedReasons, retrievedHits, retrievedBlocked := candidateRetrievedMemoryScore(input.RetrievedMemory, candidate, experiment)
	bonus += retrievedScore
	reasons = append(reasons, retrievedReasons...)
	hits = append(hits, retrievedHits...)
	blocked = retrievedBlocked
	return bonus, reasons, hits, blocked
}

func retrievedMemoryComponent(hits []CandidateRetrievedMemoryHit) float64 {
	score := 0.0
	for _, hit := range hits {
		score += hit.AppliedScore
	}
	return clampCandidate(score, -0.35, 0.18)
}

func candidateRetrievedMemoryScore(results []memory.MemoryRetrievalResult, candidate CandidateHypothesis, experiment plans.PlannedExperiment) (float64, []string, []CandidateRetrievedMemoryHit, bool) {
	if len(results) == 0 {
		return 0, nil, nil, false
	}
	score := 0.0
	reasons := []string{}
	hits := []CandidateRetrievedMemoryHit{}
	blocked := false
	seen := map[string]bool{}
	for _, result := range results {
		card := plannerRetrievedMemoryCard(result)
		key := plannerRetrievedMemoryDedupeKey(card)
		if key == "" || seen[key] {
			continue
		}
		seen[key] = true
		matchStrength, matchedFields := candidateRetrievedMemoryMatch(card, candidate, experiment)
		if matchStrength < 0.45 {
			continue
		}
		appliedScore, effect, reason, shouldBlock := candidateRetrievedMemoryEffect(card, matchStrength)
		if appliedScore == 0 || effect == "" {
			continue
		}
		score += appliedScore
		reasons = append(reasons, reason)
		if len(hits) < 4 {
			hits = append(hits, CandidateRetrievedMemoryHit{
				SourceTable:     card.SourceTable,
				SourceID:        card.SourceID,
				Kind:            card.Kind,
				Outcome:         card.Outcome,
				Mechanism:       card.Mechanism,
				Intervention:    card.Intervention,
				Lesson:          card.Lesson,
				Summary:         card.Summary,
				RetrievalReason: card.RetrievalReason,
				Score:           roundCandidateScore(card.Score),
				AppliedScore:    roundCandidateScore(appliedScore),
				Effect:          effect,
				MatchedFields:   matchedFields,
			})
		}
		if shouldBlock {
			blocked = true
		}
	}
	return clampCandidate(score, -0.35, 0.18), uniqueCandidateReasons(reasons), hits, blocked
}

func candidateRetrievedMemoryEffect(card PlannerRetrievedMemoryCard, matchStrength float64) (float64, string, string, bool) {
	confidence := clampCandidate(card.Score, 0, 1)
	if confidence == 0 {
		confidence = 0.6
	}
	weight := 0.65 + 0.35*confidence
	bucket := plannerRetrievedMemoryBucket(card)
	switch bucket {
	case "blocked_or_rejected":
		score := -clampCandidate(0.18+0.18*matchStrength*weight, 0.18, 0.35)
		return score, "blocked", "blocked by retrieved rejected option", matchStrength >= 0.72
	case "successful_strategy":
		score := clampCandidate(0.05+0.08*matchStrength*weight, 0.05, 0.14)
		return score, "bonus", "similar retrieved successful strategy", false
	case "failed_strategy":
		score := -clampCandidate(0.07+0.09*matchStrength*weight, 0.07, 0.18)
		return score, "penalty", "similar retrieved failed mechanism", false
	case "dataset_preprocessing":
		score := clampCandidate(0.03+0.05*matchStrength*weight, 0.03, 0.08)
		return score, "bonus", "retrieved dataset preprocessing analogy", false
	default:
		outcome := strings.ToLower(strings.TrimSpace(card.Outcome))
		switch {
		case containsAnyText(outcome, ExperimentPlanningOutcomeImprovedChampion, ExperimentPlanningOutcomeMinorImprovement, "success", "accepted"):
			return clampCandidate(0.03+0.04*matchStrength*weight, 0.03, 0.08), "bonus", "similar retrieved positive memory", false
		case containsAnyText(outcome, ExperimentPlanningOutcomeNoImprovement, ExperimentPlanningOutcomeFailed, "failed", "failure", "invalidated", "rejected"):
			return -clampCandidate(0.04+0.05*matchStrength*weight, 0.04, 0.10), "penalty", "similar retrieved negative memory", false
		default:
			return 0, "", "", false
		}
	}
}

func candidateRetrievedMemoryMatch(card PlannerRetrievedMemoryCard, candidate CandidateHypothesis, experiment plans.PlannedExperiment) (float64, []string) {
	mechanism := normalizeMechanism(candidate.Mechanism)
	group := mechanismGroup(mechanism)
	score := 0.0
	fields := []string{}
	cardMechanism := normalizeMechanism(card.Mechanism)
	if mechanism != "" && cardMechanism != "" && mechanismMatches(cardMechanism, mechanism, group) {
		score += 0.55
		fields = append(fields, "mechanism")
	}
	if planningMode := candidateMemoryCardValue(card, "planning_mode"); planningMode != "" && strings.EqualFold(planningMode, candidate.PlanningMode) {
		score += 0.20
		fields = append(fields, "planning_mode")
	}
	if candidateMemoryInterventionMatches(card.Intervention, candidate, experiment) {
		score += 0.20
		fields = append(fields, "intervention")
	}
	if candidateMemoryModelMatches(card, experiment) {
		score += 0.18
		fields = append(fields, "model_family")
	}
	if candidateMemoryPreprocessingMatches(card, candidate, experiment) {
		score += 0.18
		fields = append(fields, "preprocessing")
	}
	if score == 0 {
		return 0, nil
	}
	return clampCandidate(score, 0, 1), fields
}

func candidateMemoryInterventionMatches(cardIntervention string, candidate CandidateHypothesis, experiment plans.PlannedExperiment) bool {
	cardIntervention = strings.ToLower(strings.TrimSpace(cardIntervention))
	if cardIntervention == "" {
		return false
	}
	candidateText := strings.ToLower(strings.Join([]string{
		candidate.Intervention,
		candidate.ExpectedEffect,
		experiment.ClassBalancing,
		experiment.SamplingStrategy,
		experiment.ResolutionStrategy,
		experiment.AugmentationPolicy,
		experiment.Optimizer,
		experiment.Scheduler,
		plannerPreprocessingText(experiment),
		compactJSON(experiment.ClassBalancingConfig),
		compactJSON(experiment.AugmentationPolicyConfig),
	}, " "))
	for _, token := range candidateMemorySignificantTokens(cardIntervention) {
		if strings.Contains(candidateText, token) {
			return true
		}
	}
	return false
}

func candidateMemoryModelMatches(card PlannerRetrievedMemoryCard, experiment plans.PlannedExperiment) bool {
	model := strings.ToLower(strings.TrimSpace(experiment.Model))
	family := inferExperimentFamily(experiment.Model)
	if model == "" && family == "" {
		return false
	}
	text := strings.ToLower(strings.Join([]string{
		candidateMemoryCardValue(card, "model"),
		candidateMemoryCardValue(card, "models"),
		candidateMemoryCardValue(card, "best_model"),
		candidateMemoryCardValue(card, "model_family"),
	}, " "))
	if text == "" {
		return false
	}
	return (model != "" && strings.Contains(text, model)) || (family != "" && strings.Contains(text, family))
}

func candidateMemoryPreprocessingMatches(card PlannerRetrievedMemoryCard, candidate CandidateHypothesis, experiment plans.PlannedExperiment) bool {
	sourceText := strings.ToLower(strings.Join([]string{card.SourceTable, card.Kind, card.Mechanism}, " "))
	if !containsAnyText(sourceText, memory.SourceDatasetPreprocessing, memory.SourceDatasetVisualAnalysis, memory.KindDatasetPreprocessingHypothesis, "dataset_profile", "preprocessing", "visual") {
		return false
	}
	candidateText := candidateMechanismEvidenceText(candidate, experiment)
	return containsAnyText(candidateText, "preprocess", "crop", "bbox", "resolution", "image_size", "small object", "aspect")
}

func candidateMemoryCardValue(card PlannerRetrievedMemoryCard, keys ...string) string {
	records := []map[string]any{card.Metadata, card.SummaryCard}
	values := []string{}
	for _, record := range records {
		for _, key := range keys {
			if value, ok := record[key]; ok {
				if text := candidateMemoryValueText(value); text != "" {
					values = append(values, text)
				}
			}
		}
	}
	return strings.Join(values, " ")
}

func candidateMemoryValueText(value any) string {
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	case []string:
		return strings.Join(typed, " ")
	case []any:
		values := []string{}
		for _, item := range typed {
			if text := candidateMemoryValueText(item); text != "" {
				values = append(values, text)
			}
		}
		return strings.Join(values, " ")
	case map[string]any:
		values := []string{}
		for key, item := range typed {
			if text := candidateMemoryValueText(item); text != "" {
				values = append(values, key+" "+text)
			}
		}
		return strings.Join(values, " ")
	default:
		return strings.TrimSpace(fmt.Sprint(typed))
	}
}

func candidateMemorySignificantTokens(text string) []string {
	stop := map[string]bool{
		"and": true, "the": true, "for": true, "with": true, "without": true, "from": true, "this": true,
		"that": true, "into": true, "plus": true, "only": true, "use": true, "used": true, "test": true,
		"strategy": true, "mechanism": true, "experiment": true, "candidate": true,
	}
	tokens := []string{}
	for _, token := range strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !(r >= 'a' && r <= 'z') && !(r >= '0' && r <= '9') && r != '_'
	}) {
		token = strings.TrimSpace(token)
		if len(token) < 4 || stop[token] {
			continue
		}
		tokens = append(tokens, token)
	}
	return tokens
}

func uniqueCandidateReasons(values []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	return out
}

func scorecardSimilarToCandidate(scorecard PlannerStrategyScorecard, candidate CandidateHypothesis, experiment plans.PlannedExperiment) bool {
	if scorecard.PlanningMode != "" && strings.EqualFold(scorecard.PlanningMode, candidate.PlanningMode) {
		return true
	}
	changesBlob, _ := json.Marshal(scorecard.ProposedChanges)
	changes := strings.ToLower(string(changesBlob))
	model := strings.ToLower(strings.TrimSpace(experiment.Model))
	family := inferExperimentFamily(experiment.Model)
	mechanism := normalizeMechanism(candidate.Mechanism)
	return strings.Contains(changes, model) || (family != "" && strings.Contains(changes, family)) || (mechanism != "" && strings.Contains(changes, mechanism))
}

func tinyOnlyCandidate(candidate CandidateHypothesis) bool {
	keys := []string{}
	for key := range candidate.ProposedChanges {
		keys = append(keys, key)
	}
	if len(keys) == 0 {
		text := strings.ToLower(candidate.Hypothesis + " " + candidate.ExperimentConfig.Reason + " " + candidate.ExperimentConfig.Strategy)
		return containsAnyText(text, "more epoch", "lower learning", "learning rate", "batch size") &&
			!containsAnyText(text, "model", "augmentation", "preprocess", "image", "class", "balance", "scheduler", "regular")
	}
	for _, key := range keys {
		normalized := strings.ToLower(strings.TrimSpace(key))
		switch normalized {
		case "epoch", "epochs", "learning_rate", "lr", "batch_size":
		default:
			return false
		}
	}
	return true
}

func highCostWithoutEvidence(candidate CandidateHypothesis, expectedGain float64) bool {
	cost := strings.ToLower(strings.TrimSpace(candidate.CostLevel))
	if cost != "high" && cost != "very_high" && cost != "expensive" {
		return false
	}
	return expectedGain < 0.025 || len(nonEmptyStrings(candidate.EvidenceUsed)) == 0
}

func candidateDeploymentFitScore(objective ProjectObjectiveContext, experiment plans.PlannedExperiment, expectedGain float64) (float64, string) {
	if objective.PrimaryObjective != "low_latency_live_service" {
		return 0, ""
	}
	if isFastLiveModel(experiment.Model) {
		return 0.02, "candidate has a compact live-serving profile"
	}
	if expectedGain < 0.015 {
		return -0.04, "candidate should justify its live-budget tradeoff with a measurable quality gain"
	}
	return -0.01, "candidate may be viable if latency stays within the live budget"
}

func candidateRedundancyPenalty(input ExperimentPlannerInput, candidate CandidateHypothesis, experiment plans.PlannedExperiment) (float64, string) {
	model := strings.ToLower(strings.TrimSpace(experiment.Model))
	family := inferExperimentFamily(model)
	seenModel := false
	seenFamilyCount := 0
	for _, plan := range input.PriorPlans {
		for _, prior := range plan.Experiments {
			priorModel := strings.ToLower(strings.TrimSpace(prior.Model))
			if priorModel == model {
				seenModel = true
			}
			if inferExperimentFamily(priorModel) == family {
				seenFamilyCount++
			}
		}
	}
	for _, prior := range input.SourcePlan.Experiments {
		priorModel := strings.ToLower(strings.TrimSpace(prior.Model))
		if priorModel == model {
			seenModel = true
		}
		if inferExperimentFamily(priorModel) == family {
			seenFamilyCount++
		}
	}
	if seenModel && clampCandidate(candidate.NoveltyScore, 0, 1) < 0.35 && !meaningfullyChangesExperiment(candidate) {
		return 0.16, "redundant with prior model and lacks a meaningful new mechanism"
	}
	if seenFamilyCount >= 3 && clampCandidate(candidate.NoveltyScore, 0, 1) < 0.45 {
		return 0.08, "overuses a previously tested model family"
	}
	return 0, ""
}

func meaningfullyChangesExperiment(candidate CandidateHypothesis) bool {
	for key := range candidate.ProposedChanges {
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "mechanism", "intervention", "expected_effect", "model", "model_family", "architecture", "image_size", "resolution_strategy", "augmentation", "augmentation_policy", "preprocessing", "resize_strategy", "normalization", "crop", "crop_strategy", "bbox_mode", "class_balancing", "sampling_strategy", "fine_tune_strategy", "scheduler", "optimizer", "weight_decay", "dropout", "optimizer_momentum", "scheduler_step_size", "scheduler_gamma", "label_smoothing", "gradient_clip_norm", "loss":
			return true
		}
	}
	text := strings.ToLower(strings.Join([]string{
		candidate.Hypothesis,
		candidate.Mechanism,
		candidate.Intervention,
		candidate.ExpectedEffect,
		candidate.ExperimentConfig.Reason,
		candidate.ExperimentConfig.Strategy,
		candidate.ExperimentConfig.ResolutionStrategy,
		candidate.ExperimentConfig.AugmentationPolicy,
		candidate.ExperimentConfig.ClassBalancing,
		candidate.ExperimentConfig.SamplingStrategy,
		plannerPreprocessingText(candidate.ExperimentConfig),
	}, " "))
	return containsAnyText(text, "model family", "augmentation", "augment", "preprocess", "resize", "normalization", "crop", "bbox", "image size", "resolution", "weighted", "balanced", "sampler", "fine-tune", "regularization", "scheduler")
}

func plannerProposalMechanismFromCandidate(candidate CandidateHypothesis, experimentIndex int) PlannerProposalMechanism {
	return PlannerProposalMechanism{
		ExperimentIndex: experimentIndex,
		Mechanism:       normalizeMechanism(candidate.Mechanism),
		Intervention:    strings.TrimSpace(candidate.Intervention),
		EvidenceUsed:    nonEmptyStrings(candidate.EvidenceUsed),
		ExpectedEffect:  strings.TrimSpace(candidate.ExpectedEffect),
	}
}

func architectureOnlyCandidate(candidate CandidateHypothesis) bool {
	mechanism := normalizeMechanism(candidate.Mechanism)
	if mechanism != "architecture_challenge" && !proposedChangesOnlyModel(candidate.ProposedChanges) {
		return false
	}
	if !proposedChangesOnlyModel(candidate.ProposedChanges) {
		return false
	}
	text := candidateMechanismEvidenceText(candidate, candidate.ExperimentConfig)
	return !containsAnyText(text, "weighted", "balance", "sampler", "augment", "regular", "scheduler", "crop", "resolution", "preprocess", "fine-tune", "fine tune")
}

func proposedChangesOnlyModel(changes map[string]any) bool {
	if len(changes) == 0 {
		return false
	}
	for key := range changes {
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "mechanism", "model", "model_family", "architecture", "template", "epochs", "epoch", "learning_rate", "lr", "batch_size":
		default:
			return false
		}
	}
	return true
}

func sameMechanismMinorVariant(input ExperimentPlannerInput, candidate CandidateHypothesis, experiment plans.PlannedExperiment) bool {
	mechanism := normalizeMechanism(candidate.Mechanism)
	if mechanism == "" {
		return false
	}
	priors := append([]plans.PlannedExperiment(nil), input.SourcePlan.Experiments...)
	for _, plan := range input.PriorPlans {
		priors = append(priors, plan.Experiments...)
	}
	for _, prior := range priors {
		if mechanismGroup(inferExperimentMechanismTaxonomy(prior)) != mechanismGroup(mechanism) {
			continue
		}
		if experimentsDifferOnlyMinor(prior, experiment) {
			return true
		}
	}
	return false
}

func experimentsDifferOnlyMinor(left plans.PlannedExperiment, right plans.PlannedExperiment) bool {
	return strings.EqualFold(strings.TrimSpace(left.Template), strings.TrimSpace(right.Template)) &&
		strings.EqualFold(strings.TrimSpace(left.Model), strings.TrimSpace(right.Model)) &&
		left.ImageSize == right.ImageSize &&
		strings.EqualFold(strings.TrimSpace(left.ResolutionStrategy), strings.TrimSpace(right.ResolutionStrategy)) &&
		strings.EqualFold(strings.TrimSpace(left.Optimizer), strings.TrimSpace(right.Optimizer)) &&
		strings.EqualFold(strings.TrimSpace(left.Scheduler), strings.TrimSpace(right.Scheduler)) &&
		left.WeightDecay == right.WeightDecay &&
		left.Dropout == right.Dropout &&
		left.OptimizerMomentum == right.OptimizerMomentum &&
		left.SchedulerStepSize == right.SchedulerStepSize &&
		left.SchedulerGamma == right.SchedulerGamma &&
		left.LabelSmoothing == right.LabelSmoothing &&
		left.GradientClipNorm == right.GradientClipNorm &&
		strings.EqualFold(strings.TrimSpace(left.AugmentationPolicy), strings.TrimSpace(right.AugmentationPolicy)) &&
		strings.EqualFold(strings.TrimSpace(left.ClassBalancing), strings.TrimSpace(right.ClassBalancing)) &&
		strings.EqualFold(strings.TrimSpace(left.SamplingStrategy), strings.TrimSpace(right.SamplingStrategy)) &&
		left.Pretrained == right.Pretrained &&
		left.FreezeBackbone == right.FreezeBackbone &&
		strings.EqualFold(strings.TrimSpace(left.FineTuneStrategy), strings.TrimSpace(right.FineTuneStrategy)) &&
		compactJSON(left.Preprocessing) == compactJSON(right.Preprocessing) &&
		compactJSON(left.Augmentation) == compactJSON(right.Augmentation) &&
		compactJSON(left.AugmentationPolicyConfig) == compactJSON(right.AugmentationPolicyConfig) &&
		compactJSON(left.ClassBalancingConfig) == compactJSON(right.ClassBalancingConfig)
}

func inferExperimentMechanismTaxonomy(experiment plans.PlannedExperiment) string {
	if mechanism := normalizeMechanism(experiment.Mechanism); mechanism != "" {
		return mechanism
	}
	if nonDefaultText(experiment.ClassBalancing, "none") || nonDefaultText(experiment.SamplingStrategy, "none") {
		return "class_imbalance"
	}
	if experiment.Preprocessing != nil && (nonDefaultText(experiment.Preprocessing.BBoxMode, "ignore") || strings.Contains(strings.ToLower(experiment.Preprocessing.CropStrategy), "bbox")) {
		return "bbox_crop_ablation"
	}
	if (experiment.ImageSize > 0 && experiment.ImageSize != 224) || nonDefaultText(experiment.ResolutionStrategy, "fixed") ||
		(experiment.Preprocessing != nil && (nonDefaultText(experiment.Preprocessing.ResizeStrategy, "squash") || nonDefaultText(experiment.Preprocessing.CropStrategy, "none") || nonDefaultText(experiment.Preprocessing.Normalization, "imagenet"))) {
		return "resolution_crop"
	}
	if nonDefaultText(experiment.AugmentationPolicy, "none") || len(experiment.Augmentation) > 0 || experiment.AugmentationPolicyConfig != nil {
		if experiment.AugmentationPolicyConfig != nil {
			switch normalizeMechanism(experiment.AugmentationPolicyConfig.PolicyType) {
			case "randaugment", "trivialaugment", "trivialaugmentwide", "autoaugment":
				return "augmentation_auto"
			case "mixup", "cutmix":
				return "augmentation_mixed_sample"
			}
		}
		return "augmentation_basic"
	}
	if nonDefaultText(experiment.FineTuneStrategy, "head_only") || (experiment.Pretrained && !experiment.FreezeBackbone) {
		return "capacity_finetune"
	}
	if nonDefaultText(experiment.Optimizer, "adamw") || nonDefaultText(experiment.Scheduler, "none") || experiment.WeightDecay > 0 {
		return "optimizer_scheduler"
	}
	if strings.TrimSpace(experiment.Model) != "" || strings.TrimSpace(experiment.Template) != "" {
		return "architecture_challenge"
	}
	return "baseline_control"
}

func normalizeMechanism(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func mechanismGroup(mechanism string) string {
	switch normalizeMechanism(mechanism) {
	case "class_imbalance", "minority_targeting":
		return "class_imbalance"
	case "augmentation_basic", "augmentation_auto", "augmentation_mixed_sample":
		return "augmentation"
	case "resolution_crop", "bbox_crop_ablation":
		return "resolution_crop"
	default:
		return normalizeMechanism(mechanism)
	}
}

func nonDefaultText(value string, defaults ...string) bool {
	normalized := strings.ToLower(strings.TrimSpace(value))
	if normalized == "" {
		return false
	}
	for _, fallback := range defaults {
		if normalized == strings.ToLower(strings.TrimSpace(fallback)) {
			return false
		}
	}
	return true
}

func compactJSON(value any) string {
	blob, _ := json.Marshal(value)
	return string(blob)
}

func plannerPreprocessingText(experiment plans.PlannedExperiment) string {
	if experiment.Preprocessing == nil {
		return ""
	}
	blob, _ := json.Marshal(experiment.Preprocessing)
	return string(blob)
}

func candidateCostPenalty(costLevel string, experiment plans.PlannedExperiment) float64 {
	switch strings.ToLower(strings.TrimSpace(costLevel)) {
	case "very_high", "expensive":
		return 0.20
	case "high":
		return 0.14
	case "medium":
		return 0.06
	}
	if experiment.Epochs >= 25 || experiment.ImageSize >= 320 {
		return 0.08
	}
	return 0
}

func candidateRiskPenalty(risk string) float64 {
	normalized := strings.ToLower(strings.TrimSpace(risk))
	switch {
	case normalized == "":
		return 0.02
	case containsAnyText(normalized, "high", "risky", "unstable"):
		return 0.12
	case containsAnyText(normalized, "medium", "moderate"):
		return 0.06
	default:
		return 0
	}
}

func isFastLiveModel(model string) bool {
	family := inferExperimentFamily(model)
	return family == "mobilenet" || family == "efficientnet" || family == "regnet"
}

func isHigherCapacityFamily(model string) bool {
	family := inferExperimentFamily(model)
	return family == "efficientnet" || family == "convnext" || family == "swin" || family == "vit"
}

func candidateExperimentSignature(experiment plans.PlannedExperiment) string {
	augmentationBlob, _ := json.Marshal(experiment.Augmentation)
	augmentationPolicyConfigBlob, _ := json.Marshal(experiment.AugmentationPolicyConfig)
	classBalancingConfigBlob, _ := json.Marshal(experiment.ClassBalancingConfig)
	preprocessingBlob, _ := json.Marshal(experiment.Preprocessing)
	return strings.Join([]string{
		strings.ToLower(strings.TrimSpace(experiment.Template)),
		strings.ToLower(strings.TrimSpace(experiment.Model)),
		strconv.Itoa(experiment.Epochs),
		strconv.Itoa(experiment.BatchSize),
		strconv.FormatFloat(experiment.LearningRate, 'g', -1, 64),
		strconv.Itoa(experiment.ImageSize),
		strings.ToLower(strings.TrimSpace(experiment.ResolutionStrategy)),
		string(preprocessingBlob),
		strings.ToLower(strings.TrimSpace(experiment.Optimizer)),
		strings.ToLower(strings.TrimSpace(experiment.Scheduler)),
		strconv.FormatFloat(experiment.WeightDecay, 'g', -1, 64),
		strconv.FormatFloat(experiment.Dropout, 'g', -1, 64),
		strconv.FormatFloat(experiment.OptimizerMomentum, 'g', -1, 64),
		strconv.Itoa(experiment.SchedulerStepSize),
		strconv.FormatFloat(experiment.SchedulerGamma, 'g', -1, 64),
		strconv.FormatFloat(experiment.LabelSmoothing, 'g', -1, 64),
		strconv.FormatFloat(experiment.GradientClipNorm, 'g', -1, 64),
		string(augmentationBlob),
		strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy)),
		string(augmentationPolicyConfigBlob),
		strings.ToLower(strings.TrimSpace(experiment.ClassBalancing)),
		string(classBalancingConfigBlob),
		strings.ToLower(strings.TrimSpace(experiment.SamplingStrategy)),
		strconv.Itoa(experiment.EarlyStoppingPatience),
		strconv.FormatBool(experiment.Pretrained),
		strconv.FormatBool(experiment.FreezeBackbone),
		strings.ToLower(strings.TrimSpace(experiment.FineTuneStrategy)),
	}, ":")
}

func stringSliceContainsFold(values []string, target string) bool {
	target = strings.ToLower(strings.TrimSpace(target))
	for _, value := range values {
		if strings.ToLower(strings.TrimSpace(value)) == target {
			return true
		}
	}
	return false
}

func clampCandidate(value float64, minValue float64, maxValue float64) float64 {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func roundCandidateScore(value float64) float64 {
	return math.Round(value*1000) / 1000
}

func validReflectField(value reflect.Value, fieldName string) (reflect.Value, bool) {
	field := reflectField(value, fieldName)
	if !field.IsValid() {
		return reflect.Value{}, false
	}
	return field, true
}

func reflectField(value reflect.Value, fieldName string) reflect.Value {
	value = reflectIndirect(value)
	if !value.IsValid() {
		return reflect.Value{}
	}
	switch value.Kind() {
	case reflect.Struct:
		field := value.FieldByName(fieldName)
		if field.IsValid() {
			return reflectIndirect(field)
		}
	case reflect.Map:
		if value.Type().Key().Kind() != reflect.String {
			return reflect.Value{}
		}
		for _, key := range value.MapKeys() {
			if strings.EqualFold(key.String(), fieldName) || strings.EqualFold(strings.ReplaceAll(key.String(), "_", ""), strings.ReplaceAll(fieldName, "_", "")) {
				return reflectIndirect(value.MapIndex(key))
			}
		}
	}
	return reflect.Value{}
}

func reflectIndirect(value reflect.Value) reflect.Value {
	for value.IsValid() && (value.Kind() == reflect.Pointer || value.Kind() == reflect.Interface) {
		if value.IsNil() {
			return reflect.Value{}
		}
		value = value.Elem()
	}
	return value
}

func stringsFromReflectField(value reflect.Value, fieldName string) []string {
	field := reflectField(value, fieldName)
	if !field.IsValid() {
		return nil
	}
	switch field.Kind() {
	case reflect.String:
		if strings.TrimSpace(field.String()) == "" {
			return nil
		}
		return []string{field.String()}
	case reflect.Slice, reflect.Array:
		values := []string{}
		for index := 0; index < field.Len(); index++ {
			item := reflectIndirect(field.Index(index))
			if item.IsValid() && item.Kind() == reflect.String && strings.TrimSpace(item.String()) != "" {
				values = append(values, item.String())
			}
		}
		return values
	default:
		return nil
	}
}

func reflectedStringField(value reflect.Value, fieldName string) string {
	field := reflectField(value, fieldName)
	if !field.IsValid() {
		return ""
	}
	switch field.Kind() {
	case reflect.String:
		return strings.TrimSpace(field.String())
	default:
		return strings.TrimSpace(fmt.Sprint(field.Interface()))
	}
}

func reflectedBoolField(value reflect.Value, fieldName string) bool {
	field := reflectField(value, fieldName)
	if !field.IsValid() || field.Kind() != reflect.Bool {
		return false
	}
	return field.Bool()
}

func reflectedIntField(value reflect.Value, fieldName string) int {
	field := reflectField(value, fieldName)
	if !field.IsValid() {
		return 0
	}
	switch field.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return int(field.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return int(field.Uint())
	case reflect.Float32, reflect.Float64:
		return int(field.Float())
	default:
		return 0
	}
}

func candidateMaxInt(values ...int) int {
	maxValue := 0
	for _, value := range values {
		if value > maxValue {
			maxValue = value
		}
	}
	return maxValue
}
