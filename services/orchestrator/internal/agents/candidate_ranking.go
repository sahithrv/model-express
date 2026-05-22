package agents

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"model-express/services/orchestrator/internal/plans"
)

func RankPlannerCandidateHypotheses(input ExperimentPlannerInput, candidates []CandidateHypothesis, maxExperiments int) ([]CandidateRanking, []plans.PlannedExperiment) {
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
		selected = append(selected, experiment)
		selectedFamilyCounts[family]++
	}

	for index := range rankings {
		if selectedIndexes[rankings[index].CandidateIndex] {
			rankings[index].Selected = true
			rankings[index].Reasons = append(rankings[index].Reasons, "selected by deterministic backend ranking")
		}
	}
	return rankings, selected
}

func scorePlannerCandidate(input ExperimentPlannerInput, candidate CandidateHypothesis, index int, existing map[string]bool, seenProposed map[string]bool) CandidateRanking {
	experiment := candidate.ExperimentConfig
	signature := candidateExperimentSignature(experiment)
	ranking := CandidateRanking{
		CandidateIndex:      index,
		Hypothesis:          candidate.Hypothesis,
		PlanningMode:        candidate.PlanningMode,
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

	memoryBonus, memoryReasons := candidateMemoryScore(input, candidate, experiment)
	ranking.Score += memoryBonus
	ranking.Reasons = append(ranking.Reasons, memoryReasons...)
	ranking.ScoreComponents["memory_similarity"] = roundCandidateScore(memoryBonus)

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
		experiment.Reason,
		experiment.Strategy,
		experiment.ClassBalancing,
		experiment.SamplingStrategy,
		experiment.ResolutionStrategy,
		experiment.AugmentationPolicy,
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

func candidateMemoryScore(input ExperimentPlannerInput, candidate CandidateHypothesis, experiment plans.PlannedExperiment) (float64, []string) {
	bonus := 0.0
	reasons := []string{}
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
	return bonus, reasons
}

func scorecardSimilarToCandidate(scorecard PlannerStrategyScorecard, candidate CandidateHypothesis, experiment plans.PlannedExperiment) bool {
	if scorecard.PlanningMode != "" && strings.EqualFold(scorecard.PlanningMode, candidate.PlanningMode) {
		return true
	}
	changesBlob, _ := json.Marshal(scorecard.ProposedChanges)
	changes := strings.ToLower(string(changesBlob))
	model := strings.ToLower(strings.TrimSpace(experiment.Model))
	family := inferExperimentFamily(experiment.Model)
	return strings.Contains(changes, model) || (family != "" && strings.Contains(changes, family))
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
		return 0.06, "candidate fits low-latency live objective"
	}
	if expectedGain < 0.035 {
		return -0.22, "candidate is weakly aligned with the project objective"
	}
	return -0.08, "candidate must justify heavier deployment profile"
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
		case "model", "model_family", "architecture", "image_size", "resolution_strategy", "augmentation", "augmentation_policy", "preprocessing", "resize_strategy", "normalization", "crop", "crop_strategy", "bbox_mode", "class_balancing", "sampling_strategy", "fine_tune_strategy", "scheduler", "optimizer", "weight_decay", "loss":
			return true
		}
	}
	text := strings.ToLower(strings.Join([]string{
		candidate.Hypothesis,
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
		string(augmentationBlob),
		strings.ToLower(strings.TrimSpace(experiment.AugmentationPolicy)),
		strings.ToLower(strings.TrimSpace(experiment.ClassBalancing)),
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
