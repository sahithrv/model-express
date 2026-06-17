package api

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

const (
	catastrophicRunScoreFloor            = 0.001
	catastrophicTargetMetricMax          = 0.200
	catastrophicRandomLossRatioThreshold = 0.90
	catastrophicAbsoluteLossThreshold    = 2.00
)

func experimentPlannerPerformanceContext(
	targetMetric string,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
	sourcePlanID string,
) (*agents.ExperimentChampion, *agents.ExperimentChampion, []agents.ExperimentRunDelta, int, []string) {
	championSummary, hasChampion := bestSuccessfulTrainingSummaryForObjective(targetMetric, summaries, evaluations, objectiveContext)
	if !hasChampion {
		return nil, nil, []agents.ExperimentRunDelta{}, 0, []string{"No successful champion run is available yet."}
	}

	evaluationsByJob := evaluationsByJobID(evaluations)
	champion := experimentChampionFromSummaryWithEvaluation(targetMetric, championSummary, evaluationsByJob[championSummary.JobID], objectiveContext)
	baselineChampion := champion
	if baselineSummary, ok := bestSuccessfulTrainingSummaryBeforePlanForObjective(targetMetric, projectPlans, summaries, evaluations, objectiveContext, sourcePlanID); ok {
		baselineChampion = experimentChampionFromSummaryWithEvaluation(targetMetric, baselineSummary, evaluationsByJob[baselineSummary.JobID], objectiveContext)
	}
	sourcePlanDeltas := experimentRunDeltasForPlan(targetMetric, summariesForPlanID(summaries, sourcePlanID), evaluationsByJob, baselineChampion, objectiveContext)
	noImprovementRounds := consecutiveNoImprovementFollowUpRounds(targetMetric, projectPlans, summaries, evaluations, objectiveContext)
	stopSignals := experimentPlannerStopSignals(champion, noImprovementRounds)
	return &champion, &baselineChampion, sourcePlanDeltas, noImprovementRounds, stopSignals
}

func experimentChampionFromSummary(targetMetric string, summary runs.TrainingRunSummary) agents.ExperimentChampion {
	return experimentChampionFromSummaryWithEvaluation(targetMetric, summary, runs.TrainingRunEvaluation{}, agents.ProjectObjectiveContext{})
}

func experimentChampionFromSummaryWithEvaluation(
	targetMetric string,
	summary runs.TrainingRunSummary,
	evaluation runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
) agents.ExperimentChampion {
	score := holisticRunScore(targetMetric, summary, evaluation, objectiveContext)
	return agents.ExperimentChampion{
		JobID:            summary.JobID,
		PlanID:           summary.PlanID,
		Model:            summary.Model,
		TargetMetric:     "deployment_readiness",
		Score:            roundDiagnosticFloat(score),
		ScoreBasis:       "loss_heavy_deployment_readiness",
		BestMacroF1:      summary.BestMacroF1,
		BestAccuracy:     summary.BestAccuracy,
		FinalTrainLoss:   summary.FinalTrainLoss,
		FinalValLoss:     summary.FinalValLoss,
		EstimatedCostUSD: summary.EstimatedCostUSD,
		RuntimeSeconds:   summary.RuntimeSeconds,
		EpochsCompleted:  summary.EpochsCompleted,
	}
}

func experimentRunDeltasForPlan(
	targetMetric string,
	summaries []runs.TrainingRunSummary,
	evaluationsByJob map[string]runs.TrainingRunEvaluation,
	champion agents.ExperimentChampion,
	objectiveContext agents.ProjectObjectiveContext,
) []agents.ExperimentRunDelta {
	out := make([]agents.ExperimentRunDelta, 0, len(summaries))
	for _, summary := range summaries {
		score := holisticRunScore(targetMetric, summary, evaluationsByJob[summary.JobID], objectiveContext)
		out = append(out, agents.ExperimentRunDelta{
			JobID:                    summary.JobID,
			PlanID:                   summary.PlanID,
			Model:                    summary.Model,
			Status:                   summary.Status,
			TargetMetric:             "deployment_readiness",
			Score:                    roundDiagnosticFloat(score),
			ScoreBasis:               "loss_heavy_deployment_readiness",
			BestMacroF1:              summary.BestMacroF1,
			BestAccuracy:             summary.BestAccuracy,
			FinalTrainLoss:           summary.FinalTrainLoss,
			FinalValLoss:             summary.FinalValLoss,
			EstimatedCostUSD:         summary.EstimatedCostUSD,
			RuntimeSeconds:           summary.RuntimeSeconds,
			EpochsCompleted:          summary.EpochsCompleted,
			ChampionJobID:            champion.JobID,
			DeltaScoreVsChampion:     score - champion.Score,
			DeltaCostVsChampion:      summary.EstimatedCostUSD - champion.EstimatedCostUSD,
			DeltaRuntimeVsChampion:   summary.RuntimeSeconds - champion.RuntimeSeconds,
			MeaningfullyImprovedOver: score > champion.Score+plannerMinimumMeaningfulImprovement,
		})
	}
	return out
}

func consecutiveNoImprovementFollowUpRounds(
	targetMetric string,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
) int {
	orderedPlans := append([]plans.ExperimentPlan(nil), projectPlans...)
	sort.Slice(orderedPlans, func(i, j int) bool {
		if orderedPlans[i].CreatedAt.Equal(orderedPlans[j].CreatedAt) {
			return orderedPlans[i].ID < orderedPlans[j].ID
		}
		return orderedPlans[i].CreatedAt.Before(orderedPlans[j].CreatedAt)
	})

	hasChampion := false
	championScore := 0.0
	noImprovementRounds := 0
	evaluationsByJob := evaluationsByJobID(evaluations)
	for _, plan := range orderedPlans {
		planSummaries := summariesForPlanID(summaries, plan.ID)
		if !planTrainingRunsComplete(plan, planSummaries) {
			continue
		}

		best, ok := bestSuccessfulTrainingSummaryForObjective(targetMetric, planSummaries, evaluationsForPlanID(evaluations, plan.ID), objectiveContext)
		if !ok {
			if plan.SourceDecisionID != "" && hasChampion {
				noImprovementRounds++
			}
			continue
		}

		score := holisticRunScore(targetMetric, best, evaluationsByJob[best.JobID], objectiveContext)
		if !hasChampion {
			hasChampion = true
			championScore = score
			continue
		}

		if plan.SourceDecisionID != "" {
			if score > championScore+plannerMinimumMeaningfulImprovement {
				noImprovementRounds = 0
			} else {
				noImprovementRounds++
			}
		}
		if score > championScore {
			championScore = score
		}
	}
	return noImprovementRounds
}

func experimentPlannerStopSignals(champion agents.ExperimentChampion, noImprovementRounds int) []string {
	signals := []string{
		fmt.Sprintf("Current champion is %s (%s) with %s %.3f.", champion.JobID, champion.Model, champion.TargetMetric, champion.Score),
	}
	if reason, ok := nearMetricCeilingChampionStopReason(agents.ExperimentPlannerInput{
		CurrentChampion:              &champion,
		MinimumMeaningfulImprovement: plannerMinimumMeaningfulImprovement,
	}); ok {
		signals = append(signals, reason)
	}
	if noImprovementRounds > 0 {
		signals = append(signals, fmt.Sprintf("%d consecutive completed follow-up plan(s) did not improve the champion by at least %.3f.", noImprovementRounds, plannerMinimumMeaningfulImprovement))
	}
	if noImprovementRounds >= plannerNoImprovementRoundsToSelect {
		if terminalPlannerGuardsEnabled() {
			signals = append(signals, "Backend policy will select the current champion instead of scheduling another follow-up unless a future run meaningfully improves it.")
		} else {
			signals = append(signals, "No-improvement rounds are advisory only; continue by pivoting to a substantive backend-valid mechanism rather than stopping solely because the current champion remains ahead.")
		}
	}
	return signals
}

func bestSuccessfulTrainingSummary(targetMetric string, summaries []runs.TrainingRunSummary) (runs.TrainingRunSummary, bool) {
	var best runs.TrainingRunSummary
	hasBest := false
	bestScore := 0.0
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) != jobs.StatusSucceeded {
			continue
		}
		score := holisticRunScore(targetMetric, summary, runs.TrainingRunEvaluation{}, agents.ProjectObjectiveContext{})
		if !hasBest || score > bestScore || (score == bestScore && summary.EstimatedCostUSD < best.EstimatedCostUSD) {
			best = summary
			bestScore = score
			hasBest = true
		}
	}
	return best, hasBest
}

func bestSuccessfulTrainingSummaryForObjective(
	targetMetric string,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
) (runs.TrainingRunSummary, bool) {
	if len(evaluations) == 0 {
		return bestSuccessfulTrainingSummary(targetMetric, summaries)
	}
	evaluationsByJob := map[string]runs.TrainingRunEvaluation{}
	for _, evaluation := range evaluations {
		evaluationsByJob[evaluation.JobID] = evaluation
	}

	var best runs.TrainingRunSummary
	bestScore := 0.0
	hasBest := false
	for _, summary := range summaries {
		if strings.ToUpper(strings.TrimSpace(summary.Status)) != jobs.StatusSucceeded {
			continue
		}
		score := holisticRunScore(targetMetric, summary, evaluationsByJob[summary.JobID], objectiveContext)
		if !hasBest || score > bestScore || (score == bestScore && summary.EstimatedCostUSD < best.EstimatedCostUSD) {
			best = summary
			bestScore = score
			hasBest = true
		}
	}
	return best, hasBest
}

func holisticRunScore(targetMetric string, summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation, objectiveContext agents.ProjectObjectiveContext) float64 {
	score, _ := holisticRunScoreBreakdown(targetMetric, summary, evaluation, objectiveContext)
	return score
}

func holisticRunScoreBreakdown(targetMetric string, summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation, objectiveContext agents.ProjectObjectiveContext) (float64, map[string]any) {
	quality, readinessBreakdown := deploymentReadinessScoreBreakdown(targetMetric, summary, evaluation)
	latencyScore, costScore, runtimeScore := deploymentUtilityScores(summary, evaluation)
	score := quality*0.88 + latencyScore*0.06 + costScore*0.03 + runtimeScore*0.03
	weights := map[string]float64{
		"quality": 0.88,
		"latency": 0.06,
		"cost":    0.03,
		"runtime": 0.03,
	}
	if objectiveContext.PrimaryObjective == "low_latency_live_service" {
		score = quality*0.82 + latencyScore*0.10 + costScore*0.05 + runtimeScore*0.03
		weights = map[string]float64{"quality": 0.82, "latency": 0.10, "cost": 0.05, "runtime": 0.03}
	} else if objectiveContext.PrimaryObjective == "budget_sensitive" {
		score = quality*0.80 + costScore*0.12 + latencyScore*0.05 + runtimeScore*0.03
		weights = map[string]float64{"quality": 0.80, "latency": 0.05, "cost": 0.12, "runtime": 0.03}
	} else if objectiveContext.PrimaryObjective == "quality_first" {
		score = quality*0.92 + latencyScore*0.03 + costScore*0.025 + runtimeScore*0.025
		weights = map[string]float64{"quality": 0.92, "latency": 0.03, "cost": 0.025, "runtime": 0.025}
	}
	if payloadBool(readinessBreakdown, "catastrophic_quality_gate") {
		score = catastrophicRunScoreFloor
	}
	breakdown := map[string]any{
		"score":                   roundDiagnosticFloat(score),
		"quality_score":           roundDiagnosticFloat(quality),
		"latency_score":           roundDiagnosticFloat(latencyScore),
		"cost_score":              roundDiagnosticFloat(costScore),
		"runtime_score":           roundDiagnosticFloat(runtimeScore),
		"objective":               objectiveContext.PrimaryObjective,
		"target_metric":           normalizedPlannerTargetMetric(targetMetric),
		"objective_weights":       weights,
		"readiness_components":    readinessBreakdown,
		"validation_metric_score": payloadFloat(readinessBreakdown, "validation_metric_score"),
		"heldout_metric_score":    payloadFloat(readinessBreakdown, "heldout_metric_score"),
		"per_class_score":         payloadFloat(readinessBreakdown, "per_class_score"),
		"loss_health_score":       payloadFloat(readinessBreakdown, "loss_health_score"),
		"confusion_health_score":  payloadFloat(readinessBreakdown, "confusion_health_score"),
	}
	if payloadBool(readinessBreakdown, "catastrophic_quality_gate") {
		breakdown["catastrophic_quality_gate"] = true
		breakdown["catastrophic_quality_reason"] = readinessBreakdown["catastrophic_quality_reason"]
		breakdown["score_cap"] = catastrophicRunScoreFloor
	}
	return clamp01(score), breakdown
}

func deploymentUtilityScores(summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation) (float64, float64, float64) {
	latencyScore := 0.5
	if latencyMS := payloadFloat(evaluation.ModelProfile, "estimated_pipeline_latency_ms"); latencyMS > 0 {
		latencyScore = maxFloat(0, minFloat(1, 1-latencyMS/160))
	} else if latencyMS := payloadFloat(evaluation.ModelProfile, "estimated_latency_ms"); latencyMS > 0 {
		latencyScore = maxFloat(0, minFloat(1, 1-latencyMS/160))
	}
	costScore := maxFloat(0, minFloat(1, 1-summary.EstimatedCostUSD/10))
	runtimeScore := maxFloat(0, minFloat(1, 1-summary.RuntimeSeconds/1800))
	return latencyScore, costScore, runtimeScore
}

func deploymentReadinessScore(targetMetric string, summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation) float64 {
	score, _ := deploymentReadinessScoreBreakdown(targetMetric, summary, evaluation)
	return score
}

func deploymentReadinessScoreBreakdown(targetMetric string, summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation) (float64, map[string]any) {
	validationScore := validationMetricScore(targetMetric, summary, evaluation)
	heldoutScore, hasHeldoutScore := heldoutMetricScore(targetMetric, evaluation.ObjectiveProfile)
	perClassScore, hasPerClassScore := perClassMetricScore(evaluation.PerClassMetrics)
	lossScore, hasLossScore := lossHealthScore(summary, evaluation)
	confusionScore, hasConfusionScore := confusionHealthScore(evaluation.ConfusionMatrix)
	overallScore := payloadFloat(evaluation.HolisticScores, "overall_score")

	weighted := weightedScore{}
	weighted.add(validationScore, 0.12, true)
	weighted.add(heldoutScore, 0.24, hasHeldoutScore)
	weighted.add(perClassScore, 0.17, hasPerClassScore)
	weighted.add(lossScore, 0.34, hasLossScore)
	weighted.add(confusionScore, 0.09, hasConfusionScore)
	weighted.add(overallScore, 0.04, overallScore > 0)
	score := weighted.value(validationScore)
	breakdown := map[string]any{
		"score":                   roundDiagnosticFloat(score),
		"validation_metric_score": roundDiagnosticFloat(validationScore),
		"heldout_metric_score":    roundDiagnosticFloat(heldoutScore),
		"has_heldout_metric":      hasHeldoutScore,
		"per_class_score":         roundDiagnosticFloat(perClassScore),
		"has_per_class_metrics":   hasPerClassScore,
		"loss_health_score":       roundDiagnosticFloat(lossScore),
		"has_loss_health":         hasLossScore,
		"confusion_health_score":  roundDiagnosticFloat(confusionScore),
		"has_confusion_matrix":    hasConfusionScore,
		"overall_score":           roundDiagnosticFloat(overallScore),
		"component_weights": map[string]float64{
			"validation_metric": 0.12,
			"heldout_metric":    0.24,
			"per_class":         0.17,
			"loss_health":       0.34,
			"confusion_health":  0.09,
			"overall":           0.04,
		},
	}
	if capped, details := catastrophicClassificationQuality(targetMetric, summary, evaluation, validationScore, hasLossScore); capped {
		score = catastrophicRunScoreFloor
		breakdown["score"] = roundDiagnosticFloat(score)
		breakdown["catastrophic_quality_gate"] = true
		breakdown["catastrophic_quality_reason"] = "target classification metric is below the bad-quality threshold"
		breakdown["score_cap"] = catastrophicRunScoreFloor
		for key, value := range details {
			breakdown[key] = value
		}
	}
	return score, breakdown
}

type weightedScore struct {
	total  float64
	weight float64
}

func (s *weightedScore) add(value float64, weight float64, ok bool) {
	if !ok || weight <= 0 || !isFiniteFloat(value) {
		return
	}
	s.total += clamp01(value) * weight
	s.weight += weight
}

func (s weightedScore) value(fallback float64) float64 {
	if s.weight <= 0 {
		return clamp01(fallback)
	}
	return clamp01(s.total / s.weight)
}

func validationMetricScore(targetMetric string, summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation) float64 {
	macroF1 := clamp01(summary.BestMacroF1)
	accuracy := clamp01(summary.BestAccuracy)
	switch normalizedPlannerTargetMetric(targetMetric) {
	case "accuracy":
		return weightedMetricAverage([]metricComponent{{accuracy, 0.55}, {macroF1, 0.45}}, maxFloat(accuracy, macroF1))
	case "map50_95", "map50", "map":
		map50_95 := detectionMetricFromEvaluation(evaluation, "map50_95")
		map50 := detectionMetricFromEvaluation(evaluation, "map50")
		precision := detectionMetricFromEvaluation(evaluation, "precision")
		recall := detectionMetricFromEvaluation(evaluation, "recall")
		if map50_95 > 0 || map50 > 0 {
			return weightedMetricAverage(
				[]metricComponent{{map50_95, 0.66}, {map50, 0.20}, {recall, 0.09}, {precision, 0.05}},
				maxFloat(map50_95, map50),
			)
		}
		return weightedMetricAverage([]metricComponent{{macroF1, 0.65}, {accuracy, 0.35}}, maxFloat(macroF1, accuracy))
	default:
		return weightedMetricAverage([]metricComponent{{macroF1, 0.60}, {accuracy, 0.40}}, maxFloat(macroF1, accuracy))
	}
}

func catastrophicClassificationQuality(
	targetMetric string,
	summary runs.TrainingRunSummary,
	evaluation runs.TrainingRunEvaluation,
	validationScore float64,
	hasLoss bool,
) (bool, map[string]any) {
	switch normalizedPlannerTargetMetric(targetMetric) {
	case "accuracy", "macro_f1":
	default:
		return false, nil
	}
	macroF1 := clamp01(summary.BestMacroF1)
	accuracy := clamp01(summary.BestAccuracy)
	targetScore, targetMetricName := classificationTargetMetricScore(targetMetric, macroF1, accuracy)
	badMetrics := targetScore <= catastrophicTargetMetricMax
	lossValue, lossSource := worstObservedLoss(summary, evaluation)
	lossThreshold := catastrophicLossThreshold(evaluationClassCount(evaluation))
	badLoss := hasLoss && lossValue > 0 && lossValue >= lossThreshold
	details := map[string]any{
		"catastrophic_metric_threshold": roundDiagnosticFloat(catastrophicTargetMetricMax),
		"catastrophic_loss_threshold":   roundDiagnosticFloat(lossThreshold),
		"validation_metric_score":       roundDiagnosticFloat(validationScore),
		"target_metric_name":            targetMetricName,
		"target_metric_score":           roundDiagnosticFloat(targetScore),
		"worst_observed_loss":           roundDiagnosticFloat(lossValue),
		"worst_observed_loss_source":    lossSource,
		"loss_random_or_worse":          badLoss,
	}
	return badMetrics, details
}

func classificationTargetMetricScore(targetMetric string, macroF1 float64, accuracy float64) (float64, string) {
	switch normalizedPlannerTargetMetric(targetMetric) {
	case "accuracy":
		return accuracy, "accuracy"
	default:
		return macroF1, "macro_f1"
	}
}

func worstObservedLoss(summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation) (float64, string) {
	values := []struct {
		value  float64
		source string
	}{
		{summary.FinalValLoss, "final_val_loss"},
		{payloadFloat(evaluation.ObjectiveProfile, "heldout_test_loss"), "heldout_test_loss"},
		{summary.FinalTrainLoss, "final_train_loss"},
	}
	bestValue := 0.0
	bestSource := ""
	for _, candidate := range values {
		if candidate.value <= 0 || !isFiniteFloat(candidate.value) {
			continue
		}
		if candidate.value > bestValue {
			bestValue = candidate.value
			bestSource = candidate.source
		}
	}
	return bestValue, bestSource
}

func catastrophicLossThreshold(classCount int) float64 {
	randomBaseline := math.Log(float64(maxInt(classCount, 2)))
	if randomBaseline <= 0 || !isFiniteFloat(randomBaseline) {
		randomBaseline = math.Log(2)
	}
	return maxFloat(catastrophicAbsoluteLossThreshold, randomBaseline*catastrophicRandomLossRatioThreshold)
}

func detectionMetricFromEvaluation(evaluation runs.TrainingRunEvaluation, metric string) float64 {
	normalized := strings.ToLower(strings.ReplaceAll(strings.TrimSpace(metric), "-", "_"))
	objectiveKeys := map[string][]string{
		"map50_95":  {"heldout_test_map50_95", "heldout_test_map"},
		"map50":     {"heldout_test_map50"},
		"precision": {"heldout_test_precision"},
		"recall":    {"heldout_test_recall"},
	}
	for _, key := range objectiveKeys[normalized] {
		if value := payloadFloat(evaluation.ObjectiveProfile, key); value > 0 {
			return clamp01(value)
		}
	}
	detectionMetrics := payloadMap(evaluation.HolisticScores, "detection_metrics")
	holisticKeys := map[string][]string{
		"map50_95":  {"mAP50_95", "map50_95", "map"},
		"map50":     {"mAP50", "map50"},
		"precision": {"precision"},
		"recall":    {"recall"},
	}
	for _, key := range holisticKeys[normalized] {
		if value := payloadFloat(detectionMetrics, key); value > 0 {
			return clamp01(value)
		}
	}
	return 0
}

func addDetectionChampionMetrics(metrics map[string]any, evaluation runs.TrainingRunEvaluation) {
	map50_95 := detectionMetricFromEvaluation(evaluation, "map50_95")
	map50 := detectionMetricFromEvaluation(evaluation, "map50")
	precision := detectionMetricFromEvaluation(evaluation, "precision")
	recall := detectionMetricFromEvaluation(evaluation, "recall")
	if map50_95 <= 0 && map50 <= 0 {
		return
	}
	if map50_95 > 0 {
		metrics["best_map50_95"] = map50_95
		metrics["primary_metric_value"] = map50_95
		metrics["primary_metric_label"] = "mAP50-95"
	}
	if map50 > 0 {
		metrics["best_map50"] = map50
	}
	if precision > 0 {
		metrics["best_precision"] = precision
	}
	if recall > 0 {
		metrics["best_recall"] = recall
	}
	metrics["target_metric"] = "mAP50_95"
	metrics["task_type"] = "object_detection"
}

func heldoutMetricScore(targetMetric string, objectiveProfile map[string]any) (float64, bool) {
	map50_95 := payloadFloat(objectiveProfile, "heldout_test_map50_95")
	if map50_95 <= 0 {
		map50_95 = payloadFloat(objectiveProfile, "heldout_test_map")
	}
	map50 := payloadFloat(objectiveProfile, "heldout_test_map50")
	recall := payloadFloat(objectiveProfile, "heldout_test_recall")
	precision := payloadFloat(objectiveProfile, "heldout_test_precision")
	if map50_95 > 0 || map50 > 0 {
		weighted := weightedScore{}
		weighted.add(map50_95, 0.62, map50_95 > 0)
		weighted.add(map50, 0.20, map50 > 0)
		weighted.add(recall, 0.12, recall > 0)
		weighted.add(precision, 0.06, precision > 0)
		return weighted.value(maxFloat(map50_95, map50)), true
	}
	macroF1 := payloadFloat(objectiveProfile, "heldout_test_macro_f1")
	accuracy := payloadFloat(objectiveProfile, "heldout_test_accuracy")
	if macroF1 <= 0 && accuracy <= 0 {
		return 0, false
	}
	switch normalizedPlannerTargetMetric(targetMetric) {
	case "accuracy":
		return weightedMetricAverage([]metricComponent{{accuracy, 0.55}, {macroF1, 0.45}}, maxFloat(accuracy, macroF1)), true
	default:
		return weightedMetricAverage([]metricComponent{{macroF1, 0.60}, {accuracy, 0.40}}, maxFloat(macroF1, accuracy)), true
	}
}

func perClassMetricScore(metrics map[string]any) (float64, bool) {
	if score, ok := detectionPerClassMetricScore(metrics); ok {
		return score, true
	}
	worstLabel, worstRecall := worstPerClassRecall(metrics)
	macroAvg := perClassAggregateMetric(metrics, "macro avg")
	weightedAvg := perClassAggregateMetric(metrics, "weighted avg")
	if worstLabel == "" && macroAvg <= 0 && weightedAvg <= 0 {
		return 0, false
	}
	weighted := weightedScore{}
	weighted.add(worstRecall, 0.50, worstLabel != "")
	weighted.add(macroAvg, 0.35, macroAvg > 0)
	weighted.add(weightedAvg, 0.15, weightedAvg > 0)
	return weighted.value(maxFloat(worstRecall, macroAvg)), true
}

func detectionPerClassMetricScore(metrics map[string]any) (float64, bool) {
	worst := 1.0
	total := 0.0
	count := 0
	macroAvg := 0.0
	for label, rawStats := range metrics {
		normalizedLabel := strings.ToLower(strings.TrimSpace(label))
		stats := mapFromAny(rawStats)
		quality := detectionPerClassQuality(stats)
		if normalizedLabel == "macro avg" {
			macroAvg = quality
			continue
		}
		if normalizedLabel == "" || normalizedLabel == "accuracy" || strings.Contains(normalizedLabel, "avg") || quality <= 0 {
			continue
		}
		if quality < worst {
			worst = quality
		}
		total += quality
		count++
	}
	if count == 0 && macroAvg <= 0 {
		return 0, false
	}
	mean := macroAvg
	if mean <= 0 && count > 0 {
		mean = total / float64(count)
	}
	weighted := weightedScore{}
	weighted.add(worst, 0.45, count > 0)
	weighted.add(mean, 0.55, mean > 0)
	return weighted.value(maxFloat(worst, mean)), true
}

func detectionPerClassQuality(stats map[string]any) float64 {
	map50_95 := firstPayloadFloat(stats, "AP50_95", "mAP50_95", "map50_95", "map")
	map50 := firstPayloadFloat(stats, "AP50", "mAP50", "map50")
	recall := firstPayloadFloat(stats, "recall")
	precision := firstPayloadFloat(stats, "precision")
	if map50_95 <= 0 && map50 <= 0 && recall <= 0 && precision <= 0 {
		return 0
	}
	return weightedMetricAverage(
		[]metricComponent{{map50_95, 0.66}, {map50, 0.20}, {recall, 0.09}, {precision, 0.05}},
		maxFloat(map50_95, map50),
	)
}

func perClassAggregateMetric(metrics map[string]any, key string) float64 {
	stats := mapFromAny(metrics[key])
	value := payloadFloat(stats, "f1-score")
	if value <= 0 {
		value = payloadFloat(stats, "f1")
	}
	if value <= 0 {
		value = payloadFloat(stats, "recall")
	}
	return clamp01(value)
}

func firstPayloadFloat(payload map[string]any, keys ...string) float64 {
	for _, key := range keys {
		if value := payloadFloat(payload, key); value > 0 {
			return value
		}
	}
	return 0
}

func lossHealthScore(summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation) (float64, bool) {
	classCount := evaluationClassCount(evaluation)
	diagnostics := payloadMap(evaluation.HolisticScores, "training_diagnostics")
	weighted := weightedScore{}
	if severity := payloadFloat(diagnostics, "severity"); severity > 0 {
		weighted.add(1-severity, 0.18, true)
	}
	if payloadBool(diagnostics, "divergence_detected") {
		weighted.add(0.10, 0.15, true)
	}
	trainLoss := summary.FinalTrainLoss
	valLoss := summary.FinalValLoss
	if trainLoss > 0 {
		weighted.add(lossValueScore(trainLoss, classCount), 0.12, true)
	}
	if valLoss > 0 {
		weighted.add(lossValueScore(valLoss, classCount), 0.30, true)
	}
	if trainLoss > 0 && valLoss > 0 {
		gap := valLoss - trainLoss
		ratio := valLoss / maxFloat(trainLoss, 0.000001)
		weighted.add(1-minFloat(1, maxFloat(0, gap)/0.75), 0.15, true)
		weighted.add(1-minFloat(1, maxFloat(0, ratio-1)/1.50), 0.10, true)
	}
	if heldoutLoss := payloadFloat(evaluation.ObjectiveProfile, "heldout_test_loss"); heldoutLoss > 0 {
		weighted.add(lossValueScore(heldoutLoss, classCount), 0.35, true)
	}
	if weighted.weight <= 0 {
		return 0, false
	}
	return weighted.value(1), true
}

func lossValueScore(loss float64, classCount int) float64 {
	if loss <= 0 || !isFiniteFloat(loss) {
		return 0
	}
	randomBaseline := math.Log(float64(maxInt(classCount, 2)))
	if randomBaseline <= 0 {
		randomBaseline = math.Log(2)
	}
	normalized := loss / randomBaseline
	return clamp01(1 - (normalized-0.20)/1.30)
}

func confusionHealthScore(matrix [][]int) (float64, bool) {
	if len(matrix) == 0 {
		return 0, false
	}
	_, ratio := topConfusionPairRatio(matrix)
	if ratio <= 0 {
		return 1, true
	}
	return clamp01(1 - ratio/0.25), true
}

func evaluationClassCount(evaluation runs.TrainingRunEvaluation) int {
	if len(evaluation.ConfusionMatrix) > 0 {
		return len(evaluation.ConfusionMatrix)
	}
	count := 0
	for label := range evaluation.PerClassMetrics {
		normalizedLabel := strings.ToLower(strings.TrimSpace(label))
		if normalizedLabel == "" || normalizedLabel == "accuracy" || strings.Contains(normalizedLabel, "avg") {
			continue
		}
		count++
	}
	if count > 0 {
		return count
	}
	if labels := payloadStringSlice(evaluation.ModelProfile, "class_labels"); len(labels) > 0 {
		return len(labels)
	}
	return 2
}

type metricComponent struct {
	value  float64
	weight float64
}

func weightedMetricAverage(components []metricComponent, fallback float64) float64 {
	weighted := weightedScore{}
	for _, component := range components {
		weighted.add(component.value, component.weight, component.value > 0)
	}
	return weighted.value(fallback)
}

func clamp01(value float64) float64 {
	if !isFiniteFloat(value) {
		return 0
	}
	return maxFloat(0, minFloat(1, value))
}

func bestSuccessfulTrainingSummaryBeforePlan(
	targetMetric string,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	sourcePlanID string,
) (runs.TrainingRunSummary, bool) {
	return bestSuccessfulTrainingSummaryBeforePlanForObjective(targetMetric, projectPlans, summaries, nil, agents.ProjectObjectiveContext{}, sourcePlanID)
}

func bestSuccessfulTrainingSummaryBeforePlanForObjective(
	targetMetric string,
	projectPlans []plans.ExperimentPlan,
	summaries []runs.TrainingRunSummary,
	evaluations []runs.TrainingRunEvaluation,
	objectiveContext agents.ProjectObjectiveContext,
	sourcePlanID string,
) (runs.TrainingRunSummary, bool) {
	orderedPlans := append([]plans.ExperimentPlan(nil), projectPlans...)
	sort.Slice(orderedPlans, func(i, j int) bool {
		if orderedPlans[i].CreatedAt.Equal(orderedPlans[j].CreatedAt) {
			return orderedPlans[i].ID < orderedPlans[j].ID
		}
		return orderedPlans[i].CreatedAt.Before(orderedPlans[j].CreatedAt)
	})

	priorPlanIDs := map[string]bool{}
	for _, plan := range orderedPlans {
		if plan.ID == sourcePlanID {
			break
		}
		priorPlanIDs[plan.ID] = true
	}

	priorSummaries := []runs.TrainingRunSummary{}
	for _, summary := range summaries {
		if priorPlanIDs[summary.PlanID] {
			priorSummaries = append(priorSummaries, summary)
		}
	}
	return bestSuccessfulTrainingSummaryForObjective(targetMetric, priorSummaries, evaluations, objectiveContext)
}

func plannerTargetMetricValue(targetMetric string, summary runs.TrainingRunSummary, evaluation runs.TrainingRunEvaluation) float64 {
	switch normalizedPlannerTargetMetric(targetMetric) {
	case "accuracy":
		return summary.BestAccuracy
	case "map50_95", "map50", "map":
		if value := detectionMetricFromEvaluation(evaluation, "map50_95"); value > 0 {
			return value
		}
		if value := detectionMetricFromEvaluation(evaluation, "map50"); value > 0 {
			return value
		}
		return summary.BestMacroF1
	default:
		return summary.BestMacroF1
	}
}

func normalizedPlannerTargetMetric(targetMetric string) string {
	normalized := strings.ToLower(strings.TrimSpace(targetMetric))
	if normalized == "" {
		return "macro_f1"
	}
	normalized = strings.ReplaceAll(normalized, "-", "_")
	normalized = strings.ReplaceAll(normalized, "@", "")
	if normalized == "map50_95" || normalized == "map5095" || normalized == "map" {
		return "map50_95"
	}
	return normalized
}
