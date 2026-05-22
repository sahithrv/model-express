package agents

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
)

type PlannerDiagnosis struct {
	OverfittingScore           float64  `json:"overfitting_score"`
	UnderfittingScore          float64  `json:"underfitting_score"`
	PlateauScore               float64  `json:"plateau_score"`
	InstabilityScore           float64  `json:"instability_score"`
	ClassImbalanceScore        float64  `json:"class_imbalance_score"`
	MinorityClassFailureScore  float64  `json:"minority_class_failure_score"`
	GeneralizationGap          float64  `json:"generalization_gap"`
	BestMetricDeltaVsChampion  float64  `json:"best_metric_delta_vs_champion"`
	CostEfficiencyScore        float64  `json:"cost_efficiency_score"`
	LatencyPenalty             float64  `json:"latency_penalty"`
	ImprovementStagnationScore float64  `json:"improvement_stagnation_score"`
	RecommendedFailureModes    []string `json:"recommended_failure_modes"`
	DeterministicDiagnosisUsed []string `json:"deterministic_diagnosis_used"`
	Evidence                   []string `json:"evidence"`
}

func ComputePlannerDiagnosis(input ExperimentPlannerInput) PlannerDiagnosis {
	targetMetric := normalizedDiagnosisMetric(input.SourcePlan.TargetMetric)
	bestScore := 0.0
	hasBest := false
	totalCost := 0.0
	totalRuntime := 0.0
	overfitScores := []float64{}
	underfitScores := []float64{}
	gaps := []float64{}
	plateauScores := []float64{}
	instabilityScores := []float64{}

	for _, summary := range input.PlanSummaries {
		score := diagnosisSummaryMetric(summary, targetMetric)
		if strings.EqualFold(summary.Status, jobs.StatusSucceeded) && (!hasBest || score > bestScore) {
			bestScore = score
			hasBest = true
		}
		totalCost += summary.EstimatedCostUSD
		totalRuntime += summary.RuntimeSeconds

		lossGap := summary.FinalValLoss - summary.FinalTrainLoss
		if summary.FinalValLoss > 0 || summary.FinalTrainLoss > 0 {
			gaps = append(gaps, lossGap)
			overfit := clamp01((lossGap - 0.08) / 0.45)
			if summary.FinalTrainLoss > 0 && summary.FinalTrainLoss < 0.35 && summary.FinalValLoss > 0.55 {
				overfit = maxDiagnosis(overfit, 0.75)
			}
			overfitScores = append(overfitScores, overfit)

			underfit := 0.0
			if summary.FinalTrainLoss > 0.65 && summary.FinalValLoss > 0.65 && score < 0.62 {
				underfit = clamp01((0.62 - score) / 0.45)
				underfit = maxDiagnosis(underfit, clamp01((summary.FinalTrainLoss-0.55)/0.65))
			}
			underfitScores = append(underfitScores, underfit)
		}

		if metrics := input.PlanMetrics[summary.JobID]; len(metrics) > 0 {
			plateauScores = append(plateauScores, diagnosisPlateauScore(metrics, targetMetric))
			instabilityScores = append(instabilityScores, diagnosisInstabilityScore(metrics, targetMetric))
		}
	}

	deltaVsChampion := 0.0
	if hasBest {
		if input.CurrentChampion != nil && input.CurrentChampion.JobID != "" {
			deltaVsChampion = bestScore - input.CurrentChampion.Score
		} else {
			deltaVsChampion = bestScore
		}
	}

	classImbalanceScore := clamp01((input.DatasetInsights.ImbalanceRatio - 1.0) / 4.0)
	minorityFailureScore, minorityEvidence := diagnosisMinorityFailure(input.PlanEvaluations)
	if minorityFailureScore == 0 && classImbalanceScore > 0.45 && hasBest {
		for _, summary := range input.PlanSummaries {
			if summary.BestAccuracy-summary.BestMacroF1 > 0.10 {
				minorityFailureScore = maxDiagnosis(minorityFailureScore, clamp01((summary.BestAccuracy-summary.BestMacroF1)/0.30))
			}
		}
	}

	latencyPenalty, latencyEvidence := diagnosisLatencyPenalty(input.PlanEvaluations, input.ObjectiveContext)
	costEfficiency := diagnosisCostEfficiency(deltaVsChampion, totalCost, totalRuntime)
	stagnationScore := maxDiagnosis(clamp01(float64(input.NoImprovementRounds)/3.0), averageScore(plateauScores)*0.85)

	diagnosis := PlannerDiagnosis{
		OverfittingScore:           roundDiagnosis(averageScore(overfitScores)),
		UnderfittingScore:          roundDiagnosis(averageScore(underfitScores)),
		PlateauScore:               roundDiagnosis(averageScore(plateauScores)),
		InstabilityScore:           roundDiagnosis(averageScore(instabilityScores)),
		ClassImbalanceScore:        roundDiagnosis(classImbalanceScore),
		MinorityClassFailureScore:  roundDiagnosis(minorityFailureScore),
		GeneralizationGap:          roundDiagnosis(averageScore(gaps)),
		BestMetricDeltaVsChampion:  roundDiagnosis(deltaVsChampion),
		CostEfficiencyScore:        roundDiagnosis(costEfficiency),
		LatencyPenalty:             roundDiagnosis(latencyPenalty),
		ImprovementStagnationScore: roundDiagnosis(stagnationScore),
	}

	diagnosis.RecommendedFailureModes = diagnosisFailureModes(diagnosis)
	diagnosis.DeterministicDiagnosisUsed = diagnosisSignals(diagnosis)
	diagnosis.Evidence = diagnosisEvidence(input, diagnosis, bestScore, totalCost, totalRuntime, minorityEvidence, latencyEvidence)
	return diagnosis
}

func diagnosisSummaryMetric(summary runs.TrainingRunSummary, targetMetric string) float64 {
	if targetMetric == "accuracy" {
		return summary.BestAccuracy
	}
	return summary.BestMacroF1
}

func diagnosisPlateauScore(metrics []jobs.EpochMetric, targetMetric string) float64 {
	values := orderedMetricValues(metrics, targetMetric)
	if len(values) < 4 {
		return 0
	}
	bestBeforeTail := values[0]
	for _, value := range values[:len(values)-3] {
		if value > bestBeforeTail {
			bestBeforeTail = value
		}
	}
	bestTail := values[len(values)-3]
	for _, value := range values[len(values)-3:] {
		if value > bestTail {
			bestTail = value
		}
	}
	improvement := bestTail - bestBeforeTail
	if improvement >= 0.015 {
		return 0
	}
	return clamp01((0.015 - improvement) / 0.04)
}

func diagnosisInstabilityScore(metrics []jobs.EpochMetric, targetMetric string) float64 {
	values := orderedMetricValues(metrics, targetMetric)
	if len(values) < 4 {
		return 0
	}
	diffs := []float64{}
	signChanges := 0
	lastSign := 0
	for index := 1; index < len(values); index++ {
		diff := values[index] - values[index-1]
		diffs = append(diffs, math.Abs(diff))
		sign := 0
		if diff > 0.005 {
			sign = 1
		} else if diff < -0.005 {
			sign = -1
		}
		if sign != 0 && lastSign != 0 && sign != lastSign {
			signChanges++
		}
		if sign != 0 {
			lastSign = sign
		}
	}
	return clamp01(averageScore(diffs)/0.055 + float64(signChanges)*0.12)
}

func orderedMetricValues(metrics []jobs.EpochMetric, targetMetric string) []float64 {
	ordered := append([]jobs.EpochMetric(nil), metrics...)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].Epoch < ordered[j].Epoch
	})
	values := []float64{}
	for _, metric := range ordered {
		value, ok := metric.Metrics[targetMetric]
		if !ok && targetMetric != "macro_f1" {
			value, ok = metric.Metrics["macro_f1"]
		}
		if !ok && targetMetric != "accuracy" {
			value, ok = metric.Metrics["accuracy"]
		}
		if ok {
			values = append(values, value)
		}
	}
	return values
}

func diagnosisMinorityFailure(evaluations []runs.TrainingRunEvaluation) (float64, string) {
	worstRecall := 1.0
	worstLabel := ""
	found := false
	for _, evaluation := range evaluations {
		for label, raw := range evaluation.PerClassMetrics {
			normalizedLabel := strings.ToLower(strings.TrimSpace(label))
			if normalizedLabel == "" || strings.Contains(normalizedLabel, "avg") || normalizedLabel == "accuracy" {
				continue
			}
			stats, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			recall := diagnosisPayloadFloat(stats, "recall")
			if recall == 0 {
				recall = diagnosisPayloadFloat(stats, "f1-score")
			}
			if recall == 0 {
				recall = diagnosisPayloadFloat(stats, "f1")
			}
			if recall <= 0 {
				continue
			}
			found = true
			if recall < worstRecall {
				worstRecall = recall
				worstLabel = label
			}
		}
	}
	if !found {
		return 0, ""
	}
	score := clamp01((0.68 - worstRecall) / 0.48)
	if score == 0 {
		return 0, ""
	}
	return score, fmt.Sprintf("worst per-class recall/F1 is %.3f for %s", worstRecall, worstLabel)
}

func diagnosisLatencyPenalty(evaluations []runs.TrainingRunEvaluation, objective ProjectObjectiveContext) (float64, string) {
	maxLatency := 0.0
	for _, evaluation := range evaluations {
		latency := firstPositivePayloadFloat(evaluation.ModelProfile, "estimated_latency_ms", "latency_ms", "p50_latency_ms", "inference_latency_ms")
		if latency > maxLatency {
			maxLatency = latency
		}
	}
	if maxLatency <= 0 {
		return 0, ""
	}
	threshold := 160.0
	if objective.PrimaryObjective == "low_latency_live_service" {
		threshold = 80.0
	}
	penalty := clamp01((maxLatency - threshold) / threshold)
	if penalty == 0 {
		return 0, ""
	}
	return penalty, fmt.Sprintf("max estimated latency is %.1fms", maxLatency)
}

func diagnosisCostEfficiency(deltaVsChampion float64, totalCost float64, totalRuntime float64) float64 {
	if totalCost <= 0 && totalRuntime <= 0 {
		if deltaVsChampion > 0 {
			return 0.7
		}
		return 0.4
	}
	costPressure := clamp01(totalCost/0.50 + totalRuntime/3600.0)
	if deltaVsChampion <= 0 {
		return roundDiagnosis(maxDiagnosis(0, 0.45-costPressure*0.35))
	}
	return clamp01(0.45 + deltaVsChampion*8 - costPressure*0.25)
}

func diagnosisFailureModes(diagnosis PlannerDiagnosis) []string {
	modes := []string{}
	if diagnosis.OverfittingScore >= 0.55 {
		modes = append(modes, "overfitting")
	}
	if diagnosis.UnderfittingScore >= 0.55 {
		modes = append(modes, "underfitting")
	}
	if diagnosis.PlateauScore >= 0.55 {
		modes = append(modes, "plateau")
	}
	if diagnosis.InstabilityScore >= 0.55 {
		modes = append(modes, "instability")
	}
	if diagnosis.ClassImbalanceScore >= 0.45 || diagnosis.MinorityClassFailureScore >= 0.45 {
		modes = append(modes, "class_imbalance")
	}
	if diagnosis.MinorityClassFailureScore >= 0.55 {
		modes = append(modes, "minority_class_failure")
	}
	if diagnosis.CostEfficiencyScore > 0 && diagnosis.CostEfficiencyScore < 0.35 {
		modes = append(modes, "poor_cost_efficiency")
	}
	if diagnosis.LatencyPenalty >= 0.45 {
		modes = append(modes, "latency_penalty")
	}
	if diagnosis.ImprovementStagnationScore >= 0.55 {
		modes = append(modes, "improvement_stagnation")
	}
	return uniqueDiagnosisStrings(modes)
}

func diagnosisSignals(diagnosis PlannerDiagnosis) []string {
	signals := []string{}
	if diagnosis.OverfittingScore > 0 {
		signals = append(signals, fmt.Sprintf("overfitting_score=%.3f", diagnosis.OverfittingScore))
	}
	if diagnosis.UnderfittingScore > 0 {
		signals = append(signals, fmt.Sprintf("underfitting_score=%.3f", diagnosis.UnderfittingScore))
	}
	if diagnosis.PlateauScore > 0 {
		signals = append(signals, fmt.Sprintf("plateau_score=%.3f", diagnosis.PlateauScore))
	}
	if diagnosis.InstabilityScore > 0 {
		signals = append(signals, fmt.Sprintf("instability_score=%.3f", diagnosis.InstabilityScore))
	}
	if diagnosis.ClassImbalanceScore > 0 {
		signals = append(signals, fmt.Sprintf("class_imbalance_score=%.3f", diagnosis.ClassImbalanceScore))
	}
	if diagnosis.MinorityClassFailureScore > 0 {
		signals = append(signals, fmt.Sprintf("minority_class_failure_score=%.3f", diagnosis.MinorityClassFailureScore))
	}
	if diagnosis.BestMetricDeltaVsChampion != 0 {
		signals = append(signals, fmt.Sprintf("best_metric_delta_vs_champion=%.3f", diagnosis.BestMetricDeltaVsChampion))
	}
	if diagnosis.LatencyPenalty > 0 {
		signals = append(signals, fmt.Sprintf("latency_penalty=%.3f", diagnosis.LatencyPenalty))
	}
	if diagnosis.ImprovementStagnationScore > 0 {
		signals = append(signals, fmt.Sprintf("improvement_stagnation_score=%.3f", diagnosis.ImprovementStagnationScore))
	}
	return signals
}

func diagnosisEvidence(input ExperimentPlannerInput, diagnosis PlannerDiagnosis, bestScore float64, totalCost float64, totalRuntime float64, minorityEvidence string, latencyEvidence string) []string {
	evidence := []string{
		fmt.Sprintf("plan has %d summaries and %d evaluation payloads", len(input.PlanSummaries), len(input.PlanEvaluations)),
		fmt.Sprintf("best source-plan metric is %.3f with delta %.3f vs champion", bestScore, diagnosis.BestMetricDeltaVsChampion),
		fmt.Sprintf("source-plan cost %.3f USD and runtime %.0fs", totalCost, totalRuntime),
	}
	if diagnosis.GeneralizationGap != 0 {
		evidence = append(evidence, fmt.Sprintf("average validation/train loss gap is %.3f", diagnosis.GeneralizationGap))
	}
	if input.DatasetInsights.ImbalanceRatio > 0 {
		evidence = append(evidence, fmt.Sprintf("dataset imbalance ratio is %.2f", input.DatasetInsights.ImbalanceRatio))
	}
	if minorityEvidence != "" {
		evidence = append(evidence, minorityEvidence)
	}
	if latencyEvidence != "" {
		evidence = append(evidence, latencyEvidence)
	}
	if input.NoImprovementRounds > 0 {
		evidence = append(evidence, fmt.Sprintf("%d no-improvement follow-up round(s)", input.NoImprovementRounds))
	}
	return uniqueDiagnosisStrings(evidence)
}

func normalizedDiagnosisMetric(metric string) string {
	normalized := strings.ToLower(strings.TrimSpace(metric))
	if normalized == "accuracy" {
		return "accuracy"
	}
	return "macro_f1"
}

func diagnosisPayloadFloat(payload map[string]any, key string) float64 {
	switch value := payload[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case jsonNumber:
		out, _ := value.Float64()
		return out
	default:
		return 0
	}
}

func firstPositivePayloadFloat(payload map[string]any, keys ...string) float64 {
	for _, key := range keys {
		if value := diagnosisPayloadFloat(payload, key); value > 0 {
			return value
		}
	}
	return 0
}

func averageScore(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	total := 0.0
	for _, value := range values {
		total += value
	}
	return total / float64(len(values))
}

func clamp01(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func maxDiagnosis(left float64, right float64) float64 {
	if left > right {
		return left
	}
	return right
}

func roundDiagnosis(value float64) float64 {
	return math.Round(value*1000) / 1000
}

func uniqueDiagnosisStrings(values []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, trimmed)
	}
	return out
}
