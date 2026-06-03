package agents

import (
	"fmt"
	"math"
	"strings"

	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
)

const (
	defaultChampionThreshold = 0.90
	defaultCostBudgetUSD     = 10.00
)

type ExperimentReviewer struct {
	championThreshold float64
	costBudgetUSD     float64
}

func NewExperimentReviewer() ExperimentReviewer {
	return ExperimentReviewer{
		championThreshold: defaultChampionThreshold,
		costBudgetUSD:     defaultCostBudgetUSD,
	}
}

func (r ExperimentReviewer) Review(project projects.Project, plan plans.ExperimentPlan, summaries []runs.TrainingRunSummary) decisions.AgentDecisionRecommendation {
	planSummaries := summariesForPlan(plan.ID, summaries)
	payload := map[string]any{
		"project_id":            project.ID,
		"plan_id":               plan.ID,
		"target_metric":         plan.TargetMetric,
		"expected_experiments":  len(plan.Experiments),
		"reported_experiments":  len(planSummaries),
		"total_estimated_cost":  roundFloat(totalEstimatedCost(planSummaries), 6),
		"total_runtime_seconds": roundFloat(totalRuntimeSeconds(planSummaries), 3),
	}

	if plan.ID == "" {
		return decisions.AgentDecisionRecommendation{
			DecisionType: decisions.TypeWait,
			Rationale:    "No experiment plan exists yet, so there is nothing for the reviewer to compare.",
			Payload:      payload,
		}
	}

	if len(planSummaries) == 0 {
		return decisions.AgentDecisionRecommendation{
			PlanID:       plan.ID,
			DecisionType: decisions.TypeWait,
			Rationale:    "The plan has no training summaries yet. The reviewer is waiting for experiment jobs to report results.",
			Payload:      payload,
		}
	}

	if !allPlannedRunsFinished(plan, planSummaries) {
		payload["active_runs"] = activeRunCount(planSummaries)
		return decisions.AgentDecisionRecommendation{
			PlanID:       plan.ID,
			DecisionType: decisions.TypeWait,
			Rationale:    "Not all planned experiments have finished yet. The reviewer is waiting before choosing a champion.",
			Payload:      payload,
		}
	}

	successful := successfulSummaries(planSummaries)
	if len(successful) == 0 {
		if totalEstimatedCost(planSummaries) < r.costBudgetUSD {
			payload["proposed_experiments"] = fallbackExperiments()
			return decisions.AgentDecisionRecommendation{
				PlanID:       plan.ID,
				DecisionType: decisions.TypeAddExperiments,
				Rationale:    "All completed experiments failed, but the estimated spend is still below the review budget. The reviewer recommends trying a safer fallback set.",
				Payload:      payload,
			}
		}

		return decisions.AgentDecisionRecommendation{
			PlanID:       plan.ID,
			DecisionType: decisions.TypeStopProject,
			Rationale:    "All completed experiments failed and the estimated review budget has been reached.",
			Payload:      payload,
		}
	}

	best := bestSummary(plan.TargetMetric, successful)
	bestScore := summaryReadinessScore(plan.TargetMetric, best)
	payload["champion_job_id"] = best.JobID
	payload["champion_model"] = best.Model
	payload["champion_score"] = roundFloat(bestScore, 6)
	payload["champion_score_basis"] = "loss_heavy_summary_readiness"
	payload["champion_macro_f1"] = roundFloat(best.BestMacroF1, 6)
	payload["champion_accuracy"] = roundFloat(best.BestAccuracy, 6)
	payload["champion_final_train_loss"] = roundFloat(best.FinalTrainLoss, 6)
	payload["champion_final_val_loss"] = roundFloat(best.FinalValLoss, 6)
	payload["champion_estimated_cost_usd"] = roundFloat(best.EstimatedCostUSD, 6)
	payload["champion_runtime_seconds"] = roundFloat(best.RuntimeSeconds, 3)
	payload["cost_efficiency"] = roundFloat(costEfficiency(bestScore, best.EstimatedCostUSD), 6)

	if bestScore >= r.championThreshold {
		return decisions.AgentDecisionRecommendation{
			PlanID:       plan.ID,
			DecisionType: decisions.TypeSelectChampion,
			Rationale: fmt.Sprintf(
				"%s is the current champion because it reached %.3f loss-aware readiness, meeting the quality threshold for this plan.",
				best.Model,
				bestScore,
			),
			Payload: payload,
		}
	}

	if totalEstimatedCost(planSummaries) < r.costBudgetUSD {
		payload["proposed_experiments"] = followUpExperiments(best)
		return decisions.AgentDecisionRecommendation{
			PlanID:       plan.ID,
			DecisionType: decisions.TypeAddExperiments,
			Rationale: fmt.Sprintf(
				"The best run reached %.3f loss-aware readiness, which is below the %.2f champion threshold. Estimated spend is still below $%.2f, so the reviewer recommends another experiment round.",
				bestScore,
				r.championThreshold,
				r.costBudgetUSD,
			),
			Payload: payload,
		}
	}

	return decisions.AgentDecisionRecommendation{
		PlanID:       plan.ID,
		DecisionType: decisions.TypeStopProject,
		Rationale: fmt.Sprintf(
			"The best run reached %.3f loss-aware readiness, below the %.2f champion threshold, and the estimated review budget has been reached.",
			bestScore,
			r.championThreshold,
		),
		Payload: payload,
	}
}

func summariesForPlan(planID string, summaries []runs.TrainingRunSummary) []runs.TrainingRunSummary {
	out := []runs.TrainingRunSummary{}
	for _, summary := range summaries {
		if summary.PlanID == planID {
			out = append(out, summary)
		}
	}
	return out
}

func allPlannedRunsFinished(plan plans.ExperimentPlan, summaries []runs.TrainingRunSummary) bool {
	if len(summaries) < len(plan.Experiments) {
		return false
	}
	for _, summary := range summaries {
		if !isTerminalRunStatus(summary.Status) {
			return false
		}
	}
	return true
}

func activeRunCount(summaries []runs.TrainingRunSummary) int {
	count := 0
	for _, summary := range summaries {
		if !isTerminalRunStatus(summary.Status) {
			count++
		}
	}
	return count
}

func isTerminalRunStatus(status string) bool {
	switch strings.ToUpper(status) {
	case jobs.StatusSucceeded, jobs.StatusFailed:
		return true
	default:
		return false
	}
}

func successfulSummaries(summaries []runs.TrainingRunSummary) []runs.TrainingRunSummary {
	out := []runs.TrainingRunSummary{}
	for _, summary := range summaries {
		if strings.ToUpper(summary.Status) == jobs.StatusSucceeded {
			out = append(out, summary)
		}
	}
	return out
}

func bestSummary(targetMetric string, summaries []runs.TrainingRunSummary) runs.TrainingRunSummary {
	best := summaries[0]
	bestScore := summaryReadinessScore(targetMetric, best)
	for _, summary := range summaries[1:] {
		score := summaryReadinessScore(targetMetric, summary)
		if score > bestScore {
			best = summary
			bestScore = score
			continue
		}
		if score == bestScore && summary.EstimatedCostUSD < best.EstimatedCostUSD {
			best = summary
			bestScore = score
		}
	}
	return best
}

func summaryReadinessScore(targetMetric string, summary runs.TrainingRunSummary) float64 {
	metricScore := targetMetricValue(targetMetric, summary)
	lossScore, hasLoss := summaryLossHealthScore(summary)
	if !hasLoss {
		return metricScore
	}
	return metricScore*0.35 + lossScore*0.65
}

func summaryLossHealthScore(summary runs.TrainingRunSummary) (float64, bool) {
	weighted := summaryWeightedScore{}
	if summary.FinalTrainLoss > 0 {
		weighted.add(summaryLossValueScore(summary.FinalTrainLoss), 0.25, true)
	}
	if summary.FinalValLoss > 0 {
		weighted.add(summaryLossValueScore(summary.FinalValLoss), 0.45, true)
	}
	if summary.FinalTrainLoss > 0 && summary.FinalValLoss > 0 {
		gap := summary.FinalValLoss - summary.FinalTrainLoss
		ratio := summary.FinalValLoss / math.Max(summary.FinalTrainLoss, 0.000001)
		weighted.add(1-math.Min(1, math.Max(0, gap)/0.75), 0.18, true)
		weighted.add(1-math.Min(1, math.Max(0, ratio-1)/1.50), 0.12, true)
	}
	if weighted.weight <= 0 {
		return 0, false
	}
	return weighted.value(1), true
}

func summaryLossValueScore(loss float64) float64 {
	if loss <= 0 {
		return 0
	}
	normalized := loss / math.Log(2)
	return clampReviewerScore(1 - (normalized-0.20)/1.30)
}

type summaryWeightedScore struct {
	total  float64
	weight float64
}

func (s *summaryWeightedScore) add(value float64, weight float64, ok bool) {
	if !ok || weight <= 0 {
		return
	}
	s.total += clampReviewerScore(value) * weight
	s.weight += weight
}

func (s summaryWeightedScore) value(fallback float64) float64 {
	if s.weight <= 0 {
		return clampReviewerScore(fallback)
	}
	return clampReviewerScore(s.total / s.weight)
}

func clampReviewerScore(value float64) float64 {
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}

func targetMetricValue(targetMetric string, summary runs.TrainingRunSummary) float64 {
	switch normalizedTargetMetric(targetMetric) {
	case "accuracy":
		return summary.BestAccuracy
	default:
		return summary.BestMacroF1
	}
}

func normalizedTargetMetric(targetMetric string) string {
	normalized := strings.ToLower(strings.TrimSpace(targetMetric))
	if normalized == "" {
		return "macro_f1"
	}
	return normalized
}

func totalEstimatedCost(summaries []runs.TrainingRunSummary) float64 {
	total := 0.0
	for _, summary := range summaries {
		total += summary.EstimatedCostUSD
	}
	return total
}

func totalRuntimeSeconds(summaries []runs.TrainingRunSummary) float64 {
	total := 0.0
	for _, summary := range summaries {
		total += summary.RuntimeSeconds
	}
	return total
}

func costEfficiency(score float64, cost float64) float64 {
	if cost <= 0 {
		return 0
	}
	return score / cost
}

func roundFloat(value float64, places int) float64 {
	if places < 0 {
		return value
	}
	scale := math.Pow10(places)
	return math.Round(value*scale) / scale
}

func fallbackExperiments() []plans.PlannedExperiment {
	return []plans.PlannedExperiment{
		{
			Template:     "mobilenet_transfer",
			Model:        "mobilenet_v3_small",
			Epochs:       6,
			BatchSize:    16,
			LearningRate: 0.0001,
			Reason:       "Fallback to a low-cost transfer-learning baseline with a conservative learning rate.",
		},
	}
}

func followUpExperiments(best runs.TrainingRunSummary) []plans.PlannedExperiment {
	bestFamily := modelFamily(best.Model)
	experiments := []plans.PlannedExperiment{
		{
			Template:              "efficientnet_transfer",
			Model:                 "efficientnet_b0",
			Epochs:                10,
			BatchSize:             16,
			LearningRate:          0.0002,
			Reason:                "Try a quality/latency challenger with moderate augmentation instead of extending the current best model's epoch budget.",
			ImageSize:             224,
			ResolutionStrategy:    "fixed",
			Optimizer:             "adamw",
			Scheduler:             "cosine",
			WeightDecay:           0.01,
			Augmentation:          map[string]any{"horizontal_flip": true, "color_jitter": true},
			AugmentationPolicy:    "moderate",
			ClassBalancing:        "weighted_loss",
			SamplingStrategy:      "none",
			EarlyStoppingPatience: 3,
			Strategy:              "deterministic quality challenger",
			Pretrained:            true,
			FreezeBackbone:        true,
			FineTuneStrategy:      "head_only",
		},
		{
			Template:              "mobilenet_transfer",
			Model:                 "mobilenet_v3_large",
			Epochs:                10,
			BatchSize:             16,
			LearningRate:          0.0003,
			Reason:                "Test a compact live-inference challenger with class-balanced sampling rather than repeating the prior mechanism.",
			ImageSize:             224,
			ResolutionStrategy:    "low_latency",
			Optimizer:             "adamw",
			Scheduler:             "cosine",
			WeightDecay:           0.005,
			Augmentation:          map[string]any{"horizontal_flip": true},
			AugmentationPolicy:    "light",
			ClassBalancing:        "class_balanced_sampler",
			SamplingStrategy:      "class_balanced_sampler",
			EarlyStoppingPatience: 3,
			Strategy:              "deterministic deployment challenger",
			Pretrained:            true,
			FreezeBackbone:        true,
			FineTuneStrategy:      "head_only",
		},
	}
	if bestFamily == "efficientnet" {
		experiments[0] = plans.PlannedExperiment{
			Template:              "regnet_transfer",
			Model:                 "regnet_y_400mf",
			Epochs:                10,
			BatchSize:             16,
			LearningRate:          0.0002,
			Reason:                "Try a compact non-EfficientNet architecture with regularization to avoid repeating the current best family.",
			ImageSize:             224,
			ResolutionStrategy:    "low_latency",
			Optimizer:             "adamw",
			Scheduler:             "cosine",
			WeightDecay:           0.01,
			Augmentation:          map[string]any{"horizontal_flip": true, "color_jitter": true},
			AugmentationPolicy:    "moderate",
			ClassBalancing:        "weighted_loss",
			SamplingStrategy:      "none",
			EarlyStoppingPatience: 3,
			Strategy:              "deterministic architecture challenger",
			Pretrained:            true,
			FreezeBackbone:        true,
			FineTuneStrategy:      "head_only",
		}
	}
	if bestFamily == "mobilenet" {
		experiments[1] = plans.PlannedExperiment{
			Template:              "regnet_transfer",
			Model:                 "regnet_y_400mf",
			Epochs:                10,
			BatchSize:             16,
			LearningRate:          0.0002,
			Reason:                "Compare a compact architecture family against the MobileNet champion without simply increasing epochs.",
			ImageSize:             224,
			ResolutionStrategy:    "low_latency",
			Optimizer:             "adamw",
			Scheduler:             "cosine",
			WeightDecay:           0.01,
			Augmentation:          map[string]any{"horizontal_flip": true, "color_jitter": true},
			AugmentationPolicy:    "moderate",
			ClassBalancing:        "weighted_loss",
			SamplingStrategy:      "none",
			EarlyStoppingPatience: 3,
			Strategy:              "deterministic architecture challenger",
			Pretrained:            true,
			FreezeBackbone:        true,
			FineTuneStrategy:      "head_only",
		}
	}
	return experiments
}

func modelFamily(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	switch {
	case strings.HasPrefix(normalized, "mobilenet"):
		return "mobilenet"
	case strings.HasPrefix(normalized, "efficientnet"):
		return "efficientnet"
	case strings.HasPrefix(normalized, "regnet"):
		return "regnet"
	case strings.HasPrefix(normalized, "resnet"):
		return "resnet"
	default:
		return normalized
	}
}
