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
	defaultCostBudgetUSD     = 0.25
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
	bestScore := targetMetricValue(plan.TargetMetric, best)
	payload["champion_job_id"] = best.JobID
	payload["champion_model"] = best.Model
	payload["champion_score"] = roundFloat(bestScore, 6)
	payload["champion_macro_f1"] = roundFloat(best.BestMacroF1, 6)
	payload["champion_accuracy"] = roundFloat(best.BestAccuracy, 6)
	payload["champion_estimated_cost_usd"] = roundFloat(best.EstimatedCostUSD, 6)
	payload["champion_runtime_seconds"] = roundFloat(best.RuntimeSeconds, 3)
	payload["cost_efficiency"] = roundFloat(costEfficiency(bestScore, best.EstimatedCostUSD), 6)

	if bestScore >= r.championThreshold {
		return decisions.AgentDecisionRecommendation{
			PlanID:       plan.ID,
			DecisionType: decisions.TypeSelectChampion,
			Rationale: fmt.Sprintf(
				"%s is the current champion because it reached %.3f %s, meeting the quality threshold for this plan.",
				best.Model,
				bestScore,
				normalizedTargetMetric(plan.TargetMetric),
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
				"The best run reached %.3f %s, which is below the %.2f champion threshold. Estimated spend is still below $%.2f, so the reviewer recommends another experiment round.",
				bestScore,
				normalizedTargetMetric(plan.TargetMetric),
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
			"The best run reached %.3f %s, below the %.2f champion threshold, and the estimated review budget has been reached.",
			bestScore,
			normalizedTargetMetric(plan.TargetMetric),
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
	bestScore := targetMetricValue(targetMetric, best)
	for _, summary := range summaries[1:] {
		score := targetMetricValue(targetMetric, summary)
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
	model := best.Model
	if model == "" {
		model = "mobilenet_v3_small"
	}
	template := "mobilenet_transfer"
	if strings.Contains(strings.ToLower(model), "efficientnet") {
		template = "efficientnet_transfer"
	}

	return []plans.PlannedExperiment{
		{
			Template:     template,
			Model:        model,
			Epochs:       best.EpochsCompleted + 4,
			BatchSize:    12,
			LearningRate: 0.0001,
			Reason:       "Extend the current best model with a lower learning rate to see if validation performance improves.",
		},
		{
			Template:     "resnet_transfer",
			Model:        "resnet18",
			Epochs:       10,
			BatchSize:    16,
			LearningRate: 0.0002,
			Reason:       "Try a different transfer-learning family to reduce model-family bias in the search.",
		},
	}
}
