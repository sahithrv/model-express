package automl

import (
	"fmt"
	"sort"
	"strings"
)

func BuildFeedbackSummary(studyID string, targetMetric string, trials []OptimizerTrial) OptimizerFeedbackSummary {
	out := OptimizerFeedbackSummary{
		StudyID:      studyID,
		TrialCount:   len(trials),
		TargetMetric: targetMetric,
		Trend:        "insufficient_data",
	}
	if len(trials) == 0 {
		return out
	}
	ordered := append([]OptimizerTrial(nil), trials...)
	sort.Slice(ordered, func(i, j int) bool {
		return ordered[i].CreatedAt.Before(ordered[j].CreatedAt)
	})
	var best *OptimizerTrial
	for i := range ordered {
		trial := &ordered[i]
		if strings.EqualFold(strings.TrimSpace(trial.Status), "SUCCEEDED") {
			out.SucceededTrialCount++
			if best == nil || trial.Score > best.Score {
				best = trial
			}
			continue
		}
		out.FailedTrialCount++
		if trial.Error != "" && len(out.FailedParameterPatterns) < 4 {
			out.FailedParameterPatterns = append(out.FailedParameterPatterns, trial.Error)
		}
	}
	if best != nil {
		out.BestScore = best.Score
		out.BestJobID = best.JobID
		out.BestHyperparameters = mapFromAny(best.Metrics["hyperparameters"])
		out.TrainValidationGap = numberFromAny(best.Metrics["train_validation_gap"])
	}
	if out.SucceededTrialCount >= 2 {
		first := firstSucceededScore(ordered)
		last := lastSucceededScore(ordered)
		switch {
		case last > first+0.002:
			out.Trend = "improving"
		case last < first-0.002:
			out.Trend = "declining"
		default:
			out.Trend = "flat"
		}
	}
	if len(out.RecommendedNarrowing) == 0 && len(out.BestHyperparameters) > 0 {
		out.RecommendedNarrowing = append(out.RecommendedNarrowing, fmt.Sprintf("Best observed %s=%g; keep future ranges near the best validated hyperparameters.", out.TargetMetric, out.BestScore))
	}
	return out
}

func firstSucceededScore(trials []OptimizerTrial) float64 {
	for _, trial := range trials {
		if strings.EqualFold(strings.TrimSpace(trial.Status), "SUCCEEDED") {
			return trial.Score
		}
	}
	return 0
}

func lastSucceededScore(trials []OptimizerTrial) float64 {
	for i := len(trials) - 1; i >= 0; i-- {
		if strings.EqualFold(strings.TrimSpace(trials[i].Status), "SUCCEEDED") {
			return trials[i].Score
		}
	}
	return 0
}

func mapFromAny(value any) map[string]any {
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	return nil
}

func numberFromAny(value any) float64 {
	number, _ := NumberValue(value)
	return number
}
