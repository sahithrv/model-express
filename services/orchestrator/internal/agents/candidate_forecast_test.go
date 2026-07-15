package agents

import (
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/plans"
)

func TestCandidateForecastIsFrozenFromChampionAndKeepsRecommendationDeltaSeparate(t *testing.T) {
	input := ExperimentPlannerInput{
		SourcePlan: plans.ExperimentPlan{TargetMetric: "macro_f1"},
		CurrentChampion: &ExperimentChampion{
			JobID: "job_champion", TargetMetric: "macro_f1", Score: 0.72, ScoreBasis: "loss_heavy_deployment_readiness",
		},
	}
	candidates, err := freezeCandidateForecastContracts(input, []CandidateHypothesis{{ExpectedMetricImpact: 0.025}})
	if err != nil {
		t.Fatal(err)
	}
	forecast := candidates[0].Forecast
	if forecast == nil || forecast.ForecastTarget != "macro_f1" || forecast.BaselineJobID != "job_champion" || forecast.BaselineScore != 0.72 {
		t.Fatalf("forecast did not freeze champion baseline: %#v", forecast)
	}
	if forecast.PredictedDelta != 0.025 || forecast.PredictionSource != calibration.CandidatePredictionSource || forecast.Units != calibration.CandidateForecastUnits {
		t.Fatalf("forecast did not preserve candidate prediction semantics: %#v", forecast)
	}
	if forecast.ScoreBasis != "loss_heavy_deployment_readiness" || forecast.ScoreVersion != calibration.CandidateForecastScoreVersionV1 {
		t.Fatalf("forecast score identity is incomplete: %#v", forecast)
	}
}

func TestCandidateForecastRejectsMismatchedFrozenContractAndImpossiblePrediction(t *testing.T) {
	input := ExperimentPlannerInput{
		SourcePlan:      plans.ExperimentPlan{TargetMetric: "macro_f1"},
		CurrentChampion: &ExperimentChampion{JobID: "job_champion", TargetMetric: "macro_f1", Score: 0.99},
	}
	if _, err := freezeCandidateForecastContracts(input, []CandidateHypothesis{{ExpectedMetricImpact: 1.02}}); err == nil || !strings.Contains(err.Error(), "outside valid range") {
		t.Fatalf("impossible forecast was not rejected: %v", err)
	}

	input.CurrentChampion.Score = 0.70
	supplied := calibration.CandidateForecastContract{
		ForecastTarget: "macro_f1", MetricDirection: calibration.MetricDirectionHigherIsBetter,
		ScoreBasis: "macro_f1_score", ScoreVersion: calibration.CandidateForecastScoreVersionV1,
		BaselineJobID: "job_champion", BaselineScore: 0.70, PredictedDelta: 0.02,
		PredictionSource: calibration.CandidatePredictionSource, Units: "percentage_points",
		ValidRange: calibration.CandidateForecastRange{Min: 0, Max: 1},
	}
	if _, err := freezeCandidateForecastContracts(input, []CandidateHypothesis{{ExpectedMetricImpact: 0.02, Forecast: &supplied}}); err == nil || !strings.Contains(err.Error(), "backend-frozen contract") {
		t.Fatalf("mismatched forecast units were not rejected: %v", err)
	}
}

func TestPlannerPromptNamesFrozenCandidateForecastContract(t *testing.T) {
	request := experimentPlannerJSONRequest("test-model", []byte(`{}`))
	combined := ""
	for _, message := range request.Messages {
		combined += message.Content
	}
	for _, field := range []string{
		"forecast_target", "metric_direction", "score_basis", "score_version", "baseline_job_id", "baseline_score",
		"predicted_delta", "candidate.expected_metric_impact", "units", "valid_range",
	} {
		if !strings.Contains(combined, field) {
			t.Fatalf("planner prompt omitted forecast field %q", field)
		}
	}
}
