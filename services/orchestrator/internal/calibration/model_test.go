package calibration

import (
	"strings"
	"testing"
)

func TestCandidateForecastContractValidatesSignRangeAndUnits(t *testing.T) {
	valid := CandidateForecastContract{
		ForecastTarget:   "macro_f1",
		MetricDirection:  MetricDirectionHigherIsBetter,
		ScoreBasis:       "macro_f1_score",
		ScoreVersion:     CandidateForecastScoreVersionV1,
		BaselineJobID:    "job_1",
		BaselineScore:    0.70,
		PredictedDelta:   0.03,
		PredictionSource: CandidatePredictionSource,
		Units:            CandidateForecastUnits,
		ValidRange:       CandidateForecastRange{Min: 0, Max: 1},
	}
	if err := ValidateForecastContract(valid); err != nil {
		t.Fatalf("valid forecast rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*CandidateForecastContract)
		want   string
	}{
		{name: "sign", mutate: func(value *CandidateForecastContract) { value.PredictedDelta = -0.01 }, want: "non-negative"},
		{name: "range", mutate: func(value *CandidateForecastContract) { value.PredictedDelta = 1.01 }, want: "outside valid range"},
		{name: "units", mutate: func(value *CandidateForecastContract) { value.Units = "percentage_points" }, want: CandidateForecastUnits},
		{name: "source", mutate: func(value *CandidateForecastContract) {
			value.PredictionSource = "recommendation.expected_delta_vs_champion"
		}, want: CandidatePredictionSource},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			candidate := valid
			test.mutate(&candidate)
			err := ValidateForecastContract(candidate)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ValidateForecastContract() error = %v, want %q", err, test.want)
			}
		})
	}
}
