package calibration

import (
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestCalibrationReportForecastDirectionCoverageAndIntervals(t *testing.T) {
	request := testReportRequest()
	evaluation := ObservationSet{
		Candidates: []CandidateObservation{
			testCandidateObservation(0, "inv-0", MetricDirectionHigherIsBetter, 0.05, 0.04, true, true, 1),
			testCandidateObservation(1, "inv-1", MetricDirectionHigherIsBetter, 0.04, -0.02, true, true, 1),
			testCandidateObservation(2, "inv-2", MetricDirectionLowerIsBetter, -0.02, -0.06, true, true, 0),
			testCandidateObservation(3, "inv-3", MetricDirectionHigherIsBetter, 0.50, 0, false, false, 0),
		},
		Invocations: []InvocationObservation{
			testInvocationObservation("inv-0", "pricing-v1", 100),
			testInvocationObservation("inv-1", "pricing-v1", 200),
			testInvocationObservation("inv-2", "pricing-v1", 300),
			testInvocationObservation("inv-3", "pricing-v1", 400),
		},
	}
	report := BuildReport(request, ObservationSet{}, evaluation)
	cohort := findReportCohort(t, report, "overall")
	if cohort.Status != CohortAvailable || cohort.SampleSize != 4 {
		t.Fatalf("overall cohort=%#v", cohort)
	}
	assertEstimate(t, cohort.Forecast.MeanAbsoluteError, 0.11/3, 3)
	assertEstimate(t, cohort.Forecast.BiasPredictedMinusActual, 0.01, 3)
	assertEstimate(t, cohort.Forecast.CorrectImprovementDirection, 2.0/3.0, 3)
	assertEstimate(t, cohort.Forecast.MeaningfulImprovementPrecision, 0.5, 2)
	assertEstimate(t, cohort.Forecast.MeaningfulImprovementRecall, 0.5, 2)
	assertEstimate(t, cohort.Decision.EvidenceCoverage, 0.5, 4)
	assertEstimate(t, cohort.Decision.UnsupportedClaimRate, 0.5, 4)

	if cohort.Coverage.CandidateForecasts != 4 || cohort.Coverage.SelectedCandidates != 3 || cohort.Coverage.ObservedOutcomes != 3 || cohort.Coverage.UnknownOutcomes != 1 {
		t.Fatalf("coverage counts=%#v", cohort.Coverage)
	}
	assertEstimate(t, cohort.Coverage.ObservedOutcomeCoverage, 0.75, 4)
	assertEstimate(t, cohort.Coverage.SelectedOutcomeCoverage, 1, 3)
	if cohort.Forecast.CorrectImprovementDirection.ConfidenceInterval == nil ||
		cohort.Forecast.CorrectImprovementDirection.ConfidenceInterval.Method != ConfidenceMethodWilson95 ||
		cohort.Forecast.CorrectImprovementDirection.ConfidenceInterval.Lower < 0 ||
		cohort.Forecast.CorrectImprovementDirection.ConfidenceInterval.Upper > 1 {
		t.Fatalf("direction interval=%#v", cohort.Forecast.CorrectImprovementDirection.ConfidenceInterval)
	}
	if !reflect.DeepEqual(cohort.ScoreVersions, []string{CandidateForecastScoreVersionV1}) || !reflect.DeepEqual(cohort.PricingVersions, []string{"pricing-v1"}) {
		t.Fatalf("versions score=%v pricing=%v", cohort.ScoreVersions, cohort.PricingVersions)
	}
	for _, grouping := range []string{"planner_variant", "task", "mechanism", "model_family", "task_mechanism", "task_mechanism_model_family"} {
		grouped := findReportCohort(t, report, grouping)
		if grouped.Status != CohortAvailable {
			t.Fatalf("grouping %s was not reported: %#v", grouping, grouped)
		}
	}
	if !strings.Contains(report.SelectionBiasDisclosure, "unselected candidates") || report.WindowDisclosure == "" {
		t.Fatalf("missing disclosures: %#v", report)
	}

	repeated := BuildReport(request, ObservationSet{}, evaluation)
	if !reflect.DeepEqual(report, repeated) {
		t.Fatalf("report is not deterministic:\nfirst=%#v\nsecond=%#v", report, repeated)
	}
}

func TestCandidateDecisionMetadataNormalizesModelFamilyAndEvidence(t *testing.T) {
	payload := map[string]any{
		"candidate_hypotheses": []any{
			map[string]any{
				"evidence_used":     []any{"minority recall", "", "class counts"},
				"experiment_config": map[string]any{"model": "EfficientNet_B3"},
			},
		},
	}
	family, evidenceCount := CandidateDecisionMetadata(payload, 0)
	if family != "efficientnet" || evidenceCount != 2 {
		t.Fatalf("metadata family=%q evidence=%d", family, evidenceCount)
	}
	if family, evidenceCount := CandidateDecisionMetadata(payload, 2); family != "unknown" || evidenceCount != 0 {
		t.Fatalf("out-of-range metadata family=%q evidence=%d", family, evidenceCount)
	}
}

func TestCalibrationReportSuppressesUndersizedCohorts(t *testing.T) {
	request := testReportRequest()
	request.MinCohortSize = 5
	report := BuildReport(request, ObservationSet{}, ObservationSet{Candidates: []CandidateObservation{
		testCandidateObservation(0, "inv-0", MetricDirectionHigherIsBetter, 0.05, 0.04, true, true, 1),
	}})
	cohort := findReportCohort(t, report, "overall")
	if cohort.Status != CohortInsufficientEvidence || cohort.Structural != nil || cohort.Forecast != nil || !strings.Contains(cohort.SuppressionReason, "below min_cohort_size") {
		t.Fatalf("undersized cohort leaked metrics: %#v", cohort)
	}
}

func TestCalibrationReportStructuralCohortsIncludeInvocationsWithoutCandidates(t *testing.T) {
	request := testReportRequest()
	request.MinCohortSize = 5
	invocations := []InvocationObservation{}
	for index := 0; index < 5; index++ {
		invocation := testInvocationObservation("failed-"+string(rune('a'+index)), "pricing-v1", 100)
		invocation.Parsed = index != 0
		invocation.ValidationStatus = "invalid"
		invocation.FirstPassStatus = "rejected"
		invocation.EventualStatus = "rejected"
		invocations = append(invocations, invocation)
	}
	report := BuildReport(request, ObservationSet{}, ObservationSet{Invocations: invocations})
	overall := findReportCohort(t, report, "overall")
	if overall.Status != CohortAvailable || overall.InvocationSampleSize != 5 || overall.CandidateSampleSize != 0 || overall.Structural == nil || overall.Decision != nil || overall.Forecast != nil {
		t.Fatalf("invocation-driven structural cohort=%#v", overall)
	}
	assertEstimate(t, overall.Structural.ParseSuccess, 0.8, 5)
	assertEstimate(t, overall.Structural.EventualValidation, 0, 5)
}

func TestCalibrationReportUnselectedCandidateIsCoverageNotNegativeOutcome(t *testing.T) {
	request := testReportRequest()
	selected := testCandidateObservation(0, "inv-0", MetricDirectionHigherIsBetter, 0.04, 0.05, true, true, 1)
	unselected := testCandidateObservation(1, "inv-1", MetricDirectionHigherIsBetter, 0.90, 0, false, false, 1)
	report := BuildReport(request, ObservationSet{}, ObservationSet{Candidates: []CandidateObservation{selected, unselected}})
	cohort := findReportCohort(t, report, "overall")
	if cohort.Forecast.MeanAbsoluteError.SampleSize != 1 || cohort.Forecast.MeaningfulImprovementPrecision.SampleSize != 1 {
		t.Fatalf("unselected candidate entered forecast labels: %#v", cohort.Forecast)
	}
	assertEstimate(t, cohort.Coverage.ObservedOutcomeCoverage, 0.5, 2)
}

func TestCalibrationReportChronologicalPriorsUseTrainingWindowOnly(t *testing.T) {
	request := testReportRequest()
	trainingCandidate := testCandidateObservation(0, "train", MetricDirectionLowerIsBetter, -0.03, -0.06, true, true, 1)
	trainingFinalizedAt := request.TrainingWindow.End.Add(-time.Hour)
	trainingCandidate.FinalizedAt = &trainingFinalizedAt
	evaluationCandidate := testCandidateObservation(1, "eval", MetricDirectionHigherIsBetter, 0.04, -0.20, true, true, 1)
	report := BuildReport(request, ObservationSet{Candidates: []CandidateObservation{trainingCandidate}}, ObservationSet{Candidates: []CandidateObservation{evaluationCandidate}})
	if len(report.Priors) == 0 {
		t.Fatal("expected chronological priors")
	}
	var overall EmpiricalPrior
	for _, prior := range report.Priors {
		if prior.Key.Grouping == "overall" {
			overall = prior
			break
		}
	}
	if overall.SampleSize != 1 || overall.Window != request.TrainingWindow {
		t.Fatalf("overall prior=%#v", overall)
	}
	assertEstimate(t, overall.MeanNormalizedImprovement, 0.06, 1)
}

func TestCalibrationReportRequestBoundsAndReaderWindows(t *testing.T) {
	request := testReportRequest()
	reader := &recordingObservationReader{}
	report, err := GenerateReport(reader, request)
	if err != nil {
		t.Fatal(err)
	}
	if len(reader.calls) != 2 || reader.calls[0].window != request.TrainingWindow || reader.calls[1].window != request.EvaluationWindow || reader.calls[0].limit != request.Limit {
		t.Fatalf("bounded reads=%#v", reader.calls)
	}
	if report.ReadBounds.PerWindowLimit != request.Limit {
		t.Fatalf("read bounds=%#v", report.ReadBounds)
	}

	invalid := request
	invalid.TrainingWindow.End = invalid.EvaluationWindow.Start.Add(time.Second)
	if _, err := GenerateReport(reader, invalid); err == nil || !strings.Contains(err.Error(), "training window") {
		t.Fatalf("chronological overlap error=%v", err)
	}
	invalid = request
	invalid.Limit = MaximumCalibrationReadLimit + 1
	if _, err := GenerateReport(reader, invalid); err == nil || !strings.Contains(err.Error(), "limit") {
		t.Fatalf("limit error=%v", err)
	}
}

type recordingObservationReader struct {
	calls []struct {
		projectID string
		window    TimeWindow
		limit     int
	}
}

func (reader *recordingObservationReader) ReadCalibrationObservations(projectID string, window TimeWindow, limit int) (ObservationSet, error) {
	reader.calls = append(reader.calls, struct {
		projectID string
		window    TimeWindow
		limit     int
	}{projectID: projectID, window: window, limit: limit})
	return ObservationSet{}, nil
}

func testReportRequest() ReportRequest {
	trainingStart := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	evaluationStart := trainingStart.Add(30 * 24 * time.Hour)
	return ReportRequest{
		ProjectID:        "project-1",
		TrainingWindow:   TimeWindow{Start: trainingStart, End: evaluationStart},
		EvaluationWindow: TimeWindow{Start: evaluationStart, End: evaluationStart.Add(7 * 24 * time.Hour)},
		Limit:            100, MinCohortSize: 1, MeaningfulImprovement: 0.03,
	}
}

func testCandidateObservation(index int, invocationID, direction string, predicted, actualDelta float64, selected, observed bool, evidenceCount int) CandidateObservation {
	created := time.Date(2026, 2, 2, 0, 0, index, 0, time.UTC)
	candidate := CandidateProvenance{
		ID: "candidate-" + invocationID, ProjectID: "project-1", DecisionID: "decision-1",
		CandidateProvenanceCreate: CandidateProvenanceCreate{
			InvocationID: invocationID, PlannerVariantID: "variant-1", CandidateIndex: index,
			Task: "image_classification", Mechanism: "class_imbalance", Selected: selected,
			Forecast:      CandidateForecastContract{MetricDirection: direction, ScoreVersion: CandidateForecastScoreVersionV1, PredictedDelta: predicted},
			OutcomeStatus: CandidateOutcomeUnknown,
		},
		CreatedAt: created,
	}
	if observed {
		eligible := true
		finalizedAt := time.Date(2026, 2, 2, 1, 0, index, 0, time.UTC)
		candidate.OutcomeStatus = CandidateOutcomeObserved
		candidate.ActualDelta = &actualDelta
		candidate.CalibrationEligible = &eligible
		candidate.FinalizedAt = &finalizedAt
	}
	return CandidateObservation{CandidateProvenance: candidate, ModelFamily: "resnet", EvidenceCount: evidenceCount}
}

func testInvocationObservation(id, pricing string, tokens int) InvocationObservation {
	cost := float64(tokens) / 1_000_000
	return InvocationObservation{
		ID: id, PlannerVariantID: "variant-1", AttemptGroupID: id, Parsed: true,
		ValidationStatus: "valid", FirstPassStatus: "accepted", EventualStatus: "accepted", RetryOutcome: "not_needed",
		WallLatencyMS: float64(tokens), TotalTokens: tokens, ToolRounds: 1, CostUSD: &cost, PricingVersion: pricing,
		CreatedAt: time.Date(2026, 2, 2, 0, 0, 0, 0, time.UTC),
	}
}

func findReportCohort(t *testing.T, report Report, grouping string) ReportCohort {
	t.Helper()
	for _, cohort := range report.Cohorts {
		if cohort.Key.Grouping == grouping {
			return cohort
		}
	}
	t.Fatalf("missing %s cohort in %#v", grouping, report.Cohorts)
	return ReportCohort{}
}

func assertEstimate(t *testing.T, estimate MetricEstimate, want float64, sampleSize int) {
	t.Helper()
	if estimate.Value == nil || math.Abs(*estimate.Value-want) > 1e-9 || estimate.SampleSize != sampleSize || estimate.Status != EstimateAvailable {
		t.Fatalf("estimate=%#v want value=%f n=%d", estimate, want, sampleSize)
	}
}
