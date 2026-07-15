package calibration

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
)

const (
	CalibrationReportVersionV1 = "calibration_report_v1"
	ConfidenceMethodWilson95   = "wilson_95"
	ConfidenceMethodMean95     = "normal_mean_95"

	DefaultCalibrationReadLimit     = 1000
	MaximumCalibrationReadLimit     = 5000
	DefaultCalibrationMinCohortSize = 5
	MaximumCalibrationMinCohortSize = 100
	MaximumCalibrationReportCohorts = 500

	EstimateAvailable             = "available"
	EstimateNotApplicable         = "not_applicable"
	EstimateInsufficientEvidence  = "insufficient_evidence"
	CohortAvailable               = "available"
	CohortInsufficientEvidence    = "insufficient_evidence"
	SelectionBiasDisclosure       = "Observed outcomes are available only for candidates selected by the active v1 policy. This observational report is selection-biased, does not treat unselected candidates as failures, and makes no causal claim about counterfactual selections."
	ChronologicalWindowDisclosure = "Training/prior and evaluation windows are half-open [start, end); the training window must end at or before the evaluation window starts."
)

var ErrInvalidReportRequest = errors.New("invalid calibration report request")

// TimeWindow is a frozen half-open UTC interval [start, end).
type TimeWindow struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

type ReportRequest struct {
	ProjectID             string     `json:"project_id"`
	TrainingWindow        TimeWindow `json:"training_window"`
	EvaluationWindow      TimeWindow `json:"evaluation_window"`
	Limit                 int        `json:"limit"`
	MinCohortSize         int        `json:"min_cohort_size"`
	MeaningfulImprovement float64    `json:"meaningful_improvement"`
}

type CandidateObservation struct {
	CandidateProvenance
	ModelFamily   string `json:"model_family"`
	EvidenceCount int    `json:"evidence_count"`
}

// InvocationObservation is a deliberately small read projection. It excludes
// prompts, raw output, tool arguments, and other sensitive or unbounded blobs.
type InvocationObservation struct {
	ID               string    `json:"id"`
	PlannerVariantID string    `json:"planner_variant_id"`
	AttemptGroupID   string    `json:"attempt_group_id"`
	AttemptIndex     int       `json:"attempt_index"`
	Parsed           bool      `json:"parsed"`
	ValidationStatus string    `json:"validation_status"`
	FirstPassStatus  string    `json:"first_pass_status,omitempty"`
	EventualStatus   string    `json:"eventual_status,omitempty"`
	RetryOutcome     string    `json:"retry_outcome,omitempty"`
	WallLatencyMS    float64   `json:"wall_latency_ms"`
	InputTokens      int       `json:"input_tokens"`
	OutputTokens     int       `json:"output_tokens"`
	TotalTokens      int       `json:"total_tokens"`
	ToolRounds       int       `json:"tool_rounds"`
	CostUSD          *float64  `json:"cost_usd,omitempty"`
	PricingVersion   string    `json:"pricing_version,omitempty"`
	CreatedAt        time.Time `json:"created_at"`
}

type ObservationSet struct {
	Candidates           []CandidateObservation  `json:"candidates"`
	Invocations          []InvocationObservation `json:"invocations"`
	CandidatesTruncated  bool                    `json:"candidates_truncated"`
	InvocationsTruncated bool                    `json:"invocations_truncated"`
}

// ObservationReader exposes only one bounded read operation, so the shared
// API/CLI report path cannot mutate orchestrator state.
type ObservationReader interface {
	ReadCalibrationObservations(projectID string, window TimeWindow, limit int) (ObservationSet, error)
}

type ConfidenceInterval struct {
	Level  float64 `json:"level"`
	Lower  float64 `json:"lower"`
	Upper  float64 `json:"upper"`
	Method string  `json:"method"`
}

type MetricEstimate struct {
	Status             string              `json:"status"`
	Value              *float64            `json:"value,omitempty"`
	SampleSize         int                 `json:"sample_size"`
	ConfidenceInterval *ConfidenceInterval `json:"confidence_interval,omitempty"`
}

type CohortKey struct {
	Grouping         string `json:"grouping"`
	PlannerVariantID string `json:"planner_variant_id,omitempty"`
	Task             string `json:"task,omitempty"`
	Mechanism        string `json:"mechanism,omitempty"`
	ModelFamily      string `json:"model_family,omitempty"`
}

type StructuralMetrics struct {
	ParseSuccess                MetricEstimate `json:"parse_success"`
	FirstPassValidation         MetricEstimate `json:"first_pass_validation"`
	EventualValidation          MetricEstimate `json:"eventual_validation"`
	RetryRate                   MetricEstimate `json:"retry_rate"`
	AttemptsPerAcceptedDecision MetricEstimate `json:"attempts_per_accepted_decision"`
}

type DecisionMetrics struct {
	SelectionRate        MetricEstimate `json:"selection_rate"`
	RejectionRate        MetricEstimate `json:"rejection_rate"`
	EvidenceCoverage     MetricEstimate `json:"evidence_coverage"`
	UnsupportedClaimRate MetricEstimate `json:"unsupported_claim_rate"`
}

type ForecastMetrics struct {
	MeanAbsoluteError              MetricEstimate `json:"mean_absolute_error"`
	BiasPredictedMinusActual       MetricEstimate `json:"bias_predicted_minus_actual"`
	CorrectImprovementDirection    MetricEstimate `json:"correct_improvement_direction"`
	MeaningfulImprovementPrecision MetricEstimate `json:"meaningful_improvement_precision"`
	MeaningfulImprovementRecall    MetricEstimate `json:"meaningful_improvement_recall"`
}

type CoverageMetrics struct {
	CandidateForecasts      int            `json:"candidate_forecasts"`
	SelectedCandidates      int            `json:"selected_candidates"`
	ObservedOutcomes        int            `json:"observed_outcomes"`
	UnobservedOutcomes      int            `json:"unobserved_outcomes"`
	UnknownOutcomes         int            `json:"unknown_outcomes"`
	ObservedOutcomeCoverage MetricEstimate `json:"observed_outcome_coverage"`
	SelectedOutcomeCoverage MetricEstimate `json:"selected_outcome_coverage"`
}

type EfficiencyMetrics struct {
	WallLatencyMS            MetricEstimate `json:"wall_latency_ms"`
	TotalTokens              MetricEstimate `json:"total_tokens"`
	ToolRounds               MetricEstimate `json:"tool_rounds"`
	PlannerCostUSD           MetricEstimate `json:"planner_cost_usd"`
	DownstreamRuntimeSeconds MetricEstimate `json:"downstream_runtime_seconds"`
	DownstreamCostUSD        MetricEstimate `json:"downstream_cost_usd"`
}

type ReportCohort struct {
	Key                  CohortKey          `json:"key"`
	Status               string             `json:"status"`
	SuppressionReason    string             `json:"suppression_reason,omitempty"`
	SampleSize           int                `json:"sample_size"`
	CandidateSampleSize  int                `json:"candidate_sample_size"`
	InvocationSampleSize int                `json:"invocation_sample_size"`
	Window               TimeWindow         `json:"window"`
	ScoreVersions        []string           `json:"score_versions"`
	PricingVersions      []string           `json:"pricing_versions"`
	Structural           *StructuralMetrics `json:"structural,omitempty"`
	Decision             *DecisionMetrics   `json:"decision,omitempty"`
	Forecast             *ForecastMetrics   `json:"forecast,omitempty"`
	Coverage             *CoverageMetrics   `json:"coverage,omitempty"`
	Efficiency           *EfficiencyMetrics `json:"efficiency,omitempty"`
}

type EmpiricalPrior struct {
	Key                       CohortKey      `json:"key"`
	Status                    string         `json:"status"`
	SuppressionReason         string         `json:"suppression_reason,omitempty"`
	SampleSize                int            `json:"sample_size"`
	Window                    TimeWindow     `json:"window"`
	MeanNormalizedImprovement MetricEstimate `json:"mean_normalized_improvement"`
	MeaningfulImprovementRate MetricEstimate `json:"meaningful_improvement_rate"`
}

type ReportReadBounds struct {
	PerWindowLimit                 int  `json:"per_window_limit"`
	TrainingCandidatesReturned     int  `json:"training_candidates_returned"`
	TrainingInvocationsReturned    int  `json:"training_invocations_returned"`
	EvaluationCandidatesReturned   int  `json:"evaluation_candidates_returned"`
	EvaluationInvocationsReturned  int  `json:"evaluation_invocations_returned"`
	TrainingCandidatesTruncated    bool `json:"training_candidates_truncated"`
	TrainingInvocationsTruncated   bool `json:"training_invocations_truncated"`
	EvaluationCandidatesTruncated  bool `json:"evaluation_candidates_truncated"`
	EvaluationInvocationsTruncated bool `json:"evaluation_invocations_truncated"`
}

type Report struct {
	ReportVersion           string           `json:"report_version"`
	ProjectID               string           `json:"project_id"`
	GeneratedAt             time.Time        `json:"generated_at"`
	TrainingWindow          TimeWindow       `json:"training_window"`
	EvaluationWindow        TimeWindow       `json:"evaluation_window"`
	MinCohortSize           int              `json:"min_cohort_size"`
	MeaningfulImprovement   float64          `json:"meaningful_improvement"`
	ConfidenceLevel         float64          `json:"confidence_level"`
	GroupingDimensions      []string         `json:"grouping_dimensions"`
	ReadBounds              ReportReadBounds `json:"read_bounds"`
	CohortLimit             int              `json:"cohort_limit"`
	CohortsTruncated        bool             `json:"cohorts_truncated"`
	PriorsTruncated         bool             `json:"priors_truncated"`
	ScoreVersions           []string         `json:"score_versions"`
	PricingVersions         []string         `json:"pricing_versions"`
	Priors                  []EmpiricalPrior `json:"chronological_priors"`
	Cohorts                 []ReportCohort   `json:"evaluation_cohorts"`
	SelectionBiasDisclosure string           `json:"selection_bias_disclosure"`
	WindowDisclosure        string           `json:"window_disclosure"`
}

func NormalizeReportRequest(request ReportRequest) (ReportRequest, error) {
	request.ProjectID = strings.TrimSpace(request.ProjectID)
	if request.ProjectID == "" {
		return ReportRequest{}, fmt.Errorf("%w: project_id is required", ErrInvalidReportRequest)
	}
	request.TrainingWindow = normalizeWindow(request.TrainingWindow)
	request.EvaluationWindow = normalizeWindow(request.EvaluationWindow)
	if err := validateWindow("training", request.TrainingWindow); err != nil {
		return ReportRequest{}, err
	}
	if err := validateWindow("evaluation", request.EvaluationWindow); err != nil {
		return ReportRequest{}, err
	}
	if request.TrainingWindow.End.After(request.EvaluationWindow.Start) {
		return ReportRequest{}, fmt.Errorf("%w: training window must end at or before evaluation window starts", ErrInvalidReportRequest)
	}
	if request.Limit == 0 {
		request.Limit = DefaultCalibrationReadLimit
	}
	if request.Limit < 1 || request.Limit > MaximumCalibrationReadLimit {
		return ReportRequest{}, fmt.Errorf("%w: limit must be between 1 and %d", ErrInvalidReportRequest, MaximumCalibrationReadLimit)
	}
	if request.MinCohortSize == 0 {
		request.MinCohortSize = DefaultCalibrationMinCohortSize
	}
	if request.MinCohortSize < 1 || request.MinCohortSize > MaximumCalibrationMinCohortSize {
		return ReportRequest{}, fmt.Errorf("%w: min_cohort_size must be between 1 and %d", ErrInvalidReportRequest, MaximumCalibrationMinCohortSize)
	}
	if math.IsNaN(request.MeaningfulImprovement) || math.IsInf(request.MeaningfulImprovement, 0) || request.MeaningfulImprovement < 0 || request.MeaningfulImprovement > 1 {
		return ReportRequest{}, fmt.Errorf("%w: meaningful_improvement must be finite and between 0 and 1", ErrInvalidReportRequest)
	}
	return request, nil
}

func GenerateReport(reader ObservationReader, request ReportRequest) (Report, error) {
	request, err := NormalizeReportRequest(request)
	if err != nil {
		return Report{}, err
	}
	training, err := reader.ReadCalibrationObservations(request.ProjectID, request.TrainingWindow, request.Limit)
	if err != nil {
		return Report{}, fmt.Errorf("read training calibration observations: %w", err)
	}
	evaluation, err := reader.ReadCalibrationObservations(request.ProjectID, request.EvaluationWindow, request.Limit)
	if err != nil {
		return Report{}, fmt.Errorf("read evaluation calibration observations: %w", err)
	}
	report := BuildReport(request, training, evaluation)
	return report, nil
}

func BuildReport(request ReportRequest, training ObservationSet, evaluation ObservationSet) Report {
	request, err := NormalizeReportRequest(request)
	if err != nil {
		return Report{}
	}
	scoreVersions := candidateScoreVersions(append(append([]CandidateObservation{}, training.Candidates...), evaluation.Candidates...))
	pricingVersions := invocationPricingVersions(append(append([]InvocationObservation{}, training.Invocations...), evaluation.Invocations...))
	priors := buildEmpiricalPriors(training.Candidates, request)
	cohorts := buildEvaluationCohorts(evaluation, request)
	priorsTruncated := len(priors) > MaximumCalibrationReportCohorts
	cohortsTruncated := len(cohorts) > MaximumCalibrationReportCohorts
	if priorsTruncated {
		priors = priors[:MaximumCalibrationReportCohorts]
	}
	if cohortsTruncated {
		cohorts = cohorts[:MaximumCalibrationReportCohorts]
	}
	return Report{
		ReportVersion:         CalibrationReportVersionV1,
		ProjectID:             request.ProjectID,
		GeneratedAt:           request.EvaluationWindow.End,
		TrainingWindow:        request.TrainingWindow,
		EvaluationWindow:      request.EvaluationWindow,
		MinCohortSize:         request.MinCohortSize,
		MeaningfulImprovement: request.MeaningfulImprovement,
		ConfidenceLevel:       0.95,
		GroupingDimensions:    []string{"overall", "planner_variant", "task", "mechanism", "model_family", "task_mechanism", "task_mechanism_model_family"},
		ReadBounds: ReportReadBounds{
			PerWindowLimit:             request.Limit,
			TrainingCandidatesReturned: len(training.Candidates), TrainingInvocationsReturned: len(training.Invocations),
			EvaluationCandidatesReturned: len(evaluation.Candidates), EvaluationInvocationsReturned: len(evaluation.Invocations),
			TrainingCandidatesTruncated: training.CandidatesTruncated, TrainingInvocationsTruncated: training.InvocationsTruncated,
			EvaluationCandidatesTruncated: evaluation.CandidatesTruncated, EvaluationInvocationsTruncated: evaluation.InvocationsTruncated,
		},
		CohortLimit:             MaximumCalibrationReportCohorts,
		CohortsTruncated:        cohortsTruncated,
		PriorsTruncated:         priorsTruncated,
		ScoreVersions:           scoreVersions,
		PricingVersions:         pricingVersions,
		Priors:                  priors,
		Cohorts:                 cohorts,
		SelectionBiasDisclosure: SelectionBiasDisclosure,
		WindowDisclosure:        ChronologicalWindowDisclosure,
	}
}

func normalizeWindow(window TimeWindow) TimeWindow {
	return TimeWindow{Start: window.Start.UTC(), End: window.End.UTC()}
}

func validateWindow(name string, window TimeWindow) error {
	if window.Start.IsZero() || window.End.IsZero() || !window.Start.Before(window.End) {
		return fmt.Errorf("%w: %s window requires start before end", ErrInvalidReportRequest, name)
	}
	return nil
}

func buildEvaluationCohorts(observations ObservationSet, request ReportRequest) []ReportCohort {
	grouped := map[string]struct {
		key        CohortKey
		candidates []CandidateObservation
	}{encodeCohortKey(CohortKey{Grouping: "overall"}): {key: CohortKey{Grouping: "overall"}}}
	for _, candidate := range observations.Candidates {
		for _, key := range candidateCohortKeys(candidate) {
			encoded := encodeCohortKey(key)
			entry := grouped[encoded]
			entry.key = key
			entry.candidates = append(entry.candidates, candidate)
			grouped[encoded] = entry
		}
	}
	for _, invocation := range observations.Invocations {
		for _, key := range []CohortKey{
			{Grouping: "overall"},
			{Grouping: "planner_variant", PlannerVariantID: normalizedGroupValue(invocation.PlannerVariantID)},
		} {
			encoded := encodeCohortKey(key)
			entry := grouped[encoded]
			entry.key = key
			grouped[encoded] = entry
		}
	}
	keys := make([]string, 0, len(grouped))
	for key := range grouped {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]ReportCohort, 0, len(keys))
	for _, encoded := range keys {
		entry := grouped[encoded]
		invocations := cohortInvocationsForKey(entry.key, entry.candidates, observations.Invocations)
		invocationSampleSize := len(invocationGroups(invocations))
		sampleSize := len(entry.candidates)
		if invocationSampleSize > sampleSize {
			sampleSize = invocationSampleSize
		}
		cohort := ReportCohort{
			Key: entry.key, SampleSize: sampleSize, CandidateSampleSize: len(entry.candidates), InvocationSampleSize: invocationSampleSize,
			Window:        request.EvaluationWindow,
			ScoreVersions: candidateScoreVersions(entry.candidates),
		}
		cohort.PricingVersions = invocationPricingVersions(invocations)
		if len(entry.candidates) < request.MinCohortSize && invocationSampleSize < request.MinCohortSize {
			cohort.Status = CohortInsufficientEvidence
			cohort.SuppressionReason = fmt.Sprintf("candidate sample_size %d and invocation sample_size %d are below min_cohort_size %d", len(entry.candidates), invocationSampleSize, request.MinCohortSize)
			out = append(out, cohort)
			continue
		}
		cohort.Status = CohortAvailable
		if invocationSampleSize >= request.MinCohortSize {
			structural := computeStructuralMetricsFromInvocations(invocations)
			cohort.Structural = &structural
		}
		if len(entry.candidates) >= request.MinCohortSize {
			decision := computeDecisionMetrics(entry.candidates)
			forecast := computeForecastMetrics(entry.candidates, request.MeaningfulImprovement, request.EvaluationWindow)
			coverage := computeCoverageMetrics(entry.candidates, request.EvaluationWindow)
			cohort.Decision = &decision
			cohort.Forecast = &forecast
			cohort.Coverage = &coverage
		}
		efficiency := computeEfficiencyMetrics(entry.candidates, invocations, request.EvaluationWindow)
		suppressEfficiencyMetrics(&efficiency, request.MinCohortSize)
		cohort.Efficiency = &efficiency
		out = append(out, cohort)
	}
	return out
}

func candidateCohortKeys(candidate CandidateObservation) []CohortKey {
	variant := normalizedGroupValue(candidate.PlannerVariantID)
	task := normalizedGroupValue(candidate.Task)
	mechanism := normalizedGroupValue(candidate.Mechanism)
	family := normalizedGroupValue(candidate.ModelFamily)
	return []CohortKey{
		{Grouping: "overall"},
		{Grouping: "planner_variant", PlannerVariantID: variant},
		{Grouping: "task", Task: task},
		{Grouping: "mechanism", Mechanism: mechanism},
		{Grouping: "model_family", ModelFamily: family},
		{Grouping: "task_mechanism", Task: task, Mechanism: mechanism},
		{Grouping: "task_mechanism_model_family", Task: task, Mechanism: mechanism, ModelFamily: family},
	}
}

func normalizedGroupValue(value string) string {
	if value = strings.ToLower(strings.TrimSpace(value)); value != "" {
		return value
	}
	return "unknown"
}

func encodeCohortKey(key CohortKey) string {
	rank := map[string]string{
		"overall": "00", "planner_variant": "01", "task": "02", "mechanism": "03", "model_family": "04",
		"task_mechanism": "05", "task_mechanism_model_family": "06",
	}[key.Grouping]
	return strings.Join([]string{rank, key.Grouping, key.PlannerVariantID, key.Task, key.Mechanism, key.ModelFamily}, "\x00")
}

func computeStructuralMetricsFromInvocations(invocations []InvocationObservation) StructuralMetrics {
	parsed := 0
	for _, invocation := range invocations {
		if invocation.Parsed {
			parsed++
		}
	}
	groups := invocationGroups(invocations)
	firstPassAccepted, eventualAccepted, retries := 0, 0, 0
	attemptsAccepted := []float64{}
	for _, attempts := range groups {
		sort.Slice(attempts, func(i, j int) bool {
			if attempts[i].AttemptIndex == attempts[j].AttemptIndex {
				return attempts[i].CreatedAt.Before(attempts[j].CreatedAt)
			}
			return attempts[i].AttemptIndex < attempts[j].AttemptIndex
		})
		final := attempts[len(attempts)-1]
		first := attempts[0]
		firstStatus := first.FirstPassStatus
		if firstStatus == "" {
			firstStatus = first.ValidationStatus
		}
		if strings.EqualFold(firstStatus, "accepted") || strings.EqualFold(firstStatus, "valid") {
			firstPassAccepted++
		}
		eventualStatus := final.EventualStatus
		if eventualStatus == "" {
			eventualStatus = final.ValidationStatus
		}
		if strings.EqualFold(eventualStatus, "accepted") || strings.EqualFold(eventualStatus, "valid") {
			eventualAccepted++
			attemptsAccepted = append(attemptsAccepted, float64(len(attempts)))
		}
		if len(attempts) > 1 || (final.RetryOutcome != "" && !strings.EqualFold(final.RetryOutcome, "not_needed")) {
			retries++
		}
	}
	return StructuralMetrics{
		ParseSuccess:                rateEstimate(parsed, len(invocations)),
		FirstPassValidation:         rateEstimate(firstPassAccepted, len(groups)),
		EventualValidation:          rateEstimate(eventualAccepted, len(groups)),
		RetryRate:                   rateEstimate(retries, len(groups)),
		AttemptsPerAcceptedDecision: meanEstimate(attemptsAccepted),
	}
}

func cohortInvocationsForKey(key CohortKey, candidates []CandidateObservation, all []InvocationObservation) []InvocationObservation {
	switch key.Grouping {
	case "overall":
		return append([]InvocationObservation(nil), all...)
	case "planner_variant":
		out := []InvocationObservation{}
		for _, invocation := range all {
			if normalizedGroupValue(invocation.PlannerVariantID) == key.PlannerVariantID {
				out = append(out, invocation)
			}
		}
		return out
	default:
		return cohortInvocations(candidates, all)
	}
}

func suppressEfficiencyMetrics(metrics *EfficiencyMetrics, minSampleSize int) {
	if metrics == nil {
		return
	}
	metrics.WallLatencyMS = suppressEstimateBelow(metrics.WallLatencyMS, minSampleSize)
	metrics.TotalTokens = suppressEstimateBelow(metrics.TotalTokens, minSampleSize)
	metrics.ToolRounds = suppressEstimateBelow(metrics.ToolRounds, minSampleSize)
	metrics.PlannerCostUSD = suppressEstimateBelow(metrics.PlannerCostUSD, minSampleSize)
	metrics.DownstreamRuntimeSeconds = suppressEstimateBelow(metrics.DownstreamRuntimeSeconds, minSampleSize)
	metrics.DownstreamCostUSD = suppressEstimateBelow(metrics.DownstreamCostUSD, minSampleSize)
}

func suppressEstimateBelow(estimate MetricEstimate, minSampleSize int) MetricEstimate {
	if estimate.SampleSize >= minSampleSize || estimate.Status == EstimateNotApplicable {
		return estimate
	}
	return MetricEstimate{Status: EstimateInsufficientEvidence, SampleSize: estimate.SampleSize}
}

func computeDecisionMetrics(candidates []CandidateObservation) DecisionMetrics {
	selected, rejected, evidence := 0, 0, 0
	for _, candidate := range candidates {
		if candidate.Selected {
			selected++
		}
		if candidate.Rejected {
			rejected++
		}
		if candidate.EvidenceCount > 0 {
			evidence++
		}
	}
	return DecisionMetrics{
		SelectionRate:        rateEstimate(selected, len(candidates)),
		RejectionRate:        rateEstimate(rejected, len(candidates)),
		EvidenceCoverage:     rateEstimate(evidence, len(candidates)),
		UnsupportedClaimRate: rateEstimate(len(candidates)-evidence, len(candidates)),
	}
}

func computeForecastMetrics(candidates []CandidateObservation, meaningful float64, window TimeWindow) ForecastMetrics {
	errorsAbs := []float64{}
	biases := []float64{}
	directionCorrect := 0
	predictedMeaningful, actualMeaningful, truePositive := 0, 0, 0
	for _, candidate := range candidates {
		if !isObservedEligibleInWindow(candidate.CandidateProvenance, window) {
			continue
		}
		predicted := normalizedImprovement(candidate.Forecast.PredictedDelta, candidate.Forecast.MetricDirection)
		actual := normalizedImprovement(*candidate.ActualDelta, candidate.Forecast.MetricDirection)
		errorsAbs = append(errorsAbs, math.Abs(predicted-actual))
		biases = append(biases, predicted-actual)
		if sameDirection(predicted, actual) {
			directionCorrect++
		}
		predictedPositive := predicted >= meaningful
		actualPositive := actual >= meaningful
		if predictedPositive {
			predictedMeaningful++
		}
		if actualPositive {
			actualMeaningful++
		}
		if predictedPositive && actualPositive {
			truePositive++
		}
	}
	return ForecastMetrics{
		MeanAbsoluteError:              meanEstimate(errorsAbs),
		BiasPredictedMinusActual:       meanEstimate(biases),
		CorrectImprovementDirection:    rateEstimate(directionCorrect, len(errorsAbs)),
		MeaningfulImprovementPrecision: rateEstimate(truePositive, predictedMeaningful),
		MeaningfulImprovementRecall:    rateEstimate(truePositive, actualMeaningful),
	}
}

func computeCoverageMetrics(candidates []CandidateObservation, window TimeWindow) CoverageMetrics {
	selected, observed, unobserved, unknown := 0, 0, 0, 0
	for _, candidate := range candidates {
		if candidate.Selected {
			selected++
		}
		switch candidate.OutcomeStatus {
		case CandidateOutcomeObserved:
			if isObservedEligibleInWindow(candidate.CandidateProvenance, window) {
				observed++
			}
		case CandidateOutcomeUnobserved:
			unobserved++
		default:
			unknown++
		}
	}
	return CoverageMetrics{
		CandidateForecasts: len(candidates), SelectedCandidates: selected, ObservedOutcomes: observed,
		UnobservedOutcomes: unobserved, UnknownOutcomes: unknown,
		ObservedOutcomeCoverage: rateEstimate(observed, len(candidates)),
		SelectedOutcomeCoverage: rateEstimate(observed, selected),
	}
}

func computeEfficiencyMetrics(candidates []CandidateObservation, invocations []InvocationObservation, window TimeWindow) EfficiencyMetrics {
	latency, tokens, rounds, plannerCost := []float64{}, []float64{}, []float64{}, []float64{}
	for _, invocation := range invocations {
		if finiteInRange(invocation.WallLatencyMS, 0, math.MaxFloat64) {
			latency = append(latency, invocation.WallLatencyMS)
		}
		if invocation.TotalTokens >= 0 {
			tokens = append(tokens, float64(invocation.TotalTokens))
		}
		if invocation.ToolRounds >= 0 {
			rounds = append(rounds, float64(invocation.ToolRounds))
		}
		if invocation.CostUSD != nil && finiteInRange(*invocation.CostUSD, 0, math.MaxFloat64) {
			plannerCost = append(plannerCost, *invocation.CostUSD)
		}
	}
	downstreamRuntime, downstreamCost := []float64{}, []float64{}
	for _, candidate := range candidates {
		if candidate.FinalizedAt == nil || candidate.FinalizedAt.Before(window.Start) || !candidate.FinalizedAt.Before(window.End) {
			continue
		}
		if candidate.RuntimeSeconds != nil && finiteInRange(*candidate.RuntimeSeconds, 0, math.MaxFloat64) {
			downstreamRuntime = append(downstreamRuntime, *candidate.RuntimeSeconds)
		}
		if candidate.CostUSD != nil && finiteInRange(*candidate.CostUSD, 0, math.MaxFloat64) {
			downstreamCost = append(downstreamCost, *candidate.CostUSD)
		}
	}
	return EfficiencyMetrics{
		WallLatencyMS: meanEstimate(latency), TotalTokens: meanEstimate(tokens), ToolRounds: meanEstimate(rounds),
		PlannerCostUSD: meanEstimate(plannerCost), DownstreamRuntimeSeconds: meanEstimate(downstreamRuntime),
		DownstreamCostUSD: meanEstimate(downstreamCost),
	}
}

func buildEmpiricalPriors(candidates []CandidateObservation, request ReportRequest) []EmpiricalPrior {
	groups := map[string]struct {
		key    CohortKey
		values []float64
	}{encodeCohortKey(CohortKey{Grouping: "overall"}): {key: CohortKey{Grouping: "overall"}}}
	for _, candidate := range candidates {
		if !isObservedEligibleInWindow(candidate.CandidateProvenance, request.TrainingWindow) {
			continue
		}
		value := normalizedImprovement(*candidate.ActualDelta, candidate.Forecast.MetricDirection)
		keys := candidateCohortKeys(candidate)
		for _, key := range keys {
			if key.Grouping != "overall" && key.Grouping != "task" && key.Grouping != "task_mechanism" && key.Grouping != "task_mechanism_model_family" {
				continue
			}
			encoded := encodeCohortKey(key)
			entry := groups[encoded]
			entry.key = key
			entry.values = append(entry.values, value)
			groups[encoded] = entry
		}
	}
	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	out := make([]EmpiricalPrior, 0, len(keys))
	for _, encoded := range keys {
		entry := groups[encoded]
		prior := EmpiricalPrior{Key: entry.key, SampleSize: len(entry.values), Window: request.TrainingWindow}
		if len(entry.values) < request.MinCohortSize {
			prior.Status = CohortInsufficientEvidence
			prior.SuppressionReason = fmt.Sprintf("observed sample_size %d is below min_cohort_size %d", len(entry.values), request.MinCohortSize)
			prior.MeanNormalizedImprovement = notApplicableEstimate(len(entry.values))
			prior.MeaningfulImprovementRate = notApplicableEstimate(len(entry.values))
		} else {
			meaningfulCount := 0
			for _, value := range entry.values {
				if value >= request.MeaningfulImprovement {
					meaningfulCount++
				}
			}
			prior.Status = CohortAvailable
			prior.MeanNormalizedImprovement = meanEstimate(entry.values)
			prior.MeaningfulImprovementRate = rateEstimate(meaningfulCount, len(entry.values))
		}
		out = append(out, prior)
	}
	return out
}

func cohortInvocations(candidates []CandidateObservation, all []InvocationObservation) []InvocationObservation {
	acceptedIDs := map[string]bool{}
	groupIDs := map[string]bool{}
	byID := make(map[string]InvocationObservation, len(all))
	for _, invocation := range all {
		byID[invocation.ID] = invocation
	}
	for _, candidate := range candidates {
		acceptedIDs[candidate.InvocationID] = true
		if invocation, ok := byID[candidate.InvocationID]; ok {
			groupIDs[invocationGroupKey(invocation)] = true
		}
	}
	out := []InvocationObservation{}
	seen := map[string]bool{}
	for _, invocation := range all {
		if !acceptedIDs[invocation.ID] && !groupIDs[invocationGroupKey(invocation)] {
			continue
		}
		if seen[invocation.ID] {
			continue
		}
		seen[invocation.ID] = true
		out = append(out, invocation)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CreatedAt.Equal(out[j].CreatedAt) {
			return out[i].ID < out[j].ID
		}
		return out[i].CreatedAt.Before(out[j].CreatedAt)
	})
	return out
}

func invocationGroups(invocations []InvocationObservation) map[string][]InvocationObservation {
	out := map[string][]InvocationObservation{}
	for _, invocation := range invocations {
		key := invocationGroupKey(invocation)
		out[key] = append(out[key], invocation)
	}
	return out
}

func invocationGroupKey(invocation InvocationObservation) string {
	if value := strings.TrimSpace(invocation.AttemptGroupID); value != "" {
		return value
	}
	return invocation.ID
}

func normalizedImprovement(delta float64, direction string) float64 {
	if direction == MetricDirectionLowerIsBetter {
		return -delta
	}
	return delta
}

func sameDirection(predicted, actual float64) bool {
	const epsilon = 1e-12
	switch {
	case math.Abs(predicted) <= epsilon:
		return math.Abs(actual) <= epsilon
	case predicted > 0:
		return actual > 0
	default:
		return actual < 0
	}
}

func isObservedEligibleInWindow(candidate CandidateProvenance, window TimeWindow) bool {
	return candidate.Selected && candidate.OutcomeStatus == CandidateOutcomeObserved && candidate.ActualDelta != nil &&
		candidate.CalibrationEligible != nil && *candidate.CalibrationEligible && candidate.FinalizedAt != nil &&
		!candidate.FinalizedAt.Before(window.Start) && candidate.FinalizedAt.Before(window.End)
}

func rateEstimate(successes, total int) MetricEstimate {
	if total <= 0 {
		return notApplicableEstimate(0)
	}
	value := float64(successes) / float64(total)
	interval := wilson95(successes, total)
	return MetricEstimate{Status: EstimateAvailable, Value: &value, SampleSize: total, ConfidenceInterval: &interval}
}

func meanEstimate(values []float64) MetricEstimate {
	if len(values) == 0 {
		return notApplicableEstimate(0)
	}
	mean := 0.0
	for _, value := range values {
		mean += value
	}
	mean /= float64(len(values))
	interval := ConfidenceInterval{Level: 0.95, Lower: mean, Upper: mean, Method: ConfidenceMethodMean95}
	if len(values) > 1 {
		variance := 0.0
		for _, value := range values {
			variance += (value - mean) * (value - mean)
		}
		variance /= float64(len(values) - 1)
		margin := 1.96 * math.Sqrt(variance/float64(len(values)))
		interval.Lower = mean - margin
		interval.Upper = mean + margin
	}
	return MetricEstimate{Status: EstimateAvailable, Value: &mean, SampleSize: len(values), ConfidenceInterval: &interval}
}

func notApplicableEstimate(sampleSize int) MetricEstimate {
	return MetricEstimate{Status: EstimateNotApplicable, SampleSize: sampleSize}
}

func wilson95(successes, total int) ConfidenceInterval {
	if total <= 0 {
		return ConfidenceInterval{Level: 0.95, Method: ConfidenceMethodWilson95}
	}
	z := 1.96
	n := float64(total)
	p := float64(successes) / n
	denominator := 1 + z*z/n
	center := (p + z*z/(2*n)) / denominator
	margin := z * math.Sqrt((p*(1-p)+z*z/(4*n))/n) / denominator
	return ConfidenceInterval{Level: 0.95, Lower: math.Max(0, center-margin), Upper: math.Min(1, center+margin), Method: ConfidenceMethodWilson95}
}

func candidateScoreVersions(candidates []CandidateObservation) []string {
	values := []string{}
	for _, candidate := range candidates {
		values = append(values, candidate.Forecast.ScoreVersion)
	}
	return sortedUnique(values)
}

func invocationPricingVersions(invocations []InvocationObservation) []string {
	values := []string{}
	for _, invocation := range invocations {
		values = append(values, invocation.PricingVersion)
	}
	return sortedUnique(values)
}

func sortedUnique(values []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

// NormalizeModelFamily keeps report/ranker cohort keys stable without importing
// the planner package (which already depends on calibration).
func NormalizeModelFamily(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	for _, prefix := range []string{"mobilenet", "efficientnet", "regnet", "resnet", "convnext", "swin", "vit", "densenet", "yolo"} {
		if strings.HasPrefix(normalized, prefix) {
			return prefix
		}
	}
	if normalized == "" {
		return "unknown"
	}
	return normalized
}

// CandidateDecisionMetadata extracts only cohort-safe fields from the immutable
// decision payload. Malformed legacy payloads safely return unknown/zero.
func CandidateDecisionMetadata(payload map[string]any, candidateIndex int) (string, int) {
	if candidateIndex < 0 {
		return "unknown", 0
	}
	encoded, err := json.Marshal(payload["candidate_hypotheses"])
	if err != nil {
		return "unknown", 0
	}
	var items []struct {
		EvidenceUsed     []string `json:"evidence_used"`
		ExperimentConfig struct {
			Model string `json:"model"`
		} `json:"experiment_config"`
	}
	if err := json.Unmarshal(encoded, &items); err != nil || candidateIndex >= len(items) {
		return "unknown", 0
	}
	evidenceCount := 0
	for _, entry := range items[candidateIndex].EvidenceUsed {
		if strings.TrimSpace(entry) != "" {
			evidenceCount++
		}
	}
	return NormalizeModelFamily(items[candidateIndex].ExperimentConfig.Model), evidenceCount
}

func parseDecimalCost(value string) *float64 {
	parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	if err != nil || !finiteInRange(parsed, 0, math.MaxFloat64) {
		return nil
	}
	return &parsed
}
