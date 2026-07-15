package calibration

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

const (
	RankerV2PriorSnapshotVersionV1 = "ranker_v2_prior_snapshot_v1"
	RankerV2PriorMinSampleSize     = 5
	RankerV2PriorPseudoCount       = 4
	RankerV2PriorWinsorLimit       = 0.10
)

type RankerV2PriorRecord struct {
	Key                        CohortKey `json:"key"`
	SampleSize                 int       `json:"sample_size"`
	MeaningfulImprovementCount int       `json:"meaningful_improvement_count"`
	WinsorizedMeanImprovement  float64   `json:"winsorized_mean_improvement"`
	SmoothedMeanImprovement    float64   `json:"smoothed_mean_improvement"`
	SmoothedImprovementRate    float64   `json:"smoothed_improvement_rate"`
}

// RankerV2PriorSnapshot freezes every empirical input used by a shadow score.
// EvaluationWindow is metadata only; records always come from TrainingWindow.
type RankerV2PriorSnapshot struct {
	SnapshotVersion        string                `json:"snapshot_version"`
	TrainingWindow         TimeWindow            `json:"training_window"`
	EvaluationWindow       TimeWindow            `json:"evaluation_window"`
	MinSampleSize          int                   `json:"min_sample_size"`
	MeaningfulImprovement  float64               `json:"meaningful_improvement"`
	PseudoCount            int                   `json:"pseudo_count"`
	ImprovementWinsorLimit float64               `json:"improvement_winsor_limit"`
	SourceReadTruncated    bool                  `json:"source_read_truncated"`
	SourceStatus           string                `json:"source_status"`
	Records                []RankerV2PriorRecord `json:"records"`
}

type RankerV2PriorSelection struct {
	SourceLevel             string    `json:"source_level"`
	Key                     CohortKey `json:"key"`
	SampleSize              int       `json:"sample_size"`
	SmoothedMeanImprovement float64   `json:"smoothed_mean_improvement"`
	SmoothedImprovementRate float64   `json:"smoothed_improvement_rate"`
	FallbackReason          string    `json:"fallback_reason"`
}

func BuildRankerV2PriorSnapshot(
	candidates []CandidateObservation,
	trainingWindow TimeWindow,
	evaluationWindow TimeWindow,
	minSampleSize int,
	meaningfulImprovement float64,
	readTruncated bool,
) (RankerV2PriorSnapshot, error) {
	trainingWindow = normalizeWindow(trainingWindow)
	evaluationWindow = normalizeWindow(evaluationWindow)
	if err := validateWindow("ranker prior training", trainingWindow); err != nil {
		return RankerV2PriorSnapshot{}, err
	}
	if err := validateWindow("ranker shadow evaluation", evaluationWindow); err != nil {
		return RankerV2PriorSnapshot{}, err
	}
	if trainingWindow.End.After(evaluationWindow.Start) {
		return RankerV2PriorSnapshot{}, fmt.Errorf("ranker prior training window must end at or before shadow evaluation starts")
	}
	if minSampleSize <= 0 {
		minSampleSize = RankerV2PriorMinSampleSize
	}
	if minSampleSize > MaximumCalibrationMinCohortSize {
		return RankerV2PriorSnapshot{}, fmt.Errorf("ranker prior min sample size must be at most %d", MaximumCalibrationMinCohortSize)
	}
	if math.IsNaN(meaningfulImprovement) || math.IsInf(meaningfulImprovement, 0) || meaningfulImprovement < 0 || meaningfulImprovement > 1 {
		return RankerV2PriorSnapshot{}, fmt.Errorf("ranker prior meaningful improvement must be finite and between 0 and 1")
	}

	groups := map[string]struct {
		key       CohortKey
		values    []float64
		successes int
	}{encodeCohortKey(CohortKey{Grouping: "overall"}): {key: CohortKey{Grouping: "overall"}}}
	for _, candidate := range candidates {
		if !isObservedEligibleInWindow(candidate.CandidateProvenance, trainingWindow) {
			continue
		}
		rawValue := normalizedImprovement(*candidate.ActualDelta, candidate.Forecast.MetricDirection)
		value := math.Max(-RankerV2PriorWinsorLimit, math.Min(RankerV2PriorWinsorLimit, rawValue))
		for _, key := range candidateCohortKeys(candidate) {
			if key.Grouping != "overall" && key.Grouping != "task" && key.Grouping != "task_mechanism" && key.Grouping != "task_mechanism_model_family" {
				continue
			}
			encoded := encodeCohortKey(key)
			entry := groups[encoded]
			entry.key = key
			entry.values = append(entry.values, value)
			if rawValue >= meaningfulImprovement {
				entry.successes++
			}
			groups[encoded] = entry
		}
	}
	keys := make([]string, 0, len(groups))
	for key := range groups {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	records := make([]RankerV2PriorRecord, 0, len(keys))
	for _, encoded := range keys {
		entry := groups[encoded]
		mean := 0.0
		for _, value := range entry.values {
			mean += value
		}
		if len(entry.values) > 0 {
			mean /= float64(len(entry.values))
		}
		records = append(records, RankerV2PriorRecord{
			Key: entry.key, SampleSize: len(entry.values), MeaningfulImprovementCount: entry.successes,
			WinsorizedMeanImprovement: mean,
			SmoothedMeanImprovement:   mean * float64(len(entry.values)) / float64(len(entry.values)+RankerV2PriorPseudoCount),
			SmoothedImprovementRate:   float64(entry.successes+RankerV2PriorPseudoCount/2) / float64(len(entry.values)+RankerV2PriorPseudoCount),
		})
	}
	return RankerV2PriorSnapshot{
		SnapshotVersion: RankerV2PriorSnapshotVersionV1,
		TrainingWindow:  trainingWindow, EvaluationWindow: evaluationWindow,
		MinSampleSize: minSampleSize, MeaningfulImprovement: meaningfulImprovement,
		PseudoCount: RankerV2PriorPseudoCount, ImprovementWinsorLimit: RankerV2PriorWinsorLimit,
		SourceReadTruncated: readTruncated, SourceStatus: "available", Records: records,
	}, nil
}

func SelectRankerV2Prior(snapshot RankerV2PriorSnapshot, task, mechanism, modelFamily string) RankerV2PriorSelection {
	if !validRankerV2PriorSnapshot(snapshot) {
		return neutralRankerV2Prior("snapshot_unavailable_or_invalid")
	}
	task = normalizedGroupValue(task)
	mechanism = normalizedGroupValue(mechanism)
	modelFamily = normalizedGroupValue(modelFamily)
	candidates := []CohortKey{
		{Grouping: "task_mechanism_model_family", Task: task, Mechanism: mechanism, ModelFamily: modelFamily},
		{Grouping: "task_mechanism", Task: task, Mechanism: mechanism},
		{Grouping: "task", Task: task},
		{Grouping: "overall"},
	}
	byKey := make(map[string]RankerV2PriorRecord, len(snapshot.Records))
	for _, record := range snapshot.Records {
		byKey[encodeCohortKey(record.Key)] = record
	}
	for index, key := range candidates {
		record, ok := byKey[encodeCohortKey(key)]
		if !ok || record.SampleSize < snapshot.MinSampleSize {
			continue
		}
		reason := "narrow cohort met minimum sample size"
		if index > 0 {
			reason = "narrower cohort was missing or undersized; used " + strings.ReplaceAll(key.Grouping, "_", " ")
		}
		return RankerV2PriorSelection{
			SourceLevel: key.Grouping, Key: key, SampleSize: record.SampleSize,
			SmoothedMeanImprovement: record.SmoothedMeanImprovement,
			SmoothedImprovementRate: record.SmoothedImprovementRate,
			FallbackReason:          reason,
		}
	}
	return neutralRankerV2Prior("all empirical cohorts were missing or below min_sample_size")
}

func validRankerV2PriorSnapshot(snapshot RankerV2PriorSnapshot) bool {
	if snapshot.SnapshotVersion != RankerV2PriorSnapshotVersionV1 || snapshot.MinSampleSize <= 0 ||
		snapshot.TrainingWindow.Start.IsZero() || snapshot.TrainingWindow.End.IsZero() || !snapshot.TrainingWindow.Start.Before(snapshot.TrainingWindow.End) ||
		snapshot.EvaluationWindow.Start.IsZero() || snapshot.EvaluationWindow.End.IsZero() || !snapshot.EvaluationWindow.Start.Before(snapshot.EvaluationWindow.End) ||
		snapshot.TrainingWindow.End.After(snapshot.EvaluationWindow.Start) {
		return false
	}
	for _, record := range snapshot.Records {
		if record.SampleSize < 0 || record.MeaningfulImprovementCount < 0 || record.MeaningfulImprovementCount > record.SampleSize ||
			math.IsNaN(record.SmoothedMeanImprovement) || math.IsInf(record.SmoothedMeanImprovement, 0) ||
			math.IsNaN(record.SmoothedImprovementRate) || math.IsInf(record.SmoothedImprovementRate, 0) ||
			record.SmoothedImprovementRate < 0 || record.SmoothedImprovementRate > 1 {
			return false
		}
	}
	return true
}

func neutralRankerV2Prior(reason string) RankerV2PriorSelection {
	return RankerV2PriorSelection{
		SourceLevel: "neutral", Key: CohortKey{Grouping: "neutral"},
		SmoothedImprovementRate: 0.5, FallbackReason: reason,
	}
}
