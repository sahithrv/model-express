package calibration

import (
	"math"
	"testing"
	"time"
)

func TestRankerV2PriorFallbacksAndChronologicalWindows(t *testing.T) {
	request := testReportRequest()
	candidates := []CandidateObservation{}
	for index := 0; index < 5; index++ {
		candidate := testCandidateObservation(index, "inv", MetricDirectionHigherIsBetter, 0.02, 0.03, true, true, 1)
		candidate.Task = "image_classification"
		candidate.Mechanism = "class_imbalance"
		candidate.ModelFamily = "resnet"
		setPriorFinalizedAt(&candidate, request.TrainingWindow.End.Add(-time.Hour))
		candidates = append(candidates, candidate)
	}
	// This mechanism is too small, so it must fall back to the task cohort.
	small := testCandidateObservation(10, "small", MetricDirectionHigherIsBetter, 0.02, 0.01, true, true, 1)
	small.Task = "image_classification"
	small.Mechanism = "augmentation"
	small.ModelFamily = "convnext"
	setPriorFinalizedAt(&small, request.TrainingWindow.End.Add(-time.Hour))
	candidates = append(candidates, small)
	snapshot, err := BuildRankerV2PriorSnapshot(candidates, request.TrainingWindow, request.EvaluationWindow, 5, 0.01, false)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.TrainingWindow.End.After(snapshot.EvaluationWindow.Start) || snapshot.SnapshotVersion != RankerV2PriorSnapshotVersionV1 {
		t.Fatalf("snapshot windows/version=%#v", snapshot)
	}
	narrow := SelectRankerV2Prior(snapshot, "image_classification", "class_imbalance", "resnet")
	if narrow.SourceLevel != "task_mechanism_model_family" || narrow.SampleSize != 5 {
		t.Fatalf("narrow prior=%#v", narrow)
	}
	fallback := SelectRankerV2Prior(snapshot, "image_classification", "augmentation", "convnext")
	if fallback.SourceLevel != "task" || fallback.SampleSize != 6 {
		t.Fatalf("task fallback=%#v", fallback)
	}
	neutral := SelectRankerV2Prior(snapshot, "object_detection", "resolution", "yolo")
	if neutral.SourceLevel != "overall" || neutral.SampleSize != 6 {
		t.Fatalf("broad fallback=%#v", neutral)
	}
	empty := SelectRankerV2Prior(RankerV2PriorSnapshot{}, "image_classification", "augmentation", "resnet")
	if empty.SourceLevel != "neutral" || empty.SmoothedImprovementRate != 0.5 {
		t.Fatalf("empty fallback=%#v", empty)
	}
}

func TestRankerV2PriorWinsorizationResistsSingleOutlier(t *testing.T) {
	request := testReportRequest()
	candidates := []CandidateObservation{}
	for index, delta := range []float64{0.01, 0.01, 0.01, 0.01, 1.0} {
		candidate := testCandidateObservation(index, "outlier", MetricDirectionHigherIsBetter, 0.02, delta, true, true, 1)
		setPriorFinalizedAt(&candidate, request.TrainingWindow.End.Add(-time.Hour))
		candidates = append(candidates, candidate)
	}
	snapshot, err := BuildRankerV2PriorSnapshot(candidates, request.TrainingWindow, request.EvaluationWindow, 5, 0.01, false)
	if err != nil {
		t.Fatal(err)
	}
	prior := SelectRankerV2Prior(snapshot, "image_classification", "class_imbalance", "resnet")
	// (4*0.01 + 0.10) / 5 = 0.028 before smoothing; the raw 1.0 cannot dominate.
	want := 0.028 * 5.0 / 9.0
	if math.Abs(prior.SmoothedMeanImprovement-want) > 1e-9 || prior.SmoothedMeanImprovement >= 0.02 {
		t.Fatalf("outlier-resistant prior=%#v want mean=%f", prior, want)
	}
}

func setPriorFinalizedAt(candidate *CandidateObservation, value time.Time) {
	candidate.FinalizedAt = &value
}

func TestRankerV2PriorRejectsOverlappingWindows(t *testing.T) {
	request := testReportRequest()
	training := request.TrainingWindow
	training.End = request.EvaluationWindow.Start.Add(1)
	if _, err := BuildRankerV2PriorSnapshot(nil, training, request.EvaluationWindow, 5, 0.01, false); err == nil {
		t.Fatal("expected overlapping prior/evaluation windows to fail")
	}
}
