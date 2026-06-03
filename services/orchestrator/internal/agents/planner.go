package agents

import (
	"fmt"
	"math"
	"strings"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
)

const (
	PriorityBalanced     = "balanced"
	PriorityFastTraining = "fast_training"
	PriorityLowCost      = "low_cost"
	PriorityBestQuality  = "best_quality"
)

type PlanPreferences struct {
	Priority          string
	MaxWorkers        int
	TimeBudgetMinutes int
	TargetMetric      string
}

type PlanRecommendation struct {
	TargetMetric       string
	RecommendedWorkers int
	EstimatedMinutes   int
	Experiments        []plans.PlannedExperiment
	Warnings           []string
}

type DatasetPlanner struct{}

func NewDatasetPlanner() DatasetPlanner {
	return DatasetPlanner{}
}

func (p DatasetPlanner) BuildExperimentPlan(project projects.Project, dataset datasets.Dataset, preferences PlanPreferences) (PlanRecommendation, error) {
	if dataset.ProjectID != project.ID {
		return PlanRecommendation{}, fmt.Errorf("dataset does not belong to project")
	}
	if dataset.Status != datasets.StatusProfiled {
		return PlanRecommendation{}, fmt.Errorf("dataset must be profiled before planning")
	}
	if len(dataset.Profile) == 0 {
		return PlanRecommendation{}, fmt.Errorf("dataset profile is empty")
	}

	priority := normalizePriority(preferences.Priority)
	totalImages := intFromProfile(dataset.Profile, "total_images")
	classCount := intFromProfile(dataset.Profile, "class_count")
	imbalanceRatio := floatFromProfile(dataset.Profile, "imbalance_ratio")
	corruptImageCount := intFromProfile(dataset.Profile, "corrupt_image_count")
	detectionTask := profileIsYOLODetection(dataset.Profile)

	if totalImages <= 0 {
		return PlanRecommendation{}, fmt.Errorf("dataset profile has no images")
	}
	if !detectionTask && classCount < 2 {
		return PlanRecommendation{}, fmt.Errorf("classification planning requires at least two classes")
	}
	if detectionTask && classCount < 1 {
		return PlanRecommendation{}, fmt.Errorf("object-detection planning requires at least one class")
	}

	targetMetric := strings.TrimSpace(preferences.TargetMetric)
	if targetMetric == "" {
		if detectionTask {
			targetMetric = "mAP50_95"
		} else {
			targetMetric = chooseTargetMetric(imbalanceRatio)
		}
	}

	experiments := plannedExperiments(priority, totalImages, classCount)
	if detectionTask {
		experiments = plannedDetectionExperiments(priority, totalImages)
	}
	maxWorkers := preferences.MaxWorkers
	if maxWorkers <= 0 {
		maxWorkers = defaultWorkerCap(priority)
	}
	recommendedWorkers := minInt(maxWorkers, len(experiments))
	if priority == PriorityLowCost {
		recommendedWorkers = 1
	}
	if recommendedWorkers < 1 {
		recommendedWorkers = 1
	}

	estimatedMinutes := estimateMinutes(totalImages, experiments, recommendedWorkers)
	if preferences.TimeBudgetMinutes > 0 && estimatedMinutes > preferences.TimeBudgetMinutes {
		experiments = trimToBudget(experiments, priority)
		recommendedWorkers = minInt(recommendedWorkers, len(experiments))
		estimatedMinutes = estimateMinutes(totalImages, experiments, recommendedWorkers)
	}

	warnings := buildPlanningWarnings(project, totalImages, classCount, imbalanceRatio, corruptImageCount)
	if detectionTask {
		warnings = append(warnings, "YOLO object-detection evidence detected; initial plan preserves bounding-box supervision and official YOLO splits.")
	}

	return PlanRecommendation{
		TargetMetric:       targetMetric,
		RecommendedWorkers: recommendedWorkers,
		EstimatedMinutes:   estimatedMinutes,
		Experiments:        experiments,
		Warnings:           warnings,
	}, nil
}

func plannedDetectionExperiments(priority string, totalImages int) []plans.PlannedExperiment {
	baselineEpochs := 10
	challengerEpochs := 14
	if totalImages < 250 {
		baselineEpochs = 8
		challengerEpochs = 10
	}

	nano := plans.PlannedExperiment{
		Template:       "yolo11_detection",
		Model:          "yolo11n.pt",
		Mechanism:      "baseline_control",
		Intervention:   "Train a COCO-pretrained YOLO11 nano detector on the official YOLO splits.",
		EvidenceUsed:   []string{"dataset profile reports YOLO object-detection labels"},
		ExpectedEffect: "Establish a fast live-stream detector baseline with bounding-box supervision.",
		Epochs:         baselineEpochs,
		BatchSize:      8,
		LearningRate:   0.001,
		ImageSize:      640,
		Pretrained:     true,
		Reason:         "Fast YOLO detector baseline for an object-detection dataset.",
	}
	small := plans.PlannedExperiment{
		Template:       "yolo11_detection",
		Model:          "yolo11s.pt",
		Mechanism:      "architecture_challenge",
		Intervention:   "Compare YOLO11 small against the nano detector while preserving the same YOLO splits.",
		EvidenceUsed:   []string{"dataset profile reports YOLO object-detection labels"},
		ExpectedEffect: "Improve mAP/recall while staying live-friendly.",
		Epochs:         challengerEpochs,
		BatchSize:      8,
		LearningRate:   0.001,
		ImageSize:      640,
		Pretrained:     true,
		Reason:         "Small YOLO detector challenger for better mAP with moderate latency.",
	}
	medium := plans.PlannedExperiment{
		Template:       "yolo11_detection",
		Model:          "yolo11m.pt",
		Mechanism:      "architecture_challenge",
		Intervention:   "Run a YOLO11 medium quality challenger when the budget allows.",
		EvidenceUsed:   []string{"dataset profile reports YOLO object-detection labels"},
		ExpectedEffect: "Test whether additional detector capacity improves mAP enough to justify latency.",
		Epochs:         challengerEpochs,
		BatchSize:      6,
		LearningRate:   0.0008,
		ImageSize:      640,
		Pretrained:     true,
		Reason:         "Medium YOLO detector quality challenger.",
	}

	switch priority {
	case PriorityFastTraining, PriorityLowCost:
		return []plans.PlannedExperiment{nano}
	case PriorityBestQuality:
		return []plans.PlannedExperiment{nano, small, medium}
	default:
		return []plans.PlannedExperiment{nano, small}
	}
}

func plannedExperiments(priority string, totalImages int, classCount int) []plans.PlannedExperiment {
	baselineEpochs := 8
	qualityEpochs := 12
	if totalImages < 150 {
		baselineEpochs = 6
		qualityEpochs = 10
	}

	mobilenet := plans.PlannedExperiment{
		Template:     "mobilenet_transfer",
		Model:        "mobilenet_v3_small",
		Epochs:       baselineEpochs,
		BatchSize:    16,
		LearningRate: 0.0003,
		Reason:       "Fast transfer-learning baseline for an image classification dataset.",
	}

	efficientnet := plans.PlannedExperiment{
		Template:     "efficientnet_transfer",
		Model:        "efficientnet_b0",
		Epochs:       qualityEpochs,
		BatchSize:    12,
		LearningRate: 0.0002,
		Reason:       "Quality-focused transfer-learning candidate with moderate compute cost.",
	}

	resnet := plans.PlannedExperiment{
		Template:     "resnet_transfer",
		Model:        "resnet18",
		Epochs:       qualityEpochs,
		BatchSize:    12,
		LearningRate: 0.0002,
		Reason:       "Secondary quality candidate to compare architecture behavior against EfficientNet.",
	}

	switch priority {
	case PriorityFastTraining, PriorityLowCost:
		return []plans.PlannedExperiment{mobilenet}
	case PriorityBestQuality:
		if totalImages < classCount*40 {
			return []plans.PlannedExperiment{mobilenet, efficientnet}
		}
		return []plans.PlannedExperiment{mobilenet, efficientnet, resnet}
	default:
		return []plans.PlannedExperiment{mobilenet, efficientnet}
	}
}

func profileIsYOLODetection(profile map[string]any) bool {
	if strings.EqualFold(stringFromProfile(profile, "task_type"), "object_detection") && profileHasYOLOEvidence(profile) {
		return true
	}
	return profileHasYOLOEvidence(profile)
}

func profileHasYOLOEvidence(profile map[string]any) bool {
	if boolFromProfile(profile, "yolo_available") {
		return true
	}
	if summary, ok := profile["metadata_summary"].(map[string]any); ok && profileHasYOLOEvidence(summary) {
		return true
	}
	if summary, ok := profile["yolo_summary"].(map[string]any); ok {
		if boolFromProfile(summary, "available") || strings.EqualFold(stringFromProfile(summary, "format"), "yolo") || intFromProfile(summary, "label_file_count") > 0 {
			return true
		}
	}
	if counts, ok := profile["artifact_counts"].(map[string]any); ok {
		return intFromProfile(counts, "yolo_dataset_config") > 0 || intFromProfile(counts, "yolo_label_file") > 0
	}
	return false
}

func chooseTargetMetric(imbalanceRatio float64) string {
	if imbalanceRatio >= 1.5 {
		return "macro_f1"
	}
	return "accuracy"
}

func defaultWorkerCap(priority string) int {
	switch priority {
	case PriorityLowCost:
		return 1
	case PriorityFastTraining:
		return 2
	case PriorityBestQuality:
		return 3
	default:
		return 2
	}
}

func estimateMinutes(totalImages int, experiments []plans.PlannedExperiment, workers int) int {
	if workers < 1 {
		workers = 1
	}

	totalEpochs := 0
	for _, experiment := range experiments {
		totalEpochs += experiment.Epochs
	}

	workUnits := float64(totalImages * totalEpochs)
	minutes := int(math.Ceil(workUnits/850.0/float64(workers))) + len(experiments)*2
	if minutes < 5 {
		return 5
	}
	return minutes
}

func trimToBudget(experiments []plans.PlannedExperiment, priority string) []plans.PlannedExperiment {
	if len(experiments) <= 1 {
		return experiments
	}
	if priority == PriorityBestQuality {
		return experiments[:2]
	}
	return experiments[:1]
}

func buildPlanningWarnings(project projects.Project, totalImages int, classCount int, imbalanceRatio float64, corruptImageCount int) []string {
	warnings := []string{}

	if strings.TrimSpace(project.Goal) == "" {
		warnings = append(warnings, "No project goal was provided; the plan is based only on the dataset profile.")
	}
	if totalImages < 100 {
		warnings = append(warnings, "Dataset is very small; transfer learning and conservative early stopping are recommended.")
	}
	if classCount > 0 && totalImages/classCount < 30 {
		warnings = append(warnings, "Some classes may have few examples; validation metrics may be noisy.")
	}
	if imbalanceRatio >= 2 {
		warnings = append(warnings, "Class imbalance detected; macro_f1 should be preferred over raw accuracy.")
	}
	if corruptImageCount > 0 {
		warnings = append(warnings, "Corrupt images were detected during profiling and should be excluded from training.")
	}

	return warnings
}

func normalizePriority(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case PriorityFastTraining:
		return PriorityFastTraining
	case PriorityLowCost:
		return PriorityLowCost
	case PriorityBestQuality:
		return PriorityBestQuality
	default:
		return PriorityBalanced
	}
}

func intFromProfile(profile map[string]any, key string) int {
	switch value := profile[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	case jsonNumber:
		out, _ := value.Int64()
		return int(out)
	default:
		return 0
	}
}

func stringFromProfile(profile map[string]any, key string) string {
	value, _ := profile[key].(string)
	return strings.TrimSpace(value)
}

func boolFromProfile(profile map[string]any, key string) bool {
	switch value := profile[key].(type) {
	case bool:
		return value
	case string:
		return strings.EqualFold(strings.TrimSpace(value), "true")
	default:
		return false
	}
}

func floatFromProfile(profile map[string]any, key string) float64 {
	switch value := profile[key].(type) {
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

func minInt(a int, b int) int {
	if a < b {
		return a
	}
	return b
}

type jsonNumber interface {
	Float64() (float64, error)
	Int64() (int64, error)
}
