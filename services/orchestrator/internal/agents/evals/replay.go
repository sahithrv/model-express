package evals

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
)

type PlannerReplayFixture struct {
	Name         string                `json:"name"`
	Input        map[string]any        `json:"input"`
	InputSummary map[string]any        `json:"input_summary,omitempty"`
	Response     map[string]any        `json:"response,omitempty"`
	Expected     PlannerReplayExpected `json:"expected"`
}

type PlannerReplayExpected struct {
	ForbiddenMechanisms            []string `json:"forbidden_mechanisms"`
	AllowedDecisions               []string `json:"allowed_decisions"`
	AllowedAddExperimentMechanisms []string `json:"allowed_add_experiment_mechanisms"`
	MaxSelectedExperiments         int      `json:"max_selected_experiments"`
}

func LoadPlannerReplayFixture(path string) (PlannerReplayFixture, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return PlannerReplayFixture{}, err
	}
	var fixture PlannerReplayFixture
	if err := json.Unmarshal(blob, &fixture); err != nil {
		return PlannerReplayFixture{}, err
	}
	return fixture, nil
}

func ExperimentPlannerInputFromReplayFixture(fixture PlannerReplayFixture) agents.ExperimentPlannerInput {
	summary := fixture.InputSummary
	if len(summary) == 0 {
		if nested, ok := fixture.Input["input_summary"].(map[string]any); ok {
			summary = nested
		}
	}

	taskType := replayString(summary, "task_type", "image_classification")
	targetMetric := replayString(summary, "target_metric", "")
	if targetMetric == "" {
		if taskType == "object_detection" {
			targetMetric = "mAP50_95"
		} else {
			targetMetric = "macro_f1"
		}
	}

	firstScore := replayFloat(summary, "first_successful_macro_f1", 0.701)
	currentBest := replayFloat(summary, "current_best_macro_f1", 0.743)
	if score := replayFloat(summary, "first_successful_score", 0); score > 0 {
		firstScore = score
	}
	if score := replayFloat(summary, "current_best_score", 0); score > 0 {
		currentBest = score
	}
	if taskType == "object_detection" {
		if score := replayFloat(summary, "first_successful_map50_95", 0); score > 0 {
			firstScore = score
		}
		if score := replayFloat(summary, "current_best_map50_95", 0); score > 0 {
			currentBest = score
		}
	}
	runCount := replayInt(summary, "completed_training_runs", 22)
	plannerRounds := replayInt(summary, "planner_rounds", 6)
	models := replayStrings(summary, "attempted_models")
	if len(models) == 0 {
		if taskType == "object_detection" {
			models = []string{"yolo11n.pt", "yolo11s.pt", "yolo11m.pt"}
		} else {
			models = []string{"efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "mobilenet_v3_large", "resnet18", "regnet_y_400mf"}
		}
	}
	dominantMechanism := replayString(summary, "dominant_mechanism", "architecture_challenge")
	currentChampionModel := replayString(summary, "current_best_model", "")
	if currentChampionModel == "" {
		currentChampionModel = models[(runCount-1)%len(models)]
	}
	modelCatalog := replayModelCatalog(summary, "model_catalog")
	successfulStrategyMemory := replayStrategyMemory(summary, "successful_strategy_memory")
	failedStrategyMemory := replayStrategyMemory(summary, "failed_strategy_memory")
	retrievedMemory := replayRetrievedMemoryResults(summary, "retrieved_memory")

	now := time.Date(2026, 5, 31, 12, 0, 0, 0, time.UTC)
	project := projects.Project{
		ID:        "project_plateau_backbone_lottery",
		Name:      replayString(summary, "project_name", "Plateau Backbone Lottery"),
		Goal:      replayString(summary, "project_goal", "Improve macro-F1 without low-yield model-family shopping."),
		Status:    projects.StatusCreated,
		CreatedAt: now.Add(-72 * time.Hour),
		UpdatedAt: now,
	}
	dataset := datasets.Dataset{
		ID:        "dataset_plateau",
		ProjectID: project.ID,
		Name:      replayString(summary, "dataset_name", "plateau-fixture"),
		Status:    datasets.StatusProfiled,
		Profile: map[string]any{
			"task_type":          taskType,
			"class_count":        replayInt(summary, "class_count", 5),
			"total_images":       replayInt(summary, "total_images", 900),
			"imbalance_ratio":    replayFloat(summary, "imbalance_ratio", 4.2),
			"class_distribution": map[string]any{"majority": 520, "minority": 75},
		},
		CreatedAt: now.Add(-72 * time.Hour),
	}
	if taskType == "object_detection" {
		dataset.Profile["yolo_available"] = replayBool(summary, "yolo_available", true)
		dataset.Profile["object_detection_available"] = replayBool(summary, "object_detection_available", true)
		dataset.Profile["bbox_available"] = replayBool(summary, "bbox_available", true)
		if yoloSummary, ok := summary["yolo_summary"].(map[string]any); ok {
			dataset.Profile["yolo_summary"] = yoloSummary
		}
	}

	priorPlans := make([]plans.ExperimentPlan, 0, plannerRounds)
	priorJobs := make([]jobs.ExperimentJob, 0, runCount)
	priorSummaries := make([]runs.TrainingRunSummary, 0, runCount)
	sourcePlan := plans.ExperimentPlan{
		ID:           "plan_6",
		ProjectID:    project.ID,
		DatasetID:    dataset.ID,
		Status:       plans.StatusApproved,
		TargetMetric: targetMetric,
		CreatedAt:    now.Add(-12 * time.Hour),
	}

	scoreStep := 0.0
	if runCount > 1 {
		scoreStep = (currentBest - firstScore) / float64(runCount-1)
	}
	for round := 0; round < plannerRounds; round++ {
		planID := replayPlanID(round + 1)
		plan := plans.ExperimentPlan{
			ID:           planID,
			ProjectID:    project.ID,
			DatasetID:    dataset.ID,
			Status:       plans.StatusApproved,
			TargetMetric: targetMetric,
			CreatedAt:    now.Add(time.Duration(-plannerRounds+round) * time.Hour),
		}
		for index := 0; index < len(models) && index < 4; index++ {
			model := models[(round+index)%len(models)]
			plan.Experiments = append(plan.Experiments, replayTaskExperiment(model, dominantMechanism, taskType))
		}
		priorPlans = append(priorPlans, plan)
		if round == plannerRounds-1 {
			sourcePlan = plan
		}
	}

	for index := 0; index < runCount; index++ {
		model := models[index%len(models)]
		plan := priorPlans[index%len(priorPlans)]
		jobID := replayJobID(index + 1)
		completedAt := now.Add(time.Duration(index-runCount) * time.Minute)
		score := firstScore + scoreStep*float64(index)
		if index >= runCount-4 {
			score = currentBest - 0.001 + 0.00025*float64(index-(runCount-4))
		}
		if index == runCount-1 {
			score = currentBest
		}
		priorJobs = append(priorJobs, jobs.ExperimentJob{
			ID:        jobID,
			ProjectID: project.ID,
			Template:  jobs.TemplateTrainExperiment,
			Status:    jobs.StatusSucceeded,
			Config: map[string]any{
				"plan_id":          plan.ID,
				"experiment_index": index % 4,
				"model":            model,
				"mechanism":        dominantMechanism,
				"intervention":     "backbone_family_challenge",
				"evidence_used":    []any{"plateau", "model catalog"},
				"expected_effect":  "Improve macro-F1 through a different pretrained backbone.",
			},
			CreatedAt:   completedAt.Add(-20 * time.Minute),
			CompletedAt: &completedAt,
		})
		priorSummaries = append(priorSummaries, runs.TrainingRunSummary{
			JobID:            jobID,
			ProjectID:        project.ID,
			PlanID:           plan.ID,
			DatasetID:        dataset.ID,
			Model:            model,
			Status:           jobs.StatusSucceeded,
			RuntimeSeconds:   600,
			EstimatedCostUSD: 0.04,
			BestMacroF1:      score,
			BestAccuracy:     score + 0.08,
			FinalTrainLoss:   0.34,
			FinalValLoss:     0.52,
			EpochsCompleted:  12,
			CreatedAt:        completedAt.Add(-20 * time.Minute),
			UpdatedAt:        completedAt,
		})
	}

	input := agents.ExperimentPlannerInput{
		Project:    project,
		Dataset:    dataset,
		SourcePlan: sourcePlan,
		DatasetInsights: agents.DatasetPlanningInsights{
			Summary:                  replayString(summary, "dataset_summary", ""),
			TaskType:                 taskType,
			ClassCount:               replayInt(summary, "class_count", 5),
			TotalImages:              replayInt(summary, "total_images", 900),
			ImbalanceRatio:           replayFloat(summary, "imbalance_ratio", 4.2),
			RecommendedPreprocessing: replayStrings(summary, "recommended_preprocessing"),
			RecommendedAugmentations: replayStrings(summary, "recommended_augmentations"),
			RecommendedMetrics:       replayStrings(summary, "recommended_metrics"),
			LiveInferencePriorities:  replayStrings(summary, "live_inference_priorities"),
			DatasetTraits:            replayStrings(summary, "dataset_traits"),
			Constraints:              replayStrings(summary, "constraints"),
			MetadataSummary:          replayAnyMap(summary, "metadata_summary"),
			AgentSafeMetadataSummary: replayAnyMap(summary, "agent_safe_metadata_summary"),
			Artifacts:                replayAnyMapSlice(summary, "artifacts"),
		},
		DeterministicDiagnosis: agents.PlannerDiagnosis{
			PlateauScore:               0.72,
			ClassImbalanceScore:        0.62,
			MinorityClassFailureScore:  0.58,
			ImprovementStagnationScore: 0.80,
			RecommendedFailureModes:    []string{"plateau", "class_imbalance", "minority_class_failure", "improvement_stagnation"},
			DeterministicDiagnosisUsed: []string{"plateau_score=0.720", "class_imbalance_score=0.620", "minority_class_failure_score=0.580", "improvement_stagnation_score=0.800"},
			Evidence:                   []string{"22 completed runs improved macro-F1 from 0.701 to 0.743", "recent backbone attempts had negligible uplift"},
		},
		CurrentChampion: &agents.ExperimentChampion{
			JobID:        replayJobID(runCount),
			PlanID:       priorPlans[len(priorPlans)-1].ID,
			Model:        currentChampionModel,
			TargetMetric: targetMetric,
			Score:        currentBest,
			BestMacroF1:  currentBest,
			BestAccuracy: currentBest + 0.08,
		},
		NoImprovementRounds:          3,
		MinimumMeaningfulImprovement: 0.010,
		PriorPlans:                   priorPlans,
		PriorJobs:                    priorJobs,
		PriorSummaries:               priorSummaries,
		PriorMemory:                  replayAgentMemory(summary, "prior_memory"),
		MaxExperiments:               fixture.Expected.MaxSelectedExperiments,
		FollowUpRound:                plannerRounds,
		MaxFollowUpRounds:            plannerRounds,
		ModelCatalog:                 modelCatalog,
		SuccessfulStrategyMemory:     successfulStrategyMemory,
		FailedStrategyMemory:         failedStrategyMemory,
		RetrievedMemory:              retrievedMemory,
		ExistingExperimentSignatures: replayStrings(summary, "existing_experiment_signatures"),
	}
	if objectivePrimary := replayString(summary, "objective_primary", ""); objectivePrimary != "" {
		input.ObjectiveContext.PrimaryObjective = objectivePrimary
	}
	input.ObjectiveContext.GoalText = replayString(summary, "objective_goal_text", project.Goal)
	input.ObjectiveContext.DeploymentPriorities = replayStrings(summary, "deployment_priorities")
	input.ObjectiveContext.Constraints = replayStrings(summary, "objective_constraints")
	input.ObjectiveContext.MetricPreferences = replayStrings(summary, "metric_preferences")
	input.ObjectiveContext.RankingWeights = replayFloatMap(summary, "ranking_weights")
	input.ProjectTrajectory = agents.ComputeProjectTrajectoryDiagnosis(input)
	return input
}

func replayTaskExperiment(model string, mechanism string, taskType string) plans.PlannedExperiment {
	if taskType == "object_detection" {
		return plans.PlannedExperiment{
			Template:       "yolo11_detection",
			Model:          model,
			Mechanism:      mechanism,
			Intervention:   "YOLO detector replay candidate for the current object-detection task.",
			EvidenceUsed:   []string{"yolo object-detection evidence", "model catalog"},
			ExpectedEffect: "Improve mAP/recall through a detector-specific challenger.",
			Epochs:         10,
			BatchSize:      8,
			LearningRate:   0.001,
			ImageSize:      640,
			Pretrained:     true,
			Reason:         "YOLO replay candidate for object detection.",
			Strategy:       "Detector replay baseline or challenger.",
		}
	}
	return plans.PlannedExperiment{
		Template:       "transfer_learning",
		Model:          model,
		Mechanism:      mechanism,
		Intervention:   "backbone_family_challenge",
		EvidenceUsed:   []string{"plateau", "model catalog"},
		ExpectedEffect: "Improve macro-F1 through a different pretrained backbone.",
		Epochs:         12,
		BatchSize:      16,
		LearningRate:   0.0003,
		Reason:         "Prior plan tried another architecture challenger.",
		Strategy:       "Backbone/model-family exploration.",
	}
}

func replayPlanID(index int) string {
	return fmt.Sprintf("plan_%d", index)
}

func replayJobID(index int) string {
	return fmt.Sprintf("job_%02d", index)
}

func replayFloat(values map[string]any, key string, fallback float64) float64 {
	switch value := values[key].(type) {
	case float64:
		return value
	case int:
		return float64(value)
	default:
		return fallback
	}
}

func replayInt(values map[string]any, key string, fallback int) int {
	switch value := values[key].(type) {
	case float64:
		return int(value)
	case int:
		return value
	default:
		return fallback
	}
}

func replayString(values map[string]any, key string, fallback string) string {
	if value, ok := values[key].(string); ok && value != "" {
		return value
	}
	return fallback
}

func replayStrings(values map[string]any, key string) []string {
	raw, ok := values[key].([]any)
	if !ok {
		if typed, ok := values[key].([]string); ok {
			return typed
		}
		return nil
	}
	out := []string{}
	for _, item := range raw {
		if text, ok := item.(string); ok && text != "" {
			out = append(out, text)
		}
	}
	return out
}

func replayBool(values map[string]any, key string, fallback bool) bool {
	switch value := values[key].(type) {
	case bool:
		return value
	default:
		return fallback
	}
}

func replayAnyMap(values map[string]any, key string) map[string]any {
	raw, ok := values[key].(map[string]any)
	if !ok || len(raw) == 0 {
		return nil
	}
	return raw
}

func replayAnyMapSlice(values map[string]any, key string) []map[string]any {
	raw, ok := values[key].([]any)
	if !ok {
		return nil
	}
	out := make([]map[string]any, 0, len(raw))
	for _, item := range raw {
		if typed, ok := item.(map[string]any); ok && len(typed) > 0 {
			out = append(out, typed)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func replayFloatMap(values map[string]any, key string) map[string]float64 {
	raw, ok := values[key].(map[string]any)
	if !ok || len(raw) == 0 {
		return nil
	}
	out := map[string]float64{}
	for k, v := range raw {
		switch typed := v.(type) {
		case float64:
			out[k] = typed
		case int:
			out[k] = float64(typed)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func replayModelCatalog(values map[string]any, key string) []agents.SupportedModelSpec {
	raw, ok := values[key].([]any)
	if !ok {
		return nil
	}
	out := make([]agents.SupportedModelSpec, 0, len(raw))
	for _, item := range raw {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		spec := agents.SupportedModelSpec{
			Name:                  replayString(entry, "name", ""),
			Family:                replayString(entry, "family", ""),
			TaskType:              replayString(entry, "task_type", ""),
			ModelKind:             replayString(entry, "model_kind", ""),
			PretrainedWeights:     replayString(entry, "pretrained_weights", ""),
			DeploymentTier:        replayString(entry, "deployment_tier", ""),
			DefaultImageSize:      replayInt(entry, "default_image_size", 0),
			MinRecommendedImages:  replayInt(entry, "min_recommended_images", 0),
			SupportsTransfer:      replayBool(entry, "supports_transfer", false),
			TrainingEnabled:       replayBool(entry, "training_enabled", true),
			ExpectedLatencyClass:  replayString(entry, "expected_latency_class", ""),
			RecommendedUse:        replayString(entry, "recommended_use", ""),
			SupportsFineTuneModes: replayStrings(entry, "supports_fine_tune_modes"),
		}
		if spec.Name != "" {
			out = append(out, spec)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func replayStrategyMemory(values map[string]any, key string) []agents.PlannerStrategyMemory {
	raw, ok := values[key].([]any)
	if !ok {
		return nil
	}
	out := make([]agents.PlannerStrategyMemory, 0, len(raw))
	for _, item := range raw {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		memory := agents.PlannerStrategyMemory{
			MemoryID:                replayString(entry, "memory_id", ""),
			OutcomeStatus:           replayString(entry, "outcome_status", ""),
			Lesson:                  replayString(entry, "lesson", ""),
			BestModel:               replayString(entry, "best_model", ""),
			ActualDeltaVsChampion:   replayFloat(entry, "actual_delta_vs_champion", 0),
			ExpectedDeltaVsChampion: replayFloat(entry, "expected_delta_vs_champion", 0),
			TotalCostUSD:            replayFloat(entry, "total_cost_usd", 0),
			TotalRuntimeSeconds:     replayFloat(entry, "total_runtime_seconds", 0),
			ProposedModels:          replayStrings(entry, "proposed_models"),
			Tags:                    replayStrings(entry, "tags"),
		}
		if memory.MemoryID != "" || memory.Lesson != "" {
			out = append(out, memory)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func replayRetrievedMemoryResults(values map[string]any, key string) []memory.MemoryRetrievalResult {
	raw, ok := values[key].([]any)
	if !ok {
		return nil
	}
	out := make([]memory.MemoryRetrievalResult, 0, len(raw))
	for _, item := range raw {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		result := memory.MemoryRetrievalResult{
			SourceTable:     replayString(entry, "source_table", ""),
			SourceID:        replayString(entry, "source_id", ""),
			ProjectID:       replayString(entry, "project_id", ""),
			DatasetID:       replayString(entry, "dataset_id", ""),
			Kind:            replayString(entry, "kind", ""),
			Score:           replayFloat(entry, "score", 0),
			SemanticScore:   replayFloat(entry, "semantic_score", 0),
			StructuredScore: replayFloat(entry, "structured_score", 0),
			RetrievalReason: replayString(entry, "retrieval_reason", ""),
			SummaryCard:     replayAnyMap(entry, "summary_card"),
			Metadata:        replayAnyMap(entry, "metadata"),
		}
		if result.SourceTable != "" || result.SourceID != "" || len(result.SummaryCard) > 0 || len(result.Metadata) > 0 {
			out = append(out, result)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func replayAgentMemory(values map[string]any, key string) []memory.AgentMemoryRecord {
	raw, ok := values[key].([]any)
	if !ok {
		return nil
	}
	out := make([]memory.AgentMemoryRecord, 0, len(raw))
	for _, item := range raw {
		entry, ok := item.(map[string]any)
		if !ok {
			continue
		}
		record := memory.AgentMemoryRecord{
			ID:        replayString(entry, "id", replayString(entry, "memory_id", "")),
			ProjectID: replayString(entry, "project_id", ""),
			DatasetID: replayString(entry, "dataset_id", ""),
			PlanID:    replayString(entry, "plan_id", ""),
			JobID:     replayString(entry, "job_id", ""),
			AgentName: replayString(entry, "agent_name", ""),
			Kind:      replayString(entry, "kind", ""),
			Summary:   replayString(entry, "summary", ""),
			Payload:   replayAnyMap(entry, "payload"),
			Tags:      replayStrings(entry, "tags"),
		}
		if record.ID != "" || record.Summary != "" || record.Kind != "" {
			out = append(out, record)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
