package evals

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
)

type PlannerReplayFixture struct {
	Name         string                `json:"name"`
	Input        map[string]any        `json:"input"`
	InputSummary map[string]any        `json:"input_summary,omitempty"`
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

	firstScore := replayFloat(summary, "first_successful_macro_f1", 0.701)
	currentBest := replayFloat(summary, "current_best_macro_f1", 0.743)
	runCount := replayInt(summary, "completed_training_runs", 22)
	plannerRounds := replayInt(summary, "planner_rounds", 6)
	models := replayStrings(summary, "attempted_models")
	if len(models) == 0 {
		models = []string{"efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "mobilenet_v3_large", "resnet18", "regnet_y_400mf"}
	}
	dominantMechanism := replayString(summary, "dominant_mechanism", "architecture_challenge")

	now := time.Date(2026, 5, 31, 12, 0, 0, 0, time.UTC)
	project := projects.Project{
		ID:        "project_plateau_backbone_lottery",
		Name:      "Plateau Backbone Lottery",
		Goal:      "Improve macro-F1 without low-yield model-family shopping.",
		Status:    projects.StatusCreated,
		CreatedAt: now.Add(-72 * time.Hour),
		UpdatedAt: now,
	}
	dataset := datasets.Dataset{
		ID:        "dataset_plateau",
		ProjectID: project.ID,
		Name:      "plateau-fixture",
		Status:    datasets.StatusProfiled,
		Profile: map[string]any{
			"task_type":          "image_classification",
			"class_count":        5,
			"total_images":       900,
			"imbalance_ratio":    4.2,
			"class_distribution": map[string]any{"majority": 520, "minority": 75},
		},
		CreatedAt: now.Add(-72 * time.Hour),
	}

	priorPlans := make([]plans.ExperimentPlan, 0, plannerRounds)
	priorJobs := make([]jobs.ExperimentJob, 0, runCount)
	priorSummaries := make([]runs.TrainingRunSummary, 0, runCount)
	sourcePlan := plans.ExperimentPlan{
		ID:           "plan_6",
		ProjectID:    project.ID,
		DatasetID:    dataset.ID,
		Status:       plans.StatusApproved,
		TargetMetric: "macro_f1",
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
			TargetMetric: "macro_f1",
			CreatedAt:    now.Add(time.Duration(-plannerRounds+round) * time.Hour),
		}
		for index := 0; index < len(models) && index < 4; index++ {
			model := models[(round+index)%len(models)]
			plan.Experiments = append(plan.Experiments, replayArchitectureExperiment(model, dominantMechanism))
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
			TaskType:       "image_classification",
			ClassCount:     5,
			TotalImages:    900,
			ImbalanceRatio: 4.2,
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
			Model:        models[(runCount-1)%len(models)],
			TargetMetric: "macro_f1",
			Score:        currentBest,
			BestMacroF1:  currentBest,
			BestAccuracy: currentBest + 0.08,
		},
		NoImprovementRounds:          3,
		MinimumMeaningfulImprovement: 0.010,
		PriorPlans:                   priorPlans,
		PriorJobs:                    priorJobs,
		PriorSummaries:               priorSummaries,
		MaxExperiments:               fixture.Expected.MaxSelectedExperiments,
		FollowUpRound:                plannerRounds,
		MaxFollowUpRounds:            plannerRounds,
	}
	input.ProjectTrajectory = agents.ComputeProjectTrajectoryDiagnosis(input)
	return input
}

func replayArchitectureExperiment(model string, mechanism string) plans.PlannedExperiment {
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
