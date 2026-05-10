package agents

import (
	"context"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

func TestExperimentPlannerAgentValidatesAddExperiments(t *testing.T) {
	agent := NewExperimentPlannerAgent(fakeJSONGenerator{
		response: `{
			"summary": "EfficientNet is promising, but the next round should vary preprocessing and regularization.",
			"decision_type": "ADD_EXPERIMENTS",
			"rationale": "The source plan completed below target with stable but underpowered validation quality.",
			"confidence": 0.84,
			"proposed_experiments": [
				{
					"template": "efficientnet_transfer",
					"model": "efficientnet_b1",
					"epochs": 12,
					"batch_size": 16,
					"learning_rate": 0.0002,
					"reason": "Try a larger EfficientNet with stronger augmentation.",
					"image_size": 256,
					"optimizer": "adamw",
					"scheduler": "cosine",
					"weight_decay": 0.01,
					"augmentation": {"horizontal_flip": true, "color_jitter": true},
					"class_balancing": "weighted_loss",
					"early_stopping_patience": 3,
					"strategy": "promising family exploitation"
				}
			],
			"champion_job_id": "",
			"why_can_beat_champion": "It tests a stronger EfficientNet with higher resolution and regularization instead of merely extending the same baseline.",
			"expected_delta_vs_champion": 0.02,
			"stop_reason": "",
			"risks": ["higher runtime"],
			"expected_tradeoffs": ["better quality for more cost"],
			"novelty_notes": ["larger model and image size"],
			"tags": ["efficientnet", "augmentation"]
		}`,
	}, "test-model")

	trace, err := agent.PlanWithTrace(context.Background(), testExperimentPlannerInput())
	if err != nil {
		t.Fatalf("plan with trace: %v", err)
	}
	if trace.ValidationStatus != "valid" {
		t.Fatalf("expected valid trace, got %s", trace.ValidationStatus)
	}
	if trace.Recommendation.DecisionType != decisions.TypeAddExperiments {
		t.Fatalf("expected ADD_EXPERIMENTS, got %s", trace.Recommendation.DecisionType)
	}
	if trace.Recommendation.ProposedExperiments[0].ImageSize != 256 {
		t.Fatalf("expected image size to survive decode")
	}
}

func TestExperimentPlannerAgentRejectsAddWithoutExperiments(t *testing.T) {
	agent := NewExperimentPlannerAgent(fakeJSONGenerator{
		response: `{
			"summary": "More work is useful.",
			"decision_type": "ADD_EXPERIMENTS",
			"rationale": "Need stronger models.",
			"confidence": 0.75,
			"proposed_experiments": [],
			"champion_job_id": "",
			"risks": [],
			"expected_tradeoffs": [],
			"novelty_notes": [],
			"tags": []
		}`,
	}, "test-model")

	_, err := agent.Plan(context.Background(), testExperimentPlannerInput())
	if err == nil || !strings.Contains(err.Error(), "missing proposed_experiments") {
		t.Fatalf("expected missing proposed_experiments error, got %v", err)
	}
}

func testExperimentPlannerInput() ExperimentPlannerInput {
	sourceExperiment := plans.PlannedExperiment{
		Template:     "mobilenet_transfer",
		Model:        "mobilenet_v3_small",
		Epochs:       6,
		BatchSize:    16,
		LearningRate: 0.0003,
		Reason:       "baseline",
	}
	return ExperimentPlannerInput{
		SourcePlan: plans.ExperimentPlan{
			ID:           "plan_1",
			TargetMetric: "macro_f1",
			Experiments:  []plans.PlannedExperiment{sourceExperiment},
		},
		PlanJobs: []jobs.ExperimentJob{
			{
				ID:       "job_1",
				Template: jobs.TemplateTrainExperiment,
				Status:   jobs.StatusSucceeded,
				Config: map[string]any{
					"model": "mobilenet_v3_small",
				},
			},
		},
		PlanSummaries: []runs.TrainingRunSummary{
			{
				JobID:        "job_1",
				PlanID:       "plan_1",
				Model:        "mobilenet_v3_small",
				Status:       jobs.StatusSucceeded,
				BestMacroF1:  0.62,
				BestAccuracy: 0.66,
			},
		},
		PlanMetrics: map[string][]jobs.EpochMetric{
			"job_1": {
				{JobID: "job_1", Epoch: 1, Metrics: map[string]float64{"macro_f1": 0.4}},
				{JobID: "job_1", Epoch: 6, Metrics: map[string]float64{"macro_f1": 0.62}},
			},
		},
		MaxExperiments: 3,
	}
}
