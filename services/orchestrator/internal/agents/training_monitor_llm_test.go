package agents

import (
	"context"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

func TestTrainingMonitorAgentValidatesStructuredRecommendation(t *testing.T) {
	agent := NewTrainingMonitorAgent(fakeJSONGenerator{
		response: `{
			"summary": "EfficientNet is competitive and stable.",
			"recommended_action": {
				"action_type": "RANK_MODELS",
				"confidence": 0.82,
				"rationale": "The run has strong validation quality with low instability.",
				"payload": {"rank": 1},
				"requires_approval": true
			},
			"quality_summary": "Strong validation macro-F1.",
			"training_dynamics": "No obvious overfitting.",
			"cost_summary": "Moderate cost for the observed quality.",
			"risks": [],
			"findings": ["stable validation curve"],
			"rank_score": 0.78,
			"tags": ["efficientnet", "stable"]
		}`,
	}, "test-model")

	recommendation, err := agent.Evaluate(context.Background(), testTrainingMonitorInput())
	if err != nil {
		t.Fatalf("evaluate: %v", err)
	}
	if recommendation.AgentName != TrainingMonitorAgentName {
		t.Fatalf("expected agent name %s, got %s", TrainingMonitorAgentName, recommendation.AgentName)
	}
	if recommendation.RecommendedAction.ActionType != "RANK_MODELS" {
		t.Fatalf("unexpected action type %s", recommendation.RecommendedAction.ActionType)
	}
}

func TestTrainingMonitorAgentReturnsInvocationTrace(t *testing.T) {
	agent := NewTrainingMonitorAgent(fakeJSONGenerator{
		response: `{
			"summary": "EfficientNet is competitive and stable.",
			"recommended_action": {
				"action_type": "RANK_MODELS",
				"confidence": 0.82,
				"rationale": "The run has strong validation quality with low instability.",
				"payload": {"rank": 1},
				"requires_approval": true
			},
			"quality_summary": "Strong validation macro-F1.",
			"training_dynamics": "No obvious overfitting.",
			"cost_summary": "Moderate cost for the observed quality.",
			"risks": [],
			"findings": ["stable validation curve"],
			"rank_score": 0.78,
			"tags": ["efficientnet", "stable"]
		}`,
	}, "test-model")

	trace, err := agent.EvaluateWithTrace(context.Background(), testTrainingMonitorInput())
	if err != nil {
		t.Fatalf("evaluate with trace: %v", err)
	}
	if trace.ValidationStatus != "valid" {
		t.Fatalf("expected valid trace, got %s", trace.ValidationStatus)
	}
	if trace.PromptVersion != TrainingMonitorPromptVersion {
		t.Fatalf("expected prompt version %s, got %s", TrainingMonitorPromptVersion, trace.PromptVersion)
	}
	if len(trace.Request.Messages) != 2 {
		t.Fatalf("expected two prompt messages, got %d", len(trace.Request.Messages))
	}
	if string(trace.RawOutput) == "" {
		t.Fatal("expected raw output to be captured")
	}
}

func TestTrainingMonitorAgentRejectsUnknownAction(t *testing.T) {
	agent := NewTrainingMonitorAgent(fakeJSONGenerator{
		response: `{
			"summary": "Looks good.",
			"recommended_action": {
				"action_type": "RUN_ARBITRARY_CODE",
				"confidence": 0.9,
				"rationale": "Nope.",
				"payload": {},
				"requires_approval": true
			},
			"quality_summary": "ok",
			"training_dynamics": "ok",
			"cost_summary": "ok",
			"risks": [],
			"findings": [],
			"rank_score": 0.5,
			"tags": []
		}`,
	}, "test-model")

	_, err := agent.Evaluate(context.Background(), testTrainingMonitorInput())
	if err == nil || !strings.Contains(err.Error(), "invalid action_type") {
		t.Fatalf("expected invalid action_type error, got %v", err)
	}
}

type fakeJSONGenerator struct {
	response string
}

func (f fakeJSONGenerator) GenerateJSON(_ context.Context, _ llm.JSONRequest) ([]byte, error) {
	return []byte(f.response), nil
}

func testTrainingMonitorInput() TrainingMonitorInput {
	return TrainingMonitorInput{
		Plan: plans.ExperimentPlan{
			ID:           "plan_1",
			TargetMetric: "macro_f1",
		},
		Job: jobs.ExperimentJob{
			ID:       "job_1",
			Template: jobs.TemplateTrainExperiment,
			Status:   jobs.StatusSucceeded,
			Config: map[string]any{
				"model": "efficientnet_b0",
			},
		},
		Summary: runs.TrainingRunSummary{
			JobID:        "job_1",
			ProjectID:    "project_1",
			PlanID:       "plan_1",
			Model:        "efficientnet_b0",
			Status:       jobs.StatusSucceeded,
			BestMacroF1:  0.82,
			BestAccuracy: 0.84,
		},
		Metrics: []jobs.EpochMetric{
			{JobID: "job_1", Epoch: 1, Metrics: map[string]float64{"macro_f1": 0.6}},
			{JobID: "job_1", Epoch: 2, Metrics: map[string]float64{"macro_f1": 0.82}},
		},
	}
}
