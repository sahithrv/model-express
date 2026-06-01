package agents

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
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
	promptBudget, ok := trace.PromptContext["prompt_budget"].(map[string]any)
	if !ok {
		t.Fatal("expected prompt budget metadata in trace input context")
	}
	if promptBudget["approx_context_bytes"] == nil || promptBudget["approx_input_bytes"] == nil {
		t.Fatalf("expected approximate prompt size telemetry, got %#v", promptBudget)
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

func TestTrainingMonitorRejectsPlanningAuthorityPayload(t *testing.T) {
	agent := NewTrainingMonitorAgent(fakeJSONGenerator{
		response: `{
			"summary": "This run is not enough by itself.",
			"recommended_action": {
				"action_type": "RANK_MODELS",
				"confidence": 0.72,
				"rationale": "The run should inform model ranking only.",
				"payload": {"proposed_experiments": [{"model": "resnet18"}]},
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
	if err == nil || !strings.Contains(err.Error(), "forbidden scheduling or planning authority") {
		t.Fatalf("expected forbidden authority error, got %v", err)
	}
}

func TestTrainingMonitorPromptContextUsesCompactRunEvaluationCards(t *testing.T) {
	input := testTrainingMonitorInput()
	input.Plan.Experiments = []plans.PlannedExperiment{
		{
			Template:           jobs.TemplateTrainExperiment,
			Model:              "efficientnet_b0",
			Mechanism:          "regularization",
			Intervention:       "AdamW with light augmentation",
			EvidenceUsed:       []string{"validation loss gap", "stable macro-F1"},
			ExpectedEffect:     "Improve validation quality without increasing latency.",
			Epochs:             12,
			BatchSize:          32,
			LearningRate:       0.0003,
			ImageSize:          224,
			Optimizer:          "adamw",
			Scheduler:          "cosine",
			AugmentationPolicy: "light",
		},
	}
	input.Job.Config = map[string]any{
		"model":             "efficientnet_b0",
		"full_profile_dump": "SENTINEL_FULL_DUMP",
		"class_names":       []string{"class_a", "class_b", "class_c"},
	}
	input.Summary.EpochsCompleted = 12
	input.Summary.FinalTrainLoss = 0.18
	input.Summary.FinalValLoss = 0.31
	input.Summary.RuntimeSeconds = 410
	input.Summary.EstimatedCostUSD = 0.42
	input.Evaluation = &runs.TrainingRunEvaluation{
		JobID:     "job_1",
		ProjectID: "project_1",
		PlanID:    "plan_1",
		ObjectiveProfile: map[string]any{
			"target_metric": "macro_f1",
		},
		PerClassMetrics: map[string]any{
			"class_a":   map[string]any{"precision": 0.91, "recall": 0.88, "f1-score": 0.89, "support": 50},
			"class_b":   map[string]any{"precision": 0.72, "recall": 0.55, "f1-score": 0.62, "support": 12},
			"class_c":   map[string]any{"precision": 0.84, "recall": 0.79, "f1-score": 0.81, "support": 31},
			"macro avg": map[string]any{"precision": 0.82, "recall": 0.74, "f1-score": 0.77, "support": 93},
		},
		ConfusionMatrix: [][]int{
			{44, 4, 2},
			{5, 7, 0},
			{1, 4, 26},
		},
		ModelProfile: map[string]any{
			"estimated_latency_ms":                   11.4,
			"estimated_throughput_images_per_second": 87.7,
			"parameter_count":                        5300000,
			"model_size_mb":                          20.2,
			"full_profile_dump":                      "SENTINEL_FULL_DUMP",
		},
		HolisticScores: map[string]any{
			"quality_score": 0.81,
			"latency_score": 0.93,
			"overall_score": 0.84,
		},
		RecommendationSummary: "Compact model is accurate and fast.",
	}
	input.Metrics = []jobs.EpochMetric{}
	for epoch := 1; epoch <= 12; epoch++ {
		input.Metrics = append(input.Metrics, jobs.EpochMetric{
			JobID: "job_1",
			Epoch: epoch,
			Metrics: map[string]float64{
				"macro_f1":   0.50 + float64(epoch)*0.025,
				"accuracy":   0.54 + float64(epoch)*0.024,
				"train_loss": 0.90 - float64(epoch)*0.055,
				"val_loss":   0.98 - float64(epoch)*0.045,
			},
		})
	}
	input.MemoryRecords = []memory.AgentMemoryRecord{
		{AgentName: "training_monitor", Kind: memory.KindTrainingEvaluation, Summary: "Previous run plateaued.", Tags: []string{"plateau"}},
	}

	context := trainingMonitorPromptContext(input)
	for _, forbidden := range []string{"plan", "job", "summary", "run_evaluation", "epoch_metrics", "prior_memory"} {
		if _, ok := context[forbidden]; ok {
			t.Fatalf("expected compact context to omit raw key %q", forbidden)
		}
	}

	encoded, err := json.Marshal(context)
	if err != nil {
		t.Fatalf("marshal context: %v", err)
	}
	contextJSON := string(encoded)
	if strings.Contains(contextJSON, "SENTINEL_FULL_DUMP") {
		t.Fatal("compact context leaked raw profile/config dump")
	}

	dynamics := requirePromptMap(t, context, "training_dynamics_card")
	recentEpochs, ok := dynamics["recent_epochs"].([]map[string]any)
	if !ok {
		t.Fatalf("expected compact recent epochs, got %#v", dynamics["recent_epochs"])
	}
	if len(recentEpochs) != trainingMonitorMaxRecentEpochs {
		t.Fatalf("expected %d recent epochs, got %d", trainingMonitorMaxRecentEpochs, len(recentEpochs))
	}
	if recentEpochs[0]["epoch"] != 8 {
		t.Fatalf("expected recent epochs to start at 8, got %#v", recentEpochs[0]["epoch"])
	}
	if dynamics["recent_metric_delta"] == 0.0 {
		t.Fatalf("expected recent metric trend, got %#v", dynamics)
	}

	perClass := requirePromptMap(t, context, "per_class_confusion_card")
	if perClass["minority_or_imbalance_signal"] != true {
		t.Fatalf("expected minority/per-class signal, got %#v", perClass)
	}
	if !strings.Contains(contextJSON, "class_b") {
		t.Fatal("expected compact worst-class signal to be retained")
	}
	if !strings.Contains(contextJSON, "top_confusion_pairs") {
		t.Fatal("expected compact confusion signal to be retained")
	}

	deployment := requirePromptMap(t, context, "deployment_model_profile_card")
	modelProfile := requirePromptMap(t, deployment, "model_profile_summary")
	if modelProfile["estimated_latency_ms"] != 11.4 {
		t.Fatalf("expected deployment latency summary, got %#v", modelProfile)
	}

	budget := requirePromptMap(t, context, "prompt_budget")
	if budget["approx_context_bytes"] == nil || budget["approx_input_bytes"] == nil {
		t.Fatalf("expected approximate byte telemetry, got %#v", budget)
	}
}

func TestTrainingMonitorPromptContextIncludesRetrievedRunMemory(t *testing.T) {
	input := testTrainingMonitorInput()
	input.RetrievedRunMemory = []memory.MemoryRetrievalResult{
		{
			SourceTable:     memory.SourceAgentMemoryRecord,
			SourceID:        "memory_run_1",
			Kind:            memory.KindTrainingEvaluation,
			Score:           0.82,
			RetrievalReason: "same model family and plateau dynamics",
			SummaryCard: map[string]any{
				"summary":             "Prior EfficientNet run plateaued with a similar train/validation gap.",
				"model":               "efficientnet_b0",
				"training_dynamics":   "Validation macro-F1 flattened after epoch 7.",
				"lesson":              "More epochs did not improve macro-F1.",
				"full_run_evaluation": "SENTINEL_FULL_EVALUATION",
				"epoch_history":       "SENTINEL_RAW_EPOCHS",
			},
			Metadata: map[string]any{
				"model_family":   "efficientnet",
				"mechanism":      "regularization",
				"outcome":        "no_improvement",
				"embedding_text": "SENTINEL_EMBEDDING_TEXT",
				"raw_payload":    "SENTINEL_RAW_PAYLOAD",
				"image_uri":      "s3://bucket/raw-image.jpg",
			},
		},
	}

	context := trainingMonitorPromptContext(input)
	retrieved, ok := context["retrieved_run_memory"].([]map[string]any)
	if !ok {
		t.Fatalf("expected retrieved run memory in context, got %#v", context["retrieved_run_memory"])
	}
	if len(retrieved) != 1 {
		t.Fatalf("expected one retrieved memory card, got %d", len(retrieved))
	}
	card := retrieved[0]
	if card["source_table"] != memory.SourceAgentMemoryRecord || card["source_id"] != "memory_run_1" {
		t.Fatalf("unexpected source fields: %#v", card)
	}
	if card["model_family"] != "efficientnet" || card["dynamics"] == "" || card["lesson"] == "" {
		t.Fatalf("expected compact run-dynamics fields, got %#v", card)
	}

	encoded, err := json.Marshal(context)
	if err != nil {
		t.Fatalf("marshal context: %v", err)
	}
	contextJSON := string(encoded)
	for _, forbidden := range []string{
		"SENTINEL_FULL_EVALUATION",
		"SENTINEL_RAW_EPOCHS",
		"SENTINEL_EMBEDDING_TEXT",
		"SENTINEL_RAW_PAYLOAD",
		"s3://bucket/raw-image.jpg",
		"embedding_text",
		"raw_payload",
		"full_run_evaluation",
		"epoch_history",
	} {
		if strings.Contains(contextJSON, forbidden) {
			t.Fatalf("retrieved run memory leaked forbidden field/value %q in %s", forbidden, contextJSON)
		}
	}

	budget := requirePromptMap(t, context, "prompt_budget")
	if budget["retrieved_run_memory_count"] != 1 {
		t.Fatalf("expected retrieved memory count telemetry, got %#v", budget)
	}
	if budget["retrieved_run_memory_approx_bytes"] == nil {
		t.Fatalf("expected retrieved memory byte telemetry, got %#v", budget)
	}
}

func TestTrainingMonitorPromptContextOmitsEmptyRetrievedRunMemory(t *testing.T) {
	context := trainingMonitorPromptContext(testTrainingMonitorInput())
	if _, ok := context["retrieved_run_memory"]; ok {
		t.Fatalf("expected empty retrieval to omit retrieved_run_memory, got %#v", context["retrieved_run_memory"])
	}
	budget := requirePromptMap(t, context, "prompt_budget")
	if _, ok := budget["retrieved_run_memory_count"]; ok {
		t.Fatalf("expected empty retrieval to preserve prior prompt budget shape, got %#v", budget)
	}
}

func TestTrainingMonitorMemoryToolIncludesRetrievedRunMemory(t *testing.T) {
	input := testTrainingMonitorInput()
	input.MemoryRecords = []memory.AgentMemoryRecord{
		{AgentName: "training_monitor", Kind: memory.KindTrainingEvaluation, Summary: "Recent same-dataset memory."},
	}
	input.RetrievedRunMemory = []memory.MemoryRetrievalResult{
		{
			SourceTable:     memory.SourceAgentMemoryRecord,
			SourceID:        "memory_run_2",
			Kind:            memory.KindTrainingEvaluation,
			Score:           0.77,
			RetrievalReason: "same validation-loss gap",
			SummaryCard: map[string]any{
				"summary":           "Prior MobileNet run overfit in the same pattern.",
				"model":             "mobilenet_v3_small",
				"training_dynamics": "Validation loss rose while train loss fell.",
			},
			Metadata: map[string]any{
				"model_family": "mobilenet",
				"outcome":      "failed",
			},
		},
	}

	result := ExecuteTrainingMonitorInformationTool(input, TrainingMonitorToolMemoryRecords, nil, TrainingMonitorInformationToolOptions{})
	if !result.Accepted {
		t.Fatalf("expected accepted memory tool result, got %#v", result)
	}
	retrieved, ok := result.Payload["retrieved_run_memory"].([]map[string]any)
	if !ok || len(retrieved) != 1 {
		t.Fatalf("expected compact retrieved memory in tool payload, got %#v", result.Payload)
	}
	if retrieved[0]["model_family"] != "mobilenet" || retrieved[0]["retrieval_reason"] == "" {
		t.Fatalf("unexpected retrieved memory tool card: %#v", retrieved[0])
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

func requirePromptMap(t *testing.T, parent map[string]any, key string) map[string]any {
	t.Helper()
	value, ok := parent[key].(map[string]any)
	if !ok {
		t.Fatalf("expected map at %s, got %#v", key, parent[key])
	}
	return value
}
