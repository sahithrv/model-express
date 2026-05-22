package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
)

const (
	TrainingMonitorAgentName     = "training_monitor"
	TrainingMonitorAgentVersion  = "v1"
	TrainingMonitorPromptVersion = "training_monitor_v1"
)

type TrainingMonitorAgent struct {
	generator llm.JSONGenerator
	model     string
}

type TrainingMonitorInput struct {
	Plan             plans.ExperimentPlan
	Job              jobs.ExperimentJob
	Summary          runs.TrainingRunSummary
	Evaluation       *runs.TrainingRunEvaluation
	Metrics          []jobs.EpochMetric
	ObjectiveContext ProjectObjectiveContext
	MemoryRecords    []memory.AgentMemoryRecord
}

type TrainingEvaluationRecommendation struct {
	AgentName         string                        `json:"agent_name"`
	Summary           string                        `json:"summary"`
	RecommendedAction decisions.AgentActionProposal `json:"recommended_action"`
	QualitySummary    string                        `json:"quality_summary"`
	TrainingDynamics  string                        `json:"training_dynamics"`
	CostSummary       string                        `json:"cost_summary"`
	Risks             []string                      `json:"risks"`
	Findings          []string                      `json:"findings"`
	RankScore         float64                       `json:"rank_score"`
	Tags              []string                      `json:"tags"`
}

type TrainingMonitorEvaluationTrace struct {
	Recommendation   TrainingEvaluationRecommendation
	Request          llm.JSONRequest
	PromptContext    map[string]any
	RawOutput        []byte
	ParsedOutput     map[string]any
	ValidationStatus string
	ValidationError  string
	AgentVersion     string
	PromptVersion    string
}

func NewTrainingMonitorAgent(generator llm.JSONGenerator, model string) TrainingMonitorAgent {
	return TrainingMonitorAgent{
		generator: generator,
		model:     model,
	}
}

func (a TrainingMonitorAgent) Evaluate(ctx context.Context, input TrainingMonitorInput) (TrainingEvaluationRecommendation, error) {
	trace, err := a.EvaluateWithTrace(ctx, input)
	return trace.Recommendation, err
}

func (a TrainingMonitorAgent) EvaluateWithTrace(ctx context.Context, input TrainingMonitorInput) (TrainingMonitorEvaluationTrace, error) {
	trace := TrainingMonitorEvaluationTrace{
		PromptContext:    trainingMonitorPromptContext(input),
		ParsedOutput:     map[string]any{},
		ValidationStatus: memory.InvocationValidationFailed,
		AgentVersion:     TrainingMonitorAgentVersion,
		PromptVersion:    TrainingMonitorPromptVersion,
	}

	if a.generator == nil {
		err := fmt.Errorf("training monitor requires an llm generator")
		trace.ValidationError = err.Error()
		return trace, err
	}

	contextBlob, err := json.Marshal(trace.PromptContext)
	if err != nil {
		wrapped := fmt.Errorf("marshal training monitor context: %w", err)
		trace.ValidationError = wrapped.Error()
		return trace, wrapped
	}

	trace.Request = trainingMonitorJSONRequest(a.model, contextBlob)
	raw, err := a.generator.GenerateJSON(ctx, trace.Request)
	if err != nil {
		trace.ValidationError = err.Error()
		return trace, err
	}
	trace.RawOutput = append([]byte(nil), raw...)
	trace.ParsedOutput = rawOutputObject(raw)

	var recommendation TrainingEvaluationRecommendation
	if err := json.Unmarshal(raw, &recommendation); err != nil {
		wrapped := fmt.Errorf("decode training monitor recommendation: %w", err)
		trace.ValidationStatus = memory.InvocationValidationInvalid
		trace.ValidationError = wrapped.Error()
		return trace, wrapped
	}

	recommendation.AgentName = TrainingMonitorAgentName
	if err := validateTrainingEvaluation(recommendation); err != nil {
		trace.ValidationStatus = memory.InvocationValidationInvalid
		trace.ValidationError = err.Error()
		trace.Recommendation = recommendation
		return trace, err
	}

	trace.Recommendation = recommendation
	trace.ValidationStatus = memory.InvocationValidationValid
	trace.ValidationError = ""
	return trace, nil
}

func trainingMonitorJSONRequest(model string, contextBlob []byte) llm.JSONRequest {
	return llm.JSONRequest{
		Model:       model,
		Temperature: 0.2,
		Messages: []llm.Message{
			{
				Role: "system",
				Content: strings.TrimSpace(`You are the Model Express Training Monitor Agent.
Return only valid JSON. Evaluate image-classification training runs holistically.
Consider validation quality, macro-F1, accuracy, per-class metrics, confusion matrix,
train/validation gap, metric stability, plateauing, cost, runtime, inference latency,
model size, and whether the run should inform future experiments.
This agent evaluates one run only. Do not propose new experiments or plan a follow-up batch.
Produce signals that the plan-level Experiment Planning Agent can use later.`),
			},
			{
				Role: "user",
				Content: fmt.Sprintf(`Evaluate this completed training run and return JSON with this exact shape:
{
  "summary": "short human-readable summary",
  "recommended_action": {
    "action_type": "RANK_MODELS|PRUNE_RUN|STOP_PROJECT|CHANGE_PREPROCESSING",
    "confidence": 0.0,
    "rationale": "why this action is useful",
    "payload": {},
    "requires_approval": true
  },
  "quality_summary": "quality assessment",
  "training_dynamics": "overfitting/underfitting/plateau/stability assessment",
  "cost_summary": "cost and runtime assessment",
  "risks": ["risk"],
  "findings": ["finding"],
  "rank_score": 0.0,
  "tags": ["short_tag"]
}

Do not include proposed_experiments. This is not the plan-level planner.
Payload may include supporting evidence such as overfitting indicators, plateau signals, or promising settings.
Use objective_context to judge whether the run fits the user's goal. For live/real-time goals,
penalize slow or oversized models when quality is close.

Context:
%s`, string(contextBlob)),
			},
		},
	}
}

func rawOutputObject(raw []byte) map[string]any {
	out := map[string]any{}
	if len(raw) == 0 {
		return out
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func validateTrainingEvaluation(recommendation TrainingEvaluationRecommendation) error {
	if strings.TrimSpace(recommendation.Summary) == "" {
		return fmt.Errorf("training monitor recommendation missing summary")
	}
	action := recommendation.RecommendedAction
	if !allowedAgentAction(action.ActionType) {
		return fmt.Errorf("training monitor recommendation has invalid action_type %q", action.ActionType)
	}
	if action.Confidence < 0 || action.Confidence > 1 {
		return fmt.Errorf("training monitor recommendation confidence must be between 0 and 1")
	}
	if strings.TrimSpace(action.Rationale) == "" {
		return fmt.Errorf("training monitor recommendation missing action rationale")
	}
	if recommendation.RankScore < 0 || recommendation.RankScore > 1 {
		return fmt.Errorf("training monitor recommendation rank_score must be between 0 and 1")
	}
	if recommendation.Tags == nil {
		recommendation.Tags = []string{}
	}
	return nil
}

func allowedAgentAction(actionType string) bool {
	switch strings.ToUpper(strings.TrimSpace(actionType)) {
	case "CHANGE_PREPROCESSING", "PRUNE_RUN", "RANK_MODELS", "STOP_PROJECT":
		return true
	default:
		return false
	}
}

func trainingMonitorPromptContext(input TrainingMonitorInput) map[string]any {
	return map[string]any{
		"plan": map[string]any{
			"id":            input.Plan.ID,
			"target_metric": input.Plan.TargetMetric,
			"experiments":   input.Plan.Experiments,
		},
		"job": map[string]any{
			"id":       input.Job.ID,
			"template": input.Job.Template,
			"config":   input.Job.Config,
			"status":   input.Job.Status,
		},
		"summary":           input.Summary,
		"run_evaluation":    input.Evaluation,
		"objective_context": input.ObjectiveContext,
		"epoch_metrics":     compactEpochMetrics(input.Metrics),
		"prior_memory":      compactMemoryRecords(input.MemoryRecords),
	}
}

func compactEpochMetrics(metrics []jobs.EpochMetric) []map[string]any {
	out := make([]map[string]any, 0, len(metrics))
	for _, metric := range metrics {
		out = append(out, map[string]any{
			"epoch":   metric.Epoch,
			"metrics": metric.Metrics,
		})
	}
	return out
}

func compactMemoryRecords(records []memory.AgentMemoryRecord) []map[string]any {
	out := make([]map[string]any, 0, len(records))
	for _, record := range records {
		out = append(out, map[string]any{
			"agent_name": record.AgentName,
			"kind":       record.Kind,
			"summary":    record.Summary,
			"tags":       record.Tags,
		})
	}
	return out
}
