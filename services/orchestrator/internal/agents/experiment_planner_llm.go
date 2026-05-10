package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
)

const (
	ExperimentPlannerAgentName     = "experiment_planner"
	ExperimentPlannerAgentVersion  = "v1"
	ExperimentPlannerPromptVersion = "experiment_planner_v1"
)

type ExperimentPlannerAgent struct {
	generator llm.JSONGenerator
	model     string
}

type ExperimentPlannerInput struct {
	Project                      projects.Project
	Dataset                      datasets.Dataset
	SourcePlan                   plans.ExperimentPlan
	PlanJobs                     []jobs.ExperimentJob
	PlanSummaries                []runs.TrainingRunSummary
	PlanMetrics                  map[string][]jobs.EpochMetric
	CurrentChampion              *ExperimentChampion
	SourcePlanBaselineChampion   *ExperimentChampion
	SourcePlanDeltas             []ExperimentRunDelta
	NoImprovementRounds          int
	StopSignals                  []string
	MinimumMeaningfulImprovement float64
	PriorPlans                   []plans.ExperimentPlan
	PriorJobs                    []jobs.ExperimentJob
	PriorMemory                  []memory.AgentMemoryRecord
	ExistingExperimentSignatures []string
	MaxExperiments               int
	MaxFollowUpRounds            int
	FollowUpRound                int
}

type ExperimentChampion struct {
	JobID            string  `json:"job_id"`
	PlanID           string  `json:"plan_id"`
	Model            string  `json:"model"`
	TargetMetric     string  `json:"target_metric"`
	Score            float64 `json:"score"`
	BestMacroF1      float64 `json:"best_macro_f1"`
	BestAccuracy     float64 `json:"best_accuracy"`
	EstimatedCostUSD float64 `json:"estimated_cost_usd"`
	RuntimeSeconds   float64 `json:"runtime_seconds"`
	EpochsCompleted  int     `json:"epochs_completed"`
}

type ExperimentRunDelta struct {
	JobID                    string  `json:"job_id"`
	PlanID                   string  `json:"plan_id"`
	Model                    string  `json:"model"`
	Status                   string  `json:"status"`
	TargetMetric             string  `json:"target_metric"`
	Score                    float64 `json:"score"`
	BestMacroF1              float64 `json:"best_macro_f1"`
	BestAccuracy             float64 `json:"best_accuracy"`
	EstimatedCostUSD         float64 `json:"estimated_cost_usd"`
	RuntimeSeconds           float64 `json:"runtime_seconds"`
	EpochsCompleted          int     `json:"epochs_completed"`
	ChampionJobID            string  `json:"champion_job_id"`
	DeltaScoreVsChampion     float64 `json:"delta_score_vs_champion"`
	DeltaCostVsChampion      float64 `json:"delta_cost_vs_champion"`
	DeltaRuntimeVsChampion   float64 `json:"delta_runtime_vs_champion"`
	MeaningfullyImprovedOver bool    `json:"meaningfully_improved_over_champion"`
}

const (
	ExperimentPlanningOutcomeImprovedChampion = "improved_champion"
	ExperimentPlanningOutcomeMinorImprovement = "minor_improvement"
	ExperimentPlanningOutcomeNoImprovement    = "no_improvement"
	ExperimentPlanningOutcomeFailed           = "failed"
)

type ExperimentPlanningOutcome struct {
	OutcomeType             string                    `json:"outcome_type"`
	OutcomeStatus           string                    `json:"outcome_status"`
	SourceDecisionID        string                    `json:"source_decision_id"`
	SourcePlanID            string                    `json:"source_plan_id"`
	FollowUpPlanID          string                    `json:"follow_up_plan_id"`
	BaselineChampion        *ExperimentChampion       `json:"baseline_champion,omitempty"`
	ActualBestRun           *ExperimentChampion       `json:"actual_best_run,omitempty"`
	ExpectedDeltaVsChampion float64                   `json:"expected_delta_vs_champion"`
	ActualDeltaVsChampion   float64                   `json:"actual_delta_vs_champion"`
	MetExpectedDelta        bool                      `json:"met_expected_delta"`
	TotalCostUSD            float64                   `json:"total_cost_usd"`
	TotalRuntimeSeconds     float64                   `json:"total_runtime_seconds"`
	TerminalRunCount        int                       `json:"terminal_run_count"`
	SuccessfulRunCount      int                       `json:"successful_run_count"`
	FailedRunCount          int                       `json:"failed_run_count"`
	ProposedExperiments     []plans.PlannedExperiment `json:"proposed_experiments"`
	Lesson                  string                    `json:"lesson"`
	CompletedAt             time.Time                 `json:"completed_at"`
}

type ExperimentPlanningRecommendation struct {
	AgentName               string                    `json:"agent_name"`
	Summary                 string                    `json:"summary"`
	DecisionType            string                    `json:"decision_type"`
	Rationale               string                    `json:"rationale"`
	Confidence              float64                   `json:"confidence"`
	ProposedExperiments     []plans.PlannedExperiment `json:"proposed_experiments"`
	ChampionJobID           string                    `json:"champion_job_id"`
	WhyCanBeatChampion      string                    `json:"why_can_beat_champion"`
	ExpectedDeltaVsChampion float64                   `json:"expected_delta_vs_champion"`
	StopReason              string                    `json:"stop_reason"`
	Risks                   []string                  `json:"risks"`
	ExpectedTradeoffs       []string                  `json:"expected_tradeoffs"`
	NoveltyNotes            []string                  `json:"novelty_notes"`
	Tags                    []string                  `json:"tags"`
}

type ExperimentPlanningTrace struct {
	Recommendation   ExperimentPlanningRecommendation
	Request          llm.JSONRequest
	PromptContext    map[string]any
	RawOutput        []byte
	ParsedOutput     map[string]any
	ValidationStatus string
	ValidationError  string
	AgentVersion     string
	PromptVersion    string
}

func NewExperimentPlannerAgent(generator llm.JSONGenerator, model string) ExperimentPlannerAgent {
	return ExperimentPlannerAgent{
		generator: generator,
		model:     model,
	}
}

func (a ExperimentPlannerAgent) Plan(ctx context.Context, input ExperimentPlannerInput) (ExperimentPlanningRecommendation, error) {
	trace, err := a.PlanWithTrace(ctx, input)
	return trace.Recommendation, err
}

func (a ExperimentPlannerAgent) PlanWithTrace(ctx context.Context, input ExperimentPlannerInput) (ExperimentPlanningTrace, error) {
	trace := ExperimentPlanningTrace{
		PromptContext:    experimentPlannerPromptContext(input),
		ParsedOutput:     map[string]any{},
		ValidationStatus: memory.InvocationValidationFailed,
		AgentVersion:     ExperimentPlannerAgentVersion,
		PromptVersion:    ExperimentPlannerPromptVersion,
	}

	if a.generator == nil {
		err := fmt.Errorf("experiment planner requires an llm generator")
		trace.ValidationError = err.Error()
		return trace, err
	}

	contextBlob, err := json.Marshal(trace.PromptContext)
	if err != nil {
		wrapped := fmt.Errorf("marshal experiment planner context: %w", err)
		trace.ValidationError = wrapped.Error()
		return trace, wrapped
	}

	trace.Request = experimentPlannerJSONRequest(a.model, contextBlob)
	raw, err := a.generator.GenerateJSON(ctx, trace.Request)
	if err != nil {
		trace.ValidationError = err.Error()
		return trace, err
	}
	trace.RawOutput = append([]byte(nil), raw...)
	trace.ParsedOutput = rawOutputObject(raw)

	var recommendation ExperimentPlanningRecommendation
	if err := json.Unmarshal(raw, &recommendation); err != nil {
		wrapped := fmt.Errorf("decode experiment planner recommendation: %w", err)
		trace.ValidationStatus = memory.InvocationValidationInvalid
		trace.ValidationError = wrapped.Error()
		return trace, wrapped
	}

	recommendation.AgentName = ExperimentPlannerAgentName
	if err := validateExperimentPlanningRecommendation(recommendation, maxPlannerExperiments(input.MaxExperiments)); err != nil {
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

func experimentPlannerJSONRequest(model string, contextBlob []byte) llm.JSONRequest {
	return llm.JSONRequest{
		Model:       model,
		Temperature: 0.35,
		Messages: []llm.Message{
			{
				Role: "system",
				Content: strings.TrimSpace(`You are the Model Express Experiment Planning Agent.
Return only valid JSON. You run after a whole experiment plan has completed, not after one run.
Design the next image-classification experiment batch from all plan results and prior memory.
Be willing to change model family, image size, optimizer, scheduler, augmentation, class balancing,
weight decay, learning rate, batch size, and training budget when the evidence supports it.
Avoid repeating exact experiment configurations unless the repeat is explicitly intentional and justified.
Do not request direct execution. Deterministic backend policy will validate and schedule your proposal.`),
			},
			{
				Role: "user",
				Content: fmt.Sprintf(`Plan the next step and return JSON with this exact shape:
{
  "summary": "short plan-level summary",
  "decision_type": "ADD_EXPERIMENTS|SELECT_CHAMPION|STOP_PROJECT|WAIT",
  "rationale": "why this plan-level decision is best",
  "confidence": 0.0,
  "proposed_experiments": [
    {
      "template": "efficientnet_transfer",
      "model": "efficientnet_b0",
      "epochs": 10,
      "batch_size": 16,
      "learning_rate": 0.0002,
      "reason": "why this experiment is useful",
      "image_size": 224,
      "optimizer": "adamw",
      "scheduler": "cosine",
      "weight_decay": 0.01,
      "augmentation": {"horizontal_flip": true, "color_jitter": true, "random_crop": true},
      "class_balancing": "weighted_loss",
      "early_stopping_patience": 3,
      "strategy": "focused efficientnet improvement"
    }
  ],
  "champion_job_id": "",
  "why_can_beat_champion": "specific reason these experiments can beat the current champion",
  "expected_delta_vs_champion": 0.02,
  "stop_reason": "",
  "risks": ["risk"],
  "expected_tradeoffs": ["tradeoff"],
  "novelty_notes": ["what changed from prior experiments"],
  "tags": ["short_tag"]
}

Rules:
- If decision_type is ADD_EXPERIMENTS, propose 1-5 complete, novel experiments.
- Use only supported model families for now: mobilenet_v3_small, efficientnet_b0, efficientnet_b1, resnet18, resnet34.
- Use only supported optimizers: adamw, adam, sgd.
- Use only supported schedulers: none, cosine, step.
- Keep epochs between 3 and 40, batch_size between 4 and 128, image_size between 96 and 384.
- Prefer meaningful changes over just increasing epochs.
- If prior runs are weak or unstable, try model/preprocessing/regularization changes.
- If one family is promising, exploit it with controlled learning-rate, augmentation, or image-size changes.
- Compare every proposal against current_champion, source_plan_baseline_champion, and source_plan_run_deltas.
- Only use ADD_EXPERIMENTS when you can explain a concrete path to beat the current champion.
- If stop_signals say the project has repeated no-improvement follow-up rounds, prefer SELECT_CHAMPION or STOP_PROJECT.
- Set champion_job_id when selecting a champion or when a champion anchors your recommendation.
- Set why_can_beat_champion for ADD_EXPERIMENTS; set stop_reason for SELECT_CHAMPION or STOP_PROJECT.
- Do not repeat any signature listed in existing_experiment_signatures.

Context:
%s`, string(contextBlob)),
			},
		},
	}
}

func validateExperimentPlanningRecommendation(recommendation ExperimentPlanningRecommendation, maxExperiments int) error {
	if strings.TrimSpace(recommendation.Summary) == "" {
		return fmt.Errorf("experiment planner recommendation missing summary")
	}
	if strings.TrimSpace(recommendation.Rationale) == "" {
		return fmt.Errorf("experiment planner recommendation missing rationale")
	}
	if recommendation.Confidence < 0 || recommendation.Confidence > 1 {
		return fmt.Errorf("experiment planner confidence must be between 0 and 1")
	}
	switch strings.ToUpper(strings.TrimSpace(recommendation.DecisionType)) {
	case decisions.TypeAddExperiments:
		if len(recommendation.ProposedExperiments) == 0 {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing proposed_experiments")
		}
		if strings.TrimSpace(recommendation.WhyCanBeatChampion) == "" {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing why_can_beat_champion")
		}
		if len(recommendation.ProposedExperiments) > maxExperiments {
			return fmt.Errorf("experiment planner proposed %d experiments, max is %d", len(recommendation.ProposedExperiments), maxExperiments)
		}
		for index, experiment := range recommendation.ProposedExperiments {
			if err := validatePlannedExperimentShape(experiment, index); err != nil {
				return err
			}
		}
	case decisions.TypeSelectChampion, decisions.TypeStopProject, decisions.TypeWait:
	default:
		return fmt.Errorf("experiment planner has invalid decision_type %q", recommendation.DecisionType)
	}
	if recommendation.Risks == nil {
		recommendation.Risks = []string{}
	}
	if recommendation.ExpectedTradeoffs == nil {
		recommendation.ExpectedTradeoffs = []string{}
	}
	if recommendation.NoveltyNotes == nil {
		recommendation.NoveltyNotes = []string{}
	}
	if recommendation.Tags == nil {
		recommendation.Tags = []string{}
	}
	return nil
}

func validatePlannedExperimentShape(experiment plans.PlannedExperiment, index int) error {
	if strings.TrimSpace(experiment.Template) == "" {
		return fmt.Errorf("proposed experiment %d is missing template", index)
	}
	if strings.TrimSpace(experiment.Model) == "" {
		return fmt.Errorf("proposed experiment %d is missing model", index)
	}
	if experiment.Epochs < 1 {
		return fmt.Errorf("proposed experiment %d must have at least one epoch", index)
	}
	if experiment.BatchSize < 1 {
		return fmt.Errorf("proposed experiment %d must have a positive batch size", index)
	}
	if experiment.LearningRate <= 0 {
		return fmt.Errorf("proposed experiment %d must have a positive learning rate", index)
	}
	return nil
}

func experimentPlannerPromptContext(input ExperimentPlannerInput) map[string]any {
	return map[string]any{
		"project": map[string]any{
			"id":   input.Project.ID,
			"name": input.Project.Name,
			"goal": input.Project.Goal,
		},
		"dataset": map[string]any{
			"id":      input.Dataset.ID,
			"name":    input.Dataset.Name,
			"profile": input.Dataset.Profile,
		},
		"source_plan": map[string]any{
			"id":            input.SourcePlan.ID,
			"target_metric": input.SourcePlan.TargetMetric,
			"experiments":   input.SourcePlan.Experiments,
		},
		"current_champion":               input.CurrentChampion,
		"source_plan_baseline_champion":  input.SourcePlanBaselineChampion,
		"source_plan_run_deltas":         input.SourcePlanDeltas,
		"no_improvement_rounds":          input.NoImprovementRounds,
		"minimum_meaningful_improvement": input.MinimumMeaningfulImprovement,
		"stop_signals":                   input.StopSignals,
		"plan_jobs":                      compactPlannerJobs(input.PlanJobs),
		"plan_summaries":                 input.PlanSummaries,
		"plan_epoch_metrics":             compactPlannerMetrics(input.PlanMetrics),
		"prior_plans":                    compactPlannerPlans(input.PriorPlans),
		"prior_jobs":                     compactPlannerJobs(input.PriorJobs),
		"prior_memory":                   compactMemoryRecords(input.PriorMemory),
		"existing_experiment_signatures": input.ExistingExperimentSignatures,
		"max_experiments":                maxPlannerExperiments(input.MaxExperiments),
		"max_followup_rounds":            input.MaxFollowUpRounds,
		"followup_round":                 input.FollowUpRound,
	}
}

func compactPlannerPlans(projectPlans []plans.ExperimentPlan) []map[string]any {
	out := make([]map[string]any, 0, len(projectPlans))
	for _, plan := range projectPlans {
		out = append(out, map[string]any{
			"id":                 plan.ID,
			"source_decision_id": plan.SourceDecisionID,
			"target_metric":      plan.TargetMetric,
			"experiments":        plan.Experiments,
			"created_at":         plan.CreatedAt,
		})
	}
	return out
}

func compactPlannerJobs(experimentJobs []jobs.ExperimentJob) []map[string]any {
	out := make([]map[string]any, 0, len(experimentJobs))
	for _, job := range experimentJobs {
		out = append(out, map[string]any{
			"id":           job.ID,
			"template":     job.Template,
			"status":       job.Status,
			"config":       job.Config,
			"created_at":   job.CreatedAt,
			"started_at":   job.StartedAt,
			"completed_at": job.CompletedAt,
			"error":        job.Error,
		})
	}
	return out
}

func compactPlannerMetrics(metrics map[string][]jobs.EpochMetric) map[string][]map[string]any {
	out := map[string][]map[string]any{}
	for jobID, jobMetrics := range metrics {
		out[jobID] = compactEpochMetrics(jobMetrics)
	}
	return out
}

func maxPlannerExperiments(value int) int {
	if value < 1 {
		return 5
	}
	if value > 5 {
		return 5
	}
	return value
}
