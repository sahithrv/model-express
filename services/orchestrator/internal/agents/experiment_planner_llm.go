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
	ExperimentPlannerAgentVersion  = "v2"
	ExperimentPlannerPromptVersion = "experiment_planner_v3"
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
	PlanEvaluations              []runs.TrainingRunEvaluation
	PlanMetrics                  map[string][]jobs.EpochMetric
	DatasetInsights              DatasetPlanningInsights
	VisualExemplarContext        *PlannerVisualExemplarContext
	ObjectiveContext             ProjectObjectiveContext
	DeterministicDiagnosis       PlannerDiagnosis
	ModelCatalog                 []SupportedModelSpec
	CurrentChampion              *ExperimentChampion
	SourcePlanBaselineChampion   *ExperimentChampion
	SourcePlanDeltas             []ExperimentRunDelta
	NoImprovementRounds          int
	StopSignals                  []string
	MinimumMeaningfulImprovement float64
	SuccessfulStrategyMemory     []PlannerStrategyMemory
	FailedStrategyMemory         []PlannerStrategyMemory
	RejectedStrategyMemory       []RejectedPlannerOption
	StrategyScorecards           []PlannerStrategyScorecard
	PriorPlans                   []plans.ExperimentPlan
	PriorJobs                    []jobs.ExperimentJob
	PriorEvaluations             []runs.TrainingRunEvaluation
	PriorMemory                  []memory.AgentMemoryRecord
	ExistingExperimentSignatures []string
	MaxExperiments               int
	MaxFollowUpRounds            int
	FollowUpRound                int
}

type DatasetPlanningInsights struct {
	Summary                  string           `json:"summary"`
	TaskType                 string           `json:"task_type"`
	ClassCount               int              `json:"class_count"`
	TotalImages              int              `json:"total_images"`
	ImageCount               int              `json:"image_count"`
	ClassDistribution        map[string]any   `json:"class_distribution"`
	ImbalanceRatio           float64          `json:"imbalance_ratio"`
	CorruptImageCount        int              `json:"corrupt_image_count"`
	CorruptFileCount         int              `json:"corrupt_file_count"`
	WidthMin                 int              `json:"width_min,omitempty"`
	WidthMax                 int              `json:"width_max,omitempty"`
	HeightMin                int              `json:"height_min,omitempty"`
	HeightMax                int              `json:"height_max,omitempty"`
	ImageDimensionStats      map[string]any   `json:"image_dimension_stats"`
	SplitSummary             map[string]any   `json:"split_summary"`
	MetadataSummary          map[string]any   `json:"metadata_summary"`
	LeakageWarnings          []string         `json:"leakage_warnings"`
	DatasetTraits            []string         `json:"dataset_traits"`
	Artifacts                []map[string]any `json:"artifacts"`
	Constraints              []string         `json:"constraints"`
	RecommendedPreprocessing []string         `json:"recommended_preprocessing"`
	RecommendedAugmentations []string         `json:"recommended_augmentations"`
	RecommendedMetrics       []string         `json:"recommended_metrics"`
	LiveInferencePriorities  []string         `json:"live_inference_priorities"`
}

type PlannerVisualExemplarContext struct {
	Enabled        bool                   `json:"enabled"`
	EvidenceOnly   bool                   `json:"evidence_only"`
	ExemplarCount  int                    `json:"exemplar_count"`
	ClassCount     int                    `json:"class_count"`
	ByteBudget     int                    `json:"byte_budget"`
	PromptBudget   int                    `json:"prompt_budget"`
	Summary        string                 `json:"summary"`
	ObservedTraits []string               `json:"observed_traits"`
	ClassEvidence  []PlannerClassExemplar `json:"class_evidence"`
	Warnings       []string               `json:"warnings"`
	Audit          map[string]any         `json:"audit,omitempty"`
}

type PlannerClassExemplar struct {
	ClassName      string         `json:"class_name"`
	ExemplarCount  int            `json:"exemplar_count"`
	ObservedTraits []string       `json:"observed_traits"`
	Metadata       map[string]any `json:"metadata,omitempty"`
}

type PlannerStrategyMemory struct {
	MemoryID                string   `json:"memory_id"`
	OutcomeStatus           string   `json:"outcome_status"`
	Lesson                  string   `json:"lesson"`
	BestModel               string   `json:"best_model,omitempty"`
	ActualDeltaVsChampion   float64  `json:"actual_delta_vs_champion"`
	ExpectedDeltaVsChampion float64  `json:"expected_delta_vs_champion"`
	TotalCostUSD            float64  `json:"total_cost_usd"`
	TotalRuntimeSeconds     float64  `json:"total_runtime_seconds"`
	ProposedModels          []string `json:"proposed_models"`
	Tags                    []string `json:"tags"`
}

type PlannerStrategyScorecard struct {
	ID               string         `json:"id"`
	DatasetID        string         `json:"dataset_id"`
	SourceDecisionID string         `json:"source_decision_id"`
	SourcePlanID     string         `json:"source_plan_id"`
	FollowUpPlanID   string         `json:"followup_plan_id"`
	StrategyType     string         `json:"strategy_type"`
	PlanningMode     string         `json:"planning_mode"`
	DatasetTraits    map[string]any `json:"dataset_traits"`
	ObjectiveProfile map[string]any `json:"objective_profile"`
	ProposedChanges  map[string]any `json:"proposed_changes"`
	ExpectedDelta    float64        `json:"expected_delta"`
	ActualDelta      float64        `json:"actual_delta"`
	ConfidenceBefore float64        `json:"confidence_before"`
	ConfidenceAfter  float64        `json:"confidence_after"`
	CostUSD          float64        `json:"cost_usd"`
	RuntimeSeconds   float64        `json:"runtime_seconds"`
	Outcome          string         `json:"outcome"`
	Lesson           string         `json:"lesson"`
	Tags             []string       `json:"tags"`
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
	AgentName                     string                    `json:"agent_name"`
	Summary                       string                    `json:"summary"`
	DecisionType                  string                    `json:"decision_type"`
	Rationale                     string                    `json:"rationale"`
	Confidence                    float64                   `json:"confidence"`
	PlanningMode                  string                    `json:"planning_mode"`
	DeterministicDiagnosisUsed    []string                  `json:"deterministic_diagnosis_used"`
	EvidenceUsed                  []string                  `json:"evidence_used"`
	Hypothesis                    string                    `json:"hypothesis"`
	ExpectedFailureModes          []string                  `json:"expected_failure_modes"`
	DatasetPreprocessingRationale string                    `json:"dataset_preprocessing_rationale"`
	ChangedVariables              []string                  `json:"changed_variables"`
	SuccessCriteria               string                    `json:"success_criteria"`
	StopCondition                 string                    `json:"stop_condition"`
	DeploymentTradeoff            string                    `json:"deployment_tradeoff"`
	CandidateHypotheses           []CandidateHypothesis     `json:"candidate_hypotheses"`
	CandidateRankings             []CandidateRanking        `json:"candidate_rankings"`
	ProposedExperiments           []plans.PlannedExperiment `json:"proposed_experiments"`
	ChampionJobID                 string                    `json:"champion_job_id"`
	WhyCanBeatChampion            string                    `json:"why_can_beat_champion"`
	ExpectedDeltaVsChampion       float64                   `json:"expected_delta_vs_champion"`
	StopReason                    string                    `json:"stop_reason"`
	Risks                         []string                  `json:"risks"`
	ExpectedTradeoffs             []string                  `json:"expected_tradeoffs"`
	NoveltyNotes                  []string                  `json:"novelty_notes"`
	RejectedOptions               []RejectedPlannerOption   `json:"rejected_options"`
	Tags                          []string                  `json:"tags"`
}

type RejectedPlannerOption struct {
	Option      string   `json:"option"`
	Reason      string   `json:"reason"`
	Evidence    string   `json:"evidence"`
	AppliesWhen []string `json:"applies_when"`
}

type CandidateHypothesis struct {
	Hypothesis              string                  `json:"hypothesis"`
	PlanningMode            string                  `json:"planning_mode"`
	ProposedChanges         map[string]any          `json:"proposed_changes"`
	ExpectedMetricImpact    float64                 `json:"expected_metric_impact"`
	ExpectedTradeoffs       []string                `json:"expected_tradeoffs"`
	Risk                    string                  `json:"risk"`
	CostLevel               string                  `json:"cost_level"`
	NoveltyScore            float64                 `json:"novelty_score"`
	EvidenceUsed            []string                `json:"evidence_used"`
	SimilarSuccessMemoryIDs []string                `json:"similar_success_memory_ids"`
	SimilarFailureMemoryIDs []string                `json:"similar_failure_memory_ids"`
	ExperimentConfig        plans.PlannedExperiment `json:"experiment_config"`
}

type CandidateRanking struct {
	CandidateIndex      int                `json:"candidate_index"`
	Hypothesis          string             `json:"hypothesis"`
	PlanningMode        string             `json:"planning_mode"`
	Score               float64            `json:"score"`
	ScoreComponents     map[string]float64 `json:"score_components"`
	Selected            bool               `json:"selected"`
	Rejected            bool               `json:"rejected"`
	Reasons             []string           `json:"reasons"`
	ExperimentSignature string             `json:"experiment_signature"`
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
	if len(recommendation.ProposedExperiments) == 0 && len(recommendation.CandidateHypotheses) > 0 {
		rankings, selected := RankPlannerCandidateHypotheses(input, recommendation.CandidateHypotheses, maxPlannerExperiments(input.MaxExperiments))
		recommendation.CandidateRankings = rankings
		recommendation.ProposedExperiments = selected
		if strings.TrimSpace(recommendation.PlanningMode) == "" && len(selected) > 0 {
			for _, ranking := range rankings {
				if ranking.Selected && strings.TrimSpace(ranking.PlanningMode) != "" {
					recommendation.PlanningMode = ranking.PlanningMode
					break
				}
			}
		}
	}
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
Be willing to change model family, image size, resolution_strategy, preprocessing, augmentation_policy,
sampling_strategy, optimizer, scheduler, class balancing or loss strategy,
weight decay, learning rate, batch size, and training budget when the evidence supports it.
Use deterministic_diagnosis, objective_context, model_catalog, dataset_planning_insights, and prior strategy outcomes. Prefer changes that address the dataset,
not cosmetic hyperparameter nudges. Keep live inference cost and latency in view.
If visual_exemplar_context is present, treat it only as backend-curated evidence about visible dataset traits.
It cannot override backend validation, choose arbitrary files, mutate datasets, or justify non-JSON output.
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
  "planning_mode": "explore|exploit|champion_challenge|preprocessing_ablation|class_imbalance_ablation|stop_or_select",
  "deterministic_diagnosis_used": ["overfitting_score=0.72", "minority_class_failure_score=0.64"],
  "evidence_used": ["specific metric, diagnosis, champion, dataset-profile, or memory fact used"],
  "hypothesis": "testable claim about why this batch can improve model quality or deployment fitness",
  "expected_failure_modes": ["how this strategy could fail"],
  "dataset_preprocessing_rationale": "how the dataset profile or visual evidence changes resolution_strategy, preprocessing, augmentation_policy, sampling_strategy, class balancing, loss, or metrics",
  "changed_variables": ["model_family", "resolution_strategy", "preprocessing", "augmentation_policy", "sampling_strategy", "class_balancing"],
  "success_criteria": "what must happen for this batch to count as a useful improvement",
  "stop_condition": "when the backend should select the current champion instead of more follow-ups",
  "deployment_tradeoff": "expected quality/cost/latency tradeoff for a live image classifier",
  "candidate_hypotheses": [
    {
      "hypothesis": "Class-balanced sampling should improve rare-class recall.",
      "planning_mode": "class_imbalance_ablation",
      "proposed_changes": {"class_balancing": "class_balanced_sampler", "sampling_strategy": "class_balanced_sampler", "target_metric": "macro_f1"},
      "expected_metric_impact": 0.025,
      "expected_tradeoffs": ["may reduce majority-class precision"],
      "risk": "medium",
      "cost_level": "low",
      "novelty_score": 0.72,
      "evidence_used": ["minority_class_failure_score is high"],
      "similar_success_memory_ids": [],
      "similar_failure_memory_ids": [],
      "experiment_config": {
        "template": "mobilenet_transfer",
        "model": "mobilenet_v3_large",
        "epochs": 12,
        "batch_size": 16,
        "learning_rate": 0.0003,
        "reason": "Tests class-balanced sampling against minority recall failure.",
        "image_size": 224,
        "resolution_strategy": "low_latency",
        "preprocessing": {
          "resize_strategy": "preserve_aspect_pad",
          "normalization": "imagenet",
          "crop_strategy": "none",
          "bbox_mode": "ignore",
          "use_dataset_normalization": false
        },
        "optimizer": "adamw",
        "scheduler": "cosine",
        "weight_decay": 0.01,
        "augmentation": {"horizontal_flip": true, "color_jitter": true},
        "augmentation_policy": "moderate",
        "class_balancing": "class_balanced_sampler",
        "sampling_strategy": "class_balanced_sampler",
        "early_stopping_patience": 4,
        "strategy": "class imbalance ablation",
        "pretrained": true,
        "freeze_backbone": true,
        "fine_tune_strategy": "head_only"
      }
    }
  ],
  "proposed_experiments": [
    {
      "template": "efficientnet_transfer",
      "model": "efficientnet_b0",
      "epochs": 10,
      "batch_size": 16,
      "learning_rate": 0.0002,
      "reason": "why this experiment is useful",
      "image_size": 224,
      "resolution_strategy": "fixed",
      "preprocessing": {
        "resize_strategy": "random_resized_crop",
        "normalization": "imagenet",
        "crop_strategy": "random_resized_crop",
        "bbox_mode": "ignore",
        "use_dataset_normalization": false
      },
      "optimizer": "adamw",
      "scheduler": "cosine",
      "weight_decay": 0.01,
      "augmentation": {"horizontal_flip": true, "color_jitter": true, "random_crop": true},
      "augmentation_policy": "moderate",
      "class_balancing": "weighted_loss",
      "sampling_strategy": "none",
      "early_stopping_patience": 3,
      "strategy": "focused efficientnet improvement",
      "pretrained": true,
      "freeze_backbone": true,
      "fine_tune_strategy": "head_only"
    }
  ],
  "champion_job_id": "",
  "why_can_beat_champion": "specific reason these experiments can beat the current champion",
  "expected_delta_vs_champion": 0.02,
  "stop_reason": "",
  "risks": ["risk"],
  "expected_tradeoffs": ["tradeoff"],
  "novelty_notes": ["what changed from prior experiments"],
  "rejected_options": [
    {
      "option": "Same model with two more epochs",
      "reason": "Prior epochs plateaued and this does not address the diagnosis.",
      "evidence": "plateau_score is high",
      "applies_when": ["plateau", "same_model"]
    }
  ],
  "tags": ["short_tag"]
}

Rules:
- If decision_type is ADD_EXPERIMENTS, propose 1-5 complete, novel experiments.
- Prefer returning 6-12 candidate_hypotheses. The backend will score/rank candidates and select 1-5 final proposed_experiments if proposed_experiments is empty.
- If you include both candidate_hypotheses and proposed_experiments, proposed_experiments must be the strongest 1-5 after your own ranking.
- Use only model names from model_catalog.
- Use only supported optimizers: adamw, adam, sgd.
- Use only supported schedulers: none, cosine, step.
- Use only supported resolution_strategy values: fixed, low_latency, compare_224_256, high_resolution_ablation.
- Use preprocessing.resize_strategy values: squash, preserve_aspect_pad, center_crop, random_resized_crop, bbox_crop_if_available.
- Use preprocessing.normalization values: imagenet, dataset, none.
- Use preprocessing.crop_strategy values: none, center_crop, random_resized_crop, bbox_crop_if_available, bbox_crop_ablation.
- Use preprocessing.bbox_mode values: ignore, crop_if_available, crop_and_compare_full_image, use_boxes_as_metadata.
- Use augmentation_policy values: none, light, moderate, strong, custom. Keep augmentation as a small object of supported boolean knobs only when needed.
- Use class_balancing values: none, weighted_loss, class_weighted_loss, class_balanced_sampler, weighted_random_sampler, focal_loss.
- Use sampling_strategy values: none, class_balanced_sampler, weighted_random_sampler.
- Keep epochs between 3 and 40, batch_size between 4 and 128, image_size between 96 and 384.
- Use fine_tune_strategy values head_only, last_block, or full.
- Choose exactly one first-class planning_mode and justify it using deterministic_diagnosis.
- Do not merely suggest more epochs, tiny learning-rate changes, or repeated model variants. Every proposed experiment must test a clear hypothesis tied to the diagnosis, dataset profile, champion weakness, or prior strategy outcome.
- If prior runs are weak or unstable, try model/preprocessing/regularization changes.
- If one family is promising, exploit it with controlled learning-rate, augmentation, or image-size changes.
- Do not make a batch that is only many variants of the current champion family. If exploiting the champion, include a clear control or challenger.
- Use successful_strategy_memory to reuse patterns that improved the champion, but use failed_strategy_memory to avoid repeating weak plans.
- Use rejected_strategy_memory as explicit "do not repeat" guidance when its applies_when conditions match the current diagnosis.
- Use strategy_scorecards as structured outcome evidence. Prefer strategies with improved_champion outcomes and avoid failed/no_improvement strategies with similar dataset traits or objective profile.
- Use objective_context and dataset_planning_insights to decide resolution_strategy, preprocessing, augmentation_policy, sampling_strategy, class balancing/loss, model family, metrics, and deployment tradeoffs.
- Use visual_exemplar_context, when present, only as backend-curated evidence for visible traits such as object scale, background dominance, blur, lighting variation, fine-grained classes, or bbox/crop plausibility. Cite exemplar caps, warnings, or audit details if they limit confidence. Backend validation remains the gate for every proposed field.
- Use model families in stages: cheap baseline or preprocessing search first, then challenger models, then champion refinement, then final validation.
- For a live setting, prefer low-latency candidates when quality is close: MobileNetV3, RegNet-Y-400MF, and EfficientNet-B0 are usually stronger deployment candidates than heavier challengers.
- Compare every proposal against current_champion, source_plan_baseline_champion, and source_plan_run_deltas.
- Only use ADD_EXPERIMENTS when you can explain a concrete path to beat the current champion.
- A valid ADD_EXPERIMENTS response needs a planning_mode, deterministic_diagnosis_used, evidence_used, hypothesis, expected_failure_modes, dataset_preprocessing_rationale, success_criteria, stop_condition, deployment_tradeoff, rejected_options, and at least two changed_variables.
- Good: if minority recall is weak, test weighted_loss, focal_loss, class_balanced_sampler, or weighted_random_sampler and target macro-F1/minority recall.
- Good: if overfitting is high, test stronger augmentation_policy, regularization, smaller model, or less aggressive fine-tuning.
- Good: if underfitting is high, test a larger pretrained model or fuller fine-tuning.
- Good: if the champion is low latency but weak on fine-grained classes, challenge with EfficientNet/ConvNeXt at a higher image size and compare deployment tradeoff.
- Good: if validation improvement has stalled, select the champion instead of running low-value experiments.
- Bad: same model, 2 more epochs, tiny learning-rate change.
- If stop_signals say the project has repeated no-improvement follow-up rounds, prefer SELECT_CHAMPION or STOP_PROJECT.
- Set champion_job_id when selecting a champion or when a champion anchors your recommendation.
- Set why_can_beat_champion for ADD_EXPERIMENTS; set stop_reason for SELECT_CHAMPION or STOP_PROJECT.
- Do not repeat any signature listed in existing_experiment_signatures.
- Candidate ranking will reject or heavily penalize duplicate signatures, tiny-only changes, high-cost weak-justification experiments, failed strategies with similar traits, objective misalignment, and ideas not tied to deterministic_diagnosis.

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
		if strings.TrimSpace(recommendation.PlanningMode) == "" {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing planning_mode")
		}
		if err := validatePlanningModeName(recommendation.PlanningMode); err != nil {
			return err
		}
		if len(nonEmptyStrings(recommendation.DeterministicDiagnosisUsed)) == 0 {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing deterministic_diagnosis_used")
		}
		if len(nonEmptyStrings(recommendation.EvidenceUsed)) == 0 {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing evidence_used")
		}
		if strings.TrimSpace(recommendation.Hypothesis) == "" {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing hypothesis")
		}
		if len(nonEmptyStrings(recommendation.ExpectedFailureModes)) == 0 {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing expected_failure_modes")
		}
		if strings.TrimSpace(recommendation.DatasetPreprocessingRationale) == "" {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing dataset_preprocessing_rationale")
		}
		changedVariables := nonEmptyStrings(recommendation.ChangedVariables)
		if len(changedVariables) < 2 {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS needs at least two changed_variables")
		}
		if onlyMinorChangedVariables(changedVariables) {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS changed_variables are only minor tuning knobs")
		}
		if strings.TrimSpace(recommendation.SuccessCriteria) == "" {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing success_criteria")
		}
		if strings.TrimSpace(recommendation.StopCondition) == "" {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing stop_condition")
		}
		if strings.TrimSpace(recommendation.DeploymentTradeoff) == "" {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing deployment_tradeoff")
		}
		if len(recommendation.RejectedOptions) == 0 {
			return fmt.Errorf("experiment planner ADD_EXPERIMENTS missing rejected_options")
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
		if err := validatePlannerExperimentDiversity(recommendation.ProposedExperiments); err != nil {
			return err
		}
		if err := validatePlanningModeRules(recommendation); err != nil {
			return err
		}
	case decisions.TypeSelectChampion, decisions.TypeStopProject, decisions.TypeWait:
		if strings.TrimSpace(recommendation.PlanningMode) != "" {
			if err := validatePlanningModeName(recommendation.PlanningMode); err != nil {
				return err
			}
		}
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
	if recommendation.RejectedOptions == nil {
		recommendation.RejectedOptions = []RejectedPlannerOption{}
	}
	if recommendation.Tags == nil {
		recommendation.Tags = []string{}
	}
	return nil
}

func validatePlanningModeName(mode string) error {
	switch strings.ToLower(strings.TrimSpace(mode)) {
	case "explore", "exploit", "champion_challenge", "preprocessing_ablation", "class_imbalance_ablation", "stop_or_select":
		return nil
	default:
		return fmt.Errorf("experiment planner has invalid planning_mode %q", mode)
	}
}

func validatePlanningModeRules(recommendation ExperimentPlanningRecommendation) error {
	mode := strings.ToLower(strings.TrimSpace(recommendation.PlanningMode))
	switch mode {
	case "explore":
		if countExperimentFamilies(recommendation.ProposedExperiments) < 2 {
			return fmt.Errorf("experiment planner explore mode must include at least two model families")
		}
	case "exploit":
		justification := strings.ToLower(strings.Join([]string{
			recommendation.Rationale,
			recommendation.Hypothesis,
			recommendation.WhyCanBeatChampion,
		}, " "))
		if !containsAnyText(justification, "promising", "best", "champion", "prior", "previous", "worth", "strong") {
			return fmt.Errorf("experiment planner exploit mode must justify why the chosen family is worth refining")
		}
	case "champion_challenge":
		if len(nonEmptyStrings(recommendation.ExpectedTradeoffs)) == 0 {
			return fmt.Errorf("experiment planner champion_challenge mode must include expected tradeoff versus champion")
		}
		for index, experiment := range recommendation.ProposedExperiments {
			text := strings.ToLower(strings.TrimSpace(experiment.Reason + " " + experiment.Strategy))
			if !containsAnyText(text, "champion", "beat", "challenge", "tradeoff", "improve") {
				return fmt.Errorf("experiment planner champion_challenge experiment %d must explain how it can beat the current champion", index)
			}
		}
	case "preprocessing_ablation":
		changed := strings.ToLower(strings.Join(recommendation.ChangedVariables, " "))
		if !containsAnyText(changed, "preprocessing", "augmentation", "augmentation_policy", "resolution_strategy", "sampling_strategy", "image_size", "resize", "class_balancing", "weighted_loss", "crop", "normalization") {
			return fmt.Errorf("experiment planner preprocessing_ablation mode must isolate preprocessing, augmentation, image size, crop, or class-balancing changes")
		}
	case "class_imbalance_ablation":
		for index, experiment := range recommendation.ProposedExperiments {
			text := strings.ToLower(strings.TrimSpace(experiment.ClassBalancing + " " + experiment.SamplingStrategy + " " + experiment.Reason + " " + experiment.Strategy))
			if !containsAnyText(text, "weight", "weighted", "focal", "balance", "balanced", "sampler", "minority") {
				return fmt.Errorf("experiment planner class_imbalance_ablation experiment %d must include a class balancing strategy", index)
			}
		}
		criteria := strings.ToLower(recommendation.SuccessCriteria + " " + recommendation.Hypothesis)
		if !containsAnyText(criteria, "macro", "minority", "recall", "per-class", "per class", "f1") {
			return fmt.Errorf("experiment planner class_imbalance_ablation mode must target macro-F1, minority recall, or per-class metrics")
		}
	case "stop_or_select":
		if strings.ToUpper(strings.TrimSpace(recommendation.DecisionType)) == decisions.TypeAddExperiments && recommendation.ExpectedDeltaVsChampion < 0.02 {
			return fmt.Errorf("experiment planner stop_or_select mode should not propose more experiments without strong expected evidence")
		}
	}
	return nil
}

func countExperimentFamilies(experiments []plans.PlannedExperiment) int {
	families := map[string]bool{}
	for _, experiment := range experiments {
		family := inferExperimentFamily(experiment.Model)
		if family != "" {
			families[family] = true
		}
	}
	return len(families)
}

func inferExperimentFamily(model string) string {
	normalized := strings.ToLower(strings.TrimSpace(model))
	switch {
	case strings.HasPrefix(normalized, "mobilenet"):
		return "mobilenet"
	case strings.HasPrefix(normalized, "efficientnet"):
		return "efficientnet"
	case strings.HasPrefix(normalized, "regnet"):
		return "regnet"
	case strings.HasPrefix(normalized, "resnet"):
		return "resnet"
	case strings.HasPrefix(normalized, "convnext"):
		return "convnext"
	case strings.HasPrefix(normalized, "swin"):
		return "swin"
	case strings.HasPrefix(normalized, "vit"):
		return "vit"
	default:
		return normalized
	}
}

func containsAnyText(value string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(value, needle) {
			return true
		}
	}
	return false
}

func nonEmptyStrings(values []string) []string {
	out := []string{}
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			out = append(out, value)
		}
	}
	return out
}

func onlyMinorChangedVariables(values []string) bool {
	minor := map[string]bool{
		"epoch":         true,
		"epochs":        true,
		"learning_rate": true,
		"lr":            true,
		"batch_size":    true,
	}
	for _, value := range values {
		normalized := strings.ToLower(strings.TrimSpace(value))
		if !minor[normalized] {
			return false
		}
	}
	return true
}

func validatePlannerExperimentDiversity(experiments []plans.PlannedExperiment) error {
	if len(experiments) < 3 {
		return nil
	}
	models := map[string]bool{}
	for _, experiment := range experiments {
		models[strings.ToLower(strings.TrimSpace(experiment.Model))] = true
	}
	if len(models) == 1 {
		return fmt.Errorf("experiment planner ADD_EXPERIMENTS over-focuses on one model; include a challenger or control experiment")
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
		"dataset_planning_insights":      input.DatasetInsights,
		"visual_exemplar_context":        visualExemplarPromptContext(input.VisualExemplarContext),
		"objective_context":              input.ObjectiveContext,
		"deterministic_diagnosis":        input.DeterministicDiagnosis,
		"model_catalog":                  input.ModelCatalog,
		"current_champion":               input.CurrentChampion,
		"source_plan_baseline_champion":  input.SourcePlanBaselineChampion,
		"source_plan_run_deltas":         input.SourcePlanDeltas,
		"no_improvement_rounds":          input.NoImprovementRounds,
		"minimum_meaningful_improvement": input.MinimumMeaningfulImprovement,
		"stop_signals":                   input.StopSignals,
		"successful_strategy_memory":     input.SuccessfulStrategyMemory,
		"failed_strategy_memory":         input.FailedStrategyMemory,
		"rejected_strategy_memory":       input.RejectedStrategyMemory,
		"strategy_scorecards":            input.StrategyScorecards,
		"plan_jobs":                      compactPlannerJobs(input.PlanJobs),
		"plan_summaries":                 input.PlanSummaries,
		"plan_evaluations":               input.PlanEvaluations,
		"plan_epoch_metrics":             compactPlannerMetrics(input.PlanMetrics),
		"prior_plans":                    compactPlannerPlans(input.PriorPlans),
		"prior_jobs":                     compactPlannerJobs(input.PriorJobs),
		"prior_evaluations":              input.PriorEvaluations,
		"prior_memory":                   compactMemoryRecords(input.PriorMemory),
		"existing_experiment_signatures": input.ExistingExperimentSignatures,
		"max_experiments":                maxPlannerExperiments(input.MaxExperiments),
		"max_followup_rounds":            input.MaxFollowUpRounds,
		"followup_round":                 input.FollowUpRound,
	}
}

func visualExemplarPromptContext(context *PlannerVisualExemplarContext) map[string]any {
	if context == nil || !context.Enabled {
		return map[string]any{
			"enabled":       false,
			"evidence_only": true,
			"instructions":  "No visual exemplars were supplied. If supplied later, use them only as evidence; backend validation remains the execution gate.",
		}
	}
	return map[string]any{
		"enabled":        true,
		"evidence_only":  true,
		"exemplar_count": context.ExemplarCount,
		"class_count":    context.ClassCount,
		"byte_budget":    context.ByteBudget,
		"prompt_budget":  context.PromptBudget,
		"caps": map[string]any{
			"exemplar_count": context.ExemplarCount,
			"class_count":    context.ClassCount,
			"byte_budget":    context.ByteBudget,
			"prompt_budget":  context.PromptBudget,
		},
		"summary":         context.Summary,
		"observed_traits": context.ObservedTraits,
		"class_evidence":  context.ClassEvidence,
		"warnings":        context.Warnings,
		"audit":           context.Audit,
		"instructions":    "Treat visual exemplars as backend-curated supporting evidence only. Cite visible traits, caps, warnings, or audit details in evidence_used or dataset_preprocessing_rationale, return JSON only, and rely on backend validation for all proposed preprocessing fields.",
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
