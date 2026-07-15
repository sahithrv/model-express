package evals

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/llm"
)

const PlannerPairedEvalSchemaVersionV1 = "planner_paired_eval_v1"

type PlannerEvalVariant struct {
	Name                string                                 `json:"name"`
	StaticPromptVersion string                                 `json:"static_prompt_version"`
	ContextVersion      string                                 `json:"context_version"`
	RequestVariant      agents.ExperimentPlannerRequestVariant `json:"-"`
}

type PlannerEvalBudget struct {
	MaxProviderCalls int `json:"max_provider_calls"`
	MaxRequestBytes  int `json:"max_request_bytes"`
	MaxTotalTokens   int `json:"max_total_tokens,omitempty"`
}

type PlannerEvalBudgetUsage struct {
	ProviderCalls         int `json:"provider_calls"`
	ProviderCallsReserved int `json:"provider_calls_reserved"`
	RequestBytes          int `json:"request_bytes"`
	TotalTokens           int `json:"total_tokens"`
}

type PlannerPairedEvalConfig struct {
	Variants    []PlannerEvalVariant
	Repeats     int
	MaxAttempts int
	Budget      PlannerEvalBudget
	Pricing     *llm.PricingSnapshot
}

type PlannerEvalAttempt struct {
	AttemptIndex          int                           `json:"attempt_index"`
	RetryReason           string                        `json:"retry_reason,omitempty"`
	AgentVersion          string                        `json:"agent_version"`
	PromptVersion         string                        `json:"prompt_version"`
	ContextBuilderVersion string                        `json:"context_builder_version"`
	ValidatorMode         string                        `json:"validator_mode"`
	RankerMultiFidelity   bool                          `json:"ranker_multi_fidelity"`
	RequestBytes          int                           `json:"request_bytes"`
	RequestSHA256         string                        `json:"request_sha256"`
	FinalizerInputSHA256  string                        `json:"finalizer_input_sha256"`
	SystemPromptBytes     int                           `json:"system_prompt_bytes"`
	UserPromptBytes       int                           `json:"user_prompt_bytes"`
	WallLatencyMS         float64                       `json:"wall_latency_ms"`
	Usage                 *llm.Usage                    `json:"usage,omitempty"`
	DerivedCost           *llm.DerivedCost              `json:"derived_cost,omitempty"`
	ValidationStatus      string                        `json:"validation_status"`
	ValidationError       string                        `json:"validation_error,omitempty"`
	ParsedResult          map[string]any                `json:"parsed_result,omitempty"`
	ToolRounds            int                           `json:"tool_rounds"`
	ToolCalls             []agents.AgentToolCallTrace   `json:"tool_calls,omitempty"`
	ToolResults           []agents.AgentToolResultTrace `json:"tool_results,omitempty"`
	RejectedToolCalls     []agents.AgentToolResultTrace `json:"rejected_tool_calls,omitempty"`
	DryRunValidations     []map[string]any              `json:"dry_run_validations,omitempty"`
	Rubric                PlannerRubricScore            `json:"rubric"`
}

type PlannerEvalRuntime struct {
	Provider               string `json:"provider"`
	APIStyle               string `json:"api_style"`
	EffectiveAPIStyle      string `json:"effective_api_style"`
	Model                  string `json:"model"`
	ConfiguredReasoning    string `json:"configured_reasoning_effort,omitempty"`
	PlateauReasoning       string `json:"plateau_reasoning_effort,omitempty"`
	MaxToolRounds          int    `json:"max_tool_rounds"`
	MaxProviderRetries     int    `json:"max_provider_retries"`
	ToolPolicyVersion      string `json:"tool_policy_version"`
	ValidatorVersion       string `json:"validator_version"`
	RankerVersion          string `json:"ranker_version"`
	RetrievalPolicyVersion string `json:"retrieval_policy_version"`
}

type PlannerEvalRun struct {
	Variant         PlannerEvalVariant   `json:"variant"`
	RepeatIndex     int                  `json:"repeat_index"`
	Attempts        []PlannerEvalAttempt `json:"attempts"`
	FirstPassValid  bool                 `json:"first_pass_valid"`
	EventualValid   bool                 `json:"eventual_valid"`
	AcceptedAttempt int                  `json:"accepted_attempt"`
	FinalRubric     PlannerRubricScore   `json:"final_rubric"`
}

type PlannerPairedEvalSummary struct {
	RunCount            int `json:"run_count"`
	FirstPassValidCount int `json:"first_pass_valid_count"`
	EventualValidCount  int `json:"eventual_valid_count"`
	RubricPassedCount   int `json:"rubric_passed_count"`
}

type PlannerPairedEvalArtifact struct {
	SchemaVersion  string                   `json:"schema_version"`
	FixtureName    string                   `json:"fixture_name"`
	TaskType       string                   `json:"task_type"`
	ReadOnly       bool                     `json:"read_only"`
	Repeats        int                      `json:"repeats"`
	MaxAttempts    int                      `json:"max_attempts"`
	Runtime        PlannerEvalRuntime       `json:"runtime"`
	Budget         PlannerEvalBudget        `json:"budget"`
	BudgetUsage    PlannerEvalBudgetUsage   `json:"budget_usage"`
	PricingVersion string                   `json:"pricing_version,omitempty"`
	Runs           []PlannerEvalRun         `json:"runs"`
	Summary        PlannerPairedEvalSummary `json:"summary"`
}

func DefaultPlannerEvalVariants() []PlannerEvalVariant {
	return []PlannerEvalVariant{
		plannerEvalVariant("current_v1", "v1", "v1"),
		plannerEvalVariant("compact_static_prompt", "compact_v1", "v1"),
		plannerEvalVariant("context_v2", "compact_v1", "v2"),
	}
}

func plannerEvalVariant(name string, staticPromptVersion string, contextVersion string) PlannerEvalVariant {
	return PlannerEvalVariant{
		Name:                name,
		StaticPromptVersion: staticPromptVersion,
		ContextVersion:      contextVersion,
		RequestVariant: agents.ExperimentPlannerRequestVariant{
			StaticPromptVersion: staticPromptVersion,
			ContextVersion:      contextVersion,
		},
	}
}

// RunPlannerPairedEvaluation performs generation only through the supplied
// generator and has no store dependency. It cannot write projects, plans,
// jobs, memory, decisions, or champions by construction.
func RunPlannerPairedEvaluation(
	ctx context.Context,
	generator llm.JSONGenerator,
	model string,
	runtime llm.Config,
	fixture PlannerReplayFixture,
	config PlannerPairedEvalConfig,
) (PlannerPairedEvalArtifact, error) {
	if err := validatePairedEvalConfig(runtime, model, config); err != nil {
		return PlannerPairedEvalArtifact{}, err
	}
	input := ExperimentPlannerInputFromReplayFixture(fixture)
	finalizerInputBlob, err := json.Marshal(input)
	if err != nil {
		return PlannerPairedEvalArtifact{}, fmt.Errorf("encode immutable finalizer input: %w", err)
	}
	finalizerInputSHA256 := evalSHA256(finalizerInputBlob)
	rubric := PlannerRubricForFixture(fixture)
	agent := agents.NewExperimentPlannerAgentWithRuntime(generator, model, runtime, agents.PlannerInformationToolOptions{})
	artifact := PlannerPairedEvalArtifact{
		SchemaVersion: PlannerPairedEvalSchemaVersionV1,
		FixtureName:   fixture.Name,
		TaskType:      replayTaskType(input),
		ReadOnly:      true,
		Repeats:       config.Repeats,
		MaxAttempts:   config.MaxAttempts,
		Runtime: PlannerEvalRuntime{
			Provider:               runtime.Provider,
			APIStyle:               runtime.APIStyle,
			EffectiveAPIStyle:      llm.EffectiveAPIStyle(runtime.Provider, runtime.APIStyle),
			Model:                  model,
			ConfiguredReasoning:    runtime.ReasoningEffort,
			PlateauReasoning:       runtime.PlateauReasoningEffort,
			MaxToolRounds:          runtime.MaxToolRounds,
			MaxProviderRetries:     runtime.MaxRetries,
			ToolPolicyVersion:      agents.ExperimentPlannerToolPolicyVersion,
			ValidatorVersion:       agents.ExperimentPlannerValidatorVersion,
			RankerVersion:          agents.ExperimentPlannerRankerVersion,
			RetrievalPolicyVersion: agents.ExperimentPlannerRetrievalPolicyVersion,
		},
		Budget: config.Budget,
	}
	if config.Pricing != nil {
		artifact.PricingVersion = config.Pricing.PricingVersion
	}

	for repeat := 0; repeat < config.Repeats; repeat++ {
		for _, variant := range config.Variants {
			run := PlannerEvalRun{Variant: variant, RepeatIndex: repeat, AcceptedAttempt: -1}
			retryReason := ""
			for attemptIndex := 0; attemptIndex < config.MaxAttempts; attemptIndex++ {
				built, err := agent.BuildRequest(input, variant.RequestVariant)
				if err != nil {
					return artifact, fmt.Errorf("build variant %q request: %w", variant.Name, err)
				}
				requestBlob, err := json.Marshal(built.Request)
				if err != nil {
					return artifact, fmt.Errorf("measure variant %q request: %w", variant.Name, err)
				}
				if err := reserveEvalBudget(&artifact.BudgetUsage, config.Budget, len(requestBlob), runtime.MaxToolRounds); err != nil {
					return artifact, fmt.Errorf("variant %q repeat %d attempt %d: %w", variant.Name, repeat, attemptIndex, err)
				}

				started := time.Now()
				trace, _ := agent.PlanWithVariantTrace(ctx, input, variant.RequestVariant)
				latency := float64(time.Since(started).Nanoseconds()) / float64(time.Millisecond)
				actualRequestBlob, err := json.Marshal(trace.Request)
				if err != nil {
					return artifact, fmt.Errorf("measure actual variant %q request: %w", variant.Name, err)
				}
				if string(requestBlob) != string(actualRequestBlob) {
					return artifact, fmt.Errorf("variant %q request changed between budget preflight and generation", variant.Name)
				}

				providerCalls := 1 + trace.ToolRounds
				artifact.BudgetUsage.ProviderCalls += providerCalls
				artifact.BudgetUsage.RequestBytes += len(actualRequestBlob)
				if trace.Usage != nil {
					artifact.BudgetUsage.TotalTokens += trace.Usage.TotalTokens
				}
				if config.Budget.MaxTotalTokens > 0 && artifact.BudgetUsage.TotalTokens > config.Budget.MaxTotalTokens {
					return artifact, fmt.Errorf("observed token budget exceeded: %d > %d", artifact.BudgetUsage.TotalTokens, config.Budget.MaxTotalTokens)
				}

				if config.Pricing != nil && trace.Usage != nil && strings.TrimSpace(trace.Usage.RequestModel) != "" && !config.Pricing.MatchesRuntime(runtime.Provider, trace.Usage.RequestModel) {
					return artifact, fmt.Errorf("pricing snapshot %q does not match provider-reported runtime %s/%s", config.Pricing.PricingVersion, runtime.Provider, trace.Usage.RequestModel)
				}
				derivedCost, err := llm.DeriveCost(trace.Usage, config.Pricing)
				if err != nil {
					return artifact, fmt.Errorf("derive variant %q cost: %w", variant.Name, err)
				}
				rubricScore := ScorePlannerRubric(input, trace.RawOutput, rubric)
				attempt := PlannerEvalAttempt{
					AttemptIndex:          attemptIndex,
					RetryReason:           retryReason,
					AgentVersion:          trace.AgentVersion,
					PromptVersion:         trace.PromptVersion,
					ContextBuilderVersion: trace.ContextBuilderVersion,
					ValidatorMode:         trace.ValidatorMode,
					RankerMultiFidelity:   trace.RankerMultiFidelity,
					RequestBytes:          len(actualRequestBlob),
					RequestSHA256:         evalSHA256(actualRequestBlob),
					FinalizerInputSHA256:  finalizerInputSHA256,
					SystemPromptBytes:     requestMessageBytes(trace.Request, "system"),
					UserPromptBytes:       requestMessageBytes(trace.Request, "user"),
					WallLatencyMS:         latency,
					Usage:                 trace.Usage,
					DerivedCost:           derivedCost,
					ValidationStatus:      trace.ValidationStatus,
					ValidationError:       trace.ValidationError,
					ParsedResult:          trace.ParsedOutput,
					ToolRounds:            trace.ToolRounds,
					ToolCalls:             trace.ToolCalls,
					ToolResults:           trace.ToolResults,
					RejectedToolCalls:     trace.RejectedToolCalls,
					DryRunValidations:     trace.DryRunValidationResults,
					Rubric:                rubricScore,
				}
				run.Attempts = append(run.Attempts, attempt)
				if attemptIndex == 0 {
					run.FirstPassValid = rubricScore.BackendSchedulable
				}
				if rubricScore.BackendSchedulable {
					run.EventualValid = true
					run.AcceptedAttempt = attemptIndex
					run.FinalRubric = rubricScore
					break
				}
				if len(trace.RawOutput) == 0 {
					retryReason = "provider_error"
				} else {
					retryReason = "backend_validation_rejected"
				}
				run.FinalRubric = rubricScore
			}
			artifact.Runs = append(artifact.Runs, run)
			artifact.Summary.RunCount++
			if run.FirstPassValid {
				artifact.Summary.FirstPassValidCount++
			}
			if run.EventualValid {
				artifact.Summary.EventualValidCount++
			}
			if run.FinalRubric.Passed {
				artifact.Summary.RubricPassedCount++
			}
		}
	}
	return artifact, nil
}

func validatePairedEvalConfig(runtime llm.Config, model string, config PlannerPairedEvalConfig) error {
	if len(config.Variants) == 0 {
		return fmt.Errorf("paired evaluation requires at least one variant")
	}
	if config.Repeats < 1 || config.Repeats > 20 {
		return fmt.Errorf("paired evaluation repeats must be between 1 and 20")
	}
	if config.MaxAttempts < 1 || config.MaxAttempts > 3 {
		return fmt.Errorf("paired evaluation max attempts must be between 1 and 3")
	}
	maxToolRounds := runtime.MaxToolRounds
	if maxToolRounds <= 0 {
		maxToolRounds = llm.DefaultMaxToolRounds
	}
	firstPassReservation := len(config.Variants) * config.Repeats * (1 + maxToolRounds)
	if config.Budget.MaxProviderCalls < firstPassReservation {
		return fmt.Errorf("provider-call budget %d cannot cover the first-pass reservation of %d", config.Budget.MaxProviderCalls, firstPassReservation)
	}
	if config.Budget.MaxRequestBytes <= 0 {
		return fmt.Errorf("paired evaluation requires a positive request-byte budget")
	}
	if config.Pricing != nil {
		if err := config.Pricing.Validate(); err != nil {
			return fmt.Errorf("invalid evaluation pricing snapshot: %w", err)
		}
		if !config.Pricing.MatchesRuntime(runtime.Provider, model) {
			return fmt.Errorf("pricing snapshot %q does not match runtime %s/%s", config.Pricing.PricingVersion, runtime.Provider, model)
		}
	}
	for _, variant := range config.Variants {
		if strings.TrimSpace(variant.Name) == "" {
			return fmt.Errorf("paired evaluation variant name is required")
		}
	}
	return nil
}

func reserveEvalBudget(usage *PlannerEvalBudgetUsage, budget PlannerEvalBudget, requestBytes int, maxToolRounds int) error {
	if maxToolRounds <= 0 {
		maxToolRounds = llm.DefaultMaxToolRounds
	}
	providerCallReservation := 1 + maxToolRounds
	if usage.ProviderCallsReserved+providerCallReservation > budget.MaxProviderCalls {
		return fmt.Errorf("provider-call budget exhausted: reserving %d could exceed %d", providerCallReservation, budget.MaxProviderCalls)
	}
	if usage.RequestBytes+requestBytes > budget.MaxRequestBytes {
		return fmt.Errorf("request-byte budget exhausted: %d + %d > %d", usage.RequestBytes, requestBytes, budget.MaxRequestBytes)
	}
	usage.ProviderCallsReserved += providerCallReservation
	return nil
}

func requestMessageBytes(request llm.JSONRequest, role string) int {
	total := 0
	for _, message := range request.Messages {
		if message.Role == role {
			total += len([]byte(message.Content))
		}
	}
	return total
}

func evalSHA256(value []byte) string {
	digest := sha256.Sum256(value)
	return "sha256:" + hex.EncodeToString(digest[:])
}
