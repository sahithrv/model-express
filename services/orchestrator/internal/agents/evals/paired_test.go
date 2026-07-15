package evals

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/llm"
)

type pairedFakeGenerator struct {
	responses [][]byte
	requests  []llm.JSONRequest
	usage     llm.Usage
	delay     time.Duration
}

func (g *pairedFakeGenerator) GenerateJSON(ctx context.Context, request llm.JSONRequest) ([]byte, error) {
	result, err := g.GenerateJSONWithUsage(ctx, request)
	return result.RawJSON, err
}

func (g *pairedFakeGenerator) GenerateJSONWithUsage(_ context.Context, request llm.JSONRequest) (llm.JSONResult, error) {
	if g.delay > 0 {
		time.Sleep(g.delay)
	}
	g.requests = append(g.requests, request)
	index := len(g.requests) - 1
	if index >= len(g.responses) {
		index = len(g.responses) - 1
	}
	usage := g.usage
	return llm.JSONResult{RawJSON: append([]byte(nil), g.responses[index]...), Usage: &usage}, nil
}

func TestPairedEvaluationGeneratesEveryVariantAndRepeatIndependently(t *testing.T) {
	fixture := loadClassificationFixture(t)
	raw, err := ReplayPlannerResponse(fixture)
	if err != nil {
		t.Fatal(err)
	}
	generator := &pairedFakeGenerator{
		responses: [][]byte{raw},
		usage: llm.Usage{
			InputTokens: 100, OutputTokens: 25, TotalTokens: 125, RequestModel: "planner-test-model", APIStyle: llm.APIStyleChatCompletions,
		},
		delay: 2 * time.Millisecond,
	}
	runtime := llm.Config{
		Enabled:       true,
		Provider:      llm.ProviderLocal,
		Model:         "planner-test-model",
		APIStyle:      llm.APIStyleChatCompletions,
		MaxToolRounds: 1,
	}
	pricing := &llm.PricingSnapshot{
		PricingVersion:                 "paired-test-v1",
		Provider:                       llm.ProviderLocal,
		Model:                          runtime.Model,
		InputUSDPerMillionTokens:       "1",
		CachedInputUSDPerMillionTokens: "0.5",
		OutputUSDPerMillionTokens:      "2",
	}
	variants := DefaultPlannerEvalVariants()
	fixtureBefore, _ := json.Marshal(fixture)
	artifact, err := RunPlannerPairedEvaluation(context.Background(), generator, runtime.Model, runtime, fixture, PlannerPairedEvalConfig{
		Variants:    variants,
		Repeats:     2,
		MaxAttempts: 1,
		Budget: PlannerEvalBudget{
			MaxProviderCalls: 12,
			MaxRequestBytes:  1_000_000,
			MaxTotalTokens:   1_000,
		},
		Pricing: pricing,
	})
	if err != nil {
		t.Fatalf("RunPlannerPairedEvaluation() error = %v", err)
	}
	if got, want := len(generator.requests), len(variants)*2; got != want {
		t.Fatalf("independent generator calls = %d, want %d", got, want)
	}
	if got, want := len(artifact.Runs), len(variants)*2; got != want {
		t.Fatalf("runs = %d, want %d", got, want)
	}
	if !artifact.ReadOnly || artifact.Summary.FirstPassValidCount != len(artifact.Runs) || artifact.Summary.EventualValidCount != len(artifact.Runs) {
		t.Fatalf("unexpected paired summary %#v", artifact)
	}
	for runIndex, run := range artifact.Runs {
		if len(run.Attempts) != 1 {
			t.Fatalf("run %d attempts = %d", runIndex, len(run.Attempts))
		}
		attempt := run.Attempts[0]
		requestBlob, _ := json.Marshal(generator.requests[runIndex])
		if attempt.RequestBytes != len(requestBlob) || attempt.RequestSHA256 != evalSHA256(requestBlob) {
			t.Fatalf("attempt %d request measurement does not match actual built request", runIndex)
		}
		if attempt.FinalizerInputSHA256 == "" || attempt.FinalizerInputSHA256 != artifact.Runs[0].Attempts[0].FinalizerInputSHA256 {
			t.Fatalf("attempt %d did not receive the same complete finalizer input", runIndex)
		}
		if attempt.SystemPromptBytes == 0 || attempt.UserPromptBytes == 0 || attempt.WallLatencyMS < 1 {
			t.Fatalf("attempt %d missing local request/latency measurements: %#v", runIndex, attempt)
		}
		if attempt.Usage == nil || attempt.Usage.TotalTokens != 125 || attempt.DerivedCost == nil || attempt.DerivedCost.PricingVersion != pricing.PricingVersion {
			t.Fatalf("attempt %d missing usage/cost: %#v", runIndex, attempt)
		}
		if len(attempt.ParsedResult) == 0 || !attempt.Rubric.Passed {
			t.Fatalf("attempt %d missing parsed/rubric result: %#v", runIndex, attempt)
		}
	}
	fixtureAfter, _ := json.Marshal(fixture)
	if !reflect.DeepEqual(fixtureBefore, fixtureAfter) {
		t.Fatal("paired evaluation mutated fixture-backed project/plan/job/memory/champion input")
	}
}

func TestPromptOnlyVariantsUseIdenticalProductionContext(t *testing.T) {
	input := ExperimentPlannerInputFromReplayFixture(loadClassificationFixture(t))
	agent := agentsForBuildOnly()
	current, err := agent.BuildRequest(input, DefaultPlannerEvalVariants()[0].RequestVariant)
	if err != nil {
		t.Fatal(err)
	}
	compact, err := agent.BuildRequest(input, DefaultPlannerEvalVariants()[1].RequestVariant)
	if err != nil {
		t.Fatal(err)
	}
	currentSnapshot := current.PromptContext["planner_context_snapshot"].(agents.PlannerContextSnapshot)
	compactSnapshot := compact.PromptContext["planner_context_snapshot"].(agents.PlannerContextSnapshot)
	// Prompt-size estimates describe the static variant. The complete semantic
	// snapshot and immutable finalizer input remain the same.
	currentSnapshot.PromptBudget = agents.PlannerPromptBudget{}
	compactSnapshot.PromptBudget = agents.PlannerPromptBudget{}
	if !reflect.DeepEqual(currentSnapshot, compactSnapshot) {
		t.Fatal("prompt-only variants received different semantic planner context")
	}
	if current.Request.Messages[0].Content == compact.Request.Messages[0].Content {
		t.Fatal("prompt-only variants did not use distinct production static prompts")
	}
}

func TestPairedEvaluationRecordsFirstPassAndEventualValidity(t *testing.T) {
	fixture := loadClassificationFixture(t)
	valid, _ := ReplayPlannerResponse(fixture)
	generator := &pairedFakeGenerator{responses: [][]byte{[]byte(`{"summary":"invalid"}`), valid}}
	runtime := llm.Config{Provider: llm.ProviderLocal, Model: "planner-test-model", APIStyle: llm.APIStyleChatCompletions, MaxToolRounds: 1}
	artifact, err := RunPlannerPairedEvaluation(context.Background(), generator, runtime.Model, runtime, fixture, PlannerPairedEvalConfig{
		Variants:    DefaultPlannerEvalVariants()[:1],
		Repeats:     1,
		MaxAttempts: 2,
		Budget:      PlannerEvalBudget{MaxProviderCalls: 4, MaxRequestBytes: 500_000},
	})
	if err != nil {
		t.Fatal(err)
	}
	run := artifact.Runs[0]
	if run.FirstPassValid || !run.EventualValid || run.AcceptedAttempt != 1 || len(run.Attempts) != 2 {
		t.Fatalf("unexpected retry validity %#v", run)
	}
	if run.Attempts[1].RetryReason == "" {
		t.Fatalf("retry reason was not recorded: %#v", run.Attempts)
	}
}

func TestPairedEvaluationRejectsBudgetBeforeProviderCall(t *testing.T) {
	fixture := loadClassificationFixture(t)
	raw, _ := ReplayPlannerResponse(fixture)
	generator := &pairedFakeGenerator{responses: [][]byte{raw}}
	runtime := llm.Config{Provider: llm.ProviderLocal, Model: "planner-test-model", MaxToolRounds: 2}
	_, err := RunPlannerPairedEvaluation(context.Background(), generator, runtime.Model, runtime, fixture, PlannerPairedEvalConfig{
		Variants:    DefaultPlannerEvalVariants()[:1],
		Repeats:     1,
		MaxAttempts: 1,
		Budget:      PlannerEvalBudget{MaxProviderCalls: 2, MaxRequestBytes: 500_000},
	})
	if err == nil {
		t.Fatal("expected provider-call reservation budget rejection")
	}
	if len(generator.requests) != 0 {
		t.Fatalf("provider was called before budget rejection: %d calls", len(generator.requests))
	}
}

func agentsForBuildOnly() agents.ExperimentPlannerAgent {
	return agents.NewExperimentPlannerAgent(nil, "planner-test-model")
}
