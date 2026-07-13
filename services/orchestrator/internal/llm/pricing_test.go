package llm

import (
	"strings"
	"testing"
)

func TestDeriveCostRequiresPricingSnapshot(t *testing.T) {
	cost, err := DeriveCost(&Usage{InputTokens: 100, OutputTokens: 20}, nil)
	if err != nil {
		t.Fatalf("derive cost without pricing: %v", err)
	}
	if cost != nil {
		t.Fatalf("expected provider usage alone not to produce cost, got %#v", cost)
	}
}

func TestDeriveCostUsesExactArithmeticAndSeparatesCachedInput(t *testing.T) {
	snapshot := &PricingSnapshot{
		PricingVersion:                 "openai-2026-07-01",
		Provider:                       ProviderOpenAI,
		Model:                          "test-model",
		InputUSDPerMillionTokens:       "0.15",
		CachedInputUSDPerMillionTokens: "0.075",
		OutputUSDPerMillionTokens:      "0.6",
	}
	usage := &Usage{
		InputTokens:       11,
		CachedInputTokens: 3,
		OutputTokens:      7,
		ReasoningTokens:   5,
	}

	cost, err := DeriveCost(usage, snapshot)
	if err != nil {
		t.Fatalf("derive cost: %v", err)
	}
	if cost.PricingVersion != "openai-2026-07-01" {
		t.Fatalf("unexpected pricing version %q", cost.PricingVersion)
	}
	if cost.UncachedInputCostUSD != "0.0000012" {
		t.Fatalf("unexpected uncached input cost %q", cost.UncachedInputCostUSD)
	}
	if cost.CachedInputCostUSD != "0.000000225" {
		t.Fatalf("unexpected cached input cost %q", cost.CachedInputCostUSD)
	}
	if cost.OutputCostUSD != "0.0000042" {
		t.Fatalf("unexpected output cost %q", cost.OutputCostUSD)
	}
	if cost.TotalCostUSD != "0.000005625" {
		t.Fatalf("unexpected total cost %q", cost.TotalCostUSD)
	}
}

func TestDeriveCostDoesNotDoubleCountReasoningTokens(t *testing.T) {
	snapshot := &PricingSnapshot{
		PricingVersion:                 "test-v1",
		Provider:                       ProviderOpenAI,
		Model:                          "test-model",
		InputUSDPerMillionTokens:       "2.5",
		CachedInputUSDPerMillionTokens: "0.25",
		OutputUSDPerMillionTokens:      "10",
	}
	usage := &Usage{
		InputTokens:       1_200_000,
		CachedInputTokens: 200_000,
		OutputTokens:      300_000,
		ReasoningTokens:   100_000,
	}

	cost, err := DeriveCost(usage, snapshot)
	if err != nil {
		t.Fatalf("derive cost: %v", err)
	}
	if cost.UncachedInputCostUSD != "2.5" || cost.CachedInputCostUSD != "0.05" || cost.OutputCostUSD != "3" {
		t.Fatalf("unexpected cost breakdown %#v", cost)
	}
	if cost.TotalCostUSD != "5.55" {
		t.Fatalf("expected reasoning to remain part of output rather than a separate charge, got %q", cost.TotalCostUSD)
	}
}

func TestDeriveCostRejectsInvalidUsage(t *testing.T) {
	snapshot := &PricingSnapshot{
		PricingVersion:                 "test-v1",
		Provider:                       ProviderOpenAI,
		Model:                          "test-model",
		InputUSDPerMillionTokens:       "1",
		CachedInputUSDPerMillionTokens: "0.1",
		OutputUSDPerMillionTokens:      "2",
	}
	tests := []struct {
		name  string
		usage Usage
		want  string
	}{
		{name: "negative input", usage: Usage{InputTokens: -1}, want: "input_tokens must not be negative"},
		{name: "negative cached input", usage: Usage{CachedInputTokens: -1}, want: "cached_input_tokens must not be negative"},
		{name: "negative output", usage: Usage{OutputTokens: -1}, want: "output_tokens must not be negative"},
		{name: "cached exceeds input", usage: Usage{InputTokens: 2, CachedInputTokens: 3}, want: "exceed input_tokens"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cost, err := DeriveCost(&test.usage, snapshot)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected error containing %q, got cost=%#v err=%v", test.want, cost, err)
			}
		})
	}
}

func TestDeriveCostRejectsUnvalidatedSnapshot(t *testing.T) {
	usage := &Usage{InputTokens: 1, OutputTokens: 1}
	tests := []struct {
		name     string
		snapshot PricingSnapshot
		want     string
	}{
		{
			name: "missing version",
			snapshot: PricingSnapshot{
				InputUSDPerMillionTokens:       "1",
				CachedInputUSDPerMillionTokens: "0.1",
				OutputUSDPerMillionTokens:      "2",
			},
			want: "pricing_version is required",
		},
		{
			name: "negative rate",
			snapshot: PricingSnapshot{
				PricingVersion:                 "test-v1",
				Provider:                       ProviderOpenAI,
				Model:                          "test-model",
				InputUSDPerMillionTokens:       "-1",
				CachedInputUSDPerMillionTokens: "0.1",
				OutputUSDPerMillionTokens:      "2",
			},
			want: "non-negative base-10 decimal",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cost, err := DeriveCost(usage, &test.snapshot)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected error containing %q, got cost=%#v err=%v", test.want, cost, err)
			}
		})
	}
}

func TestPricingSnapshotFromEnvReturnsNoSnapshotWithoutVersion(t *testing.T) {
	t.Setenv(pricingVersionEnv, "")
	t.Setenv(inputPricePerMillionTokensEnv, "1")
	t.Setenv(cachedInputPricePerMillionTokensEnv, "0.1")
	t.Setenv(outputPricePerMillionTokensEnv, "2")

	snapshot, err := PricingSnapshotFromEnv()
	if err != nil {
		t.Fatalf("load optional pricing: %v", err)
	}
	if snapshot != nil {
		t.Fatalf("expected no pricing without a version, got %#v", snapshot)
	}
}

func TestPricingSnapshotFromEnvLoadsCompleteVersionedConfig(t *testing.T) {
	t.Setenv(pricingVersionEnv, " openai-2026-07-01 ")
	t.Setenv(pricingProviderEnv, " openai ")
	t.Setenv(pricingModelEnv, " test-model ")
	t.Setenv(inputPricePerMillionTokensEnv, " 0.15 ")
	t.Setenv(cachedInputPricePerMillionTokensEnv, " 0.075 ")
	t.Setenv(outputPricePerMillionTokensEnv, " 0.6 ")

	snapshot, err := PricingSnapshotFromEnv()
	if err != nil {
		t.Fatalf("load pricing: %v", err)
	}
	if snapshot == nil {
		t.Fatal("expected pricing snapshot")
	}
	if snapshot.PricingVersion != "openai-2026-07-01" || snapshot.Provider != ProviderOpenAI || snapshot.Model != "test-model" ||
		snapshot.InputUSDPerMillionTokens != "0.15" ||
		snapshot.CachedInputUSDPerMillionTokens != "0.075" ||
		snapshot.OutputUSDPerMillionTokens != "0.6" {
		t.Fatalf("unexpected pricing snapshot %#v", snapshot)
	}
}

func TestPricingSnapshotFromEnvRejectsIncompleteOrInvalidVersionedConfig(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		cached string
		output string
		want   string
	}{
		{name: "missing input", cached: "0.1", output: "2", want: "input_usd_per_million_tokens is required"},
		{name: "missing cached input", input: "1", output: "2", want: "cached_input_usd_per_million_tokens is required"},
		{name: "missing output", input: "1", cached: "0.1", want: "output_usd_per_million_tokens is required"},
		{name: "invalid input", input: "one", cached: "0.1", output: "2", want: "non-negative base-10 decimal"},
		{name: "negative cached input", input: "1", cached: "-0.1", output: "2", want: "non-negative base-10 decimal"},
		{name: "exponential output", input: "1", cached: "0.1", output: "2e0", want: "non-negative base-10 decimal"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv(pricingVersionEnv, "test-v1")
			t.Setenv(pricingProviderEnv, ProviderOpenAI)
			t.Setenv(pricingModelEnv, "test-model")
			t.Setenv(inputPricePerMillionTokensEnv, test.input)
			t.Setenv(cachedInputPricePerMillionTokensEnv, test.cached)
			t.Setenv(outputPricePerMillionTokensEnv, test.output)

			snapshot, err := PricingSnapshotFromEnv()
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected error containing %q, got snapshot=%#v err=%v", test.want, snapshot, err)
			}
		})
	}
}

func TestPricingSnapshotMatchesOnlyItsBoundRuntime(t *testing.T) {
	snapshot := PricingSnapshot{Provider: ProviderOpenAI, Model: "planner-model"}
	if !snapshot.MatchesRuntime(" OPENAI ", "planner-model") {
		t.Fatal("expected normalized provider and exact model to match")
	}
	if snapshot.MatchesRuntime(ProviderLocal, "planner-model") || snapshot.MatchesRuntime(ProviderOpenAI, "other-model") {
		t.Fatal("pricing snapshot matched a different provider or model")
	}
}
