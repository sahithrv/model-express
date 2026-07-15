package llm

import (
	"fmt"
	"math/big"
	"os"
	"strings"
)

const (
	pricingVersionEnv                   = "MODEL_EXPRESS_LLM_PRICING_VERSION"
	pricingProviderEnv                  = "MODEL_EXPRESS_LLM_PRICING_PROVIDER"
	pricingModelEnv                     = "MODEL_EXPRESS_LLM_PRICING_MODEL"
	inputPricePerMillionTokensEnv       = "MODEL_EXPRESS_LLM_INPUT_USD_PER_MILLION_TOKENS"
	cachedInputPricePerMillionTokensEnv = "MODEL_EXPRESS_LLM_CACHED_INPUT_USD_PER_MILLION_TOKENS"
	outputPricePerMillionTokensEnv      = "MODEL_EXPRESS_LLM_OUTPUT_USD_PER_MILLION_TOKENS"
)

// PricingSnapshot is an explicit, versioned set of token prices. Prices are
// decimal strings so cost derivation never depends on floating-point rounding.
type PricingSnapshot struct {
	PricingVersion                 string `json:"pricing_version"`
	Provider                       string `json:"provider"`
	Model                          string `json:"model"`
	InputUSDPerMillionTokens       string `json:"input_usd_per_million_tokens"`
	CachedInputUSDPerMillionTokens string `json:"cached_input_usd_per_million_tokens"`
	OutputUSDPerMillionTokens      string `json:"output_usd_per_million_tokens"`
}

// DerivedCost is an exact cost breakdown derived from provider usage and a
// validated PricingSnapshot. Reasoning tokens are already included in output
// tokens by supported providers and therefore do not have a separate charge.
type DerivedCost struct {
	PricingVersion       string `json:"pricing_version"`
	UncachedInputCostUSD string `json:"uncached_input_cost_usd"`
	CachedInputCostUSD   string `json:"cached_input_cost_usd"`
	OutputCostUSD        string `json:"output_cost_usd"`
	TotalCostUSD         string `json:"total_cost_usd"`
}

// PricingSnapshotFromEnv loads optional pricing configuration. Pricing is
// disabled when no version is configured, even if stale rate variables exist.
// Once a version is present, its provider/model binding and every rate are
// required; rates must be non-negative base-10 decimals.
func PricingSnapshotFromEnv() (*PricingSnapshot, error) {
	version := strings.TrimSpace(os.Getenv(pricingVersionEnv))
	if version == "" {
		return nil, nil
	}

	snapshot := &PricingSnapshot{
		PricingVersion:                 version,
		Provider:                       strings.ToLower(strings.TrimSpace(os.Getenv(pricingProviderEnv))),
		Model:                          strings.TrimSpace(os.Getenv(pricingModelEnv)),
		InputUSDPerMillionTokens:       strings.TrimSpace(os.Getenv(inputPricePerMillionTokensEnv)),
		CachedInputUSDPerMillionTokens: strings.TrimSpace(os.Getenv(cachedInputPricePerMillionTokensEnv)),
		OutputUSDPerMillionTokens:      strings.TrimSpace(os.Getenv(outputPricePerMillionTokensEnv)),
	}
	if err := snapshot.Validate(); err != nil {
		return nil, fmt.Errorf("load LLM pricing snapshot %q: %w", version, err)
	}
	return snapshot, nil
}

// Validate checks that a snapshot is complete and safe to use for cost
// derivation.
func (snapshot PricingSnapshot) Validate() error {
	if strings.TrimSpace(snapshot.PricingVersion) == "" {
		return fmt.Errorf("pricing_version is required")
	}
	if strings.TrimSpace(snapshot.Provider) == "" {
		return fmt.Errorf("provider is required")
	}
	if strings.TrimSpace(snapshot.Model) == "" {
		return fmt.Errorf("model is required")
	}
	prices := []struct {
		name  string
		value string
	}{
		{name: "input_usd_per_million_tokens", value: snapshot.InputUSDPerMillionTokens},
		{name: "cached_input_usd_per_million_tokens", value: snapshot.CachedInputUSDPerMillionTokens},
		{name: "output_usd_per_million_tokens", value: snapshot.OutputUSDPerMillionTokens},
	}
	for _, price := range prices {
		if strings.TrimSpace(price.value) == "" {
			return fmt.Errorf("%s is required when pricing_version is set", price.name)
		}
		if _, err := parseDecimal(price.value); err != nil {
			return fmt.Errorf("%s: %w", price.name, err)
		}
	}
	return nil
}

// MatchesRuntime prevents a versioned rate card from being applied to a
// different provider or model after runtime configuration changes.
func (snapshot PricingSnapshot) MatchesRuntime(provider string, model string) bool {
	return strings.EqualFold(strings.TrimSpace(snapshot.Provider), strings.TrimSpace(provider)) &&
		strings.TrimSpace(snapshot.Model) == strings.TrimSpace(model)
}

// DeriveCost returns no cost when pricing is not configured. When pricing is
// configured, both the snapshot and usage must be valid. Cached input is billed
// separately from uncached input, and output already includes reasoning tokens.
func DeriveCost(usage *Usage, snapshot *PricingSnapshot) (*DerivedCost, error) {
	if snapshot == nil || usage == nil {
		return nil, nil
	}
	if err := snapshot.Validate(); err != nil {
		return nil, fmt.Errorf("derive LLM cost: invalid pricing snapshot: %w", err)
	}
	if usage.InputTokens < 0 {
		return nil, fmt.Errorf("derive LLM cost: input_tokens must not be negative")
	}
	if usage.CachedInputTokens < 0 {
		return nil, fmt.Errorf("derive LLM cost: cached_input_tokens must not be negative")
	}
	if usage.OutputTokens < 0 {
		return nil, fmt.Errorf("derive LLM cost: output_tokens must not be negative")
	}
	if usage.CachedInputTokens > usage.InputTokens {
		return nil, fmt.Errorf("derive LLM cost: cached_input_tokens (%d) exceed input_tokens (%d)", usage.CachedInputTokens, usage.InputTokens)
	}

	inputRate, _ := parseDecimal(snapshot.InputUSDPerMillionTokens)
	cachedInputRate, _ := parseDecimal(snapshot.CachedInputUSDPerMillionTokens)
	outputRate, _ := parseDecimal(snapshot.OutputUSDPerMillionTokens)

	uncachedInputCost := inputRate.costForTokens(usage.InputTokens - usage.CachedInputTokens)
	cachedInputCost := cachedInputRate.costForTokens(usage.CachedInputTokens)
	outputCost := outputRate.costForTokens(usage.OutputTokens)
	totalCost := uncachedInputCost.add(cachedInputCost).add(outputCost)

	return &DerivedCost{
		PricingVersion:       strings.TrimSpace(snapshot.PricingVersion),
		UncachedInputCostUSD: uncachedInputCost.String(),
		CachedInputCostUSD:   cachedInputCost.String(),
		OutputCostUSD:        outputCost.String(),
		TotalCostUSD:         totalCost.String(),
	}, nil
}

// decimal stores an exact non-negative base-10 value as units * 10^-scale.
type decimal struct {
	units *big.Int
	scale int
}

func parseDecimal(value string) (decimal, error) {
	value = strings.TrimSpace(value)
	parts := strings.Split(value, ".")
	if len(parts) > 2 || parts[0] == "" || (len(parts) == 2 && parts[1] == "") {
		return decimal{}, fmt.Errorf("must be a non-negative base-10 decimal")
	}
	for _, part := range parts {
		for _, char := range part {
			if char < '0' || char > '9' {
				return decimal{}, fmt.Errorf("must be a non-negative base-10 decimal")
			}
		}
	}

	digits := strings.Join(parts, "")
	units, ok := new(big.Int).SetString(digits, 10)
	if !ok {
		return decimal{}, fmt.Errorf("must be a non-negative base-10 decimal")
	}
	scale := 0
	if len(parts) == 2 {
		scale = len(parts[1])
	}
	return decimal{units: units, scale: scale}, nil
}

func (value decimal) costForTokens(tokens int) decimal {
	units := new(big.Int).Mul(value.units, big.NewInt(int64(tokens)))
	return decimal{units: units, scale: value.scale + 6}
}

func (value decimal) add(other decimal) decimal {
	if value.scale == other.scale {
		return decimal{units: new(big.Int).Add(value.units, other.units), scale: value.scale}
	}
	if value.scale < other.scale {
		units := scaledUnits(value.units, other.scale-value.scale)
		return decimal{units: units.Add(units, other.units), scale: other.scale}
	}
	units := scaledUnits(other.units, value.scale-other.scale)
	return decimal{units: new(big.Int).Add(value.units, units), scale: value.scale}
}

func scaledUnits(units *big.Int, places int) *big.Int {
	multiplier := new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(places)), nil)
	return new(big.Int).Mul(units, multiplier)
}

func (value decimal) String() string {
	if value.units.Sign() == 0 {
		return "0"
	}
	digits := value.units.String()
	if value.scale == 0 {
		return digits
	}
	if len(digits) <= value.scale {
		digits = strings.Repeat("0", value.scale-len(digits)+1) + digits
	}
	wholeEnd := len(digits) - value.scale
	formatted := digits[:wholeEnd] + "." + digits[wholeEnd:]
	formatted = strings.TrimRight(formatted, "0")
	return strings.TrimSuffix(formatted, ".")
}
