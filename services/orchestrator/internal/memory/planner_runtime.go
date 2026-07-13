package memory

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

const (
	PlannerVariantIdentitySchemaV1 = "planner_variant_identity_v1"
	LegacyPlannerVariantID         = "legacy_unknown"
)

// PlannerRetrievalVariant captures the allowlisted retrieval configuration
// that can change the context presented to the planner. Credentials, URLs,
// query text, and retrieved records are deliberately not part of variant
// identity.
type PlannerRetrievalVariant struct {
	Enabled                      bool    `json:"enabled"`
	LogOnly                      bool    `json:"log_only"`
	CrossProject                 bool    `json:"cross_project"`
	MaxCards                     int     `json:"max_cards"`
	MinScore                     float64 `json:"min_score"`
	MinIndexedCards              int     `json:"min_indexed_cards"`
	QueryCacheTTLSeconds         int64   `json:"query_cache_ttl_seconds"`
	LogOnlyEmbeddings            bool    `json:"log_only_embeddings"`
	EmbeddingsEnabled            bool    `json:"embeddings_enabled"`
	EmbeddingProvider            string  `json:"embedding_provider"`
	EmbeddingEndpointFingerprint string  `json:"embedding_endpoint_fingerprint"`
	EmbeddingModel               string  `json:"embedding_model"`
	EmbeddingDimensions          int     `json:"embedding_dimensions"`
	EmbeddingMaxCallsPerDay      int     `json:"embedding_max_calls_per_day"`
}

// PlannerVariant is the canonical, allowlisted description of planner code,
// policy, and generation settings. Invocation facts such as usage, latency,
// retries, and tool rounds belong on AgentInvocation instead.
type PlannerVariant struct {
	IdentitySchemaVersion        string                  `json:"identity_schema_version"`
	AgentVersion                 string                  `json:"agent_version"`
	PromptVersion                string                  `json:"prompt_version"`
	StaticPromptVersion          string                  `json:"static_prompt_version"`
	ContextBuilderVersion        string                  `json:"context_builder_version"`
	ToolPolicyVersion            string                  `json:"tool_policy_version"`
	ValidatorVersion             string                  `json:"validator_version"`
	ValidationMode               string                  `json:"validation_mode"`
	ExecutionValidatorVersion    string                  `json:"execution_validator_version"`
	ExecutionValidationMode      string                  `json:"execution_validation_mode"`
	RankerVersion                string                  `json:"ranker_version"`
	RankerMultiFidelityEnabled   bool                    `json:"ranker_multi_fidelity_enabled"`
	RetrievalPolicyVersion       string                  `json:"retrieval_policy_version"`
	Retrieval                    PlannerRetrievalVariant `json:"retrieval"`
	RetryPolicyVersion           string                  `json:"retry_policy_version"`
	MaxBackendValidationRetries  int                     `json:"max_backend_validation_retries"`
	DecisionPolicyVersion        string                  `json:"decision_policy_version"`
	AgentMode                    string                  `json:"agent_mode"`
	TerminalPlannerGuards        bool                    `json:"terminal_planner_guards"`
	MinimumMeaningfulImprovement float64                 `json:"minimum_meaningful_improvement"`
	MaxFollowUpRounds            int                     `json:"max_followup_rounds"`
	Provider                     string                  `json:"provider"`
	APIStyle                     string                  `json:"api_style"`
	EffectiveAPIStyle            string                  `json:"effective_api_style"`
	Model                        string                  `json:"model"`
	EndpointFingerprint          string                  `json:"endpoint_fingerprint"`
	RequestTemperature           float64                 `json:"request_temperature"`
	TemperatureSent              bool                    `json:"temperature_sent"`
	RequestReasoningEffort       string                  `json:"request_reasoning_effort"`
	ReasoningEffortSent          bool                    `json:"reasoning_effort_sent"`
	ConfiguredReasoningEffort    string                  `json:"configured_reasoning_effort"`
	PlateauReasoningEffort       string                  `json:"plateau_reasoning_effort"`
	StoredResponses              bool                    `json:"stored_responses"`
	MaxToolRounds                int                     `json:"max_tool_rounds"`
	MaxSelectedExperiments       int                     `json:"max_selected_experiments"`
	MaxProviderRetries           int                     `json:"max_provider_retries"`
	RequestTimeoutMS             int64                   `json:"request_timeout_ms"`
}

// PlannerInvocationCost is present only when provider usage was priced with a
// named immutable snapshot. Usage without a pricing snapshot is not a cost.
type PlannerInvocationCost struct {
	PricingVersion                 string `json:"pricing_version"`
	Currency                       string `json:"currency"`
	Provider                       string `json:"provider"`
	Model                          string `json:"model"`
	InputTokens                    int    `json:"input_tokens"`
	CachedInputTokens              int    `json:"cached_input_tokens"`
	OutputTokens                   int    `json:"output_tokens"`
	InputUSDPerMillionTokens       string `json:"input_usd_per_million_tokens"`
	CachedInputUSDPerMillionTokens string `json:"cached_input_usd_per_million_tokens"`
	OutputUSDPerMillionTokens      string `json:"output_usd_per_million_tokens"`
	UncachedInputCostUSD           string `json:"uncached_input_cost_usd"`
	CachedInputCostUSD             string `json:"cached_input_cost_usd"`
	OutputCostUSD                  string `json:"output_cost_usd"`
	TotalCostUSD                   string `json:"total_cost_usd"`
}

// CanonicalPlannerVariantJSON returns the exact bytes hashed by
// ComputePlannerVariantID. Adding or reordering canonical fields requires a
// new identity schema version so stored IDs never silently change meaning.
func CanonicalPlannerVariantJSON(variant PlannerVariant) ([]byte, error) {
	normalized, err := normalizePlannerVariant(variant)
	if err != nil {
		return nil, err
	}
	return json.Marshal(normalized)
}

func ComputePlannerVariantID(variant PlannerVariant) (string, error) {
	canonical, err := CanonicalPlannerVariantJSON(variant)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(canonical)
	return "planner_variant_v1_" + hex.EncodeToString(digest[:]), nil
}

func normalizePlannerVariant(variant PlannerVariant) (PlannerVariant, error) {
	if strings.TrimSpace(variant.IdentitySchemaVersion) == "" {
		variant.IdentitySchemaVersion = PlannerVariantIdentitySchemaV1
	}
	if variant.IdentitySchemaVersion != PlannerVariantIdentitySchemaV1 {
		return PlannerVariant{}, fmt.Errorf("unsupported planner variant identity schema %q", variant.IdentitySchemaVersion)
	}
	if math.IsNaN(variant.RequestTemperature) || math.IsInf(variant.RequestTemperature, 0) {
		return PlannerVariant{}, fmt.Errorf("planner request temperature must be finite")
	}
	if math.IsNaN(variant.Retrieval.MinScore) || math.IsInf(variant.Retrieval.MinScore, 0) {
		return PlannerVariant{}, fmt.Errorf("planner retrieval minimum score must be finite")
	}
	if math.IsNaN(variant.MinimumMeaningfulImprovement) || math.IsInf(variant.MinimumMeaningfulImprovement, 0) || variant.MinimumMeaningfulImprovement < 0 {
		return PlannerVariant{}, fmt.Errorf("planner minimum meaningful improvement must be finite and non-negative")
	}
	if variant.MaxBackendValidationRetries < 0 || variant.MaxFollowUpRounds < 0 || variant.MaxToolRounds < 0 || variant.MaxSelectedExperiments < 0 || variant.MaxProviderRetries < 0 || variant.RequestTimeoutMS < 0 {
		return PlannerVariant{}, fmt.Errorf("planner runtime budgets must not be negative")
	}
	if variant.Retrieval.MaxCards < 0 || variant.Retrieval.MinIndexedCards < 0 || variant.Retrieval.QueryCacheTTLSeconds < 0 || variant.Retrieval.EmbeddingDimensions < 0 || variant.Retrieval.EmbeddingMaxCallsPerDay < 0 {
		return PlannerVariant{}, fmt.Errorf("planner retrieval budgets must not be negative")
	}

	variant.AgentVersion = strings.TrimSpace(variant.AgentVersion)
	variant.PromptVersion = strings.TrimSpace(variant.PromptVersion)
	variant.StaticPromptVersion = strings.TrimSpace(variant.StaticPromptVersion)
	variant.ContextBuilderVersion = strings.TrimSpace(variant.ContextBuilderVersion)
	variant.ToolPolicyVersion = strings.TrimSpace(variant.ToolPolicyVersion)
	variant.ValidatorVersion = strings.TrimSpace(variant.ValidatorVersion)
	variant.ValidationMode = strings.ToLower(strings.TrimSpace(variant.ValidationMode))
	variant.ExecutionValidatorVersion = strings.TrimSpace(variant.ExecutionValidatorVersion)
	variant.ExecutionValidationMode = strings.ToLower(strings.TrimSpace(variant.ExecutionValidationMode))
	variant.RankerVersion = strings.TrimSpace(variant.RankerVersion)
	variant.RetrievalPolicyVersion = strings.TrimSpace(variant.RetrievalPolicyVersion)
	variant.RetryPolicyVersion = strings.TrimSpace(variant.RetryPolicyVersion)
	variant.DecisionPolicyVersion = strings.TrimSpace(variant.DecisionPolicyVersion)
	variant.AgentMode = strings.ToLower(strings.TrimSpace(variant.AgentMode))
	variant.Provider = strings.ToLower(strings.TrimSpace(variant.Provider))
	variant.APIStyle = strings.ToLower(strings.TrimSpace(variant.APIStyle))
	variant.EffectiveAPIStyle = strings.ToLower(strings.TrimSpace(variant.EffectiveAPIStyle))
	variant.Model = strings.TrimSpace(variant.Model)
	variant.EndpointFingerprint = strings.ToLower(strings.TrimSpace(variant.EndpointFingerprint))
	variant.RequestReasoningEffort = strings.ToLower(strings.TrimSpace(variant.RequestReasoningEffort))
	variant.ConfiguredReasoningEffort = strings.ToLower(strings.TrimSpace(variant.ConfiguredReasoningEffort))
	variant.PlateauReasoningEffort = strings.ToLower(strings.TrimSpace(variant.PlateauReasoningEffort))
	variant.Retrieval.EmbeddingProvider = strings.ToLower(strings.TrimSpace(variant.Retrieval.EmbeddingProvider))
	variant.Retrieval.EmbeddingEndpointFingerprint = strings.ToLower(strings.TrimSpace(variant.Retrieval.EmbeddingEndpointFingerprint))
	variant.Retrieval.EmbeddingModel = strings.TrimSpace(variant.Retrieval.EmbeddingModel)
	return variant, nil
}

func ValidatePlannerInvocationCost(cost *PlannerInvocationCost) error {
	if cost == nil {
		return nil
	}
	if strings.TrimSpace(cost.PricingVersion) == "" {
		return fmt.Errorf("planner invocation cost requires pricing_version")
	}
	if cost.InputTokens < 0 || cost.CachedInputTokens < 0 || cost.OutputTokens < 0 || cost.CachedInputTokens > cost.InputTokens {
		return fmt.Errorf("planner invocation cost token counts are invalid")
	}
	for name, value := range map[string]string{
		"input_usd_per_million_tokens":        cost.InputUSDPerMillionTokens,
		"cached_input_usd_per_million_tokens": cost.CachedInputUSDPerMillionTokens,
		"output_usd_per_million_tokens":       cost.OutputUSDPerMillionTokens,
		"uncached_input_cost_usd":             cost.UncachedInputCostUSD,
		"cached_input_cost_usd":               cost.CachedInputCostUSD,
		"output_cost_usd":                     cost.OutputCostUSD,
		"total_cost_usd":                      cost.TotalCostUSD,
	} {
		if !nonnegativeDecimalString(value) {
			return fmt.Errorf("planner invocation cost %s must be a non-negative decimal", name)
		}
	}
	return nil
}

func nonnegativeDecimalString(value string) bool {
	value = strings.TrimSpace(value)
	if value == "" {
		return false
	}
	dotSeen := false
	digitSeen := false
	for _, char := range value {
		switch {
		case char >= '0' && char <= '9':
			digitSeen = true
		case char == '.' && !dotSeen:
			dotSeen = true
		default:
			return false
		}
	}
	return digitSeen && !strings.HasPrefix(value, ".") && !strings.HasSuffix(value, ".")
}

// NormalizeAgentInvocationRuntime applies legacy defaults and validates the
// coupling between a canonical component snapshot and its stored ID.
func NormalizeAgentInvocationRuntime(invocation AgentInvocation) (AgentInvocation, error) {
	if invocation.PlannerVariant != nil {
		normalized, err := normalizePlannerVariant(*invocation.PlannerVariant)
		if err != nil {
			return AgentInvocation{}, err
		}
		variantID, err := ComputePlannerVariantID(normalized)
		if err != nil {
			return AgentInvocation{}, err
		}
		if strings.TrimSpace(invocation.PlannerVariantID) != "" && invocation.PlannerVariantID != LegacyPlannerVariantID && invocation.PlannerVariantID != variantID {
			return AgentInvocation{}, fmt.Errorf("planner_variant_id does not match planner_variant")
		}
		invocation.PlannerVariant = &normalized
		invocation.PlannerVariantID = variantID
	} else {
		invocation.PlannerVariantID = strings.TrimSpace(invocation.PlannerVariantID)
		if invocation.PlannerVariantID == "" {
			invocation.PlannerVariantID = LegacyPlannerVariantID
		} else if invocation.PlannerVariantID != LegacyPlannerVariantID {
			return AgentInvocation{}, fmt.Errorf("non-legacy planner_variant_id requires planner_variant")
		}
	}

	if strings.TrimSpace(invocation.AttemptGroupID) == "" {
		invocation.AttemptGroupID = ""
		invocation.AttemptIndex = -1
	} else if invocation.AttemptIndex < 0 {
		return AgentInvocation{}, fmt.Errorf("planner attempt_index must be non-negative when attempt_group_id is present")
	}
	if math.IsNaN(invocation.WallLatencyMS) || math.IsInf(invocation.WallLatencyMS, 0) || invocation.WallLatencyMS < 0 {
		return AgentInvocation{}, fmt.Errorf("planner wall latency must be finite and non-negative")
	}
	if invocation.ProviderUsage == nil {
		invocation.ProviderUsage = map[string]any{}
	}
	if err := ValidatePlannerInvocationCost(invocation.DerivedCost); err != nil {
		return AgentInvocation{}, err
	}
	return invocation, nil
}
