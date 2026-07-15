package memory

import (
	"strings"
	"testing"
)

func representativePlannerVariant() PlannerVariant {
	return PlannerVariant{
		IdentitySchemaVersion:      PlannerVariantIdentitySchemaV1,
		AgentVersion:               "v2",
		PromptVersion:              "experiment_planner_v3",
		StaticPromptVersion:        "compact_v1",
		ContextBuilderVersion:      "planner_context_builder_v2",
		ToolPolicyVersion:          "planner_information_tools_v1",
		ValidatorVersion:           "experiment_planner_validator_v1",
		ValidationMode:             "relaxed",
		ExecutionValidatorVersion:  "execution_validation_v1",
		ExecutionValidationMode:    "enforce",
		RankerVersion:              "candidate_ranker_v1",
		RankerMultiFidelityEnabled: true,
		RetrievalPolicyVersion:     "planner_memory_retrieval_v1",
		Retrieval: PlannerRetrievalVariant{
			Enabled:                      true,
			LogOnly:                      false,
			CrossProject:                 false,
			MaxCards:                     10,
			MinScore:                     0.55,
			MinIndexedCards:              3,
			QueryCacheTTLSeconds:         3600,
			LogOnlyEmbeddings:            false,
			EmbeddingsEnabled:            true,
			EmbeddingProvider:            "openai",
			EmbeddingEndpointFingerprint: "sha256:embedding-endpoint",
			EmbeddingModel:               "text-embedding-test",
			EmbeddingDimensions:          1536,
			EmbeddingMaxCallsPerDay:      100,
		},
		RetryPolicyVersion:           "planner_backend_validation_retry_v1",
		MaxBackendValidationRetries:  1,
		DecisionPolicyVersion:        "planner_decision_postprocessor_v1",
		AgentMode:                    "autonomous",
		TerminalPlannerGuards:        true,
		MinimumMeaningfulImprovement: 0.01,
		MaxFollowUpRounds:            3,
		Provider:                     "openai",
		APIStyle:                     "responses",
		EffectiveAPIStyle:            "responses",
		Model:                        "planner-test-model",
		EndpointFingerprint:          "sha256:planner-endpoint",
		RequestTemperature:           0.35,
		TemperatureSent:              false,
		RequestReasoningEffort:       "high",
		ReasoningEffortSent:          true,
		ConfiguredReasoningEffort:    "medium",
		PlateauReasoningEffort:       "high",
		StoredResponses:              true,
		MaxToolRounds:                10,
		MaxSelectedExperiments:       5,
		MaxProviderRetries:           2,
		RequestTimeoutMS:             180000,
	}
}

func TestPlannerVariantIDStableGolden(t *testing.T) {
	variant := representativePlannerVariant()
	canonical, err := CanonicalPlannerVariantJSON(variant)
	if err != nil {
		t.Fatalf("canonical planner variant: %v", err)
	}
	wantCanonical := `{"identity_schema_version":"planner_variant_identity_v1","agent_version":"v2","prompt_version":"experiment_planner_v3","static_prompt_version":"compact_v1","context_builder_version":"planner_context_builder_v2","tool_policy_version":"planner_information_tools_v1","validator_version":"experiment_planner_validator_v1","validation_mode":"relaxed","execution_validator_version":"execution_validation_v1","execution_validation_mode":"enforce","ranker_version":"candidate_ranker_v1","ranker_multi_fidelity_enabled":true,"retrieval_policy_version":"planner_memory_retrieval_v1","retrieval":{"enabled":true,"log_only":false,"cross_project":false,"max_cards":10,"min_score":0.55,"min_indexed_cards":3,"query_cache_ttl_seconds":3600,"log_only_embeddings":false,"embeddings_enabled":true,"embedding_provider":"openai","embedding_endpoint_fingerprint":"sha256:embedding-endpoint","embedding_model":"text-embedding-test","embedding_dimensions":1536,"embedding_max_calls_per_day":100},"retry_policy_version":"planner_backend_validation_retry_v1","max_backend_validation_retries":1,"decision_policy_version":"planner_decision_postprocessor_v1","agent_mode":"autonomous","terminal_planner_guards":true,"minimum_meaningful_improvement":0.01,"max_followup_rounds":3,"provider":"openai","api_style":"responses","effective_api_style":"responses","model":"planner-test-model","endpoint_fingerprint":"sha256:planner-endpoint","request_temperature":0.35,"temperature_sent":false,"request_reasoning_effort":"high","reasoning_effort_sent":true,"configured_reasoning_effort":"medium","plateau_reasoning_effort":"high","stored_responses":true,"max_tool_rounds":10,"max_selected_experiments":5,"max_provider_retries":2,"request_timeout_ms":180000}`
	if string(canonical) != wantCanonical {
		t.Fatalf("canonical variant changed without an identity schema bump\n got: %s\nwant: %s", canonical, wantCanonical)
	}
	id, err := ComputePlannerVariantID(variant)
	if err != nil {
		t.Fatalf("compute planner variant ID: %v", err)
	}
	wantID := "planner_variant_v1_339bdce63181054b9d3f2f1ce12364fd75ef527708aac770d9ad02e50330beb4"
	if id != wantID {
		t.Fatalf("planner variant golden ID = %q, want %q", id, wantID)
	}
}

func TestPlannerVariantIDEqualConfigurationsMatch(t *testing.T) {
	left := representativePlannerVariant()
	right := representativePlannerVariant()
	right.Provider = " OPENAI "
	right.APIStyle = " Responses "
	right.RequestReasoningEffort = " HIGH "

	leftID, err := ComputePlannerVariantID(left)
	if err != nil {
		t.Fatalf("compute left ID: %v", err)
	}
	rightID, err := ComputePlannerVariantID(right)
	if err != nil {
		t.Fatalf("compute right ID: %v", err)
	}
	if leftID != rightID {
		t.Fatalf("equivalent normalized configurations produced %q and %q", leftID, rightID)
	}
}

func TestPlannerVariantIDChangesForMeaningfulConfiguration(t *testing.T) {
	base := representativePlannerVariant()
	baseID, err := ComputePlannerVariantID(base)
	if err != nil {
		t.Fatalf("compute base ID: %v", err)
	}
	tests := []struct {
		name   string
		mutate func(*PlannerVariant)
	}{
		{"agent version", func(v *PlannerVariant) { v.AgentVersion = "v3" }},
		{"prompt version", func(v *PlannerVariant) { v.PromptVersion = "experiment_planner_v4" }},
		{"static prompt", func(v *PlannerVariant) { v.StaticPromptVersion = "v1" }},
		{"context builder", func(v *PlannerVariant) { v.ContextBuilderVersion = "planner_context_builder_v3" }},
		{"tool policy", func(v *PlannerVariant) { v.ToolPolicyVersion = "planner_information_tools_v2" }},
		{"validator", func(v *PlannerVariant) { v.ValidatorVersion = "experiment_planner_validator_v2" }},
		{"validation mode", func(v *PlannerVariant) { v.ValidationMode = "strict" }},
		{"execution validation mode", func(v *PlannerVariant) { v.ExecutionValidationMode = "shadow" }},
		{"ranker", func(v *PlannerVariant) { v.RankerVersion = "candidate_ranker_v2" }},
		{"ranker setting", func(v *PlannerVariant) { v.RankerMultiFidelityEnabled = false }},
		{"retrieval policy", func(v *PlannerVariant) { v.RetrievalPolicyVersion = "planner_memory_retrieval_v2" }},
		{"retrieval enabled", func(v *PlannerVariant) { v.Retrieval.Enabled = false }},
		{"retrieval mode", func(v *PlannerVariant) { v.Retrieval.LogOnly = true }},
		{"retrieval cards", func(v *PlannerVariant) { v.Retrieval.MaxCards++ }},
		{"retrieval score", func(v *PlannerVariant) { v.Retrieval.MinScore = 0.65 }},
		{"retrieval cache ttl", func(v *PlannerVariant) { v.Retrieval.QueryCacheTTLSeconds++ }},
		{"embedding endpoint", func(v *PlannerVariant) { v.Retrieval.EmbeddingEndpointFingerprint = "sha256:other" }},
		{"embedding model", func(v *PlannerVariant) { v.Retrieval.EmbeddingModel = "text-embedding-next" }},
		{"embedding daily cap", func(v *PlannerVariant) { v.Retrieval.EmbeddingMaxCallsPerDay++ }},
		{"retry policy", func(v *PlannerVariant) { v.RetryPolicyVersion = "planner_backend_validation_retry_v2" }},
		{"backend retry budget", func(v *PlannerVariant) { v.MaxBackendValidationRetries++ }},
		{"decision policy", func(v *PlannerVariant) { v.DecisionPolicyVersion = "planner_decision_postprocessor_v2" }},
		{"agent mode", func(v *PlannerVariant) { v.AgentMode = "propose" }},
		{"terminal guards", func(v *PlannerVariant) { v.TerminalPlannerGuards = false }},
		{"meaningful improvement", func(v *PlannerVariant) { v.MinimumMeaningfulImprovement = 0.02 }},
		{"follow-up budget", func(v *PlannerVariant) { v.MaxFollowUpRounds++ }},
		{"provider", func(v *PlannerVariant) { v.Provider = "local" }},
		{"api style", func(v *PlannerVariant) { v.APIStyle = "chat_completions" }},
		{"effective api style", func(v *PlannerVariant) { v.EffectiveAPIStyle = "chat_completions" }},
		{"model", func(v *PlannerVariant) { v.Model = "planner-next-model" }},
		{"endpoint", func(v *PlannerVariant) { v.EndpointFingerprint = "sha256:other" }},
		{"temperature", func(v *PlannerVariant) { v.RequestTemperature = 0.2 }},
		{"temperature application", func(v *PlannerVariant) { v.TemperatureSent = true }},
		{"request reasoning", func(v *PlannerVariant) { v.RequestReasoningEffort = "xhigh" }},
		{"reasoning application", func(v *PlannerVariant) { v.ReasoningEffortSent = false }},
		{"configured reasoning", func(v *PlannerVariant) { v.ConfiguredReasoningEffort = "low" }},
		{"plateau reasoning", func(v *PlannerVariant) { v.PlateauReasoningEffort = "xhigh" }},
		{"stored responses", func(v *PlannerVariant) { v.StoredResponses = false }},
		{"tool budget", func(v *PlannerVariant) { v.MaxToolRounds++ }},
		{"selection budget", func(v *PlannerVariant) { v.MaxSelectedExperiments-- }},
		{"provider retry budget", func(v *PlannerVariant) { v.MaxProviderRetries++ }},
		{"request timeout", func(v *PlannerVariant) { v.RequestTimeoutMS++ }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mutated := base
			test.mutate(&mutated)
			id, err := ComputePlannerVariantID(mutated)
			if err != nil {
				t.Fatalf("compute mutated ID: %v", err)
			}
			if id == baseID {
				t.Fatalf("meaningful %s change did not affect planner variant ID", test.name)
			}
		})
	}
}

func TestPlannerVariantIDExcludesInvocationFacts(t *testing.T) {
	variant := representativePlannerVariant()
	variantID, err := ComputePlannerVariantID(variant)
	if err != nil {
		t.Fatalf("compute variant ID: %v", err)
	}
	invocation := AgentInvocation{
		PlannerVariant:   &variant,
		AttemptGroupID:   "group-one",
		AttemptIndex:     3,
		RetryReason:      "validation",
		WallLatencyMS:    12.5,
		ProviderUsage:    map[string]any{"input_tokens": 20, "tool_rounds": 2},
		PlannerVariantID: variantID,
	}
	normalized, err := NormalizeAgentInvocationRuntime(invocation)
	if err != nil {
		t.Fatalf("normalize invocation: %v", err)
	}
	if normalized.PlannerVariantID != variantID {
		t.Fatalf("invocation facts changed variant ID from %q to %q", variantID, normalized.PlannerVariantID)
	}
}

func TestLegacyInvocationAndCostInvariants(t *testing.T) {
	legacy, err := NormalizeAgentInvocationRuntime(AgentInvocation{})
	if err != nil {
		t.Fatalf("normalize legacy invocation: %v", err)
	}
	if legacy.PlannerVariantID != LegacyPlannerVariantID || legacy.AttemptIndex != -1 {
		t.Fatalf("unexpected legacy defaults %#v", legacy)
	}
	if _, err := NormalizeAgentInvocationRuntime(AgentInvocation{PlannerVariantID: "planner_variant_v1_unverifiable"}); err == nil {
		t.Fatal("expected non-legacy ID without canonical components to be rejected")
	}
	if err := ValidatePlannerInvocationCost(&PlannerInvocationCost{TotalCostUSD: "1"}); err == nil || !strings.Contains(err.Error(), "pricing_version") {
		t.Fatalf("expected unversioned cost to be rejected, got %v", err)
	}
}
