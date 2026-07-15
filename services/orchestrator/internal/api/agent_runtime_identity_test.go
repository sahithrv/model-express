package api

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/store"
)

type delayedPlannerGenerator struct {
	delay time.Duration
}

func (generator delayedPlannerGenerator) GenerateJSON(ctx context.Context, _ llm.JSONRequest) ([]byte, error) {
	timer := time.NewTimer(generator.delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-timer.C:
	}
	return []byte(`{"summary":"Wait for more evidence.","decision_type":"WAIT","rationale":"No scheduling action is warranted yet.","confidence":0.61,"risks":[],"expected_tradeoffs":[],"novelty_notes":[],"tags":["wait"]}`), nil
}

type failingAgentInvocationStore struct {
	store.Store
}

func (failingAgentInvocationStore) CreateAgentInvocation(memory.AgentInvocation) (memory.AgentInvocation, error) {
	return memory.AgentInvocation{}, errors.New("injected invocation persistence failure")
}

func TestPlannerDerivedCostRequiresVersionedPricingSnapshot(t *testing.T) {
	config := llm.Config{Provider: llm.ProviderOpenAI, Model: "test-model"}
	usage := &llm.Usage{InputTokens: 11, CachedInputTokens: 3, OutputTokens: 7, ReasoningTokens: 5, RequestModel: "provider-routed-model"}
	t.Setenv("MODEL_EXPRESS_LLM_PRICING_VERSION", "")
	t.Setenv("MODEL_EXPRESS_LLM_INPUT_USD_PER_MILLION_TOKENS", "0.15")
	t.Setenv("MODEL_EXPRESS_LLM_CACHED_INPUT_USD_PER_MILLION_TOKENS", "0.075")
	t.Setenv("MODEL_EXPRESS_LLM_OUTPUT_USD_PER_MILLION_TOKENS", "0.6")
	if cost := plannerDerivedCost(config, usage); cost != nil {
		t.Fatalf("provider usage without pricing_version produced cost %#v", cost)
	}

	t.Setenv("MODEL_EXPRESS_LLM_PRICING_VERSION", "test-model-2026-07-13")
	t.Setenv("MODEL_EXPRESS_LLM_PRICING_PROVIDER", llm.ProviderOpenAI)
	t.Setenv("MODEL_EXPRESS_LLM_PRICING_MODEL", "provider-routed-model")
	cost := plannerDerivedCost(config, usage)
	if cost == nil {
		t.Fatal("expected cost from complete versioned pricing snapshot")
	}
	if cost.PricingVersion != "test-model-2026-07-13" || cost.TotalCostUSD != "0.000005625" || cost.Model != "provider-routed-model" {
		t.Fatalf("unexpected versioned cost %#v", cost)
	}
	if err := memory.ValidatePlannerInvocationCost(cost); err != nil {
		t.Fatalf("persisted planner cost invariant failed: %v", err)
	}

	t.Setenv("MODEL_EXPRESS_LLM_PRICING_MODEL", "different-model")
	if cost := plannerDerivedCost(config, usage); cost != nil {
		t.Fatalf("mismatched model pricing produced cost %#v", cost)
	}
}

func TestExperimentPlannerVariantReflectsEffectiveRequestTransport(t *testing.T) {
	terminalGuards := true
	input := agents.ExperimentPlannerInput{
		AgentMode:                    llm.AgentModeAutonomous,
		MaxExperiments:               3,
		MaxFollowUpRounds:            4,
		MinimumMeaningfulImprovement: 0.01,
		TerminalPlannerGuardsEnabled: &terminalGuards,
		RetrievalVariant: memory.PlannerRetrievalVariant{
			MaxCards: 10,
		},
	}
	trace := agents.ExperimentPlanningTrace{
		AgentVersion:          "planner-v1",
		PromptVersion:         "prompt-v1",
		StaticPromptVersion:   "static-v1",
		ContextBuilderVersion: "context-v1",
		ValidatorMode:         "relaxed",
		Request: llm.JSONRequest{
			Model:           "planner-model",
			Temperature:     0.35,
			ReasoningEffort: llm.ReasoningEffortHigh,
		},
	}
	localConfig := llm.Config{
		Provider:        llm.ProviderLocal,
		BaseURL:         "http://local.test/v1",
		Model:           "planner-model",
		APIStyle:        llm.APIStyleResponses,
		ReasoningEffort: llm.ReasoningEffortHigh,
		MaxToolRounds:   2,
		Timeout:         time.Second,
	}
	localVariant, localID, err := experimentPlannerVariant(input, localConfig, trace)
	if err != nil {
		t.Fatalf("build local variant: %v", err)
	}
	if localVariant.APIStyle != llm.APIStyleResponses || localVariant.EffectiveAPIStyle != llm.APIStyleChatCompletions {
		t.Fatalf("configured/effective API styles were not both captured: %#v", localVariant)
	}
	if !localVariant.TemperatureSent || localVariant.ReasoningEffortSent {
		t.Fatalf("local chat request parameter application is wrong: %#v", localVariant)
	}
	if localVariant.MaxSelectedExperiments != 3 || localVariant.EndpointFingerprint == "" {
		t.Fatalf("selection budget or endpoint fingerprint missing: %#v", localVariant)
	}
	if localVariant.AgentMode != llm.AgentModeAutonomous || !localVariant.TerminalPlannerGuards || localVariant.DecisionPolicyVersion == "" {
		t.Fatalf("post-generation decision policy missing from identity: %#v", localVariant)
	}
	if localVariant.MaxFollowUpRounds != 4 || localVariant.MinimumMeaningfulImprovement != 0.01 {
		t.Fatalf("planner policy thresholds missing from identity: %#v", localVariant)
	}

	openAIConfig := localConfig
	openAIConfig.Provider = llm.ProviderOpenAI
	openAIConfig.BaseURL = "https://api.openai.com/v1"
	openAIVariant, openAIID, err := experimentPlannerVariant(input, openAIConfig, trace)
	if err != nil {
		t.Fatalf("build OpenAI variant: %v", err)
	}
	if openAIVariant.EffectiveAPIStyle != llm.APIStyleResponses || openAIVariant.TemperatureSent || !openAIVariant.ReasoningEffortSent {
		t.Fatalf("OpenAI Responses request parameter application is wrong: %#v", openAIVariant)
	}
	if openAIID == localID {
		t.Fatal("different effective transports produced the same planner variant ID")
	}
	terminalGuards = false
	_, guardsDisabledID, err := experimentPlannerVariant(input, localConfig, trace)
	if err != nil {
		t.Fatalf("build guard-disabled variant: %v", err)
	}
	if guardsDisabledID == localID {
		t.Fatal("different terminal planner guard policies produced the same planner variant ID")
	}
}

func TestMemoryRetrievalExecutesCapturedPlannerPolicy(t *testing.T) {
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_LOG_ONLY", "false")
	t.Setenv("MODEL_EXPRESS_MEMORY_RETRIEVAL_MIN_SCORE", "0.99")
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("Retrieval fixture", "Use captured retrieval policy")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	if _, err := memoryStore.UpsertMemoryEmbedding(memory.MemoryEmbeddingRecord{
		SourceTable:         memory.SourceAgentMemoryRecord,
		SourceID:            "captured_policy_memory",
		ProjectID:           project.ID,
		Kind:                memory.KindPlanningOutcome,
		Scope:               memory.ScopeProject,
		EmbeddingModel:      "fixture-embedding",
		EmbeddingDimensions: 3,
		Embedding:           []float32{1, 0, 0},
		EmbeddingText:       "class balancing improved minority recall",
		Metadata: map[string]any{
			"accepted_for_vector_memory": true,
		},
	}); err != nil {
		t.Fatalf("seed retrieval memory: %v", err)
	}
	server := newServer(memoryStore)
	query := memory.MemoryRetrievalQuery{
		ProjectID: project.ID,
		AgentName: agents.ExperimentPlannerAgentName,
		Purpose:   "experiment_planner",
		Text:      "class balancing improved minority recall",
		Kinds:     []string{memory.KindPlanningOutcome},
		Limit:     5,
	}
	policy := memory.PlannerRetrievalVariant{
		Enabled:             true,
		LogOnly:             true,
		MaxCards:            5,
		MinScore:            0,
		EmbeddingsEnabled:   false,
		EmbeddingDimensions: 1536,
	}
	results, usage := server.searchRetrievedMemory(context.Background(), query, "plan_retrieval", "", policy)
	if len(results) == 0 || !usage.LogOnly || usage.Injected {
		t.Fatalf("retrieval did not execute the captured log-only/min-score policy: results=%#v usage=%#v", results, usage)
	}
	policy.MinScore = 2
	results, _ = server.searchRetrievedMemory(context.Background(), query, "plan_retrieval", "", policy)
	if len(results) != 0 {
		t.Fatalf("captured minimum score was not applied: %#v", results)
	}
}

func TestPlannerRolloutRetrievalVariantChangesOnlyRetrievalPolicy(t *testing.T) {
	policy := calibration.DefaultPlannerRolloutPolicy()
	policy.Enabled = true
	policy.State = calibration.RolloutStateActive
	policy.StagePercent = 100
	policy.Dimensions = []string{calibration.RolloutDimensionRetrieval}
	policy.VariantValues = map[string]string{calibration.RolloutDimensionRetrieval: "log_only"}
	assignment, err := calibration.AssignPlannerRollout(policy, "project-retrieval-only")
	if err != nil {
		t.Fatal(err)
	}
	base := memory.PlannerRetrievalVariant{Enabled: false, LogOnly: false, MaxCards: 7, MinScore: 0.61}
	got, err := plannerRetrievalVariantForRollout(base, &assignment)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Enabled || !got.LogOnly || got.MaxCards != base.MaxCards || got.MinScore != base.MinScore {
		t.Fatalf("retrieval-only rollout changed unrelated retrieval settings: base=%#v got=%#v", base, got)
	}

	assignment.VariantValues[calibration.RolloutDimensionRetrieval] = "unsupported"
	if _, err := plannerRetrievalVariantForRollout(base, &assignment); err == nil {
		t.Fatal("unsupported retrieval rollout variant was accepted")
	}
}

func TestMemoryRetrievalQueryCacheIsPartitionedByEmbeddingRuntime(t *testing.T) {
	base := memory.PlannerRetrievalVariant{
		EmbeddingProvider:            "local",
		EmbeddingEndpointFingerprint: "sha256:endpoint-a",
	}
	left := memoryRetrievalQueryCachePurpose("experiment_planner", base)
	right := memoryRetrievalQueryCachePurpose("experiment_planner", base)
	if left != right {
		t.Fatalf("equal embedding runtimes produced different cache namespaces: %q and %q", left, right)
	}
	base.EmbeddingEndpointFingerprint = "sha256:endpoint-b"
	if other := memoryRetrievalQueryCachePurpose("experiment_planner", base); other == left {
		t.Fatalf("different embedding endpoints shared cache namespace %q", left)
	}
	base.EmbeddingProvider = "openai_compatible"
	if other := memoryRetrievalQueryCachePurpose("experiment_planner", base); other == left {
		t.Fatalf("different embedding providers shared cache namespace %q", left)
	}
}

func TestPlannerInvocationLatencyIsMeasuredLocally(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("Latency fixture", "Measure planner latency")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	server := newServer(memoryStore)
	config := llm.Config{
		Provider:      llm.ProviderLocal,
		BaseURL:       "http://local.test/v1",
		Model:         "planner-test-model",
		APIStyle:      llm.APIStyleChatCompletions,
		Timeout:       time.Second,
		MaxToolRounds: 1,
	}
	agent := agents.NewExperimentPlannerAgentWithRuntime(delayedPlannerGenerator{delay: 25 * time.Millisecond}, config.Model, config, agents.PlannerInformationToolOptions{})
	input := agents.ExperimentPlannerInput{
		Project:        project,
		SourcePlan:     plans.ExperimentPlan{ID: "plan_latency", ProjectID: project.ID},
		MaxExperiments: 2,
		RetrievalVariant: memory.PlannerRetrievalVariant{
			MaxCards: 10,
		},
	}

	result, err := server.runExperimentPlannerWithBackendValidationRetry(context.Background(), agent, input, config, llm.AgentModePropose)
	if err != nil {
		t.Fatalf("run planner: %v", err)
	}
	if result.Invocation.WallLatencyMS < 20 {
		t.Fatalf("locally measured wall latency = %.3fms, want at least 20ms", result.Invocation.WallLatencyMS)
	}
}

func TestPlannerDecisionFailsClosedWhenInvocationCannotPersist(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("Persistence fixture", "Require planner audit")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}
	server := newServer(failingAgentInvocationStore{Store: memoryStore})
	config := llm.Config{
		Provider:      llm.ProviderLocal,
		BaseURL:       "http://local.test/v1",
		Model:         "planner-test-model",
		APIStyle:      llm.APIStyleChatCompletions,
		Timeout:       time.Second,
		MaxToolRounds: 1,
	}
	agent := agents.NewExperimentPlannerAgentWithRuntime(delayedPlannerGenerator{}, config.Model, config, agents.PlannerInformationToolOptions{})
	input := agents.ExperimentPlannerInput{
		Project:        project,
		SourcePlan:     plans.ExperimentPlan{ID: "plan_persistence", ProjectID: project.ID},
		MaxExperiments: 2,
		RetrievalVariant: memory.PlannerRetrievalVariant{
			MaxCards: 10,
		},
	}

	result, err := server.runExperimentPlannerWithBackendValidationRetry(context.Background(), agent, input, config, llm.AgentModePropose)
	if err == nil || !strings.Contains(err.Error(), "persist experiment planner invocation") {
		t.Fatalf("expected persistence failure, got result=%#v err=%v", result, err)
	}
	if result.Payload != nil {
		t.Fatalf("planner produced a decision payload without persisted identity: %#v", result.Payload)
	}
}
