package api

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
)

func automationSettingsFromEnv() settings.AutomationSettings {
	defaultProvider := os.Getenv("MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER")
	if defaultProvider == "" {
		defaultProvider = "local"
	}
	agentMode := llm.NormalizeAgentMode(os.Getenv("MODEL_EXPRESS_AGENT_MODE"))

	return settings.AutomationSettings{
		AutoReviewExperiments:   envFlag("MODEL_EXPRESS_AUTO_REVIEW_EXPERIMENTS", false),
		AutoScheduleFollowUps:   envFlag("MODEL_EXPRESS_AUTO_SCHEDULE_FOLLOWUPS", false),
		AutoExecutePlans:        envFlag("MODEL_EXPRESS_AUTO_EXECUTE_PLANS", false),
		MaxFollowUpRounds:       maxAutoFollowUpRoundsFromEnvForMode(agentMode),
		DefaultTrainingProvider: normalizeTrainingProvider(defaultProvider),
		DefaultGPUType:          os.Getenv("MODEL_EXPRESS_DEFAULT_GPU_TYPE"),
		CostMode:                normalizeCostMode(os.Getenv("MODEL_EXPRESS_COST_MODE")),
		BudgetCapUSD:            budgetCapUSDFromEnv(),
		LLMEnabled:              envFlag("MODEL_EXPRESS_LLM_ENABLED", false),
		AgentMode:               agentMode,
		LLMProvider:             defaultLLMProviderFromEnv(),
		LLMModel:                defaultLLMModelFromEnv(),
		AutoMLEnabled:           envFlag("MODEL_EXPRESS_AUTOML_ENABLED", false),
		AutoMLSampler:           defaultAutoMLSamplerFromEnv(),
		UpdatedAt:               time.Now().UTC(),
	}
}

func (s *Server) currentAutomationSettings() settings.AutomationSettings {
	s.settingsMu.RLock()
	defer s.settingsMu.RUnlock()

	return s.automationSettings
}

func (s *Server) setAutomationSettings(automationSettings settings.AutomationSettings) {
	s.settingsMu.Lock()
	defer s.settingsMu.Unlock()

	s.automationSettings = automationSettings
}

func applyAutomationSettingsUpdate(current settings.AutomationSettings, update settings.AutomationSettingsUpdate) (settings.AutomationSettings, error) {
	if update.AutoReviewExperiments != nil {
		current.AutoReviewExperiments = *update.AutoReviewExperiments
	}
	if update.AutoScheduleFollowUps != nil {
		current.AutoScheduleFollowUps = *update.AutoScheduleFollowUps
	}
	if update.AutoExecutePlans != nil {
		current.AutoExecutePlans = *update.AutoExecutePlans
	}
	if update.MaxFollowUpRounds != nil {
		if *update.MaxFollowUpRounds < 0 {
			return settings.AutomationSettings{}, fmt.Errorf("%w: max_followup_rounds must be at least 0", store.ErrInvalidRequest)
		}
		current.MaxFollowUpRounds = *update.MaxFollowUpRounds
	}
	if update.DefaultTrainingProvider != nil {
		current.DefaultTrainingProvider = normalizeTrainingProvider(*update.DefaultTrainingProvider)
		if current.DefaultTrainingProvider == "" {
			current.DefaultTrainingProvider = "local"
		}
		if err := validateTrainingProviderConfigured(current.DefaultTrainingProvider); err != nil {
			return settings.AutomationSettings{}, err
		}
	}
	if update.DefaultGPUType != nil {
		current.DefaultGPUType = strings.TrimSpace(*update.DefaultGPUType)
	}
	if update.CostMode != nil {
		current.CostMode = normalizeCostMode(*update.CostMode)
	}
	if update.BudgetCapUSD != nil {
		if *update.BudgetCapUSD < 0 {
			return settings.AutomationSettings{}, fmt.Errorf("%w: budget_cap_usd must be at least 0", store.ErrInvalidRequest)
		}
		current.BudgetCapUSD = *update.BudgetCapUSD
	}
	if update.LLMEnabled != nil {
		current.LLMEnabled = *update.LLMEnabled
	}
	if update.AgentMode != nil {
		current.AgentMode = llm.NormalizeAgentMode(*update.AgentMode)
	}
	if update.LLMProvider != nil {
		current.LLMProvider = normalizeLLMProvider(*update.LLMProvider)
	}
	if update.LLMModel != nil {
		current.LLMModel = strings.TrimSpace(*update.LLMModel)
	}
	if update.AutoMLEnabled != nil {
		current.AutoMLEnabled = *update.AutoMLEnabled
	}
	if update.AutoMLSampler != nil {
		current.AutoMLSampler = normalizeAutoMLSampler(*update.AutoMLSampler)
	}
	if current.AgentMode == "" {
		current.AgentMode = llm.AgentModePropose
	}
	if current.LLMProvider == "" {
		current.LLMProvider = llm.ProviderOpenAI
	}
	if current.AutoMLSampler == "" {
		current.AutoMLSampler = automl.SamplerSeededRandom
	}
	if current.CostMode == "" {
		current.CostMode = "balanced"
	}
	if _, err := automl.NewSampler(current.AutoMLSampler); err != nil {
		return settings.AutomationSettings{}, fmt.Errorf("%w: %s", store.ErrInvalidRequest, err.Error())
	}

	return current, nil
}

func budgetCapUSDFromEnv() float64 {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_BUDGET_CAP_USD"))
	if value == "" {
		return 0
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || parsed < 0 {
		return 0
	}
	return parsed
}

func (s *Server) defaultExecuteExperimentPlanRequest() executeExperimentPlanRequest {
	automationSettings := s.currentAutomationSettings()
	provider := automationSettings.DefaultTrainingProvider
	if provider == "" {
		provider = "local"
	}

	return executeExperimentPlanRequest{
		Provider:          provider,
		GPUType:           automationSettings.DefaultGPUType,
		MaxConcurrentJobs: effectiveExecutionMaxConcurrentJobs(provider, 0),
	}
}

func (s *Server) defaultVisualAnalysisProvider() string {
	provider := strings.ToLower(strings.TrimSpace(s.currentAutomationSettings().DefaultTrainingProvider))
	if provider == "" {
		return "local"
	}
	return provider
}

func (s *Server) shouldAutoReviewExperimentJobs() bool {
	return s.currentAutomationSettings().AutoReviewExperiments
}

func (s *Server) shouldAutoScheduleFollowUps() bool {
	return s.currentAutomationSettings().AutoScheduleFollowUps
}

func maxAutoFollowUpRoundsFromEnv() int {
	return maxAutoFollowUpRoundsFromEnvForMode(llm.NormalizeAgentMode(os.Getenv("MODEL_EXPRESS_AGENT_MODE")))
}

func maxAutoFollowUpRoundsFromEnvForMode(agentMode string) int {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_MAX_FOLLOWUP_ROUNDS"))
	if value == "" {
		if !fastRemoteExecutionProfileEnabled() {
			return 1
		}
		if strings.EqualFold(agentMode, llm.AgentModeAutonomous) {
			return plannerAutonomousMaxFollowUpRounds
		}
		return plannerDefaultMaxFollowUpRounds
	}

	rounds, err := strconv.Atoi(value)
	if err != nil || rounds < 0 {
		if strings.EqualFold(agentMode, llm.AgentModeAutonomous) {
			return plannerAutonomousMaxFollowUpRounds
		}
		return plannerDefaultMaxFollowUpRounds
	}

	return rounds
}

func maxAutoWorkersFromEnv() int {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_MAX_AUTO_WORKERS"))
	if value == "" {
		if fastRemoteExecutionProfileEnabled() {
			return 4
		}
		return 1
	}
	count, err := strconv.Atoi(value)
	if err != nil || count < 1 {
		if fastRemoteExecutionProfileEnabled() {
			return 4
		}
		return 1
	}
	if !fastRemoteExecutionProfileEnabled() && count > 1 {
		return 1
	}
	return count
}

func effectiveExecutionMaxConcurrentJobs(provider string, requested int) int {
	provider = normalizeTrainingProvider(provider)
	if fastRemoteExecutionProfileEnabled() {
		if requested > 0 {
			return requested
		}
		return 0
	}
	switch provider {
	case "", "local", "modal":
		cap := localMaxConcurrentJobsFromEnv()
		if requested < 1 || requested > cap {
			return cap
		}
	}
	return requested
}

func localMaxConcurrentJobsFromEnv() int {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LOCAL_MAX_CONCURRENT_JOBS"))
	if value == "" {
		return 1
	}
	count, err := strconv.Atoi(value)
	if err != nil || count < 1 {
		return 1
	}
	if !fastRemoteExecutionProfileEnabled() && count > 1 {
		return 1
	}
	return count
}

func fastRemoteExecutionProfileEnabled() bool {
	for _, name := range []string{"MODEL_EXPRESS_EXECUTION_PROFILE", "MODEL_EXPRESS_V1_PROFILE"} {
		value := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
		value = strings.ReplaceAll(value, "_", "-")
		if value == "fast-remote" {
			return true
		}
	}
	return false
}

func (s *Server) maxAutoFollowUpRounds() int {
	return s.currentAutomationSettings().MaxFollowUpRounds
}

func (s *Server) shouldAutoExecuteExperimentPlans() bool {
	return s.currentAutomationSettings().AutoExecutePlans
}

func (s *Server) shouldRunLLMAgents() bool {
	return s.currentAutomationSettings().LLMEnabled
}

func defaultLLMProviderFromEnv() string {
	if provider := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_PROVIDER")); provider != "" {
		return normalizeLLMProvider(provider)
	}
	if provider := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_PROVIDER")); provider != "" {
		return normalizeLLMProvider(provider)
	}
	return llm.ProviderOpenAI
}

func defaultLLMModelFromEnv() string {
	if model := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LLM_MODEL")); model != "" {
		return model
	}
	return strings.TrimSpace(os.Getenv("MODEL_EXPRESS_VISUAL_LLM_MODEL"))
}

func defaultAutoMLSamplerFromEnv() string {
	return normalizeAutoMLSampler(os.Getenv("MODEL_EXPRESS_AUTOML_SAMPLER"))
}

func normalizeAutoMLSampler(value string) string {
	normalized := strings.ToLower(strings.TrimSpace(value))
	if normalized == "" {
		return automl.SamplerSeededRandom
	}
	switch normalized {
	case "random", "random_search":
		return automl.SamplerSeededRandom
	case "grid_search":
		return automl.SamplerGrid
	case "adaptive", "bayesian", "bayesian_optimizer":
		return automl.SamplerAdaptiveBayesian
	default:
		return normalized
	}
}

func normalizeLLMProvider(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case llm.ProviderOpenAICompatible:
		return llm.ProviderOpenAICompatible
	case llm.ProviderLocal:
		return llm.ProviderLocal
	default:
		return llm.ProviderOpenAI
	}
}
