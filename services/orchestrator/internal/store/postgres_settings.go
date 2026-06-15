package store

import (
	"context"

	"model-express/services/orchestrator/internal/settings"
)

func (s *PostgresStore) GetAutomationSettings() (settings.AutomationSettings, error) {
	const query = `
		SELECT auto_review_experiments, auto_schedule_followups, auto_execute_plans, max_followup_rounds, default_training_provider, default_gpu_type, cost_mode, budget_cap_usd, llm_enabled, agent_mode, llm_provider, llm_model, automl_enabled, automl_sampler, updated_at
		FROM automation_settings
		WHERE singleton = true
	`

	return scanAutomationSettings(s.db.QueryRowContext(context.Background(), query))
}

func (s *PostgresStore) SaveAutomationSettings(automationSettings settings.AutomationSettings) (settings.AutomationSettings, error) {
	const query = `
		INSERT INTO automation_settings (
			singleton,
			auto_review_experiments,
			auto_schedule_followups,
			auto_execute_plans,
			max_followup_rounds,
			default_training_provider,
			default_gpu_type,
			cost_mode,
			budget_cap_usd,
			llm_enabled,
			agent_mode,
			llm_provider,
			llm_model,
			automl_enabled,
			automl_sampler
		)
		VALUES (true, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
		ON CONFLICT (singleton) DO UPDATE SET
			auto_review_experiments = EXCLUDED.auto_review_experiments,
			auto_schedule_followups = EXCLUDED.auto_schedule_followups,
			auto_execute_plans = EXCLUDED.auto_execute_plans,
			max_followup_rounds = EXCLUDED.max_followup_rounds,
			default_training_provider = EXCLUDED.default_training_provider,
			default_gpu_type = EXCLUDED.default_gpu_type,
			cost_mode = EXCLUDED.cost_mode,
			budget_cap_usd = EXCLUDED.budget_cap_usd,
			llm_enabled = EXCLUDED.llm_enabled,
			agent_mode = EXCLUDED.agent_mode,
			llm_provider = EXCLUDED.llm_provider,
			llm_model = EXCLUDED.llm_model,
			automl_enabled = EXCLUDED.automl_enabled,
			automl_sampler = EXCLUDED.automl_sampler,
			updated_at = now()
		RETURNING auto_review_experiments, auto_schedule_followups, auto_execute_plans, max_followup_rounds, default_training_provider, default_gpu_type, cost_mode, budget_cap_usd, llm_enabled, agent_mode, llm_provider, llm_model, automl_enabled, automl_sampler, updated_at
	`

	return scanAutomationSettings(s.db.QueryRowContext(
		context.Background(),
		query,
		automationSettings.AutoReviewExperiments,
		automationSettings.AutoScheduleFollowUps,
		automationSettings.AutoExecutePlans,
		automationSettings.MaxFollowUpRounds,
		automationSettings.DefaultTrainingProvider,
		automationSettings.DefaultGPUType,
		automationSettings.CostMode,
		automationSettings.BudgetCapUSD,
		automationSettings.LLMEnabled,
		automationSettings.AgentMode,
		automationSettings.LLMProvider,
		automationSettings.LLMModel,
		automationSettings.AutoMLEnabled,
		automationSettings.AutoMLSampler,
	))
}
