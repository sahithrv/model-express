package store

import (
	"context"
	"encoding/json"
	"fmt"

	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/memory"
)

func (s *PostgresStore) CreateAgentDecision(projectID string, planID string, decisionType string, rationale string, payload map[string]any) (decisions.AgentDecision, error) {
	if err := s.requireProject(projectID); err != nil {
		return decisions.AgentDecision{}, err
	}
	if payload == nil {
		payload = map[string]any{}
	}

	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return decisions.AgentDecision{}, fmt.Errorf("marshal agent decision payload: %w", err)
	}

	const query = `
		INSERT INTO agent_decisions (project_id, plan_id, decision_type, rationale, payload)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, project_id, plan_id, decision_type, rationale, payload, created_at
	`

	return scanAgentDecision(s.db.QueryRowContext(
		context.Background(),
		query,
		projectID,
		planID,
		decisionType,
		rationale,
		payloadJSON,
	))
}

func (s *PostgresStore) ListProjectAgentDecisions(projectID string) ([]decisions.AgentDecision, error) {
	return s.listProjectAgentDecisions(projectID)
}

func (s *PostgresStore) ListProjectAgentDecisionActivity(projectID string, limit int) ([]decisions.AgentDecision, error) {
	return s.listProjectAgentDecisionsActivity(projectID, boundedActivityReadLimit(limit))
}

func (s *PostgresStore) listProjectAgentDecisions(projectID string) ([]decisions.AgentDecision, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, plan_id, decision_type, rationale, payload, created_at
		FROM agent_decisions
		WHERE project_id = $1
		ORDER BY created_at DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []decisions.AgentDecision{}
	for rows.Next() {
		decision, err := scanAgentDecision(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, decision)
	}

	return out, rows.Err()
}

func (s *PostgresStore) listProjectAgentDecisionsActivity(projectID string, limit int) ([]decisions.AgentDecision, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	rows, err := s.db.QueryContext(context.Background(), agentDecisionActivitySelectQuery(), projectID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []decisions.AgentDecision{}
	for rows.Next() {
		decision, err := scanAgentDecision(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, decision)
	}
	return out, rows.Err()
}

func agentDecisionActivitySelectQuery() string {
	return `
		SELECT id, project_id, plan_id, decision_type, rationale, payload, created_at
		FROM agent_decisions
		WHERE project_id = $1
		ORDER BY created_at DESC, id DESC
		LIMIT $2
	`
}

func (s *PostgresStore) CreateAgentInvocation(invocation memory.AgentInvocation) (memory.AgentInvocation, error) {
	if err := s.requireProject(invocation.ProjectID); err != nil {
		return memory.AgentInvocation{}, err
	}
	if invocation.InputMessages == nil {
		invocation.InputMessages = []map[string]string{}
	}
	if invocation.InputContext == nil {
		invocation.InputContext = map[string]any{}
	}
	if invocation.ParsedOutput == nil {
		invocation.ParsedOutput = map[string]any{}
	}
	if invocation.HumanFeedback == nil {
		invocation.HumanFeedback = map[string]any{}
	}
	if invocation.DownstreamOutcome == nil {
		invocation.DownstreamOutcome = map[string]any{}
	}

	inputMessagesJSON, err := json.Marshal(invocation.InputMessages)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation input messages: %w", err)
	}
	inputContextJSON, err := json.Marshal(invocation.InputContext)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation input context: %w", err)
	}
	parsedOutputJSON, err := json.Marshal(invocation.ParsedOutput)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation parsed output: %w", err)
	}
	humanFeedbackJSON, err := json.Marshal(invocation.HumanFeedback)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation human feedback: %w", err)
	}
	downstreamOutcomeJSON, err := json.Marshal(invocation.DownstreamOutcome)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation downstream outcome: %w", err)
	}

	const query = `
		INSERT INTO agent_invocations (
			project_id,
			dataset_id,
			plan_id,
			job_id,
			agent_name,
			agent_version,
			prompt_version,
			provider,
			model,
			input_messages,
			input_context,
			raw_output,
			parsed_output,
			validation_status,
			validation_error,
			accepted_for_memory,
			human_feedback,
			downstream_outcome
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
		RETURNING id, project_id, dataset_id, plan_id, job_id, agent_name, agent_version, prompt_version, provider, model, input_messages, input_context, raw_output, parsed_output, validation_status, validation_error, accepted_for_memory, human_feedback, downstream_outcome, created_at
	`
	return scanAgentInvocation(s.db.QueryRowContext(
		context.Background(),
		query,
		invocation.ProjectID,
		invocation.DatasetID,
		invocation.PlanID,
		invocation.JobID,
		invocation.AgentName,
		invocation.AgentVersion,
		invocation.PromptVersion,
		invocation.Provider,
		invocation.Model,
		inputMessagesJSON,
		inputContextJSON,
		invocation.RawOutput,
		parsedOutputJSON,
		invocation.ValidationStatus,
		invocation.ValidationError,
		invocation.AcceptedForMemory,
		humanFeedbackJSON,
		downstreamOutcomeJSON,
	))
}

func (s *PostgresStore) GetAgentInvocation(invocationID string) (memory.AgentInvocation, error) {
	const query = `
		SELECT id, project_id, dataset_id, plan_id, job_id, agent_name, agent_version, prompt_version, provider, model, input_messages, input_context, raw_output, parsed_output, validation_status, validation_error, accepted_for_memory, human_feedback, downstream_outcome, created_at
		FROM agent_invocations
		WHERE id = $1
	`
	return scanAgentInvocation(s.db.QueryRowContext(context.Background(), query, invocationID))
}

func (s *PostgresStore) UpdateAgentInvocationDownstreamOutcome(invocationID string, outcome map[string]any) (memory.AgentInvocation, error) {
	if outcome == nil {
		outcome = map[string]any{}
	}
	outcomeJSON, err := json.Marshal(outcome)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation downstream outcome: %w", err)
	}

	const query = `
		UPDATE agent_invocations
		SET downstream_outcome = $2
		WHERE id = $1
		RETURNING id, project_id, dataset_id, plan_id, job_id, agent_name, agent_version, prompt_version, provider, model, input_messages, input_context, raw_output, parsed_output, validation_status, validation_error, accepted_for_memory, human_feedback, downstream_outcome, created_at
	`
	return scanAgentInvocation(s.db.QueryRowContext(context.Background(), query, invocationID, outcomeJSON))
}

func (s *PostgresStore) ListProjectAgentInvocations(projectID string, filter memory.AgentInvocationFilter) ([]memory.AgentInvocation, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if filter.Limit <= 0 {
		filter.Limit = 25
	}

	const query = `
		SELECT id, project_id, dataset_id, plan_id, job_id, agent_name, agent_version, prompt_version, provider, model, input_messages, input_context, raw_output, parsed_output, validation_status, validation_error, accepted_for_memory, human_feedback, downstream_outcome, created_at
		FROM agent_invocations
		WHERE project_id = $1
			AND ($2 = '' OR dataset_id = $2)
			AND ($3 = '' OR plan_id = $3)
			AND ($4 = '' OR job_id = $4)
			AND ($5 = '' OR agent_name = $5)
		ORDER BY created_at DESC
		LIMIT $6
	`
	rows, err := s.db.QueryContext(
		context.Background(),
		query,
		projectID,
		filter.DatasetID,
		filter.PlanID,
		filter.JobID,
		filter.AgentName,
		filter.Limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []memory.AgentInvocation{}
	for rows.Next() {
		invocation, err := scanAgentInvocation(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, invocation)
	}
	return out, rows.Err()
}

func (s *PostgresStore) ListProjectAgentInvocationActivity(projectID string, limit int) ([]memory.AgentInvocationActivity, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	limit = boundedActivityReadLimit(limit)

	rows, err := s.db.QueryContext(context.Background(), agentInvocationActivitySelectQuery(), projectID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []memory.AgentInvocationActivity{}
	for rows.Next() {
		invocation, err := scanAgentInvocationActivity(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, invocation)
	}
	return out, rows.Err()
}

func agentInvocationActivitySelectQuery() string {
	return `
		SELECT
			id,
			project_id,
			plan_id,
			job_id,
			left(agent_name, 128),
			left(validation_status, 64),
			left(validation_error, 512),
			jsonb_strip_nulls(jsonb_build_object(
				'backend_validation_status', CASE
					WHEN jsonb_typeof(downstream_outcome->'backend_validation_status') = 'string'
					THEN to_jsonb(left(downstream_outcome->>'backend_validation_status', 64)) END,
				'backend_validation_error', CASE
					WHEN jsonb_typeof(downstream_outcome->'backend_validation_error') = 'string'
					THEN to_jsonb(left(downstream_outcome->>'backend_validation_error', 512)) END,
				'will_retry', CASE
					WHEN jsonb_typeof(downstream_outcome->'will_retry') = 'boolean'
						THEN downstream_outcome->'will_retry'
					WHEN jsonb_typeof(downstream_outcome->'will_retry') = 'string'
						THEN to_jsonb(left(downstream_outcome->>'will_retry', 16)) END,
				'retry_attempt', CASE
					WHEN jsonb_typeof(downstream_outcome->'retry_attempt') = 'number'
					THEN downstream_outcome->'retry_attempt' END,
				'completion_state', CASE
					WHEN jsonb_typeof(downstream_outcome->'completion_state') = 'string'
					THEN to_jsonb(left(downstream_outcome->>'completion_state', 128)) END
			)) AS activity_outcome,
			created_at
		FROM agent_invocations
		WHERE project_id = $1
		ORDER BY created_at DESC, id DESC
		LIMIT $2
	`
}

func scanAgentInvocationActivity(row rowScanner) (memory.AgentInvocationActivity, error) {
	var invocation memory.AgentInvocationActivity
	var outcomeJSON []byte
	if err := row.Scan(
		&invocation.ID,
		&invocation.ProjectID,
		&invocation.PlanID,
		&invocation.JobID,
		&invocation.AgentName,
		&invocation.ValidationStatus,
		&invocation.ValidationError,
		&outcomeJSON,
		&invocation.CreatedAt,
	); err != nil {
		return memory.AgentInvocationActivity{}, normalizeSQLError(err)
	}
	invocation.DownstreamOutcome = map[string]any{}
	if len(outcomeJSON) > 0 {
		if err := json.Unmarshal(outcomeJSON, &invocation.DownstreamOutcome); err != nil {
			return memory.AgentInvocationActivity{}, fmt.Errorf("unmarshal agent invocation activity outcome: %w", err)
		}
	}
	return invocation, nil
}
