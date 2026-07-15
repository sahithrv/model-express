package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plannervalidation"
)

const agentInvocationSelectColumns = `id, project_id, dataset_id, plan_id, job_id, agent_name, agent_version, prompt_version,
	planner_variant_id, planner_variant, validation_mode, attempt_group_id, attempt_index, retry_reason, wall_latency_ms,
	provider_usage, derived_cost, provider, model, input_messages, input_context, raw_output, parsed_output,
	validation_status, validation_error, strict_validation_verdict, validation_outcome,
	accepted_for_memory, human_feedback, downstream_outcome, created_at`

func (s *PostgresStore) CreateAgentDecision(projectID string, planID string, decisionType string, rationale string, payload map[string]any) (decisions.AgentDecision, error) {
	if payload == nil {
		payload = map[string]any{}
	}

	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return decisions.AgentDecision{}, fmt.Errorf("marshal agent decision payload: %w", err)
	}

	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return decisions.AgentDecision{}, err
	}
	defer tx.Rollback()
	if err := requireAgentProjectTx(ctx, tx, projectID); err != nil {
		return decisions.AgentDecision{}, err
	}

	decision, err := createAgentDecisionTx(ctx, tx, projectID, planID, decisionType, rationale, payloadJSON)
	if err != nil {
		return decisions.AgentDecision{}, err
	}
	if err := tx.Commit(); err != nil {
		return decisions.AgentDecision{}, err
	}
	return decision, nil
}

func createAgentDecisionTx(ctx context.Context, tx *sql.Tx, projectID string, planID string, decisionType string, rationale string, payloadJSON []byte) (decisions.AgentDecision, error) {
	const query = `
		INSERT INTO agent_decisions (project_id, plan_id, decision_type, rationale, payload)
		VALUES ($1, $2, $3, $4, $5)
		RETURNING id, project_id, plan_id, decision_type, rationale, payload, created_at
	`

	decision, err := scanAgentDecision(tx.QueryRowContext(
		ctx,
		query,
		projectID,
		planID,
		decisionType,
		rationale,
		payloadJSON,
	))
	if err != nil {
		return decisions.AgentDecision{}, err
	}
	create, err := agentDecisionRecordedEvent(decision)
	if err != nil {
		return decisions.AgentDecision{}, invalidAgentTransitionError("create agent decision transition", err)
	}
	if _, _, err := appendExecutionTransitionTx(ctx, tx, create); err != nil {
		return decisions.AgentDecision{}, err
	}
	return decision, nil
}

func (s *PostgresStore) CreateAgentDecisionWithCandidateProvenance(
	projectID string,
	planID string,
	decisionType string,
	rationale string,
	payload map[string]any,
	candidates []calibration.CandidateProvenanceCreate,
) (decisions.AgentDecision, []calibration.CandidateProvenance, error) {
	if strings.ToUpper(strings.TrimSpace(decisionType)) != decisions.TypeAddExperiments {
		return decisions.AgentDecision{}, nil, fmt.Errorf("%w: candidate provenance is only valid for ADD_EXPERIMENTS decisions", ErrInvalidRequest)
	}
	if err := validateCandidateProvenanceCreates(candidates); err != nil {
		return decisions.AgentDecision{}, nil, err
	}
	if payload == nil {
		payload = map[string]any{}
	}
	payloadJSON, err := json.Marshal(payload)
	if err != nil {
		return decisions.AgentDecision{}, nil, fmt.Errorf("marshal agent decision payload: %w", err)
	}
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return decisions.AgentDecision{}, nil, err
	}
	defer tx.Rollback()
	if err := requireAgentProjectTx(ctx, tx, projectID); err != nil {
		return decisions.AgentDecision{}, nil, err
	}
	decision, err := createAgentDecisionTx(ctx, tx, projectID, planID, decisionType, rationale, payloadJSON)
	if err != nil {
		return decisions.AgentDecision{}, nil, err
	}
	rows, err := ensureCandidateProvenanceTx(ctx, tx, decision, candidates)
	if err != nil {
		return decisions.AgentDecision{}, nil, err
	}
	if err := tx.Commit(); err != nil {
		return decisions.AgentDecision{}, nil, err
	}
	return decision, rows, nil
}

func (s *PostgresStore) EnsureCandidateProvenance(decision decisions.AgentDecision, candidates []calibration.CandidateProvenanceCreate) ([]calibration.CandidateProvenance, error) {
	if err := validateCandidateProvenanceCreates(candidates); err != nil {
		return nil, err
	}
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	var projectID string
	var decisionType string
	if err := tx.QueryRowContext(ctx, `SELECT project_id, decision_type FROM agent_decisions WHERE id = $1 FOR SHARE`, decision.ID).Scan(&projectID, &decisionType); err != nil {
		return nil, normalizeSQLError(err)
	}
	if projectID != decision.ProjectID {
		return nil, fmt.Errorf("%w: candidate decision project mismatch", ErrInvalidRequest)
	}
	if strings.ToUpper(strings.TrimSpace(decisionType)) != decisions.TypeAddExperiments {
		return nil, fmt.Errorf("%w: candidate provenance is only valid for ADD_EXPERIMENTS decisions", ErrInvalidRequest)
	}
	rows, err := ensureCandidateProvenanceTx(ctx, tx, decision, candidates)
	if err != nil {
		return nil, err
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	return rows, nil
}

func (s *PostgresStore) FinalizeCandidateOutcomes(decisionID string, updates []calibration.CandidateOutcomeUpdate) ([]calibration.CandidateProvenance, error) {
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	rows, err := listCandidateProvenanceQuery(ctx, tx, `WHERE decision_id = $1 ORDER BY candidate_index FOR UPDATE`, decisionID)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, ErrNotFound
	}
	byIndex := make(map[int]calibration.CandidateProvenance, len(rows))
	for _, row := range rows {
		byIndex[row.CandidateIndex] = row
	}
	seen := map[int]bool{}
	now := time.Now().UTC()
	for _, update := range updates {
		if seen[update.CandidateIndex] {
			return nil, fmt.Errorf("%w: duplicate candidate outcome index %d", ErrInvalidRequest, update.CandidateIndex)
		}
		seen[update.CandidateIndex] = true
		row, ok := byIndex[update.CandidateIndex]
		if !ok {
			return nil, fmt.Errorf("%w: candidate outcome index %d does not exist", ErrInvalidRequest, update.CandidateIndex)
		}
		next, err := calibration.ApplyCandidateOutcomeUpdate(row, update, now)
		if err != nil {
			return nil, fmt.Errorf("%w: %v", ErrInvalidRequest, err)
		}
		if _, err := tx.ExecContext(ctx, `
			UPDATE planner_candidate_provenance
			SET followup_plan_id=$3, experiment_id=$4, job_id=$5, attempt_id=$6,
				realized_effective_hash=$7, outcome_status=$8, actual_score=$9, actual_delta=$10,
				terminal_state=$11, cost_usd=$12, runtime_seconds=$13, calibration_eligible=$14,
				eligibility_reason=$15, finalized_at=$16
			WHERE decision_id=$1 AND candidate_index=$2
		`, decisionID, next.CandidateIndex, next.FollowUpPlanID, next.ExperimentID, next.JobID, next.AttemptID,
			next.RealizedEffectiveHash, next.OutcomeStatus, next.ActualScore, next.ActualDelta, next.TerminalState,
			next.CostUSD, next.RuntimeSeconds, next.CalibrationEligible, next.EligibilityReason, next.FinalizedAt); err != nil {
			return nil, normalizeSQLError(err)
		}
		byIndex[next.CandidateIndex] = next
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	for index := range rows {
		rows[index] = byIndex[rows[index].CandidateIndex]
	}
	return rows, nil
}

func validateCandidateProvenanceCreates(candidates []calibration.CandidateProvenanceCreate) error {
	if len(candidates) == 0 {
		return fmt.Errorf("%w: accepted planner decision requires candidate provenance", ErrInvalidRequest)
	}
	seen := map[int]bool{}
	for _, candidate := range candidates {
		if err := calibration.ValidateCandidateProvenanceCreate(candidate); err != nil {
			return fmt.Errorf("%w: %v", ErrInvalidRequest, err)
		}
		if seen[candidate.CandidateIndex] {
			return fmt.Errorf("%w: duplicate candidate_index %d", ErrInvalidRequest, candidate.CandidateIndex)
		}
		seen[candidate.CandidateIndex] = true
	}
	return nil
}

func ensureCandidateProvenanceTx(ctx context.Context, tx *sql.Tx, decision decisions.AgentDecision, candidates []calibration.CandidateProvenanceCreate) ([]calibration.CandidateProvenance, error) {
	const insert = `
		INSERT INTO planner_candidate_provenance (
			project_id, invocation_id, decision_id, planner_variant_id, candidate_index,
			requested_config_hash, accepted_spec_hash, task, mechanism,
			forecast_target, metric_direction, score_basis, score_version, baseline_job_id,
			baseline_score, predicted_delta, prediction_source, forecast_units, valid_range_min, valid_range_max,
			base_score, selection_trace_reference, selected, rejected, selection_state,
			selected_experiment_index, outcome_status, reasons
		)
		SELECT $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
			$15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28
		FROM agent_invocations
		WHERE id = $2 AND project_id = $1 AND planner_variant_id = $4
		ON CONFLICT (decision_id, candidate_index) DO NOTHING
	`
	for _, candidate := range candidates {
		reasonsJSON, err := json.Marshal(candidate.Reasons)
		if err != nil {
			return nil, fmt.Errorf("marshal candidate provenance reasons: %w", err)
		}
		result, err := tx.ExecContext(ctx, insert,
			decision.ProjectID, candidate.InvocationID, decision.ID, candidate.PlannerVariantID, candidate.CandidateIndex,
			candidate.RequestedConfigHash, candidate.AcceptedSpecHash, candidate.Task, candidate.Mechanism,
			candidate.Forecast.ForecastTarget, candidate.Forecast.MetricDirection, candidate.Forecast.ScoreBasis, candidate.Forecast.ScoreVersion, candidate.Forecast.BaselineJobID,
			candidate.Forecast.BaselineScore, candidate.Forecast.PredictedDelta, candidate.Forecast.PredictionSource, candidate.Forecast.Units, candidate.Forecast.ValidRange.Min, candidate.Forecast.ValidRange.Max,
			candidate.BaseScore, candidate.SelectionTraceReference, candidate.Selected, candidate.Rejected, candidate.SelectionState,
			candidate.SelectedExperimentIndex, candidate.OutcomeStatus, reasonsJSON,
		)
		if err != nil {
			return nil, normalizeSQLError(err)
		}
		if affected, err := result.RowsAffected(); err == nil && affected == 0 {
			var exists bool
			if err := tx.QueryRowContext(ctx, `SELECT EXISTS (SELECT 1 FROM planner_candidate_provenance WHERE decision_id = $1 AND candidate_index = $2)`, decision.ID, candidate.CandidateIndex).Scan(&exists); err != nil {
				return nil, err
			}
			if !exists {
				return nil, fmt.Errorf("%w: candidate invocation or planner variant does not match project", ErrInvalidRequest)
			}
		}
	}
	rows, err := listCandidateProvenanceQuery(ctx, tx, `WHERE decision_id = $1 ORDER BY candidate_index`, decision.ID)
	if err != nil {
		return nil, err
	}
	if len(rows) != len(candidates) {
		return nil, fmt.Errorf("%w: candidate provenance count %d does not match accepted candidate count %d", ErrInvalidRequest, len(rows), len(candidates))
	}
	byIndex := make(map[int]calibration.CandidateProvenance, len(rows))
	for _, row := range rows {
		byIndex[row.CandidateIndex] = row
	}
	for _, candidate := range candidates {
		row, ok := byIndex[candidate.CandidateIndex]
		if !ok || !calibration.CandidateProvenanceMatchesCreate(row, candidate) {
			return nil, fmt.Errorf("%w: candidate provenance at index %d conflicts with the immutable decision-time record", ErrInvalidRequest, candidate.CandidateIndex)
		}
	}
	return rows, nil
}

func (s *PostgresStore) ListDecisionCandidateProvenance(decisionID string) ([]calibration.CandidateProvenance, error) {
	return listCandidateProvenanceQuery(context.Background(), s.db, `WHERE decision_id = $1 ORDER BY candidate_index`, decisionID)
}

func (s *PostgresStore) ListProjectCandidateProvenance(projectID string) ([]calibration.CandidateProvenance, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	return listCandidateProvenanceQuery(context.Background(), s.db, `WHERE project_id = $1 ORDER BY created_at DESC, decision_id DESC, candidate_index`, projectID)
}

func (s *PostgresStore) ReadCalibrationObservations(projectID string, window calibration.TimeWindow, limit int) (calibration.ObservationSet, error) {
	if err := validateCalibrationRead(projectID, window, limit); err != nil {
		return calibration.ObservationSet{}, err
	}
	if err := s.requireProject(projectID); err != nil {
		return calibration.ObservationSet{}, err
	}
	ctx := context.Background()
	rows, err := s.db.QueryContext(ctx, `
		SELECT bounded.*, decisions.payload
		FROM (
			SELECT `+candidateProvenanceSelectColumns+`
			FROM planner_candidate_provenance
			WHERE project_id = $1 AND created_at >= $2 AND created_at < $3
			ORDER BY created_at ASC, decision_id ASC, candidate_index ASC
			LIMIT $4
		) AS bounded
		JOIN agent_decisions AS decisions ON decisions.id = bounded.decision_id
		ORDER BY bounded.created_at ASC, bounded.decision_id ASC, bounded.candidate_index ASC
	`, projectID, window.Start.UTC(), window.End.UTC(), limit+1)
	if err != nil {
		return calibration.ObservationSet{}, err
	}
	candidates := []calibration.CandidateObservation{}
	for rows.Next() {
		candidate, err := scanCalibrationCandidateObservation(rows)
		if err != nil {
			rows.Close()
			return calibration.ObservationSet{}, err
		}
		candidates = append(candidates, candidate)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return calibration.ObservationSet{}, err
	}
	rows.Close()
	candidatesTruncated := len(candidates) > limit
	if candidatesTruncated {
		candidates = candidates[:limit]
	}

	invocationRows, err := s.db.QueryContext(ctx, `
		SELECT `+agentInvocationSelectColumns+`
		FROM agent_invocations
		WHERE project_id = $1 AND agent_name = $2 AND created_at >= $3 AND created_at < $4
		ORDER BY created_at ASC, id ASC
		LIMIT $5
	`, projectID, "experiment_planner", window.Start.UTC(), window.End.UTC(), limit+1)
	if err != nil {
		return calibration.ObservationSet{}, err
	}
	defer invocationRows.Close()
	invocations := []calibration.InvocationObservation{}
	for invocationRows.Next() {
		invocation, err := scanAgentInvocation(invocationRows)
		if err != nil {
			return calibration.ObservationSet{}, err
		}
		invocations = append(invocations, calibrationInvocationObservation(invocation))
	}
	if err := invocationRows.Err(); err != nil {
		return calibration.ObservationSet{}, err
	}
	invocationsTruncated := len(invocations) > limit
	if invocationsTruncated {
		invocations = invocations[:limit]
	}
	return calibration.ObservationSet{
		Candidates: candidates, Invocations: invocations,
		CandidatesTruncated: candidatesTruncated, InvocationsTruncated: invocationsTruncated,
	}, nil
}

func scanCalibrationCandidateObservation(row rowScanner) (calibration.CandidateObservation, error) {
	var candidate calibration.CandidateProvenance
	var state candidateProvenanceScanState
	var payloadJSON []byte
	destinations := candidateProvenanceScanDestinations(&candidate, &state)
	destinations = append(destinations, &payloadJSON)
	if err := row.Scan(destinations...); err != nil {
		return calibration.CandidateObservation{}, normalizeSQLError(err)
	}
	candidate, err := finalizeCandidateProvenanceScan(candidate, state)
	if err != nil {
		return calibration.CandidateObservation{}, err
	}
	payload := map[string]any{}
	if err := json.Unmarshal(payloadJSON, &payload); err != nil {
		return calibration.CandidateObservation{}, fmt.Errorf("unmarshal calibration decision payload: %w", err)
	}
	family, evidenceCount := calibration.CandidateDecisionMetadata(payload, candidate.CandidateIndex)
	return calibration.CandidateObservation{CandidateProvenance: candidate, ModelFamily: family, EvidenceCount: evidenceCount}, nil
}

type candidateProvenanceQueryer interface {
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
}

const candidateProvenanceSelectColumns = `id, project_id, invocation_id, decision_id, planner_variant_id, candidate_index,
		requested_config_hash, accepted_spec_hash, task, mechanism,
		forecast_target, metric_direction, score_basis, score_version, baseline_job_id,
		baseline_score, predicted_delta, prediction_source, forecast_units, valid_range_min, valid_range_max,
		base_score, selection_trace_reference, selected, rejected, selection_state,
		selected_experiment_index, outcome_status, reasons,
		followup_plan_id, experiment_id, job_id, attempt_id, realized_effective_hash,
		actual_score, actual_delta, terminal_state, cost_usd, runtime_seconds,
		calibration_eligible, eligibility_reason, finalized_at, created_at`

func listCandidateProvenanceQuery(ctx context.Context, queryer candidateProvenanceQueryer, clause string, value any) ([]calibration.CandidateProvenance, error) {
	query := `SELECT ` + candidateProvenanceSelectColumns + ` FROM planner_candidate_provenance ` + clause
	rows, err := queryer.QueryContext(ctx, query, value)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []calibration.CandidateProvenance{}
	for rows.Next() {
		row, err := scanCandidateProvenance(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, row)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return out, nil
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
	var err error
	invocation, err = memory.NormalizeAgentInvocationRuntime(invocation)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("normalize agent invocation runtime: %w", err)
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
	plannerVariantJSON := []byte(`{}`)
	if invocation.PlannerVariant != nil {
		plannerVariantJSON, err = json.Marshal(invocation.PlannerVariant)
		if err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("marshal planner variant: %w", err)
		}
	}
	providerUsageJSON, err := json.Marshal(invocation.ProviderUsage)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation provider usage: %w", err)
	}
	derivedCostJSON := []byte(`{}`)
	if invocation.DerivedCost != nil {
		derivedCostJSON, err = json.Marshal(invocation.DerivedCost)
		if err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("marshal agent invocation derived cost: %w", err)
		}
	}
	strictVerdictJSON := []byte(`{}`)
	if invocation.StrictValidationVerdict != nil {
		strictVerdictJSON, err = json.Marshal(invocation.StrictValidationVerdict)
		if err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("marshal planner strict validation verdict: %w", err)
		}
	}
	validationOutcomeJSON := []byte(`{}`)
	if invocation.ValidationOutcome != nil {
		validationOutcomeJSON, err = json.Marshal(invocation.ValidationOutcome)
		if err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("marshal planner validation outcome: %w", err)
		}
	}
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return memory.AgentInvocation{}, err
	}
	defer tx.Rollback()
	if err := requireAgentProjectTx(ctx, tx, invocation.ProjectID); err != nil {
		return memory.AgentInvocation{}, err
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
			planner_variant_id,
			planner_variant,
			validation_mode,
			attempt_group_id,
			attempt_index,
			retry_reason,
			wall_latency_ms,
			provider_usage,
			derived_cost,
			provider,
			model,
			input_messages,
			input_context,
			raw_output,
			parsed_output,
			validation_status,
			validation_error,
			strict_validation_verdict,
			validation_outcome,
			accepted_for_memory,
			human_feedback,
			downstream_outcome
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29)
		RETURNING ` + agentInvocationSelectColumns + `
	`
	stored, err := scanAgentInvocation(tx.QueryRowContext(
		ctx,
		query,
		invocation.ProjectID,
		invocation.DatasetID,
		invocation.PlanID,
		invocation.JobID,
		invocation.AgentName,
		invocation.AgentVersion,
		invocation.PromptVersion,
		invocation.PlannerVariantID,
		plannerVariantJSON,
		invocation.ValidationMode,
		invocation.AttemptGroupID,
		invocation.AttemptIndex,
		invocation.RetryReason,
		invocation.WallLatencyMS,
		providerUsageJSON,
		derivedCostJSON,
		invocation.Provider,
		invocation.Model,
		inputMessagesJSON,
		inputContextJSON,
		invocation.RawOutput,
		parsedOutputJSON,
		invocation.ValidationStatus,
		invocation.ValidationError,
		strictVerdictJSON,
		validationOutcomeJSON,
		invocation.AcceptedForMemory,
		humanFeedbackJSON,
		downstreamOutcomeJSON,
	))
	if err != nil {
		return memory.AgentInvocation{}, err
	}
	create, emit, err := agentInvocationValidationEvent(stored)
	if err != nil {
		return memory.AgentInvocation{}, invalidAgentTransitionError("create agent validation transition", err)
	}
	if emit {
		if _, _, err := appendExecutionTransitionTx(ctx, tx, create); err != nil {
			return memory.AgentInvocation{}, err
		}
	}
	if err := tx.Commit(); err != nil {
		return memory.AgentInvocation{}, err
	}
	return stored, nil
}

func (s *PostgresStore) GetAgentInvocation(invocationID string) (memory.AgentInvocation, error) {
	const query = `
		SELECT ` + agentInvocationSelectColumns + `
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
	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return memory.AgentInvocation{}, err
	}
	defer tx.Rollback()

	const query = `
		UPDATE agent_invocations
		SET downstream_outcome = $2
		WHERE id = $1
		RETURNING ` + agentInvocationSelectColumns + `
	`
	updated, err := scanAgentInvocation(tx.QueryRowContext(ctx, query, invocationID, outcomeJSON))
	if err != nil {
		return memory.AgentInvocation{}, err
	}
	create, emit, err := agentInvocationValidationRetryEvent(updated, outcome)
	if err != nil {
		return memory.AgentInvocation{}, invalidAgentTransitionError("create agent validation retry transition", err)
	}
	if emit {
		if _, _, err := appendExecutionTransitionTx(ctx, tx, create); err != nil {
			return memory.AgentInvocation{}, err
		}
	}
	if err := tx.Commit(); err != nil {
		return memory.AgentInvocation{}, err
	}
	return updated, nil
}

func (s *PostgresStore) UpdateAgentInvocationValidation(invocationID string, verdict plannervalidation.Verdict, outcome plannervalidation.Outcome) (memory.AgentInvocation, error) {
	verdictJSON, err := json.Marshal(verdict)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal planner strict validation verdict: %w", err)
	}
	outcomeJSON, err := json.Marshal(outcome)
	if err != nil {
		return memory.AgentInvocation{}, fmt.Errorf("marshal planner validation outcome: %w", err)
	}
	const query = `
		UPDATE agent_invocations
		SET strict_validation_verdict = $2, validation_outcome = $3
		WHERE id = $1
		RETURNING ` + agentInvocationSelectColumns + `
	`
	return scanAgentInvocation(s.db.QueryRowContext(context.Background(), query, invocationID, verdictJSON, outcomeJSON))
}

func requireAgentProjectTx(ctx context.Context, tx *sql.Tx, projectID string) error {
	var exists bool
	if err := tx.QueryRowContext(ctx, `
		SELECT EXISTS(SELECT 1 FROM projects WHERE id = $1)
	`, projectID).Scan(&exists); err != nil {
		return err
	}
	if !exists {
		return ErrNotFound
	}
	return nil
}

func (s *PostgresStore) ListProjectAgentInvocations(projectID string, filter memory.AgentInvocationFilter) ([]memory.AgentInvocation, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if filter.Limit <= 0 {
		filter.Limit = 25
	}

	const query = `
		SELECT ` + agentInvocationSelectColumns + `
		FROM agent_invocations
		WHERE project_id = $1
			AND ($2 = '' OR dataset_id = $2)
			AND ($3 = '' OR plan_id = $3)
			AND ($4 = '' OR job_id = $4)
			AND ($5 = '' OR agent_name = $5)
			AND ($6 = '' OR planner_variant_id = $6)
		ORDER BY created_at DESC
		LIMIT $7
	`
	rows, err := s.db.QueryContext(
		context.Background(),
		query,
		projectID,
		filter.DatasetID,
		filter.PlanID,
		filter.JobID,
		filter.AgentName,
		filter.PlannerVariantID,
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
