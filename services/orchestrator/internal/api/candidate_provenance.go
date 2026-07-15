package api

import (
	"encoding/json"
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/store"
)

const candidateProvenanceSchemaPayloadKey = "candidate_provenance_schema_version"

func (s *Server) ensurePlannerCandidateProvenance(decision decisions.AgentDecision) error {
	if decision.DecisionType != decisions.TypeAddExperiments ||
		payloadString(decision.Payload, candidateProvenanceSchemaPayloadKey) != calibration.CandidateProvenanceSchemaVersionV1 {
		return nil
	}
	candidates, err := candidateProvenanceCreatesFromPayload(decision.Payload)
	if err != nil {
		return fmt.Errorf("rebuild candidate provenance for decision %s: %w", decision.ID, err)
	}
	if _, err := s.store.EnsureCandidateProvenance(decision, candidates); err != nil {
		return fmt.Errorf("ensure candidate provenance for decision %s: %w", decision.ID, err)
	}
	return nil
}

func candidateProvenanceCreatesFromPayload(payload map[string]any) ([]calibration.CandidateProvenanceCreate, error) {
	if payloadString(payload, candidateProvenanceSchemaPayloadKey) != calibration.CandidateProvenanceSchemaVersionV1 {
		return nil, fmt.Errorf("%w: decision does not declare candidate provenance schema v1", store.ErrInvalidRequest)
	}
	invocationID := payloadString(payload, "invocation_id")
	variantID := payloadString(payload, "planner_variant_id")
	rolloutCohortID := payloadString(payload, "planner_rollout_cohort_id")
	rolloutPolicyID := payloadString(payload, "planner_rollout_policy_id")
	var candidates []agents.CandidateHypothesis
	if err := decodeCandidateProvenancePayload(payload["candidate_hypotheses"], &candidates); err != nil {
		return nil, fmt.Errorf("decode candidate_hypotheses: %w", err)
	}
	var rankings []agents.CandidateRanking
	if err := decodeCandidateProvenancePayload(payload["candidate_rankings"], &rankings); err != nil {
		return nil, fmt.Errorf("decode candidate_rankings: %w", err)
	}
	var selectionTrace []agents.CandidateSelectionRound
	if err := decodeCandidateProvenancePayload(payload["candidate_selection_trace"], &selectionTrace); err != nil {
		return nil, fmt.Errorf("decode candidate_selection_trace: %w", err)
	}
	var capability execution.PlannerCapabilityCard
	if err := decodeCandidateProvenancePayload(payload["execution_capability_card"], &capability); err != nil {
		return nil, fmt.Errorf("decode execution_capability_card: %w", err)
	}
	if len(candidates) == 0 || len(rankings) != len(candidates) {
		return nil, fmt.Errorf("%w: accepted candidate and ranking counts must match", store.ErrInvalidRequest)
	}
	rankingByIndex := make(map[int]agents.CandidateRanking, len(rankings))
	rankingPositionByIndex := make(map[int]int, len(rankings))
	for position, ranking := range rankings {
		if ranking.CandidateIndex < 0 || ranking.CandidateIndex >= len(candidates) {
			return nil, fmt.Errorf("%w: ranking candidate_index %d is out of range", store.ErrInvalidRequest, ranking.CandidateIndex)
		}
		if _, exists := rankingByIndex[ranking.CandidateIndex]; exists {
			return nil, fmt.Errorf("%w: duplicate ranking candidate_index %d", store.ErrInvalidRequest, ranking.CandidateIndex)
		}
		rankingByIndex[ranking.CandidateIndex] = ranking
		rankingPositionByIndex[ranking.CandidateIndex] = position
	}

	out := make([]calibration.CandidateProvenanceCreate, 0, len(candidates))
	selectedExperimentIndexes := map[int]int{}
	for index, candidate := range candidates {
		ranking, ok := rankingByIndex[index]
		if !ok {
			return nil, fmt.Errorf("%w: candidate %d is missing ranking provenance", store.ErrInvalidRequest, index)
		}
		if candidate.Forecast == nil {
			return nil, fmt.Errorf("%w: candidate %d is missing frozen forecast", store.ErrInvalidRequest, index)
		}
		if candidate.ExpectedMetricImpact != candidate.Forecast.PredictedDelta {
			return nil, fmt.Errorf("%w: candidate %d predicted_delta must equal expected_metric_impact", store.ErrInvalidRequest, index)
		}
		requestedHash, acceptedHash, task, err := candidateExecutionIdentity(candidate.ExperimentConfig, capability)
		if err != nil {
			return nil, fmt.Errorf("candidate %d execution identity: %w", index, err)
		}
		state := calibration.CandidateSelectionUnselected
		if ranking.Selected {
			state = calibration.CandidateSelectionSelected
		} else if ranking.Rejected {
			state = calibration.CandidateSelectionRejected
		}
		mechanism := strings.TrimSpace(ranking.Mechanism)
		if mechanism == "" {
			mechanism = strings.TrimSpace(candidate.Mechanism)
		}
		create := calibration.CandidateProvenanceCreate{
			InvocationID:            invocationID,
			PlannerVariantID:        variantID,
			RolloutCohortID:         rolloutCohortID,
			RolloutPolicyID:         rolloutPolicyID,
			CandidateIndex:          index,
			RequestedConfigHash:     requestedHash,
			AcceptedSpecHash:        acceptedHash,
			Task:                    task,
			Mechanism:               mechanism,
			Forecast:                *candidate.Forecast,
			BaseScore:               ranking.BaseScore,
			SelectionTraceReference: candidateSelectionAuditReference(selectionTrace, rankingPositionByIndex[index], index),
			Selected:                ranking.Selected,
			Rejected:                ranking.Rejected,
			SelectionState:          state,
			SelectedExperimentIndex: cloneIntPointer(ranking.SelectedExperimentIndex),
			OutcomeStatus:           calibration.CandidateOutcomeUnknown,
			Reasons:                 append([]string(nil), ranking.Reasons...),
		}
		if err := calibration.ValidateCandidateProvenanceCreate(create); err != nil {
			return nil, fmt.Errorf("%w: candidate %d provenance: %v", store.ErrInvalidRequest, index, err)
		}
		if create.SelectedExperimentIndex != nil {
			if previousCandidate, exists := selectedExperimentIndexes[*create.SelectedExperimentIndex]; exists {
				return nil, fmt.Errorf("%w: candidates %d and %d map to selected_experiment_index %d", store.ErrInvalidRequest, previousCandidate, index, *create.SelectedExperimentIndex)
			}
			selectedExperimentIndexes[*create.SelectedExperimentIndex] = index
		}
		out = append(out, create)
	}
	for experimentIndex := 0; experimentIndex < len(selectedExperimentIndexes); experimentIndex++ {
		if _, ok := selectedExperimentIndexes[experimentIndex]; !ok {
			return nil, fmt.Errorf("%w: selected_experiment_index values must be contiguous and zero-based", store.ErrInvalidRequest)
		}
	}
	return out, nil
}

func candidateSelectionAuditReference(trace []agents.CandidateSelectionRound, rankingPosition int, candidateIndex int) string {
	for roundIndex, round := range trace {
		for entryIndex, entry := range round.Candidates {
			if entry.CandidateIndex == candidateIndex {
				return fmt.Sprintf("/payload/candidate_selection_trace/%d/candidates/%d", roundIndex, entryIndex)
			}
		}
	}
	return fmt.Sprintf("/payload/candidate_rankings/%d", rankingPosition)
}

func candidateExecutionIdentity(experiment plans.PlannedExperiment, capability execution.PlannerCapabilityCard) (string, string, string, error) {
	if strings.EqualFold(strings.TrimSpace(experiment.Template), jobs.TemplateLabelQualityAudit) {
		requested, err := experiment.RequestedConfig()
		if err != nil {
			return "", "", "", err
		}
		return candidateAuditExecutionIdentity(requested, capability)
	}
	provider := providerForExecutionRunner(capability.Runner)
	spec, err := buildExecutionSpecV1(experiment, provider)
	if err != nil {
		return "", "", "", err
	}
	if capability.Task != "" && spec.Task != capability.Task {
		return "", "", "", fmt.Errorf("candidate task %s does not match frozen capability task %s", spec.Task, capability.Task)
	}
	if capability.Runner != "" && spec.Runner != capability.Runner {
		return "", "", "", fmt.Errorf("candidate runner %s does not match frozen capability runner %s", spec.Runner, capability.Runner)
	}
	return spec.RequestedConfigHash, spec.AcceptedSpecHash, spec.Task, nil
}

func decodeCandidateProvenancePayload(value any, target any) error {
	if value == nil {
		return fmt.Errorf("value is missing")
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return err
	}
	return json.Unmarshal(blob, target)
}

func cloneIntPointer(value *int) *int {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func candidateAuditExecutionIdentity(requested map[string]any, capability execution.PlannerCapabilityCard) (string, string, string, error) {
	requestedHash, err := execution.CanonicalJSONHash(requested)
	if err != nil {
		return "", "", "", err
	}
	task := strings.TrimSpace(capability.Task)
	if task == "" {
		task = "dataset_audit"
	}
	acceptedHash, err := execution.CanonicalJSONHash(map[string]any{
		"schema_version":     "candidate_audit_spec_v1",
		"capability_version": capability.CapabilityVersion,
		"task":               task,
		"runner":             jobs.TemplateLabelQualityAudit,
		"accepted_config":    requested,
	})
	if err != nil {
		return "", "", "", err
	}
	return requestedHash, acceptedHash, task, nil
}
