package store

import (
	"encoding/json"
	"fmt"
	"strings"

	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/memory"
)

func agentDecisionRecordedEvent(decision decisions.AgentDecision) (execution.ExecutionEventCreate, error) {
	return execution.NewExecutionTransitionEvent(execution.ExecutionTransitionEventInput{
		Transition:   execution.TransitionAgentDecisionRecorded,
		ProjectID:    decision.ProjectID,
		PlanID:       decision.PlanID,
		DecisionID:   decision.ID,
		DecisionType: decision.DecisionType,
	})
}

func agentInvocationValidationEvent(invocation memory.AgentInvocation) (execution.ExecutionEventCreate, bool, error) {
	var transition execution.ExecutionTransition
	switch strings.ToLower(strings.TrimSpace(invocation.ValidationStatus)) {
	case memory.InvocationValidationValid:
		transition = execution.TransitionAgentValidationAccepted
	case memory.InvocationValidationInvalid:
		transition = execution.TransitionAgentValidationRejected
	case memory.InvocationValidationFailed:
		transition = execution.TransitionAgentValidationFailed
	default:
		return execution.ExecutionEventCreate{}, false, nil
	}

	create, err := execution.NewExecutionTransitionEvent(execution.ExecutionTransitionEventInput{
		Transition:   transition,
		ProjectID:    invocation.ProjectID,
		PlanID:       invocation.PlanID,
		InvocationID: invocation.ID,
		AgentName:    invocation.AgentName,
		RetryAttempt: 0,
		WillRetry:    false,
	})
	return create, true, err
}

func agentInvocationValidationRetryEvent(invocation memory.AgentInvocation, outcome map[string]any) (execution.ExecutionEventCreate, bool, error) {
	willRetry, _ := outcome["will_retry"].(bool)
	if !willRetry {
		return execution.ExecutionEventCreate{}, false, nil
	}

	sourceAttempt, ok := agentOutcomeInt(outcome["retry_attempt"])
	if !ok {
		sourceAttempt = 0
	}
	if sourceAttempt < 0 {
		return execution.ExecutionEventCreate{}, false, fmt.Errorf("retry_attempt must be nonnegative")
	}

	create, err := execution.NewExecutionTransitionEvent(execution.ExecutionTransitionEventInput{
		Transition:   execution.TransitionAgentValidationRetrying,
		ProjectID:    invocation.ProjectID,
		PlanID:       invocation.PlanID,
		InvocationID: invocation.ID,
		AgentName:    invocation.AgentName,
		// The outcome identifies the rejected source attempt. The durable retry
		// transition identifies the next attempt that was actually scheduled.
		RetryAttempt: sourceAttempt + 1,
		WillRetry:    true,
	})
	return create, true, err
}

func agentOutcomeInt(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return typed, true
	case int32:
		return int(typed), true
	case int64:
		return int(typed), true
	case float64:
		converted := int(typed)
		return converted, float64(converted) == typed
	case json.Number:
		converted, err := typed.Int64()
		return int(converted), err == nil
	default:
		return 0, false
	}
}

func invalidAgentTransitionError(operation string, err error) error {
	return fmt.Errorf("%w: %s: %v", ErrInvalidRequest, operation, err)
}
