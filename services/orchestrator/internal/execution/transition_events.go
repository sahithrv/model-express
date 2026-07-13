package execution

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// ExecutionTransition identifies a durable, state-changing activity producer.
// Values are storage/API contracts and must not be renamed after release.
type ExecutionTransition string

const (
	TransitionJobQueued               ExecutionTransition = "job.queued"
	TransitionJobAssigned             ExecutionTransition = "job.assigned"
	TransitionJobRunning              ExecutionTransition = "job.running"
	TransitionJobRetryQueued          ExecutionTransition = "job.retry_queued"
	TransitionJobCancelled            ExecutionTransition = "job.cancelled"
	TransitionJobCompleted            ExecutionTransition = "job.completed"
	TransitionJobFailed               ExecutionTransition = "job.failed"
	TransitionJobLeaseRecovered       ExecutionTransition = "job.lease_recovered"
	TransitionAgentValidationRejected ExecutionTransition = "agent.validation_rejected"
	TransitionAgentValidationRetrying ExecutionTransition = "agent.validation_retrying"
	TransitionAgentValidationAccepted ExecutionTransition = "agent.validation_accepted"
	TransitionAgentValidationFailed   ExecutionTransition = "agent.validation_failed"
	TransitionAgentDecisionRecorded   ExecutionTransition = "agent.decision_recorded"
)

const (
	EventJobQueued                = "JOB_QUEUED"
	EventJobAssigned              = "JOB_ASSIGNED"
	EventJobRunning               = "JOB_RUNNING"
	EventJobRetryQueuedTransition = "JOB_RETRY_QUEUED_TRANSITION"
	EventJobCancelled             = "JOB_CANCELLED"
	EventJobCompleted             = "JOB_COMPLETED"
	EventJobFailed                = "JOB_FAILED"
	EventJobLeaseRecovered        = "JOB_LEASE_RECOVERED"
	EventAgentValidationRejected  = "AGENT_VALIDATION_REJECTED"
	EventAgentValidationRetrying  = "AGENT_VALIDATION_RETRYING"
	EventAgentValidationAccepted  = "AGENT_VALIDATION_ACCEPTED"
	EventAgentValidationFailed    = "AGENT_VALIDATION_FAILED"
	EventAgentDecisionRecorded    = "AGENT_DECISION_RECORDED"
)

// ExecutionEventCreate is the validated input consumed by durable event
// writers. Keeping it separate from ExecutionEvent prevents callers from
// supplying IDs, cursors, or timestamps.
type ExecutionEventCreate struct {
	ProjectID      string
	PlanID         string
	EventType      string
	Message        string
	Payload        map[string]any
	IdempotencyKey string
}

// ExecutionTransitionEventInput contains identity only. The event type,
// message, category, phase, status, and severity are derived from Transition.
// In particular, this surface intentionally accepts neither arbitrary metadata
// nor a free-form message.
type ExecutionTransitionEventInput struct {
	Transition   ExecutionTransition
	ProjectID    string
	PlanID       string
	JobID        string
	AttemptID    string
	Attempt      int
	InvocationID string
	AgentName    string
	DecisionID   string
	DecisionType string
	RetryAttempt int
	WillRetry    bool
	ReasonCode   string
}

type executionTransitionDescriptor struct {
	eventType string
	category  string
	phase     string
	status    string
	severity  string
	message   string
	job       bool
	agent     bool
	decision  bool
}

var executionTransitionDescriptors = map[ExecutionTransition]executionTransitionDescriptor{
	TransitionJobQueued:               {eventType: EventJobQueued, category: "job", phase: "queue", status: "waiting", severity: "info", message: "Job queued.", job: true},
	TransitionJobAssigned:             {eventType: EventJobAssigned, category: "job", phase: "assignment", status: "active", severity: "info", message: "Job assigned to a worker.", job: true},
	TransitionJobRunning:              {eventType: EventJobRunning, category: "job", phase: "running", status: "active", severity: "info", message: "Job started running.", job: true},
	TransitionJobRetryQueued:          {eventType: EventJobRetryQueuedTransition, category: "job", phase: "retry", status: "waiting", severity: "warning", message: "Job queued for retry.", job: true},
	TransitionJobCancelled:            {eventType: EventJobCancelled, category: "job", phase: "cancellation", status: "cancelled", severity: "warning", message: "Job cancelled.", job: true},
	TransitionJobCompleted:            {eventType: EventJobCompleted, category: "job", phase: "completion", status: "succeeded", severity: "success", message: "Job completed.", job: true},
	TransitionJobFailed:               {eventType: EventJobFailed, category: "job", phase: "failure", status: "failed", severity: "error", message: "Job failed.", job: true},
	TransitionJobLeaseRecovered:       {eventType: EventJobLeaseRecovered, category: "job", phase: "recovery", status: "waiting", severity: "warning", message: "Expired job lease recovered.", job: true},
	TransitionAgentValidationRejected: {eventType: EventAgentValidationRejected, category: "agent", phase: "validation", status: "blocked", severity: "warning", message: "Agent draft rejected by validation.", agent: true},
	TransitionAgentValidationRetrying: {eventType: EventAgentValidationRetrying, category: "agent", phase: "validation", status: "active", severity: "warning", message: "Agent draft rejected; validation retry scheduled.", agent: true},
	TransitionAgentValidationAccepted: {eventType: EventAgentValidationAccepted, category: "agent", phase: "validation", status: "succeeded", severity: "success", message: "Agent draft accepted by validation.", agent: true},
	TransitionAgentValidationFailed:   {eventType: EventAgentValidationFailed, category: "agent", phase: "validation", status: "failed", severity: "error", message: "Agent validation failed.", agent: true},
	TransitionAgentDecisionRecorded:   {eventType: EventAgentDecisionRecorded, category: "agent", phase: "decision", status: "succeeded", severity: "success", message: "Agent decision recorded.", decision: true},
}

// IsDurableTransitionEventType reports whether eventType belongs to the closed
// PR4 producer contract. Legacy event types intentionally return false.
func IsDurableTransitionEventType(eventType string) bool {
	for _, descriptor := range executionTransitionDescriptors {
		if descriptor.eventType == eventType {
			return true
		}
	}
	return false
}

func DurableTransitionEventTypes() []string {
	types := make([]string, 0, len(executionTransitionDescriptors))
	for _, descriptor := range executionTransitionDescriptors {
		types = append(types, descriptor.eventType)
	}
	sort.Strings(types)
	return types
}

// NewExecutionTransitionEvent builds the only payload shape used by durable
// transition producers. Repeated calls for the same semantic identity return
// the same idempotency key.
func NewExecutionTransitionEvent(input ExecutionTransitionEventInput) (ExecutionEventCreate, error) {
	descriptor, ok := executionTransitionDescriptors[input.Transition]
	if !ok {
		return ExecutionEventCreate{}, fmt.Errorf("unsupported execution transition %q", input.Transition)
	}

	projectID, err := executionTransitionToken("project_id", input.ProjectID, true)
	if err != nil {
		return ExecutionEventCreate{}, err
	}
	planID, err := executionTransitionToken("plan_id", input.PlanID, false)
	if err != nil {
		return ExecutionEventCreate{}, err
	}
	jobID, err := executionTransitionToken("job_id", input.JobID, descriptor.job)
	if err != nil {
		return ExecutionEventCreate{}, err
	}
	attemptID, err := executionTransitionToken("attempt_id", input.AttemptID, false)
	if err != nil {
		return ExecutionEventCreate{}, err
	}
	invocationID, err := executionTransitionToken("invocation_id", input.InvocationID, descriptor.agent)
	if err != nil {
		return ExecutionEventCreate{}, err
	}
	agentName, err := executionTransitionToken("agent_name", input.AgentName, descriptor.agent)
	if err != nil {
		return ExecutionEventCreate{}, err
	}
	decisionID, err := executionTransitionToken("decision_id", input.DecisionID, descriptor.decision)
	if err != nil {
		return ExecutionEventCreate{}, err
	}
	decisionType, err := executionTransitionToken("decision_type", input.DecisionType, descriptor.decision)
	if err != nil {
		return ExecutionEventCreate{}, err
	}
	reasonCode, err := executionTransitionReasonCode(input.ReasonCode)
	if err != nil {
		return ExecutionEventCreate{}, err
	}
	if input.Attempt < 0 {
		return ExecutionEventCreate{}, fmt.Errorf("attempt must be nonnegative")
	}
	if input.RetryAttempt < 0 {
		return ExecutionEventCreate{}, fmt.Errorf("retry_attempt must be nonnegative")
	}

	payload := map[string]any{
		"category": descriptor.category,
		"phase":    descriptor.phase,
		"status":   descriptor.status,
		"severity": descriptor.severity,
	}
	if jobID != "" {
		payload["job_id"] = jobID
		payload["attempt"] = input.Attempt
	}
	if attemptID != "" {
		payload["attempt_id"] = attemptID
	}
	if invocationID != "" {
		payload["invocation_id"] = invocationID
	}
	if agentName != "" {
		payload["agent_name"] = agentName
	}
	if decisionID != "" {
		payload["decision_id"] = decisionID
		payload["decision_type"] = decisionType
	}
	// Validation attempt zero is meaningful (the initial draft) and must remain
	// distinguishable from a missing retry identity.
	if descriptor.agent {
		payload["retry_attempt"] = input.RetryAttempt
	}
	if input.Transition == TransitionAgentValidationRejected || input.Transition == TransitionAgentValidationRetrying {
		payload["will_retry"] = input.WillRetry
	}
	if reasonCode != "" {
		payload["reason_code"] = reasonCode
	}

	identity := []string{string(input.Transition), projectID}
	if descriptor.job {
		// Plan and reason are descriptive metadata, not lifecycle identity. This
		// makes terminal reconciliation collide with the original authoritative
		// completed/failed/cancelled event even if its diagnostic context differs.
		identity = append(identity, jobID, attemptID, strconv.Itoa(input.Attempt))
	} else if descriptor.agent {
		// Invocation plus retry number preserves validation-attempt identity.
		identity = append(identity, invocationID, strconv.Itoa(input.RetryAttempt))
	} else {
		identity = append(identity, decisionID, decisionType)
	}
	return ExecutionEventCreate{
		ProjectID:      projectID,
		PlanID:         planID,
		EventType:      descriptor.eventType,
		Message:        descriptor.message,
		Payload:        payload,
		IdempotencyKey: executionTransitionIdempotencyKey(identity),
	}, nil
}

func executionTransitionToken(field string, value string, required bool) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		if required {
			return "", fmt.Errorf("%s is required", field)
		}
		return "", nil
	}
	if len([]rune(value)) > 160 {
		return "", fmt.Errorf("%s exceeds 160 characters", field)
	}
	for _, char := range value {
		if unicode.IsLetter(char) || unicode.IsDigit(char) {
			continue
		}
		switch char {
		case '_', '-', '.', ':', '@':
			continue
		default:
			return "", fmt.Errorf("%s contains unsafe characters", field)
		}
	}
	return value, nil
}

func executionTransitionReasonCode(value string) (string, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return "", nil
	}
	if len(value) > 64 || value[0] < 'a' || value[0] > 'z' {
		return "", fmt.Errorf("reason_code must be a bounded lowercase identifier")
	}
	for _, char := range value {
		if (char >= 'a' && char <= 'z') || (char >= '0' && char <= '9') || char == '_' {
			continue
		}
		return "", fmt.Errorf("reason_code must be a bounded lowercase identifier")
	}
	return value, nil
}

func executionTransitionIdempotencyKey(fields []string) string {
	var canonical strings.Builder
	for _, field := range fields {
		canonical.WriteString(strconv.Itoa(len(field)))
		canonical.WriteByte(':')
		canonical.WriteString(field)
		canonical.WriteByte('|')
	}
	sum := sha256.Sum256([]byte(canonical.String()))
	return "transition:v1:" + hex.EncodeToString(sum[:])
}
