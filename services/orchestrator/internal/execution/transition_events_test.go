package execution

import (
	"reflect"
	"strings"
	"testing"
)

func TestNewExecutionTransitionEventIsTypedBoundedAndDeterministic(t *testing.T) {
	input := ExecutionTransitionEventInput{
		Transition: TransitionJobRetryQueued,
		ProjectID:  "project_1",
		PlanID:     "plan_2",
		JobID:      "job_3",
		AttemptID:  "job_3:attempt-2",
		Attempt:    2,
		ReasonCode: "lease_expired",
	}
	first, err := NewExecutionTransitionEvent(input)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewExecutionTransitionEvent(input)
	if err != nil {
		t.Fatal(err)
	}
	if first.IdempotencyKey != second.IdempotencyKey {
		t.Fatalf("idempotency key changed: %q != %q", first.IdempotencyKey, second.IdempotencyKey)
	}
	if len(first.IdempotencyKey) > 200 || !strings.HasPrefix(first.IdempotencyKey, "transition:v1:") {
		t.Fatalf("unexpected idempotency key %q", first.IdempotencyKey)
	}
	if first.EventType != EventJobRetryQueuedTransition || first.Message != "Job queued for retry." {
		t.Fatalf("unexpected event: %#v", first)
	}
	wantPayload := map[string]any{
		"category":    "job",
		"phase":       "retry",
		"status":      "waiting",
		"severity":    "warning",
		"job_id":      "job_3",
		"attempt_id":  "job_3:attempt-2",
		"attempt":     2,
		"reason_code": "lease_expired",
	}
	if !reflect.DeepEqual(first.Payload, wantPayload) {
		t.Fatalf("unexpected allowlisted payload: %#v", first.Payload)
	}
}

func TestExecutionTransitionIdempotencyScopesAttemptsAndRetries(t *testing.T) {
	base := ExecutionTransitionEventInput{Transition: TransitionJobAssigned, ProjectID: "project_1", JobID: "job_1", AttemptID: "job_1:attempt-1", Attempt: 1}
	first, err := NewExecutionTransitionEvent(base)
	if err != nil {
		t.Fatal(err)
	}
	nextAttempt := base
	nextAttempt.Attempt = 2
	nextAttempt.AttemptID = "job_1:attempt-2"
	second, err := NewExecutionTransitionEvent(nextAttempt)
	if err != nil {
		t.Fatal(err)
	}
	if first.IdempotencyKey == second.IdempotencyKey {
		t.Fatal("different attempts shared an idempotency key")
	}
	changedReason := base
	changedReason.PlanID = "plan_diagnostic_context_changed"
	changedReason.ReasonCode = "lease_expired"
	sameTransition, err := NewExecutionTransitionEvent(changedReason)
	if err != nil {
		t.Fatal(err)
	}
	if first.IdempotencyKey != sameTransition.IdempotencyKey {
		t.Fatal("diagnostic plan/reason changed semantic transition identity")
	}

	retry := ExecutionTransitionEventInput{Transition: TransitionAgentValidationRetrying, ProjectID: "project_1", InvocationID: "invocation_1", AgentName: "experiment_planner", RetryAttempt: 1, WillRetry: true}
	retryOne, err := NewExecutionTransitionEvent(retry)
	if err != nil {
		t.Fatal(err)
	}
	retry.RetryAttempt = 2
	retryTwo, err := NewExecutionTransitionEvent(retry)
	if err != nil {
		t.Fatal(err)
	}
	if retryOne.IdempotencyKey == retryTwo.IdempotencyKey {
		t.Fatal("validation retry attempts shared an idempotency key")
	}
	if retryOne.Payload["retry_attempt"] != 1 || retryOne.Payload["will_retry"] != true {
		t.Fatalf("retry identity missing from payload: %#v", retryOne.Payload)
	}
	retry.AgentName = "renamed_planner"
	retrySameAttempt, err := NewExecutionTransitionEvent(retry)
	if err != nil {
		t.Fatal(err)
	}
	if retryTwo.IdempotencyKey != retrySameAttempt.IdempotencyKey {
		t.Fatal("agent display name changed validation attempt identity")
	}
}

func TestNewExecutionTransitionEventRejectsUnboundedOrUnsafeInput(t *testing.T) {
	tests := []struct {
		name  string
		input ExecutionTransitionEventInput
	}{
		{name: "unsupported transition", input: ExecutionTransitionEventInput{Transition: "job.secret", ProjectID: "project_1", JobID: "job_1"}},
		{name: "missing project", input: ExecutionTransitionEventInput{Transition: TransitionJobQueued, JobID: "job_1"}},
		{name: "missing job", input: ExecutionTransitionEventInput{Transition: TransitionJobQueued, ProjectID: "project_1"}},
		{name: "missing invocation", input: ExecutionTransitionEventInput{Transition: TransitionAgentValidationAccepted, ProjectID: "project_1", AgentName: "planner"}},
		{name: "missing agent", input: ExecutionTransitionEventInput{Transition: TransitionAgentValidationAccepted, ProjectID: "project_1", InvocationID: "invocation_1"}},
		{name: "missing decision", input: ExecutionTransitionEventInput{Transition: TransitionAgentDecisionRecorded, ProjectID: "project_1", DecisionType: "WAIT"}},
		{name: "missing decision type", input: ExecutionTransitionEventInput{Transition: TransitionAgentDecisionRecorded, ProjectID: "project_1", DecisionID: "decision_1"}},
		{name: "negative attempt", input: ExecutionTransitionEventInput{Transition: TransitionJobAssigned, ProjectID: "project_1", JobID: "job_1", Attempt: -1}},
		{name: "uri", input: ExecutionTransitionEventInput{Transition: TransitionJobFailed, ProjectID: "project_1", JobID: "s3://private/job"}},
		{name: "path", input: ExecutionTransitionEventInput{Transition: TransitionJobFailed, ProjectID: "project_1", JobID: "/tmp/private"}},
		{name: "secret-shaped free text", input: ExecutionTransitionEventInput{Transition: TransitionJobFailed, ProjectID: "project_1", JobID: "job_1", ReasonCode: "failed with sk-secret-secret-secret"}},
		{name: "secret-shaped token", input: ExecutionTransitionEventInput{Transition: TransitionJobFailed, ProjectID: "project_1", JobID: "job_1", ReasonCode: "sk-secret-secret-secret"}},
		{name: "too long", input: ExecutionTransitionEventInput{Transition: TransitionJobFailed, ProjectID: "project_1", JobID: "job_1", ReasonCode: strings.Repeat("a", 161)}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewExecutionTransitionEvent(test.input); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestExecutionTransitionDescriptorsNeverExposeCallerContent(t *testing.T) {
	inputs := []ExecutionTransitionEventInput{
		{Transition: TransitionJobQueued, ProjectID: "project_1", JobID: "job_1"},
		{Transition: TransitionJobAssigned, ProjectID: "project_1", JobID: "job_1", Attempt: 1},
		{Transition: TransitionJobRunning, ProjectID: "project_1", JobID: "job_1", Attempt: 1},
		{Transition: TransitionJobRetryQueued, ProjectID: "project_1", JobID: "job_1", Attempt: 1},
		{Transition: TransitionJobCancelled, ProjectID: "project_1", JobID: "job_1", Attempt: 1},
		{Transition: TransitionJobCompleted, ProjectID: "project_1", JobID: "job_1", Attempt: 1},
		{Transition: TransitionJobFailed, ProjectID: "project_1", JobID: "job_1", Attempt: 1},
		{Transition: TransitionJobLeaseRecovered, ProjectID: "project_1", JobID: "job_1", Attempt: 1},
		{Transition: TransitionAgentValidationRejected, ProjectID: "project_1", InvocationID: "invocation_1", AgentName: "planner"},
		{Transition: TransitionAgentValidationRetrying, ProjectID: "project_1", InvocationID: "invocation_1", AgentName: "planner", RetryAttempt: 1, WillRetry: true},
		{Transition: TransitionAgentValidationAccepted, ProjectID: "project_1", InvocationID: "invocation_1", AgentName: "planner"},
		{Transition: TransitionAgentValidationFailed, ProjectID: "project_1", InvocationID: "invocation_1", AgentName: "planner"},
		{Transition: TransitionAgentDecisionRecorded, ProjectID: "project_1", DecisionID: "decision_1", DecisionType: "WAIT"},
	}
	allowed := map[string]bool{
		"category": true, "phase": true, "status": true, "severity": true,
		"job_id": true, "attempt_id": true, "attempt": true,
		"invocation_id": true, "agent_name": true, "retry_attempt": true,
		"decision_id": true, "decision_type": true,
		"will_retry": true, "reason_code": true,
	}
	streamAllowed := map[string]bool{}
	for _, key := range SafeExecutionEventMetadataKeys() {
		streamAllowed[key] = true
	}
	for _, input := range inputs {
		event, err := NewExecutionTransitionEvent(input)
		if err != nil {
			t.Fatalf("%s: %v", input.Transition, err)
		}
		if len([]rune(event.Message)) > 220 {
			t.Fatalf("%s message exceeds bound", input.Transition)
		}
		for key := range event.Payload {
			if !allowed[key] {
				t.Fatalf("%s emitted disallowed metadata key %q", input.Transition, key)
			}
			if !streamAllowed[key] {
				t.Fatalf("%s emitted metadata key %q that v2 would discard", input.Transition, key)
			}
		}
	}
}

func TestIsDurableTransitionEventTypeExcludesLegacyTypes(t *testing.T) {
	for _, eventType := range []string{
		EventJobQueued,
		EventJobAssigned,
		EventJobRunning,
		EventJobRetryQueuedTransition,
		EventJobCancelled,
		EventJobCompleted,
		EventJobFailed,
		EventJobLeaseRecovered,
		EventAgentValidationRejected,
		EventAgentValidationRetrying,
		EventAgentValidationAccepted,
		EventAgentValidationFailed,
		EventAgentDecisionRecorded,
		EventJobProgressBoundary,
	} {
		if !IsDurableTransitionEventType(eventType) {
			t.Fatalf("new event type %q was not recognized", eventType)
		}
	}
	for _, legacy := range []string{EventJobRetryQueued, EventExecutionCancelled, EventJobsQueued, ""} {
		if IsDurableTransitionEventType(legacy) {
			t.Fatalf("legacy event type %q was marked durable", legacy)
		}
	}
}

func TestJobProgressBoundaryEventIsSafeAndRevisionIdempotent(t *testing.T) {
	current, total := int64(2), int64(5)
	input := JobProgressBoundaryEventInput{
		ProjectID: "project_1", PlanID: "plan_1", JobID: "job_1",
		AttemptID: "job_1:attempt-1", Attempt: 1, TaxonomyVersion: 1,
		Stage: "training", DetailCode: "epoch.complete", Status: "running",
		Current: &current, Total: &total, Unit: "epoch", Revision: 9,
	}
	first, err := NewJobProgressBoundaryEvent(input)
	if err != nil {
		t.Fatal(err)
	}
	second, err := NewJobProgressBoundaryEvent(input)
	if err != nil {
		t.Fatal(err)
	}
	if first.IdempotencyKey != second.IdempotencyKey || first.EventType != EventJobProgressBoundary {
		t.Fatalf("boundary event is not stable: %#v %#v", first, second)
	}
	allowed := map[string]bool{}
	for _, key := range SafeExecutionEventMetadataKeys() {
		allowed[key] = true
	}
	for key := range first.Payload {
		if !allowed[key] {
			t.Fatalf("progress event key %q is not stream allowlisted: %#v", key, first.Payload)
		}
	}
	if first.Payload["current"] != int64(2) || first.Payload["total"] != int64(5) || first.Payload["revision"] != int64(9) {
		t.Fatalf("progress boundary payload=%#v", first.Payload)
	}
}

func TestAgentDecisionTransitionPreservesTypedIdentity(t *testing.T) {
	input := ExecutionTransitionEventInput{
		Transition:   TransitionAgentDecisionRecorded,
		ProjectID:    "project_1",
		PlanID:       "plan_1",
		DecisionID:   "decision_1",
		DecisionType: "SELECT_CHAMPION",
	}
	event, err := NewExecutionTransitionEvent(input)
	if err != nil {
		t.Fatal(err)
	}
	if event.EventType != EventAgentDecisionRecorded || event.Payload["decision_id"] != "decision_1" || event.Payload["decision_type"] != "SELECT_CHAMPION" {
		t.Fatalf("decision identity missing: %#v", event)
	}
	changed := input
	changed.DecisionType = "WAIT"
	other, err := NewExecutionTransitionEvent(changed)
	if err != nil {
		t.Fatal(err)
	}
	if event.IdempotencyKey == other.IdempotencyKey {
		t.Fatal("different decision transitions shared an idempotency key")
	}
}
