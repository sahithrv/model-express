package store

import (
	"encoding/json"
	"errors"
	"reflect"
	"sort"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/memory"
)

func TestMemoryAgentDecisionAndValidationProducersAreDurableAndAllowlisted(t *testing.T) {
	s := NewMemoryStore()
	project, err := s.CreateProject("agent transitions", "")
	if err != nil {
		t.Fatal(err)
	}
	secret := "secret-prompt-output-storage-uri"

	decision, err := s.CreateAgentDecision(project.ID, "plan_1", "WAIT", secret, map[string]any{
		"prompt": secret, "raw_output": secret, "storage_uri": "s3://private/model",
	})
	if err != nil {
		t.Fatal(err)
	}
	invocations := []memory.AgentInvocation{
		{ProjectID: project.ID, PlanID: "plan_1", AgentName: "planner", ValidationStatus: memory.InvocationValidationValid, RawOutput: secret, ValidationError: secret},
		{ProjectID: project.ID, PlanID: "plan_1", AgentName: "planner", ValidationStatus: memory.InvocationValidationInvalid, RawOutput: secret, ValidationError: secret},
		{ProjectID: project.ID, PlanID: "plan_1", AgentName: "planner", ValidationStatus: memory.InvocationValidationFailed, RawOutput: secret, ValidationError: secret},
		{ProjectID: project.ID, PlanID: "plan_1", AgentName: "planner", ValidationStatus: "unknown", RawOutput: secret, ValidationError: secret},
	}
	for _, invocation := range invocations {
		if _, err := s.CreateAgentInvocation(invocation); err != nil {
			t.Fatal(err)
		}
	}

	events := memoryDurableAgentEvents(s)
	if len(events) != 4 {
		t.Fatalf("durable agent event count = %d, want 4: %#v", len(events), events)
	}
	wantTypes := []string{
		execution.EventAgentDecisionRecorded,
		execution.EventAgentValidationAccepted,
		execution.EventAgentValidationRejected,
		execution.EventAgentValidationFailed,
	}
	for index, event := range events {
		if event.EventType != wantTypes[index] {
			t.Fatalf("event %d type = %q, want %q", index, event.EventType, wantTypes[index])
		}
		assertAgentEventAllowlisted(t, event, secret)
	}
	if got := events[0].Payload["decision_id"]; got != decision.ID {
		t.Fatalf("decision event identity = %#v, want %q", got, decision.ID)
	}
}

func TestMemoryAgentValidationRetryProducerIsAttemptScopedAndIdempotent(t *testing.T) {
	s := NewMemoryStore()
	project, _ := s.CreateProject("validation retry", "")
	invocation, err := s.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID: project.ID, PlanID: "plan_1", AgentName: "experiment_planner",
		ValidationStatus: memory.InvocationValidationInvalid,
		ValidationError:  "private rejected output",
	})
	if err != nil {
		t.Fatal(err)
	}

	firstOutcome := map[string]any{
		"will_retry": true, "retry_attempt": 0,
		"backend_validation_error": "private rejected output",
		"raw_output":               "private model response",
	}
	if _, err := s.UpdateAgentInvocationDownstreamOutcome(invocation.ID, firstOutcome); err != nil {
		t.Fatal(err)
	}
	if _, err := s.UpdateAgentInvocationDownstreamOutcome(invocation.ID, firstOutcome); err != nil {
		t.Fatal(err)
	}

	secondOutcome := map[string]any{"will_retry": true, "retry_attempt": float64(1)}
	if _, err := s.UpdateAgentInvocationDownstreamOutcome(invocation.ID, secondOutcome); err != nil {
		t.Fatal(err)
	}

	events := memoryDurableAgentEvents(s)
	if len(events) != 3 {
		t.Fatalf("events after duplicate retry callback = %d, want rejection + two retries: %#v", len(events), events)
	}
	initialAttempt, initialAttemptOK := agentOutcomeInt(events[0].Payload["retry_attempt"])
	if events[0].EventType != execution.EventAgentValidationRejected || !initialAttemptOK || initialAttempt != 0 {
		t.Fatalf("initial rejection lost source attempt identity: %#v", events[0])
	}
	for index, wantAttempt := range []int{1, 2} {
		event := events[index+1]
		retryAttempt, retryAttemptOK := agentOutcomeInt(event.Payload["retry_attempt"])
		if event.EventType != execution.EventAgentValidationRetrying || !retryAttemptOK || retryAttempt != wantAttempt || event.Payload["will_retry"] != true {
			t.Fatalf("retry event %d = %#v, want scheduled attempt %d", index, event, wantAttempt)
		}
		assertAgentEventAllowlisted(t, event, "private")
	}
	if events[1].IdempotencyKey == events[2].IdempotencyKey {
		t.Fatal("different retry attempts shared an idempotency key")
	}
}

func TestMemoryAgentProducerFailureLeavesRowsAndEventsUnchanged(t *testing.T) {
	s := NewMemoryStore()
	project, _ := s.CreateProject("agent rollback", "")

	if _, err := s.CreateAgentDecision(project.ID, "", "unsafe/type", "private", nil); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("unsafe decision type error = %v", err)
	}
	decisions, err := s.ListProjectAgentDecisions(project.ID)
	if err != nil || len(decisions) != 0 {
		t.Fatalf("failed decision producer committed row: rows=%#v err=%v", decisions, err)
	}
	if _, err := s.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID: project.ID, AgentName: "unsafe/agent", ValidationStatus: memory.InvocationValidationValid,
	}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("unsafe invocation identity error = %v", err)
	}
	invocations, err := s.ListProjectAgentInvocations(project.ID, memory.AgentInvocationFilter{})
	if err != nil || len(invocations) != 0 {
		t.Fatalf("failed invocation producer committed row: rows=%#v err=%v", invocations, err)
	}

	invocation, err := s.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID: project.ID, AgentName: "planner", ValidationStatus: "",
		DownstreamOutcome: map[string]any{"existing": "value"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.UpdateAgentInvocationDownstreamOutcome(invocation.ID, map[string]any{
		"will_retry": true, "retry_attempt": -1,
	}); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("invalid retry identity error = %v", err)
	}
	reloaded, err := s.GetAgentInvocation(invocation.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(reloaded.DownstreamOutcome, map[string]any{"existing": "value"}) {
		t.Fatalf("failed retry event committed outcome: %#v", reloaded.DownstreamOutcome)
	}
	if events := memoryDurableAgentEvents(s); len(events) != 0 {
		t.Fatalf("failed producers committed events: %#v", events)
	}
}

func memoryDurableAgentEvents(s *MemoryStore) []execution.ExecutionEvent {
	events := make([]execution.ExecutionEvent, 0, len(s.executionEvents))
	for _, event := range s.executionEvents {
		if execution.IsDurableTransitionEventType(event.EventType) && strings.HasPrefix(event.EventType, "AGENT_") {
			events = append(events, event)
		}
	}
	sort.Slice(events, func(i, j int) bool { return events[i].Sequence < events[j].Sequence })
	return events
}

func assertAgentEventAllowlisted(t *testing.T, event execution.ExecutionEvent, forbidden string) {
	t.Helper()
	allowed := map[string]bool{
		"category": true, "phase": true, "status": true, "severity": true,
		"invocation_id": true, "agent_name": true, "retry_attempt": true,
		"will_retry": true, "decision_id": true, "decision_type": true,
	}
	for key := range event.Payload {
		if !allowed[key] {
			t.Fatalf("event payload included non-allowlisted key %q: %#v", key, event.Payload)
		}
	}
	encoded, err := json.Marshal(event.Payload)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(encoded), forbidden) || strings.Contains(event.Message, forbidden) {
		t.Fatalf("event exposed caller content: message=%q payload=%s", event.Message, encoded)
	}
}
