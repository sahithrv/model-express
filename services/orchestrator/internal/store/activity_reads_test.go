package store

import (
	"encoding/json"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/memory"
)

func TestMemoryStoreAgentInvocationActivityIsSlimAndBounded(t *testing.T) {
	s := NewMemoryStore()
	project, err := s.CreateProject("activity", "")
	if err != nil {
		t.Fatal(err)
	}
	largeMarker := strings.Repeat("private-prompt-output-", 200)
	for index := 0; index < 4; index++ {
		_, err := s.CreateAgentInvocation(memory.AgentInvocation{
			ProjectID:     project.ID,
			AgentName:     "experiment_planner",
			InputMessages: []map[string]string{{"role": "system", "content": largeMarker}},
			InputContext:  map[string]any{"private_context": largeMarker},
			RawOutput:     largeMarker,
			ParsedOutput:  map[string]any{"private_output": largeMarker},
			HumanFeedback: map[string]any{"private_feedback": largeMarker},
			DownstreamOutcome: map[string]any{
				"backend_validation_status": "rejected",
				"backend_validation_error":  "duplicate model",
				"will_retry":                true,
				"retry_attempt":             index,
				"completion_state":          "validation_complete",
				"private_tool_payload":      largeMarker,
				"unrelated_nested_value":    map[string]any{"private": largeMarker},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	rows, err := s.ListProjectAgentInvocationActivity(project.ID, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 {
		t.Fatalf("activity projection count = %d, want 2", len(rows))
	}
	if rows[0].DownstreamOutcome["retry_attempt"] != 3 {
		t.Fatalf("newest invocation missing from bounded projection: %#v", rows)
	}
	if len(rows[0].DownstreamOutcome) != 5 {
		t.Fatalf("downstream projection keys = %#v", rows[0].DownstreamOutcome)
	}
	blob, err := json.Marshal(rows)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(blob), largeMarker[:80]) || strings.Contains(string(blob), "private_tool_payload") {
		t.Fatalf("activity projection included a large private field: %s", blob)
	}
}

func TestMemoryStoreAgentDecisionActivityIsBounded(t *testing.T) {
	s := NewMemoryStore()
	project, _ := s.CreateProject("decisions", "")
	for index := 0; index < 5; index++ {
		if _, err := s.CreateAgentDecision(project.ID, "", "WAIT", "decision", map[string]any{"index": index}); err != nil {
			t.Fatal(err)
		}
	}
	rows, err := s.ListProjectAgentDecisionActivity(project.ID, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 2 || rows[0].Payload["index"] != 4 || rows[1].Payload["index"] != 3 {
		t.Fatalf("decision activity was not newest-first and bounded: %#v", rows)
	}
}

func TestAgentInvocationActivityQueryIsSlimAndBounded(t *testing.T) {
	query := agentInvocationActivitySelectQuery()
	for _, required := range []string{
		"jsonb_build_object",
		"backend_validation_status",
		"backend_validation_error",
		"will_retry",
		"retry_attempt",
		"completion_state",
		"ORDER BY created_at DESC, id DESC",
		"LIMIT $2",
	} {
		if !strings.Contains(query, required) {
			t.Fatalf("invocation activity query omitted %q: %s", required, query)
		}
	}
	for _, forbidden := range []string{"input_messages", "input_context", "raw_output", "parsed_output", "human_feedback"} {
		if strings.Contains(query, forbidden) {
			t.Fatalf("invocation activity query selected %q: %s", forbidden, query)
		}
	}
}

func TestAgentDecisionActivityQueryIsBounded(t *testing.T) {
	query := agentDecisionActivitySelectQuery()
	for _, required := range []string{"WHERE project_id = $1", "ORDER BY created_at DESC, id DESC", "LIMIT $2"} {
		if !strings.Contains(query, required) {
			t.Fatalf("decision activity query omitted %q: %s", required, query)
		}
	}
}
