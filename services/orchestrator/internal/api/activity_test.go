package api

import (
	"encoding/json"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/store"
)

func TestActivityValidationRejectionWithRetryIsSanitized(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("activity demo", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}

	base64Blob := strings.Repeat("A", 120)
	invocation, err := memoryStore.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID: project.ID,
		PlanID:    "plan_3",
		AgentName: agents.ExperimentPlannerAgentName,
		InputMessages: []map[string]string{{
			"role":    "system",
			"content": "hidden planner prompt C:\\Users\\Sahith\\datasets\\private " + base64Blob,
		}},
		InputContext: map[string]any{
			"hidden_tool_payload": map[string]any{"storage_uri": "s3://private-bucket/dataset.zip"},
		},
		RawOutput:        "raw llm response " + base64Blob,
		ParsedOutput:     map[string]any{"full_json": strings.Repeat("{\"oversized\":true}", 20)},
		ValidationStatus: memory.InvocationValidationInvalid,
		ValidationError:  "draft repeated one model too often using file:///tmp/private.png",
	})
	if err != nil {
		t.Fatalf("create invocation: %v", err)
	}
	if _, err := memoryStore.UpdateAgentInvocationDownstreamOutcome(invocation.ID, map[string]any{
		"backend_validation_status": "rejected",
		"backend_validation_error":  "over-focused on one model; check C:\\Users\\Sahith\\datasets\\private and s3://private-bucket/dataset.zip " + base64Blob,
		"retry_attempt":             0,
		"will_retry":                true,
		"rejected_tool_calls": []map[string]any{{
			"name":      "secret_tool",
			"arguments": map[string]any{"local_path": "C:\\Users\\Sahith\\secret"},
		}},
	}); err != nil {
		t.Fatalf("update invocation outcome: %v", err)
	}

	events, err := server.listProjectActivityEvents(project.ID, 10)
	if err != nil {
		t.Fatalf("list activity: %v", err)
	}
	var validationEvent *agentActivityEvent
	for index := range events {
		if events[index].Type == "planner.validation_rejected" {
			validationEvent = &events[index]
			break
		}
	}
	if validationEvent == nil {
		t.Fatalf("expected planner validation activity, got %#v", events)
	}
	if validationEvent.Status != "active" || validationEvent.Severity != "warning" {
		t.Fatalf("expected retrying validation to be active warning, got %#v", validationEvent)
	}
	if got := validationEvent.Metadata["will_retry"]; got != true {
		t.Fatalf("expected will_retry metadata, got %#v", validationEvent.Metadata)
	}
	if !strings.Contains(validationEvent.Title, "retrying") {
		t.Fatalf("expected retry title, got %q", validationEvent.Title)
	}

	blob, err := json.Marshal(events)
	if err != nil {
		t.Fatalf("marshal activity: %v", err)
	}
	body := string(blob)
	for _, forbidden := range []string{
		"hidden planner prompt",
		"raw llm response",
		"hidden_tool_payload",
		"secret_tool",
		"C:\\Users\\Sahith",
		"s3://private-bucket",
		"file:///tmp/private.png",
		base64Blob[:80],
	} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("activity stream leaked %q in %s", forbidden, body)
		}
	}
}

func TestActivityExecutionEventMetadataIsAllowlisted(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("activity metadata", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}

	if _, err := memoryStore.CreateExecutionEvent(project.ID, "plan_9", execution.EventJobsQueued, "Queued jobs from s3://private-bucket/dataset.zip at C:\\Users\\Sahith\\dataset.", map[string]any{
		"job_ids":           []string{"job_1", "job_2"},
		"open_job_count":    2,
		"provider":          "local",
		"dataset_checksum":  strings.Repeat("b", 64),
		"dataset_cache_key": "cache/private",
		"storage_uri":       "s3://private-bucket/dataset.zip",
		"raw_payload":       map[string]any{"local_path": "C:\\Users\\Sahith\\dataset"},
	}); err != nil {
		t.Fatalf("create execution event: %v", err)
	}

	events, err := server.listProjectActivityEvents(project.ID, 10)
	if err != nil {
		t.Fatalf("list activity: %v", err)
	}
	var queuedEvent *agentActivityEvent
	for index := range events {
		if events[index].Type == "plan.queued" {
			queuedEvent = &events[index]
			break
		}
	}
	if queuedEvent == nil {
		t.Fatalf("expected plan queued activity, got %#v", events)
	}
	if _, ok := queuedEvent.Metadata["storage_uri"]; ok {
		t.Fatalf("storage_uri should not be exposed: %#v", queuedEvent.Metadata)
	}
	if _, ok := queuedEvent.Metadata["dataset_checksum"]; ok {
		t.Fatalf("dataset_checksum should not be exposed: %#v", queuedEvent.Metadata)
	}
	if queuedEvent.Metadata["open_job_count"] != 2 {
		t.Fatalf("expected tiny job metadata, got %#v", queuedEvent.Metadata)
	}

	blob, err := json.Marshal(events)
	if err != nil {
		t.Fatalf("marshal activity: %v", err)
	}
	body := string(blob)
	for _, forbidden := range []string{"s3://private-bucket", "C:\\Users\\Sahith", "raw_payload", strings.Repeat("b", 64)} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("activity stream leaked %q in %s", forbidden, body)
		}
	}
}

func TestActivityDispatcherIdleEventIsVisibleAndAllowlisted(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	server := newServer(memoryStore)
	project, err := memoryStore.CreateProject("activity dispatcher", "")
	if err != nil {
		t.Fatalf("create project: %v", err)
	}

	if _, err := memoryStore.CreateExecutionEvent(project.ID, "", execution.EventDispatcherIdleExit, "Dispatcher idle after checking C:\\Users\\Sahith\\private.", map[string]any{
		"dispatcher":            "modal",
		"slot_count":            0,
		"desired_slot_count":    0,
		"registered_slot_count": 2,
		"active_slot_count":     0,
		"idle_seconds":          30.5,
		"idle_exit_seconds":     30,
		"storage_uri":           "s3://private-bucket/dataset.zip",
	}); err != nil {
		t.Fatalf("create dispatcher event: %v", err)
	}

	events, err := server.listProjectActivityEvents(project.ID, 10)
	if err != nil {
		t.Fatalf("list activity: %v", err)
	}
	var dispatcherEvent *agentActivityEvent
	for index := range events {
		if events[index].Type == "dispatcher.idle_exit" {
			dispatcherEvent = &events[index]
			break
		}
	}
	if dispatcherEvent == nil {
		t.Fatalf("expected dispatcher idle activity, got %#v", events)
	}
	if dispatcherEvent.Status != "succeeded" || dispatcherEvent.Severity != "success" {
		t.Fatalf("expected succeeded dispatcher idle event, got %#v", dispatcherEvent)
	}
	if dispatcherEvent.Metadata["slot_count"] != 0 {
		t.Fatalf("expected dispatcher slot metadata, got %#v", dispatcherEvent.Metadata)
	}
	if _, ok := dispatcherEvent.Metadata["storage_uri"]; ok {
		t.Fatalf("storage_uri should not be exposed: %#v", dispatcherEvent.Metadata)
	}
	blob, err := json.Marshal(events)
	if err != nil {
		t.Fatalf("marshal activity: %v", err)
	}
	body := string(blob)
	for _, forbidden := range []string{"C:\\Users\\Sahith", "s3://private-bucket"} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("activity stream leaked %q in %s", forbidden, body)
		}
	}
}
