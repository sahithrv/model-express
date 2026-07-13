package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/agents"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/store"
)

type activityReadSpyStore struct {
	store.Store
	invocationActivityCalls int
	decisionActivityCalls   int
	legacyInvocationCalls   int
	legacyDecisionCalls     int
	invocationLimit         int
	decisionLimit           int
}

func (s *activityReadSpyStore) ListProjectAgentInvocationActivity(projectID string, limit int) ([]memory.AgentInvocationActivity, error) {
	s.invocationActivityCalls++
	s.invocationLimit = limit
	return s.Store.ListProjectAgentInvocationActivity(projectID, limit)
}

func (s *activityReadSpyStore) ListProjectAgentDecisionActivity(projectID string, limit int) ([]decisions.AgentDecision, error) {
	s.decisionActivityCalls++
	s.decisionLimit = limit
	return s.Store.ListProjectAgentDecisionActivity(projectID, limit)
}

func (s *activityReadSpyStore) ListProjectAgentInvocations(projectID string, filter memory.AgentInvocationFilter) ([]memory.AgentInvocation, error) {
	s.legacyInvocationCalls++
	return nil, errors.New("legacy invocation activity read used")
}

func (s *activityReadSpyStore) ListProjectAgentDecisions(projectID string) ([]decisions.AgentDecision, error) {
	s.legacyDecisionCalls++
	return nil, errors.New("legacy decision activity read used")
}

func TestActivityUsesOnlyBoundedProjectionReads(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("projection spy", "")
	invocation, err := memoryStore.CreateAgentInvocation(memory.AgentInvocation{
		ProjectID:        project.ID,
		AgentName:        agents.ExperimentPlannerAgentName,
		ValidationStatus: memory.InvocationValidationInvalid,
		ValidationError:  "duplicate proposal",
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := memoryStore.UpdateAgentInvocationDownstreamOutcome(invocation.ID, map[string]any{
		"backend_validation_status": "rejected",
		"backend_validation_error":  "duplicate proposal",
		"will_retry":                true,
		"retry_attempt":             1,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := memoryStore.CreateAgentDecision(project.ID, "", decisions.TypeWait, "wait", nil); err != nil {
		t.Fatal(err)
	}

	spy := &activityReadSpyStore{Store: memoryStore}
	server := newServer(spy)
	events, stats, err := server.listProjectActivityEventsWithStats(project.ID, 10)
	if err != nil {
		t.Fatal(err)
	}
	if spy.invocationActivityCalls != 1 || spy.decisionActivityCalls != 1 || spy.legacyInvocationCalls != 0 || spy.legacyDecisionCalls != 0 {
		t.Fatalf("unexpected activity read calls: %#v", spy)
	}
	if spy.invocationLimit != 50 || spy.decisionLimit != 50 || stats.StoreCallCount != 4 {
		t.Fatalf("activity bounds/stats = invocation %d decision %d stats %#v", spy.invocationLimit, spy.decisionLimit, stats)
	}
	foundRetry := false
	for _, event := range events {
		if event.Type == "planner.validation_rejected" && event.Metadata["will_retry"] == true {
			foundRetry = true
		}
	}
	if !foundRetry {
		t.Fatalf("bounded projection lost retry activity: %#v", events)
	}
}

func TestV1ActivityIgnoresDualWrittenDurableTransitions(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("dual write compatibility", "")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := memoryStore.CreateExecutionTransition(execution.ExecutionTransitionEventInput{
		Transition: execution.TransitionJobAssigned,
		ProjectID:  project.ID,
		JobID:      "job_1",
		AttemptID:  "job_1:attempt-1",
		Attempt:    1,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := memoryStore.CreateExecutionEvent(project.ID, "", execution.EventWorkersActive, "Workers active.", nil); err != nil {
		t.Fatal(err)
	}

	events, err := newServer(memoryStore).listProjectActivityEvents(project.ID, 10)
	if err != nil {
		t.Fatal(err)
	}
	foundLegacy := false
	for _, event := range events {
		if event.Type == "system.event" && event.Message == "Job assigned to a worker." {
			t.Fatalf("durable transition leaked into v1 as a duplicate: %#v", events)
		}
		if event.Type == "workers.active" {
			foundLegacy = true
		}
	}
	if !foundLegacy {
		t.Fatalf("legacy execution events were filtered with durable transitions: %#v", events)
	}
}

func TestV1ActivityPreservesRetryAndLegacyEventsAcrossDurableBursts(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("v1 retry compatibility", "")
	dataset, _ := memoryStore.CreateDataset(project.ID, "dataset", "memory://dataset", "checksum", 1)
	worker, _ := memoryStore.RegisterWorker(project.ID, "worker", "gpu")
	job, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := memoryStore.PollJob(worker.ID, store.JobPollFilter{}); err != nil {
		t.Fatal(err)
	}
	if _, requeued, err := memoryStore.RetryJob(job.ID, "retry", store.RetryJobOptions{}); err != nil || !requeued {
		t.Fatalf("retry requeued=%t err=%v", requeued, err)
	}
	if _, requeued, err := memoryStore.RetryJob(job.ID, "attempts exhausted", store.RetryJobOptions{ForceFail: true}); err != nil || requeued {
		t.Fatalf("retry exhaustion requeued=%t err=%v", requeued, err)
	}
	if _, err := memoryStore.CreateExecutionEvent(project.ID, "", execution.EventWorkersActive, "Workers active.", nil); err != nil {
		t.Fatal(err)
	}
	for index := 0; index < 80; index++ {
		if _, _, err := memoryStore.CreateExecutionTransition(execution.ExecutionTransitionEventInput{
			Transition: execution.TransitionJobAssigned,
			ProjectID:  project.ID,
			JobID:      "burst_job_" + strconv.Itoa(index),
			Attempt:    1,
		}); err != nil {
			t.Fatal(err)
		}
	}

	events, err := newServer(memoryStore).listProjectActivityEvents(project.ID, 20)
	if err != nil {
		t.Fatal(err)
	}
	foundRetry := false
	foundLegacy := false
	foundFailure := false
	for _, event := range events {
		foundRetry = foundRetry || event.Type == "job.retrying"
		foundLegacy = foundLegacy || event.Type == "workers.active"
		foundFailure = foundFailure || event.Type == "system.failed"
	}
	if !foundRetry || !foundLegacy || !foundFailure {
		t.Fatalf("v1 activity lost retry/legacy events across durable burst: %#v", events)
	}
}

func TestV1ActivityProducerCoverage(t *testing.T) {
	// Execution-event projections already originate in the durable log. Job,
	// invocation-validation, and decision projections are covered by the typed
	// PR4 producers. Only the rolling count summary is intentionally derived
	// current state rather than a transition.
	coverage := map[string]string{
		"system.event":                     "existing durable execution-event row",
		"planner.started":                  "existing durable execution-event row",
		"planner.decision_recorded":        "typed agent-decision transition",
		"champion.decision_recorded":       "typed agent-decision transition",
		"planner.stopped":                  "typed agent-decision transition",
		"planner.waiting":                  "typed agent-decision transition",
		"agent.outcome_recorded":           "existing durable execution-event row",
		"planner.blocked":                  "existing durable execution-event row",
		"planner.validation_failed":        "typed agent-validation transition",
		"planner.validation_rejected":      "typed agent-validation transition",
		"agent.validation_rejected":        "typed agent-validation transition",
		"agent.failed":                     "existing durable execution-event row",
		"plan.queued":                      "existing durable execution-event row",
		"workers.required":                 "existing durable execution-event row",
		"workers.starting":                 "existing durable execution-event row",
		"workers.active":                   "existing durable execution-event row",
		"dispatcher.status":                "existing durable execution-event row",
		"dispatcher.idle_exit":             "existing durable execution-event row",
		"job.retrying":                     "typed job retry transition",
		"job.queued":                       "typed job queue transition",
		"job.running":                      "typed job assignment/running transition",
		"job.completed":                    "typed job completion transition",
		"job.failed":                       "typed job failure transition",
		"system.failed":                    "existing durable execution-event row",
		"memory.retrieval_logged":          "existing durable execution-event row",
		"champion.selected":                "existing durable execution-event row",
		"champion.feedback_recorded":       "existing durable execution-event row",
		"dataset.visual_analysis_queued":   "existing durable execution-event row",
		"dataset.visual_analysis_recorded": "existing durable execution-event row",
		"project.reopened":                 "typed decision or existing durable execution-event row",
		"jobs.status_counts":               "documented derived aggregate; not a transition",
	}

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve activity test source path")
	}
	source, err := os.ReadFile(filepath.Join(filepath.Dir(filename), "activity.go"))
	if err != nil {
		t.Fatal(err)
	}
	pattern := regexp.MustCompile(`(?:activity\.Type|eventType)\s*(?::=|=)\s*"([^"]+)"|Type:\s*"([^"]+)"`)
	discovered := map[string]bool{}
	for _, match := range pattern.FindAllStringSubmatch(string(source), -1) {
		activityType := match[1]
		if activityType == "" {
			activityType = match[2]
		}
		discovered[activityType] = true
	}
	for activityType := range discovered {
		if strings.TrimSpace(coverage[activityType]) == "" {
			t.Errorf("v1 activity type %q has no durable producer or documented exclusion", activityType)
		}
	}
	if coverage["jobs.status_counts"] == "" || !strings.Contains(coverage["jobs.status_counts"], "not a transition") {
		t.Fatal("derived jobs.status_counts exclusion must remain explicit")
	}
}

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

func TestActivityMetadataKeepsLegacyErrorAliasesOnly(t *testing.T) {
	metadata := activityMetadataFromPayload(map[string]any{
		"backend_validation_error": "invalid draft at s3://private/data",
		"error":                    "worker failed at /tmp/private/model.bin",
		"last_error":               "older error",
	})
	for _, rawKey := range []string{"backend_validation_error", "error", "last_error"} {
		if _, ok := metadata[rawKey]; ok {
			t.Fatalf("v1 metadata exposed new raw key %q: %#v", rawKey, metadata)
		}
	}
	if metadata["validation_error"] == nil || metadata["error_summary"] == nil {
		t.Fatalf("legacy error aliases missing: %#v", metadata)
	}
	blob, _ := json.Marshal(metadata)
	if strings.Contains(string(blob), "s3://") || strings.Contains(string(blob), "/tmp/private") {
		t.Fatalf("legacy error aliases were not sanitized: %s", blob)
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

func TestActivityTickDiagnosticsContainOnlyBoundedMetrics(t *testing.T) {
	gin.SetMode(gin.TestMode)
	logDir := t.TempDir()
	t.Setenv("MODEL_EXPRESS_LOG_DIR", logDir)
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("diagnostics", "")
	if _, err := memoryStore.CreateExecutionEvent(project.ID, "", execution.EventJobsQueued, "safe event", map[string]any{
		"open_job_count": 1,
		"storage_uri":    "s3://private-bucket/secret.zip",
	}); err != nil {
		t.Fatal(err)
	}

	router := NewRouter(memoryStore)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/activity-stream?limit=10", nil).WithContext(ctx)
	response := httptest.NewRecorder()
	router.ServeHTTP(response, req)

	body, err := os.ReadFile(filepath.Join(logDir, "orchestrator.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	var tick map[string]any
	for _, line := range strings.Split(strings.TrimSpace(string(body)), "\n") {
		var record map[string]any
		if json.Unmarshal([]byte(line), &record) == nil && record["event"] == "activity_stream_tick" {
			tick = record
		}
	}
	if tick == nil {
		t.Fatalf("activity tick diagnostic missing: %s", body)
	}
	if tick["store_call_count"] != float64(4) || tick["events_returned"] != float64(1) || tick["response_bytes"].(float64) <= 0 {
		t.Fatalf("activity tick metrics = %#v", tick)
	}
	for _, forbidden := range []string{"s3://private-bucket", "storage_uri", "safe event", project.ID} {
		if strings.Contains(string(body), forbidden) {
			t.Fatalf("diagnostics leaked %q: %s", forbidden, body)
		}
	}
}

type cursorStateStore struct {
	store.Store
	state      execution.ExecutionEventCursorState
	afterCalls int
}

func (s *cursorStateStore) GetExecutionEventCursorState(context.Context) (execution.ExecutionEventCursorState, error) {
	return s.state, nil
}

func (s *cursorStateStore) ListProjectExecutionEventsAfter(ctx context.Context, projectID string, cursor int64, limit int) ([]execution.ExecutionEvent, error) {
	s.afterCalls++
	return s.Store.ListProjectExecutionEventsAfter(ctx, projectID, cursor, limit)
}

func TestExecutionEventV2RouteFlagAndCursorRecovery(t *testing.T) {
	gin.SetMode(gin.TestMode)
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("v2", "")
	for index := 0; index < 5; index++ {
		_, _ = memoryStore.CreateExecutionEvent(project.ID, "", "EVENT", "event", map[string]any{"index": index})
	}

	t.Setenv("MODEL_EXPRESS_ACTIVITY_STREAM_V2_ENABLED", "false")
	disabled := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(disabled, httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/events/stream/v2?cursor=bad", nil))
	if disabled.Code != http.StatusNotFound {
		t.Fatalf("disabled v2 route status = %d, want 404", disabled.Code)
	}

	t.Setenv("MODEL_EXPRESS_ACTIVITY_STREAM_V2_ENABLED", "true")
	invalid := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(invalid, httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/events/stream/v2?cursor=bad", nil))
	if invalid.Code != http.StatusBadRequest || !strings.Contains(invalid.Body.String(), "invalid_cursor") {
		t.Fatalf("invalid cursor response = %d %s", invalid.Code, invalid.Body.String())
	}

	ahead := httptest.NewRecorder()
	NewRouter(memoryStore).ServeHTTP(ahead, httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/events/stream/v2?cursor=99", nil))
	if ahead.Code != http.StatusConflict || !strings.Contains(ahead.Body.String(), "cursor_ahead") {
		t.Fatalf("ahead cursor response = %d %s", ahead.Code, ahead.Body.String())
	}

	floorStore := &cursorStateStore{Store: memoryStore, state: execution.ExecutionEventCursorState{
		LastSequence:          5,
		RetainedSequenceFloor: 3,
	}}
	tooOld := httptest.NewRecorder()
	NewRouter(floorStore).ServeHTTP(tooOld, httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/events/stream/v2?cursor=1", nil))
	if tooOld.Code != http.StatusGone || !strings.Contains(tooOld.Body.String(), "cursor_too_old") || !strings.Contains(tooOld.Body.String(), "resync") {
		t.Fatalf("too-old cursor response = %d %s", tooOld.Code, tooOld.Body.String())
	}
}

func TestExecutionEventV2BurstReconnectHasNoGapOrDuplicate(t *testing.T) {
	gin.SetMode(gin.TestMode)
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("burst", "")
	for index := 0; index < 7; index++ {
		_, _ = memoryStore.CreateExecutionEvent(project.ID, "", "EVENT", "event", map[string]any{"attempt": index})
	}
	server := &Server{store: memoryStore}

	response := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(response)
	c.Request = httptest.NewRequest(http.MethodGet, "/", nil)
	cursor, more, err := server.writeProjectExecutionEventV2Pages(c, project.ID, 0, 2, executionEventV2MaxPages)
	if err != nil || more || cursor != 7 {
		t.Fatalf("burst catch-up cursor=%d more=%v err=%v", cursor, more, err)
	}
	stream := response.Body.String()
	for sequence := 1; sequence <= 7; sequence++ {
		if strings.Count(stream, "id: "+strconv.Itoa(sequence)+"\n") != 1 {
			t.Fatalf("sequence %d missing or duplicated in %s", sequence, stream)
		}
	}

	reconnect := httptest.NewRecorder()
	reconnectContext, _ := gin.CreateTestContext(reconnect)
	reconnectContext.Request = httptest.NewRequest(http.MethodGet, "/", nil)
	cursor, more, err = server.writeProjectExecutionEventV2Pages(reconnectContext, project.ID, 4, 2, executionEventV2MaxPages)
	if err != nil || more || cursor != 7 {
		t.Fatalf("reconnect cursor=%d more=%v err=%v", cursor, more, err)
	}
	if strings.Contains(reconnect.Body.String(), "id: 4\n") || strings.Count(reconnect.Body.String(), "id: 5\n") != 1 {
		t.Fatalf("reconnect duplicated or skipped a cursor: %s", reconnect.Body.String())
	}
}

func TestExecutionEventV2ProjectionSanitizesHistoricalRows(t *testing.T) {
	base64Blob := strings.Repeat("A", 120)
	projected := executionEventV2Projection(execution.ExecutionEvent{
		ID:        "execution_event_1",
		ProjectID: "project_1",
		EventType: execution.EventJobsQueued,
		Message:   "read s3://private-bucket/data.zip at C:\\Users\\Private\\data " + base64Blob,
		Payload: map[string]any{
			"job_id":      "job_1",
			"storage_uri": "s3://private-bucket/data.zip",
			"local_path":  "/Users/private/data",
			"raw_payload": map[string]any{"secret": "do-not-emit"},
			"reason": []any{
				map[string]any{"raw_output": "nested-confidential-text"},
				"safe scalar",
			},
		},
		CreatedAt: time.Now().UTC(),
		Sequence:  9,
	})
	blob, err := json.Marshal(projected)
	if err != nil {
		t.Fatal(err)
	}
	body := string(blob)
	for _, forbidden := range []string{"s3://private-bucket", "C:\\Users\\Private", base64Blob[:80], "storage_uri", "local_path", "raw_payload", "do-not-emit", "raw_output", "nested-confidential-text", `"payload"`} {
		if strings.Contains(body, forbidden) {
			t.Fatalf("v2 projection leaked %q: %s", forbidden, body)
		}
	}
	if !strings.Contains(body, `"job_id":"job_1"`) || !strings.Contains(body, `"sequence":9`) {
		t.Fatalf("v2 projection lost safe identity: %s", body)
	}
}

type blockingCursorStore struct {
	store.Store
	started chan struct{}
}

type blockingProjectValidationStore struct {
	store.Store
	started chan struct{}
}

func (s *blockingProjectValidationStore) GetProjectContext(ctx context.Context, _ string) (projects.Project, error) {
	close(s.started)
	<-ctx.Done()
	return projects.Project{}, ctx.Err()
}

func TestExecutionEventV2InitialValidationHonorsCancellation(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("cancel validation", "")
	blocking := &blockingProjectValidationStore{Store: memoryStore, started: make(chan struct{})}
	server := &Server{store: blocking}
	ctx, cancel := context.WithCancel(context.Background())
	response := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(response)
	c.Params = []gin.Param{{Key: "id", Value: project.ID}}
	c.Request = httptest.NewRequest(http.MethodGet, "/", nil).WithContext(ctx)
	done := make(chan struct{})
	go func() {
		server.streamProjectExecutionEventsV2(c)
		close(done)
	}()
	<-blocking.started
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("initial v2 validation did not observe request cancellation")
	}
	if response.Body.Len() != 0 {
		t.Fatalf("cancelled validation wrote a response: %q", response.Body.String())
	}
}

func (s *blockingCursorStore) ListProjectExecutionEventsAfter(ctx context.Context, _ string, _ int64, _ int) ([]execution.ExecutionEvent, error) {
	close(s.started)
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestExecutionEventV2CatchUpCancelsInFlightStoreRead(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("cancel in flight", "")
	blocking := &blockingCursorStore{Store: memoryStore, started: make(chan struct{})}
	server := &Server{store: blocking}
	ctx, cancel := context.WithCancel(context.Background())
	response := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(response)
	c.Request = httptest.NewRequest(http.MethodGet, "/", nil).WithContext(ctx)
	done := make(chan error, 1)
	go func() {
		_, _, err := server.writeProjectExecutionEventV2Pages(c, project.ID, 0, 10, 2)
		done <- err
	}()
	<-blocking.started
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("in-flight cancellation err = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("in-flight cursor read did not observe request cancellation")
	}
}

func TestExecutionEventV2CatchUpHonorsCancellationBeforeStoreRead(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("cancel", "")
	spy := &cursorStateStore{Store: memoryStore}
	server := &Server{store: spy}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	response := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(response)
	c.Request = httptest.NewRequest(http.MethodGet, "/", nil).WithContext(ctx)
	_, _, err := server.writeProjectExecutionEventV2Pages(c, project.ID, 0, 10, 2)
	if !errors.Is(err, context.Canceled) || spy.afterCalls != 0 || response.Body.Len() != 0 {
		t.Fatalf("cancelled catch-up err=%v calls=%d body=%q", err, spy.afterCalls, response.Body.String())
	}
}

func TestExecutionEventCursorHeaderTakesPrecedence(t *testing.T) {
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodGet, "/?cursor=7", nil)
	c.Request.Header.Set("Last-Event-ID", "4")
	cursor, err := executionEventCursorFromRequest(c)
	if err != nil || cursor != 4 {
		t.Fatalf("cursor = %d, err=%v", cursor, err)
	}
}

func TestExecutionEventV2KeepaliveIsAnIdleComment(t *testing.T) {
	response := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(response)
	writeExecutionEventV2Keepalive(c)
	if response.Body.String() != ": keepalive\n\n" {
		t.Fatalf("keepalive = %q", response.Body.String())
	}
}
