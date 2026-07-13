package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/store"
)

func TestJobProgressCallbackAuthenticatesAndPersistsBoundary(t *testing.T) {
	memoryStore, router, assigned := newProgressCallbackFixture(t)
	before := time.Now().UTC()
	resp := postProgressCallback(t, router, assigned, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"taxonomy_version":    jobs.ProgressTaxonomyVersion,
		"stage":               jobs.ProgressStageRemoteScheduled,
		"detail_code":         "provider.submitted",
		"status":              jobs.ProgressStatusRunning,
		"revision":            3,
		"message":             "Remote training was scheduled.",
		"metadata":            map[string]any{"provider": "modal"},
	}, true)
	if resp.Code != http.StatusOK {
		t.Fatalf("valid progress status=%d body=%s", resp.Code, resp.Body.String())
	}
	var payload struct {
		Progress     jobs.JobProgress `json:"progress"`
		Updated      bool             `json:"updated"`
		EventCreated bool             `json:"event_created"`
	}
	if err := json.Unmarshal(resp.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if !payload.Updated || !payload.EventCreated || payload.Progress.Stage != jobs.ProgressStageRemoteScheduled || payload.Progress.Revision != 3 {
		t.Fatalf("valid progress response=%#v", payload)
	}
	if payload.Progress.HeartbeatAt.Before(before) || payload.Progress.HeartbeatAt.After(time.Now().UTC()) {
		t.Fatalf("heartbeat does not use server receipt time: %s", payload.Progress.HeartbeatAt)
	}
	events, err := memoryStore.ListProjectExecutionEvents(assigned.ProjectID, 100)
	if err != nil {
		t.Fatal(err)
	}
	boundaries := 0
	for _, event := range events {
		if event.EventType == execution.EventJobProgressBoundary {
			boundaries++
			if event.Payload["message"] != nil || event.Payload["metadata"] != nil || event.Payload["provider"] != nil {
				t.Fatalf("boundary event retained snapshot-only callback content: %#v", event)
			}
		}
	}
	if boundaries != 1 {
		t.Fatalf("boundary event count=%d events=%#v", boundaries, events)
	}
}

func TestJobProgressCallbackRejectsMissingInvalidAndStaleAttempts(t *testing.T) {
	memoryStore, router, assigned := newProgressCallbackFixture(t)
	body := progressCallbackBody(t, assigned, map[string]any{
		"stage": jobs.ProgressStageEnvironmentStarting, "revision": 3,
	})

	missing := httptest.NewRecorder()
	missingReq := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/progress", bytes.NewReader(body))
	missingReq.Header.Set("Content-Type", "application/json")
	router.ServeHTTP(missing, missingReq)
	if missing.Code != http.StatusUnauthorized {
		t.Fatalf("missing token status=%d body=%s", missing.Code, missing.Body.String())
	}

	invalid := httptest.NewRecorder()
	invalidReq := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/progress", bytes.NewReader(body))
	invalidReq.Header.Set("Content-Type", "application/json")
	invalidReq.Header.Set(callbackTokenHeader, "invalid-token")
	router.ServeHTTP(invalid, invalidReq)
	if invalid.Code != http.StatusUnauthorized {
		t.Fatalf("invalid token status=%d body=%s", invalid.Code, invalid.Body.String())
	}

	if _, _, err := memoryStore.RetryJob(assigned.ID, "retry", store.RetryJobOptions{}); err != nil {
		t.Fatal(err)
	}
	stale := httptest.NewRecorder()
	staleReq := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/progress", bytes.NewReader(body))
	staleReq.Header.Set("Content-Type", "application/json")
	setCallbackToken(t, staleReq, assigned)
	router.ServeHTTP(stale, staleReq)
	if stale.Code != http.StatusConflict || !strings.Contains(stale.Body.String(), "stale_attempt") {
		t.Fatalf("stale attempt status=%d body=%s", stale.Code, stale.Body.String())
	}
}

func TestJobProgressCallbackDuplicateOlderHeartbeatAndTerminalNonRegression(t *testing.T) {
	memoryStore, router, assigned := newProgressCallbackFixture(t)
	first := postProgressCallback(t, router, assigned, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"taxonomy_version":    jobs.ProgressTaxonomyVersion, "stage": jobs.ProgressStageDataLoading,
		"status": jobs.ProgressStatusRunning, "revision": 3,
	}, true)
	if first.Code != http.StatusOK {
		t.Fatalf("first status=%d body=%s", first.Code, first.Body.String())
	}
	heartbeat := postProgressCallback(t, router, assigned, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"taxonomy_version":    jobs.ProgressTaxonomyVersion, "stage": jobs.ProgressStageDataLoading,
		"status": jobs.ProgressStatusRunning, "revision": 4, "message": "Data loading is active.",
	}, true)
	if heartbeat.Code != http.StatusOK || !strings.Contains(heartbeat.Body.String(), `"updated":true`) || !strings.Contains(heartbeat.Body.String(), `"event_created":false`) {
		t.Fatalf("heartbeat status=%d body=%s", heartbeat.Code, heartbeat.Body.String())
	}
	for _, revision := range []int{4, 3} {
		ignored := postProgressCallback(t, router, assigned, map[string]any{
			"training_attempt_id": callbackAttemptID(t, assigned),
			"taxonomy_version":    jobs.ProgressTaxonomyVersion, "stage": jobs.ProgressStageTraining,
			"status": jobs.ProgressStatusRunning, "revision": revision,
		}, true)
		if ignored.Code != http.StatusOK || !strings.Contains(ignored.Body.String(), `"updated":false`) {
			t.Fatalf("revision %d status=%d body=%s", revision, ignored.Code, ignored.Body.String())
		}
	}

	terminal, err := memoryStore.UpsertJobProgress(assigned.ID, jobs.JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: jobs.ProgressTaxonomyVersion,
		Stage: jobs.ProgressStageCompleted, Status: jobs.ProgressStatusCompleted, Revision: 10,
	})
	if err != nil {
		t.Fatal(err)
	}
	postTerminal := postProgressCallback(t, router, assigned, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"taxonomy_version":    jobs.ProgressTaxonomyVersion, "stage": jobs.ProgressStageFinalizing,
		"status": jobs.ProgressStatusRunning, "revision": 11,
	}, true)
	if postTerminal.Code != http.StatusOK || !strings.Contains(postTerminal.Body.String(), `"updated":false`) {
		t.Fatalf("post-terminal status=%d body=%s", postTerminal.Code, postTerminal.Body.String())
	}
	after, err := memoryStore.GetJobProgress(assigned.ID, 1)
	if err != nil || after.Stage != terminal.Stage || after.Revision != terminal.Revision {
		t.Fatalf("terminal progress regressed: before=%#v after=%#v err=%v", terminal, after, err)
	}
}

func TestJobProgressCallbackStrictValidation(t *testing.T) {
	_, router, assigned := newProgressCallbackFixture(t)
	base := map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"taxonomy_version":    jobs.ProgressTaxonomyVersion,
		"stage":               jobs.ProgressStageTraining, "status": jobs.ProgressStatusRunning, "revision": 3,
	}
	tooLongMessage := strings.Repeat("x", jobs.JobProgressMaxMessageBytes+1)
	tooLongDetail := strings.Repeat("x", jobs.JobProgressMaxDetailCodeBytes+1)
	cases := []struct {
		name  string
		patch map[string]any
	}{
		{"taxonomy", map[string]any{"taxonomy_version": 2}},
		{"terminal stage", map[string]any{"stage": jobs.ProgressStageCompleted, "status": jobs.ProgressStatusCompleted}},
		{"queued stage", map[string]any{"stage": jobs.ProgressStageQueued, "status": jobs.ProgressStatusQueued}},
		{"incompatible status", map[string]any{"status": jobs.ProgressStatusCompleted}},
		{"negative current", map[string]any{"current": -1, "total": 2, "unit": "epoch"}},
		{"reversed range", map[string]any{"current": 3, "total": 2, "unit": "epoch"}},
		{"partial range", map[string]any{"current": 1, "unit": "epoch"}},
		{"range missing unit", map[string]any{"current": 1, "total": 2}},
		{"unit without range", map[string]any{"unit": "epoch"}},
		{"unknown unit", map[string]any{"current": 1, "total": 2, "unit": "fortnight"}},
		{"unsafe message", map[string]any{"message": "loading s3://private/data"}},
		{"control message", map[string]any{"message": "line one\nline two"}},
		{"long message", map[string]any{"message": tooLongMessage}},
		{"invalid detail", map[string]any{"detail_code": "epoch complete"}},
		{"long detail", map[string]any{"detail_code": tooLongDetail}},
		{"unknown metadata", map[string]any{"metadata": map[string]any{"arbitrary": "value"}}},
		{"unsafe metadata", map[string]any{"metadata": map[string]any{"provider": "file:///private/model"}}},
		{"nested metadata", map[string]any{"metadata": map[string]any{"provider": map[string]any{"name": "modal"}}}},
		{"metadata token type", map[string]any{"metadata": map[string]any{"provider": []string{"modal"}}}},
		{"metadata boolean type", map[string]any{"metadata": map[string]any{"early_stopped": "false"}}},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			body := copyPayloadMap(base)
			for key, value := range test.patch {
				body[key] = value
			}
			resp := postProgressCallback(t, router, assigned, body, true)
			if resp.Code != http.StatusBadRequest {
				t.Fatalf("status=%d body=%s", resp.Code, resp.Body.String())
			}
		})
	}

	unknown := append(progressCallbackBody(t, assigned, map[string]any{"stage": jobs.ProgressStageTraining, "revision": 3})[:0:0], []byte(fmt.Sprintf(
		`{"training_attempt_id":%q,"taxonomy_version":1,"stage":"training","status":"running","revision":3,"unknown":true}`,
		callbackAttemptID(t, assigned),
	))...)
	for name, body := range map[string][]byte{
		"unknown field": unknown,
		"trailing json": append(progressCallbackBody(t, assigned, map[string]any{"stage": jobs.ProgressStageTraining, "revision": 3}), []byte(` {}`)...),
	} {
		t.Run(name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/progress", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			setCallbackToken(t, req, assigned)
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)
			if resp.Code != http.StatusBadRequest {
				t.Fatalf("status=%d body=%s", resp.Code, resp.Body.String())
			}
		})
	}

	// Once syntax and required fields are valid, attempt authentication wins
	// over semantic validation and does not disclose callback details.
	unauthenticatedInvalid := postProgressCallback(t, router, assigned, map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"taxonomy_version":    jobs.ProgressTaxonomyVersion, "stage": jobs.ProgressStageCompleted,
		"status": jobs.ProgressStatusCompleted, "revision": 3,
	}, false)
	if unauthenticatedInvalid.Code != http.StatusUnauthorized {
		t.Fatalf("unauthenticated semantic error status=%d body=%s", unauthenticatedInvalid.Code, unauthenticatedInvalid.Body.String())
	}
}

func newProgressCallbackFixture(t *testing.T) (*store.MemoryStore, *gin.Engine, jobs.ExperimentJob) {
	t.Helper()
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("progress callback", "")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := memoryStore.CreateDataset(project.ID, "dataset", "memory://dataset", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	worker, err := memoryStore.RegisterWorker(project.ID, "worker", "gpu")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := memoryStore.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID}); err != nil {
		t.Fatal(err)
	}
	router := NewRouter(memoryStore)
	assigned := pollJobForCallback(t, router, worker.ID, `{}`)
	return memoryStore, router, assigned
}

func postProgressCallback(t *testing.T, router *gin.Engine, assigned jobs.ExperimentJob, payload map[string]any, authenticated bool) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/jobs/"+assigned.ID+"/progress", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if authenticated {
		setCallbackToken(t, req, assigned)
	}
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	return resp
}

func progressCallbackBody(t *testing.T, assigned jobs.ExperimentJob, patch map[string]any) []byte {
	t.Helper()
	payload := map[string]any{
		"training_attempt_id": callbackAttemptID(t, assigned),
		"taxonomy_version":    jobs.ProgressTaxonomyVersion,
		"stage":               jobs.ProgressStageTraining,
		"status":              jobs.ProgressStatusRunning,
		"revision":            3,
	}
	for key, value := range patch {
		payload[key] = value
	}
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	return body
}
