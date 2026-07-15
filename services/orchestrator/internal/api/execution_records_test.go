package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/store"
)

func TestExecutionObservationCallbackIsAuthenticatedIdempotentAndAttemptScoped(t *testing.T) {
	gin.SetMode(gin.TestMode)
	s := store.NewMemoryStore()
	project, _ := s.CreateProject("p", "g")
	dataset, _ := s.CreateDataset(project.ID, "d", "s3://bucket/data", "sha", 1)
	spec, err := execution.BuildExecutionSpecV1("image_classification", "local_simulator", map[string]any{"model": "resnet18", "epochs": 3}, map[string]any{"model": "resnet18", "epochs": 3})
	if err != nil {
		t.Fatal(err)
	}
	payload, _ := spec.Payload()
	job, err := s.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{execution.ExecutionSpecConfigKey: payload, "provider": "local", "dataset_id": dataset.ID})
	if err != nil {
		t.Fatal(err)
	}
	worker, _ := s.RegisterWorker(project.ID, "w", "local")
	assigned, err := s.PollJob(worker.ID, store.JobPollFilter{})
	if err != nil {
		t.Fatal(err)
	}
	server := newServer(s)
	router := gin.New()
	router.POST("/jobs/:id/execution-observations", server.reportRealizationObservation)
	router.GET("/jobs/:id/execution-record", server.getJobExecutionRecord)
	attemptOne := jobConfigString(assigned.Config, "active_attempt_id")
	record, _ := s.GetJobExecutionRecord(job.ID)
	body := map[string]any{"training_attempt_id": attemptOne, "schema_version": execution.ExecutionObservationSchemaV1, "stage": "INITIALIZED", "idempotency_key": "init-1", "realized_config": record.AcceptedSpec.AcceptedSpec}

	post := func(token string, value map[string]any) *httptest.ResponseRecorder {
		data, _ := json.Marshal(value)
		req := httptest.NewRequest(http.MethodPost, "/jobs/"+job.ID+"/execution-observations", bytes.NewReader(data))
		req.Header.Set("Content-Type", "application/json")
		if token != "" {
			req.Header.Set(callbackTokenHeader, token)
		}
		response := httptest.NewRecorder()
		router.ServeHTTP(response, req)
		return response
	}
	if got := post("", body).Code; got != http.StatusUnauthorized {
		t.Fatalf("unauthenticated status=%d", got)
	}
	tokenOne := server.callbackToken(job.ID, attemptOne)
	if got := post(tokenOne, body).Code; got != http.StatusCreated {
		t.Fatalf("initial status=%d", got)
	}
	if got := post(tokenOne, body).Code; got != http.StatusOK {
		t.Fatalf("duplicate status=%d", got)
	}

	if _, requeued, err := s.RetryJob(job.ID, "retry", store.RetryJobOptions{}); err != nil || !requeued {
		t.Fatalf("retry requeued=%v err=%v", requeued, err)
	}
	if _, err := s.PollJob(worker.ID, store.JobPollFilter{}); err != nil {
		t.Fatal(err)
	}
	body["idempotency_key"] = "late-init"
	if got := post(tokenOne, body).Code; got != http.StatusConflict {
		t.Fatalf("stale callback status=%d", got)
	}

	readReq := httptest.NewRequest(http.MethodGet, "/jobs/"+job.ID+"/execution-record", nil)
	readResponse := httptest.NewRecorder()
	router.ServeHTTP(readResponse, readReq)
	if readResponse.Code != http.StatusOK {
		t.Fatalf("read status=%d body=%s", readResponse.Code, readResponse.Body.String())
	}
	var read execution.ExecutionRecord
	if err := json.Unmarshal(readResponse.Body.Bytes(), &read); err != nil {
		t.Fatal(err)
	}
	if len(read.Attempts) != 2 || len(read.Attempts[0].Observations) != 1 {
		t.Fatalf("record=%#v", read)
	}
}
