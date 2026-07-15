package api

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/store"
)

type realizationObservationRequest struct {
	TrainingAttemptID  string         `json:"training_attempt_id" binding:"required"`
	SchemaVersion      string         `json:"schema_version"`
	Stage              string         `json:"stage" binding:"required"`
	IdempotencyKey     string         `json:"idempotency_key" binding:"required"`
	RealizedConfig     map[string]any `json:"realized_config" binding:"required"`
	FrameworkArguments map[string]any `json:"framework_arguments"`
	Evidence           map[string]any `json:"evidence"`
	AdjustmentPolicy   string         `json:"adjustment_policy"`
	Simulated          bool           `json:"simulated"`
}

func (s *Server) reportRealizationObservation(c *gin.Context) {
	var req realizationObservationRequest
	if !bindJSON(c, &req) {
		return
	}
	if _, ok := s.validateJobCallback(c, c.Param("id"), req.TrainingAttemptID, "execution_realization", map[string]any{"stage": req.Stage, "idempotency_key": req.IdempotencyKey}); !ok {
		return
	}
	observation, created, err := s.store.AppendRealizationObservation(c.Param("id"), execution.RealizationObservationCreate{
		AttemptID: strings.TrimSpace(req.TrainingAttemptID), SchemaVersion: firstNonEmptyString(strings.TrimSpace(req.SchemaVersion), execution.ExecutionObservationSchemaV1), Stage: req.Stage, IdempotencyKey: strings.TrimSpace(req.IdempotencyKey), RealizedConfig: req.RealizedConfig, FrameworkArguments: req.FrameworkArguments, Evidence: req.Evidence, AdjustmentPolicy: strings.TrimSpace(req.AdjustmentPolicy), Simulated: req.Simulated,
	})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	status := http.StatusOK
	if created {
		status = http.StatusCreated
	}
	c.JSON(status, observation)
}

func (s *Server) getJobExecutionRecord(c *gin.Context) {
	record, err := s.store.GetJobExecutionRecord(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, record)
}

func (s *Server) listProjectExecutionRecords(c *gin.Context) {
	limit := queryInt(c, "limit", 50, 1, 100)
	offset := queryInt(c, "offset", 0, 0, 1_000_000_000)
	items, err := s.store.ListProjectExecutionRecords(c.Param("id"), store.PageOptions{Limit: limit + 1, Offset: offset})
	if err != nil {
		writeStoreError(c, err)
		return
	}
	page, hasMore := pageHasMore(items, limit)
	c.JSON(http.StatusOK, pagedListPayload("execution_records", page, limit, offset, hasMore))
}
