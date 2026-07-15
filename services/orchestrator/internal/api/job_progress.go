package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/store"
)

type reportJobProgressRequest struct {
	TrainingAttemptID string         `json:"training_attempt_id"`
	TaxonomyVersion   *int           `json:"taxonomy_version"`
	Stage             string         `json:"stage"`
	DetailCode        string         `json:"detail_code,omitempty"`
	Status            string         `json:"status"`
	Current           *int64         `json:"current,omitempty"`
	Total             *int64         `json:"total,omitempty"`
	Unit              string         `json:"unit,omitempty"`
	Message           string         `json:"message,omitempty"`
	Revision          *int64         `json:"revision"`
	Metadata          map[string]any `json:"metadata,omitempty"`
}

func (s *Server) reportJobProgress(c *gin.Context) {
	var req reportJobProgressRequest
	if !bindStrictProgressJSON(c, &req) {
		return
	}
	if req.TaxonomyVersion == nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "taxonomy_version is required"})
		return
	}
	if req.Revision == nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "revision is required"})
		return
	}

	update := jobs.JobProgressUpsert{
		TaxonomyVersion: *req.TaxonomyVersion,
		Stage:           req.Stage,
		DetailCode:      req.DetailCode,
		Status:          req.Status,
		Current:         req.Current,
		Total:           req.Total,
		Unit:            req.Unit,
		Message:         req.Message,
		Revision:        *req.Revision,
		Metadata:        req.Metadata,
	}
	job, ok := s.validateJobCallback(
		c,
		c.Param("id"),
		req.TrainingAttemptID,
		"progress",
		map[string]any{"stage": strings.TrimSpace(req.Stage), "revision": *req.Revision},
	)
	if !ok {
		return
	}
	// Validate the callback-only taxonomy after attempt authentication and
	// before any store composition. The store repeats this check so direct
	// callers get the same authority guard.
	if _, err := jobs.NormalizeWorkerJobProgressUpsert(update); err != nil {
		writeStoreError(c, fmt.Errorf("%w: invalid worker progress: %v", store.ErrInvalidRequest, err))
		return
	}

	result, err := s.store.ReportJobProgress(job.ID, req.TrainingAttemptID, update)
	if err != nil {
		if errors.Is(err, store.ErrStaleAttempt) {
			c.JSON(http.StatusConflict, gin.H{
				"error":               "stale training attempt",
				"status":              "stale_attempt",
				"job_id":              job.ID,
				"training_attempt_id": strings.TrimSpace(req.TrainingAttemptID),
			})
			return
		}
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"progress":      result.Progress,
		"updated":       result.Updated,
		"event_created": result.EventCreated,
	})
}

func bindStrictProgressJSON(c *gin.Context, value any) bool {
	decoder := json.NewDecoder(c.Request.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(value); err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) || strings.Contains(strings.ToLower(err.Error()), "request body too large") {
			c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "JSON request body too large"})
			return false
		}
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return false
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("multiple JSON values are not allowed")
		}
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return false
	}
	return true
}
