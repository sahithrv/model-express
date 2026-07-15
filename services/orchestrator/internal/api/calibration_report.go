package api

import (
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/calibration"
)

const (
	defaultCalibrationEvaluationWindow = 30 * 24 * time.Hour
	defaultCalibrationTrainingWindow   = 90 * 24 * time.Hour
)

func (s *Server) getProjectCalibrationReport(c *gin.Context) {
	request, err := calibrationReportRequestFromQuery(c, time.Now().UTC())
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	report, err := calibration.GenerateReport(s.store, request)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, report)
}

func calibrationReportRequestFromQuery(c *gin.Context, now time.Time) (calibration.ReportRequest, error) {
	evaluationEnd, err := calibrationQueryTime(c, "evaluation_end", now.UTC())
	if err != nil {
		return calibration.ReportRequest{}, err
	}
	evaluationStart, err := calibrationQueryTime(c, "evaluation_start", evaluationEnd.Add(-defaultCalibrationEvaluationWindow))
	if err != nil {
		return calibration.ReportRequest{}, err
	}
	trainingEnd, err := calibrationQueryTime(c, "training_end", evaluationStart)
	if err != nil {
		return calibration.ReportRequest{}, err
	}
	trainingStart, err := calibrationQueryTime(c, "training_start", trainingEnd.Add(-defaultCalibrationTrainingWindow))
	if err != nil {
		return calibration.ReportRequest{}, err
	}
	limit, err := calibrationQueryInt(c, "limit", calibration.DefaultCalibrationReadLimit)
	if err != nil {
		return calibration.ReportRequest{}, err
	}
	minCohortSize, err := calibrationQueryInt(c, "min_cohort_size", calibration.DefaultCalibrationMinCohortSize)
	if err != nil {
		return calibration.ReportRequest{}, err
	}
	meaningful, err := calibrationQueryFloat(c, "meaningful_improvement", 0.01)
	if err != nil {
		return calibration.ReportRequest{}, err
	}
	return calibration.NormalizeReportRequest(calibration.ReportRequest{
		ProjectID:        c.Param("id"),
		TrainingWindow:   calibration.TimeWindow{Start: trainingStart, End: trainingEnd},
		EvaluationWindow: calibration.TimeWindow{Start: evaluationStart, End: evaluationEnd},
		Limit:            limit, MinCohortSize: minCohortSize, MeaningfulImprovement: meaningful,
	})
}

func calibrationQueryTime(c *gin.Context, name string, fallback time.Time) (time.Time, error) {
	value := strings.TrimSpace(c.Query(name))
	if value == "" {
		return fallback.UTC(), nil
	}
	parsed, err := time.Parse(time.RFC3339Nano, value)
	if err != nil {
		return time.Time{}, fmt.Errorf("%s must be RFC3339: %w", name, err)
	}
	return parsed.UTC(), nil
}

func calibrationQueryInt(c *gin.Context, name string, fallback int) (int, error) {
	value := strings.TrimSpace(c.Query(name))
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return 0, fmt.Errorf("%s must be an integer", name)
	}
	return parsed, nil
}

func calibrationQueryFloat(c *gin.Context, name string, fallback float64) (float64, error) {
	value := strings.TrimSpace(c.Query(name))
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) {
		return 0, fmt.Errorf("%s must be a finite number", name)
	}
	return parsed, nil
}
