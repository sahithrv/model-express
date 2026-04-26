package api

import (
	"errors"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/store"
)

type createProjectRequest struct {
	Name string `json:"name" binding:"required"`
	Goal string `json:"goal" binding:"required"`
}

type registerWorkerRequest struct {
	Name    string `json:"name" binding:"required"`
	GPUType string `json:"gpu_type"`
}

type createJobRequest struct {
	Template string         `json:"template" binding:"required"`
	Config   map[string]any `json:"config"`
}

type reportMetricRequest struct {
	Epoch   int                `json:"epoch" binding:"required"`
	Metrics map[string]float64 `json:"metrics" binding:"required"`
}

type completeJobRequest struct {
	MLflowRunID string `json:"mlflow_run_id"`
}

type failJobRequest struct {
	Error string `json:"error" binding:"required"`
}

type pollJobResponse struct {
	Job *jobs.ExperimentJob `json:"job"`
}

func (s *Server) health(c *gin.Context) {
	c.JSON(http.StatusOK, healthResponse{
		Status:    "ok",
		Service:   "orchestrator",
		Timestamp: time.Now().UTC(),
	})
}

func (s *Server) createProject(c *gin.Context) {
	var req createProjectRequest
	if !bindJSON(c, &req) {
		return
	}

	project := s.store.CreateProject(req.Name, req.Goal)
	c.JSON(http.StatusCreated, project)
}

func (s *Server) listProjects(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"projects": s.store.ListProjects()})
}

func (s *Server) getProject(c *gin.Context) {
	project, err := s.store.GetProject(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, project)
}

func (s *Server) createJob(c *gin.Context) {
	var req createJobRequest
	if !bindJSON(c, &req) {
		return
	}

	if req.Config == nil {
		req.Config = map[string]any{}
	}

	job, err := s.store.CreateJob(c.Param("id"), req.Template, req.Config)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, job)
}

func (s *Server) getJob(c *gin.Context) {
	job, err := s.store.GetJob(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) reportMetric(c *gin.Context) {
	var req reportMetricRequest
	if !bindJSON(c, &req) {
		return
	}

	metric, err := s.store.ReportMetric(c.Param("id"), req.Epoch, req.Metrics)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, metric)
}

func (s *Server) completeJob(c *gin.Context) {
	var req completeJobRequest
	if !bindJSON(c, &req) {
		return
	}

	job, err := s.store.CompleteJob(c.Param("id"), req.MLflowRunID)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) failJob(c *gin.Context) {
	var req failJobRequest
	if !bindJSON(c, &req) {
		return
	}

	job, err := s.store.FailJob(c.Param("id"), req.Error)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, job)
}

func (s *Server) registerWorker(c *gin.Context) {
	var req registerWorkerRequest
	if !bindJSON(c, &req) {
		return
	}

	worker := s.store.RegisterWorker(req.Name, req.GPUType)
	c.JSON(http.StatusCreated, worker)
}

func (s *Server) heartbeatWorker(c *gin.Context) {
	worker, err := s.store.HeartbeatWorker(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, worker)
}

func (s *Server) pollJob(c *gin.Context) {
	job, err := s.store.PollJob(c.Param("id"))
	if err == nil {
		c.JSON(http.StatusOK, pollJobResponse{Job: job})
		return
	}

	if errors.Is(err, store.ErrNoJob) {
		c.JSON(http.StatusOK, pollJobResponse{Job: nil})
		return
	}

	writeStoreError(c, err)
}

func bindJSON(c *gin.Context, value any) bool {
	if err := c.ShouldBindJSON(value); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return false
	}

	return true
}

func writeStoreError(c *gin.Context, err error) {
	if errors.Is(err, store.ErrNotFound) {
		c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
		return
	}

	c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
}
