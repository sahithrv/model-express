package api

import (
	"errors"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/store"
)

type policyPreviewResponse struct {
	policies.EffectivePolicy
	ReadOnly      bool `json:"read_only"`
	AuditRecorded bool `json:"audit_recorded"`
}

func (s *Server) previewProjectExperimentPolicy(c *gin.Context) {
	if strings.TrimSpace(c.Query("account_id")) != "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "account_id is derived from the authenticated project and cannot be supplied",
		})
		return
	}
	projectID := strings.TrimSpace(c.Param("id"))
	project, err := s.store.GetProject(projectID)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	datasetID := strings.TrimSpace(c.Query("dataset_id"))
	jobID := strings.TrimSpace(c.Query("job_id"))
	experimentJobID := strings.TrimSpace(c.Query("experiment_job_id"))
	if jobID != "" && experimentJobID != "" && jobID != experimentJobID {
		c.JSON(http.StatusBadRequest, gin.H{"error": "job_id and experiment_job_id must match when both are supplied"})
		return
	}
	if jobID == "" {
		jobID = experimentJobID
	}
	if jobID != "" {
		job, getErr := s.store.GetJob(jobID)
		if getErr != nil || job.ProjectID != project.ID {
			c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
			return
		}
		jobDatasetID, _ := job.Config["dataset_id"].(string)
		jobDatasetID = strings.TrimSpace(jobDatasetID)
		if datasetID != "" && jobDatasetID != "" && datasetID != jobDatasetID {
			c.JSON(http.StatusBadRequest, gin.H{"error": "dataset_id does not match the selected job"})
			return
		}
		if datasetID == "" {
			datasetID = jobDatasetID
		}
	}

	task := strings.TrimSpace(c.Query("task"))
	if datasetID != "" {
		dataset, getErr := s.store.GetDataset(datasetID)
		if getErr != nil || dataset.ProjectID != project.ID {
			c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
			return
		}
		if task == "" {
			profileTask, _ := dataset.Profile["task_type"].(string)
			if entry, ok := catalog.Resolve("tasks", profileTask); ok {
				task = entry.ID
			}
		}
	}

	result, err := policies.NewResolver(s.store).Resolve(policies.ScopeContext{
		AccountID:       project.AccountID,
		ProjectID:       project.ID,
		DatasetID:       datasetID,
		ExperimentJobID: jobID,
		Task:            task,
		Runner:          strings.TrimSpace(c.Query("runner")),
	})
	if err != nil {
		writePolicyPreviewError(c, result, err)
		return
	}
	c.JSON(http.StatusOK, policyPreviewResponse{
		EffectivePolicy: result,
		ReadOnly:        true,
		AuditRecorded:   false,
	})
}

func writePolicyPreviewError(c *gin.Context, result policies.EffectivePolicy, err error) {
	if errors.Is(err, store.ErrNotFound) {
		c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
		return
	}
	var policyErr *policies.PolicyError
	if !errors.As(err, &policyErr) {
		if errors.Is(err, store.ErrInvalidRequest) {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	status := http.StatusUnprocessableEntity
	if policyErr.Code == policies.ReasonUnknownCatalogIdentifier {
		status = http.StatusBadRequest
	}
	payload := gin.H{
		"error":                 policyErr.Error(),
		"code":                  policyErr.Code,
		"findings":              policyErr.Findings,
		"effective_policy_hash": policyErr.EffectivePolicyHash,
		"policy_evaluation_id":  policyErr.PolicyEvaluationID,
		"blocked_dimensions":    policyErr.BlockedDimensions,
		"contributing_scopes":   policyErr.ContributingScopes,
		"read_only":             true,
		"audit_recorded":        false,
	}
	if result.EffectivePolicyHash != "" {
		payload["decision"] = result.Decision
		payload["catalog_version"] = result.CatalogVersion
		payload["snapshot"] = result.Snapshot
		payload["permitted_catalog"] = result.PermittedCatalog
		payload["permitted_counts"] = result.PermittedCounts
		payload["blocked_dimensions"] = result.BlockedDimensions
	}
	c.JSON(status, payload)
}
