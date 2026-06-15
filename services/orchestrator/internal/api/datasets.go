package api

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/datasets"
	datasetmetadata "model-express/services/orchestrator/internal/datasets/metadata"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/store"
)

type updateDatasetProfileRequest struct {
	Profile map[string]any `json:"profile" binding:"required"`
}

type activeDatasetMetadataImportGetter interface {
	GetActiveDatasetMetadataImport(datasetID string) (datasets.DatasetMetadataImport, error)
}

type activeDatasetMetadataImportGetterWithContext interface {
	GetActiveDatasetMetadataImport(ctx context.Context, datasetID string) (datasets.DatasetMetadataImport, error)
}

type importDatasetMetadataRequest struct {
	StrictMode      bool                                 `json:"strict_mode"`
	SourceKind      string                               `json:"source_kind"`
	Sources         []datasetMetadataSourceRequest       `json:"sources"`
	Inventory       datasetmetadata.DatasetFileInventory `json:"inventory"`
	WorkerDiscovery map[string]any                       `json:"worker_discovery"`
	Warnings        []datasets.MetadataIssue             `json:"warnings"`
}

type datasetMetadataSourceRequest struct {
	RelativePath   string   `json:"relative_path"`
	StorageURI     string   `json:"storage_uri"`
	ChecksumSHA256 string   `json:"checksum_sha256"`
	SizeBytes      int64    `json:"size_bytes"`
	DeclaredFormat string   `json:"declared_format"`
	ContentBase64  string   `json:"content_base64"`
	Warnings       []string `json:"warnings"`
}

func (s *Server) importDatasetMetadata(c *gin.Context) {
	var req importDatasetMetadataRequest
	if !bindJSON(c, &req) {
		return
	}

	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	parseRequest, warnings, err := datasetMetadataParseRequest(req)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if parseRequest.SourceKind == "" && len(req.WorkerDiscovery) > 0 {
		parseRequest.SourceKind = datasets.MetadataSourceKindWorkerDiscovery
	}
	importRecord, err := datasetmetadata.NewService().Parse(dataset, parseRequest)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	importRecord.Warnings = append(importRecord.Warnings, normalizeMetadataIssues(req.Warnings)...)
	importRecord.Warnings = append(importRecord.Warnings, warnings...)
	importRecord.Summary = datasetmetadata.BuildSummary(importRecord)
	importRecord.AgentSafeSummary = datasetmetadata.BuildAgentSafeSummary(importRecord.Summary)

	stored, err := s.store.CreateDatasetMetadataImport(importRecord)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"metadata_import":    stored,
		"summary":            stored.Summary,
		"agent_safe_summary": stored.AgentSafeSummary,
	})
}

func (s *Server) listDatasetMetadataImports(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	imports, err := s.store.ListDatasetMetadataImports(dataset.ID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"dataset_id":        dataset.ID,
		"metadata_imports":  imports,
		"source_of_truth":   "dataset_metadata_imports",
		"safe_summary_only": false,
	})
}

func (s *Server) getDatasetMetadataImport(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	importRecord, err := s.store.GetDatasetMetadataImport(c.Param("import_id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if importRecord.DatasetID != dataset.ID {
		writeStoreError(c, store.ErrNotFound)
		return
	}
	c.JSON(http.StatusOK, importRecord)
}

func (s *Server) getDatasetMetadataSummary(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	importRecord, err := s.store.GetActiveDatasetMetadataImport(dataset.ID)
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			c.JSON(http.StatusOK, gin.H{
				"dataset_id":         dataset.ID,
				"metadata_import_id": "",
				"summary":            nil,
				"agent_safe_summary": nil,
				"source_of_truth":    "dataset_metadata_imports",
				"safe_summary_only":  true,
				"message":            "No active dataset metadata import has been recorded.",
			})
			return
		}
		writeStoreError(c, err)
		return
	}
	if strings.EqualFold(c.Query("safe"), "false") || strings.EqualFold(c.Query("agent_safe"), "false") {
		c.JSON(http.StatusOK, gin.H{
			"dataset_id":           dataset.ID,
			"metadata_import_id":   importRecord.ID,
			"summary":              importRecord.Summary,
			"agent_safe_summary":   importRecord.AgentSafeSummary,
			"source_of_truth":      "dataset_metadata_imports",
			"raw_sources_included": false,
		})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"dataset_id":         dataset.ID,
		"metadata_import_id": importRecord.ID,
		"summary":            importRecord.AgentSafeSummary,
		"agent_safe_summary": importRecord.AgentSafeSummary,
		"source_of_truth":    "dataset_metadata_imports",
		"safe_summary_only":  true,
	})
}

func (s *Server) getDatasetMetadataBundle(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}
	limit := queryInt(c, "limit", 1000, 1, 1000)
	offset := queryInt(c, "offset", 0, 0, 1_000_000_000)
	importID := strings.TrimSpace(c.Query("metadata_import_id"))
	if importID == "" {
		importID = strings.TrimSpace(c.Query("import_id"))
	}
	includeAnnotations := false
	for _, include := range strings.Split(c.Query("include"), ",") {
		if strings.EqualFold(strings.TrimSpace(include), "bbox") || strings.EqualFold(strings.TrimSpace(include), "annotations") {
			includeAnnotations = true
		}
	}
	bundle, err := s.store.GetDatasetMetadataBundle(dataset.ID, importID, includeAnnotations, limit, offset)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	purpose := strings.TrimSpace(c.Query("purpose"))
	if purpose != "" {
		bundle.Purpose = purpose
	}
	c.JSON(http.StatusOK, gin.H{"bundle": bundle})
}

func datasetMetadataParseRequest(req importDatasetMetadataRequest) (datasetmetadata.ImportRequest, []datasets.MetadataIssue, error) {
	out := datasetmetadata.ImportRequest{
		StrictMode: req.StrictMode,
		SourceKind: strings.TrimSpace(req.SourceKind),
		Inventory:  req.Inventory,
		Sources:    make([]datasetmetadata.SourceInput, 0, len(req.Sources)),
	}
	warnings := []datasets.MetadataIssue{}
	var totalBytes int64
	for _, source := range req.Sources {
		input := datasetmetadata.SourceInput{
			RelativePath:   source.RelativePath,
			StorageURI:     source.StorageURI,
			ChecksumSHA256: source.ChecksumSHA256,
			SizeBytes:      source.SizeBytes,
			DeclaredFormat: source.DeclaredFormat,
		}
		for _, warning := range source.Warnings {
			warnings = append(warnings, datasets.MetadataIssue{
				Severity: "warning",
				Code:     strings.TrimSpace(warning),
				Message:  metadataWarningMessage(warning),
			})
		}
		if strings.TrimSpace(source.ContentBase64) != "" {
			decoded, err := base64.StdEncoding.DecodeString(source.ContentBase64)
			if err != nil {
				return datasetmetadata.ImportRequest{}, nil, fmt.Errorf("%w: metadata source content_base64 is invalid", store.ErrInvalidRequest)
			}
			if len(decoded) > datasetMetadataMaxSourceBytes {
				warnings = append(warnings, datasets.MetadataIssue{Severity: "warning", Code: "backend_source_size_cap", Message: "metadata source content exceeded backend source byte cap and was skipped"})
			} else if totalBytes+int64(len(decoded)) > datasetMetadataMaxTotalSourceBytes {
				warnings = append(warnings, datasets.MetadataIssue{Severity: "warning", Code: "backend_total_source_size_cap", Message: "metadata source content exceeded backend total byte cap and was skipped"})
			} else {
				input.Content = decoded
				totalBytes += int64(len(decoded))
			}
		}
		out.Sources = append(out.Sources, input)
	}
	return out, warnings, nil
}

func normalizeMetadataIssues(issues []datasets.MetadataIssue) []datasets.MetadataIssue {
	out := make([]datasets.MetadataIssue, 0, len(issues))
	for _, issue := range issues {
		issue.Code = strings.TrimSpace(issue.Code)
		issue.Message = strings.TrimSpace(issue.Message)
		if issue.Code == "" && issue.Message == "" {
			continue
		}
		if issue.Severity == "" {
			issue.Severity = "warning"
		}
		out = append(out, issue)
	}
	return out
}

func metadataWarningMessage(code string) string {
	code = strings.TrimSpace(code)
	switch code {
	case "unsupported_metadata_format":
		return "metadata candidate was marked unsupported by worker discovery"
	case "content_skipped_source_size_cap":
		return "metadata candidate content was skipped by the worker source byte cap"
	case "content_skipped_total_size_cap":
		return "metadata candidate content was skipped by the worker total byte cap"
	default:
		if code == "" {
			return "metadata candidate warning"
		}
		return strings.ReplaceAll(code, "_", " ")
	}
}

func (s *Server) createDataset(c *gin.Context) {
	var req createDatasetRequest
	if !bindJSON(c, &req) {
		return
	}

	dataset, err := s.store.CreateDataset(
		c.Param("id"),
		req.Name,
		req.StorageURI,
		req.ChecksumSHA256,
		req.SizeBytes,
	)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	if _, err := s.store.CreateJob(dataset.ProjectID, jobs.TemplateProfileDataset, map[string]any{
		"dataset_id": dataset.ID,
	}); err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusCreated, dataset)
}

func (s *Server) listProjectDatasets(c *gin.Context) {
	datasets, err := s.store.ListProjectDatasets(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"datasets": datasets})
}

func (s *Server) getDataset(c *gin.Context) {
	dataset, err := s.store.GetDataset(c.Param("id"))
	if err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, dataset)
}

func (s *Server) updateDatasetProfile(c *gin.Context) {
	var req updateDatasetProfileRequest
	if !bindJSON(c, &req) {
		return
	}

	dataset, err := s.store.UpdateDatasetProfile(c.Param("id"), req.Profile)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	s.indexMemoryCard(context.Background(), memory.NewDatasetProfileMemoryCard(dataset))

	visualQueued, err := s.maybeQueueInitialDatasetVisualAnalysis(dataset.ID)
	if err != nil {
		writeStoreError(c, err)
		return
	}
	if visualQueued {
		c.JSON(http.StatusOK, gin.H{
			"dataset":                         dataset,
			"initial_experiment_plan_status":  "waiting_for_visual_analysis",
			"visual_analysis_required_before": "initial_experiment_plan",
		})
		return
	}

	if err := s.createInitialPlanForDataset(dataset.ID); err != nil {
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, dataset)
}
