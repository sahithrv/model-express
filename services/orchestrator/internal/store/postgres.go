package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/automl"
	"model-express/services/orchestrator/internal/datasets"
	"model-express/services/orchestrator/internal/decisions"
	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/projects"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/strategies"
	"model-express/services/orchestrator/internal/workers"
)

type PostgresStore struct {
	db *sql.DB
}

type rowScanner interface {
	Scan(dest ...any) error
}

func (s *PostgresStore) CreateAgentMemoryRecord(record memory.AgentMemoryRecord) (memory.AgentMemoryRecord, error) {
	if err := s.requireProject(record.ProjectID); err != nil {
		return memory.AgentMemoryRecord{}, err
	}
	if record.Payload == nil {
		record.Payload = map[string]any{}
	}
	if record.Tags == nil {
		record.Tags = []string{}
	}

	payloadJSON, err := json.Marshal(record.Payload)
	if err != nil {
		return memory.AgentMemoryRecord{}, fmt.Errorf("marshal agent memory payload: %w", err)
	}
	tagsJSON, err := json.Marshal(record.Tags)
	if err != nil {
		return memory.AgentMemoryRecord{}, fmt.Errorf("marshal agent memory tags: %w", err)
	}

	const query = `
		INSERT INTO agent_memory_records (invocation_id, project_id, dataset_id, plan_id, job_id, agent_name, kind, summary, payload, tags)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		RETURNING id, invocation_id, project_id, dataset_id, plan_id, job_id, agent_name, kind, summary, payload, tags, created_at
	`
	return scanAgentMemoryRecord(s.db.QueryRowContext(
		context.Background(),
		query,
		record.InvocationID,
		record.ProjectID,
		record.DatasetID,
		record.PlanID,
		record.JobID,
		record.AgentName,
		record.Kind,
		record.Summary,
		payloadJSON,
		tagsJSON,
	))
}

func (s *PostgresStore) ListProjectAgentMemoryRecords(projectID string, filter memory.AgentMemoryFilter) ([]memory.AgentMemoryRecord, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}
	if filter.Limit <= 0 {
		filter.Limit = 25
	}

	const query = `
		SELECT id, invocation_id, project_id, dataset_id, plan_id, job_id, agent_name, kind, summary, payload, tags, created_at
		FROM agent_memory_records
		WHERE project_id = $1
			AND ($2 = '' OR dataset_id = $2)
			AND ($3 = '' OR plan_id = $3)
			AND ($4 = '' OR job_id = $4)
			AND ($5 = '' OR kind = $5)
		ORDER BY created_at DESC
		LIMIT $6
	`
	rows, err := s.db.QueryContext(
		context.Background(),
		query,
		projectID,
		filter.DatasetID,
		filter.PlanID,
		filter.JobID,
		filter.Kind,
		filter.Limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []memory.AgentMemoryRecord{}
	for rows.Next() {
		record, err := scanAgentMemoryRecord(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, record)
	}
	return out, rows.Err()
}

func (s *PostgresStore) insertDatasetVisualAnalysis(analysis datasets.DatasetVisualAnalysis, validationStatus string) (datasets.DatasetVisualAnalysis, error) {
	dataset, err := s.GetDataset(analysis.DatasetID)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, err
	}
	if analysis.ProjectID == "" {
		analysis.ProjectID = dataset.ProjectID
	}
	if analysis.ProjectID != dataset.ProjectID {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("%w: dataset_id does not belong to this project", ErrInvalidRequest)
	}
	if analysis.DatasetName == "" {
		analysis.DatasetName = dataset.Name
	}
	if analysis.SchemaVersion == "" {
		analysis.SchemaVersion = datasets.VisualAnalysisSchemaVersion
	}
	if analysis.AgentName == "" {
		analysis.AgentName = datasets.VisualAnalysisAgentName
	}
	if analysis.TriggerReason == "" {
		analysis.TriggerReason = datasets.VisualTriggerManual
	}
	if analysis.TriggerDetails == nil {
		analysis.TriggerDetails = map[string]any{}
	}
	if analysis.CoverageReport.PerClassCounts == nil {
		analysis.CoverageReport.PerClassCounts = map[string]int{}
	}
	if analysis.ClassesToWatch == nil {
		analysis.ClassesToWatch = []datasets.ClassWatchItem{}
	}
	if analysis.VisualTraits == nil {
		analysis.VisualTraits = []datasets.VisualTrait{}
	}
	if analysis.PreprocessingHypotheses == nil {
		analysis.PreprocessingHypotheses = []datasets.PreprocessingHypothesis{}
	}
	if analysis.Cautions == nil {
		analysis.Cautions = []datasets.VisualCaution{}
	}
	if analysis.Limitations == nil {
		analysis.Limitations = []string{}
	}
	if analysis.ValidationErrors == nil {
		analysis.ValidationErrors = []string{}
	}

	triggerDetailsJSON, err := json.Marshal(analysis.TriggerDetails)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis trigger details: %w", err)
	}
	coverageReportJSON, err := json.Marshal(analysis.CoverageReport)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis coverage report: %w", err)
	}
	classesToWatchJSON, err := json.Marshal(analysis.ClassesToWatch)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis classes to watch: %w", err)
	}
	visualTraitsJSON, err := json.Marshal(analysis.VisualTraits)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis traits: %w", err)
	}
	preprocessingHypothesesJSON, err := json.Marshal(analysis.PreprocessingHypotheses)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis preprocessing hypotheses: %w", err)
	}
	cautionsJSON, err := json.Marshal(analysis.Cautions)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis cautions: %w", err)
	}
	limitationsJSON, err := json.Marshal(analysis.Limitations)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis limitations: %w", err)
	}
	validationErrorsJSON, err := json.Marshal(analysis.ValidationErrors)
	if err != nil {
		return datasets.DatasetVisualAnalysis{}, fmt.Errorf("marshal visual analysis validation errors: %w", err)
	}

	query := `
		INSERT INTO dataset_visual_analyses (
			project_id,
			dataset_id,
			dataset_name,
			schema_version,
			analysis_version,
			prompt_version,
			agent_name,
			agent_version,
			provider,
			model,
			trigger_reason,
			trigger_details,
			source_job_id,
			source_invocation_id,
			profile_schema_version,
			profile_fingerprint,
			total_images,
			images_analyzed,
			coverage_report,
			classes_to_watch,
			confidence,
			visual_traits,
			preprocessing_hypotheses,
			cautions,
			limitations,
			validation_status,
			validation_errors
		)
		VALUES (
			$1,
			$2,
			$3,
			$4,
			COALESCE((SELECT MAX(analysis_version) + 1 FROM dataset_visual_analyses WHERE dataset_id = $2), 1),
			$5,
			$6,
			$7,
			$8,
			$9,
			$10,
			$11,
			$12,
			$13,
			$14,
			$15,
			$16,
			$17,
			$18,
			$19,
			$20,
			$21,
			$22,
			$23,
			$24,
			$25,
			$26
		)
		RETURNING ` + datasetVisualAnalysisSelectColumns() + `
	`
	return scanDatasetVisualAnalysis(s.db.QueryRowContext(
		context.Background(),
		query,
		analysis.ProjectID,
		analysis.DatasetID,
		analysis.DatasetName,
		analysis.SchemaVersion,
		analysis.PromptVersion,
		analysis.AgentName,
		analysis.AgentVersion,
		analysis.Provider,
		analysis.Model,
		string(analysis.TriggerReason),
		triggerDetailsJSON,
		analysis.SourceJobID,
		analysis.SourceInvocationID,
		analysis.ProfileSchemaVersion,
		analysis.ProfileFingerprint,
		analysis.TotalImages,
		analysis.ImagesAnalyzed,
		coverageReportJSON,
		classesToWatchJSON,
		analysis.Confidence,
		visualTraitsJSON,
		preprocessingHypothesesJSON,
		cautionsJSON,
		limitationsJSON,
		validationStatus,
		validationErrorsJSON,
	))
}

func (s *PostgresStore) requireProject(projectID string) error {
	var exists bool
	if err := s.db.QueryRowContext(context.Background(), `
		SELECT EXISTS(SELECT 1 FROM projects WHERE id = $1)
	`, projectID).Scan(&exists); err != nil {
		return err
	}

	if !exists {
		return ErrNotFound
	}

	return nil
}

func (s *PostgresStore) requireProjectDataset(projectID string) error {
	var exists bool
	if err := s.db.QueryRowContext(context.Background(), `
		SELECT EXISTS(SELECT 1 FROM datasets WHERE project_id = $1)
	`, projectID).Scan(&exists); err != nil {
		return err
	}

	if !exists {
		return fmt.Errorf("%w: project must have a dataset before workers or jobs can be created", ErrInvalidRequest)
	}

	return nil
}

func (s *PostgresStore) requireDatasetBelongsToProject(projectID string, datasetID string) error {
	if datasetID == "" {
		return fmt.Errorf("%w: dataset_id is required", ErrInvalidRequest)
	}

	var exists bool
	if err := s.db.QueryRowContext(context.Background(), `
		SELECT EXISTS(SELECT 1 FROM datasets WHERE id = $1 AND project_id = $2)
	`, datasetID, projectID).Scan(&exists); err != nil {
		return err
	}

	if !exists {
		return fmt.Errorf("%w: dataset_id does not belong to this project", ErrInvalidRequest)
	}

	return nil
}

func (s *PostgresStore) requireDatasetConfig(projectID string, config map[string]any) error {
	value, ok := config["dataset_id"]
	if !ok {
		return fmt.Errorf("%w: job config must include dataset_id", ErrInvalidRequest)
	}

	datasetID, ok := value.(string)
	if !ok || datasetID == "" {
		return fmt.Errorf("%w: dataset_id must be a non-empty string", ErrInvalidRequest)
	}

	return s.requireDatasetBelongsToProject(projectID, datasetID)
}

func scanProject(row rowScanner) (projects.Project, error) {
	var project projects.Project
	if err := row.Scan(
		&project.ID,
		&project.Name,
		&project.Goal,
		&project.Status,
		&project.CreatedAt,
		&project.UpdatedAt,
	); err != nil {
		return projects.Project{}, normalizeSQLError(err)
	}

	return project, nil
}

func scanDataset(row rowScanner) (datasets.Dataset, error) {
	var dataset datasets.Dataset
	var profileJSON []byte
	var profiledAt sql.NullTime

	if err := row.Scan(
		&dataset.ID,
		&dataset.ProjectID,
		&dataset.Name,
		&dataset.StorageURI,
		&dataset.ChecksumSHA256,
		&dataset.SizeBytes,
		&profileJSON,
		&dataset.Status,
		&dataset.CreatedAt,
		&profiledAt,
	); err != nil {
		return datasets.Dataset{}, normalizeSQLError(err)
	}

	dataset.Profile = map[string]any{}
	if len(profileJSON) > 0 {
		if err := json.Unmarshal(profileJSON, &dataset.Profile); err != nil {
			return datasets.Dataset{}, fmt.Errorf("unmarshal dataset profile: %w", err)
		}
	}

	if profiledAt.Valid {
		dataset.ProfiledAt = &profiledAt.Time
	}

	return dataset, nil
}

func scanDatasetMetadataImport(row rowScanner) (datasets.DatasetMetadataImport, error) {
	var importRecord datasets.DatasetMetadataImport
	var summaryJSON []byte
	var agentSafeSummaryJSON []byte
	var warningsJSON []byte
	var errorsJSON []byte
	var completedAt sql.NullTime
	if err := row.Scan(
		&importRecord.ID,
		&importRecord.DatasetID,
		&importRecord.ProjectID,
		&importRecord.Status,
		&importRecord.ImportVersion,
		&importRecord.DatasetChecksumSHA256,
		&importRecord.ParserRegistryVersion,
		&importRecord.SourceKind,
		&importRecord.Active,
		&importRecord.StrictMode,
		&summaryJSON,
		&agentSafeSummaryJSON,
		&warningsJSON,
		&errorsJSON,
		&importRecord.CreatedAt,
		&completedAt,
	); err != nil {
		return datasets.DatasetMetadataImport{}, normalizeSQLError(err)
	}
	if len(summaryJSON) > 0 {
		if err := json.Unmarshal(summaryJSON, &importRecord.Summary); err != nil {
			return datasets.DatasetMetadataImport{}, fmt.Errorf("unmarshal dataset metadata summary: %w", err)
		}
	}
	if len(agentSafeSummaryJSON) > 0 {
		if err := json.Unmarshal(agentSafeSummaryJSON, &importRecord.AgentSafeSummary); err != nil {
			return datasets.DatasetMetadataImport{}, fmt.Errorf("unmarshal dataset metadata agent-safe summary: %w", err)
		}
	}
	importRecord.Warnings = []datasets.MetadataIssue{}
	if len(warningsJSON) > 0 {
		if err := json.Unmarshal(warningsJSON, &importRecord.Warnings); err != nil {
			return datasets.DatasetMetadataImport{}, fmt.Errorf("unmarshal dataset metadata warnings: %w", err)
		}
	}
	importRecord.Errors = []datasets.MetadataIssue{}
	if len(errorsJSON) > 0 {
		if err := json.Unmarshal(errorsJSON, &importRecord.Errors); err != nil {
			return datasets.DatasetMetadataImport{}, fmt.Errorf("unmarshal dataset metadata errors: %w", err)
		}
	}
	if completedAt.Valid {
		importRecord.CompletedAt = &completedAt.Time
	}
	return importRecord, nil
}

func scanDatasetMetadataSource(row rowScanner) (datasets.DatasetMetadataSource, error) {
	var source datasets.DatasetMetadataSource
	var warningsJSON []byte
	var errorsJSON []byte
	var rawPreviewJSON []byte
	if err := row.Scan(
		&source.ID,
		&source.ImportID,
		&source.DatasetID,
		&source.RelativePath,
		&source.StorageURI,
		&source.ChecksumSHA256,
		&source.SizeBytes,
		&source.DeclaredFormat,
		&source.DetectedFormat,
		&source.ParserName,
		&source.ParserVersion,
		&source.Status,
		&warningsJSON,
		&errorsJSON,
		&rawPreviewJSON,
	); err != nil {
		return datasets.DatasetMetadataSource{}, normalizeSQLError(err)
	}
	source.Warnings = []datasets.MetadataIssue{}
	if len(warningsJSON) > 0 {
		if err := json.Unmarshal(warningsJSON, &source.Warnings); err != nil {
			return datasets.DatasetMetadataSource{}, fmt.Errorf("unmarshal dataset metadata source warnings: %w", err)
		}
	}
	source.Errors = []datasets.MetadataIssue{}
	if len(errorsJSON) > 0 {
		if err := json.Unmarshal(errorsJSON, &source.Errors); err != nil {
			return datasets.DatasetMetadataSource{}, fmt.Errorf("unmarshal dataset metadata source errors: %w", err)
		}
	}
	source.RawPreview = map[string]any{}
	if len(rawPreviewJSON) > 0 {
		if err := json.Unmarshal(rawPreviewJSON, &source.RawPreview); err != nil {
			return datasets.DatasetMetadataSource{}, fmt.Errorf("unmarshal dataset metadata source raw preview: %w", err)
		}
	}
	return source, nil
}

func scanDatasetClass(row rowScanner) (datasets.DatasetClass, error) {
	var class datasets.DatasetClass
	var classIndex sql.NullInt64
	var metadataJSON []byte
	if err := row.Scan(
		&class.ID,
		&class.ImportID,
		&class.DatasetID,
		&class.ClassKey,
		&class.ClassName,
		&classIndex,
		&class.ParentClassKey,
		&class.SourceID,
		&metadataJSON,
	); err != nil {
		return datasets.DatasetClass{}, normalizeSQLError(err)
	}
	if classIndex.Valid {
		value := int(classIndex.Int64)
		class.ClassIndex = &value
	}
	class.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &class.Metadata); err != nil {
			return datasets.DatasetClass{}, fmt.Errorf("unmarshal dataset class metadata: %w", err)
		}
	}
	return class, nil
}

func scanDatasetManifestRecord(row rowScanner) (datasets.DatasetManifestRecord, error) {
	var record datasets.DatasetManifestRecord
	var metadataJSON []byte
	if err := row.Scan(
		&record.ID,
		&record.ImportID,
		&record.DatasetID,
		&record.SampleKey,
		&record.MediaType,
		&record.RelativePath,
		&record.StorageURI,
		&record.LabelKey,
		&record.LabelName,
		&record.Split,
		&record.Width,
		&record.Height,
		&record.DurationMS,
		&record.FrameCount,
		&record.ChecksumSHA256,
		&record.SourceID,
		&metadataJSON,
	); err != nil {
		return datasets.DatasetManifestRecord{}, normalizeSQLError(err)
	}
	record.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &record.Metadata); err != nil {
			return datasets.DatasetManifestRecord{}, fmt.Errorf("unmarshal dataset manifest metadata: %w", err)
		}
	}
	return record, nil
}

func scanDatasetAnnotation(row rowScanner) (datasets.DatasetAnnotation, error) {
	var annotation datasets.DatasetAnnotation
	var bboxJSON []byte
	var metadataJSON []byte
	var confidence sql.NullFloat64
	if err := row.Scan(
		&annotation.ID,
		&annotation.ImportID,
		&annotation.DatasetID,
		&annotation.SampleKey,
		&annotation.AnnotationType,
		&annotation.LabelKey,
		&annotation.LabelName,
		&bboxJSON,
		&confidence,
		&annotation.SourceID,
		&metadataJSON,
	); err != nil {
		return datasets.DatasetAnnotation{}, normalizeSQLError(err)
	}
	annotation.BBox = map[string]any{}
	if len(bboxJSON) > 0 {
		if err := json.Unmarshal(bboxJSON, &annotation.BBox); err != nil {
			return datasets.DatasetAnnotation{}, fmt.Errorf("unmarshal dataset annotation bbox: %w", err)
		}
	}
	if confidence.Valid {
		annotation.Confidence = &confidence.Float64
	}
	annotation.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &annotation.Metadata); err != nil {
			return datasets.DatasetAnnotation{}, fmt.Errorf("unmarshal dataset annotation metadata: %w", err)
		}
	}
	return annotation, nil
}

func scanDatasetSplit(row rowScanner) (datasets.DatasetSplit, error) {
	var split datasets.DatasetSplit
	var metadataJSON []byte
	if err := row.Scan(
		&split.ID,
		&split.ImportID,
		&split.DatasetID,
		&split.SplitName,
		&split.SampleCount,
		&split.SourceID,
		&metadataJSON,
	); err != nil {
		return datasets.DatasetSplit{}, normalizeSQLError(err)
	}
	split.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &split.Metadata); err != nil {
			return datasets.DatasetSplit{}, fmt.Errorf("unmarshal dataset split metadata: %w", err)
		}
	}
	return split, nil
}

func scanDatasetVisualAnalysis(row rowScanner) (datasets.DatasetVisualAnalysis, error) {
	var analysis datasets.DatasetVisualAnalysis
	var triggerDetailsJSON []byte
	var coverageReportJSON []byte
	var classesToWatchJSON []byte
	var visualTraitsJSON []byte
	var preprocessingHypothesesJSON []byte
	var cautionsJSON []byte
	var limitationsJSON []byte
	var validationErrorsJSON []byte
	var triggerReason string

	if err := row.Scan(
		&analysis.ID,
		&analysis.ProjectID,
		&analysis.DatasetID,
		&analysis.DatasetName,
		&analysis.SchemaVersion,
		&analysis.AnalysisVersion,
		&analysis.PromptVersion,
		&analysis.AgentName,
		&analysis.AgentVersion,
		&analysis.Provider,
		&analysis.Model,
		&triggerReason,
		&triggerDetailsJSON,
		&analysis.SourceJobID,
		&analysis.SourceInvocationID,
		&analysis.ProfileSchemaVersion,
		&analysis.ProfileFingerprint,
		&analysis.TotalImages,
		&analysis.ImagesAnalyzed,
		&coverageReportJSON,
		&classesToWatchJSON,
		&analysis.Confidence,
		&visualTraitsJSON,
		&preprocessingHypothesesJSON,
		&cautionsJSON,
		&limitationsJSON,
		&analysis.ValidationStatus,
		&validationErrorsJSON,
		&analysis.CreatedAt,
		&analysis.UpdatedAt,
	); err != nil {
		return datasets.DatasetVisualAnalysis{}, normalizeSQLError(err)
	}

	analysis.TriggerReason = datasets.VisualReanalysisTrigger(triggerReason)
	analysis.TriggerDetails = map[string]any{}
	if len(triggerDetailsJSON) > 0 {
		if err := json.Unmarshal(triggerDetailsJSON, &analysis.TriggerDetails); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis trigger details: %w", err)
		}
	}
	if len(coverageReportJSON) > 0 {
		if err := json.Unmarshal(coverageReportJSON, &analysis.CoverageReport); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis coverage report: %w", err)
		}
	}
	analysis.ClassesToWatch = []datasets.ClassWatchItem{}
	if len(classesToWatchJSON) > 0 {
		if err := json.Unmarshal(classesToWatchJSON, &analysis.ClassesToWatch); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis classes to watch: %w", err)
		}
	}
	analysis.VisualTraits = []datasets.VisualTrait{}
	if len(visualTraitsJSON) > 0 {
		if err := json.Unmarshal(visualTraitsJSON, &analysis.VisualTraits); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis traits: %w", err)
		}
	}
	analysis.PreprocessingHypotheses = []datasets.PreprocessingHypothesis{}
	if len(preprocessingHypothesesJSON) > 0 {
		if err := json.Unmarshal(preprocessingHypothesesJSON, &analysis.PreprocessingHypotheses); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis preprocessing hypotheses: %w", err)
		}
	}
	analysis.Cautions = []datasets.VisualCaution{}
	if len(cautionsJSON) > 0 {
		if err := json.Unmarshal(cautionsJSON, &analysis.Cautions); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis cautions: %w", err)
		}
	}
	analysis.Limitations = []string{}
	if len(limitationsJSON) > 0 {
		if err := json.Unmarshal(limitationsJSON, &analysis.Limitations); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis limitations: %w", err)
		}
	}
	analysis.ValidationErrors = []string{}
	if len(validationErrorsJSON) > 0 {
		if err := json.Unmarshal(validationErrorsJSON, &analysis.ValidationErrors); err != nil {
			return datasets.DatasetVisualAnalysis{}, fmt.Errorf("unmarshal visual analysis validation errors: %w", err)
		}
	}

	return analysis, nil
}

func scanWorker(row rowScanner) (workers.Worker, error) {
	var worker workers.Worker
	if err := row.Scan(
		&worker.ID,
		&worker.ProjectID,
		&worker.Name,
		&worker.Status,
		&worker.GPUType,
		&worker.LastHeartbeat,
		&worker.CurrentJobID,
	); err != nil {
		return workers.Worker{}, normalizeSQLError(err)
	}

	if time.Since(worker.LastHeartbeat) > workers.HeartbeatLimit {
		worker.Status = workers.StatusOffline
	}

	return worker, nil
}

func scanJob(row rowScanner) (jobs.ExperimentJob, error) {
	var job jobs.ExperimentJob
	var configJSON []byte
	var startedAt sql.NullTime
	var completedAt sql.NullTime
	var leaseExpiresAt sql.NullTime
	var leaseLastHeartbeatAt sql.NullTime

	if err := row.Scan(
		&job.ID,
		&job.ProjectID,
		&job.WorkerID,
		&job.Template,
		&job.Status,
		&configJSON,
		&job.MLflowRunID,
		&job.Error,
		&job.Attempt,
		&job.MaxAttempts,
		&job.LeaseOwnerWorkerID,
		&leaseExpiresAt,
		&leaseLastHeartbeatAt,
		&job.CreatedAt,
		&startedAt,
		&completedAt,
	); err != nil {
		return jobs.ExperimentJob{}, normalizeSQLError(err)
	}

	if len(configJSON) > 0 {
		if err := json.Unmarshal(configJSON, &job.Config); err != nil {
			return jobs.ExperimentJob{}, fmt.Errorf("unmarshal job config: %w", err)
		}
	}

	if startedAt.Valid {
		job.StartedAt = &startedAt.Time
	}
	if completedAt.Valid {
		job.CompletedAt = &completedAt.Time
	}
	if leaseExpiresAt.Valid {
		job.LeaseExpiresAt = &leaseExpiresAt.Time
	}
	if leaseLastHeartbeatAt.Valid {
		job.LeaseLastHeartbeatAt = &leaseLastHeartbeatAt.Time
	}
	if job.MaxAttempts < 1 {
		job.MaxAttempts = defaultJobMaxAttempts
	}

	return job, nil
}

func scanMetric(row rowScanner) (jobs.EpochMetric, error) {
	var metric jobs.EpochMetric
	var metricsJSON []byte

	if err := row.Scan(&metric.JobID, &metric.Epoch, &metricsJSON, &metric.CreatedAt); err != nil {
		return jobs.EpochMetric{}, normalizeSQLError(err)
	}

	if err := json.Unmarshal(metricsJSON, &metric.Metrics); err != nil {
		return jobs.EpochMetric{}, fmt.Errorf("unmarshal metrics: %w", err)
	}

	return metric, nil
}

func scanTrainingRunSummary(row rowScanner) (runs.TrainingRunSummary, error) {
	var summary runs.TrainingRunSummary
	var datasetMaterializationJSON []byte
	var stageTelemetryJSON []byte
	if err := row.Scan(
		&summary.JobID,
		&summary.ProjectID,
		&summary.PlanID,
		&summary.DatasetID,
		&summary.Model,
		&summary.Provider,
		&summary.GPUType,
		&summary.Status,
		&summary.RuntimeSeconds,
		&summary.EstimatedCostUSD,
		&summary.BestMacroF1,
		&summary.BestAccuracy,
		&summary.FinalTrainLoss,
		&summary.FinalValLoss,
		&summary.EpochsCompleted,
		&summary.ModalFunctionCallID,
		&summary.ModalInputID,
		&datasetMaterializationJSON,
		&stageTelemetryJSON,
		&summary.CreatedAt,
		&summary.UpdatedAt,
	); err != nil {
		return runs.TrainingRunSummary{}, normalizeSQLError(err)
	}
	summary.DatasetMaterialization = map[string]any{}
	if len(datasetMaterializationJSON) > 0 {
		if err := json.Unmarshal(datasetMaterializationJSON, &summary.DatasetMaterialization); err != nil {
			return runs.TrainingRunSummary{}, fmt.Errorf("unmarshal dataset materialization: %w", err)
		}
	}
	summary.StageTelemetry = map[string]any{}
	if len(stageTelemetryJSON) > 0 {
		if err := json.Unmarshal(stageTelemetryJSON, &summary.StageTelemetry); err != nil {
			return runs.TrainingRunSummary{}, fmt.Errorf("unmarshal stage telemetry: %w", err)
		}
	}

	return summary, nil
}

func scanTrainingRunEvaluation(row rowScanner) (runs.TrainingRunEvaluation, error) {
	var evaluation runs.TrainingRunEvaluation
	var objectiveProfileJSON []byte
	var perClassMetricsJSON []byte
	var confusionMatrixJSON []byte
	var modelProfileJSON []byte
	var holisticScoresJSON []byte
	if err := row.Scan(
		&evaluation.JobID,
		&evaluation.ProjectID,
		&evaluation.PlanID,
		&evaluation.DatasetID,
		&objectiveProfileJSON,
		&perClassMetricsJSON,
		&confusionMatrixJSON,
		&modelProfileJSON,
		&holisticScoresJSON,
		&evaluation.RecommendationSummary,
		&evaluation.CreatedAt,
		&evaluation.UpdatedAt,
	); err != nil {
		return runs.TrainingRunEvaluation{}, normalizeSQLError(err)
	}

	evaluation.ObjectiveProfile = map[string]any{}
	if len(objectiveProfileJSON) > 0 {
		if err := json.Unmarshal(objectiveProfileJSON, &evaluation.ObjectiveProfile); err != nil {
			return runs.TrainingRunEvaluation{}, fmt.Errorf("unmarshal objective profile: %w", err)
		}
	}
	evaluation.PerClassMetrics = map[string]any{}
	if len(perClassMetricsJSON) > 0 {
		if err := json.Unmarshal(perClassMetricsJSON, &evaluation.PerClassMetrics); err != nil {
			return runs.TrainingRunEvaluation{}, fmt.Errorf("unmarshal per-class metrics: %w", err)
		}
	}
	evaluation.ConfusionMatrix = [][]int{}
	if len(confusionMatrixJSON) > 0 {
		if err := json.Unmarshal(confusionMatrixJSON, &evaluation.ConfusionMatrix); err != nil {
			return runs.TrainingRunEvaluation{}, fmt.Errorf("unmarshal confusion matrix: %w", err)
		}
	}
	evaluation.ModelProfile = map[string]any{}
	if len(modelProfileJSON) > 0 {
		if err := json.Unmarshal(modelProfileJSON, &evaluation.ModelProfile); err != nil {
			return runs.TrainingRunEvaluation{}, fmt.Errorf("unmarshal model profile: %w", err)
		}
	}
	evaluation.HolisticScores = map[string]any{}
	if len(holisticScoresJSON) > 0 {
		if err := json.Unmarshal(holisticScoresJSON, &evaluation.HolisticScores); err != nil {
			return runs.TrainingRunEvaluation{}, fmt.Errorf("unmarshal holistic scores: %w", err)
		}
	}
	return evaluation, nil
}

func scanProjectChampion(row rowScanner) (runs.ProjectChampion, error) {
	var champion runs.ProjectChampion
	var metricsJSON []byte
	var evaluationJSON []byte
	var deploymentProfileJSON []byte
	if err := row.Scan(
		&champion.ID,
		&champion.ProjectID,
		&champion.DatasetID,
		&champion.PlanID,
		&champion.JobID,
		&champion.SourceDecisionID,
		&champion.SelectionReason,
		&metricsJSON,
		&evaluationJSON,
		&deploymentProfileJSON,
		&champion.CreatedAt,
		&champion.UpdatedAt,
	); err != nil {
		return runs.ProjectChampion{}, normalizeSQLError(err)
	}

	champion.Metrics = map[string]any{}
	if len(metricsJSON) > 0 {
		if err := json.Unmarshal(metricsJSON, &champion.Metrics); err != nil {
			return runs.ProjectChampion{}, fmt.Errorf("unmarshal champion metrics: %w", err)
		}
	}
	champion.Evaluation = map[string]any{}
	if len(evaluationJSON) > 0 {
		if err := json.Unmarshal(evaluationJSON, &champion.Evaluation); err != nil {
			return runs.ProjectChampion{}, fmt.Errorf("unmarshal champion evaluation: %w", err)
		}
	}
	champion.DeploymentProfile = map[string]any{}
	if len(deploymentProfileJSON) > 0 {
		if err := json.Unmarshal(deploymentProfileJSON, &champion.DeploymentProfile); err != nil {
			return runs.ProjectChampion{}, fmt.Errorf("unmarshal champion deployment profile: %w", err)
		}
	}
	return champion, nil
}

func scanChampionExport(row rowScanner) (runs.ChampionExport, error) {
	var export runs.ChampionExport
	var metadataJSON []byte
	var validationErrorsJSON []byte
	if err := row.Scan(
		&export.ID,
		&export.ProjectID,
		&export.ChampionID,
		&export.JobID,
		&export.Status,
		&export.Format,
		&export.ArtifactURI,
		&metadataJSON,
		&validationErrorsJSON,
		&export.CreatedAt,
		&export.UpdatedAt,
	); err != nil {
		return runs.ChampionExport{}, normalizeSQLError(err)
	}

	export.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &export.Metadata); err != nil {
			return runs.ChampionExport{}, fmt.Errorf("unmarshal champion export metadata: %w", err)
		}
	}
	export.ValidationErrors = []string{}
	if len(validationErrorsJSON) > 0 {
		if err := json.Unmarshal(validationErrorsJSON, &export.ValidationErrors); err != nil {
			return runs.ChampionExport{}, fmt.Errorf("unmarshal champion export validation errors: %w", err)
		}
	}
	return export, nil
}

func scanChampionDemoPrediction(row rowScanner) (runs.ChampionDemoPrediction, error) {
	var prediction runs.ChampionDemoPrediction
	var imageMetadataJSON []byte
	var topKJSON []byte
	var confidence sql.NullFloat64
	var latencyMS sql.NullFloat64
	var correct sql.NullBool
	if err := row.Scan(
		&prediction.ID,
		&prediction.ProjectID,
		&prediction.ChampionID,
		&prediction.JobID,
		&prediction.DatasetID,
		&prediction.ImageURI,
		&prediction.ImageID,
		&imageMetadataJSON,
		&prediction.Status,
		&prediction.PredictedLabel,
		&prediction.TrueLabel,
		&confidence,
		&topKJSON,
		&latencyMS,
		&correct,
		&prediction.Error,
		&prediction.CreatedAt,
	); err != nil {
		return runs.ChampionDemoPrediction{}, normalizeSQLError(err)
	}

	prediction.ImageMetadata = map[string]any{}
	if len(imageMetadataJSON) > 0 {
		if err := json.Unmarshal(imageMetadataJSON, &prediction.ImageMetadata); err != nil {
			return runs.ChampionDemoPrediction{}, fmt.Errorf("unmarshal champion demo prediction image metadata: %w", err)
		}
	}
	prediction.TopK = []runs.DemoPredictionTopK{}
	if len(topKJSON) > 0 {
		if err := json.Unmarshal(topKJSON, &prediction.TopK); err != nil {
			return runs.ChampionDemoPrediction{}, fmt.Errorf("unmarshal champion demo prediction top-k: %w", err)
		}
	}
	if confidence.Valid {
		prediction.Confidence = &confidence.Float64
	}
	if latencyMS.Valid {
		prediction.LatencyMS = &latencyMS.Float64
	}
	if correct.Valid {
		prediction.Correct = &correct.Bool
	}
	return prediction, nil
}

func scanChampionFeedback(row rowScanner) (runs.ChampionFeedback, error) {
	var feedback runs.ChampionFeedback
	var predictionSnapshotJSON []byte
	var metricsSnapshotJSON []byte
	var metadataJSON []byte
	if err := row.Scan(
		&feedback.ID,
		&feedback.ProjectID,
		&feedback.ChampionID,
		&feedback.PredictionID,
		&feedback.JobID,
		&feedback.DatasetID,
		&feedback.ImageURI,
		&feedback.ImageID,
		&feedback.Rating,
		&feedback.Message,
		&predictionSnapshotJSON,
		&metricsSnapshotJSON,
		&metadataJSON,
		&feedback.CreatedAt,
	); err != nil {
		return runs.ChampionFeedback{}, normalizeSQLError(err)
	}

	feedback.PredictionSnapshot = map[string]any{}
	if len(predictionSnapshotJSON) > 0 {
		if err := json.Unmarshal(predictionSnapshotJSON, &feedback.PredictionSnapshot); err != nil {
			return runs.ChampionFeedback{}, fmt.Errorf("unmarshal champion feedback prediction snapshot: %w", err)
		}
	}
	feedback.MetricsSnapshot = map[string]any{}
	if len(metricsSnapshotJSON) > 0 {
		if err := json.Unmarshal(metricsSnapshotJSON, &feedback.MetricsSnapshot); err != nil {
			return runs.ChampionFeedback{}, fmt.Errorf("unmarshal champion feedback metrics snapshot: %w", err)
		}
	}
	feedback.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &feedback.Metadata); err != nil {
			return runs.ChampionFeedback{}, fmt.Errorf("unmarshal champion feedback metadata: %w", err)
		}
	}
	return feedback, nil
}

func scanAgentDecision(row rowScanner) (decisions.AgentDecision, error) {
	var decision decisions.AgentDecision
	var payloadJSON []byte

	if err := row.Scan(
		&decision.ID,
		&decision.ProjectID,
		&decision.PlanID,
		&decision.DecisionType,
		&decision.Rationale,
		&payloadJSON,
		&decision.CreatedAt,
	); err != nil {
		return decisions.AgentDecision{}, normalizeSQLError(err)
	}

	decision.Payload = map[string]any{}
	if len(payloadJSON) > 0 {
		if err := json.Unmarshal(payloadJSON, &decision.Payload); err != nil {
			return decisions.AgentDecision{}, fmt.Errorf("unmarshal agent decision payload: %w", err)
		}
	}

	return decision, nil
}

func workerRequirementSelectColumns() string {
	return "id, project_id, plan_id, provider, gpu_type, target_count, status, source, dataset_id, dataset_checksum, dataset_cache_key, dataset_materialization_status, cold_cache_policy, max_concurrent_jobs, max_cold_dataset_materializations, last_error, created_at, updated_at"
}

func scanWorkerRequirement(row rowScanner) (execution.WorkerRequirement, error) {
	var requirement execution.WorkerRequirement
	if err := row.Scan(
		&requirement.ID,
		&requirement.ProjectID,
		&requirement.PlanID,
		&requirement.Provider,
		&requirement.GPUType,
		&requirement.TargetCount,
		&requirement.Status,
		&requirement.Source,
		&requirement.DatasetID,
		&requirement.DatasetChecksum,
		&requirement.DatasetCacheKey,
		&requirement.DatasetMaterializationStatus,
		&requirement.ColdCachePolicy,
		&requirement.MaxConcurrentJobs,
		&requirement.MaxColdDatasetMaterializations,
		&requirement.LastError,
		&requirement.CreatedAt,
		&requirement.UpdatedAt,
	); err != nil {
		return execution.WorkerRequirement{}, normalizeSQLError(err)
	}
	return requirement, nil
}

func scanExecutionEvent(row rowScanner) (execution.ExecutionEvent, error) {
	var event execution.ExecutionEvent
	var payloadJSON []byte
	if err := row.Scan(
		&event.ID,
		&event.ProjectID,
		&event.PlanID,
		&event.EventType,
		&event.Message,
		&payloadJSON,
		&event.CreatedAt,
	); err != nil {
		return execution.ExecutionEvent{}, normalizeSQLError(err)
	}
	event.Payload = map[string]any{}
	if len(payloadJSON) > 0 {
		if err := json.Unmarshal(payloadJSON, &event.Payload); err != nil {
			return execution.ExecutionEvent{}, fmt.Errorf("unmarshal execution event payload: %w", err)
		}
	}
	return event, nil
}

func scanMemoryEmbeddingRecord(row rowScanner) (memory.MemoryEmbeddingRecord, error) {
	var record memory.MemoryEmbeddingRecord
	var embeddingText string
	var embeddingTextHash string
	var summaryJSON []byte
	var metadataJSON []byte
	if err := row.Scan(
		&record.ID,
		&record.SourceTable,
		&record.SourceID,
		&record.ProjectID,
		&record.DatasetID,
		&record.PlanID,
		&record.JobID,
		&record.InvocationID,
		&record.Kind,
		&record.Scope,
		&record.EmbeddingModel,
		&record.EmbeddingDimensions,
		&embeddingText,
		&record.EmbeddingText,
		&embeddingTextHash,
		&summaryJSON,
		&metadataJSON,
		&record.QualityScore,
		&record.OutcomeScore,
		&record.CreatedAt,
		&record.UpdatedAt,
	); err != nil {
		return memory.MemoryEmbeddingRecord{}, normalizeSQLError(err)
	}
	embedding, err := parsePostgresVectorLiteral(embeddingText)
	if err != nil {
		return memory.MemoryEmbeddingRecord{}, fmt.Errorf("parse memory embedding vector: %w", err)
	}
	record.Embedding = embedding
	record.EmbeddingTextHash = embeddingTextHash
	record.SummaryCard = map[string]any{}
	if len(summaryJSON) > 0 {
		if err := json.Unmarshal(summaryJSON, &record.SummaryCard); err != nil {
			return memory.MemoryEmbeddingRecord{}, fmt.Errorf("unmarshal memory embedding summary_card: %w", err)
		}
	}
	record.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &record.Metadata); err != nil {
			return memory.MemoryEmbeddingRecord{}, fmt.Errorf("unmarshal memory embedding metadata: %w", err)
		}
	}
	return record, nil
}

func scanMemoryEmbeddingUsageEvent(row rowScanner) (memory.MemoryEmbeddingUsageEvent, error) {
	var event memory.MemoryEmbeddingUsageEvent
	var providerUsageJSON []byte
	var metadataJSON []byte
	if err := row.Scan(
		&event.ID,
		&event.ProjectID,
		&event.DatasetID,
		&event.PlanID,
		&event.JobID,
		&event.InvocationID,
		&event.Purpose,
		&event.RetrievalPurpose,
		&event.SourceTable,
		&event.SourceID,
		&event.EmbeddingModel,
		&event.EmbeddingDimensions,
		&event.InputBytes,
		&event.ProviderCallCount,
		&event.RetrievedCount,
		&event.Injected,
		&event.LogOnly,
		&event.Cached,
		&event.Skipped,
		&event.SkipReason,
		&event.SourceTextHash,
		&event.QueryHash,
		&providerUsageJSON,
		&metadataJSON,
		&event.CreatedAt,
	); err != nil {
		return memory.MemoryEmbeddingUsageEvent{}, normalizeSQLError(err)
	}
	event.ProviderUsage = map[string]any{}
	if len(providerUsageJSON) > 0 {
		if err := json.Unmarshal(providerUsageJSON, &event.ProviderUsage); err != nil {
			return memory.MemoryEmbeddingUsageEvent{}, fmt.Errorf("unmarshal memory embedding usage provider_usage: %w", err)
		}
	}
	event.Metadata = map[string]any{}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &event.Metadata); err != nil {
			return memory.MemoryEmbeddingUsageEvent{}, fmt.Errorf("unmarshal memory embedding usage metadata: %w", err)
		}
	}
	return event, nil
}

func scanAgentMemoryRecord(row rowScanner) (memory.AgentMemoryRecord, error) {
	var record memory.AgentMemoryRecord
	var payloadJSON []byte
	var tagsJSON []byte
	if err := row.Scan(
		&record.ID,
		&record.InvocationID,
		&record.ProjectID,
		&record.DatasetID,
		&record.PlanID,
		&record.JobID,
		&record.AgentName,
		&record.Kind,
		&record.Summary,
		&payloadJSON,
		&tagsJSON,
		&record.CreatedAt,
	); err != nil {
		return memory.AgentMemoryRecord{}, normalizeSQLError(err)
	}
	record.Payload = map[string]any{}
	if len(payloadJSON) > 0 {
		if err := json.Unmarshal(payloadJSON, &record.Payload); err != nil {
			return memory.AgentMemoryRecord{}, fmt.Errorf("unmarshal agent memory payload: %w", err)
		}
	}
	record.Tags = []string{}
	if len(tagsJSON) > 0 {
		if err := json.Unmarshal(tagsJSON, &record.Tags); err != nil {
			return memory.AgentMemoryRecord{}, fmt.Errorf("unmarshal agent memory tags: %w", err)
		}
	}
	return record, nil
}

func scanAgentInvocation(row rowScanner) (memory.AgentInvocation, error) {
	var invocation memory.AgentInvocation
	var inputMessagesJSON []byte
	var inputContextJSON []byte
	var parsedOutputJSON []byte
	var humanFeedbackJSON []byte
	var downstreamOutcomeJSON []byte

	if err := row.Scan(
		&invocation.ID,
		&invocation.ProjectID,
		&invocation.DatasetID,
		&invocation.PlanID,
		&invocation.JobID,
		&invocation.AgentName,
		&invocation.AgentVersion,
		&invocation.PromptVersion,
		&invocation.Provider,
		&invocation.Model,
		&inputMessagesJSON,
		&inputContextJSON,
		&invocation.RawOutput,
		&parsedOutputJSON,
		&invocation.ValidationStatus,
		&invocation.ValidationError,
		&invocation.AcceptedForMemory,
		&humanFeedbackJSON,
		&downstreamOutcomeJSON,
		&invocation.CreatedAt,
	); err != nil {
		return memory.AgentInvocation{}, normalizeSQLError(err)
	}

	invocation.InputMessages = []map[string]string{}
	if len(inputMessagesJSON) > 0 {
		if err := json.Unmarshal(inputMessagesJSON, &invocation.InputMessages); err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("unmarshal agent invocation input messages: %w", err)
		}
	}
	invocation.InputContext = map[string]any{}
	if len(inputContextJSON) > 0 {
		if err := json.Unmarshal(inputContextJSON, &invocation.InputContext); err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("unmarshal agent invocation input context: %w", err)
		}
	}
	invocation.ParsedOutput = map[string]any{}
	if len(parsedOutputJSON) > 0 {
		if err := json.Unmarshal(parsedOutputJSON, &invocation.ParsedOutput); err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("unmarshal agent invocation parsed output: %w", err)
		}
	}
	invocation.HumanFeedback = map[string]any{}
	if len(humanFeedbackJSON) > 0 {
		if err := json.Unmarshal(humanFeedbackJSON, &invocation.HumanFeedback); err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("unmarshal agent invocation human feedback: %w", err)
		}
	}
	invocation.DownstreamOutcome = map[string]any{}
	if len(downstreamOutcomeJSON) > 0 {
		if err := json.Unmarshal(downstreamOutcomeJSON, &invocation.DownstreamOutcome); err != nil {
			return memory.AgentInvocation{}, fmt.Errorf("unmarshal agent invocation downstream outcome: %w", err)
		}
	}

	return invocation, nil
}

func scanStrategyScorecard(row rowScanner) (strategies.StrategyScorecard, error) {
	var scorecard strategies.StrategyScorecard
	var diagnosisTriggersJSON []byte
	var evidenceUsedJSON []byte
	var datasetTraitsJSON []byte
	var objectiveProfileJSON []byte
	var proposedChangesJSON []byte
	var tagsJSON []byte
	if err := row.Scan(
		&scorecard.ID,
		&scorecard.ProjectID,
		&scorecard.DatasetID,
		&scorecard.SourceDecisionID,
		&scorecard.SourcePlanID,
		&scorecard.FollowUpPlanID,
		&scorecard.StrategyType,
		&scorecard.PlanningMode,
		&scorecard.Mechanism,
		&scorecard.Intervention,
		&diagnosisTriggersJSON,
		&evidenceUsedJSON,
		&scorecard.ExpectedEffect,
		&datasetTraitsJSON,
		&objectiveProfileJSON,
		&proposedChangesJSON,
		&scorecard.ExpectedDelta,
		&scorecard.ActualDelta,
		&scorecard.ConfidenceBefore,
		&scorecard.ConfidenceAfter,
		&scorecard.CostUSD,
		&scorecard.RuntimeSeconds,
		&scorecard.Outcome,
		&scorecard.Lesson,
		&tagsJSON,
		&scorecard.CreatedAt,
	); err != nil {
		return strategies.StrategyScorecard{}, normalizeSQLError(err)
	}
	scorecard.DiagnosisTriggers = []string{}
	if len(diagnosisTriggersJSON) > 0 {
		if err := json.Unmarshal(diagnosisTriggersJSON, &scorecard.DiagnosisTriggers); err != nil {
			return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard diagnosis_triggers: %w", err)
		}
	}
	scorecard.EvidenceUsed = []string{}
	if len(evidenceUsedJSON) > 0 {
		if err := json.Unmarshal(evidenceUsedJSON, &scorecard.EvidenceUsed); err != nil {
			return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard evidence_used: %w", err)
		}
	}
	if err := json.Unmarshal(datasetTraitsJSON, &scorecard.DatasetTraits); err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard dataset_traits: %w", err)
	}
	if err := json.Unmarshal(objectiveProfileJSON, &scorecard.ObjectiveProfile); err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard objective_profile: %w", err)
	}
	if err := json.Unmarshal(proposedChangesJSON, &scorecard.ProposedChanges); err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard proposed_changes: %w", err)
	}
	if err := json.Unmarshal(tagsJSON, &scorecard.Tags); err != nil {
		return strategies.StrategyScorecard{}, fmt.Errorf("unmarshal strategy scorecard tags: %w", err)
	}
	return scorecard, nil
}

func scanOptimizerStudy(row rowScanner) (automl.OptimizerStudy, error) {
	var study automl.OptimizerStudy
	var intentJSON []byte
	var searchSpaceJSON []byte
	var strategyJSON []byte
	if err := row.Scan(
		&study.ID,
		&study.ProjectID,
		&study.PlanID,
		&study.DatasetID,
		&study.SourceDecisionID,
		&study.ExperimentIndex,
		&study.Model,
		&intentJSON,
		&study.Sampler,
		&study.Seed,
		&searchSpaceJSON,
		&strategyJSON,
		&study.CreatedAt,
	); err != nil {
		return automl.OptimizerStudy{}, normalizeSQLError(err)
	}
	if len(intentJSON) > 0 {
		if err := json.Unmarshal(intentJSON, &study.Intent); err != nil {
			return automl.OptimizerStudy{}, fmt.Errorf("unmarshal automl intent: %w", err)
		}
	}
	if len(searchSpaceJSON) > 0 {
		if err := json.Unmarshal(searchSpaceJSON, &study.SearchSpace); err != nil {
			return automl.OptimizerStudy{}, fmt.Errorf("unmarshal automl search space: %w", err)
		}
	}
	study.StrategySnapshot = map[string]any{}
	if len(strategyJSON) > 0 {
		if err := json.Unmarshal(strategyJSON, &study.StrategySnapshot); err != nil {
			return automl.OptimizerStudy{}, fmt.Errorf("unmarshal automl strategy snapshot: %w", err)
		}
	}
	return study, nil
}

func scanOptimizerSuggestion(row rowScanner) (automl.OptimizerSuggestion, error) {
	var suggestion automl.OptimizerSuggestion
	var valuesJSON []byte
	var finalValuesJSON []byte
	var provenanceJSON []byte
	var validationErrorsJSON []byte
	if err := row.Scan(
		&suggestion.ID,
		&suggestion.StudyID,
		&suggestion.ProjectID,
		&suggestion.PlanID,
		&suggestion.DatasetID,
		&suggestion.JobID,
		&suggestion.ExperimentIndex,
		&suggestion.Model,
		&suggestion.Sampler,
		&suggestion.Seed,
		&valuesJSON,
		&finalValuesJSON,
		&provenanceJSON,
		&suggestion.ValidationStatus,
		&validationErrorsJSON,
		&suggestion.CreatedAt,
	); err != nil {
		return automl.OptimizerSuggestion{}, normalizeSQLError(err)
	}
	suggestion.Values = map[string]any{}
	if len(valuesJSON) > 0 {
		if err := json.Unmarshal(valuesJSON, &suggestion.Values); err != nil {
			return automl.OptimizerSuggestion{}, fmt.Errorf("unmarshal automl values: %w", err)
		}
	}
	suggestion.FinalValues = map[string]any{}
	if len(finalValuesJSON) > 0 {
		if err := json.Unmarshal(finalValuesJSON, &suggestion.FinalValues); err != nil {
			return automl.OptimizerSuggestion{}, fmt.Errorf("unmarshal automl final values: %w", err)
		}
	}
	suggestion.Provenance = map[string]automl.HyperparameterProvenance{}
	if len(provenanceJSON) > 0 {
		if err := json.Unmarshal(provenanceJSON, &suggestion.Provenance); err != nil {
			return automl.OptimizerSuggestion{}, fmt.Errorf("unmarshal automl provenance: %w", err)
		}
	}
	suggestion.ValidationErrors = []string{}
	if len(validationErrorsJSON) > 0 {
		if err := json.Unmarshal(validationErrorsJSON, &suggestion.ValidationErrors); err != nil {
			return automl.OptimizerSuggestion{}, fmt.Errorf("unmarshal automl validation errors: %w", err)
		}
	}
	return suggestion, nil
}

func scanOptimizerTrial(row rowScanner) (automl.OptimizerTrial, error) {
	var trial automl.OptimizerTrial
	var metricsJSON []byte
	if err := row.Scan(
		&trial.ID,
		&trial.StudyID,
		&trial.SuggestionID,
		&trial.ProjectID,
		&trial.PlanID,
		&trial.DatasetID,
		&trial.JobID,
		&trial.Status,
		&trial.TargetMetric,
		&trial.Score,
		&metricsJSON,
		&trial.Error,
		&trial.CreatedAt,
		&trial.UpdatedAt,
	); err != nil {
		return automl.OptimizerTrial{}, normalizeSQLError(err)
	}
	trial.Metrics = map[string]any{}
	if len(metricsJSON) > 0 {
		if err := json.Unmarshal(metricsJSON, &trial.Metrics); err != nil {
			return automl.OptimizerTrial{}, fmt.Errorf("unmarshal automl trial metrics: %w", err)
		}
	}
	return trial, nil
}

func scanAutomationSettings(row rowScanner) (settings.AutomationSettings, error) {
	var automationSettings settings.AutomationSettings
	if err := row.Scan(
		&automationSettings.AutoReviewExperiments,
		&automationSettings.AutoScheduleFollowUps,
		&automationSettings.AutoExecutePlans,
		&automationSettings.MaxFollowUpRounds,
		&automationSettings.DefaultTrainingProvider,
		&automationSettings.DefaultGPUType,
		&automationSettings.CostMode,
		&automationSettings.BudgetCapUSD,
		&automationSettings.LLMEnabled,
		&automationSettings.AgentMode,
		&automationSettings.LLMProvider,
		&automationSettings.LLMModel,
		&automationSettings.AutoMLEnabled,
		&automationSettings.AutoMLSampler,
		&automationSettings.UpdatedAt,
	); err != nil {
		return settings.AutomationSettings{}, normalizeSQLError(err)
	}

	return automationSettings, nil
}

func scanExperimentPlan(row rowScanner) (plans.ExperimentPlan, error) {
	var plan plans.ExperimentPlan
	var experimentsJSON []byte
	var warningsJSON []byte

	if err := row.Scan(
		&plan.ID,
		&plan.ProjectID,
		&plan.DatasetID,
		&plan.Status,
		&plan.SourceDecisionID,
		&plan.TargetMetric,
		&plan.RecommendedWorkers,
		&plan.EstimatedMinutes,
		&experimentsJSON,
		&warningsJSON,
		&plan.CreatedAt,
	); err != nil {
		return plans.ExperimentPlan{}, normalizeSQLError(err)
	}

	plan.Experiments = []plans.PlannedExperiment{}
	if len(experimentsJSON) > 0 {
		if err := json.Unmarshal(experimentsJSON, &plan.Experiments); err != nil {
			return plans.ExperimentPlan{}, fmt.Errorf("unmarshal planned experiments: %w", err)
		}
	}
	if plan.Experiments == nil {
		plan.Experiments = []plans.PlannedExperiment{}
	}

	plan.Warnings = []string{}
	if len(warningsJSON) > 0 {
		if err := json.Unmarshal(warningsJSON, &plan.Warnings); err != nil {
			return plans.ExperimentPlan{}, fmt.Errorf("unmarshal experiment plan warnings: %w", err)
		}
	}
	if plan.Warnings == nil {
		plan.Warnings = []string{}
	}

	return plan, nil
}

func normalizeSQLError(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return ErrNotFound
	}

	return err
}

func selectJobSQL(column string) string {
	return fmt.Sprintf(`
		SELECT `+jobSelectColumns()+`
		FROM experiment_jobs
		WHERE %s = $1
	`, column)
}

func jobSelectColumns() string {
	return "id, project_id, worker_id, template, status, config, mlflow_run_id, error, attempt, max_attempts, lease_owner_worker_id, lease_expires_at, lease_last_heartbeat_at, created_at, started_at, completed_at"
}

func datasetVisualAnalysisSelectColumns() string {
	return "id, project_id, dataset_id, dataset_name, schema_version, analysis_version, prompt_version, agent_name, agent_version, provider, model, trigger_reason, trigger_details, source_job_id, source_invocation_id, profile_schema_version, profile_fingerprint, total_images, images_analyzed, coverage_report, classes_to_watch, confidence, visual_traits, preprocessing_hypotheses, cautions, limitations, validation_status, validation_errors, created_at, updated_at"
}

func datasetMetadataImportSelectColumns() string {
	return "id, dataset_id, project_id, status, import_version, dataset_checksum_sha256, parser_registry_version, source_kind, active, strict_mode, summary, agent_safe_summary, warnings, errors, created_at, completed_at"
}

func memoryEmbeddingSelectColumns() string {
	return "id, source_table, source_id, project_id, dataset_id, plan_id, job_id, invocation_id, kind, scope, embedding_model, embedding_dimensions, embedding::text, embedding_text, embedding_text_hash, summary_card, metadata, quality_score, outcome_score, created_at, updated_at"
}

func strategyScorecardSelectColumns() string {
	return "id, project_id, dataset_id, source_decision_id, source_plan_id, followup_plan_id, strategy_type, planning_mode, mechanism, intervention, diagnosis_triggers, evidence_used, expected_effect, dataset_traits, objective_profile, proposed_changes, expected_delta, actual_delta, confidence_before, confidence_after, cost_usd, runtime_seconds, outcome, lesson, tags, created_at"
}

func automlStudySelectColumns() string {
	return "id, project_id, plan_id, dataset_id, source_decision_id, experiment_index, model, intent, sampler, seed, search_space, strategy_snapshot, created_at"
}

func automlSuggestionSelectColumns() string {
	return "id, study_id, project_id, plan_id, dataset_id, job_id, experiment_index, model, sampler, seed, values, final_values, provenance, validation_status, validation_errors, created_at"
}

func automlTrialSelectColumns() string {
	return "id, study_id, suggestion_id, project_id, plan_id, dataset_id, job_id, status, target_metric, score, metrics, error, created_at, updated_at"
}

func newPostgresTrainingRunSummaryFromJob(job jobs.ExperimentJob, now time.Time) runs.TrainingRunSummary {
	provider := postgresConfigString(job.Config, "provider")
	if provider == "" {
		provider = "local"
	}

	return runs.TrainingRunSummary{
		JobID:                  job.ID,
		ProjectID:              job.ProjectID,
		PlanID:                 postgresConfigString(job.Config, "plan_id"),
		DatasetID:              postgresConfigString(job.Config, "dataset_id"),
		Model:                  postgresConfigString(job.Config, "model"),
		Provider:               provider,
		GPUType:                postgresConfigString(job.Config, "gpu_type"),
		Status:                 job.Status,
		DatasetMaterialization: map[string]any{},
		StageTelemetry:         map[string]any{},
		CreatedAt:              now,
		UpdatedAt:              now,
	}
}

func applyPostgresTrainingRunSummaryUpdate(summary *runs.TrainingRunSummary, update runs.TrainingRunSummaryUpdate, now time.Time) {
	if update.Model != "" {
		summary.Model = update.Model
	}
	if update.Provider != "" {
		summary.Provider = update.Provider
	}
	if update.GPUType != "" {
		summary.GPUType = update.GPUType
	}
	if update.Status != "" {
		summary.Status = update.Status
	}
	if update.RuntimeSeconds != nil {
		summary.RuntimeSeconds = *update.RuntimeSeconds
	}
	if update.EstimatedCostUSD != nil {
		summary.EstimatedCostUSD = *update.EstimatedCostUSD
	}
	if update.BestMacroF1 != nil {
		summary.BestMacroF1 = *update.BestMacroF1
	}
	if update.BestAccuracy != nil {
		summary.BestAccuracy = *update.BestAccuracy
	}
	if update.FinalTrainLoss != nil {
		summary.FinalTrainLoss = *update.FinalTrainLoss
	}
	if update.FinalValLoss != nil {
		summary.FinalValLoss = *update.FinalValLoss
	}
	if update.EpochsCompleted != nil {
		summary.EpochsCompleted = *update.EpochsCompleted
	}
	if update.ModalFunctionCallID != "" {
		summary.ModalFunctionCallID = update.ModalFunctionCallID
	}
	if update.ModalInputID != "" {
		summary.ModalInputID = update.ModalInputID
	}
	if update.DatasetMaterialization != nil {
		summary.DatasetMaterialization = copyAnyMap(update.DatasetMaterialization)
	}
	if update.StageTelemetry != nil {
		summary.StageTelemetry = copyAnyMap(update.StageTelemetry)
	}

	summary.UpdatedAt = now
}

func postgresConfigString(config map[string]any, key string) string {
	value, _ := config[key].(string)
	return value
}

func postgresVectorLiteral(values []float32) (string, error) {
	if len(values) == 0 {
		return "", fmt.Errorf("%w: embedding is required", ErrInvalidRequest)
	}
	var builder strings.Builder
	builder.WriteByte('[')
	for index, value := range values {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return "", fmt.Errorf("%w: embedding contains non-finite value", ErrInvalidRequest)
		}
		if index > 0 {
			builder.WriteByte(',')
		}
		builder.WriteString(strconv.FormatFloat(float64(value), 'g', -1, 32))
	}
	builder.WriteByte(']')
	return builder.String(), nil
}

func parsePostgresVectorLiteral(value string) ([]float32, error) {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil, fmt.Errorf("%w: vector text is empty", ErrInvalidRequest)
	}
	value = strings.TrimPrefix(value, "[")
	value = strings.TrimSuffix(value, "]")
	if strings.TrimSpace(value) == "" {
		return nil, fmt.Errorf("%w: vector text is empty", ErrInvalidRequest)
	}
	parts := strings.Split(value, ",")
	out := make([]float32, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		number, err := strconv.ParseFloat(part, 32)
		if err != nil {
			return nil, err
		}
		if math.IsNaN(number) || math.IsInf(number, 0) {
			return nil, fmt.Errorf("%w: vector contains non-finite value", ErrInvalidRequest)
		}
		out = append(out, float32(number))
	}
	return out, nil
}
