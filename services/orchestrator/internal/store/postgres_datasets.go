package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/datasets"
	datasetmetadata "model-express/services/orchestrator/internal/datasets/metadata"
)

func (s *PostgresStore) CreateDataset(projectID string, name string, storageURI string, checksumSHA256 string, sizeBytes int64) (datasets.Dataset, error) {
	if err := s.requireProject(projectID); err != nil {
		return datasets.Dataset{}, err
	}
	if name == "" {
		return datasets.Dataset{}, fmt.Errorf("%w: name is required", ErrInvalidRequest)
	}
	if storageURI == "" {
		return datasets.Dataset{}, fmt.Errorf("%w: storage_uri is required", ErrInvalidRequest)
	}

	const query = `
		INSERT INTO datasets (project_id, name, storage_uri, checksum_sha256, size_bytes, status)
		VALUES ($1, $2, $3, $4, $5, $6)
		RETURNING id, project_id, name, storage_uri, checksum_sha256, size_bytes, profile, status, created_at, profiled_at
	`

	return scanDataset(s.db.QueryRowContext(
		context.Background(),
		query,
		projectID,
		name,
		storageURI,
		checksumSHA256,
		sizeBytes,
		datasets.StatusRegistered,
	))
}

func (s *PostgresStore) GetDataset(id string) (datasets.Dataset, error) {
	const query = `
		SELECT id, project_id, name, storage_uri, checksum_sha256, size_bytes, profile, status, created_at, profiled_at
		FROM datasets
		WHERE id = $1
	`

	return scanDataset(s.db.QueryRowContext(context.Background(), query, id))
}

func (s *PostgresStore) ListProjectDatasets(projectID string) ([]datasets.Dataset, error) {
	if err := s.requireProject(projectID); err != nil {
		return nil, err
	}

	const query = `
		SELECT id, project_id, name, storage_uri, checksum_sha256, size_bytes, profile, status, created_at, profiled_at
		FROM datasets
		WHERE project_id = $1
		ORDER BY created_at DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query, projectID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []datasets.Dataset{}
	for rows.Next() {
		dataset, err := scanDataset(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, dataset)
	}

	return out, rows.Err()
}

func (s *PostgresStore) UpdateDatasetProfile(id string, profile map[string]any) (datasets.Dataset, error) {
	profileJSON, err := json.Marshal(profile)
	if err != nil {
		return datasets.Dataset{}, fmt.Errorf("marshal dataset profile: %w", err)
	}

	const query = `
		UPDATE datasets
		SET profile = $1, status = $2, profiled_at = now()
		WHERE id = $3
		RETURNING id, project_id, name, storage_uri, checksum_sha256, size_bytes, profile, status, created_at, profiled_at
	`

	return scanDataset(s.db.QueryRowContext(context.Background(), query, profileJSON, datasets.StatusProfiled, id))
}

func (s *PostgresStore) CreateDatasetMetadataImport(importRecord datasets.DatasetMetadataImport) (datasets.DatasetMetadataImport, error) {
	dataset, err := s.GetDataset(importRecord.DatasetID)
	if err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	if importRecord.ProjectID == "" {
		importRecord.ProjectID = dataset.ProjectID
	}
	if importRecord.ProjectID != dataset.ProjectID {
		return datasets.DatasetMetadataImport{}, fmt.Errorf("%w: dataset_id does not belong to this project", ErrInvalidRequest)
	}
	if importRecord.Status == "" {
		importRecord.Status = datasets.MetadataImportStatusSucceeded
	}
	if importRecord.Status == datasets.MetadataImportStatusFailed {
		importRecord.Active = false
	}
	if importRecord.ParserRegistryVersion == "" {
		importRecord.ParserRegistryVersion = datasets.MetadataParserRegistryV1
	}
	if importRecord.SourceKind == "" {
		importRecord.SourceKind = datasets.MetadataSourceKindUploadedSidecar
	}
	importRecord.DatasetChecksumSHA256 = firstNonEmptyText(importRecord.DatasetChecksumSHA256, dataset.ChecksumSHA256)

	ctx := context.Background()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	defer tx.Rollback()

	if err := tx.QueryRowContext(ctx, `SELECT 'dataset_metadata_import_' || nextval('dataset_metadata_import_id_seq')`).Scan(&importRecord.ID); err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	if err := tx.QueryRowContext(ctx, `
		SELECT COALESCE(MAX(import_version), 0) + 1
		FROM dataset_metadata_imports
		WHERE dataset_id = $1
	`, importRecord.DatasetID).Scan(&importRecord.ImportVersion); err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	now := time.Now().UTC()
	if importRecord.CreatedAt.IsZero() {
		importRecord.CreatedAt = now
	}
	if importRecord.CompletedAt == nil {
		completedAt := now
		importRecord.CompletedAt = &completedAt
	}
	counters := map[string]int{}
	rewriteDatasetMetadataImportIDs(&importRecord, func(prefix string) string {
		counters[prefix]++
		return fmt.Sprintf("%s_%s_%d", importRecord.ID, strings.TrimPrefix(prefix, "dataset_"), counters[prefix])
	})
	importRecord.Summary = datasetmetadata.BuildSummary(importRecord)
	importRecord.AgentSafeSummary = datasetmetadata.BuildAgentSafeSummary(importRecord.Summary)

	summaryJSON, err := json.Marshal(importRecord.Summary)
	if err != nil {
		return datasets.DatasetMetadataImport{}, fmt.Errorf("marshal dataset metadata summary: %w", err)
	}
	agentSafeSummaryJSON, err := json.Marshal(importRecord.AgentSafeSummary)
	if err != nil {
		return datasets.DatasetMetadataImport{}, fmt.Errorf("marshal dataset metadata agent-safe summary: %w", err)
	}
	warningsJSON, err := json.Marshal(importRecord.Warnings)
	if err != nil {
		return datasets.DatasetMetadataImport{}, fmt.Errorf("marshal dataset metadata warnings: %w", err)
	}
	errorsJSON, err := json.Marshal(importRecord.Errors)
	if err != nil {
		return datasets.DatasetMetadataImport{}, fmt.Errorf("marshal dataset metadata errors: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `
		INSERT INTO dataset_metadata_imports (
			id, dataset_id, project_id, status, import_version, dataset_checksum_sha256,
			parser_registry_version, source_kind, active, strict_mode, summary,
			agent_safe_summary, warnings, errors, created_at, completed_at
		)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, false, $9, $10, $11, $12, $13, $14, $15)
	`, importRecord.ID, importRecord.DatasetID, importRecord.ProjectID, importRecord.Status, importRecord.ImportVersion,
		importRecord.DatasetChecksumSHA256, importRecord.ParserRegistryVersion, importRecord.SourceKind,
		importRecord.StrictMode, summaryJSON, agentSafeSummaryJSON, warningsJSON, errorsJSON,
		importRecord.CreatedAt, importRecord.CompletedAt); err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	if err := insertDatasetMetadataSourcesTx(ctx, tx, importRecord.Sources); err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	if err := insertDatasetClassesTx(ctx, tx, importRecord.Classes); err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	if err := insertDatasetManifestRecordsTx(ctx, tx, importRecord.ManifestRecords); err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	if err := insertDatasetAnnotationsTx(ctx, tx, importRecord.Annotations); err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	if err := insertDatasetSplitsTx(ctx, tx, importRecord.Splits); err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	if importRecord.Active {
		if _, err := tx.ExecContext(ctx, `
			UPDATE dataset_metadata_imports
			SET active = false
			WHERE dataset_id = $1 AND active = true
		`, importRecord.DatasetID); err != nil {
			return datasets.DatasetMetadataImport{}, err
		}
		if _, err := tx.ExecContext(ctx, `
			UPDATE dataset_metadata_imports
			SET active = true
			WHERE id = $1
		`, importRecord.ID); err != nil {
			return datasets.DatasetMetadataImport{}, err
		}
	}
	if err := tx.Commit(); err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	return s.getDatasetMetadataImportHeader(importRecord.ID)
}

func (s *PostgresStore) GetDatasetMetadataImport(importID string) (datasets.DatasetMetadataImport, error) {
	importRecord, err := s.getDatasetMetadataImportHeader(importID)
	if err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	return s.hydrateDatasetMetadataImport(importRecord)
}

func (s *PostgresStore) GetActiveDatasetMetadataImport(datasetID string) (datasets.DatasetMetadataImport, error) {
	return s.getActiveDatasetMetadataImportHeader(datasetID)
}

func (s *PostgresStore) ListDatasetMetadataImports(datasetID string) ([]datasets.DatasetMetadataImport, error) {
	if _, err := s.GetDataset(datasetID); err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT `+datasetMetadataImportSelectColumns()+`
		FROM dataset_metadata_imports
		WHERE dataset_id = $1
		ORDER BY import_version DESC, created_at DESC
	`, datasetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []datasets.DatasetMetadataImport{}
	for rows.Next() {
		importRecord, err := scanDatasetMetadataImport(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, importRecord)
	}
	return out, rows.Err()
}

func (s *PostgresStore) GetDatasetMetadataBundle(datasetID string, importID string, includeAnnotations bool, limit int, offset int) (datasets.DatasetMetadataBundle, error) {
	var importRecord datasets.DatasetMetadataImport
	var err error
	if importID == "" {
		importRecord, err = s.getActiveDatasetMetadataImportHeader(datasetID)
	} else {
		importRecord, err = s.getDatasetMetadataImportHeader(importID)
	}
	if err != nil {
		return datasets.DatasetMetadataBundle{}, err
	}
	if importRecord.DatasetID != datasetID {
		return datasets.DatasetMetadataBundle{}, ErrNotFound
	}
	if limit <= 0 || limit > 1000 {
		limit = 1000
	}
	if offset < 0 {
		offset = 0
	}
	var total int
	if err := s.db.QueryRowContext(context.Background(), `
		SELECT COUNT(*)
		FROM dataset_manifest_records
		WHERE dataset_id = $1 AND import_id = $2
	`, datasetID, importRecord.ID).Scan(&total); err != nil {
		return datasets.DatasetMetadataBundle{}, err
	}
	classes, err := s.listDatasetClasses(importRecord.ID)
	if err != nil {
		return datasets.DatasetMetadataBundle{}, err
	}
	records, err := s.listDatasetManifestRecords(importRecord.ID, limit, offset)
	if err != nil {
		return datasets.DatasetMetadataBundle{}, err
	}
	annotations := []datasets.DatasetAnnotation{}
	if includeAnnotations {
		annotations, err = s.listDatasetAnnotations(importRecord.ID)
		if err != nil {
			return datasets.DatasetMetadataBundle{}, err
		}
	}
	splits, err := s.listDatasetSplits(importRecord.ID)
	if err != nil {
		return datasets.DatasetMetadataBundle{}, err
	}
	var nextOffset *int
	if offset+len(records) < total {
		next := offset + len(records)
		nextOffset = &next
	}
	return datasets.DatasetMetadataBundle{
		DatasetID:       datasetID,
		ImportID:        importRecord.ID,
		ImportVersion:   importRecord.ImportVersion,
		Purpose:         "training",
		Classes:         classes,
		ManifestRecords: records,
		Annotations:     annotations,
		Splits:          splits,
		Limit:           limit,
		Offset:          offset,
		NextOffset:      nextOffset,
		TotalRecords:    total,
	}, nil
}

func (s *PostgresStore) getDatasetMetadataImportHeader(importID string) (datasets.DatasetMetadataImport, error) {
	return scanDatasetMetadataImport(s.db.QueryRowContext(context.Background(), `
		SELECT `+datasetMetadataImportSelectColumns()+`
		FROM dataset_metadata_imports
		WHERE id = $1
	`, importID))
}

func (s *PostgresStore) getActiveDatasetMetadataImportHeader(datasetID string) (datasets.DatasetMetadataImport, error) {
	return scanDatasetMetadataImport(s.db.QueryRowContext(context.Background(), `
		SELECT `+datasetMetadataImportSelectColumns()+`
		FROM dataset_metadata_imports
		WHERE dataset_id = $1 AND active = true
		ORDER BY import_version DESC, created_at DESC
		LIMIT 1
	`, datasetID))
}

func (s *PostgresStore) hydrateDatasetMetadataImport(importRecord datasets.DatasetMetadataImport) (datasets.DatasetMetadataImport, error) {
	var err error
	importRecord.Sources, err = s.listDatasetMetadataSources(importRecord.ID)
	if err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	importRecord.Classes, err = s.listDatasetClasses(importRecord.ID)
	if err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	importRecord.ManifestRecords, err = s.listDatasetManifestRecords(importRecord.ID, 0, 0)
	if err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	importRecord.Annotations, err = s.listDatasetAnnotations(importRecord.ID)
	if err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	importRecord.Splits, err = s.listDatasetSplits(importRecord.ID)
	if err != nil {
		return datasets.DatasetMetadataImport{}, err
	}
	return importRecord, nil
}

func insertDatasetMetadataSourcesTx(ctx context.Context, tx *sql.Tx, sources []datasets.DatasetMetadataSource) error {
	for _, source := range sources {
		warningsJSON, err := json.Marshal(source.Warnings)
		if err != nil {
			return fmt.Errorf("marshal dataset metadata source warnings: %w", err)
		}
		errorsJSON, err := json.Marshal(source.Errors)
		if err != nil {
			return fmt.Errorf("marshal dataset metadata source errors: %w", err)
		}
		rawPreviewJSON, err := json.Marshal(emptyMapIfNil(source.RawPreview))
		if err != nil {
			return fmt.Errorf("marshal dataset metadata source raw preview: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO dataset_metadata_sources (
				id, import_id, dataset_id, relative_path, storage_uri, checksum_sha256,
				size_bytes, declared_format, detected_format, parser_name, parser_version,
				status, warnings, errors, raw_preview
			)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
		`, source.ID, source.ImportID, source.DatasetID, source.RelativePath, source.StorageURI,
			source.ChecksumSHA256, source.SizeBytes, source.DeclaredFormat, source.DetectedFormat,
			source.ParserName, source.ParserVersion, source.Status, warningsJSON, errorsJSON, rawPreviewJSON); err != nil {
			return err
		}
	}
	return nil
}

func insertDatasetClassesTx(ctx context.Context, tx *sql.Tx, classes []datasets.DatasetClass) error {
	for _, class := range classes {
		metadataJSON, err := json.Marshal(emptyMapIfNil(class.Metadata))
		if err != nil {
			return fmt.Errorf("marshal dataset class metadata: %w", err)
		}
		var classIndex any
		if class.ClassIndex != nil {
			classIndex = *class.ClassIndex
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO dataset_classes (
				id, import_id, dataset_id, class_key, class_name, class_index,
				parent_class_key, source_id, metadata
			)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
		`, class.ID, class.ImportID, class.DatasetID, class.ClassKey, class.ClassName, classIndex,
			class.ParentClassKey, class.SourceID, metadataJSON); err != nil {
			return err
		}
	}
	return nil
}

func insertDatasetManifestRecordsTx(ctx context.Context, tx *sql.Tx, records []datasets.DatasetManifestRecord) error {
	for _, record := range records {
		metadataJSON, err := json.Marshal(emptyMapIfNil(record.Metadata))
		if err != nil {
			return fmt.Errorf("marshal dataset manifest metadata: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO dataset_manifest_records (
				id, import_id, dataset_id, sample_key, media_type, relative_path,
				storage_uri, label_key, label_name, split, width, height, duration_ms,
				frame_count, checksum_sha256, source_id, metadata
			)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
		`, record.ID, record.ImportID, record.DatasetID, record.SampleKey, record.MediaType,
			record.RelativePath, record.StorageURI, record.LabelKey, record.LabelName, record.Split,
			record.Width, record.Height, record.DurationMS, record.FrameCount, record.ChecksumSHA256,
			record.SourceID, metadataJSON); err != nil {
			return err
		}
	}
	return nil
}

func insertDatasetAnnotationsTx(ctx context.Context, tx *sql.Tx, annotations []datasets.DatasetAnnotation) error {
	for _, annotation := range annotations {
		bboxJSON, err := json.Marshal(emptyMapIfNil(annotation.BBox))
		if err != nil {
			return fmt.Errorf("marshal dataset annotation bbox: %w", err)
		}
		metadataJSON, err := json.Marshal(emptyMapIfNil(annotation.Metadata))
		if err != nil {
			return fmt.Errorf("marshal dataset annotation metadata: %w", err)
		}
		var confidence any
		if annotation.Confidence != nil {
			confidence = *annotation.Confidence
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO dataset_annotations (
				id, import_id, dataset_id, sample_key, annotation_type, label_key,
				label_name, bbox, confidence, source_id, metadata
			)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
		`, annotation.ID, annotation.ImportID, annotation.DatasetID, annotation.SampleKey,
			annotation.AnnotationType, annotation.LabelKey, annotation.LabelName, bboxJSON,
			confidence, annotation.SourceID, metadataJSON); err != nil {
			return err
		}
	}
	return nil
}

func insertDatasetSplitsTx(ctx context.Context, tx *sql.Tx, splits []datasets.DatasetSplit) error {
	for _, split := range splits {
		metadataJSON, err := json.Marshal(emptyMapIfNil(split.Metadata))
		if err != nil {
			return fmt.Errorf("marshal dataset split metadata: %w", err)
		}
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO dataset_splits (
				id, import_id, dataset_id, split_name, sample_count, source_id, metadata
			)
			VALUES ($1, $2, $3, $4, $5, $6, $7)
		`, split.ID, split.ImportID, split.DatasetID, split.SplitName, split.SampleCount,
			split.SourceID, metadataJSON); err != nil {
			return err
		}
	}
	return nil
}

func (s *PostgresStore) listDatasetMetadataSources(importID string) ([]datasets.DatasetMetadataSource, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT id, import_id, dataset_id, relative_path, storage_uri, checksum_sha256,
			size_bytes, declared_format, detected_format, parser_name, parser_version,
			status, warnings, errors, raw_preview
		FROM dataset_metadata_sources
		WHERE import_id = $1
		ORDER BY id
	`, importID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []datasets.DatasetMetadataSource{}
	for rows.Next() {
		source, err := scanDatasetMetadataSource(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, source)
	}
	return out, rows.Err()
}

func (s *PostgresStore) listDatasetClasses(importID string) ([]datasets.DatasetClass, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT id, import_id, dataset_id, class_key, class_name, class_index,
			parent_class_key, source_id, metadata
		FROM dataset_classes
		WHERE import_id = $1
		ORDER BY class_index NULLS LAST, class_key
	`, importID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []datasets.DatasetClass{}
	for rows.Next() {
		class, err := scanDatasetClass(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, class)
	}
	return out, rows.Err()
}

func (s *PostgresStore) listDatasetManifestRecords(importID string, limit int, offset int) ([]datasets.DatasetManifestRecord, error) {
	if limit <= 0 {
		limit = 1000000
	}
	if offset < 0 {
		offset = 0
	}
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT id, import_id, dataset_id, sample_key, media_type, relative_path,
			storage_uri, label_key, label_name, split, width, height, duration_ms,
			frame_count, checksum_sha256, source_id, metadata
		FROM dataset_manifest_records
		WHERE import_id = $1
		ORDER BY sample_key
		LIMIT $2 OFFSET $3
	`, importID, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []datasets.DatasetManifestRecord{}
	for rows.Next() {
		record, err := scanDatasetManifestRecord(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, record)
	}
	return out, rows.Err()
}

func (s *PostgresStore) listDatasetAnnotations(importID string) ([]datasets.DatasetAnnotation, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT id, import_id, dataset_id, sample_key, annotation_type, label_key,
			label_name, bbox, confidence, source_id, metadata
		FROM dataset_annotations
		WHERE import_id = $1
		ORDER BY sample_key, id
	`, importID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []datasets.DatasetAnnotation{}
	for rows.Next() {
		annotation, err := scanDatasetAnnotation(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, annotation)
	}
	return out, rows.Err()
}

func (s *PostgresStore) listDatasetSplits(importID string) ([]datasets.DatasetSplit, error) {
	rows, err := s.db.QueryContext(context.Background(), `
		SELECT id, import_id, dataset_id, split_name, sample_count, source_id, metadata
		FROM dataset_splits
		WHERE import_id = $1
		ORDER BY CASE split_name WHEN 'train' THEN 0 WHEN 'val' THEN 1 WHEN 'test' THEN 2 ELSE 9 END, split_name
	`, importID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []datasets.DatasetSplit{}
	for rows.Next() {
		split, err := scanDatasetSplit(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, split)
	}
	return out, rows.Err()
}

func (s *PostgresStore) CreateDatasetVisualAnalysis(analysis datasets.DatasetVisualAnalysis) (datasets.DatasetVisualAnalysis, error) {
	return s.insertDatasetVisualAnalysis(analysis, datasets.VisualValidationStatusAccepted)
}

func (s *PostgresStore) RejectDatasetVisualAnalysis(analysis datasets.DatasetVisualAnalysis) (datasets.DatasetVisualAnalysis, error) {
	return s.insertDatasetVisualAnalysis(analysis, datasets.VisualValidationStatusRejected)
}

func (s *PostgresStore) GetLatestAcceptedDatasetVisualAnalysis(datasetID string) (datasets.DatasetVisualAnalysis, error) {
	query := `
		SELECT ` + datasetVisualAnalysisSelectColumns() + `
		FROM dataset_visual_analyses
		WHERE dataset_id = $1 AND validation_status = $2
		ORDER BY analysis_version DESC, created_at DESC
		LIMIT 1
	`
	return scanDatasetVisualAnalysis(s.db.QueryRowContext(context.Background(), query, datasetID, datasets.VisualValidationStatusAccepted))
}

func (s *PostgresStore) ListDatasetVisualAnalyses(datasetID string) ([]datasets.DatasetVisualAnalysis, error) {
	if _, err := s.GetDataset(datasetID); err != nil {
		return nil, err
	}

	query := `
		SELECT ` + datasetVisualAnalysisSelectColumns() + `
		FROM dataset_visual_analyses
		WHERE dataset_id = $1
		ORDER BY analysis_version DESC, created_at DESC
	`
	rows, err := s.db.QueryContext(context.Background(), query, datasetID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []datasets.DatasetVisualAnalysis{}
	for rows.Next() {
		analysis, err := scanDatasetVisualAnalysis(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, analysis)
	}
	return out, rows.Err()
}
