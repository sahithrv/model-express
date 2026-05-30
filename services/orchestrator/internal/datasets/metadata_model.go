package datasets

import "time"

const (
	MetadataImportStatusSucceeded = "SUCCEEDED"
	MetadataImportStatusFailed    = "FAILED"
	MetadataImportStatusPartial   = "PARTIAL"

	MetadataSourceStatusParsed             = "parsed"
	MetadataSourceStatusSkippedUnsupported = "skipped_unsupported"
	MetadataSourceStatusError              = "error"

	MetadataSourceKindUploadedSidecar = "uploaded_sidecar"
	MetadataSourceKindWorkerDiscovery = "worker_discovery"
	MetadataSourceKindInferred        = "inferred"

	MetadataFormatImageFolder   = "image_folder"
	MetadataFormatSplitFolder   = "split_folder"
	MetadataFormatCSVManifest   = "csv_manifest"
	MetadataFormatCUBSidecars   = "cub_sidecars"
	MetadataFormatPascalVOC     = "pascal_voc"
	MetadataFormatUnsupported   = "unsupported"
	MetadataFormatUnknown       = "unknown"
	MetadataMediaTypeImage      = "image"
	MetadataMediaTypeVideo      = "video"
	MetadataAnnotationBBox      = "bbox"
	MetadataAnnotationAttribute = "attribute"
	MetadataAnnotationKeypoint  = "keypoint"
	MetadataParserRegistryV1    = "metadata_registry_v1"
)

type DatasetMetadataImport struct {
	ID                    string                          `json:"id"`
	DatasetID             string                          `json:"dataset_id"`
	ProjectID             string                          `json:"project_id"`
	Status                string                          `json:"status"`
	ImportVersion         int                             `json:"import_version"`
	DatasetChecksumSHA256 string                          `json:"dataset_checksum_sha256,omitempty"`
	ParserRegistryVersion string                          `json:"parser_registry_version"`
	SourceKind            string                          `json:"source_kind"`
	Active                bool                            `json:"active"`
	StrictMode            bool                            `json:"strict_mode"`
	Summary               DatasetMetadataSummary          `json:"summary"`
	AgentSafeSummary      AgentSafeDatasetMetadataSummary `json:"agent_safe_summary"`
	Warnings              []MetadataIssue                 `json:"warnings"`
	Errors                []MetadataIssue                 `json:"errors"`
	Sources               []DatasetMetadataSource         `json:"sources,omitempty"`
	Classes               []DatasetClass                  `json:"classes,omitempty"`
	ManifestRecords       []DatasetManifestRecord         `json:"manifest_records,omitempty"`
	Annotations           []DatasetAnnotation             `json:"annotations,omitempty"`
	Splits                []DatasetSplit                  `json:"splits,omitempty"`
	CreatedAt             time.Time                       `json:"created_at"`
	CompletedAt           *time.Time                      `json:"completed_at,omitempty"`
}

type DatasetMetadataSource struct {
	ID             string          `json:"id"`
	ImportID       string          `json:"import_id"`
	DatasetID      string          `json:"dataset_id"`
	RelativePath   string          `json:"relative_path,omitempty"`
	StorageURI     string          `json:"storage_uri,omitempty"`
	ChecksumSHA256 string          `json:"checksum_sha256,omitempty"`
	SizeBytes      int64           `json:"size_bytes"`
	DeclaredFormat string          `json:"declared_format,omitempty"`
	DetectedFormat string          `json:"detected_format,omitempty"`
	ParserName     string          `json:"parser_name,omitempty"`
	ParserVersion  string          `json:"parser_version,omitempty"`
	Status         string          `json:"status"`
	Warnings       []MetadataIssue `json:"warnings"`
	Errors         []MetadataIssue `json:"errors"`
	RawPreview     map[string]any  `json:"raw_preview,omitempty"`
}

type DatasetClass struct {
	ID             string         `json:"id"`
	ImportID       string         `json:"import_id"`
	DatasetID      string         `json:"dataset_id"`
	ClassKey       string         `json:"class_key"`
	ClassName      string         `json:"class_name"`
	ClassIndex     *int           `json:"class_index,omitempty"`
	ParentClassKey string         `json:"parent_class_key,omitempty"`
	SourceID       string         `json:"source_id,omitempty"`
	Metadata       map[string]any `json:"metadata,omitempty"`
}

type DatasetManifestRecord struct {
	ID             string         `json:"id"`
	ImportID       string         `json:"import_id"`
	DatasetID      string         `json:"dataset_id"`
	SampleKey      string         `json:"sample_key"`
	MediaType      string         `json:"media_type"`
	RelativePath   string         `json:"relative_path,omitempty"`
	StorageURI     string         `json:"storage_uri,omitempty"`
	LabelKey       string         `json:"label_key,omitempty"`
	LabelName      string         `json:"label_name,omitempty"`
	Split          string         `json:"split,omitempty"`
	Width          int            `json:"width,omitempty"`
	Height         int            `json:"height,omitempty"`
	DurationMS     int64          `json:"duration_ms,omitempty"`
	FrameCount     int64          `json:"frame_count,omitempty"`
	ChecksumSHA256 string         `json:"checksum_sha256,omitempty"`
	SourceID       string         `json:"source_id,omitempty"`
	Metadata       map[string]any `json:"metadata,omitempty"`
}

type DatasetAnnotation struct {
	ID             string         `json:"id"`
	ImportID       string         `json:"import_id"`
	DatasetID      string         `json:"dataset_id"`
	SampleKey      string         `json:"sample_key"`
	AnnotationType string         `json:"annotation_type"`
	LabelKey       string         `json:"label_key,omitempty"`
	LabelName      string         `json:"label_name,omitempty"`
	BBox           map[string]any `json:"bbox,omitempty"`
	Confidence     *float64       `json:"confidence,omitempty"`
	SourceID       string         `json:"source_id,omitempty"`
	Metadata       map[string]any `json:"metadata,omitempty"`
}

type DatasetSplit struct {
	ID          string         `json:"id"`
	ImportID    string         `json:"import_id"`
	DatasetID   string         `json:"dataset_id"`
	SplitName   string         `json:"split_name"`
	SampleCount int            `json:"sample_count"`
	SourceID    string         `json:"source_id,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

type DatasetMetadataSummary struct {
	DatasetID              string          `json:"dataset_id,omitempty"`
	ImportID               string          `json:"import_id,omitempty"`
	Status                 string          `json:"status,omitempty"`
	ImportVersion          int             `json:"import_version,omitempty"`
	ParserRegistryVersion  string          `json:"parser_registry_version,omitempty"`
	SourceKind             string          `json:"source_kind,omitempty"`
	SourceCount            int             `json:"source_count"`
	ParsedSourceCount      int             `json:"parsed_source_count"`
	UnsupportedSourceCount int             `json:"unsupported_source_count"`
	ErrorSourceCount       int             `json:"error_source_count"`
	SourceFormats          []string        `json:"source_formats,omitempty"`
	ClassCount             int             `json:"class_count"`
	SampleCount            int             `json:"sample_count"`
	LabeledSampleCount     int             `json:"labeled_sample_count"`
	MissingLabelCount      int             `json:"missing_label_count"`
	SplitCounts            map[string]int  `json:"split_counts,omitempty"`
	OfficialSplitAvailable bool            `json:"official_split_available"`
	AnnotationCounts       map[string]int  `json:"annotation_counts,omitempty"`
	BBoxAnnotationCount    int             `json:"bbox_annotation_count"`
	BBoxSampleCount        int             `json:"bbox_sample_count"`
	BBoxCoverageRatio      float64         `json:"bbox_coverage_ratio"`
	Warnings               []MetadataIssue `json:"warnings,omitempty"`
	Errors                 []MetadataIssue `json:"errors,omitempty"`
	Capabilities           map[string]bool `json:"capabilities,omitempty"`
	CreatedAt              *time.Time      `json:"created_at,omitempty"`
	CompletedAt            *time.Time      `json:"completed_at,omitempty"`
}

type AgentSafeDatasetMetadataSummary struct {
	DatasetID              string          `json:"dataset_id,omitempty"`
	ImportID               string          `json:"import_id,omitempty"`
	Status                 string          `json:"status,omitempty"`
	ImportVersion          int             `json:"import_version,omitempty"`
	SourceKind             string          `json:"source_kind,omitempty"`
	SourceCount            int             `json:"source_count"`
	ParsedSourceCount      int             `json:"parsed_source_count"`
	UnsupportedSourceCount int             `json:"unsupported_source_count"`
	ErrorSourceCount       int             `json:"error_source_count"`
	SourceFormats          []string        `json:"source_formats,omitempty"`
	ClassCount             int             `json:"class_count"`
	SampleCount            int             `json:"sample_count"`
	LabeledSampleCount     int             `json:"labeled_sample_count"`
	MissingLabelCount      int             `json:"missing_label_count"`
	SplitCounts            map[string]int  `json:"split_counts,omitempty"`
	OfficialSplitAvailable bool            `json:"official_split_available"`
	AnnotationCounts       map[string]int  `json:"annotation_counts,omitempty"`
	BBoxAnnotationCount    int             `json:"bbox_annotation_count"`
	BBoxSampleCount        int             `json:"bbox_sample_count"`
	BBoxCoverageRatio      float64         `json:"bbox_coverage_ratio"`
	Warnings               []MetadataIssue `json:"warnings,omitempty"`
	Errors                 []MetadataIssue `json:"errors,omitempty"`
	Capabilities           map[string]bool `json:"capabilities,omitempty"`
}

type MetadataIssue struct {
	Severity string `json:"severity"`
	Code     string `json:"code"`
	Message  string `json:"message"`
	SourceID string `json:"source_id,omitempty"`
	Count    int    `json:"count,omitempty"`
}

type DatasetMetadataBundle struct {
	DatasetID       string                  `json:"dataset_id"`
	ImportID        string                  `json:"import_id"`
	ImportVersion   int                     `json:"import_version"`
	Purpose         string                  `json:"purpose"`
	Classes         []DatasetClass          `json:"classes"`
	ManifestRecords []DatasetManifestRecord `json:"manifest_records"`
	Annotations     []DatasetAnnotation     `json:"annotations,omitempty"`
	Splits          []DatasetSplit          `json:"splits"`
	Limit           int                     `json:"limit"`
	Offset          int                     `json:"offset"`
	NextOffset      *int                    `json:"next_offset,omitempty"`
	TotalRecords    int                     `json:"total_records"`
}
