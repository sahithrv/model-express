package datasets

import (
	"time"

	"model-express/services/orchestrator/internal/plans"
)

const (
	StatusRegistered = "REGISTERED"
	StatusProfiled   = "PROFILED"
)

type Dataset struct {
	ID             string         `json:"id"`
	ProjectID      string         `json:"project_id"`
	Name           string         `json:"name"`
	StorageURI     string         `json:"storage_uri"`
	ChecksumSHA256 string         `json:"checksum_sha256,omitempty"`
	SizeBytes      int64          `json:"size_bytes"`
	Profile        map[string]any `json:"profile"`
	Status         string         `json:"status"`
	CreatedAt      time.Time      `json:"created_at"`
	ProfiledAt     *time.Time     `json:"profiled_at,omitempty"`
}

type DatasetProfile struct {
	SchemaVersion       string         `json:"schema_version,omitempty"`
	TaskType            string         `json:"task_type"`
	ClassNames          []string       `json:"class_names"`
	ClassCount          int            `json:"class_count"`
	ImageCount          int            `json:"image_count"`
	TotalImages         int            `json:"total_images"`
	ClassDistribution   map[string]int `json:"class_distribution"`
	ImagesPerClass      map[string]int `json:"images_per_class"`
	ImbalanceRatio      float64        `json:"imbalance_ratio"`
	ImageDimensionStats map[string]any `json:"image_dimension_stats"`
	CorruptFileCount    int            `json:"corrupt_file_count"`
	CorruptImageCount   int            `json:"corrupt_image_count"`
	SplitSummary        map[string]any `json:"split_summary"`
	MetadataSummary     map[string]any `json:"metadata_summary"`
	LeakageWarnings     []string       `json:"leakage_warnings"`
	DatasetTraits       []string       `json:"dataset_traits"`
	Artifacts           []Artifact     `json:"artifacts"`
}

type Artifact struct {
	ArtifactType   string         `json:"artifact_type"`
	Path           string         `json:"path"`
	Format         string         `json:"format,omitempty"`
	Description    string         `json:"description,omitempty"`
	DetectedSchema map[string]any `json:"detected_schema,omitempty"`
}

type VisualExemplar struct {
	ID          string         `json:"id,omitempty"`
	ClassName   string         `json:"class_name"`
	URI         string         `json:"uri"`
	Width       int            `json:"width,omitempty"`
	Height      int            `json:"height,omitempty"`
	SizeBytes   int64          `json:"size_bytes,omitempty"`
	MimeType    string         `json:"mime_type,omitempty"`
	Split       string         `json:"split,omitempty"`
	Label       string         `json:"label,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
	Description string         `json:"description,omitempty"`
}

type VisualReanalysisTrigger string

const (
	VisualTriggerInitialProfile       VisualReanalysisTrigger = "initial_profile"
	VisualTriggerDeficiencyReanalysis VisualReanalysisTrigger = "deficiency_reanalysis"
	VisualTriggerManual               VisualReanalysisTrigger = "manual"
)

const (
	VisualAnalysisSchemaVersion = "dataset_visual_analysis_v1"
	VisualAnalysisAgentName     = "visual_dataset_analysis"

	VisualValidationStatusAccepted = "accepted"
	VisualValidationStatusRejected = "rejected"
)

type DatasetVisualAnalysis struct {
	ID                      string                    `json:"id"`
	ProjectID               string                    `json:"project_id"`
	DatasetID               string                    `json:"dataset_id"`
	DatasetName             string                    `json:"dataset_name"`
	SchemaVersion           string                    `json:"schema_version"`
	AnalysisVersion         int                       `json:"analysis_version"`
	PromptVersion           string                    `json:"prompt_version"`
	AgentName               string                    `json:"agent_name"`
	AgentVersion            string                    `json:"agent_version"`
	Provider                string                    `json:"provider"`
	Model                   string                    `json:"model"`
	TriggerReason           VisualReanalysisTrigger   `json:"trigger_reason"`
	TriggerDetails          map[string]any            `json:"trigger_details,omitempty"`
	SourceJobID             string                    `json:"source_job_id,omitempty"`
	SourceInvocationID      string                    `json:"source_invocation_id,omitempty"`
	ProfileSchemaVersion    string                    `json:"profile_schema_version,omitempty"`
	ProfileFingerprint      string                    `json:"profile_fingerprint,omitempty"`
	TotalImages             int                       `json:"total_images"`
	ImagesAnalyzed          int                       `json:"images_analyzed"`
	CoverageReport          VisualCoverageReport      `json:"coverage_report"`
	ClassesToWatch          []ClassWatchItem          `json:"classes_to_watch"`
	Confidence              string                    `json:"confidence"`
	VisualTraits            []VisualTrait             `json:"visual_traits"`
	PreprocessingHypotheses []PreprocessingHypothesis `json:"preprocessing_hypotheses"`
	Cautions                []VisualCaution           `json:"cautions"`
	Limitations             []string                  `json:"limitations"`
	ValidationStatus        string                    `json:"validation_status"`
	ValidationErrors        []string                  `json:"validation_errors,omitempty"`
	CreatedAt               time.Time                 `json:"created_at"`
	UpdatedAt               time.Time                 `json:"updated_at"`
}

type VisualCoverageReport struct {
	SelectionStrategy    string         `json:"selection_strategy"`
	SelectionBasis       []string       `json:"selection_basis"`
	ImagesAvailable      int            `json:"images_available"`
	ImagesAnalyzed       int            `json:"images_analyzed"`
	ClassesTotal         int            `json:"classes_total"`
	ClassesCovered       int            `json:"classes_covered"`
	ClassCoverageRatio   float64        `json:"class_coverage_ratio"`
	PerClassCounts       map[string]int `json:"per_class_counts,omitempty"`
	HardExampleCount     int            `json:"hard_example_count"`
	EdgeCaseCount        int            `json:"edge_case_count"`
	HighDetailImageCount int            `json:"high_detail_image_count"`
	Limitations          []string       `json:"limitations"`
}

type VisualTrait struct {
	Trait           string   `json:"trait"`
	Level           string   `json:"level"`
	Confidence      string   `json:"confidence"`
	Evidence        []string `json:"evidence"`
	ExampleImageIDs []string `json:"example_image_ids,omitempty"`
	AffectedClasses []string `json:"affected_classes,omitempty"`
	Notes           string   `json:"notes,omitempty"`
}

type ClassWatchItem struct {
	ClassName       string   `json:"class_name"`
	Reason          string   `json:"reason"`
	RelatedClasses  []string `json:"related_classes,omitempty"`
	Evidence        []string `json:"evidence"`
	ExampleImageIDs []string `json:"example_image_ids,omitempty"`
	Confidence      string   `json:"confidence"`
}

type PreprocessingHypothesis struct {
	ID                          string                          `json:"id"`
	Mechanism                   string                          `json:"mechanism"`
	Summary                     string                          `json:"summary"`
	Evidence                    []string                        `json:"evidence"`
	SuggestedPreprocessing      *plans.Preprocessing            `json:"suggested_preprocessing,omitempty"`
	SuggestedImageSizes         []int                           `json:"suggested_image_sizes,omitempty"`
	SuggestedAugmentationPolicy string                          `json:"suggested_augmentation_policy,omitempty"`
	SuggestedAugmentationConfig *plans.AugmentationPolicyConfig `json:"suggested_augmentation_policy_config,omitempty"`
	ExpectedEffect              string                          `json:"expected_effect"`
	Risk                        string                          `json:"risk,omitempty"`
	Confidence                  string                          `json:"confidence"`
	SupportStatus               string                          `json:"support_status"`
	UnsupportedReason           string                          `json:"unsupported_reason,omitempty"`
}

type VisualCaution struct {
	Operation       string   `json:"operation"`
	Reason          string   `json:"reason"`
	Severity        string   `json:"severity"`
	Confidence      string   `json:"confidence"`
	AffectedClasses []string `json:"affected_classes,omitempty"`
	ExampleImageIDs []string `json:"example_image_ids,omitempty"`
}

type PlannerVisualExemplarContext struct {
	Enabled                 bool                      `json:"enabled"`
	EvidenceOnly            bool                      `json:"evidence_only"`
	AnalysisID              string                    `json:"analysis_id,omitempty"`
	AnalysisVersion         int                       `json:"analysis_version,omitempty"`
	TriggerReason           string                    `json:"trigger_reason,omitempty"`
	ImagesAnalyzed          int                       `json:"images_analyzed"`
	ClassCoverage           VisualCoverageReport      `json:"class_coverage"`
	Summary                 string                    `json:"summary"`
	ObservedTraits          []string                  `json:"observed_traits"`
	ClassesToWatch          []ClassWatchItem          `json:"classes_to_watch,omitempty"`
	PreprocessingHypotheses []PreprocessingHypothesis `json:"preprocessing_hypotheses,omitempty"`
	Cautions                []VisualCaution           `json:"cautions,omitempty"`
	Limitations             []string                  `json:"limitations,omitempty"`
	RawImagesIncluded       bool                      `json:"raw_images_included"`
	PromptBudget            int                       `json:"prompt_budget"`
}

type VisualSampleManifestItem struct {
	ImageID        string         `json:"image_id"`
	ClassName      string         `json:"class_name,omitempty"`
	ClassLabel     string         `json:"class,omitempty"`
	Width          int            `json:"width,omitempty"`
	Height         int            `json:"height,omitempty"`
	SelectionBasis []string       `json:"selection_basis,omitempty"`
	DetailLevel    string         `json:"detail_level,omitempty"`
	HasBBox        bool           `json:"has_bbox,omitempty"`
	BBox           map[string]any `json:"bbox,omitempty"`
	Metadata       map[string]any `json:"metadata,omitempty"`
}

const (
	ArtifactImageRoot      = "image_root"
	ArtifactClassFolder    = "class_folder"
	ArtifactAnnotationXML  = "annotation_xml"
	ArtifactAnnotationJSON = "annotation_json"
	ArtifactLabelsCSV      = "labels_csv"
	ArtifactSplitFile      = "split_file"
	ArtifactMetadataFolder = "metadata_folder"
	ArtifactBoundingBoxes  = "bounding_boxes"
	ArtifactClassHierarchy = "class_hierarchy"
)
