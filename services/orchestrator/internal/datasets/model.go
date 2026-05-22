package datasets

import "time"

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
