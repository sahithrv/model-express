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
