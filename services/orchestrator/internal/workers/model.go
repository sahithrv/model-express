package workers

import (
	"time"
)

const (
	StatusIdle     = "IDLE"
	StatusRunning  = "RUNNING"
	StatusOffline  = "OFFLINE"
	HeartbeatLimit = 30 * time.Second
)

type Worker struct {
	ID            string    `json:"id"`
	ProjectID     string    `json:"project_id"`
	Name          string    `json:"name"`
	Status        string    `json:"status"`
	GPUType       string    `json:"gpu_type"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
	CurrentJobID  string    `json:"current_job_id,omitempty"`
}
