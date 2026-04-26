package workers

import (
	"time"
)

const (
	StatusIdle    = "IDLE"
	StatusRunning = "RUNNING"
)

type Worker struct {
	ID            string    `json:"id"`
	Name          string    `json:"name"`
	Status        string    `json:"status"`
	GPUType       string    `json:"gpu_type"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
	CurrentJobID  string    `json:"current_job_id,omitempty"`
}
