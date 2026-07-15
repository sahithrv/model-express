package projects

import (
	"time"
)

const (
	StatusCreated = "CREATED"
)

type Project struct {
	ID        string    `json:"id"`
	AccountID string    `json:"account_id"`
	Name      string    `json:"name"`
	Goal      string    `json:"goal"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}
