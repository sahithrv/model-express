package store

import (
	"context"

	"model-express/services/orchestrator/internal/policies"
	"model-express/services/orchestrator/internal/projects"
)

func (s *PostgresStore) CreateProject(name string, goal string) (projects.Project, error) {
	const query = `
		INSERT INTO projects (account_id, name, goal, status)
		VALUES ($1, $2, $3, $4)
		RETURNING id, account_id, name, goal, status, created_at, updated_at
	`

	return scanProject(s.db.QueryRowContext(context.Background(), query, policies.LocalDefaultAccountID, name, goal, projects.StatusCreated))
}

func (s *PostgresStore) GetProject(id string) (projects.Project, error) {
	return s.GetProjectContext(context.Background(), id)
}

func (s *PostgresStore) GetProjectContext(ctx context.Context, id string) (projects.Project, error) {
	const query = `
		SELECT id, account_id, name, goal, status, created_at, updated_at
		FROM projects
		WHERE id = $1
	`

	return scanProject(s.db.QueryRowContext(ctx, query, id))
}

func (s *PostgresStore) ListProjects() ([]projects.Project, error) {
	const query = `
		SELECT id, account_id, name, goal, status, created_at, updated_at
		FROM projects
		ORDER BY created_at DESC
	`

	rows, err := s.db.QueryContext(context.Background(), query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	out := []projects.Project{}
	for rows.Next() {
		project, err := scanProject(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, project)
	}

	return out, rows.Err()
}
