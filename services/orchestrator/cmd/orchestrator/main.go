package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"model-express/services/orchestrator/internal/api"
	"model-express/services/orchestrator/internal/config"
	"model-express/services/orchestrator/internal/store"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	repoRoot := os.Getenv("MODEL_EXPRESS_ROOT")
	if repoRoot == "" {
		repoRoot = filepath.Clean(filepath.Join("..", ".."))
	}
	if err := config.LoadRepoEnv(repoRoot); err != nil {
		logger.Warn("failed to load repo env files", "error", err)
	}

	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		databaseURL = "postgres://model_express:model_express@localhost:5432/model_express?sslmode=disable"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	postgresStore, err := store.NewPostgresStore(ctx, databaseURL)
	if err != nil {
		logger.Error("failed to connect to postgres", "error", err)
		os.Exit(1)
	}
	defer postgresStore.Close()

	router := api.NewRouter(postgresStore)

	addr := ":8080"
	server := &http.Server{
		Addr:              addr,
		Handler:           router,
		ReadHeaderTimeout: 5 * time.Second,
	}

	logger.Info("starting orchestrator", "addr", addr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		logger.Error("orchestrator stopped", "error", err)
		os.Exit(1)
	}
}
