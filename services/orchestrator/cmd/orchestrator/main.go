package main

import (
	"log/slog"
	"net/http"
	"os"
	"time"

	"model-express/services/orchestrator/internal/api"
	"model-express/services/orchestrator/internal/store"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

	memoryStore := store.NewMemoryStore()
	router := api.NewRouter(memoryStore)

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
