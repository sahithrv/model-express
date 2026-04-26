package api

import (
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/store"
)

type Server struct {
	store *store.MemoryStore
}

func NewRouter(store *store.MemoryStore) *gin.Engine {
	server := &Server{store: store}

	router := gin.Default()

	router.GET("/healthz", server.health)

	router.POST("/projects", server.createProject)
	router.GET("/projects", server.listProjects)
	router.GET("/projects/:id", server.getProject)
	router.POST("/projects/:id/jobs", server.createJob)

	router.GET("/jobs/:id", server.getJob)
	router.POST("/jobs/:id/metrics", server.reportMetric)
	router.POST("/jobs/:id/complete", server.completeJob)
	router.POST("/jobs/:id/fail", server.failJob)

	router.POST("/workers/register", server.registerWorker)
	router.POST("/workers/:id/heartbeat", server.heartbeatWorker)
	router.POST("/workers/:id/poll", server.pollJob)

	return router
}

type healthResponse struct {
	Status    string    `json:"status"`
	Service   string    `json:"service"`
	Timestamp time.Time `json:"timestamp"`
}
