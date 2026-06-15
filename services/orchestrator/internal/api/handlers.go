package api

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/settings"
	"model-express/services/orchestrator/internal/store"
)

type createProjectRequest struct {
	Name string `json:"name" binding:"required"`
	Goal string `json:"goal"`
}

type createJobRequest struct {
	Template string         `json:"template" binding:"required"`
	Config   map[string]any `json:"config"`
}

type createDatasetRequest struct {
	Name           string `json:"name" binding:"required"`
	StorageURI     string `json:"storage_uri" binding:"required"`
	ChecksumSHA256 string `json:"checksum_sha256"`
	SizeBytes      int64  `json:"size_bytes"`
}

const (
	callbackTokenHeader = "X-Model-Express-Callback-Token"

	llmExperimentPlannerDecisionSource     = "llm_experiment_planner"
	costPolicyChampionDecisionSource       = "cost_policy_budget_stop"
	userCancelChampionDecisionSource       = "user_cancel_best_available"
	minLLMDecisionConfidence               = 0.50
	maxLLMPlannerExperiments               = 5
	plannerMinimumMeaningfulImprovement    = 0.005
	plannerAutonomousMeaningfulImprovement = 0.010
	championSelectionOverrideMinDelta      = 0.025
	plannerNoImprovementRoundsToSelect     = 2
	plannerDefaultMaxFollowUpRounds        = 10
	plannerAutonomousMaxFollowUpRounds     = 3
	plannerBackendValidationRetryLimit     = 1
	plannerDefaultMaxToolRounds            = 10

	modalOOMRetryHistoryKey = "modal_oom_retry_history"

	visualAnalysisDefaultCooldownMinutes      = 360
	visualAnalysisDefaultMaxRunsPerProfile    = 3
	visualAnalysisDefaultLowMacroF1Threshold  = 0.55
	visualAnalysisDefaultWorstRecallThreshold = 0.40
	visualAnalysisDefaultConfusionThreshold   = 0.20

	datasetMetadataMaxSourceBytes      = 2_000_000
	datasetMetadataMaxTotalSourceBytes = 10_000_000

	memoryRetrievalDefaultMaxCards = 10
	memoryRetrievalDefaultMinScore = 0.55
)

var (
	errNoNovelFollowUpExperiments      = fmt.Errorf("%w: no novel follow-up experiments", store.ErrInvalidRequest)
	errChampionSelectedFollowUpBlocked = fmt.Errorf("%w: champion selected guard", errNoNovelFollowUpExperiments)
	modalGPUEscalationLadder           = []string{"T4", "L4", "A10", "L40S", "A100-40GB", "A100-80GB"}
)

func callbackSecretFromEnv() []byte {
	if secret := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_CALLBACK_SECRET")); secret != "" {
		return []byte(secret)
	}
	if secret := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_API_TOKEN")); secret != "" {
		return []byte(secret)
	}
	secret := make([]byte, 32)
	if _, err := rand.Read(secret); err == nil {
		return secret
	}
	sum := sha256.Sum256([]byte(fmt.Sprintf("model-express-callback-secret:%d", time.Now().UnixNano())))
	return sum[:]
}

func (s *Server) callbackToken(jobID string, trainingAttemptID string) string {
	mac := hmac.New(sha256.New, s.callbackSecret)
	_, _ = io.WriteString(mac, strings.TrimSpace(jobID))
	_, _ = mac.Write([]byte{0})
	_, _ = io.WriteString(mac, strings.TrimSpace(trainingAttemptID))
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func callbackTokenFromRequest(c *gin.Context) string {
	if token := strings.TrimSpace(c.GetHeader(callbackTokenHeader)); token != "" {
		return token
	}
	return bearerToken(c.GetHeader("Authorization"))
}

func secureTokenEqual(actual string, expected string) bool {
	actual = strings.TrimSpace(actual)
	expected = strings.TrimSpace(expected)
	return actual != "" && expected != "" && hmac.Equal([]byte(actual), []byte(expected))
}

const (
	defaultProjectJobsLimit         = 100
	maxProjectJobsLimit             = 500
	defaultJobMetricsLimit          = 200
	maxJobMetricsLimit              = 2000
	defaultTrainingSummariesLimit   = 100
	maxTrainingSummariesLimit       = 500
	defaultTrainingEvaluationsLimit = 50
	maxTrainingEvaluationsLimit     = 200
)

func (s *Server) health(c *gin.Context) {
	c.JSON(http.StatusOK, healthResponse{
		Status:    "ok",
		Service:   "orchestrator",
		Timestamp: time.Now().UTC(),
	})
}

func (s *Server) getAutomationSettings(c *gin.Context) {
	c.JSON(http.StatusOK, s.currentAutomationSettings())
}

func (s *Server) updateAutomationSettings(c *gin.Context) {
	var req settings.AutomationSettingsUpdate
	if !bindJSON(c, &req) {
		return
	}

	current := s.currentAutomationSettings()
	updated, err := applyAutomationSettingsUpdate(current, req)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	saved, err := s.store.SaveAutomationSettings(updated)
	if err != nil {
		writeStoreError(c, err)
		return
	}

	s.setAutomationSettings(saved)
	c.JSON(http.StatusOK, saved)
}

func (s *Server) getProjectChampion(c *gin.Context) {
	champion, err := s.store.GetProjectChampion(c.Param("id"))
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			c.JSON(http.StatusOK, gin.H{"champion": nil})
			return
		}
		writeStoreError(c, err)
		return
	}

	c.JSON(http.StatusOK, gin.H{"champion": champion})
}
