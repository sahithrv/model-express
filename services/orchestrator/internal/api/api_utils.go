package api

import (
	"encoding/json"
	"errors"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/llm"
	"model-express/services/orchestrator/internal/plans"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

func copyPayloadMap(input map[string]any) map[string]any {
	out := make(map[string]any, len(input))
	for key, value := range input {
		out[key] = value
	}
	return out
}

func roundDiagnosticFloat(value float64) float64 {
	if !isFiniteFloat(value) {
		return 0
	}
	return math.Round(value*1_000_000) / 1_000_000
}

func isFiniteFloat(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func mapFromStruct(value any) (map[string]any, error) {
	blob, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	out := map[string]any{}
	if err := json.Unmarshal(blob, &out); err != nil {
		return nil, err
	}
	return out, nil
}

func payloadString(payload map[string]any, key string) string {
	value, _ := payload[key].(string)
	return value
}

func payloadStringSlice(payload map[string]any, key string) []string {
	value, ok := payload[key]
	if !ok || value == nil {
		return nil
	}
	if typed, ok := value.([]string); ok {
		return append([]string(nil), typed...)
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	out := []string{}
	if err := json.Unmarshal(blob, &out); err == nil {
		return out
	}
	values, ok := value.([]any)
	if !ok {
		return nil
	}
	for _, item := range values {
		if text, ok := item.(string); ok {
			out = append(out, text)
		}
	}
	return out
}

func nonEmptyStringValues(values []string) []string {
	out := []string{}
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			out = append(out, value)
		}
	}
	return out
}

func payloadMap(payload map[string]any, key string) map[string]any {
	value, ok := payload[key]
	if !ok || value == nil {
		return map[string]any{}
	}
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return map[string]any{}
	}
	out := map[string]any{}
	if err := json.Unmarshal(blob, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func mapFromAny(value any) map[string]any {
	if value == nil {
		return map[string]any{}
	}
	if typed, ok := value.(map[string]any); ok {
		return typed
	}
	blob, err := json.Marshal(value)
	if err != nil {
		return map[string]any{}
	}
	out := map[string]any{}
	if err := json.Unmarshal(blob, &out); err != nil {
		return map[string]any{}
	}
	return out
}

func payloadFloat(payload map[string]any, key string) float64 {
	value, ok := payloadFloatValue(payload, key)
	if !ok {
		return 0
	}
	return value
}

func payloadFloatValue(payload map[string]any, key string) (float64, bool) {
	switch value := payload[key].(type) {
	case float64:
		return value, isFiniteFloat(value)
	case float32:
		out := float64(value)
		return out, isFiniteFloat(out)
	case int:
		return float64(value), true
	case int64:
		return float64(value), true
	case json.Number:
		out, err := value.Float64()
		return out, err == nil && isFiniteFloat(out)
	default:
		return 0, false
	}
}

func payloadBool(payload map[string]any, key string) bool {
	switch value := payload[key].(type) {
	case bool:
		return value
	case string:
		parsed, ok := envFlagValueFromString(value)
		return ok && parsed
	default:
		return false
	}
}

func maxFloat(left float64, right float64) float64 {
	if left > right {
		return left
	}
	return right
}

func minFloat(left float64, right float64) float64 {
	if left < right {
		return left
	}
	return right
}

func maxInt(left int, right int) int {
	if left > right {
		return left
	}
	return right
}

func profileString(profile map[string]any, key string) string {
	value, _ := profile[key].(string)
	return value
}

func profileInt(profile map[string]any, key string) int {
	switch value := profile[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	case json.Number:
		out, _ := value.Int64()
		return int(out)
	default:
		return 0
	}
}

func profileFloat(profile map[string]any, key string) float64 {
	switch value := profile[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case json.Number:
		out, _ := value.Float64()
		return out
	default:
		return 0
	}
}

func profileBool(profile map[string]any, key string) bool {
	switch value := profile[key].(type) {
	case bool:
		return value
	case string:
		normalized := strings.ToLower(strings.TrimSpace(value))
		return normalized == "true" || normalized == "yes" || normalized == "1"
	default:
		return false
	}
}

func profileMap(profile map[string]any, key string) map[string]any {
	value, ok := profile[key].(map[string]any)
	if ok {
		return value
	}
	return map[string]any{}
}

func profileStringSlice(profile map[string]any, key string) []string {
	values, ok := profile[key].([]string)
	if ok {
		return values
	}
	rawValues, ok := profile[key].([]any)
	if !ok {
		return []string{}
	}
	out := []string{}
	for _, raw := range rawValues {
		if value, ok := raw.(string); ok {
			out = append(out, value)
		}
	}
	return out
}

func profileMapSlice(profile map[string]any, key string) []map[string]any {
	values, ok := profile[key].([]map[string]any)
	if ok {
		return values
	}
	rawValues, ok := profile[key].([]any)
	if !ok {
		return []map[string]any{}
	}
	out := []map[string]any{}
	for _, raw := range rawValues {
		if value, ok := raw.(map[string]any); ok {
			out = append(out, value)
		}
	}
	return out
}

func metadataBool(metadata map[string]any, key string) bool {
	switch value := metadata[key].(type) {
	case bool:
		return value
	case string:
		normalized := strings.ToLower(strings.TrimSpace(value))
		return normalized == "true" || normalized == "yes" || normalized == "1"
	default:
		return false
	}
}

func containsString(values []string, target string) bool {
	target = strings.ToLower(strings.TrimSpace(target))
	for _, value := range values {
		if strings.ToLower(strings.TrimSpace(value)) == target {
			return true
		}
	}
	return false
}

func uniqueStrings(values []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		key := strings.ToLower(trimmed)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, trimmed)
	}
	return out
}

func jobsForPlan(projectJobs []jobs.ExperimentJob, planID string) []jobs.ExperimentJob {
	out := []jobs.ExperimentJob{}
	for _, job := range projectJobs {
		if configString(job.Config, "plan_id") == planID {
			out = append(out, job)
		}
	}
	return out
}

func summariesForPlanID(summaries []runs.TrainingRunSummary, planID string) []runs.TrainingRunSummary {
	out := []runs.TrainingRunSummary{}
	for _, summary := range summaries {
		if summary.PlanID == planID {
			out = append(out, summary)
		}
	}
	return out
}

func evaluationsForPlanID(evaluations []runs.TrainingRunEvaluation, planID string) []runs.TrainingRunEvaluation {
	out := []runs.TrainingRunEvaluation{}
	for _, evaluation := range evaluations {
		if evaluation.PlanID == planID {
			out = append(out, evaluation)
		}
	}
	return out
}

func evaluationsByJobID(evaluations []runs.TrainingRunEvaluation) map[string]runs.TrainingRunEvaluation {
	out := map[string]runs.TrainingRunEvaluation{}
	for _, evaluation := range evaluations {
		if strings.TrimSpace(evaluation.JobID) == "" {
			continue
		}
		out[evaluation.JobID] = evaluation
	}
	return out
}

func minInt(left int, right int) int {
	if left < right {
		return left
	}
	return right
}

func configString(config map[string]any, key string) string {
	value, _ := config[key].(string)
	return value
}

func latestExperimentPlan(projectPlans []plans.ExperimentPlan) (plans.ExperimentPlan, bool) {
	if len(projectPlans) == 0 {
		return plans.ExperimentPlan{}, false
	}

	sort.Slice(projectPlans, func(i, j int) bool {
		return projectPlans[i].CreatedAt.After(projectPlans[j].CreatedAt)
	})

	return projectPlans[0], true
}

func followUpPlanForDecision(projectPlans []plans.ExperimentPlan, decisionID string) (plans.ExperimentPlan, bool) {
	for _, plan := range projectPlans {
		if plan.SourceDecisionID == decisionID {
			return plan, true
		}
	}

	return plans.ExperimentPlan{}, false
}

func (s *Server) ensureOpenJob(projectID string, template string, config map[string]any, matches func(jobs.ExperimentJob) bool) (jobs.ExperimentJob, error) {
	projectJobs, err := s.store.ListProjectJobs(projectID)
	if err != nil {
		return jobs.ExperimentJob{}, err
	}
	for _, job := range projectJobs {
		if job.Template != template {
			continue
		}
		if job.Status != jobs.StatusQueued && job.Status != jobs.StatusAssigned && job.Status != jobs.StatusRunning {
			continue
		}
		if matches(job) {
			return job, nil
		}
	}
	return s.store.CreateJob(projectID, template, config)
}

func jobConfigString(config map[string]any, key string) string {
	value, _ := config[key].(string)
	return strings.TrimSpace(value)
}

func copyMap(values map[string]any) map[string]any {
	out := map[string]any{}
	for key, value := range values {
		out[key] = value
	}
	return out
}

func firstString(values map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := values[key]; ok {
			if text, ok := value.(string); ok && strings.TrimSpace(text) != "" {
				return strings.TrimSpace(text)
			}
		}
	}
	return ""
}

func maxInt64(a int64, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func plannerMinimumMeaningfulImprovementFromEnv(agentMode string) float64 {
	value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PLANNER_MIN_MEANINGFUL_DELTA"))
	if value != "" {
		parsed, err := strconv.ParseFloat(value, 64)
		if err == nil && parsed > 0 {
			return parsed
		}
	}
	if strings.EqualFold(agentMode, llm.AgentModeAutonomous) {
		return plannerAutonomousMeaningfulImprovement
	}
	return plannerMinimumMeaningfulImprovement
}

func terminalPlannerGuardsEnabled() bool {
	return envFlag("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", false)
}

func terminalPlannerGuardsEnabledForMode(agentMode string) bool {
	if strings.EqualFold(agentMode, llm.AgentModeAutonomous) {
		return envFlag("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", true)
	}
	return envFlag("MODEL_EXPRESS_TERMINAL_PLANNER_GUARDS", false)
}

func plannerStrictValidationEnabled() bool {
	return envFlag("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION", false)
}

func plannerRelaxedValidationWarning(err error) string {
	if err == nil {
		return ""
	}
	return plannerRelaxedValidationWarningText(err.Error())
}

func plannerRelaxedValidationWarningText(message string) string {
	message = strings.TrimSpace(message)
	message = strings.TrimPrefix(message, store.ErrInvalidRequest.Error()+": ")
	message = strings.TrimPrefix(message, errNoNovelFollowUpExperiments.Error()+": ")
	message = strings.TrimSpace(message)
	if message == "" {
		message = "planner soft validation did not pass"
	}
	if strings.HasPrefix(message, "Relaxed planner validation: ") {
		return message
	}
	return "Relaxed planner validation: " + message
}

func envFlag(name string, defaultValue bool) bool {
	if value, ok := envFlagValue(name); ok {
		return value
	}

	return defaultValue
}

func envFlagValue(name string) (bool, bool) {
	return envFlagValueFromString(os.Getenv(name))
}

func envFlagValueFromString(input string) (bool, bool) {
	value := strings.ToLower(strings.TrimSpace(input))
	switch value {
	case "1", "true", "yes", "on":
		return true, true
	case "0", "false", "no", "off":
		return false, true
	}

	return false, false
}

func envInt(name string, defaultValue int, minValue int, maxValue int) int {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	if parsed < minValue {
		return minValue
	}
	if maxValue >= minValue && parsed > maxValue {
		return maxValue
	}
	return parsed
}

func queryInt(c *gin.Context, key string, defaultValue int, minValue int, maxValue int) int {
	value := strings.TrimSpace(c.Query(key))
	if value == "" {
		return defaultValue
	}
	parsed, err := strconv.Atoi(value)
	if err != nil {
		return defaultValue
	}
	if parsed < minValue {
		return minValue
	}
	if maxValue >= minValue && parsed > maxValue {
		return maxValue
	}
	return parsed
}

func queryBool(c *gin.Context, key string) bool {
	value, ok := envFlagValueFromString(c.Query(key))
	return ok && value
}

func pageHasMore[T any](items []T, limit int) ([]T, bool) {
	if limit <= 0 || len(items) <= limit {
		return items, false
	}
	return items[:limit], true
}

func pageLatestWindowHasMore[T any](items []T, limit int) ([]T, bool) {
	if limit <= 0 || len(items) <= limit {
		return items, false
	}
	return items[len(items)-limit:], true
}

func pagedListPayload(key string, value any, limit int, offset int, hasMore bool) gin.H {
	payload := gin.H{
		key:        value,
		"limit":    limit,
		"offset":   offset,
		"has_more": hasMore,
	}
	if hasMore {
		payload["next_offset"] = offset + limit
	}
	return payload
}

func envFloat(name string, defaultValue float64, minValue float64, maxValue float64) float64 {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return defaultValue
	}
	parsed, err := strconv.ParseFloat(value, 64)
	if err != nil {
		return defaultValue
	}
	if parsed < minValue {
		return minValue
	}
	if maxValue >= minValue && parsed > maxValue {
		return maxValue
	}
	return parsed
}

func bindJSON(c *gin.Context, value any) bool {
	if err := c.ShouldBindJSON(value); err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) || strings.Contains(strings.ToLower(err.Error()), "request body too large") {
			c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "JSON request body too large"})
			return false
		}
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return false
	}

	return true
}

func bindOptionalJSON(c *gin.Context, value any) bool {
	if c.Request.Body == nil {
		return true
	}
	if err := c.ShouldBindJSON(value); err != nil {
		if errors.Is(err, io.EOF) {
			return true
		}
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) || strings.Contains(strings.ToLower(err.Error()), "request body too large") {
			c.JSON(http.StatusRequestEntityTooLarge, gin.H{"error": "JSON request body too large"})
			return false
		}
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return false
	}
	return true
}

func writeStoreError(c *gin.Context, err error) {
	if errors.Is(err, store.ErrNotFound) {
		c.JSON(http.StatusNotFound, gin.H{"error": "not found"})
		return
	}
	if errors.Is(err, store.ErrInvalidRequest) {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
}
