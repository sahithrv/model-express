package calibration

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

const (
	PlannerRolloutAssignmentSchemaVersionV1 = "planner_rollout_assignment_v1"
	PlannerRolloutCohortVersionV1           = "planner_project_cohort_v1"
	PlannerRolloutPolicyVersionV1           = "planner_rollout_policy_v1"
	PlannerRolloutLegacyIdentity            = "legacy_unknown"
	PlannerRolloutLastApprovedPolicyV1      = "planner_policy_ranker_v1_approved"
	PlannerRolloutCandidatePolicyV1         = "planner_policy_ranker_v2_candidate"
	PlannerRolloutRankerV2VariantV1         = "candidate_ranker_v2_v1"

	RolloutDimensionRanker    = "ranker"
	RolloutDimensionPrompt    = "prompt"
	RolloutDimensionContext   = "context"
	RolloutDimensionRetrieval = "retrieval"

	RolloutStateDisabled     = "disabled"
	RolloutStateActive       = "active"
	RolloutStatePaused       = "paused"
	RolloutStateManualReview = "manual_review"
	RolloutStateRolledBack   = "rolled_back"

	RolloutActionHold         = "hold"
	RolloutActionAdvance      = "advance"
	RolloutActionPause        = "pause"
	RolloutActionManualReview = "manual_review"
	RolloutActionApproved     = "approved"
)

var guardedRolloutStages = []int{5, 25, 50, 100}

type PlannerRolloutPolicy struct {
	PolicyVersion        string            `json:"policy_version"`
	Enabled              bool              `json:"enabled"`
	PolicyID             string            `json:"policy_id"`
	LastApprovedPolicyID string            `json:"last_approved_policy_id"`
	StagePercent         int               `json:"stage_percent"`
	State                string            `json:"state"`
	Dimensions           []string          `json:"dimensions"`
	Factorial            bool              `json:"factorial"`
	Rollback             bool              `json:"rollback"`
	VariantValues        map[string]string `json:"variant_values"`
}

type PlannerRolloutAssignment struct {
	SchemaVersion        string            `json:"schema_version"`
	CohortVersion        string            `json:"cohort_version"`
	CohortID             string            `json:"cohort_id"`
	CohortBucket         int               `json:"cohort_bucket"`
	PolicyID             string            `json:"policy_id"`
	LastApprovedPolicyID string            `json:"last_approved_policy_id"`
	StagePercent         int               `json:"stage_percent"`
	State                string            `json:"state"`
	Treatment            bool              `json:"treatment"`
	Dimensions           []string          `json:"dimensions"`
	Factorial            bool              `json:"factorial"`
	VariantValues        map[string]string `json:"variant_values,omitempty"`
	ManualReviewRequired bool              `json:"manual_review_required"`
}

type PlannerRolloutPromotionThresholds struct {
	MinimumSampleSize             int     `json:"minimum_sample_size"`
	MinimumObservedOutcomeSamples int     `json:"minimum_observed_outcome_samples"`
	FirstPassValidityMargin       float64 `json:"first_pass_validity_margin"`
	EventualValidityMargin        float64 `json:"eventual_validity_margin"`
	ObservedOutcomeMargin         float64 `json:"observed_outcome_margin"`
	MaximumCostIncreaseRate       float64 `json:"maximum_cost_increase_rate"`
	MaximumLatencyIncreaseRate    float64 `json:"maximum_latency_increase_rate"`
	MaximumFailureRate            float64 `json:"maximum_failure_rate"`
	MaximumRetryRate              float64 `json:"maximum_retry_rate"`
}

type PlannerRolloutPromotionObservation struct {
	SampleSize                     int     `json:"sample_size"`
	ObservedOutcomeSampleSize      int     `json:"observed_outcome_sample_size"`
	SafetyRegressions              int     `json:"safety_regressions"`
	TaskCompatibilityRegressions   int     `json:"task_compatibility_regressions"`
	FirstPassValidityDelta         float64 `json:"first_pass_validity_delta"`
	EventualValidityDelta          float64 `json:"eventual_validity_delta"`
	ObservedOutcomeDelta           float64 `json:"observed_outcome_delta"`
	CostIncreaseRate               float64 `json:"cost_increase_rate"`
	LatencyIncreaseRate            float64 `json:"latency_increase_rate"`
	FailureRate                    float64 `json:"failure_rate"`
	RetryRate                      float64 `json:"retry_rate"`
	UnexecutedCounterfactualClaims int     `json:"unexecuted_counterfactual_claims"`
}

type PlannerRolloutPromotionDecision struct {
	CurrentStagePercent int                               `json:"current_stage_percent"`
	NextStagePercent    int                               `json:"next_stage_percent"`
	Action              string                            `json:"action"`
	Thresholds          PlannerRolloutPromotionThresholds `json:"thresholds"`
	Reasons             []string                          `json:"reasons,omitempty"`
}

func DefaultPlannerRolloutPolicy() PlannerRolloutPolicy {
	return PlannerRolloutPolicy{
		PolicyVersion:        PlannerRolloutPolicyVersionV1,
		PolicyID:             PlannerRolloutCandidatePolicyV1,
		LastApprovedPolicyID: PlannerRolloutLastApprovedPolicyV1,
		StagePercent:         5,
		State:                RolloutStateDisabled,
		Dimensions:           []string{RolloutDimensionRanker},
		VariantValues:        map[string]string{RolloutDimensionRanker: PlannerRolloutRankerV2VariantV1},
	}
}

func PlannerRolloutPolicyFromEnvironment() (PlannerRolloutPolicy, error) {
	policy := DefaultPlannerRolloutPolicy()
	policy.Enabled = rolloutEnvironmentFlag("MODEL_EXPRESS_PLANNER_ROLLOUT_ENABLED")
	if policy.Enabled {
		policy.State = RolloutStateActive
	}
	if value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PLANNER_ROLLOUT_POLICY_ID")); value != "" {
		policy.PolicyID = value
	}
	if value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PLANNER_ROLLOUT_LAST_APPROVED_POLICY_ID")); value != "" {
		policy.LastApprovedPolicyID = value
	}
	if value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PLANNER_ROLLOUT_STAGE_PERCENT")); value != "" {
		stage, err := strconv.Atoi(value)
		if err != nil {
			return PlannerRolloutPolicy{}, fmt.Errorf("planner rollout stage must be one of 5, 25, 50, or 100")
		}
		policy.StagePercent = stage
	}
	if value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PLANNER_ROLLOUT_STATE")); value != "" {
		policy.State = strings.ToLower(value)
	}
	if value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PLANNER_ROLLOUT_DIMENSIONS")); value != "" {
		policy.Dimensions = splitRolloutDimensions(value)
	}
	policy.Factorial = rolloutEnvironmentFlag("MODEL_EXPRESS_PLANNER_ROLLOUT_FACTORIAL")
	policy.Rollback = rolloutEnvironmentFlag("MODEL_EXPRESS_PLANNER_ROLLOUT_ROLLBACK")
	for dimension, environmentName := range map[string]string{
		RolloutDimensionRanker:    "MODEL_EXPRESS_PLANNER_ROLLOUT_RANKER_VERSION",
		RolloutDimensionPrompt:    "MODEL_EXPRESS_PLANNER_ROLLOUT_PROMPT_VERSION",
		RolloutDimensionContext:   "MODEL_EXPRESS_PLANNER_ROLLOUT_CONTEXT_VERSION",
		RolloutDimensionRetrieval: "MODEL_EXPRESS_PLANNER_ROLLOUT_RETRIEVAL_VARIANT",
	} {
		if value := strings.TrimSpace(os.Getenv(environmentName)); value != "" {
			policy.VariantValues[dimension] = value
		}
	}
	if err := ValidatePlannerRolloutPolicy(policy); err != nil {
		return PlannerRolloutPolicy{}, err
	}
	return policy, nil
}

func ValidatePlannerRolloutPolicy(policy PlannerRolloutPolicy) error {
	if strings.TrimSpace(policy.PolicyVersion) == "" {
		policy.PolicyVersion = PlannerRolloutPolicyVersionV1
	}
	if policy.PolicyVersion != PlannerRolloutPolicyVersionV1 {
		return fmt.Errorf("unsupported planner rollout policy version %q", policy.PolicyVersion)
	}
	if strings.TrimSpace(policy.PolicyID) == "" || strings.TrimSpace(policy.LastApprovedPolicyID) == "" {
		return fmt.Errorf("planner rollout policy and last-approved policy identities are required")
	}
	if !validGuardedRolloutStage(policy.StagePercent) {
		return fmt.Errorf("planner rollout stage must be one of 5, 25, 50, or 100")
	}
	switch policy.State {
	case RolloutStateDisabled, RolloutStateActive, RolloutStatePaused, RolloutStateManualReview:
	default:
		return fmt.Errorf("planner rollout state %q is invalid", policy.State)
	}
	if unknown := unknownRolloutDimensions(policy.Dimensions); len(unknown) > 0 {
		return fmt.Errorf("planner rollout contains unsupported dimensions: %s", strings.Join(unknown, ", "))
	}
	dimensions := normalizedRolloutDimensions(policy.Dimensions)
	if len(dimensions) == 0 {
		return fmt.Errorf("planner rollout requires one experimental dimension")
	}
	if len(dimensions) > 1 && !policy.Factorial {
		return fmt.Errorf("planner rollout cannot change %d dimensions without explicit factorial mode", len(dimensions))
	}
	for _, dimension := range dimensions {
		if strings.TrimSpace(policy.VariantValues[dimension]) == "" {
			return fmt.Errorf("planner rollout dimension %q requires an explicit variant value", dimension)
		}
		if dimension == RolloutDimensionRanker && policy.VariantValues[dimension] != PlannerRolloutRankerV2VariantV1 {
			return fmt.Errorf("planner rollout ranker variant %q is unsupported", policy.VariantValues[dimension])
		}
	}
	return nil
}

func AssignPlannerRollout(policy PlannerRolloutPolicy, projectID string) (PlannerRolloutAssignment, error) {
	if err := ValidatePlannerRolloutPolicy(policy); err != nil {
		return PlannerRolloutAssignment{}, err
	}
	projectID = strings.TrimSpace(projectID)
	if projectID == "" {
		return PlannerRolloutAssignment{}, fmt.Errorf("planner rollout project identity is required")
	}
	bucket := plannerRolloutBucket(projectID)
	dimensions := normalizedRolloutDimensions(policy.Dimensions)
	assignment := PlannerRolloutAssignment{
		SchemaVersion:        PlannerRolloutAssignmentSchemaVersionV1,
		CohortVersion:        PlannerRolloutCohortVersionV1,
		CohortID:             fmt.Sprintf("%s_%02d", PlannerRolloutCohortVersionV1, bucket),
		CohortBucket:         bucket,
		PolicyID:             policy.LastApprovedPolicyID,
		LastApprovedPolicyID: policy.LastApprovedPolicyID,
		StagePercent:         policy.StagePercent,
		State:                policy.State,
		Dimensions:           dimensions,
		Factorial:            policy.Factorial,
		VariantValues:        copyRolloutValues(policy.VariantValues, dimensions),
	}
	if !policy.Enabled {
		assignment.State = RolloutStateDisabled
		return assignment, nil
	}
	if policy.Rollback {
		assignment.State = RolloutStateRolledBack
		return assignment, nil
	}
	if policy.State == RolloutStatePaused || policy.State == RolloutStateManualReview {
		assignment.ManualReviewRequired = true
		return assignment, nil
	}
	if bucket < policy.StagePercent {
		assignment.PolicyID = policy.PolicyID
		assignment.Treatment = true
	}
	return assignment, nil
}

func ValidatePlannerRolloutAssignment(assignment PlannerRolloutAssignment) error {
	if assignment.SchemaVersion != PlannerRolloutAssignmentSchemaVersionV1 || assignment.CohortVersion != PlannerRolloutCohortVersionV1 {
		return fmt.Errorf("planner rollout assignment schema or cohort version is invalid")
	}
	if assignment.CohortBucket < 0 || assignment.CohortBucket > 99 || strings.TrimSpace(assignment.CohortID) == "" {
		return fmt.Errorf("planner rollout cohort identity is invalid")
	}
	wantCohortID := fmt.Sprintf("%s_%02d", PlannerRolloutCohortVersionV1, assignment.CohortBucket)
	if assignment.CohortID != wantCohortID {
		return fmt.Errorf("planner rollout cohort identity does not match its bucket")
	}
	if strings.TrimSpace(assignment.PolicyID) == "" || strings.TrimSpace(assignment.LastApprovedPolicyID) == "" {
		return fmt.Errorf("planner rollout assignment policy identity is required")
	}
	if !validGuardedRolloutStage(assignment.StagePercent) {
		return fmt.Errorf("planner rollout assignment stage is invalid")
	}
	switch assignment.State {
	case RolloutStateDisabled, RolloutStateActive, RolloutStatePaused, RolloutStateManualReview, RolloutStateRolledBack:
	default:
		return fmt.Errorf("planner rollout assignment state %q is invalid", assignment.State)
	}
	if unknown := unknownRolloutDimensions(assignment.Dimensions); len(unknown) > 0 {
		return fmt.Errorf("planner rollout assignment contains unsupported dimensions: %s", strings.Join(unknown, ", "))
	}
	if len(assignment.Dimensions) > 1 && !assignment.Factorial {
		return fmt.Errorf("planner rollout assignment contains unintended simultaneous experiments")
	}
	for _, dimension := range normalizedRolloutDimensions(assignment.Dimensions) {
		if strings.TrimSpace(assignment.VariantValues[dimension]) == "" {
			return fmt.Errorf("planner rollout assignment dimension %q has no variant value", dimension)
		}
		if dimension == RolloutDimensionRanker && assignment.VariantValues[dimension] != PlannerRolloutRankerV2VariantV1 {
			return fmt.Errorf("planner rollout assignment ranker variant %q is unsupported", assignment.VariantValues[dimension])
		}
	}
	if assignment.Treatment && assignment.PolicyID == assignment.LastApprovedPolicyID {
		return fmt.Errorf("planner rollout treatment must identify a policy distinct from the last approved policy")
	}
	if !assignment.Treatment && assignment.PolicyID != assignment.LastApprovedPolicyID {
		return fmt.Errorf("planner rollout control must use the last approved policy")
	}
	return nil
}

func PlannerRolloutTreatsDimension(assignment *PlannerRolloutAssignment, dimension string) bool {
	if assignment == nil || !assignment.Treatment {
		return false
	}
	dimension = strings.ToLower(strings.TrimSpace(dimension))
	for _, active := range assignment.Dimensions {
		if active == dimension {
			return true
		}
	}
	return false
}

func PlannerRolloutVariantValue(assignment *PlannerRolloutAssignment, dimension string) string {
	if !PlannerRolloutTreatsDimension(assignment, dimension) {
		return ""
	}
	return strings.TrimSpace(assignment.VariantValues[dimension])
}

func DefaultPlannerRolloutPromotionThresholds(stagePercent int) (PlannerRolloutPromotionThresholds, error) {
	minimumSamples := map[int]int{5: 50, 25: 100, 50: 200, 100: 400}
	minimum, ok := minimumSamples[stagePercent]
	if !ok {
		return PlannerRolloutPromotionThresholds{}, fmt.Errorf("planner rollout stage must be one of 5, 25, 50, or 100")
	}
	return PlannerRolloutPromotionThresholds{
		MinimumSampleSize: minimum, MinimumObservedOutcomeSamples: minimum,
		FirstPassValidityMargin: -0.02, EventualValidityMargin: -0.005, ObservedOutcomeMargin: -0.01,
		MaximumCostIncreaseRate: 0.10, MaximumLatencyIncreaseRate: 0.10,
		MaximumFailureRate: 0.05, MaximumRetryRate: 0.20,
	}, nil
}

func EvaluatePlannerRolloutPromotion(stagePercent int, observed PlannerRolloutPromotionObservation) (PlannerRolloutPromotionDecision, error) {
	thresholds, err := DefaultPlannerRolloutPromotionThresholds(stagePercent)
	if err != nil {
		return PlannerRolloutPromotionDecision{}, err
	}
	if err := validatePlannerRolloutPromotionObservation(observed); err != nil {
		return PlannerRolloutPromotionDecision{}, err
	}
	decision := PlannerRolloutPromotionDecision{
		CurrentStagePercent: stagePercent,
		NextStagePercent:    stagePercent,
		Action:              RolloutActionHold,
		Thresholds:          thresholds,
	}
	criticalReasons := []string{}
	if observed.SafetyRegressions != 0 {
		criticalReasons = append(criticalReasons, "safety parity regressed")
	}
	if observed.TaskCompatibilityRegressions != 0 {
		criticalReasons = append(criticalReasons, "task-compatibility parity regressed")
	}
	if observed.UnexecutedCounterfactualClaims != 0 {
		criticalReasons = append(criticalReasons, "outcomes were claimed for unexecuted counterfactuals")
	}
	if len(criticalReasons) > 0 {
		decision.Action = RolloutActionManualReview
		decision.Reasons = criticalReasons
		return decision, nil
	}
	if observed.SampleSize < thresholds.MinimumSampleSize || observed.ObservedOutcomeSampleSize < thresholds.MinimumObservedOutcomeSamples {
		decision.Reasons = append(decision.Reasons, fmt.Sprintf("adequate samples require %d invocations and %d observed outcomes", thresholds.MinimumSampleSize, thresholds.MinimumObservedOutcomeSamples))
		return decision, nil
	}
	if observed.FirstPassValidityDelta < thresholds.FirstPassValidityMargin {
		decision.Reasons = append(decision.Reasons, "first-pass validity is inferior")
	}
	if observed.EventualValidityDelta < thresholds.EventualValidityMargin {
		decision.Reasons = append(decision.Reasons, "eventual validity is inferior")
	}
	if observed.ObservedOutcomeDelta < thresholds.ObservedOutcomeMargin {
		decision.Reasons = append(decision.Reasons, "observed executed outcomes are inferior")
	}
	if observed.CostIncreaseRate > thresholds.MaximumCostIncreaseRate {
		decision.Reasons = append(decision.Reasons, "cost increase exceeds the bound")
	}
	if observed.LatencyIncreaseRate > thresholds.MaximumLatencyIncreaseRate {
		decision.Reasons = append(decision.Reasons, "latency increase exceeds the bound")
	}
	if observed.FailureRate > thresholds.MaximumFailureRate {
		decision.Reasons = append(decision.Reasons, "failure rate exceeds the bound")
	}
	if observed.RetryRate > thresholds.MaximumRetryRate {
		decision.Reasons = append(decision.Reasons, "retry rate exceeds the bound")
	}
	if len(decision.Reasons) > 0 {
		decision.Action = RolloutActionPause
		return decision, nil
	}
	if stagePercent == 100 {
		decision.Action = RolloutActionApproved
		return decision, nil
	}
	decision.Action = RolloutActionAdvance
	decision.NextStagePercent = nextGuardedRolloutStage(stagePercent)
	return decision, nil
}

func validatePlannerRolloutPromotionObservation(observed PlannerRolloutPromotionObservation) error {
	if observed.SampleSize < 0 || observed.ObservedOutcomeSampleSize < 0 || observed.SafetyRegressions < 0 ||
		observed.TaskCompatibilityRegressions < 0 || observed.UnexecutedCounterfactualClaims < 0 {
		return fmt.Errorf("planner rollout observations cannot contain negative counts")
	}
	for _, metric := range []struct {
		name  string
		value float64
	}{
		{"first_pass_validity_delta", observed.FirstPassValidityDelta},
		{"eventual_validity_delta", observed.EventualValidityDelta},
		{"observed_outcome_delta", observed.ObservedOutcomeDelta},
	} {
		name, value := metric.name, metric.value
		if math.IsNaN(value) || math.IsInf(value, 0) || value < -1 || value > 1 {
			return fmt.Errorf("planner rollout %s must be finite and between -1 and 1", name)
		}
	}
	for _, metric := range []struct {
		name  string
		value float64
	}{
		{"cost_increase_rate", observed.CostIncreaseRate},
		{"latency_increase_rate", observed.LatencyIncreaseRate},
	} {
		name, value := metric.name, metric.value
		if math.IsNaN(value) || math.IsInf(value, 0) || value < -1 {
			return fmt.Errorf("planner rollout %s must be finite and at least -1", name)
		}
	}
	for _, metric := range []struct {
		name  string
		value float64
	}{
		{"failure_rate", observed.FailureRate},
		{"retry_rate", observed.RetryRate},
	} {
		name, value := metric.name, metric.value
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 1 {
			return fmt.Errorf("planner rollout %s must be finite and between 0 and 1", name)
		}
	}
	return nil
}

func GuardedPlannerRolloutStages() []int {
	return append([]int(nil), guardedRolloutStages...)
}

func plannerRolloutBucket(projectID string) int {
	digest := sha256.Sum256([]byte(PlannerRolloutCohortVersionV1 + "\x00" + projectID))
	return int(binary.BigEndian.Uint64(digest[:8]) % 100)
}

func validGuardedRolloutStage(stage int) bool {
	for _, candidate := range guardedRolloutStages {
		if stage == candidate {
			return true
		}
	}
	return false
}

func nextGuardedRolloutStage(stage int) int {
	for index, candidate := range guardedRolloutStages {
		if candidate == stage && index+1 < len(guardedRolloutStages) {
			return guardedRolloutStages[index+1]
		}
	}
	return stage
}

func splitRolloutDimensions(value string) []string {
	return normalizedRolloutDimensions(strings.Split(value, ","))
}

func normalizedRolloutDimensions(values []string) []string {
	allowed := map[string]bool{
		RolloutDimensionRanker: true, RolloutDimensionPrompt: true,
		RolloutDimensionContext: true, RolloutDimensionRetrieval: true,
	}
	seen := map[string]bool{}
	out := []string{}
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if value == "" || seen[value] || !allowed[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func unknownRolloutDimensions(values []string) []string {
	allowed := map[string]bool{
		RolloutDimensionRanker: true, RolloutDimensionPrompt: true,
		RolloutDimensionContext: true, RolloutDimensionRetrieval: true,
	}
	seen := map[string]bool{}
	out := []string{}
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if value == "" || allowed[value] || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func copyRolloutValues(values map[string]string, dimensions []string) map[string]string {
	out := map[string]string{}
	for _, dimension := range dimensions {
		out[dimension] = strings.TrimSpace(values[dimension])
	}
	return out
}

func rolloutEnvironmentFlag(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
