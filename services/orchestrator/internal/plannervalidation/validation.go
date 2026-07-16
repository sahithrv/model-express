package plannervalidation

import (
	"fmt"
	"math"
	"os"
	"strings"
	"sync/atomic"

	"model-express/services/orchestrator/internal/diagnostics"
)

const (
	ModeRelaxed      = "relaxed"
	ModeShadowStrict = "shadow_strict"
	ModeStrict       = "strict"

	VerdictSchemaVersionV1 = "planner_strict_validation_verdict_v1"
	OutcomeSchemaVersionV1 = "planner_validation_outcome_v1"

	VerdictNotEvaluated = "not_evaluated"
	VerdictPassed       = "passed"
	VerdictWouldBlock   = "would_block"

	CategoryMissingEvidence   = "missing_evidence"
	CategoryMechanismMismatch = "mechanism_mismatch"
	CategoryInvalidTaskModel  = "invalid_task_model"
	CategoryProposalNoOp      = "proposal_no_op"
	CategoryStrictContract    = "strict_contract"

	FirstPassAccepted = "accepted"
	FirstPassRejected = "rejected"
	FirstPassUnknown  = "unknown"
	EventualAccepted  = "accepted"
	EventualRejected  = "rejected"
	EventualPending   = "pending"
	RetryNotNeeded    = "not_needed"
	RetryScheduled    = "retry_scheduled"
	RetryExhausted    = "retry_exhausted"
	RetryAccepted     = "accepted_after_retry"

	StrictDefaultRolloutPolicyVersion = "planner_strict_default_rollout_v1"
	RelaxedRollbackPolicyVersion      = "planner_relaxed_rollback_v1"
	RelaxedRollbackDiagnosticEvent    = "planner_validation_relaxed_rollback"
	DefaultMaxRetriesPerAttemptGroup  = 2
)

var relaxedRollbackDiagnosticCount atomic.Uint64

type ModeResolution struct {
	Mode     string
	Source   string
	Rollback bool
}

type StrictRolloutThresholds struct {
	PolicyVersion                  string  `json:"policy_version"`
	MinimumSampleSize              int     `json:"minimum_sample_size"`
	MinimumEventualValidityRate    float64 `json:"minimum_eventual_validity_rate"`
	MaximumRetryRate               float64 `json:"maximum_retry_rate"`
	MaximumUnsupportedProposalRate float64 `json:"maximum_unsupported_proposal_rate"`
	MaximumUnsafeSchedules         int     `json:"maximum_unsafe_schedules"`
	MaximumPostValidationEscapes   int     `json:"maximum_post_validation_escapes"`
	MaximumRetriesPerAttemptGroup  int     `json:"maximum_retries_per_attempt_group"`
}

type StrictRolloutObservation struct {
	SampleSize              int     `json:"sample_size"`
	EventualValidityRate    float64 `json:"eventual_validity_rate"`
	RetryRate               float64 `json:"retry_rate"`
	UnsupportedProposalRate float64 `json:"unsupported_proposal_rate"`
	UnsafeSchedules         int     `json:"unsafe_schedules"`
	PostValidationEscapes   int     `json:"post_validation_escapes"`
}

type StrictRolloutAssessment struct {
	Passed     bool                     `json:"passed"`
	Thresholds StrictRolloutThresholds  `json:"thresholds"`
	Observed   StrictRolloutObservation `json:"observed"`
	Violations []string                 `json:"violations,omitempty"`
}

// Finding is a stable, typed planner-validation result. Message is retained
// for diagnosis, while callers aggregate by Code and Category.
type Finding struct {
	Code     string `json:"code"`
	Category string `json:"category"`
	Stage    string `json:"stage"`
	Message  string `json:"message"`
}

// Verdict records what the strict validator found. In shadow_strict mode a
// would_block verdict is observational and never changes the relaxed result.
type Verdict struct {
	SchemaVersion string    `json:"schema_version"`
	Mode          string    `json:"mode"`
	Status        string    `json:"status"`
	WouldBlock    bool      `json:"would_block"`
	Findings      []Finding `json:"findings,omitempty"`
}

// Outcome records attempt-group semantics in addition to the per-attempt
// validation_status already stored on the invocation.
type Outcome struct {
	SchemaVersion   string `json:"schema_version"`
	Mode            string `json:"mode"`
	FirstPassStatus string `json:"first_pass_status"`
	EventualStatus  string `json:"eventual_status"`
	RetryOutcome    string `json:"retry_outcome"`
}

// Check names a strict rule before executing it, avoiding error-string based
// metric classification.
type Check struct {
	Code     string
	Category string
	Stage    string
	Validate func() error
}

// NormalizeMode is the only normalization policy for planner validation.
// Unknown values fail safely to the compatibility-preserving relaxed mode.
func NormalizeMode(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case ModeShadowStrict:
		return ModeShadowStrict
	case ModeStrict:
		return ModeStrict
	default:
		return ModeRelaxed
	}
}

// ResolveModeFromEnvironment centralizes planner validation configuration.
// Strict is the safe default for new configurations. The old boolean remains
// a one-release compatibility alias when the mode variable is absent.
func ResolveModeFromEnvironment() ModeResolution {
	if value := strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE"))); value != "" {
		switch value {
		case ModeRelaxed:
			return ModeResolution{Mode: ModeRelaxed, Source: "mode_environment", Rollback: true}
		case ModeShadowStrict:
			return ModeResolution{Mode: ModeShadowStrict, Source: "mode_environment"}
		case ModeStrict:
			return ModeResolution{Mode: ModeStrict, Source: "mode_environment"}
		default:
			return ModeResolution{Mode: ModeStrict, Source: "invalid_mode_environment"}
		}
	}
	if value := strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION"))); value != "" {
		switch value {
		case "1", "true", "yes", "on":
			return ModeResolution{Mode: ModeStrict, Source: "legacy_boolean_alias"}
		case "0", "false", "no", "off":
			return ModeResolution{Mode: ModeRelaxed, Source: "legacy_boolean_alias", Rollback: true}
		default:
			return ModeResolution{Mode: ModeStrict, Source: "invalid_legacy_boolean_alias"}
		}
	}
	return ModeResolution{Mode: ModeStrict, Source: "strict_default"}
}

func ModeFromEnvironment() string {
	resolution := ResolveModeFromEnvironment()
	if resolution.Rollback {
		count := relaxedRollbackDiagnosticCount.Add(1)
		diagnostics.Event("warn", RelaxedRollbackDiagnosticEvent, map[string]any{
			"count":          count,
			"mode":           resolution.Mode,
			"source":         resolution.Source,
			"policy_version": RelaxedRollbackPolicyVersion,
		})
	}
	return resolution.Mode
}

func RelaxedRollbackDiagnosticCount() uint64 {
	return relaxedRollbackDiagnosticCount.Load()
}

func DefaultStrictRolloutThresholds() StrictRolloutThresholds {
	return StrictRolloutThresholds{
		PolicyVersion:                  StrictDefaultRolloutPolicyVersion,
		MinimumSampleSize:              100,
		MinimumEventualValidityRate:    0.98,
		MaximumRetryRate:               0.20,
		MaximumUnsupportedProposalRate: 0.05,
		MaximumUnsafeSchedules:         0,
		MaximumPostValidationEscapes:   0,
		MaximumRetriesPerAttemptGroup:  DefaultMaxRetriesPerAttemptGroup,
	}
}

func AssessStrictRollout(observed StrictRolloutObservation) StrictRolloutAssessment {
	thresholds := DefaultStrictRolloutThresholds()
	assessment := StrictRolloutAssessment{
		Passed:     true,
		Thresholds: thresholds,
		Observed:   observed,
	}
	addViolation := func(condition bool, message string) {
		if condition {
			assessment.Passed = false
			assessment.Violations = append(assessment.Violations, message)
		}
	}
	addViolation(observed.SampleSize < 0 || observed.UnsafeSchedules < 0 || observed.PostValidationEscapes < 0,
		"strict rollout counts must be non-negative")
	addViolation(!validRolloutRate(observed.EventualValidityRate), "eventual validity rate must be finite and between 0 and 1")
	addViolation(!validRolloutRate(observed.RetryRate), "retry rate must be finite and between 0 and 1")
	addViolation(!validRolloutRate(observed.UnsupportedProposalRate), "unsupported proposal rate must be finite and between 0 and 1")
	addViolation(observed.SampleSize < thresholds.MinimumSampleSize,
		fmt.Sprintf("sample size %d is below %d", observed.SampleSize, thresholds.MinimumSampleSize))
	addViolation(observed.EventualValidityRate < thresholds.MinimumEventualValidityRate,
		fmt.Sprintf("eventual validity rate %.4f is below %.4f", observed.EventualValidityRate, thresholds.MinimumEventualValidityRate))
	addViolation(observed.RetryRate > thresholds.MaximumRetryRate,
		fmt.Sprintf("retry rate %.4f exceeds %.4f", observed.RetryRate, thresholds.MaximumRetryRate))
	addViolation(observed.UnsupportedProposalRate > thresholds.MaximumUnsupportedProposalRate,
		fmt.Sprintf("unsupported proposal rate %.4f exceeds %.4f", observed.UnsupportedProposalRate, thresholds.MaximumUnsupportedProposalRate))
	addViolation(observed.UnsafeSchedules > thresholds.MaximumUnsafeSchedules,
		fmt.Sprintf("unsafe schedules %d exceeds %d", observed.UnsafeSchedules, thresholds.MaximumUnsafeSchedules))
	addViolation(observed.PostValidationEscapes > thresholds.MaximumPostValidationEscapes,
		fmt.Sprintf("post-validation escapes %d exceeds %d", observed.PostValidationEscapes, thresholds.MaximumPostValidationEscapes))
	return assessment
}

func validRolloutRate(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0 && value <= 1
}

func IsStrict(mode string) bool {
	return NormalizeMode(mode) == ModeStrict
}

func IsShadow(mode string) bool {
	return NormalizeMode(mode) == ModeShadowStrict
}

// Evaluate executes one shared strict implementation for shadow_strict and
// strict. Relaxed mode deliberately skips the extra checks. All checks run so
// a shadow verdict can quantify multiple typed failure classes in one pass.
func Evaluate(mode string, checks []Check) (Verdict, error) {
	mode = NormalizeMode(mode)
	verdict := Verdict{
		SchemaVersion: VerdictSchemaVersionV1,
		Mode:          mode,
		Status:        VerdictNotEvaluated,
		Findings:      []Finding{},
	}
	if mode == ModeRelaxed {
		return verdict, nil
	}

	for _, check := range checks {
		if check.Validate == nil {
			continue
		}
		if err := check.Validate(); err != nil {
			verdict.Findings = append(verdict.Findings, Finding{
				Code:     strings.TrimSpace(check.Code),
				Category: strings.TrimSpace(check.Category),
				Stage:    strings.TrimSpace(check.Stage),
				Message:  err.Error(),
			})
		}
	}
	if len(verdict.Findings) == 0 {
		verdict.Status = VerdictPassed
		return verdict, nil
	}
	verdict.Status = VerdictWouldBlock
	verdict.WouldBlock = true
	if mode == ModeStrict {
		return verdict, EvaluationError{Findings: verdict.Findings}
	}
	return verdict, nil
}

// Merge combines validation stages without losing typed findings. A relaxed
// not-evaluated verdict never overwrites an evaluated verdict.
func Merge(left, right Verdict) Verdict {
	if left.SchemaVersion == "" || left.Status == VerdictNotEvaluated {
		return right
	}
	if right.SchemaVersion == "" || right.Status == VerdictNotEvaluated {
		return left
	}
	merged := left
	merged.Mode = NormalizeMode(right.Mode)
	merged.Findings = append(append([]Finding(nil), left.Findings...), right.Findings...)
	merged.WouldBlock = len(merged.Findings) > 0
	if merged.WouldBlock {
		merged.Status = VerdictWouldBlock
	} else {
		merged.Status = VerdictPassed
	}
	return merged
}

type EvaluationError struct {
	Findings []Finding
}

func (err EvaluationError) Error() string {
	if len(err.Findings) == 0 {
		return "strict planner validation failed"
	}
	if len(err.Findings) == 1 {
		return err.Findings[0].Message
	}
	return fmt.Sprintf("%s (and %d additional strict validation findings)", err.Findings[0].Message, len(err.Findings)-1)
}
