package plannervalidation

import (
	"fmt"
	"os"
	"strings"
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
)

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

// ModeFromEnvironment centralizes planner validation configuration. The old
// boolean remains a one-release compatibility alias when the mode variable is
// absent; all runtime callers consume this function rather than either env var.
func ModeFromEnvironment() string {
	if value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PLANNER_VALIDATION_MODE")); value != "" {
		return NormalizeMode(value)
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_STRICT_PLANNER_VALIDATION"))) {
	case "1", "true", "yes", "on":
		return ModeStrict
	default:
		return ModeRelaxed
	}
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
