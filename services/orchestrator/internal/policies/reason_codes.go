package policies

import "fmt"

type ReasonCode string

const (
	ReasonCatalogIDDenied             ReasonCode = "POLICY_CATALOG_ID_DENIED"
	ReasonCatalogAttributeDenied      ReasonCode = "POLICY_CATALOG_ATTRIBUTE_DENIED"
	ReasonFieldValueDenied            ReasonCode = "POLICY_FIELD_VALUE_DENIED"
	ReasonNotInCompatibilityProfile   ReasonCode = "POLICY_NOT_IN_COMPATIBILITY_PROFILE"
	ReasonUnknownCatalogIdentifier    ReasonCode = "POLICY_UNKNOWN_CATALOG_IDENTIFIER"
	ReasonProfileVersionUnavailable   ReasonCode = "POLICY_PROFILE_VERSION_UNAVAILABLE"
	ReasonNoValidConfiguration        ReasonCode = "POLICY_NO_VALID_CONFIGURATION"
	ReasonChangedAfterQueue           ReasonCode = "POLICY_CHANGED_AFTER_QUEUE"
	ReasonWorkerCapabilityUnavailable ReasonCode = "POLICY_WORKER_CAPABILITY_UNAVAILABLE"
)

type Finding struct {
	Code            ReasonCode `json:"code"`
	Catalog         string     `json:"catalog,omitempty"`
	ID              string     `json:"id,omitempty"`
	FieldPath       string     `json:"field_path,omitempty"`
	Origin          string     `json:"origin,omitempty"`
	Scope           Scope      `json:"scope,omitempty"`
	SubjectID       string     `json:"subject_id,omitempty"`
	PolicyVersionID string     `json:"policy_version_id,omitempty"`
	RuleID          string     `json:"rule_id,omitempty"`
	ProfileKey      string     `json:"profile_key,omitempty"`
	ProfileVersion  string     `json:"profile_version,omitempty"`
	Attribute       string     `json:"attribute,omitempty"`
	AttributeValue  string     `json:"attribute_value,omitempty"`
	CauseCatalog    string     `json:"cause_catalog,omitempty"`
	CauseID         string     `json:"cause_id,omitempty"`
	Remediation     string     `json:"remediation,omitempty"`
}

type PolicyError struct {
	Code                ReasonCode          `json:"code"`
	Message             string              `json:"error"`
	PolicyEvaluationID  string              `json:"policy_evaluation_id,omitempty"`
	EffectivePolicyHash string              `json:"effective_policy_hash,omitempty"`
	Findings            []Finding           `json:"findings,omitempty"`
	BlockedDimensions   []string            `json:"blocked_dimensions,omitempty"`
	ContributingScopes  []ScopeContribution `json:"contributing_scopes,omitempty"`
}

func (e *PolicyError) Error() string {
	if e == nil {
		return ""
	}
	if e.Message != "" {
		return e.Message
	}
	return fmt.Sprintf("experiment policy failed with %s", e.Code)
}

func policyError(code ReasonCode, message string, findings ...Finding) error {
	return &PolicyError{Code: code, Message: message, Findings: findings}
}
