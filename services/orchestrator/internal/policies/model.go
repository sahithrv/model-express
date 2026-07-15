package policies

import (
	"encoding/json"
	"errors"
	"time"

	"model-express/services/orchestrator/internal/catalog"
)

const (
	PolicySchemaVersionV1               = "model_express_experiment_policy.v1"
	CompatibilityProfileSchemaVersionV1 = "model_express_compatibility_profile.v1"
	EffectiveSnapshotSchemaVersionV1    = "model_express_effective_policy.v1"
	ImplicitAllowAllProfileKey          = "allow_all_v0"
	ImplicitAllowAllProfileVersion      = "0.0.0"
	LocalDefaultAccountID               = "account_local_default"
)

var (
	ErrRevisionConflict  = errors.New("experiment policy binding revision conflict")
	ErrImmutableConflict = errors.New("immutable experiment policy record already exists")
)

type Scope string

const (
	ScopeAccount Scope = "account"
	ScopeProject Scope = "project"
	ScopeDataset Scope = "dataset"
	ScopeRun     Scope = "run"
)

func ScopeRank(scope Scope) int {
	switch scope {
	case ScopeAccount:
		return 0
	case ScopeProject:
		return 1
	case ScopeDataset:
		return 2
	case ScopeRun:
		return 3
	default:
		return 99
	}
}

type ScopeContext struct {
	AccountID       string `json:"account_id"`
	ProjectID       string `json:"project_id,omitempty"`
	DatasetID       string `json:"dataset_id,omitempty"`
	ExperimentJobID string `json:"experiment_job_id,omitempty"`
	Task            string `json:"task,omitempty"`
	Runner          string `json:"runner,omitempty"`
}

type ProfileRef struct {
	ID      string `json:"id"`
	Version string `json:"version"`
}

type Metadata struct {
	DisplayName string `json:"display_name,omitempty"`
	Description string `json:"description,omitempty"`
}

type Selector struct {
	Kind      string   `json:"kind"`
	Catalog   string   `json:"catalog,omitempty"`
	IDs       []string `json:"ids,omitempty"`
	Attribute string   `json:"attribute,omitempty"`
	Values    []any    `json:"values,omitempty"`
	Field     string   `json:"field,omitempty"`
}

const (
	SelectorCatalogIDs             = "catalog_ids"
	SelectorCatalogAttributeValues = "catalog_attribute_values"
	SelectorFieldValues            = "field_values"
	EffectDeny                     = "deny"
	EffectAllow                    = "allow"
)

type Rule struct {
	ID       string   `json:"id"`
	Effect   string   `json:"effect"`
	Selector Selector `json:"selector"`
}

type PolicyDocument struct {
	SchemaVersion string       `json:"schema_version"`
	ProfileRefs   []ProfileRef `json:"profile_refs"`
	Rules         []Rule       `json:"rules"`
	Metadata      Metadata     `json:"metadata,omitempty"`
}

type CompatibilityProfileDocument struct {
	SchemaVersion  string   `json:"schema_version"`
	CatalogVersion string   `json:"catalog_version"`
	Rules          []Rule   `json:"rules"`
	Metadata       Metadata `json:"metadata,omitempty"`
}

type CompatibilityProfile struct {
	ID              string                       `json:"id"`
	ProfileKey      string                       `json:"profile_key"`
	SemanticVersion string                       `json:"semantic_version"`
	SchemaVersion   string                       `json:"schema_version"`
	CatalogVersion  string                       `json:"catalog_version"`
	Document        CompatibilityProfileDocument `json:"document"`
	CanonicalJSON   json.RawMessage              `json:"canonical_json"`
	DocumentHash    string                       `json:"document_hash"`
	OwnerAccountID  string                       `json:"owner_account_id,omitempty"`
	CreatedAt       time.Time                    `json:"created_at"`
	CreatedBy       string                       `json:"created_by"`
}

type PolicyVersion struct {
	ID             string          `json:"id"`
	OwnerAccountID string          `json:"owner_account_id"`
	SchemaVersion  string          `json:"schema_version"`
	Revision       int64           `json:"revision"`
	Document       PolicyDocument  `json:"document"`
	CanonicalJSON  json.RawMessage `json:"canonical_json"`
	DocumentHash   string          `json:"document_hash"`
	CreatedAt      time.Time       `json:"created_at"`
	CreatedBy      string          `json:"created_by"`
}

type Binding struct {
	ID              string     `json:"id"`
	Scope           Scope      `json:"scope"`
	SubjectID       string     `json:"subject_id"`
	PolicyVersionID string     `json:"policy_version_id"`
	Revision        int64      `json:"revision"`
	Active          bool       `json:"active"`
	SupersedesID    string     `json:"supersedes_id,omitempty"`
	SupersededAt    *time.Time `json:"superseded_at,omitempty"`
	CreatedAt       time.Time  `json:"created_at"`
	CreatedBy       string     `json:"created_by"`
}

type BindingWrite struct {
	Scope            Scope
	SubjectID        string
	PolicyVersionID  string
	ExpectedRevision *int64
	CreatedBy        string
}

type PolicySource struct {
	BindingID       string `json:"binding_id"`
	BindingRevision int64  `json:"binding_revision"`
	Scope           Scope  `json:"scope"`
	SubjectID       string `json:"subject_id"`
	PolicyVersionID string `json:"policy_version_id"`
	PolicyRevision  int64  `json:"policy_revision"`
	PolicyHash      string `json:"policy_hash"`
}

type ProfileSource struct {
	ID           string `json:"id"`
	Version      string `json:"version"`
	DocumentHash string `json:"document_hash"`
}

type FieldDenial struct {
	Field           string `json:"field"`
	CanonicalValue  string `json:"canonical_value"`
	Scope           Scope  `json:"scope"`
	SubjectID       string `json:"subject_id"`
	PolicyVersionID string `json:"policy_version_id"`
	RuleID          string `json:"rule_id"`
}

type ResolvedDefault struct {
	Field      string `json:"field"`
	Catalog    string `json:"catalog"`
	ID         string `json:"id"`
	Origin     string `json:"origin"`
	OriginalID string `json:"original_id,omitempty"`
}

type EffectiveSnapshot struct {
	SchemaVersion         string              `json:"schema_version"`
	CatalogVersion        string              `json:"catalog_version"`
	Context               ScopeContext        `json:"context"`
	ImplicitProfile       *ProfileRef         `json:"implicit_profile,omitempty"`
	CompatibilityProfiles []ProfileSource     `json:"compatibility_profiles"`
	PolicySources         []PolicySource      `json:"policy_sources"`
	PermittedCatalog      map[string][]string `json:"permitted_catalog"`
	DeniedCatalog         map[string][]string `json:"denied_catalog"`
	FieldDenials          []FieldDenial       `json:"field_denials"`
	ResolvedDefaults      []ResolvedDefault   `json:"resolved_defaults"`
	RequiredCatalogs      []string            `json:"required_catalogs"`
}

type EffectivePolicy struct {
	Decision            string                     `json:"decision"`
	CatalogVersion      string                     `json:"catalog_version"`
	EffectivePolicyHash string                     `json:"effective_policy_hash"`
	Snapshot            EffectiveSnapshot          `json:"snapshot"`
	PermittedCatalog    map[string][]catalog.Entry `json:"permitted_catalog"`
	PermittedCounts     map[string]int             `json:"permitted_counts"`
	Findings            []Finding                  `json:"findings"`
	BlockedDimensions   []string                   `json:"blocked_dimensions,omitempty"`
	ContributingScopes  []ScopeContribution        `json:"contributing_scopes,omitempty"`
}

type ScopeContribution struct {
	Scope           Scope  `json:"scope"`
	SubjectID       string `json:"subject_id"`
	PolicyVersionID string `json:"policy_version_id"`
}

// PromptPolicyCard is the complete, server-owned selectable policy surface
// supplied to proposal generators. It intentionally contains permitted values
// rather than a list of denied identifiers.
type PromptPolicyCard struct {
	SchemaVersion         string                     `json:"schema_version"`
	CatalogVersion        string                     `json:"catalog_version"`
	EffectivePolicyHash   string                     `json:"effective_policy_hash"`
	ImplicitProfile       *ProfileRef                `json:"implicit_profile,omitempty"`
	CompatibilityProfiles []ProfileSource            `json:"compatibility_profiles"`
	PolicySources         []PolicySource             `json:"policy_sources"`
	PermittedCatalog      map[string][]catalog.Entry `json:"permitted_catalog"`
	FieldDenials          []FieldDenial              `json:"field_denials"`
	ResolvedDefaults      []ResolvedDefault          `json:"resolved_defaults"`
	RequiredCatalogs      []string                   `json:"required_catalogs"`
	Instruction           string                     `json:"instruction"`
}

const (
	DecisionAllowed = "allowed"
	DecisionDenied  = "denied"
)

type CapabilityUse struct {
	Catalog   string `json:"catalog"`
	ID        string `json:"id"`
	FieldPath string `json:"field_path"`
	Origin    string `json:"origin"`
}

type Evaluation struct {
	ID                      string          `json:"id"`
	Operation               string          `json:"operation"`
	Decision                string          `json:"decision"`
	AccountID               string          `json:"account_id,omitempty"`
	ProjectID               string          `json:"project_id,omitempty"`
	DatasetID               string          `json:"dataset_id,omitempty"`
	PlanID                  string          `json:"plan_id,omitempty"`
	JobID                   string          `json:"job_id,omitempty"`
	AgentInvocationID       string          `json:"agent_invocation_id,omitempty"`
	ChampionExportID        string          `json:"champion_export_id,omitempty"`
	CandidateConfigHash     string          `json:"candidate_config_hash,omitempty"`
	CatalogVersion          string          `json:"catalog_version"`
	CompatibilityProfiles   []ProfileSource `json:"compatibility_profiles"`
	PolicySources           []PolicySource  `json:"policy_sources"`
	EffectiveSnapshot       json.RawMessage `json:"effective_snapshot"`
	EffectivePolicyHash     string          `json:"effective_policy_hash"`
	RequestedCapabilityUses []CapabilityUse `json:"requested_capability_uses"`
	EffectiveCapabilityUses []CapabilityUse `json:"effective_capability_uses"`
	ReasonCodes             []ReasonCode    `json:"reason_codes"`
	Findings                []Finding       `json:"findings"`
	ActorID                 string          `json:"actor_id,omitempty"`
	RequestID               string          `json:"request_id,omitempty"`
	CreatedAt               time.Time       `json:"created_at"`
}

type PersistenceReference struct {
	EvaluationID        string `json:"policy_evaluation_id"`
	EffectivePolicyHash string `json:"effective_policy_hash"`
	Status              string `json:"policy_status,omitempty"`
}

type Repository interface {
	GetCompatibilityProfile(profileKey string, semanticVersion string) (CompatibilityProfile, error)
	GetExperimentPolicyVersion(id string) (PolicyVersion, error)
	ListActiveExperimentPolicyBindings(scope ScopeContext) ([]Binding, error)
	CreateExperimentPolicyEvaluation(evaluation Evaluation) (Evaluation, error)
}
