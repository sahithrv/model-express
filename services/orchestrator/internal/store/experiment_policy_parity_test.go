package store

import (
	"context"
	"database/sql"
	"errors"
	"os"
	"reflect"
	"sort"
	"strconv"
	"testing"
	"time"

	"model-express/services/orchestrator/internal/catalog"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/policies"
)

type policyParitySummary struct {
	AccountID           string
	PolicyDocumentHash  string
	ProfileDocumentHash string
	BindingRevision     int64
	BindingSuperseded   bool
	StaleWriteRejected  bool
	Decision            string
	PermittedCatalog    map[string][]string
	FindingKeys         []string
	EvaluationDecision  string
	EvaluationCount     int
}

func TestMemoryExperimentPolicyPersistenceAndResolutionParityFixture(t *testing.T) {
	summary := buildPolicyParityFixture(t, NewMemoryStore(), "memory_"+strconv.FormatInt(time.Now().UnixNano(), 10))
	if summary.AccountID != policies.LocalDefaultAccountID || summary.BindingRevision != 2 || !summary.BindingSuperseded || !summary.StaleWriteRejected {
		t.Fatalf("memory ownership/binding summary = %#v", summary)
	}
	if summary.Decision != policies.DecisionAllowed || summary.EvaluationDecision != policies.DecisionAllowed || summary.EvaluationCount != 1 {
		t.Fatalf("memory resolution/audit summary = %#v", summary)
	}
	if containsID(summary.PermittedCatalog["resize_strategies"], "yolo_letterbox") || containsID(summary.PermittedCatalog["augmentation_policies"], "strong") {
		t.Fatalf("memory policy semantics were not applied: %#v", summary.PermittedCatalog)
	}
}

func TestMemoryExperimentPolicyDefinitionsAndBindingsAreImmutableCopies(t *testing.T) {
	persistence := NewMemoryStore()
	project, err := persistence.CreateProject("immutable policy", "")
	if err != nil {
		t.Fatal(err)
	}
	profileInput := policies.CompatibilityProfile{
		ProfileKey: "immutable_profile", SemanticVersion: "1.0.0",
		Document: policies.CompatibilityProfileDocument{
			SchemaVersion:  policies.CompatibilityProfileSchemaVersionV1,
			CatalogVersion: catalog.CatalogV1().CatalogVersion,
			Rules: []policies.Rule{{
				ID: "allow_resnet", Effect: policies.EffectAllow,
				Selector: policies.Selector{Kind: policies.SelectorCatalogIDs, Catalog: "models", IDs: []string{"resnet18"}},
			}},
		},
	}
	profile, err := persistence.CreateCompatibilityProfile(profileInput)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := persistence.CreateCompatibilityProfile(profileInput); !errors.Is(err, policies.ErrImmutableConflict) {
		t.Fatalf("duplicate profile error = %v", err)
	}
	profile.Document.Rules[0].Selector.IDs[0] = "vit_b_16"
	reloadedProfile, err := persistence.GetCompatibilityProfile(profile.ProfileKey, profile.SemanticVersion)
	if err != nil {
		t.Fatal(err)
	}
	if got := reloadedProfile.Document.Rules[0].Selector.IDs[0]; got != "resnet18" {
		t.Fatalf("stored profile mutated through return value: %s", got)
	}
	version, err := persistence.CreateExperimentPolicyVersion(policies.PolicyVersion{
		OwnerAccountID: project.AccountID,
		Document: policies.PolicyDocument{
			SchemaVersion: policies.PolicySchemaVersionV1,
			Rules: []policies.Rule{{
				ID: "deny_vit", Effect: policies.EffectDeny,
				Selector: policies.Selector{Kind: policies.SelectorCatalogIDs, Catalog: "models", IDs: []string{"vit_b_16"}},
			}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	expected := int64(0)
	binding, err := persistence.SetExperimentPolicyBinding(policies.BindingWrite{
		Scope: policies.ScopeProject, SubjectID: project.ID,
		PolicyVersionID: version.ID, ExpectedRevision: &expected,
	})
	if err != nil {
		t.Fatal(err)
	}
	binding.PolicyVersionID = "mutated"
	active, err := persistence.ListActiveExperimentPolicyBindings(policies.ScopeContext{AccountID: project.AccountID, ProjectID: project.ID})
	if err != nil {
		t.Fatal(err)
	}
	if len(active) != 1 || active[0].PolicyVersionID != version.ID {
		t.Fatalf("stored binding mutated through return value: %#v", active)
	}
}

func TestPostgresExperimentPolicyParityIntegration(t *testing.T) {
	databaseURL := os.Getenv("MODEL_EXPRESS_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("MODEL_EXPRESS_TEST_DATABASE_URL is not set")
	}
	postgres, err := NewPostgresStore(context.Background(), databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer postgres.Close()
	suffix := strconv.FormatInt(time.Now().UnixNano(), 10)
	want := buildPolicyParityFixture(t, NewMemoryStore(), "memory_"+suffix)
	got := buildPolicyParityFixture(t, postgres, "postgres_"+suffix)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("memory/postgres policy behavior differs:\nmemory=%#v\npostgres=%#v", want, got)
	}
}

func TestPostgresExperimentPolicyMigrationAndParityMinimalIntegration(t *testing.T) {
	databaseURL := os.Getenv("MODEL_EXPRESS_TEST_POLICY_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("MODEL_EXPRESS_TEST_POLICY_DATABASE_URL is not set")
	}
	db, err := sql.Open("pgx", databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := db.PingContext(context.Background()); err != nil {
		t.Fatal(err)
	}
	for _, migration := range []string{"001_init.sql", "013_attempt_execution_records.sql", "024_experiment_policies.sql", "024_experiment_policies.sql"} {
		sqlText, err := migrationFiles.ReadFile("migrations/" + migration)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := db.ExecContext(context.Background(), string(sqlText)); err != nil {
			t.Fatalf("apply %s: %v", migration, err)
		}
	}
	postgres := &PostgresStore{db: db}
	suffix := strconv.FormatInt(time.Now().UnixNano(), 10)
	want := buildPolicyParityFixtureWithoutJob(t, NewMemoryStore(), "memory_minimal_"+suffix)
	got := buildPolicyParityFixtureWithoutJob(t, postgres, "postgres_minimal_"+suffix)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("minimal memory/postgres policy behavior differs:\nmemory=%#v\npostgres=%#v", want, got)
	}
}

func buildPolicyParityFixture(t *testing.T, persistence Store, suffix string) policyParitySummary {
	t.Helper()
	return buildPolicyParityFixtureWithJobOption(t, persistence, suffix, true)
}

func buildPolicyParityFixtureWithoutJob(t *testing.T, persistence Store, suffix string) policyParitySummary {
	t.Helper()
	return buildPolicyParityFixtureWithJobOption(t, persistence, suffix, false)
}

func buildPolicyParityFixtureWithJobOption(t *testing.T, persistence Store, suffix string, includeJob bool) policyParitySummary {
	t.Helper()
	project, err := persistence.CreateProject("policy parity "+suffix, "")
	if err != nil {
		t.Fatal(err)
	}
	dataset, err := persistence.CreateDataset(project.ID, "dataset", "memory://dataset/"+suffix, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	jobID := ""
	if includeJob {
		job, err := persistence.CreateJob(project.ID, jobs.TemplateTrainExperiment, map[string]any{"dataset_id": dataset.ID})
		if err != nil {
			t.Fatal(err)
		}
		jobID = job.ID
	}
	profile, err := persistence.CreateCompatibilityProfile(policies.CompatibilityProfile{
		ProfileKey:      "parity_" + suffix,
		SemanticVersion: "1.0.0",
		Document: policies.CompatibilityProfileDocument{
			SchemaVersion:  policies.CompatibilityProfileSchemaVersionV1,
			CatalogVersion: catalog.CatalogV1().CatalogVersion,
			Rules: []policies.Rule{{
				ID: "allow_models", Effect: policies.EffectAllow,
				Selector: policies.Selector{Kind: policies.SelectorCatalogIDs, Catalog: "models", IDs: []string{"resnet18"}},
			}},
		},
		CreatedBy: "parity",
	})
	if err != nil {
		t.Fatal(err)
	}
	reloadedProfile, err := persistence.GetCompatibilityProfile(profile.ProfileKey, profile.SemanticVersion)
	if err != nil {
		t.Fatal(err)
	}
	policyDocument := policies.PolicyDocument{
		SchemaVersion: policies.PolicySchemaVersionV1,
		ProfileRefs:   []policies.ProfileRef{},
		Rules: []policies.Rule{
			{
				ID: "deny_letterbox", Effect: policies.EffectDeny,
				Selector: policies.Selector{Kind: policies.SelectorCatalogIDs, Catalog: "resize_strategies", IDs: []string{"letterbox"}},
			},
			{
				ID: "deny_erasing", Effect: policies.EffectDeny,
				Selector: policies.Selector{Kind: policies.SelectorCatalogIDs, Catalog: "augmentation_operations", IDs: []string{"random_erasing"}},
			},
		},
	}
	version, err := persistence.CreateExperimentPolicyVersion(policies.PolicyVersion{
		OwnerAccountID: project.AccountID,
		Document:       policyDocument,
		CreatedBy:      "parity",
	})
	if err != nil {
		t.Fatal(err)
	}
	reloadedVersion, err := persistence.GetExperimentPolicyVersion(version.ID)
	if err != nil {
		t.Fatal(err)
	}
	expected := int64(0)
	binding, err := persistence.SetExperimentPolicyBinding(policies.BindingWrite{
		Scope: policies.ScopeProject, SubjectID: project.ID,
		PolicyVersionID: version.ID, ExpectedRevision: &expected, CreatedBy: "parity",
	})
	if err != nil {
		t.Fatal(err)
	}
	replacement, err := persistence.CreateExperimentPolicyVersion(policies.PolicyVersion{
		OwnerAccountID: project.AccountID, Document: policyDocument, CreatedBy: "parity replacement",
	})
	if err != nil {
		t.Fatal(err)
	}
	staleRevision := int64(0)
	_, staleErr := persistence.SetExperimentPolicyBinding(policies.BindingWrite{
		Scope: policies.ScopeProject, SubjectID: project.ID,
		PolicyVersionID: replacement.ID, ExpectedRevision: &staleRevision, CreatedBy: "parity stale",
	})
	staleRejected := errors.Is(staleErr, policies.ErrRevisionConflict)
	if !staleRejected {
		t.Fatalf("stale policy binding write error = %v", staleErr)
	}
	expected = binding.Revision
	binding, err = persistence.SetExperimentPolicyBinding(policies.BindingWrite{
		Scope: policies.ScopeProject, SubjectID: project.ID,
		PolicyVersionID: replacement.ID, ExpectedRevision: &expected, CreatedBy: "parity replacement",
	})
	if err != nil {
		t.Fatal(err)
	}
	version = replacement
	reloadedVersion, err = persistence.GetExperimentPolicyVersion(version.ID)
	if err != nil {
		t.Fatal(err)
	}
	resolver := policies.NewResolver(persistence)
	result, evaluation, err := resolver.ResolveAndRecord(policies.ScopeContext{
		AccountID: project.AccountID, ProjectID: project.ID,
		DatasetID: dataset.ID, ExperimentJobID: jobID,
	}, "parity_preview", "parity", suffix)
	if err != nil {
		t.Fatal(err)
	}
	reloadedEvaluation, err := persistence.GetExperimentPolicyEvaluation(evaluation.ID)
	if err != nil {
		t.Fatal(err)
	}
	evaluations, err := persistence.ListExperimentPolicyEvaluations(project.ID)
	if err != nil {
		t.Fatal(err)
	}
	findingKeys := make([]string, 0, len(result.Findings))
	for _, finding := range result.Findings {
		findingKeys = append(findingKeys, string(finding.Code)+"/"+finding.Catalog+"/"+finding.ID+"/"+finding.Origin)
	}
	sort.Strings(findingKeys)
	return policyParitySummary{
		AccountID:           project.AccountID,
		PolicyDocumentHash:  reloadedVersion.DocumentHash,
		ProfileDocumentHash: reloadedProfile.DocumentHash,
		BindingRevision:     binding.Revision,
		BindingSuperseded:   binding.SupersedesID != "",
		StaleWriteRejected:  staleRejected,
		Decision:            result.Decision,
		PermittedCatalog:    result.Snapshot.PermittedCatalog,
		FindingKeys:         findingKeys,
		EvaluationDecision:  reloadedEvaluation.Decision,
		EvaluationCount:     len(evaluations),
	}
}

func containsID(values []string, expected string) bool {
	for _, value := range values {
		if value == expected {
			return true
		}
	}
	return false
}
