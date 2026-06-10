package store

import (
	"strings"
	"testing"
)

func TestPollJobCandidateQueryPushesProviderAndTemplateFiltersIntoSQL(t *testing.T) {
	query, args := pollJobCandidateQuery("project_1", JobPollFilter{
		Provider:                            "modal",
		Templates:                           []string{"train_experiment"},
		IncludeUnspecifiedProviderTemplates: []string{"profile_dataset"},
	})

	if !strings.Contains(query, "lower(template) IN") {
		t.Fatalf("expected template filter in query: %s", query)
	}
	if !strings.Contains(query, "config->>'provider'") {
		t.Fatalf("expected provider filter in query: %s", query)
	}
	if !strings.Contains(query, "FOR UPDATE SKIP LOCKED") || !strings.Contains(query, "LIMIT 1") {
		t.Fatalf("expected locked single-row poll query: %s", query)
	}
	if got := len(args); got != 5 {
		t.Fatalf("expected 5 query args, got %d: %#v", got, args)
	}
	if args[2] != "train_experiment" || args[3] != "modal" || args[4] != "profile_dataset" {
		t.Fatalf("unexpected filter args: %#v", args)
	}
}
