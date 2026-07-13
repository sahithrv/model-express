package jobs

import (
	"math"
	"strings"
	"testing"
)

func TestNormalizeJobProgressUpsertValidatesTaxonomyAndBounds(t *testing.T) {
	current := int64(2)
	total := int64(5)
	update, err := NormalizeJobProgressUpsert(JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: ProgressTaxonomyVersion,
		Stage: " TRAINING ", Status: " RUNNING ", DetailCode: " EPOCH ",
		Current: &current, Total: &total, Unit: " EPOCHS ", Revision: 4,
		Message: " Epoch 2 of 5 ", Metadata: map[string]any{"provider": "modal", "warm": true},
	})
	if err != nil {
		t.Fatal(err)
	}
	if update.Stage != ProgressStageTraining || update.Status != ProgressStatusRunning || update.DetailCode != "epoch" || update.Unit != "epochs" {
		t.Fatalf("tokens were not normalized: %#v", update)
	}
	if update.Message != "Epoch 2 of 5" || update.Metadata["provider"] != "modal" {
		t.Fatalf("safe fields were not preserved: %#v", update)
	}

	cases := []JobProgressUpsert{
		{Attempt: -1, TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageQueued, Status: ProgressStatusQueued},
		{Attempt: 1, TaxonomyVersion: 2, Stage: ProgressStageQueued, Status: ProgressStatusQueued},
		{Attempt: 1, TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageCompleted, Status: ProgressStatusRunning},
		{Attempt: 1, TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Message: strings.Repeat("x", JobProgressMaxMessageBytes+1)},
		{Attempt: 1, TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Metadata: map[string]any{"secret": map[string]any{"token": "no"}}},
		{Attempt: 1, TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Metadata: map[string]any{"metric": math.NaN()}},
		{Attempt: 1, TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Message: "loading s3://private-bucket/dataset"},
		{Attempt: 1, TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Metadata: map[string]any{"source": "C:\\Users\\private\\dataset"}},
		{Attempt: 1, TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Metadata: map[string]any{"api_key": "plain-secret"}},
	}
	for index, candidate := range cases {
		if _, err := NormalizeJobProgressUpsert(candidate); err == nil {
			t.Fatalf("case %d unexpectedly accepted: %#v", index, candidate)
		}
	}
}

func TestNormalizeJobProgressUpsertCopiesMutableValues(t *testing.T) {
	current := int64(1)
	values := []string{"a"}
	update, err := NormalizeJobProgressUpsert(JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: ProgressTaxonomyVersion,
		Stage: ProgressStageTraining, Status: ProgressStatusRunning,
		Current: &current, Metadata: map[string]any{"labels": values},
	})
	if err != nil {
		t.Fatal(err)
	}
	current = 9
	values[0] = "mutated"
	if *update.Current != 1 || update.Metadata["labels"].([]string)[0] != "a" {
		t.Fatalf("normalized snapshot retained caller-owned values: %#v", update)
	}
}

func TestNormalizeWorkerJobProgressUpsertEnforcesCallbackContract(t *testing.T) {
	current, total := int64(1), int64(10)
	valid, err := NormalizeWorkerJobProgressUpsert(JobProgressUpsert{
		Attempt: 1, TaxonomyVersion: ProgressTaxonomyVersion,
		Stage: ProgressStageTraining, Status: ProgressStatusRunning,
		Current: &current, Total: &total, Unit: "epoch", Revision: 3,
		DetailCode: "epoch.complete", Message: "Epoch one finished.",
		Metadata: map[string]any{"provider": "modal", "early_stopped": false},
	})
	if err != nil {
		t.Fatal(err)
	}
	if valid.Stage != ProgressStageTraining || valid.Revision != 3 || valid.Unit != "epoch" {
		t.Fatalf("normalized worker progress=%#v", valid)
	}

	invalid := []JobProgressUpsert{
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageQueued, Status: ProgressStatusQueued, Revision: 3},
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageCompleted, Status: ProgressStatusCompleted, Revision: 3},
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusCompleted, Revision: 3},
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Revision: 0},
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Revision: 3, Current: &current},
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Revision: 3, Current: &current, Total: &total},
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Revision: 3, Unit: "epoch"},
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Revision: 3, Current: &current, Total: &total, Unit: "fortnight"},
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Revision: 3, Metadata: map[string]any{"arbitrary": "value"}},
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Revision: 3, Metadata: map[string]any{"provider": []string{"modal"}}},
		{TaxonomyVersion: ProgressTaxonomyVersion, Stage: ProgressStageTraining, Status: ProgressStatusRunning, Revision: 3, Metadata: map[string]any{"early_stopped": "false"}},
	}
	for index, update := range invalid {
		if _, err := NormalizeWorkerJobProgressUpsert(update); err == nil {
			t.Errorf("invalid worker update %d was accepted: %#v", index, update)
		}
	}
}
