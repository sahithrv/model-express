package jobs

import "testing"

func TestWithExecutionSpecStatusMarksVersionedAndLegacyTrainingJobs(t *testing.T) {
	legacy := WithExecutionSpecStatus(ExperimentJob{
		Template: TemplateTrainExperiment,
		Config:   map[string]any{"model": "resnet18"},
	})
	if legacy.ExecutionSpecStatus != "legacy_unversioned" {
		t.Fatalf("legacy training job status = %q", legacy.ExecutionSpecStatus)
	}
	versioned := WithExecutionSpecStatus(ExperimentJob{
		Template: TemplateTrainExperiment,
		Config: map[string]any{
			"execution_spec_v1": map[string]any{"schema_version": "execution_spec_v1"},
		},
	})
	if versioned.ExecutionSpecStatus != "versioned" {
		t.Fatalf("versioned training job status = %q", versioned.ExecutionSpecStatus)
	}
	notApplicable := WithExecutionSpecStatus(ExperimentJob{
		Template: TemplateProfileDataset,
		Config:   map[string]any{},
	})
	if notApplicable.ExecutionSpecStatus != "" {
		t.Fatalf("non-training job should not have execution spec status: %#v", notApplicable)
	}
}
