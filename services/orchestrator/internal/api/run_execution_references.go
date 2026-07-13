package api

import (
	"errors"
	"reflect"
	"strings"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

const runExecutionReferencesSchemaV1 = "run_execution_references_v1"

func (s *Server) executionReferencesForJob(jobID string, existing *runs.ExecutionArtifactReferences) *runs.ExecutionArtifactReferences {
	var references runs.ExecutionArtifactReferences
	if existing != nil {
		references = *existing
	}
	record, err := s.store.GetJobExecutionRecord(jobID)
	recordMissing := errors.Is(err, store.ErrNotFound)
	if err == nil {
		evidence := execution.DeriveEvidenceEligibility(&record, execution.LegacyEvidencePolicyAllow)
		references.SchemaVersion = runExecutionReferencesSchemaV1
		references.LifecycleStatus = evidence.LifecycleStatus
		references.FidelityVerdict = evidence.FidelityVerdict
		references.CapabilityVersion = evidence.CapabilityVersion
		references.AcceptedSpecHash = evidence.AcceptedSpecHash
		references.RealizedEffectiveHash = evidence.RealizedEffectiveHash
		references.AdjustmentReasonCodes = append([]string(nil), evidence.AdjustmentReasonCodes...)
		references.ExecutionRecordRef = "/jobs/" + strings.TrimSpace(jobID) + "/execution-record"
	} else if !errors.Is(err, store.ErrNotFound) {
		return existing
	}

	if job, jobErr := s.store.GetJob(jobID); jobErr == nil {
		if recordMissing && job.Template == jobs.TemplateTrainExperiment {
			references.SchemaVersion = runExecutionReferencesSchemaV1
			references.FidelityVerdict = execution.ExecutionVerdictUnverified
		}
		if exports, exportErr := s.store.ListProjectChampionExports(job.ProjectID); exportErr == nil {
			for _, export := range exports {
				if export.JobID != jobID || export.Status != runs.ChampionExportStatusReady {
					continue
				}
				manifestURI := firstNonEmptyString(
					payloadString(export.Metadata, "export_manifest_uri"),
					payloadString(export.Metadata, "manifest_uri"),
				)
				if manifestURI != "" {
					references.ChampionExportManifestURI = manifestURI
					references.PreprocessingContractRef = manifestURI + "#/metadata/preprocessing_contract"
				}
				break
			}
		}
	}
	if reflect.DeepEqual(references, runs.ExecutionArtifactReferences{}) {
		return nil
	}
	if references.SchemaVersion == "" {
		references.SchemaVersion = runExecutionReferencesSchemaV1
	}
	return &references
}

func compactTrainingRunSummaries(summaries []runs.TrainingRunSummary) []map[string]any {
	out := make([]map[string]any, 0, len(summaries))
	for _, summary := range summaries {
		out = append(out, map[string]any{
			"job_id":               summary.JobID,
			"project_id":           summary.ProjectID,
			"status":               summary.Status,
			"execution_references": summary.ExecutionReferences,
		})
	}
	return out
}
