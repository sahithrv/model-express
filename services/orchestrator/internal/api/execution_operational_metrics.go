package api

import (
	"strings"

	"model-express/services/orchestrator/internal/execution"
	"model-express/services/orchestrator/internal/jobs"
	"model-express/services/orchestrator/internal/runs"
	"model-express/services/orchestrator/internal/store"
)

const executionOperationalMetricsSchemaV1 = "execution_fidelity_operational_metrics_v1"

type executionFidelityOperationalMetrics struct {
	SchemaVersion                     string         `json:"schema_version"`
	ValidationMode                    string         `json:"validation_mode"`
	UnsupportedProposals              int            `json:"unsupported_proposals"`
	UnsupportedFindings               int            `json:"unsupported_findings"`
	PendingAttempts                   int            `json:"pending_attempts"`
	InitializedAttempts               int            `json:"initialized_attempts"`
	NotRealizedAttempts               int            `json:"not_realized_attempts"`
	MatchedVerdicts                   int            `json:"matched_verdicts"`
	MismatchVerdicts                  int            `json:"mismatch_verdicts"`
	ApprovedAdjustments               int            `json:"approved_adjustments"`
	SimulatedVerdicts                 int            `json:"simulated_verdicts"`
	LegacyUnverifiedJobs              int            `json:"legacy_unverified_jobs"`
	UnverifiedReads                   int64          `json:"unverified_reads"`
	SuccessfulWithoutFinalRealization int            `json:"successful_without_final_realization"`
	SuccessfulMismatchVerdicts        int            `json:"successful_mismatch_verdicts"`
	MatchedByTask                     map[string]int `json:"matched_by_task"`
}

func (s *Server) executionOperationalMetrics(projectID string, eventLimit int) (executionFidelityOperationalMetrics, error) {
	metrics := executionFidelityOperationalMetrics{
		SchemaVersion:  executionOperationalMetricsSchemaV1,
		ValidationMode: executionValidationMode(),
		MatchedByTask:  map[string]int{},
	}
	projectJobs, err := s.store.ListProjectJobs(projectID)
	if err != nil {
		return metrics, err
	}
	records, err := s.store.ListProjectExecutionRecords(projectID, store.PageOptions{Limit: max(1, len(projectJobs)+1)})
	if err != nil {
		return metrics, err
	}
	events, err := s.store.ListProjectExecutionEvents(projectID, eventLimit)
	if err != nil {
		return metrics, err
	}

	recordsByJob := make(map[string]execution.ExecutionRecord, len(records))
	for _, record := range records {
		recordsByJob[record.AcceptedSpec.JobID] = record
		for _, attempt := range record.Attempts {
			switch strings.ToUpper(strings.TrimSpace(attempt.LifecycleStatus)) {
			case execution.ExecutionLifecyclePending:
				metrics.PendingAttempts++
			case execution.ExecutionLifecycleInitialized:
				metrics.InitializedAttempts++
			case execution.ExecutionLifecycleNotRealized:
				metrics.NotRealizedAttempts++
			}
			if attempt.FidelityVerdict == nil {
				continue
			}
			switch strings.ToUpper(strings.TrimSpace(*attempt.FidelityVerdict)) {
			case execution.ExecutionVerdictMatched:
				metrics.MatchedVerdicts++
				metrics.MatchedByTask[record.AcceptedSpec.Task]++
			case execution.ExecutionVerdictMismatch:
				metrics.MismatchVerdicts++
			case execution.ExecutionVerdictApprovedAdjustment:
				metrics.ApprovedAdjustments++
			case execution.ExecutionVerdictSimulated:
				metrics.SimulatedVerdicts++
			}
		}
	}

	for _, job := range projectJobs {
		if job.Template != jobs.TemplateTrainExperiment {
			continue
		}
		record, versioned := recordsByJob[job.ID]
		if !versioned {
			metrics.LegacyUnverifiedJobs++
			continue
		}
		if job.Status == jobs.StatusSucceeded {
			eligibility := execution.DeriveEvidenceEligibility(&record, execution.LegacyEvidencePolicyVisibleOnly)
			if eligibility.LifecycleStatus != execution.ExecutionLifecycleFinalized {
				metrics.SuccessfulWithoutFinalRealization++
			}
			if eligibility.FidelityVerdict == execution.ExecutionVerdictMismatch {
				metrics.SuccessfulMismatchVerdicts++
			}
		}
	}

	for _, report := range executionValidationReports(projectJobs, events) {
		if !report.WouldBlock {
			continue
		}
		metrics.UnsupportedProposals++
		for _, finding := range report.Findings {
			if finding.WouldBlock {
				metrics.UnsupportedFindings++
			}
		}
	}
	s.fidelityMetricsMu.Lock()
	metrics.UnverifiedReads = s.unverifiedReadsByProject[projectID]
	s.fidelityMetricsMu.Unlock()
	return metrics, nil
}

func (s *Server) recordUnverifiedExecutionRead(projectID string, references ...*runs.ExecutionArtifactReferences) {
	count := int64(0)
	for _, reference := range references {
		if reference != nil && reference.FidelityVerdict == execution.ExecutionVerdictUnverified {
			count++
		}
	}
	if count == 0 || strings.TrimSpace(projectID) == "" {
		return
	}
	s.fidelityMetricsMu.Lock()
	s.unverifiedReadsByProject[projectID] += count
	s.fidelityMetricsMu.Unlock()
}
