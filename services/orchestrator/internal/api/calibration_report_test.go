package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/memory"
	"model-express/services/orchestrator/internal/store"
)

func TestCalibrationReportAPIIsBoundedAndReadOnly(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, err := memoryStore.CreateProject("calibration", "read-only report")
	if err != nil {
		t.Fatal(err)
	}
	router := NewRouter(memoryStore)
	path := "/projects/" + project.ID + "/calibration-report?training_start=2026-01-01T00:00:00Z&training_end=2026-02-01T00:00:00Z&evaluation_start=2026-02-01T00:00:00Z&evaluation_end=2026-03-01T00:00:00Z&limit=17&min_cohort_size=3&meaningful_improvement=0.02"
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, path, nil)
	router.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	var report calibration.Report
	if err := json.Unmarshal(recorder.Body.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.ReportVersion != calibration.CalibrationReportVersionV1 || report.ReadBounds.PerWindowLimit != 17 || report.MinCohortSize != 3 || report.MeaningfulImprovement != 0.02 {
		t.Fatalf("report bounds/version=%#v", report)
	}
	if report.SelectionBiasDisclosure == "" || report.WindowDisclosure == "" || report.CohortLimit != calibration.MaximumCalibrationReportCohorts {
		t.Fatalf("report disclosures/bounds=%#v", report)
	}
	decisions, err := memoryStore.ListProjectAgentDecisions(project.ID)
	if err != nil || len(decisions) != 0 {
		t.Fatalf("report mutated decisions: %#v err=%v", decisions, err)
	}
	invocations, err := memoryStore.ListProjectAgentInvocations(project.ID, memory.AgentInvocationFilter{Limit: 1})
	if err != nil || len(invocations) != 0 {
		t.Fatalf("report mutated invocations: %#v err=%v", invocations, err)
	}
}

func TestCalibrationReportAPIRejectsUnboundedOrOverlappingQueries(t *testing.T) {
	memoryStore := store.NewMemoryStore()
	project, _ := memoryStore.CreateProject("calibration", "bounds")
	router := NewRouter(memoryStore)
	for _, query := range []string{
		"limit=5001",
		"training_start=2026-01-01T00:00:00Z&training_end=2026-03-01T00:00:00Z&evaluation_start=2026-02-01T00:00:00Z&evaluation_end=2026-04-01T00:00:00Z",
		"evaluation_start=not-a-time",
	} {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(http.MethodGet, "/projects/"+project.ID+"/calibration-report?"+query, nil)
		router.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusBadRequest || !strings.Contains(recorder.Body.String(), "error") {
			t.Fatalf("query=%s status=%d body=%s", query, recorder.Code, recorder.Body.String())
		}
	}
}
