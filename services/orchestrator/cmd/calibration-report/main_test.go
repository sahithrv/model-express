package main

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestCalibrationReportCLIOptionsAreChronologicalAndBounded(t *testing.T) {
	now := time.Date(2026, 7, 14, 12, 0, 0, 0, time.UTC)
	options, err := parseOptions([]string{
		"-project", "project-1", "-limit", "25", "-min-cohort-size", "7",
		"-training-start", "2026-01-01T00:00:00Z", "-training-end", "2026-04-01T00:00:00Z",
		"-evaluation-start", "2026-04-01T00:00:00Z", "-evaluation-end", "2026-05-01T00:00:00Z",
	}, now)
	if err != nil {
		t.Fatal(err)
	}
	if options.Request.ProjectID != "project-1" || options.Request.Limit != 25 || options.Request.MinCohortSize != 7 || options.Request.TrainingWindow.End.After(options.Request.EvaluationWindow.Start) {
		t.Fatalf("options=%#v", options)
	}
	for _, args := range [][]string{
		{"-project", "project-1", "-limit", "5001"},
		{"-project", "project-1", "-max-json-bytes", "10"},
		{"-project", "project-1", "-training-start", "2026-01-01T00:00:00Z", "-training-end", "2026-05-01T00:00:00Z", "-evaluation-start", "2026-04-01T00:00:00Z"},
	} {
		if _, err := parseOptions(args, now); err == nil {
			t.Fatalf("expected bounded option error for %v", args)
		}
	}
}

func TestCalibrationReportCLIOutputBound(t *testing.T) {
	var output bytes.Buffer
	if err := writeBoundedJSON(&output, map[string]any{"ok": true}, 1024); err != nil || !strings.HasSuffix(output.String(), "\n") {
		t.Fatalf("write output=%q err=%v", output.String(), err)
	}
	if err := writeBoundedJSON(&output, strings.Repeat("x", 2000), 1024); err == nil || !strings.Contains(err.Error(), "max-json-bytes") {
		t.Fatalf("output bound error=%v", err)
	}
}
