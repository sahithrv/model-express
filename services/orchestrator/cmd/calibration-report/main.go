package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/config"
	"model-express/services/orchestrator/internal/store"
)

const (
	defaultMaxReportJSONBytes = 4 * 1024 * 1024
	maximumReportJSONBytes    = 16 * 1024 * 1024
)

type cliOptions struct {
	Request      calibration.ReportRequest
	DatabaseURL  string
	MaxJSONBytes int
}

func main() {
	if err := run(os.Args[1:], os.Stdout, time.Now().UTC()); err != nil {
		fmt.Fprintln(os.Stderr, "calibration-report:", err)
		os.Exit(1)
	}
}

func run(args []string, output io.Writer, now time.Time) error {
	options, err := parseOptions(args, now)
	if err != nil {
		return err
	}
	repoRoot := os.Getenv("MODEL_EXPRESS_ROOT")
	if repoRoot == "" {
		repoRoot = filepath.Clean(filepath.Join("..", ".."))
	}
	if err := config.LoadRepoEnv(repoRoot); err != nil {
		return fmt.Errorf("load repository environment: %w", err)
	}
	if options.DatabaseURL == "" {
		options.DatabaseURL = os.Getenv("DATABASE_URL")
	}
	if options.DatabaseURL == "" {
		options.DatabaseURL = "postgres://model_express:model_express@localhost:5432/model_express?sslmode=disable"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	postgresStore, err := store.NewPostgresStore(ctx, options.DatabaseURL)
	if err != nil {
		return fmt.Errorf("connect to orchestrator database: %w", err)
	}
	defer postgresStore.Close()
	report, err := calibration.GenerateReport(postgresStore, options.Request)
	if err != nil {
		return err
	}
	return writeBoundedJSON(output, report, options.MaxJSONBytes)
}

func parseOptions(args []string, now time.Time) (cliOptions, error) {
	flags := flag.NewFlagSet("calibration-report", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	projectID := flags.String("project", "", "project ID (required)")
	databaseURL := flags.String("database-url", "", "Postgres URL; defaults to DATABASE_URL")
	evaluationEndText := flags.String("evaluation-end", now.UTC().Format(time.RFC3339Nano), "evaluation window end (RFC3339)")
	evaluationStartText := flags.String("evaluation-start", "", "evaluation window start (RFC3339; defaults to 30 days before end)")
	trainingEndText := flags.String("training-end", "", "training/prior window end (RFC3339; defaults to evaluation start)")
	trainingStartText := flags.String("training-start", "", "training/prior window start (RFC3339; defaults to 90 days before training end)")
	limit := flags.Int("limit", calibration.DefaultCalibrationReadLimit, "maximum candidate and invocation rows read per window")
	minCohortSize := flags.Int("min-cohort-size", calibration.DefaultCalibrationMinCohortSize, "minimum cohort size before metrics are shown")
	meaningful := flags.Float64("meaningful-improvement", 0.01, "normalized improvement threshold for precision/recall")
	maxJSONBytes := flags.Int("max-json-bytes", defaultMaxReportJSONBytes, "maximum output bytes")
	if err := flags.Parse(args); err != nil {
		return cliOptions{}, err
	}
	if flags.NArg() != 0 {
		return cliOptions{}, fmt.Errorf("unexpected positional arguments: %s", strings.Join(flags.Args(), " "))
	}
	evaluationEnd, err := parseRequiredTime("evaluation-end", *evaluationEndText)
	if err != nil {
		return cliOptions{}, err
	}
	evaluationStart := evaluationEnd.Add(-30 * 24 * time.Hour)
	if strings.TrimSpace(*evaluationStartText) != "" {
		evaluationStart, err = parseRequiredTime("evaluation-start", *evaluationStartText)
		if err != nil {
			return cliOptions{}, err
		}
	}
	trainingEnd := evaluationStart
	if strings.TrimSpace(*trainingEndText) != "" {
		trainingEnd, err = parseRequiredTime("training-end", *trainingEndText)
		if err != nil {
			return cliOptions{}, err
		}
	}
	trainingStart := trainingEnd.Add(-90 * 24 * time.Hour)
	if strings.TrimSpace(*trainingStartText) != "" {
		trainingStart, err = parseRequiredTime("training-start", *trainingStartText)
		if err != nil {
			return cliOptions{}, err
		}
	}
	request, err := calibration.NormalizeReportRequest(calibration.ReportRequest{
		ProjectID:        *projectID,
		TrainingWindow:   calibration.TimeWindow{Start: trainingStart, End: trainingEnd},
		EvaluationWindow: calibration.TimeWindow{Start: evaluationStart, End: evaluationEnd},
		Limit:            *limit, MinCohortSize: *minCohortSize, MeaningfulImprovement: *meaningful,
	})
	if err != nil {
		return cliOptions{}, err
	}
	if *maxJSONBytes < 1024 || *maxJSONBytes > maximumReportJSONBytes {
		return cliOptions{}, fmt.Errorf("max-json-bytes must be between 1024 and %d", maximumReportJSONBytes)
	}
	return cliOptions{Request: request, DatabaseURL: strings.TrimSpace(*databaseURL), MaxJSONBytes: *maxJSONBytes}, nil
}

func parseRequiredTime(name, value string) (time.Time, error) {
	parsed, err := time.Parse(time.RFC3339Nano, strings.TrimSpace(value))
	if err != nil {
		return time.Time{}, fmt.Errorf("%s must be RFC3339: %w", name, err)
	}
	return parsed.UTC(), nil
}

func writeBoundedJSON(output io.Writer, value any, maxBytes int) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(encoded)+1 > maxBytes {
		return fmt.Errorf("report is %d bytes; max-json-bytes is %d", len(encoded)+1, maxBytes)
	}
	_, err = output.Write(append(encoded, '\n'))
	return err
}
