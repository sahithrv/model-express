package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"model-express/services/orchestrator/internal/agents/evals"
	"model-express/services/orchestrator/internal/llm"
)

const defaultMaxJSONLBytes = 2 * 1024 * 1024

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "planner-eval:", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		live             = flag.Bool("live", false, "run opt-in paired provider evaluation instead of deterministic rubric scoring")
		fixturePaths     = flag.String("fixtures", "", "comma-separated fixture JSON paths; defaults to the complete embedded scenario corpus")
		repeats          = flag.Int("repeats", 1, "live generations per variant")
		maxAttempts      = flag.Int("max-attempts", 1, "live attempts per variant/repeat (1-3)")
		maxProviderCalls = flag.Int("max-provider-calls", 0, "required hard live provider-call reservation budget")
		maxRequestBytes  = flag.Int("max-request-bytes", 0, "required hard live initial-request byte budget")
		maxTotalTokens   = flag.Int("max-total-tokens", 0, "optional observed live token ceiling")
		maxJSONLBytes    = flag.Int("max-jsonl-bytes", defaultMaxJSONLBytes, "maximum bytes per emitted JSONL record")
		timeout          = flag.Duration("timeout", 10*time.Minute, "overall live evaluation timeout")
		checkBaseline    = flag.Bool("check-baseline", false, "compare deterministic results with the checked baseline")
		gateEfficiency   = flag.Bool("gate-efficiency", false, "fail baseline checks on response-byte regressions instead of warning")
		baselinePath     = flag.String("baseline", "", "optional baseline JSON path; defaults to the embedded checked baseline")
		updateBaseline   = flag.String("update-baseline", "", "write a reviewed deterministic baseline to this path")
	)
	flag.Parse()

	if *maxJSONLBytes < 1024 || *maxJSONLBytes > 16*1024*1024 {
		return fmt.Errorf("max-jsonl-bytes must be between 1024 and 16777216")
	}
	if *gateEfficiency && !*checkBaseline {
		return errors.New("-gate-efficiency requires -check-baseline")
	}
	fixtures, err := loadFixtures(*fixturePaths, *live)
	if err != nil {
		return err
	}
	if !*live {
		artifact, err := evals.EvaluatePlannerRubricFixtures(fixtures)
		if err != nil {
			return err
		}
		if strings.TrimSpace(*updateBaseline) != "" {
			if *checkBaseline || *gateEfficiency {
				return errors.New("-update-baseline cannot be combined with -check-baseline or -gate-efficiency")
			}
			return evals.WritePlannerRubricBaseline(filepath.Clean(*updateBaseline), artifact)
		}
		if *checkBaseline {
			var baseline evals.PlannerRubricBaseline
			if strings.TrimSpace(*baselinePath) == "" {
				baseline, err = evals.LoadCheckedPlannerRubricBaseline()
			} else {
				baseline, err = evals.LoadPlannerRubricBaseline(filepath.Clean(*baselinePath))
			}
			if err != nil {
				return err
			}
			comparison := evals.ComparePlannerRubricBaselineWithOptions(baseline, artifact, evals.PlannerBaselineGateOptions{
				GateEfficiency: *gateEfficiency,
			})
			if err := writeBoundedJSONL(comparison, *maxJSONLBytes); err != nil {
				return err
			}
			if !comparison.Passed {
				return errors.New("planner rubric baseline tolerances were exceeded")
			}
			return nil
		}
		return writeBoundedJSONL(artifact, *maxJSONLBytes)
	}
	if *checkBaseline || *gateEfficiency || strings.TrimSpace(*updateBaseline) != "" || strings.TrimSpace(*baselinePath) != "" {
		return errors.New("baseline flags are available only for deterministic evaluation")
	}

	if !liveEvalEnabled() {
		return errors.New("live evaluation requires MODEL_EXPRESS_PLANNER_EVAL_LIVE=true")
	}
	if *maxProviderCalls <= 0 || *maxRequestBytes <= 0 {
		return errors.New("live evaluation requires positive -max-provider-calls and -max-request-bytes budgets")
	}
	runtime := llm.ConfigFromEnv(true, "", "")
	pricing, err := llm.PricingSnapshotFromEnv()
	if err != nil {
		return err
	}
	client := llm.NewClient(runtime)
	variants := evals.DefaultPlannerEvalVariants()
	remaining := evals.PlannerEvalBudget{
		MaxProviderCalls: *maxProviderCalls,
		MaxRequestBytes:  *maxRequestBytes,
		MaxTotalTokens:   *maxTotalTokens,
	}
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	tokenBudgetEnabled := *maxTotalTokens > 0

	for fixtureIndex, fixture := range fixtures {
		artifact, err := evals.RunPlannerPairedEvaluation(ctx, client, runtime.Model, runtime, fixture, evals.PlannerPairedEvalConfig{
			Variants:    variants,
			Repeats:     *repeats,
			MaxAttempts: *maxAttempts,
			Budget:      remaining,
			Pricing:     pricing,
		})
		if err != nil {
			return err
		}
		if err := writeBoundedJSONL(artifact, *maxJSONLBytes); err != nil {
			return err
		}
		remaining.MaxProviderCalls -= artifact.BudgetUsage.ProviderCallsReserved
		remaining.MaxRequestBytes -= artifact.BudgetUsage.RequestBytes
		if tokenBudgetEnabled {
			remaining.MaxTotalTokens -= artifact.BudgetUsage.TotalTokens
			if remaining.MaxTotalTokens <= 0 && fixtureIndex+1 < len(fixtures) {
				return errors.New("observed live token budget is exhausted before all fixtures ran")
			}
		}
	}
	return nil
}

func loadFixtures(value string, live bool) ([]evals.PlannerReplayFixture, error) {
	if strings.TrimSpace(value) == "" {
		if live {
			return evals.LoadStarterPlannerRubricFixtures()
		}
		return evals.LoadPlannerRubricFixtures()
	}
	fixtures := []evals.PlannerReplayFixture{}
	for _, path := range strings.Split(value, ",") {
		path = filepath.Clean(strings.TrimSpace(path))
		if path == "." || path == "" {
			return nil, errors.New("fixture path must not be empty")
		}
		fixture, err := evals.LoadPlannerReplayFixture(path)
		if err != nil {
			return nil, fmt.Errorf("load fixture %s: %w", path, err)
		}
		fixtures = append(fixtures, fixture)
	}
	return fixtures, nil
}

func writeBoundedJSONL(value any, maxBytes int) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if len(encoded)+1 > maxBytes {
		return fmt.Errorf("JSONL record is %d bytes; limit is %d", len(encoded)+1, maxBytes)
	}
	_, err = os.Stdout.Write(append(encoded, '\n'))
	return err
}

func liveEvalEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_PLANNER_EVAL_LIVE"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
