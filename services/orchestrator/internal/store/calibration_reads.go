package store

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"model-express/services/orchestrator/internal/calibration"
	"model-express/services/orchestrator/internal/memory"
)

func validateCalibrationRead(projectID string, window calibration.TimeWindow, limit int) error {
	if strings.TrimSpace(projectID) == "" {
		return fmt.Errorf("%w: calibration project_id is required", ErrInvalidRequest)
	}
	if window.Start.IsZero() || window.End.IsZero() || !window.Start.Before(window.End) {
		return fmt.Errorf("%w: calibration read window requires start before end", ErrInvalidRequest)
	}
	if limit < 1 || limit > calibration.MaximumCalibrationReadLimit {
		return fmt.Errorf("%w: calibration read limit must be between 1 and %d", ErrInvalidRequest, calibration.MaximumCalibrationReadLimit)
	}
	return nil
}

func calibrationInvocationObservation(invocation memory.AgentInvocation) calibration.InvocationObservation {
	observation := calibration.InvocationObservation{
		ID: invocation.ID, PlannerVariantID: invocation.PlannerVariantID,
		AttemptGroupID: invocation.AttemptGroupID, AttemptIndex: invocation.AttemptIndex,
		Parsed: len(invocation.ParsedOutput) > 0, ValidationStatus: invocation.ValidationStatus,
		WallLatencyMS: invocation.WallLatencyMS, CreatedAt: invocation.CreatedAt.UTC(),
	}
	if invocation.ValidationOutcome != nil {
		observation.FirstPassStatus = invocation.ValidationOutcome.FirstPassStatus
		observation.EventualStatus = invocation.ValidationOutcome.EventualStatus
		observation.RetryOutcome = invocation.ValidationOutcome.RetryOutcome
	}
	observation.InputTokens = calibrationInt(invocation.ProviderUsage["input_tokens"])
	observation.OutputTokens = calibrationInt(invocation.ProviderUsage["output_tokens"])
	observation.TotalTokens = calibrationInt(invocation.ProviderUsage["total_tokens"])
	observation.ToolRounds = calibrationInt(invocation.ProviderUsage["tool_rounds"])
	if invocation.DerivedCost != nil {
		observation.PricingVersion = strings.TrimSpace(invocation.DerivedCost.PricingVersion)
		observation.CostUSD = calibrationDecimal(invocation.DerivedCost.TotalCostUSD)
	}
	return observation
}

func calibrationInt(value any) int {
	switch typed := value.(type) {
	case int:
		return max(typed, 0)
	case int64:
		if typed > 0 {
			return int(typed)
		}
	case float64:
		if typed > 0 && typed <= float64(math.MaxInt) {
			return int(typed)
		}
	case string:
		parsed, err := strconv.Atoi(strings.TrimSpace(typed))
		if err == nil && parsed > 0 {
			return parsed
		}
	}
	return 0
}

func calibrationDecimal(value string) *float64 {
	parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	if err != nil || math.IsNaN(parsed) || math.IsInf(parsed, 0) || parsed < 0 {
		return nil
	}
	return &parsed
}
