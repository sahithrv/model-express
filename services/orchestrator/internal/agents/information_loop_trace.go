package agents

import (
	"encoding/json"
	"strings"

	"model-express/services/orchestrator/internal/llm"
)

type AgentToolCallTrace struct {
	CallID string `json:"call_id"`
	Name   string `json:"name"`
}

type AgentToolResultTrace struct {
	CallID   string         `json:"call_id"`
	Name     string         `json:"name"`
	Accepted bool           `json:"accepted"`
	Error    string         `json:"error,omitempty"`
	Summary  map[string]any `json:"summary,omitempty"`
}

func llmToolDefinitions(specs []AgentInformationToolSpec) []llm.ToolDefinition {
	out := make([]llm.ToolDefinition, 0, len(specs))
	for _, spec := range specs {
		out = append(out, llm.ToolDefinition{
			Type:        "function",
			Name:        spec.Name,
			Description: spec.Description,
			Parameters:  spec.Parameters,
			Strict:      true,
		})
	}
	return out
}

func toolCallTraces(calls []llm.ToolCall) []AgentToolCallTrace {
	out := make([]AgentToolCallTrace, 0, len(calls))
	for _, call := range calls {
		out = append(out, AgentToolCallTrace{
			CallID: call.CallID,
			Name:   call.Name,
		})
	}
	return out
}

func toolResultTraces(results []llm.ToolResult) ([]AgentToolResultTrace, []AgentToolResultTrace, []map[string]any) {
	out := make([]AgentToolResultTrace, 0, len(results))
	rejected := []AgentToolResultTrace{}
	dryRuns := []map[string]any{}
	for _, result := range results {
		trace := AgentToolResultTrace{
			CallID:   result.CallID,
			Name:     result.Name,
			Accepted: result.Error == "",
			Error:    result.Error,
		}
		payload := map[string]any{}
		if len(result.Output) > 0 {
			_ = json.Unmarshal(result.Output, &payload)
		}
		if accepted, ok := payload["accepted"].(bool); ok {
			trace.Accepted = accepted
		}
		if trace.Error == "" {
			trace.Error = stringFromAny(payload["error"])
		}
		trace.Summary = compactToolTraceSummary(payload)
		out = append(out, trace)
		if !trace.Accepted || strings.TrimSpace(trace.Error) != "" {
			rejected = append(rejected, trace)
		}
		if dryRun, ok := payload["payload"].(map[string]any); ok {
			if validation, ok := dryRun["dry_run_validation"].(map[string]any); ok {
				dryRuns = append(dryRuns, compactAnyMap(validation, 12))
			}
		}
		if validation, ok := payload["dry_run_validation"].(map[string]any); ok {
			dryRuns = append(dryRuns, compactAnyMap(validation, 12))
		}
	}
	return out, rejected, dryRuns
}

func compactToolTraceSummary(payload map[string]any) map[string]any {
	if len(payload) == 0 {
		return nil
	}
	summary := map[string]any{}
	for _, key := range []string{
		"accepted",
		"tool_name",
		"tool",
		"error",
		"sample_count",
		"validation_status",
		"proposed_experiment_count",
		"candidate_count",
	} {
		if value, ok := payload[key]; ok {
			summary[key] = compactAnyValue(value)
		}
	}
	if payloadMap, ok := payload["payload"].(map[string]any); ok {
		for _, key := range []string{"dry_run_validation", "message"} {
			if value, ok := payloadMap[key]; ok {
				summary[key] = compactAnyValue(value)
			}
		}
	}
	return compactNonZeroMap(summary)
}
