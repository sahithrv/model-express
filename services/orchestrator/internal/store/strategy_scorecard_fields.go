package store

import (
	"strings"

	"model-express/services/orchestrator/internal/strategies"
)

func hydrateStrategyScorecardMechanismFields(scorecard strategies.StrategyScorecardCreate) (string, string, []string, []string, string) {
	proposedChanges := emptyMapIfNil(scorecard.ProposedChanges)
	mechanism := strings.TrimSpace(scorecard.Mechanism)
	intervention := strings.TrimSpace(scorecard.Intervention)
	diagnosisTriggers := cleanStringList(scorecard.DiagnosisTriggers)
	evidenceUsed := cleanStringList(scorecard.EvidenceUsed)
	expectedEffect := strings.TrimSpace(scorecard.ExpectedEffect)

	if mechanism == "" {
		mechanism = firstNonEmptyString(proposedChanges, "mechanism", "mechanism_group")
	}
	if intervention == "" {
		intervention = firstNonEmptyString(proposedChanges, "intervention")
	}
	if len(diagnosisTriggers) == 0 {
		diagnosisTriggers = stringsFromAnyValue(proposedChanges["diagnosis_triggers"])
	}
	if len(evidenceUsed) == 0 {
		evidenceUsed = stringsFromAnyValue(proposedChanges["evidence_used"])
	}
	if expectedEffect == "" {
		expectedEffect = firstNonEmptyString(proposedChanges, "expected_effect")
	}

	for _, key := range []string{"proposal_mechanisms", "candidate_hypotheses", "proposed_experiments"} {
		for _, item := range mapsFromAnySlice(proposedChanges[key]) {
			if mechanism == "" {
				mechanism = firstNonEmptyString(item, "mechanism", "mechanism_group")
			}
			if intervention == "" {
				intervention = firstNonEmptyString(item, "intervention")
			}
			if len(diagnosisTriggers) == 0 {
				diagnosisTriggers = stringsFromAnyValue(item["diagnosis_triggers"])
			}
			if len(evidenceUsed) == 0 {
				evidenceUsed = stringsFromAnyValue(item["evidence_used"])
			}
			if expectedEffect == "" {
				expectedEffect = firstNonEmptyString(item, "expected_effect")
			}
			if mechanism != "" && intervention != "" && len(diagnosisTriggers) > 0 && len(evidenceUsed) > 0 && expectedEffect != "" {
				return mechanism, intervention, diagnosisTriggers, evidenceUsed, expectedEffect
			}
		}
	}

	return mechanism, intervention, diagnosisTriggers, evidenceUsed, expectedEffect
}

func firstNonEmptyString(values map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := values[key].(string); ok {
			if normalized := strings.TrimSpace(value); normalized != "" {
				return normalized
			}
		}
	}
	return ""
}

func stringsFromAnyValue(value any) []string {
	switch typed := value.(type) {
	case []string:
		return cleanStringList(typed)
	case []any:
		out := make([]string, 0, len(typed))
		for _, item := range typed {
			if text, ok := item.(string); ok {
				out = append(out, text)
			}
		}
		return cleanStringList(out)
	default:
		return nil
	}
}

func mapsFromAnySlice(value any) []map[string]any {
	switch typed := value.(type) {
	case []map[string]any:
		return typed
	case []any:
		out := make([]map[string]any, 0, len(typed))
		for _, item := range typed {
			if mapped, ok := item.(map[string]any); ok {
				out = append(out, mapped)
			}
		}
		return out
	default:
		return nil
	}
}

func cleanStringList(values []string) []string {
	out := []string{}
	seen := map[string]bool{}
	for _, value := range values {
		normalized := strings.TrimSpace(value)
		if normalized == "" {
			continue
		}
		key := strings.ToLower(normalized)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, normalized)
	}
	return out
}
