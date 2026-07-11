package agents

import (
	"encoding/json"
	"testing"

	"model-express/services/orchestrator/internal/execution"
)

func TestPlannerContextIncludesBoundedExecutionCapabilitiesAndFeedback(t *testing.T) {
	card, err := execution.BuildPlannerCapabilityCard(
		"object_detection", "modal_ultralytics", execution.ValidationModeEnforce, []string{"yolo11"},
	)
	if err != nil {
		t.Fatalf("build capability card: %v", err)
	}
	input := ExperimentPlannerInput{
		ExecutionCapabilityCard: card,
		ExecutionEnforcementFeedback: []execution.EnforcementFeedback{{
			Task: "object_detection", Runner: "modal_ultralytics", ModelFamily: "yolo11",
			Field: "class_balancing", Classification: "unsupported", ReasonCode: "task_not_applicable",
			Count: 3, SuggestedAlternative: "omit class_balancing and tune an executed detector field",
		}},
	}
	snapshot := BuildPlannerContextSnapshot(input)
	if snapshot.ExecutionCapabilities.Task != "object_detection" || snapshot.ExecutionCapabilities.Runner != "modal_ultralytics" {
		t.Fatalf("planner received the wrong task capability card: %#v", snapshot.ExecutionCapabilities)
	}
	if len(snapshot.EnforcementFeedback) != 1 || snapshot.EnforcementFeedback[0].Field != "class_balancing" {
		t.Fatalf("planner lost durable enforcement feedback: %#v", snapshot.EnforcementFeedback)
	}
	blob, err := json.Marshal(snapshot.ExecutionCapabilities)
	if err != nil {
		t.Fatalf("marshal capability card: %v", err)
	}
	if len(blob) > 16*1024 {
		t.Fatalf("capability card exceeded bounded prompt budget: %d bytes", len(blob))
	}
}
