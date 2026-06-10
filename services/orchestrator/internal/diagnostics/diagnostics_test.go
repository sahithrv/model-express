package diagnostics

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEventWritesRedactedJSONL(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("MODEL_EXPRESS_LOG_DIR", dir)

	Event("error", "llm_failed", map[string]any{
		"project_id":     "project_1",
		"api_key":        "sk-secret",
		"request_url":    "https://api.openai.com/v1/responses",
		"error":          "400 Bad Request: Invalid schema",
		"sample_base64":  "data:image/jpeg;base64,/9j/4AAQSkZJRg",
		"nested":         map[string]any{"authorization": "Bearer token"},
		"raw_output":     `{"too":"much"}`,
		"validation_ids": []string{"a", "b"},
	})

	body, err := os.ReadFile(filepath.Join(dir, "orchestrator.jsonl"))
	if err != nil {
		t.Fatalf("read log: %v", err)
	}
	if strings.Contains(string(body), "sk-secret") || strings.Contains(string(body), "api.openai.com") || strings.Contains(string(body), "/9j/4AAQSkZJRg") {
		t.Fatalf("expected sensitive fields to be redacted, got %s", body)
	}
	var record map[string]any
	if err := json.Unmarshal(bytesTrimSpace(body), &record); err != nil {
		t.Fatalf("decode log: %v", err)
	}
	if record["event"] != "llm_failed" || record["component"] != "orchestrator" {
		t.Fatalf("unexpected record: %#v", record)
	}
	if record["project_id"] != "project_1" {
		t.Fatalf("expected project id, got %#v", record["project_id"])
	}
}

func TestEventRotatesJSONL(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("MODEL_EXPRESS_LOG_DIR", dir)
	t.Setenv("MODEL_EXPRESS_LOG_MAX_BYTES", "1")
	t.Setenv("MODEL_EXPRESS_LOG_MAX_FILES", "2")

	Event("info", "first", map[string]any{"payload": "x"})
	Event("info", "second", map[string]any{"payload": "y"})

	if _, err := os.Stat(filepath.Join(dir, "orchestrator.jsonl")); err != nil {
		t.Fatalf("current log missing: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "orchestrator.jsonl.1")); err != nil {
		t.Fatalf("rotated log missing: %v", err)
	}
}

func bytesTrimSpace(value []byte) []byte {
	return []byte(strings.TrimSpace(string(value)))
}
