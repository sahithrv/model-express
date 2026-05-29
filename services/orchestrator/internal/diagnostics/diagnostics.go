package diagnostics

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

const (
	maxLogStringLength = 1800
	maxLogArrayItems   = 24
	maxLogMapItems     = 80
)

var (
	writeMu      sync.Mutex
	unsafeTextRE = regexp.MustCompile(`(?i)(data:image/[a-z0-9.+-]+;base64,|base64\s*[:=,]|bearer\s+[a-z0-9._\-]+|aws_access_key|[A-Za-z]:\\|file://|s3://|/Users/|/home/|/tmp/|\\\\|/9j/4AAQSkZJRg|iVBORw0KGgo)`)
)

// Event appends one bounded JSONL diagnostic record for local debugging.
func Event(level string, event string, fields map[string]any) {
	if strings.TrimSpace(event) == "" {
		return
	}
	record := map[string]any{
		"ts":        time.Now().UTC().Format(time.RFC3339Nano),
		"level":     normalizeLevel(level),
		"component": "orchestrator",
		"event":     event,
	}
	for key, value := range fields {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		record[key] = safeValue(key, value, 0)
	}

	encoded, err := json.Marshal(record)
	if err != nil {
		return
	}

	dir := logDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return
	}
	path := filepath.Join(dir, "orchestrator.jsonl")

	writeMu.Lock()
	defer writeMu.Unlock()
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return
	}
	defer file.Close()
	_, _ = file.Write(append(encoded, '\n'))
}

func logDir() string {
	if value := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_LOG_DIR")); value != "" {
		return value
	}
	if root := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_ROOT")); root != "" {
		return filepath.Join(root, "artifacts", "logs")
	}
	if root := discoverRepoRoot(); root != "" {
		return filepath.Join(root, "artifacts", "logs")
	}
	return filepath.Join(os.TempDir(), "model-express", "logs")
}

func discoverRepoRoot() string {
	cwd, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if exists(filepath.Join(cwd, "services")) && exists(filepath.Join(cwd, "apps")) {
			return cwd
		}
		parent := filepath.Dir(cwd)
		if parent == cwd {
			return ""
		}
		cwd = parent
	}
}

func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func normalizeLevel(level string) string {
	switch strings.ToLower(strings.TrimSpace(level)) {
	case "debug", "info", "warn", "error":
		return strings.ToLower(strings.TrimSpace(level))
	default:
		return "info"
	}
}

func safeValue(key string, value any, depth int) any {
	if isSensitiveKey(key) {
		return "[redacted]"
	}
	if depth > 4 {
		return "[truncated]"
	}
	switch typed := value.(type) {
	case nil, bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		return typed
	case string:
		return safeString(typed)
	case error:
		return safeString(typed.Error())
	case []string:
		limit := min(len(typed), maxLogArrayItems)
		out := make([]any, 0, limit)
		for _, item := range typed[:limit] {
			out = append(out, safeString(item))
		}
		return out
	case []any:
		limit := min(len(typed), maxLogArrayItems)
		out := make([]any, 0, limit)
		for _, item := range typed[:limit] {
			out = append(out, safeValue("", item, depth+1))
		}
		return out
	case map[string]any:
		out := map[string]any{}
		count := 0
		for childKey, childValue := range typed {
			if count >= maxLogMapItems {
				out["_truncated"] = true
				break
			}
			out[childKey] = safeValue(childKey, childValue, depth+1)
			count++
		}
		return out
	default:
		encoded, err := json.Marshal(typed)
		if err != nil {
			return safeString("")
		}
		return safeString(string(encoded))
	}
}

func safeString(value string) string {
	text := strings.TrimSpace(value)
	if text == "" {
		return ""
	}
	text = unsafeTextRE.ReplaceAllString(text, "[redacted]")
	if len(text) > maxLogStringLength {
		text = strings.TrimSpace(text[:maxLogStringLength-1]) + "..."
	}
	return text
}

func isSensitiveKey(key string) bool {
	normalized := strings.ToLower(strings.TrimSpace(key))
	for _, fragment := range []string{
		"api_key",
		"authorization",
		"base64",
		"credential",
		"image",
		"password",
		"prompt",
		"raw_output",
		"secret",
		"storage_uri",
		"token",
		"uri",
		"url",
	} {
		if strings.Contains(normalized, fragment) {
			return true
		}
	}
	return false
}
