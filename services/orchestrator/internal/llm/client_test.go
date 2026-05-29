package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

func TestGenerateJSONChatPathUnchanged(t *testing.T) {
	var seenPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenPath = r.URL.Path
		if r.Method != http.MethodPost {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected chat completions path, got %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("expected bearer auth header, got %q", got)
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		if body["model"] != "test-model" {
			t.Errorf("expected model test-model, got %#v", body["model"])
		}
		if body["temperature"] != 0.2 {
			t.Errorf("expected temperature 0.2, got %#v", body["temperature"])
		}
		if _, ok := body["messages"]; !ok {
			t.Errorf("expected chat request messages")
		}
		if _, ok := body["input"]; ok {
			t.Errorf("chat request should not include responses input")
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"{\"ok\":true}"}}]}`)
	}))
	defer server.Close()

	client := NewClient(Config{
		Enabled:  true,
		Provider: ProviderOpenAI,
		BaseURL:  server.URL,
		APIKey:   "test-key",
		Model:    "test-model",
	})

	raw, err := client.GenerateJSON(context.Background(), JSONRequest{
		Temperature: 0.2,
		Messages: []Message{
			{Role: "system", Content: "Return JSON."},
			{Role: "user", Content: "Go."},
		},
	})
	if err != nil {
		t.Fatalf("GenerateJSON returned error: %v", err)
	}
	if string(raw) != `{"ok":true}` {
		t.Fatalf("expected chat content unchanged, got %s", raw)
	}
	if seenPath != "/chat/completions" {
		t.Fatalf("expected chat path, got %q", seenPath)
	}
}

func TestGenerateJSONResponsesPathAndFinalJSONExtraction(t *testing.T) {
	var seenPath string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenPath = r.URL.Path
		if r.URL.Path != "/responses" {
			t.Errorf("expected responses path, got %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		if _, ok := body["input"]; !ok {
			t.Errorf("expected responses input")
		}
		if _, ok := body["messages"]; ok {
			t.Errorf("responses request should not include chat messages")
		}
		if _, ok := body["temperature"]; ok {
			t.Errorf("responses request should omit temperature for reasoning-model compatibility")
		}
		reasoning, ok := body["reasoning"].(map[string]any)
		if !ok || reasoning["effort"] != ReasoningEffortLow {
			t.Errorf("expected low reasoning effort, got %#v", body["reasoning"])
		}
		if body["store"] != true {
			t.Errorf("expected store=true, got %#v", body["store"])
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"id":"resp_final",
			"output":[
				{"type":"message","content":[{"type":"output_text","text":"Here is the result: {\"ok\":true,\"nested\":{\"a\":1}}"}]}
			]
		}`)
	}))
	defer server.Close()

	client := NewClient(Config{
		Enabled:         true,
		Provider:        ProviderOpenAI,
		BaseURL:         server.URL,
		APIKey:          "test-key",
		Model:           "test-model",
		APIStyle:        APIStyleResponses,
		ReasoningEffort: ReasoningEffortLow,
		StoredResponses: true,
	})

	raw, err := client.GenerateJSON(context.Background(), JSONRequest{
		Temperature: 0.2,
		Messages:    []Message{{Role: "user", Content: "Return JSON."}},
	})
	if err != nil {
		t.Fatalf("GenerateJSON returned error: %v", err)
	}
	if string(raw) != `{"ok":true,"nested":{"a":1}}` {
		t.Fatalf("expected extracted JSON, got %s", raw)
	}
	if seenPath != "/responses" {
		t.Fatalf("expected responses path, got %q", seenPath)
	}
}

func TestGenerateJSONWithToolsParsesFunctionCallsAndSendsPreviousResponseID(t *testing.T) {
	var mu sync.Mutex
	requestCount := 0
	previousResponseIDSent := false
	toolOutputSent := false
	toolDeclared := false

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		requestCount++

		if r.URL.Path != "/responses" {
			t.Errorf("expected responses path, got %s", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request body: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		switch requestCount {
		case 1:
			if _, ok := body["previous_response_id"]; ok {
				t.Errorf("first response request should not include previous_response_id")
			}
			tools, _ := body["tools"].([]any)
			if len(tools) == 1 {
				if tool, ok := tools[0].(map[string]any); ok && tool["name"] == "dataset_profile" && tool["type"] == "function" {
					toolDeclared = true
				}
			}
			fmt.Fprint(w, `{
				"id":"resp_1",
				"output":[
					{"type":"function_call","id":"fc_1","call_id":"call_1","name":"dataset_profile","arguments":"{\"dataset_id\":\"ds1\"}"}
				]
			}`)
		case 2:
			if body["previous_response_id"] == "resp_1" {
				previousResponseIDSent = true
			}
			input, _ := body["input"].([]any)
			for _, item := range input {
				mapped, ok := item.(map[string]any)
				if !ok {
					continue
				}
				output, _ := mapped["output"].(string)
				if mapped["type"] == "function_call_output" && mapped["call_id"] == "call_1" && strings.Contains(output, `"rows":12`) {
					toolOutputSent = true
				}
			}
			fmt.Fprint(w, `{
				"id":"resp_2",
				"previous_response_id":"resp_1",
				"output":[
					{"type":"message","content":[{"type":"output_text","text":"{\"answer\":true}"}]}
				]
			}`)
		default:
			t.Errorf("unexpected extra response request %d", requestCount)
			http.Error(w, "unexpected request", http.StatusTooManyRequests)
		}
	}))
	defer server.Close()

	client := NewClient(Config{
		Enabled:         true,
		Provider:        ProviderOpenAI,
		BaseURL:         server.URL,
		APIKey:          "test-key",
		Model:           "test-model",
		APIStyle:        APIStyleResponses,
		StoredResponses: true,
		MaxToolRounds:   2,
	})

	result, err := client.GenerateJSONWithTools(context.Background(), ToolLoopRequest{
		JSONRequest: JSONRequest{
			Temperature: 0.2,
			Messages:    []Message{{Role: "user", Content: "Need a dataset profile."}},
		},
		Tools: []ToolDefinition{
			{
				Name:        "dataset_profile",
				Description: "Return a bounded dataset profile.",
				Parameters: map[string]any{
					"type":       "object",
					"properties": map[string]any{"dataset_id": map[string]any{"type": "string"}},
					"required":   []any{"dataset_id"},
				},
			},
		},
		ToolAnswerer: InformationToolFunc(func(_ context.Context, call ToolCall) (ToolResult, error) {
			if call.CallID != "call_1" {
				t.Errorf("expected call_1, got %q", call.CallID)
			}
			if call.Name != "dataset_profile" {
				t.Errorf("expected dataset_profile, got %q", call.Name)
			}
			if string(call.Arguments) != `{"dataset_id":"ds1"}` {
				t.Errorf("expected parsed arguments, got %s", call.Arguments)
			}
			return ToolResult{
				CallID: call.CallID,
				Name:   call.Name,
				Output: json.RawMessage(`{"rows":12}`),
			}, nil
		}),
	})
	if err != nil {
		t.Fatalf("GenerateJSONWithTools returned error: %v", err)
	}
	if string(result.FinalJSON) != `{"answer":true}` {
		t.Fatalf("expected final JSON, got %s", result.FinalJSON)
	}
	if result.ToolRounds != 1 {
		t.Fatalf("expected one tool round, got %d", result.ToolRounds)
	}
	if len(result.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %d", len(result.ToolCalls))
	}

	mu.Lock()
	defer mu.Unlock()
	if requestCount != 2 {
		t.Fatalf("expected two responses requests, got %d", requestCount)
	}
	if !toolDeclared {
		t.Fatalf("expected function tool definition in first request")
	}
	if !previousResponseIDSent {
		t.Fatalf("expected previous_response_id on second request")
	}
	if !toolOutputSent {
		t.Fatalf("expected function_call_output input on second request")
	}
}

func TestUnsupportedProvidersIgnoreResponsesStyleAndUseChat(t *testing.T) {
	for _, provider := range []string{ProviderLocal, ProviderOpenAICompatible} {
		t.Run(provider, func(t *testing.T) {
			var seenPath string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				seenPath = r.URL.Path
				if r.URL.Path != "/chat/completions" {
					t.Errorf("expected chat path, got %s", r.URL.Path)
				}
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"{\"ok\":true}"}}]}`)
			}))
			defer server.Close()

			apiKey := ""
			if provider != ProviderLocal {
				apiKey = "test-key"
			}
			client := NewClient(Config{
				Enabled:  true,
				Provider: provider,
				BaseURL:  server.URL,
				APIKey:   apiKey,
				Model:    "test-model",
				APIStyle: APIStyleResponses,
			})
			raw, err := client.GenerateJSON(context.Background(), JSONRequest{
				Messages: []Message{{Role: "user", Content: "Return JSON."}},
			})
			if err != nil {
				t.Fatalf("GenerateJSON returned error: %v", err)
			}
			if string(raw) != `{"ok":true}` {
				t.Fatalf("expected chat JSON, got %s", raw)
			}
			if seenPath != "/chat/completions" {
				t.Fatalf("expected chat path, got %q", seenPath)
			}
		})
	}
}
