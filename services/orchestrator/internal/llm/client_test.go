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

func TestGenerateJSONWithUsageCapturesChatUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("expected chat completions path, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"choices":[{"message":{"role":"assistant","content":"{\"ok\":true}"}}],
			"usage":{
				"prompt_tokens":11,
				"prompt_tokens_details":{"cached_tokens":3},
				"completion_tokens":7,
				"completion_tokens_details":{"reasoning_tokens":2},
				"total_tokens":18
			}
		}`)
	}))
	defer server.Close()

	client := NewClient(Config{
		Enabled:  true,
		Provider: ProviderOpenAI,
		BaseURL:  server.URL,
		APIKey:   "test-key",
		Model:    "test-model",
	})

	result, err := client.GenerateJSONWithUsage(context.Background(), JSONRequest{
		Messages: []Message{{Role: "user", Content: "Return JSON."}},
	})
	if err != nil {
		t.Fatalf("GenerateJSONWithUsage returned error: %v", err)
	}
	if string(result.RawJSON) != `{"ok":true}` {
		t.Fatalf("expected chat content unchanged, got %s", result.RawJSON)
	}
	if result.Usage == nil {
		t.Fatal("expected usage to be captured")
	}
	if result.Usage.InputTokens != 11 || result.Usage.OutputTokens != 7 || result.Usage.TotalTokens != 18 {
		t.Fatalf("unexpected chat usage: %#v", result.Usage)
	}
	if result.Usage.CachedInputTokens != 3 || result.Usage.ReasoningTokens != 2 {
		t.Fatalf("unexpected chat token breakdown: %#v", result.Usage)
	}
	if result.Usage.RequestModel != "test-model" || result.Usage.APIStyle != APIStyleChatCompletions {
		t.Fatalf("unexpected chat usage metadata: %#v", result.Usage)
	}
}

func TestGenerateJSONChatRetriesTransientHTTPFailure(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			http.Error(w, "temporarily overloaded", http.StatusTooManyRequests)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"{\"ok\":true}"}}]}`)
	}))
	defer server.Close()

	client := NewClient(Config{
		Enabled:    true,
		Provider:   ProviderOpenAI,
		BaseURL:    server.URL,
		APIKey:     "test-key",
		Model:      "test-model",
		MaxRetries: 1,
	})

	raw, err := client.GenerateJSON(context.Background(), JSONRequest{
		Messages: []Message{{Role: "user", Content: "Return JSON."}},
	})
	if err != nil {
		t.Fatalf("GenerateJSON returned error: %v", err)
	}
	if attempts != 2 {
		t.Fatalf("expected retry after transient failure, got %d attempts", attempts)
	}
	if string(raw) != `{"ok":true}` {
		t.Fatalf("expected retried content, got %s", raw)
	}
}

func TestGenerateJSONWithUsageUsesRequestModelOverride(t *testing.T) {
	testCases := []struct {
		name          string
		apiStyle      string
		expectedPath  string
		expectedStyle string
	}{
		{
			name:          "chat",
			apiStyle:      APIStyleChatCompletions,
			expectedPath:  "/chat/completions",
			expectedStyle: APIStyleChatCompletions,
		},
		{
			name:          "responses",
			apiStyle:      APIStyleResponses,
			expectedPath:  "/responses",
			expectedStyle: APIStyleResponses,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != tc.expectedPath {
					t.Errorf("expected %s path, got %s", tc.expectedPath, r.URL.Path)
				}

				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Errorf("decode request body: %v", err)
					return
				}
				if body["model"] != "request-model" {
					t.Errorf("expected request model override, got %#v", body["model"])
					return
				}

				w.Header().Set("Content-Type", "application/json")
				if tc.apiStyle == APIStyleResponses {
					fmt.Fprint(w, `{
						"id":"resp_override",
						"output":[
							{"type":"message","content":[{"type":"output_text","text":"{\"ok\":true}"}]}
						],
						"usage":{
							"input_tokens":5,
							"input_tokens_details":{"cached_tokens":1},
							"output_tokens":4,
							"output_tokens_details":{"reasoning_tokens":2},
							"total_tokens":9
						}
					}`)
					return
				}

				fmt.Fprint(w, `{
					"choices":[{"message":{"role":"assistant","content":"{\"ok\":true}"}}],
					"usage":{
						"prompt_tokens":5,
						"prompt_tokens_details":{"cached_tokens":1},
						"completion_tokens":4,
						"completion_tokens_details":{"reasoning_tokens":2},
						"total_tokens":9
					}
				}`)
			}))
			defer server.Close()

			client := NewClient(Config{
				Enabled:  true,
				Provider: ProviderOpenAI,
				BaseURL:  server.URL,
				APIKey:   "test-key",
				Model:    "config-model",
				APIStyle: tc.apiStyle,
			})

			result, err := client.GenerateJSONWithUsage(context.Background(), JSONRequest{
				Model:       "request-model",
				Messages:    []Message{{Role: "user", Content: "Return JSON."}},
				Temperature: 0.2,
			})
			if err != nil {
				t.Fatalf("GenerateJSONWithUsage returned error: %v", err)
			}
			if string(result.RawJSON) != `{"ok":true}` {
				t.Fatalf("expected JSON payload, got %s", result.RawJSON)
			}
			if result.Usage == nil {
				t.Fatal("expected usage to be captured")
			}
			if result.Usage.RequestModel != "request-model" {
				t.Fatalf("expected request-model usage metadata, got %#v", result.Usage)
			}
			if result.Usage.APIStyle != tc.expectedStyle {
				t.Fatalf("expected %s usage metadata, got %#v", tc.expectedStyle, result.Usage)
			}
		})
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
			],
			"usage":{
				"input_tokens":21,
				"input_tokens_details":{"cached_tokens":4},
				"output_tokens":9,
				"output_tokens_details":{"reasoning_tokens":5},
				"total_tokens":30
			}
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

	result, err := client.GenerateJSONWithUsage(context.Background(), JSONRequest{
		Temperature: 0.2,
		Messages:    []Message{{Role: "user", Content: "Return JSON."}},
	})
	if err != nil {
		t.Fatalf("GenerateJSONWithUsage returned error: %v", err)
	}
	if string(result.RawJSON) != `{"ok":true,"nested":{"a":1}}` {
		t.Fatalf("expected extracted JSON, got %s", result.RawJSON)
	}
	if result.Usage == nil {
		t.Fatal("expected usage to be captured")
	}
	if result.Usage.InputTokens != 21 || result.Usage.OutputTokens != 9 || result.Usage.TotalTokens != 30 {
		t.Fatalf("unexpected responses usage: %#v", result.Usage)
	}
	if result.Usage.CachedInputTokens != 4 || result.Usage.ReasoningTokens != 5 {
		t.Fatalf("unexpected responses token breakdown: %#v", result.Usage)
	}
	if result.Usage.RequestModel != "test-model" || result.Usage.APIStyle != APIStyleResponses {
		t.Fatalf("unexpected responses usage metadata: %#v", result.Usage)
	}
	if seenPath != "/responses" {
		t.Fatalf("expected responses path, got %q", seenPath)
	}
}

func TestGenerateJSONResponsesRetriesTransientHTTPFailure(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts == 1 {
			http.Error(w, "temporary upstream failure", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"id":"resp_final",
			"output":[{"type":"message","content":[{"type":"output_text","text":"{\"ok\":true}"}]}]
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
		StoredResponses: true,
		MaxRetries:      1,
	})

	result, err := client.GenerateJSONWithUsage(context.Background(), JSONRequest{
		Messages: []Message{{Role: "user", Content: "Return JSON."}},
	})
	if err != nil {
		t.Fatalf("GenerateJSONWithUsage returned error: %v", err)
	}
	if attempts != 2 {
		t.Fatalf("expected retry after transient failure, got %d attempts", attempts)
	}
	if string(result.RawJSON) != `{"ok":true}` {
		t.Fatalf("expected retried content, got %s", result.RawJSON)
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
				],
				"usage":{
					"input_tokens":80,
					"input_tokens_details":{"cached_tokens":12},
					"output_tokens":30,
					"output_tokens_details":{"reasoning_tokens":7},
					"total_tokens":110
				}
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
				],
				"usage":{
					"input_tokens":120,
					"input_tokens_details":{"cached_tokens":8},
					"output_tokens":40,
					"output_tokens_details":{"reasoning_tokens":3},
					"total_tokens":160
				}
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
	if result.Usage == nil {
		t.Fatal("expected aggregate usage to be captured")
	}
	if result.Usage.InputTokens != 200 || result.Usage.OutputTokens != 70 || result.Usage.TotalTokens != 270 {
		t.Fatalf("unexpected aggregate usage: %#v", result.Usage)
	}
	if result.Usage.CachedInputTokens != 20 || result.Usage.ReasoningTokens != 10 {
		t.Fatalf("unexpected aggregate token breakdown: %#v", result.Usage)
	}
	if result.Usage.RequestModel != "test-model" || result.Usage.APIStyle != APIStyleResponses || result.Usage.ToolRounds != 1 {
		t.Fatalf("unexpected aggregate usage metadata: %#v", result.Usage)
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
