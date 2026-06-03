package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"model-express/services/orchestrator/internal/diagnostics"
)

type OpenAICompatibleClient struct {
	config     Config
	httpClient *http.Client
}

func NewClient(config Config) OpenAICompatibleClient {
	return newOpenAICompatibleClient(config)
}

func NewOpenAICompatibleClient(config Config) OpenAICompatibleClient {
	return newOpenAICompatibleClient(config)
}

func newOpenAICompatibleClient(config Config) OpenAICompatibleClient {
	timeout := config.Timeout
	if timeout <= 0 {
		timeout = ConfigFromEnv(config.Enabled, config.Provider, config.Model).Timeout
	}
	return OpenAICompatibleClient{
		config: config,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

func (c OpenAICompatibleClient) GenerateJSON(ctx context.Context, req JSONRequest) ([]byte, error) {
	result, err := c.GenerateJSONWithUsage(ctx, req)
	if err != nil {
		return nil, err
	}
	return result.RawJSON, nil
}

func (c OpenAICompatibleClient) GenerateJSONWithUsage(ctx context.Context, req JSONRequest) (JSONResult, error) {
	if c.useResponses() {
		result, err := c.CreateResponse(ctx, ResponseRequest{
			Model:              req.Model,
			Messages:           req.Messages,
			Temperature:        req.Temperature,
			ReasoningEffort:    req.ReasoningEffort,
			PreviousResponseID: req.PreviousResponseID,
		})
		if err != nil {
			return JSONResult{}, err
		}
		if len(result.FunctionCalls) > 0 {
			return JSONResult{}, fmt.Errorf("llm response requested %d information tool calls but no answerer was provided", len(result.FunctionCalls))
		}
		if len(result.FinalJSON) == 0 {
			return JSONResult{}, fmt.Errorf("llm response content was empty")
		}
		return JSONResult{RawJSON: result.FinalJSON, Usage: result.Usage}, nil
	}
	raw, usage, err := c.generateChatJSONWithUsage(ctx, req)
	if err != nil {
		return JSONResult{}, err
	}
	return JSONResult{RawJSON: raw, Usage: usage}, nil
}

func (c OpenAICompatibleClient) GenerateJSONWithTools(ctx context.Context, req ToolLoopRequest) (ToolLoopResult, error) {
	if !c.useResponses() {
		result, err := c.GenerateJSONWithUsage(ctx, req.JSONRequest)
		return ToolLoopResult{FinalJSON: result.RawJSON, Usage: result.Usage}, err
	}

	model, err := c.modelForRequest(req.JSONRequest)
	if err != nil {
		return ToolLoopResult{}, err
	}
	maxToolRounds := req.MaxToolRounds
	if maxToolRounds <= 0 {
		maxToolRounds = c.config.MaxToolRounds
	}
	if maxToolRounds <= 0 {
		maxToolRounds = DefaultMaxToolRounds
	}

	input := responseInputFromMessages(req.Messages)
	previousResponseID := req.PreviousResponseID
	result := ToolLoopResult{}

	for {
		response, err := c.postResponse(ctx, responsesCall{
			model:              model,
			input:              input,
			temperature:        req.Temperature,
			reasoningEffort:    firstNonEmpty(req.ReasoningEffort, c.config.ReasoningEffort),
			previousResponseID: previousResponseID,
			tools:              req.Tools,
			store:              c.config.StoredResponses,
		})
		if err != nil {
			return result, err
		}

		result.ResponseID = response.ID
		result.PreviousResponseID = response.PreviousResponseID
		result.Usage = mergeUsage(result.Usage, response.Usage)
		if len(response.FunctionCalls) == 0 {
			if len(response.FinalJSON) == 0 {
				return result, fmt.Errorf("llm response content was empty")
			}
			result.FinalJSON = response.FinalJSON
			if result.Usage != nil {
				result.Usage.ToolRounds = result.ToolRounds
			}
			return result, nil
		}
		if req.ToolAnswerer == nil {
			return result, fmt.Errorf("llm response requested %d information tool calls but no answerer was provided", len(response.FunctionCalls))
		}
		if result.ToolRounds >= maxToolRounds {
			return result, fmt.Errorf("llm exceeded max tool rounds (%d)", maxToolRounds)
		}

		result.ToolCalls = append(result.ToolCalls, response.FunctionCalls...)
		toolResults := make([]ToolResult, 0, len(response.FunctionCalls))
		for _, call := range response.FunctionCalls {
			toolResult, err := req.ToolAnswerer.AnswerInformationToolCall(ctx, call)
			if err != nil {
				toolResult = ToolResult{
					CallID: call.CallID,
					Name:   call.Name,
					Error:  err.Error(),
				}
			}
			if toolResult.CallID == "" {
				toolResult.CallID = call.CallID
			}
			if toolResult.Name == "" {
				toolResult.Name = call.Name
			}
			toolResults = append(toolResults, toolResult)
		}
		result.ToolResults = append(result.ToolResults, toolResults...)
		result.ToolRounds++

		toolOutputItems := responseInputFromToolResults(toolResults)
		if c.config.StoredResponses && response.ID != "" {
			previousResponseID = response.ID
			input = toolOutputItems
			continue
		}

		input = append(input, response.outputItemsAsInput()...)
		input = append(input, toolOutputItems...)
	}
}

func (c OpenAICompatibleClient) CreateResponse(ctx context.Context, req ResponseRequest) (ResponseResult, error) {
	if !c.useResponses() {
		raw, usage, err := c.generateChatJSONWithUsage(ctx, JSONRequest{
			Model:       req.Model,
			Messages:    req.Messages,
			Temperature: req.Temperature,
		})
		return ResponseResult{FinalJSON: raw, Text: string(raw), Usage: usage}, err
	}

	model, err := c.modelForRequest(JSONRequest{Model: req.Model})
	if err != nil {
		return ResponseResult{}, err
	}
	input := responseInputFromMessages(req.Messages)
	input = append(input, responseInputFromToolResults(req.ToolOutputs)...)
	store := c.config.StoredResponses
	if req.Store != nil {
		store = *req.Store
	}
	return c.postResponse(ctx, responsesCall{
		model:              model,
		input:              input,
		temperature:        req.Temperature,
		reasoningEffort:    firstNonEmpty(req.ReasoningEffort, c.config.ReasoningEffort),
		previousResponseID: req.PreviousResponseID,
		tools:              req.Tools,
		store:              store,
	})
}

func (c OpenAICompatibleClient) generateChatJSON(ctx context.Context, req JSONRequest) ([]byte, error) {
	raw, _, err := c.generateChatJSONWithUsage(ctx, req)
	return raw, err
}

func (c OpenAICompatibleClient) generateChatJSONWithUsage(ctx context.Context, req JSONRequest) ([]byte, *Usage, error) {
	if !c.config.Enabled {
		return nil, nil, fmt.Errorf("llm is disabled")
	}
	if c.config.BaseURL == "" {
		return nil, nil, fmt.Errorf("MODEL_EXPRESS_LLM_BASE_URL is required for provider %s", c.config.Provider)
	}
	if c.config.Model == "" && req.Model == "" {
		return nil, nil, fmt.Errorf("MODEL_EXPRESS_LLM_MODEL is required")
	}
	if c.config.APIKey == "" && c.config.Provider != ProviderLocal {
		return nil, nil, fmt.Errorf("MODEL_EXPRESS_LLM_API_KEY is required for provider %s", c.config.Provider)
	}

	model := req.Model
	if model == "" {
		model = c.config.Model
	}

	body := chatCompletionRequest{
		Model:       model,
		Messages:    req.Messages,
		Temperature: req.Temperature,
	}

	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, nil, fmt.Errorf("marshal llm request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.config.BaseURL+"/chat/completions", bytes.NewReader(encoded))
	if err != nil {
		return nil, nil, fmt.Errorf("build llm request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.config.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, nil, fmt.Errorf("call llm: %w", err)
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("read llm response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, nil, fmt.Errorf("llm request failed: %s: %s", resp.Status, string(responseBody))
	}

	var decoded chatCompletionResponse
	if err := json.Unmarshal(responseBody, &decoded); err != nil {
		return nil, nil, fmt.Errorf("decode llm response: %w", err)
	}
	if len(decoded.Choices) == 0 {
		return nil, nil, fmt.Errorf("llm response had no choices")
	}

	content := strings.TrimSpace(decoded.Choices[0].Message.Content)
	if content == "" {
		return nil, nil, fmt.Errorf("llm response content was empty")
	}

	return []byte(content), chatUsageFromChatCompletion(decoded, body.Model, APIStyleChatCompletions), nil
}

func (c OpenAICompatibleClient) postResponse(ctx context.Context, call responsesCall) (ResponseResult, error) {
	if !c.config.Enabled {
		return ResponseResult{}, fmt.Errorf("llm is disabled")
	}
	if c.config.BaseURL == "" {
		return ResponseResult{}, fmt.Errorf("MODEL_EXPRESS_LLM_BASE_URL is required for provider %s", c.config.Provider)
	}
	if call.model == "" {
		return ResponseResult{}, fmt.Errorf("MODEL_EXPRESS_LLM_MODEL is required")
	}
	if c.config.APIKey == "" && c.config.Provider != ProviderLocal {
		return ResponseResult{}, fmt.Errorf("MODEL_EXPRESS_LLM_API_KEY is required for provider %s", c.config.Provider)
	}

	body := responsesRequest{
		Model:              call.model,
		Input:              call.input,
		PreviousResponseID: call.previousResponseID,
		Store:              &call.store,
		Tools:              responseToolDefinitions(call.tools),
	}
	if call.reasoningEffort != "" {
		body.Reasoning = &responsesReasoning{Effort: call.reasoningEffort}
	}

	encoded, err := json.Marshal(body)
	if err != nil {
		return ResponseResult{}, fmt.Errorf("marshal llm responses request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.config.BaseURL+"/responses", bytes.NewReader(encoded))
	if err != nil {
		return ResponseResult{}, fmt.Errorf("build llm responses request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.config.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return ResponseResult{}, fmt.Errorf("call llm responses: %w", err)
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return ResponseResult{}, fmt.Errorf("read llm responses response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		diagnostics.Event("error", "llm_responses_http_error", map[string]any{
			"provider":    c.config.Provider,
			"model":       call.model,
			"api_style":   APIStyleResponses,
			"status":      resp.Status,
			"status_code": resp.StatusCode,
			"body":        string(responseBody),
			"tool_count":  len(call.tools),
		})
		return ResponseResult{}, fmt.Errorf("llm responses request failed: %s: %s", resp.Status, string(responseBody))
	}

	var decoded responsesResponse
	if err := json.Unmarshal(responseBody, &decoded); err != nil {
		return ResponseResult{}, fmt.Errorf("decode llm responses response: %w", err)
	}
	result := decoded.result()
	if result.Usage != nil {
		result.Usage.RequestModel = call.model
		result.Usage.APIStyle = APIStyleResponses
	}
	if len(result.FinalJSON) == 0 && len(result.FunctionCalls) == 0 {
		return ResponseResult{}, fmt.Errorf("llm response content was empty")
	}
	return result, nil
}

func (c OpenAICompatibleClient) modelForRequest(req JSONRequest) (string, error) {
	if !c.config.Enabled {
		return "", fmt.Errorf("llm is disabled")
	}
	if c.config.BaseURL == "" {
		return "", fmt.Errorf("MODEL_EXPRESS_LLM_BASE_URL is required for provider %s", c.config.Provider)
	}
	if c.config.APIKey == "" && c.config.Provider != ProviderLocal {
		return "", fmt.Errorf("MODEL_EXPRESS_LLM_API_KEY is required for provider %s", c.config.Provider)
	}
	model := req.Model
	if model == "" {
		model = c.config.Model
	}
	if model == "" {
		return "", fmt.Errorf("MODEL_EXPRESS_LLM_MODEL is required")
	}
	return model, nil
}

func (c OpenAICompatibleClient) useResponses() bool {
	return strings.ToLower(strings.TrimSpace(c.config.Provider)) == ProviderOpenAI &&
		NormalizeAPIStyle(c.config.APIStyle) == APIStyleResponses
}

type chatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
	Usage *chatCompletionUsage `json:"usage"`
}

type chatCompletionUsage struct {
	PromptTokens            int                           `json:"prompt_tokens"`
	PromptTokensDetails     chatCompletionTokenBreak      `json:"prompt_tokens_details"`
	CompletionTokens        int                           `json:"completion_tokens"`
	CompletionTokensDetails chatCompletionCompletionBreak `json:"completion_tokens_details"`
	TotalTokens             int                           `json:"total_tokens"`
}

type chatCompletionTokenBreak struct {
	CachedTokens int `json:"cached_tokens"`
}

type chatCompletionCompletionBreak struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type ResponseRequest struct {
	Model              string
	Messages           []Message
	Temperature        float64
	ReasoningEffort    string
	PreviousResponseID string
	Tools              []ToolDefinition
	ToolOutputs        []ToolResult
	Store              *bool
}

type ResponseResult struct {
	ID                 string
	PreviousResponseID string
	Text               string
	FinalJSON          []byte
	FunctionCalls      []ToolCall
	Usage              *Usage

	outputItems []json.RawMessage
}

func (r ResponseResult) outputItemsAsInput() []any {
	out := make([]any, 0, len(r.outputItems))
	for _, item := range r.outputItems {
		out = append(out, item)
	}
	return out
}

type ToolLoopRequest struct {
	JSONRequest
	Tools         []ToolDefinition
	ToolAnswerer  InformationToolAnswerer
	MaxToolRounds int
}

type ToolLoopResult struct {
	FinalJSON          []byte
	ResponseID         string
	PreviousResponseID string
	ToolRounds         int
	ToolCalls          []ToolCall
	ToolResults        []ToolResult
	Usage              *Usage
}

type InformationToolAnswerer interface {
	AnswerInformationToolCall(ctx context.Context, call ToolCall) (ToolResult, error)
}

type InformationToolFunc func(ctx context.Context, call ToolCall) (ToolResult, error)

func (f InformationToolFunc) AnswerInformationToolCall(ctx context.Context, call ToolCall) (ToolResult, error) {
	return f(ctx, call)
}

type ToolDefinition struct {
	Type        string         `json:"type,omitempty"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      bool           `json:"strict,omitempty"`
}

type ToolCall struct {
	ID           string          `json:"id,omitempty"`
	CallID       string          `json:"call_id"`
	Name         string          `json:"name"`
	Arguments    json.RawMessage `json:"arguments,omitempty"`
	RawArguments string          `json:"raw_arguments,omitempty"`
}

type ToolResult struct {
	CallID string          `json:"call_id"`
	Name   string          `json:"name,omitempty"`
	Output json.RawMessage `json:"output,omitempty"`
	Error  string          `json:"error,omitempty"`
}

type responsesCall struct {
	model              string
	input              []any
	temperature        float64
	reasoningEffort    string
	previousResponseID string
	tools              []ToolDefinition
	store              bool
}

type responsesRequest struct {
	Model              string                    `json:"model"`
	Input              []any                     `json:"input"`
	PreviousResponseID string                    `json:"previous_response_id,omitempty"`
	Store              *bool                     `json:"store,omitempty"`
	Reasoning          *responsesReasoning       `json:"reasoning,omitempty"`
	Tools              []responsesToolDefinition `json:"tools,omitempty"`
}

type responsesReasoning struct {
	Effort string `json:"effort"`
}

type responsesMessageInput struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type responsesFunctionCallOutput struct {
	Type   string `json:"type"`
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

type responsesToolDefinition struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      bool           `json:"strict,omitempty"`
}

type responsesResponse struct {
	ID                 string                `json:"id"`
	PreviousResponseID string                `json:"previous_response_id"`
	OutputText         string                `json:"output_text"`
	Output             []responsesOutputItem `json:"output"`
	Usage              *responsesUsage       `json:"usage"`
}

type responsesUsage struct {
	InputTokens         int                  `json:"input_tokens"`
	InputTokensDetails  responsesUsageInput  `json:"input_tokens_details"`
	OutputTokens        int                  `json:"output_tokens"`
	OutputTokensDetails responsesUsageOutput `json:"output_tokens_details"`
	TotalTokens         int                  `json:"total_tokens"`
}

type responsesUsageInput struct {
	CachedTokens int `json:"cached_tokens"`
}

type responsesUsageOutput struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type responsesOutputItem struct {
	Type      string                   `json:"type"`
	ID        string                   `json:"id"`
	CallID    string                   `json:"call_id"`
	Name      string                   `json:"name"`
	Arguments string                   `json:"arguments"`
	Text      string                   `json:"text"`
	Content   []responsesOutputContent `json:"content"`
	Raw       json.RawMessage          `json:"-"`
}

type responsesOutputContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func (i *responsesOutputItem) UnmarshalJSON(data []byte) error {
	type alias responsesOutputItem
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*i = responsesOutputItem(decoded)
	i.Raw = append(i.Raw[:0], data...)
	return nil
}

func (r responsesResponse) result() ResponseResult {
	result := ResponseResult{
		ID:                 r.ID,
		PreviousResponseID: r.PreviousResponseID,
		outputItems:        make([]json.RawMessage, 0, len(r.Output)),
		Usage:              responsesUsageToUsage(r.Usage),
	}

	var textParts []string
	if strings.TrimSpace(r.OutputText) != "" {
		textParts = append(textParts, r.OutputText)
	}
	for _, item := range r.Output {
		if len(item.Raw) > 0 {
			result.outputItems = append(result.outputItems, append(json.RawMessage(nil), item.Raw...))
		}
		switch item.Type {
		case "function_call":
			result.FunctionCalls = append(result.FunctionCalls, ToolCall{
				ID:           item.ID,
				CallID:       item.CallID,
				Name:         item.Name,
				Arguments:    parseToolArguments(item.Arguments),
				RawArguments: item.Arguments,
			})
		case "message":
			for _, content := range item.Content {
				if content.Type == "output_text" || content.Type == "text" || content.Type == "" {
					if strings.TrimSpace(content.Text) != "" {
						textParts = append(textParts, content.Text)
					}
				}
			}
		case "output_text":
			if strings.TrimSpace(item.Text) != "" {
				textParts = append(textParts, item.Text)
			}
		}
	}

	result.Text = strings.TrimSpace(strings.Join(textParts, "\n"))
	if result.Text != "" {
		result.FinalJSON = extractFinalJSON(result.Text)
	}
	return result
}

func chatUsageFromChatCompletion(response chatCompletionResponse, requestModel string, apiStyle string) *Usage {
	if response.Usage == nil {
		return nil
	}
	return &Usage{
		InputTokens:       response.Usage.PromptTokens,
		OutputTokens:      response.Usage.CompletionTokens,
		TotalTokens:       response.Usage.TotalTokens,
		CachedInputTokens: response.Usage.PromptTokensDetails.CachedTokens,
		ReasoningTokens:   response.Usage.CompletionTokensDetails.ReasoningTokens,
		RequestModel:      requestModel,
		APIStyle:          apiStyle,
	}
}

func responsesUsageToUsage(response *responsesUsage) *Usage {
	if response == nil {
		return nil
	}
	return &Usage{
		InputTokens:       response.InputTokens,
		OutputTokens:      response.OutputTokens,
		TotalTokens:       response.TotalTokens,
		CachedInputTokens: response.InputTokensDetails.CachedTokens,
		ReasoningTokens:   response.OutputTokensDetails.ReasoningTokens,
	}
}

func mergeUsage(existing *Usage, next *Usage) *Usage {
	if next == nil {
		return existing
	}
	if existing == nil {
		clone := *next
		return &clone
	}
	existing.InputTokens += next.InputTokens
	existing.OutputTokens += next.OutputTokens
	existing.TotalTokens += next.TotalTokens
	existing.CachedInputTokens += next.CachedInputTokens
	existing.ReasoningTokens += next.ReasoningTokens
	if existing.RequestModel == "" {
		existing.RequestModel = next.RequestModel
	}
	if existing.APIStyle == "" {
		existing.APIStyle = next.APIStyle
	}
	if existing.ToolRounds == 0 {
		existing.ToolRounds = next.ToolRounds
	}
	return existing
}

func responseInputFromMessages(messages []Message) []any {
	input := make([]any, 0, len(messages))
	for _, message := range messages {
		input = append(input, responsesMessageInput{
			Role:    message.Role,
			Content: message.Content,
		})
	}
	return input
}

func responseInputFromToolResults(results []ToolResult) []any {
	input := make([]any, 0, len(results))
	for _, result := range results {
		input = append(input, responsesFunctionCallOutput{
			Type:   "function_call_output",
			CallID: result.CallID,
			Output: toolResultOutput(result),
		})
	}
	return input
}

func responseToolDefinitions(tools []ToolDefinition) []responsesToolDefinition {
	if len(tools) == 0 {
		return nil
	}
	out := make([]responsesToolDefinition, 0, len(tools))
	for _, tool := range tools {
		toolType := strings.TrimSpace(tool.Type)
		if toolType == "" {
			toolType = "function"
		}
		parameters := tool.Parameters
		if parameters == nil && toolType == "function" {
			parameters = map[string]any{
				"type":                 "object",
				"properties":           map[string]any{},
				"additionalProperties": false,
			}
		}
		out = append(out, responsesToolDefinition{
			Type:        toolType,
			Name:        tool.Name,
			Description: tool.Description,
			Parameters:  parameters,
			Strict:      tool.Strict,
		})
	}
	return out
}

func toolResultOutput(result ToolResult) string {
	if result.Error != "" {
		encoded, err := json.Marshal(map[string]string{"error": result.Error})
		if err == nil {
			return string(encoded)
		}
		return result.Error
	}
	if len(result.Output) == 0 {
		return "{}"
	}
	return string(result.Output)
}

func parseToolArguments(arguments string) json.RawMessage {
	trimmed := strings.TrimSpace(arguments)
	if trimmed == "" {
		return json.RawMessage(`{}`)
	}
	if json.Valid([]byte(trimmed)) {
		return json.RawMessage(trimmed)
	}
	encoded, err := json.Marshal(trimmed)
	if err != nil {
		return json.RawMessage(`""`)
	}
	return encoded
}

func extractFinalJSON(text string) []byte {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil
	}
	if json.Valid([]byte(trimmed)) {
		return []byte(trimmed)
	}
	if fenced := extractFencedJSON(trimmed); fenced != "" && json.Valid([]byte(fenced)) {
		return []byte(fenced)
	}
	if segment := extractBalancedJSON(trimmed); segment != "" && json.Valid([]byte(segment)) {
		return []byte(segment)
	}
	return []byte(trimmed)
}

func extractFencedJSON(text string) string {
	start := strings.Index(text, "```")
	if start < 0 {
		return ""
	}
	afterFence := text[start+3:]
	newline := strings.IndexByte(afterFence, '\n')
	if newline < 0 {
		return ""
	}
	body := afterFence[newline+1:]
	end := strings.Index(body, "```")
	if end < 0 {
		return ""
	}
	return strings.TrimSpace(body[:end])
}

func extractBalancedJSON(text string) string {
	for start := 0; start < len(text); start++ {
		if text[start] != '{' && text[start] != '[' {
			continue
		}
		if segment := balancedJSONFrom(text[start:]); segment != "" {
			return segment
		}
	}
	return ""
}

func balancedJSONFrom(text string) string {
	stack := make([]byte, 0, 8)
	inString := false
	escaped := false
	for i := 0; i < len(text); i++ {
		ch := text[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			switch ch {
			case '\\':
				escaped = true
			case '"':
				inString = false
			}
			continue
		}
		switch ch {
		case '"':
			inString = true
		case '{':
			stack = append(stack, '}')
		case '[':
			stack = append(stack, ']')
		case '}', ']':
			if len(stack) == 0 || stack[len(stack)-1] != ch {
				return ""
			}
			stack = stack[:len(stack)-1]
			if len(stack) == 0 {
				return strings.TrimSpace(text[:i+1])
			}
		}
	}
	return ""
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}
