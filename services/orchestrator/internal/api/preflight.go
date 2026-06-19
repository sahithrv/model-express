package api

import (
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/embeddings"
	"model-express/services/orchestrator/internal/llm"
)

type cloudPreflightRequest struct {
	Stage            string `json:"stage"`
	Live             bool   `json:"live"`
	AutomaticTunnels bool   `json:"automatic_tunnels"`
}

type cloudPreflightResponse struct {
	Status string                `json:"status"`
	Checks []cloudPreflightCheck `json:"checks"`
}

type cloudPreflightCheck struct {
	ID          string         `json:"id"`
	Status      string         `json:"status"`
	Message     string         `json:"message"`
	Remediation string         `json:"remediation,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

func (s *Server) preflightCloud(c *gin.Context) {
	req := cloudPreflightRequest{Stage: "manual"}
	if !bindOptionalJSON(c, &req) {
		return
	}
	response := s.runCloudPreflight(req)
	status := http.StatusOK
	if response.Status == "failed" {
		status = http.StatusBadRequest
	}
	c.JSON(status, response)
}

func (s *Server) runCloudPreflight(req cloudPreflightRequest) cloudPreflightResponse {
	builder := cloudPreflightBuilder{}
	stage := strings.TrimSpace(req.Stage)
	if stage == "" {
		stage = "manual"
	}

	profile := cloudV1Profile()
	if profile == "cloud" {
		builder.ok("cloud_profile", "Cloud v1 profile is enabled.", map[string]any{"stage": stage})
	} else {
		builder.fail(
			"cloud_profile",
			"Cloud Agentic Demo requires the cloud v1 profile.",
			"Set MODEL_EXPRESS_V1_PROFILE=cloud and start with MODEL_EXPRESS_ENV_FILE=.env.v1.cloud.",
			map[string]any{"stage": stage, "profile": profile},
		)
	}

	automation := s.currentAutomationSettings()
	provider := normalizeTrainingProvider(automation.DefaultTrainingProvider)
	if provider == "modal" {
		builder.ok("training_provider", "Default training provider is Modal.", map[string]any{"provider": provider})
	} else {
		builder.fail(
			"training_provider",
			"Cloud Agentic Demo requires Modal as the default training provider.",
			"Set MODEL_EXPRESS_DEFAULT_TRAINING_PROVIDER=modal or update Automation Settings to Modal.",
			map[string]any{"provider": provider},
		)
	}

	config := llm.ConfigFromEnv(automation.LLMEnabled, automation.LLMProvider, automation.LLMModel)
	if !config.Enabled {
		builder.fail(
			"openai_enabled",
			"OpenAI agent runtime is disabled.",
			"Set MODEL_EXPRESS_LLM_ENABLED=true.",
			nil,
		)
	} else if config.Provider != llm.ProviderOpenAI {
		builder.fail(
			"openai_provider",
			"Cloud Agentic Demo requires the OpenAI provider.",
			"Set MODEL_EXPRESS_LLM_PROVIDER=openai.",
			map[string]any{"provider": config.Provider},
		)
	} else {
		builder.ok("openai_provider", "OpenAI provider is configured.", map[string]any{"provider": config.Provider})
	}
	if config.APIStyle == llm.APIStyleResponses {
		builder.ok("openai_responses", "OpenAI Responses API style is configured.", nil)
	} else {
		builder.fail(
			"openai_responses",
			"Cloud Agentic Demo requires the OpenAI Responses API.",
			"Set MODEL_EXPRESS_LLM_API_STYLE=responses.",
			map[string]any{"api_style": config.APIStyle},
		)
	}
	if config.StoredResponses {
		builder.ok("openai_stored_responses", "Stored Responses are enabled.", nil)
	} else {
		builder.fail(
			"openai_stored_responses",
			"Cloud Agentic Demo requires stored Responses to preserve the agent workflow.",
			"Set MODEL_EXPRESS_LLM_STORED_RESPONSES=true.",
			nil,
		)
	}
	if strings.TrimSpace(config.Model) == "" {
		builder.fail("openai_model", "OpenAI model is not configured.", "Set MODEL_EXPRESS_LLM_MODEL.", nil)
	} else {
		builder.ok("openai_model", "OpenAI model is configured.", map[string]any{"model": config.Model})
	}
	if strings.TrimSpace(config.APIKey) == "" {
		builder.fail(
			"openai_key",
			"Cloud Agentic Demo requires an OpenAI API key.",
			"Set MODEL_EXPRESS_LLM_API_KEY or OPENAI_API_KEY and run preflight again.",
			nil,
		)
	} else {
		builder.ok("openai_key", "OpenAI API key is configured.", map[string]any{"source": config.APIKeySource})
	}
	if source := modalAuthSource(); source != "" {
		builder.ok("modal_auth", "Modal authentication is configured.", map[string]any{"source": source})
	} else {
		builder.fail(
			"modal_auth",
			"Cloud Agentic Demo requires Modal authentication.",
			"Run Modal auth setup or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET.",
			nil,
		)
	}

	builder.addOrchestratorURLCheck(firstEnvValue("MODAL_ORCHESTRATOR_URL", "MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL"), req.AutomaticTunnels)
	if (orchestratorExposedMode() || req.AutomaticTunnels) && strings.TrimSpace(os.Getenv("MODEL_EXPRESS_API_TOKEN")) == "" {
		builder.fail(
			"api_token",
			"Public orchestrator exposure requires MODEL_EXPRESS_API_TOKEN.",
			"Restart Mission Control and the orchestrator so they use the same generated local API token.",
			nil,
		)
	} else {
		builder.ok("api_token", "Public orchestrator API token requirement is satisfied.", map[string]any{"required": orchestratorExposedMode() || req.AutomaticTunnels})
	}
	if req.AutomaticTunnels {
		builder.ok("automatic_tunnels", "Automatic Cloudflare tunnels are enabled.", nil)
	} else {
		builder.fail(
			"automatic_tunnels",
			"Automatic Cloudflare tunnels are disabled.",
			"Allow Mission Control to create Modal tunnels automatically or configure safe public URLs.",
			nil,
		)
	}

	builder.addS3EndpointCheck(os.Getenv("S3_ENDPOINT_URL"), req.AutomaticTunnels)
	builder.addModalS3EndpointCheck(
		firstEnvValue("MODAL_S3_ENDPOINT_URL", "MODEL_EXPRESS_MODAL_S3_ENDPOINT_URL"),
		os.Getenv("S3_ENDPOINT_URL"),
		req.AutomaticTunnels,
	)

	bucket := firstEnvValue("S3_BUCKET", "MODEL_EXPRESS_ARTIFACT_BUCKET")
	if strings.TrimSpace(bucket) == "" {
		builder.fail("s3_bucket", "S3 bucket is not configured.", "Set S3_BUCKET and MODEL_EXPRESS_ARTIFACT_BUCKET.", nil)
	} else {
		builder.ok("s3_bucket", "S3 bucket is configured.", map[string]any{"bucket": bucket})
	}

	builder.addCredentialCheck("s3_upload_credentials", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", false)
	builder.addCredentialCheck("modal_s3_credentials", "MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID", "MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY", true)
	builder.addMemoryRetrievalCheck()

	status := "ok"
	for _, check := range builder.checks {
		if check.Status == "failed" {
			status = "failed"
			break
		}
	}
	return cloudPreflightResponse{Status: status, Checks: builder.checks}
}

type cloudPreflightBuilder struct {
	checks []cloudPreflightCheck
}

func (b *cloudPreflightBuilder) ok(id string, message string, metadata map[string]any) {
	b.checks = append(b.checks, cloudPreflightCheck{ID: id, Status: "ok", Message: message, Metadata: metadata})
}

func (b *cloudPreflightBuilder) fail(id string, message string, remediation string, metadata map[string]any) {
	b.checks = append(b.checks, cloudPreflightCheck{ID: id, Status: "failed", Message: message, Remediation: remediation, Metadata: metadata})
}

func (b *cloudPreflightBuilder) warn(id string, message string, metadata map[string]any) {
	b.checks = append(b.checks, cloudPreflightCheck{ID: id, Status: "warning", Message: message, Metadata: metadata})
}

func (b *cloudPreflightBuilder) addURLCheck(id string, name string, value string) {
	parsed, err := validateCloudPublicURL(value)
	if err != nil {
		b.fail(
			id,
			name+" is not a public HTTPS URL reachable by Modal.",
			"Set "+name+" to a public HTTPS origin and do not use private/local hosts or service console ports.",
			map[string]any{"error": err.Error()},
		)
		return
	}
	b.ok(id, name+" is a public HTTPS origin.", map[string]any{
		"scheme": parsed.Scheme,
		"host":   parsed.Hostname(),
	})
}

func (b *cloudPreflightBuilder) addOrchestratorURLCheck(value string, automaticTunnels bool) {
	if strings.TrimSpace(value) == "" && automaticTunnels {
		b.warn("modal_orchestrator_url", "Orchestrator tunnel will be created automatically by Mission Control at worker start.", map[string]any{"automatic_tunnel": true})
		return
	}
	b.addURLCheck("modal_orchestrator_url", "MODAL_ORCHESTRATOR_URL", value)
}

func (b *cloudPreflightBuilder) addS3EndpointCheck(value string, automaticTunnels bool) {
	text := strings.TrimSpace(value)
	if text == "" {
		if automaticTunnels && modalS3TunnelConfigured() {
			b.warn("s3_endpoint", "Local S3 endpoint will be exposed through an automatic tunnel at worker start.", map[string]any{"automatic_tunnel": true})
			return
		}
		b.fail(
			"s3_endpoint",
			"S3 endpoint is not configured.",
			"Set S3_ENDPOINT_URL to a public HTTPS S3/R2 endpoint or enable MODEL_EXPRESS_MODAL_TUNNEL_S3 for local MinIO tunneling.",
			nil,
		)
		return
	}
	parsed, err := validateCloudPublicURL(text)
	if err == nil {
		b.ok("s3_endpoint", "S3_ENDPOINT_URL is a public HTTPS origin.", map[string]any{
			"scheme": parsed.Scheme,
			"host":   parsed.Hostname(),
		})
		return
	}
	if automaticTunnels && modalS3TunnelConfigured() {
		localTarget, parseErr := url.Parse(text)
		if parseErr == nil && localTarget.Hostname() != "" && preflightLoopbackHost(localTarget.Hostname()) && localTarget.Port() != "9001" {
			b.warn("s3_endpoint", "Local S3 endpoint will be exposed through an automatic tunnel at worker start.", map[string]any{"automatic_tunnel": true})
			return
		}
	}
	b.fail(
		"s3_endpoint",
		"S3_ENDPOINT_URL is not a public HTTPS URL reachable by Modal.",
		"Set S3_ENDPOINT_URL to a public HTTPS origin and do not use private/local hosts or service console ports.",
		map[string]any{"error": err.Error()},
	)
}

func (b *cloudPreflightBuilder) addModalS3EndpointCheck(value string, s3Endpoint string, automaticTunnels bool) {
	if strings.TrimSpace(value) != "" {
		b.addURLCheck("modal_s3_endpoint", "MODAL_S3_ENDPOINT_URL", value)
		return
	}
	if parsed, err := validateCloudPublicURL(s3Endpoint); err == nil {
		b.ok("modal_s3_endpoint", "Modal workers will use S3_ENDPOINT_URL.", map[string]any{
			"scheme": parsed.Scheme,
			"host":   parsed.Hostname(),
		})
		return
	}
	if automaticTunnels && modalS3TunnelConfigured() {
		if !safeAutomaticLocalS3Target(s3Endpoint) {
			b.fail(
				"modal_s3_endpoint",
				"Automatic S3 tunnel target is not safe for Modal.",
				"Use the loopback MinIO API endpoint on port 9000 and never tunnel the MinIO console port 9001.",
				nil,
			)
			return
		}
		b.warn("modal_s3_endpoint", "S3 tunnel will be created automatically by Mission Control at worker start.", map[string]any{"automatic_tunnel": true})
		return
	}
	b.fail(
		"modal_s3_endpoint",
		"MODAL_S3_ENDPOINT_URL is not configured.",
		"Set MODAL_S3_ENDPOINT_URL, use a public HTTPS S3_ENDPOINT_URL, or enable MODEL_EXPRESS_MODAL_TUNNEL_S3 for local MinIO tunneling.",
		nil,
	)
}

func (b *cloudPreflightBuilder) addCredentialCheck(id string, accessName string, secretName string, modal bool) {
	accessKey := strings.TrimSpace(os.Getenv(accessName))
	secretKey := strings.TrimSpace(os.Getenv(secretName))
	if modal && accessKey == "" && secretKey == "" {
		accessKey = strings.TrimSpace(os.Getenv("AWS_ACCESS_KEY_ID"))
		secretKey = strings.TrimSpace(os.Getenv("AWS_SECRET_ACCESS_KEY"))
	}
	if accessKey == "" || secretKey == "" {
		b.fail(
			id,
			"S3 credentials are incomplete.",
			"Set "+accessName+" and "+secretName+" to scoped S3 credentials.",
			map[string]any{"access_key_present": accessKey != "", "secret_key_present": secretKey != ""},
		)
		return
	}
	if accessKey == "model_express" && secretKey == "model_express_password" && !envFlag("MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE", false) {
		b.fail(
			id,
			"Cloud storage refuses default local MinIO root credentials.",
			"Use scoped S3 credentials or set MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE=true only for an explicit advanced demo fallback.",
			nil,
		)
		return
	}
	b.ok(id, "Scoped S3 credentials are configured.", map[string]any{
		"access_key_present": true,
		"secret_key_present": true,
	})
}

func (b *cloudPreflightBuilder) addMemoryRetrievalCheck() {
	config := embeddings.ConfigFromEnv()
	metadata := map[string]any{
		"retrieval_enabled":  config.RetrievalEnabled,
		"embeddings_enabled": config.EmbeddingsEnabled,
		"provider":           config.Provider,
		"model":              config.Model,
	}
	if config.APIKeySource != "" {
		metadata["api_key_source"] = config.APIKeySource
	}
	if !config.RetrievalEnabled && !config.EmbeddingsEnabled {
		b.warn("memory_retrieval", "Memory retrieval is disabled.", metadata)
		return
	}
	if config.EmbeddingsEnabled {
		if err := config.ReadyForIndexing(); err != nil {
			b.fail(
				"memory_retrieval",
				"Memory retrieval is enabled but embedding configuration is incomplete.",
				"Set MODEL_EXPRESS_MEMORY_EMBEDDING_PROVIDER=openai, MODEL_EXPRESS_MEMORY_EMBEDDING_MODEL=text-embedding-3-small, and MODEL_EXPRESS_MEMORY_EMBEDDING_API_KEY or OPENAI_API_KEY.",
				map[string]any{"error": err.Error()},
			)
			return
		}
	}
	if config.RetrievalEnabled && !config.EmbeddingsEnabled {
		b.warn("memory_retrieval", "Memory retrieval is enabled, but memory embeddings are disabled; only existing indexed memory can be used.", metadata)
		return
	}
	if config.RetrievalEnabled {
		b.ok("memory_retrieval", "Memory retrieval is enabled and embedding configuration is ready.", metadata)
		return
	}
	b.warn("memory_retrieval", "Memory embeddings are enabled but retrieval is disabled.", metadata)
}

func validateCloudPublicURL(value string) (*url.URL, error) {
	text := strings.TrimSpace(value)
	if text == "" {
		return nil, errPreflight("empty")
	}
	parsed, err := url.Parse(text)
	if err != nil || parsed.Scheme == "" || parsed.Hostname() == "" {
		return nil, errPreflight("invalid URL")
	}
	if parsed.Scheme != "https" {
		return nil, errPreflight("must use https")
	}
	if parsed.User != nil {
		return nil, errPreflight("must not include credentials")
	}
	if parsed.Path != "" && parsed.Path != "/" || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, errPreflight("must be an origin without path, query, or fragment")
	}
	if preflightPrivateOrLocalHost(parsed.Hostname()) {
		return nil, errPreflight("private or local host")
	}
	switch parsed.Port() {
	case "5432", "5000", "9001":
		return nil, errPreflight("unsafe service port")
	}
	return parsed, nil
}

type preflightError string

func errPreflight(message string) error {
	return preflightError(message)
}

func (e preflightError) Error() string {
	return string(e)
}

func preflightPrivateOrLocalHost(hostname string) bool {
	host := strings.Trim(strings.ToLower(strings.TrimSpace(hostname)), "[]")
	if preflightLoopbackHost(host) || host == "0.0.0.0" || host == "::" || host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast()
}

func preflightLoopbackHost(host string) bool {
	return listenHostIsLoopback(host) || strings.EqualFold(strings.Trim(host, "[]"), "localhost")
}

func modalS3TunnelConfigured() bool {
	return envFlag("MODEL_EXPRESS_MODAL_TUNNEL_S3", false) || envFlag("MODEL_EXPRESS_ALLOW_MODAL_MINIO_TUNNEL", false)
}

func modalAuthSource() string {
	if strings.TrimSpace(os.Getenv("MODAL_TOKEN_ID")) != "" && strings.TrimSpace(os.Getenv("MODAL_TOKEN_SECRET")) != "" {
		return "MODAL_TOKEN_ID/MODAL_TOKEN_SECRET"
	}
	if strings.TrimSpace(os.Getenv("MODAL_PROFILE")) != "" {
		return "MODAL_PROFILE"
	}
	if configured := strings.TrimSpace(os.Getenv("MODAL_CONFIG_PATH")); configured != "" {
		if _, err := os.Stat(configured); err == nil {
			return "MODAL_CONFIG_PATH"
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		if _, err := os.Stat(home + string(os.PathSeparator) + ".modal.toml"); err == nil {
			return "modal_config"
		}
	}
	return ""
}

func safeAutomaticLocalS3Target(value string) bool {
	text := strings.TrimSpace(value)
	if text == "" {
		return true
	}
	parsed, err := url.Parse(text)
	if err != nil || parsed.Hostname() == "" {
		return false
	}
	return preflightLoopbackHost(parsed.Hostname()) && parsed.Port() != "9001"
}

func firstEnvValue(names ...string) string {
	for _, name := range names {
		if value := strings.TrimSpace(os.Getenv(name)); value != "" {
			return value
		}
	}
	return ""
}

func cloudV1Profile() string {
	value := strings.ToLower(strings.TrimSpace(os.Getenv("MODEL_EXPRESS_V1_PROFILE")))
	value = strings.ReplaceAll(value, "_", "-")
	if value == "cloud-v1" || value == "v1-cloud" {
		return "cloud"
	}
	return value
}
