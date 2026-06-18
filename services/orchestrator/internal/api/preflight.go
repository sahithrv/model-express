package api

import (
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/gin-gonic/gin"

	"model-express/services/orchestrator/internal/llm"
)

type cloudPreflightRequest struct {
	Stage string `json:"stage"`
	Live  bool   `json:"live"`
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

	builder.addURLCheck("modal_orchestrator_url", "MODAL_ORCHESTRATOR_URL", firstEnvValue("MODAL_ORCHESTRATOR_URL", "MODEL_EXPRESS_MODAL_ORCHESTRATOR_URL"))
	if orchestratorExposedMode() && strings.TrimSpace(os.Getenv("MODEL_EXPRESS_API_TOKEN")) == "" {
		builder.fail(
			"api_token",
			"Public orchestrator exposure requires MODEL_EXPRESS_API_TOKEN.",
			"Set MODEL_EXPRESS_API_TOKEN to a long random value and restart Mission Control and the orchestrator.",
			nil,
		)
	} else {
		builder.ok("api_token", "Public orchestrator API token requirement is satisfied.", map[string]any{"required": orchestratorExposedMode()})
	}

	builder.addURLCheck("s3_endpoint", "S3_ENDPOINT_URL", os.Getenv("S3_ENDPOINT_URL"))
	builder.addURLCheck("modal_s3_endpoint", "MODAL_S3_ENDPOINT_URL", firstEnvValue("MODAL_S3_ENDPOINT_URL", "MODEL_EXPRESS_MODAL_S3_ENDPOINT_URL"))

	bucket := firstEnvValue("S3_BUCKET", "MODEL_EXPRESS_ARTIFACT_BUCKET")
	if strings.TrimSpace(bucket) == "" {
		builder.fail("s3_bucket", "S3 bucket is not configured.", "Set S3_BUCKET and MODEL_EXPRESS_ARTIFACT_BUCKET.", nil)
	} else {
		builder.ok("s3_bucket", "S3 bucket is configured.", map[string]any{"bucket": bucket})
	}

	builder.addCredentialCheck("s3_upload_credentials", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", false)
	builder.addCredentialCheck("modal_s3_credentials", "MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID", "MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY", true)

	if envFlag("MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED", false) || envFlag("MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED", false) {
		builder.fail(
			"memory_bootstrap",
			"Cloud v1 release does not enable memory bootstrap or embeddings.",
			"Set MODEL_EXPRESS_MEMORY_RETRIEVAL_ENABLED=false and MODEL_EXPRESS_MEMORY_EMBEDDINGS_ENABLED=false.",
			nil,
		)
	} else {
		builder.ok("memory_bootstrap", "Memory bootstrap and embeddings are disabled for v1.", nil)
	}

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
	if listenHostIsLoopback(host) || host == "0.0.0.0" || host == "::" || host == "localhost" || strings.HasSuffix(host, ".localhost") {
		return true
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast()
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
