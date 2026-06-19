package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLoadRepoEnvGeneratesAndReusesLocalAPIToken(t *testing.T) {
	userDataDir := t.TempDir()
	t.Setenv("MODEL_EXPRESS_USER_DATA_DIR", userDataDir)
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "")

	if err := LoadRepoEnv(t.TempDir()); err != nil {
		t.Fatalf("load repo env: %v", err)
	}
	token := os.Getenv("MODEL_EXPRESS_API_TOKEN")
	if len(token) != 43 || strings.ContainsAny(token, "+/=") {
		t.Fatalf("expected generated base64url API token shape")
	}
	configPath := filepath.Join(userDataDir, localConfigFile)
	contents, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read local config: %v", err)
	}
	if !strings.Contains(string(contents), token) {
		t.Fatalf("expected local config to contain generated token")
	}

	if err := os.Setenv("MODEL_EXPRESS_API_TOKEN", ""); err != nil {
		t.Fatalf("clear token: %v", err)
	}
	if err := ensureLocalAPITokenEnv(); err != nil {
		t.Fatalf("reload local token: %v", err)
	}
	if os.Getenv("MODEL_EXPRESS_API_TOKEN") != token {
		t.Fatalf("expected persisted token to be reused")
	}
}

func TestLoadRepoEnvGeneratesLocalRuntimeStorageEnv(t *testing.T) {
	userDataDir := t.TempDir()
	t.Setenv("MODEL_EXPRESS_USER_DATA_DIR", userDataDir)
	t.Setenv("MODEL_EXPRESS_API_TOKEN", "")
	t.Setenv("S3_ENDPOINT_URL", "")
	t.Setenv("AWS_ACCESS_KEY_ID", "")
	t.Setenv("AWS_SECRET_ACCESS_KEY", "")
	t.Setenv("MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID", "")
	t.Setenv("MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY", "")
	t.Setenv("MODEL_EXPRESS_MODAL_TUNNEL_S3", "")
	t.Setenv("MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE", "")

	if err := LoadRepoEnv(t.TempDir()); err != nil {
		t.Fatalf("load repo env: %v", err)
	}

	accessKey := os.Getenv("AWS_ACCESS_KEY_ID")
	secretKey := os.Getenv("AWS_SECRET_ACCESS_KEY")
	if accessKey == "" || secretKey == "" {
		t.Fatalf("expected generated local MinIO credentials")
	}
	if accessKey == "model_express" || secretKey == "model_express_password" {
		t.Fatalf("expected generated credentials, not shared defaults")
	}
	if os.Getenv("S3_ENDPOINT_URL") != defaultS3EndpointURL {
		t.Fatalf("expected local S3 endpoint, got %q", os.Getenv("S3_ENDPOINT_URL"))
	}
	if os.Getenv("S3_BUCKET") != defaultS3Bucket || os.Getenv("MODEL_EXPRESS_ARTIFACT_BUCKET") != defaultS3Bucket {
		t.Fatalf("expected default bucket env")
	}
	if os.Getenv("MODEL_EXPRESS_ARTIFACT_PREFIX") != defaultArtifactPrefix {
		t.Fatalf("expected default artifact prefix")
	}
	if os.Getenv("MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID") != accessKey || os.Getenv("MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY") != secretKey {
		t.Fatalf("expected modal storage credentials to match generated local credentials")
	}
	if os.Getenv("MODEL_EXPRESS_MODAL_TUNNEL_S3") != "true" {
		t.Fatalf("expected automatic S3 tunnel flag")
	}
	if os.Getenv("MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE") != "true" {
		t.Fatalf("expected RC local-only root storage allowance")
	}
}
