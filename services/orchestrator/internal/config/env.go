package config

import (
	"bufio"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const localConfigFile = "model-express.local.json"
const defaultS3EndpointURL = "http://127.0.0.1:9000"
const defaultS3Bucket = "model-express"
const defaultArtifactPrefix = "model-express/artifacts"

func LoadRepoEnv(repoRoot string) error {
	envFile := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_ENV_FILE"))
	files := []string{
		filepath.Join(repoRoot, ".env"),
		filepath.Join(repoRoot, ".env.local"),
	}
	if envFile != "" {
		files = []string{resolveEnvFile(repoRoot, envFile)}
	}

	for _, file := range files {
		if err := loadEnvFile(file); err != nil {
			return err
		}
	}
	if err := ensureLocalAPITokenEnv(); err != nil {
		return err
	}
	if err := ensureLocalRuntimeEnv(); err != nil {
		return err
	}
	return nil
}

func loadEnvFile(path string) error {
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		if _, exists := os.LookupEnv(key); exists {
			continue
		}
		value = strings.TrimSpace(value)
		if len(value) >= 2 {
			if (strings.HasPrefix(value, `"`) && strings.HasSuffix(value, `"`)) ||
				(strings.HasPrefix(value, `'`) && strings.HasSuffix(value, `'`)) {
				value = value[1 : len(value)-1]
			}
		}
		if err := os.Setenv(key, value); err != nil {
			return err
		}
	}
	return scanner.Err()
}

func resolveEnvFile(repoRoot string, envFile string) string {
	if filepath.IsAbs(envFile) {
		return envFile
	}
	return filepath.Join(repoRoot, envFile)
}

func ensureLocalAPITokenEnv() error {
	if strings.TrimSpace(os.Getenv("MODEL_EXPRESS_API_TOKEN")) != "" {
		return nil
	}
	configPath := localConfigPath()
	_, statErr := os.Stat(configPath)
	configMissing := os.IsNotExist(statErr)
	config, err := readLocalConfig()
	if err != nil {
		return err
	}
	token, _ := config["model_express_api_token"].(string)
	token = strings.TrimSpace(token)
	if token == "" {
		generated, err := generateLocalAPIToken()
		if err != nil {
			return err
		}
		token = generated
		config["model_express_api_token"] = token
		config["updated_at"] = time.Now().UTC().Format(time.RFC3339)
		if configMissing {
			created, err := writeNewLocalConfig(config)
			if err != nil {
				return err
			}
			if !created {
				reread, err := readLocalConfig()
				if err != nil {
					return err
				}
				existing, _ := reread["model_express_api_token"].(string)
				if existing = strings.TrimSpace(existing); existing != "" {
					token = existing
				} else if err := writeLocalConfig(config); err != nil {
					return err
				}
			}
		} else {
			if err := writeLocalConfig(config); err != nil {
				return err
			}
		}
	}
	return os.Setenv("MODEL_EXPRESS_API_TOKEN", token)
}

func readLocalConfig() (map[string]any, error) {
	config := map[string]any{}
	contents, err := os.ReadFile(localConfigPath())
	if os.IsNotExist(err) {
		return config, nil
	}
	if err != nil {
		return nil, err
	}
	if len(strings.TrimSpace(string(contents))) == 0 {
		return config, nil
	}
	if err := json.Unmarshal(contents, &config); err != nil {
		return nil, err
	}
	if config == nil {
		config = map[string]any{}
	}
	return config, nil
}

func writeLocalConfig(config map[string]any) error {
	path := localConfigPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	payload, err := localConfigPayload(config)
	if err != nil {
		return err
	}
	tmp := filepath.Join(filepath.Dir(path), "."+filepath.Base(path)+".tmp")
	if err := os.WriteFile(tmp, payload, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func writeNewLocalConfig(config map[string]any) (bool, error) {
	path := localConfigPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return false, err
	}
	payload, err := localConfigPayload(config)
	if err != nil {
		return false, err
	}
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if os.IsExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer file.Close()
	if _, err := file.Write(payload); err != nil {
		return false, err
	}
	return true, nil
}

func localConfigPayload(config map[string]any) ([]byte, error) {
	payload, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return nil, err
	}
	return append(payload, '\n'), nil
}

func generateLocalAPIToken() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

func ensureLocalRuntimeEnv() error {
	config, err := readLocalConfig()
	if err != nil {
		return err
	}
	changed := false

	minioAccessKey, _ := config["minio_access_key"].(string)
	minioAccessKey = strings.TrimSpace(minioAccessKey)
	if minioAccessKey == "" {
		generated, err := generateMinioAccessKey()
		if err != nil {
			return err
		}
		minioAccessKey = generated
		config["minio_access_key"] = minioAccessKey
		changed = true
	}

	minioSecretKey, _ := config["minio_secret_key"].(string)
	minioSecretKey = strings.TrimSpace(minioSecretKey)
	if minioSecretKey == "" {
		generated, err := generateRuntimeSecret()
		if err != nil {
			return err
		}
		minioSecretKey = generated
		config["minio_secret_key"] = minioSecretKey
		changed = true
	}

	bucket, _ := config["s3_bucket"].(string)
	bucket = strings.TrimSpace(bucket)
	if bucket == "" {
		bucket = defaultS3Bucket
		config["s3_bucket"] = bucket
		changed = true
	}

	artifactPrefix, _ := config["artifact_prefix"].(string)
	artifactPrefix = normalizeArtifactPrefix(artifactPrefix)
	if artifactPrefix == "" {
		artifactPrefix = defaultArtifactPrefix
	}
	if config["artifact_prefix"] != artifactPrefix {
		config["artifact_prefix"] = artifactPrefix
		changed = true
	}

	if changed {
		config["updated_at"] = time.Now().UTC().Format(time.RFC3339)
		if err := writeLocalConfig(config); err != nil {
			return err
		}
	}

	appManaged := envFlag("MODEL_EXPRESS_APP_MANAGED_LOCAL_RUNTIME", false)
	setRuntimeEnvDefault("S3_BUCKET", bucket, appManaged)
	setRuntimeEnvDefault("MODEL_EXPRESS_ARTIFACT_BUCKET", bucket, appManaged)
	setRuntimeEnvDefault("MODEL_EXPRESS_ARTIFACT_PREFIX", artifactPrefix, appManaged)
	setRuntimeEnvDefault("AWS_ACCESS_KEY_ID", minioAccessKey, appManaged)
	setRuntimeEnvDefault("AWS_SECRET_ACCESS_KEY", minioSecretKey, appManaged)
	setRuntimeEnvDefault("MODEL_EXPRESS_MODAL_AWS_ACCESS_KEY_ID", minioAccessKey, appManaged)
	setRuntimeEnvDefault("MODEL_EXPRESS_MODAL_AWS_SECRET_ACCESS_KEY", minioSecretKey, appManaged)
	setRuntimeEnvDefault("AWS_DEFAULT_REGION", "us-east-1", false)
	if appManaged || strings.TrimSpace(os.Getenv("S3_ENDPOINT_URL")) == "" {
		if err := os.Setenv("S3_ENDPOINT_URL", defaultS3EndpointURL); err != nil {
			return err
		}
	}
	if strings.TrimSpace(os.Getenv("MODEL_EXPRESS_MODAL_TUNNEL_S3")) == "" && localRuntimeS3Endpoint() {
		if err := os.Setenv("MODEL_EXPRESS_MODAL_TUNNEL_S3", "true"); err != nil {
			return err
		}
	}
	if strings.TrimSpace(os.Getenv("MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE")) == "" && usingGeneratedLocalMinIOCredentials(minioAccessKey, minioSecretKey) {
		if err := os.Setenv("MODEL_EXPRESS_ALLOW_MODAL_ROOT_STORAGE", "true"); err != nil {
			return err
		}
	}
	return nil
}

func setRuntimeEnvDefault(name string, value string, override bool) {
	if strings.TrimSpace(value) == "" {
		return
	}
	if override || strings.TrimSpace(os.Getenv(name)) == "" {
		_ = os.Setenv(name, value)
	}
}

func usingGeneratedLocalMinIOCredentials(accessKey string, secretKey string) bool {
	return strings.TrimSpace(os.Getenv("AWS_ACCESS_KEY_ID")) == accessKey &&
		strings.TrimSpace(os.Getenv("AWS_SECRET_ACCESS_KEY")) == secretKey
}

func localRuntimeS3Endpoint() bool {
	value := strings.TrimSpace(os.Getenv("S3_ENDPOINT_URL"))
	return value == "" || strings.HasPrefix(value, "http://127.0.0.1:9000") || strings.HasPrefix(value, "http://localhost:9000")
}

func generateMinioAccessKey() (string, error) {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return "mx" + hex.EncodeToString(buf), nil
}

func generateRuntimeSecret() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}

func normalizeArtifactPrefix(value string) string {
	parts := strings.FieldsFunc(strings.TrimSpace(value), func(r rune) bool { return r == '/' })
	return strings.Join(parts, "/")
}

func envFlag(name string, defaultValue bool) bool {
	value := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	if value == "" {
		return defaultValue
	}
	switch value {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return defaultValue
	}
}

func localConfigPath() string {
	return filepath.Join(modelExpressUserDataDir(), localConfigFile)
}

func modelExpressUserDataDir() string {
	if configured := strings.TrimSpace(os.Getenv("MODEL_EXPRESS_USER_DATA_DIR")); configured != "" {
		return configured
	}
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return filepath.Join(os.TempDir(), "model-express")
	}
	switch runtime.GOOS {
	case "windows":
		base := strings.TrimSpace(os.Getenv("APPDATA"))
		if base == "" {
			base = filepath.Join(home, "AppData", "Roaming")
		}
		return filepath.Join(base, "Model Express")
	case "darwin":
		return filepath.Join(home, "Library", "Application Support", "Model Express")
	default:
		base := strings.TrimSpace(os.Getenv("XDG_CONFIG_HOME"))
		if base == "" {
			base = filepath.Join(home, ".config")
		}
		return filepath.Join(base, "model-express")
	}
}
