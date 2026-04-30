package config

import (
	"bufio"
	"os"
	"path/filepath"
	"strings"
)

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
