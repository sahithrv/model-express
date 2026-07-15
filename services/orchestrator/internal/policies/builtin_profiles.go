package policies

import (
	"encoding/json"
	"fmt"
	"time"
)

type generatedCompatibilityProfile struct {
	ProfileKey      string                       `json:"profile_key"`
	SemanticVersion string                       `json:"semantic_version"`
	Document        CompatibilityProfileDocument `json:"document"`
}

// BuiltinCompatibilityProfiles returns immutable, generator-validated profile
// records. Persistence layers seed these records without making them mutable.
func BuiltinCompatibilityProfiles() []CompatibilityProfile {
	var generated []generatedCompatibilityProfile
	if err := json.Unmarshal([]byte(generatedCompatibilityProfilesJSON), &generated); err != nil {
		panic(fmt.Errorf("decode generated compatibility profiles: %w", err))
	}
	profiles := make([]CompatibilityProfile, 0, len(generated))
	for _, input := range generated {
		document, canonical, hash, err := NormalizeCompatibilityProfileDocument(input.Document)
		if err != nil {
			panic(fmt.Errorf("normalize generated compatibility profile %s@%s: %w", input.ProfileKey, input.SemanticVersion, err))
		}
		profiles = append(profiles, CompatibilityProfile{
			ID:         "compatibility_profile_builtin_" + input.ProfileKey + "_" + input.SemanticVersion,
			ProfileKey: input.ProfileKey, SemanticVersion: input.SemanticVersion,
			SchemaVersion: document.SchemaVersion, CatalogVersion: document.CatalogVersion,
			Document: document, CanonicalJSON: canonical, DocumentHash: hash,
			CreatedAt: time.Unix(0, 0).UTC(), CreatedBy: "builtin_generator",
		})
	}
	return profiles
}
