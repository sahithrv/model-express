package policies

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"

	"model-express/services/orchestrator/internal/catalog"
)

var (
	identifierPattern = regexp.MustCompile(`^[a-z][a-z0-9_.-]{0,127}$`)
	fieldPattern      = regexp.MustCompile(`^[a-z][a-z0-9_.-]{0,255}$`)
	semverPattern     = regexp.MustCompile(`^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$`)
)

type fieldValueKind int

const (
	fieldValueAny fieldValueKind = iota
	fieldValueBoolean
	fieldValueInteger
	fieldValueNumber
)

var typedPolicyFields = map[string]fieldValueKind{
	"pretrained":                              fieldValueBoolean,
	"freeze_backbone":                         fieldValueBoolean,
	"use_dataset_normalization":               fieldValueBoolean,
	"preprocessing.use_dataset_normalization": fieldValueBoolean,
	"epochs":                  fieldValueInteger,
	"batch_size":              fieldValueInteger,
	"image_size":              fieldValueInteger,
	"early_stopping_patience": fieldValueInteger,
	"learning_rate":           fieldValueNumber,
	"weight_decay":            fieldValueNumber,
	"dropout":                 fieldValueNumber,
	"label_smoothing":         fieldValueNumber,
	"gradient_clip_norm":      fieldValueNumber,
}

var attributeCatalogs = map[string]string{
	"family":            "model_families",
	"deployment_tier":   "deployment_tiers",
	"latency_tier":      "latency_tiers",
	"fine_tuning_modes": "fine_tuning_modes",
}

func NormalizePolicyDocument(input PolicyDocument) (PolicyDocument, json.RawMessage, string, error) {
	if input.SchemaVersion != PolicySchemaVersionV1 {
		return PolicyDocument{}, nil, "", fmt.Errorf("policy schema_version must be %q", PolicySchemaVersionV1)
	}
	normalized := PolicyDocument{
		SchemaVersion: PolicySchemaVersionV1,
		ProfileRefs:   make([]ProfileRef, 0, len(input.ProfileRefs)),
		Rules:         make([]Rule, 0, len(input.Rules)),
		Metadata:      normalizeMetadata(input.Metadata),
	}
	if err := ValidateMetadata(normalized.Metadata); err != nil {
		return PolicyDocument{}, nil, "", err
	}
	seenProfiles := map[string]struct{}{}
	for _, ref := range input.ProfileRefs {
		ref.ID = strings.TrimSpace(ref.ID)
		ref.Version = strings.TrimSpace(ref.Version)
		if !identifierPattern.MatchString(ref.ID) {
			return PolicyDocument{}, nil, "", fmt.Errorf("invalid compatibility profile id %q", ref.ID)
		}
		if !semverPattern.MatchString(ref.Version) {
			return PolicyDocument{}, nil, "", fmt.Errorf("invalid compatibility profile version %q", ref.Version)
		}
		key := ref.ID + "@" + ref.Version
		if _, ok := seenProfiles[key]; ok {
			return PolicyDocument{}, nil, "", fmt.Errorf("duplicate compatibility profile ref %s", key)
		}
		seenProfiles[key] = struct{}{}
		normalized.ProfileRefs = append(normalized.ProfileRefs, ref)
	}
	sort.Slice(normalized.ProfileRefs, func(i, j int) bool {
		if normalized.ProfileRefs[i].ID != normalized.ProfileRefs[j].ID {
			return normalized.ProfileRefs[i].ID < normalized.ProfileRefs[j].ID
		}
		return normalized.ProfileRefs[i].Version < normalized.ProfileRefs[j].Version
	})

	seenRules := map[string]struct{}{}
	for _, rule := range input.Rules {
		rule.ID = strings.TrimSpace(rule.ID)
		if !identifierPattern.MatchString(rule.ID) {
			return PolicyDocument{}, nil, "", fmt.Errorf("invalid policy rule id %q", rule.ID)
		}
		if _, ok := seenRules[rule.ID]; ok {
			return PolicyDocument{}, nil, "", fmt.Errorf("duplicate policy rule id %q", rule.ID)
		}
		seenRules[rule.ID] = struct{}{}
		if strings.TrimSpace(rule.Effect) != EffectDeny {
			return PolicyDocument{}, nil, "", fmt.Errorf("user policy rule %q must use effect %q", rule.ID, EffectDeny)
		}
		selector, err := normalizeSelector(rule.Selector, false)
		if err != nil {
			return PolicyDocument{}, nil, "", fmt.Errorf("policy rule %q: %w", rule.ID, err)
		}
		normalized.Rules = append(normalized.Rules, Rule{ID: rule.ID, Effect: EffectDeny, Selector: selector})
	}
	sort.Slice(normalized.Rules, func(i, j int) bool { return normalized.Rules[i].ID < normalized.Rules[j].ID })
	canonical, hash, err := canonicalJSONAndHash(normalized)
	return normalized, canonical, hash, err
}

func NormalizeCompatibilityProfileDocument(input CompatibilityProfileDocument) (CompatibilityProfileDocument, json.RawMessage, string, error) {
	if input.SchemaVersion != CompatibilityProfileSchemaVersionV1 {
		return CompatibilityProfileDocument{}, nil, "", fmt.Errorf("compatibility profile schema_version must be %q", CompatibilityProfileSchemaVersionV1)
	}
	if input.CatalogVersion != catalog.CatalogV1().CatalogVersion {
		return CompatibilityProfileDocument{}, nil, "", fmt.Errorf("compatibility profile catalog_version %q is unavailable", input.CatalogVersion)
	}
	normalized := CompatibilityProfileDocument{
		SchemaVersion:  CompatibilityProfileSchemaVersionV1,
		CatalogVersion: input.CatalogVersion,
		Rules:          make([]Rule, 0, len(input.Rules)),
		Metadata:       normalizeMetadata(input.Metadata),
	}
	if err := ValidateMetadata(normalized.Metadata); err != nil {
		return CompatibilityProfileDocument{}, nil, "", err
	}
	seenRules := map[string]struct{}{}
	for _, rule := range input.Rules {
		rule.ID = strings.TrimSpace(rule.ID)
		if !identifierPattern.MatchString(rule.ID) {
			return CompatibilityProfileDocument{}, nil, "", fmt.Errorf("invalid compatibility profile rule id %q", rule.ID)
		}
		if _, ok := seenRules[rule.ID]; ok {
			return CompatibilityProfileDocument{}, nil, "", fmt.Errorf("duplicate compatibility profile rule id %q", rule.ID)
		}
		seenRules[rule.ID] = struct{}{}
		if strings.TrimSpace(rule.Effect) != EffectAllow {
			return CompatibilityProfileDocument{}, nil, "", fmt.Errorf("compatibility profile rule %q must use effect %q", rule.ID, EffectAllow)
		}
		selector, err := normalizeSelector(rule.Selector, true)
		if err != nil {
			return CompatibilityProfileDocument{}, nil, "", fmt.Errorf("compatibility profile rule %q: %w", rule.ID, err)
		}
		if selector.Kind == SelectorFieldValues {
			return CompatibilityProfileDocument{}, nil, "", fmt.Errorf("compatibility profile rule %q cannot use field_values", rule.ID)
		}
		normalized.Rules = append(normalized.Rules, Rule{ID: rule.ID, Effect: EffectAllow, Selector: selector})
	}
	if len(normalized.Rules) == 0 {
		return CompatibilityProfileDocument{}, nil, "", fmt.Errorf("compatibility profile must contain at least one allow rule")
	}
	sort.Slice(normalized.Rules, func(i, j int) bool { return normalized.Rules[i].ID < normalized.Rules[j].ID })
	canonical, hash, err := canonicalJSONAndHash(normalized)
	return normalized, canonical, hash, err
}

func ValidateProfileIdentity(profileKey string, semanticVersion string) error {
	if !identifierPattern.MatchString(strings.TrimSpace(profileKey)) {
		return fmt.Errorf("invalid compatibility profile key %q", profileKey)
	}
	if !semverPattern.MatchString(strings.TrimSpace(semanticVersion)) {
		return fmt.Errorf("invalid compatibility profile semantic version %q", semanticVersion)
	}
	return nil
}

func normalizeSelector(input Selector, profile bool) (Selector, error) {
	kind := strings.TrimSpace(input.Kind)
	switch kind {
	case SelectorCatalogIDs:
		category := strings.ToLower(strings.TrimSpace(input.Catalog))
		if !catalogCategoryExists(category) {
			return Selector{}, unknownCatalogError(category, "")
		}
		if len(input.IDs) == 0 {
			return Selector{}, fmt.Errorf("catalog_ids requires at least one id")
		}
		ids := make([]string, 0, len(input.IDs))
		seen := map[string]struct{}{}
		for _, requested := range input.IDs {
			entry, ok := catalog.Resolve(category, requested)
			if !ok {
				return Selector{}, unknownCatalogError(category, strings.TrimSpace(requested))
			}
			if _, ok := seen[entry.ID]; ok {
				continue
			}
			seen[entry.ID] = struct{}{}
			ids = append(ids, entry.ID)
		}
		sort.Strings(ids)
		return Selector{Kind: kind, Catalog: category, IDs: ids}, nil

	case SelectorCatalogAttributeValues:
		category := strings.ToLower(strings.TrimSpace(input.Catalog))
		attribute := strings.ToLower(strings.TrimSpace(input.Attribute))
		if !catalogCategoryExists(category) {
			return Selector{}, unknownCatalogError(category, "")
		}
		if !identifierPattern.MatchString(attribute) {
			return Selector{}, fmt.Errorf("invalid catalog attribute %q", input.Attribute)
		}
		if len(input.Values) == 0 {
			return Selector{}, fmt.Errorf("catalog_attribute_values requires at least one value")
		}
		values := make([]any, 0, len(input.Values))
		seen := map[string]struct{}{}
		for _, raw := range input.Values {
			value, ok := raw.(string)
			if !ok || !identifierPattern.MatchString(strings.ToLower(strings.TrimSpace(value))) {
				return Selector{}, fmt.Errorf("catalog attribute values must be identifiers")
			}
			canonicalValue, err := canonicalAttributeValue(attribute, value)
			if err != nil {
				return Selector{}, err
			}
			if !catalogAttributeValueExists(category, attribute, canonicalValue) {
				return Selector{}, unknownCatalogError(category+"."+attribute, canonicalValue)
			}
			if _, ok := seen[canonicalValue]; ok {
				continue
			}
			seen[canonicalValue] = struct{}{}
			values = append(values, canonicalValue)
		}
		sort.Slice(values, func(i, j int) bool { return values[i].(string) < values[j].(string) })
		return Selector{Kind: kind, Catalog: category, Attribute: attribute, Values: values}, nil

	case SelectorFieldValues:
		if profile {
			return Selector{}, fmt.Errorf("field_values is not valid in a compatibility profile")
		}
		field := strings.ToLower(strings.TrimSpace(input.Field))
		if !fieldPattern.MatchString(field) {
			return Selector{}, fmt.Errorf("invalid policy field %q", input.Field)
		}
		if len(input.Values) == 0 {
			return Selector{}, fmt.Errorf("field_values requires at least one value")
		}
		values := make([]any, 0, len(input.Values))
		seen := map[string]struct{}{}
		for _, value := range input.Values {
			canonical, err := normalizeFieldValue(field, value)
			if err != nil {
				return Selector{}, err
			}
			key, err := canonicalValueString(canonical)
			if err != nil {
				return Selector{}, err
			}
			if _, ok := seen[key]; ok {
				continue
			}
			seen[key] = struct{}{}
			values = append(values, canonical)
		}
		sort.Slice(values, func(i, j int) bool {
			left, _ := canonicalValueString(values[i])
			right, _ := canonicalValueString(values[j])
			return left < right
		})
		return Selector{Kind: kind, Field: field, Values: values}, nil
	default:
		return Selector{}, fmt.Errorf("unsupported selector kind %q", input.Kind)
	}
}

func normalizeFieldValue(field string, value any) (any, error) {
	kind := typedPolicyFields[field]
	switch typed := value.(type) {
	case bool:
		if kind != fieldValueAny && kind != fieldValueBoolean {
			return nil, fmt.Errorf("field %q requires a numeric value", field)
		}
		return typed, nil
	case string:
		if kind == fieldValueBoolean {
			return nil, fmt.Errorf("field %q requires boolean values; string-valued booleans are rejected", field)
		}
		if kind == fieldValueInteger || kind == fieldValueNumber {
			return nil, fmt.Errorf("field %q requires numeric values", field)
		}
		if !utf8.ValidString(typed) {
			return nil, fmt.Errorf("field %q contains invalid UTF-8", field)
		}
		return typed, nil
	case float64:
		if math.IsNaN(typed) || math.IsInf(typed, 0) {
			return nil, fmt.Errorf("field %q contains a non-finite number", field)
		}
		if kind == fieldValueBoolean {
			return nil, fmt.Errorf("field %q requires boolean values", field)
		}
		if kind == fieldValueInteger && math.Trunc(typed) != typed {
			return nil, fmt.Errorf("field %q requires integer values", field)
		}
		return typed, nil
	case float32:
		return normalizeFieldValue(field, float64(typed))
	case int:
		return normalizeFieldValue(field, float64(typed))
	case int8:
		return normalizeFieldValue(field, float64(typed))
	case int16:
		return normalizeFieldValue(field, float64(typed))
	case int32:
		return normalizeFieldValue(field, float64(typed))
	case int64:
		return normalizeFieldValue(field, float64(typed))
	case uint:
		return normalizeFieldValue(field, float64(typed))
	case uint8:
		return normalizeFieldValue(field, float64(typed))
	case uint16:
		return normalizeFieldValue(field, float64(typed))
	case uint32:
		return normalizeFieldValue(field, float64(typed))
	case uint64:
		if typed > 1<<53 {
			return nil, fmt.Errorf("field %q integer exceeds exact JSON range", field)
		}
		return normalizeFieldValue(field, float64(typed))
	case json.Number:
		number, err := typed.Float64()
		if err != nil {
			return nil, fmt.Errorf("field %q contains invalid number: %w", field, err)
		}
		return normalizeFieldValue(field, number)
	default:
		return nil, fmt.Errorf("field %q values must be string, number, integer, or boolean", field)
	}
}

func normalizeMetadata(input Metadata) Metadata {
	return Metadata{
		DisplayName: strings.TrimSpace(input.DisplayName),
		Description: strings.TrimSpace(input.Description),
	}
}

func ValidateMetadata(metadata Metadata) error {
	if utf8.RuneCountInString(metadata.DisplayName) > 120 {
		return fmt.Errorf("policy display_name exceeds 120 characters")
	}
	if utf8.RuneCountInString(metadata.Description) > 1000 {
		return fmt.Errorf("policy description exceeds 1000 characters")
	}
	return nil
}

func canonicalJSONAndHash(value any) (json.RawMessage, string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, "", err
	}
	var generic any
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&generic); err != nil {
		return nil, "", err
	}
	canonical, err := json.Marshal(generic)
	if err != nil {
		return nil, "", err
	}
	hash := sha256.Sum256(canonical)
	return json.RawMessage(canonical), "sha256:" + hex.EncodeToString(hash[:]), nil
}

func HashCanonicalJSON(data []byte) (string, error) {
	var value any
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return "", err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err == nil {
		return "", fmt.Errorf("multiple JSON values are not allowed")
	} else if err != io.EOF {
		return "", err
	}
	_, hash, err := canonicalJSONAndHash(value)
	if err != nil {
		return "", err
	}
	return hash, nil
}

func canonicalValueString(value any) (string, error) {
	data, err := json.Marshal(value)
	return string(data), err
}

func catalogCategoryExists(category string) bool {
	_, ok := catalog.CatalogV1().Categories[category]
	return ok
}

func unknownCatalogError(category string, id string) error {
	finding := Finding{
		Code:        ReasonUnknownCatalogIdentifier,
		Catalog:     category,
		ID:          id,
		Origin:      "policy",
		Remediation: "Use an identifier from the pinned Model Express catalog.",
	}
	message := fmt.Sprintf("unknown policy catalog %q", category)
	if id != "" {
		message = fmt.Sprintf("unknown policy catalog identifier %s/%s", category, id)
	}
	return policyError(ReasonUnknownCatalogIdentifier, message, finding)
}

func canonicalAttributeValue(attribute string, value string) (string, error) {
	value = strings.ToLower(strings.TrimSpace(value))
	if category := attributeCatalogs[attribute]; category != "" {
		entry, ok := catalog.Resolve(category, value)
		if !ok {
			return "", unknownCatalogError(category, value)
		}
		return entry.ID, nil
	}
	return value, nil
}

func catalogAttributeValueExists(category string, attribute string, value string) bool {
	for _, entry := range catalog.Entries(category) {
		if attributeContains(entry.Attributes[attribute], value) {
			return true
		}
	}
	return false
}

func attributeContains(raw any, expected string) bool {
	switch value := raw.(type) {
	case string:
		return strings.EqualFold(strings.TrimSpace(value), expected)
	case []string:
		for _, item := range value {
			if strings.EqualFold(strings.TrimSpace(item), expected) {
				return true
			}
		}
	case []any:
		for _, item := range value {
			if text, ok := item.(string); ok && strings.EqualFold(strings.TrimSpace(text), expected) {
				return true
			}
		}
	}
	return false
}

func cloneJSON[T any](input T) T {
	data, err := json.Marshal(input)
	if err != nil {
		return input
	}
	var output T
	if err := json.Unmarshal(data, &output); err != nil {
		return input
	}
	return output
}

func CloneCompatibilityProfile(input CompatibilityProfile) CompatibilityProfile {
	return cloneJSON(input)
}

func ClonePolicyVersion(input PolicyVersion) PolicyVersion {
	return cloneJSON(input)
}

func CloneBinding(input Binding) Binding {
	return cloneJSON(input)
}

func CloneEvaluation(input Evaluation) Evaluation {
	return cloneJSON(input)
}
