package plans

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

func (experiment *PlannedExperiment) UnmarshalJSON(data []byte) error {
	type experimentJSON PlannedExperiment
	var decoded experimentJSON
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*experiment = PlannedExperiment(decoded)
	experiment.presentFields = presentJSONFields(data)
	return nil
}

func (experiment PlannedExperiment) MarshalJSON() ([]byte, error) {
	type experimentJSON PlannedExperiment
	return marshalPresenceAwareStruct(experimentJSON(experiment), experiment.presentFields)
}

func (config *AugmentationPolicyConfig) UnmarshalJSON(data []byte) error {
	type configJSON AugmentationPolicyConfig
	var decoded configJSON
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*config = AugmentationPolicyConfig(decoded)
	config.presentFields = presentJSONFields(data)
	return nil
}

func (config AugmentationPolicyConfig) MarshalJSON() ([]byte, error) {
	type configJSON AugmentationPolicyConfig
	return marshalPresenceAwareStruct(configJSON(config), config.presentFields)
}

func (preprocessing *Preprocessing) UnmarshalJSON(data []byte) error {
	type preprocessingJSON Preprocessing
	var decoded preprocessingJSON
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*preprocessing = Preprocessing(decoded)
	preprocessing.presentFields = presentJSONFields(data)
	return nil
}

func (preprocessing Preprocessing) MarshalJSON() ([]byte, error) {
	type preprocessingJSON Preprocessing
	return marshalPresenceAwareStruct(preprocessingJSON(preprocessing), preprocessing.presentFields)
}

func (experiment PlannedExperiment) IsFieldPresent(path string) bool {
	root, nested, hasNested := strings.Cut(path, ".")
	if !hasNested {
		return hasPresenceField(experiment.presentFields, root)
	}
	switch root {
	case "augmentation":
		_, ok := experiment.Augmentation[nested]
		return ok
	case "augmentation_policy_config":
		return experiment.AugmentationPolicyConfig != nil &&
			hasPresenceField(experiment.AugmentationPolicyConfig.presentFields, nested)
	case "class_balancing_config":
		_, ok := experiment.ClassBalancingConfig[nested]
		return ok
	case "preprocessing":
		return experiment.Preprocessing != nil &&
			hasPresenceField(experiment.Preprocessing.presentFields, nested)
	default:
		return false
	}
}

func (experiment *PlannedExperiment) MarkFieldPresent(path string) {
	if experiment == nil {
		return
	}
	root, nested, hasNested := strings.Cut(path, ".")
	experiment.presentFields = withPresenceField(experiment.presentFields, root)
	if !hasNested {
		return
	}
	switch root {
	case "augmentation_policy_config":
		if experiment.AugmentationPolicyConfig != nil {
			experiment.AugmentationPolicyConfig.presentFields = withPresenceField(
				experiment.AugmentationPolicyConfig.presentFields,
				nested,
			)
		}
	case "preprocessing":
		if experiment.Preprocessing != nil {
			experiment.Preprocessing.presentFields = withPresenceField(
				experiment.Preprocessing.presentFields,
				nested,
			)
		}
	}
}

func (experiment *PlannedExperiment) ClearFieldPresence(path string) {
	if experiment == nil {
		return
	}
	root, nested, hasNested := strings.Cut(path, ".")
	if !hasNested {
		experiment.presentFields = withoutPresenceField(experiment.presentFields, root)
		return
	}
	switch root {
	case "augmentation_policy_config":
		if experiment.AugmentationPolicyConfig != nil {
			experiment.AugmentationPolicyConfig.presentFields = withoutPresenceField(
				experiment.AugmentationPolicyConfig.presentFields,
				nested,
			)
		}
	case "preprocessing":
		if experiment.Preprocessing != nil {
			experiment.Preprocessing.presentFields = withoutPresenceField(
				experiment.Preprocessing.presentFields,
				nested,
			)
		}
	}
}

func (experiment PlannedExperiment) RequestedConfig() (map[string]any, error) {
	data, err := json.Marshal(experiment)
	if err != nil {
		return nil, fmt.Errorf("marshal requested experiment config: %w", err)
	}
	var config map[string]any
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("decode requested experiment config: %w", err)
	}
	return config, nil
}

func marshalPresenceAwareStruct(value any, presentFields map[string]struct{}) ([]byte, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	if len(presentFields) == 0 {
		return data, nil
	}
	var payload map[string]any
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil, err
	}
	structValue := reflect.ValueOf(value)
	structType := structValue.Type()
	for index := 0; index < structType.NumField(); index++ {
		field := structType.Field(index)
		jsonName := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonName == "" || jsonName == "-" || !hasPresenceField(presentFields, jsonName) {
			continue
		}
		payload[jsonName] = structValue.Field(index).Interface()
	}
	return json.Marshal(payload)
}

func presentJSONFields(data []byte) map[string]struct{} {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(data, &payload); err != nil {
		return nil
	}
	fields := make(map[string]struct{}, len(payload))
	for field := range payload {
		fields[field] = struct{}{}
	}
	return fields
}

func hasPresenceField(fields map[string]struct{}, field string) bool {
	_, ok := fields[field]
	return ok
}

func withPresenceField(fields map[string]struct{}, field string) map[string]struct{} {
	out := make(map[string]struct{}, len(fields)+1)
	for existing := range fields {
		out[existing] = struct{}{}
	}
	out[field] = struct{}{}
	return out
}

func withoutPresenceField(fields map[string]struct{}, field string) map[string]struct{} {
	out := make(map[string]struct{}, len(fields))
	for existing := range fields {
		if existing != field {
			out[existing] = struct{}{}
		}
	}
	return out
}
