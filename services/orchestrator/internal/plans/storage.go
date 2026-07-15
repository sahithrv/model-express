package plans

import (
	"bytes"
	"encoding/json"
	"fmt"
)

const StoredExperimentsSchemaV1 = "planned_experiments.v1"

type StoredExperimentsEnvelopeV1 struct {
	SchemaVersion     string              `json:"schema_version"`
	CapabilityVersion string              `json:"capability_version"`
	Experiments       []PlannedExperiment `json:"experiments"`
}

func MarshalVersionedExperiments(
	experiments []PlannedExperiment,
	capabilityVersion string,
) ([]byte, error) {
	return json.Marshal(StoredExperimentsEnvelopeV1{
		SchemaVersion:     StoredExperimentsSchemaV1,
		CapabilityVersion: capabilityVersion,
		Experiments:       experiments,
	})
}

func UnmarshalStoredExperiments(data []byte) (
	experiments []PlannedExperiment,
	capabilityVersion string,
	legacyUnversioned bool,
	err error,
) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return []PlannedExperiment{}, "", true, nil
	}
	if trimmed[0] == '[' {
		if err := json.Unmarshal(trimmed, &experiments); err != nil {
			return nil, "", false, fmt.Errorf("decode legacy planned experiments: %w", err)
		}
		return experiments, "", true, nil
	}
	var envelope StoredExperimentsEnvelopeV1
	if err := json.Unmarshal(trimmed, &envelope); err != nil {
		return nil, "", false, fmt.Errorf("decode versioned planned experiments: %w", err)
	}
	if envelope.SchemaVersion != StoredExperimentsSchemaV1 {
		return nil, "", false, fmt.Errorf(
			"unsupported planned experiments schema %q",
			envelope.SchemaVersion,
		)
	}
	if envelope.Experiments == nil {
		envelope.Experiments = []PlannedExperiment{}
	}
	return envelope.Experiments, envelope.CapabilityVersion, false, nil
}
