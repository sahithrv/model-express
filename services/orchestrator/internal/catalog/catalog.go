package catalog

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
)

const SchemaVersionV1 = "model_express_catalog.v1"

type Document struct {
	SchemaVersion  string             `json:"schema_version"`
	CatalogVersion string             `json:"catalog_version"`
	Categories     map[string][]Entry `json:"categories"`
}

type Entry struct {
	ID           string         `json:"id"`
	Aliases      []string       `json:"aliases"`
	Available    bool           `json:"available"`
	Tasks        []string       `json:"tasks"`
	Runners      []string       `json:"runners"`
	Attributes   map[string]any `json:"attributes"`
	Implies      []string       `json:"implies"`
	EquivalentTo []string       `json:"equivalent_to"`
}

var (
	catalogOnce       sync.Once
	catalogV1Document Document
	catalogV1Error    error
)

func CatalogV1() Document {
	catalogOnce.Do(func() {
		catalogV1Document, catalogV1Error = parseCatalogV1()
	})
	if catalogV1Error != nil {
		panic(catalogV1Error)
	}
	return catalogV1Document
}

func GeneratedCatalogJSONV1() string {
	return generatedCatalogJSONV1
}

func Entries(category string) []Entry {
	document := CatalogV1()
	entries := document.Categories[category]
	return append([]Entry(nil), entries...)
}

func Resolve(category, value string) (Entry, bool) {
	normalized := normalizeID(value)
	for _, entry := range Entries(category) {
		if normalizeID(entry.ID) == normalized {
			return entry, true
		}
		for _, alias := range entry.Aliases {
			if normalizeID(alias) == normalized {
				return entry, true
			}
		}
	}
	return Entry{}, false
}

func AllowedValues(category string) map[string]bool {
	return AllowedValuesMatching(category, nil)
}

func AllowedValuesMatching(category string, include func(Entry) bool) map[string]bool {
	out := map[string]bool{}
	for _, entry := range Entries(category) {
		if !entry.Available || (include != nil && !include(entry)) {
			continue
		}
		out[normalizeID(entry.ID)] = true
		for _, alias := range entry.Aliases {
			out[normalizeID(alias)] = true
		}
	}
	return out
}

func CanonicalIDs(category string, availableOnly bool) []string {
	out := []string{}
	for _, entry := range Entries(category) {
		if availableOnly && !entry.Available {
			continue
		}
		out = append(out, entry.ID)
	}
	sort.Strings(out)
	return out
}

func AttributeBool(entry Entry, name string) bool {
	value, _ := entry.Attributes[name].(bool)
	return value
}

func parseCatalogV1() (Document, error) {
	var document Document
	if err := json.Unmarshal([]byte(generatedCatalogJSONV1), &document); err != nil {
		return Document{}, fmt.Errorf("decode embedded model express catalog: %w", err)
	}
	if document.SchemaVersion != SchemaVersionV1 {
		return Document{}, fmt.Errorf("unsupported model express catalog schema %q", document.SchemaVersion)
	}
	if strings.TrimSpace(document.CatalogVersion) == "" {
		return Document{}, fmt.Errorf("model express catalog version is required")
	}
	if len(document.Categories) == 0 {
		return Document{}, fmt.Errorf("model express catalog categories are required")
	}
	return document, nil
}

func normalizeID(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}
