package metadata

import (
	"bytes"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/xml"
	"fmt"
	"io"
	"path"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"model-express/services/orchestrator/internal/datasets"
)

const (
	parserVersion      = "v1"
	maxPreviewBytes    = 2048
	defaultSourceKind  = datasets.MetadataSourceKindUploadedSidecar
	inferredSourceID   = "inferred_inventory"
	cubParserName      = "cub_sidecars"
	imageFolderParser  = "image_folder_inventory"
	csvParserName      = "csv_manifest"
	vocParserName      = "pascal_voc_xml"
	splitFolderParser  = "split_folder_inventory"
	issueSeverityWarn  = "warning"
	issueSeverityError = "error"
)

type ImportRequest struct {
	StrictMode bool
	SourceKind string
	Sources    []SourceInput
	Inventory  DatasetFileInventory
}

type SourceInput struct {
	RelativePath   string
	StorageURI     string
	ChecksumSHA256 string
	SizeBytes      int64
	DeclaredFormat string
	Content        []byte
}

type DatasetFileInventory struct {
	Files []InventoryFile `json:"files"`
}

type InventoryFile struct {
	RelativePath   string `json:"relative_path"`
	SizeBytes      int64  `json:"size_bytes,omitempty"`
	ChecksumSHA256 string `json:"checksum_sha256,omitempty"`
	Width          int    `json:"width,omitempty"`
	Height         int    `json:"height,omitempty"`
}

type Service struct{}

func NewService() Service {
	return Service{}
}

func (Service) Parse(dataset datasets.Dataset, req ImportRequest) (datasets.DatasetMetadataImport, error) {
	now := time.Now().UTC()
	sourceKind := strings.TrimSpace(req.SourceKind)
	if sourceKind == "" {
		sourceKind = defaultSourceKind
	}
	out := datasets.DatasetMetadataImport{
		DatasetID:             dataset.ID,
		ProjectID:             dataset.ProjectID,
		DatasetChecksumSHA256: dataset.ChecksumSHA256,
		ParserRegistryVersion: datasets.MetadataParserRegistryV1,
		SourceKind:            sourceKind,
		StrictMode:            req.StrictMode,
		Warnings:              []datasets.MetadataIssue{},
		Errors:                []datasets.MetadataIssue{},
		Sources:               []datasets.DatasetMetadataSource{},
		Classes:               []datasets.DatasetClass{},
		ManifestRecords:       []datasets.DatasetManifestRecord{},
		Annotations:           []datasets.DatasetAnnotation{},
		Splits:                []datasets.DatasetSplit{},
		CreatedAt:             now,
		CompletedAt:           &now,
	}

	inventory, inventoryWarnings, inventoryErrors := normalizeInventory(req.Inventory)
	out.Warnings = append(out.Warnings, inventoryWarnings...)
	out.Errors = append(out.Errors, inventoryErrors...)
	pathResolver := newInventoryPathResolver(inventory)

	sourceContents := map[string][]byte{}
	for index, input := range req.Sources {
		source, content, warnings, errors := normalizeSource(dataset.ID, index, input)
		out.Warnings = append(out.Warnings, warnings...)
		out.Errors = append(out.Errors, errors...)
		sourceContents[source.ID] = content
		out.Sources = append(out.Sources, source)
	}

	if len(inventory.Files) > 0 && shouldInferInventoryLabels(out.Sources) {
		parsed, warnings := parseInventory(dataset.ID, inventory)
		out.Warnings = append(out.Warnings, warnings...)
		out.Classes = append(out.Classes, parsed.Classes...)
		out.ManifestRecords = append(out.ManifestRecords, parsed.ManifestRecords...)
		out.Splits = append(out.Splits, parsed.Splits...)
	}

	if hasCUBSidecars(out.Sources) {
		parsed, warnings, errors := parseCUBSidecars(dataset.ID, out.Sources, sourceContents, pathResolver)
		out.Warnings = append(out.Warnings, warnings...)
		out.Errors = append(out.Errors, errors...)
		out.Classes = append(out.Classes, parsed.Classes...)
		out.ManifestRecords = append(out.ManifestRecords, parsed.ManifestRecords...)
		out.Annotations = append(out.Annotations, parsed.Annotations...)
		out.Splits = append(out.Splits, parsed.Splits...)
		markCUBSources(&out)
	}

	for index := range out.Sources {
		source := &out.Sources[index]
		if source.Status != "" && source.Status != datasets.MetadataSourceStatusSkippedUnsupported {
			continue
		}
		content := sourceContents[source.ID]
		switch source.DetectedFormat {
		case datasets.MetadataFormatCSVManifest:
			parsed, warnings, errors := parseCSVManifest(dataset.ID, *source, content, pathResolver)
			source.ParserName = csvParserName
			source.ParserVersion = parserVersion
			if len(errors) > 0 {
				source.Status = datasets.MetadataSourceStatusError
				source.Errors = append(source.Errors, errors...)
			} else {
				source.Status = datasets.MetadataSourceStatusParsed
			}
			source.Warnings = append(source.Warnings, warnings...)
			out.Warnings = append(out.Warnings, warnings...)
			out.Errors = append(out.Errors, errors...)
			out.Classes = append(out.Classes, parsed.Classes...)
			out.ManifestRecords = append(out.ManifestRecords, parsed.ManifestRecords...)
			out.Splits = append(out.Splits, parsed.Splits...)
		case datasets.MetadataFormatPascalVOC:
			parsed, warnings, errors := parsePascalVOC(dataset.ID, *source, content, pathResolver)
			source.ParserName = vocParserName
			source.ParserVersion = parserVersion
			if len(errors) > 0 {
				source.Status = datasets.MetadataSourceStatusError
				source.Errors = append(source.Errors, errors...)
			} else {
				source.Status = datasets.MetadataSourceStatusParsed
			}
			source.Warnings = append(source.Warnings, warnings...)
			out.Warnings = append(out.Warnings, warnings...)
			out.Errors = append(out.Errors, errors...)
			out.Classes = append(out.Classes, parsed.Classes...)
			out.ManifestRecords = append(out.ManifestRecords, parsed.ManifestRecords...)
			out.Annotations = append(out.Annotations, parsed.Annotations...)
		case datasets.MetadataFormatImageFolder, datasets.MetadataFormatSplitFolder, datasets.MetadataFormatCUBSidecars:
			if source.Status == "" {
				source.Status = datasets.MetadataSourceStatusParsed
			}
		default:
			source.DetectedFormat = datasets.MetadataFormatUnsupported
			source.Status = datasets.MetadataSourceStatusSkippedUnsupported
			warning := issue("unsupported_source", "metadata source type is unsupported and was skipped", source.ID)
			source.Warnings = append(source.Warnings, warning)
			out.Warnings = append(out.Warnings, warning)
		}
	}

	out.Classes = normalizeClasses(dataset.ID, out.Classes)
	out.ManifestRecords, out.Warnings = normalizeManifestRecords(dataset.ID, out.ManifestRecords, out.Warnings)
	out.Annotations, out.Warnings = normalizeAnnotations(dataset.ID, out.Annotations, out.Warnings)
	out.Splits = deriveSplits(dataset.ID, out.ManifestRecords, out.Splits)
	if req.StrictMode {
		for _, source := range out.Sources {
			if source.Status == datasets.MetadataSourceStatusSkippedUnsupported && strings.TrimSpace(source.DeclaredFormat) != "" {
				out.Errors = append(out.Errors, errorIssue("declared_format_unsupported", "strict import failed because a declared metadata format is unsupported", source.ID))
			}
		}
	}
	if len(out.Errors) > 0 {
		out.Status = datasets.MetadataImportStatusFailed
		out.Active = false
	} else if hasSourceErrors(out.Sources) {
		out.Status = datasets.MetadataImportStatusPartial
		out.Active = true
	} else {
		out.Status = datasets.MetadataImportStatusSucceeded
		out.Active = true
	}
	out.Summary = BuildSummary(out)
	out.AgentSafeSummary = BuildAgentSafeSummary(out.Summary)
	if len(out.ManifestRecords) == 0 && len(out.Classes) == 0 && len(out.Annotations) == 0 {
		warning := issue("no_metadata_records", "no normalized classes, samples, or annotations were imported", "")
		out.Warnings = append(out.Warnings, warning)
		out.Summary.Warnings = append(out.Summary.Warnings, warning)
		out.AgentSafeSummary.Warnings = append(out.AgentSafeSummary.Warnings, warning)
	}
	return out, nil
}

func BuildSummary(importRecord datasets.DatasetMetadataImport) datasets.DatasetMetadataSummary {
	summary := datasets.DatasetMetadataSummary{
		DatasetID:             importRecord.DatasetID,
		ImportID:              importRecord.ID,
		Status:                importRecord.Status,
		ImportVersion:         importRecord.ImportVersion,
		ParserRegistryVersion: importRecord.ParserRegistryVersion,
		SourceKind:            importRecord.SourceKind,
		SourceFormats:         []string{},
		SplitCounts:           map[string]int{},
		AnnotationCounts:      map[string]int{},
		Warnings:              append([]datasets.MetadataIssue(nil), importRecord.Warnings...),
		Errors:                append([]datasets.MetadataIssue(nil), importRecord.Errors...),
		Capabilities:          map[string]bool{},
	}
	if !importRecord.CreatedAt.IsZero() {
		createdAt := importRecord.CreatedAt
		summary.CreatedAt = &createdAt
	}
	if importRecord.CompletedAt != nil {
		completedAt := *importRecord.CompletedAt
		summary.CompletedAt = &completedAt
	}
	formats := map[string]bool{}
	for _, source := range importRecord.Sources {
		summary.SourceCount++
		switch source.Status {
		case datasets.MetadataSourceStatusParsed:
			summary.ParsedSourceCount++
		case datasets.MetadataSourceStatusSkippedUnsupported:
			summary.UnsupportedSourceCount++
		case datasets.MetadataSourceStatusError:
			summary.ErrorSourceCount++
		}
		format := strings.TrimSpace(source.DetectedFormat)
		if format == "" {
			format = strings.TrimSpace(source.DeclaredFormat)
		}
		if format != "" && format != datasets.MetadataFormatUnknown {
			formats[format] = true
		}
	}
	for format := range formats {
		summary.SourceFormats = append(summary.SourceFormats, format)
	}
	sort.Strings(summary.SourceFormats)
	summary.ClassCount = len(importRecord.Classes)
	summary.SampleCount = len(importRecord.ManifestRecords)
	bboxSamples := map[string]bool{}
	for _, record := range importRecord.ManifestRecords {
		if strings.TrimSpace(record.LabelKey) != "" || strings.TrimSpace(record.LabelName) != "" {
			summary.LabeledSampleCount++
		}
		if split := strings.TrimSpace(record.Split); split != "" {
			summary.SplitCounts[split]++
		}
	}
	summary.MissingLabelCount = summary.SampleCount - summary.LabeledSampleCount
	for _, split := range importRecord.Splits {
		if split.SplitName != "" && split.SampleCount > summary.SplitCounts[split.SplitName] {
			summary.SplitCounts[split.SplitName] = split.SampleCount
		}
	}
	for _, annotation := range importRecord.Annotations {
		annotationType := strings.TrimSpace(annotation.AnnotationType)
		if annotationType == "" {
			annotationType = "unknown"
		}
		summary.AnnotationCounts[annotationType]++
		if annotationType == datasets.MetadataAnnotationBBox {
			summary.BBoxAnnotationCount++
			if annotation.SampleKey != "" {
				bboxSamples[annotation.SampleKey] = true
			}
		}
	}
	summary.BBoxSampleCount = len(bboxSamples)
	if summary.SampleCount > 0 {
		summary.BBoxCoverageRatio = float64(summary.BBoxSampleCount) / float64(summary.SampleCount)
	}
	summary.OfficialSplitAvailable = hasOfficialSplits(summary.SplitCounts)
	summary.Capabilities["classification_labels"] = summary.LabeledSampleCount > 0 || summary.ClassCount > 0
	summary.Capabilities["official_splits"] = summary.OfficialSplitAvailable
	summary.Capabilities["bbox_annotations"] = summary.BBoxAnnotationCount > 0
	summary.Capabilities["attribute_annotations"] = summary.AnnotationCounts[datasets.MetadataAnnotationAttribute] > 0
	summary.Capabilities["keypoint_annotations"] = summary.AnnotationCounts[datasets.MetadataAnnotationKeypoint] > 0
	return summary
}

func BuildAgentSafeSummary(summary datasets.DatasetMetadataSummary) datasets.AgentSafeDatasetMetadataSummary {
	return datasets.AgentSafeDatasetMetadataSummary{
		DatasetID:              summary.DatasetID,
		ImportID:               summary.ImportID,
		Status:                 summary.Status,
		ImportVersion:          summary.ImportVersion,
		SourceKind:             summary.SourceKind,
		SourceCount:            summary.SourceCount,
		ParsedSourceCount:      summary.ParsedSourceCount,
		UnsupportedSourceCount: summary.UnsupportedSourceCount,
		ErrorSourceCount:       summary.ErrorSourceCount,
		SourceFormats:          append([]string(nil), summary.SourceFormats...),
		ClassCount:             summary.ClassCount,
		SampleCount:            summary.SampleCount,
		LabeledSampleCount:     summary.LabeledSampleCount,
		MissingLabelCount:      summary.MissingLabelCount,
		SplitCounts:            copyIntMap(summary.SplitCounts),
		OfficialSplitAvailable: summary.OfficialSplitAvailable,
		AnnotationCounts:       copyIntMap(summary.AnnotationCounts),
		BBoxAnnotationCount:    summary.BBoxAnnotationCount,
		BBoxSampleCount:        summary.BBoxSampleCount,
		BBoxCoverageRatio:      summary.BBoxCoverageRatio,
		Warnings:               safeIssues(summary.Warnings),
		Errors:                 safeIssues(summary.Errors),
		Capabilities:           copyBoolMap(summary.Capabilities),
	}
}

type parsedRecords struct {
	Classes         []datasets.DatasetClass
	ManifestRecords []datasets.DatasetManifestRecord
	Annotations     []datasets.DatasetAnnotation
	Splits          []datasets.DatasetSplit
}

func normalizeSource(datasetID string, index int, input SourceInput) (datasets.DatasetMetadataSource, []byte, []datasets.MetadataIssue, []datasets.MetadataIssue) {
	warnings := []datasets.MetadataIssue{}
	errors := []datasets.MetadataIssue{}
	relativePath, err := safeRelativePath(input.RelativePath)
	if err != nil {
		errors = append(errors, errorIssue("unsafe_path", "metadata source path was rejected as unsafe", ""))
		relativePath = fmt.Sprintf("source_%d", index+1)
	}
	content := append([]byte(nil), input.Content...)
	checksum := strings.TrimSpace(input.ChecksumSHA256)
	if checksum == "" && len(content) > 0 {
		sum := sha256.Sum256(content)
		checksum = hex.EncodeToString(sum[:])
	}
	sizeBytes := input.SizeBytes
	if sizeBytes == 0 && len(content) > 0 {
		sizeBytes = int64(len(content))
	}
	source := datasets.DatasetMetadataSource{
		ID:             fmt.Sprintf("source_%d", index+1),
		DatasetID:      datasetID,
		RelativePath:   relativePath,
		StorageURI:     strings.TrimSpace(input.StorageURI),
		ChecksumSHA256: checksum,
		SizeBytes:      sizeBytes,
		DeclaredFormat: normalizeFormat(input.DeclaredFormat),
		DetectedFormat: datasets.MetadataFormatUnknown,
		Status:         "",
		Warnings:       []datasets.MetadataIssue{},
		Errors:         []datasets.MetadataIssue{},
		RawPreview:     rawPreview(content),
	}
	source.DetectedFormat = detectFormat(source.DeclaredFormat, relativePath, content)
	return source, content, warnings, errors
}

func normalizeInventory(inventory DatasetFileInventory) (DatasetFileInventory, []datasets.MetadataIssue, []datasets.MetadataIssue) {
	out := DatasetFileInventory{Files: []InventoryFile{}}
	warnings := []datasets.MetadataIssue{}
	errors := []datasets.MetadataIssue{}
	seen := map[string]bool{}
	for _, file := range inventory.Files {
		relativePath, err := safeRelativePath(file.RelativePath)
		if err != nil {
			errors = append(errors, datasets.MetadataIssue{Severity: issueSeverityError, Code: "unsafe_inventory_path", Message: "inventory path was rejected as unsafe"})
			continue
		}
		if seen[relativePath] {
			warnings = append(warnings, issue("duplicate_inventory_path", "duplicate inventory path was ignored", ""))
			continue
		}
		seen[relativePath] = true
		file.RelativePath = relativePath
		out.Files = append(out.Files, file)
	}
	return out, warnings, errors
}

type inventoryPathResolver struct {
	exact     map[string]string
	suffix    map[string]string
	ambiguous map[string]bool
}

func newInventoryPathResolver(inventory DatasetFileInventory) inventoryPathResolver {
	resolver := inventoryPathResolver{
		exact:     map[string]string{},
		suffix:    map[string]string{},
		ambiguous: map[string]bool{},
	}
	for _, file := range inventory.Files {
		relativePath := file.RelativePath
		if relativePath == "" {
			continue
		}
		resolver.addExact(relativePath, relativePath)
		resolver.addExact(strings.ToLower(relativePath), relativePath)
		resolver.addLookup(strings.ToLower(relativePath), relativePath)
		parts := strings.Split(relativePath, "/")
		for index := 1; index < len(parts); index++ {
			suffix := strings.Join(parts[index:], "/")
			resolver.addLookup(suffix, relativePath)
			resolver.addLookup(strings.ToLower(suffix), relativePath)
		}
	}
	return resolver
}

func (resolver inventoryPathResolver) Resolve(relativePath string) string {
	if relativePath == "" {
		return relativePath
	}
	if resolved := resolver.exact[relativePath]; resolved != "" {
		return resolved
	}
	if resolved := resolver.exact[strings.ToLower(relativePath)]; resolved != "" {
		return resolved
	}
	for _, folder := range imageRootFolders() {
		prefixed := path.Join(folder, relativePath)
		if resolved := resolver.exact[prefixed]; resolved != "" {
			return resolved
		}
		if resolved := resolver.exact[strings.ToLower(prefixed)]; resolved != "" {
			return resolved
		}
	}
	if resolved := resolver.resolveLookup(relativePath); resolved != "" {
		return resolved
	}
	if resolved := resolver.resolveLookup(strings.ToLower(relativePath)); resolved != "" {
		return resolved
	}
	return relativePath
}

func (resolver inventoryPathResolver) addExact(key string, relativePath string) {
	if key == "" || resolver.ambiguous[key] {
		return
	}
	if existing := resolver.exact[key]; existing != "" && existing != relativePath {
		delete(resolver.exact, key)
		resolver.ambiguous[key] = true
		return
	}
	resolver.exact[key] = relativePath
}

func (resolver inventoryPathResolver) addLookup(key string, relativePath string) {
	if key == "" || resolver.ambiguous[key] {
		return
	}
	if existing := resolver.suffix[key]; existing != "" && existing != relativePath {
		delete(resolver.suffix, key)
		resolver.ambiguous[key] = true
		return
	}
	resolver.suffix[key] = relativePath
}

func (resolver inventoryPathResolver) resolveLookup(key string) string {
	if resolver.ambiguous[key] {
		return ""
	}
	return resolver.suffix[key]
}

func inventoryLabelParts(relativePath string, wrapperPrefix string) []string {
	parts := strings.Split(relativePath, "/")
	if wrapperPrefix != "" && len(parts) >= 2 && strings.EqualFold(parts[0], wrapperPrefix) {
		parts = parts[1:]
	}
	if len(parts) >= 3 && isImageRootFolder(parts[0]) {
		return parts[1:]
	}
	return parts
}

func isImageRootFolder(value string) bool {
	normalized := strings.ToLower(strings.TrimSpace(value))
	for _, folder := range imageRootFolders() {
		if normalized == strings.ToLower(folder) {
			return true
		}
	}
	return false
}

func imageRootFolders() []string {
	return []string{"images", "image", "imgs", "img", "JPEGImages", "data"}
}

func shouldInferInventoryLabels(sources []datasets.DatasetMetadataSource) bool {
	if hasCUBSidecars(sources) {
		return false
	}
	for _, source := range sources {
		if source.DetectedFormat == datasets.MetadataFormatCSVManifest {
			return false
		}
	}
	return true
}

func parseInventory(datasetID string, inventory DatasetFileInventory) (parsedRecords, []datasets.MetadataIssue) {
	records := parsedRecords{}
	warnings := []datasets.MetadataIssue{}
	imageFiles := []InventoryFile{}
	for _, file := range inventory.Files {
		if !isImagePath(file.RelativePath) {
			continue
		}
		imageFiles = append(imageFiles, file)
	}
	wrapperPrefix := inventoryWrapperPrefix(imageFiles)
	for _, file := range imageFiles {
		parts := inventoryLabelParts(file.RelativePath, wrapperPrefix)
		if len(parts) < 2 {
			warnings = append(warnings, issue("unlabeled_inventory_image", "image inventory item could not be mapped to a class folder", inferredSourceID))
			continue
		}
		split := ""
		className := parts[0]
		parserName := imageFolderParser
		if normalizedSplit := normalizeSplit(parts[0]); normalizedSplit != "" && len(parts) >= 3 {
			split = normalizedSplit
			className = parts[1]
			parserName = splitFolderParser
		}
		classKey := classKey(className)
		records.Classes = append(records.Classes, datasets.DatasetClass{
			DatasetID: datasetID,
			ClassKey:  classKey,
			ClassName: className,
			SourceID:  inferredSourceID,
			Metadata:  map[string]any{"parser": parserName},
		})
		records.ManifestRecords = append(records.ManifestRecords, datasets.DatasetManifestRecord{
			DatasetID:      datasetID,
			SampleKey:      sampleKey(file.RelativePath),
			MediaType:      datasets.MetadataMediaTypeImage,
			RelativePath:   file.RelativePath,
			LabelKey:       classKey,
			LabelName:      className,
			Split:          split,
			Width:          file.Width,
			Height:         file.Height,
			ChecksumSHA256: file.ChecksumSHA256,
			SourceID:       inferredSourceID,
			Metadata:       map[string]any{"size_bytes": file.SizeBytes, "parser": parserName},
		})
	}
	return records, warnings
}

func inventoryWrapperPrefix(files []InventoryFile) string {
	prefix := ""
	for _, file := range files {
		parts := strings.Split(file.RelativePath, "/")
		if len(parts) < 3 {
			return ""
		}
		first := strings.TrimSpace(parts[0])
		if first == "" || isImageRootFolder(first) || normalizeSplit(first) != "" {
			return ""
		}
		if prefix == "" {
			prefix = first
			continue
		}
		if !strings.EqualFold(prefix, first) {
			return ""
		}
	}
	return prefix
}

func parseCSVManifest(datasetID string, source datasets.DatasetMetadataSource, content []byte, resolver inventoryPathResolver) (parsedRecords, []datasets.MetadataIssue, []datasets.MetadataIssue) {
	records := parsedRecords{}
	warnings := []datasets.MetadataIssue{}
	errors := []datasets.MetadataIssue{}
	if len(bytes.TrimSpace(content)) == 0 {
		errors = append(errors, errorIssue("empty_csv_manifest", "CSV manifest is empty", source.ID))
		return records, warnings, errors
	}
	reader := csv.NewReader(bytes.NewReader(content))
	reader.FieldsPerRecord = -1
	rows, err := reader.ReadAll()
	if err != nil {
		errors = append(errors, errorIssue("malformed_csv_manifest", "CSV manifest could not be parsed", source.ID))
		return records, warnings, errors
	}
	if len(rows) < 2 {
		errors = append(errors, errorIssue("csv_manifest_without_rows", "CSV manifest has no data rows", source.ID))
		return records, warnings, errors
	}
	header := normalizedHeader(rows[0])
	pathIndex := firstHeaderIndex(header, "relative_path", "filepaths", "filepath", "file_path", "path", "image", "image_path", "filename", "file")
	labelIndex := firstHeaderIndex(header, "labels", "label", "label_name", "class", "class_name", "category", "target")
	splitIndex := firstHeaderIndex(header, "data_set", "dataset", "set", "split", "subset", "partition")
	widthIndex := firstHeaderIndex(header, "width", "image_width")
	heightIndex := firstHeaderIndex(header, "height", "image_height")
	if pathIndex < 0 {
		errors = append(errors, errorIssue("csv_manifest_missing_path", "CSV manifest requires a path-like column", source.ID))
		return records, warnings, errors
	}
	for rowIndex, row := range rows[1:] {
		relativePath := cell(row, pathIndex)
		relativePath, err = safeRelativePath(relativePath)
		if err != nil {
			warnings = append(warnings, issue("csv_manifest_unsafe_path", "CSV row path was skipped because it is unsafe", source.ID))
			continue
		}
		relativePath = resolver.Resolve(relativePath)
		labelName := cell(row, labelIndex)
		labelKey := classKey(labelName)
		split := normalizeSplit(cell(row, splitIndex))
		if labelName == "" {
			warnings = append(warnings, issue("csv_manifest_missing_label", "CSV row is missing a label", source.ID))
		} else {
			records.Classes = append(records.Classes, datasets.DatasetClass{
				DatasetID: datasetID,
				ClassKey:  labelKey,
				ClassName: labelName,
				SourceID:  source.ID,
			})
		}
		records.ManifestRecords = append(records.ManifestRecords, datasets.DatasetManifestRecord{
			DatasetID:    datasetID,
			SampleKey:    sampleKey(relativePath),
			MediaType:    mediaTypeForPath(relativePath),
			RelativePath: relativePath,
			LabelKey:     labelKey,
			LabelName:    labelName,
			Split:        split,
			Width:        parsePositiveInt(cell(row, widthIndex)),
			Height:       parsePositiveInt(cell(row, heightIndex)),
			SourceID:     source.ID,
			Metadata:     map[string]any{"csv_row": rowIndex + 2},
		})
	}
	return records, warnings, errors
}

type vocAnnotation struct {
	Filename string      `xml:"filename"`
	Objects  []vocObject `xml:"object"`
}

type vocObject struct {
	Name string  `xml:"name"`
	BBox vocBBox `xml:"bndbox"`
}

type vocBBox struct {
	XMin float64 `xml:"xmin"`
	YMin float64 `xml:"ymin"`
	XMax float64 `xml:"xmax"`
	YMax float64 `xml:"ymax"`
}

func parsePascalVOC(datasetID string, source datasets.DatasetMetadataSource, content []byte, resolver inventoryPathResolver) (parsedRecords, []datasets.MetadataIssue, []datasets.MetadataIssue) {
	records := parsedRecords{}
	warnings := []datasets.MetadataIssue{}
	errors := []datasets.MetadataIssue{}
	var doc vocAnnotation
	if err := xml.Unmarshal(content, &doc); err != nil {
		errors = append(errors, errorIssue("malformed_pascal_voc", "Pascal VOC XML could not be parsed", source.ID))
		return records, warnings, errors
	}
	filename, err := safeRelativePath(doc.Filename)
	if err != nil || filename == "" {
		errors = append(errors, errorIssue("pascal_voc_missing_filename", "Pascal VOC XML is missing a safe filename", source.ID))
		return records, warnings, errors
	}
	filename = resolver.Resolve(filename)
	sampleKey := sampleKey(filename)
	labels := map[string]string{}
	for _, object := range doc.Objects {
		labelName := strings.TrimSpace(object.Name)
		labelKey := classKey(labelName)
		if labelName == "" {
			warnings = append(warnings, issue("pascal_voc_missing_object_label", "VOC object without a label was skipped", source.ID))
			continue
		}
		if object.BBox.XMax <= object.BBox.XMin || object.BBox.YMax <= object.BBox.YMin {
			warnings = append(warnings, issue("pascal_voc_invalid_bbox", "VOC object has a non-positive-area bounding box and was skipped", source.ID))
			continue
		}
		labels[labelKey] = labelName
		records.Classes = append(records.Classes, datasets.DatasetClass{
			DatasetID: datasetID,
			ClassKey:  labelKey,
			ClassName: labelName,
			SourceID:  source.ID,
		})
		records.Annotations = append(records.Annotations, datasets.DatasetAnnotation{
			DatasetID:      datasetID,
			SampleKey:      sampleKey,
			AnnotationType: datasets.MetadataAnnotationBBox,
			LabelKey:       labelKey,
			LabelName:      labelName,
			BBox: map[string]any{
				"xmin": object.BBox.XMin,
				"ymin": object.BBox.YMin,
				"xmax": object.BBox.XMax,
				"ymax": object.BBox.YMax,
			},
			SourceID: source.ID,
		})
	}
	record := datasets.DatasetManifestRecord{
		DatasetID:    datasetID,
		SampleKey:    sampleKey,
		MediaType:    datasets.MetadataMediaTypeImage,
		RelativePath: filename,
		SourceID:     source.ID,
		Metadata:     map[string]any{"parser": vocParserName},
	}
	if len(labels) == 1 {
		for key, name := range labels {
			record.LabelKey = key
			record.LabelName = name
		}
	} else if len(labels) > 1 {
		warnings = append(warnings, issue("pascal_voc_multi_object_label", "VOC XML has multiple object labels, so no classification label was inferred", source.ID))
	}
	records.ManifestRecords = append(records.ManifestRecords, record)
	return records, warnings, errors
}

func parseCUBSidecars(datasetID string, sources []datasets.DatasetMetadataSource, contents map[string][]byte, resolver inventoryPathResolver) (parsedRecords, []datasets.MetadataIssue, []datasets.MetadataIssue) {
	records := parsedRecords{}
	warnings := []datasets.MetadataIssue{}
	errors := []datasets.MetadataIssue{}
	sourceByBase := map[string]datasets.DatasetMetadataSource{}
	for _, source := range sources {
		base := strings.ToLower(path.Base(source.RelativePath))
		sourceByBase[base] = source
	}
	imagesSource, okImages := sourceByBase["images.txt"]
	labelsSource, okLabels := sourceByBase["image_class_labels.txt"]
	classesSource, okClasses := sourceByBase["classes.txt"]
	if !okImages || !okLabels || !okClasses {
		return records, warnings, errors
	}
	idToPath := parseCUBTwoColumnText(contents[imagesSource.ID])
	idToClassID := parseCUBTwoColumnText(contents[labelsSource.ID])
	classIDToName := parseCUBTwoColumnText(contents[classesSource.ID])
	idToSplit := map[string]string{}
	if splitSource, ok := sourceByBase["train_test_split.txt"]; ok {
		for id, value := range parseCUBTwoColumnText(contents[splitSource.ID]) {
			if value == "1" {
				idToSplit[id] = "train"
			} else if value == "0" {
				idToSplit[id] = "test"
			}
		}
	}
	idToBBox := map[string]map[string]any{}
	if bboxSource, ok := sourceByBase["bounding_boxes.txt"]; ok {
		idToBBox = parseCUBBBoxText(contents[bboxSource.ID])
	}
	partIDToName := map[string]string{}
	if partsSource, ok := sourceByBase["parts.txt"]; ok {
		partIDToName = parseCUBTwoColumnText(contents[partsSource.ID])
	}
	partLocs := map[string][]cubPartLocation{}
	partLocsSource, hasPartLocs := sourceByBase["part_locs.txt"]
	if hasPartLocs {
		partLocs = parseCUBPartLocsText(contents[partLocsSource.ID])
	}
	if len(idToPath) == 0 || len(idToClassID) == 0 || len(classIDToName) == 0 {
		errors = append(errors, errorIssue("malformed_cub_sidecars", "CUB sidecar files are missing required rows", imagesSource.ID))
		return records, warnings, errors
	}
	for imageID, relativePath := range idToPath {
		relativePath, err := safeRelativePath(relativePath)
		if err != nil {
			warnings = append(warnings, issue("cub_unsafe_image_path", "CUB image path was skipped because it is unsafe", imagesSource.ID))
			continue
		}
		relativePath = resolver.Resolve(relativePath)
		classID := idToClassID[imageID]
		className := classIDToName[classID]
		if className == "" {
			warnings = append(warnings, issue("cub_orphan_class_label", "CUB image label references an unknown class", labelsSource.ID))
		}
		ckey := classKey(className)
		classIndex := parsePositiveInt(classID)
		if className != "" {
			records.Classes = append(records.Classes, datasets.DatasetClass{
				DatasetID:  datasetID,
				ClassKey:   ckey,
				ClassName:  className,
				ClassIndex: &classIndex,
				SourceID:   classesSource.ID,
				Metadata:   map[string]any{"cub_class_id": classID},
			})
		}
		records.ManifestRecords = append(records.ManifestRecords, datasets.DatasetManifestRecord{
			DatasetID:    datasetID,
			SampleKey:    sampleKey(relativePath),
			MediaType:    datasets.MetadataMediaTypeImage,
			RelativePath: relativePath,
			LabelKey:     ckey,
			LabelName:    className,
			Split:        idToSplit[imageID],
			SourceID:     imagesSource.ID,
			Metadata:     map[string]any{"cub_image_id": imageID, "cub_class_id": classID},
		})
		if bbox, ok := idToBBox[imageID]; ok {
			records.Annotations = append(records.Annotations, datasets.DatasetAnnotation{
				DatasetID:      datasetID,
				SampleKey:      sampleKey(relativePath),
				AnnotationType: datasets.MetadataAnnotationBBox,
				LabelKey:       ckey,
				LabelName:      className,
				BBox:           bbox,
				SourceID:       "bounding_boxes.txt",
			})
		}
		for _, partLoc := range partLocs[imageID] {
			partName := partIDToName[partLoc.PartID]
			metadata := map[string]any{
				"parser":       cubParserName,
				"cub_image_id": imageID,
				"cub_part_id":  partLoc.PartID,
				"x":            partLoc.X,
				"y":            partLoc.Y,
				"visible":      partLoc.Visible,
			}
			if partName != "" {
				metadata["part_name"] = partName
			}
			records.Annotations = append(records.Annotations, datasets.DatasetAnnotation{
				DatasetID:      datasetID,
				SampleKey:      sampleKey(relativePath),
				AnnotationType: datasets.MetadataAnnotationKeypoint,
				LabelKey:       ckey,
				LabelName:      className,
				SourceID:       partLocsSource.ID,
				Metadata:       metadata,
			})
		}
	}
	return records, warnings, errors
}

func markCUBSources(importRecord *datasets.DatasetMetadataImport) {
	cubFiles := map[string]bool{
		"images.txt":             true,
		"image_class_labels.txt": true,
		"train_test_split.txt":   true,
		"classes.txt":            true,
		"bounding_boxes.txt":     true,
		"parts.txt":              true,
		"part_locs.txt":          true,
	}
	for index := range importRecord.Sources {
		if cubFiles[strings.ToLower(path.Base(importRecord.Sources[index].RelativePath))] {
			importRecord.Sources[index].DetectedFormat = datasets.MetadataFormatCUBSidecars
			importRecord.Sources[index].ParserName = cubParserName
			importRecord.Sources[index].ParserVersion = parserVersion
			importRecord.Sources[index].Status = datasets.MetadataSourceStatusParsed
		}
	}
}

func normalizeClasses(datasetID string, classes []datasets.DatasetClass) []datasets.DatasetClass {
	byKey := map[string]datasets.DatasetClass{}
	for _, class := range classes {
		class.DatasetID = datasetID
		class.ClassName = strings.TrimSpace(class.ClassName)
		class.ClassKey = classKey(firstNonEmpty(class.ClassKey, class.ClassName))
		if class.ClassKey == "" {
			continue
		}
		if class.Metadata == nil {
			class.Metadata = map[string]any{}
		}
		if existing, ok := byKey[class.ClassKey]; ok {
			if existing.ClassName == "" && class.ClassName != "" {
				existing.ClassName = class.ClassName
			}
			if existing.ClassIndex == nil && class.ClassIndex != nil {
				existing.ClassIndex = class.ClassIndex
			}
			byKey[class.ClassKey] = existing
			continue
		}
		byKey[class.ClassKey] = class
	}
	out := make([]datasets.DatasetClass, 0, len(byKey))
	for _, class := range byKey {
		out = append(out, class)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].ClassIndex != nil && out[j].ClassIndex != nil && *out[i].ClassIndex != *out[j].ClassIndex {
			return *out[i].ClassIndex < *out[j].ClassIndex
		}
		return out[i].ClassKey < out[j].ClassKey
	})
	return out
}

func normalizeManifestRecords(datasetID string, records []datasets.DatasetManifestRecord, warnings []datasets.MetadataIssue) ([]datasets.DatasetManifestRecord, []datasets.MetadataIssue) {
	byKey := map[string]datasets.DatasetManifestRecord{}
	for _, record := range records {
		record.DatasetID = datasetID
		record.SampleKey = firstNonEmpty(record.SampleKey, sampleKey(record.RelativePath))
		if record.SampleKey == "" {
			warnings = append(warnings, issue("manifest_record_missing_sample_key", "manifest row without sample key was skipped", record.SourceID))
			continue
		}
		if record.MediaType == "" {
			record.MediaType = mediaTypeForPath(record.RelativePath)
		}
		record.Split = normalizeSplit(record.Split)
		record.LabelKey = classKey(firstNonEmpty(record.LabelKey, record.LabelName))
		if record.Metadata == nil {
			record.Metadata = map[string]any{}
		}
		if existing, ok := byKey[record.SampleKey]; ok {
			if existing.LabelKey != "" && record.LabelKey != "" && existing.LabelKey != record.LabelKey {
				warnings = append(warnings, issue("conflicting_sample_label", "sample has conflicting labels; first label was kept", record.SourceID))
				continue
			}
			if existing.LabelKey == "" && record.LabelKey != "" {
				existing.LabelKey = record.LabelKey
				existing.LabelName = record.LabelName
			}
			if existing.Split == "" && record.Split != "" {
				existing.Split = record.Split
			}
			if existing.RelativePath == "" && record.RelativePath != "" {
				existing.RelativePath = record.RelativePath
			}
			byKey[record.SampleKey] = existing
			continue
		}
		byKey[record.SampleKey] = record
	}
	out := make([]datasets.DatasetManifestRecord, 0, len(byKey))
	for _, record := range byKey {
		out = append(out, record)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].SampleKey < out[j].SampleKey })
	return out, warnings
}

func normalizeAnnotations(datasetID string, annotations []datasets.DatasetAnnotation, warnings []datasets.MetadataIssue) ([]datasets.DatasetAnnotation, []datasets.MetadataIssue) {
	out := []datasets.DatasetAnnotation{}
	seen := map[string]bool{}
	for _, annotation := range annotations {
		annotation.DatasetID = datasetID
		annotation.SampleKey = strings.TrimSpace(annotation.SampleKey)
		if annotation.SampleKey == "" {
			warnings = append(warnings, issue("annotation_missing_sample_key", "annotation without sample key was skipped", annotation.SourceID))
			continue
		}
		annotation.AnnotationType = strings.TrimSpace(annotation.AnnotationType)
		if annotation.AnnotationType == "" {
			annotation.AnnotationType = "unknown"
		}
		annotation.LabelKey = classKey(firstNonEmpty(annotation.LabelKey, annotation.LabelName))
		if annotation.Metadata == nil {
			annotation.Metadata = map[string]any{}
		}
		key := annotation.SampleKey + "\x00" + annotation.AnnotationType + "\x00" + annotation.LabelKey + "\x00" + fmt.Sprint(annotation.BBox) + "\x00" + fmt.Sprint(annotation.Metadata)
		if seen[key] {
			continue
		}
		seen[key] = true
		out = append(out, annotation)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].SampleKey < out[j].SampleKey })
	return out, warnings
}

func deriveSplits(datasetID string, records []datasets.DatasetManifestRecord, explicit []datasets.DatasetSplit) []datasets.DatasetSplit {
	counts := map[string]int{}
	for _, split := range explicit {
		name := normalizeSplit(split.SplitName)
		if name != "" && split.SampleCount > counts[name] {
			counts[name] = split.SampleCount
		}
	}
	for _, record := range records {
		if split := normalizeSplit(record.Split); split != "" {
			counts[split]++
		}
	}
	out := make([]datasets.DatasetSplit, 0, len(counts))
	for split, count := range counts {
		out = append(out, datasets.DatasetSplit{
			DatasetID:   datasetID,
			SplitName:   split,
			SampleCount: count,
			Metadata:    map[string]any{},
		})
	}
	sort.Slice(out, func(i, j int) bool { return splitRank(out[i].SplitName) < splitRank(out[j].SplitName) })
	return out
}

func detectFormat(declaredFormat string, relativePath string, content []byte) string {
	if declaredFormat != "" && declaredFormat != datasets.MetadataFormatUnknown {
		return declaredFormat
	}
	base := strings.ToLower(path.Base(relativePath))
	ext := strings.ToLower(path.Ext(relativePath))
	switch {
	case base == "images.txt" || base == "image_class_labels.txt" || base == "train_test_split.txt" || base == "classes.txt" || base == "bounding_boxes.txt" || base == "parts.txt" || base == "part_locs.txt":
		return datasets.MetadataFormatCUBSidecars
	case ext == ".csv":
		return datasets.MetadataFormatCSVManifest
	case ext == ".xml" && bytes.Contains(bytes.ToLower(content), []byte("<annotation")):
		return datasets.MetadataFormatPascalVOC
	case ext == ".xml":
		return datasets.MetadataFormatPascalVOC
	default:
		return datasets.MetadataFormatUnsupported
	}
}

func normalizeFormat(value string) string {
	normalized := strings.ToLower(strings.TrimSpace(value))
	normalized = strings.ReplaceAll(normalized, "-", "_")
	normalized = strings.ReplaceAll(normalized, " ", "_")
	switch normalized {
	case "", "auto":
		return ""
	case "csv", "labels_csv", "manifest_csv":
		return datasets.MetadataFormatCSVManifest
	case "voc", "pascal", "pascal_xml", "pascal_voc_xml":
		return datasets.MetadataFormatPascalVOC
	case "cub", "cub_200", "cub_200_2011":
		return datasets.MetadataFormatCUBSidecars
	case "imagefolder":
		return datasets.MetadataFormatImageFolder
	case "splitfolder":
		return datasets.MetadataFormatSplitFolder
	default:
		return normalized
	}
}

func safeRelativePath(value string) (string, error) {
	value = strings.TrimSpace(strings.ReplaceAll(value, "\\", "/"))
	if value == "" {
		return "", fmt.Errorf("empty path")
	}
	if strings.Contains(value, "\x00") || strings.Contains(value, ":") || strings.HasPrefix(value, "/") {
		return "", fmt.Errorf("unsafe path")
	}
	cleaned := path.Clean(value)
	if cleaned == "." || strings.HasPrefix(cleaned, "../") || cleaned == ".." {
		return "", fmt.Errorf("unsafe path")
	}
	return cleaned, nil
}

func rawPreview(content []byte) map[string]any {
	if len(content) == 0 {
		return map[string]any{}
	}
	limit := min(len(content), maxPreviewBytes)
	return map[string]any{
		"size_bytes":      len(content),
		"preview":         string(content[:limit]),
		"truncated":       len(content) > limit,
		"preview_charset": "utf-8",
	}
}

func parseCUBTwoColumnText(content []byte) map[string]string {
	out := map[string]string{}
	for _, line := range strings.Split(string(content), "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) < 2 {
			continue
		}
		out[fields[0]] = strings.Join(fields[1:], " ")
	}
	return out
}

func parseCUBBBoxText(content []byte) map[string]map[string]any {
	out := map[string]map[string]any{}
	for _, line := range strings.Split(string(content), "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) < 5 {
			continue
		}
		x, _ := strconv.ParseFloat(fields[1], 64)
		y, _ := strconv.ParseFloat(fields[2], 64)
		w, _ := strconv.ParseFloat(fields[3], 64)
		h, _ := strconv.ParseFloat(fields[4], 64)
		if w <= 0 || h <= 0 {
			continue
		}
		out[fields[0]] = map[string]any{"xmin": x, "ymin": y, "xmax": x + w, "ymax": y + h}
	}
	return out
}

type cubPartLocation struct {
	PartID  string
	X       float64
	Y       float64
	Visible bool
}

func parseCUBPartLocsText(content []byte) map[string][]cubPartLocation {
	out := map[string][]cubPartLocation{}
	for _, line := range strings.Split(string(content), "\n") {
		fields := strings.Fields(strings.TrimSpace(line))
		if len(fields) < 5 {
			continue
		}
		x, xErr := strconv.ParseFloat(fields[2], 64)
		y, yErr := strconv.ParseFloat(fields[3], 64)
		if xErr != nil || yErr != nil {
			continue
		}
		visible := fields[4] == "1" || strings.EqualFold(fields[4], "true")
		out[fields[0]] = append(out[fields[0]], cubPartLocation{
			PartID:  fields[1],
			X:       x,
			Y:       y,
			Visible: visible,
		})
	}
	return out
}

func hasCUBSidecars(sources []datasets.DatasetMetadataSource) bool {
	seen := map[string]bool{}
	for _, source := range sources {
		seen[strings.ToLower(path.Base(source.RelativePath))] = true
	}
	return seen["images.txt"] && seen["image_class_labels.txt"] && seen["classes.txt"]
}

func normalizedHeader(header []string) []string {
	out := make([]string, len(header))
	for i, value := range header {
		out[i] = strings.ToLower(strings.TrimSpace(value))
		out[i] = strings.ReplaceAll(out[i], "-", "_")
		out[i] = strings.ReplaceAll(out[i], " ", "_")
	}
	return out
}

func firstHeaderIndex(header []string, names ...string) int {
	for _, name := range names {
		for index, value := range header {
			if value == name {
				return index
			}
		}
	}
	return -1
}

func cell(row []string, index int) string {
	if index < 0 || index >= len(row) {
		return ""
	}
	return strings.TrimSpace(row[index])
}

func classKey(value string) string {
	value = strings.TrimSpace(value)
	value = strings.TrimSuffix(value, "/")
	if value == "" {
		return ""
	}
	var builder strings.Builder
	lastUnderscore := false
	for _, r := range strings.ToLower(value) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			builder.WriteRune(r)
			lastUnderscore = false
			continue
		}
		if !lastUnderscore {
			builder.WriteRune('_')
			lastUnderscore = true
		}
	}
	return strings.Trim(builder.String(), "_")
}

func sampleKey(relativePath string) string {
	relativePath = strings.TrimSpace(strings.ReplaceAll(relativePath, "\\", "/"))
	return strings.TrimPrefix(path.Clean(relativePath), "./")
}

func normalizeSplit(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "train", "training":
		return "train"
	case "val", "valid", "validation", "dev":
		return "val"
	case "test", "testing", "holdout", "heldout":
		return "test"
	default:
		return ""
	}
}

func splitRank(split string) int {
	switch split {
	case "train":
		return 0
	case "val":
		return 1
	case "test":
		return 2
	default:
		return 9
	}
}

func mediaTypeForPath(relativePath string) string {
	ext := strings.ToLower(path.Ext(relativePath))
	switch ext {
	case ".mp4", ".mov", ".avi", ".mkv", ".webm":
		return datasets.MetadataMediaTypeVideo
	default:
		return datasets.MetadataMediaTypeImage
	}
}

func isImagePath(relativePath string) bool {
	switch strings.ToLower(path.Ext(relativePath)) {
	case ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff":
		return true
	default:
		return false
	}
}

func parsePositiveInt(value string) int {
	parsed, err := strconv.Atoi(strings.TrimSpace(value))
	if err != nil || parsed < 0 {
		return 0
	}
	return parsed
}

func hasSourceErrors(sources []datasets.DatasetMetadataSource) bool {
	for _, source := range sources {
		if source.Status == datasets.MetadataSourceStatusError {
			return true
		}
	}
	return false
}

func hasOfficialSplits(counts map[string]int) bool {
	return counts["train"] > 0 && (counts["val"] > 0 || counts["test"] > 0)
}

func issue(code string, message string, sourceID string) datasets.MetadataIssue {
	return datasets.MetadataIssue{Severity: issueSeverityWarn, Code: code, Message: message, SourceID: sourceID}
}

func errorIssue(code string, message string, sourceID string) datasets.MetadataIssue {
	return datasets.MetadataIssue{Severity: issueSeverityError, Code: code, Message: message, SourceID: sourceID}
}

func safeIssues(issues []datasets.MetadataIssue) []datasets.MetadataIssue {
	out := make([]datasets.MetadataIssue, 0, len(issues))
	for _, issue := range issues {
		out = append(out, datasets.MetadataIssue{
			Severity: issue.Severity,
			Code:     issue.Code,
			Message:  issue.Message,
			Count:    issue.Count,
		})
	}
	return out
}

func copyIntMap(in map[string]int) map[string]int {
	out := map[string]int{}
	for key, value := range in {
		out[key] = value
	}
	return out
}

func copyBoolMap(in map[string]bool) map[string]bool {
	out := map[string]bool{}
	for key, value := range in {
		out[key] = value
	}
	return out
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func readAllLimited(reader io.Reader, limit int64) ([]byte, error) {
	return io.ReadAll(io.LimitReader(reader, limit))
}
