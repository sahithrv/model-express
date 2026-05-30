package metadata

import (
	"testing"

	"model-express/services/orchestrator/internal/datasets"
)

func TestServiceParsesCSVManifestAndInventorySplits(t *testing.T) {
	dataset := datasets.Dataset{ID: "dataset_1", ProjectID: "project_1", ChecksumSHA256: "dataset_sha"}
	csvContent := []byte("path,label,split,width,height\ntrain/cat/one.jpg,Cat,train,320,240\nvalidation/dog/two.jpg,Dog,valid,128,128\n")

	importRecord, err := NewService().Parse(dataset, ImportRequest{
		Sources: []SourceInput{{
			RelativePath:   "metadata/labels.csv",
			DeclaredFormat: datasets.MetadataFormatCSVManifest,
			Content:        csvContent,
		}},
		Inventory: DatasetFileInventory{Files: []InventoryFile{
			{RelativePath: "train/cat/one.jpg", SizeBytes: 10},
			{RelativePath: "validation/dog/two.jpg", SizeBytes: 11},
		}},
	})
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if importRecord.Status != datasets.MetadataImportStatusSucceeded {
		t.Fatalf("status = %s", importRecord.Status)
	}
	if importRecord.Summary.ClassCount != 2 || importRecord.Summary.SampleCount != 2 {
		t.Fatalf("summary counts = %#v", importRecord.Summary)
	}
	if importRecord.Summary.SplitCounts["train"] != 1 || importRecord.Summary.SplitCounts["val"] != 1 {
		t.Fatalf("split counts = %#v", importRecord.Summary.SplitCounts)
	}
	if !importRecord.Summary.OfficialSplitAvailable {
		t.Fatalf("expected official split availability in %#v", importRecord.Summary)
	}
	if importRecord.AgentSafeSummary.BBoxAnnotationCount != 0 {
		t.Fatalf("unexpected bbox count in safe summary: %#v", importRecord.AgentSafeSummary)
	}
}

func TestServiceInfersClassesInsideTopLevelImageContainer(t *testing.T) {
	dataset := datasets.Dataset{ID: "dataset_1", ProjectID: "project_1"}
	importRecord, err := NewService().Parse(dataset, ImportRequest{
		Inventory: DatasetFileInventory{Files: []InventoryFile{
			{RelativePath: "images/cat/one.jpg", SizeBytes: 10},
			{RelativePath: "images/dog/two.jpg", SizeBytes: 11},
			{RelativePath: "data/bird/three.jpg", SizeBytes: 12},
		}},
	})
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if importRecord.Summary.ClassCount != 3 || importRecord.Summary.SampleCount != 3 {
		t.Fatalf("summary counts = %#v", importRecord.Summary)
	}
	if hasClass(importRecord.Classes, "images") || hasClass(importRecord.Classes, "data") {
		t.Fatalf("container directory was treated as a class: %#v", importRecord.Classes)
	}
	for _, classKey := range []string{"cat", "dog", "bird"} {
		if !hasClass(importRecord.Classes, classKey) {
			t.Fatalf("missing inferred class %q in %#v", classKey, importRecord.Classes)
		}
	}
}

func TestServiceParsesCUBSidecarsAndPascalVOCBBox(t *testing.T) {
	dataset := datasets.Dataset{ID: "dataset_1", ProjectID: "project_1"}
	importRecord, err := NewService().Parse(dataset, ImportRequest{
		Sources: []SourceInput{
			{RelativePath: "images.txt", Content: []byte("1 birds/cardinal.jpg\n")},
			{RelativePath: "image_class_labels.txt", Content: []byte("1 7\n")},
			{RelativePath: "classes.txt", Content: []byte("7 Cardinal\n")},
			{RelativePath: "train_test_split.txt", Content: []byte("1 0\n")},
			{RelativePath: "bounding_boxes.txt", Content: []byte("1 10 20 30 40\n")},
			{RelativePath: "annotations/cardinal.xml", DeclaredFormat: "pascal_voc_xml", Content: []byte(`<annotation><filename>birds/cardinal.jpg</filename><object><name>Cardinal</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>20</xmax><ymax>30</ymax></bndbox></object></annotation>`)},
		},
	})
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if importRecord.Summary.BBoxAnnotationCount != 2 || importRecord.Summary.BBoxSampleCount != 1 {
		t.Fatalf("bbox summary = %#v", importRecord.Summary)
	}
	if !importRecord.AgentSafeSummary.Capabilities["bbox_annotations"] {
		t.Fatalf("expected bbox capability in safe summary: %#v", importRecord.AgentSafeSummary.Capabilities)
	}
	if importRecord.Summary.SplitCounts["test"] != 1 {
		t.Fatalf("expected CUB test split, got %#v", importRecord.Summary.SplitCounts)
	}
}

func TestServiceCanonicalizesSidecarPathsAgainstInventory(t *testing.T) {
	dataset := datasets.Dataset{ID: "dataset_1", ProjectID: "project_1"}
	importRecord, err := NewService().Parse(dataset, ImportRequest{
		Sources: []SourceInput{
			{RelativePath: "images.txt", Content: []byte("1 Cardinal/cardinal.jpg\n")},
			{RelativePath: "image_class_labels.txt", Content: []byte("1 7\n")},
			{RelativePath: "classes.txt", Content: []byte("7 Cardinal\n")},
			{RelativePath: "bounding_boxes.txt", Content: []byte("1 10 20 30 40\n")},
		},
		Inventory: DatasetFileInventory{Files: []InventoryFile{
			{RelativePath: "images/Cardinal/cardinal.jpg", SizeBytes: 10},
		}},
	})
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if importRecord.Status != datasets.MetadataImportStatusSucceeded {
		t.Fatalf("status = %s, errors = %#v", importRecord.Status, importRecord.Errors)
	}
	if importRecord.Summary.SampleCount != 1 {
		t.Fatalf("expected canonicalized inventory/CUB records to dedupe to one sample, got %#v", importRecord.ManifestRecords)
	}
	if hasClass(importRecord.Classes, "images") {
		t.Fatalf("image container was treated as a class: %#v", importRecord.Classes)
	}
	if len(importRecord.Classes) != 1 || importRecord.Classes[0].ClassIndex == nil || *importRecord.Classes[0].ClassIndex != 7 {
		t.Fatalf("expected authoritative CUB class metadata, got %#v", importRecord.Classes)
	}
	if got := importRecord.ManifestRecords[0].RelativePath; got != "images/Cardinal/cardinal.jpg" {
		t.Fatalf("relative path = %q", got)
	}
	if got := importRecord.Annotations[0].SampleKey; got != "images/Cardinal/cardinal.jpg" {
		t.Fatalf("annotation sample key = %q", got)
	}
}

func TestServiceCanonicalizesCSVAndVOCPathsThroughImageRoots(t *testing.T) {
	dataset := datasets.Dataset{ID: "dataset_1", ProjectID: "project_1"}
	importRecord, err := NewService().Parse(dataset, ImportRequest{
		Sources: []SourceInput{
			{
				RelativePath:   "metadata/labels.csv",
				DeclaredFormat: datasets.MetadataFormatCSVManifest,
				Content:        []byte("path,label\ncat/one.jpg,Cat\n"),
			},
			{
				RelativePath:   "annotations/dog.xml",
				DeclaredFormat: datasets.MetadataFormatPascalVOC,
				Content:        []byte(`<annotation><filename>dog/two.jpg</filename><object><name>Dog</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>20</ymax></bndbox></object></annotation>`),
			},
		},
		Inventory: DatasetFileInventory{Files: []InventoryFile{
			{RelativePath: "images/cat/one.jpg", SizeBytes: 10},
			{RelativePath: "JPEGImages/dog/two.jpg", SizeBytes: 11},
		}},
	})
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if importRecord.Summary.SampleCount != 2 {
		t.Fatalf("sample count = %d, records = %#v", importRecord.Summary.SampleCount, importRecord.ManifestRecords)
	}
	if !hasRecordPath(importRecord.ManifestRecords, "images/cat/one.jpg") {
		t.Fatalf("CSV path was not canonicalized through images/: %#v", importRecord.ManifestRecords)
	}
	if !hasRecordPath(importRecord.ManifestRecords, "JPEGImages/dog/two.jpg") {
		t.Fatalf("VOC path was not canonicalized through JPEGImages/: %#v", importRecord.ManifestRecords)
	}
	if hasClass(importRecord.Classes, "images") || hasClass(importRecord.Classes, "jpegimages") {
		t.Fatalf("inventory labels should not be inferred when a CSV manifest is present: %#v", importRecord.Classes)
	}
}

func TestServiceInfersInventoryLabelsUnderSingleWrapperAndImageRoot(t *testing.T) {
	dataset := datasets.Dataset{ID: "dataset_1", ProjectID: "project_1"}
	importRecord, err := NewService().Parse(dataset, ImportRequest{
		Inventory: DatasetFileInventory{Files: []InventoryFile{
			{RelativePath: "archive/images/cat/one.jpg", SizeBytes: 10},
			{RelativePath: "archive/images/dog/two.jpg", SizeBytes: 11},
		}},
	})
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if importRecord.Summary.ClassCount != 2 || !hasClass(importRecord.Classes, "cat") || !hasClass(importRecord.Classes, "dog") {
		t.Fatalf("expected wrapped image-root classes to be inferred, got %#v", importRecord.Classes)
	}
	if hasClass(importRecord.Classes, "archive") || hasClass(importRecord.Classes, "images") {
		t.Fatalf("wrapper/image root was treated as a class: %#v", importRecord.Classes)
	}
}

func TestServiceParsesCUBPartLocationsAsKeypoints(t *testing.T) {
	dataset := datasets.Dataset{ID: "dataset_1", ProjectID: "project_1"}
	importRecord, err := NewService().Parse(dataset, ImportRequest{
		Sources: []SourceInput{
			{RelativePath: "images.txt", Content: []byte("1 Cardinal/cardinal.jpg\n")},
			{RelativePath: "image_class_labels.txt", Content: []byte("1 7\n")},
			{RelativePath: "classes.txt", Content: []byte("7 Cardinal\n")},
			{RelativePath: "parts/parts.txt", Content: []byte("1 beak\n2 crown\n")},
			{RelativePath: "parts/part_locs.txt", Content: []byte("1 1 12.5 22.5 1\n1 2 30 40 0\n")},
		},
		Inventory: DatasetFileInventory{Files: []InventoryFile{
			{RelativePath: "JPEGImages/Cardinal/cardinal.jpg", SizeBytes: 10},
		}},
	})
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if got := importRecord.Summary.AnnotationCounts[datasets.MetadataAnnotationKeypoint]; got != 2 {
		t.Fatalf("keypoint annotation count = %d, summary = %#v", got, importRecord.Summary)
	}
	if !importRecord.Summary.Capabilities["keypoint_annotations"] || !importRecord.AgentSafeSummary.Capabilities["keypoint_annotations"] {
		t.Fatalf("expected keypoint capability, summary = %#v safe = %#v", importRecord.Summary.Capabilities, importRecord.AgentSafeSummary.Capabilities)
	}
	for _, annotation := range importRecord.Annotations {
		if annotation.AnnotationType != datasets.MetadataAnnotationKeypoint {
			continue
		}
		if annotation.SampleKey != "JPEGImages/Cardinal/cardinal.jpg" {
			t.Fatalf("keypoint sample key = %q", annotation.SampleKey)
		}
		if annotation.Metadata["part_name"] == "" {
			t.Fatalf("missing part metadata in %#v", annotation.Metadata)
		}
		return
	}
	t.Fatal("expected a keypoint annotation")
}

func TestServiceRejectsUnsafeSourcePath(t *testing.T) {
	dataset := datasets.Dataset{ID: "dataset_1", ProjectID: "project_1"}
	importRecord, err := NewService().Parse(dataset, ImportRequest{
		StrictMode: true,
		Sources: []SourceInput{{
			RelativePath: "../labels.csv",
			Content:      []byte("path,label\ncat/one.jpg,cat\n"),
		}},
	})
	if err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if importRecord.Status != datasets.MetadataImportStatusFailed {
		t.Fatalf("expected failed import, got %s", importRecord.Status)
	}
	if len(importRecord.Errors) == 0 {
		t.Fatal("expected unsafe path error")
	}
}

func hasClass(classes []datasets.DatasetClass, classKey string) bool {
	for _, class := range classes {
		if class.ClassKey == classKey {
			return true
		}
	}
	return false
}

func hasRecordPath(records []datasets.DatasetManifestRecord, relativePath string) bool {
	for _, record := range records {
		if record.RelativePath == relativePath {
			return true
		}
	}
	return false
}
