# Backend Dataset Metadata Roadmap

## Summary
Keep the current ImageFolder flow working and keep `datasets.profile` as the legacy profile surface. Add backend-owned normalized metadata imports beside it, with provenance, safe summaries, and worker bundles. Workers discover/transport sidecars and execute training; Go backend parses, validates, persists, summarizes, and gates what agents/workers can see.

Parallel shape:
- PR 1 is the database/store foundation.
- PR 2 builds parser/summary contracts on top of PR 1 types.
- PRs 3 and 4 can run in parallel after PR 2.
- PRs 5, 6, and 8 can run mostly in parallel after PRs 1-4 define contracts.
- PR 7 depends on PR 5 plus the normalized records.
- PR 9 is UI-only after PR 5 endpoints exist.
- PR 10 is deferred format expansion.

## Backend Schema Foundation

Use one migration, likely `003_dataset_metadata.sql`, with additive tables and indexes. Do not activate the existing `dataset_profiles` table.

Core tables:
- `dataset_metadata_imports`: one row per import attempt/version. Columns: `id`, `dataset_id`, `project_id`, `status`, `import_version`, `dataset_checksum_sha256`, `parser_registry_version`, `source_kind`, `active`, `strict_mode`, `summary jsonb`, `agent_safe_summary jsonb`, `warnings jsonb`, `errors jsonb`, `created_at`, `completed_at`.
- `dataset_metadata_sources`: one row per candidate sidecar or inferred source. Columns: `id`, `import_id`, `dataset_id`, `relative_path`, `storage_uri`, `checksum_sha256`, `size_bytes`, `declared_format`, `detected_format`, `parser_name`, `parser_version`, `status`, `warnings jsonb`, `errors jsonb`, `raw_preview jsonb`.
- `dataset_classes`: normalized class vocabulary. Columns: `id`, `import_id`, `dataset_id`, `class_key`, `class_name`, `class_index`, `parent_class_key`, `source_id`, `metadata jsonb`.
- `dataset_manifest_records`: canonical image/video sample manifest. Columns: `id`, `import_id`, `dataset_id`, `sample_key`, `media_type`, `relative_path`, `storage_uri`, `label_key`, `label_name`, `split`, `width`, `height`, `duration_ms`, `frame_count`, `checksum_sha256`, `source_id`, `metadata jsonb`.
- `dataset_annotations`: optional annotations per sample. Columns: `id`, `import_id`, `dataset_id`, `sample_key`, `annotation_type`, `label_key`, `label_name`, `bbox jsonb`, `confidence`, `source_id`, `metadata jsonb`.
- `dataset_splits`: explicit split declarations and counts. Columns: `id`, `import_id`, `dataset_id`, `split_name`, `sample_count`, `source_id`, `metadata jsonb`.

Indexes:
- partial unique active import per dataset: `(dataset_id) WHERE active`.
- lookup indexes on `(dataset_id, import_id)`, `(dataset_id, sample_key)`, `(dataset_id, split)`, `(dataset_id, label_key)`, `(dataset_id, annotation_type)`.
- source uniqueness scoped to import: `(import_id, relative_path)`.

Schema defaults:
- Use repo style text IDs with sequences.
- Store raw source files in object storage or dataset archive provenance, not unbounded Postgres blobs.
- Persist only bounded `raw_preview` for debugging, never sent to agents.

## PR 1 - Schema, Store, And Domain Types
1. Goal: Add normalized metadata persistence without changing existing dataset registration/training behavior.
2. Files/modules: `internal/datasets/model.go`, new metadata model file, `internal/store/{store,memory,postgres}.go`, migrations.
3. DB: add the schema above and migration/index tests.
4. Go types/interfaces: `DatasetMetadataImport`, `DatasetMetadataSource`, `DatasetClass`, `DatasetManifestRecord`, `DatasetAnnotation`, `DatasetSplit`, `DatasetMetadataSummary`, `AgentSafeDatasetMetadataSummary`; store CRUD/list/get-active methods.
5. Parser/adapter design: none yet, only persisted contracts.
6. API changes: none.
7. Validation: store validates dataset/project ownership, active import replacement, enum normalization, and JSON defaults.
8. Worker impact: none.
9. Agent impact: none.
10. UI impact: none.
11. Tests: memory/postgres CRUD, active import swap, migration idempotence.
12. Risks/tradeoffs: larger store interface; keep methods narrowly scoped.
13. Acceptance: existing tests pass and a metadata import can be persisted/retrieved in memory and Postgres.

## PR 2 - Parser Registry, Detection, Validation, Summaries
1. Goal: Build backend parse/validate/summarize engine independent of API and worker upload.
2. Files/modules: new `internal/datasets/metadata` package or equivalent.
3. DB: none beyond PR 1.
4. Go types/interfaces: `MetadataParser`, `MetadataFormat`, `MetadataParseInput`, `MetadataParseOutput`, `MetadataIssue`, `DatasetFileInventory`.
5. Parser design: registry chooses parser by declared format, known filename, extension, and bounded sniffing. Unknown files become `skipped_unsupported` warnings.
6. API changes: none.
7. Validation: reject unsafe paths/path traversal; warn on missing labels, orphan rows, unresolved images, conflicting labels, partial splits.
8. Worker impact: none.
9. Agent impact: summary generator strips paths, URIs, raw rows, and source contents.
10. UI impact: none.
11. Tests: detector matrix, summary generation, warning/error taxonomy, safe-summary scrub tests.
12. Risks/tradeoffs: detection ambiguity; declared format wins only when parser succeeds.
13. Acceptance: engine can return normalized records plus safe summaries from in-memory fixtures.

## PR 3 - Classification MVP Parsers
1. Goal: Support the smallest useful metadata set for image classification.
2. Files/modules: parser package and fixtures.
3. DB: uses PR 1 tables.
4. Go types/interfaces: format constants for `image_folder`, `split_folder`, `csv_manifest`, `cub_sidecars`.
5. Parser design: support ImageFolder, `train/val/test/class/image`, generic CSV with bounded aliases, and CUB files: `images.txt`, `image_class_labels.txt`, `train_test_split.txt`, `classes.txt`, optional `bounding_boxes.txt`.
6. API changes: none.
7. Validation: normalize `valid|validation` to `val`, preserve official splits, tolerate unknown extra files with warnings.
8. Worker impact: none.
9. Agent impact: summaries can report class distribution, split counts, official split availability, missing/orphan counts.
10. UI impact: none.
11. Tests: CSV aliases, CUB joins, split-folder detection, conflict handling.
12. Risks/tradeoffs: generic CSV cannot cover every schema; keep MVP aliases explicit.
13. Acceptance: non-folder labels and official splits normalize into canonical manifest records.

## PR 4 - Pascal VOC And BBox Evidence
1. Goal: Add optional bbox support as metadata, not as the whole model.
2. Files/modules: parser package, bbox validation helpers, fixtures.
3. DB: uses `dataset_annotations`.
4. Go types/interfaces: bbox helper type and annotation constants.
5. Parser design: Pascal VOC XML resolves sample by `filename`; stores object labels and `bbox` as annotations. If exactly one object label exists, it may contribute a classification label with provenance.
6. API changes: none.
7. Validation: positive-area boxes only; warn on unresolved image or multi-object classification ambiguity.
8. Worker impact: none yet.
9. Agent impact: summaries expose bbox coverage/counts only.
10. UI impact: none.
11. Tests: valid VOC, invalid bbox, missing filename, multi-object XML.
12. Risks/tradeoffs: VOC is detection-native, so classification inference stays conservative.
13. Acceptance: backend can prove bbox availability for planner gates from normalized annotations.

## PR 5 - Import API And Registration/Profile Integration
1. Goal: Expose backend metadata import endpoints and wire imports into dataset lifecycle.
2. Files/modules: `api/router.go`, `api/handlers.go`, store, metadata service.
3. DB: transactional writes to PR 1 tables; previous active import deactivated only after successful replacement.
4. Go types/interfaces: import request/response DTOs, source payload DTOs.
5. Parser design: import service runs registry, validates, stores records, stores provenance and summaries.
6. API changes: add `POST /datasets/:id/metadata/imports`, `GET /datasets/:id/metadata/imports`, `GET /datasets/:id/metadata/summary`, `GET /datasets/:id/metadata/imports/:import_id`.
7. Validation: non-strict default warns for unsupported files; strict mode can fail declared unsupported/malformed sources.
8. Worker impact: worker can call this before posting profile.
9. Agent impact: summary endpoint defaults to agent-safe output.
10. UI impact: endpoint ready for display/manual import.
11. Tests: unsupported file warnings, malformed source handling, active import swap, no raw path leakage in safe summary.
12. Risks/tradeoffs: inline source caps avoid adding Go archive/S3 extraction in MVP.
13. Acceptance: dataset registration/profile flow remains compatible and metadata import is additive.

## PR 6 - Worker Metadata Discovery Handoff
1. Goal: Let workers discover sidecars and hand bounded candidates to backend.
2. Files/modules: `worker/datasets/profiler.py`, new discovery helper, `worker/orchestrator_client.py`, Modal/local profile paths.
3. DB: none.
4. Go types/interfaces: none.
5. Parser design: worker does not semantically parse; it sends inventory plus capped candidate bytes/checksums/hints.
6. API changes: Python client calls PR 5 import endpoint, falls back gracefully if unavailable.
7. Validation: backend authoritative; worker only rejects unsafe paths before transport.
8. Worker impact: profile job imports metadata, posts profile JSON, then completes.
9. Agent impact: initial planner can see metadata summary after profiling.
10. UI impact: none.
11. Tests: Python discovery, unsupported-file warning payload, old-backend fallback.
12. Risks/tradeoffs: large sidecars skipped with explicit warnings.
13. Acceptance: sidecars in uploaded folders are imported without breaking ImageFolder-only datasets.

## PR 7 - Worker Training Bundle And Official Split Use
1. Goal: Let training consume normalized metadata only when required.
2. Files/modules: API bundle handler, Python client, `training/modal_app.py`, dataset loader helpers.
3. DB: read PR 1 normalized tables.
4. Go types/interfaces: `DatasetMetadataBundle`, paged records, worker purpose enum.
5. Parser design: no parsing; bundle emits active normalized records.
6. API changes: add `GET /datasets/:id/metadata/bundle?purpose=training&include=bbox&limit=&offset=`.
7. Validation: relative paths only, bounded pages, active import/dataset match.
8. Worker impact: training honors official splits and label mappings when `metadata_import_id`/bundle exists; otherwise legacy random ImageFolder split remains.
9. Agent impact: none.
10. UI impact: none.
11. Tests: official split honored, CSV/CUB labels override folders, bbox crop can read normalized boxes, fallback path unchanged.
12. Risks/tradeoffs: full manifests may be large; cap/paginate and fail clearly when unsupported scale is hit.
13. Acceptance: first useful backend metadata affects training behavior through a backend-owned bundle.

## PR 8 - Agent-Safe Planner Context And Backend Gates
1. Goal: Make compact metadata summaries planner-visible without raw sidecars.
2. Files/modules: `experiment_planner_llm.go`, `planner_information_tools.go`, `backend_gated_methods.go`, API planner input assembly.
3. DB: read active metadata summary.
4. Go types/interfaces: add metadata summary fields to `DatasetPlanningInsights`/`PlannerDatasetCard`.
5. Parser design: use PR 2 safe summary.
6. API changes: add planner tool `dataset_metadata_summary`; no raw detail tool in MVP.
7. Validation: prompt/snapshot tests ensure paths, URIs, source rows, and raw content are excluded.
8. Worker impact: none.
9. Agent impact: planner can cite official splits, class imbalance, bbox coverage, orphan/missing-label warnings, unsupported metadata warnings.
10. UI impact: none.
11. Tests: prompt scrub, bbox gate uses normalized bbox evidence, summary caps.
12. Risks/tradeoffs: planner may want more detail; defer bounded detail tools.
13. Acceptance: LLM receives compact summaries only, while backend remains scheduling authority.

## PR 9 - Mission Control Metadata Status
1. Goal: Show metadata import state and warnings to operators.
2. Files/modules: `apps/mission-control/src/App.tsx`, `types.ts`, `vite-env.d.ts`, Electron preload/main if manual upload is included.
3. DB: none.
4. Go types/interfaces: none.
5. Parser design: UI does not parse.
6. API changes: consume PR 5 summary/import endpoints.
7. Validation: display unsupported files as warnings, not fatal app errors.
8. Worker impact: none.
9. Agent impact: none.
10. UI impact: dataset intelligence shows metadata status, split counts, class counts, bbox coverage, warnings/errors.
11. Tests: TS build, defensive rendering for missing endpoints and empty summaries.
12. Risks/tradeoffs: manual upload can be deferred if status display is enough.
13. Acceptance: user can tell which metadata was used and which files were skipped.

## PR 10 - Format Expansion
1. Goal: Add broader formats after MVP contracts are proven.
2. Files/modules: parser adapters, fixtures, docs.
3. DB: reuse schema unless video fields prove insufficient.
4. Go types/interfaces: formats for COCO JSON, YOLO TXT, OpenImages CSV, CVAT, Label Studio, JSONL, YAML, generic JSON/XML, video clip/frame metadata.
5. Parser design: one conservative adapter per format; Parquet remains unsupported with warning unless a lightweight reader is approved.
6. API changes: none.
7. Validation: format-specific actionable errors; unknown files never break registration.
8. Worker impact: discovery recognizes more sidecars.
9. Agent impact: summaries add hierarchy, confidence, video duration/frame stats when present.
10. UI impact: display additional source formats defensively.
11. Tests: fixture suite per adapter, large-file caps, video skeleton fixtures.
12. Risks/tradeoffs: export variants are messy; prefer partial import plus warnings.
13. Acceptance: unsupported files produce clear warnings like “not used because type is unsupported,” never hard crashes.

## Assumptions And Defaults
- Backend parsing/validation is authoritative.
- Workers never invent plans, mutate datasets, or parse metadata into strategy.
- Default imports are robust and non-strict.
- Raw metadata is preserved by provenance/checksum/storage reference, not dumped into LLM context.
- MVP is PRs 1-8; PRs 9-10 are operator polish and expansion.
