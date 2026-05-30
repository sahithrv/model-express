CREATE SEQUENCE IF NOT EXISTS dataset_metadata_import_id_seq;
CREATE SEQUENCE IF NOT EXISTS dataset_metadata_source_id_seq;
CREATE SEQUENCE IF NOT EXISTS dataset_class_id_seq;
CREATE SEQUENCE IF NOT EXISTS dataset_manifest_record_id_seq;
CREATE SEQUENCE IF NOT EXISTS dataset_annotation_id_seq;
CREATE SEQUENCE IF NOT EXISTS dataset_split_id_seq;

CREATE TABLE IF NOT EXISTS dataset_metadata_imports (
  id text PRIMARY KEY DEFAULT 'dataset_metadata_import_' || nextval('dataset_metadata_import_id_seq'),
  dataset_id text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  status text NOT NULL DEFAULT '',
  import_version integer NOT NULL DEFAULT 1,
  dataset_checksum_sha256 text NOT NULL DEFAULT '',
  parser_registry_version text NOT NULL DEFAULT '',
  source_kind text NOT NULL DEFAULT '',
  active boolean NOT NULL DEFAULT false,
  strict_mode boolean NOT NULL DEFAULT false,
  summary jsonb NOT NULL DEFAULT '{}'::jsonb,
  agent_safe_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
  warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
  errors jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS dataset_metadata_sources (
  id text PRIMARY KEY DEFAULT 'dataset_metadata_source_' || nextval('dataset_metadata_source_id_seq'),
  import_id text NOT NULL REFERENCES dataset_metadata_imports(id) ON DELETE CASCADE,
  dataset_id text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  relative_path text NOT NULL DEFAULT '',
  storage_uri text NOT NULL DEFAULT '',
  checksum_sha256 text NOT NULL DEFAULT '',
  size_bytes bigint NOT NULL DEFAULT 0,
  declared_format text NOT NULL DEFAULT '',
  detected_format text NOT NULL DEFAULT '',
  parser_name text NOT NULL DEFAULT '',
  parser_version text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT '',
  warnings jsonb NOT NULL DEFAULT '[]'::jsonb,
  errors jsonb NOT NULL DEFAULT '[]'::jsonb,
  raw_preview jsonb NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE(import_id, relative_path)
);

CREATE TABLE IF NOT EXISTS dataset_classes (
  id text PRIMARY KEY DEFAULT 'dataset_class_' || nextval('dataset_class_id_seq'),
  import_id text NOT NULL REFERENCES dataset_metadata_imports(id) ON DELETE CASCADE,
  dataset_id text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  class_key text NOT NULL DEFAULT '',
  class_name text NOT NULL DEFAULT '',
  class_index integer,
  parent_class_key text NOT NULL DEFAULT '',
  source_id text NOT NULL DEFAULT '',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS dataset_manifest_records (
  id text PRIMARY KEY DEFAULT 'dataset_manifest_record_' || nextval('dataset_manifest_record_id_seq'),
  import_id text NOT NULL REFERENCES dataset_metadata_imports(id) ON DELETE CASCADE,
  dataset_id text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  sample_key text NOT NULL DEFAULT '',
  media_type text NOT NULL DEFAULT 'image',
  relative_path text NOT NULL DEFAULT '',
  storage_uri text NOT NULL DEFAULT '',
  label_key text NOT NULL DEFAULT '',
  label_name text NOT NULL DEFAULT '',
  split text NOT NULL DEFAULT '',
  width integer NOT NULL DEFAULT 0,
  height integer NOT NULL DEFAULT 0,
  duration_ms bigint NOT NULL DEFAULT 0,
  frame_count bigint NOT NULL DEFAULT 0,
  checksum_sha256 text NOT NULL DEFAULT '',
  source_id text NOT NULL DEFAULT '',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS dataset_annotations (
  id text PRIMARY KEY DEFAULT 'dataset_annotation_' || nextval('dataset_annotation_id_seq'),
  import_id text NOT NULL REFERENCES dataset_metadata_imports(id) ON DELETE CASCADE,
  dataset_id text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  sample_key text NOT NULL DEFAULT '',
  annotation_type text NOT NULL DEFAULT '',
  label_key text NOT NULL DEFAULT '',
  label_name text NOT NULL DEFAULT '',
  bbox jsonb NOT NULL DEFAULT '{}'::jsonb,
  confidence double precision,
  source_id text NOT NULL DEFAULT '',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS dataset_splits (
  id text PRIMARY KEY DEFAULT 'dataset_split_' || nextval('dataset_split_id_seq'),
  import_id text NOT NULL REFERENCES dataset_metadata_imports(id) ON DELETE CASCADE,
  dataset_id text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  split_name text NOT NULL DEFAULT '',
  sample_count integer NOT NULL DEFAULT 0,
  source_id text NOT NULL DEFAULT '',
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_metadata_imports_active_dataset ON dataset_metadata_imports(dataset_id) WHERE active;
CREATE INDEX IF NOT EXISTS idx_dataset_metadata_imports_dataset_created_at ON dataset_metadata_imports(dataset_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dataset_metadata_imports_project_created_at ON dataset_metadata_imports(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dataset_metadata_sources_dataset_import ON dataset_metadata_sources(dataset_id, import_id);
CREATE INDEX IF NOT EXISTS idx_dataset_classes_dataset_import ON dataset_classes(dataset_id, import_id);
CREATE INDEX IF NOT EXISTS idx_dataset_classes_dataset_label ON dataset_classes(dataset_id, class_key);
CREATE INDEX IF NOT EXISTS idx_dataset_manifest_records_dataset_import ON dataset_manifest_records(dataset_id, import_id);
CREATE INDEX IF NOT EXISTS idx_dataset_manifest_records_dataset_sample ON dataset_manifest_records(dataset_id, sample_key);
CREATE INDEX IF NOT EXISTS idx_dataset_manifest_records_dataset_split ON dataset_manifest_records(dataset_id, split);
CREATE INDEX IF NOT EXISTS idx_dataset_manifest_records_dataset_label ON dataset_manifest_records(dataset_id, label_key);
CREATE INDEX IF NOT EXISTS idx_dataset_annotations_dataset_import ON dataset_annotations(dataset_id, import_id);
CREATE INDEX IF NOT EXISTS idx_dataset_annotations_dataset_sample ON dataset_annotations(dataset_id, sample_key);
CREATE INDEX IF NOT EXISTS idx_dataset_annotations_dataset_type ON dataset_annotations(dataset_id, annotation_type);
CREATE INDEX IF NOT EXISTS idx_dataset_splits_dataset_import ON dataset_splits(dataset_id, import_id);
