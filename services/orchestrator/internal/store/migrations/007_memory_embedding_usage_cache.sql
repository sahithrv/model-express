CREATE SEQUENCE IF NOT EXISTS memory_embedding_usage_event_id_seq;
CREATE SEQUENCE IF NOT EXISTS memory_retrieval_query_cache_id_seq;

ALTER TABLE agent_memory_embeddings
  ADD COLUMN IF NOT EXISTS embedding_text_hash text NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_agent_memory_embeddings_source_model_hash
  ON agent_memory_embeddings(project_id, source_table, source_id, embedding_model, embedding_dimensions, embedding_text_hash);

CREATE TABLE IF NOT EXISTS memory_embedding_usage_events (
  id text PRIMARY KEY DEFAULT 'memory_embedding_usage_event_' || nextval('memory_embedding_usage_event_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL DEFAULT '',
  plan_id text NOT NULL DEFAULT '',
  job_id text NOT NULL DEFAULT '',
  invocation_id text NOT NULL DEFAULT '',
  purpose text NOT NULL,
  retrieval_purpose text NOT NULL DEFAULT '',
  source_table text NOT NULL DEFAULT '',
  source_id text NOT NULL DEFAULT '',
  embedding_model text NOT NULL DEFAULT '',
  embedding_dimensions integer NOT NULL DEFAULT 0,
  input_bytes integer NOT NULL DEFAULT 0,
  provider_call_count integer NOT NULL DEFAULT 0,
  retrieved_count integer NOT NULL DEFAULT 0,
  injected boolean NOT NULL DEFAULT false,
  log_only boolean NOT NULL DEFAULT false,
  cached boolean NOT NULL DEFAULT false,
  skipped boolean NOT NULL DEFAULT false,
  skip_reason text NOT NULL DEFAULT '',
  source_text_hash text NOT NULL DEFAULT '',
  query_hash text NOT NULL DEFAULT '',
  provider_usage jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_embedding_usage_events_project_purpose_created
  ON memory_embedding_usage_events(project_id, purpose, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_embedding_usage_events_project_retrieval_created
  ON memory_embedding_usage_events(project_id, retrieval_purpose, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_embedding_usage_events_source
  ON memory_embedding_usage_events(project_id, source_table, source_id, embedding_model);

CREATE TABLE IF NOT EXISTS memory_retrieval_query_cache (
  id text PRIMARY KEY DEFAULT 'memory_retrieval_query_cache_' || nextval('memory_retrieval_query_cache_id_seq'),
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL DEFAULT '',
  purpose text NOT NULL,
  embedding_model text NOT NULL,
  embedding_dimensions integer NOT NULL DEFAULT 0,
  normalized_query_hash text NOT NULL,
  query_text text NOT NULL,
  embedding_json jsonb NOT NULL DEFAULT '[]'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL,
  UNIQUE (project_id, dataset_id, purpose, embedding_model, embedding_dimensions, normalized_query_hash)
);

CREATE INDEX IF NOT EXISTS idx_memory_retrieval_query_cache_project_expires
  ON memory_retrieval_query_cache(project_id, expires_at);

