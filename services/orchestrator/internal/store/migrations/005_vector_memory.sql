CREATE EXTENSION IF NOT EXISTS vector;

CREATE SEQUENCE IF NOT EXISTS agent_memory_embedding_id_seq;

CREATE TABLE IF NOT EXISTS agent_memory_embeddings (
  id text PRIMARY KEY DEFAULT 'memory_embedding_' || nextval('agent_memory_embedding_id_seq'),
  source_table text NOT NULL,
  source_id text NOT NULL,
  project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dataset_id text NOT NULL DEFAULT '',
  plan_id text NOT NULL DEFAULT '',
  job_id text NOT NULL DEFAULT '',
  invocation_id text NOT NULL DEFAULT '',
  kind text NOT NULL,
  scope text NOT NULL DEFAULT '',
  embedding_model text NOT NULL,
  embedding_dimensions integer NOT NULL DEFAULT 1536,
  embedding vector(1536) NOT NULL,
  embedding_text text NOT NULL,
  summary_card jsonb NOT NULL DEFAULT '{}'::jsonb,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  quality_score double precision NOT NULL DEFAULT 0,
  outcome_score double precision NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (source_table, source_id, embedding_model)
);

ALTER TABLE agent_memory_embeddings ADD COLUMN IF NOT EXISTS plan_id text NOT NULL DEFAULT '';
ALTER TABLE agent_memory_embeddings ADD COLUMN IF NOT EXISTS job_id text NOT NULL DEFAULT '';
ALTER TABLE agent_memory_embeddings ADD COLUMN IF NOT EXISTS invocation_id text NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_agent_memory_embeddings_project_kind_updated ON agent_memory_embeddings(project_id, kind, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_memory_embeddings_project_dataset_kind ON agent_memory_embeddings(project_id, dataset_id, kind);
CREATE INDEX IF NOT EXISTS idx_agent_memory_embeddings_source ON agent_memory_embeddings(source_table, source_id);
CREATE INDEX IF NOT EXISTS idx_agent_memory_embeddings_mechanism ON agent_memory_embeddings((metadata->>'mechanism')) WHERE metadata ? 'mechanism';
CREATE INDEX IF NOT EXISTS idx_agent_memory_embeddings_outcome ON agent_memory_embeddings((metadata->>'outcome')) WHERE metadata ? 'outcome';
CREATE INDEX IF NOT EXISTS idx_agent_memory_embeddings_vector_cosine ON agent_memory_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
