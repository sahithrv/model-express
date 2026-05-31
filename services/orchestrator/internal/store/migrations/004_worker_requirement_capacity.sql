ALTER TABLE worker_requirements ADD COLUMN IF NOT EXISTS dataset_id text NOT NULL DEFAULT '';
ALTER TABLE worker_requirements ADD COLUMN IF NOT EXISTS dataset_checksum text NOT NULL DEFAULT '';
ALTER TABLE worker_requirements ADD COLUMN IF NOT EXISTS dataset_cache_key text NOT NULL DEFAULT '';
ALTER TABLE worker_requirements ADD COLUMN IF NOT EXISTS dataset_materialization_status text NOT NULL DEFAULT '';
ALTER TABLE worker_requirements ADD COLUMN IF NOT EXISTS cold_cache_policy text NOT NULL DEFAULT '';
ALTER TABLE worker_requirements ADD COLUMN IF NOT EXISTS max_concurrent_jobs integer NOT NULL DEFAULT 0;
ALTER TABLE worker_requirements ADD COLUMN IF NOT EXISTS max_cold_dataset_materializations integer NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_worker_requirements_provider_status ON worker_requirements(provider, status);
CREATE INDEX IF NOT EXISTS idx_worker_requirements_dataset_cache ON worker_requirements(dataset_cache_key, provider);
