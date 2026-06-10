ALTER TABLE automation_settings ADD COLUMN IF NOT EXISTS cost_mode text NOT NULL DEFAULT 'balanced';
ALTER TABLE automation_settings ADD COLUMN IF NOT EXISTS budget_cap_usd double precision NOT NULL DEFAULT 0;
