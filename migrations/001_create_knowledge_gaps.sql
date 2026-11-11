-- Migration 001: Create knowledge_gaps table
-- Phase 1 — Knowledge Learning System
--
-- INSTRUCTIONS:
--   1. Take a database backup BEFORE running this migration:
--        pg_dump -h <DB_HOST> -U <DB_USER> -d <DB_NAME> -f backup_pre_migration_001.sql
--   2. Run this migration as the DB owner:
--        psql -h <DB_HOST> -U <DB_USER> -d <DB_NAME> -f migrations/001_create_knowledge_gaps.sql
--   3. Verify: SELECT COUNT(*) FROM knowledge_gaps;  -- should return 0
--   4. THEN deploy the new application version.
--
-- IDEMPOTENT: safe to run multiple times. CREATE TABLE IF NOT EXISTS and
-- CREATE INDEX IF NOT EXISTS will no-op if already applied.
--
-- ROLLBACK (run only if you need to undo this migration before any data exists):
--   DROP TABLE IF EXISTS knowledge_gaps;
--
-- PRIVACY NOTES:
--   - This table stores normalized question topics only (stop-word stripped,
--     max 30 words, max 300 chars). No session IDs, no user IDs, no raw text.
--   - Rows can be individually deleted via the admin API (DELETE /knowledge-gaps/{id}).
--   - There is no automated retention policy in Phase 1. Admins manage retention
--     manually by setting status to 'dismissed' or deleting records.

BEGIN;

CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id              SERIAL PRIMARY KEY,
    topic           VARCHAR(100)  NOT NULL,
    normalized_question TEXT      NOT NULL,
    frequency       INTEGER       NOT NULL DEFAULT 1,
    flag_reason     VARCHAR(50)   NOT NULL,
    first_seen      TIMESTAMP WITHOUT TIME ZONE,
    last_seen       TIMESTAMP WITHOUT TIME ZONE,
    status          VARCHAR(30)   NOT NULL DEFAULT 'needs_review',
    admin_notes     TEXT,
    source_reference TEXT
);

-- Indexes used by the admin list/filter queries
CREATE INDEX IF NOT EXISTS ix_knowledge_gaps_topic
    ON knowledge_gaps(topic);

CREATE INDEX IF NOT EXISTS ix_knowledge_gaps_flag_reason
    ON knowledge_gaps(flag_reason);

CREATE INDEX IF NOT EXISTS ix_knowledge_gaps_status
    ON knowledge_gaps(status);

COMMIT;

-- Verify
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'knowledge_gaps'
    ) THEN
        RAISE EXCEPTION 'Migration 001 failed: knowledge_gaps table was not created';
    END IF;
    RAISE NOTICE 'Migration 001 applied successfully.';
END $$;
