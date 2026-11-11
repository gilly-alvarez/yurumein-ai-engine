-- 002_chat_session_management.sql
-- Adds pinned and archived state flags to chat_sessions.
-- Idempotent: safe to run more than once.
-- Run as: sudo -u postgres psql -d ai_db -f migrations/002_chat_session_management.sql

ALTER TABLE chat_sessions
  ADD COLUMN IF NOT EXISTS pinned   BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS archived BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_active
  ON chat_sessions (user_id, archived, pinned DESC, updated_at DESC);
