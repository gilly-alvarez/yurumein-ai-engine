-- Migration 004: Lexicon sense metadata columns
-- ===============================================
-- Addresses P1.5 — minimum schema support for per-entry sense metadata.
--
-- All new columns are NULLABLE. No existing row values are changed.
-- Fabricated defaults are not set — unknown fields stay NULL until explicitly
-- populated by an owner-approved correction workflow.
--
-- COLUMNS ADDED:
--   part_of_speech  TEXT     — grammatical category (noun, verb, adjective, etc.)
--                              populated only from verified source annotations
--   sense_index     INTEGER  — ordinal sense within a Garífuna headword when multiple
--                              meanings exist (1-based). NULL = unknown ordering.
--   source_id       TEXT     — references source_registry.yaml id (e.g. DICCIONARIO_GARIFUNA)
--                              indicating which source the entry was derived from
--   confidence      TEXT     — editorial confidence label; allowed values not enforced
--                              here (application decides the vocabulary). Suggested:
--                              'verified', 'probable', 'uncertain', 'flagged'.
--                              NULL = not yet assessed.
--
-- OWNER DECISIONS NEEDED BEFORE POPULATING:
--   1. What is the authority source for part_of_speech? The Diccionario uses Spanish
--      grammatical classifications (Sus. = sustantivo, Vb. = verbo, etc.). Decide
--      whether to store the original abbreviation or expand to full English terms.
--   2. How is sense_index assigned? Only the Diccionario has numbered senses for some
--      entries. For the remaining sources, ordering is undocumented.
--   3. For source_id, the source_registry.yaml IDs (DICCIONARIO_GARIFUNA, LIBRO_SERIE_I,
--      etc.) should be used. Decide if single-source entries record the row source
--      or whether multi-source entries are allowed.
--   4. For confidence, establish the full vocabulary and encode it as a CHECK constraint
--      once the value set is stable.
--
-- INSTRUCTIONS:
--   1. Run migration 003 first.
--   2. Take a backup BEFORE running.
--   3. Run as DB owner:
--        psql -h <DB_HOST> -U <DB_USER> -d ai_db -f migrations/004_lexicon_sense_metadata.sql
--
-- IDEMPOTENT: safe to run more than once.
-- ===============================================

BEGIN;

ALTER TABLE yurumein_lexicon
  ADD COLUMN IF NOT EXISTS part_of_speech TEXT,
  ADD COLUMN IF NOT EXISTS sense_index    INTEGER,
  ADD COLUMN IF NOT EXISTS source_id      TEXT,
  ADD COLUMN IF NOT EXISTS confidence     TEXT;

COMMIT;

-- VERIFY
DO $$
DECLARE
  missing_cols text[] := ARRAY[]::text[];
  c text;
BEGIN
  FOREACH c IN ARRAY ARRAY['part_of_speech','sense_index','source_id','confidence'] LOOP
    IF NOT EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_name = 'yurumein_lexicon' AND column_name = c
    ) THEN
      missing_cols := array_append(missing_cols, c);
    END IF;
  END LOOP;

  IF array_length(missing_cols, 1) > 0 THEN
    RAISE EXCEPTION 'Migration 004 failed: missing columns: %', array_to_string(missing_cols, ', ');
  END IF;

  RAISE NOTICE 'Migration 004 applied successfully. All sense metadata columns present.';
  RAISE NOTICE 'All columns are NULL for existing rows — populate only after owner-approved correction workflow.';
END $$;
