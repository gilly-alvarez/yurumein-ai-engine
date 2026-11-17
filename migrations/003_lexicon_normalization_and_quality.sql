-- Migration 003: Garífuna normalization column + quality_flag
-- ============================================================
-- Addresses:
--   P0.1 — Replace full table scan with indexed exact NFC lookup on garifuna_norm.
--   P0.2 — Add quality_flag; quarantine confirmed malformed rows; mark missing-EN for review.
--
-- INSTRUCTIONS:
--   1. Take a full backup BEFORE running:
--        pg_dump -h <DB_HOST> -U <DB_USER> -d ai_db -f backup_pre_003_$(date +%Y%m%d).sql
--   2. Run as DB owner:
--        psql -h <DB_HOST> -U <DB_USER> -d ai_db -f migrations/003_lexicon_normalization_and_quality.sql
--   3. After running, verify (see VERIFY block at the end).
--   4. THEN deploy the new application version.
--
-- IDEMPOTENT: safe to run more than once.
--
-- ROLLBACK:
--   Application rollback: revert to the previous app version. The columns are
--   not referenced by old code, so leaving them in place is safe.
--   Schema rollback (only if explicitly instructed):
--     ALTER TABLE yurumein_lexicon DROP COLUMN IF EXISTS garifuna_norm;
--     ALTER TABLE yurumein_lexicon DROP COLUMN IF EXISTS quality_flag;
--     DROP FUNCTION IF EXISTS yurumein_garifuna_norm_update();
--
-- NOTES ON garifuna_norm SEMANTICS:
--   garifuna_norm = lower(garifuna)
--   This is exact NFC lowercase — diacritics are PRESERVED (ü stays ü, é stays é).
--   It is the primary lookup key for Garífuna reverse translation.
--   A SEPARATE folded fallback (diacritic-stripped, in Python) handles unaccented
--   queries and is identified as match_type="folded" in API responses.
--   The folded fallback must NEVER silently override an exact match.
--
-- NOTES ON quality_flag VALUES:
--   'ok'     — no known issue; default for all existing rows
--   'review' — low confidence; serves lookups but needs owner verification
--              (currently: rows with quality_flag='ok' AND empty english column)
--   'bad'    — confirmed malformed; EXCLUDED from all public lookups
--              (currently: confirmed contaminated IDs + rows with null/empty garifuna)
--
-- CONFLICT NOTE — missing-English count:
--   The Phase 4B audit recorded "~8 known empty-English rows" in one section and
--   listed 5 specific IDs (4397, 4508, 5330, 11076, 11113) in another. The 'review'
--   UPDATE below marks ALL rows where english IS NULL or empty. Owner must verify
--   whether the unlisted rows are intentional (Garífuna-only entries) or errors.
--   Use: SELECT id, garifuna, spanish FROM yurumein_lexicon WHERE quality_flag='review';
--
-- OWNER DECISIONS NEEDED BEFORE RUNNING:
--   1. Confirm that id=4487 should be 'bad' (OCR error; previously in review queue).
--   2. Confirm that id=4676 should be 'bad' (geographic text; in _CONTAMINATED_LEXICON_IDS).
--   3. Review rows marked 'review' to decide which are intentional Garífuna-only entries.
-- ============================================================

BEGIN;

-- ----------------------------------------------------------------
-- PART A: garifuna_norm column
-- ----------------------------------------------------------------
-- Prefer a stored generated column (PG 13+) for automatic NFC-aware maintenance.
-- PG 12.x falls back to a generated column without NFC normalization (normalize()
-- was added in PG 13); NFC must then be guaranteed at write time.
-- PG < 12 falls back to a plain column + trigger.
--
-- normalize(garifuna, NFC) is IMMUTABLE (confirmed on PG 13+), so it may be used
-- in a generated column expression without an extension.

DO $$
DECLARE
  v_pg_ver   integer;
  v_col_exists boolean;
BEGIN
  v_pg_ver     := current_setting('server_version_num')::integer;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'yurumein_lexicon' AND column_name = 'garifuna_norm'
  ) INTO v_col_exists;

  IF v_col_exists THEN
    RAISE NOTICE 'garifuna_norm already exists — skipping column creation.';

  ELSIF v_pg_ver >= 130000 THEN
    -- PG 13+: NFC-aware stored generated column.
    -- normalize(garifuna, NFC) recomposes any NFD-encoded input to NFC before
    -- lowercasing, so garifuna_norm always matches nfc_normalize_term() in Python
    -- regardless of how the garifuna column was originally encoded.
    EXECUTE
      'ALTER TABLE yurumein_lexicon '
      'ADD COLUMN garifuna_norm TEXT GENERATED ALWAYS AS (lower(normalize(garifuna, NFC))) STORED';
    RAISE NOTICE 'garifuna_norm added as GENERATED ALWAYS AS (lower(normalize(garifuna, NFC))) STORED (PG %).', v_pg_ver / 10000;

  ELSIF v_pg_ver >= 120000 THEN
    -- PG 12.x: stored generated column without NFC normalization.
    -- normalize() is not available on PG 12; NFC correctness depends on the
    -- garifuna column containing NFC-encoded data at write time.
    EXECUTE
      'ALTER TABLE yurumein_lexicon '
      'ADD COLUMN garifuna_norm TEXT GENERATED ALWAYS AS (lower(garifuna)) STORED';
    RAISE WARNING 'PG 12 detected: garifuna_norm uses lower(garifuna) without NFC normalization. '
                  'Ensure garifuna column is written as NFC or upgrade to PG 13+ and re-run this migration.';
    RAISE NOTICE 'garifuna_norm added as GENERATED ALWAYS AS (lower(garifuna)) STORED (PG 12 — NFC not enforced).';

  ELSE
    -- PG < 12: add plain column, backfill, attach trigger.
    -- normalize() is not available; NFC must be guaranteed at write time.
    EXECUTE 'ALTER TABLE yurumein_lexicon ADD COLUMN garifuna_norm TEXT';
    UPDATE yurumein_lexicon SET garifuna_norm = lower(garifuna) WHERE garifuna IS NOT NULL;
    RAISE NOTICE 'garifuna_norm added as plain column and backfilled (PG %).', v_pg_ver / 10000;
    RAISE WARNING 'PG < 12 detected: garifuna_norm uses lower(garifuna) without NFC normalization. '
                  'Upgrade to PG 13+ for full NFC correctness.';

    -- Trigger to keep garifuna_norm in sync with garifuna on future writes.
    CREATE OR REPLACE FUNCTION yurumein_garifuna_norm_update()
    RETURNS TRIGGER LANGUAGE plpgsql AS $func$
    BEGIN
      NEW.garifuna_norm := lower(NEW.garifuna);
      RETURN NEW;
    END;
    $func$;

    DROP TRIGGER IF EXISTS trig_garifuna_norm ON yurumein_lexicon;
    CREATE TRIGGER trig_garifuna_norm
      BEFORE INSERT OR UPDATE OF garifuna ON yurumein_lexicon
      FOR EACH ROW EXECUTE FUNCTION yurumein_garifuna_norm_update();

    RAISE NOTICE 'Trigger trig_garifuna_norm created for PG < 12 maintenance.';
  END IF;
END $$;

-- Index for fast reverse-lookup by NFC-lowercase form.
CREATE INDEX IF NOT EXISTS idx_yurumein_lexicon_garifuna_norm
  ON yurumein_lexicon (garifuna_norm);

-- ----------------------------------------------------------------
-- PART B: quality_flag column
-- ----------------------------------------------------------------

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'yurumein_lexicon' AND column_name = 'quality_flag'
  ) THEN
    ALTER TABLE yurumein_lexicon
      ADD COLUMN quality_flag TEXT NOT NULL DEFAULT 'ok';
    RAISE NOTICE 'quality_flag column added with default ok.';
  ELSE
    RAISE NOTICE 'quality_flag already exists — skipping column creation.';
  END IF;
END $$;

-- CHECK constraint: allow only the three defined values.
-- Named so it can be dropped independently if the value set expands.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.table_constraints
    WHERE table_name = 'yurumein_lexicon'
      AND constraint_name = 'chk_yurumein_lexicon_quality_flag'
  ) THEN
    ALTER TABLE yurumein_lexicon
      ADD CONSTRAINT chk_yurumein_lexicon_quality_flag
      CHECK (quality_flag IN ('ok', 'review', 'bad'));
    RAISE NOTICE 'CHECK constraint chk_yurumein_lexicon_quality_flag added.';
  ELSE
    RAISE NOTICE 'CHECK constraint already exists — skipping.';
  END IF;
END $$;

-- ----------------------------------------------------------------
-- PART C: Initial quality_flag values
-- ----------------------------------------------------------------
-- Step 1: quarantine confirmed malformed rows.
--   id=4676 — Spanish geographic description in garifuna column
--              (in _CONTAMINATED_LEXICON_IDS; confirmed in Phase 4C audit)
--   id=4487 — OCR error (in owner-approval queue; confirmed as bad data)
--   NULL or empty garifuna — no Garífuna form; cannot serve any lookup
UPDATE yurumein_lexicon
SET quality_flag = 'bad'
WHERE (id IN (4676, 4487) OR garifuna IS NULL OR garifuna = '')
  AND quality_flag != 'bad';  -- idempotent

-- Step 2: flag rows with no English translation for owner review.
-- These may still serve GAR→ES or ES→GAR lookups, but EN lookups cannot
-- return meaningful results. Marked 'review', not 'bad' — still served.
-- See CONFLICT NOTE above: the exact count is unresolved in the audit.
UPDATE yurumein_lexicon
SET quality_flag = 'review'
WHERE quality_flag = 'ok'
  AND (english IS NULL OR english = '');

COMMIT;

-- ----------------------------------------------------------------
-- VERIFY (run manually after migration)
-- ----------------------------------------------------------------
DO $$
DECLARE
  v_col_norm     boolean;
  v_col_flag     boolean;
  v_bad_count    integer;
  v_review_count integer;
  v_gen_expr     text;
  v_nfc_expr     boolean;
BEGIN
  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'yurumein_lexicon' AND column_name = 'garifuna_norm'
  ) INTO v_col_norm;

  SELECT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'yurumein_lexicon' AND column_name = 'quality_flag'
  ) INTO v_col_flag;

  SELECT COUNT(*) FROM yurumein_lexicon WHERE quality_flag = 'bad'    INTO v_bad_count;
  SELECT COUNT(*) FROM yurumein_lexicon WHERE quality_flag = 'review' INTO v_review_count;

  -- Retrieve the actual generation expression to confirm NFC normalization.
  SELECT pg_get_expr(d.adbin, d.adrelid)
    INTO v_gen_expr
    FROM pg_attribute a
    JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
   WHERE a.attrelid = 'yurumein_lexicon'::regclass AND a.attname = 'garifuna_norm';

  v_nfc_expr := v_gen_expr ILIKE '%normalize%';

  IF NOT v_col_norm THEN
    RAISE EXCEPTION 'Migration 003 failed: garifuna_norm column missing';
  END IF;
  IF NOT v_col_flag THEN
    RAISE EXCEPTION 'Migration 003 failed: quality_flag column missing';
  END IF;

  RAISE NOTICE 'Migration 003 applied successfully.';
  RAISE NOTICE '  garifuna_norm       : present';
  RAISE NOTICE '  generation_expr     : %', coalesce(v_gen_expr, '(plain column or trigger — not a generated column)');
  RAISE NOTICE '  NFC normalization   : %  (expected: true on PG 13+)', v_nfc_expr;
  RAISE NOTICE '  quality_flag        : present';
  RAISE NOTICE '  bad rows            : %  (expected >= 2: ids 4676 + 4487 + null-garifuna rows)', v_bad_count;
  RAISE NOTICE '  review rows         : %  (owner: verify intentional vs. erroneous Garífuna-only entries)', v_review_count;

  IF NOT v_nfc_expr THEN
    RAISE WARNING 'garifuna_norm expression does not include NFC normalization. '
                  'Exact lookups from nfc_normalize_term() may fail for NFD-encoded input. '
                  'Re-run this migration on PG 13+ to fix.';
  END IF;
END $$;
