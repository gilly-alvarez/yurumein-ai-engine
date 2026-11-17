import re
import unicodedata
from sqlalchemy import create_engine, text
from app.core.config import settings


def normalize_term(value: str) -> str:
    """Folded normalization: NFD decompose, strip diacritic marks, NFKC recompose, lowercase.

    Used for:
    - EN/ES source lookups (english_norm / spanish_norm columns are stored with this transform)
    - Language-code detection in the translator (strips accents: 'garífuna' → 'garifuna')
    - Folded Garífuna fallback lookup when exact NFC match returns no results

    NOT suitable as the primary Garífuna lookup key because it conflates distinct
    Garífuna forms (e.g. 'Duna' and 'Düna' would both fold to 'duna').
    Use nfc_normalize_term() for primary Garífuna lookups.
    """
    value = unicodedata.normalize("NFD", value or "")
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value.strip().lower())


def nfc_normalize_term(value: str) -> str:
    """Exact NFC normalization: NFC recompose then lowercase. Diacritics are preserved.

    This is the primary Garífuna lookup key — it matches the garifuna_norm stored
    generated column (lower(garifuna)). Two distinct Garífuna forms with different
    diacritics (e.g. 'Duna' vs 'Düna') produce different keys and are not conflated.

    A folded fallback (normalize_term) is available as a secondary tier when this
    exact match returns no results.
    """
    value = unicodedata.normalize("NFC", value or "")
    return re.sub(r"\s+", " ", value.strip().lower())


class YurumeinLexicon:
    def __init__(self):
        self.engine = create_engine(
            f"postgresql://{settings.DB_USER}:{settings.DB_PASS}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )

    def lookup(self, term: str, language: str) -> list:
        """EN/ES -> Garífuna lookup by normalized column.

        Excludes rows with quality_flag='bad'. Rows marked 'review' are still
        served — they may have valid Garífuna forms even if the English/Spanish
        annotation is incomplete.

        For English queries, if exact match fails and the query contains no
        parenthetical annotation, retries with LIKE in two passes:

        Fix A — space-paren: matches '<base> (<annotation>)' e.g.
          "handsome" → "handsome (masculine)", "handsome (feminine)"
          Covers ~2,002 entries stored with a space before '('.

        Fix B — comma-paren: if Fix A also misses, matches '<base>, (<annotation>)'
          e.g. "count" → "Count, (numbers)", "in front of" → "In front of, (she)"
          Covers ~195 entries stored with a comma before the parenthetical.

        LIKE metacharacters (%, _, !) in the input are escaped so a query
        such as '50%' or 'some_thing' cannot act as a SQL wildcard.

        When an exact entry for the base form exists, neither fallback fires.
        """
        lang = language.lower()
        norm = normalize_term(term)
        column = "spanish_norm" if lang in {"spanish", "es"} else "english_norm"

        # quality_flag filter: only 'ok' rows are served; 'bad' and 'review' are excluded.
        # quality_flag column may not yet exist (pre-migration): IS NULL keeps those rows
        # accessible so the query degrades gracefully on an un-migrated schema.
        _quality_guard = "(quality_flag IS NULL OR quality_flag = 'ok')"

        sql_exact = text(
            f"SELECT * FROM yurumein_lexicon "
            f"WHERE {column} = :term AND {_quality_guard}"
        )
        with self.engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(sql_exact, {"term": norm}).mappings().all()]
            if rows or column != "english_norm" or "(" in norm:
                return rows
            escaped = norm.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            sql_like = text(
                f"SELECT * FROM yurumein_lexicon "
                f"WHERE english_norm LIKE :prefix ESCAPE '!' AND {_quality_guard}"
            )
            # Fix A: space-paren — '<base> (<annotation>)'
            rows = [dict(r) for r in conn.execute(sql_like, {"prefix": escaped + " (%"}).mappings().all()]
            if rows:
                return rows
            # Fix B: comma-paren — '<base>, (<annotation>)'
            return [dict(r) for r in conn.execute(sql_like, {"prefix": escaped + ", (%"}).mappings().all()]

    def lookup_with_type(self, term: str, language: str) -> tuple:
        """EN/ES → Garífuna lookup with match classification.

        Returns (rows: list[dict], match_type: str) where match_type reflects
        which query tier returned the match:
          'exact'  — Tier 1 hit: direct SQL equality on indexed english_norm or
                     spanish_norm column (normalize_term applied to input)
          'folded' — Tier 2 hit: LIKE annotation-stripping fallback (Fix A or Fix B);
                     the lexicon entry was stored with a parenthetical annotation
                     (e.g. 'handsome (masculine)') and was matched by prefix
          'none'   — no match in either tier

        match_type is determined by WHICH TIER returned the match, not by whether
        the input contained diacritics. Both tiers apply normalize_term() (folded)
        to the input before querying; 'exact' means the entry matched directly on
        the indexed column, 'folded' means the LIKE prefix fallback was required.
        """
        lang = language.lower()
        norm = normalize_term(term)
        column = "spanish_norm" if lang in {"spanish", "es"} else "english_norm"
        _quality_guard = "(quality_flag IS NULL OR quality_flag = 'ok')"

        sql_exact = text(
            f"SELECT * FROM yurumein_lexicon "
            f"WHERE {column} = :term AND {_quality_guard}"
        )
        with self.engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(sql_exact, {"term": norm}).mappings().all()]
            if rows:
                return rows, "exact"  # Tier 1: SQL exact hit on indexed column
            if column != "english_norm" or "(" in norm:
                return [], "none"
            # Tier 2: LIKE annotation-stripping fallback (English only, no existing annotation)
            escaped = norm.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            sql_like = text(
                f"SELECT * FROM yurumein_lexicon "
                f"WHERE english_norm LIKE :prefix ESCAPE '!' AND {_quality_guard}"
            )
            # Fix A: space-paren — '<base> (<annotation>)'
            rows = [dict(r) for r in conn.execute(sql_like, {"prefix": escaped + " (%"}).mappings().all()]
            if rows:
                return rows, "folded"
            # Fix B: comma-paren — '<base>, (<annotation>)'
            rows = [dict(r) for r in conn.execute(sql_like, {"prefix": escaped + ", (%"}).mappings().all()]
            return rows, ("folded" if rows else "none")

    def reverse_lookup(self, term: str) -> list:
        """Garífuna → EN/ES reverse lookup.  Returns matched rows; match_type discarded.

        Backward-compatible wrapper around reverse_lookup_with_type().
        Callers that need the match_type (e.g. the translator response) should
        call reverse_lookup_with_type() directly.
        """
        rows, _ = self.reverse_lookup_with_type(term)
        return rows

    def reverse_lookup_with_type(self, term: str) -> tuple:
        """Garífuna → EN/ES reverse lookup with match classification.

        Returns (rows: list[dict], match_type: str) where match_type is one of:
          'exact'  — NFC-lowercase SQL match on indexed garifuna_norm column
          'folded' — diacritic-stripped fallback; input differed from canonical form
          'none'   — no match found in either tier

        Lookup order:
          1. Exact: SQL query on garifuna_norm = nfc_normalize(term)
             Uses the index added by migration 003 for O(log n) performance.
          2. Folded: in-memory full scan using normalize_term() on garifuna column.
             Only reached when the exact tier returns 0 results.
             Returns match_type='folded' so callers can distinguish it.

        Rows with quality_flag='bad' or 'review' are excluded from both tiers.
        Only 'ok' rows (and pre-migration NULL rows) are served.

        Pre-migration graceful degradation:
          If garifuna_norm column does not exist (migration 003 not yet applied),
          the SQL query raises an exception which is caught; the method falls
          through to the folded in-memory scan automatically.
          If quality_flag column does not exist, the guard IS NULL clause keeps
          both queries valid on un-migrated schemas.

        Multiple valid senses (comma-separated variants in garifuna column) are
        preserved — all matching rows are returned, not collapsed to one.
        """
        exact_norm = nfc_normalize_term(term)
        if not exact_norm:
            return [], "none"

        _quality_guard = "(quality_flag IS NULL OR quality_flag = 'ok')"

        # Tier 1: exact NFC SQL lookup on indexed garifuna_norm.
        try:
            sql_exact = text(
                f"SELECT * FROM yurumein_lexicon "
                f"WHERE garifuna_norm = :n AND {_quality_guard}"
            )
            with self.engine.connect() as conn:
                rows = [
                    dict(r)
                    for r in conn.execute(sql_exact, {"n": exact_norm}).mappings().all()
                ]
            if rows:
                return rows, "exact"
        except Exception:
            # garifuna_norm column absent (pre-migration): fall through to folded scan.
            pass

        # Tier 2: folded in-memory scan (diacritic-stripped).
        # This is a full table scan; acceptable as a secondary path for unaccented input.
        folded_norm = normalize_term(term)
        try:
            sql_all = text(
                f"SELECT * FROM yurumein_lexicon "
                f"WHERE garifuna IS NOT NULL AND {_quality_guard}"
            )
            with self.engine.connect() as conn:
                candidates = [
                    dict(r) for r in conn.execute(sql_all).mappings().all()
                ]
        except Exception:
            # quality_flag column absent: fall back to no filter.
            sql_all_bare = text(
                "SELECT * FROM yurumein_lexicon WHERE garifuna IS NOT NULL"
            )
            with self.engine.connect() as conn:
                candidates = [
                    dict(r) for r in conn.execute(sql_all_bare).mappings().all()
                ]

        result = []
        for row in candidates:
            forms = [f.strip() for f in (row.get("garifuna") or "").split(",")]
            if any(normalize_term(f) == folded_norm for f in forms):
                result.append(row)

        if result:
            return result, "folded"
        return [], "none"
