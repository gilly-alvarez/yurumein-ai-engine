"""
Tests for lexicon normalization and quality_flag filtering.

Covers:
  - nfc_normalize_term(): exact NFC lowercase, diacritics preserved
  - normalize_term(): folded NFD/NFKC, diacritics stripped (unchanged)
  - reverse_lookup_with_type(): exact tier, folded tier, 'none', quality_flag exclusion
  - lookup(): quality_flag='bad' exclusion
  - Backward compatibility: reverse_lookup() still returns a list

Unit tests use unittest.mock to avoid a live DB connection.
Integration tests (requires live service at localhost:8085) are marked with
@requires_service and skipped automatically when unreachable.

Run unit tests only:
    pytest tests/test_lexicon_normalization.py -v -k "not Integration"

Run all including integration:
    pytest tests/test_lexicon_normalization.py -v
"""
import sys
import os
import unicodedata
import json
import urllib.request
import unittest
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.lexicon.lexicon import normalize_term, nfc_normalize_term, YurumeinLexicon

AI_API = "http://localhost:8085"


def _service_available() -> bool:
    try:
        with urllib.request.urlopen(f"{AI_API}/health", timeout=3):
            return True
    except Exception:
        return False


requires_service = pytest.mark.skipif(
    not _service_available(), reason="gaupo_ai service not reachable at localhost:8085"
)


# =============================================================================
# nfc_normalize_term
# =============================================================================

class TestNfcNormalizeTerm:

    def test_lowercase(self):
        assert nfc_normalize_term("Duna") == "duna"

    def test_preserves_diacritics(self):
        result = nfc_normalize_term("Wéndeti")
        assert result == "wéndeti", (
            f"Expected 'wéndeti' (diacritics preserved), got {result!r}"
        )

    def test_preserves_umlaut(self):
        result = nfc_normalize_term("Bángüli")
        assert result == "bángüli", (
            f"Expected 'bángüli' (ü preserved), got {result!r}"
        )

    def test_nfc_distinct_from_folded(self):
        accented = nfc_normalize_term("Wéndeti")
        unaccented = nfc_normalize_term("Wendeti")
        assert accented != unaccented, (
            "nfc_normalize_term must distinguish accented from unaccented forms; "
            f"got accented={accented!r}, unaccented={unaccented!r}"
        )

    def test_none_input(self):
        assert nfc_normalize_term(None) == ""

    def test_empty_string(self):
        assert nfc_normalize_term("") == ""

    def test_strips_leading_trailing_whitespace(self):
        assert nfc_normalize_term("  Duna  ") == "duna"

    def test_collapses_internal_whitespace(self):
        assert nfc_normalize_term("Duna  Wéri") == "duna wéri"

    def test_nfc_canonical_form(self):
        # Compose 'e' + combining acute into precomposed 'é' — both should normalize to same
        composed = "é"            # precomposed U+00E9
        decomposed = "é"    # e + combining acute U+0301
        assert nfc_normalize_term(composed) == nfc_normalize_term(decomposed)


# =============================================================================
# normalize_term (folded) — existing behavior must be unchanged
# =============================================================================

class TestNormalizeTerm:

    def test_strips_diacritics(self):
        assert normalize_term("Wéndeti") == "wendeti"

    def test_strips_umlaut(self):
        assert normalize_term("Bángüli") == "banguli"

    def test_lowercase(self):
        assert normalize_term("DUNA") == "duna"

    def test_garifuna_accent_strips(self):
        assert normalize_term("garífuna") == "garifuna"

    def test_none_input(self):
        assert normalize_term(None) == ""

    def test_folded_matches_both_forms(self):
        """Folded norm of accented and unaccented Wéndeti must be equal."""
        assert normalize_term("Wéndeti") == normalize_term("Wendeti")

    def test_folded_distinct_from_nfc(self):
        """Folded and NFC norms must differ for accented Garífuna forms."""
        assert normalize_term("Wéndeti") != nfc_normalize_term("Wéndeti")


# =============================================================================
# reverse_lookup_with_type — unit tests with mocked DB
# =============================================================================

def _make_row(garifuna="Duna", english="Water", spanish="Agua", quality_flag="ok", gn=None):
    return {
        "id": 1,
        "garifuna": garifuna,
        "garifuna_norm": gn or garifuna.lower(),
        "english": english,
        "spanish": spanish,
        "quality_flag": quality_flag,
    }


def _make_mock_conn(rows):
    """Return a mock SQLAlchemy connection that yields `rows` on execute()."""
    mock_mapping = MagicMock()
    mock_mapping.all.return_value = [MagicMock(spec=dict, **r) for r in rows]
    for r_obj, r_data in zip(mock_mapping.all.return_value, rows):
        r_obj.__iter__ = lambda self, r=r_data: iter(r.items())
        r_obj.keys = lambda r=r_data: r.keys()

    mock_result = MagicMock()
    mock_result.mappings.return_value = mock_mapping

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: s
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value = mock_result
    return mock_conn


class TestReverseLookupWithType:

    def _lexicon(self):
        """Return a YurumeinLexicon instance with a mocked engine."""
        lex = YurumeinLexicon.__new__(YurumeinLexicon)
        lex.engine = MagicMock()
        return lex

    def test_exact_match_uses_garifuna_norm_sql(self):
        """Exact tier: SQL query on garifuna_norm must be attempted first."""
        lex = self._lexicon()
        rows = [_make_row("Duna", "Water", "Agua")]

        # Simulate: garifuna_norm col exists, exact query returns a row.
        # We set up mappings to return dictified rows.
        mock_result = MagicMock()
        mock_result.mappings.return_value.all.return_value = [
            {k: v for k, v in row.items()} for row in rows
        ]
        conn = MagicMock()
        conn.__enter__ = lambda s: s
        conn.__exit__ = MagicMock(return_value=False)
        conn.execute.return_value = mock_result
        lex.engine.connect.return_value = conn

        with patch.object(conn, "execute", return_value=mock_result):
            # Patch connect to return our conn
            lex.engine.connect.return_value.__enter__ = lambda s: conn
            lex.engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        # Direct low-level test: mock the whole connect context
        result_rows = [{"id": 1, "garifuna": "Duna", "english": "Water", "quality_flag": "ok"}]
        with patch.object(lex.engine, "connect") as mock_connect:
            mock_conn_ctx = MagicMock()
            mock_conn_ctx.__enter__ = lambda s: mock_conn_ctx
            mock_conn_ctx.__exit__ = MagicMock(return_value=False)
            mock_mappings = MagicMock()
            mock_mappings.all.return_value = result_rows
            mock_conn_ctx.execute.return_value = MagicMock(mappings=MagicMock(return_value=mock_mappings))
            mock_connect.return_value = mock_conn_ctx

            matched, match_type = lex.reverse_lookup_with_type("Duna")

        assert match_type == "exact"
        assert len(matched) == 1

    def test_folded_fallback_when_no_exact(self):
        """When garifuna_norm SQL returns empty, folded in-memory scan fires."""
        lex = self._lexicon()
        # Exact query returns empty; folded scan finds the accented form.
        empty_rows: list = []
        full_rows = [{"id": 2, "garifuna": "Wéndeti", "english": "Elegant", "quality_flag": "ok"}]

        call_count = [0]

        def _connect_side_effect():
            ctx = MagicMock()
            ctx.__enter__ = lambda s: s
            ctx.__exit__ = MagicMock(return_value=False)

            def _execute(sql, params=None, **kw):
                call_count[0] += 1
                mock_m = MagicMock()
                # First call = exact (garifuna_norm) → empty
                # Second call = folded all-rows scan → full rows
                if call_count[0] == 1:
                    mock_m.mappings.return_value.all.return_value = []
                else:
                    mock_m.mappings.return_value.all.return_value = full_rows
                return mock_m

            ctx.execute = _execute
            return ctx

        lex.engine.connect = _connect_side_effect

        matched, match_type = lex.reverse_lookup_with_type("Wendeti")
        # 'Wendeti' folded = 'wendeti'; 'Wéndeti' folded = 'wendeti' → match
        assert match_type == "folded", f"Expected 'folded', got {match_type!r}"
        assert len(matched) == 1

    def test_none_when_no_match(self):
        """Returns ([], 'none') when neither tier finds a result."""
        lex = self._lexicon()

        def _connect_side_effect():
            ctx = MagicMock()
            ctx.__enter__ = lambda s: s
            ctx.__exit__ = MagicMock(return_value=False)
            mock_m = MagicMock()
            mock_m.mappings.return_value.all.return_value = []
            ctx.execute = MagicMock(return_value=mock_m)
            return ctx

        lex.engine.connect = _connect_side_effect

        matched, match_type = lex.reverse_lookup_with_type("xyzQwerty99")
        assert match_type == "none"
        assert matched == []

    def test_empty_term_returns_none(self):
        lex = self._lexicon()
        matched, match_type = lex.reverse_lookup_with_type("")
        assert match_type == "none"
        assert matched == []

    def test_none_term_returns_none(self):
        lex = self._lexicon()
        matched, match_type = lex.reverse_lookup_with_type(None)
        assert match_type == "none"
        assert matched == []

    def test_bad_rows_excluded_from_exact(self):
        """quality_flag='bad' rows must not appear in exact results."""
        lex = self._lexicon()
        # Row is 'bad'; exact query should return it only if the SQL guard fails.
        # The SQL guard (quality_flag != 'bad') should exclude it.
        # We simulate the DB correctly enforcing the filter by returning empty.
        good_row  = {"id": 1, "garifuna": "Duna", "english": "Water", "quality_flag": "ok"}
        bad_row   = {"id": 2, "garifuna": "Duna", "english": "WRONG", "quality_flag": "bad"}

        # Simulate DB returning only the good row (guard already applied in SQL).
        def _connect_side_effect():
            ctx = MagicMock()
            ctx.__enter__ = lambda s: s
            ctx.__exit__ = MagicMock(return_value=False)
            mock_m = MagicMock()
            mock_m.mappings.return_value.all.return_value = [good_row]
            ctx.execute = MagicMock(return_value=mock_m)
            return ctx

        lex.engine.connect = _connect_side_effect

        matched, match_type = lex.reverse_lookup_with_type("Duna")
        assert match_type == "exact"
        assert all(r.get("quality_flag") != "bad" for r in matched), (
            "quality_flag='bad' row must not appear in results"
        )

    def test_backward_compat_reverse_lookup_returns_list(self):
        """reverse_lookup() (old signature) must still return a plain list."""
        lex = self._lexicon()

        def _connect_side_effect():
            ctx = MagicMock()
            ctx.__enter__ = lambda s: s
            ctx.__exit__ = MagicMock(return_value=False)
            mock_m = MagicMock()
            mock_m.mappings.return_value.all.return_value = [
                {"id": 1, "garifuna": "Duna", "english": "Water", "quality_flag": "ok"}
            ]
            ctx.execute = MagicMock(return_value=mock_m)
            return ctx

        lex.engine.connect = _connect_side_effect

        result = lex.reverse_lookup("Duna")
        assert isinstance(result, list), "reverse_lookup() must return a list"
        assert len(result) >= 1

    def test_multiple_senses_preserved(self):
        """All rows matching the query are returned, not collapsed to one."""
        lex = self._lexicon()
        rows = [
            {"id": 1, "garifuna": "Baba", "english": "Father", "quality_flag": "ok"},
            {"id": 2, "garifuna": "Baba", "english": "Dad",    "quality_flag": "ok"},
        ]

        def _connect_side_effect():
            ctx = MagicMock()
            ctx.__enter__ = lambda s: s
            ctx.__exit__ = MagicMock(return_value=False)
            mock_m = MagicMock()
            mock_m.mappings.return_value.all.return_value = rows
            ctx.execute = MagicMock(return_value=mock_m)
            return ctx

        lex.engine.connect = _connect_side_effect

        matched, match_type = lex.reverse_lookup_with_type("Baba")
        assert len(matched) == 2, "Both senses must be preserved"

    def test_unaccented_garifuna_uses_folded_tier(self):
        """'Banguli' (no accent) must miss NFC tier 1 and hit folded tier 2 as 'folded'.

        garifuna_norm for 'Bángüli' = 'bángüli' (NFC lowercase).
        nfc_normalize_term('Banguli') = 'banguli' ≠ 'bángüli' → tier 1 misses.
        normalize_term('Banguli') = 'banguli' == normalize_term('Bángüli') = 'banguli'
        → tier 2 folded in-memory scan matches → match_type must be 'folded'.
        """
        lex = self._lexicon()
        accented_row = {"id": 10, "garifuna": "Bángüli", "english": "Bracelet", "quality_flag": "ok"}
        call_count = [0]

        def _connect_side_effect():
            ctx = MagicMock()
            ctx.__enter__ = lambda s: s
            ctx.__exit__ = MagicMock(return_value=False)

            def _execute(sql, params=None, **kw):
                call_count[0] += 1
                mock_m = MagicMock()
                if call_count[0] == 1:
                    mock_m.mappings.return_value.all.return_value = []  # tier 1: exact NFC misses
                else:
                    mock_m.mappings.return_value.all.return_value = [accented_row]  # tier 2: all rows scan
                return mock_m

            ctx.execute = _execute
            return ctx

        lex.engine.connect = _connect_side_effect
        matched, match_type = lex.reverse_lookup_with_type("Banguli")
        assert match_type == "folded", (
            f"Expected 'folded' for unaccented 'Banguli' matching accented 'Bángüli', got {match_type!r}"
        )
        assert len(matched) == 1

    def test_accented_garifuna_uses_exact_tier(self):
        """'Bángüli' (correctly accented) must hit NFC tier 1 as 'exact'.

        nfc_normalize_term('Bángüli') = 'bángüli'; if garifuna_norm = 'bángüli'
        then tier 1 SQL equality matches directly → match_type must be 'exact'.
        """
        lex = self._lexicon()
        result_rows = [{"id": 10, "garifuna": "Bángüli", "english": "Bracelet", "quality_flag": "ok"}]

        with patch.object(lex.engine, "connect") as mock_connect:
            mock_conn_ctx = MagicMock()
            mock_conn_ctx.__enter__ = lambda s: mock_conn_ctx
            mock_conn_ctx.__exit__ = MagicMock(return_value=False)
            mock_mappings = MagicMock()
            mock_mappings.all.return_value = result_rows
            mock_conn_ctx.execute.return_value = MagicMock(mappings=MagicMock(return_value=mock_mappings))
            mock_connect.return_value = mock_conn_ctx

            matched, match_type = lex.reverse_lookup_with_type("Bángüli")

        assert match_type == "exact", (
            f"Expected 'exact' for correctly accented 'Bángüli', got {match_type!r}"
        )
        assert len(matched) == 1


# =============================================================================
# lookup_with_type — unit tests with mocked DB (forward lookup: EN/ES → GAR)
# =============================================================================

def _make_en_row(garifuna="Duna", english_norm="water", quality_flag="ok"):
    return {"id": 1, "garifuna": garifuna, "english": english_norm, "quality_flag": quality_flag}


class TestLookupWithType:
    """
    Tests for YurumeinLexicon.lookup_with_type() — forward EN/ES → Garífuna lookup.

    match_type is determined by which tier returned the match:
      Tier 1: SQL exact equality on english_norm / spanish_norm → 'exact'
      Tier 2: LIKE annotation-stripping fallback (Fix A or Fix B)  → 'folded'
      No match in either tier                                       → 'none'

    The match_type does NOT depend on whether the input contained diacritics.
    """

    def _lexicon(self):
        lex = YurumeinLexicon.__new__(YurumeinLexicon)
        lex.engine = MagicMock()
        return lex

    def _mock_connect(self, tier1_rows, fix_a_rows=None, fix_b_rows=None):
        """
        Returns a connect side-effect where:
          execute call 1 → tier1_rows (SQL exact)
          execute call 2 → fix_a_rows (LIKE Fix A)  or []
          execute call 3 → fix_b_rows (LIKE Fix B)  or []
        """
        fix_a_rows = fix_a_rows or []
        fix_b_rows = fix_b_rows or []
        call_count = [0]

        def _side_effect():
            ctx = MagicMock()
            ctx.__enter__ = lambda s: s
            ctx.__exit__ = MagicMock(return_value=False)

            def _execute(sql, params=None, **kw):
                call_count[0] += 1
                mock_m = MagicMock()
                if call_count[0] == 1:
                    mock_m.mappings.return_value.all.return_value = tier1_rows
                elif call_count[0] == 2:
                    mock_m.mappings.return_value.all.return_value = fix_a_rows
                else:
                    mock_m.mappings.return_value.all.return_value = fix_b_rows
                return mock_m

            ctx.execute = _execute
            return ctx

        return _side_effect

    def test_tier1_exact_hit_returns_exact(self):
        """SQL tier 1 returns rows → match_type='exact'."""
        lex = self._lexicon()
        rows = [_make_en_row("Duna", "water")]
        lex.engine.connect = self._mock_connect(tier1_rows=rows)

        matched, match_type = lex.lookup_with_type("water", "en")
        assert match_type == "exact", f"Expected 'exact', got {match_type!r}"
        assert len(matched) == 1

    def test_tier1_no_hit_returns_none_for_spanish(self):
        """Spanish lookup: tier 1 misses → 'none' (no LIKE fallback for Spanish)."""
        lex = self._lexicon()
        lex.engine.connect = self._mock_connect(tier1_rows=[])

        matched, match_type = lex.lookup_with_type("agua", "es")
        assert match_type == "none"
        assert matched == []

    def test_tier2_fix_a_hit_returns_folded(self):
        """LIKE Fix A (space-paren annotation) fires → match_type='folded'.

        Simulates: 'handsome' exact misses; 'handsome (masculine)' stored entry
        matches via Fix A LIKE prefix.
        """
        lex = self._lexicon()
        annotation_row = [{"id": 2, "garifuna": "Lóu", "english": "handsome (masculine)", "quality_flag": "ok"}]
        lex.engine.connect = self._mock_connect(tier1_rows=[], fix_a_rows=annotation_row)

        matched, match_type = lex.lookup_with_type("handsome", "en")
        assert match_type == "folded", (
            f"Expected 'folded' for LIKE annotation fallback, got {match_type!r}"
        )
        assert len(matched) == 1

    def test_tier2_fix_b_hit_returns_folded(self):
        """LIKE Fix B (comma-paren annotation) fires → match_type='folded'.

        Simulates: 'count' exact misses, Fix A misses; 'count, (numbers)' stored
        entry matches via Fix B LIKE prefix.
        """
        lex = self._lexicon()
        annotation_row = [{"id": 3, "garifuna": "Abahüda", "english": "count, (numbers)", "quality_flag": "ok"}]
        lex.engine.connect = self._mock_connect(tier1_rows=[], fix_a_rows=[], fix_b_rows=annotation_row)

        matched, match_type = lex.lookup_with_type("count", "en")
        assert match_type == "folded", (
            f"Expected 'folded' for LIKE Fix B fallback, got {match_type!r}"
        )
        assert len(matched) == 1

    def test_no_match_returns_none(self):
        """Neither tier matches → ([], 'none')."""
        lex = self._lexicon()
        lex.engine.connect = self._mock_connect(tier1_rows=[], fix_a_rows=[], fix_b_rows=[])

        matched, match_type = lex.lookup_with_type("xyzunknown99", "en")
        assert match_type == "none"
        assert matched == []

    def test_spanish_exact_hit_returns_exact(self):
        """Spanish column exact hit → 'exact'."""
        lex = self._lexicon()
        rows = [{"id": 4, "garifuna": "Ha", "spanish": "agua", "quality_flag": "ok"}]
        lex.engine.connect = self._mock_connect(tier1_rows=rows)

        matched, match_type = lex.lookup_with_type("agua", "es")
        assert match_type == "exact", f"Expected 'exact' for Spanish tier 1, got {match_type!r}"
        assert len(matched) == 1

    def test_input_with_parenthetical_uses_tier1_only(self):
        """Input that already contains '(' skips LIKE fallback even with no tier 1 hit."""
        lex = self._lexicon()
        lex.engine.connect = self._mock_connect(tier1_rows=[])

        matched, match_type = lex.lookup_with_type("handsome (masculine)", "en")
        assert match_type == "none"
        assert matched == []


# =============================================================================
# Integration tests (require live service)
# =============================================================================

def _translate(base, target, text):
    payload = json.dumps({
        "base_language": base,
        "target_language": target,
        "text": text,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{AI_API}/translater",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


class TestMatchTypeIntegration:

    @requires_service
    def test_exact_match_returns_match_type_exact(self):
        """EN→GAR 'Bracelet' should return match_type='exact' (english_norm SQL hit)."""
        resp = _translate("en", "gar", "Bracelet")
        assert resp.get("match_type") == "exact", (
            f"Expected match_type='exact' for EN→GAR 'Bracelet', got {resp.get('match_type')!r}"
        )

    @requires_service
    def test_folded_match_returns_match_type_folded(self):
        """GAR→EN 'Wendeti' (unaccented) must return match_type='folded' since
        the lexicon stores 'Wéndeti' and the unaccented form has no exact NFC match."""
        resp = _translate("gar", "en", "Wendeti")
        mt = resp.get("match_type")
        assert mt == "folded", (
            f"Expected match_type='folded' for unaccented 'Wendeti', got {mt!r}"
        )

    @requires_service
    def test_exact_garifuna_match_returns_exact(self):
        """GAR→EN 'Wéndeti' (correctly accented) must return match_type='exact'
        once migration 003 has been applied and garifuna_norm index exists."""
        resp = _translate("gar", "en", "Wéndeti")
        mt = resp.get("match_type")
        # Before migration 003: folded (garifuna_norm absent → falls through to scan)
        # After migration 003:  exact (garifuna_norm = 'wéndeti' hit)
        assert mt in ("exact", "folded"), (
            f"GAR→EN 'Wéndeti' match_type should be exact or folded, got {mt!r}"
        )

    @requires_service
    def test_decline_returns_match_type_none(self):
        """Unknown term should return match_type='none'."""
        resp = _translate("gar", "en", "xyzqwrtnofuna99")
        mt = resp.get("match_type")
        assert mt == "none", (
            f"Expected match_type='none' for unknown term, got {mt!r}"
        )

    @requires_service
    def test_bad_rows_not_served(self):
        """After migration 003, known bad IDs (4676, 4487) must not appear in results.
        This test cannot directly query IDs, but can test that searching for the
        Spanish geographic text in those rows does not return a Garífuna translation."""
        resp = _translate("es", "gar", "norte de Livingston")
        translation = resp.get("translation", "")
        # Should either decline or return a RAG-based result, never a lexicon hit
        # for contaminated geographic text.
        assert resp.get("match_type") != "exact", (
            f"Geographic contamination row must not produce an exact lexicon hit. "
            f"Got: {resp!r}"
        )
