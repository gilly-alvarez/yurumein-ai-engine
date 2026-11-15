"""
Targeted tests for the Phase 4B Translator numbers-retrieval and validation fixes.

Fix 1 — Retrieval: _RE_NUMBERS_INTENT regex detects numbers queries and
  triggers source_targeted_search("%DICCIONARIO GARIFUNA%") to guarantee
  DICCIONARIO p.203 ("NUMERU LIDAN GARIFUNA") enters the retrieval pool for
  all casing variants ("numbers in garifuna language" / "Numbers in Garifuna
  Language" / "NUMBERS IN GARIFUNA LANGUAGE").

Fix 2 — Validation gate: _SKIP_VALIDATION_TOKENS excludes "garifuna" /
  "garífuna" from _output_contains_garifuna_forms() so that Spanish output
  such as "NUMEROS EN IDIOMA GARIFUNA" is rejected (it previously passed
  because "garifuna" was in _garifuna_words from contaminated lexicon entries).

Root cause confirmed:
  - keyword_search("numbers in garifuna language"): extracts terms ["numbers",
    "garifuna", "language"]; DICCIONARIO p.203 scores 1/3 (only "garifuna"
    matches — the chunk has "NUMEROS"/"IDIOMA" in Spanish, not English),
    never reaching top-5.
  - similarity_search: lowercase queries score lower against the uppercase
    "NUMERU LIDAN GARIFUNA" chunk than uppercase queries do.
  - nodes.py is NOT in the Translator path; _CONTEXTUAL_QUERY_EXPANSIONS
    would have no effect here.

Tests are split into:
  A. Pure-logic (no DB, no API): regex and skip-token set
  B. DB-only (requires localhost:5432, no OpenAI key): retrieval simulation
  C. Live (requires valid OpenAI key): end-to-end retrieval
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# A. Pure-logic tests — no DB, no API key required
# ---------------------------------------------------------------------------

class TestNumbersIntentRegex:
    """_RE_NUMBERS_INTENT must fire on all casing variants and EN/ES forms."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from app.services.translater.translater import _RE_NUMBERS_INTENT
        self.re = _RE_NUMBERS_INTENT

    def _match(self, s):
        return bool(self.re.search(s))

    # Must trigger
    def test_lowercase_en(self):       assert self._match("numbers in garifuna language")
    def test_mixedcase_en(self):       assert self._match("Numbers in Garifuna Language")
    def test_uppercase_en(self):       assert self._match("NUMBERS IN GARIFUNA LANGUAGE")
    def test_number_singular(self):    assert self._match("what is the number for one")
    def test_count_en(self):           assert self._match("how do you count in garifuna")
    def test_just_count(self):         assert self._match("count")
    def test_numero_es(self):          assert self._match("numero en garifuna")
    def test_numeros_es(self):         assert self._match("numeros en idioma garifuna")
    def test_lidan_garifuna(self):     assert self._match("lidan garifuna")
    def test_contar_es(self):          assert self._match("contar en garifuna")

    # Must NOT trigger
    def test_color_query(self):        assert not self._match("what is the color blue in garifuna")
    def test_animal_query(self):       assert not self._match("what is dog in garifuna")
    def test_greeting(self):           assert not self._match("how do you say hello")
    def test_empty(self):              assert not self._match("")


class TestSkipValidationTokens:
    """_SKIP_VALIDATION_TOKENS must contain garifuna/garífuna and nothing else harmful."""

    @pytest.fixture(autouse=True)
    def _load(self):
        from app.services.translater.translater import _SKIP_VALIDATION_TOKENS
        self.tokens = _SKIP_VALIDATION_TOKENS

    def test_garifuna_in_set(self):
        assert "garifuna" in self.tokens

    def test_garifuna_accented_in_set(self):
        assert "garífuna" in self.tokens

    def test_garifuna_blocks_spanish_leak(self):
        """Confirm that the standard Spanish-leak output would be rejected after the fix."""
        # Simulate _output_contains_garifuna_forms logic with skip tokens applied
        translation = "NUMEROS EN IDIOMA GARIFUNA"
        # Build a minimal word set that includes "garifuna" (the contamination source)
        fake_words = {"garifuna", "aban", "biama", "urüwa"}
        result = False
        for segment in {translation.strip().lower()}:
            for token in segment.split():
                w = token.strip(".,;:!?-").lower()
                if len(w) > 2 and w in fake_words and w not in self.tokens:
                    result = True
        assert result is False, (
            "NUMEROS EN IDIOMA GARIFUNA should be REJECTED after skip-token fix, got PASSED"
        )

    def test_valid_garifuna_still_passes(self):
        """A real Garífuna number word (Aban) must still pass validation."""
        from app.services.translater.translater import _SKIP_VALIDATION_TOKENS
        fake_words = {"garifuna", "aban", "biama", "urüwa"}
        translation = "Aban"
        result = False
        for segment in {translation.strip().lower()}:
            for token in segment.split():
                w = token.strip(".,;:!?-").lower()
                if len(w) > 2 and w in fake_words and w not in _SKIP_VALIDATION_TOKENS:
                    result = True
        assert result is True, "Aban (a real Garífuna number) should PASS validation"


# ---------------------------------------------------------------------------
# B. DB-only tests — no OpenAI key, requires localhost:5432
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vs():
    """VectorStoreManager — skipped when DB is unreachable."""
    try:
        from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import VectorStoreManager
        store = VectorStoreManager()
        return store
    except Exception:
        pytest.skip("VectorStoreManager unavailable — DB or OpenAI not reachable")


class TestKeywordSearchNeverSurfacesDictionariop203:
    """
    Confirm the keyword_search root cause: DICCIONARIO p.203 is NOT in top-5
    for any casing of "numbers in garifuna language".
    This validates the need for the source_targeted_search fallback.
    """

    QUERIES = [
        "numbers in garifuna language",
        "Numbers in Garifuna Language",
        "NUMBERS IN GARIFUNA LANGUAGE",
    ]

    def test_diccionario_absent_from_keyword_top5(self, vs):
        """DICCIONARIO GARIFUNA must not appear in keyword top-5 for any casing."""
        for query in self.QUERIES:
            results = vs.keyword_search(query=query, k=5)
            sources = [(r.get("metadata") or {}).get("source", "") for r in results]
            diccionario_hits = [s for s in sources if "DICCIONARIO GARIFUNA" in s]
            assert len(diccionario_hits) == 0, (
                f"DICCIONARIO p.203 unexpectedly in keyword top-5 for [{query}]: {diccionario_hits}"
            )
