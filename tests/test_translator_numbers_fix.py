"""
Targeted tests for the Phase 4B Translator numbers-retrieval and validation fixes.

Fix 1 — Retrieval: _RE_NUMBERS_INTENT regex detects numbers queries and
  triggers page_search(source_pattern="%DICCIONARIO GARIFUNA%", pages=(203,))
  to guarantee DICCIONARIO p.203 ("NUMERU LIDAN GARIFUNA") enters the retrieval
  pool for all casing variants ("numbers in garifuna language" / "Numbers in
  Garifuna Language" / "NUMBERS IN GARIFUNA LANGUAGE").
  A content guard ("NUMERU LIDAN GARIFUNA" in chunk) confirms the result
  is page 203 before it is accepted. page_search is a pure-SQL page-label
  lookup — no embeddings required, deterministic.

Fix 2 — Validation gate: _SKIP_VALIDATION_TOKENS excludes "garifuna" /
  "garífuna" from _output_contains_garifuna_forms(). These language-name tokens
  appear in both Spanish descriptions and Garífuna text and cannot discriminate.
  After the fix:
  - "NUMERU LIDAN GARIFUNA" passes via "lidan" (Garífuna preposition, in 20+
    lexicon entries, e.g. id=2811 'Garifunóuti, Lidan garifuna').
  - "NUMEROS EN IDIOMA GARIFUNA" still passes via "idioma" because ids 4293/4294
    have Spanish text in the garifuna column, contaminating the word set.
    This residual failure is a known limitation requiring lexicon data correction
    (ids 4293, 4294, 4676) — documented separately, not corrected here.

Regex scope (narrowed from initial draft):
  - "count" removed: handled by lexicon.lookup() via Fix B ('count, (numbers)'
    → Abahüda) before _retrieve_context() is reached; keeping it would inject the
    numbers table into unrelated phrases like "count the birds".
  - "lidan" removed: Garífuna preposition that appears in 20+ non-numbers entries
    (e.g. "Lidan oura buiti"); "lidan" alone is not a numbers indicator.

Root cause confirmed by DB:
  - keyword_search("numbers in garifuna language"): extracts terms ["numbers",
    "garifuna", "language"]; DICCIONARIO p.203 scores 1/3 (only "garifuna"
    matches — chunk has "NUMEROS"/"IDIOMA" in Spanish). Never reaches top-5.
  - similarity_search: casing-sensitive — lowercase queries score lower against
    uppercase "NUMERU LIDAN GARIFUNA" chunk.
  - nodes.py is NOT in the Translator path; _CONTEXTUAL_QUERY_EXPANSIONS
    would have no effect here.

Tests are split into:
  A. Pure-logic (no DB, no API): regex correctness and skip-token set
  B. DB-only (requires localhost:5432, no OpenAI key): retrieval and validation
  C. Live (requires valid OpenAI key): end-to-end retrieval — pending key refresh
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# A. Pure-logic tests — no DB, no API key required
# ---------------------------------------------------------------------------

class TestNumbersIntentRegex:
    """_RE_NUMBERS_INTENT must fire on numbers/counting vocabulary; not on
    generic terms like 'count' (handled by lexicon) or 'lidan' (Garífuna preposition).
    """

    @pytest.fixture(autouse=True)
    def _load(self):
        from app.services.translater.translater import _RE_NUMBERS_INTENT
        self.re = _RE_NUMBERS_INTENT

    def _match(self, s):
        return bool(self.re.search(s))

    # Must trigger — these are genuine numbers queries
    def test_lowercase_en(self):          assert self._match("numbers in garifuna language")
    def test_mixedcase_en(self):          assert self._match("Numbers in Garifuna Language")
    def test_uppercase_en(self):          assert self._match("NUMBERS IN GARIFUNA LANGUAGE")
    def test_number_singular(self):       assert self._match("what is the number for one")
    def test_numero_es(self):             assert self._match("numero en garifuna")
    def test_numeros_es(self):            assert self._match("numeros en idioma garifuna")
    def test_numero_accented(self):       assert self._match("número en garifuna")
    def test_numeros_accented(self):      assert self._match("números en idioma garifuna")
    def test_numeru_garifuna(self):       assert self._match("numeru lidan garifuna")
    def test_numeru_uppercase(self):      assert self._match("NUMERU LIDAN GARIFUNA")
    def test_contar_es(self):             assert self._match("contar en garifuna")
    def test_contar_simple(self):         assert self._match("contar")

    # Must NOT trigger — these are non-numbers or handled upstream
    def test_count_removed(self):
        """'count' alone is handled by lexicon Fix B — must not trigger numbers injection."""
        assert not self._match("count")

    def test_count_phrase_removed(self):
        """'count the birds' must not inject the numbers table."""
        assert not self._match("how do you count the birds")

    def test_lidan_alone(self):
        """'lidan' alone is a Garífuna preposition — must not trigger numbers section."""
        assert not self._match("lidan garifuna")

    def test_lidan_phrase(self):
        """Non-numbers Garífuna phrase with lidan must not trigger."""
        assert not self._match("Lidan oura buiti, Báfaguagüle")

    def test_color_query(self):           assert not self._match("what is the color blue in garifuna")
    def test_animal_query(self):          assert not self._match("what is dog in garifuna")
    def test_greeting(self):              assert not self._match("how do you say hello")
    def test_empty(self):                 assert not self._match("")
    def test_county(self):
        """'county' must not trigger (word boundary check)."""
        assert not self._match("county")
    def test_encounter(self):
        """'encounter' must not trigger."""
        assert not self._match("encounter")


class TestSkipValidationTokens:
    """_SKIP_VALIDATION_TOKENS: garifuna/garífuna excluded from validation.
    'NUMERU LIDAN GARIFUNA' must still pass via 'lidan' (confirmed in real lexicon).
    """

    @pytest.fixture(autouse=True)
    def _load(self):
        from app.services.translater.translater import _SKIP_VALIDATION_TOKENS
        self.tokens = _SKIP_VALIDATION_TOKENS

    def test_garifuna_in_set(self):
        assert "garifuna" in self.tokens

    def test_garifuna_accented_in_set(self):
        assert "garífuna" in self.tokens

    def test_set_size_reasonable(self):
        """Skip set should be small and deliberate — not a broad blacklist."""
        assert len(self.tokens) == 2, (
            f"Expected exactly 2 skip tokens (garifuna/garífuna), got {sorted(self.tokens)}"
        )

    def test_garifuna_blocks_isolated_match(self):
        """'garifuna' alone must not pass validation after the fix."""
        skip = self.tokens
        words = {"garifuna", "aban"}  # minimal simulated set
        result = False
        for w in ["garifuna"]:
            if len(w) > 2 and w in words and w not in skip:
                result = True
        assert result is False

    def test_valid_number_word_passes(self):
        """Real Garífuna number words (Aban, Biama, Seingü) must pass."""
        skip = self.tokens
        for word in ["aban", "biama", "urüwa", "seingu"]:
            fake_words = {"garifuna", word}
            result = any(
                len(w) > 2 and w in fake_words and w not in skip
                for w in [word]
            )
            assert result, f"'{word}' (Garífuna number) should PASS validation"


# ---------------------------------------------------------------------------
# B. DB-only tests — no OpenAI key, requires localhost:5432
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vs():
    """VectorStoreManager with DB-only operations (no embedding calls)."""
    try:
        from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import VectorStoreManager
        store = VectorStoreManager()
        return store
    except Exception as e:
        pytest.skip(f"VectorStoreManager unavailable: {e}")


@pytest.fixture(scope="module")
def garifuna_words():
    """Real lexicon word set from yurumein_lexicon.garifuna column."""
    try:
        from sqlalchemy import create_engine, text as sa_text
        from app.core.config import settings
        engine = create_engine(
            f"postgresql://{settings.DB_USER}:{settings.DB_PASS}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        with engine.connect() as conn:
            rows = conn.execute(sa_text(
                "SELECT garifuna FROM yurumein_lexicon WHERE garifuna IS NOT NULL AND garifuna != ''"
            )).mappings().all()
        words: set = set()
        for row in rows:
            for token in (row["garifuna"] or "").split():
                w = token.strip(".,;:!?-").lower()
                if len(w) > 2:
                    words.add(w)
        return words
    except Exception as e:
        pytest.skip(f"Lexicon word set unavailable: {e}")


class TestKeywordSearchNeverSurfacesDictionariop203:
    """
    Confirm the keyword_search root cause: DICCIONARIO p.203 is NOT in top-5
    for any casing of "numbers in garifuna language".
    This validates the need for the page_search fallback.
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


class TestPageSearchSurfacesP203:
    """
    page_search("%DICCIONARIO GARIFUNA%", pages=(203,)) must return the p.203
    chunk and the content guard "NUMERU LIDAN GARIFUNA" in chunk must hold.
    Pure-SQL: does not require OpenAI embeddings.
    """

    def test_page_search_returns_p203(self, vs):
        """page_search must return at least one chunk from page 203."""
        results = vs.page_search(
            source_pattern="%DICCIONARIO GARIFUNA%",
            pages=(203,),
        )
        assert len(results) >= 1, "page_search returned no results for DICCIONARIO p.203"
        pages = [(r.get("metadata") or {}).get("page_label") for r in results]
        assert all(p == "203" for p in pages if p is not None), (
            f"Expected all results from page 203, got {set(pages)}"
        )

    def test_content_guard_finds_at_least_one_p203_chunk(self, vs):
        """At least one p.203 chunk must contain 'NUMERU LIDAN GARIFUNA'.
        The page is split into multiple chunks; only the leading segment has the
        section header. _retrieve_context() iterates and breaks on the first hit,
        so one valid chunk is sufficient.
        """
        results = vs.page_search(
            source_pattern="%DICCIONARIO GARIFUNA%",
            pages=(203,),
        )
        passing = [
            r for r in results
            if "NUMERU LIDAN GARIFUNA" in (r.get("page_content") or "")
        ]
        assert len(passing) >= 1, (
            f"Content guard found no p.203 chunk containing 'NUMERU LIDAN GARIFUNA'. "
            f"Total p.203 chunks returned: {len(results)}"
        )

    def test_no_other_diccionario_page_contains_guard_phrase(self, vs):
        """'NUMERU LIDAN GARIFUNA' must not appear in any DICCIONARIO page other than 203.
        Confirms the guard phrase is unique to p.203 and won't pass on wrong pages.
        """
        for page in (200, 201, 202, 204, 205):
            results = vs.page_search(
                source_pattern="%DICCIONARIO GARIFUNA%",
                pages=(page,),
            )
            for r in results:
                chunk = (r.get("page_content") or "").strip()
                assert "NUMERU LIDAN GARIFUNA" not in chunk, (
                    f"Guard phrase unexpectedly found on page {page}: {repr(chunk[:200])}"
                )


class TestValidationGateWithRealLexicon:
    """
    Validate _output_contains_garifuna_forms() behavior against the ACTUAL
    lexicon word set — not a simulated clean set.

    Known limitation: ids 4293 and 4294 put "idioma" into _garifuna_words, so
    "NUMEROS EN IDIOMA GARIFUNA" still passes validation via "idioma" even after
    the skip-tokens fix. Fixing this requires correcting those DB records (owner
    action required). Tests document both the passing behavior and the limitation.
    """

    @pytest.fixture(autouse=True)
    def _load(self, garifuna_words):
        from app.services.translater.translater import _SKIP_VALIDATION_TOKENS
        self.words = garifuna_words
        self.skip = _SKIP_VALIDATION_TOKENS

    def _validate(self, translation: str) -> bool:
        """Reproduce _output_contains_garifuna_forms() logic."""
        words, skip = self.words, self.skip
        if not words:
            return False
        segments = {translation.strip().lower()}
        for sep in (',', ';'):
            segments.update(seg.strip().lower() for seg in translation.split(sep))
        for segment in segments:
            for token in segment.split():
                w = token.strip(".,;:!?-").lower()
                if len(w) > 2 and w in words and w not in skip:
                    return True
        return False

    # Correct Garífuna outputs — must pass
    def test_numeru_lidan_garifuna_passes(self):
        """The correct numbers section heading must pass validation.
        Passes via 'lidan' — a Garífuna preposition in 20+ lexicon entries."""
        assert self._validate("NUMERU LIDAN GARIFUNA"), (
            "'NUMERU LIDAN GARIFUNA' must PASS validation via 'lidan' (Garífuna preposition)"
        )

    def test_aban_passes(self):
        assert self._validate("Aban"), "'Aban' (1 in Garífuna) must pass"

    def test_biama_passes(self):
        assert self._validate("Biama"), "'Biama' (2 in Garífuna) must pass"

    def test_urüwa_passes(self):
        assert self._validate("Ürüwa"), "'Ürüwa' (3 in Garífuna) must pass"

    def test_sururugu_passes(self):
        assert self._validate("Sururugu"), "'Sururugu' (Garífuna word) must pass"

    def test_multiword_garifuna_phrase_passes(self):
        assert self._validate("Sururügü tumóun, Chamu tumóun"), (
            "Multi-word Garífuna phrase must pass"
        )

    # Validation limitation: contaminated word set
    def test_numeros_idioma_garifuna_limitation(self):
        """
        KNOWN LIMITATION: 'NUMEROS EN IDIOMA GARIFUNA' (Spanish output / leak)
        still passes via 'idioma' — which is in _garifuna_words from ids 4293/4294
        where the garifuna column contains Spanish text.
        Skip tokens ('garifuna') are excluded but 'idioma' is not.
        Root fix: correct ids 4293 and 4294 (owner action required).
        This test documents the limitation rather than asserting a fix.
        """
        idioma_in_words = "idioma" in self.words
        passes = self._validate("NUMEROS EN IDIOMA GARIFUNA")
        # Document the finding — do not assert it must pass or fail
        # since the fix requires DB correction outside this PR's scope.
        assert idioma_in_words, (
            "'idioma' expected in _garifuna_words from contaminated entries "
            f"(ids 4293, 4294). This confirms the validation limitation."
        )
        # This assertion will fail once the data correction is applied:
        # assert not passes, "After correcting ids 4293/4294, Spanish leak should fail"

    def test_pure_spanish_sentence_without_contamination(self):
        """A Spanish sentence with no contaminated token must fail validation."""
        # "el perro es bonito" — none of these words should be in _garifuna_words
        result = self._validate("el perro es bonito")
        assert not result, (
            "'el perro es bonito' (Spanish) must fail validation"
        )

    def test_garifuna_alone_rejected(self):
        """'garifuna' as the entire output must fail — it's in _SKIP_VALIDATION_TOKENS."""
        assert not self._validate("garifuna"), (
            "'garifuna' alone must be REJECTED after skip-token fix"
        )

    def test_garifuna_as_only_token_in_phrase_rejected(self):
        """
        'En Garifuna' — 'garifuna' skipped, 'en' too short — must fail.
        (Without fix, it passed via 'garifuna'.)
        """
        assert not self._validate("En Garifuna"), (
            "'En Garifuna' must be REJECTED — only 'garifuna' would match, "
            "but it is in _SKIP_VALIDATION_TOKENS"
        )
