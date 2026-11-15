"""
Targeted tests for the Phase 4B/4C Translator numbers-retrieval and validation fixes.

Fix 1 — Retrieval (Phase 4B): _RE_NUMBERS_INTENT detects numbers queries and
  triggers page_search(source_pattern="%DICCIONARIO GARIFUNA%", pages=(203,))
  to guarantee DICCIONARIO p.203 ("NUMERU LIDAN GARIFUNA") enters the retrieval
  pool for all casing variants. page_search is a pure-SQL page-label lookup — no
  embeddings required, deterministic. Content guard confirms p.203.

Fix 2 — Validation gate (Phase 4B): _SKIP_VALIDATION_TOKENS excludes
  "garifuna"/"garífuna" from _output_contains_garifuna_forms(). These tokens
  appear in both Spanish and Garífuna text and cannot discriminate.

Fix 3 — Contamination filter (Phase 4C): _CONTAMINATED_LEXICON_IDS excludes
  23 records with Spanish text in the garifuna column from _load_garifuna_words().
  Source: Phase 4C audit. Key contaminating records:
    - 4293/4294: Spanish-language annotations injecting "idioma", "español"
    - 6326, 7511-7512, 8923, 9090, 9099, 9131, 9132: alphabet meta-commentary
    - 4583, 4676, 5145, 5593, 5595, 7471: geographic/biographical Spanish text
    - 4090, 5142: Spanish institutional headings
    - 5269, 7927: other meta-linguistic Spanish
    - 6857: Spanish description containing isolated Garífuna words
  After this fix:
    - "idioma" is no longer in _garifuna_words — from contaminated entries only
    - "NUMEROS EN IDIOMA GARIFUNA" is now correctly REJECTED

Regex scope (narrowed in Phase 4B, unchanged):
  - "count" excluded: handled upstream by lexicon Fix B
  - "lidan" excluded: Garífuna preposition in 20+ non-numbers contexts

Root cause (Phase 4B, confirmed by DB):
  - keyword_search scores DICCIONARIO p.203 too low for English queries
  - similarity_search is casing-sensitive; lowercase queries miss the chunk
  - nodes.py is NOT in the Translator path (_CONTEXTUAL_QUERY_EXPANSIONS
    has no effect here)

Tests:
  A. Pure-logic (no DB, no API): regex, skip-tokens, contamination-ID set
  B. DB-only (requires localhost:5432): retrieval confirmation and validation
  C. Live (requires valid OpenAI key): end-to-end — pending key rotation
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


class TestContaminatedLexiconIds:
    """_CONTAMINATED_LEXICON_IDS must contain the full set of 23 audited records.
    These are the records whose garifuna columns contain Spanish text (alphabet
    meta-commentary, geographic descriptions, language annotations) that would
    otherwise inject Spanish tokens like 'idioma', 'consonantes', 'sustituida'
    into the validation word set.
    """

    @pytest.fixture(autouse=True)
    def _load(self):
        from app.services.translater.translater import _CONTAMINATED_LEXICON_IDS
        self.ids = _CONTAMINATED_LEXICON_IDS

    def test_set_size(self):
        assert len(self.ids) == 23, (
            f"Expected 23 contaminated IDs from Phase 4C audit, got {len(self.ids)}"
        )

    def test_contains_idioma_sources(self):
        """Records confirmed to inject 'idioma' into the word set."""
        for expected in (4293, 4294, 6326, 7511, 8923, 9099, 9132):
            assert expected in self.ids, f"id={expected} (idioma source) missing from set"

    def test_contains_alphabet_meta_sources(self):
        """Records with alphabet meta-commentary in garifuna column."""
        for expected in (4188, 5269, 7512, 8922, 9090, 9131):
            assert expected in self.ids, f"id={expected} (alphabet meta) missing from set"

    def test_contains_geographic_sources(self):
        """Records with Spanish geographic descriptions in garifuna column."""
        for expected in (4583, 4676, 5145, 5593, 5595, 7471):
            assert expected in self.ids, f"id={expected} (geographic) missing from set"

    def test_set_is_frozenset(self):
        """Must be a frozenset so it cannot be mutated at runtime."""
        from app.services.translater.translater import _CONTAMINATED_LEXICON_IDS as cids
        assert isinstance(cids, frozenset)


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
    """Real lexicon word set, built exactly as _load_garifuna_words() does.
    Excludes _CONTAMINATED_LEXICON_IDS so the fixture tests the corrected set."""
    try:
        from sqlalchemy import create_engine, text as sa_text
        from app.core.config import settings
        from app.services.translater.translater import _CONTAMINATED_LEXICON_IDS
        engine = create_engine(
            f"postgresql://{settings.DB_USER}:{settings.DB_PASS}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        with engine.connect() as conn:
            rows = conn.execute(sa_text(
                "SELECT garifuna FROM yurumein_lexicon "
                "WHERE garifuna IS NOT NULL AND garifuna != '' "
                "AND id != ALL(:excluded)"
            ), {"excluded": list(_CONTAMINATED_LEXICON_IDS)}).mappings().all()
        words: set = set()
        for row in rows:
            for token in (row["garifuna"] or "").split():
                w = token.strip(".,;:!?-()").lower()
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


class TestRetrievalContextInclusion:
    """
    Task 3: Confirm that the p.203 chunk enters the final LLM context string,
    not merely returned by the database query. Tests _retrieve_context() directly.
    """

    @pytest.fixture(autouse=True)
    def _translater(self):
        try:
            from app.services.translater.translater import Translater
            self.t = Translater()
        except Exception as e:
            pytest.skip(f"Translater unavailable: {e}")

    def test_numbers_query_includes_p203_in_context(self):
        """'numbers in garifuna language' must produce a context string containing
        'NUMERU LIDAN GARIFUNA' (the p.203 content guard phrase)."""
        ctx = self.t._retrieve_context("numbers in garifuna language")
        assert "NUMERU LIDAN GARIFUNA" in ctx, (
            "p.203 chunk must be present in _retrieve_context() output for numbers query"
        )

    def test_numbers_query_uppercase_includes_p203(self):
        """Uppercase numbers query must also include p.203."""
        ctx = self.t._retrieve_context("NUMBERS IN GARIFUNA LANGUAGE")
        assert "NUMERU LIDAN GARIFUNA" in ctx

    def test_numeru_query_includes_p203(self):
        """'numeru lidan garifuna' triggers the numbers intent and must include p.203."""
        ctx = self.t._retrieve_context("numeru lidan garifuna")
        assert "NUMERU LIDAN GARIFUNA" in ctx

    def test_non_numbers_query_does_not_inject_p203(self):
        """An unrelated query must NOT include DICCIONARIO p.203 in context.
        This confirms the page_search fallback fires only on numbers intent.
        """
        ctx = self.t._retrieve_context("how do you say hello")
        assert "NUMERU LIDAN GARIFUNA" not in ctx, (
            "p.203 must not be injected for non-numbers queries"
        )

    def test_color_query_does_not_inject_p203(self):
        """Color query must not include p.203."""
        ctx = self.t._retrieve_context("what is the color blue in garifuna")
        assert "NUMERU LIDAN GARIFUNA" not in ctx

    def test_lidan_alone_does_not_inject_p203(self):
        """'lidan garifuna' must NOT inject p.203 — 'lidan' was explicitly excluded
        from _RE_NUMBERS_INTENT to avoid broad triggers."""
        ctx = self.t._retrieve_context("lidan garifuna")
        assert "NUMERU LIDAN GARIFUNA" not in ctx, (
            "'lidan garifuna' must not trigger numbers retrieval — lidan is a preposition"
        )


class TestValidationGateWithRealLexicon:
    """
    Validate _output_contains_garifuna_forms() against the contamination-filtered
    lexicon word set (Phase 4C fix applied).

    The garifuna_words fixture mirrors _load_garifuna_words() exactly:
    it excludes _CONTAMINATED_LEXICON_IDS. Tests verify that after Phase 4C:
      - "NUMEROS EN IDIOMA GARIFUNA" is REJECTED (idioma no longer in word set)
      - Legitimate Garífuna single-word and multi-word outputs are ACCEPTED
      - Language-only phrases are REJECTED (via _SKIP_VALIDATION_TOKENS)
      - Diacritics, masculine/feminine forms, and place names are handled correctly
    """

    @pytest.fixture(autouse=True)
    def _load(self, garifuna_words):
        from app.services.translater.translater import _SKIP_VALIDATION_TOKENS
        self.words = garifuna_words
        self.skip = _SKIP_VALIDATION_TOKENS

    def _validate(self, translation: str) -> bool:
        """Reproduce _output_contains_garifuna_forms() with Phase 4C strip chars."""
        words, skip = self.words, self.skip
        if not words:
            return False
        segments = {translation.strip().lower()}
        for sep in (',', ';'):
            segments.update(seg.strip().lower() for seg in translation.split(sep))
        for segment in segments:
            for token in segment.split():
                w = token.strip(".,;:!?-()").lower()
                if len(w) > 2 and w in words and w not in skip:
                    return True
        return False

    # Phase 4C: Spanish section headings must be REJECTED
    def test_numeros_idioma_garifuna_rejected(self):
        """Phase 4C fix: 'NUMEROS EN IDIOMA GARIFUNA' must be REJECTED.
        'idioma' was the contamination token from ids 4293/4294. After excluding
        those records from the word set, 'idioma' is no longer present.
        """
        assert "idioma" not in self.words, (
            "'idioma' must not be in the filtered word set "
            "(contaminated IDs 4293, 4294, 6326, 7511, 8923, 9099, 9132 excluded)"
        )
        assert not self._validate("NUMEROS EN IDIOMA GARIFUNA"), (
            "'NUMEROS EN IDIOMA GARIFUNA' (Spanish section heading) must be REJECTED"
        )

    def test_idioma_absent_from_word_set(self):
        """'idioma' must not appear in the contamination-filtered word set."""
        assert "idioma" not in self.words

    def test_consonantes_absent_from_word_set(self):
        """'consonantes' must not appear in the filtered word set."""
        assert "consonantes" not in self.words

    def test_sustituida_absent_from_word_set(self):
        """'sustituida' must not appear in the filtered word set."""
        assert "sustituida" not in self.words

    # Correct Garífuna outputs — must PASS
    def test_numeru_lidan_garifuna_passes(self):
        """The numbers section heading in Garífuna must pass validation.
        Passes via 'lidan' — a Garífuna preposition in 20+ clean lexicon entries."""
        assert self._validate("NUMERU LIDAN GARIFUNA"), (
            "'NUMERU LIDAN GARIFUNA' must PASS validation via 'lidan'"
        )

    def test_lidan_in_word_set(self):
        """'lidan' must remain in the word set from clean entries."""
        assert "lidan" in self.words, "'lidan' must be in filtered word set"

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

    def test_diacritics_preserved_urüwa(self):
        """Garífuna words with ü must pass — diacritics are part of the form."""
        assert self._validate("Ürüwa"), "Diacritic form 'Ürüwa' must pass"

    def test_diacritics_preserved_bürüraütü(self):
        """Multi-token output with diacritics (thin, masculine) must pass."""
        assert self._validate("Bürüraütü"), "'Bürüraütü' (thin, masculine) must pass"

    def test_masculine_form_passes(self):
        """Masculine Garífuna form must pass independently."""
        assert self._validate("Aturiahatu"), "'Aturiahatu' (student, masculine) must pass"

    def test_feminine_form_passes(self):
        """Feminine Garífuna form must pass independently."""
        assert self._validate("Aturiahati"), "'Aturiahati' (student, feminine) must pass"

    def test_hamuga_passes(self):
        """'Hamuga' appears in 8+ clean entries — must pass after contamination filter."""
        assert self._validate("Hamuga"), (
            "'Hamuga' must pass — it appears in clean entries (ids 452, 1710, 1711, ...)"
        )

    # Must REJECT
    def test_pure_spanish_sentence_rejected(self):
        """Spanish sentence with no Garífuna tokens must fail."""
        assert not self._validate("el perro es bonito"), (
            "'el perro es bonito' (Spanish) must be REJECTED"
        )

    def test_mixed_language_heading_rejected(self):
        """Spanish with shared word must fail: 'idioma' gone, 'garifuna' skipped."""
        assert not self._validate("idioma garifuna"), (
            "'idioma garifuna' must be REJECTED — idioma excluded, garifuna skipped"
        )

    def test_garifuna_alone_rejected(self):
        """'garifuna' alone must fail — in _SKIP_VALIDATION_TOKENS."""
        assert not self._validate("garifuna")

    def test_en_garifuna_rejected(self):
        """'En Garifuna' must fail — garifuna skipped, 'en' too short."""
        assert not self._validate("En Garifuna")


class TestValidationGateAdversarial:
    """
    Documents KNOWN LIMITATIONS of the validation word-set approach.

    These tests record the validator's actual behaviour against adversarial
    inputs drawn from the production lexicon.  Assertions prefixed KNOWN_LIMIT
    record cases the current word-set tokenisation cannot block:

      1. Garífuna loanwords from Spanish (agua, baile) appear in both languages.
      2. Spanish parenthetical glosses inside the garifuna column contribute
         tokens that are indistinguishable from Garífuna vocabulary.
      3. Spanish descriptions inside mixed-content garifuna rows contribute
         common Spanish prepositions (del) as word-set tokens.

    None of these are regressions from Phase 4C.  Fixing them requires either
    parenthetical-content stripping or a comprehensive mixed-content audit —
    neither is in scope for this PR.

    The critical Phase 4C failure (Spanish institutional headings accepted via
    'idioma') is fully fixed and covered in TestValidationGateWithRealLexicon.
    """

    @pytest.fixture(autouse=True)
    def _load(self, garifuna_words):
        from app.services.translater.translater import _SKIP_VALIDATION_TOKENS
        self.words = garifuna_words
        self.skip = _SKIP_VALIDATION_TOKENS

    def _validate(self, translation: str) -> bool:
        words, skip = self.words, self.skip
        if not words:
            return False
        segments = {translation.strip().lower()}
        for sep in (',', ';'):
            segments.update(seg.strip().lower() for seg in translation.split(sep))
        for segment in segments:
            for token in segment.split():
                w = token.strip(".,;:!?-()").lower()
                if len(w) > 2 and w in words and w not in skip:
                    return True
        return False

    # --- Confirmed Phase 4C improvements ---

    def test_phase4c_key_fix_numeros_idioma_rejected(self):
        """REGRESSION GUARD: the Phase 4C target failure must remain rejected."""
        assert not self._validate("NUMEROS EN IDIOMA GARIFUNA")

    def test_phase4c_idioma_absent(self):
        """REGRESSION GUARD: 'idioma' must not be in the filtered word set."""
        assert "idioma" not in self.words

    def test_spanish_number_words_rejected(self):
        """Spanish number words (uno–cinco) have no overlap with Garífuna lexicon."""
        assert not self._validate("Uno, dos, tres, cuatro, cinco")

    def test_spanish_place_phrase_rejected(self):
        """Spanish phrase with no lexicon overlap must be rejected."""
        assert not self._validate("Pueblo nuevo barrio garifuna")

    def test_spanish_fundacion_rejected(self):
        assert not self._validate("Fundación de Livingston")

    def test_spanish_nature_phrase_rejected(self):
        assert not self._validate("La luna y el sol")

    # --- Known limitations: loanwords / parenthetical glosses ---

    def test_known_limit_agua_in_word_set(self):
        """KNOWN LIMITATION: 'agua' is in the filtered word set.
        It enters via lexicon entries where it functions as a Garífuna verb
        form meaning 'put/throw' (ids 1094, 1585, 4165).  Separately, id=4504
        contributes it via Spanish context.  Because 'agua' has a documented
        Garífuna use, it cannot be excluded without a source-grounded audit.
        Consequence: Spanish phrases containing 'agua' can pass the gate.
        """
        assert "agua" in self.words  # documents the limitation

    def test_known_limit_del_in_word_set(self):
        """KNOWN LIMITATION: 'del' is in the filtered word set.
        It enters via 8 records whose garifuna column contains Spanish
        descriptions (e.g. 'del cazabe', 'del pozo', 'en la profundidad del').
        Root cause: mixed-content entries where Spanish context precedes a
        Garífuna word in the same column value.  Fix requires parenthetical
        stripping or a broader mixed-content audit — not in scope for this PR.
        Consequence: Spanish phrases containing 'del' can pass the gate.
        Source IDs: 5107, 5144, 6242, 6244, 6245, 7381, 7678, 8946.
        """
        assert "del" in self.words  # documents the limitation

    def test_known_limit_hombre_in_word_set(self):
        """KNOWN LIMITATION: 'hombre' is in the filtered word set.
        It enters via parenthetical Spanish glosses in ids 304, 3815, 8849
        (e.g. 'Yámani (wügürí - hombre)').  The strip of '()' extracts
        'hombre' as a bare token.
        """
        assert "hombre" in self.words  # documents the limitation

    def test_known_limit_baile_in_word_set(self):
        """KNOWN LIMITATION: 'baile' is in the filtered word set.
        It enters from id=4085 'Abinani min pol (Baile que se realiza…)'
        — a Garífuna phrase followed by a parenthetical Spanish explanation.
        """
        assert "baile" in self.words  # documents the limitation


# ---------------------------------------------------------------------------
# D. Language-code normalization regression tests (Task 1 fix)
# ---------------------------------------------------------------------------

class TestLanguageCodeNormalization:
    """
    normalize_term() applied to target_language in get_response() must
    canonicalize all documented Garífuna language-code variants to a form
    present in the _gar frozenset. Without this fix, 'Garífuna'/'garífuna'
    (accented label sent by the frontend) yielded is_garifuna_target=False,
    silently bypassing the lexicon-first path, context-evidence check, and
    output validator — the validation bypass confirmed in the Phase 4C audit.
    """

    def test_garifuna_accented_lower_normalizes(self):
        """'garífuna' (accented, lowercase) must normalize to 'garifuna'."""
        from app.services.lexicon.lexicon import normalize_term
        assert normalize_term("garífuna") == "garifuna"

    def test_garifuna_accented_title_normalizes(self):
        """'Garífuna' (accented + title-case) must normalize to 'garifuna'."""
        from app.services.lexicon.lexicon import normalize_term
        assert normalize_term("Garífuna") == "garifuna"

    def test_garifuna_title_normalizes(self):
        from app.services.lexicon.lexicon import normalize_term
        assert normalize_term("Garifuna") == "garifuna"

    def test_garifuna_lower_unchanged(self):
        from app.services.lexicon.lexicon import normalize_term
        assert normalize_term("garifuna") == "garifuna"

    def test_gar_code_unchanged(self):
        from app.services.lexicon.lexicon import normalize_term
        assert normalize_term("gar") == "gar"

    def test_all_five_variants_in_gar_frozenset(self):
        """All five documented Garífuna code variants must resolve to a _gar member."""
        from app.services.lexicon.lexicon import normalize_term
        _gar = frozenset({"garifuna", "gar"})
        for v in ("gar", "garifuna", "Garifuna", "garífuna", "Garífuna"):
            assert normalize_term(v) in _gar, (
                f"Language variant '{v}' must normalize to a _gar member; "
                f"got '{normalize_term(v)}'"
            )

    def test_unsupported_codes_not_in_gar(self):
        """Unknown language labels must not activate the Garífuna path."""
        from app.services.lexicon.lexicon import normalize_term
        _gar = frozenset({"garifuna", "gar"})
        for code in ("Spanish", "english", "EN", "es", "xyz", ""):
            assert normalize_term(code) not in _gar, (
                f"Code '{code}' must not activate the Garífuna path"
            )


class TestLanguageCodeNormalizationWithLexicon:
    """
    DB-backed: each Garífuna target-language variant must activate the
    lexicon-first path in get_response(). Tests 'water' (guaranteed lexicon
    HIT, 0 s, no OpenAI call) so the pipeline returns immediately on the hit.
    If normalization fails, is_garifuna_target=False → lexicon skipped → DECLINE
    or unvalidated OpenAI output instead of 'Duna'.
    Requires localhost:5432.
    """

    VARIANTS = ["gar", "garifuna", "Garifuna", "garífuna", "Garífuna"]

    @pytest.fixture(autouse=True)
    def _translater(self):
        try:
            from app.services.translater.translater import Translater
            self.t = Translater()
        except Exception as e:
            pytest.skip(f"Translater unavailable: {e}")

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_water_lexicon_hit_for_all_garifuna_variants(self, variant):
        """'water' must return 'Duna' via the lexicon-first path for every variant.
        A DECLINE or non-Duna result means normalize_term() failed to activate
        is_garifuna_target=True and the lexicon path was skipped."""
        result = self.t.get_response({
            "text": "water",
            "base_language": "English",
            "target_language": variant,
            "ui_language": "en",
        })
        assert (result.translation or "").strip() == "Duna", (
            f"target_language='{variant}' must return 'Duna' via lexicon; "
            f"got: {result.translation!r}"
        )


# ---------------------------------------------------------------------------
# E. Context-grounding regression tests (Task 2 fix)
# ---------------------------------------------------------------------------

class TestOutputGroundingCheck:
    """
    _output_grounded_in_context() must require at least one meaningful LLM
    output token to appear in the retrieved RAG context. This guards against
    the word-set check accepting a valid Garífuna word returned for the wrong
    meaning when the RAG context is completely unrelated to the requested term.

    Canonical case — smartphone → Mibi:
      - 'mibi' is in the Garífuna word set (a real word, different meaning)
      - RAG context for 'smartphone' is number exercises (no 'mibi')
      - word-set check: PASS (mibi in set) ← insufficient alone
      - grounding check: FAIL (mibi not in context) → DECLINE
    End-to-end confirmation is in the Phase 4C benchmark (requires VPS + OpenAI key).
    """

    @pytest.fixture(autouse=True)
    def _translater(self):
        try:
            from app.services.translater.translater import Translater
            self.t = Translater()
        except Exception as e:
            pytest.skip(f"Translater unavailable: {e}")

    # Synthetic context approximating number-exercise pages returned for 'smartphone'
    # (LIBRO GARIFUNA II, pp. 143-144): no 'mibi' present.
    _NUMBER_EXERCISE_CTX = (
        "Ejercicio # 1  720 730  aban isein urüwa buiti seingü "
        "seingü-aban seingü-biama weyu müda fufun biama urüwa"
    )

    # Synthetic p.203 context from DICCIONARIO GARIFUNA.
    _P203_CTX = (
        "NUMERU LIDAN GARIFUNA aban isein urüwa buiti seingü "
        "seingü-aban seingü-biama naguanu weyu müda fufun biama"
    )

    def test_mibi_not_grounded_in_number_exercise_context(self):
        """'Mibi' is not in a typical number-exercise context → grounding fails.
        This is the smartphone/Mibi failure case that motivates this check."""
        assert not self.t._output_grounded_in_context("Mibi", self._NUMBER_EXERCISE_CTX)

    def test_numeru_lidan_garifuna_grounded_in_p203_context(self):
        """'NUMERU LIDAN GARIFUNA' must be grounded via 'numeru' or 'lidan' in p203."""
        assert self.t._output_grounded_in_context("NUMERU LIDAN GARIFUNA", self._P203_CTX)

    def test_aban_grounded_in_context_containing_aban(self):
        """'Aban' (1 in Garífuna) must be grounded when context contains 'aban'."""
        assert self.t._output_grounded_in_context("Aban", self._NUMBER_EXERCISE_CTX)

    def test_empty_context_always_fails(self):
        assert not self.t._output_grounded_in_context("Duna", "")
        assert not self.t._output_grounded_in_context("NUMERU LIDAN GARIFUNA", "")

    def test_empty_translation_fails(self):
        assert not self.t._output_grounded_in_context("", self._P203_CTX)

    def test_short_tokens_not_counted(self):
        """Tokens of length ≤ 2 must not count as grounding evidence."""
        ctx = "en la de el"  # all tokens len ≤ 2
        assert not self.t._output_grounded_in_context("en la", ctx)

    def test_skip_tokens_not_counted_as_grounding(self):
        """'garifuna'/'garífuna' must not count as grounding evidence even if in context."""
        ctx = "garifuna garífuna lidan"
        assert not self.t._output_grounded_in_context("GARIFUNA", ctx)

    def test_non_skip_token_in_context_grounds(self):
        """'lidan' (not in skip tokens) in context is sufficient grounding evidence."""
        ctx = "garifuna garífuna lidan"
        assert self.t._output_grounded_in_context("NUMERU LIDAN GARIFUNA", ctx)

    def test_one_token_in_context_is_sufficient(self):
        """A single grounded token suffices — not all translation tokens need to match."""
        ctx = "weyu agumuraha"  # contains 'weyu' and 'agumuraha'
        assert self.t._output_grounded_in_context("Agumuraha Weyu", ctx)

    def test_token_with_punctuation_stripped_before_check(self):
        """Punctuation on tokens must be stripped before the context substring check."""
        ctx = "duna lidan weyu"
        assert self.t._output_grounded_in_context("Duna.", ctx)
        assert self.t._output_grounded_in_context("(Duna)", ctx)
