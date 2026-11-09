"""
Regression suite for source-role governance.

Unit tests (no network, no DB) cover the deterministic components:
  - lexical_parser: translation and definition extraction
  - intent_router: intent classification
  - source_classifier: source role mapping and citation display

Integration tests (require the live API at localhost:8085 and the
PostgreSQL lexicon at the configured DB) are skipped automatically when
the services are unavailable.

Run with:
    pytest tests/test_source_roles.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import re
import urllib.request
import urllib.parse
import json


# ─── helpers ────────────────────────────────────────────────────────────────

AI_API = "http://localhost:8085"


def _service_available() -> bool:
    try:
        with urllib.request.urlopen(f"{AI_API}/health", timeout=3):
            return True
    except Exception:
        return False


def _ask_chat(question: str, lang: str = "en") -> str:
    data = urllib.parse.urlencode(
        {"user_prompt": question, "user_id": "test_regression", "language": lang}
    ).encode()
    req = urllib.request.Request(
        f"{AI_API}/chat",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read()).get("response", "")


def _ask_retrieval(question: str, k: int = 5) -> list:
    q = urllib.parse.quote(question)
    with urllib.request.urlopen(
        f"{AI_API}/retrieval_check_hybrid?query={q}&k={k}", timeout=30
    ) as r:
        return json.loads(r.read()).get("matches", [])


requires_service = pytest.mark.skipif(
    not _service_available(), reason="gaupo_ai service not reachable at localhost:8085"
)


# ─── UNIT TESTS: lexical_parser ─────────────────────────────────────────────

from app.services.chatbot.lexical_parser import parse_lexical_request, parse_definition_request


class TestLexicalParser:
    def test_translate_english_bare(self):
        r = parse_lexical_request("How do you say water in Garifuna?")
        assert r.matched
        assert r.term == "water"
        assert r.language == "english"

    def test_translate_english_quoted_single(self):
        r = parse_lexical_request("How do you say 'water' in Garifuna?")
        assert r.matched
        assert r.term == "water", f"expected 'water', got {r.term!r}"

    def test_translate_english_quoted_double(self):
        r = parse_lexical_request('How do you say "water" in Garifuna?')
        assert r.matched
        assert r.term == "water", f"expected 'water', got {r.term!r}"

    def test_translate_spanish(self):
        r = parse_lexical_request("¿Cómo se dice agua en garífuna?")
        assert r.matched
        assert r.language == "spanish"
        assert "agua" in r.term

    def test_no_match_on_pronunciation_question(self):
        r = parse_lexical_request("How do Garifuna vowels work?")
        assert not r.matched

    def test_definition_what_does_mean(self):
        r = parse_definition_request('What does "Dábula" mean?')
        assert r.matched
        assert r.language == "garifuna"
        # Quotes should be stripped
        assert '"' not in r.term, f"unexpected quotes in {r.term!r}"

    def test_definition_define_explicit(self):
        r = parse_definition_request("Define wüwüñü")
        assert r.matched
        assert r.term == "wüwüñü"

    def test_definition_que_significa(self):
        r = parse_definition_request("¿Qué significa Dábula?")
        assert r.matched

    def test_definition_no_match_on_grammar_question(self):
        r = parse_definition_request("Explain Garifuna sentence construction")
        assert not r.matched


# ─── UNIT TESTS: intent_router ───────────────────────────────────────────────

from app.services.chatbot.intent_router import Intent, detect_intent, preferred_roles_for


class TestIntentRouter:
    def test_numbering(self):
        assert detect_intent("Teach me basic Garifuna numbers") == Intent.NUMBERING

    def test_pronunciation(self):
        assert detect_intent("How do you pronounce Garifuna vowels?") == Intent.PRONUNCIATION

    def test_sentence_construction(self):
        assert detect_intent("Explain Garifuna sentence construction") == Intent.SENTENCE_CONSTRUCTION

    def test_advanced_grammar(self):
        assert detect_intent("What is Garifuna grammar like?") == Intent.ADVANCED_GRAMMAR

    def test_cultural_historical(self):
        assert detect_intent("What are the origins of the Garifuna people?") == Intent.CULTURAL_HISTORICAL

    def test_definition(self):
        assert detect_intent("What does wüwüñü mean?") == Intent.DEFINE

    def test_foundational_learn(self):
        assert detect_intent("Teach me basic Garifuna phrases") == Intent.FOUNDATIONAL_LANG

    def test_general_fallback(self):
        assert detect_intent("Tell me about Yurumein") == Intent.GENERAL

    def test_pronunciation_prefers_serie_i(self):
        roles = preferred_roles_for(Intent.PRONUNCIATION)
        assert roles[0] == "FOUNDATIONAL_LANGUAGE_EDUCATION"

    def test_cultural_prefers_garifuna_ii(self):
        roles = preferred_roles_for(Intent.CULTURAL_HISTORICAL)
        assert roles[0] == "ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT"

    def test_sentence_construction_prefers_garifuna_ii(self):
        roles = preferred_roles_for(Intent.SENTENCE_CONSTRUCTION)
        assert roles[0] == "ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT"


# ─── UNIT TESTS: source_classifier ──────────────────────────────────────────

from app.services.chatbot.source_classifier import classify_source, source_display_name


class TestSourceClassifier:
    def test_serie_i_recognized(self):
        m = classify_source("app/services/utils/RAG_pipeline/data/new_docs/LIBRO SERIE I.pdf")
        assert m is not None
        assert m.role == "FOUNDATIONAL_LANGUAGE_EDUCATION"
        assert m.display == "Libro Serie I"

    def test_garifuna_ii_recognized(self):
        m = classify_source("app/services/utils/RAG_pipeline/data/new_docs/LIBRO GARIFUNA II FORMATO DIGITAL.pdf")
        assert m is not None
        assert m.role == "ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT"

    def test_diccionario_recognized(self):
        m = classify_source("app/services/utils/RAG_pipeline/data/new_docs/DICCIONARIO GARIFUNA.pdf")
        assert m is not None
        assert m.role == "DICTIONARY"

    def test_unknown_source_returns_none(self):
        assert classify_source("some/random/file.pdf") is None

    def test_citation_uses_page_label(self):
        meta = {
            "source": "app/.../LIBRO SERIE I.pdf",
            "page": 3,
            "page_label": "4",
        }
        citation = source_display_name(meta)
        assert "Libro Serie I" in citation
        assert "page 4" in citation  # page_label (1-indexed), not page (0-indexed)

    def test_citation_falls_back_to_page_plus_one(self):
        meta = {
            "source": "app/.../LIBRO GARIFUNA II FORMATO DIGITAL.pdf",
            "page": 29,
        }
        citation = source_display_name(meta)
        assert "Libro Garífuna II" in citation
        assert "page 30" in citation


# ─── INTEGRATION TESTS: reverse lookup ───────────────────────────────────────

try:
    from app.services.lexicon.lexicon import YurumeinLexicon, normalize_term
    _lexicon_importable = True
except Exception:
    _lexicon_importable = False

requires_lexicon = pytest.mark.skipif(
    not _lexicon_importable, reason="lexicon module not importable"
)


class TestReverseLookup:
    @requires_lexicon
    def test_exact_match_duna(self):
        # "Dábula" (table) lives in Diccionario Garífuna (RAG), not the
        # structured lexicon. Use "Duna" (water), confirmed in yurumein_lexicon.
        rows = YurumeinLexicon().reverse_lookup("Duna")
        assert len(rows) > 0
        assert all(normalize_term(r["garifuna"]) == "duna" for r in rows)

    @requires_lexicon
    def test_no_substring_match(self):
        # "duna" (water) must not match "aduna" or other words containing "duna"
        rows = YurumeinLexicon().reverse_lookup("duna")
        for r in rows:
            assert normalize_term(r["garifuna"]) == "duna", (
                f"substring match leaked: {r['garifuna']!r}"
            )

    @requires_lexicon
    def test_empty_term_returns_empty(self):
        assert YurumeinLexicon().reverse_lookup("") == []

    @requires_lexicon
    def test_nonexistent_term_returns_empty(self):
        assert YurumeinLexicon().reverse_lookup("superconductor") == []

    @requires_lexicon
    def test_accented_initial_character(self):
        # lower("Ibehabu") = "ibehabu"; the old prefix filter
        # "lower(garifuna) LIKE i%" misses accented I because i-acute != i.
        rows = YurumeinLexicon().reverse_lookup("Ibehabu")
        assert len(rows) > 0, "Accented initial character lookup returned empty"
        assert any(r["english"] == "Abundance" for r in rows)

    @requires_lexicon
    def test_second_variant_comma_separated(self):
        # Weiyali is the 2nd form in Aruguti, Weiyali, Ko (Grandfather)
        # and also in Weiriei, Weiyali (Old man).
        # The old whole-field normalize comparison would miss both.
        rows = YurumeinLexicon().reverse_lookup("Weiyali")
        assert len(rows) >= 1, "Second comma-separated variant returned empty"
        forms_per_row = [
            [f.strip() for f in (r["garifuna"] or "").split(",")]
            for r in rows
        ]
        assert all(
            any(f.lower() == "weiyali" for f in forms)
            for forms in forms_per_row
        ), "Non-matching row leaked: " + str([r["garifuna"] for r in rows])


# ─── INTEGRATION TESTS: live API ─────────────────────────────────────────────

class TestLiveRetrieval:
    @requires_service
    def test_water_uses_lexicon(self):
        answer = _ask_chat("How do you say water in Garifuna?")
        assert re.search(r"[Dd]una", answer), f"Expected 'Duna' in response, got: {answer[:200]}"

    @requires_service
    def test_water_quoted_uses_lexicon(self):
        answer = _ask_chat("How do you say 'water' in Garifuna?")
        assert re.search(r"[Dd]una", answer), f"Expected 'Duna' in response, got: {answer[:200]}"

    @requires_service
    def test_vowels_cites_serie_i(self):
        answer = _ask_chat("How do you pronounce Garifuna vowels?")
        assert re.search(r"[Ss]erie\s*I", answer), (
            f"Expected Libro Serie I citation, got: {answer[:300]}"
        )

    @requires_service
    def test_numbers_cites_serie_i(self):
        # LLM must either cite Libro Serie I or return the correct Garífuna numbers
        # (Aban, Biama, Üruwa, etc.) — the numbers themselves are uniquely from Serie I
        # and their presence confirms the right content was retrieved.
        answer = _ask_chat("Teach me basic Garifuna numbers")
        cites_serie_i = bool(re.search(r"[Ss]erie\s*I", answer))
        has_garifuna_numbers = bool(re.search(r"\bAban\b|\bBiama\b|\bÜruwa\b|\bGádürü\b", answer))
        assert cites_serie_i or has_garifuna_numbers, (
            f"Expected Serie I citation or Garífuna number terms (Aban/Biama/Üruwa), "
            f"got: {answer[:300]}"
        )

    @requires_service
    def test_dabula_definition(self):
        answer = _ask_chat('What does "Dábula" mean?')
        assert re.search(r"table|mesa", answer, re.I), (
            f"Expected 'table' in response, got: {answer[:200]}"
        )

    @requires_service
    def test_sentence_construction_retrieves_garifuna_ii(self):
        # Check the retrieval layer, not LLM output — GPT-4o may deflect
        # to "not enough information" non-deterministically even when the
        # right chunk is returned. The retrieval guarantee is what matters.
        matches = _ask_retrieval("Explain Garifuna sentence construction", k=5)
        sources = [m.get("metadata", {}).get("source", "") for m in matches]
        assert any("GARIFUNA II" in s.upper() for s in sources), (
            f"Expected at least one Libro Garifuna II chunk in top-5, "
            f"got sources: {sources}"
        )

    @requires_service
    def test_origins_attributes_to_author(self):
        answer = _ask_chat("What does Libro II say about Garifuna origins?")
        assert re.search(r"Enr[ií]quez|[Aa]uthor|[Aa]ccording", answer), (
            f"Expected attribution, got: {answer[:300]}"
        )

    @requires_service
    def test_missing_word_acknowledges_gap(self):
        answer = _ask_chat("What is the Garifuna word for 'superconductor'?")
        assert re.search(
            r"not find|no.*translat|don.t have|unable|insufficient|not.*available|"
            r"not documented|no information|not provide|do not provide|documents do not",
            answer, re.I
        ), f"Expected acknowledgement of gap, got: {answer[:200]}"

    @requires_service
    def test_pronunciation_retrieval_includes_serie_i(self):
        matches = _ask_retrieval("How do Garifuna vowels work?", k=5)
        sources = [m.get("metadata", {}).get("source", "") for m in matches]
        assert any("SERIE I" in s.upper() for s in sources), (
            f"Expected at least one Serie I chunk in top-5, got sources: {sources}"
        )

    @requires_service
    def test_chuluguau_badari_retrievable(self):
        # Regression: "Ch\u00fcl\u00fcgua\u00fc badari" (llegó tú novio) is
        # an exercise example on page 29 of Libro Garifuna II. keyword_search
        # finds it at rank 1 but the old hybrid_score = keyword * 0.3 buried it
        # below rank 20. Fix: keyword-only entries now score keyword * 0.6.
        sentence = "Ch\u00fcl\u00fcgua\u00fc badari"
        matches = _ask_retrieval(sentence, k=10)
        page29 = [
            m for m in matches
            if "GARIFUNA II" in m.get("metadata", {}).get("source", "").upper()
            and str(m.get("metadata", {}).get("page_label", "")) == "29"
        ]
        assert len(page29) >= 1, (
            "Page 29 of Libro Garifuna II not found in top-10. "
            "Keyword-only retrieval scoring may be broken."
        )
        combined = " ".join(m.get("content", "") for m in page29)
        assert re.search(r"badari", combined, re.I), (
            "Page-29 chunk does not contain expected sentence"
        )


    @requires_service
    def test_duguu_duration_not_falsely_attributed_to_garifuna_ii(self):
        # Libro Garífuna II mentions Dügü and Chugú only as a brief list
        # (page 6) and in a house-types table (page 105). It never states
        # the seven-day duration or the animal offerings. Those details
        # exist only in Libro Serie I, page 128.
        #
        # Failure mode: LLM attributes Serie I content to "Libro Garífuna II"
        # in the response body, following the user's framing.
        #
        # Test strategy: scan sentence by sentence. A sentence that mentions
        # Garífuna II AND a duration (seven / siete) WITHOUT a negation word
        # is a false attribution. A correct response either says Garífuna II
        # does not specify (negation present) or names Serie I as the source.
        answer = _ask_chat(
            "According to Libro Garifuna II, how long does the Duguu ceremony last?"
        )
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        for sentence in sentences:
            names_garifuna_ii = bool(re.search(r"gar[ií]funa\s+ii", sentence, re.I))
            claims_duration = bool(re.search(r"\bseven\b|\bsiete\b|7\s*days?", sentence, re.I))
            has_negation = bool(re.search(
                r"\b(?:not|no|never|doesn'?t|does\s+not|without|"
                r"not\s+specif|not\s+state|not\s+mention|no\s+specif)\b",
                sentence, re.I,
            ))
            assert not (names_garifuna_ii and claims_duration and not has_negation), (
                "LLM made an affirmative attribution of seven-day duration to "
                "Libro Garífuna II. That claim exists only in Libro Serie I, "
                f"page 128.\nOffending sentence: {sentence!r}\n"
                f"Full response: {answer[:400]}"
            )

    @requires_service
    def test_page_citations_are_plausible(self):
        answer = _ask_chat("How do Garifuna syllables work?")
        pages = [int(m) for m in re.findall(r"page\s+(\d+)", answer, re.I)]
        for p in pages:
            assert 1 <= p <= 200, f"Implausible page number: {p}"

    @requires_service
    def test_vowel_includes_u_umlaut(self):
        # Libro Serie I p.4 explicitly documents six vowels: a, e, i, o, u, ü.
        # The ü (guttural dental) must appear in the vowel explanation.
        # Page 4 is retrieved at rank 1 for vowel queries — omission is a generation error.
        answer = _ask_chat("What are the Garifuna vowels?")
        assert re.search(r"[üÜ]|u[\s-]?umlaut|guttural|diéresis", answer, re.I), (
            f"Response omits ü — Libro Serie I p.4 documents six vowels including ü:\n"
            f"{answer[:400]}"
        )

    @requires_service
    def test_duguu_building_is_gayunari_unconstrained(self):
        # Libro Garífuna II p.105 documents: Dügü → Gayunari; Chugú → Dabuyaba.
        # This test uses a general (non-source-constrained) query to prevent the specific
        # regression where the LLM incorrectly assigns Dabuyaba to the Dügü ceremony
        # (Dabuyaba is sourced from Diccionario as "ceremonial house" without ceremony
        # specificity, which caused mis-attribution when Garífuna II p.105 was not retrieved).
        answer = _ask_chat("What ceremonial building is used for the Dügü ceremony?")
        # Gayunari must be named as Dügü's ceremonial building (Diccionario p.43; Garífuna II p.105).
        # If the LLM incorrectly assigns Dabuyaba to Dügü, Gayunari will be absent.
        assert re.search(r"Gayunari", answer, re.I), (
            f"Expected Gayunari as Dügü's ceremonial building (Diccionario p.43; Garífuna II p.105).\n"
            f"Response: {answer[:400]}"
        )


# ─── UNIT + INTEGRATION TESTS: explicit page retrieval ───────────────────────

class TestPageRetrieval:
    # Regression tests for explicit book-and-page retrieval (Phase 2 fix).

    # -- unit: parser -----------------------------------------------------------

    def test_parse_page_request_serie_i_two_pages(self):
        from app.services.chatbot.lexical_parser import parse_page_request
        r = parse_page_request("Show me pages 21 and 24 of Libro Serie I")
        assert r.matched
        assert r.source_pattern == "%LIBRO SERIE I%"
        assert r.display_name == "Libro Serie I"
        assert 21 in r.pages
        assert 24 in r.pages

    def test_parse_page_request_garifuna_ii(self):
        from app.services.chatbot.lexical_parser import parse_page_request
        r = parse_page_request("What is on page 29 of Libro Garífuna II?")
        assert r.matched
        assert r.source_pattern == "%GARIFUNA II%"
        assert 29 in r.pages

    def test_parse_page_request_no_match_without_book(self):
        from app.services.chatbot.lexical_parser import parse_page_request
        r = parse_page_request("What is on page 21?")
        assert not r.matched

    def test_parse_page_request_ordinary_question_no_match(self):
        from app.services.chatbot.lexical_parser import parse_page_request
        for q in [
            "How do Garifuna vowels work?",
            "Teach me basic Garifuna numbers",
            "What does wüwüñü mean?",
            "Explain Garifuna sentence construction",
        ]:
            r = parse_page_request(q)
            assert not r.matched, f"parse_page_request incorrectly matched: {q!r}"

    # -- integration: live API --------------------------------------------------

    @requires_service
    def test_serie_i_pages_21_and_24_retrieved(self):
        answer = _ask_chat("Show me what is on pages 21 and 24 of Libro Serie I")
        # Pages 21 and 24 contain vocabulary (Bunidi, Biniwali, Funayumati, etc.).
        # The response must not falsely claim these pages are unavailable.
        assert not re.search(
            r"not.*available|not.*present|not.*found|not.*index|do not have.*page",
            answer, re.I
        ), f"Response falsely claims page unavailable: {answer[:300]}"

    @requires_service
    def test_serie_i_page_8_retrieved(self):
        answer = _ask_chat("What is on page 8 of Libro Serie I?")
        assert not re.search(
            r"not.*available|not.*present|not.*found|not.*index|do not have.*page",
            answer, re.I
        ), f"Response falsely claims page unavailable: {answer[:300]}"

    @requires_service
    def test_nonexistent_page_response(self):
        answer = _ask_chat("What is on page 999 of Libro Serie I?")
        assert re.search(
            r"not.*available|not.*present|not.*found|not.*index|unavailable|"
            r"does not exist|not in.*collection|those pages|page.*not",
            answer, re.I
        ), f"Expected not-found response for page 999, got: {answer[:300]}"

    @requires_service
    def test_ordinary_question_uses_existing_pipeline(self):
        # An ordinary grammar question (no book name + page number) must use
        # the standard hybrid retrieval path, not page-filtered lookup.
        from app.services.chatbot.lexical_parser import parse_page_request
        ordinary_queries = [
            "How do Garifuna vowels work?",
            "Explain Garifuna sentence construction",
            "Teach me basic Garifuna numbers",
        ]
        for q in ordinary_queries:
            r = parse_page_request(q)
            assert not r.matched, f"Ordinary question triggered page path: {q!r}"
        matches = _ask_retrieval("How do Garifuna vowels work?", k=5)
        assert len(matches) > 0, "No retrieval results for ordinary question"


# ─── PHASE 2 REGRESSION TESTS ────────────────────────────────────────────────

class TestPhase2Retrieval:
    """
    Regression suite for Phase 2 fixes: source-constrained retrieval,
    multi-part evidence, and citation integrity.
    """

    # ── Scenario 1: explicit pages (already covered by TestPageRetrieval; ──
    # ──             repeat here as a Phase-2 named regression) ──────────────

    @requires_service
    def test_scenario1_pages_21_and_24(self):
        # "Using Libro Serie I, show five vocabulary words from pages 21 and 24."
        # This query hits path 2.5 (parse_page_request matches) in global_retrieval_process.
        # We verify via the chat response, which reflects the actual workflow path.
        answer = _ask_chat(
            "Using Libro Serie I, show five vocabulary words from pages 21 and 24."
        )
        # Must NOT falsely say the pages are unavailable
        assert not re.search(
            r"not.*available|not.*found|not.*index|do not have.*page",
            answer, re.I,
        ), f"False unavailable claim: {answer[:300]}"
        # Must cite Serie I
        assert re.search(r"[Ss]erie\s*I", answer), (
            f"Expected Serie I citation: {answer[:300]}"
        )
        # Must reference pages 21 or 24 somewhere in the response
        assert re.search(r"\b(21|24)\b", answer), (
            f"Expected page 21 or 24 in response: {answer[:400]}"
        )

    # ── Scenario 2: source-constrained retrieval ─────────────────────────────

    @requires_service
    def test_scenario2_garifuna_ii_duguu_retrieval_constrained(self):
        # "According to Libro Garífuna II, what is Dügü?"
        # Path 2.6 (parse_book_request) restricts retrieval to Garífuna II in
        # global_retrieval_process. Verify via the Source: citations in the response —
        # the chat path runs the full workflow including source-constrained retrieval.
        # Note: /retrieval_check_hybrid bypasses path 2.6 and cannot test this.
        answer = _ask_chat("According to Libro Garifuna II, what is Duguu?")
        # Response must cite Garífuna II (not exclusively other sources)
        assert re.search(r"gar[ií]funa\s+ii", answer, re.I), (
            f"Expected Garífuna II attribution in response: {answer[:400]}"
        )
        # Must NOT attribute a Garífuna-II answer to only Diccionario or Serie I
        source_lines = re.findall(r"\*\*Source:\*\*\s*([^\n]+)", answer)
        if source_lines:
            all_non_garifuna_ii = all(
                "garifuna ii" not in s.lower() and "garífuna ii" not in s.lower()
                for s in source_lines
            )
            assert not all_non_garifuna_ii, (
                f"All Source: citations are from other books; Garífuna II missing.\n"
                f"Source lines: {source_lines}\nResponse: {answer[:400]}"
            )

    @requires_service
    def test_scenario2_garifuna_ii_duguu_no_false_disclaimer(self):
        # With source-constrained retrieval, the LLM should not falsely say
        # "Libro Garífuna II contains no information about Dügü."
        # Garífuna II p.5 and p.105 do mention Dügü / Dügü ceremonies.
        answer = _ask_chat(
            "According to Libro Garifuna II, what is Duguu?"
        )
        assert not re.search(
            r"(?:garífuna\s+ii|garifuna\s+ii).*(?:do not|does not|no information|not contain|not found|not available)",
            answer, re.I,
        ), (
            f"LLM falsely claimed Libro Garífuna II has no Dügü info: {answer[:400]}"
        )
        # Must cite Garífuna II
        assert re.search(r"gar[ií]funa\s+ii", answer, re.I), (
            f"Expected Garífuna II attribution: {answer[:300]}"
        )

    # ── Scenario 3: ceremony vs ceremonial building ──────────────────────────

    @requires_service
    def test_scenario3_duguu_is_ceremony_not_building(self):
        # "Explain Dügü, Chugú, Gayunari, and Dabuyaba."
        # Dügü and Chugú are CEREMONIES.
        # Gayunari and Dabuyaba are the CEREMONIAL BUILDINGS/SPACES.
        # The LLM must NOT define Dügü or Chugú primarily as "a house" or "a building."
        answer = _ask_chat("Explain Duguu, Chugu, Gayunari, and Dabuyaba.")
        sentences = re.split(r"(?<=[.!?])\s+", answer)
        for sentence in sentences:
            names_duguu_or_chugu = bool(
                re.search(r"\b[Dd][üu]g[üu][üu]?\b|\b[Cc]hugu\b|\b[Cc]hug[uú]\b", sentence)
            )
            calls_it_house = bool(
                re.search(
                    r"\b(?:a\s+house|a\s+building|a\s+structure|tipo\s+de\s+casa|casa\s+cerem)",
                    sentence, re.I,
                )
            )
            has_ceremony_context = bool(
                re.search(r"\bcerem|ritual|ancestral|spiritual|religious", sentence, re.I)
            )
            assert not (names_duguu_or_chugu and calls_it_house and not has_ceremony_context), (
                f"LLM defined a ceremony (Dügü/Chugú) as a building without ceremony "
                f"context.\nOffending sentence: {sentence!r}\nFull: {answer[:400]}"
            )

    @requires_service
    def test_scenario3_garifuna_ii_retrieved(self):
        # Libro Garífuna II, page 105 documents the specific ceremony→building mappings:
        #   Dügü (ceremony) → Gayunari (its ceremonial house)
        #   Chugú (ceremony) → Dabuyaba (its ceremonial house)
        # This test verifies those specific pairings via a source-constrained query
        # (path 2.6: 'Libro Garifuna II' triggers _source_constrained_hybrid_retrieve).
        # Generic ceremony/building distinction is insufficient; specific names must appear.
        answer = _ask_chat(
            "According to Libro Garifuna II, what ceremonial buildings are associated "
            "with the Duguu and Chugu ceremonies?"
        )
        # Both ceremonial building names must be in the response
        assert re.search(r"Gayunari", answer, re.I), (
            f"Response does not mention Gayunari (Dügü’s ceremonial building).\n"
            f"Response: {answer[:500]}"
        )
        assert re.search(r"Dabuyaba", answer, re.I), (
            f"Response does not mention Dabuyaba (Chugú’s ceremonial building).\n"
            f"Response: {answer[:500]}"
        )
        # Garífuna II must be cited (source-constrained retrieval should deliver it)
        assert re.search(r"gar[íí]funa\s+ii", answer, re.I), (
            f"Response does not cite Garífuna II.\nResponse: {answer[:400]}"
        )
        # Page 105 specifically — this is the documented source for these mappings
        assert re.search(r"\b105\b", answer), (
            f"Response does not cite page 105 of Garífuna II — specific source unverified.\n"
            f"Response: {answer[:500]}"
        )

    # ── Scenario 4: multi-part evidence retrieval ────────────────────────────

    @requires_service
    def test_scenario4_vowels_and_vocab_retrieval(self):
        # 'Teach me the Garífuna vowels and give five documented examples.'
        # Verifies that global_retrieval_process (path 4b: multi-part decomposition)
        # retrieves evidence for BOTH sub-queries and surfaces both topics in the response.
        # Note: /retrieval_check_hybrid BYPASSES path 4b — this test must use _ask_chat.
        answer = _ask_chat(
            "Teach me the Garifuna vowels and give five documented vocabulary examples."
        )
        # Topic 1: vowel content from Serie I must appear
        has_vowel_content = bool(re.search(
            r"\bvowel|\bvocal|\bpronunci|\ba,\s*e,\s*i", answer, re.I
        ))
        # Topic 2: vocabulary content must appear — either a numbered word list (with or
        # without bold/definitions), a Diccionario/Lexicon citation, or an explicit
        # statement that vocabulary was retrieved but translations are unavailable.
        has_vocab_content = bool(
            re.search(r"\d+\.\s+\*\*[A-Za-záéíóúüñÁÉÍÓÚÜÑ]", answer) or
            re.search(r"[Dd]iccionario|[Ll]exicon", answer) or
            re.search(r"\d+\.\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñA-Z]", answer) or
            re.search(r"vocabular|word|palabra|translat", answer, re.I)
        )
        assert has_vowel_content, (
            f"Multi-part retrieval did not produce vowel content in response:\n{answer[:400]}"
        )
        assert has_vocab_content, (
            f"Multi-part retrieval did not produce vocabulary content in response:\n{answer[:400]}"
        )
        # Serie I must be cited — it is the vowel source
        assert re.search(r"[Ss]erie\s*I", answer), (
            f"Serie I not cited — vowel evidence may not have been retrieved:\n{answer[:400]}"
        )
        # ü must appear — Libro Serie I p.4 lists six vowels including ü
        assert re.search(r"[üÜ]|u[\s-]?umlaut|guttural", answer, re.I), (
            f"Response omits ü — Serie I p.4 is retrieved at rank 1 with ü listed:\n"
            f"{answer[:400]}"
        )

    @requires_service
    def test_scenario4_no_fabricated_vocabulary(self):
        # The LLM must not present vocabulary items with fabricated definitions.
        # A citation for the vowel section (Serie I) does NOT cover the vocabulary section.
        # Vocabulary words paired with definitions must be backed by Diccionario/Lexicon.
        answer = _ask_chat(
            "Teach me the Garifuna vowels and give five documented vocabulary examples."
        )
        # Detect "1. **Word** — Definition" items (vocabulary with inline definitions)
        has_defined_vocab = bool(re.search(
            r"\d+\.\s+\*\*\w[^*]+\*\*\s*[-–—]\s*\w",
            answer,
        ))
        # Diccionario or Lexicon must be cited when vocabulary definitions are provided
        has_diccionario_citation = bool(re.search(
            r"[Dd]iccionario|[Ll]exicon", answer
        ))
        acknowledges_limitation = bool(re.search(
            r"not.*translat|no.*translat|not.*provid|not.*definit|"
            r"without.*translat|unable.*translat|don.?t.*have.*translat",
            answer, re.I,
        ))
        assert not has_defined_vocab or has_diccionario_citation or acknowledges_limitation, (
            f"Response provides vocabulary definitions without Diccionario/Lexicon citation.\n"
            f"A citation from Serie I does not verify vocabulary — Serie I is a grammar book.\n"
            f"Response (first 600 chars):\n{answer[:600]}"
        )


    @requires_service
    def test_scenario4_vocabulary_five_sourced_or_states_unavailable(self):
        # When asked for five vocabulary examples from the Diccionario, the response must
        # either: (a) provide five word-meaning pairs backed by retrieved context, OR
        # (b) explicitly state when fewer than five with definitions are available.
        # Words listed without definitions do not count.
        # Query uses source-constrained path 2.6 (Diccionario named) to reliably
        # retrieve bilingual dictionary entries — avoids the lexical-parser path.
        answer = _ask_chat(
            "According to the Diccionario Garifuna, show me five example words and what they mean."
        )
        # Count items that have an actual definition (not 'not provided' or similar)
        all_items = re.findall(
            r"\d+\.\s+\*\*[^*]+\*\*\s*[-–—]\s*([^\n]+)",
            answer,
        )
        sourced = [
            d for d in all_items
            if not re.search(
                r"not provided|not available|no translation|context does not|not in.*context",
                d, re.I,
            )
        ]
        acknowledges_fewer = bool(re.search(
            r"only\s+\d+|fewer than|could only|just \d+|\d+.*available|available.*\d+"
            r"|do not provide|not provide|not available|insufficient|does not contain|no.*translat",
            answer, re.I,
        ))
        assert len(sourced) >= 5 or acknowledges_fewer, (
            f"Expected 5 sourced vocabulary pairs or explicit acknowledgment of fewer.\n"
            f"Found {len(sourced)} sourced items (of {len(all_items)} listed).\n"
            f"Response: {answer[:600]}"
        )

    # ── Scenario 5: specific page verification ───────────────────────────────

    @requires_service
    def test_scenario5_page_29_garifuna_ii(self):
        # "Verify Chülüguaü badari on page 29 of Libro Garífuna II."
        # Already covered by TestLiveRetrieval::test_chuluguau_badari_retrievable
        # but repeated here with the explicit page form.
        answer = _ask_chat(
            "Verify the sentence on page 29 of Libro Garifuna II that starts with Chulugua badari."
        )
        assert not re.search(
            r"not.*available|not.*found|not.*index|do not have.*page",
            answer, re.I,
        ), f"Page 29 falsely reported unavailable: {answer[:300]}"
        assert re.search(r"[Gg]ar[ií]funa\s+II|page\s+29", answer), (
            f"Expected page 29 / Garífuna II reference: {answer[:300]}"
        )

    # ── Scenario 6: nonexistent page → precise unavailable response ──────────

    @requires_service
    def test_scenario6_nonexistent_page_precise_response(self):
        # Already tested in TestPageRetrieval but repeated here for Phase 3 naming.
        answer = _ask_chat("What is on page 999 of Libro Serie I?")
        assert re.search(
            r"not.*available|not.*present|not.*found|not.*index|unavailable|"
            r"does not exist|not in.*collection|those pages|page.*not",
            answer, re.I,
        ), f"Expected not-found response for page 999, got: {answer[:300]}"
        # Must NOT fabricate a citation for a nonexistent page.
        assert not re.search(r"\*\*Source:.*page\s+999", answer, re.I), (
            f"Fabricated citation for nonexistent page: {answer[:300]}"
        )

    # ── Unit: parse_book_request ─────────────────────────────────────────────

    def test_parse_book_request_garifuna_ii(self):
        from app.services.chatbot.lexical_parser import parse_book_request
        r = parse_book_request("According to Libro Garifuna II, what is Duguu?")
        assert r.matched
        assert r.source_pattern == "%GARIFUNA II%"
        assert r.display_name == "Libro Garífuna II"

    def test_parse_book_request_serie_i(self):
        from app.services.chatbot.lexical_parser import parse_book_request
        r = parse_book_request("What does Libro Serie I say about vowels?")
        assert r.matched
        assert r.source_pattern == "%LIBRO SERIE I%"

    def test_parse_book_request_no_match_without_book(self):
        from app.services.chatbot.lexical_parser import parse_book_request
        r = parse_book_request("Explain Garifuna vowels")
        assert not r.matched

    def test_parse_book_request_no_match_ambiguous_two_books(self):
        from app.services.chatbot.lexical_parser import parse_book_request
        r = parse_book_request("Compare Libro Serie I with Libro Garifuna II")
        assert not r.matched, "Two-book query should not trigger source constraint"

    def test_parse_book_request_defers_to_page_request(self):
        # When pages are specified, parse_page_request handles it; parse_book_request
        # still returns matched (nodes.py uses page_result first).
        from app.services.chatbot.lexical_parser import parse_book_request, parse_page_request
        q = "Show me page 21 of Libro Serie I"
        page_r = parse_page_request(q)
        book_r = parse_book_request(q)
        assert page_r.matched, "Page request should match"
        assert book_r.matched, "Book request also matches; nodes.py uses page path first"

    def test_validate_citations_exact_page_check(self):
        # Unit test: _validate_citations must check exact (book, page) pairs,
        # detect both **Source:** and [Source: ...] formats, and replace
        # unverified citations in-place with [UNVERIFIED: ...].
        from app.services.utils.RAG_pipeline.workflow.nodes import Nodes as WorkflowNodes
        inst = object.__new__(WorkflowNodes)  # bypass __init__; only testing the method

        retrieved_docs = [{
            "metadata": {
                "source": "app/services/utils/RAG_pipeline/data/new_docs/LIBRO SERIE I.pdf",
                "page_label": "21",
            }
        }]

        # Correct book + correct page → unchanged
        result = inst._validate_citations("**Source:** Libro Serie I — page 21", retrieved_docs)
        assert "[UNVERIFIED" not in result, f"Valid citation was incorrectly marked: {result}"

        # Correct book + wrong page → UNVERIFIED
        result = inst._validate_citations("**Source:** Libro Serie I — page 99", retrieved_docs)
        assert "[UNVERIFIED" in result, f"Wrong page not flagged: {result}"

        # Wrong book entirely → UNVERIFIED
        result = inst._validate_citations("**Source:** Libro Garífuna II — page 21", retrieved_docs)
        assert "[UNVERIFIED" in result, f"Wrong book not flagged: {result}"

        # [Source: ...] bracket format, wrong page → UNVERIFIED
        result = inst._validate_citations("[Source: Libro Serie I — page 99]", retrieved_docs)
        assert "[UNVERIFIED" in result, f"Bracket format not checked: {result}"

        # [Source: ...] bracket format, correct page → unchanged
        result = inst._validate_citations("[Source: Libro Serie I — page 21]", retrieved_docs)
        assert "[UNVERIFIED" not in result, f"Valid bracket citation was incorrectly flagged: {result}"


# ─── PHASE 5 REGRESSION TESTS ────────────────────────────────────────────────

class TestPhase5Regression:
    """
    Regression suite for all 10 original user test questions plus paraphrases.
    Covers: direct-answer relevance, cultural accuracy, source grounding,
    citation integrity, beginner teaching usefulness.
    """

    # ── Unit: Phase 2 parser fix ─────────────────────────────────────────────

    def test_how_do_i_say_parser_matches(self):
        r = parse_lexical_request("How do I say water in Garifuna?")
        assert r.matched, "New 'how do I say' pattern must match"
        assert r.term == "water"
        assert r.language == "english"
        assert r.pattern == "how_do_i_say"

    def test_how_do_i_say_parser_case_insensitive(self):
        r = parse_lexical_request("how do i say house in garifuna")
        assert r.matched
        assert r.term == "house"

    def test_how_do_i_say_does_not_match_pronunciation_question(self):
        r = parse_lexical_request("How do I say the alphabet in Garifuna correctly?")
        # This should match (term = "the alphabet") — verify it doesn't explode
        # and returns a usable term
        assert r.matched
        assert "alphabet" in r.term

    # ── Unit: Phase 1 source string fix ─────────────────────────────────────

    def test_build_lexicon_context_source_string(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import Nodes as WorkflowNodes
        inst = object.__new__(WorkflowNodes)
        rows = [{"garifuna": "Duna"}]
        ctx = inst._build_lexicon_context("water", "english", rows)
        assert "Source: Yurumein Lexicon" in ctx, (
            f"_build_lexicon_context must output 'Source: Yurumein Lexicon' (not underscore).\n"
            f"Got: {ctx[:200]}"
        )
        assert "yurumein_lexicon" not in ctx, (
            f"Underscore form must not appear in lexicon context output: {ctx[:200]}"
        )

    def test_build_definition_context_source_string(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import Nodes as WorkflowNodes
        inst = object.__new__(WorkflowNodes)
        rows = [{"garifuna": "Duna", "english": "Water", "spanish": "Agua"}]
        ctx = inst._build_definition_context("Duna", rows)
        assert "Source: Yurumein Lexicon" in ctx, (
            f"_build_definition_context must output 'Source: Yurumein Lexicon'.\n"
            f"Got: {ctx[:200]}"
        )
        assert "yurumein_lexicon" not in ctx, (
            f"Underscore form must not appear in definition context output: {ctx[:200]}"
        )

    def test_lexicon_path_clears_retrieved_docs_key(self):
        # The lexical path must return retrieved_docs=[] to prevent stale docs
        # from prior turns triggering [UNVERIFIED: Yurumein Lexicon] in validation.
        # This is a structural check on the return value, not a live API call.
        import asyncio
        from app.services.utils.RAG_pipeline.workflow.nodes import Nodes as WorkflowNodes
        inst = object.__new__(WorkflowNodes)
        # Patch the lexicon lookup to avoid a DB call
        class _FakeLexicon:
            def lookup(self, term, lang):
                return [{"garifuna": "Duna"}]
        import app.services.utils.RAG_pipeline.workflow.nodes as _nodes_mod
        original = _nodes_mod.YurumeinLexicon
        _nodes_mod.YurumeinLexicon = lambda: _FakeLexicon()
        try:
            state = {
                "user_query": "How do you say water in Garifuna?",
                "messages": [],
                "retrieved_docs": [{"metadata": {"source": "some/old/RAG/doc.pdf"}}],
            }
            result = asyncio.get_event_loop().run_until_complete(
                inst.global_retrieval_process(state)
            )
            assert "retrieved_docs" in result, (
                "Lexical path must explicitly set retrieved_docs to clear stale state"
            )
            assert result["retrieved_docs"] == [], (
                f"Lexical path must set retrieved_docs=[] to prevent stale UNVERIFIED tags.\n"
                f"Got: {result.get('retrieved_docs')}"
            )
        finally:
            _nodes_mod.YurumeinLexicon = original

    # ── Live: Q1 — "How do you say hello in Garífuna?" ───────────────────────

    @requires_service
    def test_q1_hello_mabuiga_or_honest_gap(self):
        # Original question 1: "How do you say hello in Garífuna?"
        # Mábuiga is documented at Diccionario Garífuna p.174 as "Saludes → Mábuiga".
        # If retrieval succeeds, Mábuiga must appear. If not, the response must
        # acknowledge the gap — not fabricate or substitute another expression.
        answer = _ask_chat("How do you say hello in Garifuna?")
        has_mabuiga = bool(re.search(r"m[áa]buiga|mabuiga", answer, re.I))
        acknowledges_gap = bool(re.search(
            r"not.*find|no.*translat|don.?t have|not.*available|not.*document|"
            r"not.*approv|not.*source|no.*word.*for\s+hello|documents do not",
            answer, re.I,
        ))
        # Must never present a fabricated or unverified expression without a gap disclaimer
        assert has_mabuiga or acknowledges_gap, (
            f"Response neither contains Mábuiga nor acknowledges the source gap.\n"
            f"Response: {answer[:400]}"
        )
        # EID tags must never appear
        assert not re.search(r"\[EID-\d+:", answer), (
            f"Internal EID tag leaked into response: {answer[:400]}"
        )

    @requires_service
    def test_q1_hello_paraphrase_garifuna_word_for(self):
        # Paraphrase: "What is the Garífuna word for hello?"
        answer = _ask_chat("What is the Garifuna word for hello?")
        has_mabuiga = bool(re.search(r"m[áa]buiga|mabuiga", answer, re.I))
        acknowledges_gap = bool(re.search(
            r"not.*find|no.*translat|don.?t have|not.*available|not.*document|"
            r"documents do not|not.*source|no.*equivalent",
            answer, re.I,
        ))
        assert has_mabuiga or acknowledges_gap, (
            f"Paraphrase response neither contains Mábuiga nor acknowledges gap.\n"
            f"Response: {answer[:400]}"
        )

    @requires_service
    def test_q1_hello_no_fabricated_unverified_expression(self):
        # The response must not present a Garífuna expression for "hello" without
        # a cited source. General-knowledge disclaimers ("commonly used") are not
        # sufficient — only retrieved evidence counts.
        answer = _ask_chat("How do you say hello in Garifuna?")
        # If a non-Mábuiga Garífuna expression is claimed (e.g., "Buiti"), it must
        # either cite a retrieved source or be wrapped in a fallback prefix.
        garifuna_greetings = re.findall(
            r'\b(?:buiti|lidan|seremei|sereina|wagucha|hounisa)\b',
            answer, re.I,
        )
        for expr in garifuna_greetings:
            # Each such expression must be accompanied by a source citation or
            # a "general knowledge" fallback prefix somewhere in the answer
            has_citation = bool(re.search(r"\*\*Source:\*\*|\[Source:", answer))
            has_fallback = bool(re.search(
                r"general knowledge|my general knowledge|not.*approv|may not be.*verify|"
                r"not.*document|may be.*used.*community|community.*expression",
                answer, re.I,
            ))
            assert has_citation or has_fallback, (
                f"Expression '{expr}' presented without retrieved-source citation or "
                f"fallback prefix.\nResponse: {answer[:500]}"
            )

    # ── Live: Q2 — "How do I introduce myself in Garífuna?" ─────────────────

    @requires_service
    def test_q2_introduce_myself_niribei_or_honest_gap(self):
        # Original question 2: "How do I introduce myself in Garífuna?"
        # Libro Garífuna II p.108 documents: "Le niribei: Juan Martínez"
        # (pattern for stating one's name).
        answer = _ask_chat("How do I introduce myself in Garifuna?")
        has_niribei = bool(re.search(r"niribei|le niribei", answer, re.I))
        acknowledges_gap = bool(re.search(
            r"not.*find|not.*available|not.*document|documents do not|"
            r"no.*information|insufficient|not.*source",
            answer, re.I,
        ))
        assert has_niribei or acknowledges_gap, (
            f"Response neither contains 'Le niribei' (Garífuna II p.108) nor acknowledges gap.\n"
            f"Response: {answer[:400]}"
        )
        assert not re.search(r"\[EID-\d+:", answer), (
            f"Internal EID tag leaked into response: {answer[:400]}"
        )

    @requires_service
    def test_q2_introduce_paraphrase_my_name_is(self):
        # Paraphrase: "How do Garífuna people introduce themselves?"
        # This unambiguously targets self-introduction content.
        # Libro Garífuna II p.108 documents "Le niribei: [name]" for name exchange.
        answer = _ask_chat("How do Garifuna people introduce themselves?")
        has_niribei = bool(re.search(r"niribei|le niribei", answer, re.I))
        has_cited_response = bool(re.search(r"\*\*Source:\*\*|\[Source:", answer))
        acknowledges_gap = bool(re.search(
            r"not.*find|not.*available|not.*document|documents do not|"
            r"no.*information|insufficient",
            answer, re.I,
        ))
        assert has_niribei or has_cited_response or acknowledges_gap, (
            f"Response neither contains 'niribei', cites a source, nor acknowledges gap.\n"
            f"Response: {answer[:400]}"
        )

    @requires_service
    def test_q2_introduce_does_not_reference_pageant(self):
        # p.116 is a formal pageant speech; a beginner asking to introduce themselves
        # should NOT receive content about beauty pageant acceptance speeches.
        answer = _ask_chat("How do I introduce myself in Garifuna?")
        assert not re.search(
            r"pageant|queen|candidat|coronat|beauty.*contest",
            answer, re.I,
        ), (
            f"Response references a beauty pageant context, which is irrelevant to "
            f"personal self-introduction.\nResponse: {answer[:400]}"
        )

    # ── Live: Q3 — "Where do Garífuna come from?" ────────────────────────────

    @requires_service
    def test_q3_origins_mentions_st_vincent_or_yurumein(self):
        # Original question 3: "Where do Garífuna come from?"
        # Answer must lead with geographic origin (St. Vincent / Yurumein),
        # not lead with linguistic composition percentages.
        answer = _ask_chat("Where do Garifuna come from?")
        has_geography = bool(re.search(
            r"st\.?\s*vincent|yurumein|saint\s+vincent|caribbean|central\s+america|"
            r"honduras|belize|guatemala|nicaragua",
            answer, re.I,
        ))
        assert has_geography, (
            f"Origins response does not mention St. Vincent/Yurumein or Central America.\n"
            f"Response: {answer[:400]}"
        )
        assert not re.search(r"\[EID-\d+:", answer), (
            f"Internal EID tag leaked into response: {answer[:400]}"
        )

    @requires_service
    def test_q3_origins_paraphrase_history(self):
        # Paraphrase: "Tell me about the history of the Garífuna people"
        answer = _ask_chat("Tell me about the history of the Garifuna people")
        has_geography = bool(re.search(
            r"st\.?\s*vincent|yurumein|saint\s+vincent|caribbean|arawak|carib|"
            r"west\s+african|1797|central\s+america",
            answer, re.I,
        ))
        assert has_geography, (
            f"History response does not mention geographic/ethnic origins.\n"
            f"Response: {answer[:400]}"
        )

    @requires_service
    def test_q3_origins_does_not_lead_with_linguistics_only(self):
        # The first paragraph must not exclusively discuss linguistic percentages
        # (e.g., "60% Arawak, 20% Carib") without first mentioning people/geography.
        answer = _ask_chat("Where do Garifuna come from?")
        paragraphs = answer.split("\n\n")
        first_para = paragraphs[0] if paragraphs else answer
        leads_with_pure_linguistics = bool(
            re.search(r"\d+\s*%.*arawak|\d+\s*%.*carib|percent.*vocab", first_para, re.I)
        ) and not bool(re.search(
            r"st\.?\s*vincent|yurumein|island|caribbean|africa|people|group|community",
            first_para, re.I,
        ))
        assert not leads_with_pure_linguistics, (
            f"Origins response leads with linguistic percentages without mentioning "
            f"geographic/ethnic context in the first paragraph.\n"
            f"First paragraph: {first_para[:300]}"
        )

    # ── Live: Citation integrity across all 3 key questions ─────────────────

    @requires_service
    def test_no_eid_tags_in_hello_response(self):
        answer = _ask_chat("How do you say hello in Garifuna?")
        assert not re.search(r"\[EID-\d+:", answer), (
            f"EID tag leaked into hello response: {answer[:400]}"
        )

    @requires_service
    def test_no_eid_tags_in_introduction_response(self):
        answer = _ask_chat("How do I introduce myself in Garifuna?")
        assert not re.search(r"\[EID-\d+:", answer), (
            f"EID tag leaked into introduction response: {answer[:400]}"
        )

    @requires_service
    def test_no_eid_tags_in_origins_response(self):
        answer = _ask_chat("Where do Garifuna come from?")
        assert not re.search(r"\[EID-\d+:", answer), (
            f"EID tag leaked into origins response: {answer[:400]}"
        )

    # ── Retrieval: Q1/Q2/Q3 retrieve relevant sources ───────────────────────

    @requires_service
    def test_q1_hello_retrieval_includes_diccionario(self):
        # "How do you say hello" should retrieve Diccionario chunks (Mábuiga at p.174)
        matches = _ask_retrieval("How do you say hello in Garifuna?", k=5)
        sources = [m.get("metadata", {}).get("source", "") for m in matches]
        assert any("DICCIONARIO" in s.upper() for s in sources), (
            f"Expected Diccionario chunk for hello query (Mábuiga at p.174).\n"
            f"Got sources: {sources}"
        )

    @requires_service
    def test_q2_introduce_retrieval_includes_garifuna_ii(self):
        # Introduction queries should surface Libro Garífuna II (p.108: Le niribei)
        matches = _ask_retrieval("How do I introduce myself in Garifuna?", k=5)
        sources = [m.get("metadata", {}).get("source", "") for m in matches]
        assert any("GARIFUNA II" in s.upper() for s in sources), (
            f"Expected Garífuna II chunk for introduction query (p.108: Le niribei).\n"
            f"Got sources: {sources}"
        )

    @requires_service
    def test_q3_origins_retrieval_includes_garifuna_ii(self):
        # Origins queries should surface Libro Garífuna II (historical background)
        matches = _ask_retrieval("Where do Garifuna come from?", k=5)
        sources = [m.get("metadata", {}).get("source", "") for m in matches]
        assert any("GARIFUNA II" in s.upper() for s in sources), (
            f"Expected Garífuna II chunk for origins query.\n"
            f"Got sources: {sources}"
        )
