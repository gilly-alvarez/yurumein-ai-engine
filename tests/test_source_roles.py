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
        answer = _ask_chat("Teach me basic Garifuna numbers")
        assert re.search(r"[Ss]erie\s*I", answer), (
            f"Expected Libro Serie I citation, got: {answer[:300]}"
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
