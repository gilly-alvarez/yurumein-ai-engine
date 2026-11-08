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
    def test_exact_match_dabula(self):
        rows = YurumeinLexicon().reverse_lookup("Dábula")
        assert len(rows) > 0
        assert all(
            normalize_term(r["garifuna"]) == normalize_term("Dábula") for r in rows
        )

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
    def test_sentence_construction_cites_garifuna_ii(self):
        answer = _ask_chat("Explain Garifuna sentence construction")
        assert re.search(r"[Gg]ar[ií]funa\s+II|[Ll]ibro\s+II", answer), (
            f"Expected Libro Garifuna II citation, got: {answer[:300]}"
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
            r"not find|no.*translat|don.t have|unable|insufficient|not.*available|not documented|no information",
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
    def test_page_citations_are_plausible(self):
        answer = _ask_chat("How do Garifuna syllables work?")
        pages = [int(m) for m in re.findall(r"page\s+(\d+)", answer, re.I)]
        for p in pages:
            assert 1 <= p <= 200, f"Implausible page number: {p}"
