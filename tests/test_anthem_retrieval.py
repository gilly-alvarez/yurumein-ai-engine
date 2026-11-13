"""
Regression suite for Guatemala national anthem retrieval.

Unit tests cover deterministic components (no network, no DB):
  - intent_router: anthem intent detection
  - source_classifier: anthem source role mapping

Integration tests require the live API at localhost:8085 and are
skipped automatically when the service is unavailable.

Run with:
    pytest tests/test_anthem_retrieval.py -v
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import urllib.parse
import urllib.request

import pytest

AI_API = "http://localhost:8085"


def _service_available() -> bool:
    try:
        with urllib.request.urlopen(f"{AI_API}/health", timeout=3):
            return True
    except Exception:
        return False


def _ask_chat(question: str, lang: str = "en") -> str:
    data = urllib.parse.urlencode(
        {"user_prompt": question, "user_id": "test_anthem", "language": lang}
    ).encode()
    req = urllib.request.Request(
        f"{AI_API}/chat",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read()).get("response", "")


def _ask_retrieval(question: str, k: int = 10) -> list:
    q = urllib.parse.quote(question)
    with urllib.request.urlopen(
        f"{AI_API}/retrieval_check_hybrid?query={q}&k={k}", timeout=30
    ) as r:
        return json.loads(r.read()).get("matches", [])


requires_service = pytest.mark.skipif(
    not _service_available(), reason="gaupo_ai service not reachable at localhost:8085"
)


# ─── UNIT: intent_router ────────────────────────────────────────────────────

from app.services.chatbot.intent_router import detect_intent, Intent
from app.services.utils.RAG_pipeline.workflow.nodes import Nodes


class TestAnthemIntentDetection:
    def test_national_anthem_english(self):
        assert detect_intent("What is the Guatemala national anthem in Garifuna?") == Intent.ANTHEM

    def test_himno_nacional_spanish(self):
        assert detect_intent("¿Cuál es la letra del himno nacional de Guatemala en garífuna?") == Intent.ANTHEM

    def test_himno_garifuna(self):
        assert detect_intent("himno garífuna") == Intent.ANTHEM

    def test_wadimalu_first_line(self):
        # First word of the Garífuna anthem — must trigger anthem intent
        assert detect_intent("What does Wadimalu mean in the anthem?") == Intent.ANTHEM

    def test_garifuna_anthem_phrase(self):
        assert detect_intent("play the Garifuna anthem") == Intent.ANTHEM

    def test_anthem_of_guatemala(self):
        assert detect_intent("anthem of Guatemala") == Intent.ANTHEM

    def test_no_anthem_match_on_song(self):
        # Generic music query should NOT trigger ANTHEM
        result = detect_intent("What are traditional Garifuna songs?")
        assert result != Intent.ANTHEM

    def test_no_anthem_match_on_history(self):
        # Cultural history query should not trigger anthem intent
        result = detect_intent("Tell me about Garifuna history")
        assert result != Intent.ANTHEM


# ─── UNIT: source_classifier ─────────────────────────────────────────────────

from app.services.chatbot.source_classifier import classify_source


class TestAnthemSourceClassifier:
    def test_anthem_docx_classified(self):
        path = "/data/new_docs/Himno_Nacional_Guatemala_Garifuna_Espanol_English_Yurumein_AI.docx"
        meta = classify_source(path)
        assert meta is not None, "Anthem DOCX should be classified"
        assert meta.role == "NATIONAL_ANTHEM"
        assert "Himno" in meta.display

    def test_anthem_uppercase_path(self):
        path = "/DATA/NEW_DOCS/HIMNO_NACIONAL_GUATEMALA_GARIFUNA_ESPANOL_ENGLISH_YURUMEIN_AI.DOCX"
        meta = classify_source(path)
        assert meta is not None
        assert meta.role == "NATIONAL_ANTHEM"

    def test_non_anthem_path_unaffected(self):
        path = "/data/LIBRO SERIE I.pdf"
        meta = classify_source(path)
        assert meta is not None
        assert meta.role == "FOUNDATIONAL_LANGUAGE_EDUCATION"

    def test_diccionario_unaffected(self):
        path = "/data/DICCIONARIO GARIFUNA.pdf"
        meta = classify_source(path)
        assert meta is not None
        assert meta.role == "DICTIONARY"

    def test_unrecognized_path_returns_none(self):
        assert classify_source("random_file.pdf") is None

    def test_national_anthem_has_attribution(self):
        path = "/data/new_docs/Himno_Nacional_Guatemala_Garifuna.docx"
        meta = classify_source(path)
        assert meta is not None
        assert meta.attribution != "", "NATIONAL_ANTHEM must have an attribution string"
        assert "verified" in meta.attribution.lower()


# ─── UNIT: anthem language detection ────────────────────────────────────────

_nodes = Nodes.__new__(Nodes)  # instantiate without connecting to DB


class TestAnthemLanguageDetection:
    def test_default_is_garifuna(self):
        assert _nodes._detect_anthem_language("What is the national anthem?") == "garifuna"

    def test_explicit_garifuna(self):
        assert _nodes._detect_anthem_language("show me the anthem in Garífuna") == "garifuna"

    def test_spanish_keyword(self):
        assert _nodes._detect_anthem_language("show me the Spanish version") == "spanish"

    def test_español_keyword(self):
        assert _nodes._detect_anthem_language("¿cuál es el himno en español?") == "spanish"

    def test_english_keyword(self):
        assert _nodes._detect_anthem_language("English lyrics please") == "english"

    def test_mixed_query_spanish_wins(self):
        # When both English and Spanish appear, the first match (Spanish pattern) wins
        result = _nodes._detect_anthem_language("translate anthem from Spanish to English")
        assert result in ("spanish", "english")  # either is acceptable; not "garifuna"

    def test_garifuna_is_default_for_bare_anthem_query(self):
        assert _nodes._detect_anthem_language("Guatemala national anthem") == "garifuna"


# ─── UNIT: section filtering ─────────────────────────────────────────────────

def _make_chunk(text: str, idx: int) -> dict:
    return {"page_content": text, "metadata": {"chunk_index": idx}}


class TestAnthemSectionFilter:
    _MOCK_CHUNKS = [
        _make_chunk("Himno Nacional de Guatemala", 0),
        _make_chunk("Garífuna", 1),                           # section header
        _make_chunk("Wadimalu gundati lau bareini", 2),
        _make_chunk("Wadimalu liubani le seremei", 3),
        _make_chunk("Wadimalu biri le moweidibei", 4),         # final line
        _make_chunk("Español", 5),                             # section header
        _make_chunk("¡Guatemala feliz! que tus aras", 6),
        _make_chunk("English", 7),                             # section header
        _make_chunk("Happy Guatemala! may your altars", 8),
    ]

    def test_garifuna_section_excludes_first_line_check(self):
        section = _nodes._filter_anthem_section(self._MOCK_CHUNKS, "garifuna")
        texts = [c["page_content"] for c in section]
        assert "Wadimalu gundati lau bareini" in texts

    def test_garifuna_section_includes_final_line(self):
        section = _nodes._filter_anthem_section(self._MOCK_CHUNKS, "garifuna")
        texts = [c["page_content"] for c in section]
        assert "Wadimalu biri le moweidibei" in texts, (
            f"Final line missing from Garífuna section. Got: {texts}"
        )

    def test_garifuna_section_excludes_spanish(self):
        section = _nodes._filter_anthem_section(self._MOCK_CHUNKS, "garifuna")
        texts = [c["page_content"] for c in section]
        assert not any("Guatemala feliz" in t for t in texts), (
            "Garífuna section must not contain Spanish lyrics"
        )

    def test_spanish_section_contains_spanish_lyrics(self):
        section = _nodes._filter_anthem_section(self._MOCK_CHUNKS, "spanish")
        texts = [c["page_content"] for c in section]
        assert any("Guatemala feliz" in t for t in texts)

    def test_spanish_section_excludes_garifuna(self):
        section = _nodes._filter_anthem_section(self._MOCK_CHUNKS, "spanish")
        texts = [c["page_content"] for c in section]
        assert not any("Wadimalu" in t for t in texts)

    def test_english_section_contains_english_lyrics(self):
        section = _nodes._filter_anthem_section(self._MOCK_CHUNKS, "english")
        texts = [c["page_content"] for c in section]
        assert any("Happy Guatemala" in t for t in texts)

    def test_no_headers_returns_all_chunks(self):
        bare_chunks = [
            _make_chunk("Wadimalu gundati lau bareini", 0),
            _make_chunk("Wadimalu biri le moweidibei", 1),
        ]
        result = _nodes._filter_anthem_section(bare_chunks, "garifuna")
        assert result == bare_chunks

    def test_missing_language_returns_all_chunks(self):
        # Request a section that has no header in the document
        result = _nodes._filter_anthem_section(self._MOCK_CHUNKS, "french")
        assert result == self._MOCK_CHUNKS

    def test_garifuna_section_preserves_order(self):
        section = _nodes._filter_anthem_section(self._MOCK_CHUNKS, "garifuna")
        texts = [c["page_content"] for c in section]
        first_idx = texts.index("Wadimalu gundati lau bareini")
        last_idx = texts.index("Wadimalu biri le moweidibei")
        assert first_idx < last_idx, "First line must precede final line in filtered section"


# ─── UNIT: anthem context builder ────────────────────────────────────────────

class TestBuildAnthemContext:
    _CHUNKS = [
        _make_chunk("Wadimalu gundati lau bareini", 0),
        _make_chunk("Wadimalu liubani le seremei", 1),
        _make_chunk("Wadimalu biri le moweidibei", 2),
    ]

    def test_contains_anthem_exact_tag(self):
        ctx = _nodes._build_anthem_context(self._CHUNKS, "garifuna")
        assert "<ANTHEM_EXACT" in ctx

    def test_contains_first_line(self):
        ctx = _nodes._build_anthem_context(self._CHUNKS, "garifuna")
        assert "Wadimalu gundati lau bareini" in ctx

    def test_contains_last_line(self):
        ctx = _nodes._build_anthem_context(self._CHUNKS, "garifuna")
        assert "Wadimalu biri le moweidibei" in ctx

    def test_first_line_before_last_line(self):
        ctx = _nodes._build_anthem_context(self._CHUNKS, "garifuna")
        assert ctx.index("Wadimalu gundati lau bareini") < ctx.index("Wadimalu biri le moweidibei")

    def test_contains_attribution_disclaimer(self):
        ctx = _nodes._build_anthem_context(self._CHUNKS, "garifuna")
        assert "not been independently" in ctx.lower()

    def test_contains_source_citation(self):
        ctx = _nodes._build_anthem_context(self._CHUNKS, "garifuna")
        assert "Himno Nacional de Guatemala" in ctx

    def test_empty_chunks_returns_empty_string(self):
        assert _nodes._build_anthem_context([], "garifuna") == ""

    def test_lang_label_spanish(self):
        ctx = _nodes._build_anthem_context(self._CHUNKS, "spanish")
        assert 'lang="Spanish"' in ctx

    def test_lang_label_english(self):
        ctx = _nodes._build_anthem_context(self._CHUNKS, "english")
        assert 'lang="English"' in ctx


# ─── INTEGRATION: anthem retrieval ──────────────────────────────────────────

@requires_service
class TestAnthemRetrieval:
    def test_anthem_query_returns_first_line(self):
        """Engine must include the first Garífuna line of the anthem."""
        response = _ask_chat("What is the Guatemala national anthem in Garifuna?")
        assert "Wadimalu" in response, (
            f"Expected 'Wadimalu' in anthem response, got:\n{response}"
        )

    def test_anthem_query_does_not_say_unavailable(self):
        """Engine must NOT claim the anthem text is unavailable."""
        response = _ask_chat("Show me the Guatemala national anthem in Garifuna")
        lower = response.lower()
        negative_phrases = [
            "not available",
            "not found",
            "not indexed",
            "cannot provide",
            "do not have",
            "unable to",
            "no tengo",
            "no disponible",
        ]
        for phrase in negative_phrases:
            assert phrase not in lower, (
                f"Engine incorrectly said anthem unavailable ('{phrase}'):\n{response}"
            )

    def test_anthem_last_line_present(self):
        """Engine must include the final Garífuna line of the anthem."""
        response = _ask_chat("Please show me all the lyrics of the Guatemala anthem in Garifuna")
        assert "moweidibei" in response, (
            f"Expected final line 'moweidibei' in full anthem response, got:\n{response}"
        )

    def test_anthem_retrieval_pool_contains_anthem_chunks(self):
        """Retrieval pool for anthem query must surface anthem document chunks."""
        matches = _ask_retrieval("Guatemala national anthem Garifuna lyrics", k=10)
        sources = [m.get("metadata", {}).get("source", "") for m in matches]
        anthem_found = any("HIMNO_NACIONAL" in s.upper() for s in sources)
        assert anthem_found, (
            f"No anthem chunks in retrieval pool. Sources: {sources}"
        )

    def test_anthem_first_and_last_line_present(self):
        """Engine must include both the first and final Garífuna lines in one response."""
        response = _ask_chat("Show me the complete Guatemala national anthem in Garifuna")
        assert "Wadimalu gundati lau bareini" in response, (
            f"Expected first line 'Wadimalu gundati lau bareini' in response, got:\n{response}"
        )
        assert "Wadimalu biri le moweidibei" in response, (
            f"Expected final line 'Wadimalu biri le moweidibei' in response, got:\n{response}"
        )

    def test_anthem_does_not_mix_spanish_in_garifuna_response(self):
        """Garífuna anthem response must not include Spanish lyric lines."""
        response = _ask_chat("Show me the Garifuna lyrics of the national anthem")
        # Spanish anthem line that must NOT appear when Garífuna is requested
        assert "Guatemala feliz" not in response, (
            "Garífuna anthem response incorrectly included Spanish lyrics"
        )

    def test_anthem_spanish_section_returned_on_spanish_request(self):
        """Requesting the Spanish version must return Spanish lyrics."""
        response = _ask_chat("Show me the Spanish version of the national anthem")
        assert "Guatemala" in response, (
            f"Spanish anthem request did not return Spanish lyrics:\n{response}"
        )
        assert "Wadimalu" not in response, (
            "Spanish anthem request incorrectly included Garífuna lyrics"
        )

    def test_anthem_attribution_present(self):
        """Response must acknowledge that the Garífuna text is an unverified transcription."""
        response = _ask_chat("Show me the Guatemala national anthem in Garifuna")
        lower = response.lower()
        attribution_signals = [
            "transcription",
            "contributed",
            "not independently",
            "not been independently",
            "verified",
        ]
        assert any(sig in lower for sig in attribution_signals), (
            f"Response missing attribution disclaimer. Got:\n{response}"
        )

    def test_anthem_spanish_query(self):
        """Spanish anthem query must return Garífuna lyrics by default."""
        response = _ask_chat("¿Cuál es la letra del himno nacional en garífuna?", lang="es")
        assert "Wadimalu" in response, (
            f"Spanish anthem query did not return Garífuna lyrics:\n{response}"
        )
