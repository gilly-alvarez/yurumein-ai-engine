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

    def test_anthem_spanish_query(self):
        """Spanish anthem query must also return Garífuna lyrics."""
        response = _ask_chat("¿Cuál es la letra del himno nacional en garífuna?", lang="es")
        assert "Wadimalu" in response, (
            f"Spanish anthem query did not return Garífuna lyrics:\n{response}"
        )
