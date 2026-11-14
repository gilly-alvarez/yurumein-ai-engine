"""
Regression tests for Garífuna → Spanish/English translation routing.

Root cause fixed: language-code string matching used exact "garifuna" string;
the API sends the ISO-style code "gar". Three bugs patched:
  1. translater.py:157-158 — is_garifuna_source/target now accept "gar"
  2. translater.py:177     — col selection now accepts "en" for English target
  3. lexicon.py:25         — Spanish lookup now accepts "es" for spanish_norm

These tests cover both directions and diacritic variants.
All are integration tests: they require the gaupo_ai service at localhost:8085
and a live PostgreSQL lexicon. They are skipped automatically when the service
is unreachable.

Run with:
    pytest tests/test_reverse_translation_routing.py -v
"""
import sys
import os
import json
import urllib.request
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AI_API = "http://localhost:8085"


def _service_available() -> bool:
    try:
        with urllib.request.urlopen(f"{AI_API}/health", timeout=3):
            return True
    except Exception:
        return False


def _translate(base_language: str, target_language: str, text: str, ui_language: str = "en") -> str:
    payload = json.dumps({
        "base_language": base_language,
        "target_language": target_language,
        "text": text,
        "ui_language": ui_language,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{AI_API}/translater",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read()).get("translation", "")


requires_service = pytest.mark.skipif(
    not _service_available(), reason="gaupo_ai service not reachable at localhost:8085"
)


# =============================================================================
# GAR → Spanish
# =============================================================================

class TestGarifunaToSpanish:

    @requires_service
    def test_wendeti_accented_to_spanish(self):
        """GAR→ES Wéndeti must return Elegante (from lexicon, not LLM phrase)."""
        result = _translate("gar", "es", "Wéndeti", ui_language="es")
        assert "elegante" in result.lower(), (
            f"Expected 'Elegante' in GAR→ES 'Wéndeti', got: {result!r}"
        )
        assert "bien vestido" not in result.lower(), (
            f"LLM phrase 'bien vestido' must not appear after routing fix. Got: {result!r}"
        )

    @requires_service
    def test_wendeti_unaccented_to_spanish(self):
        """GAR→ES Wendeti (no diacritics, normalizes same) must return Elegante."""
        result = _translate("gar", "es", "Wendeti", ui_language="es")
        assert "elegante" in result.lower(), (
            f"Expected 'Elegante' in GAR→ES 'Wendeti', got: {result!r}"
        )

    @requires_service
    def test_banguli_accented_to_spanish(self):
        """GAR→ES Bángüli must include jewelry terms (Pulsera, Brazalete, or Alhajas).
        Note: 'Esclava' may also appear — in Spanish it is a recognized name for a
        rigid bracelet (Diccionario de americanismos, ASALE; Diccionario p.81). The
        Phase 3B correction was English-only: 'Slave'/'Enslaved woman' → 'Bracelet'.
        The Spanish column 'Esclava' was intentionally preserved."""
        result = _translate("gar", "es", "Bángüli", ui_language="es")
        result_lower = result.lower()
        assert any(w in result_lower for w in ("pulsera", "brazalete", "alhaja")), (
            f"Expected jewelry term (pulsera/brazalete/alhaja) in GAR→ES 'Bángüli', got: {result!r}"
        )

    @requires_service
    def test_baba_to_spanish(self):
        """GAR→ES Baba must return Padre or Papá (regression guard for father entries)."""
        result = _translate("gar", "es", "Baba", ui_language="es")
        result_lower = result.lower()
        assert "padre" in result_lower or "papá" in result_lower or "papa" in result_lower, (
            f"Expected 'Padre'/'Papá' in GAR→ES 'Baba', got: {result!r}"
        )


# =============================================================================
# GAR → English
# =============================================================================

class TestGarifunaToEnglish:

    @requires_service
    def test_wendeti_accented_to_english(self):
        """GAR→EN Wéndeti must return Elegant (English column, not Spanish phrase)."""
        result = _translate("gar", "en", "Wéndeti", ui_language="en")
        assert "elegant" in result.lower(), (
            f"Expected 'Elegant' in GAR→EN 'Wéndeti', got: {result!r}"
        )

    @requires_service
    def test_wendeti_unaccented_to_english(self):
        """GAR→EN Wendeti (no diacritics) must return Elegant."""
        result = _translate("gar", "en", "Wendeti", ui_language="en")
        assert "elegant" in result.lower(), (
            f"Expected 'Elegant' in GAR→EN 'Wendeti', got: {result!r}"
        )

    @requires_service
    def test_banguli_accented_to_english(self):
        """GAR→EN Bángüli must return Bracelet or Jewelry, NOT Slave or Enslaved."""
        result = _translate("gar", "en", "Bángüli", ui_language="en")
        result_lower = result.lower()
        assert "bracelet" in result_lower or "jewelry" in result_lower, (
            f"Expected 'Bracelet'/'Jewelry' in GAR→EN 'Bángüli', got: {result!r}"
        )
        assert "slave" not in result_lower and "enslaved" not in result_lower, (
            f"'Slave'/'Enslaved' must not appear in GAR→EN 'Bángüli'. Got: {result!r}"
        )

    @requires_service
    def test_banguli_unaccented_to_english(self):
        """GAR→EN Banguli (no diacritics, normalizes to same key as Bángüli/Bangüli) must return a valid result."""
        result = _translate("gar", "en", "Banguli", ui_language="en")
        result_lower = result.lower()
        assert "bracelet" in result_lower or "jewelry" in result_lower, (
            f"Expected 'Bracelet'/'Jewelry' in GAR→EN 'Banguli', got: {result!r}"
        )

    @requires_service
    def test_duna_to_english(self):
        """GAR→EN Duna must return Water (regression guard for existing behavior)."""
        result = _translate("gar", "en", "Duna", ui_language="en")
        assert "water" in result.lower(), (
            f"Expected 'Water' in GAR→EN 'Duna', got: {result!r}"
        )

    @requires_service
    def test_baba_to_english(self):
        """GAR→EN Baba must return father or dad."""
        result = _translate("gar", "en", "Baba", ui_language="en")
        result_lower = result.lower()
        assert "father" in result_lower or "dad" in result_lower, (
            f"Expected 'father'/'dad' in GAR→EN 'Baba', got: {result!r}"
        )

    @requires_service
    def test_unknown_word_no_invention(self):
        """Unknown Garífuna word must not produce an invented translation."""
        result = _translate("gar", "en", "xyzqwrtnofuna", ui_language="en")
        result_lower = result.lower()
        assert (
            "not available" in result_lower
            or "no " in result_lower
            or result_lower.strip() == ""
            or "verified" in result_lower
        ), (
            f"Unknown word should produce a decline message, not an invented translation. Got: {result!r}"
        )


# =============================================================================
# Forward direction: ES/EN → GAR (regression guard — must not break)
# =============================================================================

class TestForwardLookupRegression:

    @requires_service
    def test_elegante_spanish_to_garifuna(self):
        """ES→GAR Elegante must still return Wéndeti after routing fix."""
        result = _translate("es", "gar", "Elegante", ui_language="es")
        assert "wéndeti" in result.lower() or "wendeti" in result.lower(), (
            f"Expected 'Wéndeti' in ES→GAR 'Elegante', got: {result!r}"
        )

    @requires_service
    def test_elegant_english_to_garifuna(self):
        """EN→GAR Elegant must return Wéndeti."""
        result = _translate("en", "gar", "Elegant", ui_language="en")
        assert "wéndeti" in result.lower() or "wendeti" in result.lower(), (
            f"Expected 'Wéndeti' in EN→GAR 'Elegant', got: {result!r}"
        )

    @requires_service
    def test_bracelet_english_to_garifuna(self):
        """EN→GAR Bracelet must return Bángüli."""
        result = _translate("en", "gar", "Bracelet", ui_language="en")
        assert "bángüli" in result.lower() or "banguli" in result.lower(), (
            f"Expected 'Bángüli' in EN→GAR 'Bracelet', got: {result!r}"
        )

    @requires_service
    def test_pulsera_spanish_to_garifuna(self):
        """ES→GAR Pulsera must return Bángüli."""
        result = _translate("es", "gar", "Pulsera", ui_language="es")
        assert "bángüli" in result.lower() or "banguli" in result.lower(), (
            f"Expected 'Bángüli' in ES→GAR 'Pulsera', got: {result!r}"
        )
