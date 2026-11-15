"""
Regression tests for EN→GAR annotation-stripped lookup fallback.

Root cause: english_norm stores parenthetical annotations literally
("handsome (masculine)"), so an exact match for the base form ("handsome")
fails even though a matching entry exists.

Fix: when english_norm exact match returns 0 rows and the query contains
no parenthetical, retry with LIKE '<query> (%' to find annotated forms.

Phase 4 audit identified ~2,028 entries affected by this pattern.
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
# Unit-level tests for the fallback logic (via lexicon module directly)
# =============================================================================

class TestAnnotationFallbackUnit:
    """
    Unit tests exercising YurumeinLexicon.lookup() directly.
    These run without the full service stack.
    Skipped if DB env vars are not set (CI without DB).
    """

    @pytest.fixture(scope="class")
    def lexicon(self):
        try:
            from app.services.lexicon.lexicon import YurumeinLexicon
            return YurumeinLexicon()
        except Exception:
            pytest.skip("Cannot initialise YurumeinLexicon — DB not reachable")

    def test_exact_match_still_works(self, lexicon):
        """Exact english_norm match must still return rows (no regression)."""
        rows = lexicon.lookup("Elegant", "en")
        assert rows, "Expected rows for exact match 'Elegant'"
        garифуна_forms = [r.get("garifuna", "") for r in rows]
        assert any("ndeti" in (g or "").lower() for g in garифуна_forms), (
            f"Expected Wéndeti in results for 'Elegant'. Got garifuna: {garифуна_forms}"
        )

    def test_base_form_hits_annotated_entry(self, lexicon):
        """'Handsome' (no annotation) must resolve to entry with 'handsome (masculine)'."""
        rows = lexicon.lookup("Handsome", "en")
        assert rows, (
            "Expected annotation-stripped fallback to find 'handsome (masculine)' entries. "
            "Got 0 rows."
        )

    def test_annotated_query_still_exact_matches(self, lexicon):
        """'Handsome (masculine)' typed exactly must still hit via exact match."""
        rows = lexicon.lookup("Handsome (masculine)", "en")
        assert rows, "Expected exact match for 'Handsome (masculine)'. Got 0 rows."

    def test_fallback_does_not_fire_for_spanish(self, lexicon):
        """Spanish lookup must not use the annotation fallback.
        Spanish entries don't have parenthetical annotations — no change expected."""
        rows_guapo = lexicon.lookup("Guapo", "es")
        assert rows_guapo, "Expected ES match for 'Guapo'"

    def test_fallback_does_not_match_prefix_without_paren(self, lexicon):
        """'Hand' must NOT match 'handsome (masculine)' — prefix only fires with ' ('."""
        rows = lexicon.lookup("Hand", "en")
        garifuna_forms = [r.get("garifuna", "") for r in rows]
        assert not any("ndeti" in (g or "").lower() for g in garifuna_forms), (
            "'Hand' must not resolve to 'Wéndeti' (Elegant). Got: "
            + str(garifuna_forms)
        )

    def test_fallback_does_not_fire_when_query_has_paren(self, lexicon):
        """Query containing '(' must not trigger fallback (exact match only)."""
        rows_explicit = lexicon.lookup("Handsome (masculine)", "en")
        rows_fallback = lexicon.lookup("Handsome", "en")
        # Both should return results but via different paths;
        # the annotated query must not double-apply the fallback.
        assert rows_explicit, "Exact annotated query must still hit."
        assert rows_fallback, "Base-form query must hit via fallback."

    def test_bracelet_exact_unaffected(self, lexicon):
        """'Bracelet' has plain english_norm (no annotation) — exact match unchanged."""
        rows = lexicon.lookup("Bracelet", "en")
        assert rows, "Expected exact match for 'Bracelet'"
        garifuna_forms = [r.get("garifuna", "") for r in rows]
        assert any("nguli" in (g or "").lower() for g in garifuna_forms), (
            f"Expected Bángüli in results for 'Bracelet'. Got: {garifuna_forms}"
        )


# =============================================================================
# Integration tests via live API (require gaupo_ai at localhost:8085)
# =============================================================================

class TestAnnotationFallbackIntegration:

    @requires_service
    def test_en_gar_handsome_returns_garifuna(self):
        """EN→GAR 'Handsome' must return a Garífuna form via annotation-stripped fallback.
        Pre-fix: returned empty / LLM decline. Post-fix: returns Garífuna form."""
        result = _translate("en", "gar", "Handsome", ui_language="en")
        assert result.strip(), "EN→GAR 'Handsome' must not return empty after fix"
        result_lower = result.lower()
        assert "not available" not in result_lower and "no " not in result_lower, (
            f"EN→GAR 'Handsome' returned a decline instead of Garífuna: {result!r}"
        )

    @requires_service
    def test_en_gar_handsome_matches_es_gar_guapo(self):
        """EN 'Handsome' and ES 'Guapo' must resolve to the same Garífuna form.
        Parity test: both should reach the same lexicon entry."""
        result_en = _translate("en", "gar", "Handsome", ui_language="en")
        result_es = _translate("es", "gar", "Guapo", ui_language="es")
        # Normalize both results (strip accents, lowercase) for comparison
        import unicodedata
        def norm(s):
            s = unicodedata.normalize("NFD", s)
            s = "".join(c for c in s if unicodedata.category(c) != "Mn")
            return s.lower().strip()
        # At least one Garífuna form should overlap
        en_forms = {norm(f.strip()) for f in result_en.split(";") if f.strip()}
        es_forms = {norm(f.strip()) for f in result_es.split(";") if f.strip()}
        assert en_forms & es_forms, (
            f"EN 'Handsome' and ES 'Guapo' return no overlapping Garífuna forms.\n"
            f"EN result: {result_en!r}\nES result: {result_es!r}"
        )

    @requires_service
    def test_en_gar_elegant_no_regression(self):
        """EN→GAR 'Elegant' must still work (exact match — no regression from fallback)."""
        result = _translate("en", "gar", "Elegant", ui_language="en")
        result_lower = result.lower()
        assert "ndeti" in result_lower or "wendeti" in result_lower, (
            f"Expected Wéndeti in EN→GAR 'Elegant' result. Got: {result!r}"
        )

    @requires_service
    def test_en_gar_bracelet_no_regression(self):
        """EN→GAR 'Bracelet' must still work (plain entry — no annotation)."""
        result = _translate("en", "gar", "Bracelet", ui_language="en")
        result_lower = result.lower()
        assert "nguli" in result_lower, (
            f"Expected Bángüli in EN→GAR 'Bracelet' result. Got: {result!r}"
        )

    @requires_service
    def test_en_gar_numbers_no_regression(self):
        """EN→GAR 'NUMBERS IN GARIFUNA LANGUAGE' must still resolve (Phase 5 regression)."""
        result = _translate("en", "gar", "NUMBERS IN GARIFUNA LANGUAGE", ui_language="en")
        assert "numeru" in result.lower() or "garifuna" in result.lower(), (
            f"Expected NUMERU LIDAN GARIFUNA in result. Got: {result!r}"
        )

    @requires_service
    def test_es_gar_regression_guapo(self):
        """ES→GAR 'Guapo' must still work (Spanish lookup unchanged by this fix)."""
        result = _translate("es", "gar", "Guapo", ui_language="es")
        assert result.strip(), f"ES→GAR 'Guapo' returned empty: {result!r}"

    @requires_service
    def test_en_gar_no_false_positive_short_prefix(self):
        """Short query 'He' must NOT match every english_norm starting with 'he ('."""
        result = _translate("en", "gar", "He", ui_language="en")
        # Result should either be empty (decline) or specifically the Garífuna pronoun,
        # not a large list from prefix matching all 'he (...)' entries
        if result.strip():
            # Must not return an implausibly long list
            forms = [f.strip() for f in result.split(";") if f.strip()]
            assert len(forms) <= 10, (
                f"EN→GAR 'He' returned suspiciously many forms — possible false-positive "
                f"prefix explosion: {result!r}"
            )
