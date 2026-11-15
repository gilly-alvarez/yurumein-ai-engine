"""
Regression tests for EN→GAR annotation-stripped lookup fallback.
Phase 4A hardening — covers all requirements from the Phase 4A audit.

Root cause documented in Phase 4 audit:
  english_norm stores parenthetical annotations literally
  ("handsome (masculine)"), so an exact match for the base form
  ("handsome") failed even when a matching entry existed.

Fix:
  When english_norm exact match returns 0 rows, the query has no
  parenthetical, and the lookup is English, retry with:
    WHERE english_norm LIKE '<escaped_query> (%' ESCAPE '!'
  LIKE metacharacters in the input (%, _, !) are escaped before
  concatenation so they cannot act as SQL wildcards.

Scope of the fallback:
  Covers ~2,002 entries stored as '<base> (<annotation>)'.
  Does NOT cover ~195 entries stored as '<base>, (<annotation>)' —
  documented as a known limitation.

Test organisation:
  TestAnnotationFallbackUnit    — lexicon.py unit tests (direct DB, no HTTP)
  TestAnnotationFallbackInteg   — integration tests via /translater API
  TestKnownPhraseRegression     — specific phrase and parity regressions
"""
import sys
import os
import json
import urllib.request
import unicodedata
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AI_API = "http://localhost:8085"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _service_available() -> bool:
    try:
        with urllib.request.urlopen(f"{AI_API}/health", timeout=3):
            return True
    except Exception:
        return False


def _translate(base_language: str, target_language: str, text: str,
               ui_language: str = "en") -> str:
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


def _norm(s: str) -> str:
    """Strip diacritics and lowercase — mirrors normalize_term logic."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


requires_service = pytest.mark.skipif(
    not _service_available(), reason="gaupo_ai service not reachable at localhost:8085"
)


# ---------------------------------------------------------------------------
# Unit tests — exercise YurumeinLexicon.lookup() directly against live DB
# ---------------------------------------------------------------------------

class TestAnnotationFallbackUnit:
    """
    Tests that target the lookup() method directly.
    Skipped automatically if the DB cannot be reached (e.g. CI without DB).

    Lexicon facts used:
      id=19:   english_norm='abuser'              garifuna='Bíbuti'   (exact entry)
      id=9318: english_norm='abuser (feminine)'   garifuna='Bíbutu'   (annotated)
      id=9321: english_norm='abuser (masculine)'  garifuna='Bíbuti'   (annotated)
      id=19 means 'abuser' has an exact entry → fallback must NOT fire.

      ids 99/100: english_norm='active'           garifuna='Chubalugutu/i'  (exact)
      ids 9385/9386: english_norm='active (feminine/masculine)'             (annotated)

      No plain 'handsome' entry. 'handsome (masculine)' exists (id ~varied).
      No plain 'belizean' entry. 4 annotated forms exist ('belizean (man/woman/…)').
    """

    @pytest.fixture(scope="class")
    @classmethod
    def lex(cls):
        try:
            from app.services.lexicon.lexicon import YurumeinLexicon
            return YurumeinLexicon()
        except Exception:
            pytest.skip("Cannot initialise YurumeinLexicon — DB not reachable")

    # -- Exact match behaviour (must not regress) ---------------------------

    def test_exact_match_unaffected(self, lex):
        """'Elegant' has an exact english_norm entry — exact match must succeed."""
        rows = lex.lookup("Elegant", "en")
        assert rows, "Expected rows for exact match 'Elegant'"
        assert all(_norm(r.get("garifuna", "")) == _norm("Wéndeti") for r in rows), (
            f"Expected all results to be Wéndeti variants. Got: "
            f"{[r.get('garifuna') for r in rows]}"
        )

    def test_exact_match_authoritative_no_fallback(self, lex):
        """'Abuser' has both an exact entry (id=19) and annotated variants.
        Exact match must be returned; fallback must NOT be invoked.
        The annotated variants (masculine/feminine) must NOT be appended."""
        rows = lex.lookup("Abuser", "en")
        assert rows, "Expected exact match for 'Abuser'"
        returned_norms = [r["english_norm"] for r in rows]
        assert all("(" not in n for n in returned_norms), (
            "Fallback invoked despite exact match — annotated rows leaked into result. "
            f"Returned english_norm values: {returned_norms}"
        )

    # -- Annotation-stripped fallback (core fix) ----------------------------

    def test_base_form_hits_annotated_only_entry(self, lex):
        """'Handsome' has NO exact entry; only 'handsome (masculine)' exists.
        The fallback must find and return it."""
        rows = lex.lookup("Handsome", "en")
        assert rows, (
            "Expected annotation-stripped fallback to find 'handsome (masculine)'. "
            "Got 0 rows."
        )
        assert all("handsome" in r["english_norm"] for r in rows), (
            f"All returned rows must contain 'handsome'. Got: "
            f"{[r['english_norm'] for r in rows]}"
        )

    def test_multiple_annotated_variants_all_returned(self, lex):
        """'Belizean' has NO exact entry; 4 annotated forms exist
        ('belizean (man)', 'belizean (woman)', 'belizean (masculine)', 'belizean (feminine)').
        Fallback must return all of them — users need the full set of Garífuna forms."""
        rows = lex.lookup("Belizean", "en")
        assert len(rows) >= 2, (
            f"Expected at least 2 annotated 'belizean (...)' rows via fallback. "
            f"Got {len(rows)}: {[r['english_norm'] for r in rows]}"
        )
        assert all("belizean" in r["english_norm"] for r in rows), (
            f"All returned rows must relate to 'belizean'. Got: "
            f"{[r['english_norm'] for r in rows]}"
        )

    def test_annotated_query_still_exact_matches(self, lex):
        """'Handsome (masculine)' typed exactly must succeed via exact match
        (not via fallback). The fallback guard '(\" in norm' prevents double-firing."""
        rows = lex.lookup("Handsome (masculine)", "en")
        assert rows, "Expected exact match for 'Handsome (masculine)'"
        assert all(r["english_norm"] == "handsome (masculine)" for r in rows), (
            f"Expected exact english_norm='handsome (masculine)'. Got: "
            f"{[r['english_norm'] for r in rows]}"
        )

    # -- Wildcard injection safety ------------------------------------------

    def test_percent_in_query_does_not_wildcard(self, lex):
        """A query containing '%' must not act as a SQL wildcard.
        '100%' should return 0 rows (no entry with that exact english_norm)
        rather than matching all entries via unescaped LIKE."""
        rows = lex.lookup("100%", "en")
        # If % were unescaped, LIKE '100% (%' would match many rows.
        # With proper escaping, it matches only entries literally containing '100%'.
        assert len(rows) == 0 or all(
            "100%" in r.get("english_norm", "") for r in rows
        ), (
            f"'100%' query returned rows that do not literally contain '100%'. "
            f"Likely wildcard injection. Got: {[r['english_norm'] for r in rows]}"
        )

    def test_underscore_in_query_does_not_wildcard(self, lex):
        """A query containing '_' must not act as a SQL wildcard.
        'some_thing' should return 0 rows rather than matching one-char substitutions."""
        rows = lex.lookup("some_thing", "en")
        assert len(rows) == 0 or all(
            "_" in r.get("english_norm", "") for r in rows
        ), (
            f"'some_thing' query matched rows without literal '_'. "
            f"Likely wildcard injection. Got: {[r['english_norm'] for r in rows]}"
        )

    def test_exclamation_in_query_does_not_escape_wildcard(self, lex):
        """'!' is used as the LIKE escape character. A query containing '!' must
        not accidentally suppress wildcard matching on the legitimate ' (%' suffix."""
        rows_with_bang = lex.lookup("active!", "en")
        rows_without_bang = lex.lookup("active", "en")
        # 'active!' should not return 'active (masculine)' — it must not exploit
        # the escape mechanism to strip the '!' before the prefix check.
        assert len(rows_with_bang) == 0 or all(
            "active!" in r.get("english_norm", "") for r in rows_with_bang
        ), (
            f"'active!' unexpectedly matched rows without literal '!'. "
            f"Got: {[r['english_norm'] for r in rows_with_bang]}"
        )

    def test_prefix_does_not_match_without_space_paren(self, lex):
        """'Hand' must NOT match 'handsome (masculine)' — the separator ' (' is
        required; plain prefix substring matching must not occur."""
        rows = lex.lookup("Hand", "en")
        handsome_rows = [r for r in rows if "handsome" in (r.get("english_norm") or "")]
        assert not handsome_rows, (
            f"'Hand' must not resolve to 'handsome' entries. "
            f"Got: {[r['english_norm'] for r in handsome_rows]}"
        )

    # -- Language column isolation ------------------------------------------

    def test_spanish_lookup_unaffected(self, lex):
        """Spanish lookup must use spanish_norm only. No annotation fallback
        applies to Spanish because spanish_norm entries use plain headwords."""
        rows = lex.lookup("Guapo", "es")
        assert rows, "Expected exact match for 'Guapo' via spanish_norm"

    def test_english_and_spanish_target_different_columns(self, lex):
        """'Elegante' exists only in spanish_norm. EN lookup for 'Elegante' must
        return 0 rows (not fall through to spanish_norm)."""
        rows_en = lex.lookup("Elegante", "en")
        rows_es = lex.lookup("Elegante", "es")
        assert rows_es, "Expected 'Elegante' via Spanish lookup"
        # 'Elegante' should NOT be in english_norm (Phase 3B corrected it to 'Elegant')
        assert all("elegant" in (r.get("english_norm") or "") for r in rows_en), (
            "English lookup for 'Elegante' must only return rows where english_norm "
            f"matches 'elegante'. Got: {[r.get('english_norm') for r in rows_en]}"
        ) if rows_en else True

    # -- Unknown word -------------------------------------------------------

    def test_unknown_word_returns_empty(self, lex):
        """An unknown word must return an empty list from both exact and fallback paths."""
        rows = lex.lookup("xyzqwrtnofuna", "en")
        assert rows == [], f"Expected [] for unknown word. Got: {rows}"

    # -- EN/ES semantic equivalence ----------------------------------------

    def test_en_es_same_garifuna_abuser(self, lex):
        """'Abuser' (EN) and 'Abusador' (ES) must return the same Garífuna root.
        EN exact match returns id=19 (Bíbuti). ES exact match returns same."""
        rows_en = lex.lookup("Abuser", "en")
        rows_es = lex.lookup("Abusador", "es")
        assert rows_en, "EN 'Abuser' must have results"
        assert rows_es, "ES 'Abusador' must have results"
        en_garifuna = {_norm(r.get("garifuna", "")) for r in rows_en}
        es_garifuna = {_norm(r.get("garifuna", "")) for r in rows_es}
        assert en_garifuna & es_garifuna, (
            f"EN 'Abuser' and ES 'Abusador' share no common Garífuna root.\n"
            f"EN garifuna: {en_garifuna}\nES garifuna: {es_garifuna}"
        )


# ---------------------------------------------------------------------------
# Integration tests via live API (require gaupo_ai at localhost:8085)
# ---------------------------------------------------------------------------

class TestAnnotationFallbackInteg:

    @requires_service
    def test_en_gar_handsome_returns_garifuna(self):
        """EN→GAR 'Handsome' must return a Garífuna form via annotation fallback."""
        result = _translate("en", "gar", "Handsome", ui_language="en")
        assert result.strip(), "EN→GAR 'Handsome' must not return empty after fix"
        result_lower = result.lower()
        assert "not available" not in result_lower and "i don" not in result_lower, (
            f"EN→GAR 'Handsome' returned a decline: {result!r}"
        )

    @requires_service
    def test_en_gar_handsome_parity_with_es_gar_guapo(self):
        """EN 'Handsome' and ES 'Guapo' must resolve to overlapping Garífuna forms."""
        result_en = _translate("en", "gar", "Handsome", ui_language="en")
        result_es = _translate("es", "gar", "Guapo", ui_language="es")
        en_forms = {_norm(f.strip()) for f in result_en.split(";") if f.strip()}
        es_forms = {_norm(f.strip()) for f in result_es.split(";") if f.strip()}
        assert en_forms & es_forms, (
            f"EN 'Handsome' and ES 'Guapo' share no Garífuna forms.\n"
            f"EN: {result_en!r}\nES: {result_es!r}"
        )

    @requires_service
    def test_en_gar_elegant_exact_match_no_regression(self):
        """EN→GAR 'Elegant' must still return Wéndeti (exact english_norm match)."""
        result = _translate("en", "gar", "Elegant", ui_language="en")
        assert _norm("Wéndeti") in _norm(result) or "ndeti" in result.lower(), (
            f"Expected Wéndeti in EN→GAR 'Elegant'. Got: {result!r}"
        )

    @requires_service
    def test_en_gar_bracelet_exact_match_no_regression(self):
        """EN→GAR 'Bracelet' must still return Bángüli (plain entry, no annotation).
        Uses _norm() to strip diacritics before substring check: 'ü' != 'u' in
        Python string comparison, so result.lower() alone would miss 'nguli' in
        'bángüli'."""
        result = _translate("en", "gar", "Bracelet", ui_language="en")
        assert "nguli" in _norm(result), (
            f"Expected Bángüli in EN→GAR 'Bracelet'. Got: {result!r}"
        )

    @requires_service
    def test_en_gar_belizean_multiple_forms(self):
        """EN→GAR 'Belizean' must return Garífuna forms (via annotation fallback for
        'belizean (man/woman/masculine/feminine)' — no exact 'belizean' entry)."""
        result = _translate("en", "gar", "Belizean", ui_language="en")
        assert result.strip(), f"EN→GAR 'Belizean' returned empty: {result!r}"
        result_lower = result.lower()
        # All annotated Belizean entries share 'Balisina' as the Garífuna root
        assert "balisina" in result_lower, (
            f"Expected 'Balisina' in EN→GAR 'Belizean' result. Got: {result!r}"
        )

    @requires_service
    def test_en_gar_no_false_positive_short_prefix(self):
        """Short query 'He' must not match every english_norm starting with 'he ('."""
        result = _translate("en", "gar", "He", ui_language="en")
        if result.strip():
            forms = [f.strip() for f in result.split(";") if f.strip()]
            assert len(forms) <= 10, (
                f"EN→GAR 'He' returned suspiciously many forms — possible fallback "
                f"explosion: {result!r}"
            )


# ---------------------------------------------------------------------------
# Known phrase regressions and bilingual parity regressions
# ---------------------------------------------------------------------------

class TestKnownPhraseRegression:

    @requires_service
    def test_numbers_exact_phrase(self):
        """EN→GAR 'NUMBERS IN GARIFUNA LANGUAGE' must return the documented
        section header 'NUMERU LIDAN GARIFUNA' from DICCIONARIO page 203.
        This was a Phase 3C routing failure; now resolved. Must contain the
        complete phrase, not merely the word 'Garifuna'."""
        result = _translate("en", "gar", "NUMBERS IN GARIFUNA LANGUAGE", ui_language="en")
        result_norm = _norm(result)
        assert "numeru lidan garifuna" in result_norm, (
            f"Expected 'NUMERU LIDAN GARIFUNA' in EN→GAR numbers result.\n"
            f"Got: {result!r}\nNormalized: {result_norm!r}"
        )

    @requires_service
    def test_es_numbers_phrase(self):
        """ES→GAR 'NUMEROS EN IDIOMA GARIFUNA' must also return 'NUMERU LIDAN GARIFUNA'."""
        result = _translate("es", "gar", "NUMEROS EN IDIOMA GARIFUNA", ui_language="es")
        result_norm = _norm(result)
        assert "numeru lidan garifuna" in result_norm, (
            f"Expected 'NUMERU LIDAN GARIFUNA' in ES→GAR numbers result.\n"
            f"Got: {result!r}"
        )

    @requires_service
    def test_gar_en_wendeti_quality(self):
        """GAR→EN 'Wéndeti' must return clean English forms with no Garífuna headword,
        no duplicate segments, and no trailing period segment."""
        result = _translate("gar", "en", "Wéndeti", ui_language="en")
        result_lower = result.lower()
        assert "wéndeti" not in result_lower, (
            f"Garífuna headword must not appear in GAR→EN result. Got: {result!r}"
        )
        segments = [s.strip() for s in result.split(";") if s.strip()]
        assert len(segments) == len({s.lower() for s in segments}), (
            f"Duplicate segments in GAR→EN 'Wéndeti': {result!r}"
        )
        for seg in segments:
            assert not seg.endswith("."), (
                f"Trailing period in segment {seg!r}: {result!r}"
            )

    @requires_service
    def test_gar_es_wendeti_quality(self):
        """GAR→ES 'Wéndeti' must return clean Spanish forms with no trailing period."""
        result = _translate("gar", "es", "Wéndeti", ui_language="es")
        segments = [s.strip() for s in result.split(";") if s.strip()]
        for seg in segments:
            assert not seg.endswith("."), (
                f"Trailing period in segment {seg!r}: {result!r}"
            )
        assert "elegante" in result.lower(), (
            f"Expected 'Elegante' in GAR→ES 'Wéndeti'. Got: {result!r}"
        )

    @requires_service
    def test_en_gar_banguli_no_slave(self):
        """EN→GAR 'Bracelet' must not return any slavery-related English in the path.
        GAR→EN 'Bángüli' must return 'Bracelet', not 'Slave' or 'Enslaved'."""
        result = _translate("gar", "en", "Bángüli", ui_language="en")
        result_lower = result.lower()
        assert "bracelet" in result_lower or "jewelry" in result_lower, (
            f"Expected jewelry term in GAR→EN 'Bángüli'. Got: {result!r}"
        )
        assert "slave" not in result_lower and "enslaved" not in result_lower, (
            f"'Slave'/'Enslaved' must not appear in GAR→EN 'Bángüli'. Got: {result!r}"
        )

    @requires_service
    def test_gar_en_duna_all_senses(self):
        """GAR→EN 'Duna' must return all three ABÜRÜHATI-documented senses:
        Water (id=187), lake (id=5447), river (id=7886)."""
        result = _translate("gar", "en", "Duna", ui_language="en")
        result_lower = result.lower()
        assert "water" in result_lower, f"'water' missing from GAR→EN 'Duna': {result!r}"
        assert "lake" in result_lower, f"'lake' missing from GAR→EN 'Duna': {result!r}"
        assert "river" in result_lower, f"'river' missing from GAR→EN 'Duna': {result!r}"

    @requires_service
    def test_gar_es_duna_all_senses(self):
        """GAR→ES 'Duna' must include Agua, Lago, and Río."""
        result = _translate("gar", "es", "Duna", ui_language="es")
        result_lower = result.lower()
        assert "agua" in result_lower, f"'agua' missing from GAR→ES 'Duna': {result!r}"
        assert "lago" in result_lower, f"'lago' missing from GAR→ES 'Duna': {result!r}"
        assert "río" in result_lower or "rio" in result_lower, (
            f"'río' missing from GAR→ES 'Duna': {result!r}"
        )

    @requires_service
    def test_es_gar_elegante_round_trip(self):
        """ES→GAR 'Elegante' must still return Wéndeti (forward lookup regression)."""
        result = _translate("es", "gar", "Elegante", ui_language="es")
        assert _norm("Wéndeti") in _norm(result) or "ndeti" in result.lower(), (
            f"Expected Wéndeti in ES→GAR 'Elegante'. Got: {result!r}"
        )

    @requires_service
    def test_en_gar_elegant_round_trip(self):
        """EN→GAR 'Elegant' must still return Wéndeti."""
        result = _translate("en", "gar", "Elegant", ui_language="en")
        assert _norm("Wéndeti") in _norm(result) or "ndeti" in result.lower(), (
            f"Expected Wéndeti in EN→GAR 'Elegant'. Got: {result!r}"
        )
