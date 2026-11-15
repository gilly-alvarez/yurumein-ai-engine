"""
Phase 4B tests: Fix B (comma-paren lookup fallback) and Fix VA (validation anchor).

Fix B — comma-paren LIKE fallback
  Lexicon entries stored as '<base>, (<annotation>)' were inaccessible when a
  user typed the base form.  Fix B adds a second LIKE pass with ', (%' so that
  "count" finds "Count, (numbers)", "thin" finds "Thin, (you)", etc.

  Known entries used:
    id=555:  english_norm='armenia, (honduras)'           → Arüménia
    id=1784: english_norm='with all his heart, (they)'    → Tau sun hanigi
    id=1785: english_norm='with all his heart, (she)'     → Tau sun tanigi
    id=1786: english_norm='with all his heart, (he)'      → Tau sun lanigi
    id=1878: english_norm='count, (numbers)'              → Abahüda
    id=2275: english_norm='thin, (you)'                   → Bürüraütü / Bürüraütu...
    id=2276: english_norm='thin, (you)'                   → Bürüraüti...
    id=2285: english_norm='insane, (she)'                 → Lülüti tau
    id=2286: english_norm='insane, (he)'                  → Lülüti lau
    id=2290: english_norm='demon, (she)'                  → Máfia, (hiñaru)
    id=2291: english_norm='demon, (he)'                   → Máfia, (eyeri)

Fix VA — validation anchor ("garifuna" excluded from output token check)
  _output_contains_garifuna_forms() previously accepted "NÚMEROS EN IDIOMA
  GARIFUNA" because "garifuna" is a token in the lexicon's garifuna column
  (from entries like id=2811 "Garifunóuti, Lidan garifuna").  After Fix VA,
  "garifuna" / "garífuna" are excluded from the validation token set so that
  only distinctly Garífuna vocabulary anchors the check.

  "NUMERU LIDAN GARIFUNA" must still pass because "lidan" is a Garífuna word
  found in the lexicon (id=2773 "Idan, Lidan, Luagu" for Spanish "En").
"""
import sys
import os
import unicodedata

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower().strip()


# ---------------------------------------------------------------------------
# Fix B — comma-paren fallback: YurumeinLexicon.lookup() unit tests
# ---------------------------------------------------------------------------

class TestCommaParen:
    """
    Direct unit tests for Fix B.  Skipped if DB is not reachable.
    All tests verify that the comma-paren LIKE fallback ('<base>, (%') fires
    correctly and that existing exact-match and space-paren behaviour is
    preserved.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def lex(cls):
        try:
            from app.services.lexicon.lexicon import YurumeinLexicon
            return YurumeinLexicon()
        except Exception:
            pytest.skip("Cannot initialise YurumeinLexicon — DB not reachable")

    # -- Fix B fires when neither exact nor Fix A match -----------------------

    def test_count_numbers_via_comma_paren(self, lex):
        """'Count' has no exact entry and no space-paren entries.
        Fix B must find 'Count, (numbers)' (id=1878)."""
        rows = lex.lookup("Count", "en")
        assert rows, "Expected Fix B to find 'Count, (numbers)' for query 'Count'"
        norms = [r["english_norm"] for r in rows]
        assert any("count" in n for n in norms), (
            f"Expected english_norm to contain 'count'. Got: {norms}"
        )
        assert any("(" in n for n in norms), (
            "Expected annotated comma-paren rows to be returned"
        )

    def test_with_all_his_heart_all_genders(self, lex):
        """'With all his heart' has three comma-paren entries (he/she/they).
        All three must be returned so the caller gets the complete gendered set."""
        rows = lex.lookup("With all his heart", "en")
        assert len(rows) >= 3, (
            f"Expected at least 3 gendered forms for 'With all his heart'. "
            f"Got {len(rows)}: {[r['english_norm'] for r in rows]}"
        )

    def test_thin_you_returned(self, lex):
        """'Thin' has no exact entry.  Fix B must find 'Thin, (you)' rows."""
        rows = lex.lookup("Thin", "en")
        assert rows, "Expected Fix B to find 'Thin, (you)' rows"
        norms = [r["english_norm"] for r in rows]
        assert all("thin" in n for n in norms), (
            f"All returned rows must relate to 'thin'. Got: {norms}"
        )

    def test_demon_she_and_he_returned(self, lex):
        """'Demon' has no exact entry.  Fix B must return both gendered forms."""
        rows = lex.lookup("Demon", "en")
        assert len(rows) >= 2, (
            f"Expected at least 2 gendered 'demon' rows. Got {len(rows)}"
        )
        garifuna_values = [r.get("garifuna", "") for r in rows]
        # Both 'hiñaru' (she) and 'eyeri' (he) variants of Máfia must appear
        combined = " ".join(garifuna_values).lower()
        assert "mafia" in _norm(combined) or any(
            "demon" in n for n in [r["english_norm"] for r in rows]
        ), f"Expected demon entries. Got garifuna values: {garifuna_values}"

    def test_insane_she_and_he_returned(self, lex):
        """'Insane' has no exact entry. Fix B returns 'Insane, (she)' and
        'Insane, (he)'."""
        rows = lex.lookup("Insane", "en")
        assert len(rows) >= 2, (
            f"Expected at least 2 gendered 'insane' rows via Fix B. Got {len(rows)}"
        )

    def test_armenia_place_qualifier(self, lex):
        """'Armenia' has no plain English entry; only 'Armenia, (Honduras)'.
        Fix B must find it."""
        rows = lex.lookup("Armenia", "en")
        assert rows, "Expected Fix B to find 'Armenia, (Honduras)'"
        garifuna = rows[0].get("garifuna", "")
        assert garifuna, "Expected non-empty Garífuna value for Armenia"

    # -- Fix B does NOT fire when exact match succeeds -------------------------

    def test_exact_match_blocks_comma_paren_fallback(self, lex):
        """'Elegant' has an exact entry.  Neither Fix A nor Fix B must fire.
        Only the exact match row must be returned."""
        rows = lex.lookup("Elegant", "en")
        assert rows, "Expected exact match for 'Elegant'"
        norms = [r["english_norm"] for r in rows]
        assert all("(" not in n for n in norms), (
            f"Exact match present — neither fallback should fire. Got: {norms}"
        )

    # -- Fix B does NOT fire when Fix A already succeeded ----------------------

    def test_fix_a_takes_priority_over_fix_b(self, lex):
        """'Handsome' has no exact entry but HAS space-paren entries.
        Fix A must fire; Fix B must NOT add comma-paren rows on top."""
        rows = lex.lookup("Handsome", "en")
        assert rows, "Expected Fix A to find 'handsome (masculine)'"
        norms = [r["english_norm"] for r in rows]
        # All results must have space-paren pattern, not comma-paren
        assert all(" (" in n for n in norms), (
            f"Expected only space-paren rows from Fix A. Got: {norms}"
        )
        assert not any(", (" in n for n in norms), (
            f"Fix B must not add comma-paren rows when Fix A succeeds. Got: {norms}"
        )

    # -- Fix B wildcard safety -------------------------------------------------

    def test_percent_in_query_does_not_wildcard_fix_b(self, lex):
        """A '%' in the base query must remain a literal in the Fix B LIKE pattern."""
        rows = lex.lookup("50%", "en")
        # No entry '50%, (...)' exists; must return 0 rows, not match everything
        assert rows == [], f"Expected 0 rows for '50%' via Fix B. Got: {len(rows)}"

    def test_underscore_in_query_does_not_wildcard_fix_b(self, lex):
        """A '_' in the base query must remain a literal in the Fix B LIKE pattern."""
        rows = lex.lookup("some_word", "en")
        assert rows == [], f"Expected 0 rows for 'some_word' via Fix B. Got: {len(rows)}"

    # -- Spanish lookup unaffected --------------------------------------------

    def test_spanish_lookup_unaffected_by_fix_b(self, lex):
        """Fix B only fires for English. Spanish lookups must use exact match only."""
        rows = lex.lookup("Elegante", "es")
        assert rows, "Expected exact Spanish match for 'Elegante'"
        norms = [r["spanish_norm"] for r in rows]
        assert all("elegante" in n for n in norms), (
            f"Spanish lookup must return exact match only. Got: {norms}"
        )


# ---------------------------------------------------------------------------
# Fix VA — validation anchor: _output_contains_garifuna_forms() unit tests
#
# Scope of Fix VA:
#   Removes "garifuna" (and "garífuna") from the token set used to anchor
#   output validation.  These proper nouns appear in both Garífuna and
#   Spanish/English contexts, so they are unreliable as sole validation
#   anchors.
#
# Known limitation (documented, not addressed by Fix VA):
#   The validation word set has pre-existing contamination: some lexicon
#   entries store Spanish annotation text in the garifuna column (e.g.
#   ids 4293/4294 have "(en Idioma Español) Hamuga..."), which causes
#   Spanish words like "idioma", "casa", "madera" to be treated as valid
#   Garífuna tokens.  As a result, "NÚMEROS EN IDIOMA GARIFUNA" still passes
#   validation via "idioma".  Eliminating this contamination requires either:
#     (a) a separate validation-overhaul PR (cleaning the word set), or
#     (b) a system prompt change (requires owner authorization per CLAUDE.md)
#   That work is deferred and tracked separately.
#
# What Fix VA DOES guarantee:
#   - An output consisting ONLY of the word "garifuna" / "garífuna" is rejected.
#   - "NUMERU LIDAN GARIFUNA" continues to pass via the token "lidan".
#   - "Aban", "Biama" and other undisputed Garífuna number words pass.
#   - Entries where the lexicon lookup catches "Garifuna"→"Garifuna" before
#     the LLM is called are unaffected.
# ---------------------------------------------------------------------------

class TestValidationAnchor:
    """
    Unit tests for the _output_contains_garifuna_forms() change.

    These tests instantiate the Translater and exercise the validation method
    directly (no LLM call, no network).  Skipped if DB is not reachable.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def tr(cls):
        try:
            from app.services.translater.translater import Translater
            return Translater()
        except Exception:
            pytest.skip("Cannot instantiate Translater — DB or config not reachable")

    # -- Fix VA: "garifuna" / "garífuna" no longer anchor validation -----------

    def test_only_garifuna_token_rejected(self, tr):
        """An output of only 'garifuna' must be rejected.
        Before Fix VA this passed because 'garifuna' was in the word set."""
        result = tr._output_contains_garifuna_forms("garifuna")
        assert result is False, (
            "Bare 'garifuna' must not pass Garífuna output validation after Fix VA"
        )

    def test_only_garifuna_accented_rejected(self, tr):
        """The accented form 'garífuna' must also be excluded from anchoring."""
        result = tr._output_contains_garifuna_forms("garífuna")
        assert result is False

    def test_english_phrase_garifuna_only_anchor_rejected(self, tr):
        """'In the Garifuna language' — 'garifuna' is the only potential anchor.
        After Fix VA it is excluded, so the phrase must be rejected."""
        result = tr._output_contains_garifuna_forms("In the Garifuna language")
        assert result is False

    def test_garifuna_as_last_token_rejected(self, tr):
        """A phrase whose only word-set match is 'garifuna' at the end
        must fail validation.  Covers the original numbers-phrase Mode 1
        variant 'NUMBERS IN THE GARIFUNA' (hypothetical, 'garifuna' as sole anchor)."""
        result = tr._output_contains_garifuna_forms("Numbers in the Garifuna")
        assert result is False

    # -- Fix VA: Correct Garífuna output still passes --------------------------

    def test_numeru_lidan_garifuna_passes(self, tr):
        """'NUMERU LIDAN GARIFUNA' must pass — 'lidan' is an exclusive
        Garífuna word in the lexicon (id=2773 garifuna='Idan, Lidan, Luagu').
        It must pass even after 'garifuna' is excluded from anchoring."""
        result = tr._output_contains_garifuna_forms("NUMERU LIDAN GARIFUNA")
        assert result is True, (
            "'NUMERU LIDAN GARIFUNA' must pass validation via the 'lidan' anchor"
        )

    def test_lidan_alone_passes(self, tr):
        """'lidan' alone must pass — it is exclusively Garífuna and not
        excluded by Fix VA."""
        result = tr._output_contains_garifuna_forms("lidan")
        assert result is True, "'lidan' (Garífuna preposition) must pass validation"

    def test_aban_biama_pass(self, tr):
        """The Garífuna number words 'aban' (1) and 'biama' (2) are in the
        lexicon word set and must pass validation."""
        assert tr._output_contains_garifuna_forms("Aban") is True, \
            "'Aban' (Garífuna for 1) must pass validation"
        assert tr._output_contains_garifuna_forms("Biama") is True, \
            "'Biama' (Garífuna for 2) must pass validation"

    def test_multiple_garifuna_words_pass(self, tr):
        """Multiple Garífuna number words must pass."""
        result = tr._output_contains_garifuna_forms("Aban, Biama, Gaduru")
        assert result is True, "Expected Garífuna number words to pass validation"

    def test_garifunaouti_passes(self, tr):
        """'Garifunóuti' (the Garífuna word meaning 'in Garífuna') must still
        pass — it is a distinct token (after lowercase strip) from the
        excluded 'garifuna'."""
        result = tr._output_contains_garifuna_forms("Garifunóuti")
        assert result is True, (
            "'Garifunóuti' must pass validation as a distinct Garífuna token"
        )

    # -- Empty / trivial cases ------------------------------------------------

    def test_empty_string_rejected(self, tr):
        """Empty string must be rejected."""
        result = tr._output_contains_garifuna_forms("")
        assert result is False

    def test_very_short_tokens_rejected(self, tr):
        """Tokens of length ≤ 2 are always filtered; output of only short
        words must be rejected."""
        result = tr._output_contains_garifuna_forms("En la")
        assert result is False

    # -- Fix VA does not affect context evidence check -------------------------

    def test_context_evidence_unaffected(self, tr):
        """_context_has_garifuna_evidence() checks context for Garífuna
        evidence.  Fix VA only modifies _output_contains_garifuna_forms()
        and must not alter the context check."""
        context = "NUMERU LIDAN GARIFUNA – 0 Gararoguaü 1 Aban 2 Biama"
        result = tr._context_has_garifuna_evidence(context)
        assert result is True, (
            "_context_has_garifuna_evidence() must accept context with Garífuna words"
        )
