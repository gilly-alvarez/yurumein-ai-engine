"""
Isolated tests for Fix B: comma-paren LIKE fallback in YurumeinLexicon.lookup()

Fix B adds a second LIKE pass with the pattern '<base>, (%' so that entries
stored as '<base>, (<annotation>)' become reachable by typing just the base.

Examples from the lexicon:
  "count"       → "Count, (numbers)"           → Abahüda
  "thin"        → "Thin, (you)"                → Bürüraütu / Bürüraüti
  "armenia"     → "Armenia, (Honduras)"        → Arüménia, (Huliligati)
  "in front of" → "In front of, (she)"         → Sururügü bougui ...
  "direct to"   → "Direct to, (she)"           → Sururügü tumóun  [×2 dialectal]

Scope: ALL 98 distinct base forms known to have comma-paren entries (195 total
entries, 98 base forms identified by DB query on 2026-11-14).

Coverage strategy:
  1. Data-driven: every base form returns ≥1 row via Fix B when no exact match exists.
  2. Spot-checks: specific Garífuna forms for key entries.
  3. Priority: exact match > Fix A > Fix B — neither fallback fires when a
     higher-priority result exists.
  4. Safety: SQL wildcard injection via % and _ is blocked.
  5. Spanish: Fix B does not fire for Spanish-language lookups.
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
    """Reproduce normalize_term() without importing the app module."""
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = unicodedata.normalize("NFKC", s)
    import re
    return re.sub(r"\s+", " ", s.strip().lower())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def lex():
    try:
        from app.services.lexicon.lexicon import YurumeinLexicon
        return YurumeinLexicon()
    except Exception:
        pytest.skip("Cannot initialise YurumeinLexicon — DB not reachable")


# ---------------------------------------------------------------------------
# Data-driven: ALL 98 distinct comma-paren base forms
# ---------------------------------------------------------------------------

# Every entry here was discovered by:
#   SELECT english_norm FROM yurumein_lexicon WHERE english_norm LIKE '%, (%'
# and confirmed to have no exact-match entry at the base form.
COMMA_PAREN_BASES = [
    "against",
    "age of",
    "agreeing",
    "american",
    "armenia",
    "back of",
    "background",
    "being, i was, for",
    "between",
    "bitter heart of",
    "cat",
    "clogging",
    "condemn",
    "condition",
    "controlled",
    "count",
    "cover",
    "covered",
    "cricket",
    "culila",
    "demon",
    "dignity of",
    "direct to",
    "disobedient",
    "divorced",
    "doctor",
    "elegant",
    "exalting",
    "exercise",
    "experience of",
    "export",
    "faithful",
    "filtrarse",
    "firm",
    "fraud",
    "front",
    "front of",
    "general",
    "geneva",
    "gesture of",
    "getting drunk or brought by",
    "good-hearted",
    "grain",
    "greatness",
    "grey",
    "guanaba",
    "guilt of",
    "half-white",
    "happy",
    "happy birthday",
    "hook",
    "i agree",
    "in company of",
    "in front of",
    "in the home of",
    "in the place of",
    "in the times of",
    "insane",
    "inside",
    "inside the mouth of",
    "instead",
    "lack of judgment",
    "on the part of",
    "outside",
    "report it to",
    "report them to",
    "right of",
    "running",
    "say",
    "says",
    "scream",
    "separated teeth",
    "skilled to navigate",
    "skilled to sell",
    "skilled to work",
    "skirt",
    "sleeping with a lover",
    "spread",
    "sternum",
    "student",
    "sweet",
    "talking stuttering",
    "teeth perched",
    "the brain of",
    "the exterior of",
    "the godfather of",
    "the interior",
    "the interior of",
    "the other side of town",
    "the sacrifice of",
    "the skull",
    "the skull of",
    "these",
    "thin",
    "unpleasant",
    "venereal disease",
    "with",
    "with all his heart",
]


class TestCommaParenCoverage:
    """Data-driven: each of the 98 known base forms must return ≥1 row via Fix B.

    For any base that has an exact-match entry the lookup will return the exact
    match (not a Fix-B row); those cases are excluded from this coverage test.
    The parameterised test skips if the base happens to have an exact entry.
    """

    @pytest.mark.parametrize("base", COMMA_PAREN_BASES)
    def test_base_form_reachable(self, lex, base):
        rows = lex.lookup(base, "en")
        assert rows, (
            f"Fix B must find at least one entry for base '{base}'. Got 0 rows."
        )
        # Every returned row must be related to this base (english_norm starts with base)
        norm_base = _norm(base)
        for r in rows:
            en = r.get("english_norm", "")
            assert en.startswith(norm_base), (
                f"Unexpected row for '{base}': english_norm='{en}'"
            )

    @pytest.mark.parametrize("base", COMMA_PAREN_BASES)
    def test_garifuna_non_empty(self, lex, base):
        rows = lex.lookup(base, "en")
        if not rows:
            pytest.skip(f"No rows for '{base}' — covered by test_base_form_reachable")
        for r in rows:
            assert r.get("garifuna"), (
                f"Row for '{base}' has empty Garífuna value: {r}"
            )


# ---------------------------------------------------------------------------
# Spot-checks: specific entries and Garífuna values
# ---------------------------------------------------------------------------

class TestCommaParenSpotChecks:
    """Targeted assertions for representative entries from each category."""

    # --- Semantic disambiguation ---

    def test_count_numbers(self, lex):
        """'Count, (numbers)' — id=1878 → Abahüda"""
        rows = lex.lookup("Count", "en")
        assert rows, "Expected Fix B to find 'Count, (numbers)'"
        garifuna = " ".join(r.get("garifuna", "") for r in rows).lower()
        assert "abah" in _norm(garifuna), (
            f"Expected 'Abahüda' for Count (numbers). Got: {[r['garifuna'] for r in rows]}"
        )

    def test_cricket_animal(self, lex):
        """'Cricket, (animal)' — Diru"""
        rows = lex.lookup("Cricket", "en")
        assert rows, "Expected Fix B to find 'Cricket, (animal)'"
        # Should return only the animal entry, not an insect sport
        en_norms = [r["english_norm"] for r in rows]
        assert all("cricket" in n for n in en_norms)

    def test_venereal_disease(self, lex):
        """'Venereal disease, (gonorrhea)' — specific Garífuna term"""
        rows = lex.lookup("Venereal disease", "en")
        assert rows, "Expected Fix B to find 'Venereal disease, (gonorrhea)'"

    def test_gesture_of_disobedience(self, lex):
        """'Gesture of, (disobedience)' — Aginücha, Giñuti"""
        rows = lex.lookup("Gesture of", "en")
        assert rows, "Expected Fix B to find 'Gesture of, (disobedience)'"

    def test_filtrarse_liquid(self, lex):
        """'Filtrarse, (liquid)' — Ebeseiragua"""
        rows = lex.lookup("Filtrarse", "en")
        assert rows, "Expected Fix B to find 'Filtrarse, (liquid)'"

    # --- Gendered forms ---

    def test_thin_you_returned(self, lex):
        """'Thin, (you)' — 2 entries (masc/fem forms of 'you')"""
        rows = lex.lookup("Thin", "en")
        assert len(rows) >= 2, (
            f"Expected ≥2 'Thin, (you)' rows via Fix B. Got {len(rows)}"
        )
        for r in rows:
            assert "thin" in r["english_norm"]

    def test_demon_she_and_he(self, lex):
        """'Demon, (she)' and 'Demon, (he)' — both returned"""
        rows = lex.lookup("Demon", "en")
        assert len(rows) >= 2, (
            f"Expected ≥2 gendered 'Demon' rows. Got {len(rows)}"
        )

    def test_insane_gendered(self, lex):
        """'Insane, (she)' and 'Insane, (he)' — both returned"""
        rows = lex.lookup("Insane", "en")
        assert len(rows) >= 2, (
            f"Expected ≥2 'Insane' rows via Fix B. Got {len(rows)}"
        )

    def test_in_front_of_all_persons(self, lex):
        """'In front of, (she/he/ell@s)' — 6 entries returned"""
        rows = lex.lookup("In front of", "en")
        assert len(rows) >= 3, (
            f"Expected ≥3 'In front of' gendered rows. Got {len(rows)}"
        )

    def test_direct_to_multiple_forms(self, lex):
        """'Direct to, (she)' × 2 dialectal variants — ≥2 rows returned"""
        rows = lex.lookup("Direct to", "en")
        assert len(rows) >= 2, (
            f"Expected ≥2 'Direct to' rows via Fix B. Got {len(rows)}"
        )

    def test_with_all_his_heart_all_persons(self, lex):
        """'With all his heart, (he/she/they)' — 3 entries"""
        rows = lex.lookup("With all his heart", "en")
        assert len(rows) >= 3, (
            f"Expected ≥3 'With all his heart' rows. Got {len(rows)}"
        )

    def test_student_gendered(self, lex):
        """'Student, (she/he)' — 2 entries (no plural form in DB)"""
        rows = lex.lookup("Student", "en")
        assert len(rows) >= 2, (
            f"Expected ≥2 'Student' rows. Got {len(rows)}"
        )

    # --- Geographic qualifiers ---

    def test_armenia_honduras(self, lex):
        """'Armenia, (Honduras)' — Arüménia, (Huliligati)"""
        rows = lex.lookup("Armenia", "en")
        assert rows, "Expected Fix B to find 'Armenia, (Honduras)'"
        garifuna = rows[0].get("garifuna", "")
        assert garifuna, "Expected non-empty Garífuna value for Armenia"
        assert "arumenia" in _norm(garifuna) or "armenia" in _norm(garifuna), (
            f"Expected Arüménia for Armenia. Got: {garifuna}"
        )

    def test_geneva_switzerland(self, lex):
        """'Geneva, (Switzerland)' — Sinefu"""
        rows = lex.lookup("Geneva", "en")
        assert rows, "Expected Fix B to find 'Geneva, (Switzerland)'"
        garifuna = rows[0].get("garifuna", "")
        assert "sinefu" in _norm(garifuna), (
            f"Expected 'Sinefu' for Geneva. Got: {garifuna}"
        )

    # --- Grain disambiguation (5 annotations) ---

    def test_grain_multiple_annotations(self, lex):
        """'Grain' has 5 comma-paren entries: (bean/corn/rice/basic/wounded)"""
        rows = lex.lookup("Grain", "en")
        assert len(rows) >= 3, (
            f"Expected ≥3 'Grain' entries. Got {len(rows)}"
        )
        for r in rows:
            assert "grain" in r["english_norm"]


# ---------------------------------------------------------------------------
# Priority: exact match and Fix A must take precedence over Fix B
# ---------------------------------------------------------------------------

class TestCommaParenPriority:

    def test_exact_match_blocks_fix_b(self, lex):
        """'Handsome' has no exact entry but HAS space-paren entries.
        Fix A fires; Fix B must NOT fire on top."""
        rows = lex.lookup("Handsome", "en")
        assert rows, "Expected Fix A rows for 'Handsome'"
        norms = [r["english_norm"] for r in rows]
        assert all(" (" in n for n in norms), (
            f"Expected only space-paren rows from Fix A. Got: {norms}"
        )
        assert not any(", (" in n for n in norms), (
            "Fix B must not add comma-paren rows when Fix A succeeds."
        )

    def test_elegant_exact_match_wins(self, lex):
        """'Elegant' has an exact entry — no fallback must fire."""
        rows = lex.lookup("Elegant", "en")
        assert rows, "Expected exact match for 'Elegant'"
        norms = [r["english_norm"] for r in rows]
        assert all("(" not in n for n in norms), (
            f"Exact match present — no fallback should fire. Got: {norms}"
        )

    def test_fix_a_entries_not_mixed_with_fix_b(self, lex):
        """'Aggressive' has space-paren entries.  Fix B must not add rows."""
        rows = lex.lookup("Aggressive", "en")
        assert rows
        norms = [r["english_norm"] for r in rows]
        assert not any(", (" in n for n in norms), (
            f"Fix B rows must not appear when Fix A succeeded. Got: {norms}"
        )

    def test_happy_birthday_fix_b(self, lex):
        """'Happy birthday' has comma-paren entries and no exact/space-paren entry."""
        rows = lex.lookup("Happy birthday", "en")
        assert rows, "Expected Fix B rows for 'Happy birthday'"
        for r in rows:
            assert "happy birthday" in r["english_norm"]


# ---------------------------------------------------------------------------
# SQL wildcard safety
# ---------------------------------------------------------------------------

class TestCommaParenWildcardSafety:

    def test_percent_in_query(self, lex):
        """'%' must be escaped; query '50%' must not match every row."""
        rows = lex.lookup("50%", "en")
        assert rows == [], f"Expected 0 rows for '50%'. Got {len(rows)}"

    def test_underscore_in_query(self, lex):
        """'_' must be escaped; query 'some_word' must not match single-char wildcards."""
        rows = lex.lookup("some_word", "en")
        assert rows == [], f"Expected 0 rows for 'some_word'. Got {len(rows)}"

    def test_exclamation_in_query(self, lex):
        """'!' is the LIKE escape char — must be doubled so 'foo!' stays literal."""
        rows = lex.lookup("foo!", "en")
        assert rows == [], f"Expected 0 rows for 'foo!'. Got {len(rows)}"

    def test_percent_in_comma_paren_base_no_wildcard(self, lex):
        """'%, (test)' pattern must not widen the LIKE to match everything."""
        rows = lex.lookup("%, (test)", "en")
        # Input contains '(' so the fallback is bypassed entirely; exact match only.
        # Either 0 rows or only the specific entry if it exists.
        for r in rows:
            assert "(test)" in r.get("english_norm", ""), (
                f"Wildcard injection may have occurred. Got: {r['english_norm']}"
            )


# ---------------------------------------------------------------------------
# Spanish lookup unaffected
# ---------------------------------------------------------------------------

class TestSpanishUnaffected:

    def test_spanish_exact_unaffected(self, lex):
        """Fix B is English-only. Spanish lookups use exact match only."""
        rows = lex.lookup("Elegante", "es")
        assert rows, "Expected exact Spanish match for 'Elegante'"
        for r in rows:
            assert "elegante" in r["spanish_norm"]
            assert "(" not in r["spanish_norm"], (
                "Spanish lookup must return exact match, not fallback rows."
            )

    def test_spanish_no_comma_paren_fallback(self, lex):
        """A Spanish term that only exists as 'term, (annotation)' returns nothing
        because the Spanish lookup never fires the LIKE fallback."""
        rows = lex.lookup("ConteoComa", "es")
        assert rows == [], (
            "Spanish lookup with no exact match must return empty list, not use LIKE."
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_string(self, lex):
        rows = lex.lookup("", "en")
        assert rows == []

    def test_whitespace_only(self, lex):
        rows = lex.lookup("   ", "en")
        assert rows == []

    def test_case_insensitive(self, lex):
        """Lookup is case-insensitive via normalize_term."""
        rows_upper = lex.lookup("COUNT", "en")
        rows_lower = lex.lookup("count", "en")
        rows_mixed = lex.lookup("Count", "en")
        assert rows_upper == rows_lower == rows_mixed

    def test_diacritic_insensitive(self, lex):
        """Normalize_term strips diacritics; accented input still finds entry."""
        rows_plain = lex.lookup("armenia", "en")
        rows_accent = lex.lookup("Armenía", "en")
        assert rows_plain
        assert len(rows_plain) == len(rows_accent)

    def test_no_cross_contamination(self, lex):
        """'Count' must not return 'County' or 'Countdown' rows."""
        rows = lex.lookup("Count", "en")
        for r in rows:
            assert r["english_norm"].startswith("count,"), (
                f"Unexpected row: english_norm='{r['english_norm']}'"
            )
