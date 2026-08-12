"""
Unit tests for Chatbot Lexical Integration Phase 1 — post-reconciliation.

Row shape contract (Decision 1):
  Each row's garifuna field is an opaque documented source entry.
  No splitting on commas, semicolons, or any other punctuation.
  Deduplication applies only to exact duplicate complete strings.

Normalization contract (Decision 2):
  normalize_term() does accent-folding (NFD + strip Mn + NFKC + lowercase).
  "árbol" → "arbol", "Cómo" → "como", etc.
  Garífuna text itself is never normalized or modified.

Tests:
   1. English single match → LEXICON_CONTEXT returned
   2. Accept (one row, comma-separated garifuna) → preserved as one opaque entry
   3. Spanish single match → LEXICON_CONTEXT with Spanish label
   4. Spanish accented term (árbol) → raw term preserved in context; normalization contract verified
   5. Parser match + zero lexicon rows → existing RAG fallback
   6. Unrelated cultural question → lexicon never queried, existing RAG path
   7. Unsupported lexical phrasing → existing RAG path
   8. Lexicon hit skips PGVector retrieval
   9. Duplicate complete garifuna strings across rows → deduplicated (wedding ring pattern)
  10. Multiple distinct complete garifuna strings → all preserved (wedding ring pattern)

Normalization:
  11. normalize_term strips diacritics from Spanish lookup keys
  12. normalize_term preserves terms that carry no diacritics
  13. normalize_term handles None / empty input
  14. normalize_term collapses internal whitespace
"""

import asyncio
from unittest.mock import patch

from app.services.lexicon.lexicon import normalize_term
from app.services.utils.RAG_pipeline.workflow.nodes import Nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _make_state(query: str) -> dict:
    return {"messages": [], "user_query": query, "file": None}


def _make_row(garifuna: str, english: str = "word", spanish: str = None) -> dict:
    """Build a mock lexicon row using the same normalization as production."""
    return {
        "id": 1,
        "english": english,
        "english_norm": normalize_term(english),
        "spanish": spanish,
        "spanish_norm": normalize_term(spanish) if spanish else None,
        "garifuna": garifuna,
    }


_RAG_DOCS = [
    {
        "page_content": "The Garífuna are an Afro-Indigenous people of the Caribbean coast.",
        "metadata": {"source": "test.pdf"},
        "vector_score": 0.1,
        "keyword_score": 0.5,
        "hybrid_score": 0.3,
        "retrieval_source": "vector",
        "initial_rank": 1,
    }
]

_PATCH = "app.services.utils.RAG_pipeline.workflow.nodes.YurumeinLexicon"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestChatbotLexicalIntegration:

    def setup_method(self):
        self.nodes = Nodes()

    # 1. English single match → LEXICON_CONTEXT returned
    def test_english_single_match_returns_lexicon_context(self):
        rows = [_make_row("Duna", english="water")]

        with patch(_PATCH) as MockLexicon:
            MockLexicon.return_value.lookup.return_value = rows
            result = _run(self.nodes.global_retrieval_process(
                _make_state("How do you say water in Garífuna?")
            ))

        ctx = result["context"]
        assert "<LEXICON_CONTEXT>" in ctx
        assert "</LEXICON_CONTEXT>" in ctx
        assert "water" in ctx
        assert "Duna" in ctx
        assert "English" in ctx
        assert "yurumein_lexicon" in ctx

    # 2. Accept → one row, comma-separated garifuna → preserved as one opaque entry
    def test_accept_one_row_garifuna_preserved_as_opaque_entry(self):
        # Production reality: one DB row whose garifuna field is a comma-separated
        # string. The integration must not split it.
        rows = [_make_row("Anhan, Ánhara, Resibi, Resibiwati", english="accept")]

        with patch(_PATCH) as MockLexicon:
            MockLexicon.return_value.lookup.return_value = rows
            result = _run(self.nodes.global_retrieval_process(
                _make_state("How do you say accept in Garífuna?")
            ))

        ctx = result["context"]
        assert "<LEXICON_CONTEXT>" in ctx
        # Full string must appear as one documented entry
        assert "- Anhan, Ánhara, Resibi, Resibiwati" in ctx
        # Must not be split into four separate bullets
        assert ctx.count("- Anhan") == 1
        assert ctx.count("- Ánhara") == 0
        assert ctx.count("- Resibi") == 0
        assert ctx.count("- Resibiwati") == 0

    # 3. Spanish single match → LEXICON_CONTEXT with Spanish label
    def test_spanish_single_match_returns_lexicon_context(self):
        rows = [_make_row("Duna", spanish="agua")]

        with patch(_PATCH) as MockLexicon:
            MockLexicon.return_value.lookup.return_value = rows
            result = _run(self.nodes.global_retrieval_process(
                _make_state("¿Cómo se dice agua en garífuna?")
            ))

        ctx = result["context"]
        assert "<LEXICON_CONTEXT>" in ctx
        assert "agua" in ctx
        assert "Duna" in ctx
        assert "Spanish" in ctx

    # 4. Spanish accented term → raw term preserved in context
    def test_spanish_accented_term_raw_form_preserved_in_context(self):
        # The parser extracts "árbol" (raw, with accent).
        # normalize_term("árbol") → "arbol" happens inside lookup() against the DB.
        # The context must show "árbol" (the user's raw term), not "arbol".
        rows = [_make_row("Nuiñu", spanish="árbol")]

        with patch(_PATCH) as MockLexicon:
            MockLexicon.return_value.lookup.return_value = rows
            result = _run(self.nodes.global_retrieval_process(
                _make_state("¿Cómo se dice árbol en garífuna?")
            ))

        ctx = result["context"]
        # Raw accented term is visible in the context (not the normalized form)
        assert "árbol" in ctx
        assert "arbol" not in ctx.split("árbol")[0]  # "árbol" not silently replaced
        assert "Nuiñu" in ctx

    # 5. Parser match + zero lexicon rows → RP10 hybrid retrieval (not _vector_retrieve)
    def test_parser_match_zero_rows_triggers_hybrid_retrieval(self):
        # RP10: lexical MISS must call _hybrid_retrieve with the extracted term,
        # not the full conversational sentence. _vector_retrieve is the generic
        # RAG path and must NOT be called for a lexical miss.
        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_hybrid_retrieve", return_value=_RAG_DOCS) as mock_hybrid, \
             patch.object(self.nodes, "_vector_retrieve") as mock_vector:
            MockLexicon.return_value.lookup.return_value = []
            result = _run(self.nodes.global_retrieval_process(
                _make_state("How do you say xenolith in Garífuna?")
            ))

        mock_hybrid.assert_called_once()
        mock_vector.assert_not_called()
        query_used = mock_hybrid.call_args.kwargs["query"]
        assert "xenolith" in query_used
        # RP10: retrieval query is the extracted term alone — not the full
        # conversational sentence and not "term garifuna". Adding "garifuna"
        # boosts overview/history documents over vocabulary documents.
        assert "How do you say" not in query_used
        assert query_used.strip() == "xenolith"
        assert "<LEXICON_MISS>" in result["context"]
        assert "<LEXICON_CONTEXT>" not in result["context"]

    # 6. Unrelated cultural question → lexicon never instantiated, RAG path used
    def test_cultural_question_skips_lexicon_uses_rag(self):
        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_vector_retrieve", return_value=_RAG_DOCS):
            result = _run(self.nodes.global_retrieval_process(
                _make_state("What is the history of the Garífuna people?")
            ))
            MockLexicon.assert_not_called()

        assert "<LEXICON_CONTEXT>" not in result["context"]

    # 7. Unsupported lexical phrasing → RAG path, lexicon never queried
    def test_unsupported_lexical_phrasing_uses_rag(self):
        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_vector_retrieve", return_value=_RAG_DOCS):
            result = _run(self.nodes.global_retrieval_process(
                _make_state("What does hello mean in Garífuna?")
            ))
            MockLexicon.assert_not_called()

        assert "<LEXICON_CONTEXT>" not in result["context"]

    # 8. Lexicon hit skips _vector_retrieve entirely
    def test_lexicon_hit_skips_pgvector_retrieval(self):
        rows = [_make_row("Duna", english="water")]

        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_vector_retrieve") as mock_retrieve:
            MockLexicon.return_value.lookup.return_value = rows
            _run(self.nodes.global_retrieval_process(
                _make_state("How do you say water in Garífuna?")
            ))

        mock_retrieve.assert_not_called()

    # 9. Duplicate complete garifuna strings across rows → deduplicated (wedding ring pattern)
    def test_duplicate_complete_garifuna_strings_are_deduplicated(self):
        # Two distinct Spanish phrasings (different spanish_norm) that produce
        # the same Garífuna form. Only one bullet should appear.
        rows = [
            _make_row("Bagü lani amareiduni", english="wedding ring"),
            _make_row("Bagü lani amareiduni", english="wedding ring"),  # exact garifuna duplicate
        ]

        with patch(_PATCH) as MockLexicon:
            MockLexicon.return_value.lookup.return_value = rows
            result = _run(self.nodes.global_retrieval_process(
                _make_state("How do you say wedding ring in Garífuna?")
            ))

        assert result["context"].count("- Bagü lani amareiduni") == 1

    # 10. Multiple distinct complete garifuna strings → all preserved (wedding ring pattern)
    def test_multiple_distinct_garifuna_rows_all_preserved(self):
        # Two DB rows with distinct Garífuna values for the same English term.
        # Both must appear as separate documented entries.
        rows = [
            _make_row("Bagü lani amareiduni", english="wedding ring"),
            _make_row("Bagü lun amareiduni", english="wedding ring"),
        ]

        with patch(_PATCH) as MockLexicon:
            MockLexicon.return_value.lookup.return_value = rows
            result = _run(self.nodes.global_retrieval_process(
                _make_state("How do you say wedding ring in Garífuna?")
            ))

        ctx = result["context"]
        assert "- Bagü lani amareiduni" in ctx
        assert "- Bagü lun amareiduni" in ctx
        assert ctx.count("- ") == 2


# ---------------------------------------------------------------------------
# Normalization contract tests
# ---------------------------------------------------------------------------

class TestNormalizationContract:
    """
    Direct unit tests for normalize_term's accent-folding behavior.

    These confirm that local normalization now matches the production contract:
    lookup keys are accent-folded so that "árbol" resolves to "arbol" in the DB.
    Garífuna text is never passed through normalize_term.
    """

    # 11. Spanish accented chars are folded for lookup keys
    def test_spanish_accented_chars_are_accent_folded(self):
        assert normalize_term("árbol") == "arbol"
        assert normalize_term("corazón") == "corazon"
        assert normalize_term("aceptar") == "aceptar"   # no accent → unchanged

    # 12. English terms without diacritics are unaffected
    def test_english_terms_without_diacritics_unchanged(self):
        assert normalize_term("water") == "water"
        assert normalize_term("Wedding Ring") == "wedding ring"
        assert normalize_term("Accept") == "accept"

    # 13. None / empty input returns empty string
    def test_none_and_empty_input_return_empty_string(self):
        assert normalize_term(None) == ""
        assert normalize_term("") == ""
        assert normalize_term("   ") == ""

    # 14. Internal whitespace is collapsed
    def test_internal_whitespace_collapsed(self):
        assert normalize_term("wedding  ring") == "wedding ring"
        assert normalize_term("  buenos  días  ") == "buenos dias"


# ---------------------------------------------------------------------------
# RP10 fallback retrieval-query construction (Phase 4)
# ---------------------------------------------------------------------------

class TestRP10LexicalMissFallback:
    """
    Verifies RP10: on a lexical MISS the retrieval query is built from the
    extracted term, not from the full conversational sentence.

    Tests A–D exercise English MISS variants.
    Test E–F verify generic RAG queries remain completely untouched.
    Test G verifies the Spanish MISS path constructs a language-appropriate query.
    """

    def setup_method(self):
        self.nodes = Nodes()

    # A. Lexicon HIT — deterministic result, secondary RAG must not be called
    def test_A_lexicon_hit_uses_deterministic_result_only(self):
        rows = [_make_row("Adurugutu", english="abandoned")]

        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_hybrid_retrieve") as mock_hybrid, \
             patch.object(self.nodes, "_vector_retrieve") as mock_vector:
            MockLexicon.return_value.lookup.return_value = rows
            result = _run(self.nodes.global_retrieval_process(
                _make_state("What is the garifuna word for abandoned?")
            ))

        ctx = result["context"]
        assert "<LEXICON_CONTEXT>" in ctx
        assert "abandoned" in ctx
        assert "Adurugutu" in ctx
        mock_hybrid.assert_not_called()
        mock_vector.assert_not_called()

    # B. Lexicon MISS, long form — fallback uses extracted term, not full sentence
    def test_B_lexical_miss_father_long_form_builds_term_query(self):
        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_hybrid_retrieve", return_value=_RAG_DOCS) as mock_hybrid:
            MockLexicon.return_value.lookup.return_value = []
            result = _run(self.nodes.global_retrieval_process(
                _make_state("How do you say father in garifuna?")
            ))

        mock_hybrid.assert_called_once()
        query_used = mock_hybrid.call_args.kwargs["query"]
        # RP10: query is the extracted term alone — not "father garifuna".
        # "garifuna" as a keyword floods the corpus (present in every document
        # title/context) and boosts overview/history chunks over vocabulary chunks.
        assert query_used.strip() == "father"
        assert "How do you say" not in query_used
        assert "<LEXICON_MISS>" in result["context"]
        assert "<LEXICON_CONTEXT>" not in result["context"]

    # C. Lexicon MISS, short form — same RP10 behavior
    def test_C_lexical_miss_father_short_form_same_behavior(self):
        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_hybrid_retrieve", return_value=_RAG_DOCS) as mock_hybrid:
            MockLexicon.return_value.lookup.return_value = []
            result = _run(self.nodes.global_retrieval_process(
                _make_state("How do you say father?")
            ))

        mock_hybrid.assert_called_once()
        query_used = mock_hybrid.call_args.kwargs["query"]
        assert query_used.strip() == "father"
        assert "How do you say" not in query_used
        assert "<LEXICON_MISS>" in result["context"]

    # D. Lexicon MISS, repeated punctuation — same RP10 behavior
    def test_D_lexical_miss_father_repeated_punctuation_same_behavior(self):
        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_hybrid_retrieve", return_value=_RAG_DOCS) as mock_hybrid:
            MockLexicon.return_value.lookup.return_value = []
            result = _run(self.nodes.global_retrieval_process(
                _make_state("How do you say father??")
            ))

        mock_hybrid.assert_called_once()
        query_used = mock_hybrid.call_args.kwargs["query"]
        assert query_used.strip() == "father"
        assert "<LEXICON_MISS>" in result["context"]

    # E. Cultural query — generic _vector_retrieve path, not _hybrid_retrieve
    def test_E_cultural_query_uses_vector_retrieve_not_hybrid(self):
        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_hybrid_retrieve") as mock_hybrid, \
             patch.object(self.nodes, "_vector_retrieve", return_value=_RAG_DOCS) as mock_vector:
            result = _run(self.nodes.global_retrieval_process(
                _make_state("What is the Dügü?")
            ))

        MockLexicon.assert_not_called()
        mock_hybrid.assert_not_called()
        mock_vector.assert_called_once()
        assert "<LEXICON_MISS>" not in result["context"]
        assert "<LEXICON_CONTEXT>" not in result["context"]

    # F. Historical query — same generic path
    def test_F_historical_query_uses_vector_retrieve_not_hybrid(self):
        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_hybrid_retrieve") as mock_hybrid, \
             patch.object(self.nodes, "_vector_retrieve", return_value=_RAG_DOCS) as mock_vector:
            result = _run(self.nodes.global_retrieval_process(
                _make_state("What are the origins of the Garífuna people?")
            ))

        MockLexicon.assert_not_called()
        mock_hybrid.assert_not_called()
        mock_vector.assert_called_once()
        assert "<LEXICON_MISS>" not in result["context"]

    # G. Spanish lexical MISS — query uses the extracted Spanish term, not the
    # full Spanish conversational sentence, and includes "garifuna" as signal
    def test_G_spanish_lexical_miss_uses_term_based_query(self):
        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_hybrid_retrieve", return_value=_RAG_DOCS) as mock_hybrid:
            MockLexicon.return_value.lookup.return_value = []
            result = _run(self.nodes.global_retrieval_process(
                _make_state("¿Cómo se dice padre en garífuna?")
            ))

        mock_hybrid.assert_called_once()
        query_used = mock_hybrid.call_args.kwargs["query"]
        # RP10: extracted Spanish term used as retrieval signal, not the
        # full conversational sentence. Term alone is correct for both
        # English and Spanish (same corpus-omnipresence argument for "garifuna").
        assert query_used.strip() == "padre"
        assert "Cómo" not in query_used
        assert "dice" not in query_used
        assert "<LEXICON_MISS>" in result["context"]
        assert "<LEXICON_CONTEXT>" not in result["context"]
