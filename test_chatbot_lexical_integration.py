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

    # 5. Parser match + zero lexicon rows → falls through to existing RAG
    def test_parser_match_zero_rows_falls_through_to_rag(self):
        with patch(_PATCH) as MockLexicon, \
             patch.object(self.nodes, "_vector_retrieve", return_value=_RAG_DOCS) as mock_retrieve:
            MockLexicon.return_value.lookup.return_value = []
            result = _run(self.nodes.global_retrieval_process(
                _make_state("How do you say xenolith in Garífuna?")
            ))

        mock_retrieve.assert_called_once()
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
