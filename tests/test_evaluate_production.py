"""
Unit tests for scripts/evaluate_production.py

Covers all pure (no-DB, no-LLM) logic functions:
  - detect_retrieval_path
  - source_key_in_docs
  - passage_in_docs
  - doc_summary
  - compute_summary

Does NOT require a database connection or OpenAI API key.
"""

import importlib.util
import os
import sys
import pytest

# ─── Import the script module ─────────────────────────────────────────────────
# The script lives in scripts/, not in the app package, so we load it directly.

_SCRIPT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "scripts", "evaluate_production.py"
)
_spec = importlib.util.spec_from_file_location("evaluate_production", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)

# Patch sys.modules so relative imports inside the script resolve correctly.
# (The script does sys.path.insert on import, so app.* is resolvable from repo root.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_spec.loader.exec_module(_mod)

detect_retrieval_path = _mod.detect_retrieval_path
source_key_in_docs    = _mod.source_key_in_docs
passage_in_docs       = _mod.passage_in_docs
doc_summary           = _mod.doc_summary
compute_summary       = _mod.compute_summary
QUESTIONS             = _mod.QUESTIONS
SRC_LIBRO_II          = _mod.SRC_LIBRO_II
SRC_DICCIONARIO       = _mod.SRC_DICCIONARIO
SRC_SERIE_I           = _mod.SRC_SERIE_I
SRC_ANTHEM            = _mod.SRC_ANTHEM
SRC_INSTRUMENTS       = _mod.SRC_INSTRUMENTS
SRC_LEXICON           = _mod.SRC_LEXICON
SRC_OVERVIEW          = _mod.SRC_OVERVIEW


# ─── detect_retrieval_path ────────────────────────────────────────────────────

class TestDetectRetrievalPath:
    def test_lexicon_context(self):
        ctx = "<LEXICON_CONTEXT>\nSource: Yurumein Lexicon\n</LEXICON_CONTEXT>"
        assert detect_retrieval_path(ctx) == "lexicon"

    def test_lexicon_note_is_miss(self):
        ctx = "<LEXICON_NOTE>\nThe term 'automobile' was searched ...\n</LEXICON_NOTE>\n[EID-1:DICC]"
        assert detect_retrieval_path(ctx) == "lexicon_miss_hybrid"

    def test_lexicon_context_takes_priority_over_note(self):
        # If both appear (shouldn't happen in practice) LEXICON_CONTEXT wins.
        ctx = "<LEXICON_CONTEXT>foo</LEXICON_CONTEXT>\n<LEXICON_NOTE>bar</LEXICON_NOTE>"
        assert detect_retrieval_path(ctx) == "lexicon"

    def test_anthem_exact(self):
        ctx = '<ANTHEM_EXACT lang="Garífuna">\nWadimalu...\n</ANTHEM_EXACT>'
        assert detect_retrieval_path(ctx) == "anthem_ordered"

    def test_page_lookup(self):
        ctx = "[Source: Libro Serie I, Page 4 | Rank: 1 | Retrieval: page_lookup]\nContent."
        assert detect_retrieval_path(ctx) == "page_lookup"

    def test_page_not_found(self):
        ctx = "[Source: Libro Serie I | Retrieval: page_lookup]\nNOTE: The requested page(s) 500 ..."
        assert detect_retrieval_path(ctx) == "page_lookup"

    def test_general_hybrid(self):
        ctx = "[EID-1:GAR_II:p5 | Source: Libro Garífuna II — page 5 | Retrieval: vector]\nContent."
        assert detect_retrieval_path(ctx) == "hybrid"

    def test_empty_context(self):
        assert detect_retrieval_path("") == "hybrid"

    def test_no_relevant_markers(self):
        assert detect_retrieval_path("No relevant documents found.") == "hybrid"


# ─── source_key_in_docs ──────────────────────────────────────────────────────

class TestSourceKeyInDocs:
    def _doc(self, source: str, content: str = "text") -> dict:
        return {"page_content": content, "metadata": {"source": source}}

    def test_exact_match(self):
        docs = [self._doc("LIBRO GARIFUNA II FORMATO DIGITAL")]
        assert source_key_in_docs(docs, SRC_LIBRO_II) is True

    def test_case_insensitive(self):
        docs = [self._doc("libro garifuna ii formato digital")]
        assert source_key_in_docs(docs, SRC_LIBRO_II) is True

    def test_substring_match(self):
        docs = [self._doc("/var/www/data/DICCIONARIO GARIFUNA (DIMUREI AGEI).pdf")]
        assert source_key_in_docs(docs, SRC_DICCIONARIO) is True

    def test_miss(self):
        docs = [self._doc("LIBRO SERIE I")]
        assert source_key_in_docs(docs, SRC_DICCIONARIO) is False

    def test_empty_docs(self):
        assert source_key_in_docs([], SRC_LIBRO_II) is False

    def test_missing_metadata(self):
        docs = [{"page_content": "some text"}]
        assert source_key_in_docs(docs, SRC_LIBRO_II) is False

    def test_multiple_docs_one_matches(self):
        docs = [
            self._doc("LIBRO SERIE I"),
            self._doc("LIBRO GARIFUNA II FORMATO DIGITAL"),
        ]
        assert source_key_in_docs(docs, SRC_LIBRO_II) is True

    def test_anthem_source_key(self):
        docs = [self._doc("HIMNO_NACIONAL_GUATEMALA_GARIFUNA.docx")]
        assert source_key_in_docs(docs, SRC_ANTHEM) is True

    def test_instruments_source_key(self):
        docs = [self._doc("/docs/MUSICAL INSTRUMENTS GARIFUNA.pdf")]
        assert source_key_in_docs(docs, SRC_INSTRUMENTS) is True


# ─── passage_in_docs ─────────────────────────────────────────────────────────

class TestPassageInDocs:
    def _doc(self, source: str, content: str) -> dict:
        return {"page_content": content, "metadata": {"source": source}}

    def test_fragment_found_in_right_source(self):
        docs = [self._doc("DICCIONARIO GARIFUNA", "Padre. Sus. com. Baba, Tata, Úguchilin.")]
        assert passage_in_docs(docs, "Baba", SRC_DICCIONARIO) is True

    def test_fragment_case_insensitive(self):
        docs = [self._doc("DICCIONARIO GARIFUNA", "baba tata uguchilin")]
        assert passage_in_docs(docs, "BABA", SRC_DICCIONARIO) is True

    def test_fragment_in_wrong_source(self):
        # Fragment present but in a different source than expected.
        docs = [self._doc("LIBRO SERIE I", "Baba means father.")]
        assert passage_in_docs(docs, "Baba", SRC_DICCIONARIO) is False

    def test_fragment_not_found(self):
        docs = [self._doc("DICCIONARIO GARIFUNA", "Madre. Sus. com. Nana.")]
        assert passage_in_docs(docs, "Baba", SRC_DICCIONARIO) is False

    def test_empty_docs(self):
        assert passage_in_docs([], "Satuye", SRC_LIBRO_II) is False

    def test_multiple_docs_correct_one_has_fragment(self):
        docs = [
            self._doc("LIBRO SERIE I", "No baba here."),
            self._doc("DICCIONARIO GARIFUNA", "Padre: Baba, Tata, Úguchilin."),
        ]
        assert passage_in_docs(docs, "Baba", SRC_DICCIONARIO) is True

    def test_partial_fragment_match(self):
        # Fragment "afric" matches "africano" in content.
        docs = [self._doc("LIBRO GARIFUNA II FORMATO DIGITAL", "herencia africana occidental")]
        assert passage_in_docs(docs, "afric", SRC_LIBRO_II) is True

    def test_yurumein_fragment(self):
        docs = [self._doc("LIBRO GARIFUNA II FORMATO DIGITAL", "llamaron a San Vicente 'Yurumein'")]
        assert passage_in_docs(docs, "Yurumein", SRC_LIBRO_II) is True


# ─── doc_summary ─────────────────────────────────────────────────────────────

class TestDocSummary:
    def test_known_source_classified(self):
        doc = {
            "page_content": "Padre. Sus. com. Baba, Tata, Úguchilin.",
            "metadata": {
                "source": "DICCIONARIO GARIFUNA (DIMUREI AGEI).pdf",
                "page_label": "144",
            },
            "hybrid_score": 0.823456,
            "retrieval_source": "hybrid",
        }
        s = doc_summary(doc)
        assert s["source_role"] == "DICTIONARY"
        assert s["source_display"] == "Diccionario Garífuna"
        assert s["page_label"] == "144"
        assert s["hybrid_score"] == 0.8235
        assert s["retrieval_source"] == "hybrid"
        assert "Baba" in s["content_preview"]

    def test_unknown_source(self):
        doc = {
            "page_content": "Some content.",
            "metadata": {"source": "UNKNOWN_DOC.pdf"},
            "hybrid_score": 0.3,
            "retrieval_source": "vector",
        }
        s = doc_summary(doc)
        assert s["source_role"] is None
        assert s["source_display"] == "unknown"

    def test_content_truncated_at_140(self):
        doc = {
            "page_content": "A" * 200,
            "metadata": {"source": "LIBRO SERIE I"},
            "hybrid_score": 0.5,
        }
        s = doc_summary(doc)
        assert len(s["content_preview"]) <= 140

    def test_newlines_replaced(self):
        doc = {
            "page_content": "line1\nline2\nline3",
            "metadata": {"source": "LIBRO GARIFUNA II FORMATO DIGITAL"},
            "hybrid_score": 0.4,
        }
        s = doc_summary(doc)
        assert "\n" not in s["content_preview"]

    def test_missing_fields_handled(self):
        s = doc_summary({})
        assert s["source_role"] is None
        assert s["hybrid_score"] == 0.0
        assert s["content_preview"] == ""


# ─── compute_summary ─────────────────────────────────────────────────────────

def _make_result(
    qid: str,
    cat: str,
    expected_key: str | None,
    expected_frag: str | None,
    hit10: bool | None,
    hit_ctx: bool | None,
    pfound10: bool | None = None,
    pfound_ctx: bool | None = None,
    top10_applicable: bool = True,
    error: str | None = None,
) -> dict:
    return {
        "id": qid,
        "category": cat,
        "expected_source_key": expected_key,
        "expected_passage_fragment": expected_frag,
        "top10_applicable": top10_applicable,
        "hit_at_10": hit10,
        "hit_in_context": hit_ctx,
        "passage_found_at_10": pfound10,
        "passage_found_in_context": pfound_ctx,
        "error": error,
    }


class TestComputeSummary:
    def test_all_hits(self):
        results = [
            _make_result("Q1", "CatA", "SRC", None, True, True),
            _make_result("Q2", "CatA", "SRC", None, True, True),
        ]
        s = compute_summary(results)
        assert s["total_questions"] == 2
        assert s["errors"] == 0
        assert s["sourced_questions"] == 2
        assert s["edge_case_questions"] == 0
        assert s["source_hit_rate_at_10"] == 1.0
        assert s["source_hit_rate_in_context"] == 1.0

    def test_all_misses(self):
        results = [
            _make_result("Q1", "CatA", "SRC", None, False, False),
            _make_result("Q2", "CatA", "SRC", None, False, False),
        ]
        s = compute_summary(results)
        assert s["source_hit_rate_at_10"] == 0.0
        assert s["source_hit_rate_in_context"] == 0.0

    def test_edge_cases_excluded_from_hit_rate(self):
        results = [
            _make_result("Q1", "CatA", "SRC", None, True, True),
            _make_result("E1", "Edge", None,  None, None, None),
        ]
        s = compute_summary(results)
        assert s["sourced_questions"] == 1
        assert s["edge_case_questions"] == 1
        assert s["source_hit_rate_at_10"] == 1.0

    def test_anthem_path_excluded_from_top10(self):
        # anthem_ordered path: top10_applicable=False.
        results = [
            _make_result("Q1", "Anthem", "SRC_ANTHEM", None, None, True, top10_applicable=False),
            _make_result("Q2", "Hybrid", "SRC_OTHER", None, True, True, top10_applicable=True),
        ]
        s = compute_summary(results)
        assert s["hybrid_path_questions"] == 1
        assert s["source_hit_rate_at_10"] == 1.0   # only Q2 counted
        assert s["source_hit_rate_in_context"] == 1.0  # both counted

    def test_passage_recall(self):
        results = [
            _make_result("Q1", "CatA", "SRC", "fragment", True, True, True, True),
            _make_result("Q2", "CatA", "SRC", "fragment", True, False, True, False),
            _make_result("Q3", "CatA", "SRC", None, True, True),  # no fragment
        ]
        s = compute_summary(results)
        assert s["passage_recall_at_10"] == 1.0     # 2/2 passage questions hit@10
        assert s["passage_recall_in_context"] == 0.5  # 1/2

    def test_error_counted(self):
        results = [
            _make_result("Q1", "CatA", "SRC", None, None, None, error="Connection refused"),
            _make_result("Q2", "CatA", "SRC", None, True, True),
        ]
        s = compute_summary(results)
        assert s["errors"] == 1

    def test_no_sourced_questions_returns_none_rates(self):
        results = [_make_result("E1", "Edge", None, None, None, None)]
        s = compute_summary(results)
        assert s["source_hit_rate_at_10"] is None
        assert s["source_hit_rate_in_context"] is None

    def test_category_breakdown(self):
        results = [
            _make_result("Q1", "Baliceaux", "SRC", None, True, True),
            _make_result("Q2", "Baliceaux", "SRC", None, False, False),
            _make_result("Q3", "Anthem",    "SRC", None, None, True, top10_applicable=False),
        ]
        s = compute_summary(results)
        b = s["by_category"]["Baliceaux"]
        a = s["by_category"]["Anthem"]
        assert b["total"] == 2
        assert b["hit_at_10"] == 1
        assert b["hit_in_context"] == 1
        assert a["hit_in_context"] == 1


# ─── QUESTIONS list integrity ─────────────────────────────────────────────────

class TestQuestionsIntegrity:
    def test_exactly_40_questions(self):
        assert len(QUESTIONS) == 40

    def test_all_have_required_fields(self):
        required = {"id", "category", "language", "query",
                    "expected_source_key", "expected_passage_fragment", "notes"}
        for q in QUESTIONS:
            missing = required - q.keys()
            assert not missing, f"{q['id']} is missing fields: {missing}"

    def test_ids_unique(self):
        ids = [q["id"] for q in QUESTIONS]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"

    def test_eight_categories(self):
        cats = {q["category"] for q in QUESTIONS}
        assert len(cats) == 8, f"Expected 8 categories, got {len(cats)}: {cats}"

    def test_five_per_category(self):
        from collections import Counter
        counts = Counter(q["category"] for q in QUESTIONS)
        for cat, count in counts.items():
            assert count == 5, f"Category '{cat}' has {count} questions (expected 5)"

    def test_languages_are_valid(self):
        for q in QUESTIONS:
            assert q["language"] in ("en", "es"), f"{q['id']} has invalid language: {q['language']}"

    def test_expected_source_keys_are_known_constants(self):
        known = {
            SRC_LIBRO_II, SRC_DICCIONARIO, SRC_SERIE_I,
            SRC_ANTHEM, SRC_INSTRUMENTS, SRC_LEXICON,
            SRC_OVERVIEW, None,
        }
        for q in QUESTIONS:
            assert q["expected_source_key"] in known, (
                f"{q['id']} has unknown source key: {q['expected_source_key']!r}"
            )

    def test_edge_cases_have_null_source(self):
        edge = [q for q in QUESTIONS if q["category"] == "Edge_Cases"]
        for q in edge:
            assert q["expected_source_key"] is None, (
                f"Edge case {q['id']} should have null expected_source_key"
            )

    def test_anthem_questions_all_target_anthem_source(self):
        anthem_q = [q for q in QUESTIONS if q["category"] == "National_Anthem"]
        for q in anthem_q:
            assert q["expected_source_key"] == SRC_ANTHEM, (
                f"Anthem question {q['id']} has unexpected source key: {q['expected_source_key']}"
            )

    def test_no_empty_queries(self):
        for q in QUESTIONS:
            assert q["query"].strip(), f"{q['id']} has an empty query"
