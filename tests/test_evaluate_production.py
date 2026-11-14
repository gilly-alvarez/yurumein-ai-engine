"""
Unit tests for scripts/evaluate_production.py

Tests pure-logic functions only — no network, no database, no LLM calls.
Imports the script as a module via importlib so it can be run from the
repo root without altering sys.path permanently.

Run from repo root:
    .venv/bin/python3 -m pytest tests/test_evaluate_production.py -v
"""

import importlib.util
import os
import sys

import pytest

# ─── Load the script module ───────────────────────────────────────────────────

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "evaluate_production.py",
)

spec = importlib.util.spec_from_file_location("evaluate_production", _SCRIPT)
_mod = importlib.util.module_from_spec(spec)

# The script does top-level imports that reach the application.
# Insert the repo root so relative app imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

spec.loader.exec_module(_mod)

# ─── Import symbols from module ───────────────────────────────────────────────

detect_retrieval_path = _mod.detect_retrieval_path
source_key_in_docs    = _mod.source_key_in_docs
any_source_in_docs    = _mod.any_source_in_docs
passage_in_docs       = _mod.passage_in_docs
any_passage_in_docs   = _mod.any_passage_in_docs
doc_summary           = _mod.doc_summary
compute_summary       = _mod.compute_summary
QUESTIONS             = _mod.QUESTIONS

SRC_DICCIONARIO       = _mod.SRC_DICCIONARIO
SRC_SERIE_I           = _mod.SRC_SERIE_I
SRC_LIBRO_II          = _mod.SRC_LIBRO_II
SRC_LEXICON           = _mod.SRC_LEXICON
SRC_ANTHEM            = _mod.SRC_ANTHEM
SRC_INSTRUMENTS       = _mod.SRC_INSTRUMENTS
SRC_OVERVIEW          = _mod.SRC_OVERVIEW
SRC_BALICEAUX         = _mod.SRC_BALICEAUX
SRC_CHATOYER          = _mod.SRC_CHATOYER
SRC_HISTORIA          = _mod.SRC_HISTORIA
SRC_GARIFUNA_IDENTITY = _mod.SRC_GARIFUNA_IDENTITY
SRC_ARCHIVO_BILINGUE  = _mod.SRC_ARCHIVO_BILINGUE
SRC_ARCHIVO           = _mod.SRC_ARCHIVO


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _doc(source: str, content: str, page_label=None) -> dict:
    return {
        "page_content": content,
        "metadata": {"source": source, "page_label": page_label},
        "hybrid_score": 0.9,
        "retrieval_source": "hybrid",
    }


# ─── TestDetectRetrievalPath ──────────────────────────────────────────────────

class TestDetectRetrievalPath:
    def test_lexicon_hit(self):
        ctx = "Result from SQL lexicon lookup\n<LEXICON_CONTEXT>...</LEXICON_CONTEXT>"
        assert detect_retrieval_path(ctx) == "lexicon"

    def test_lexicon_miss(self):
        ctx = "No lexicon match found.\n<LEXICON_NOTE>term not in pilot</LEXICON_NOTE>"
        assert detect_retrieval_path(ctx) == "lexicon_miss_hybrid"

    def test_anthem_exact(self):
        ctx = '<ANTHEM_EXACT lang="Garífuna">Louverture...</ANTHEM_EXACT>'
        assert detect_retrieval_path(ctx) == "anthem_ordered"

    def test_anthem_no_lang_attribute(self):
        # Even without attributes, the tag prefix must match.
        ctx = "<ANTHEM_EXACT>Some anthem text</ANTHEM_EXACT>"
        assert detect_retrieval_path(ctx) == "anthem_ordered"

    def test_page_lookup(self):
        ctx = "page_lookup: source=%LIBRO SERIE I%, pages=(5,)"
        assert detect_retrieval_path(ctx) == "page_lookup"

    def test_hybrid_fallback(self):
        ctx = "Some retrieved context from vector search."
        assert detect_retrieval_path(ctx) == "hybrid"

    def test_empty_context(self):
        assert detect_retrieval_path("") == "hybrid"

    def test_lexicon_note_takes_priority_over_hybrid(self):
        ctx = "Hybrid context here. <LEXICON_NOTE>miss</LEXICON_NOTE>"
        assert detect_retrieval_path(ctx) == "lexicon_miss_hybrid"

    def test_lexicon_context_beats_all(self):
        ctx = "<LEXICON_CONTEXT>data</LEXICON_CONTEXT> page_lookup included"
        assert detect_retrieval_path(ctx) == "lexicon"


# ─── TestSourceKeyInDocs ─────────────────────────────────────────────────────

class TestSourceKeyInDocs:
    def test_exact_match(self):
        docs = [_doc("/app/data/DICCIONARIO GARIFUNA.pdf", "content")]
        assert source_key_in_docs(docs, SRC_DICCIONARIO)

    def test_case_insensitive(self):
        docs = [_doc("/app/data/diccionario garifuna.pdf", "content")]
        assert source_key_in_docs(docs, "DICCIONARIO GARIFUNA")

    def test_partial_path_match(self):
        docs = [_doc("/full/path/to/LIBRO SERIE I.pdf", "content")]
        assert source_key_in_docs(docs, SRC_SERIE_I)

    def test_no_match(self):
        docs = [_doc("/app/data/SOME_OTHER_FILE.pdf", "content")]
        assert not source_key_in_docs(docs, SRC_DICCIONARIO)

    def test_empty_docs(self):
        assert not source_key_in_docs([], SRC_LIBRO_II)

    def test_new_doc_baliceaux(self):
        docs = [_doc("/app/.../Baliceaux_Yurumein_AI_Historical_Brief.docx", "content")]
        assert source_key_in_docs(docs, SRC_BALICEAUX)

    def test_new_doc_chatoyer(self):
        docs = [_doc("/app/.../Yurumein_AI_Chief_Joseph_Chatoyer_EN_ES.docx", "text")]
        assert source_key_in_docs(docs, SRC_CHATOYER)

    def test_new_doc_historia(self):
        docs = [_doc("/app/.../Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt", "text")]
        assert source_key_in_docs(docs, SRC_HISTORIA)

    def test_new_doc_identity(self):
        docs = [_doc("/app/.../Yurumein_Garifuna_Identity_EN_ES.docx", "text")]
        assert source_key_in_docs(docs, SRC_GARIFUNA_IDENTITY)

    def test_new_doc_archivo_bilingue(self):
        docs = [_doc("/app/.../Yurumei_2024_YurumeinAI_Archivo_Bilingue.docx", "text")]
        assert source_key_in_docs(docs, SRC_ARCHIVO_BILINGUE)

    def test_new_doc_archivo_plain(self):
        docs = [_doc("/app/.../Yurumei_2024_Archivo.docx", "text")]
        assert source_key_in_docs(docs, SRC_ARCHIVO)

    def test_archivo_key_does_not_match_bilingue(self):
        # SRC_ARCHIVO ("Yurumei_2024_Archivo.docx") must NOT match the Bilingue path.
        docs = [_doc("/app/.../Yurumei_2024_YurumeinAI_Archivo_Bilingue.docx", "text")]
        assert not source_key_in_docs(docs, SRC_ARCHIVO)

    def test_instruments_underscore_key_matches(self):
        # Fixed: Musical_Instruments_Dataset (underscore) matches the actual filename.
        docs = [_doc("/app/.../Garifuna_Musical_Instruments_Dataset.pdf.pdf", "text")]
        assert source_key_in_docs(docs, SRC_INSTRUMENTS)

    def test_instruments_space_key_does_not_match(self):
        # Regression guard: the old "MUSICAL INSTRUMENTS" (space) constant was wrong.
        docs = [_doc("/app/.../Garifuna_Musical_Instruments_Dataset.pdf.pdf", "text")]
        assert not source_key_in_docs(docs, "MUSICAL INSTRUMENTS")

    def test_anthem_key_matches(self):
        docs = [_doc(
            "/app/.../Himno_Nacional_Guatemala_Garifuna_Espanol_English_Yurumein_AI.docx",
            "text",
        )]
        assert source_key_in_docs(docs, SRC_ANTHEM)

    def test_missing_metadata(self):
        doc = {"page_content": "content"}
        assert not source_key_in_docs([doc], SRC_DICCIONARIO)


# ─── TestAnySourceInDocs ─────────────────────────────────────────────────────

class TestAnySourceInDocs:
    def test_first_key_matches(self):
        docs = [_doc("/app/.../Baliceaux_Yurumein_AI_Historical_Brief.docx", "text")]
        assert any_source_in_docs(docs, [SRC_BALICEAUX, SRC_CHATOYER])

    def test_second_key_matches(self):
        docs = [_doc("/app/.../Yurumein_AI_Chief_Joseph_Chatoyer_EN_ES.docx", "text")]
        assert any_source_in_docs(docs, [SRC_BALICEAUX, SRC_CHATOYER])

    def test_no_key_matches(self):
        docs = [_doc("/app/.../DICCIONARIO GARIFUNA.pdf", "text")]
        assert not any_source_in_docs(docs, [SRC_BALICEAUX, SRC_CHATOYER])

    def test_empty_keys(self):
        docs = [_doc("/app/.../DICCIONARIO GARIFUNA.pdf", "text")]
        assert not any_source_in_docs(docs, [])

    def test_empty_docs(self):
        assert not any_source_in_docs([], [SRC_BALICEAUX, SRC_CHATOYER])


# ─── TestPassageInDocs ────────────────────────────────────────────────────────

class TestPassageInDocs:
    def test_fragment_and_source_match(self):
        docs = [_doc(
            "/app/.../Baliceaux_Yurumein_AI_Historical_Brief.docx",
            "In 1796 the British transferred thousands of Garífuna to Baliceaux.",
        )]
        assert passage_in_docs(docs, "1796", SRC_BALICEAUX)

    def test_fragment_case_insensitive(self):
        docs = [_doc(
            "/app/.../Baliceaux_Yurumein_AI_Historical_Brief.docx",
            "In 1796 the British transferred...",
        )]
        assert passage_in_docs(docs, "1796", SRC_BALICEAUX)

    def test_fragment_missing(self):
        docs = [_doc(
            "/app/.../Baliceaux_Yurumein_AI_Historical_Brief.docx",
            "No date mentioned here.",
        )]
        assert not passage_in_docs(docs, "1796", SRC_BALICEAUX)

    def test_source_mismatch(self):
        docs = [_doc(
            "/app/.../DICCIONARIO GARIFUNA.pdf",
            "1796 — but wrong source.",
        )]
        assert not passage_in_docs(docs, "1796", SRC_BALICEAUX)

    def test_fragment_found_in_second_doc(self):
        docs = [
            _doc("/app/.../DICCIONARIO GARIFUNA.pdf", "unrelated"),
            _doc("/app/.../Yurumein_AI_Chief_Joseph_Chatoyer_EN_ES.docx", "Satuye was a chief"),
        ]
        assert passage_in_docs(docs, "Satuye", SRC_CHATOYER)

    def test_empty_docs(self):
        assert not passage_in_docs([], "1796", SRC_BALICEAUX)

    def test_1773_fragment_chatoyer(self):
        docs = [_doc(
            "/app/.../Yurumein_AI_Chief_Joseph_Chatoyer_EN_ES.docx",
            "The 1773 peace agreement established boundaries.",
        )]
        assert passage_in_docs(docs, "1773", SRC_CHATOYER)

    def test_ben_petersen_fragment_identity(self):
        docs = [_doc(
            "/app/.../Yurumein_Garifuna_Identity_EN_ES.docx",
            "Curated reference derived from A Story About the Garifuna (Ben Petersen, 2012/2013)",
        )]
        assert passage_in_docs(docs, "Ben Petersen", SRC_GARIFUNA_IDENTITY)

    def test_mayuru_fragment_archivo(self):
        docs = [_doc(
            "/app/.../Yurumei_2024_Archivo.docx",
            "El documental reconoce a Marcos Sánchez Díaz, también llamado Mayuru.",
        )]
        assert passage_in_docs(docs, "Mayuru", SRC_ARCHIVO)

    def test_hijos_de_livingston_fragment(self):
        docs = [_doc(
            "/app/.../Yurumei_2024_YurumeinAI_Archivo_Bilingue.docx",
            "Hijos de Livingston | Livingston (La Buga), Izabal, Guatemala.",
        )]
        assert passage_in_docs(docs, "Hijos de Livingston", SRC_ARCHIVO_BILINGUE)


# ─── TestAnyPassageInDocs ─────────────────────────────────────────────────────

class TestAnyPassageInDocs:
    def test_fragment_in_primary_source(self):
        docs = [_doc(
            "/app/.../Baliceaux_Yurumein_AI_Historical_Brief.docx",
            "1796 — the British transferred captives.",
        )]
        assert any_passage_in_docs(docs, "1796", [SRC_BALICEAUX, SRC_CHATOYER])

    def test_fragment_in_secondary_source(self):
        docs = [_doc(
            "/app/.../Yurumein_AI_Chief_Joseph_Chatoyer_EN_ES.docx",
            "In 1796, Baliceaux became a holding site.",
        )]
        assert any_passage_in_docs(docs, "1796", [SRC_BALICEAUX, SRC_CHATOYER])

    def test_fragment_not_in_any_valid_source(self):
        docs = [_doc(
            "/app/.../DICCIONARIO GARIFUNA.pdf",
            "1796 — wrong source for this question.",
        )]
        assert not any_passage_in_docs(docs, "1796", [SRC_BALICEAUX, SRC_CHATOYER])

    def test_empty_sources(self):
        docs = [_doc("/app/.../Baliceaux_Yurumein_AI_Historical_Brief.docx", "1796")]
        assert not any_passage_in_docs(docs, "1796", [])


# ─── TestDocSummary ───────────────────────────────────────────────────────────

class TestDocSummary:
    def _make_doc(self, source, content, page_label=None):
        return {
            "page_content": content,
            "metadata": {"source": source, "page_label": page_label},
            "hybrid_score": 0.75,
            "retrieval_source": "hybrid",
        }

    def test_known_source(self):
        d = self._make_doc("/app/.../DICCIONARIO GARIFUNA.pdf", "word")
        s = doc_summary(d)
        assert s["source_role"] is not None
        assert s["source_display"] != "unclassified"

    def test_unknown_source(self):
        d = self._make_doc("/app/.../Baliceaux_Yurumein_AI_Historical_Brief.docx", "text")
        s = doc_summary(d)
        assert s["source_role"] is None
        assert s["source_display"] == "unclassified"

    def test_content_preview_truncated(self):
        long = "x" * 300
        d = self._make_doc("/app/.../file.pdf", long)
        s = doc_summary(d)
        assert len(s["content_preview"]) <= 140

    def test_missing_metadata(self):
        d = {"page_content": "content", "hybrid_score": 0.5}
        s = doc_summary(d)
        assert s["source_path"] == ""
        assert s["source_role"] is None

    def test_score_rounded(self):
        d = self._make_doc("/app/.../file.pdf", "content")
        d["hybrid_score"] = 0.123456789
        s = doc_summary(d)
        assert s["hybrid_score"] == round(0.123456789, 4)

    def test_newlines_stripped_from_preview(self):
        d = self._make_doc("/app/.../file.pdf", "line1\nline2\nline3")
        s = doc_summary(d)
        assert "\n" not in s["content_preview"]


# ─── TestComputeSummary ───────────────────────────────────────────────────────

class TestComputeSummary:
    def _base_sourced(self, **overrides) -> dict:
        r = {
            "category": "Test",
            "is_edge_case": False,
            "top10_applicable": True,
            "primary_hit_at_10": True,
            "any_hit_at_10": True,
            "primary_hit_in_context": True,
            "any_hit_in_context": True,
            "expected_passage_fragment": None,
            "passage_found_at_10": None,
            "passage_found_in_context": None,
            "error": None,
        }
        r.update(overrides)
        return r

    def _base_edge(self, **overrides) -> dict:
        r = {
            "category": "Edge_Cases",
            "is_edge_case": True,
            "top10_applicable": False,
            "primary_hit_at_10": None,
            "any_hit_at_10": None,
            "primary_hit_in_context": None,
            "any_hit_in_context": None,
            "expected_passage_fragment": None,
            "passage_found_at_10": None,
            "passage_found_in_context": None,
            "error": None,
        }
        r.update(overrides)
        return r

    def test_all_hits(self):
        results = [self._base_sourced() for _ in range(4)]
        s = compute_summary(results)
        assert s["primary_source_hit_rate_at_10"] == 1.0
        assert s["any_source_hit_rate_at_10"] == 1.0
        assert s["primary_source_hit_rate_in_context"] == 1.0
        assert s["any_source_hit_rate_in_context"] == 1.0

    def test_all_misses(self):
        results = [
            self._base_sourced(
                primary_hit_at_10=False,
                any_hit_at_10=False,
                primary_hit_in_context=False,
                any_hit_in_context=False,
            )
            for _ in range(4)
        ]
        s = compute_summary(results)
        assert s["primary_source_hit_rate_at_10"] == 0.0
        assert s["primary_source_hit_rate_in_context"] == 0.0

    def test_mixed(self):
        results = [
            self._base_sourced(
                primary_hit_at_10=True,
                any_hit_at_10=True,
                primary_hit_in_context=True,
                any_hit_in_context=True,
            ),
            self._base_sourced(
                primary_hit_at_10=False,
                any_hit_at_10=False,
                primary_hit_in_context=False,
                any_hit_in_context=False,
            ),
        ]
        s = compute_summary(results)
        assert s["primary_source_hit_rate_at_10"] == 0.5
        assert s["primary_source_hit_rate_in_context"] == 0.5

    def test_edge_cases_excluded_from_sourced_counts(self):
        results = [
            self._base_sourced(),
            self._base_edge(),
        ]
        s = compute_summary(results)
        assert s["total_questions"] == 2
        assert s["sourced_questions"] == 1
        assert s["edge_case_questions"] == 1

    def test_edge_cases_do_not_affect_hit_rate(self):
        results = [
            self._base_sourced(
                primary_hit_at_10=True,
                any_hit_at_10=True,
                primary_hit_in_context=True,
                any_hit_in_context=True,
            ),
            self._base_edge(),
        ]
        s = compute_summary(results)
        assert s["primary_source_hit_rate_at_10"] == 1.0
        assert s["primary_source_hit_rate_in_context"] == 1.0

    def test_passage_recall(self):
        results = [
            self._base_sourced(
                expected_passage_fragment="abc",
                passage_found_at_10=True,
                passage_found_in_context=True,
            ),
            self._base_sourced(
                expected_passage_fragment="xyz",
                passage_found_at_10=False,
                passage_found_in_context=False,
            ),
        ]
        s = compute_summary(results)
        assert s["passage_recall_at_10"] == 0.5
        assert s["passage_recall_in_context"] == 0.5

    def test_no_passage_fragments(self):
        results = [self._base_sourced(expected_passage_fragment=None) for _ in range(3)]
        s = compute_summary(results)
        assert s["passage_recall_at_10"] is None
        assert s["passage_recall_in_context"] is None

    def test_errors_counted(self):
        results = [
            self._base_sourced(error="some error"),
            self._base_sourced(),
        ]
        s = compute_summary(results)
        assert s["errors"] == 1

    def test_non_hybrid_excluded_from_top10(self):
        results = [
            self._base_sourced(
                top10_applicable=True,
                primary_hit_at_10=True,
                any_hit_at_10=True,
                primary_hit_in_context=True,
                any_hit_in_context=True,
            ),
            self._base_sourced(
                top10_applicable=False,
                primary_hit_at_10=None,
                any_hit_at_10=None,
                primary_hit_in_context=True,
                any_hit_in_context=True,
            ),
        ]
        s = compute_summary(results)
        assert s["hybrid_path_questions"] == 1
        assert s["primary_source_hit_rate_at_10"] == 1.0
        assert s["primary_source_hit_rate_in_context"] == 1.0

    def test_any_vs_primary_distinction(self):
        results = [
            self._base_sourced(
                primary_hit_at_10=False,
                any_hit_at_10=True,
                primary_hit_in_context=False,
                any_hit_in_context=True,
            ),
        ]
        s = compute_summary(results)
        assert s["primary_source_hit_rate_at_10"] == 0.0
        assert s["any_source_hit_rate_at_10"] == 1.0
        assert s["primary_source_hit_rate_in_context"] == 0.0
        assert s["any_source_hit_rate_in_context"] == 1.0

    def test_category_breakdown_present(self):
        results = [self._base_sourced(category="Baliceaux") for _ in range(3)]
        s = compute_summary(results)
        assert "Baliceaux" in s["by_category"]
        assert s["by_category"]["Baliceaux"]["total"] == 3

    def test_empty_results(self):
        s = compute_summary([])
        assert s["total_questions"] == 0
        assert s["errors"] == 0


# ─── TestLexicalPathScoring ──────────────────────────────────────────────────

class TestLexicalPathScoring:
    """
    Unit tests for the lexicon_structured_hit scoring path introduced to fix
    C6Q1 (father / Baba, Tata, Úguchilin).

    The lexical path in evaluate_question fires when detect_retrieval_path()
    returns "lexicon" (<LEXICON_CONTEXT> tag present).  It sets:
      lexicon_structured_hit = True
      primary_hit_in_context  = True
      any_hit_in_context      = True
      passage_found_in_context = (fragment in ctx_string) if fragment else not set
    retrieved_docs=[] is intentional; source-based scoring on [] would be a
    false MISS, so lexical hits skip it entirely.
    """

    def _lexical_record(self, **overrides) -> dict:
        """Simulate a record produced by the lexical path in evaluate_question."""
        r = {
            "category": "Language",
            "is_edge_case": False,
            "top10_applicable": False,
            "primary_hit_at_10": None,
            "any_hit_at_10": None,
            "primary_hit_in_context": True,
            "any_hit_in_context": True,
            "expected_passage_fragment": None,
            "passage_found_at_10": None,
            "passage_found_in_context": None,
            "lexicon_structured_hit": True,
            "error": None,
        }
        r.update(overrides)
        return r

    def _hybrid_record(self, **overrides) -> dict:
        r = {
            "category": "Language",
            "is_edge_case": False,
            "top10_applicable": True,
            "primary_hit_at_10": True,
            "any_hit_at_10": True,
            "primary_hit_in_context": True,
            "any_hit_in_context": True,
            "expected_passage_fragment": None,
            "passage_found_at_10": None,
            "passage_found_in_context": None,
            "lexicon_structured_hit": False,
            "error": None,
        }
        r.update(overrides)
        return r

    def test_lexical_path_hits_key_in_summary(self):
        s = compute_summary([])
        assert "lexical_path_hits" in s

    def test_lexical_path_hits_zero_on_empty(self):
        s = compute_summary([])
        assert s["lexical_path_hits"] == 0

    def test_lexical_hit_counted(self):
        s = compute_summary([self._lexical_record()])
        assert s["lexical_path_hits"] == 1

    def test_non_lexical_not_counted(self):
        s = compute_summary([self._hybrid_record()])
        assert s["lexical_path_hits"] == 0

    def test_field_absent_treated_as_false(self):
        # Records written before this fix lack lexicon_structured_hit entirely.
        r = self._hybrid_record()
        del r["lexicon_structured_hit"]
        s = compute_summary([r])
        assert s["lexical_path_hits"] == 0

    def test_multiple_lexical_hits_accumulated(self):
        results = [self._lexical_record(), self._lexical_record(), self._hybrid_record()]
        s = compute_summary(results)
        assert s["lexical_path_hits"] == 2

    def test_lexical_hit_excluded_from_hybrid_path_count(self):
        # top10_applicable=False → NOT in hybrid_path_questions.
        results = [self._lexical_record(), self._hybrid_record()]
        s = compute_summary(results)
        assert s["hybrid_path_questions"] == 1

    def test_lexical_hit_contributes_to_primary_ctx_rate(self):
        # primary_hit_in_context=True on lexical records → counted in rate.
        results = [
            self._lexical_record(),
            self._hybrid_record(primary_hit_in_context=False),
        ]
        s = compute_summary(results)
        assert s["primary_source_hit_rate_in_context"] == 0.5

    def test_lexical_hit_contributes_to_any_ctx_rate(self):
        results = [
            self._lexical_record(),
            self._hybrid_record(any_hit_in_context=False),
        ]
        s = compute_summary(results)
        assert s["any_source_hit_rate_in_context"] == 0.5

    def test_lexical_passage_hit_scored_in_ctx(self):
        # Fragment found in LEXICON_CONTEXT string → passage_found_in_context=True.
        results = [
            self._lexical_record(
                expected_passage_fragment="Baba",
                passage_found_in_context=True,
            )
        ]
        s = compute_summary(results)
        assert s["passage_recall_in_context"] == 1.0

    def test_lexical_passage_miss_scored_in_ctx(self):
        results = [
            self._lexical_record(
                expected_passage_fragment="Baba",
                passage_found_in_context=False,
            )
        ]
        s = compute_summary(results)
        assert s["passage_recall_in_context"] == 0.0

    def test_lexical_record_excluded_from_passage_at_10(self):
        # Lexical path: top10_applicable=False → not in passage_recall_at_10 denominator.
        results = [
            self._lexical_record(
                expected_passage_fragment="Baba",
                passage_found_in_context=True,
            )
        ]
        s = compute_summary(results)
        # No top10-applicable passage questions → rate is None.
        assert s["passage_recall_at_10"] is None


# ─── TestQuestionsIntegrity ───────────────────────────────────────────────────

class TestQuestionsIntegrity:
    EXPECTED_CATEGORIES = {
        "Baliceaux",
        "Chatoyer_Satuye",
        "Yurumein_Identity",
        "Livingston_Community",
        "National_Anthem",
        "Language",
        "Cultural_Traditions",
        "Edge_Cases",
        "Document_Coverage",
    }

    def test_total_question_count(self):
        assert len(QUESTIONS) == 46

    def test_all_categories_present(self):
        cats = {q["category"] for q in QUESTIONS}
        assert cats == self.EXPECTED_CATEGORIES

    def test_question_ids_unique(self):
        ids = [q["id"] for q in QUESTIONS]
        assert len(ids) == len(set(ids)), "Duplicate question IDs found"

    def test_all_ids_match_pattern(self):
        import re
        pattern = re.compile(r"^C\d+Q\d+$")
        for q in QUESTIONS:
            assert pattern.match(q["id"]), f"Bad id format: {q['id']}"

    def test_valid_source_keys_is_list(self):
        for q in QUESTIONS:
            assert isinstance(q["valid_source_keys"], list), (
                f"{q['id']}: valid_source_keys must be a list"
            )

    def test_edge_cases_have_empty_source_keys(self):
        for q in QUESTIONS:
            if q["category"] == "Edge_Cases":
                assert q["valid_source_keys"] == [], (
                    f"{q['id']}: edge case must have valid_source_keys=[]"
                )

    def test_sourced_questions_have_non_empty_source_keys(self):
        for q in QUESTIONS:
            if q["category"] != "Edge_Cases":
                assert q["valid_source_keys"], (
                    f"{q['id']}: sourced question must have at least one valid source key"
                )

    def test_valid_source_keys_are_known_constants(self):
        known = {
            SRC_DICCIONARIO, SRC_SERIE_I, SRC_LIBRO_II, SRC_LEXICON,
            SRC_ANTHEM, SRC_INSTRUMENTS, SRC_OVERVIEW,
            SRC_BALICEAUX, SRC_CHATOYER, SRC_HISTORIA,
            SRC_GARIFUNA_IDENTITY, SRC_ARCHIVO_BILINGUE, SRC_ARCHIVO,
        }
        for q in QUESTIONS:
            for k in q["valid_source_keys"]:
                assert k in known, (
                    f"{q['id']}: unknown source key '{k}'. "
                    f"Add it to the known set or use a defined constant."
                )

    def test_expected_passage_fragment_is_string_or_none(self):
        for q in QUESTIONS:
            frag = q["expected_passage_fragment"]
            assert frag is None or isinstance(frag, str), (
                f"{q['id']}: expected_passage_fragment must be str or None"
            )

    def test_all_questions_have_notes(self):
        for q in QUESTIONS:
            assert q.get("notes"), f"{q['id']}: missing notes"

    def test_language_field_valid(self):
        for q in QUESTIONS:
            assert q["language"] in ("en", "es"), (
                f"{q['id']}: language must be 'en' or 'es'"
            )

    def test_baliceaux_uses_dedicated_source(self):
        for q in QUESTIONS:
            if q["category"] == "Baliceaux":
                assert SRC_BALICEAUX in q["valid_source_keys"], (
                    f"{q['id']}: Baliceaux questions must list SRC_BALICEAUX"
                )

    def test_chatoyer_uses_dedicated_source(self):
        for q in QUESTIONS:
            if q["category"] == "Chatoyer_Satuye":
                assert SRC_CHATOYER in q["valid_source_keys"], (
                    f"{q['id']}: Chatoyer questions must list SRC_CHATOYER"
                )

    def test_livingston_uses_archive_sources(self):
        for q in QUESTIONS:
            if q["category"] == "Livingston_Community":
                has_archive = (
                    SRC_ARCHIVO_BILINGUE in q["valid_source_keys"]
                    or SRC_ARCHIVO in q["valid_source_keys"]
                    or SRC_LIBRO_II in q["valid_source_keys"]
                )
                assert has_archive, (
                    f"{q['id']}: Livingston questions must reference at least one "
                    f"archive or Libro II source"
                )

    def test_anthem_category_uses_anthem_source(self):
        for q in QUESTIONS:
            if q["category"] == "National_Anthem":
                assert SRC_ANTHEM in q["valid_source_keys"], (
                    f"{q['id']}: Anthem questions must include SRC_ANTHEM"
                )

    def test_document_coverage_has_six_questions(self):
        dc = [q for q in QUESTIONS if q["category"] == "Document_Coverage"]
        assert len(dc) == 6

    def test_document_coverage_covers_all_new_docs(self):
        dc = [q for q in QUESTIONS if q["category"] == "Document_Coverage"]
        all_keys = [k for q in dc for k in q["valid_source_keys"]]
        new_docs = {
            SRC_BALICEAUX, SRC_CHATOYER, SRC_HISTORIA,
            SRC_GARIFUNA_IDENTITY, SRC_ARCHIVO_BILINGUE, SRC_ARCHIVO,
        }
        for doc in new_docs:
            assert doc in all_keys, (
                f"No Document_Coverage question covers '{doc}'"
            )

    def test_edge_cases_has_five_questions(self):
        edge = [q for q in QUESTIONS if q["category"] == "Edge_Cases"]
        assert len(edge) == 5

    def test_instruments_constant_uses_underscore(self):
        # Guard against regression to old "MUSICAL INSTRUMENTS" (space) constant.
        assert " " not in SRC_INSTRUMENTS, (
            "SRC_INSTRUMENTS must not contain spaces — the actual file uses underscores"
        )
        assert "Musical_Instruments_Dataset" in SRC_INSTRUMENTS

    def test_historia_diagnostic_note_present(self):
        historia_qs = [
            q for q in QUESTIONS if SRC_HISTORIA in q.get("valid_source_keys", [])
        ]
        assert historia_qs, "No question found that lists SRC_HISTORIA"
        any_note = any(
            "pending_cultural_review" in q["notes"].lower()
            or "publication_allowed" in q["notes"].lower()
            for q in historia_qs
        )
        assert any_note, (
            "At least one question listing SRC_HISTORIA must note its "
            "staging/review status (publication_allowed: false / pending_cultural_review)"
        )


# ─── TestRestrictionScoring ───────────────────────────────────────────────────

class TestRestrictionScoring:
    """
    Unit tests for restriction_expected / restriction_verified scoring.

    C9Q3 (Historia) is a restriction-check question, not a regular sourced
    question. Its success criterion is that the restricted source is ABSENT
    from the public context, not present.
    """

    def _restriction_record(self, verified: bool) -> dict:
        return {
            "category": "Document_Coverage",
            "is_edge_case": False,
            "restriction_expected": True,
            "restriction_verified": verified,
            "top10_applicable": False,
            "primary_hit_at_10": None,
            "any_hit_at_10": None,
            "primary_hit_in_context": None,
            "any_hit_in_context": None,
            "expected_passage_fragment": None,
            "passage_found_at_10": None,
            "passage_found_in_context": None,
            "lexicon_structured_hit": False,
            "error": None,
        }

    def _sourced_record(self, **overrides) -> dict:
        r = {
            "category": "Test",
            "is_edge_case": False,
            "restriction_expected": False,
            "restriction_verified": None,
            "top10_applicable": True,
            "primary_hit_at_10": True,
            "any_hit_at_10": True,
            "primary_hit_in_context": True,
            "any_hit_in_context": True,
            "expected_passage_fragment": None,
            "passage_found_at_10": None,
            "passage_found_in_context": None,
            "lexicon_structured_hit": False,
            "error": None,
        }
        r.update(overrides)
        return r

    # ── Question metadata ────────────────────────────────────────────────────

    def test_c9q3_has_restriction_expected(self):
        c9q3 = next((q for q in QUESTIONS if q["id"] == "C9Q3"), None)
        assert c9q3 is not None, "C9Q3 not found in QUESTIONS"
        assert c9q3.get("restriction_expected") is True, (
            "C9Q3 must have restriction_expected=True — Historia is a restricted document"
        )

    def test_c9q3_restriction_note_explains_outcome(self):
        c9q3 = next(q for q in QUESTIONS if q["id"] == "C9Q3")
        note = c9q3["notes"].lower()
        assert "restricted_ok" in note or "restriction_leak" in note, (
            "C9Q3 notes must explain RESTRICTED_OK / RESTRICTION_LEAK outcomes"
        )

    # ── compute_summary — restriction counts ─────────────────────────────────

    def test_restricted_ok_counted(self):
        results = [self._restriction_record(verified=True)]
        s = compute_summary(results)
        assert s["restriction_questions"] == 1
        assert s["restriction_verified_count"] == 1
        assert s["restriction_leak_count"] == 0

    def test_restriction_leak_counted(self):
        results = [self._restriction_record(verified=False)]
        s = compute_summary(results)
        assert s["restriction_questions"] == 1
        assert s["restriction_verified_count"] == 0
        assert s["restriction_leak_count"] == 1

    def test_mixed_restriction_results(self):
        results = [
            self._restriction_record(verified=True),
            self._restriction_record(verified=False),
            self._restriction_record(verified=True),
        ]
        s = compute_summary(results)
        assert s["restriction_questions"] == 3
        assert s["restriction_verified_count"] == 2
        assert s["restriction_leak_count"] == 1

    def test_no_restriction_questions_zero_counts(self):
        results = [self._sourced_record()]
        s = compute_summary(results)
        assert s["restriction_questions"] == 0
        assert s["restriction_verified_count"] == 0
        assert s["restriction_leak_count"] == 0

    # ── Exclusion from regular sourced metrics ───────────────────────────────

    def test_restriction_question_excluded_from_sourced_count(self):
        # A restriction_expected record must NOT inflate the sourced hit-rate
        # denominator.  Only the regular sourced record counts.
        results = [
            self._sourced_record(),            # 1 sourced hit
            self._restriction_record(True),    # restriction-only — excluded
        ]
        s = compute_summary(results)
        assert s["sourced_questions"] == 1
        assert s["primary_source_hit_rate_in_context"] == 1.0

    def test_restriction_question_does_not_dilute_hit_rate(self):
        # With two sourced hits and one restriction question (not a hit or miss),
        # the hit rate must still be 2/2 = 1.0, not 2/3.
        results = [
            self._sourced_record(),
            self._sourced_record(),
            self._restriction_record(True),
        ]
        s = compute_summary(results)
        assert s["primary_source_hit_rate_in_context"] == 1.0

    def test_restriction_verified_none_not_counted_as_leak(self):
        # restriction_verified=None (run error) must NOT be counted as a leak.
        r = self._restriction_record(verified=True)
        r["restriction_verified"] = None
        s = compute_summary([r])
        assert s["restriction_leak_count"] == 0
        assert s["restriction_verified_count"] == 0
