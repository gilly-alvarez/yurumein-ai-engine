#!/usr/bin/env python3
"""
Stage A: Production Retrieval Benchmark — Yurumein AI
======================================================

Measures whether the production hybrid retrieval pipeline consistently
surfaces the right evidence for 40 culturally-grounded questions across
8 categories.

Does NOT call the LLM. Does NOT modify any data.
Requires a live connection to the production PGVector database.

Usage (run from repo root on the production VPS):
    python scripts/evaluate_production.py

Output:
    stdout            — human-readable per-question summary and totals
    stage_a_results.json — machine-readable full results

What this measures
------------------
  hit_at_10          : expected source appeared in the raw top-10 hybrid pool
                       (step 4 of global_retrieval_process, before top-up / boost)
  hit_in_context     : expected source appeared in the final top-5 context
                       (after top-up, low-info filter, role boost, expansion injection)
  passage_found_*    : specific text fragment found in the expected source chunk
                       (only for questions where a ground-truth fragment is recorded)
  top10_applicable   : False for anthem/lexicon/page paths (hybrid pool not used)

What this does NOT measure
--------------------------
  - LLM answer quality, factual accuracy, or citation reliability (Stage B)
  - Sources not in the production PGVector collection
  - Lexicon path hits (retrieved_docs=[] for structured-lexicon responses)

Safety
------
  - All database operations are SELECT-only on existing tables.
  - Embedding API calls are made (OpenAI text-embedding-3-small, ~40 questions × 2).
  - PGVector init may issue idempotent CREATE TABLE IF NOT EXISTS statements;
    this is safe on production where tables already exist.
  - Script aborts if the target collection has fewer than 1,000 chunks
    (protects against running on the empty local dev database).
  - No vectors are added, deleted, or modified.
  - No application code is restarted or redeployed.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

# Add repo root so app.* imports resolve regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text

from app.core.config import settings
from app.services.chatbot.source_classifier import classify_source
from app.services.utils.RAG_pipeline.workflow.nodes import Nodes

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

MIN_COLLECTION_CHUNKS = 1_000
REPORT_FILE = "stage_a_results.json"

# Source-key substrings for expected-source matching.
# These are the uppercase fragments checked against the source path in chunk metadata.
# They match exactly the keys used in source_classifier._ROLE_MAP.
SRC_SERIE_I     = "LIBRO SERIE I"
SRC_LIBRO_II    = "LIBRO GARIFUNA II FORMATO DIGITAL"
SRC_DICCIONARIO = "DICCIONARIO GARIFUNA"
SRC_LEXICON     = "YURUMEIN LEXICON"
SRC_ANTHEM      = "HIMNO_NACIONAL_GUATEMALA_GARIFUNA"
SRC_INSTRUMENTS = "MUSICAL INSTRUMENTS"
SRC_OVERVIEW    = "GARIFUNA LANGUAGE OVERVIEW"


# ─── 40-QUESTION EVALUATION SET ───────────────────────────────────────────────
#
# expected_source_key  : substring that must appear (case-insensitive) in the
#                        source path of a retrieved chunk to count as a hit.
#                        None for edge cases where no source hit is expected.
#
# expected_passage_fragment : short substring (case-insensitive) that must
#                        appear in the page_content of a hit chunk from the
#                        expected source.  None when passage not independently
#                        verified — only source-level hit is scored.
#
# notes                : human-readable rationale and provenance for expected values.

QUESTIONS = [
    # ── Category 1: Baliceaux ─────────────────────────────────────────────────
    {
        "id": "C1Q1", "category": "Baliceaux", "language": "en",
        "query": "What was Baliceaux and what happened to the Garifuna there?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Baliceaux: internment island before 1797 deportation. Passage not independently verified.",
    },
    {
        "id": "C1Q2", "category": "Baliceaux", "language": "es",
        "query": "¿Qué fue Baliceaux y cuál fue su significado para el pueblo garífuna?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Spanish query; same expected source.",
    },
    {
        "id": "C1Q3", "category": "Baliceaux", "language": "en",
        "query": "Describe the conditions on Baliceaux before the 1797 Garifuna exile.",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": "1797",
        "notes": "1797 is a documented date in Libro Garífuna II; passage fragment is minimal.",
    },
    {
        "id": "C1Q4", "category": "Baliceaux", "language": "es",
        "query": "¿Cuántos garífunas murieron en Baliceaux y por qué?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Death toll not independently verified; source-level hit only.",
    },
    {
        "id": "C1Q5", "category": "Baliceaux", "language": "en",
        "query": "How did the internment on Baliceaux lead to the Garifuna deportation to Central America?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Deportation chain from Baliceaux to Honduras; source-level only.",
    },

    # ── Category 2: Chief Joseph Chatoyer / Satuye ────────────────────────────
    {
        "id": "C2Q1", "category": "Chatoyer_Satuye", "language": "en",
        "query": "Who was Satuye and what role did he play in Garifuna resistance against the British?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": "Satuye",
        "notes": (
            "Libro Garífuna II p.8 names Satuye (Chatoyer) as the last leader who "
            "fought the British before the 1797 deportation."
        ),
    },
    {
        "id": "C2Q2", "category": "Chatoyer_Satuye", "language": "es",
        "query": "¿Quién fue Joseph Chatoyer y por qué es un símbolo de la identidad garífuna?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Spanish query about Chatoyer's symbolic significance.",
    },
    {
        "id": "C2Q3", "category": "Chatoyer_Satuye", "language": "en",
        "query": "Describe the Black Carib resistance against British colonization of St. Vincent.",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Black Carib Wars; St. Vincent / Yurumein colonial resistance context.",
    },
    {
        "id": "C2Q4", "category": "Chatoyer_Satuye", "language": "es",
        "query": "¿Qué fue la guerra de los caribes negros y cuándo ocurrió?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Spanish query; Carib Wars 1795. Source-level only.",
    },
    {
        "id": "C2Q5", "category": "Chatoyer_Satuye", "language": "en",
        "query": "What happened to Satuye at the end of the Carib Wars of 1795?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Satuye's fate; passage not independently verified.",
    },

    # ── Category 3: Yurumein and Garifuna identity ────────────────────────────
    {
        "id": "C3Q1", "category": "Yurumein_Identity", "language": "en",
        "query": "What does the name Yurumein mean and why is it significant to the Garifuna?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": "Yurumein",
        "notes": "Yurumein = Garifuna name for St. Vincent; documented in Libro Garífuna II p.6.",
    },
    {
        "id": "C3Q2", "category": "Yurumein_Identity", "language": "es",
        "query": "¿Cuáles son los orígenes étnicos del pueblo garífuna?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "African + Arawak + Carib origins; Libro Garífuna II.",
    },
    {
        "id": "C3Q3", "category": "Yurumein_Identity", "language": "en",
        "query": "Explain the African, Arawak, and Carib heritage of the Garifuna people.",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": "afric",
        "notes": (
            "Libro Garífuna II p.5 documents African-Arawak-Carib heritage. "
            "Fragment 'afric' is a minimal case-insensitive anchor."
        ),
    },
    {
        "id": "C3Q4", "category": "Yurumein_Identity", "language": "es",
        "query": "¿Qué fue la deportación de 1797 y cómo definió la identidad garífuna moderna?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": "1797",
        "notes": "1797 deportation as identity anchor; documented in Libro Garífuna II.",
    },
    {
        "id": "C3Q5", "category": "Yurumein_Identity", "language": "en",
        "query": "Where are Garifuna communities found today, and how did they get there after 1797?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Diaspora: Honduras, Belize, Guatemala, Nicaragua; Libro Garífuna II.",
    },

    # ── Category 4: Livingston and community ──────────────────────────────────
    {
        "id": "C4Q1", "category": "Livingston_Community", "language": "en",
        "query": "What is the significance of Livingston, Guatemala in Garifuna history and culture?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": (
            "Lívingston is the home community of ABÜRÜHATI (Elmer Mauricio Enríquez Bermúdez). "
            "Coverage depends on whether the source names it explicitly."
        ),
    },
    {
        "id": "C4Q2", "category": "Livingston_Community", "language": "es",
        "query": "¿Qué comunidades garífunas se encuentran en la costa atlántica de Guatemala?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Guatemala Garifuna communities; Libro Garífuna II.",
    },
    {
        "id": "C4Q3", "category": "Livingston_Community", "language": "en",
        "query": "Describe the Garifuna presence along the Caribbean coast of Guatemala.",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Caribbean coast of Guatemala; Libro Garífuna II.",
    },
    {
        "id": "C4Q4", "category": "Livingston_Community", "language": "es",
        "query": "¿Quién es Elmer Mauricio Enríquez Bermúdez y cuál es su legado cultural?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": "Enríquez",
        "notes": (
            "ABÜRÜHATI attribution appears in Libro Garífuna II. "
            "Fragment matches the author surname in the retrieved chunk."
        ),
    },
    {
        "id": "C4Q5", "category": "Livingston_Community", "language": "en",
        "query": "What educational materials about Garifuna language and culture were produced by the community in Guatemala?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Broad educational resources query; Libro Garífuna II or Libro Serie I.",
    },

    # ── Category 5: Guatemala national anthem ─────────────────────────────────
    # All five queries are designed to trigger Intent.ANTHEM → anthem_ordered path.
    {
        "id": "C5Q1", "category": "National_Anthem", "language": "en",
        "query": "Show me the Garifuna version of the Guatemala national anthem.",
        "expected_source_key": SRC_ANTHEM,
        "expected_passage_fragment": None,
        "notes": "Triggers anthem_ordered path. retrieval_path must be anthem_ordered.",
    },
    {
        "id": "C5Q2", "category": "National_Anthem", "language": "es",
        "query": "¿Cuál es la letra del himno nacional de Guatemala en versión garífuna?",
        "expected_source_key": SRC_ANTHEM,
        "expected_passage_fragment": None,
        "notes": "Spanish anthem query; triggers anthem_ordered path.",
    },
    {
        "id": "C5Q3", "category": "National_Anthem", "language": "en",
        "query": "What document in this knowledge base contains the Garifuna national anthem of Guatemala?",
        "expected_source_key": SRC_ANTHEM,
        "expected_passage_fragment": None,
        "notes": "Document identification query; anthem intent should fire.",
    },
    {
        "id": "C5Q4", "category": "National_Anthem", "language": "en",
        "query": "Recite the full Garifuna national anthem of Guatemala.",
        "expected_source_key": SRC_ANTHEM,
        "expected_passage_fragment": None,
        "notes": "Full recitation request; triggers anthem_ordered path.",
    },
    {
        "id": "C5Q5", "category": "National_Anthem", "language": "es",
        "query": "Dame la letra completa del himno nacional de Guatemala en garífuna.",
        "expected_source_key": SRC_ANTHEM,
        "expected_passage_fragment": None,
        "notes": "Full lyrics in Spanish; triggers anthem_ordered path.",
    },

    # ── Category 6: Garifuna language ─────────────────────────────────────────
    {
        "id": "C6Q1", "category": "Language", "language": "en",
        "query": "How do you say 'father' in Garifuna?",
        "expected_source_key": SRC_DICCIONARIO,
        "expected_passage_fragment": "Baba",
        "notes": (
            "Lexical parser fires ('how do you say X in Garifuna'). "
            "Lexicon covers A–An only → lexicon miss → hybrid RAG with LEXICON_NOTE. "
            "Diccionario p.144: Baba, Tata, Úguchilin."
        ),
    },
    {
        "id": "C6Q2", "category": "Language", "language": "es",
        "query": "¿Cuántas vocales tiene el idioma garífuna y cuáles son?",
        "expected_source_key": SRC_SERIE_I,
        "expected_passage_fragment": None,
        "notes": (
            "PRONUNCIATION intent → preferred FOUNDATIONAL_LANGUAGE_EDUCATION. "
            "Libro Serie I p.4–5: six vowels including ü."
        ),
    },
    {
        "id": "C6Q3", "category": "Language", "language": "en",
        "query": "What linguistic families is the Garifuna language descended from?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Arawak-Carib-African roots; Libro Garífuna II. CULTURAL_HISTORICAL intent.",
    },
    {
        "id": "C6Q4", "category": "Language", "language": "es",
        "query": "¿Cómo se pronuncia la letra ü en el idioma garífuna?",
        "expected_source_key": SRC_SERIE_I,
        "expected_passage_fragment": None,
        "notes": "ü phonology; Libro Serie I covers pronunciation. PRONUNCIATION intent.",
    },
    {
        "id": "C6Q5", "category": "Language", "language": "en",
        "query": "What are the numbers one through three in Garifuna?",
        "expected_source_key": SRC_SERIE_I,
        "expected_passage_fragment": "Aban",
        "notes": (
            "NUMBERING intent → contextual expansion 'Aban biama urüwa'. "
            "Libro Serie I p.8: Aban=1, Biáma=2, Ürüwa=3."
        ),
    },

    # ── Category 7: Cultural traditions ───────────────────────────────────────
    {
        "id": "C7Q1", "category": "Cultural_Traditions", "language": "en",
        "query": "What is the Dügü ceremony and what is its cultural significance?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": (
            "CULTURAL_HISTORICAL intent. Contextual expansion 'Gayunari Dabuyaba' fires. "
            "Libro Garífuna II covers Dügü; Diccionario p.43 has ceremony-house mapping."
        ),
    },
    {
        "id": "C7Q2", "category": "Cultural_Traditions", "language": "es",
        "query": "¿Qué es la punta y cuál es su importancia en la cultura garífuna?",
        "expected_source_key": SRC_LIBRO_II,
        "expected_passage_fragment": None,
        "notes": "Punta dance/music; Libro Garífuna II. CULTURAL_HISTORICAL intent.",
    },
    {
        "id": "C7Q3", "category": "Cultural_Traditions", "language": "en",
        "query": "Describe the musical instruments used in traditional Garifuna ceremonies.",
        "expected_source_key": SRC_INSTRUMENTS,
        "expected_passage_fragment": None,
        "notes": "Musical Instruments dataset (3 chunks). Passage not verified at chunk level.",
    },
    {
        "id": "C7Q4", "category": "Cultural_Traditions", "language": "es",
        "query": "¿Qué es el Dabuyaba y en qué ceremonias se utiliza?",
        "expected_source_key": SRC_DICCIONARIO,
        "expected_passage_fragment": "Dabuyaba",
        "notes": (
            "Diccionario p.43: 'Dabuyaba' = house used in Chugú ceremony. "
            "Contextual expansion 'Gayunari Dabuyaba' should surface this chunk."
        ),
    },
    {
        "id": "C7Q5", "category": "Cultural_Traditions", "language": "en",
        "query": "What role does the garaon drum play in Garifuna music and ceremony?",
        "expected_source_key": SRC_INSTRUMENTS,
        "expected_passage_fragment": None,
        "notes": "Garaon drum; Musical Instruments dataset. Source-level only.",
    },

    # ── Category 8: Edge cases ─────────────────────────────────────────────────
    # expected_source_key=None means no specific source hit is expected.
    # These questions test behavior at the edges of coverage.
    {
        "id": "C8Q1", "category": "Edge_Cases", "language": "en",
        "query": "What does Libro Serie I say on page 500?",
        "expected_source_key": None,
        "expected_passage_fragment": None,
        "notes": (
            "parse_page_request fires: source=%LIBRO SERIE I%, pages=(500,). "
            "Page 500 does not exist → page-not-found context. No retrieved_docs."
        ),
    },
    {
        "id": "C8Q2", "category": "Edge_Cases", "language": "en",
        "query": "What is the Garifuna word for automobile?",
        "expected_source_key": None,
        "expected_passage_fragment": None,
        "notes": (
            "Lexical parser fires ('what is the Garifuna word for X'). "
            "'Automobile' not in lexicon (A–An) or Diccionario. "
            "Expect lexicon miss + generic hybrid retrieval with low relevance."
        ),
    },
    {
        "id": "C8Q3", "category": "Edge_Cases", "language": "es",
        "query": "¿Qué dice el Libro Garífuna II en la página 999?",
        "expected_source_key": None,
        "expected_passage_fragment": None,
        "notes": (
            "parse_page_request fires: source=%GARIFUNA II%, pages=(999,). "
            "Page 999 does not exist → page-not-found context."
        ),
    },
    {
        "id": "C8Q4", "category": "Edge_Cases", "language": "en",
        "query": "Was the Garifuna language influenced by ancient Sumerian or Egyptian writing systems?",
        "expected_source_key": None,
        "expected_passage_fragment": None,
        "notes": (
            "Unsupported claim. Sources document Arawak/Carib/European contact only. "
            "Any retrieved docs are low-relevance noise."
        ),
    },
    {
        "id": "C8Q5", "category": "Edge_Cases", "language": "en",
        "query": "Tell me about the Garifuna diaspora community in Paris, France.",
        "expected_source_key": None,
        "expected_passage_fragment": None,
        "notes": (
            "Paris not documented in indexed sources. CULTURAL_HISTORICAL intent fires. "
            "Retrieval may surface general diaspora chunks (Honduras/Belize) — record what appears."
        ),
    },
]

assert len(QUESTIONS) == 40, f"Expected 40 questions, got {len(QUESTIONS)}"


# ─── PRE-FLIGHT ───────────────────────────────────────────────────────────────

def preflight_check() -> dict:
    """
    Verify database connectivity and collection state before evaluation.

    Returns an info dict for inclusion in the JSON report.
    Raises RuntimeError on any blocking failure.
    """
    collection_name = (
        getattr(settings, "COLLECTION_NAME", None) or "document_collection"
    )
    conn_str = (
        f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASS}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    )
    engine = create_engine(conn_str)

    info: dict = {
        "db_host": settings.DB_HOST,
        "db_name": settings.DB_NAME,
        "collection_name": collection_name,
        "chunk_count": 0,
        "collections_found": [],
        "openai_key_set": bool(settings.OPENAI_API_KEY),
    }

    if not info["openai_key_set"]:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — embedding calls required by the "
            "semantic search will fail."
        )

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT name FROM langchain_pg_collection ORDER BY name")
            ).mappings().all()
            info["collections_found"] = [r["name"] for r in rows]

            count_row = conn.execute(
                text(
                    """
                    SELECT COUNT(*) AS n
                    FROM langchain_pg_embedding e
                    JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                    WHERE c.name = :col
                    """
                ),
                {"col": collection_name},
            ).mappings().one_or_none()

            info["chunk_count"] = int(count_row["n"]) if count_row else 0
    except Exception as exc:
        raise RuntimeError(f"Database pre-flight failed: {exc}") from exc

    if collection_name not in info["collections_found"]:
        raise RuntimeError(
            f"Collection '{collection_name}' was not found in the database. "
            f"Collections present: {info['collections_found']}. "
            f"Verify COLLECTION_NAME in .env."
        )

    if info["chunk_count"] < MIN_COLLECTION_CHUNKS:
        raise RuntimeError(
            f"Collection '{collection_name}' contains only {info['chunk_count']} chunks "
            f"(minimum {MIN_COLLECTION_CHUNKS} required). "
            f"This looks like the empty local development database — it has 0 indexed documents. "
            f"Run the benchmark on the production VPS only."
        )

    return info


# ─── PURE HELPER FUNCTIONS ────────────────────────────────────────────────────
# These functions are pure (no DB, no I/O) and are covered by unit tests.

def source_key_in_docs(docs: list[dict], source_key: str) -> bool:
    """Return True if any doc's source path contains source_key (case-insensitive)."""
    key_upper = source_key.upper()
    return any(
        key_upper in (doc.get("metadata") or {}).get("source", "").upper()
        for doc in docs
    )


def passage_in_docs(docs: list[dict], fragment: str, source_key: str) -> bool:
    """
    Return True if the fragment appears (case-insensitive) in a doc that also
    matches the expected source key.  Both conditions must hold in the same doc.
    """
    key_upper = source_key.upper()
    frag_lower = fragment.lower()
    for doc in docs:
        source_path = (doc.get("metadata") or {}).get("source", "").upper()
        if key_upper not in source_path:
            continue
        if frag_lower in (doc.get("page_content") or "").lower():
            return True
    return False


def detect_retrieval_path(context: str) -> str:
    """
    Infer which retrieval path was taken from the returned context string.

    Returns one of:
      "lexicon"            — structured lexicon hit (retrieved_docs=[])
      "lexicon_miss_hybrid"— lexical parser fired but lexicon returned nothing;
                             fell through to hybrid RAG with LEXICON_NOTE injected
      "anthem_ordered"     — deterministic source-ordered anthem retrieval
      "page_lookup"        — explicit page request (found or not-found)
      "hybrid"             — general hybrid RAG path (most queries)
    """
    if "<LEXICON_CONTEXT>" in context:
        return "lexicon"
    if "<LEXICON_NOTE>" in context:
        return "lexicon_miss_hybrid"
    if "<ANTHEM_EXACT" in context:
        return "anthem_ordered"
    if "page_lookup" in context:
        return "page_lookup"
    return "hybrid"


def doc_summary(doc: dict) -> dict:
    """Compact, JSON-serialisable representation of a retrieved document."""
    meta = doc.get("metadata") or {}
    source_path = meta.get("source", "")
    sm = classify_source(source_path)
    return {
        "source_path": source_path,
        "source_role": sm.role if sm else None,
        "source_display": sm.display if sm else "unknown",
        "page_label": meta.get("page_label"),
        "hybrid_score": round(float(doc.get("hybrid_score") or 0.0), 4),
        "retrieval_source": doc.get("retrieval_source"),
        "content_preview": (doc.get("page_content") or "")[:140].replace("\n", " "),
    }


def compute_summary(results: list[dict]) -> dict:
    """
    Compute aggregate retrieval metrics from per-question results.

    Metrics explained:
      source_hit_rate_at_10    — fraction of sourced questions where the
                                  expected source appeared in the raw top-10
                                  hybrid pool (excludes anthem/lexicon/page paths
                                  where the hybrid pool is not used)
      source_hit_rate_in_context — fraction of sourced questions where the
                                   expected source appeared in the final top-5
      passage_recall_at_10     — fraction of passage-annotated questions where
                                  the expected fragment appeared in the top-10
      passage_recall_in_context — same but for the final top-5
    """
    total = len(results)
    errors = sum(1 for r in results if r["error"])

    sourced = [r for r in results if r["expected_source_key"] is not None]
    sourced_top10 = [r for r in sourced if r["top10_applicable"]]

    hits10 = sum(1 for r in sourced_top10 if r.get("hit_at_10") is True)
    hits_ctx = sum(1 for r in sourced if r.get("hit_in_context") is True)

    passage_q_top10 = [r for r in sourced_top10 if r["expected_passage_fragment"]]
    passage_q_ctx = [r for r in sourced if r["expected_passage_fragment"]]
    phits10 = sum(1 for r in passage_q_top10 if r.get("passage_found_at_10") is True)
    phits_ctx = sum(1 for r in passage_q_ctx if r.get("passage_found_in_context") is True)

    categories: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {
                "total": 0, "hit_at_10": 0, "hit_in_context": 0, "errors": 0
            }
        categories[cat]["total"] += 1
        if r["error"]:
            categories[cat]["errors"] += 1
        if r.get("hit_at_10") is True:
            categories[cat]["hit_at_10"] += 1
        if r.get("hit_in_context") is True:
            categories[cat]["hit_in_context"] += 1

    return {
        "total_questions": total,
        "errors": errors,
        "sourced_questions": len(sourced),
        "edge_case_questions": total - len(sourced),
        "hybrid_path_questions": len(sourced_top10),
        "source_hit_rate_at_10": (
            round(hits10 / len(sourced_top10), 3) if sourced_top10 else None
        ),
        "source_hit_rate_in_context": (
            round(hits_ctx / len(sourced), 3) if sourced else None
        ),
        "passage_recall_at_10": (
            round(phits10 / len(passage_q_top10), 3) if passage_q_top10 else None
        ),
        "passage_recall_in_context": (
            round(phits_ctx / len(passage_q_ctx), 3) if passage_q_ctx else None
        ),
        "by_category": categories,
    }


# ─── EVALUATOR ────────────────────────────────────────────────────────────────

async def evaluate_question(nodes: Nodes, q: dict) -> dict:
    """
    Run one question through the full production retrieval pipeline and score it.

    Steps:
      1. Call global_retrieval_process() to get the production final context (top-5).
      2. For hybrid-path questions, also call _hybrid_retrieve(top_k=10) to measure
         the raw pool before top-up / filter / boost.  This mirrors exactly the
         step-4 pool in global_retrieval_process.
      3. Score expected source and passage fragment against both pools.
    """
    record: dict = {
        "id": q["id"],
        "category": q["category"],
        "query": q["query"],
        "language": q["language"],
        "expected_source_key": q["expected_source_key"],
        "expected_passage_fragment": q["expected_passage_fragment"],
        "notes": q["notes"],
        "retrieval_path": None,
        "top10_applicable": True,
        "top10_pool": [],
        "top5_context": [],
        "hit_at_10": None,
        "hit_in_context": None,
        "passage_found_at_10": None,
        "passage_found_in_context": None,
        "error": None,
        "elapsed_s": None,
    }

    t0 = time.monotonic()
    try:
        state: dict = {"user_query": q["query"], "messages": []}
        prod_result = await nodes.global_retrieval_process(state)

        ctx_string = prod_result.get("context", "")
        retrieval_path = detect_retrieval_path(ctx_string)
        record["retrieval_path"] = retrieval_path

        final_docs: list[dict] = prod_result.get("retrieved_docs") or []
        record["top5_context"] = [doc_summary(d) for d in final_docs]

        # Raw top-10: only meaningful for hybrid-family paths.
        top10_applicable = retrieval_path in ("hybrid", "lexicon_miss_hybrid")
        record["top10_applicable"] = top10_applicable

        raw_top10: list[dict] = []
        if top10_applicable:
            retrieval_query = nodes._contextualize_query(q["query"], [])
            raw_top10 = nodes._hybrid_retrieve(query=retrieval_query, top_k=10)
            record["top10_pool"] = [doc_summary(d) for d in raw_top10]
        elif retrieval_path == "anthem_ordered":
            # For anthem, final_docs is already the full ordered set (no top-10 concept).
            raw_top10 = final_docs
            record["top10_pool"] = record["top5_context"]

        # ── Scoring ──────────────────────────────────────────────────────────
        expected_key = q["expected_source_key"]
        expected_frag = q["expected_passage_fragment"]

        if expected_key is None:
            # Edge case: no source expected. Leave all scores as None.
            pass
        else:
            record["hit_at_10"] = source_key_in_docs(raw_top10, expected_key)
            record["hit_in_context"] = source_key_in_docs(final_docs, expected_key)

            if expected_frag:
                record["passage_found_at_10"] = passage_in_docs(
                    raw_top10, expected_frag, expected_key
                )
                record["passage_found_in_context"] = passage_in_docs(
                    final_docs, expected_frag, expected_key
                )

    except Exception as exc:
        record["error"] = str(exc)

    record["elapsed_s"] = round(time.monotonic() - t0, 2)
    return record


# ─── MAIN ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 72)
    print("  Yurumein AI — Stage A: Production Retrieval Benchmark")
    print("=" * 72)

    # 1. Pre-flight.
    print("\n[1/4] Running pre-flight checks ...")
    try:
        pf = preflight_check()
    except RuntimeError as exc:
        print(f"\n  ABORTED — pre-flight failed:\n  {exc}")
        sys.exit(1)

    print(f"  DB host      : {pf['db_host']}")
    print(f"  DB name      : {pf['db_name']}")
    print(f"  Collection   : {pf['collection_name']}")
    print(f"  Chunks       : {pf['chunk_count']:,}")
    print(f"  Collections  : {', '.join(pf['collections_found'])}")
    print(f"  OpenAI key   : SET")

    # 2. Initialise retrieval engine (triggers lazy VectorStoreManager init).
    print("\n[2/4] Initialising retrieval engine ...")
    nodes = Nodes()
    _ = nodes.vector_store  # eager init so the first question is not slower
    print("  Engine ready.")

    # 3. Evaluate questions.
    print(f"\n[3/4] Evaluating {len(QUESTIONS)} questions ...")
    results: list[dict] = []

    for i, q in enumerate(QUESTIONS, 1):
        prefix = f"  [{i:02d}/{len(QUESTIONS)}] {q['id']:<6} ({q['category']})"
        print(f"{prefix} ...", end="", flush=True)
        r = await evaluate_question(nodes, q)

        if r["error"]:
            status = "ERROR"
        elif r["expected_source_key"] is None:
            status = "EDGE"
        elif r.get("hit_in_context") is True:
            status = "HIT"
        else:
            status = "MISS"

        print(f" {status:<4} [{r['elapsed_s']}s]")
        if r["error"]:
            print(f"    ! {r['error']}")

        results.append(r)

    # 4. Summary.
    print("\n[4/4] Computing summary ...")
    summary = compute_summary(results)

    print("\n" + "─" * 72)
    print(f"  Questions evaluated          : {summary['total_questions']}")
    print(f"  Retrieval errors             : {summary['errors']}")
    print(f"  Sourced questions            : {summary['sourced_questions']}")
    print(f"  Edge-case questions          : {summary['edge_case_questions']}")
    print(f"  Hybrid-path questions        : {summary['hybrid_path_questions']}")
    print()
    print(f"  Source hit-rate@10 (hybrid)  : {summary['source_hit_rate_at_10']}")
    print(f"  Source hit-rate in context   : {summary['source_hit_rate_in_context']}")
    print(f"  Passage recall@10            : {summary['passage_recall_at_10']}")
    print(f"  Passage recall in context    : {summary['passage_recall_in_context']}")
    print("─" * 72)
    print("  Category breakdown:")
    for cat, m in summary["by_category"].items():
        n = m["total"]
        print(
            f"    {cat:<30} "
            f"hit@10={m['hit_at_10']}/{n}  "
            f"ctx={m['hit_in_context']}/{n}  "
            f"err={m['errors']}"
        )

    # Write JSON report.
    report = {
        "benchmark": "stage_a_retrieval",
        "version": "1.0",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "preflight": pf,
        "summary": summary,
        "results": results,
    }
    with open(REPORT_FILE, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\n  Full results written to: {REPORT_FILE}")
    print("=" * 72)

    if summary["errors"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
