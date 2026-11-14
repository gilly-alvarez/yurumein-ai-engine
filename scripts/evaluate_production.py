#!/usr/bin/env python3
"""
Stage A: Production Retrieval Benchmark — Yurumein AI
======================================================

Measures whether the production hybrid retrieval pipeline consistently
surfaces the right evidence for 46 culturally-grounded questions across
9 categories.

Does NOT call the LLM. Does NOT modify any data.
Requires a live connection to the production PGVector database.

Usage (run from repo root on the production VPS):
    python scripts/evaluate_production.py

No command-line arguments are accepted. Output is written to:
    stage_a_results.json  — machine-readable per-question results

What this measures
------------------
  primary_hit_at_10      : primary (first) valid source in raw top-10 hybrid pool
  any_hit_at_10          : ANY listed valid source in raw top-10 hybrid pool
  primary_hit_in_context : primary source in the final top-5 context passed to LLM
  any_hit_in_context     : ANY listed valid source in the final top-5 context
  passage_found_*        : expected text fragment found inside a valid-source chunk
                           (only for questions where a ground-truth fragment is recorded)
  top10_applicable       : False for anthem / lexicon / page paths

Edge cases (Category 8)
-----------------------
  valid_source_keys = [] — no source hit is expected.
  These are DIAGNOSTIC tests that record what the system returns for queries
  about missing pages, unsupported claims, or unknown terms.
  They are NOT scored as pass/fail. The JSON marks them: "is_edge_case": true.
  A missing source for an edge case is not a retrieval failure.

Production database (as verified 2026-11-13)
--------------------------------------------
  document_collection — 13 sources, ~6,160 chunks total:
    4972  DICCIONARIO GARIFUNA.pdf
     467  LIBRO SERIE I.pdf
     376  LIBRO GARIFUNA II FORMATO DIGITAL.pdf
      58  Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt
      39  Yurumein Lexicon Pilot A-Ab-An.docx
      36  Yurumei_2024_YurumeinAI_Archivo_Bilingue.docx
      36  Yurumei_2024_Archivo.docx
      26  Yurumein_Garifuna_Identity_EN_ES.docx
      21  Yurumein_AI_Chief_Joseph_Chatoyer_EN_ES.docx
      20  Himno_Nacional_Guatemala_Garifuna_Espanol_English_Yurumein_AI.docx
      16  Baliceaux_Yurumein_AI_Historical_Brief.docx
       6  Garifuna Language Overview.pdf
       3  Garifuna_Musical_Instruments_Dataset.pdf.pdf

  Classification note: the 7 new documents (all except the original 6)
  are NOT registered in source_classifier._ROLE_MAP.  They will be retrieved
  by vector/keyword search but will NOT receive a Source label in the LLM
  context, will NOT be boosted by role-based scoring, and CANNOT be top-up
  targeted.  The benchmark measures whether they appear in retrieved_docs
  regardless of their classification status.

  Special note — Historia document:
  Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt carries metadata
  publication_allowed: false and review_state: pending_cultural_review.
  It is indexed in PGVector and WILL be retrieved, but it is a staging draft.

Safety
------
  All database operations are SELECT-only on existing tables.
  Embedding API calls are made (~46 questions × 2 = ~92 calls to
  OpenAI text-embedding-3-small).
  PGVector __init__ may issue idempotent CREATE TABLE IF NOT EXISTS;
  this is safe on production where tables already exist.
  Script aborts if chunk count < 1,000 (protects against local dev DB).
  No vectors added, deleted, or modified.
  No application code restarted.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text

from app.core.config import settings
from app.services.chatbot.source_classifier import classify_source
from app.services.utils.RAG_pipeline.workflow.nodes import Nodes

# ─── CONSTANTS ────────────────────────────────────────────────────────────────

MIN_COLLECTION_CHUNKS = 1_000
REPORT_FILE = "stage_a_results.json"

# Source-key substrings for expected-source matching.
# Each constant is checked via case-insensitive substring match against the
# source path stored in chunk metadata.  Choose substrings that are unique
# to exactly one indexed document.
#
# Original six sources (source_classifier.py recognises all of these):
SRC_DICCIONARIO  = "DICCIONARIO GARIFUNA"
SRC_SERIE_I      = "LIBRO SERIE I"
SRC_LIBRO_II     = "LIBRO GARIFUNA II FORMATO DIGITAL"
SRC_LEXICON      = "YURUMEIN LEXICON"
SRC_ANTHEM       = "HIMNO_NACIONAL_GUATEMALA_GARIFUNA"
# NOTE: Musical Instruments filename uses underscores; "MUSICAL INSTRUMENTS"
# (space) does NOT match the path "Garifuna_Musical_Instruments_Dataset.pdf.pdf".
# Using the correct substring:
SRC_INSTRUMENTS  = "Musical_Instruments_Dataset"
SRC_OVERVIEW     = "GARIFUNA LANGUAGE OVERVIEW"

# Seven new documents (source_classifier.py does NOT know these roles):
SRC_BALICEAUX         = "Baliceaux_Yurumein_AI_Historical_Brief"
SRC_CHATOYER          = "Chief_Joseph_Chatoyer"
SRC_HISTORIA          = "Historia_ES_EN"
SRC_GARIFUNA_IDENTITY = "Garifuna_Identity_EN_ES"
SRC_ARCHIVO_BILINGUE  = "Archivo_Bilingue"
# Include .docx extension to distinguish from Archivo_Bilingue:
SRC_ARCHIVO           = "Yurumei_2024_Archivo.docx"


# ─── 46-QUESTION EVALUATION SET ───────────────────────────────────────────────
#
# valid_source_keys : list of source-key substrings that are acceptable sources
#                     for this question.  EMPTY LIST = edge case (diagnostic only).
#                     First entry = primary (most specific) expected source.
#
# expected_passage_fragment : short substring (case-insensitive) verified to
#                     appear in at least one chunk of a valid source document.
#                     None = source-level hit only; no passage scoring.
#
# notes             : provenance and grounding for expected values.

QUESTIONS = [
    # ── Category 1: Baliceaux ──────────────────────────────────────────────────
    # Primary: dedicated Baliceaux historical brief.
    # Secondary: Chatoyer doc (chunk 5 covers Baliceaux–Roatán diaspora);
    #            Garifuna Identity doc (chunk 4 covers 1796 Baliceaux confinement).
    {
        "id": "C1Q1", "category": "Baliceaux", "language": "en",
        "query": "What was Baliceaux and what happened to the Garifuna there?",
        "valid_source_keys": [SRC_BALICEAUX, SRC_CHATOYER, SRC_GARIFUNA_IDENTITY],
        "expected_passage_fragment": None,
        "notes": (
            "Baliceaux Historical Brief is the primary dedicated source. "
            "Chatoyer doc chunk 5 also covers Baliceaux. "
            "Garifuna Identity doc chunk 4 names Baliceaux."
        ),
    },
    {
        "id": "C1Q2", "category": "Baliceaux", "language": "es",
        "query": "¿Qué fue Baliceaux y cuál fue su significado para el pueblo garífuna?",
        "valid_source_keys": [SRC_BALICEAUX, SRC_CHATOYER, SRC_GARIFUNA_IDENTITY],
        "expected_passage_fragment": None,
        "notes": "Spanish query; same valid sources.",
    },
    {
        "id": "C1Q3", "category": "Baliceaux", "language": "en",
        "query": "What happened on Baliceaux in 1796 when the British transferred Garifuna captives there?",
        "valid_source_keys": [SRC_BALICEAUX, SRC_CHATOYER, SRC_GARIFUNA_IDENTITY],
        "expected_passage_fragment": "1796",
        "notes": (
            "Fragment '1796' verified in Baliceaux brief chunk 2: "
            "'1796 Following the defeat of Garífuna resistance on Saint Vincent, "
            "British authorities forcibly transferred thousands to Baliceaux.'"
        ),
    },
    {
        "id": "C1Q4", "category": "Baliceaux", "language": "es",
        "query": "¿Cuántos garífunas murieron en Baliceaux y cuál fue la causa?",
        "valid_source_keys": [SRC_BALICEAUX, SRC_CHATOYER, SRC_GARIFUNA_IDENTITY],
        "expected_passage_fragment": None,
        "notes": (
            "Death-toll estimates appear in brief chunk 2 ('Estimates vary across "
            "historical accounts and should be attributed rather than presented as exact'). "
            "Passage not verified at a specific numeric claim."
        ),
    },
    {
        "id": "C1Q5", "category": "Baliceaux", "language": "en",
        "query": "How did the internment on Baliceaux lead to the Garifuna deportation to Central America in 1797?",
        "valid_source_keys": [SRC_BALICEAUX, SRC_CHATOYER, SRC_GARIFUNA_IDENTITY],
        "expected_passage_fragment": "1797",
        "notes": (
            "Fragment '1797' verified in Baliceaux brief chunk 2: '1797 Survi...'. "
            "Also in Chatoyer doc chunk 5 and Garifuna Identity chunk 4."
        ),
    },

    # ── Category 2: Chief Joseph Chatoyer / Satuye ────────────────────────────
    # Primary: dedicated Chatoyer document.
    # Secondary: Garifuna Identity doc; Libro Garífuna II.
    {
        "id": "C2Q1", "category": "Chatoyer_Satuye", "language": "en",
        "query": "Who was Satuye and what role did he play in Garifuna resistance against the British?",
        "valid_source_keys": [SRC_CHATOYER, SRC_GARIFUNA_IDENTITY, SRC_LIBRO_II],
        "expected_passage_fragment": "Satuye",
        "notes": (
            "Fragment 'Satuye' verified in Chatoyer doc chunk 1: "
            "'also known as Satuye or Satuyé, was a leading Garífuna (Garinagu) chief'."
        ),
    },
    {
        "id": "C2Q2", "category": "Chatoyer_Satuye", "language": "es",
        "query": "¿Quién fue Joseph Chatoyer y por qué es un símbolo de la identidad garífuna?",
        "valid_source_keys": [SRC_CHATOYER, SRC_GARIFUNA_IDENTITY, SRC_LIBRO_II],
        "expected_passage_fragment": None,
        "notes": "Spanish query; Chatoyer doc is primary.",
    },
    {
        "id": "C2Q3", "category": "Chatoyer_Satuye", "language": "en",
        "query": "Describe the Black Carib resistance against British colonization of Saint Vincent.",
        "valid_source_keys": [SRC_CHATOYER, SRC_GARIFUNA_IDENTITY, SRC_LIBRO_II],
        "expected_passage_fragment": None,
        "notes": "Black Carib Wars context. Chatoyer doc chunk 2: 'colonial struggle'.",
    },
    {
        "id": "C2Q4", "category": "Chatoyer_Satuye", "language": "es",
        "query": "¿Qué ocurrió en la guerra de 1795 entre los garífunas y los británicos?",
        "valid_source_keys": [SRC_CHATOYER, SRC_LIBRO_II],
        "expected_passage_fragment": None,
        "notes": "1795 uprising. Chatoyer doc chunk 4 covers this.",
    },
    {
        "id": "C2Q5", "category": "Chatoyer_Satuye", "language": "en",
        "query": "What was the outcome of the Carib Wars for the Garifuna people after Chatoyer's death?",
        "valid_source_keys": [SRC_CHATOYER, SRC_GARIFUNA_IDENTITY, SRC_BALICEAUX, SRC_LIBRO_II],
        "expected_passage_fragment": None,
        "notes": (
            "Post-war deportation. Chatoyer doc chunk 5: "
            "'Baliceaux, Roatán, and the Garífuna diaspora'. Multiple sources cover this."
        ),
    },

    # ── Category 3: Yurumein and Garifuna identity ────────────────────────────
    # Primary: Garifuna Identity doc; also Historia, Chatoyer, Libro II.
    {
        "id": "C3Q1", "category": "Yurumein_Identity", "language": "en",
        "query": "What does the name Yurumein mean and why is it significant to the Garifuna?",
        "valid_source_keys": [SRC_GARIFUNA_IDENTITY, SRC_HISTORIA, SRC_CHATOYER, SRC_LIBRO_II],
        "expected_passage_fragment": "Yurumein",
        "notes": (
            "Fragment 'Yurumein' verified in multiple sources. "
            "Historia chunk 4: 'Yurumein es el nombre garífuna de San Vicente'. "
            "Chatoyer chunk 2: 'Yurumein was the homeland of the Garinagu'."
        ),
    },
    {
        "id": "C3Q2", "category": "Yurumein_Identity", "language": "es",
        "query": "¿Cuáles son los orígenes étnicos del pueblo garífuna?",
        "valid_source_keys": [SRC_GARIFUNA_IDENTITY, SRC_HISTORIA, SRC_CHATOYER, SRC_LIBRO_II],
        "expected_passage_fragment": None,
        "notes": "African + Arawak + Carib origins. All four sources cover this.",
    },
    {
        "id": "C3Q3", "category": "Yurumein_Identity", "language": "en",
        "query": "Explain the African, Arawak, and Carib heritage of the Garifuna people.",
        "valid_source_keys": [SRC_GARIFUNA_IDENTITY, SRC_HISTORIA, SRC_CHATOYER, SRC_LIBRO_II],
        "expected_passage_fragment": "afric",
        "notes": (
            "Fragment 'afric' (case-insensitive; matches 'africana', 'African') "
            "verified in Identity chunk 2 and Chatoyer chunk 2."
        ),
    },
    {
        "id": "C3Q4", "category": "Yurumein_Identity", "language": "es",
        "query": "¿Qué fue la deportación de 1797 y cómo definió la identidad garífuna moderna?",
        "valid_source_keys": [SRC_GARIFUNA_IDENTITY, SRC_HISTORIA, SRC_BALICEAUX, SRC_CHATOYER, SRC_LIBRO_II],
        "expected_passage_fragment": "1797",
        "notes": (
            "Fragment '1797' verified in Baliceaux brief chunk 2, "
            "Garifuna Identity chunk 4 ('April 1797, survivors were forcibly transported'), "
            "Chatoyer chunk 5."
        ),
    },
    {
        "id": "C3Q5", "category": "Yurumein_Identity", "language": "en",
        "query": "Where are Garifuna communities found today, and how did they get there?",
        "valid_source_keys": [SRC_GARIFUNA_IDENTITY, SRC_HISTORIA, SRC_ARCHIVO_BILINGUE, SRC_ARCHIVO, SRC_LIBRO_II],
        "expected_passage_fragment": None,
        "notes": "Diaspora: Honduras, Belize, Guatemala, Nicaragua. Multiple sources.",
    },

    # ── Category 4: Livingston and community ──────────────────────────────────
    # Primary: community archive documents (Archivo Bilingue and Archivo).
    # Secondary: Libro Garífuna II; Garifuna Identity.
    {
        "id": "C4Q1", "category": "Livingston_Community", "language": "en",
        "query": "What is the significance of Livingston, Guatemala in Garifuna history and community life?",
        "valid_source_keys": [SRC_ARCHIVO_BILINGUE, SRC_ARCHIVO, SRC_GARIFUNA_IDENTITY, SRC_LIBRO_II],
        "expected_passage_fragment": "Livingston",
        "notes": (
            "Fragment 'Livingston' verified in Archivo Bilingue chunk 0: "
            "'Hijos de Livingston | Livingston (La Buga), Izabal, Guatemala'. "
            "Same in Archivo chunk 0."
        ),
    },
    {
        "id": "C4Q2", "category": "Livingston_Community", "language": "es",
        "query": "¿Qué comunidades garífunas se encuentran en la costa atlántica de Guatemala?",
        "valid_source_keys": [SRC_ARCHIVO_BILINGUE, SRC_ARCHIVO, SRC_GARIFUNA_IDENTITY, SRC_LIBRO_II],
        "expected_passage_fragment": None,
        "notes": "Guatemala Garifuna communities. Archive docs cover La Buga / Izabal.",
    },
    {
        "id": "C4Q3", "category": "Livingston_Community", "language": "en",
        "query": "Describe the oral history of Garifuna arrival and settlement in Livingston, Guatemala.",
        "valid_source_keys": [SRC_ARCHIVO_BILINGUE, SRC_ARCHIVO, SRC_GARIFUNA_IDENTITY],
        "expected_passage_fragment": None,
        "notes": (
            "Archive docs cover Yurumei ceremony, Mayuru founding Livingston. "
            "Garifuna Identity doc covers settlement after deportation."
        ),
    },
    {
        "id": "C4Q4", "category": "Livingston_Community", "language": "es",
        "query": "¿Quién es Elmer Mauricio Enríquez Bermúdez y cuál es su legado cultural garífuna?",
        "valid_source_keys": [SRC_LIBRO_II, SRC_DICCIONARIO],
        "expected_passage_fragment": "Enríquez",
        "notes": (
            "ABÜRÜHATI attribution. Fragment 'Enríquez' expected in Libro Garífuna II. "
            "Archive documents do not specifically reference him."
        ),
    },
    {
        "id": "C4Q5", "category": "Livingston_Community", "language": "en",
        "query": "What is the Yurumei ceremony and how does it connect the Livingston community to Garifuna heritage?",
        "valid_source_keys": [SRC_ARCHIVO_BILINGUE, SRC_ARCHIVO],
        "expected_passage_fragment": "Yurumei",
        "notes": (
            "Fragment 'Yurumei' verified in Archivo Bilingue chunk 3: "
            "'en Livingston, Izabal, el Yurumei conmemora la travesía'. "
            "Dedicated archive source."
        ),
    },

    # ── Category 5: Guatemala national anthem ─────────────────────────────────
    # All five queries trigger Intent.ANTHEM → anthem_ordered retrieval path.
    {
        "id": "C5Q1", "category": "National_Anthem", "language": "en",
        "query": "Show me the Garifuna version of the Guatemala national anthem.",
        "valid_source_keys": [SRC_ANTHEM],
        "expected_passage_fragment": None,
        "notes": "Triggers anthem_ordered path. retrieval_path must be 'anthem_ordered'.",
    },
    {
        "id": "C5Q2", "category": "National_Anthem", "language": "es",
        "query": "¿Cuál es la letra del himno nacional de Guatemala en versión garífuna?",
        "valid_source_keys": [SRC_ANTHEM],
        "expected_passage_fragment": None,
        "notes": "Spanish anthem query; triggers anthem_ordered path.",
    },
    {
        "id": "C5Q3", "category": "National_Anthem", "language": "en",
        "query": "What document in this knowledge base contains the Garifuna national anthem of Guatemala?",
        "valid_source_keys": [SRC_ANTHEM],
        "expected_passage_fragment": None,
        "notes": "Document identification query; anthem intent should fire.",
    },
    {
        "id": "C5Q4", "category": "National_Anthem", "language": "en",
        "query": "Recite the full Garifuna national anthem of Guatemala.",
        "valid_source_keys": [SRC_ANTHEM],
        "expected_passage_fragment": None,
        "notes": "Full recitation request; triggers anthem_ordered path.",
    },
    {
        "id": "C5Q5", "category": "National_Anthem", "language": "es",
        "query": "Dame la letra completa del himno nacional de Guatemala en garífuna.",
        "valid_source_keys": [SRC_ANTHEM],
        "expected_passage_fragment": None,
        "notes": "Full lyrics in Spanish; triggers anthem_ordered path.",
    },

    # ── Category 6: Garifuna language ─────────────────────────────────────────
    {
        "id": "C6Q1", "category": "Language", "language": "en",
        "query": "How do you say 'father' in Garifuna?",
        "valid_source_keys": [SRC_DICCIONARIO],
        "expected_passage_fragment": "Baba",
        "notes": (
            "LEXICAL PATH HIT: lexical parser fires. "
            "Full yurumein_lexicon SQL table (6720+ rows) includes English lookup. "
            "Row id=6720: english='father' → garifuna='Baba, Tata, Úguchilin'. "
            "Diccionario p.144 (SIGNIFICADO): Baba, Tata, Úguchilin (documented authority). "
            "Retrieved_docs=[] is intentional (lexical path); Stage A scores this via "
            "lexicon_structured_hit, not retrieved_docs matching. "
            "Fragment 'Baba' IS in the LEXICON_CONTEXT response string."
        ),
    },
    {
        "id": "C6Q2", "category": "Language", "language": "es",
        "query": "¿Cuántas vocales tiene el idioma garífuna y cuáles son?",
        "valid_source_keys": [SRC_SERIE_I],
        "expected_passage_fragment": None,
        "notes": "PRONUNCIATION intent → preferred FOUNDATIONAL_LANGUAGE_EDUCATION. Libro Serie I p.4–5.",
    },
    {
        "id": "C6Q3", "category": "Language", "language": "en",
        "query": "What linguistic families is the Garifuna language descended from?",
        "valid_source_keys": [SRC_GARIFUNA_IDENTITY, SRC_HISTORIA, SRC_LIBRO_II],
        "expected_passage_fragment": None,
        "notes": "Arawak-Carib roots. Identity doc and Historia cover this explicitly.",
    },
    {
        "id": "C6Q4", "category": "Language", "language": "es",
        "query": "¿Cómo se pronuncia la letra ü en el idioma garífuna?",
        "valid_source_keys": [SRC_SERIE_I],
        "expected_passage_fragment": None,
        "notes": "ü phonology; Libro Serie I covers pronunciation. PRONUNCIATION intent.",
    },
    {
        "id": "C6Q5", "category": "Language", "language": "en",
        "query": "What are the numbers one through three in Garifuna?",
        "valid_source_keys": [SRC_SERIE_I],
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
        "valid_source_keys": [SRC_LIBRO_II, SRC_DICCIONARIO],
        "expected_passage_fragment": None,
        "notes": (
            "CULTURAL_HISTORICAL intent. Contextual expansion 'Gayunari Dabuyaba' fires. "
            "Libro Garífuna II covers Dügü. Diccionario p.43: Dügü→Gayunari, Chugú→Dabuyaba."
        ),
    },
    {
        "id": "C7Q2", "category": "Cultural_Traditions", "language": "es",
        "query": "¿Qué es la punta y cuál es su importancia en la cultura garífuna?",
        "valid_source_keys": [SRC_LIBRO_II, SRC_ARCHIVO_BILINGUE, SRC_ARCHIVO],
        "expected_passage_fragment": None,
        "notes": "Punta dance/music. Archive docs mention music and dance in Livingston.",
    },
    {
        "id": "C7Q3", "category": "Cultural_Traditions", "language": "en",
        "query": "Describe the musical instruments used in traditional Garifuna ceremonies.",
        "valid_source_keys": [SRC_INSTRUMENTS],
        "expected_passage_fragment": None,
        "notes": (
            "Musical Instruments dataset (3 chunks). "
            "NOTE: source path uses underscores (Musical_Instruments_Dataset), "
            "NOT spaces — classifier does not recognise this source."
        ),
    },
    {
        "id": "C7Q4", "category": "Cultural_Traditions", "language": "es",
        "query": "¿Qué es el Dabuyaba y en qué ceremonias se utiliza?",
        "valid_source_keys": [SRC_DICCIONARIO, SRC_LIBRO_II],
        "expected_passage_fragment": "Dabuyaba",
        "notes": (
            "Fragment 'Dabuyaba' expected in Diccionario p.43: "
            "'Casa en donde se realiza ceremonia ancestral. Chugu, Dabuyaba'. "
            "Contextual expansion 'Gayunari Dabuyaba' should surface this chunk."
        ),
    },
    {
        "id": "C7Q5", "category": "Cultural_Traditions", "language": "en",
        "query": "What role does the garaon drum play in Garifuna music and ceremony?",
        "valid_source_keys": [SRC_INSTRUMENTS],
        "expected_passage_fragment": None,
        "notes": "Garaon drum. Musical Instruments dataset (unclassified source).",
    },

    # ── Category 8: Edge cases (diagnostic only — not scored pass/fail) ────────
    # valid_source_keys = [] for all edge cases.
    # These record system behavior for queries at the boundaries of coverage.
    {
        "id": "C8Q1", "category": "Edge_Cases", "language": "en",
        "query": "What does Libro Serie I say on page 500?",
        "valid_source_keys": [],
        "expected_passage_fragment": None,
        "notes": (
            "DIAGNOSTIC. parse_page_request fires: source=%LIBRO SERIE I%, pages=(500,). "
            "Page 500 does not exist → page-not-found context, no retrieved_docs."
        ),
    },
    {
        "id": "C8Q2", "category": "Edge_Cases", "language": "en",
        "query": "What is the Garifuna word for automobile?",
        "valid_source_keys": [],
        "expected_passage_fragment": None,
        "notes": (
            "DIAGNOSTIC. Lexical parser fires ('what is the Garifuna word for X'). "
            "'Automobile' not in lexicon (A–An) or Diccionario. "
            "Expect lexicon miss + generic hybrid retrieval."
        ),
    },
    {
        "id": "C8Q3", "category": "Edge_Cases", "language": "es",
        "query": "¿Qué dice el Libro Garífuna II en la página 999?",
        "valid_source_keys": [],
        "expected_passage_fragment": None,
        "notes": (
            "DIAGNOSTIC. parse_page_request fires: source=%GARIFUNA II%, pages=(999,). "
            "Page 999 does not exist → page-not-found context."
        ),
    },
    {
        "id": "C8Q4", "category": "Edge_Cases", "language": "en",
        "query": "Was the Garifuna language influenced by ancient Sumerian or Egyptian writing systems?",
        "valid_source_keys": [],
        "expected_passage_fragment": None,
        "notes": (
            "DIAGNOSTIC. Unsupported claim; sources cover Arawak/Carib/European contact only. "
            "Any retrieved docs are low-relevance noise."
        ),
    },
    {
        "id": "C8Q5", "category": "Edge_Cases", "language": "en",
        "query": "Tell me about the Garifuna diaspora community in Paris, France.",
        "valid_source_keys": [],
        "expected_passage_fragment": None,
        "notes": (
            "DIAGNOSTIC. Paris not documented in indexed sources. "
            "CULTURAL_HISTORICAL intent fires. May retrieve general diaspora chunks."
        ),
    },

    # ── Category 9: Document coverage — targeted retrieval tests ─────────────
    # One question per new document.  Each query is designed to surface content
    # unique to that document.  Passage fragments are verified against actual
    # indexed chunk content.
    {
        "id": "C9Q1", "category": "Document_Coverage", "language": "en",
        "query": "What is specifically documented about Baliceaux as a site of confinement in 1796?",
        "valid_source_keys": [SRC_BALICEAUX, SRC_CHATOYER, SRC_GARIFUNA_IDENTITY],
        "expected_passage_fragment": "1796",
        "notes": (
            "TARGET: Baliceaux_Yurumein_AI_Historical_Brief.docx (16 chunks). "
            "Fragment '1796' verified in chunk 2."
        ),
    },
    {
        "id": "C9Q2", "category": "Document_Coverage", "language": "en",
        "query": "What was the 1773 peace agreement between the British and the Garifuna on Saint Vincent?",
        "valid_source_keys": [SRC_CHATOYER],
        "expected_passage_fragment": "1773",
        "notes": (
            "TARGET: Yurumein_AI_Chief_Joseph_Chatoyer_EN_ES.docx (21 chunks). "
            "Fragment '1773' verified in chunk 3: "
            "'The 1773 peace agreement established boundaries between "
            "British-controlled and Garífuna-held lands'."
        ),
    },
    {
        "id": "C9Q3", "category": "Document_Coverage", "language": "es",
        "query": "¿Qué información contiene el documento yurumein-history-001 sobre los orígenes garífunas?",
        "valid_source_keys": [SRC_HISTORIA],
        "expected_passage_fragment": "yurumein-history-001",
        "restriction_expected": True,
        "notes": (
            "TARGET: Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt (58 chunks). "
            "Fragment 'yurumein-history-001' verified in chunk 0 (document_id field). "
            "RESTRICTION: publication_allowed: false + pending_cultural_review. "
            "Historia must NOT appear in public context. "
            "RESTRICTED_OK = Historia absent (security success). "
            "RESTRICTION_LEAK = Historia present (security failure)."
        ),
    },
    {
        "id": "C9Q4", "category": "Document_Coverage", "language": "en",
        "query": "What does the Ben Petersen account say about Garifuna language and cultural continuity?",
        "valid_source_keys": [SRC_GARIFUNA_IDENTITY],
        "expected_passage_fragment": "Ben Petersen",
        "notes": (
            "TARGET: Yurumein_Garifuna_Identity_EN_ES.docx (26 chunks). "
            "Fragment 'Ben Petersen' verified in chunk 0: "
            "'Curated reference derived from A Story About the Garifuna "
            "(Ben Petersen, 2012/2013 circulation)'."
        ),
    },
    {
        "id": "C9Q5", "category": "Document_Coverage", "language": "es",
        "query": "¿Qué documenta el archivo cultural de los Hijos de Livingston sobre la memoria garífuna?",
        "valid_source_keys": [SRC_ARCHIVO_BILINGUE, SRC_ARCHIVO],
        "expected_passage_fragment": "Hijos de Livingston",
        "notes": (
            "TARGET: Yurumei_2024_YurumeinAI_Archivo_Bilingue.docx (36 chunks). "
            "Fragment 'Hijos de Livingston' verified in chunk 0 and chunk 2. "
            "NOTE: Yurumei_2024_Archivo.docx has identical first chunks — "
            "both are valid sources for this query."
        ),
    },
    {
        "id": "C9Q6", "category": "Document_Coverage", "language": "es",
        "query": "¿Quién fue Marcos Sánchez Díaz, conocido como Mayuru, y cuál fue su papel en la fundación de Livingston?",
        "valid_source_keys": [SRC_ARCHIVO, SRC_ARCHIVO_BILINGUE],
        "expected_passage_fragment": "Mayuru",
        "notes": (
            "TARGET: Yurumei_2024_Archivo.docx (36 chunks). "
            "Fragment 'Mayuru' verified in chunk 5: "
            "'El documental reconoce a Marcos Sánchez Díaz, también llamado Mayuru, "
            "como figura fundadora de Livingston'. "
            "NOTE: Archivo_Bilingue has identical content — also listed as valid."
        ),
    },
]

assert len(QUESTIONS) == 46, f"Expected 46 questions, got {len(QUESTIONS)}"


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
        "source_inventory": [],
        "collections_found": [],
        "openai_key_set": bool(settings.OPENAI_API_KEY),
    }

    if not info["openai_key_set"]:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — embedding calls required by "
            "semantic search will fail."
        )

    try:
        with engine.connect() as conn:
            coll_rows = conn.execute(
                text("SELECT name FROM langchain_pg_collection ORDER BY name")
            ).mappings().all()
            info["collections_found"] = [r["name"] for r in coll_rows]

            count_row = conn.execute(
                text(
                    "SELECT COUNT(*) AS n FROM langchain_pg_embedding e "
                    "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                    "WHERE c.name = :col"
                ),
                {"col": collection_name},
            ).mappings().one_or_none()
            info["chunk_count"] = int(count_row["n"]) if count_row else 0

            src_rows = conn.execute(
                text(
                    "SELECT e.cmetadata->>'source' AS src, COUNT(*) AS n "
                    "FROM langchain_pg_embedding e "
                    "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                    "WHERE c.name = :col "
                    "GROUP BY e.cmetadata->>'source' ORDER BY n DESC"
                ),
                {"col": collection_name},
            ).mappings().all()
            info["source_inventory"] = [
                {"source": r["src"], "chunks": int(r["n"])} for r in src_rows
            ]
    except Exception as exc:
        raise RuntimeError(f"Database pre-flight failed: {exc}") from exc

    if collection_name not in info["collections_found"]:
        raise RuntimeError(
            f"Collection '{collection_name}' not found in the database. "
            f"Collections present: {info['collections_found']}. "
            f"Verify COLLECTION_NAME in .env."
        )

    if info["chunk_count"] < MIN_COLLECTION_CHUNKS:
        raise RuntimeError(
            f"Collection '{collection_name}' has only {info['chunk_count']} chunks "
            f"(minimum {MIN_COLLECTION_CHUNKS} required). "
            f"This looks like the empty local development database — "
            f"run the benchmark on the production VPS only."
        )

    return info


# ─── PURE HELPER FUNCTIONS ────────────────────────────────────────────────────

def source_key_in_docs(docs: list[dict], source_key: str) -> bool:
    """Return True if any doc's source path contains source_key (case-insensitive)."""
    key_upper = source_key.upper()
    return any(
        key_upper in (doc.get("metadata") or {}).get("source", "").upper()
        for doc in docs
    )


def any_source_in_docs(docs: list[dict], source_keys: list[str]) -> bool:
    """Return True if any doc matches any of the provided source keys."""
    return any(source_key_in_docs(docs, k) for k in source_keys)


def passage_in_docs(docs: list[dict], fragment: str, source_key: str) -> bool:
    """
    Return True if fragment appears (case-insensitive) in a doc that also
    matches source_key.  Both conditions must hold in the same doc.
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


def any_passage_in_docs(docs: list[dict], fragment: str, source_keys: list[str]) -> bool:
    """Return True if fragment appears in a doc matching any of source_keys."""
    return any(passage_in_docs(docs, fragment, k) for k in source_keys)


def detect_retrieval_path(context: str) -> str:
    """
    Infer which retrieval path was taken from the returned context string.

    "lexicon"            — structured lexicon hit (retrieved_docs=[])
    "lexicon_miss_hybrid"— lexical parser fired, lexicon returned nothing;
                           fell through to hybrid RAG with LEXICON_NOTE
    "anthem_ordered"     — deterministic source-ordered anthem retrieval
    "page_lookup"        — explicit page request (found or not-found)
    "hybrid"             — general hybrid RAG path
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
        "source_display": sm.display if sm else "unclassified",
        "page_label": meta.get("page_label"),
        "hybrid_score": round(float(doc.get("hybrid_score") or 0.0), 4),
        "retrieval_source": doc.get("retrieval_source"),
        "content_preview": (doc.get("page_content") or "")[:140].replace("\n", " "),
    }


def compute_summary(results: list[dict]) -> dict:
    """
    Compute aggregate retrieval metrics.

    primary_source_hit_rate_at_10    : fraction of sourced+hybrid questions where the
                                       PRIMARY valid source appeared in the raw top-10
    any_source_hit_rate_at_10        : same but any valid source
    primary_source_hit_rate_in_context: fraction of all sourced questions where the
                                       PRIMARY valid source appeared in the final top-5
    any_source_hit_rate_in_context   : same but any valid source
    passage_recall_at_10/in_context  : passage fragment found in any valid-source chunk
    """
    total = len(results)
    errors = sum(1 for r in results if r["error"])

    edge         = [r for r in results if r["is_edge_case"]]
    restricted   = [r for r in results if r.get("restriction_expected")]
    # Sourced excludes both edge-cases and restriction-only questions from hit-rate metrics.
    sourced      = [r for r in results if not r["is_edge_case"] and not r.get("restriction_expected")]

    sourced_hybrid = [r for r in sourced if r["top10_applicable"]]
    lexical_hits   = sum(1 for r in sourced if r.get("lexicon_structured_hit") is True)

    restriction_verified_count = sum(1 for r in restricted if r.get("restriction_verified") is True)
    restriction_leak_count     = sum(1 for r in restricted if r.get("restriction_verified") is False)

    primary10  = sum(1 for r in sourced_hybrid if r.get("primary_hit_at_10") is True)
    any10      = sum(1 for r in sourced_hybrid if r.get("any_hit_at_10") is True)
    primary_ctx = sum(1 for r in sourced if r.get("primary_hit_in_context") is True)
    any_ctx     = sum(1 for r in sourced if r.get("any_hit_in_context") is True)

    passage_q_top10 = [r for r in sourced_hybrid if r["expected_passage_fragment"]]
    passage_q_ctx   = [r for r in sourced if r["expected_passage_fragment"]]
    phits10  = sum(1 for r in passage_q_top10 if r.get("passage_found_at_10") is True)
    phits_ctx = sum(1 for r in passage_q_ctx  if r.get("passage_found_in_context") is True)

    categories: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {
                "total": 0,
                "is_edge_case_category": r["is_edge_case"],
                "primary_hit_at_10": 0,
                "any_hit_at_10": 0,
                "primary_hit_in_context": 0,
                "any_hit_in_context": 0,
                "errors": 0,
            }
        categories[cat]["total"] += 1
        if r["error"]:
            categories[cat]["errors"] += 1
        if r.get("primary_hit_at_10") is True:
            categories[cat]["primary_hit_at_10"] += 1
        if r.get("any_hit_at_10") is True:
            categories[cat]["any_hit_at_10"] += 1
        if r.get("primary_hit_in_context") is True:
            categories[cat]["primary_hit_in_context"] += 1
        if r.get("any_hit_in_context") is True:
            categories[cat]["any_hit_in_context"] += 1

    return {
        "total_questions": total,
        "errors": errors,
        "sourced_questions": len(sourced),
        "edge_case_questions": len(edge),
        "restriction_questions": len(restricted),
        "restriction_verified_count": restriction_verified_count,
        "restriction_leak_count": restriction_leak_count,
        "hybrid_path_questions": len(sourced_hybrid),
        "lexical_path_hits": lexical_hits,
        "primary_source_hit_rate_at_10": (
            round(primary10 / len(sourced_hybrid), 3) if sourced_hybrid else None
        ),
        "any_source_hit_rate_at_10": (
            round(any10 / len(sourced_hybrid), 3) if sourced_hybrid else None
        ),
        "primary_source_hit_rate_in_context": (
            round(primary_ctx / len(sourced), 3) if sourced else None
        ),
        "any_source_hit_rate_in_context": (
            round(any_ctx / len(sourced), 3) if sourced else None
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

    For sourced questions (valid_source_keys non-empty):
      1. Call global_retrieval_process() → production final context (top-5)
      2. For hybrid-path questions, call _hybrid_retrieve(top_k=10) → raw pool
      3. Score primary and any-source hits against both pools
      4. Score passage fragment against any valid-source doc

    For edge cases (valid_source_keys empty):
      1. Run retrieval; record all returned docs and the retrieval path
      2. No hit/miss scoring — purely diagnostic
    """
    is_edge_case        = not q["valid_source_keys"]
    restriction_expected = q.get("restriction_expected", False)
    primary_key         = q["valid_source_keys"][0] if q["valid_source_keys"] else None

    record: dict = {
        "id": q["id"],
        "category": q["category"],
        "query": q["query"],
        "language": q["language"],
        "is_edge_case": is_edge_case,
        "restriction_expected": restriction_expected,
        "restriction_verified": None,
        "valid_source_keys": q["valid_source_keys"],
        "primary_source_key": primary_key,
        "expected_passage_fragment": q["expected_passage_fragment"],
        "notes": q["notes"],
        "retrieval_path": None,
        "top10_applicable": True,
        "top10_pool": [],
        "top5_context": [],
        "primary_hit_at_10": None,
        "any_hit_at_10": None,
        "primary_hit_in_context": None,
        "any_hit_in_context": None,
        "passage_found_at_10": None,
        "passage_found_in_context": None,
        "lexicon_structured_hit": False,
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

        top10_applicable = retrieval_path in ("hybrid", "lexicon_miss_hybrid")
        record["top10_applicable"] = top10_applicable

        raw_top10: list[dict] = []
        if top10_applicable:
            retrieval_query = nodes._contextualize_query(q["query"], [])
            raw_top10 = nodes._hybrid_retrieve(query=retrieval_query, top_k=10)
            record["top10_pool"] = [doc_summary(d) for d in raw_top10]
        elif retrieval_path == "anthem_ordered":
            raw_top10 = final_docs
            record["top10_pool"] = record["top5_context"]

        # Restriction check — runs before regular scoring.
        # For restriction_expected questions (e.g. C9Q3 / Historia):
        #   RESTRICTED_OK   = restricted source absent from public context (success)
        #   RESTRICTION_LEAK = restricted source present in public context (failure)
        if restriction_expected:
            historia_in_context = source_key_in_docs(final_docs, primary_key or "")
            record["restriction_verified"] = not historia_in_context

        # Scoring — skipped for edge cases and restriction-only questions.
        if not is_edge_case and not restriction_expected:
            valid_keys = q["valid_source_keys"]
            frag = q["expected_passage_fragment"]

            if retrieval_path == "lexicon":
                # Structured lexical hit: the pipeline answered via LEXICON_CONTEXT,
                # not via retrieved_docs (which is intentionally [] on the lexical path).
                # Score as a full retrieval success; source matching on empty docs
                # would produce a false MISS.
                record["lexicon_structured_hit"] = True
                record["primary_hit_in_context"] = True
                record["any_hit_in_context"] = True
                if frag:
                    record["passage_found_in_context"] = frag.lower() in ctx_string.lower()
            else:
                record["primary_hit_at_10"] = (
                    source_key_in_docs(raw_top10, primary_key) if top10_applicable else None
                )
                record["any_hit_at_10"] = (
                    any_source_in_docs(raw_top10, valid_keys) if top10_applicable else None
                )
                record["primary_hit_in_context"] = source_key_in_docs(final_docs, primary_key)
                record["any_hit_in_context"] = any_source_in_docs(final_docs, valid_keys)

                if frag:
                    record["passage_found_at_10"] = (
                        any_passage_in_docs(raw_top10, frag, valid_keys)
                        if top10_applicable else None
                    )
                    record["passage_found_in_context"] = any_passage_in_docs(
                        final_docs, frag, valid_keys
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
    print(f"  Sources      : {len(pf['source_inventory'])}")
    for s in pf["source_inventory"]:
        import os as _os
        fname = _os.path.basename(s["source"] or "")
        print(f"    {s['chunks']:>5}  {fname}")

    print("\n[2/4] Initialising retrieval engine ...")
    nodes = Nodes()
    _ = nodes.vector_store
    print("  Engine ready.")

    print(f"\n[3/4] Evaluating {len(QUESTIONS)} questions ...")
    results: list[dict] = []

    for i, q in enumerate(QUESTIONS, 1):
        prefix = f"  [{i:02d}/{len(QUESTIONS)}] {q['id']:<6} ({q['category']})"
        print(f"{prefix} ...", end="", flush=True)
        r = await evaluate_question(nodes, q)

        if r["error"]:
            status = "ERROR"
        elif r["is_edge_case"]:
            status = "DIAG"
        elif r.get("restriction_expected"):
            status = "RESTRICTED_OK" if r.get("restriction_verified") else "RESTRICTION_LEAK"
        elif r.get("lexicon_structured_hit"):
            status = "LEXICON"
        elif r.get("primary_hit_in_context") is True:
            status = "HIT"
        elif r.get("any_hit_in_context") is True:
            status = "ANY"
        else:
            status = "MISS"

        print(f" {status:<4} [{r['elapsed_s']}s]")
        if r["error"]:
            print(f"    ! {r['error']}")

        results.append(r)

    print("\n[4/4] Computing summary ...")
    summary = compute_summary(results)

    print("\n" + "─" * 72)
    print(f"  Questions evaluated                  : {summary['total_questions']}")
    print(f"  Retrieval errors                     : {summary['errors']}")
    print(f"  Sourced questions                    : {summary['sourced_questions']}")
    print(f"  Edge-case questions (diagnostic)     : {summary['edge_case_questions']}")
    print(f"  Restriction-check questions          : {summary['restriction_questions']}")
    print(f"    RESTRICTED_OK (source absent)      : {summary['restriction_verified_count']}")
    print(f"    RESTRICTION_LEAK (source present!) : {summary['restriction_leak_count']}")
    print(f"  Hybrid-path questions                : {summary['hybrid_path_questions']}")
    print(f"  Lexical-path structured hits         : {summary['lexical_path_hits']}")
    print()
    print(f"  Primary source hit-rate@10 (hybrid)  : {summary['primary_source_hit_rate_at_10']}")
    print(f"  Any source hit-rate@10 (hybrid)      : {summary['any_source_hit_rate_at_10']}")
    print(f"  Primary source hit-rate in context   : {summary['primary_source_hit_rate_in_context']}")
    print(f"  Any source hit-rate in context       : {summary['any_source_hit_rate_in_context']}")
    print(f"  Passage recall@10                    : {summary['passage_recall_at_10']}")
    print(f"  Passage recall in context            : {summary['passage_recall_in_context']}")
    print("─" * 72)
    print("  Category breakdown:")
    for cat, m in summary["by_category"].items():
        n = m["total"]
        edge_tag = " [diagnostic]" if m["is_edge_case_category"] else ""
        print(
            f"    {cat:<30} "
            f"p10={m['primary_hit_at_10']}/{n}  "
            f"a10={m['any_hit_at_10']}/{n}  "
            f"ctx={m['primary_hit_in_context']}/{n}  "
            f"any_ctx={m['any_hit_in_context']}/{n}"
            f"{edge_tag}"
        )

    report = {
        "benchmark": "stage_a_retrieval",
        "version": "2.0",
        "run_at": datetime.now(timezone.utc).isoformat(),
        "output_file": REPORT_FILE,
        "edge_case_note": (
            "Questions with is_edge_case=true are diagnostic tests only. "
            "They record system behavior for queries about missing pages, "
            "unsupported claims, or unknown terms. "
            "They are NOT scored as retrieval pass/fail."
        ),
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
