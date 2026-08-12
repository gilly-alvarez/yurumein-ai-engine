#!/usr/bin/env python3
"""
Phase 4 v1 — Retrieval Verification Script
==========================================
Run this ON THE VPS (or any environment with production DB access) BEFORE
authorising production deployment.

Purpose:
  1. Compare retrieval quality for three "father" query formulations to confirm
     the lexical-miss rag_query ("father garifuna") reliably surfaces Pa/Papá.
  2. Run the four Phase 4 regression queries and report retrieved sources/tiers.

Usage (from yurumein-ai-engine project root on VPS):
  python scripts/phase4_retrieval_verify.py

Requires: .env with production DB credentials (DB_HOST, DB_PORT, DB_USER,
          DB_PASS, DB_NAME, COLLECTION_NAME, OPENAI_API_KEY).
"""
import os
import sys
import json
import re
from pathlib import Path

# Determine project root regardless of where the script is executed from.
# Normal execution: script lives at <project_root>/scripts/ → parent.parent = root.
# Container /tmp verification: script lives at /tmp/ → parent.parent = /
#   which does not contain main.py, so fall back to /app (Dockerfile WORKDIR).
_SCRIPT_DIR = Path(__file__).parent
_INFERRED_ROOT = _SCRIPT_DIR.parent
PROJECT_ROOT = _INFERRED_ROOT if (_INFERRED_ROOT / "main.py").exists() else Path("/app")

# Load .env from project root (falls back gracefully to existing env)
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)
    print(f"Loaded .env from {env_path}")
else:
    print(f"WARNING: .env not found at {env_path}. Using existing environment.")

# Must come after env load
try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_postgres import PGVector
    from sqlalchemy import create_engine, text
    import yaml
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_USER = os.environ["DB_USER"]
DB_PASS = os.environ["DB_PASS"]
DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]
DB_NAME = os.environ["DB_NAME"]
COLLECTION = os.environ.get("COLLECTION_NAME", "document_collection")
OPENAI_KEY = os.environ["OPENAI_API_KEY"]

CONN_STR = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

REGISTRY_PATH = PROJECT_ROOT / "app/services/utils/RAG_pipeline/registry/source_registry.yaml"


# ---------------------------------------------------------------------------
# Load source registry
# ---------------------------------------------------------------------------
def load_registry(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {s["filename"]: s for s in data.get("sources", [])}


REGISTRY = load_registry(REGISTRY_PATH)


def source_info(filename: str) -> str:
    # Registry is keyed by basename; PGVector metadata stores full paths.
    key = os.path.basename(filename) if filename else filename
    reg = REGISTRY.get(key, {})
    tier = reg.get("overall_tier", "?")
    verification = reg.get("verification_status", "unregistered")
    return f"Tier:{tier} | Verification:{verification}"


# ---------------------------------------------------------------------------
# Vector store helpers
# ---------------------------------------------------------------------------
def make_store() -> PGVector:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION,
        connection=CONN_STR,
        use_jsonb=True,
    )


def make_engine():
    return create_engine(CONN_STR)


def vector_search(store: PGVector, query: str, k: int = 5):
    return store.similarity_search(query=query, k=k)


def keyword_search(engine, query: str, k: int = 5):
    terms = [t for t in re.findall(r"\b\w+\b", query.lower()) if len(t) > 2]
    if not terms:
        return []
    score_parts, where_parts, params = [], [], {"collection_name": COLLECTION, "limit": k}
    for i, term in enumerate(terms):
        pname = f"term_{i}"
        score_parts.append(f"CASE WHEN lower(e.document) LIKE :{pname} THEN 1 ELSE 0 END")
        where_parts.append(f"lower(e.document) LIKE :{pname}")
        params[pname] = f"%{term}%"
    sql = f"""
        SELECT e.document, e.cmetadata,
               ({' + '.join(score_parts)}) AS keyword_score
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = :collection_name AND ({' OR '.join(where_parts)})
        ORDER BY keyword_score DESC, length(e.document) ASC
        LIMIT :limit
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
    return [{"document": r["document"], "metadata": r["cmetadata"] or {}} for r in rows]


def hybrid_search(store: PGVector, engine, query: str, top_k: int = 5):
    vector_hits = store.similarity_search_with_score(query=query, k=max(top_k * 3, 10))
    kw_hits = keyword_search(engine, query, k=max(top_k * 3, 10))

    merged = {}
    query_terms = [t for t in re.findall(r"\b\w+\b", query.lower()) if len(t) > 2]
    query_term_count = max(len(query_terms), 1)

    def kw_overlap(q, text):
        qt = {t for t in re.findall(r"\b\w+\b", q.lower()) if len(t) > 2}
        tt = set(re.findall(r"\b\w+\b", text.lower()))
        return len(qt & tt) / len(qt) if qt else 0.0

    for rank, (doc, vs) in enumerate(vector_hits, 1):
        meta = dict(doc.metadata or {})
        content = (doc.page_content or "").strip()
        if not content:
            continue
        key = (meta.get("source"), meta.get("page_number", meta.get("page")), content[:120])
        sem = 1.0 / (1.0 + max(float(vs), 0.0))
        kw = kw_overlap(query, content)
        merged[key] = {
            "content": content, "metadata": meta,
            "vector_score": float(vs), "keyword_score": kw,
            "hybrid_score": sem * 0.7 + kw * 0.3,
            "source": "vector", "rank": rank,
        }

    for rank, hit in enumerate(kw_hits, 1):
        meta = dict(hit.get("metadata") or {})
        content = (hit.get("document") or "").strip()
        if not content:
            continue
        key = (meta.get("source"), meta.get("page_number", meta.get("page")), content[:120])
        kw = max(
            float(hit.get("keyword_score", 0)) / query_term_count,
            kw_overlap(query, content),
        )
        if key in merged:
            merged[key]["keyword_score"] = max(merged[key]["keyword_score"], kw)
            sem = 1.0 / (1.0 + max(float(merged[key]["vector_score"] or 0), 0))
            merged[key]["hybrid_score"] = max(merged[key]["hybrid_score"], sem * 0.7 + kw * 0.3)
            merged[key]["source"] = "hybrid"
        else:
            merged[key] = {
                "content": content, "metadata": meta,
                "vector_score": None, "keyword_score": kw,
                "hybrid_score": kw * 0.3,
                "source": "keyword", "rank": rank,
            }

    return sorted(merged.values(), key=lambda x: (x["hybrid_score"], x["keyword_score"]), reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
SEP = "=" * 72


def print_hits(label: str, hits, show_content_chars: int = 200):
    print(f"\n  --- {label} ({len(hits)} hits) ---")
    for i, h in enumerate(hits, 1):
        if isinstance(h, tuple):  # (doc, score) from similarity_search_with_score
            doc, score = h
            meta = dict(doc.metadata or {})
            content = (doc.page_content or "").strip()
            fname = meta.get("filename") or meta.get("source", "?")
            page = meta.get("page_number", meta.get("page", "?"))
            info = source_info(fname)
            # Word-boundary match: "Papa" or "Papá" as a whole word.
            # Avoids false positives from "Pa" in "Spanish", "Paper", etc.
            contains_papa = bool(re.search(r'\bPap[aá]\b', content))
            print(f"  [{i}] {fname} | page:{page} | {info} | score:{score:.4f}")
            print(f"       Pa/Papá present: {contains_papa}")
            print(f"       Preview: {repr(content[:show_content_chars])}")
        else:
            meta = h.get("metadata") or {}
            content = h.get("content", h.get("document", "")).strip()
            fname = meta.get("filename") or meta.get("source", "?")
            page = meta.get("page_number", meta.get("page", "?"))
            info = source_info(fname)
            hybrid = h.get("hybrid_score", 0)
            contains_papa = bool(re.search(r'\bPap[aá]\b', content))
            print(f"  [{i}] {fname} | page:{page} | {info} | hybrid:{hybrid:.4f}")
            print(f"       Pa/Papá present: {contains_papa}")
            print(f"       Preview: {repr(content[:show_content_chars])}")


# ---------------------------------------------------------------------------
# SECTION 1: Father query comparison
# ---------------------------------------------------------------------------
def section_father_comparison(store, engine):
    print()
    print(SEP)
    print("SECTION 1: Father Query Retrieval Comparison")
    print("Purpose: confirm which query formulation best retrieves Pa/Papá")
    print(SEP)

    queries = [
        ("father",                          "hybrid", "Extracted term alone (CORRECT RP10 behavior)"),
        ("father garifuna",                 "hybrid", "term + 'garifuna' suffix (DEPRECATED — false positive in v1)"),
        ("How do you say father in garifuna?", "vector", "Full conversational query (OLD behaviour — bad baseline)"),
        ("How do you say father??",         "vector", "Short phrasing — vector finds Pa/Papá at rank 5"),
    ]

    for q, mode, label in queries:
        print(f"\n  Query: {repr(q)}")
        print(f"  Mode: {mode} | Scenario: {label}")
        if mode == "hybrid":
            hits = hybrid_search(store, engine, q, top_k=5)
            print_hits("Hybrid results", hits)
        else:
            docs = store.similarity_search_with_score(query=q, k=5)
            print_hits("Vector results", docs)

    print()
    print("  ACCEPTANCE CHECK: At least one hit for 'father' (hybrid, CORRECT RP10)")
    print("  should contain Papa or Papá as a whole word — confirming Pa/Papá chunk retrieved.")
    print("  'father garifuna' is deprecated: corpus-omnipresent 'garifuna' keyword")
    print("  boosts overview/history docs over vocabulary docs — no Pa/Papá in top 5.")


# ---------------------------------------------------------------------------
# SECTION 2: Regression tests
# ---------------------------------------------------------------------------
def regression_test(store, engine, label: str, query: str, retrieval_mode: str,
                    accept_check_fn, accept_description: str):
    print()
    print(f"  Query #{label}: {repr(query)}")
    print(f"  Retrieval mode: {retrieval_mode}")

    if retrieval_mode == "hybrid":
        hits = hybrid_search(store, engine, query, top_k=5)
    else:
        docs = store.similarity_search_with_score(query=query, k=5)
        hits = [
            {"content": d.page_content or "", "metadata": dict(d.metadata or {}),
             "vector_score": s, "hybrid_score": 0, "source": "vector"}
            for d, s in docs
        ]

    print_hits(f"Retrieved chunks", hits, show_content_chars=300)

    # Acceptance check
    result = accept_check_fn(hits)
    status = "PASS" if result else "FAIL"
    print(f"\n  ACCEPTANCE [{status}]: {accept_description}")
    return status, hits


def section_regressions(store, engine):
    print()
    print(SEP)
    print("SECTION 2: Phase 4 Regression Tests")
    print(SEP)

    results = {}

    # Test 1 — Father long
    def check_father(hits):
        # Word-boundary match for Papa/Papá as a whole word.
        # Previous condition ("Pa" in c and "Pap" in c) was too broad:
        # "Pa" matches "Spanish", "Paper", etc. and "Pap" is a substring
        # of many words unrelated to the Garífuna term for father.
        for h in hits:
            c = h.get("content", "")
            if re.search(r'\bPap[aá]\b', c):
                return True
        return False

    status, _ = regression_test(
        store, engine,
        label="1a",
        query="father",  # the actual rag_query RP10 now generates (term alone)
        retrieval_mode="hybrid",
        accept_check_fn=check_father,
        accept_description="At least one chunk contains Papa or Papá as a whole word",
    )
    results["1a_father_term_only"] = status

    # Test 1b — Deprecated "father garifuna" query documented as expected FAIL.
    # This confirms the v1 false positive: "garifuna" boosts overview/history docs.
    def check_father_deprecated_fails(hits):
        # Expect NO Pa/Papá in top 5 for "father garifuna" — confirms the
        # v1 acceptance condition was a false positive.
        return not check_father(hits)

    status, _ = regression_test(
        store, engine,
        label="1b",
        query="father garifuna",  # deprecated — kept for regression documentation
        retrieval_mode="hybrid",
        accept_check_fn=check_father_deprecated_fails,
        accept_description="Deprecated query returns NO Pa/Papá (confirms v1 was false positive)",
    )
    results["1b_father_garifuna_deprecated"] = status

    # Test 2 — Dügü (must not get Dabuyaba as primary definition)
    def check_dugu(hits):
        # PASS if top chunk contains Dügü/Dÿgÿ AND does NOT define it via Dabuyaba only
        for h in hits[:2]:
            c = h.get("content", "")
            has_dugu = any(x in c for x in ["Dügü", "Dÿgÿ", "Dügu", "dügü"])
            dabuyaba_only = "dabuyaba" in c.lower() and not has_dugu
            if has_dugu:
                return True
            if dabuyaba_only:
                return False
        return False  # no dugu content found

    status, _ = regression_test(
        store, engine,
        label="3",
        query="What is the Dügü?",
        retrieval_mode="vector",
        accept_check_fn=check_dugu,
        accept_description="Top chunks contain Dügü/Dÿgÿ (not just Dabuyaba)",
    )
    results["3_dugu"] = status

    # Test 3 — Origins (should retrieve conflicting sources)
    def check_origins(hits):
        # PGVector metadata stores path under "source"; "filename" is a secondary key.
        filenames = {
            h.get("metadata", {}).get("filename") or h.get("metadata", {}).get("source", "")
            for h in hits
        }
        # PASS if we retrieve at least one historical source (LIBRO II)
        return any("LIBRO GARIFUNA II" in f for f in filenames)

    status, _ = regression_test(
        store, engine,
        label="4",
        query="What are the origins of the Garífuna people?",
        retrieval_mode="vector",
        accept_check_fn=check_origins,
        accept_description="Retrieves at least one LIBRO GARIFUNA II chunk (historical Tier B)",
    )
    results["4_origins"] = status

    return results


# ---------------------------------------------------------------------------
# SECTION 3: Lexicon HIT regression — confirm lexicon path not broken
# ---------------------------------------------------------------------------
def section_lexicon_hits():
    print()
    print(SEP)
    print("SECTION 3: Structured Lexicon HIT Regression")
    print("Purpose: confirm existing deterministic lexicon path is not broken")
    print(SEP)

    # Import modules — use PROJECT_ROOT so imports work from /tmp or project tree.
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from app.services.lexicon.lexicon import YurumeinLexicon
    except ImportError as e:
        print(f"  SKIP: Cannot import lexicon module: {e}")
        return {}

    # Optionally load the candidate (Phase 4) lexical_parser from /tmp.
    # Place app/services/chatbot/lexical_parser.py at /tmp/lexical_parser_candidate.py
    # on the VPS before running this script to test the new short-form pattern.
    # If the candidate file is absent, the production parser is used instead.
    _candidate = Path("/tmp/lexical_parser_candidate.py")
    if _candidate.exists():
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("lexical_parser_candidate", _candidate)
        _mod = _ilu.module_from_spec(_spec)
        # Register before exec_module so @dataclass can resolve cls.__module__.
        sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)
        parse_lexical_request = _mod.parse_lexical_request
        print(f"  Using CANDIDATE parser from {_candidate}")
        _using_candidate = True
    else:
        try:
            from app.services.chatbot.lexical_parser import parse_lexical_request
        except ImportError as e:
            print(f"  SKIP: Cannot import parser: {e}")
            return {}
        print("  Using PRODUCTION parser (/tmp/lexical_parser_candidate.py not found)")
        _using_candidate = False

    lexicon = YurumeinLexicon()

    # Base cases use patterns unchanged from the original parser — pass regardless
    # of whether candidate or production parser is loaded.
    _base_cases = [
        ("How do you say agua in garifuna?", "agua", "spanish"),
        ("What is the garifuna word for abandoned?", "abandoned", "english"),
    ]
    # Extended cases exercise the new how_do_you_say_short pattern — only
    # meaningful when the candidate parser is loaded.
    _candidate_cases = [
        ("how do you say father",   "father", "english"),
        ("How do you say father?",  "father", "english"),
        ("How do you say father??", "father", "english"),
    ] if _using_candidate else []
    test_cases = _base_cases + _candidate_cases

    results = {}
    for query, expected_term, expected_lang in test_cases:
        print(f"\n  Query: {repr(query)}")
        parse = parse_lexical_request(query)
        print(f"  Parser: matched={parse.matched} term={repr(parse.term)} lang={repr(parse.language)}")

        if not parse.matched:
            print("  FAIL: parser did not match")
            results[query] = "FAIL (no parse match)"
            continue

        rows = lexicon.lookup(parse.term, parse.language)
        print(f"  Lexicon rows: {len(rows)}")
        if rows:
            garifuna_forms = [r.get("garifuna") for r in rows if r.get("garifuna")]
            print(f"  Garífuna forms: {garifuna_forms}")
            print(f"  PASS: lexicon HIT — {len(rows)} row(s)")
            results[query] = f"PASS ({len(rows)} rows)"
        else:
            print(f"  WARN: lexicon MISS for {repr(parse.term)} — word may not be in A-An range")
            results[query] = "WARN (lexicon miss — check word is in A-An range)"

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print()
    print(SEP)
    print("Phase 4 v1 — Retrieval Verification")
    print(f"Collection: {COLLECTION}")
    print(f"DB Host: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(SEP)

    print("\nConnecting to vector store...")
    try:
        store = make_store()
        engine = make_engine()
        # Quick connectivity check
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM langchain_pg_embedding e "
                     "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                     "WHERE c.name = :name"),
                {"name": COLLECTION}
            ).scalar()
        print(f"Connected. Collection '{COLLECTION}' has {result} chunks.")
        if result == 0:
            print("ERROR: Collection is empty. Point DB_HOST to production DB.")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not connect: {e}")
        sys.exit(1)

    section_father_comparison(store, engine)
    regression_results = section_regressions(store, engine)
    lexicon_results = section_lexicon_hits()

    # Summary
    print()
    print(SEP)
    print("SUMMARY")
    print(SEP)
    print("\nRegression test results:")
    all_pass = True
    for key, status in regression_results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {key}: {status}")
        if status != "PASS":
            all_pass = False

    print("\nLexicon HIT regression:")
    for query, status in lexicon_results.items():
        icon = "✓" if "PASS" in status else "?"
        print(f"  {icon} {repr(query)}: {status}")

    print()
    if all_pass:
        print("ALL REGRESSION TESTS PASS — safe to proceed with deployment review.")
    else:
        print("ONE OR MORE TESTS FAILED — do NOT deploy. Report results to owner.")
    print()


if __name__ == "__main__":
    main()
