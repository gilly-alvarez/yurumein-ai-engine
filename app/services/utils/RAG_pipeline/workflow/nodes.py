import os
import re
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..DB.session_manager import SessionManager
from ..prompt.prompt import system_prompt, user_prompt
from ..vectore_db.pgvector_db import VectorStoreManager
from .state import ChatState
from app.services.chatbot.lexical_parser import parse_lexical_request, parse_definition_request, parse_page_request, parse_book_request
from app.services.chatbot.intent_router import Intent, detect_intent, preferred_roles_for
from app.services.chatbot.source_classifier import classify_source, source_display_name
from app.services.lexicon.lexicon import YurumeinLexicon

# Chunks shorter than this are likely blank worksheet lines or bare headers.
_MIN_CHUNK_CHARS = 80

# Boost multipliers applied by preferred-role rank after retrieval.
_BOOST_BY_RANK = [1.4, 1.25, 1.1]

# Minimum number of chunks from the top preferred source that must appear
# in the pool before we skip the targeted top-up search.
_MIN_PREFERRED_COVERAGE = 2

# Source-path ILIKE patterns used for targeted top-up retrieval.
# Keyed by the source role strings defined in source_classifier.py.
_ROLE_TO_SOURCE_PATTERN: dict[str, str] = {
    "FOUNDATIONAL_LANGUAGE_EDUCATION":        "%LIBRO SERIE I%",
    "ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT": "%LIBRO GARIFUNA II%",
    "DICTIONARY":                             "%DICCIONARIO GARIFUNA%",
}

# Bilingual retrieval expansion for lexicon misses.
# When the lexical parser extracts a term that returns no lexicon rows, these
# expansions run as supplementary RAG queries to surface the documented equivalent.
#
# SOURCE NOTES (must be preserved by any future editor):
#   "hello" / "greetings" →  "Saludos Mábugaü Mábuigaü"
#       Diccionario Garífuna p.174: "Saludos. Sus. com.  Mábugaü, Mábuigaü"
#       "Saludos" means greetings/salutations (a noun), not a conversational "hello."
#       The mapping surfaces the documented form; the LLM must note the distinction.
#       FLAG: Whether Mábugaü functions as a standalone greeting requires
#       native-speaker confirmation.
#
# NATIVE-SPEAKER REVIEW REQUIRED before any new entry is added.
_BILINGUAL_RETRIEVAL_EXPANSIONS: dict[str, str] = {
    "hello":     "Saludos Mábugaü Mábuigaü",
    "greetings": "Saludos Mábugaü Mábuigaü",
    "greeting":  "Saludos Mábugaü Mábuigaü",
    "hi":        "Saludos Mábugaü Mábuigaü",
}

# Contextual query expansions for non-lexical-parser phrases.
# Applied in the RAG path when the user query matches the pattern.
# Each entry: (compiled regex, supplementary retrieval query, source note)
#
# SOURCE NOTES:
#   introduce myself → "niribei Ká biri nombre"
#       Garífuna II p.108: "Ká biri?" = "¿Cuál es tú nombre?" (What is your name?)
#                          "Le niribei: Juan Martínez" = "Mi nombre es Juan Martínez"
#       This is a NAME-EXCHANGE Q&A used in everyday conversation —
#       NOT a formal self-introduction. The LLM must present it accurately.
#
#   Dügü / Chugú ceremony → "Gayunari Dabuyaba"
#       Root cause: Libro Serie I p.128 describes Dügü and Chugú together ("Dügü y/o Chugú")
#       and mentions "templo Dabuyaba" without distinguishing which ceremony uses which house.
#       This misleads the LLM into attributing Dabuyaba to Dügü.
#       Fix: supplement the pool with "Gayunari Dabuyaba" to surface:
#         - Diccionario p.43: "Casa en donde se realiza ceremonia ancestral. Dügÿ, Gayunari"
#                              "Casa en donde se realiza ceremonia ancestral. Chugu, Dabuyaba"
#         - Garífuna II p.105: same pairing
#       With these in context the LLM receives the definitive ceremony/building distinction.
#       SOURCE: Diccionario Garífuna p.43 and Libro Garífuna II p.105
_CONTEXTUAL_QUERY_EXPANSIONS: list[tuple] = [
    (
        re.compile(r'\bintroduce\s+(?:myself|yourself)\b|\bhow\s+(?:do\s+I|to)\s+(?:say|tell)\s+(?:someone\s+)?my\s+name\b', re.IGNORECASE),
        "niribei Ká biri nombre",
        "Garífuna II p.108: name-exchange Q&A",
    ),
    (
        re.compile(r'\b[Dd][üu]{1,2}[gG][üu]{1,2}\b|\b[Cc]hug[uú]\b|\bGayunari\b|\bDabuyaba\b', re.IGNORECASE),
        "Gayunari Dabuyaba",
        "Diccionario p.43 / Garífuna II p.105: Dügü→Gayunari, Chugú→Dabuyaba",
    ),
    (
        # Counting / numbers queries: surface Serie I p.8 number list.
        # SOURCE: Libro Serie I p.8 — Aban=Uno(1), Biáma=Dos(2), Ürüwa=Tres(3)
        re.compile(r'\b(?:count|number[s]?|numeros?|contar)\b', re.IGNORECASE),
        "Aban biama urüwa",
        "Serie I p.8/p.45: Aban=1 (Uno), Biáma=2 (Dos), Ürüwa=3 (Tres)",
    ),
]


class Nodes:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self._vector_store = None

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = VectorStoreManager()
        return self._vector_store

    def _get_retrieval_query(self, messages, user_query: str) -> str:
        if user_query:
            return user_query.strip()

        for message in reversed(messages or []):
            if isinstance(message, dict):
                content = message.get("content", "")
            elif isinstance(message, BaseMessage):
                content = message.content
            else:
                content = str(message)

            if content:
                return content.strip()

        return ""

    # Signals that a query is a contextual follow-up and needs topic enrichment.
    # Includes English and Spanish referential terms and continuation phrases.
    _FOLLOWUP_SIGNALS = re.compile(
        r"\b(they|them|their|theirs|it\b|its\b|this|that|these|those|he|she|him|her"
        r"|tell\s+me\s+more|more\s+about|what\s+happened|and\s+(?:what|where|when|how|why)"
        r"|eso|esto|ese|esa|esos|esas|ello|ellos|ellas"
        r"|h[aá]blame|cu[eé]ntame|dime\s+m[aá]s|m[aá]s\s+sobre|expl[ií]came)\b",
        re.IGNORECASE,
    )

    def _contextualize_query(self, query: str, messages: list[dict]) -> str:
        """
        When a query contains genuine follow-up signals, prepend the previous user
        query as the topic anchor so retrieval surfaces the same document neighborhood.

        Gated strictly on follow-up signals — short questions without referential
        language are left unchanged to avoid contaminating unrelated queries with
        a prior topic.

        Anchor strategy: use the previous user/human query (most reliable — it
        contains the original topic in the user's own retrieval-friendly language).
        Falls back to the last assistant reply, stripped of markdown noise, only
        when no prior user query is available.
        """
        if not messages:
            return query
        # Only enrich when genuine follow-up signals are present.
        # Do NOT enrich based on query length alone.
        if not self._FOLLOWUP_SIGNALS.search(query):
            return query

        # Primary: use the last user/human query in history as topic anchor.
        for msg in reversed(messages):
            if msg.get("role") in ("user", "human"):
                content = (msg.get("content") or "").strip()
                if content and len(content.split()) >= 3:
                    return f"{content[:200]} {query}"

        # Fallback: last assistant reply, stripped of citation/markdown noise.
        last_ai = None
        for msg in reversed(messages):
            if msg.get("role") in ("assistant", "ai"):
                last_ai = (msg.get("content") or "").strip()
                break
        if not last_ai:
            return query
        clean = re.sub(r"\*\*Source:.*", "", last_ai, flags=re.IGNORECASE | re.DOTALL)
        clean = re.sub(r"\[(?:Source|EID)[^\]]*\]", "", clean)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean) if len(s.strip()) >= 20]
        if not sentences or sentences[0].startswith(("**", "[")):
            return query
        return f"{sentences[0][:150]} {query}"

    def _keyword_overlap_score(self, query: str, text: str) -> float:
        query_terms = {
            term for term in re.findall(r"\b\w+\b", query.lower()) if len(term) > 2
        }
        if not query_terms:
            return 0.0

        text_terms = set(re.findall(r"\b\w+\b", text.lower()))
        return len(query_terms.intersection(text_terms)) / len(query_terms)

    def _doc_key(self, content: str, metadata: dict) -> tuple:
        return (
            metadata.get("source"),
            metadata.get("page_number", metadata.get("page")),
            content[:120],
        )

    def _hybrid_retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        vector_hits = self.vector_store.similarity_search_with_score(
            query=query,
            k=max(top_k * 3, 10),
        )
        keyword_hits = self.vector_store.keyword_search(
            query=query,
            k=max(top_k * 3, 10),
        )

        merged: dict[tuple, dict] = {}
        query_term_count = max(
            len({term for term in re.findall(r"\b\w+\b", query.lower()) if len(term) > 2}),
            1,
        )

        for rank, (doc, vector_score) in enumerate(vector_hits, start=1):
            metadata = dict(doc.metadata or {})
            content = (doc.page_content or "").strip()
            if not content:
                continue

            key = self._doc_key(content, metadata)
            semantic_score = 1.0 / (1.0 + max(float(vector_score), 0.0))
            keyword_score = self._keyword_overlap_score(query, content)
            merged[key] = {
                "page_content": content,
                "metadata": metadata,
                "vector_score": float(vector_score),
                "keyword_score": keyword_score,
                "hybrid_score": (semantic_score * 0.7) + (keyword_score * 0.3),
                "retrieval_source": "vector",
                "initial_rank": rank,
            }

        for rank, hit in enumerate(keyword_hits, start=1):
            metadata = dict(hit.get("metadata") or {})
            content = (hit.get("document") or "").strip()
            if not content:
                continue

            key = self._doc_key(content, metadata)
            keyword_score = max(
                float(hit.get("keyword_score", 0.0)) / query_term_count,
                self._keyword_overlap_score(query, content),
            )

            if key in merged:
                merged[key]["keyword_score"] = max(merged[key]["keyword_score"], keyword_score)
                semantic_score = 1.0 / (
                    1.0 + max(float(merged[key]["vector_score"] or 0.0), 0.0)
                )
                merged[key]["hybrid_score"] = max(
                    merged[key]["hybrid_score"],
                    (semantic_score * 0.7) + (keyword_score * 0.3),
                )
                merged[key]["retrieval_source"] = "hybrid"
            else:
                merged[key] = {
                    "page_content": content,
                    "metadata": metadata,
                    "vector_score": None,
                    "keyword_score": keyword_score,
                    "hybrid_score": keyword_score * 0.6,
                    "retrieval_source": "keyword",
                    "initial_rank": rank,
                }

        ranked_results = sorted(
            merged.values(),
            key=lambda item: (item["hybrid_score"], item["keyword_score"]),
            reverse=True,
        )
        return ranked_results[:top_k]

    def _vector_retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        docs = self.vector_store.similarity_search(query=query, k=top_k)
        results = []
        for index, doc in enumerate(docs, start=1):
            results.append(
                {
                    "page_content": (doc.page_content or "").strip(),
                    "metadata": dict(doc.metadata or {}),
                    "vector_score": None,
                    "keyword_score": 0.0,
                    "hybrid_score": 0.0,
                    "retrieval_source": "vector",
                    "initial_rank": index,
                }
            )
        return [item for item in results if item["page_content"]]

    def _topup_from_preferred_source(
        self,
        query: str,
        pool: list[dict],
        preferred_roles: list[str],
    ) -> list[dict]:
        """
        Guarantee that the pool contains at least _MIN_PREFERRED_COVERAGE chunks
        from the top preferred source role.

        If the pool already has enough coverage, returns it unchanged.
        Otherwise runs a targeted semantic search scoped to that source and
        merges the new chunks in (deduplicating by doc key).
        """
        if not preferred_roles:
            return pool

        # Count pool coverage per role.
        coverage: dict[str, int] = {}
        existing_keys = set()
        for doc in pool:
            source = (doc.get("metadata") or {}).get("source", "")
            meta = classify_source(source)
            if meta:
                coverage[meta.role] = coverage.get(meta.role, 0) + 1
            existing_keys.add(self._doc_key(
                doc.get("page_content", ""), doc.get("metadata") or {}
            ))

        # Top-up for each preferred role that is underrepresented.
        additional: list[dict] = []
        for role in preferred_roles[:2]:  # check top-2 preferred roles
            if coverage.get(role, 0) >= _MIN_PREFERRED_COVERAGE:
                continue

            source_pattern = _ROLE_TO_SOURCE_PATTERN.get(role)
            if not source_pattern:
                continue

            targeted = self.vector_store.source_targeted_search(
                query=query,
                source_pattern=source_pattern,
                k=5,
            )
            for hit in targeted:
                content = hit.get("page_content", "").strip()
                metadata = hit.get("metadata") or {}
                if not content:
                    continue
                key = self._doc_key(content, metadata)
                if key in existing_keys:
                    continue

                # Convert distance (lower = better) to a hybrid score
                # comparable to the existing pool items.
                dist = hit.get("distance", 1.0)
                semantic_score = 1.0 / (1.0 + max(dist, 0.0))
                additional.append({
                    "page_content": content,
                    "metadata": metadata,
                    "vector_score": dist,
                    "keyword_score": 0.0,
                    "hybrid_score": semantic_score * 0.7,
                    "retrieval_source": "targeted",
                    "initial_rank": len(pool) + len(additional) + 1,
                })
                existing_keys.add(key)

        return pool + additional

    def _filter_low_info(self, docs: list[dict]) -> list[dict]:
        """Deprioritize chunks that are likely worksheet noise (too short)."""
        meaningful = [d for d in docs if len(d.get("page_content", "")) >= _MIN_CHUNK_CHARS]
        return meaningful if meaningful else docs

    def _boost_by_source_role(self, docs: list[dict], preferred_roles: list[str]) -> list[dict]:
        """
        Re-rank docs by applying a multiplier to chunks from preferred sources.
        First preferred role gets the highest boost; subsequent roles get
        progressively smaller boosts. Unpreferred sources are not penalized.
        """
        role_boost: dict[str, float] = {}
        for i, role in enumerate(preferred_roles):
            boost = _BOOST_BY_RANK[i] if i < len(_BOOST_BY_RANK) else 1.05
            role_boost[role] = boost

        def effective_score(doc: dict) -> float:
            base = doc.get("hybrid_score") or 0.0
            source_path = (doc.get("metadata") or {}).get("source", "")
            meta = classify_source(source_path)
            if meta and meta.role in role_boost:
                return base * role_boost[meta.role]
            return base

        return sorted(docs, key=effective_score, reverse=True)

    _EID_ABBREV = {
        "FOUNDATIONAL_LANGUAGE_EDUCATION":        "SERIE_I",
        "ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT": "GAR_II",
        "DICTIONARY":                             "DICC",
        "STRUCTURED_LEXICON":                     "LEXICON",
    }

    def _build_context(self, docs: list[dict]) -> str:
        parts = []
        for index, doc in enumerate(docs, start=1):
            metadata = doc.get("metadata") or {}
            source_citation = source_display_name(metadata)
            retrieval_source = doc.get("retrieval_source", "vector")
            source_path = metadata.get("source", "")
            page_label = metadata.get("page_label", "")
            meta = classify_source(source_path)
            abbrev = self._EID_ABBREV.get(meta.role, "DOC") if meta else "DOC"
            eid = f"EID-{index}:{abbrev}:p{page_label}" if page_label else f"EID-{index}:{abbrev}"
            parts.append(
                f"[{eid} | Source: {source_citation} | Retrieval: {retrieval_source}]\n"
                f"{doc.get('page_content', '')}"
            )
        return "\n\n".join(parts)

    def _build_lexicon_miss_context(self, term: str, language: str) -> str:
        """Inject a deterministic note when the lexicon was searched but returned nothing.

        This signals the LLM that it must not substitute a general-knowledge phrase.
        The note is prepended to whatever RAG context follows.
        """
        return (
            "<LEXICON_NOTE>\n"
            f"The term '{term}' ({language}) was searched in the verified Garífuna lexicon "
            f"(yurumein_lexicon) but no entry was found. "
            f"If the retrieved documents below contain a documented Garífuna equivalent, "
            f"cite that source and note any distinction between the documented meaning and "
            f"the user's exact term (e.g., 'greetings/salutations' vs. a conversational greeting). "
            f"Do not supply a translation from general knowledge or community expressions — "
            f"these are not verified by the approved sources.\n"
            "</LEXICON_NOTE>"
        )

    def _build_lexicon_context(self, term: str, language: str, rows: list[dict]) -> str:
        forms = list(dict.fromkeys(r["garifuna"] for r in rows if r.get("garifuna")))
        if not forms:
            return ""
        forms_text = "\n".join(f"- {f}" for f in forms)
        return (
            "<LEXICON_CONTEXT>\n"
            f"Source: Yurumein Lexicon\n"
            f"Query term: {term}\n"
            f"Source language: {language.capitalize()}\n"
            f"Documented Garífuna forms:\n{forms_text}\n"
            "</LEXICON_CONTEXT>"
        )

    def _build_definition_context(self, term: str, rows: list[dict]) -> str:
        """Build lexicon context for a Garífuna → EN/ES reverse lookup."""
        if not rows:
            return ""
        parts = []
        for row in rows:
            line = [f"Garífuna: {row['garifuna']}"]
            if row.get("english"):
                line.append(f"English: {row['english']}")
            if row.get("spanish"):
                line.append(f"Spanish: {row['spanish']}")
            parts.append(" | ".join(line))
        forms_text = "\n".join(f"- {p}" for p in parts)
        return (
            "<LEXICON_CONTEXT>\n"
            f"Source: Yurumein Lexicon\n"
            f"Query: {term}\n"
            f"Documented entries:\n{forms_text}\n"
            "</LEXICON_CONTEXT>"
        )

    def _build_page_context(
        self, display_name: str, pages: tuple, chunks: list[dict]
    ) -> str:
        parts = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata") or {}
            page_label = metadata.get("page_label", "?")
            parts.append(
                f"[Source: {display_name}, Page {page_label} | Rank: {index} | Retrieval: page_lookup]\n"
                f"{chunk.get('page_content', '')}"
            )
        return "\n\n".join(parts)

    def _build_page_not_found_context(
        self, display_name: str, pages: tuple
    ) -> str:
        pages_str = ", ".join(str(p) for p in pages)
        return (
            f"[Source: {display_name} | Retrieval: page_lookup]\n"
            f"NOTE: The requested page(s) {pages_str} of {display_name} were not found "
            f"in the indexed document collection. Those specific page numbers are not "
            f"present. Inform the user accurately that those pages are not available."
        )

    def _source_constrained_hybrid_retrieve(
        self, query: str, source_pattern: str, top_k: int = 10
    ) -> list[dict]:
        """Hybrid retrieval restricted to a single source document."""
        vector_hits = self.vector_store.source_targeted_search(
            query=query, source_pattern=source_pattern, k=max(top_k * 2, 10)
        )
        keyword_hits = self.vector_store.source_keyword_search(
            query=query, source_pattern=source_pattern, k=max(top_k * 2, 10)
        )

        query_term_count = max(
            len({t for t in re.findall(r"\b\w+\b", query.lower()) if len(t) > 2}), 1
        )
        merged: dict[tuple, dict] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            metadata = dict(hit.get("metadata") or {})
            content = (hit.get("page_content") or "").strip()
            if not content:
                continue
            dist = float(hit.get("distance", 1.0))
            semantic_score = 1.0 / (1.0 + max(dist, 0.0))
            keyword_score = self._keyword_overlap_score(query, content)
            key = self._doc_key(content, metadata)
            merged[key] = {
                "page_content": content,
                "metadata": metadata,
                "vector_score": dist,
                "keyword_score": keyword_score,
                "hybrid_score": (semantic_score * 0.7) + (keyword_score * 0.3),
                "retrieval_source": "targeted",
                "initial_rank": rank,
            }

        for rank, hit in enumerate(keyword_hits, start=1):
            metadata = dict(hit.get("metadata") or {})
            content = (hit.get("document") or "").strip()
            if not content:
                continue
            keyword_score = max(
                float(hit.get("keyword_score", 0.0)) / query_term_count,
                self._keyword_overlap_score(query, content),
            )
            key = self._doc_key(content, metadata)
            if key in merged:
                merged[key]["keyword_score"] = max(merged[key]["keyword_score"], keyword_score)
                sem = 1.0 / (1.0 + max(float(merged[key].get("vector_score") or 0.0), 0.0))
                merged[key]["hybrid_score"] = max(
                    merged[key]["hybrid_score"], (sem * 0.7) + (keyword_score * 0.3)
                )
                merged[key]["retrieval_source"] = "hybrid"
            else:
                merged[key] = {
                    "page_content": content,
                    "metadata": metadata,
                    "vector_score": None,
                    "keyword_score": keyword_score,
                    "hybrid_score": keyword_score * 0.6,
                    "retrieval_source": "keyword",
                    "initial_rank": rank,
                }

        return sorted(merged.values(), key=lambda d: d["hybrid_score"], reverse=True)[:top_k]

    # Pattern: "... and give/show/list/provide ..." splits into two sub-queries.
    _MULTI_PART_RE = re.compile(
        r"\band\s+(?:also\s+)?(?:give|show|list|provide|include)\b",
        re.IGNORECASE,
    )

    def _decompose_multi_part_query(self, query: str) -> list[str]:
        """
        Split a multi-part query into up to 2 sub-queries for separate retrieval.

        Only splits on "... and give/show/list/provide ..." — conservative enough
        to avoid splitting natural-language compound questions.
        """
        parts = self._MULTI_PART_RE.split(query, maxsplit=1)
        if len(parts) == 2 and len(parts[0].strip()) > 15 and len(parts[1].strip()) > 10:
            return [parts[0].strip(), parts[1].strip()]
        return [query]

    def _validate_citations(self, response: str, retrieved_docs: list[dict]) -> str:
        """
        Post-generate validation of every **Source:** and [Source: ...] citation.

        Builds an exact lookup of (book_display_lower → {page_label}) from retrieved_docs,
        then checks every citation in the response against that set. Citations that cannot
        be verified — wrong book, wrong page, or book not retrieved at all — are replaced
        in-place with [UNVERIFIED: ...] so they are never presented as confirmed sources.
        """
        if not retrieved_docs:
            return response

        # Build lookup: book_display_lower → {page_label_str, ...}
        retrieved_pages: dict[str, set[str]] = {}
        for doc in retrieved_docs:
            sm = classify_source((doc.get("metadata") or {}).get("source", ""))
            if sm:
                page = str((doc.get("metadata") or {}).get("page_label", "") or "")
                retrieved_pages.setdefault(sm.display.lower(), set()).add(page)

        def _claim_valid(claim: str) -> bool:
            claim_l = claim.strip().lower()
            matched_book = next((b for b in retrieved_pages if b in claim_l), None)
            if not matched_book:
                return False
            claimed_pages = re.findall(r"\b(\d+)\b", claim_l)
            if not claimed_pages:
                return True  # book-level citation, no page demanded — accept
            return any(p in retrieved_pages[matched_book] for p in claimed_pages)

        def _fix_bold(m: re.Match) -> str:
            claim = m.group(1).strip()
            if _claim_valid(claim):
                return m.group(0)
            return f"**Source:** [UNVERIFIED: {claim}]"

        def _fix_bracket(m: re.Match) -> str:
            claim = m.group(1).strip()
            if _claim_valid(claim):
                return m.group(0)
            return f"[UNVERIFIED: {claim}]"

        result = re.sub(r"\*\*Source:\*\*\s*([^\n*]+)", _fix_bold, response, flags=re.IGNORECASE)
        result = re.sub(r"\[Source:\s*([^\]]+)\]", _fix_bracket, result, flags=re.IGNORECASE)
        return result

    async def decide_retrieval_path(self, state: ChatState) -> dict:
        if state.get("file"):
            return {"next_node": "file_process"}
        return {"next_node": "global_retrieval_process"}

    async def file_process(self, state: ChatState) -> dict:
        file = state.get("file")
        if not file:
            return {"context": "No file provided"}

        try:
            name = file.filename.lower()

            if name.endswith(".pdf"):
                file_bytes = await file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                try:
                    loader = PyPDFLoader(tmp_path)
                    pages = loader.load()
                    text = "\n".join([p.page_content or "" for p in pages])
                    return {"context": text if text.strip() else "PDF appears empty"}
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            if name.endswith(".txt"):
                file_bytes = await file.read()
                text = file_bytes.decode("utf-8", errors="ignore")
                return {"context": text if text.strip() else "Text file appears empty"}

            return {"context": "Unsupported file format"}
        except Exception as e:
            print(f"File processing error: {e}")
            return {"context": f"Error processing file: {str(e)}"}

    async def global_retrieval_process(self, state: ChatState) -> dict:
        """
        Source-aware retrieval.

        Priority order:
          1. Exact translation match  → structured lexicon (direct, no RAG)
          2. Definition match         → lexicon reverse lookup, then RAG
          3. All other queries        → hybrid RAG with:
             a. Broad hybrid retrieve (pool of 10)
             b. Targeted top-up for preferred sources when underrepresented
             c. Low-info chunk filter
             d. Source-role score boosting
        """
        try:
            messages = state.get("messages", [])
            user_query = state.get("user_query", "")

            query = self._get_retrieval_query(messages, user_query)
            if not query:
                return {"context": "No query provided"}

            # Build a context-enriched retrieval query for follow-up turns.
            # `query` is preserved for intent routing and all parsers; only
            # the main hybrid search and top-up use the enriched form so
            # specialised paths (lexical, page, book) are unaffected.
            retrieval_query = self._contextualize_query(query, messages)

            # Track lexicon-miss state for downstream LEXICON_NOTE injection.
            _lexicon_miss_term: str | None = None
            _lexicon_miss_language: str | None = None
            _extra_expansion_queries: list[str] = []

            # 1. Translation request: exact pattern → lexicon, skip RAG.
            # retrieved_docs is explicitly cleared so stale docs from a prior turn
            # do not cause _validate_citations to tag the lexicon citation as UNVERIFIED.
            parse_result = parse_lexical_request(query)
            if parse_result.matched:
                rows = YurumeinLexicon().lookup(parse_result.term, parse_result.language)
                if rows:
                    ctx = self._build_lexicon_context(parse_result.term, parse_result.language, rows)
                    if ctx:
                        return {"context": ctx, "retrieved_docs": []}
                else:
                    # Lexicon miss: record for LEXICON_NOTE and check bilingual expansion.
                    _lexicon_miss_term = parse_result.term
                    _lexicon_miss_language = parse_result.language
                    expansion = _BILINGUAL_RETRIEVAL_EXPANSIONS.get(parse_result.term.strip().lower())
                    if expansion:
                        _extra_expansion_queries.append(expansion)

            # 2. Definition request: attempt reverse lexicon lookup first.
            def_result = parse_definition_request(query)
            if def_result.matched:
                rows = YurumeinLexicon().reverse_lookup(def_result.term)
                if rows:
                    ctx = self._build_definition_context(def_result.term, rows)
                    if ctx:
                        return {"context": ctx, "retrieved_docs": []}

            # 2.5. Explicit page request: deterministic metadata-filtered lookup.
            page_result = parse_page_request(query)
            if page_result.matched:
                chunks = self.vector_store.page_search(
                    page_result.source_pattern, page_result.pages
                )
                if chunks:
                    return {
                        "context": self._build_page_context(
                            page_result.display_name, page_result.pages, chunks
                        ),
                        "retrieved_docs": chunks,
                    }
                return {
                    "context": self._build_page_not_found_context(
                        page_result.display_name, page_result.pages
                    ),
                }

            # 2.6. Source-constrained retrieval: user named a book but no specific pages.
            # parse_page_request (2.5) has already returned if pages were specified;
            # so if we reach here, book_result matches a name-only reference.
            book_result = parse_book_request(query)
            if book_result.matched:
                sub_queries = self._decompose_multi_part_query(query)
                constrained: dict[tuple, dict] = {}
                for sub_q in sub_queries:
                    for chunk in self._source_constrained_hybrid_retrieve(
                        sub_q, book_result.source_pattern, top_k=8
                    ):
                        key = self._doc_key(chunk.get("page_content", ""), chunk.get("metadata") or {})
                        if key not in constrained or chunk.get("hybrid_score", 0) > constrained[key].get("hybrid_score", 0):
                            constrained[key] = chunk
                # Apply contextual expansion within the constrained source pool.
                # Runs the same expansions as path 2.7 but keyword-scoped to the named
                # source, so "According to Libro X, what is Duguu?" surfaces p.105.
                for _pattern, _exp_q, _note in _CONTEXTUAL_QUERY_EXPANSIONS:
                    if _pattern.search(query):
                        kw_hits = self.vector_store.source_keyword_search(
                            _exp_q, book_result.source_pattern, k=5
                        )
                        for hit in kw_hits:
                            content = (hit.get("document") or "").strip()
                            if not content:
                                continue
                            metadata = dict(hit.get("metadata") or {})
                            keyword_score = float(hit.get("keyword_score", 0.0))
                            key = self._doc_key(content, metadata)
                            if key not in constrained:
                                constrained[key] = {
                                    "page_content": content,
                                    "metadata": metadata,
                                    "vector_score": None,
                                    "keyword_score": keyword_score,
                                    "hybrid_score": keyword_score * 0.6,
                                    "retrieval_source": "keyword_expansion",
                                    "initial_rank": len(constrained) + 1,
                                }
                        break

                if constrained:
                    sorted_constrained = sorted(constrained.values(), key=lambda d: d.get("hybrid_score", 0), reverse=True)
                    final_constrained = self._filter_low_info(sorted_constrained)[:5]
                    return {
                        "context": self._build_context(final_constrained),
                        "retrieved_docs": final_constrained,
                    }
                # No results from the named source → fall through to general retrieval.

            # 2.7. Contextual query expansion: for specific phrase patterns whose
            # natural-language form does not match the indexed vocabulary, supplement
            # the pool with a targeted query that surfaces the documented content.
            # Expansions are source-verified (see _CONTEXTUAL_QUERY_EXPANSIONS).
            for _pattern, _exp_q, _note in _CONTEXTUAL_QUERY_EXPANSIONS:
                if _pattern.search(query):
                    _extra_expansion_queries.append(_exp_q)
                    break  # one expansion per query is sufficient

            # 3. Detect intent and select preferred source roles.
            intent = detect_intent(query)
            preferred = preferred_roles_for(intent)

            # 4. Broad hybrid retrieval.
            pool = self._hybrid_retrieve(query=retrieval_query, top_k=10)
            if not pool:
                return {"context": "No relevant documents found"}

            # 4a. Run any supplementary expansion queries and merge into pool.
            # Expansion queries are keyword-tuned phrases crafted to surface specific
            # documented content. We use keyword_search (not hybrid) so that the
            # exact Garífuna vocabulary in the expansion phrase drives retrieval
            # rather than semantic similarity to unrelated pages.
            # Track expansion doc keys separately so we can guarantee their presence
            # in final_docs — without that guarantee, the citation validator may tag
            # citations from these docs as UNVERIFIED even when the LLM correctly cites them.
            pool_keys = {self._doc_key(d.get("page_content", ""), d.get("metadata") or {}) for d in pool}
            _expansion_doc_keys: set[tuple] = set()
            for exp_q in _extra_expansion_queries:
                kw_hits = self.vector_store.keyword_search(exp_q, k=5)
                for hit in kw_hits:
                    content = (hit.get("document") or "").strip()
                    if not content:
                        continue
                    metadata = dict(hit.get("metadata") or {})
                    keyword_score = float(hit.get("keyword_score", 0.0))
                    doc = {
                        "page_content": content,
                        "metadata": metadata,
                        "vector_score": None,
                        "keyword_score": keyword_score,
                        "hybrid_score": keyword_score * 0.6,
                        "retrieval_source": "keyword_expansion",
                    }
                    key = self._doc_key(content, metadata)
                    if key not in pool_keys:
                        pool.append(doc)
                        pool_keys.add(key)
                    _expansion_doc_keys.add(key)

            # 4b. Multi-part query: also retrieve for the second sub-query so each
            #     distinct topic gets its own evidence in the pool.
            sub_queries = self._decompose_multi_part_query(query)
            if len(sub_queries) > 1:
                for sub_q in sub_queries[1:]:
                    sub_pool = self._hybrid_retrieve(query=sub_q, top_k=5)
                    for doc in sub_pool:
                        key = self._doc_key(doc.get("page_content", ""), doc.get("metadata") or {})
                        if not any(
                            self._doc_key(p.get("page_content", ""), p.get("metadata") or {}) == key
                            for p in pool
                        ):
                            pool.append(doc)

            # 5. Guarantee preferred-source coverage via targeted top-up.
            #    This ensures content from the right book reaches the pool
            #    even when it doesn't rank in the global top-10.
            if preferred:
                pool = self._topup_from_preferred_source(retrieval_query, pool, preferred)

            # 6. Filter low-information chunks.
            filtered = self._filter_low_info(pool)

            # 7. Boost and rank; take final top-5.
            if preferred:
                ranked = self._boost_by_source_role(filtered, preferred)
            else:
                ranked = filtered

            final_docs = ranked[:5]

            # 7b. Guarantee at least one chunk from the top preferred role is present.
            # This ensures that intent-routed queries (e.g. cultural → Garífuna II)
            # always surface book-specific context even when Diccionario terms score
            # higher in the global hybrid pool.
            if preferred and final_docs:
                top_role = preferred[0]
                has_preferred = any(
                    (m := classify_source((d.get("metadata") or {}).get("source", "")))
                    and m.role == top_role
                    for d in final_docs
                )
                if not has_preferred:
                    for doc in ranked[5:]:
                        m = classify_source((doc.get("metadata") or {}).get("source", ""))
                        if m and m.role == top_role:
                            final_docs = final_docs[:-1] + [doc]
                            break

            # 7c. Guarantee expansion docs appear in final_docs.
            # Expansion docs contain the evidence the LLM will cite; if they fall
            # outside the top-5 by score (or were filtered by _filter_low_info),
            # the citation validator will flag citations from them as UNVERIFIED.
            # We search the full pool (not just ranked) so filtered docs are still
            # eligible for injection.
            if _expansion_doc_keys and final_docs:
                final_keys = {self._doc_key(d.get("page_content", ""), d.get("metadata") or {}) for d in final_docs}
                missing_expansions = [
                    d for d in pool
                    if self._doc_key(d.get("page_content", ""), d.get("metadata") or {}) in _expansion_doc_keys
                    and self._doc_key(d.get("page_content", ""), d.get("metadata") or {}) not in final_keys
                ]
                if missing_expansions:
                    # Inject the best-scored missing expansion doc, replacing the last final doc.
                    best_exp = max(missing_expansions, key=lambda d: d.get("hybrid_score", 0))
                    final_docs = final_docs[:-1] + [best_exp]

            rag_context = self._build_context(final_docs)

            # Prepend LEXICON_NOTE when the lexical path fired but returned no rows.
            # This deterministic signal prevents the LLM from substituting a
            # general-knowledge translation for a term not in the approved sources.
            if _lexicon_miss_term:
                note = self._build_lexicon_miss_context(_lexicon_miss_term, _lexicon_miss_language or "")
                rag_context = note + "\n\n" + rag_context

            return {
                "context": rag_context,
                "retrieved_docs": final_docs,
            }
        except Exception as e:
            print(f"Retrieval error: {e}")
            return {"context": f"Error retrieving documents: {str(e)}"}

    async def trim_chat_history(self, state: ChatState) -> dict:
        """Get chat history and return messages."""
        try:
            session_id = state.get("thread_id")
            manager = SessionManager()
            user_id = state.get("user_id")
            print(session_id, user_id)

            if not session_id or not user_id:
                return {"messages": []}

            messages = await manager.get_recent_session_history(session_id, user_id, limit=30)
            print(messages)
            return {"messages": messages}
        except Exception as e:
            print(f"Error in trim_chat_history: {e}")
            return {"messages": []}

    async def llm_call(self, state: ChatState) -> dict:
        """Generate response with retrieved context and chat history."""
        try:
            context = state.get("context", "No additional context available.")
            user_query = state.get("user_query", "")
            messages = state.get("messages", [])

            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    formatted_messages.append(f"{msg.get('role', '')} : {msg.get('content', '')}")

            chat_history_formatted = "\n".join(formatted_messages)

            _lang_map = {"en": "English", "es": "Spanish"}
            _lang_code = (state.get("language") or "").strip().lower()
            _response_lang = _lang_map.get(_lang_code)
            effective_system = system_prompt
            if _response_lang:
                effective_system = system_prompt + (
                    f"\n\n### Response Language\n"
                    f"Respond in {_response_lang}. "
                    f"If the user explicitly asks for a response in a different language in their current message, honor that request. "
                    f"Always preserve Garífuna words exactly as documented — do not translate them unless asked."
                )

            template = ChatPromptTemplate.from_messages(
                [("system", effective_system), ("human", user_prompt)]
            )

            prompt = template.format_messages(
                chat_history=chat_history_formatted or "No previous chat history.",
                context=context,
                query=user_query,
            )
            response = await self.llm.ainvoke(prompt)
            validated = self._validate_citations(
                response.content, state.get("retrieved_docs") or []
            )
            return {"response": validated}
        except Exception as e:
            print(f"LLM call error: {e}")
            return {"response": f"Error generating response: {str(e)}"}

    async def generate_session_title(self, first_message: str) -> str:
        """Generate session title."""
        try:
            prompt = (
                f"Generate a short, descriptive title (max 6 words) for: "
                f"'{first_message}'. Return only the title."
            )
            messages = [HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            title = response.content.strip().strip('"').strip("'")
            return title[:100]
        except Exception as e:
            print(f"Error generating title: {e}")
            return first_message[:50] + "..." if len(first_message) > 50 else first_message
