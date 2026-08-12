import os
import re
import tempfile
import yaml

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from ..DB.session_manager import SessionManager
from ..prompt.prompt import system_prompt, user_prompt
from ..vectore_db.pgvector_db import VectorStoreManager
from .state import ChatState
from app.services.chatbot.lexical_parser import parse_lexical_request
from app.services.lexicon.lexicon import YurumeinLexicon

# ---------------------------------------------------------------------------
# Source Registry — loaded once at module import.
# Keyed by filename (matches metadata["filename"] stored in PGVector chunks).
# Returns {} on any failure so retrieval continues without registry info.
# ---------------------------------------------------------------------------
_REGISTRY_PATH = os.path.join(
    os.path.dirname(__file__), "..", "registry", "source_registry.yaml"
)


def _load_source_registry() -> dict:
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        registry = {s["filename"]: s for s in data.get("sources", [])}
        print(f"[source_registry] Loaded {len(registry)} entries.")
        return registry
    except Exception as exc:
        print(f"[source_registry] Failed to load ({exc}). Retrieval continues without registry.")
        return {}


_SOURCE_REGISTRY: dict = _load_source_registry()


def _registry_header(filename: str) -> str:
    """Return a compact authority annotation from the registry for a given filename."""
    reg = _SOURCE_REGISTRY.get(filename)
    if not reg:
        return f"Tier: ? | Canonical: (unregistered)"
    canonical = reg.get("canonical_title") or filename
    tier = reg.get("overall_tier", "?")
    verification = reg.get("verification_status", "")
    domain_authority = reg.get("domain_authority") or {}
    non_na_domains = {
        k: v for k, v in domain_authority.items()
        if v and str(v).upper() not in ("N/A", "NONE", "")
    }
    domain_str = (" | Domain: " + " ".join(f"{k}:{v}" for k, v in non_na_domains.items())
                  if non_na_domains else "")
    verification_str = f" | Verification: {verification}" if verification else ""
    return f"Tier: {tier}{verification_str} | Canonical: {canonical}{domain_str}"


class Nodes:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.6)
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
                    "hybrid_score": keyword_score * 0.3,
                    "retrieval_source": "keyword",
                    "initial_rank": rank,
                }

        ranked_results = sorted(
            merged.values(),
            key=lambda item: (item["hybrid_score"], item["keyword_score"]),
            reverse=True,
        )
        return ranked_results[:top_k]

    def _build_context(self, docs: list[dict]) -> str:
        parts = []
        for index, doc in enumerate(docs, start=1):
            metadata = doc.get("metadata") or {}
            filename = metadata.get("filename") or metadata.get("source") or "unknown"
            page_number = metadata.get("page_number", metadata.get("page"))
            page_label = f" | Page: {page_number}" if page_number is not None else ""
            retrieval_source = doc.get("retrieval_source", "vector")
            authority = _registry_header(filename)
            parts.append(
                f"[Doc {index} | File: {filename}{page_label}"
                f" | {authority}"
                f" | Retrieval: {retrieval_source} | Rank: {index}]\n"
                f"{doc.get('page_content', '')}"
            )
        return "\n\n".join(parts)

    def _build_lexicon_context(self, term: str, language: str, rows: list[dict]) -> str:
        forms = list(dict.fromkeys(r["garifuna"] for r in rows if r.get("garifuna")))
        if not forms:
            return ""
        forms_text = "\n".join(f"- {f}" for f in forms)
        return (
            "<LEXICON_CONTEXT>\n"
            f"Source: yurumein_lexicon\n"
            f"Query term: {term}\n"
            f"Source language: {language.capitalize()}\n"
            f"Documented Garífuna forms:\n{forms_text}\n"
            "</LEXICON_CONTEXT>"
        )

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
        """Retrieve context: lexicon-first for translation queries, RAG otherwise.

        Lexical-hit path: structured lexicon entry found → deterministic lexicon answer.
        Lexical-miss path (RP10): term recognised but not in lexicon → hybrid retrieval
          using the extracted term as the query signal (not the full conversational
          sentence). Context is prefixed with <LEXICON_MISS> so generation can
          distinguish "not in lexicon" from "term does not exist in Garífuna."
        Generic RAG path: non-lexical query → vector retrieval.
        """
        try:
            messages = state.get("messages", [])
            user_query = state.get("user_query", "")

            query = self._get_retrieval_query(messages, user_query)
            if not query:
                return {"context": "No query provided"}

            parse_result = parse_lexical_request(query)
            if parse_result.matched:
                rows = YurumeinLexicon().lookup(parse_result.term, parse_result.language)
                if rows:
                    # Lexical HIT — deterministic structured-lexicon answer.
                    lexicon_context = self._build_lexicon_context(
                        parse_result.term, parse_result.language, rows
                    )
                    if lexicon_context:
                        return {"context": lexicon_context}
                else:
                    # Lexical MISS (RP10): structured lexicon has no entry for this term.
                    # Use the extracted term alone as the retrieval signal — not the full
                    # conversational query and not "term garifuna".
                    # Rationale: "garifuna" is near-omnipresent in this corpus, so appending
                    # it boosts overview/history documents (which say "garifuna" hundreds of
                    # times) over vocabulary documents (which do not). The term alone produces
                    # a semantically targeted embedding that correctly surfaces vocabulary
                    # content in LIBRO SERIE I and DICCIONARIO chunks.
                    rag_query = parse_result.term
                    print(
                        f"[lexical_miss] term={parse_result.term!r} language={parse_result.language!r} "
                        f"rag_query={rag_query!r}"
                    )
                    retrieved_docs = self._hybrid_retrieve(query=rag_query, top_k=5)
                    miss_preamble = (
                        "<LEXICON_MISS>\n"
                        f"The Yurumein structured lexicon contains no entry for "
                        f"'{parse_result.term}' ({parse_result.language}). "
                        "Absence from the structured lexicon does NOT mean the term "
                        "has no Garífuna equivalent. "
                        "The following documentary evidence from approved language "
                        "sources was retrieved as secondary evidence.\n"
                        "</LEXICON_MISS>\n\n"
                    )
                    if retrieved_docs:
                        return {
                            "context": miss_preamble + self._build_context(retrieved_docs),
                            "retrieved_docs": retrieved_docs,
                        }
                    return {
                        "context": (
                            miss_preamble
                            + "No documentary evidence found in available sources."
                        ),
                    }

            retrieved_docs = self._vector_retrieve(query=query, top_k=5)
            if not retrieved_docs:
                return {"context": "No relevant documents found"}

            return {
                "context": self._build_context(retrieved_docs),
                "retrieved_docs": retrieved_docs,
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
            template = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", user_prompt)]
            )

            prompt = template.format_messages(
                chat_history=chat_history_formatted or "No previous chat history.",
                context=context,
                query=user_query,
            )
            response = await self.llm.ainvoke(prompt)
            return {"response": response.content}
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
