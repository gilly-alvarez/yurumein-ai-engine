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
from app.services.chatbot.lexical_parser import parse_lexical_request, parse_definition_request
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

    def _build_context(self, docs: list[dict]) -> str:
        parts = []
        for index, doc in enumerate(docs, start=1):
            metadata = doc.get("metadata") or {}
            source_citation = source_display_name(metadata)
            retrieval_source = doc.get("retrieval_source", "vector")
            parts.append(
                f"[Source: {source_citation} | Rank: {index} | Retrieval: {retrieval_source}]\n"
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
            f"Source: yurumein_lexicon\n"
            f"Query: {term}\n"
            f"Documented entries:\n{forms_text}\n"
            "</LEXICON_CONTEXT>"
        )

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

            # 1. Translation request: exact pattern → lexicon, skip RAG.
            parse_result = parse_lexical_request(query)
            if parse_result.matched:
                rows = YurumeinLexicon().lookup(parse_result.term, parse_result.language)
                if rows:
                    ctx = self._build_lexicon_context(parse_result.term, parse_result.language, rows)
                    if ctx:
                        return {"context": ctx}

            # 2. Definition request: attempt reverse lexicon lookup first.
            def_result = parse_definition_request(query)
            if def_result.matched:
                rows = YurumeinLexicon().reverse_lookup(def_result.term)
                if rows:
                    ctx = self._build_definition_context(def_result.term, rows)
                    if ctx:
                        return {"context": ctx}

            # 3. Detect intent and select preferred source roles.
            intent = detect_intent(query)
            preferred = preferred_roles_for(intent)

            # 4. Broad hybrid retrieval.
            pool = self._hybrid_retrieve(query=query, top_k=10)
            if not pool:
                return {"context": "No relevant documents found"}

            # 5. Guarantee preferred-source coverage via targeted top-up.
            #    This ensures content from the right book reaches the pool
            #    even when it doesn't rank in the global top-10.
            if preferred:
                pool = self._topup_from_preferred_source(query, pool, preferred)

            # 6. Filter low-information chunks.
            filtered = self._filter_low_info(pool)

            # 7. Boost and rank; take final top-5.
            if preferred:
                ranked = self._boost_by_source_role(filtered, preferred)
            else:
                ranked = filtered

            final_docs = ranked[:5]

            return {
                "context": self._build_context(final_docs),
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
