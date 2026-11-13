from typing import List, Optional
import re

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from ..processed_data.processed_data import ProcessedData
from .....core.config import settings


class VectorStoreManager:
    def __init__(self, collection_name: Optional[str] = None):
        self.connection_string = (
            f"postgresql+psycopg2://{settings.DB_USER}:{settings.DB_PASS}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )
        self.collection_name = (
            collection_name
            or getattr(settings, "COLLECTION_NAME", None)
            or "document_collection"
        )
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection_string,
            use_jsonb=True,
        )
        self._engine = create_engine(self.connection_string)

        print(f"Connected to PGVector collection: {self.collection_name}")

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 50,
    ) -> List[str]:
        """Add documents to the PostgreSQL database."""
        if not documents:
            return []

        try:
            ids = self.vector_store.add_documents(documents)
            print(f"Successfully added {len(ids)} documents")
            return ids
        except Exception as e:
            print(f"Error adding documents: {e}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Basic semantic search."""
        try:
            return self.vector_store.similarity_search(query=query, k=k)
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 8,
    ) -> List[tuple[Document, float]]:
        """Semantic search with raw vector distance score."""
        try:
            return self.vector_store.similarity_search_with_score(query=query, k=k)
        except Exception as e:
            print(f"Search with score error: {e}")
            return []

    def source_targeted_search(
        self,
        query: str,
        source_pattern: str,
        k: int = 5,
    ) -> List[dict]:
        """
        Semantic search scoped to chunks whose source path matches source_pattern.

        source_pattern is an ILIKE expression, e.g. '%LIBRO SERIE I%'.
        Returns dicts with page_content, metadata, and cosine distance score.

        Used to guarantee that content from a preferred source enters the
        retrieval pool for intent-routed queries, regardless of whether it
        scored in the top-k of the global hybrid search.
        """
        try:
            query_embedding = self.embeddings.embed_query(query)
            vec_literal = "[" + ",".join(str(v) for v in query_embedding) + "]"

            sql = f"""
                SELECT
                    e.document,
                    e.cmetadata,
                    (e.embedding <=> '{vec_literal}'::vector) AS distance
                FROM langchain_pg_embedding AS e
                JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
                WHERE c.name = :collection_name
                  AND e.cmetadata->>'source' ILIKE :source_pattern
                  AND length(e.document) >= 80
                ORDER BY e.embedding <=> '{vec_literal}'::vector
                LIMIT :k
            """

            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(sql),
                    {
                        "collection_name": self.collection_name,
                        "source_pattern": source_pattern,
                        "k": k,
                    },
                ).mappings().all()

            return [
                {
                    "page_content": row["document"] or "",
                    "metadata": row["cmetadata"] or {},
                    "distance": float(row["distance"] or 1.0),
                }
                for row in rows
                if (row["document"] or "").strip()
            ]
        except Exception as e:
            print(f"Source targeted search error: {e}")
            return []

    def keyword_search(self, query: str, k: int = 8) -> List[dict]:
        """Lexical search over stored chunk text for exact-term recall."""
        terms = [term for term in re.findall(r"\b\w+\b", query.lower()) if len(term) > 2]
        if not terms:
            return []

        score_parts = []
        where_parts = []
        params = {"collection_name": self.collection_name, "limit": k}
        for index, term in enumerate(terms):
            param_name = f"term_{index}"
            score_parts.append(
                f"CASE WHEN lower(e.document) LIKE :{param_name} THEN 1 ELSE 0 END"
            )
            where_parts.append(f"lower(e.document) LIKE :{param_name}")
            params[param_name] = f"%{term}%"

        sql = f"""
            SELECT
                e.document,
                e.cmetadata,
                ({' + '.join(score_parts)}) AS keyword_score
            FROM langchain_pg_embedding AS e
            JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
            WHERE c.name = :collection_name
              AND ({' OR '.join(where_parts)})
            ORDER BY keyword_score DESC, length(e.document) ASC
            LIMIT :limit
        """

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text(sql), params).mappings().all()
            return [
                {
                    "document": row["document"],
                    "metadata": row["cmetadata"] or {},
                    "keyword_score": float(row["keyword_score"] or 0.0),
                }
                for row in rows
            ]
        except Exception as e:
            print(f"Keyword search error: {e}")
            return []

    def page_search(self, source_pattern: str, pages: tuple) -> list[dict]:
        """
        Return ALL chunks from specific pages of a source, ordered by page then
        length DESC. Used for explicit book-and-page requests; no k limit.
        """
        if not pages:
            return []
        sql = """
            SELECT e.document, e.cmetadata
            FROM langchain_pg_embedding AS e
            JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
            WHERE c.name = :collection_name
              AND e.cmetadata->>'source' ILIKE :source_pattern
              AND (e.cmetadata->>'page_label')::int = ANY(:pages)
            ORDER BY (e.cmetadata->>'page_label')::int,
                     length(e.document) DESC
        """
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(sql),
                    {
                        "collection_name": self.collection_name,
                        "source_pattern": source_pattern,
                        "pages": list(pages),
                    },
                ).mappings().all()
            return [
                {
                    "page_content": row["document"] or "",
                    "metadata": row["cmetadata"] or {},
                }
                for row in rows
                if (row["document"] or "").strip()
            ]
        except Exception as e:
            print(f"Page search error: {e}")
            return []

    def source_ordered_search(self, source_pattern: str) -> list[dict]:
        """
        Return ALL chunks from a source in original document order (chunk_index).

        No vector distance is computed — this is a metadata-only query. Used for
        documents where complete, ordered coverage is required, such as the national
        anthem, where semantic ranking would reorder or silently drop stanzas.

        Returns dicts with page_content and metadata. No k limit.
        """
        sql = """
            SELECT e.document, e.cmetadata
            FROM langchain_pg_embedding AS e
            JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
            WHERE c.name = :collection_name
              AND e.cmetadata->>'source' ILIKE :source_pattern
            ORDER BY (e.cmetadata->>'chunk_index')::int
        """
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(sql),
                    {
                        "collection_name": self.collection_name,
                        "source_pattern": source_pattern,
                    },
                ).mappings().all()
            return [
                {
                    "page_content": row["document"] or "",
                    "metadata": row["cmetadata"] or {},
                }
                for row in rows
                if (row["document"] or "").strip()
            ]
        except Exception as e:
            print(f"Source ordered search error: {e}")
            return []

    def source_keyword_search(self, query: str, source_pattern: str, k: int = 8) -> list[dict]:
        """Keyword search scoped to a specific source document (ILIKE pattern)."""
        terms = [term for term in re.findall(r"\b\w+\b", query.lower()) if len(term) > 2]
        if not terms:
            return []

        score_parts = []
        where_parts = []
        params = {
            "collection_name": self.collection_name,
            "source_pattern": source_pattern,
            "limit": k,
        }
        for index, term in enumerate(terms):
            pname = f"term_{index}"
            score_parts.append(f"CASE WHEN lower(e.document) LIKE :{pname} THEN 1 ELSE 0 END")
            where_parts.append(f"lower(e.document) LIKE :{pname}")
            params[pname] = f"%{term}%"

        sql = f"""
            SELECT e.document, e.cmetadata, ({' + '.join(score_parts)}) AS keyword_score
            FROM langchain_pg_embedding AS e
            JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
            WHERE c.name = :collection_name
              AND e.cmetadata->>'source' ILIKE :source_pattern
              AND ({' OR '.join(where_parts)})
            ORDER BY keyword_score DESC, length(e.document) ASC
            LIMIT :limit
        """
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text(sql), params).mappings().all()
            return [
                {
                    "document": row["document"],
                    "metadata": row["cmetadata"] or {},
                    "keyword_score": float(row["keyword_score"] or 0.0),
                }
                for row in rows
            ]
        except Exception as e:
            print(f"Source keyword search error: {e}")
            return []

    def delete_collection(self):
        """Clear all data in this collection."""
        self.vector_store.delete_collection()
        print("Collection deleted")


def upload_docs_pipeline(file_paths: Optional[List[str]] = None):
    """Process and upload docs with richer chunk metadata."""
    try:
        processor = ProcessedData(file_paths=file_paths)
        chunks = processor.chunking()

        if not chunks:
            return []

        v_store = VectorStoreManager()
        ids = v_store.add_documents(chunks)
        return ids
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise
