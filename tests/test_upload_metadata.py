"""
Tests for upload metadata integrity and publication restriction.

Covers:
  1. ProcessedData.load_all_documents: original_filename survives loader metadata
     - Tested with actual TextLoader (injects source=UUID_path)
     - Tested with simulated PyPDFLoader shape (source + page fields)
     - Tested with simulated UnstructuredWordDocumentLoader shape (source only)
  2. page_number is correctly derived from loader-supplied page even after fix
  3. upload_docs_pipeline unconditionally stamps publication_allowed=False /
     review_state=pending regardless of loader or base metadata
  4. Restriction filter blocks pending chunks from public context
  5. Historia name preserved through merge enables source-path guard
  6. Route level: authorized upload passes sanitized original_filename to pipeline
  7. Disk path uses UUID; original name is metadata-only

No live database or VPS access — storage calls mocked; TextLoader test uses a
real temp file.

Run from repo root:
    pytest tests/test_upload_metadata.py -v
"""

import io
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

from langchain_core.documents import Document


# ===========================================================================
# 1. load_all_documents — original_filename survives loader metadata injection
# ===========================================================================

class TestLoaderMetadataMergeOrder:
    """
    Document loaders inject their own 'source' key pointing at the UUID disk
    path.  The merge order fix must ensure base identity fields (source,
    filename, section) win after the merge, not before.
    """

    def test_text_loader_does_not_overwrite_source(self):
        """
        Real TextLoader integration: creates a temp file, verifies that
        source/filename in the resulting Document metadata is the original_filename,
        NOT the UUID temp path that TextLoader injects.
        """
        from app.services.utils.RAG_pipeline.processed_data.processed_data import ProcessedData

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Ibagari buiganü Garínagu luma Surínama.\n" * 10)
            uuid_path = f.name

        try:
            processor = ProcessedData(
                file_paths=[uuid_path],
                original_filename="Garifuna_Language_Overview.txt",
            )
            docs = processor.load_all_documents()
            assert len(docs) >= 1
            for doc in docs:
                assert doc.metadata["source"] == "Garifuna_Language_Overview.txt", (
                    f"source was {doc.metadata['source']!r}; expected original_filename"
                )
                assert doc.metadata["filename"] == "Garifuna_Language_Overview.txt"
                assert doc.metadata["section"] == "Garifuna_Language_Overview.txt"
                # UUID path must not appear anywhere in source or filename
                assert os.path.basename(uuid_path) not in doc.metadata["source"]
        finally:
            os.unlink(uuid_path)

    def test_text_loader_without_original_filename_falls_back_to_basename(self):
        """
        When no original_filename is set (bulk-load path), source is derived
        from the actual file basename — not broken by the fix.
        """
        from app.services.utils.RAG_pipeline.processed_data.processed_data import ProcessedData

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8", prefix="LIBRO_SERIE_I_"
        ) as f:
            f.write("Cultural heritage text.\n" * 5)
            path = f.name

        try:
            processor = ProcessedData(file_paths=[path])
            docs = processor.load_all_documents()
            assert len(docs) >= 1
            # source should be the basename, not an empty string or UUID artifact
            assert docs[0].metadata["source"] == os.path.basename(path)
        finally:
            os.unlink(path)

    def _make_processor_with_mock_loader(self, loader_meta: dict, original_filename: str):
        """
        Returns (processor, docs) where the loader is mocked to return a single
        Document with the given loader_meta.  Used to test PDF and DOCX shapes
        without binary files.
        """
        from app.services.utils.RAG_pipeline.processed_data.processed_data import ProcessedData

        fake_doc = Document(
            page_content="Garifuna cultural content for testing.",
            metadata=loader_meta,
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            uuid_path = f.name

        try:
            processor = ProcessedData(
                file_paths=[uuid_path],
                original_filename=original_filename,
            )
            with patch.object(processor, "load_all_file_paths", return_value=[uuid_path]):
                # Patch the loader class so load() returns our fake doc
                with patch(
                    "app.services.utils.RAG_pipeline.processed_data.processed_data.PyPDFLoader"
                ) as mock_cls:
                    mock_cls.return_value.load.return_value = [fake_doc]
                    docs = processor.load_all_documents()
        finally:
            os.unlink(uuid_path)

        return docs

    def test_pdf_loader_source_does_not_overwrite_original_filename(self):
        """
        PyPDFLoader injects {"source": uuid_path, "page": 0}.
        source must be the original filename; page_number must be derived correctly.
        """
        uuid_path = "/tmp/abc123def456.pdf"
        docs = self._make_processor_with_mock_loader(
            loader_meta={"source": uuid_path, "page": 2},
            original_filename="Garifuna_History_EN.pdf",
        )
        assert len(docs) >= 1
        doc = docs[0]
        assert doc.metadata["source"] == "Garifuna_History_EN.pdf", (
            f"source was {doc.metadata['source']!r}"
        )
        assert doc.metadata["filename"] == "Garifuna_History_EN.pdf"
        assert "abc123def456" not in doc.metadata["source"]
        # page_number should be page+1 = 3
        assert doc.metadata.get("page_number") == 3

    def test_docx_loader_source_does_not_overwrite_original_filename(self):
        """
        UnstructuredWordDocumentLoader injects {"source": uuid_path}.
        source must be the original filename; no page (docx has no page field).
        """
        from app.services.utils.RAG_pipeline.processed_data.processed_data import ProcessedData

        uuid_path = "/tmp/abc123def456.docx"
        fake_doc = Document(
            page_content="Garínagu cultural content.",
            metadata={"source": uuid_path},
        )

        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            tmp_docx = f.name

        try:
            processor = ProcessedData(
                file_paths=[tmp_docx],
                original_filename="Garifuna_Cultural_Guide.docx",
            )
            with patch.object(processor, "load_all_file_paths", return_value=[tmp_docx]):
                with patch(
                    "app.services.utils.RAG_pipeline.processed_data.processed_data.UnstructuredWordDocumentLoader"
                ) as mock_cls:
                    mock_cls.return_value.load.return_value = [fake_doc]
                    docs = processor.load_all_documents()
        finally:
            os.unlink(tmp_docx)

        assert len(docs) >= 1
        assert docs[0].metadata["source"] == "Garifuna_Cultural_Guide.docx"
        assert docs[0].metadata.get("page_number") is None  # no page in docx loader

    def test_loader_page_field_is_preserved_after_merge(self):
        """
        page (from loader) must remain in metadata for page-aware queries,
        even though source is overridden.
        """
        from app.services.utils.RAG_pipeline.processed_data.processed_data import ProcessedData

        uuid_path = "/tmp/paged_uuid.pdf"
        fake_doc = Document(
            page_content="Page content here.",
            metadata={"source": uuid_path, "page": 4},
        )

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_pdf = f.name

        try:
            processor = ProcessedData(
                file_paths=[tmp_pdf],
                original_filename="Yurumein_Language_Book.pdf",
            )
            with patch.object(processor, "load_all_file_paths", return_value=[tmp_pdf]):
                with patch(
                    "app.services.utils.RAG_pipeline.processed_data.processed_data.PyPDFLoader"
                ) as mock_cls:
                    mock_cls.return_value.load.return_value = [fake_doc]
                    docs = processor.load_all_documents()
        finally:
            os.unlink(tmp_pdf)

        assert docs[0].metadata["source"] == "Yurumein_Language_Book.pdf"
        assert docs[0].metadata["page"] == 4       # loader field preserved
        assert docs[0].metadata["page_number"] == 5  # page + 1


# ===========================================================================
# 2. Historia filename through merge enables restriction guard
# ===========================================================================

class TestHistoriaFilenamePreservedThroughMerge:

    def test_historia_name_survives_text_loader_injection(self):
        """
        Real TextLoader integration with a Historia-named original_filename.
        The source-path guard must fire on the resulting document.
        """
        from app.services.utils.RAG_pipeline.processed_data.processed_data import ProcessedData
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Historia cultural content.\n" * 10)
            uuid_path = f.name

        try:
            processor = ProcessedData(
                file_paths=[uuid_path],
                original_filename="Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt",
            )
            docs = processor.load_all_documents()
            assert len(docs) >= 1
            for doc in docs:
                assert _is_publication_restricted(doc.metadata) is True, (
                    f"Historia chunk not blocked; source={doc.metadata.get('source')!r}"
                )
        finally:
            os.unlink(uuid_path)


# ===========================================================================
# 3. upload_docs_pipeline — publication_allowed unconditionally False
# ===========================================================================

class TestPipelinePublicationDefault:

    def _run_pipeline(self, chunks=None, original_filename=None):
        """Run pipeline with mocked ProcessedData and VectorStoreManager."""
        if chunks is None:
            chunks = [Document(
                page_content="garifuna content",
                metadata={"source": "doc.txt"},
            )]
        added_docs = []

        def fake_add(docs, **kw):
            added_docs.extend(docs)
            return ["fake-id"] * len(docs)

        with (
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.ProcessedData") as pd,
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.VectorStoreManager") as vs,
        ):
            pd.return_value.chunking.return_value = chunks
            vs.return_value.add_documents.side_effect = fake_add
            from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline
            upload_docs_pipeline(file_paths=["/tmp/abc.txt"], original_filename=original_filename)

        return added_docs

    def test_publication_allowed_false_on_new_upload(self):
        docs = self._run_pipeline()
        assert docs[0].metadata["publication_allowed"] is False

    def test_review_state_pending_on_new_upload(self):
        docs = self._run_pipeline()
        assert docs[0].metadata["review_state"] == "pending_cultural_review"

    def test_publication_allowed_cannot_be_set_true_by_loader(self):
        """Hard assignment — not setdefault — means even a chunk that arrives
        with publication_allowed=True gets overwritten to False."""
        chunk = Document(
            page_content="content",
            metadata={"source": "doc.txt", "publication_allowed": True},
        )
        docs = self._run_pipeline(chunks=[chunk])
        assert docs[0].metadata["publication_allowed"] is False, (
            "publication_allowed=True from loader must be unconditionally overridden"
        )

    def test_truthy_review_state_overwritten(self):
        """Any loader-supplied review_state is replaced."""
        chunk = Document(
            page_content="content",
            metadata={"source": "doc.txt", "review_state": "approved"},
        )
        docs = self._run_pipeline(chunks=[chunk])
        assert docs[0].metadata["review_state"] == "pending_cultural_review"

    def test_all_chunks_get_restriction_flag(self):
        chunks = [
            Document(page_content="chunk 1", metadata={"source": "doc.txt"}),
            Document(page_content="chunk 2", metadata={"source": "doc.txt", "publication_allowed": True}),
            Document(page_content="chunk 3", metadata={"source": "doc.txt", "review_state": "approved"}),
        ]
        docs = self._run_pipeline(chunks=chunks)
        assert all(d.metadata["publication_allowed"] is False for d in docs)
        assert all(d.metadata["review_state"] == "pending_cultural_review" for d in docs)

    def test_original_filename_passed_to_processor(self):
        with (
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.ProcessedData") as pd,
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.VectorStoreManager"),
        ):
            pd.return_value.chunking.return_value = []
            from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline
            upload_docs_pipeline(
                file_paths=["/tmp/abc.pdf"],
                original_filename="Real_Doc.pdf",
            )
        pd.assert_called_once_with(
            file_paths=["/tmp/abc.pdf"],
            original_filename="Real_Doc.pdf",
        )

    def test_empty_chunk_list_returns_empty(self):
        with (
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.ProcessedData") as pd,
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.VectorStoreManager") as vs,
        ):
            pd.return_value.chunking.return_value = []
            from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline
            result = upload_docs_pipeline(file_paths=["/tmp/abc.pdf"])
        assert result == []
        vs.return_value.add_documents.assert_not_called()


# ===========================================================================
# 4. Restriction filter — pending chunks blocked from public retrieval
# ===========================================================================

class TestPendingChunkBlocked:

    def test_pending_chunk_blocked_by_publication_allowed_false(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        assert _is_publication_restricted(
            {"publication_allowed": False, "review_state": "pending_cultural_review"}
        ) is True

    def test_pending_chunk_filtered_from_retrieval_set(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        docs = [
            {"metadata": {"publication_allowed": False, "review_state": "pending_cultural_review"},
             "page_content": "pending content"},
            {"metadata": {"publication_allowed": True}, "page_content": "approved content"},
        ]
        public = [d for d in docs if not _is_publication_restricted(d["metadata"])]
        assert len(public) == 1
        assert public[0]["page_content"] == "approved content"

    def test_historia_upload_double_blocked(self):
        """Historia name + pending state: both guards fire independently."""
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        meta = {
            "source": "Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt",
            "publication_allowed": False,
            "review_state": "pending_cultural_review",
        }
        assert _is_publication_restricted(meta) is True


# ===========================================================================
# 5. Route level — authorized upload passes sanitized original_filename
# ===========================================================================

class TestUploadRoutePassesOriginalName:

    def _do_upload_with_pipeline_capture(self, filename: str, content: bytes = b"x"):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.core.config import settings

        ADMIN_KEY = "test-key-abc"
        settings.ADMIN_API_KEY = ADMIN_KEY
        app = FastAPI()
        from app.services.chatbot.cahtbot_router import router
        app.include_router(router)

        captured = {}

        def fake_pipeline(file_paths, original_filename=None, **kw):
            captured["original_filename"] = original_filename
            captured["disk_path"] = file_paths[0] if file_paths else ""

        with (
            patch("app.services.chatbot.cahtbot_router.upload_docs_pipeline", side_effect=fake_pipeline),
            patch("os.path.exists", return_value=False),
        ):
            with TestClient(app) as client:
                resp = client.post(
                    "/upload_docs",
                    headers={"X-Admin-Key": ADMIN_KEY},
                    files={"file": (filename, io.BytesIO(content), "text/plain")},
                )
        return resp, captured

    def test_original_name_reaches_pipeline(self):
        resp, captured = self._do_upload_with_pipeline_capture("My_Garifuna_Doc.txt")
        assert resp.status_code == 200
        assert captured["original_filename"] == "My_Garifuna_Doc.txt"

    def test_path_traversal_stripped_to_basename(self):
        resp, captured = self._do_upload_with_pipeline_capture("../../etc/shadow.txt")
        assert resp.status_code == 200
        name = captured["original_filename"]
        assert "/" not in name and "\\" not in name
        assert name == "shadow.txt"

    def test_disk_path_is_uuid_not_original_name(self):
        import re
        _, captured = self._do_upload_with_pipeline_capture("Real_Document_Name.txt")
        disk = os.path.basename(captured.get("disk_path", ""))
        assert re.match(r"^[0-9a-f]{32}\.txt$", disk), f"Expected UUID, got: {disk}"
        assert "Real_Document_Name" not in captured["disk_path"]

    def test_response_shows_original_name(self):
        resp, _ = self._do_upload_with_pipeline_capture("Cultural_Archive.txt")
        assert resp.status_code == 200
        assert resp.json()["filename"] == "Cultural_Archive.txt"
        assert resp.json()["status"] == "success"
        assert "review" in resp.json()["message"].lower() or "vectorized" in resp.json()["message"].lower()
