"""
Tests for upload metadata integrity and publication approval workflow.

Covers:
  1. ProcessedData.load_all_documents: original_filename survives loader metadata
     - Actual TextLoader (injects source=UUID_path)
     - Simulated PyPDFLoader shape (source + page fields)
     - Simulated UnstructuredWordDocumentLoader shape (source only)
  2. page_number correctly derived from loader page field after merge fix
  3. upload_docs_pipeline — approved upload stamps publication_allowed=True / approved
  4. upload_docs_pipeline — confirmed=False raises ValueError immediately
  5. upload_docs_pipeline — confirmed required (no default); TypeError if omitted
  6. Restriction filter: existing pending/restricted docs remain blocked
  7. Historia name preserved through merge keeps source-path guard active
  8. Route level: Historia upload rejected before pipeline; zero chunks → 400
  9. Route level: authorized upload passes confirmed=True; original filename preserved

Policy: review before upload; upload == approved for publication.
Existing DB documents are not touched by these tests (all storage is mocked).

Run from repo root:
    pytest tests/test_upload_metadata.py -v
"""

import io
import os
import re
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from langchain_core.documents import Document


# ===========================================================================
# 1. load_all_documents — original_filename survives loader metadata injection
# ===========================================================================

class TestLoaderMetadataMergeOrder:
    """
    Document loaders inject their own 'source' key pointing at the UUID disk
    path.  The merge order fix ensures base identity fields (source, filename,
    section) always win after the merge.
    """

    def test_text_loader_does_not_overwrite_source(self):
        """Real TextLoader: source/filename must be original_filename, not UUID path."""
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
                assert doc.metadata["source"] == "Garifuna_Language_Overview.txt"
                assert doc.metadata["filename"] == "Garifuna_Language_Overview.txt"
                assert doc.metadata["section"] == "Garifuna_Language_Overview.txt"
                assert os.path.basename(uuid_path) not in doc.metadata["source"]
        finally:
            os.unlink(uuid_path)

    def test_text_loader_without_original_filename_falls_back_to_basename(self):
        """Bulk-load path: source is derived from the actual file basename."""
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
            assert docs[0].metadata["source"] == os.path.basename(path)
        finally:
            os.unlink(path)

    def _processor_with_mock_loader(self, loader_meta: dict, original_filename: str, suffix=".pdf"):
        """Return (processor, temp_path) with a mocked loader returning one doc."""
        from app.services.utils.RAG_pipeline.processed_data.processed_data import ProcessedData

        fake_doc = Document(
            page_content="Garifuna cultural content for testing.",
            metadata=loader_meta,
        )

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            tmp_path = f.name

        processor = ProcessedData(file_paths=[tmp_path], original_filename=original_filename)
        return processor, tmp_path, fake_doc

    def test_pdf_loader_source_does_not_overwrite_original_filename(self):
        """PyPDFLoader injects {source: uuid_path, page: N}; source must be original."""
        processor, tmp_path, fake_doc = self._processor_with_mock_loader(
            {"source": "/tmp/abc123def456.pdf", "page": 2},
            "Garifuna_History_EN.pdf",
        )
        try:
            with patch.object(processor, "load_all_file_paths", return_value=[tmp_path]):
                with patch(
                    "app.services.utils.RAG_pipeline.processed_data.processed_data.PyPDFLoader"
                ) as mock_cls:
                    mock_cls.return_value.load.return_value = [fake_doc]
                    docs = processor.load_all_documents()
        finally:
            os.unlink(tmp_path)

        assert docs[0].metadata["source"] == "Garifuna_History_EN.pdf"
        assert "abc123def456" not in docs[0].metadata["source"]
        assert docs[0].metadata.get("page_number") == 3  # page+1

    def test_docx_loader_source_does_not_overwrite_original_filename(self):
        """UnstructuredWordDocumentLoader injects {source: uuid_path}; source must be original."""
        from app.services.utils.RAG_pipeline.processed_data.processed_data import ProcessedData

        fake_doc = Document(
            page_content="Garínagu cultural content.",
            metadata={"source": "/tmp/abc123def456.docx"},
        )
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            tmp_path = f.name

        try:
            processor = ProcessedData(file_paths=[tmp_path], original_filename="Garifuna_Cultural_Guide.docx")
            with patch.object(processor, "load_all_file_paths", return_value=[tmp_path]):
                with patch(
                    "app.services.utils.RAG_pipeline.processed_data.processed_data.UnstructuredWordDocumentLoader"
                ) as mock_cls:
                    mock_cls.return_value.load.return_value = [fake_doc]
                    docs = processor.load_all_documents()
        finally:
            os.unlink(tmp_path)

        assert docs[0].metadata["source"] == "Garifuna_Cultural_Guide.docx"
        assert docs[0].metadata.get("page_number") is None

    def test_loader_page_field_preserved_after_merge(self):
        """page key from loader must survive even though source is overridden."""
        processor, tmp_path, fake_doc = self._processor_with_mock_loader(
            {"source": "/tmp/paged_uuid.pdf", "page": 4},
            "Yurumein_Language_Book.pdf",
        )
        try:
            with patch.object(processor, "load_all_file_paths", return_value=[tmp_path]):
                with patch(
                    "app.services.utils.RAG_pipeline.processed_data.processed_data.PyPDFLoader"
                ) as mock_cls:
                    mock_cls.return_value.load.return_value = [fake_doc]
                    docs = processor.load_all_documents()
        finally:
            os.unlink(tmp_path)

        assert docs[0].metadata["source"] == "Yurumein_Language_Book.pdf"
        assert docs[0].metadata["page"] == 4
        assert docs[0].metadata["page_number"] == 5


# ===========================================================================
# 2. Historia filename preserved through merge — source-path guard still fires
# ===========================================================================

class TestHistoriaFilenamePreservedThroughMerge:

    def test_historia_name_survives_text_loader_injection(self):
        """
        Real TextLoader integration: even after the merge-order fix, Historia
        chunks must still be flagged as restricted by the source-path guard.
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
                assert _is_publication_restricted(doc.metadata) is True
        finally:
            os.unlink(uuid_path)


# ===========================================================================
# 3. upload_docs_pipeline — approved upload → publication_allowed=True
# ===========================================================================

class TestPipelineApprovalStamping:
    """
    Policy: upload == approved for publication.
    Every chunk from a confirmed upload must be stamped publication_allowed=True
    and review_state="approved", unconditionally.
    """

    def _run_pipeline(self, chunks=None, original_filename=None, confirmed=True):
        if chunks is None:
            chunks = [Document(page_content="garifuna content", metadata={"source": "doc.txt"})]
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
            upload_docs_pipeline(
                file_paths=["/tmp/abc.txt"],
                original_filename=original_filename,
                confirmed=confirmed,
            )

        return added_docs

    def test_approved_upload_publication_allowed_true(self):
        docs = self._run_pipeline()
        assert docs[0].metadata["publication_allowed"] is True

    def test_approved_upload_review_state_approved(self):
        docs = self._run_pipeline()
        assert docs[0].metadata["review_state"] == "approved"

    def test_all_chunks_stamped_approved(self):
        chunks = [
            Document(page_content="chunk 1", metadata={"source": "doc.txt"}),
            Document(page_content="chunk 2", metadata={"source": "doc.txt"}),
            Document(page_content="chunk 3", metadata={"source": "doc.txt"}),
        ]
        docs = self._run_pipeline(chunks=chunks)
        assert all(d.metadata["publication_allowed"] is True for d in docs)
        assert all(d.metadata["review_state"] == "approved" for d in docs)

    def test_loader_cannot_override_publication_allowed(self):
        """Even if a loader injects publication_allowed=False, approved upload wins."""
        chunk = Document(
            page_content="content",
            metadata={"source": "doc.txt", "publication_allowed": False},
        )
        docs = self._run_pipeline(chunks=[chunk])
        assert docs[0].metadata["publication_allowed"] is True

    def test_loader_cannot_override_review_state(self):
        """Loader-injected review_state is replaced by 'approved'."""
        chunk = Document(
            page_content="content",
            metadata={"source": "doc.txt", "review_state": "pending_cultural_review"},
        )
        docs = self._run_pipeline(chunks=[chunk])
        assert docs[0].metadata["review_state"] == "approved"

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
                confirmed=True,
            )
        pd.assert_called_once_with(file_paths=["/tmp/abc.pdf"], original_filename="Real_Doc.pdf")

    def test_empty_chunk_list_returns_empty(self):
        with (
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.ProcessedData") as pd,
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.VectorStoreManager") as vs,
        ):
            pd.return_value.chunking.return_value = []
            from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline
            result = upload_docs_pipeline(file_paths=["/tmp/abc.pdf"], confirmed=True)
        assert result == []
        vs.return_value.add_documents.assert_not_called()


# ===========================================================================
# 4. upload_docs_pipeline — confirmed=False / missing raises
# ===========================================================================

class TestPipelineConfirmationRequired:

    def test_confirmed_false_raises_value_error(self):
        """confirmed=False must raise before any file I/O or indexing."""
        with (
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.ProcessedData") as pd,
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.VectorStoreManager") as vs,
        ):
            from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline
            with pytest.raises(ValueError, match="confirmation"):
                upload_docs_pipeline(file_paths=["/tmp/abc.txt"], confirmed=False)
        pd.assert_not_called()
        vs.assert_not_called()

    def test_confirmed_required_keyword_only(self):
        """confirmed is keyword-only (*); passing it positionally must raise TypeError."""
        from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline
        with pytest.raises(TypeError):
            upload_docs_pipeline(["/tmp/abc.txt"], "doc.txt", True)  # positional after *

    def test_omitting_confirmed_raises_type_error(self):
        """Calling the pipeline without confirmed must raise TypeError immediately."""
        from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline
        with pytest.raises(TypeError):
            upload_docs_pipeline(file_paths=["/tmp/abc.txt"])


# ===========================================================================
# 5. Restriction filter — existing pending/restricted docs remain blocked
# ===========================================================================

class TestExistingRestrictionFilterUnchanged:
    """
    _is_publication_restricted is unchanged.  Existing DB chunks with
    publication_allowed=False, pending_cultural_review state, or a Historia
    source path are still blocked from public context.
    """

    def test_publication_allowed_false_still_blocked(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        assert _is_publication_restricted({"publication_allowed": False}) is True

    def test_pending_cultural_review_still_blocked(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        assert _is_publication_restricted({"review_state": "pending_cultural_review"}) is True

    def test_historia_source_still_blocked(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        assert _is_publication_restricted(
            {"source": "Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt"}
        ) is True

    def test_historia_null_publication_flag_still_blocked(self):
        """Source-path guard fires even when publication_allowed=None."""
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        assert _is_publication_restricted(
            {
                "source": "data/Yurumein_AI_Historia_ES_EN_v1.2.txt",
                "publication_allowed": None,
                "review_state": None,
            }
        ) is True

    def test_approved_doc_passes_filter(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        assert _is_publication_restricted(
            {"publication_allowed": True, "review_state": "approved"}
        ) is False

    def test_pending_chunk_filtered_from_retrieval_set(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        docs = [
            {"metadata": {"publication_allowed": False, "review_state": "pending_cultural_review"},
             "page_content": "pending content"},
            {"metadata": {"publication_allowed": True, "review_state": "approved"},
             "page_content": "approved content"},
        ]
        public = [d for d in docs if not _is_publication_restricted(d["metadata"])]
        assert len(public) == 1
        assert public[0]["page_content"] == "approved content"

    def test_test_doc_still_blocked(self):
        """Previous test document (uploaded during deployment verification) stays blocked."""
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        meta = {
            "source": "Yurumein_Upload_Test_20261114.txt",
            "publication_allowed": False,
            "review_state": "pending_cultural_review",
        }
        assert _is_publication_restricted(meta) is True


# ===========================================================================
# 6. Route level — approval workflow end-to-end
# ===========================================================================

class TestUploadRouteApprovalWorkflow:

    def _build_client(self, admin_key="test-key-abc"):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.core.config import settings
        settings.ADMIN_API_KEY = admin_key
        app = FastAPI()
        from app.services.chatbot.cahtbot_router import router
        app.include_router(router)
        return TestClient(app)

    def _upload(self, client, filename, content=b"x", confirmed="true", ids=None):
        captured = {}

        def fake_pipeline(file_paths, original_filename=None, **kwargs):
            captured["original_filename"] = original_filename
            captured["disk_path"] = file_paths[0] if file_paths else ""
            captured["confirmed"] = kwargs.get("confirmed")
            return ids if ids is not None else ["fake-id"]

        with (
            patch("app.services.chatbot.cahtbot_router.upload_docs_pipeline", side_effect=fake_pipeline),
            patch("os.path.exists", return_value=False),
        ):
            data = {}
            if confirmed is not None:
                data["confirmed"] = confirmed
            resp = client.post(
                "/upload_docs",
                headers={"X-Admin-Key": "test-key-abc"},
                files={"file": (filename, io.BytesIO(content), "text/plain")},
                data=data,
            )
        return resp, captured

    def test_original_name_reaches_pipeline(self):
        with self._build_client() as client:
            resp, captured = self._upload(client, "My_Garifuna_Doc.txt")
        assert resp.status_code == 200
        assert captured["original_filename"] == "My_Garifuna_Doc.txt"

    def test_path_traversal_stripped_to_basename(self):
        with self._build_client() as client:
            resp, captured = self._upload(client, "../../etc/shadow.txt")
        assert resp.status_code == 200
        name = captured["original_filename"]
        assert "/" not in name and "\\" not in name
        assert name == "shadow.txt"

    def test_disk_path_is_uuid_not_original_name(self):
        with self._build_client() as client:
            _, captured = self._upload(client, "Real_Document_Name.txt")
        disk = os.path.basename(captured.get("disk_path", ""))
        assert re.match(r"^[0-9a-f]{32}\.txt$", disk), f"Expected UUID, got: {disk}"
        assert "Real_Document_Name" not in captured["disk_path"]

    def test_response_message_reflects_approval(self):
        with self._build_client() as client:
            resp, _ = self._upload(client, "Cultural_Archive.txt")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["message"] == "Document uploaded and approved for Yurumein AI."

    def test_no_confirmation_returns_422(self):
        with self._build_client() as client:
            resp, captured = self._upload(client, "doc.txt", confirmed=None)
        assert resp.status_code == 422
        assert captured == {}

    def test_false_confirmation_returns_422(self):
        with self._build_client() as client:
            resp, captured = self._upload(client, "doc.txt", confirmed="false")
        assert resp.status_code == 422
        assert captured == {}

    def test_historia_filename_rejected_400(self):
        with self._build_client() as client:
            resp, captured = self._upload(
                client, "Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt"
            )
        assert resp.status_code == 400
        detail = resp.json()["detail"].lower()
        assert "restricted" in detail
        assert captured == {}

    def test_zero_chunks_returns_400(self):
        """Pipeline returns [] → 400; no success message shown."""
        with self._build_client() as client:
            resp, _ = self._upload(client, "empty.txt", ids=[])
        assert resp.status_code == 400
        assert resp.json()["status"] != "success" if "status" in resp.json() else True

    def test_pipeline_receives_confirmed_true(self):
        with self._build_client() as client:
            _, captured = self._upload(client, "doc.txt")
        assert captured.get("confirmed") is True
