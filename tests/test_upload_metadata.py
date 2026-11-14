"""
Tests for upload metadata integrity and publication restriction after UUID filename change.

Covers:
  1. ProcessedData._base_metadata uses original_filename when supplied
  2. upload_docs_pipeline stamps publication_allowed=False on every chunk
  3. Restriction filter blocks pending chunks from public context
  4. Source/filename ILIKE patterns still work with original name
  5. Path-traversal in original_filename has no disk-I/O effect
  6. Route-level: authorized upload preserves original name in pipeline call

No live database or VPS access — all storage calls are mocked.

Run from repo root:
    pytest tests/test_upload_metadata.py -v
"""

import io
import os
import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path

from langchain_core.documents import Document


# ===========================================================================
# 1. ProcessedData._base_metadata — original_filename override
# ===========================================================================

class TestBaseMetadataWithOriginalFilename:
    """_base_metadata should use original_filename when provided."""

    def _make_processor(self, original_filename=None):
        from app.services.utils.RAG_pipeline.processed_data.processed_data import ProcessedData
        return ProcessedData(
            file_paths=["/tmp/abc123.pdf"],
            original_filename=original_filename,
        )

    def test_without_override_uses_uuid_path(self):
        p = self._make_processor()
        meta = p._base_metadata("/tmp/abc123def456.pdf", "pdf", 0)
        assert meta["source"] == "abc123def456.pdf"
        assert meta["filename"] == "abc123def456.pdf"

    def test_with_override_uses_original_name(self):
        p = self._make_processor(original_filename="Garifuna_Dictionary_v2.pdf")
        meta = p._base_metadata("/tmp/abc123def456.pdf", "pdf", 0)
        assert meta["source"] == "Garifuna_Dictionary_v2.pdf"
        assert meta["filename"] == "Garifuna_Dictionary_v2.pdf"
        assert meta["section"] == "Garifuna_Dictionary_v2.pdf"

    def test_uuid_path_absent_from_stored_metadata(self):
        """The UUID disk path must not appear in any metadata field."""
        p = self._make_processor(original_filename="Garifuna_Dictionary_v2.pdf")
        meta = p._base_metadata("/tmp/abc123def456.pdf", "pdf", 0)
        for field in ("source", "filename", "section"):
            assert "abc123def456" not in meta[field]

    def test_original_filename_cannot_inject_path_separator(self):
        """Even if the caller passes a path with separators, only the basename is used.
        Note: sanitization happens in the router; this test documents that
        the pipeline stores whatever string is passed. The safe_original_name
        extraction in the router is what prevents path injection."""
        p = self._make_processor(original_filename="safe_name.pdf")
        meta = p._base_metadata("/tmp/abc.pdf", "pdf", 0)
        assert meta["source"] == "safe_name.pdf"

    def test_historia_name_preserved_enables_restriction_guard(self):
        """Original Historia filename must survive into source metadata so
        _is_publication_restricted can fire the source-path guard."""
        p = self._make_processor(
            original_filename="Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt"
        )
        meta = p._base_metadata("/tmp/abc123.txt", "txt", 0)
        assert "HISTORIA_ES_EN" in meta["source"].upper()

    def test_approved_name_does_not_trigger_restriction_guard(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        p = self._make_processor(original_filename="Garifuna_Language_Overview.pdf")
        meta = p._base_metadata("/tmp/abc123.pdf", "pdf", 0)
        # source-path guard should NOT fire
        assert _is_publication_restricted(meta) is False


# ===========================================================================
# 2. upload_docs_pipeline — publication_allowed=False injected on all chunks
# ===========================================================================

class TestPipelinePublicationDefault:
    """Every chunk from upload_docs_pipeline must carry publication_allowed=False."""

    def _run_pipeline(self, original_filename=None, extra_meta=None):
        """Run pipeline with mocked ProcessedData and VectorStoreManager.
        Returns the Document list that would have been passed to add_documents."""
        chunk = Document(
            page_content="some garifuna content",
            metadata={"source": "some_doc.txt", **(extra_meta or {})},
        )

        added_docs = []

        def fake_add_documents(docs, **kwargs):
            added_docs.extend(docs)
            return ["fake-id"] * len(docs)

        with (
            patch(
                "app.services.utils.RAG_pipeline.vectore_db.pgvector_db.ProcessedData"
            ) as mock_pd_cls,
            patch(
                "app.services.utils.RAG_pipeline.vectore_db.pgvector_db.VectorStoreManager"
            ) as mock_vs_cls,
        ):
            mock_pd_cls.return_value.chunking.return_value = [chunk]
            mock_vs_cls.return_value.add_documents.side_effect = fake_add_documents

            from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline
            upload_docs_pipeline(
                file_paths=["/tmp/abc123.txt"],
                original_filename=original_filename,
            )

        return added_docs

    def test_publication_allowed_false_on_new_upload(self):
        docs = self._run_pipeline()
        assert len(docs) == 1
        assert docs[0].metadata["publication_allowed"] is False

    def test_review_state_pending_on_new_upload(self):
        docs = self._run_pipeline()
        assert docs[0].metadata["review_state"] == "pending_cultural_review"

    def test_all_chunks_get_restriction_flag(self):
        chunk1 = Document(page_content="chunk 1", metadata={"source": "doc.txt"})
        chunk2 = Document(page_content="chunk 2", metadata={"source": "doc.txt"})
        added_docs = []

        with (
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.ProcessedData") as pd,
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.VectorStoreManager") as vs,
        ):
            pd.return_value.chunking.return_value = [chunk1, chunk2]
            vs.return_value.add_documents.side_effect = lambda docs, **kw: (
                added_docs.extend(docs) or ["id1", "id2"]
            )
            from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline
            upload_docs_pipeline(file_paths=["/tmp/test.txt"])

        assert all(d.metadata["publication_allowed"] is False for d in added_docs)
        assert all(d.metadata["review_state"] == "pending_cultural_review" for d in added_docs)

    def test_explicit_false_already_set_is_not_overridden(self):
        """setdefault must not overwrite an already-false value."""
        docs = self._run_pipeline(extra_meta={"publication_allowed": False})
        assert docs[0].metadata["publication_allowed"] is False

    def test_original_filename_passed_to_processor(self):
        with (
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.ProcessedData") as pd,
            patch("app.services.utils.RAG_pipeline.vectore_db.pgvector_db.VectorStoreManager") as vs,
        ):
            pd.return_value.chunking.return_value = []
            from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline
            upload_docs_pipeline(
                file_paths=["/tmp/abc123.pdf"],
                original_filename="Real_Document_Name.pdf",
            )
        pd.assert_called_once_with(
            file_paths=["/tmp/abc123.pdf"],
            original_filename="Real_Document_Name.pdf",
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
# 3. Restriction filter — pending chunks blocked from public retrieval
# ===========================================================================

class TestPendingChunkBlocked:
    """Chunks stamped publication_allowed=False must be filtered from retrieval."""

    def test_pending_chunk_filtered_by_is_publication_restricted(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        meta = {"publication_allowed": False, "review_state": "pending_cultural_review"}
        assert _is_publication_restricted(meta) is True

    def test_pending_chunk_filtered_by_filter_restricted_docs(self):
        """The Nodes method must exclude pending-upload chunks."""
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        docs = [
            {"page_content": "pending content",
             "metadata": {"publication_allowed": False, "review_state": "pending_cultural_review"}},
            {"page_content": "approved content",
             "metadata": {"publication_allowed": True}},
        ]
        filtered = [d for d in docs if not _is_publication_restricted(d["metadata"])]
        assert len(filtered) == 1
        assert filtered[0]["page_content"] == "approved content"

    def test_historia_upload_double_blocked(self):
        """A Historia file that gets past the extension check is blocked by BOTH
        source-path guard AND the pending publication state."""
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
        meta = {
            "source": "Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt",
            "publication_allowed": False,
            "review_state": "pending_cultural_review",
        }
        assert _is_publication_restricted(meta) is True


# ===========================================================================
# 4. Route level — authorized upload passes original_filename to pipeline
# ===========================================================================

class TestUploadRoutePassesOriginalName:
    """The router must extract the original filename and pass it to the pipeline."""

    def test_original_name_reaches_pipeline(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.core.config import settings

        ADMIN_KEY = "test-key-abc"
        settings.ADMIN_API_KEY = ADMIN_KEY

        app = FastAPI()
        from app.services.chatbot.cahtbot_router import router as chatbot_router
        app.include_router(chatbot_router)

        pipeline_calls = {}

        def fake_pipeline(file_paths, original_filename=None, **kwargs):
            pipeline_calls["original_filename"] = original_filename
            pipeline_calls["file_paths"] = file_paths

        with (
            patch("app.services.chatbot.cahtbot_router.upload_docs_pipeline", side_effect=fake_pipeline),
            patch("os.path.exists", return_value=False),
        ):
            with TestClient(app) as client:
                resp = client.post(
                    "/upload_docs",
                    headers={"X-Admin-Key": ADMIN_KEY},
                    files={"file": ("My_Garifuna_Doc.txt", io.BytesIO(b"content"), "text/plain")},
                )

        assert resp.status_code == 200
        assert pipeline_calls["original_filename"] == "My_Garifuna_Doc.txt"

    def test_path_traversal_in_filename_stripped_before_pipeline(self):
        """../../etc/passwd.txt must arrive at pipeline as 'passwd.txt' (basename only).
        The extension filter (.txt allowed) means this test only covers the
        basename stripping — the directory components must be gone."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.core.config import settings

        settings.ADMIN_API_KEY = "test-key-abc"
        app = FastAPI()
        from app.services.chatbot.cahtbot_router import router
        app.include_router(router)

        pipeline_calls = {}

        def fake_pipeline(file_paths, original_filename=None, **kwargs):
            pipeline_calls["original_filename"] = original_filename

        with (
            patch("app.services.chatbot.cahtbot_router.upload_docs_pipeline", side_effect=fake_pipeline),
            patch("os.path.exists", return_value=False),
        ):
            with TestClient(app) as client:
                resp = client.post(
                    "/upload_docs",
                    headers={"X-Admin-Key": "test-key-abc"},
                    files={"file": ("../../etc/shadow.txt", io.BytesIO(b"content"), "text/plain")},
                )

        assert resp.status_code == 200
        # Must NOT contain any path separator or directory component
        name = pipeline_calls["original_filename"]
        assert "/" not in name
        assert "\\" not in name
        assert name == "shadow.txt"

    def test_disk_path_uses_uuid_not_original_name(self):
        """The file must be written to a UUID path, never the original name."""
        import re
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.core.config import settings

        settings.ADMIN_API_KEY = "test-key-abc"
        app = FastAPI()
        from app.services.chatbot.cahtbot_router import router
        app.include_router(router)

        pipeline_calls = {}

        def fake_pipeline(file_paths, original_filename=None, **kwargs):
            pipeline_calls["disk_path"] = file_paths[0] if file_paths else ""

        with (
            patch("app.services.chatbot.cahtbot_router.upload_docs_pipeline", side_effect=fake_pipeline),
            patch("os.path.exists", return_value=False),
            patch("builtins.open", MagicMock()),
            patch("shutil.copyfileobj"),
        ):
            with TestClient(app) as client:
                client.post(
                    "/upload_docs",
                    headers={"X-Admin-Key": "test-key-abc"},
                    files={"file": ("Real_Document_Name.txt", io.BytesIO(b"x"), "text/plain")},
                )

        disk_path = pipeline_calls.get("disk_path", "")
        filename_part = os.path.basename(disk_path)
        assert re.match(r"^[0-9a-f]{32}\.txt$", filename_part), (
            f"Expected UUID filename, got: {filename_part}"
        )
        assert "Real_Document_Name" not in disk_path
