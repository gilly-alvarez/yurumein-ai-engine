"""
Route-level security tests for admin-gated endpoints via FastAPI TestClient.

Endpoints under test:
  GET  /retrieval_check          — requires X-Admin-Key, filters restricted docs
  GET  /retrieval_check_hybrid   — requires X-Admin-Key, filters restricted docs
  POST /upload_docs              — requires X-Admin-Key, server-side filename

All DB and pipeline calls are mocked so no production state is touched.

Design follows test_integration_phase1.py:
  - A minimal FastAPI app is built per test using the real chatbot router.
  - settings.ADMIN_API_KEY is set directly before mounting the router.
  - Patches target the module-level names in cahtbot_router (not the imported origin).

Run from repo root:
    pytest tests/test_admin_endpoint_routes.py -v
"""

import io
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

ADMIN_KEY = "test-admin-key-xyz"

# ---------------------------------------------------------------------------
# App builder
# ---------------------------------------------------------------------------

def _build_app(admin_api_key: str = ADMIN_KEY) -> FastAPI:
    """Minimal FastAPI app with the chatbot router and ADMIN_API_KEY pre-set."""
    from app.core.config import settings
    settings.ADMIN_API_KEY = admin_api_key

    app = FastAPI()
    from app.services.chatbot.cahtbot_router import router as chatbot_router
    app.include_router(chatbot_router)
    return app


# ---------------------------------------------------------------------------
# Document helpers (match the shape returned by VectorStoreManager and
# node._hybrid_retrieve respectively)
# ---------------------------------------------------------------------------

def _vector_doc(source: str, content: str = "some content") -> MagicMock:
    """LangChain Document-like object for similarity_search results."""
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source}
    return doc


def _hybrid_doc(source: str, content: str = "some content") -> dict:
    """Dict shape returned by node._hybrid_retrieve."""
    return {
        "page_content": content,
        "metadata": {"source": source},
        "retrieval_source": "vector",
        "hybrid_score": 0.85,
        "vector_score": 0.9,
        "keyword_score": 0.5,
    }


HISTORIA_PATH = "/app/data/Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt"
APPROVED_PATH = "/app/data/Yurumein_Garifuna_Identity_EN_ES.docx"


# ===========================================================================
# 1. /retrieval_check — HTTP-layer auth
# ===========================================================================

class TestRetrievalCheckAuth:

    def test_no_key_returns_403(self):
        with TestClient(_build_app()) as client:
            resp = client.get("/retrieval_check?query=hello")
        assert resp.status_code == 403

    def test_wrong_key_returns_403(self):
        with TestClient(_build_app()) as client:
            resp = client.get(
                "/retrieval_check?query=hello",
                headers={"X-Admin-Key": "wrong-key"},
            )
        assert resp.status_code == 403

    def test_unconfigured_server_returns_503(self):
        with TestClient(_build_app(admin_api_key="")) as client:
            resp = client.get(
                "/retrieval_check?query=hello",
                headers={"X-Admin-Key": ADMIN_KEY},
            )
        assert resp.status_code == 503

    def test_missing_query_returns_400(self):
        with patch(
            "app.services.chatbot.cahtbot_router.VectorStoreManager"
        ) as mock_vs_cls:
            mock_vs_cls.return_value.similarity_search.return_value = []
            with TestClient(_build_app()) as client:
                resp = client.get(
                    "/retrieval_check",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
        assert resp.status_code == 422  # FastAPI requires query param


# ===========================================================================
# 2. /retrieval_check — restriction filter at route level
# ===========================================================================

class TestRetrievalCheckRestrictionFilter:

    def test_approved_doc_returned(self):
        with patch(
            "app.services.chatbot.cahtbot_router.VectorStoreManager"
        ) as mock_vs_cls:
            mock_vs_cls.return_value.similarity_search.return_value = [
                _vector_doc(APPROVED_PATH, "garifuna identity content"),
            ]
            with TestClient(_build_app()) as client:
                resp = client.get(
                    "/retrieval_check?query=garifuna",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["matches_found"] == 1
        assert body["matches"][0]["content"] == "garifuna identity content"

    def test_historia_doc_filtered_even_with_valid_key(self):
        with patch(
            "app.services.chatbot.cahtbot_router.VectorStoreManager"
        ) as mock_vs_cls:
            mock_vs_cls.return_value.similarity_search.return_value = [
                _vector_doc(HISTORIA_PATH, "restricted historia content"),
            ]
            with TestClient(_build_app()) as client:
                resp = client.get(
                    "/retrieval_check?query=historia",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["matches_found"] == 0
        assert body["matches"] == []

    def test_mixed_results_historia_stripped(self):
        with patch(
            "app.services.chatbot.cahtbot_router.VectorStoreManager"
        ) as mock_vs_cls:
            mock_vs_cls.return_value.similarity_search.return_value = [
                _vector_doc(HISTORIA_PATH, "historia text"),
                _vector_doc(APPROVED_PATH, "approved text"),
                _vector_doc(HISTORIA_PATH, "another historia chunk"),
            ]
            with TestClient(_build_app()) as client:
                resp = client.get(
                    "/retrieval_check?query=garifuna",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["matches_found"] == 1
        assert body["matches"][0]["content"] == "approved text"

    def test_historia_null_metadata_still_filtered(self):
        """Production state: pa=None, rs=None — source-path guard must still fire."""
        doc = MagicMock()
        doc.page_content = "historia content"
        doc.metadata = {
            "source": HISTORIA_PATH,
            "publication_allowed": None,
            "review_state": None,
        }
        with patch(
            "app.services.chatbot.cahtbot_router.VectorStoreManager"
        ) as mock_vs_cls:
            mock_vs_cls.return_value.similarity_search.return_value = [doc]
            with TestClient(_build_app()) as client:
                resp = client.get(
                    "/retrieval_check?query=historia",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
        assert resp.status_code == 200
        assert resp.json()["matches_found"] == 0


# ===========================================================================
# 3. /retrieval_check_hybrid — HTTP-layer auth
# ===========================================================================

class TestRetrievalCheckHybridAuth:

    def test_no_key_returns_403(self):
        with TestClient(_build_app()) as client:
            resp = client.get("/retrieval_check_hybrid?query=hello")
        assert resp.status_code == 403

    def test_wrong_key_returns_403(self):
        with TestClient(_build_app()) as client:
            resp = client.get(
                "/retrieval_check_hybrid?query=hello",
                headers={"X-Admin-Key": "wrong"},
            )
        assert resp.status_code == 403

    def test_unconfigured_server_returns_503(self):
        with TestClient(_build_app(admin_api_key="")) as client:
            resp = client.get(
                "/retrieval_check_hybrid?query=hello",
                headers={"X-Admin-Key": ADMIN_KEY},
            )
        assert resp.status_code == 503


# ===========================================================================
# 4. /retrieval_check_hybrid — restriction filter at route level
# ===========================================================================

class TestRetrievalCheckHybridRestrictionFilter:

    def _make_mock_node(self, raw_docs: list) -> MagicMock:
        """Node mock whose _filter_restricted_docs applies the real restriction logic."""
        from app.services.utils.RAG_pipeline.workflow.nodes import _is_publication_restricted

        node = MagicMock()
        node._hybrid_retrieve.return_value = raw_docs
        node._filter_restricted_docs.side_effect = lambda docs: [
            d for d in docs if not _is_publication_restricted(d.get("metadata") or {})
        ]
        return node

    def test_approved_doc_returned(self):
        raw = [_hybrid_doc(APPROVED_PATH, "approved hybrid content")]
        with patch(
            "app.services.chatbot.cahtbot_router.create_workflow"
        ) as mock_cw:
            mock_cw.return_value = (MagicMock(), self._make_mock_node(raw))
            with TestClient(_build_app()) as client:
                resp = client.get(
                    "/retrieval_check_hybrid?query=garifuna",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["matches_found"] == 1
        assert body["matches"][0]["content"] == "approved hybrid content"

    def test_historia_filtered_even_with_valid_key(self):
        raw = [_hybrid_doc(HISTORIA_PATH, "historia hybrid content")]
        with patch(
            "app.services.chatbot.cahtbot_router.create_workflow"
        ) as mock_cw:
            mock_cw.return_value = (MagicMock(), self._make_mock_node(raw))
            with TestClient(_build_app()) as client:
                resp = client.get(
                    "/retrieval_check_hybrid?query=historia",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["matches_found"] == 0
        assert body["matches"] == []

    def test_mixed_results_historia_stripped(self):
        raw = [
            _hybrid_doc(HISTORIA_PATH, "historia chunk"),
            _hybrid_doc(APPROVED_PATH, "approved chunk"),
        ]
        with patch(
            "app.services.chatbot.cahtbot_router.create_workflow"
        ) as mock_cw:
            mock_cw.return_value = (MagicMock(), self._make_mock_node(raw))
            with TestClient(_build_app()) as client:
                resp = client.get(
                    "/retrieval_check_hybrid?query=garifuna",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
        assert resp.status_code == 200
        body = resp.json()
        assert body["matches_found"] == 1
        assert body["matches"][0]["content"] == "approved chunk"

    def test_historia_null_metadata_still_filtered(self):
        raw_doc = {
            "page_content": "historia content",
            "metadata": {
                "source": HISTORIA_PATH,
                "publication_allowed": None,
                "review_state": None,
            },
            "retrieval_source": "vector",
            "hybrid_score": 0.8,
            "vector_score": 0.9,
            "keyword_score": 0.5,
        }
        with patch(
            "app.services.chatbot.cahtbot_router.create_workflow"
        ) as mock_cw:
            mock_cw.return_value = (MagicMock(), self._make_mock_node([raw_doc]))
            with TestClient(_build_app()) as client:
                resp = client.get(
                    "/retrieval_check_hybrid?query=historia",
                    headers={"X-Admin-Key": ADMIN_KEY},
                )
        assert resp.status_code == 200
        assert resp.json()["matches_found"] == 0


# ===========================================================================
# 5. /upload_docs — HTTP-layer auth
# ===========================================================================

class TestUploadDocsAuth:

    def test_no_key_returns_403(self):
        with TestClient(_build_app()) as client:
            resp = client.post(
                "/upload_docs",
                files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
            )
        assert resp.status_code == 403

    def test_wrong_key_returns_403(self):
        with TestClient(_build_app()) as client:
            resp = client.post(
                "/upload_docs",
                headers={"X-Admin-Key": "wrong-key"},
                files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
            )
        assert resp.status_code == 403

    def test_unconfigured_server_returns_503(self):
        with TestClient(_build_app(admin_api_key="")) as client:
            resp = client.post(
                "/upload_docs",
                headers={"X-Admin-Key": ADMIN_KEY},
                files={"file": ("test.txt", io.BytesIO(b"content"), "text/plain")},
            )
        assert resp.status_code == 503

    def test_no_file_returns_422(self):
        with TestClient(_build_app()) as client:
            resp = client.post(
                "/upload_docs",
                headers={"X-Admin-Key": ADMIN_KEY},
            )
        assert resp.status_code == 422

    def test_disallowed_extension_returns_400(self):
        with TestClient(_build_app()) as client:
            resp = client.post(
                "/upload_docs",
                headers={"X-Admin-Key": ADMIN_KEY},
                files={"file": ("evil.py", io.BytesIO(b"import os"), "text/plain")},
                data={"confirmed": "true"},
            )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]


# ===========================================================================
# 6. /upload_docs — authorized upload with mocked pipeline
# ===========================================================================

class TestUploadDocsAuthorized:

    def _do_upload(
        self,
        filename: str,
        content: bytes = b"test content",
        confirmed: str = "true",
        ids: list = None,
    ) -> tuple:
        """Run an authorized upload; return (response, pipeline_kwargs_list)."""
        captured = []

        def fake_pipeline(file_paths, original_filename=None, **kwargs):
            captured.append({"file_paths": file_paths, "original_filename": original_filename, **kwargs})
            return ids if ids is not None else ["fake-id-1"]

        data = {}
        if confirmed is not None:
            data["confirmed"] = confirmed

        with (
            patch(
                "app.services.chatbot.cahtbot_router.upload_docs_pipeline",
                side_effect=fake_pipeline,
            ),
            patch("os.path.exists", return_value=False),
        ):
            with TestClient(_build_app()) as client:
                resp = client.post(
                    "/upload_docs",
                    headers={"X-Admin-Key": ADMIN_KEY},
                    files={"file": (filename, io.BytesIO(content), "text/plain")},
                    data=data,
                )

        return resp, captured

    def test_authorized_txt_upload_succeeds(self):
        resp, captured = self._do_upload("document.txt")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["filename"] == "document.txt"
        assert len(captured) == 1

    def test_authorized_pdf_upload_succeeds(self):
        resp, _ = self._do_upload("document.pdf", b"%PDF-content")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_success_message_reflects_approval(self):
        resp, _ = self._do_upload("document.txt")
        assert resp.json()["message"] == "Document uploaded and approved for Yurumein AI."

    def test_success_includes_chunks_indexed(self):
        resp, _ = self._do_upload("document.txt", ids=["id1", "id2", "id3"])
        assert resp.json()["chunks_indexed"] == 3

    def test_missing_confirmation_returns_422(self):
        """No confirmed field → 422 before the pipeline is called."""
        captured = []

        def fake_pipeline(*a, **kw):
            captured.append(True)
            return ["id"]

        with (
            patch("app.services.chatbot.cahtbot_router.upload_docs_pipeline", side_effect=fake_pipeline),
            patch("os.path.exists", return_value=False),
        ):
            with TestClient(_build_app()) as client:
                resp = client.post(
                    "/upload_docs",
                    headers={"X-Admin-Key": ADMIN_KEY},
                    files={"file": ("doc.txt", io.BytesIO(b"content"), "text/plain")},
                )
        assert resp.status_code == 422
        assert "confirmation" in resp.json()["detail"].lower()
        assert captured == [], "Pipeline must not be called without confirmation"

    def test_false_confirmation_returns_422(self):
        resp, captured = self._do_upload("document.txt", confirmed="false")
        assert resp.status_code == 422
        assert captured == []

    def test_pipeline_called_with_confirmed_true(self):
        """Route must pass confirmed=True to the pipeline."""
        _, captured = self._do_upload("document.txt")
        assert len(captured) == 1
        assert captured[0].get("confirmed") is True

    def test_historia_filename_rejected_before_pipeline(self):
        """Known restricted source must be blocked even with confirmation."""
        resp, captured = self._do_upload(
            "Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt"
        )
        assert resp.status_code == 400
        assert "restricted" in resp.json()["detail"].lower()
        assert captured == [], "Pipeline must not be called for restricted source"

    def test_historia_filename_case_insensitive(self):
        resp, captured = self._do_upload("historia_es_en_draft.txt")
        assert resp.status_code == 400
        assert captured == []

    def test_zero_chunks_returns_400(self):
        """Pipeline returns [] → no content extracted → 400, not success."""
        resp, _ = self._do_upload("empty.txt", ids=[])
        assert resp.status_code == 400
        assert "extracted" in resp.json()["detail"].lower() or "empty" in resp.json()["detail"].lower()

    def test_server_generated_path_does_not_contain_client_filename(self):
        """The server must not use the client-supplied path for disk I/O."""
        resp, _ = self._do_upload("../../etc/passwd")
        # .passwd is not in ALLOWED_EXTENSIONS → 400
        assert resp.status_code == 400

    def test_server_generated_path_does_not_contain_client_stem(self):
        """UUID-based filename: pipeline receives a UUID path, not the client stem."""
        _, captured = self._do_upload("my_secret_name.txt")
        assert len(captured) == 1
        disk_path = captured[0]["file_paths"][0]
        assert disk_path.endswith(".txt")
        assert "my_secret_name" not in disk_path
        import re
        assert re.match(r"^[0-9a-f]{32}\.txt$", os.path.basename(disk_path))

    def test_pipeline_not_called_without_key(self):
        captured = []

        with patch(
            "app.services.chatbot.cahtbot_router.upload_docs_pipeline",
            side_effect=lambda fp, **kw: captured.append(fp),
        ):
            with TestClient(_build_app()) as client:
                client.post(
                    "/upload_docs",
                    files={"file": ("doc.txt", io.BytesIO(b"content"), "text/plain")},
                    data={"confirmed": "true"},
                )
        assert captured == [], "Pipeline must not be called on unauthenticated request"
