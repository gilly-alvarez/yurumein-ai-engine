"""
Integration tests — Phase 1 Knowledge Learning System
Verifies real HTTP flows via FastAPI TestClient without touching
OpenAI, PGVector, or PostgreSQL.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI


# ---------------------------------------------------------------------------
# Shared app fixture — builds a minimal FastAPI app with just the two routers
# under test so tests are independent of the full main.py startup chain.
# ---------------------------------------------------------------------------

def _build_app(admin_api_key: str = "test-admin-key-xyz") -> FastAPI:
    """Return a FastAPI instance with knowledge-gap + chatbot routers mounted."""
    from app.core.config import settings

    settings.ADMIN_API_KEY = admin_api_key

    app = FastAPI()

    from app.services.knowledge_gaps.knowledge_gap_router import router as gap_router
    app.include_router(gap_router)

    return app


# ---------------------------------------------------------------------------
# 1. Admin API authentication — real HTTP layer
# ---------------------------------------------------------------------------

class TestAdminAPIHTTP:
    """Verify 403/503/200 responses from actual HTTP requests via TestClient."""

    def test_list_gaps_no_key_returns_403(self):
        app = _build_app(admin_api_key="real-secret")
        with TestClient(app) as client:
            resp = client.get("/knowledge-gaps")
        assert resp.status_code == 403

    def test_list_gaps_wrong_key_returns_403(self):
        app = _build_app(admin_api_key="real-secret")
        with TestClient(app) as client:
            resp = client.get("/knowledge-gaps", headers={"X-Admin-Key": "wrong"})
        assert resp.status_code == 403

    def test_list_gaps_unconfigured_key_returns_503(self):
        app = _build_app(admin_api_key="")
        with TestClient(app) as client:
            resp = client.get("/knowledge-gaps", headers={"X-Admin-Key": "anything"})
        assert resp.status_code == 503

    def test_report_no_key_returns_403(self):
        app = _build_app(admin_api_key="real-secret")
        with TestClient(app) as client:
            resp = client.get("/knowledge-gaps/report")
        assert resp.status_code == 403

    def test_patch_no_key_returns_403(self):
        app = _build_app(admin_api_key="real-secret")
        with TestClient(app) as client:
            resp = client.patch("/knowledge-gaps/1", json={"status": "resolved"})
        assert resp.status_code == 403

    def test_delete_no_key_returns_403(self):
        app = _build_app(admin_api_key="real-secret")
        with TestClient(app) as client:
            resp = client.delete("/knowledge-gaps/1")
        assert resp.status_code == 403

    def test_list_gaps_correct_key_reaches_manager(self):
        """Correct key passes auth; manager is called (mocked to avoid DB)."""
        app = _build_app(admin_api_key="real-secret")

        mock_mgr = MagicMock()
        mock_mgr.list_gaps = AsyncMock(return_value={"total": 0, "items": []})

        with patch(
            "app.services.knowledge_gaps.knowledge_gap_router.get_gap_manager",
            return_value=mock_mgr,
        ):
            with TestClient(app) as client:
                resp = client.get(
                    "/knowledge-gaps", headers={"X-Admin-Key": "real-secret"}
                )
        assert resp.status_code == 200
        assert resp.json() == {"total": 0, "items": []}
        mock_mgr.list_gaps.assert_awaited_once()

    def test_report_correct_key_reaches_manager(self):
        app = _build_app(admin_api_key="real-secret")

        mock_mgr = MagicMock()
        mock_mgr.get_report = AsyncMock(return_value={"new_this_week": 0})

        with patch(
            "app.services.knowledge_gaps.knowledge_gap_router.get_gap_manager",
            return_value=mock_mgr,
        ):
            with TestClient(app) as client:
                resp = client.get(
                    "/knowledge-gaps/report", headers={"X-Admin-Key": "real-secret"}
                )
        assert resp.status_code == 200
        assert resp.json()["new_this_week"] == 0

    def test_patch_correct_key_404_on_missing_gap(self):
        app = _build_app(admin_api_key="real-secret")

        mock_mgr = MagicMock()
        mock_mgr.update_gap = AsyncMock(return_value=None)

        with patch(
            "app.services.knowledge_gaps.knowledge_gap_router.get_gap_manager",
            return_value=mock_mgr,
        ):
            with TestClient(app) as client:
                resp = client.patch(
                    "/knowledge-gaps/999",
                    headers={"X-Admin-Key": "real-secret"},
                    json={"status": "resolved"},
                )
        assert resp.status_code == 404

    def test_delete_correct_key_404_on_missing_gap(self):
        app = _build_app(admin_api_key="real-secret")

        mock_mgr = MagicMock()
        mock_mgr.delete_gap = AsyncMock(return_value=False)

        with patch(
            "app.services.knowledge_gaps.knowledge_gap_router.get_gap_manager",
            return_value=mock_mgr,
        ):
            with TestClient(app) as client:
                resp = client.delete(
                    "/knowledge-gaps/999",
                    headers={"X-Admin-Key": "real-secret"},
                )
        assert resp.status_code == 404

    def test_invalid_status_returns_400(self):
        app = _build_app(admin_api_key="real-secret")

        mock_mgr = MagicMock()
        mock_mgr.list_gaps = AsyncMock(return_value={"total": 0, "items": []})

        with patch(
            "app.services.knowledge_gaps.knowledge_gap_router.get_gap_manager",
            return_value=mock_mgr,
        ):
            with TestClient(app) as client:
                resp = client.get(
                    "/knowledge-gaps?status=not_a_real_status",
                    headers={"X-Admin-Key": "real-secret"},
                )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 2. /chat and /chatstrem — gap detection failure does not break response
# ---------------------------------------------------------------------------

class TestChatEndpointResilience:
    """Verify /chat still returns a response when gap detection raises an exception."""

    def test_chat_completes_when_gap_detection_raises(self):
        """BackgroundTask exception must never surface to the HTTP caller."""
        from fastapi import FastAPI
        from app.services.chatbot.cahtbot_router import router as chat_router

        app = FastAPI()
        app.include_router(chat_router)

        fake_result = {
            "response": "The Garífuna people have a rich history.",
            "retrieved_docs": [{"hybrid_score": 0.9, "content": "..."}],
            "context": "Some context",
            "citations_stripped": 0,
        }

        with (
            patch(
                "app.services.chatbot.cahtbot_router.run_workflow",
                new=AsyncMock(return_value=fake_result),
            ),
            patch(
                "app.services.chatbot.cahtbot_router.get_gap_manager",
            ) as mock_get_mgr,
            patch(
                "app.services.chatbot.cahtbot_router.SessionManager"
            ) as mock_sm_cls,
        ):
            # Make gap manager's record_gap raise to confirm isolation
            mock_mgr = MagicMock()
            mock_mgr.record_gap = AsyncMock(side_effect=RuntimeError("DB down"))
            mock_get_mgr.return_value = mock_mgr

            # Mock SessionManager instance methods
            mock_sm = AsyncMock()
            mock_sm.save_message = AsyncMock()
            mock_sm.create_session = AsyncMock()
            mock_sm.update_session = AsyncMock()
            mock_sm_cls.return_value = mock_sm

            with (
                patch(
                    "app.services.chatbot.cahtbot_router.create_workflow"
                ) as mock_cw,
            ):
                mock_node = MagicMock()
                mock_node.generate_session_title = AsyncMock(return_value="Test Session")
                mock_wf = MagicMock()
                mock_cw.return_value = (mock_wf, mock_node)

                with TestClient(app) as client:
                    resp = client.post(
                        "/chat",
                        data={"user_prompt": "Tell me about the Garífuna people", "user_id": "test-user"},
                    )

        assert resp.status_code == 200
        body = resp.json()
        assert "response" in body
        assert body["response"] == "The Garífuna people have a rich history."

    def test_chat_response_unaffected_by_no_gap_signal(self):
        """When no gap signals fire, /chat still returns normally."""
        from fastapi import FastAPI
        from app.services.chatbot.cahtbot_router import router as chat_router

        app = FastAPI()
        app.include_router(chat_router)

        # Good evidence — no gap should be recorded
        fake_result = {
            "response": "The Garífuna are an Afro-Indigenous people.",
            "retrieved_docs": [{"hybrid_score": 0.85, "content": "..."}],
            "context": "RAG context here",
            "citations_stripped": 0,
        }

        with (
            patch(
                "app.services.chatbot.cahtbot_router.run_workflow",
                new=AsyncMock(return_value=fake_result),
            ),
            patch(
                "app.services.chatbot.cahtbot_router.get_gap_manager",
            ) as mock_get_mgr,
            patch(
                "app.services.chatbot.cahtbot_router.SessionManager"
            ) as mock_sm_cls,
            patch(
                "app.services.chatbot.cahtbot_router.create_workflow"
            ) as mock_cw,
        ):
            mock_mgr = MagicMock()
            mock_mgr.record_gap = AsyncMock()
            mock_get_mgr.return_value = mock_mgr

            mock_sm = AsyncMock()
            mock_sm.save_message = AsyncMock()
            mock_sm.create_session = AsyncMock()
            mock_sm.update_session = AsyncMock()
            mock_sm_cls.return_value = mock_sm

            mock_node = MagicMock()
            mock_node.generate_session_title = AsyncMock(return_value="Session")
            mock_cw.return_value = (MagicMock(), mock_node)

            with TestClient(app) as client:
                resp = client.post(
                    "/chat",
                    data={"user_prompt": "Tell me about Garífuna music", "user_id": "u1"},
                )

        assert resp.status_code == 200
        # record_gap must NOT have been called (no gap signals)
        mock_mgr.record_gap.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. PII scrubbing in normalization
# ---------------------------------------------------------------------------

class TestNormalizePIIScrubbing:
    """Verify that PII patterns are removed before storing normalized questions."""

    def test_email_removed_from_normalized(self):
        from app.services.utils.RAG_pipeline.DB.knowledge_gap_manager import KnowledgeGapManager
        result = KnowledgeGapManager.normalize("Contact me at user@example.com about garifuna")
        assert "@" not in result
        assert "example" not in result or "garifuna" in result

    def test_long_digit_string_removed(self):
        from app.services.utils.RAG_pipeline.DB.knowledge_gap_manager import KnowledgeGapManager
        result = KnowledgeGapManager.normalize("My ID 123456789 and garifuna history")
        assert "123456789" not in result
        assert "garifuna" in result

    def test_phone_number_removed(self):
        from app.services.utils.RAG_pipeline.DB.knowledge_gap_manager import KnowledgeGapManager
        result = KnowledgeGapManager.normalize("Call 555-867-5309 for garifuna lessons")
        assert "555" not in result
        assert "garifuna" in result

    def test_normal_question_unaffected(self):
        from app.services.utils.RAG_pipeline.DB.knowledge_gap_manager import KnowledgeGapManager
        result = KnowledgeGapManager.normalize("What is the history of garifuna settlements")
        assert "garifuna" in result
        assert "history" in result
        assert "settlements" in result


# ---------------------------------------------------------------------------
# 4. Report frequency_note present
# ---------------------------------------------------------------------------

class TestReportMetricLabeling:
    """Verify get_report() output includes accurate frequency_note."""

    async def test_report_includes_frequency_note(self):
        from app.services.utils.RAG_pipeline.DB.knowledge_gap_manager import KnowledgeGapManager
        from unittest.mock import AsyncMock, MagicMock, patch

        with patch(
            "app.services.utils.RAG_pipeline.DB.knowledge_gap_manager.SessionManager"
        ):
            mgr = KnowledgeGapManager()

        mock_db = AsyncMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)
        mock_db.execute = AsyncMock(return_value=MagicMock(
            all=MagicMock(return_value=[]),
            scalar=MagicMock(return_value=0),
            scalars=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[]))),
        ))
        mgr._sm = MagicMock()
        mgr._sm.AsyncSessionLocal = MagicMock(return_value=mock_db)

        report = await mgr.get_report()

        assert "frequency_note" in report
        note = report["frequency_note"]
        assert "detection events" in note
        assert "not unique users" in note
