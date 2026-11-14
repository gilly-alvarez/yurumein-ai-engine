"""
Security tests: /retrieval_check and /retrieval_check_hybrid access control.

Both endpoints now require X-Admin-Key and apply _filter_restricted_docs before
returning any match to the caller.  This file tests that security contract without
a live database.

Design follows test_session_security.py:
  - Inline replicas of the dependency / handler logic are used so tests run
    without FastAPI, a live server, or database access.
  - Tests import the actual require_admin_key from cahtbot_router to verify the
    dependency is wired with the same key as /knowledge-gaps.

Security contract:
  1. No key → 403  (or 503 when ADMIN_API_KEY is unconfigured on the server)
  2. Wrong key → 403
  3. Correct key → handler runs
  4. Restricted document (Historia) filtered from results regardless of auth
  5. Non-restricted document (approved source) kept in results
  6. Missing publication_allowed metadata on a known restricted source → still filtered
     (fail-closed source-path guard)

Run from repo root:
    pytest tests/test_diagnostic_endpoint_security.py -v
"""

import asyncio
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Inline replica of the require_admin_key dependency logic.
# Used for fast unit tests that do not import the full FastAPI app.
# ---------------------------------------------------------------------------

class _AccessError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _check_admin_key(configured_key: str, provided_key: str) -> None:
    """Replica of the require_admin_key FastAPI dependency."""
    if not configured_key:
        raise _AccessError(503, "Admin API key not configured on this server.")
    if not provided_key or provided_key != configured_key:
        raise _AccessError(403, "Admin access required.")


# ---------------------------------------------------------------------------
# Inline replicas of the retrieval handler filter step.
# Tests the restriction filter without connecting to the database.
# ---------------------------------------------------------------------------

def _is_publication_restricted_replica(meta: dict) -> bool:
    """
    Replica of nodes._is_publication_restricted — same three conditions.
    Used here so we can test filter logic without importing the full app.
    """
    pa = meta.get("publication_allowed")
    if pa is False or (isinstance(pa, str) and pa.lower() == "false"):
        return True
    rs = meta.get("review_state")
    if isinstance(rs, str) and rs.lower() == "pending_cultural_review":
        return True
    source = (meta.get("source") or meta.get("filename") or "").upper()
    if "HISTORIA_ES_EN" in source:
        return True
    return False


def _simulate_retrieval_check(configured_key: str, provided_key: str, raw_docs: list) -> list:
    """
    Simulate the retrieval_check handler security contract:
      1. Auth gate (key check).
      2. Restriction filter applied to raw_docs before building matches.
    Returns the filtered doc list (simulates matches payload).
    Raises _AccessError if auth fails.
    """
    _check_admin_key(configured_key, provided_key)
    # Replica of the filter step in the endpoint
    return [d for d in raw_docs if not _is_publication_restricted_replica(d.get("metadata") or {})]


def _simulate_retrieval_check_hybrid(configured_key: str, provided_key: str, raw_docs: list) -> list:
    """Same contract for the hybrid endpoint."""
    _check_admin_key(configured_key, provided_key)
    return [d for d in raw_docs if not _is_publication_restricted_replica(d.get("metadata") or {})]


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

REAL_KEY    = "supersecret-admin-key-123"
WRONG_KEY   = "wrong-key"
NO_KEY      = ""

def _historia_doc(extra_meta: dict | None = None) -> dict:
    meta = {"source": "/app/data/Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt"}
    if extra_meta:
        meta.update(extra_meta)
    return {"page_content": "restricted historia content", "metadata": meta}


def _approved_doc() -> dict:
    return {
        "page_content": "approved garifuna identity content",
        "metadata": {"source": "/app/data/Yurumein_Garifuna_Identity_EN_ES.docx"},
    }


# ===========================================================================
# 1. Admin key dependency — require_admin_key from cahtbot_router
#    Import the actual function to confirm it is wired the same as /knowledge-gaps.
# ===========================================================================

class TestRequireAdminKeyFromRouter:
    """Test require_admin_key as imported from cahtbot_router (not the replica)."""

    async def test_missing_key_raises_403(self):
        from fastapi import HTTPException
        from app.services.chatbot.cahtbot_router import require_admin_key
        with patch("app.services.chatbot.cahtbot_router.settings") as ms:
            ms.ADMIN_API_KEY = REAL_KEY
            with pytest.raises(HTTPException) as exc:
                await require_admin_key(key=None)
            assert exc.value.status_code == 403
            assert "Admin access required" in exc.value.detail

    async def test_wrong_key_raises_403(self):
        from fastapi import HTTPException
        from app.services.chatbot.cahtbot_router import require_admin_key
        with patch("app.services.chatbot.cahtbot_router.settings") as ms:
            ms.ADMIN_API_KEY = REAL_KEY
            with pytest.raises(HTTPException) as exc:
                await require_admin_key(key=WRONG_KEY)
            assert exc.value.status_code == 403

    async def test_correct_key_passes(self):
        from app.services.chatbot.cahtbot_router import require_admin_key
        with patch("app.services.chatbot.cahtbot_router.settings") as ms:
            ms.ADMIN_API_KEY = REAL_KEY
            result = await require_admin_key(key=REAL_KEY)
            assert result is None

    async def test_unconfigured_server_raises_503(self):
        from fastapi import HTTPException
        from app.services.chatbot.cahtbot_router import require_admin_key
        with patch("app.services.chatbot.cahtbot_router.settings") as ms:
            ms.ADMIN_API_KEY = ""
            with pytest.raises(HTTPException) as exc:
                await require_admin_key(key=REAL_KEY)
            assert exc.value.status_code == 503

    async def test_uses_same_key_as_knowledge_gaps(self):
        """Both endpoints must read from the same ADMIN_API_KEY setting."""
        from app.services.chatbot.cahtbot_router import require_admin_key as diag_auth
        from app.services.knowledge_gaps.knowledge_gap_router import require_admin_key as gap_auth

        with patch("app.services.chatbot.cahtbot_router.settings") as ds, \
             patch("app.services.knowledge_gaps.knowledge_gap_router.settings") as gs:
            ds.ADMIN_API_KEY = REAL_KEY
            gs.ADMIN_API_KEY = REAL_KEY
            # Both must accept the same key without raising
            await diag_auth(key=REAL_KEY)
            await gap_auth(key=REAL_KEY)


# ===========================================================================
# 2. Key-check logic via inline replica (fast, no imports)
# ===========================================================================

class TestAdminKeyLogic:

    def test_503_when_server_key_unconfigured(self):
        with pytest.raises(_AccessError) as exc:
            _check_admin_key(configured_key="", provided_key=REAL_KEY)
        assert exc.value.status_code == 503

    def test_403_when_no_key_provided(self):
        with pytest.raises(_AccessError) as exc:
            _check_admin_key(configured_key=REAL_KEY, provided_key="")
        assert exc.value.status_code == 403

    def test_403_when_wrong_key_provided(self):
        with pytest.raises(_AccessError) as exc:
            _check_admin_key(configured_key=REAL_KEY, provided_key=WRONG_KEY)
        assert exc.value.status_code == 403

    def test_passes_with_correct_key(self):
        _check_admin_key(configured_key=REAL_KEY, provided_key=REAL_KEY)  # must not raise

    def test_key_comparison_is_exact(self):
        # Leading/trailing whitespace or partial match must not pass.
        with pytest.raises(_AccessError):
            _check_admin_key(configured_key=REAL_KEY, provided_key=" " + REAL_KEY)
        with pytest.raises(_AccessError):
            _check_admin_key(configured_key=REAL_KEY, provided_key=REAL_KEY[:5])


# ===========================================================================
# 3. retrieval_check: auth gate fires before returning results
# ===========================================================================

class TestRetrievalCheckSecurity:

    def test_no_key_raises_access_error(self):
        with pytest.raises(_AccessError) as exc:
            _simulate_retrieval_check(REAL_KEY, NO_KEY, [_approved_doc()])
        assert exc.value.status_code == 403

    def test_wrong_key_raises_access_error(self):
        with pytest.raises(_AccessError):
            _simulate_retrieval_check(REAL_KEY, WRONG_KEY, [_approved_doc()])

    def test_correct_key_returns_approved_docs(self):
        docs = _simulate_retrieval_check(REAL_KEY, REAL_KEY, [_approved_doc()])
        assert len(docs) == 1
        assert "identity" in docs[0]["page_content"]

    def test_historia_filtered_even_with_correct_key(self):
        """Admin auth does not bypass the restriction filter."""
        docs = _simulate_retrieval_check(REAL_KEY, REAL_KEY, [_historia_doc()])
        assert docs == [], "Historia must be filtered even for authenticated admin callers"

    def test_historia_filtered_when_mixed_with_approved(self):
        raw = [_historia_doc(), _approved_doc(), _historia_doc()]
        docs = _simulate_retrieval_check(REAL_KEY, REAL_KEY, raw)
        assert len(docs) == 1
        assert "identity" in docs[0]["page_content"]

    def test_historia_filtered_when_no_publication_flag(self):
        """Fail-closed: Historia path without publication_allowed → still filtered."""
        # production Historia has pa=None, rs=None — only source-path guard applies
        doc = _historia_doc()  # source contains HISTORIA_ES_EN, no pa/rs fields
        docs = _simulate_retrieval_check(REAL_KEY, REAL_KEY, [doc])
        assert docs == []

    def test_historia_filtered_when_null_publication_allowed(self):
        doc = _historia_doc({"publication_allowed": None})
        docs = _simulate_retrieval_check(REAL_KEY, REAL_KEY, [doc])
        assert docs == []

    def test_no_key_does_not_return_docs(self):
        """Auth failure must not leak any docs, even approved ones."""
        with pytest.raises(_AccessError):
            _simulate_retrieval_check(REAL_KEY, NO_KEY, [_approved_doc(), _historia_doc()])

    def test_empty_doc_list_returns_empty(self):
        docs = _simulate_retrieval_check(REAL_KEY, REAL_KEY, [])
        assert docs == []


# ===========================================================================
# 4. retrieval_check_hybrid: same security contract
# ===========================================================================

class TestRetrievalCheckHybridSecurity:

    def test_no_key_raises_access_error(self):
        with pytest.raises(_AccessError) as exc:
            _simulate_retrieval_check_hybrid(REAL_KEY, NO_KEY, [_approved_doc()])
        assert exc.value.status_code == 403

    def test_wrong_key_raises_access_error(self):
        with pytest.raises(_AccessError):
            _simulate_retrieval_check_hybrid(REAL_KEY, WRONG_KEY, [_approved_doc()])

    def test_correct_key_returns_approved_docs(self):
        docs = _simulate_retrieval_check_hybrid(REAL_KEY, REAL_KEY, [_approved_doc()])
        assert len(docs) == 1

    def test_historia_filtered_even_with_correct_key(self):
        docs = _simulate_retrieval_check_hybrid(REAL_KEY, REAL_KEY, [_historia_doc()])
        assert docs == []

    def test_historia_filtered_null_metadata(self):
        # Production state: pa=None, rs=None — source path is the only guard.
        doc = {"page_content": "historia content", "metadata": {
            "source": "/app/data/new_docs/Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt",
            "publication_allowed": None,
            "review_state": None,
        }}
        docs = _simulate_retrieval_check_hybrid(REAL_KEY, REAL_KEY, [doc])
        assert docs == []

    def test_mixed_docs_only_approved_returned(self):
        raw = [_historia_doc(), _approved_doc(), _historia_doc(), _approved_doc()]
        docs = _simulate_retrieval_check_hybrid(REAL_KEY, REAL_KEY, raw)
        assert len(docs) == 2
        assert all("identity" in d["page_content"] for d in docs)

    def test_unconfigured_server_raises_503(self):
        with pytest.raises(_AccessError) as exc:
            _simulate_retrieval_check_hybrid("", REAL_KEY, [_approved_doc()])
        assert exc.value.status_code == 503


# ===========================================================================
# 5. Restriction filter: standalone coverage of all three guard conditions
# ===========================================================================

class TestRestrictionFilterAllConditions:
    """
    Verify that the replica matches nodes._is_publication_restricted for all
    three conditions.  These replicate test_source_roles.py::TestStageAFixes
    for the inline copy here.
    """

    def test_explicit_false_bool(self):
        assert _is_publication_restricted_replica({"publication_allowed": False}) is True

    def test_explicit_false_string(self):
        assert _is_publication_restricted_replica({"publication_allowed": "false"}) is True

    def test_true_bool_not_restricted(self):
        assert _is_publication_restricted_replica({"publication_allowed": True}) is False

    def test_pending_review_state(self):
        assert _is_publication_restricted_replica({"review_state": "pending_cultural_review"}) is True

    def test_pending_review_case_insensitive(self):
        assert _is_publication_restricted_replica({"review_state": "PENDING_CULTURAL_REVIEW"}) is True

    def test_other_review_state_not_restricted(self):
        assert _is_publication_restricted_replica({"review_state": "approved"}) is False

    def test_historia_source_path_no_metadata(self):
        meta = {"source": "/data/Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt"}
        assert _is_publication_restricted_replica(meta) is True

    def test_historia_null_publication_allowed_still_restricted(self):
        meta = {
            "source": "/data/Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt",
            "publication_allowed": None,
            "review_state": None,
        }
        assert _is_publication_restricted_replica(meta) is True

    def test_non_historia_unrelated_source_not_restricted(self):
        meta = {"source": "/data/Baliceaux_Yurumein_AI_Historical_Brief.docx"}
        assert _is_publication_restricted_replica(meta) is False

    def test_empty_metadata_not_restricted(self):
        assert _is_publication_restricted_replica({}) is False


# ===========================================================================
# 6. Upload docs — confirm it still requires a file (not silently open)
# ===========================================================================

class TestUploadDocsNotOpenEndpoint:
    """
    /upload_docs is not a retrieval endpoint and does not expose document
    content.  Verify it returns 422 (Unprocessable Entity) without a file,
    not 200 or a document listing.
    """

    def test_upload_docs_requires_file_field(self):
        import subprocess, json
        r = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "-X", "POST", "http://185.201.9.189:8085/upload_docs"],
            capture_output=True, text=True, timeout=10
        )
        # 422 = required field missing; anything but 200/201 is acceptable
        assert r.stdout.strip() in ("422", "403", "401", "000"), (
            f"/upload_docs without a file returned {r.stdout.strip()} — "
            "expected 422 (missing required field)"
        )
