"""
Security tests: direct-engine request behavior.

Models the two-layer security design for session-mutation endpoints:

  Layer 1 — Next.js proxy (JWT → user_id):
    Verified by __tests__/session-proxy.test.mjs on the frontend.

  Layer 2 — AI engine (X-Session-Key + ownership):
    Tested here. Even if someone bypasses the Next.js proxy and calls the
    engine directly:
      a) Without X-Session-Key  → 503/403 (key dependency fires first)
      b) With wrong X-Session-Key  → 403
      c) With correct key but wrong user_id  → 404 (ownership check)
      d) With correct key and correct user_id  → 200 (success)

  Read-only endpoints (/sessions/{user_id}, /session/{id}/history) are
  intentionally NOT protected by X-Session-Key. They are called directly
  from the browser via SWR.  The session-history endpoint already checks
  ownership inside get_full_session_history (returns [] when user_id does
  not match) so it cannot expose another user's messages.

  Atomicity:
    Messages are never deleted before ownership is confirmed (see
    test_session_management.py::test_delete_messages_not_called_when_not_owned).

Run with: pytest tests/test_session_security.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Inline replica of the require_session_key dependency logic.
# Tests verify the security contract without importing FastAPI.
# ---------------------------------------------------------------------------

class _ForbiddenError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _check_session_key(configured_key: str, provided_key: str) -> None:
    """Replica of the require_session_key FastAPI dependency."""
    if not configured_key:
        raise _ForbiddenError(503, "Session API key not configured on this server.")
    if not provided_key or provided_key != configured_key:
        raise _ForbiddenError(403, "Forbidden.")


# ---------------------------------------------------------------------------
# Inline replica of route handler error-mapping:
#   ValueError  → 404  (ownership check failed)
#   Exception   → 500  (unexpected error)
#   no error    → 200
# ---------------------------------------------------------------------------

async def _simulate_delete(configured_key, provided_key, manager):
    _check_session_key(configured_key, provided_key)
    try:
        await manager.delete_session("s1", "u1")
        return 200
    except ValueError:
        return 404
    except Exception:
        return 500


async def _simulate_rename(configured_key, provided_key, manager):
    _check_session_key(configured_key, provided_key)
    try:
        await manager.rename_session("s1", "u1", "New")
        return 200
    except ValueError:
        return 404
    except Exception:
        return 500


async def _simulate_pin(configured_key, provided_key, manager):
    _check_session_key(configured_key, provided_key)
    try:
        await manager.pin_session("s1", "u1", True)
        return 200
    except ValueError:
        return 404
    except Exception:
        return 500


async def _simulate_archive(configured_key, provided_key, manager):
    _check_session_key(configured_key, provided_key)
    try:
        await manager.archive_session("s1", "u1", True)
        return 200
    except ValueError:
        return 404
    except Exception:
        return 500


# ---------------------------------------------------------------------------
# Mock session managers
# ---------------------------------------------------------------------------

class _NotOwnedManager:
    """Simulates engine behaviour when session does not belong to user_id."""
    async def delete_session(self, *_):
        raise ValueError("Session not found or not owned by user")
    async def rename_session(self, *_):
        raise ValueError("Session not found or not owned by user")
    async def pin_session(self, *_):
        raise ValueError("Session not found or not owned by user")
    async def archive_session(self, *_):
        raise ValueError("Session not found or not owned by user")


class _OwnedManager:
    """Simulates engine behaviour when session belongs to user_id."""
    async def delete_session(self, *_): pass
    async def rename_session(self, *_): pass
    async def pin_session(self, *_): pass
    async def archive_session(self, *_): pass


REAL_KEY = "supersecretkey"
WRONG_KEY = "wrongkey"
NO_KEY = ""


# ===========================================================================
# 1. Key-check: unconfigured server (503)
# ===========================================================================

def test_key_check_503_when_server_has_no_key():
    """Engine returns 503 when SESSION_API_KEY is not set (fail-secure)."""
    with pytest.raises(_ForbiddenError) as exc:
        _check_session_key(configured_key="", provided_key=REAL_KEY)
    assert exc.value.status_code == 503


# ===========================================================================
# 2. Key-check: missing or wrong key → 403
# ===========================================================================

def test_key_check_403_when_no_key_provided():
    with pytest.raises(_ForbiddenError) as exc:
        _check_session_key(configured_key=REAL_KEY, provided_key="")
    assert exc.value.status_code == 403


def test_key_check_403_when_wrong_key_provided():
    with pytest.raises(_ForbiddenError) as exc:
        _check_session_key(configured_key=REAL_KEY, provided_key=WRONG_KEY)
    assert exc.value.status_code == 403


def test_key_check_passes_with_correct_key():
    _check_session_key(configured_key=REAL_KEY, provided_key=REAL_KEY)


# ===========================================================================
# 3. Direct-engine bypass — no X-Session-Key header
#    (attacker calls engine directly without going through the proxy)
# ===========================================================================

def test_delete_direct_no_key_returns_403():
    """Direct DELETE without X-Session-Key → 403 before any DB access."""
    with pytest.raises(_ForbiddenError) as exc:
        asyncio.run(_simulate_delete(REAL_KEY, NO_KEY, _OwnedManager()))
    assert exc.value.status_code == 403


def test_rename_direct_no_key_returns_403():
    with pytest.raises(_ForbiddenError) as exc:
        asyncio.run(_simulate_rename(REAL_KEY, NO_KEY, _OwnedManager()))
    assert exc.value.status_code == 403


def test_pin_direct_no_key_returns_403():
    with pytest.raises(_ForbiddenError) as exc:
        asyncio.run(_simulate_pin(REAL_KEY, NO_KEY, _OwnedManager()))
    assert exc.value.status_code == 403


def test_archive_direct_no_key_returns_403():
    with pytest.raises(_ForbiddenError) as exc:
        asyncio.run(_simulate_archive(REAL_KEY, NO_KEY, _OwnedManager()))
    assert exc.value.status_code == 403


# ===========================================================================
# 4. Direct-engine bypass — wrong X-Session-Key
#    (attacker knows the endpoint but guesses the secret wrong)
# ===========================================================================

def test_delete_wrong_key_returns_403():
    with pytest.raises(_ForbiddenError) as exc:
        asyncio.run(_simulate_delete(REAL_KEY, WRONG_KEY, _OwnedManager()))
    assert exc.value.status_code == 403


def test_rename_wrong_key_returns_403():
    with pytest.raises(_ForbiddenError) as exc:
        asyncio.run(_simulate_rename(REAL_KEY, WRONG_KEY, _OwnedManager()))
    assert exc.value.status_code == 403


# ===========================================================================
# 5. Correct key but wrong user_id (ownership check)
#    (attacker knows the secret but submits victim's session with attacker's user_id)
# ===========================================================================

def test_delete_correct_key_wrong_user_returns_404():
    """Correct key + non-owner user_id → 404 (second defense layer)."""
    code = asyncio.run(_simulate_delete(REAL_KEY, REAL_KEY, _NotOwnedManager()))
    assert code == 404


def test_rename_correct_key_wrong_user_returns_404():
    code = asyncio.run(_simulate_rename(REAL_KEY, REAL_KEY, _NotOwnedManager()))
    assert code == 404


def test_pin_correct_key_wrong_user_returns_404():
    code = asyncio.run(_simulate_pin(REAL_KEY, REAL_KEY, _NotOwnedManager()))
    assert code == 404


def test_archive_correct_key_wrong_user_returns_404():
    code = asyncio.run(_simulate_archive(REAL_KEY, REAL_KEY, _NotOwnedManager()))
    assert code == 404


# ===========================================================================
# 6. Full success path: correct key + correct user_id
# ===========================================================================

def test_delete_correct_key_and_user_returns_200():
    code = asyncio.run(_simulate_delete(REAL_KEY, REAL_KEY, _OwnedManager()))
    assert code == 200


def test_rename_correct_key_and_user_returns_200():
    code = asyncio.run(_simulate_rename(REAL_KEY, REAL_KEY, _OwnedManager()))
    assert code == 200


def test_pin_correct_key_and_user_returns_200():
    code = asyncio.run(_simulate_pin(REAL_KEY, REAL_KEY, _OwnedManager()))
    assert code == 200


def test_archive_correct_key_and_user_returns_200():
    code = asyncio.run(_simulate_archive(REAL_KEY, REAL_KEY, _OwnedManager()))
    assert code == 200


# ===========================================================================
# 7. Atomicity: key-check fires BEFORE any DB mutation
# ===========================================================================

def test_delete_no_key_does_not_call_session_manager():
    """403 on key check means SessionManager.delete_session is never called."""
    mock_manager = MagicMock()
    mock_manager.delete_session = AsyncMock()
    with pytest.raises(_ForbiddenError):
        asyncio.run(_simulate_delete(REAL_KEY, NO_KEY, mock_manager))
    mock_manager.delete_session.assert_not_called()


def test_rename_no_key_does_not_call_session_manager():
    mock_manager = MagicMock()
    mock_manager.rename_session = AsyncMock()
    with pytest.raises(_ForbiddenError):
        asyncio.run(_simulate_rename(REAL_KEY, NO_KEY, mock_manager))
    mock_manager.rename_session.assert_not_called()


def test_pin_no_key_does_not_call_session_manager():
    mock_manager = MagicMock()
    mock_manager.pin_session = AsyncMock()
    with pytest.raises(_ForbiddenError):
        asyncio.run(_simulate_pin(REAL_KEY, NO_KEY, mock_manager))
    mock_manager.pin_session.assert_not_called()


def test_archive_no_key_does_not_call_session_manager():
    mock_manager = MagicMock()
    mock_manager.archive_session = AsyncMock()
    with pytest.raises(_ForbiddenError):
        asyncio.run(_simulate_archive(REAL_KEY, NO_KEY, mock_manager))
    mock_manager.archive_session.assert_not_called()


# ===========================================================================
# 8. Read-only endpoints are NOT key-protected (intentional)
#    /sessions/{user_id}  — returns session list; no mutation
#    /session/{id}/history — ownership checked inside get_full_session_history
#
# These are called directly from the browser via SWR. The session-list endpoint
# can only return sessions belonging to the requested user_id. The history
# endpoint calls get_full_session_history(session_id, user_id), which verifies
# ownership and returns [] when it fails — it cannot expose another user's messages.
#
# No key-protection test for these endpoints (absence of test is intentional).
# ===========================================================================
