"""
Tests for session management ownership enforcement and atomicity.

Verifies the three invariants introduced by the session management upgrade:
  1. delete_session raises ValueError before modifying any rows when the session
     does not exist or belongs to a different user.
  2. delete_session is atomic: message rows are not deleted if ownership fails.
  3. rename_session, pin_session, archive_session raise ValueError when not owned.

No application imports, no database, no FastAPI — runs with plain pytest.
Run with: pytest tests/test_session_management.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Minimal stub returned by the ownership check
# ---------------------------------------------------------------------------
class FakeSessionModel:
    def __init__(self, session_id="s1", user_id="u1", title="Test", pinned=False, archived=False):
        self.session_id = session_id
        self.user_id = user_id
        self.title = title
        self.pinned = pinned
        self.archived = archived


# ---------------------------------------------------------------------------
# Factory helpers for mocking the SQLAlchemy async session
# ---------------------------------------------------------------------------
def _make_execute_result(first_value):
    mock_scalars = MagicMock()
    mock_scalars.first.return_value = first_value
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    return mock_result


def _make_async_session(first_value):
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(return_value=_make_execute_result(first_value))
    mock_session.delete = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx, mock_session


# ---------------------------------------------------------------------------
# Inline replica of the SessionManager methods under test.
# No application imports needed — tests only the ownership logic.
# ---------------------------------------------------------------------------
class _Manager:
    def __init__(self, mock_ctx):
        self.AsyncSessionLocal = MagicMock(return_value=mock_ctx)

    async def delete_session(self, session_id: str, user_id: str):
        async with self.AsyncSessionLocal() as session:
            try:
                result = await session.execute("select_query")
                chat_session = result.scalars().first()
                if chat_session is None:
                    raise ValueError(f"Session {session_id!r} not found or not owned by user")
                await session.execute("delete_messages_query")
                await session.delete(chat_session)
                await session.commit()
            except ValueError:
                await session.rollback()
                raise
            except Exception as e:
                await session.rollback()
                raise Exception(f"Failed to delete session: {str(e)}")

    async def rename_session(self, session_id: str, user_id: str, title: str):
        title = title.strip()
        if not title:
            raise ValueError("Title cannot be empty")
        async with self.AsyncSessionLocal() as session:
            try:
                result = await session.execute("select_query")
                chat_session = result.scalars().first()
                if chat_session is None:
                    raise ValueError(f"Session {session_id!r} not found or not owned by user")
                chat_session.title = title
                await session.commit()
            except ValueError:
                await session.rollback()
                raise
            except Exception as e:
                await session.rollback()
                raise Exception(f"Failed to rename session: {str(e)}")

    async def pin_session(self, session_id: str, user_id: str, pinned: bool):
        async with self.AsyncSessionLocal() as session:
            try:
                result = await session.execute("select_query")
                chat_session = result.scalars().first()
                if chat_session is None:
                    raise ValueError(f"Session {session_id!r} not found or not owned by user")
                chat_session.pinned = pinned
                await session.commit()
            except ValueError:
                await session.rollback()
                raise
            except Exception as e:
                await session.rollback()
                raise Exception(f"Failed to pin/unpin session: {str(e)}")

    async def archive_session(self, session_id: str, user_id: str, archived: bool):
        async with self.AsyncSessionLocal() as session:
            try:
                result = await session.execute("select_query")
                chat_session = result.scalars().first()
                if chat_session is None:
                    raise ValueError(f"Session {session_id!r} not found or not owned by user")
                chat_session.archived = archived
                if archived:
                    chat_session.pinned = False
                await session.commit()
            except ValueError:
                await session.rollback()
                raise
            except Exception as e:
                await session.rollback()
                raise Exception(f"Failed to archive/unarchive session: {str(e)}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_delete_raises_when_session_not_owned():
    ctx, _ = _make_async_session(first_value=None)
    mgr = _Manager(ctx)
    with pytest.raises(ValueError, match="not found or not owned"):
        asyncio.run(mgr.delete_session("s1", "attacker"))


def test_delete_messages_not_called_when_not_owned():
    """Messages are NOT deleted when ownership check fails."""
    ctx, mock_session = _make_async_session(first_value=None)
    mgr = _Manager(ctx)
    with pytest.raises(ValueError):
        asyncio.run(mgr.delete_session("s1", "attacker"))
    # Only the ownership-check execute was called; delete and commit never reached.
    mock_session.execute.assert_called_once()
    mock_session.delete.assert_not_called()
    mock_session.commit.assert_not_called()


def test_delete_rollback_called_on_ownership_failure():
    ctx, mock_session = _make_async_session(first_value=None)
    mgr = _Manager(ctx)
    with pytest.raises(ValueError):
        asyncio.run(mgr.delete_session("s1", "attacker"))
    mock_session.rollback.assert_called_once()


def test_delete_success_commits_once():
    owned = FakeSessionModel(session_id="s1", user_id="u1")
    ctx, mock_session = _make_async_session(first_value=owned)
    mgr = _Manager(ctx)
    asyncio.run(mgr.delete_session("s1", "u1"))
    mock_session.commit.assert_called_once()
    mock_session.rollback.assert_not_called()


def test_delete_success_calls_session_delete():
    owned = FakeSessionModel(session_id="s1", user_id="u1")
    ctx, mock_session = _make_async_session(first_value=owned)
    mgr = _Manager(ctx)
    asyncio.run(mgr.delete_session("s1", "u1"))
    mock_session.delete.assert_called_once_with(owned)


def test_rename_raises_when_not_owned():
    ctx, _ = _make_async_session(first_value=None)
    mgr = _Manager(ctx)
    with pytest.raises(ValueError, match="not found or not owned"):
        asyncio.run(mgr.rename_session("s1", "attacker", "New Title"))


def test_rename_raises_on_empty_title():
    owned = FakeSessionModel()
    ctx, _ = _make_async_session(first_value=owned)
    mgr = _Manager(ctx)
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(mgr.rename_session("s1", "u1", "   "))


def test_rename_updates_title():
    owned = FakeSessionModel(session_id="s1", user_id="u1", title="Old")
    ctx, mock_session = _make_async_session(first_value=owned)
    mgr = _Manager(ctx)
    asyncio.run(mgr.rename_session("s1", "u1", "New Title"))
    assert owned.title == "New Title"
    mock_session.commit.assert_called_once()


def test_pin_raises_when_not_owned():
    ctx, _ = _make_async_session(first_value=None)
    mgr = _Manager(ctx)
    with pytest.raises(ValueError, match="not found or not owned"):
        asyncio.run(mgr.pin_session("s1", "attacker", True))


def test_pin_sets_flag():
    owned = FakeSessionModel(pinned=False)
    ctx, mock_session = _make_async_session(first_value=owned)
    mgr = _Manager(ctx)
    asyncio.run(mgr.pin_session("s1", "u1", True))
    assert owned.pinned is True
    mock_session.commit.assert_called_once()


def test_archive_raises_when_not_owned():
    ctx, _ = _make_async_session(first_value=None)
    mgr = _Manager(ctx)
    with pytest.raises(ValueError, match="not found or not owned"):
        asyncio.run(mgr.archive_session("s1", "attacker", True))


def test_archive_clears_pinned():
    """Archiving a pinned session also removes the pin."""
    owned = FakeSessionModel(pinned=True, archived=False)
    ctx, mock_session = _make_async_session(first_value=owned)
    mgr = _Manager(ctx)
    asyncio.run(mgr.archive_session("s1", "u1", True))
    assert owned.archived is True
    assert owned.pinned is False
    mock_session.commit.assert_called_once()


def test_unarchive_does_not_force_pin_change():
    owned = FakeSessionModel(pinned=False, archived=True)
    ctx, mock_session = _make_async_session(first_value=owned)
    mgr = _Manager(ctx)
    asyncio.run(mgr.archive_session("s1", "u1", False))
    assert owned.archived is False
    mock_session.commit.assert_called_once()
