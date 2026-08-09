"""
Test environment bootstrap.

This conftest runs before any test file is collected. It stubs all
third-party packages that are not installed in the test Python environment
so that app modules can be imported without live database or API connections.

Required because:
  - The project targets Python 3.11; this file may run under Python 3.9
    (system Python on macOS with Xcode CLI tools).
  - app/core/config.py uses `Any | None` syntax (Python 3.10+), so it
    must be replaced entirely with a pre-built settings stub.
  - None of the project's pip dependencies (openai, sqlalchemy, langchain_*,
    pydantic, etc.) are installed on the bare system Python.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ── Repo root on sys.path ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Env vars required by any settings-aware code ──────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("MODEL_NAME", "gpt-5")
os.environ.setdefault("COLLECTION_NAME", "test_collection")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_PASS", "test")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "testdb")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stub(name: str, **attrs) -> MagicMock:
    """Create a MagicMock, set named attributes, register in sys.modules."""
    m = MagicMock()
    m.__name__ = name
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# ── Working BaseModel substitute (pydantic) ───────────────────────────────────
# Subclasses must accept keyword args and expose them as attributes.
# model_dump() is used by translater_router.py.

class _MockBaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    def model_dump(self):
        return dict(self.__dict__)


# ── Working BaseSettings substitute (pydantic-settings) ──────────────────────

class _MockBaseSettings:
    class Config:
        env_file = ".env"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── SQL text wrapper: preserves original string in str() ─────────────────────
# load_lexicon.py stores _INSERT_SQL = text("...") at module level, and the
# test inspects str(_INSERT_SQL) to verify the SQL is non-destructive.

class _SQLText:
    def __init__(self, sql: str):
        self._sql = sql

    def __str__(self) -> str:
        return self._sql

    def __repr__(self) -> str:
        return f"text({self._sql!r})"


# ── Register all missing packages ────────────────────────────────────────────

_stub("pydantic", BaseModel=_MockBaseModel)
_stub("pydantic_settings", BaseSettings=_MockBaseSettings)

_stub("sqlalchemy",
      text=_SQLText,
      create_engine=MagicMock())
_stub("sqlalchemy.orm")
_stub("sqlalchemy.ext")
_stub("sqlalchemy.ext.asyncio")
_stub("sqlalchemy.future")


# sqlalchemy.exc needs a real OperationalError class that can be raised and caught.
class _OperationalError(Exception):
    def __init__(self, statement=None, params=None, orig=None, **kw):
        super().__init__(statement)

_exc_mod = MagicMock()
_exc_mod.OperationalError = _OperationalError
_exc_mod.DatabaseError = Exception
sys.modules["sqlalchemy.exc"] = _exc_mod

_stub("openai")

_stub("langchain_core")
_stub("langchain_core.documents")
_stub("langchain_core.messages")
_stub("langchain_core.prompts")
_stub("langchain_core.output_parsers")

_stub("langchain_openai")
_stub("langchain_postgres")

_stub("langchain_community")
_stub("langchain_community.document_loaders")

_stub("langchain_text_splitters")

_stub("pytesseract")
_stub("PIL")

_stub("asyncpg")
_stub("psycopg2")
_stub("psycopg")


# ── Stub app.core.config ─────────────────────────────────────────────────────
# config.py uses `Any | None` annotation syntax (Python 3.10+), which is a
# SyntaxError on 3.9. Replace the whole module with a pre-built settings object.

_settings_obj = _MockBaseSettings(
    OPENAI_API_KEY="test-key",
    MODEL_NAME="gpt-5",
    COLLECTION_NAME="test_collection",
    DB_USER="test",
    DB_PASS="test",
    DB_HOST="localhost",
    DB_PORT="5432",
    DB_NAME="testdb",
)
_config_mod = MagicMock()
_config_mod.settings = _settings_obj
sys.modules["app.core.config"] = _config_mod
