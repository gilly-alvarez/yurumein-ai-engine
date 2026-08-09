#!/usr/bin/env python3
"""
Phase 1 translator tests.

All external I/O is mocked — no live DB, no OpenAI API key required.
conftest.py stubs missing third-party packages before this file is collected.

Tests cover every decision point in the Phase 1 translation flow:

  Garífuna target
    ├─ lexicon hit  → return directly (no vector, no LLM)
    └─ lexicon miss → keyword_search
          ├─ context found → LLM (grounded)
          └─ no context    → decline (no LLM)

  Non-Garífuna target (no lexicon gate)
    ├─ context found → LLM (grounded)
    └─ no context    → decline (no LLM)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# conftest.py has already registered all sys.modules stubs, so this import
# is safe on Python 3.9 without project dependencies installed.
from app.services.translater.translater import Translater

DECLINE_MSG = (
    "This word or phrase is not currently available in the Yurumein AI "
    "verified Garífuna database. Please try another word or phrase, or "
    "consult a Garífuna language expert."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_response(translation: str) -> MagicMock:
    """Build a mock OpenAI completion whose first choice contains the given translation."""
    choice = MagicMock()
    choice.message.content = json.dumps({"translation": translation})
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def t():
    """
    Translater instance with all three external dependencies replaced by
    MagicMock. Patches stay active for the duration of each test so that
    constructor calls inside __init__ are intercepted.
    """
    mock_vsm = MagicMock()
    mock_lex = MagicMock()
    mock_client = MagicMock()

    with (
        patch("app.services.translater.translater.VectorStoreManager", return_value=mock_vsm),
        patch("app.services.translater.translater.YurumeinLexicon", return_value=mock_lex),
        patch("app.services.translater.translater.OpenAI", return_value=mock_client),
    ):
        translater = Translater()
        yield translater, mock_vsm, mock_lex, mock_client


# ---------------------------------------------------------------------------
# T1 — Garífuna target, exact lexicon hit
# ---------------------------------------------------------------------------

class TestT1LexiconHit:
    def test_returns_lexicon_garifuna_directly(self, t):
        tr, vsm, lex, client = t
        lex.lookup.return_value = [{"garifuna": "Ünabu"}]

        result = tr.get_response({
            "text": "down",
            "base_language": "english",
            "target_language": "garifuna",
        })

        lex.lookup.assert_called_once_with("down", "english")
        vsm.keyword_search.assert_not_called()
        client.chat.completions.create.assert_not_called()
        assert result.translation == "Ünabu"

    def test_uses_first_row_when_multiple_hits(self, t):
        """Only r[0]['garifuna'] is used even when multiple rows match."""
        tr, _, lex, client = t
        lex.lookup.return_value = [{"garifuna": "Ünabu"}, {"garifuna": "Ynabu"}]

        result = tr.get_response({
            "text": "down",
            "base_language": "english",
            "target_language": "garifuna",
        })

        client.chat.completions.create.assert_not_called()
        assert result.translation == "Ünabu"

    def test_spanish_base_language_forwarded_to_lexicon(self, t):
        tr, _, lex, _ = t
        lex.lookup.return_value = [{"garifuna": "Igiruti"}]

        tr.get_response({
            "text": "abandonado",
            "base_language": "spanish",
            "target_language": "garifuna",
        })

        lex.lookup.assert_called_once_with("abandonado", "spanish")


# ---------------------------------------------------------------------------
# T2 — Garífuna target, lexicon miss, keyword hit → LLM
# ---------------------------------------------------------------------------

class TestT2LexiconMissKeywordHit:
    def test_llm_called_once_with_context(self, t):
        tr, vsm, lex, client = t
        lex.lookup.return_value = []
        vsm.keyword_search.return_value = [
            {"document": "Ünabu — below, down", "keyword_score": 0.8}
        ]
        client.chat.completions.create.return_value = _openai_response("Ünabu")

        result = tr.get_response({
            "text": "down",
            "base_language": "english",
            "target_language": "garifuna",
        })

        vsm.keyword_search.assert_called_once_with(query="down", k=5)
        client.chat.completions.create.assert_called_once()
        assert result.translation == "Ünabu"

    def test_retrieved_context_embedded_in_system_prompt(self, t):
        """Context text must appear inside the single system message sent to the LLM."""
        tr, vsm, lex, client = t
        lex.lookup.return_value = []
        vsm.keyword_search.return_value = [
            {"document": "CHUNK_SENTINEL", "keyword_score": 1.0}
        ]
        client.chat.completions.create.return_value = _openai_response("x")

        tr.get_response({
            "text": "word",
            "base_language": "english",
            "target_language": "garifuna",
        })

        kwargs = client.chat.completions.create.call_args.kwargs
        system_msg = kwargs["messages"][0]
        assert system_msg["role"] == "system"
        assert "CHUNK_SENTINEL" in system_msg["content"]

    def test_only_one_system_message_sent(self, t):
        """Context is embedded in a single system message, not injected as a second one."""
        tr, vsm, lex, client = t
        lex.lookup.return_value = []
        vsm.keyword_search.return_value = [{"document": "ctx", "keyword_score": 0.5}]
        client.chat.completions.create.return_value = _openai_response("x")

        tr.get_response({
            "text": "word",
            "base_language": "english",
            "target_language": "garifuna",
        })

        kwargs = client.chat.completions.create.call_args.kwargs
        system_msgs = [m for m in kwargs["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1

    def test_multiple_chunks_joined_in_context(self, t):
        tr, vsm, lex, client = t
        lex.lookup.return_value = []
        vsm.keyword_search.return_value = [
            {"document": "CHUNK_A", "keyword_score": 1.0},
            {"document": "CHUNK_B", "keyword_score": 0.5},
        ]
        client.chat.completions.create.return_value = _openai_response("x")

        tr.get_response({
            "text": "word",
            "base_language": "english",
            "target_language": "garifuna",
        })

        kwargs = client.chat.completions.create.call_args.kwargs
        system_content = kwargs["messages"][0]["content"]
        assert "CHUNK_A" in system_content
        assert "CHUNK_B" in system_content


# ---------------------------------------------------------------------------
# T3 — Garífuna target, lexicon miss, no keyword results → decline, no LLM
# ---------------------------------------------------------------------------

class TestT3LexiconMissNoContext:
    def test_decline_returned_without_llm_call(self, t):
        tr, vsm, lex, client = t
        lex.lookup.return_value = []
        vsm.keyword_search.return_value = []

        result = tr.get_response({
            "text": "xyzzy",
            "base_language": "english",
            "target_language": "garifuna",
        })

        client.chat.completions.create.assert_not_called()
        assert result.translation == DECLINE_MSG

    def test_decline_message_is_exact(self, t):
        """The decline text must match the constant exactly — no variation."""
        tr, vsm, lex, _ = t
        lex.lookup.return_value = []
        vsm.keyword_search.return_value = []

        result = tr.get_response({
            "text": "nothing",
            "base_language": "english",
            "target_language": "garifuna",
        })

        assert result.translation == DECLINE_MSG


# ---------------------------------------------------------------------------
# T4 — Non-Garífuna target: lexicon gate skipped, keyword search runs
# ---------------------------------------------------------------------------

class TestT4NonGarifunaTarget:
    def test_english_target_skips_lexicon(self, t):
        tr, vsm, lex, client = t
        vsm.keyword_search.return_value = [{"document": "ctx", "keyword_score": 0.5}]
        client.chat.completions.create.return_value = _openai_response("hello")

        tr.get_response({
            "text": "Ünabu",
            "base_language": "garifuna",
            "target_language": "english",
        })

        lex.lookup.assert_not_called()
        vsm.keyword_search.assert_called_once()
        client.chat.completions.create.assert_called_once()

    def test_spanish_target_skips_lexicon(self, t):
        tr, vsm, lex, client = t
        vsm.keyword_search.return_value = [{"document": "ctx", "keyword_score": 0.5}]
        client.chat.completions.create.return_value = _openai_response("abajo")

        tr.get_response({
            "text": "down",
            "base_language": "english",
            "target_language": "spanish",
        })

        lex.lookup.assert_not_called()
        vsm.keyword_search.assert_called_once_with(query="down", k=5)


# ---------------------------------------------------------------------------
# T5 — Non-Garífuna target, no keyword results → decline, no LLM
# ---------------------------------------------------------------------------

class TestT5NonGarifunaNoContext:
    def test_decline_without_llm(self, t):
        tr, vsm, lex, client = t
        vsm.keyword_search.return_value = []

        result = tr.get_response({
            "text": "untranslatable",
            "base_language": "english",
            "target_language": "spanish",
        })

        lex.lookup.assert_not_called()
        client.chat.completions.create.assert_not_called()
        assert result.translation == DECLINE_MSG


# ---------------------------------------------------------------------------
# T6 — Exception handling
# ---------------------------------------------------------------------------

class TestT6Exceptions:
    def test_lexicon_db_error_raises_runtime_error(self, t):
        """Any exception from lexicon.lookup must be wrapped in RuntimeError."""
        tr, _, lex, client = t
        lex.lookup.side_effect = Exception("connection refused")

        with pytest.raises(RuntimeError, match="Translation failed"):
            tr.get_response({
                "text": "down",
                "base_language": "english",
                "target_language": "garifuna",
            })

        client.chat.completions.create.assert_not_called()

    def test_keyword_search_error_returns_decline(self, t):
        """_retrieve_context swallows exceptions → empty context → decline path."""
        tr, vsm, lex, client = t
        lex.lookup.return_value = []
        vsm.keyword_search.side_effect = Exception("DB unavailable")

        result = tr.get_response({
            "text": "down",
            "base_language": "english",
            "target_language": "garifuna",
        })

        client.chat.completions.create.assert_not_called()
        assert result.translation == DECLINE_MSG

    def test_openai_error_raises_runtime_error(self, t):
        tr, vsm, lex, client = t
        lex.lookup.return_value = []
        vsm.keyword_search.return_value = [{"document": "ctx", "keyword_score": 1.0}]
        client.chat.completions.create.side_effect = Exception("API error")

        with pytest.raises(RuntimeError, match="Translation failed"):
            tr.get_response({
                "text": "down",
                "base_language": "english",
                "target_language": "garifuna",
            })


# ---------------------------------------------------------------------------
# T7 — load_lexicon.py: idempotency and additive-only behaviour
# ---------------------------------------------------------------------------

class TestT7Loader:
    """
    Tests for scripts/load_lexicon.py.
    Uses temp CSV files and a fully mocked DB engine — no live PostgreSQL needed.
    """

    def _ctx(self, mock_conn: MagicMock):
        """Build a context-manager mock that yields mock_conn."""
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_conn)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    def _lex(self, rowcounts=None):
        """
        Return (mock_lexicon, list_of_mock_conns).
        engine.begin() is called once per valid DB row via side_effect, returning
        a fresh context manager each time. rowcounts entries are int (rowcount
        returned by execute) or Exception instance (raised by execute).
        Defaults to [1] — one successful insert.
        """
        if rowcounts is None:
            rowcounts = [1]
        conns = []
        ctxs = []
        for rc in rowcounts:
            conn = MagicMock()
            if isinstance(rc, Exception):
                conn.execute.side_effect = rc
            else:
                conn.execute.return_value = MagicMock(rowcount=rc)
            conns.append(conn)
            ctxs.append(self._ctx(conn))

        lex = MagicMock()
        lex.engine.begin.side_effect = ctxs
        return lex, conns

    def _csv(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "lexicon.csv"
        p.write_text(content, encoding="utf-8")
        return p

    # -- helpers end --

    def test_all_rows_inserted_on_first_run(self, tmp_path):
        csv = self._csv(tmp_path,
            "spanish,spanish_norm,garifuna,english,english_norm\n"
            "Abajo,abajo,Ünabu,Down,down\n"
            "Arriba,arriba,Ünaruhu,Up,up\n"
        )
        lex, conns = self._lex(rowcounts=[1, 1])

        with (
            patch("scripts.load_lexicon.YurumeinLexicon", return_value=lex),
            patch("scripts.load_lexicon.normalize_term", side_effect=str.lower),
        ):
            from scripts.load_lexicon import load
            load(csv)

        assert lex.engine.begin.call_count == 2
        assert all(c.execute.call_count == 1 for c in conns)

    def test_second_run_skips_existing_rows(self, tmp_path):
        """rowcount=0 means WHERE NOT EXISTS blocked the insert — counted as skipped."""
        csv = self._csv(tmp_path,
            "spanish,spanish_norm,garifuna,english,english_norm\n"
            "Abajo,abajo,Ünabu,Down,down\n"
            "Arriba,arriba,Ünaruhu,Up,up\n"
        )
        lex, conns = self._lex(rowcounts=[0, 0])

        with (
            patch("scripts.load_lexicon.YurumeinLexicon", return_value=lex),
            patch("scripts.load_lexicon.normalize_term", side_effect=str.lower),
        ):
            from scripts.load_lexicon import load
            load(csv)

        # execute is called for each row (SQL runs), but rowcount=0 → skipped
        assert lex.engine.begin.call_count == 2
        assert all(c.execute.call_count == 1 for c in conns)

    def test_empty_garifuna_rows_skipped_without_db_call(self, tmp_path):
        """A row with no garifuna value must not reach the DB."""
        csv = self._csv(tmp_path,
            "spanish,spanish_norm,garifuna,english,english_norm\n"
            "Abajo,abajo,,Down,down\n"
            "Arriba,arriba,Ünaruhu,Up,up\n"
        )
        lex, conns = self._lex()  # default: one context for the one valid row

        with (
            patch("scripts.load_lexicon.YurumeinLexicon", return_value=lex),
            patch("scripts.load_lexicon.normalize_term", side_effect=str.lower),
        ):
            from scripts.load_lexicon import load
            load(csv)

        # Only 1 valid row → only 1 engine.begin() and 1 execute call
        assert lex.engine.begin.call_count == 1
        assert conns[0].execute.call_count == 1

    def test_missing_csv_exits_nonzero(self, tmp_path):
        missing = tmp_path / "does_not_exist.csv"
        lex, _ = self._lex()

        with patch("scripts.load_lexicon.YurumeinLexicon", return_value=lex):
            from scripts.load_lexicon import load
            with pytest.raises(SystemExit) as exc:
                load(missing)
            assert exc.value.code == 1

    def test_one_row_fails_between_two_valid_rows(self, tmp_path):
        """
        Per-row transactions: a single row failure must not roll back previously
        committed rows or prevent later rows from being attempted. Each row runs
        in its own engine.begin() context.
        """
        csv = self._csv(tmp_path,
            "spanish,spanish_norm,garifuna,english,english_norm\n"
            "Abajo,abajo,Ünabu,Down,down\n"
            "Malo,malo,Error,Bad,bad\n"
            "Arriba,arriba,Ünaruhu,Up,up\n"
        )
        # Row 1 succeeds, row 2 raises on execute, row 3 succeeds
        lex, conns = self._lex(rowcounts=[1, Exception("constraint violation"), 1])

        with (
            patch("scripts.load_lexicon.YurumeinLexicon", return_value=lex),
            patch("scripts.load_lexicon.normalize_term", side_effect=str.lower),
        ):
            from scripts.load_lexicon import load
            with pytest.raises(SystemExit) as exc:
                load(csv)
            assert exc.value.code == 1  # exits non-zero: errors > 0

        # All three transactions were opened independently
        assert lex.engine.begin.call_count == 3
        # Row 1's transaction committed: execute reached
        assert conns[0].execute.call_count == 1
        # Row 2's transaction attempted: execute raised (counted as one error)
        assert conns[1].execute.call_count == 1
        # Row 3's transaction committed independently despite row 2's failure
        assert conns[2].execute.call_count == 1

    def test_insert_sql_is_non_destructive(self):
        """The module-level INSERT SQL must use WHERE NOT EXISTS and no DELETE/TRUNCATE/DROP."""
        from scripts.load_lexicon import _INSERT_SQL
        sql = str(_INSERT_SQL)
        assert "WHERE NOT EXISTS" in sql
        assert "DELETE" not in sql.upper()
        assert "TRUNCATE" not in sql.upper()
        assert "DROP" not in sql.upper()
