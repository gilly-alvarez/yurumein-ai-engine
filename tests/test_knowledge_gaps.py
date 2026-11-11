"""
Unit tests for the Phase 1 Knowledge Learning System.

All tests are deterministic and require no external services (no DB, no OpenAI).
Gap detection logic is tested against the signals defined in the spec.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.utils.RAG_pipeline.DB.knowledge_gap_manager import KnowledgeGapManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager() -> KnowledgeGapManager:
    """Return a KnowledgeGapManager whose SessionManager is fully mocked."""
    with patch(
        "app.services.utils.RAG_pipeline.DB.knowledge_gap_manager.SessionManager"
    ):
        return KnowledgeGapManager()


# ---------------------------------------------------------------------------
# 1. Normalize / classify — pure functions, no DB needed
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_lowercases_and_strips_punctuation(self):
        result = KnowledgeGapManager.normalize("What is the Garífuna word for 'father'?")
        assert "garífuna" in result or "garifuna" in result.lower()
        assert "'" not in result
        assert "?" not in result

    def test_removes_stop_words(self):
        result = KnowledgeGapManager.normalize("what is the history of the Garifuna people")
        assert "what" not in result.split()
        assert "is" not in result.split()
        assert "the" not in result.split()
        assert "of" not in result.split()
        # Content words preserved
        assert "history" in result
        assert "garifuna" in result

    def test_empty_input_returns_empty(self):
        assert KnowledgeGapManager.normalize("") == ""
        assert KnowledgeGapManager.normalize("   ") == ""

    def test_truncates_to_30_words(self):
        long_query = " ".join([f"word{i}" for i in range(50)])
        result = KnowledgeGapManager.normalize(long_query)
        assert len(result.split()) <= 30


class TestClassifyTopic:
    def test_language_query(self):
        assert KnowledgeGapManager.classify_topic("garifuna word translate") == "Language & Vocabulary"

    def test_history_query(self):
        assert KnowledgeGapManager.classify_topic("history origin carib afric") == "Historical Origins"

    def test_cultural_query(self):
        assert KnowledgeGapManager.classify_topic("dugu ceremony ritual tradition") == "Cultural Practices"

    def test_community_query(self):
        assert KnowledgeGapManager.classify_topic("elder diaspora community ancestor") == "Community & People"

    def test_education_query(self):
        assert KnowledgeGapManager.classify_topic("learn lesson teach class") == "Education"

    def test_unknown_falls_back(self):
        assert KnowledgeGapManager.classify_topic("xyz abc qrs") == "General Knowledge"


# ---------------------------------------------------------------------------
# 2. Jaccard deduplication
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical_sets(self):
        s = {"garifuna", "language", "word"}
        assert KnowledgeGapManager._jaccard(s, s) == 1.0

    def test_disjoint_sets(self):
        assert KnowledgeGapManager._jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        s1 = {"garifuna", "word", "father"}
        s2 = {"garifuna", "word", "mother"}
        score = KnowledgeGapManager._jaccard(s1, s2)
        assert 0.0 < score < 1.0

    def test_empty_sets_return_zero(self):
        assert KnowledgeGapManager._jaccard(set(), {"a"}) == 0.0
        assert KnowledgeGapManager._jaccard({"a"}, set()) == 0.0


# ---------------------------------------------------------------------------
# 3. Gap signal detection (tests cahtbot_router logic via imports)
# ---------------------------------------------------------------------------

# Import the detection function
from app.services.chatbot.cahtbot_router import _detect_and_record_gap, _GAP_PHRASES


class TestGapSignalDetection:
    """Verify that the correct signals trigger gap recording."""

    @pytest.mark.asyncio
    async def test_no_retrieved_docs_non_lexicon_triggers_gap(self):
        with patch(
            "app.services.chatbot.cahtbot_router.get_gap_manager"
        ) as mock_factory:
            mock_mgr = AsyncMock()
            mock_factory.return_value = mock_mgr

            await _detect_and_record_gap(
                user_prompt="How was the Garifuna language formed?",
                response_text="I found some information about language formation.",
                retrieved_docs=[],
                context="Some general context",  # no LEXICON_CONTEXT sentinel
                citations_stripped=0,
            )

            mock_mgr.record_gap.assert_awaited_once()
            call_args = mock_mgr.record_gap.call_args
            assert call_args[1]["flag_reason"] == "no_retrieved_docs"

    @pytest.mark.asyncio
    async def test_lexicon_hit_with_no_docs_does_not_trigger_gap(self):
        with patch(
            "app.services.chatbot.cahtbot_router.get_gap_manager"
        ) as mock_factory:
            mock_mgr = AsyncMock()
            mock_factory.return_value = mock_mgr

            await _detect_and_record_gap(
                user_prompt="What does 'baba' mean?",
                response_text="'Baba' means father in Garífuna.",
                retrieved_docs=[],
                context="<LEXICON_CONTEXT>baba = father</LEXICON_CONTEXT>",
                citations_stripped=0,
            )

            mock_mgr.record_gap.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_citation_stripped_triggers_gap(self):
        with patch(
            "app.services.chatbot.cahtbot_router.get_gap_manager"
        ) as mock_factory:
            mock_mgr = AsyncMock()
            mock_factory.return_value = mock_mgr

            await _detect_and_record_gap(
                user_prompt="Explain Van Sertima's critics",
                response_text="Some historians disagree.",
                retrieved_docs=[{"hybrid_score": 0.6, "metadata": {}}],
                context="Some context",
                citations_stripped=2,
            )

            mock_mgr.record_gap.assert_awaited_once()
            assert mock_mgr.record_gap.call_args[1]["flag_reason"] == "citation_rejected"

    @pytest.mark.asyncio
    async def test_gap_acknowledged_phrase_triggers_gap(self):
        with patch(
            "app.services.chatbot.cahtbot_router.get_gap_manager"
        ) as mock_factory:
            mock_mgr = AsyncMock()
            mock_factory.return_value = mock_mgr

            await _detect_and_record_gap(
                user_prompt="When was the first Garifuna settlement in Honduras?",
                response_text="I do not have information about the exact date.",
                retrieved_docs=[{"hybrid_score": 0.65, "metadata": {}}],
                context="Partial context",
                citations_stripped=0,
            )

            mock_mgr.record_gap.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_good_answer_no_gap_recorded(self):
        with patch(
            "app.services.chatbot.cahtbot_router.get_gap_manager"
        ) as mock_factory:
            mock_mgr = AsyncMock()
            mock_factory.return_value = mock_mgr

            await _detect_and_record_gap(
                user_prompt="What does the word 'hagiya' mean?",
                response_text="'Hagiya' means 'thank you' in Garífuna.",
                retrieved_docs=[{"hybrid_score": 0.82, "metadata": {}}],
                context="Lexicon context with hagiya",
                citations_stripped=0,
            )

            mock_mgr.record_gap.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_low_evidence_thin_docs_triggers_gap(self):
        with patch(
            "app.services.chatbot.cahtbot_router.get_gap_manager"
        ) as mock_factory:
            mock_mgr = AsyncMock()
            mock_factory.return_value = mock_mgr

            await _detect_and_record_gap(
                user_prompt="Describe pre-Columbian Garifuna trade routes",
                response_text="Trade information is limited in sources.",
                retrieved_docs=[{"hybrid_score": 0.18, "metadata": {}},
                                {"hybrid_score": 0.21, "metadata": {}}],
                context="Thin context",
                citations_stripped=0,
            )

            mock_mgr.record_gap.assert_awaited_once()
            assert mock_mgr.record_gap.call_args[1]["flag_reason"] == "low_evidence"

    @pytest.mark.asyncio
    async def test_citation_rejected_takes_priority_over_no_docs(self):
        """citation_rejected is higher-priority than no_retrieved_docs."""
        with patch(
            "app.services.chatbot.cahtbot_router.get_gap_manager"
        ) as mock_factory:
            mock_mgr = AsyncMock()
            mock_factory.return_value = mock_mgr

            await _detect_and_record_gap(
                user_prompt="Cite sources on Garifuna origin",
                response_text="Some answer.",
                retrieved_docs=[],
                context="No real context",
                citations_stripped=1,
            )

            assert mock_mgr.record_gap.call_args[1]["flag_reason"] == "citation_rejected"


# ---------------------------------------------------------------------------
# 4. Privacy / governance — user statements must NOT modify knowledge base
# ---------------------------------------------------------------------------

class TestPrivacyAndGovernance:
    """The gap system must never automatically treat user statements as knowledge."""

    @pytest.mark.asyncio
    async def test_user_statement_does_not_auto_publish(self):
        """
        A good answer (no gap signals) must never call record_gap.
        There is no code path from a successful user message to a DB write.
        """
        with patch(
            "app.services.chatbot.cahtbot_router.get_gap_manager"
        ) as mock_factory:
            mock_mgr = AsyncMock()
            mock_factory.return_value = mock_mgr

            # User "injects" a statement — but the response is well-sourced
            await _detect_and_record_gap(
                user_prompt="I know Garifuna comes from the moon",
                response_text="The Garifuna language evolved from Island Carib.",
                retrieved_docs=[{"hybrid_score": 0.78, "metadata": {}}],
                context="Rich context",
                citations_stripped=0,
            )

            # The gap manager's record_gap must NOT be called
            mock_mgr.record_gap.assert_not_awaited()

    def test_gap_record_uses_normalized_prompt_not_full_text(self):
        """normalize() strips punctuation that could expose PII-like patterns."""
        # normalize() is a pure static method — no DB needed
        result = KnowledgeGapManager.normalize("user@example.com asked about cultural practices")
        assert "@" not in result
        # Content words preserved
        assert "cultural" in result
        assert "practices" in result

    def test_gap_record_never_stores_session_or_user_id(self):
        """KnowledgeGapModel has no session_id or user_id column."""
        from app.services.utils.RAG_pipeline.DB.knowledge_gap_manager import KnowledgeGapModel
        col_names = [c.key for c in KnowledgeGapModel.__table__.columns]
        assert "session_id" not in col_names
        assert "user_id" not in col_names
        assert "user_email" not in col_names


# ---------------------------------------------------------------------------
# 5. _validate_citations backward-compat (make sure refactor didn't break it)
# ---------------------------------------------------------------------------

class TestValidateCitationsBackwardCompat:
    """Verify that _validate_citations still returns a string (not a tuple)."""

    def setup_method(self):
        from app.services.utils.RAG_pipeline.workflow.nodes import Nodes
        self.node = object.__new__(Nodes)

    def test_returns_string(self):
        result = self.node._validate_citations("Hello world.", [])
        assert isinstance(result, str)

    def test_strips_fabricated_citation_on_empty_docs(self):
        text = "Some answer.\n**Source:** Diccionario Garífuna — page 50"
        result = self.node._validate_citations(text, [])
        assert "Source:" not in result
        assert "Diccionario" not in result

    def test_with_count_returns_tuple(self):
        text = "Some answer.\n**Source:** Diccionario Garífuna — page 50"
        result = self.node._validate_citations_with_count(text, [])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], int)

    def test_with_count_tracks_stripped_count(self):
        text = "Answer.\n**Source:** Fabricated Book — page 99"
        _, count = self.node._validate_citations_with_count(text, [])
        assert count == 1

    def test_no_citations_stripped_when_valid(self):
        # Yurumein Lexicon citations are always preserved regardless of retrieved_docs
        text = "Answer.\n**Source:** Yurumein Lexicon"
        _, count = self.node._validate_citations_with_count(text, [])
        assert count == 0


# ---------------------------------------------------------------------------
# 6. Admin API authentication
# ---------------------------------------------------------------------------

class TestAdminAuth:
    """Verify that require_admin_key returns 403/503 for unauthenticated access."""

    async def test_missing_key_raises_403(self):
        from fastapi import HTTPException
        from app.services.knowledge_gaps.knowledge_gap_router import require_admin_key
        from unittest.mock import patch

        with patch("app.services.knowledge_gaps.knowledge_gap_router.settings") as ms:
            ms.ADMIN_API_KEY = "real-secret-key-123"
            with pytest.raises(HTTPException) as exc:
                await require_admin_key(key=None)
            assert exc.value.status_code == 403

    async def test_wrong_key_raises_403(self):
        from fastapi import HTTPException
        from app.services.knowledge_gaps.knowledge_gap_router import require_admin_key
        from unittest.mock import patch

        with patch("app.services.knowledge_gaps.knowledge_gap_router.settings") as ms:
            ms.ADMIN_API_KEY = "real-secret-key-123"
            with pytest.raises(HTTPException) as exc:
                await require_admin_key(key="wrong-key")
            assert exc.value.status_code == 403

    async def test_correct_key_passes(self):
        from app.services.knowledge_gaps.knowledge_gap_router import require_admin_key
        from unittest.mock import patch

        with patch("app.services.knowledge_gaps.knowledge_gap_router.settings") as ms:
            ms.ADMIN_API_KEY = "real-secret-key-123"
            # Must not raise
            result = await require_admin_key(key="real-secret-key-123")
            assert result is None

    async def test_unconfigured_key_raises_503(self):
        from fastapi import HTTPException
        from app.services.knowledge_gaps.knowledge_gap_router import require_admin_key
        from unittest.mock import patch

        with patch("app.services.knowledge_gaps.knowledge_gap_router.settings") as ms:
            ms.ADMIN_API_KEY = ""
            with pytest.raises(HTTPException) as exc:
                await require_admin_key(key="any-key")
            assert exc.value.status_code == 503


# ---------------------------------------------------------------------------
# 7. Frequency cooldown — anchored to last_incremented, not last_seen
# ---------------------------------------------------------------------------

def _make_gap(now, last_incremented_offset_h, last_seen_offset_h=None):
    """Helper: build a mock KnowledgeGapModel with the given time offsets."""
    from app.services.utils.RAG_pipeline.DB.knowledge_gap_manager import KnowledgeGapModel
    from unittest.mock import MagicMock
    from datetime import timedelta
    gap = MagicMock(spec=KnowledgeGapModel)
    gap.normalized_question = "garifuna language origin"
    gap.frequency = 2
    gap.flag_reason = "no_retrieved_docs"
    gap.status = "needs_review"
    gap.last_incremented = now - timedelta(hours=last_incremented_offset_h)
    gap.last_seen = now - timedelta(hours=(last_seen_offset_h or last_incremented_offset_h))
    return gap


async def _run_record_gap(gap):
    """Helper: run record_gap against a mocked DB returning the given gap."""
    from app.services.utils.RAG_pipeline.DB.knowledge_gap_manager import KnowledgeGapManager
    from unittest.mock import AsyncMock, MagicMock, patch
    with patch("app.services.utils.RAG_pipeline.DB.knowledge_gap_manager.SessionManager"):
        mgr = KnowledgeGapManager()
    mgr._find_similar = AsyncMock(return_value=gap)
    mock_db = AsyncMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=False)
    mock_db.commit = AsyncMock()
    mgr._sm = MagicMock()
    mgr._sm.AsyncSessionLocal = MagicMock(return_value=mock_db)
    await mgr.record_gap(question="garifuna language origin", flag_reason="no_retrieved_docs")
    return gap


class TestFrequencyCooldown:
    """Verify 24-hour cooldown anchored to last_incremented (not last_seen)."""

    async def test_within_24h_of_last_incremented_does_not_increment(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        gap = _make_gap(now, last_incremented_offset_h=1)  # incremented 1h ago
        await _run_record_gap(gap)
        assert gap.frequency == 2  # unchanged

    async def test_after_24h_of_last_incremented_increments(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        gap = _make_gap(now, last_incremented_offset_h=25)  # incremented 25h ago
        await _run_record_gap(gap)
        assert gap.frequency == 3  # incremented

    async def test_last_seen_always_updated_regardless_of_cooldown(self):
        """last_seen must reflect actual recency even when frequency is not incremented."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        gap = _make_gap(now, last_incremented_offset_h=1, last_seen_offset_h=1)
        original_last_seen = gap.last_seen
        result = await _run_record_gap(gap)
        # last_seen updated even though cooldown blocked frequency increment
        assert result.last_seen != original_last_seen

    async def test_frequent_activity_cannot_indefinitely_block_increment(self):
        """The original bug: last_seen was used as cooldown anchor, so frequent
        activity (every 23h) could reset the window and prevent frequency growth.
        With last_incremented as anchor, the question still increments after 24h
        even if last_seen was updated repeatedly."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        from app.services.utils.RAG_pipeline.DB.knowledge_gap_manager import KnowledgeGapModel
        from unittest.mock import MagicMock
        gap = MagicMock(spec=KnowledgeGapModel)
        gap.normalized_question = "garifuna language origin"
        gap.frequency = 2
        gap.flag_reason = "no_retrieved_docs"
        gap.status = "needs_review"
        # last_incremented is 25 hours ago — cooldown window is open
        gap.last_incremented = now - timedelta(hours=25)
        # last_seen is only 1 hour ago — simulates frequent activity updating last_seen
        gap.last_seen = now - timedelta(hours=1)
        result = await _run_record_gap(gap)
        # Must increment because last_incremented > 24h ago, regardless of last_seen
        assert result.frequency == 3

    async def test_last_incremented_updated_when_frequency_increments(self):
        """When an increment fires, last_incremented must be set to now."""
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        gap = _make_gap(now, last_incremented_offset_h=25)
        result = await _run_record_gap(gap)
        assert result.frequency == 3
        # last_incremented must have been reset (no longer 25h ago)
        assert (now - result.last_incremented).total_seconds() < 5
