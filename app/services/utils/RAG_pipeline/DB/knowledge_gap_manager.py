from typing import List, Dict, Optional
from datetime import datetime, timezone, timedelta
import re
from sqlalchemy import Column, String, DateTime, Integer, Text, func
from sqlalchemy.future import select
from .session_manager import Base, SessionManager

# Patterns removed before normalization to avoid storing identifiable tokens.
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(r"\b\d[\d\s\-().]{6,}\d\b")
_LONG_DIGIT_RE = re.compile(r"\b\d{5,}\b")


_STOP_WORDS = frozenset({
    "the", "a", "an", "in", "on", "to", "for", "of", "and", "or", "but",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
    "how", "what", "where", "when", "why", "who", "which", "about", "with",
    "el", "la", "los", "las", "un", "una", "en", "de", "por", "para", "y",
    "o", "es", "son", "fue", "cómo", "qué", "cuál", "cuáles", "dónde",
    "cuándo", "quién", "quiénes", "se", "del", "al",
})

_TOPIC_RULES = [
    (
        ["translat", "vocabular", "word", "mean", "pronoun", "garifuna", "spell",
         "term", "traducir", "vocabulario", "palabra", "significar", "pronunciar"],
        "Language & Vocabulary",
    ),
    (
        ["history", "origin", "afric", "carib", "coloni", "yurumein", "settl",
         "migrat", "historia", "origen", "africano", "caribe", "coloniz"],
        "Historical Origins",
    ),
    (
        ["ceremony", "ritual", "dugu", "tradition", "cultural", "food", "music",
         "danc", "ceremonia", "tradicion", "cultural", "comida", "música", "baile"],
        "Cultural Practices",
    ),
    (
        ["community", "people", "elder", "diaspora", "leader", "ancestor",
         "comunidad", "pueblo", "anciano", "diáspora", "líder", "ancestro"],
        "Community & People",
    ),
    (
        ["teach", "learn", "lesson", "class", "study", "educat",
         "enseñar", "aprender", "lección", "clase", "estudiar", "educación"],
        "Education",
    ),
]

VALID_STATUSES = frozenset(
    {"needs_review", "research_needed", "verified", "resolved", "dismissed"}
)
VALID_FLAG_REASONS = frozenset(
    {"no_retrieved_docs", "low_evidence", "citation_rejected",
     "gap_acknowledged", "recurring_question"}
)


class KnowledgeGapModel(Base):
    __tablename__ = "knowledge_gaps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    topic = Column(String(100), nullable=False, index=True)
    normalized_question = Column(Text, nullable=False)
    frequency = Column(Integer, default=1, nullable=False)
    flag_reason = Column(String(50), nullable=False, index=True)
    first_seen = Column(
        DateTime(timezone=False),
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )
    last_seen = Column(
        DateTime(timezone=False),
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )
    status = Column(String(30), default="needs_review", nullable=False, index=True)
    admin_notes = Column(Text, nullable=True)
    source_reference = Column(Text, nullable=True)


def _gap_to_dict(gap: KnowledgeGapModel) -> Dict:
    return {
        "id": gap.id,
        "topic": gap.topic,
        "normalized_question": gap.normalized_question,
        "frequency": gap.frequency,
        "flag_reason": gap.flag_reason,
        "first_seen": gap.first_seen.isoformat() if gap.first_seen else None,
        "last_seen": gap.last_seen.isoformat() if gap.last_seen else None,
        "status": gap.status,
        "admin_notes": gap.admin_notes,
        "source_reference": gap.source_reference,
    }


class KnowledgeGapManager:
    def __init__(self):
        self._sm = SessionManager()

    @staticmethod
    def normalize(question: str) -> str:
        """Lowercase, strip PII patterns and punctuation, remove stop words.

        PII scrubbing applied before tokenization:
        - Email addresses (token containing @)
        - Phone-like digit sequences (7+ digit runs with spaces/dashes/parens)
        - Long standalone digit strings (5+ digits, e.g. IDs, zip codes)

        What remains is topic keywords only — no names, emails, or identifiers
        are actively filtered (names are common words that remain). The admin
        review queue should not be treated as a verbatim conversation log.
        """
        q = question.lower()
        q = _EMAIL_RE.sub(" ", q)
        q = _PHONE_RE.sub(" ", q)
        q = _LONG_DIGIT_RE.sub(" ", q)
        q = re.sub(r"[^\w\s]", " ", q)
        words = [w for w in q.split() if w and w not in _STOP_WORDS and len(w) > 1]
        return " ".join(words[:30])

    @staticmethod
    def classify_topic(normalized: str) -> str:
        words = normalized.split()
        for keywords, topic in _TOPIC_RULES:
            if any(any(kw in w for kw in keywords) for w in words):
                return topic
        return "General Knowledge"

    @staticmethod
    def _jaccard(set1: set, set2: set) -> float:
        if not set1 or not set2:
            return 0.0
        union = len(set1 | set2)
        return len(set1 & set2) / union if union else 0.0

    async def _find_similar(self, normalized: str, db) -> Optional[KnowledgeGapModel]:
        """Find an open gap with Jaccard word-overlap >= 0.5 (checks last 100 open gaps)."""
        new_words = set(normalized.split())
        if len(new_words) < 2:
            return None
        stmt = (
            select(KnowledgeGapModel)
            .where(KnowledgeGapModel.status.in_(["needs_review", "research_needed"]))
            .order_by(KnowledgeGapModel.last_seen.desc())
            .limit(100)
        )
        result = await db.execute(stmt)
        best_score, best_match = 0.0, None
        for candidate in result.scalars().all():
            score = self._jaccard(new_words, set(candidate.normalized_question.split()))
            if score >= 0.5 and score > best_score:
                best_score, best_match = score, candidate
        return best_match

    async def record_gap(self, question: str, flag_reason: str) -> None:
        """Upsert a knowledge gap signal. Never stores PII — only normalized topic words.

        `frequency` counts detection events (chat signals), NOT unique users.
        No user or session identifiers are stored, so per-user counting is not
        possible. High frequency indicates the topic is asked often; it does not
        mean many different people asked it.

        24-hour cooldown: frequency increments at most once per 24-hour window
        per normalized topic. This dampens rapid-fire repetition from any single
        session but does not enforce strict per-user limits.

        Deduplication: Jaccard word-overlap >= 0.5 on normalized questions.
        Cross-language note: Spanish/English dedup is approximate. If the same
        concept is asked in both languages with shared roots (e.g. "garifuna"),
        they will often deduplicate. Fully non-overlapping phrasings may create
        separate entries — this is a known Phase 1 limitation; no extra LLM call
        is used to avoid adding latency per user message.
        """
        try:
            normalized = self.normalize(question)
            if not normalized:
                return
            topic = self.classify_topic(normalized)
            now = datetime.now(timezone.utc).replace(tzinfo=None)

            async with self._sm.AsyncSessionLocal() as db:
                existing = await self._find_similar(normalized, db)
                if existing:
                    # Rate-limit frequency: at most one increment per 24-hour window.
                    hours_since_last = (now - existing.last_seen).total_seconds() / 3600
                    if hours_since_last >= 24:
                        existing.frequency += 1
                    existing.last_seen = now
                    # Escalate reason if a stronger signal arrives for the same topic
                    if (flag_reason == "citation_rejected"
                            and existing.flag_reason not in ("citation_rejected",)):
                        existing.flag_reason = flag_reason
                    if (existing.frequency >= 3
                            and existing.flag_reason not in ("citation_rejected",)):
                        existing.flag_reason = "recurring_question"
                    await db.commit()
                else:
                    db.add(KnowledgeGapModel(
                        topic=topic,
                        normalized_question=normalized[:300],
                        flag_reason=flag_reason,
                        status="needs_review",
                        frequency=1,
                        first_seen=now,
                        last_seen=now,
                    ))
                    await db.commit()
        except Exception as e:
            print(f"[knowledge-gap] record_gap failed: {e}")

    async def list_gaps(
        self,
        status: Optional[str] = None,
        flag_reason: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict:
        async with self._sm.AsyncSessionLocal() as db:
            base_where = []
            if status:
                base_where.append(KnowledgeGapModel.status == status)
            if flag_reason:
                base_where.append(KnowledgeGapModel.flag_reason == flag_reason)

            count_stmt = select(func.count(KnowledgeGapModel.id))
            for w in base_where:
                count_stmt = count_stmt.where(w)
            total = (await db.execute(count_stmt)).scalar() or 0

            stmt = select(KnowledgeGapModel).order_by(KnowledgeGapModel.last_seen.desc())
            for w in base_where:
                stmt = stmt.where(w)
            stmt = stmt.limit(limit).offset(offset)
            rows = (await db.execute(stmt)).scalars().all()
            return {"total": total, "items": [_gap_to_dict(g) for g in rows]}

    async def update_gap(
        self,
        gap_id: int,
        status: Optional[str] = None,
        admin_notes: Optional[str] = None,
        source_reference: Optional[str] = None,
    ) -> Optional[Dict]:
        async with self._sm.AsyncSessionLocal() as db:
            result = await db.execute(
                select(KnowledgeGapModel).where(KnowledgeGapModel.id == gap_id)
            )
            gap = result.scalars().first()
            if gap is None:
                return None
            if status is not None:
                gap.status = status
            if admin_notes is not None:
                gap.admin_notes = admin_notes
            if source_reference is not None:
                gap.source_reference = source_reference
            await db.commit()
            return _gap_to_dict(gap)

    async def delete_gap(self, gap_id: int) -> bool:
        async with self._sm.AsyncSessionLocal() as db:
            result = await db.execute(
                select(KnowledgeGapModel).where(KnowledgeGapModel.id == gap_id)
            )
            gap = result.scalars().first()
            if gap is None:
                return False
            await db.delete(gap)
            await db.commit()
            return True

    async def get_report(self) -> Dict:
        async with self._sm.AsyncSessionLocal() as db:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            week_ago = now - timedelta(days=7)

            status_rows = (await db.execute(
                select(KnowledgeGapModel.status, func.count(KnowledgeGapModel.id))
                .group_by(KnowledgeGapModel.status)
            )).all()

            reason_rows = (await db.execute(
                select(KnowledgeGapModel.flag_reason, func.count(KnowledgeGapModel.id))
                .group_by(KnowledgeGapModel.flag_reason)
            )).all()

            new_this_week = (await db.execute(
                select(func.count(KnowledgeGapModel.id))
                .where(KnowledgeGapModel.first_seen >= week_ago)
            )).scalar() or 0

            top_topics = (await db.execute(
                select(
                    KnowledgeGapModel.topic,
                    func.sum(KnowledgeGapModel.frequency).label("total"),
                )
                .group_by(KnowledgeGapModel.topic)
                .order_by(func.sum(KnowledgeGapModel.frequency).desc())
                .limit(10)
            )).all()

            top_gaps_rows = (await db.execute(
                select(KnowledgeGapModel)
                .where(KnowledgeGapModel.status.in_(["needs_review", "research_needed"]))
                .order_by(KnowledgeGapModel.frequency.desc())
                .limit(10)
            )).scalars().all()

            return {
                "generated_at": now.isoformat(),
                "period_days": 7,
                "new_this_week": new_this_week,
                "frequency_note": (
                    "frequency counts detection events (chat signals), not unique users. "
                    "No user identifiers are stored. High frequency means the topic "
                    "triggered gap signals often; it does not confirm distinct users."
                ),
                "status_breakdown": {r[0]: r[1] for r in status_rows},
                "flag_reason_breakdown": {r[0]: r[1] for r in reason_rows},
                "top_topics": [{"topic": r[0], "detection_events": int(r[1])} for r in top_topics],
                "top_unresolved_gaps": [_gap_to_dict(g) for g in top_gaps_rows],
            }

    async def create_tables(self) -> None:
        async with self._sm.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("Knowledge gap table created/verified")


_default_manager: Optional[KnowledgeGapManager] = None


def get_gap_manager() -> KnowledgeGapManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = KnowledgeGapManager()
    return _default_manager
