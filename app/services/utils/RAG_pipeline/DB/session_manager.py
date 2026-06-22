from typing import List, Dict, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import Column, String, DateTime, Integer, Text, delete, update
from sqlalchemy.orm import declarative_base
from sqlalchemy.future import select
from langchain_core.messages import HumanMessage, AIMessage
from .....core.config import settings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Base = declarative_base()


class ChatSessionModel(Base):
    __tablename__ = "chat_sessions"

    session_id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)

    created_at = Column(
        DateTime(timezone=False),
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )

    updated_at = Column(
        DateTime(timezone=False),
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    timestamp = Column(
        DateTime(timezone=False),
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
    )


class SessionManager:
    def __init__(self):
        # Database credentials from your URL
        self.DB_USER = settings.DB_USER
        self.DB_PASS = settings.DB_PASS
        self.DB_HOST = settings.DB_HOST
        self.DB_PORT = settings.DB_PORT
        self.DB_NAME = settings.DB_NAME
        # Sync URL (for psycopg2 if needed)
        self.DATABASE_URL = f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

        # Async engine configuration
        self.async_engine = create_async_engine(
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}",
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
            pool_recycle=300,
            connect_args={
                "statement_cache_size": 0,
            },
        )

        self.AsyncSessionLocal = async_sessionmaker(
            bind=self.async_engine,
            expire_on_commit=False,
            class_=AsyncSession,
            autoflush=True,
            autocommit=False,
        )
        self._initialized = True

    async def create_tables(self):
        """Initialize database tables if they don't exist"""

        async with self.async_engine.begin() as conn:

            await conn.run_sync(Base.metadata.create_all)

        print("Chat session tables created/verified")

    async def create_session(self, session_id: str, user_id: str, title: str):
        """Create a new chat session"""

        async with self.AsyncSessionLocal() as session:

            try:

                # Check if session already exists

                existing = await session.execute(
                    select(ChatSessionModel).filter_by(
                        session_id=session_id, user_id=user_id
                    )
                )

                if existing.scalars().first():

                    print(f"Session {session_id} already exists")

                    return

                session_model = ChatSessionModel(
                    session_id=session_id,
                    user_id=user_id,
                    title=title,
                )
                session.add(session_model)
                await session.commit()
                print(f"✓ Created session: {session_id}")
            except Exception as e:

                await session.rollback()

                raise Exception(f"Failed to create session: {str(e)}")

    async def update_session(self, session_id: str, user_id: str):
        """Update session timestamp"""

        async with self.AsyncSessionLocal() as session:
            try:
                stmt = (
                    update(ChatSessionModel)
                    .where(
                        ChatSessionModel.session_id == session_id,
                        ChatSessionModel.user_id == user_id,
                    )
                    .values(updated_at=datetime.utcnow())
                )
                await session.execute(stmt)
                await session.commit()
                print(f"✓ Updated session: {session_id}")

            except Exception as e:

                await session.rollback()

                print(f"Failed to update session: {str(e)}")

    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ):
        """Save a message to the database"""

        async with self.AsyncSessionLocal() as session:

            try:

                message = ChatMessageModel(
                    session_id=session_id,
                    role=role,
                    content=content,
                )
                session.add(message)
                await session.commit()
                print(f"✓ Saved {role} message for session: {session_id}")
            except Exception as e:
                await session.rollback()
                raise Exception(f"Failed to save message: {str(e)}")

    async def get_full_session_history(self, session_id: str, user_id: str) -> List:
        """Get all messages for a session as LangChain message objects"""

        async with self.AsyncSessionLocal() as session:

            try:

                # Verify session exists and belongs to user

                session_stmt = select(ChatSessionModel).filter_by(
                    session_id=session_id, user_id=user_id
                )
                session_result = await session.execute(session_stmt)

                session_model = session_result.scalars().first()

                if not session_model:

                    print(f"Session not found: {session_id}")

                    return []
                # Get messages with explicit connection management

                messages_stmt = (
                    select(ChatMessageModel)
                    .filter_by(session_id=session_id)
                    .order_by(ChatMessageModel.timestamp.asc())
                )

                messages_result = await session.execute(messages_stmt)

                messages = list(messages_result.scalars().all())
                # Convert to LangChain format

                result = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    }
                    for msg in messages
                ]


                print(
                    f"✓ Retrieved {len(result)} messages for session: {session_id}"
                )

                return result

            except Exception as e:

                print(f"Error retrieving messages for session {session_id}: {e}")

                return []

    async def get_recent_session_history(
        self, session_id: str, user_id: str, limit: int = 20
    ) -> List[Dict]:
        """Get recent chat history with metadata"""
        async with self.AsyncSessionLocal() as session:
            try:
                # FIX: Store result before checking
                session_stmt = select(ChatSessionModel).filter_by(
                    session_id=session_id, user_id=user_id
                )
                session_result = await session.execute(session_stmt)
                session_model = session_result.scalars().first()

                if not session_model:
                    print(f"Session not found: {session_id}")
                    return []

                messages_stmt = (
                    select(ChatMessageModel)
                    .filter_by(session_id=session_id.strip())
                    .order_by(ChatMessageModel.timestamp.desc())
                    .limit(limit)
                )
                messages_result = await session.execute(messages_stmt)
                messages = messages_result.scalars().all()

                messages_chronological = list(reversed(messages))

                result = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                    for msg in messages_chronological
                ]

                print(f"✓ Retrieved recent history ({len(result)} messages)")
                return result

            except Exception as e:
                print(f"Error retrieving recent history: {e}")
                return []

    async def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get all sessions for a user"""
        async with self.AsyncSessionLocal() as session:
            try:
                sessions_stmt = (
                    select(ChatSessionModel)
                    .filter_by(user_id=user_id.strip())
                    .order_by(ChatSessionModel.updated_at.desc())
                )
                sessions_result = await session.execute(sessions_stmt)
                sessions = sessions_result.scalars().all()

                result = [
                    {
                        "session_id": s.session_id,
                        "title": s.title,
                        "created_at": (
                            s.created_at.isoformat() if s.created_at else None
                        ),
                        "updated_at": (
                            s.updated_at.isoformat() if s.updated_at else None
                        ),
                    }
                    for s in sessions
                ]
                print(f"✓ Retrieved {len(result)} sessions for user: {user_id}")
                return result

            except Exception as e:
                print(f"Error retrieving sessions for user {user_id}: {e}")
                return []

    async def geta_ll_sessions(self, user_id: str) -> List[Dict]:
        """Alias for get_user_sessions"""
        return await self.get_user_sessions(user_id)

    async def delete_session(self, session_id: str, user_id: str):
        """Delete a session and all its messages"""

        async with self.AsyncSessionLocal() as session:

            try:

                # Delete messages first

                delete_messages_result = await session.execute(
                    delete(ChatMessageModel).where(
                        ChatMessageModel.session_id == session_id
                    )
                )
                delete_session_result = await session.execute(
                    delete(ChatSessionModel).where(
                        ChatSessionModel.session_id == session_id,
                        ChatSessionModel.user_id == user_id,
                    )
                )
                await session.commit()
                print(f"✓ Deleted session: {session_id}")
            except Exception as e:

                await session.rollback()

                raise Exception(f"Failed to delete session: {str(e)}")

    async def close(self):
        """Close database connections properly"""

        try:

            if hasattr(self, "async_engine") and self.async_engine:

                await self.async_engine.dispose()
                print("✓ Async engine disposed")
            if hasattr(self, "sync_engine") and self.sync_engine:
                self.sync_engine.dispose()
                print("✓ Sync engine disposed")
        except Exception as e:
            print(f"Error during connection closure: {e}")

    async def __aenter__(self):
        """Async context manager support"""

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Auto-close when exiting context"""

        await self.close()


async def initialize_db():
    """Initialize all database tables"""

    manager = SessionManager()

    await manager.create_tables()

    return manager
