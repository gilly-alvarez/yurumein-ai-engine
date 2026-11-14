
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form, Depends, Security, status
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from typing import Annotated, Optional, List, AsyncGenerator
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import json
import uuid
import asyncio
from ..utils.RAG_pipeline.workflow.workflow import create_workflow, run_workflow
from ..utils.RAG_pipeline.workflow.nodes import _is_publication_restricted
from ..utils.RAG_pipeline.DB.session_manager import SessionManager
from ..utils.RAG_pipeline.DB.knowledge_gap_manager import get_gap_manager
from ..utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline, VectorStoreManager
from ..chatbot.chatbot_schema import ChatbotResponse,ChatHistoryResponse,ChatSessionResponse,RetrievalCheckResponse,HybridRetrievalCheckResponse
from ...core.config import settings

router = APIRouter()

# ---------------------------------------------------------------------------
# Session-mutation authentication
# ---------------------------------------------------------------------------
_session_key_header = APIKeyHeader(name="X-Session-Key", auto_error=False)


async def require_session_key(key: Optional[str] = Security(_session_key_header)) -> None:
    """Require X-Session-Key on session-mutation endpoints.

    Passed server-side by the Next.js proxy — never from the browser.
    Returns 503 when SESSION_API_KEY is not configured (fail-secure default).
    Returns 403 for any missing or incorrect key.
    """
    configured = settings.SESSION_API_KEY
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Session API key not configured on this server.",
        )
    if not key or key != configured:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden.",
        )

# ---------------------------------------------------------------------------
# Admin authentication (diagnostic endpoints)
# ---------------------------------------------------------------------------
_admin_key_header = APIKeyHeader(name="X-Admin-Key", auto_error=False)


async def require_admin_key(key: Optional[str] = Security(_admin_key_header)) -> None:
    """Require X-Admin-Key on diagnostic endpoints.

    Mirrors the identical dependency in knowledge_gap_router so diagnostic
    routes use the same credential store without a cross-module dependency.
    Returns 503 when ADMIN_API_KEY is not configured (fail-secure default).
    Returns 403 for any missing or incorrect key.
    """
    configured = settings.ADMIN_API_KEY
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key not configured on this server.",
        )
    if not key or key != configured:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )


_GAP_PHRASES = (
    # English
    "do not have information", "don't have information",
    "do not provide information",
    "not available in", "not documented", "no documentation",
    "no information available", "unable to find", "cannot find",
    # Spanish — original
    "no encuentro información", "no tengo información",
    "no está documentado", "sin información",
    # Spanish — observed in production responses
    "no proporcionan información",
    "no proporciona información",
    "no documentan",
    "no hay información",
)


async def _detect_and_record_gap(
    user_prompt: str,
    response_text: str,
    retrieved_docs: list,
    context: str,
    citations_stripped: int,
) -> None:
    """Background task: detect knowledge gap signals and persist if found.

    Stores only normalized topic words — no PII, no session ID, no raw query.
    Does not modify any existing knowledge, lexicon, or verified content.
    """
    try:
        is_lexicon_hit = "<LEXICON_CONTEXT>" in (context or "")
        no_docs = len(retrieved_docs) == 0 and not is_lexicon_hit

        low_evidence = False
        if 0 < len(retrieved_docs) <= 2:
            scores = [d.get("hybrid_score", 0.0) for d in retrieved_docs]
            if max(scores, default=0.0) < 0.3:
                low_evidence = True

        citation_rejected = citations_stripped > 0
        gap_acknowledged = any(p in response_text.lower() for p in _GAP_PHRASES)

        if not any([no_docs, low_evidence, citation_rejected, gap_acknowledged]):
            return

        if citation_rejected:
            flag_reason = "citation_rejected"
        elif no_docs:
            flag_reason = "no_retrieved_docs"
        elif low_evidence:
            flag_reason = "low_evidence"
        else:
            flag_reason = "gap_acknowledged"

        await get_gap_manager().record_gap(question=user_prompt, flag_reason=flag_reason)
    except Exception as e:
        print(f"[knowledge-gap] detection error: {e}")


async def get_session_manager() -> SessionManager:
    """Dependency that returns the SessionManager singleton"""
    return SessionManager()


@router.post('/chatstrem', response_class=StreamingResponse)
async def chat_stream(
    user_prompt: Annotated[str, Form()],
    user_id: Annotated[str, Form()],
    session_id: Annotated[Optional[str], Form()] = None,
    upload_file: Annotated[Optional[UploadFile], File()] = None,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    FIXED: Stream chat responses in real-time
    """
    try:
        if not session_id:
            session_id = str(uuid.uuid4())
            is_new_session = True
        else:
            is_new_session = False

        state = {
            "thread_id": session_id,
            "user_id": user_id,
            "user_query": user_prompt,
            "file": upload_file,
            "session_title": None,
            "context": None,
            "messages": [],
            "response": None,
            "retrieved_docs": None,
            "next_node": None,
            "citations_stripped": 0,
        }

        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id
            }
        }

        async def generate() -> AsyncGenerator[str, None]:
            full_response = ""
            # Capture gap-detection data from workflow events
            _gap_retrieved_docs: list = []
            _gap_context: str = ""
            _gap_citations_stripped: int = 0

            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'is_new': is_new_session})}\n\n"

                workflow, node_instance = create_workflow()

                async for event in workflow.astream(state, config=config):
                    # Capture retrieval state from whichever retrieval node fires
                    for _rnode in ("global_retrieval_process", "file_process"):
                        if _rnode in event:
                            _n = event[_rnode]
                            _gap_retrieved_docs = _n.get("retrieved_docs") or []
                            _gap_context = _n.get("context") or ""
                    if "llm_call" in event:
                        llm_output = event["llm_call"]
                        response_text = llm_output.get("response", "")
                        _gap_citations_stripped = llm_output.get("citations_stripped") or 0

                        for char in response_text:
                            full_response += char
                            yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
                            await asyncio.sleep(0.01)

                await session_manager.save_message(
                    session_id=session_id,
                    role="user",
                    content=user_prompt,
                )

                if is_new_session:
                    session_title = await node_instance.generate_session_title(user_prompt)
                    await session_manager.create_session(
                        session_id=session_id,
                        user_id=user_id,
                        title=session_title
                    )
                    yield f"data: {json.dumps({'type': 'title', 'title': session_title})}\n\n"
                else:
                    await session_manager.update_session(session_id, user_id)

                await session_manager.save_message(
                    session_id=session_id,
                    role="assistant",
                    content=full_response
                )

                # Fire gap detection as a background task — never blocks streaming response
                asyncio.create_task(_detect_and_record_gap(
                    user_prompt=user_prompt,
                    response_text=full_response,
                    retrieved_docs=_gap_retrieved_docs,
                    context=_gap_context,
                    citations_stripped=_gap_citations_stripped,
                ))

            except Exception as e:
                error_msg = f"Error during chat processing: {str(e)}"
                print(error_msg)
                yield f"data: {json.dumps({'type': 'error', 'message': error_msg})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat streaming failed: {str(e)}")


@router.post('/chat', response_model=ChatbotResponse)
async def chat_simple(
    background_tasks: BackgroundTasks,
    user_prompt: Annotated[str, Form()],
    user_id: Annotated[str, Form()],
    session_id: Annotated[Optional[str], Form()] = None,
    language: Annotated[Optional[str], Form()] = None,
    upload_file: Annotated[UploadFile, File()] = None,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    SAFE: Non-streaming chat endpoint with optional file upload
    """
    try:
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
            is_new_session = True
        else:
            is_new_session = False

        state = {
            "thread_id": session_id,
            "user_id": user_id,
            "user_query": user_prompt,
            "file": upload_file,
            "session_title": None,
            "context": None,
            "messages": [],
            "response": None,
            "retrieved_docs": None,
            "next_node": None,
            "language": language,
            "citations_stripped": 0,
        }
        result = await run_workflow(state)
        response_text = result.get("response", "")
        retrieved_docs = result.get("retrieved_docs") or []
        context = result.get("context") or ""
        citations_stripped = result.get("citations_stripped") or 0
        print(response_text)

        background_tasks.add_task(
            _detect_and_record_gap,
            user_prompt=user_prompt,
            response_text=response_text,
            retrieved_docs=retrieved_docs,
            context=context,
            citations_stripped=citations_stripped,
        )
        await session_manager.save_message(
            session_id=session_id,
            role="user",
            content=user_prompt,
        )

        if is_new_session:
            _, node_instance = create_workflow()
            session_title = await node_instance.generate_session_title(user_prompt)

            await session_manager.create_session(
                session_id=session_id,
                user_id=user_id,
                title=session_title
            )
            print(f"New session created with title: {session_title}")
        else:
            await session_manager.update_session(session_id, user_id)

        await session_manager.save_message(
            session_id=session_id,
            role="assistant",
            content=response_text
        )

        file_status = (
            f"File '{upload_file.filename}' processed"
            if upload_file else
            "No file uploaded"
        )

        return ChatbotResponse(
            session_id=session_id,
            response=response_text,
            file_status=file_status
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")
#done
@router.get('/sessions/{user_id}', response_model=List[ChatSessionResponse])
async def get_user_sessions(
    user_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get all chat sessions for a user"""
    try:
        sessions = await session_manager.get_user_sessions(user_id)
        return sessions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")

#done
@router.get('/session/{session_id}/history', response_model=ChatHistoryResponse)
async def get_session_history(
    session_id: str,
    user_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get full chat history for a session"""
    try:
        # FIX: Use get_full_session_history instead
        messages = await session_manager.get_full_session_history(session_id, user_id)
        
        # Now messages are already dicts
        message_dicts = [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg.get("timestamp")
            }
            for msg in messages
        ]

        return ChatHistoryResponse(
            session_id=session_id,
            messages=message_dicts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.delete('/session/{session_id}', dependencies=[Depends(require_session_key)])
async def delete_session(
    session_id: str,
    user_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Delete a chat session. Returns 404 if the session does not exist or belongs to a different user."""
    try:
        await session_manager.delete_session(session_id, user_id)
        return {"message": "Session deleted successfully", "session_id": session_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.patch('/session/{session_id}/rename', dependencies=[Depends(require_session_key)])
async def rename_session(
    session_id: str,
    user_id: str,
    title: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Rename a chat session. Returns 404 if not owned by user."""
    try:
        await session_manager.rename_session(session_id, user_id, title)
        return {"message": "Session renamed", "session_id": session_id, "title": title.strip()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rename session: {str(e)}")


@router.patch('/session/{session_id}/pin', dependencies=[Depends(require_session_key)])
async def pin_session(
    session_id: str,
    user_id: str,
    pinned: bool,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Pin or unpin a chat session. Returns 404 if not owned by user."""
    try:
        await session_manager.pin_session(session_id, user_id, pinned)
        return {"message": "Session updated", "session_id": session_id, "pinned": pinned}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pin session: {str(e)}")


@router.patch('/session/{session_id}/archive', dependencies=[Depends(require_session_key)])
async def archive_session(
    session_id: str,
    user_id: str,
    archived: bool,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Archive or unarchive a chat session. Returns 404 if not owned by user."""
    try:
        await session_manager.archive_session(session_id, user_id, archived)
        return {"message": "Session updated", "session_id": session_id, "archived": archived}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to archive session: {str(e)}")


@router.post('/upload_docs', dependencies=[Depends(require_admin_key)])
async def upload_docs_RAG(file: UploadFile = File(...)):
    """Upload and process documents for RAG. Requires X-Admin-Key."""
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.doc', '.xlsx', '.xls', '.csv', '.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    CURRENT_DIR = Path(__file__).resolve().parent
    UPLOAD_DIR = CURRENT_DIR / ".." / "utils" / "RAG_pipeline" / "data" / "new_docs"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # Use a server-generated name so the client-supplied path cannot escape the upload directory.
    safe_filename = f"{uuid.uuid4().hex}{file_ext}"
    file_path = str(UPLOAD_DIR / safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"File saved: {file_path}")

        upload_docs_pipeline(file_paths=[file_path])

        if os.path.exists(file_path):
            os.remove(file_path)


        return {
            "status": "success",
            "filename": file.filename,
            "message": "File successfully uploaded, processed, and vectorized."
        }

    except Exception as e:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        await file.close()


@router.get('/retrieval_check', response_model=RetrievalCheckResponse,
            dependencies=[Depends(require_admin_key)])
async def retrieval_check(query: str, k: int = 5):
    """Check direct vector DB retrieval without running the chat workflow.

    Requires X-Admin-Key. Publication-restricted documents are excluded from
    results even for authenticated administrators — use an offline workflow
    for auditing restricted source content directly.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    if k < 1 or k > 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")

    try:
        vector_store = VectorStoreManager()
        docs = vector_store.similarity_search(query=query.strip(), k=k)

        # Filter publication-restricted documents before returning to caller.
        docs = [d for d in docs if not _is_publication_restricted(d.metadata or {})]

        matches = [
            {
                "rank": index,
                "content": doc.page_content,
                "metadata": doc.metadata or {},
            }
            for index, doc in enumerate(docs, start=1)
        ]

        return RetrievalCheckResponse(
            query=query.strip(),
            matches_found=len(matches),
            matches=matches,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval check failed: {str(e)}")


@router.get('/retrieval_check_hybrid', response_model=HybridRetrievalCheckResponse,
            dependencies=[Depends(require_admin_key)])
async def retrieval_check_hybrid(query: str, k: int = 5):
    """Check merged hybrid retrieval results without running the chat workflow.

    Requires X-Admin-Key. Publication-restricted documents are excluded from
    results even for authenticated administrators — use an offline workflow
    for auditing restricted source content directly.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    if k < 1 or k > 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")

    try:
        workflow, node_instance = create_workflow()
        _ = workflow
        docs = node_instance._hybrid_retrieve(query=query.strip(), top_k=k)

        # Filter publication-restricted documents before returning to caller.
        docs = node_instance._filter_restricted_docs(docs)

        matches = [
            {
                "rank": index,
                "content": doc.get("page_content", ""),
                "metadata": doc.get("metadata", {}),
                "retrieval_source": doc.get("retrieval_source", "vector"),
                "hybrid_score": float(doc.get("hybrid_score", 0.0)),
                "vector_score": (
                    float(doc["vector_score"])
                    if doc.get("vector_score") is not None
                    else None
                ),
                "keyword_score": float(doc.get("keyword_score", 0.0)),
            }
            for index, doc in enumerate(docs, start=1)
        ]

        return HybridRetrievalCheckResponse(
            query=query.strip(),
            matches_found=len(matches),
            matches=matches,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid retrieval check failed: {str(e)}")


@router.get('/health')
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Chatbot API",
        "version": "1.0.0"
    }


