
import os
import shutil
from pathlib import Path 
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import StreamingResponse
from typing import Annotated, Optional, List, AsyncGenerator
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
import json
import uuid
import asyncio
from ..utils.RAG_pipeline.workflow.workflow import create_workflow, run_workflow
from ..utils.RAG_pipeline.DB.session_manager import SessionManager
from ..utils.RAG_pipeline.vectore_db.pgvector_db import upload_docs_pipeline, VectorStoreManager
from ..chatbot.chatbot_schema import ChatbotResponse,ChatHistoryResponse,ChatSessionResponse,RetrievalCheckResponse,HybridRetrievalCheckResponse

router = APIRouter()



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
            "next_node": None
        }

        config = {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id
            }
        }

        async def generate() -> AsyncGenerator[str, None]:
            full_response = ""

            try:
                yield f"data: {json.dumps({'type': 'session', 'session_id': session_id, 'is_new': is_new_session})}\n\n"

                workflow, node_instance = create_workflow()

                async for event in workflow.astream(state, config=config):
                    if "llm_call" in event:
                        llm_output = event["llm_call"]
                        response_text = llm_output.get("response", "")

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
    user_prompt: Annotated[str, Form()],
    user_id: Annotated[str, Form()],
    session_id: Annotated[Optional[str], Form()] = None,
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
            "next_node": None
        }
        result = await run_workflow(state)
        response_text = result.get("response", "")
        print(response_text)
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


@router.delete('/session/{session_id}')
async def delete_session(
    session_id: str,
    user_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Delete a chat session"""
    try:
        await session_manager.delete_session(session_id, user_id)
        return {"message": "Session deleted successfully", "session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.post('/upload_docs')
async def upload_docs_RAG(file: UploadFile = File(...)):
    """
   Upload and process documents for RAG
    """
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
    file_path = os.path.join(UPLOAD_DIR, file.filename)

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


@router.get('/retrieval_check', response_model=RetrievalCheckResponse)
async def retrieval_check(query: str, k: int = 5):
    """Check direct vector DB retrieval without running the chat workflow."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    if k < 1 or k > 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")

    try:
        vector_store = VectorStoreManager()
        docs = vector_store.similarity_search(query=query.strip(), k=k)

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


@router.get('/retrieval_check_hybrid', response_model=HybridRetrievalCheckResponse)
async def retrieval_check_hybrid(query: str, k: int = 5):
    """Check merged hybrid retrieval results without running the chat workflow."""
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query is required")

    if k < 1 or k > 20:
        raise HTTPException(status_code=400, detail="k must be between 1 and 20")

    try:
        workflow, node_instance = create_workflow()
        _ = workflow
        docs = node_instance._hybrid_retrieve(query=query.strip(), top_k=k)

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


