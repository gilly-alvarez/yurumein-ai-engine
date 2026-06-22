from typing import Optional, List, Any, TypedDict, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from datetime import datetime
from operator import add

class ChatState(TypedDict):
    """State for chat workflow - Using TypedDict for LangGraph compatibility"""
    thread_id: str  
    user_id: str
    session_title: Optional[str]
    file: Optional[Any]
    user_query: str
    context: Optional[str]
    messages: Annotated[List[dict], add]  # Merge from parallel nodes
    response: Optional[str]
    retrieved_docs: Optional[List[dict]]
    next_node: Optional[str]

class ChatSession(BaseModel):
    """Model for chat session metadata"""
    session_id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

class ChatMessage(BaseModel):
    """Model for individual chat message"""
    session_id: str
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_name: Optional[str] = None