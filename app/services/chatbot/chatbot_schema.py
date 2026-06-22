from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class ChatbotResponse(BaseModel):
    session_id: str
    response: str
    file_status: str

class ChatSessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: Optional[str]
    updated_at: Optional[str]

class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]

class RetrievalMatchResponse(BaseModel):
    rank: int
    content: str
    metadata: dict

class RetrievalCheckResponse(BaseModel):
    query: str
    matches_found: int
    matches: List[RetrievalMatchResponse]

class HybridRetrievalMatchResponse(BaseModel):
    rank: int
    content: str
    metadata: dict
    retrieval_source: str
    hybrid_score: float
    vector_score: Optional[float]
    keyword_score: float

class HybridRetrievalCheckResponse(BaseModel):
    query: str
    matches_found: int
    matches: List[HybridRetrievalMatchResponse]
