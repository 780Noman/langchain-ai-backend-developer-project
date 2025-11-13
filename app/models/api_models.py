from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID

class QueryRequest(BaseModel):
    question: str
    conversation_id: Optional[UUID] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    conversation_id: UUID
