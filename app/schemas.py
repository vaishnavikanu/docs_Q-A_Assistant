from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


# Document Schemas
class DocumentBase(BaseModel):
    doc_id: str = Field(..., min_length=1, max_length=50, description="Unique document identifier, e.g. D-01")
    title: str = Field(..., min_length=1, max_length=255, description="Document title")
    content: str = Field(..., min_length=1, description="Document body text")


class DocumentCreate(DocumentBase):
    pass


class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    content: Optional[str] = Field(None, min_length=1)


class DocumentResponse(DocumentBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Q&A Schemas
class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="User question in plain English (1 to 500 characters)"
    )


class RetrievedChunkDetail(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    bm25_score: float = 0.0
    cosine_score: Optional[float] = None
    rrf_score: float = 0.0
    rank: int = 0


class DecisionDetail(BaseModel):
    passed_evidence_gate: bool
    gate_reason: str
    flagged_terms: List[str] = []
    verification_passed: Optional[bool] = None
    verification_error: Optional[str] = None


class AnswerResponse(BaseModel):
    status: str = Field(
        ...,
        description="One of: answered, not_covered, needs_clarification, degraded, abstain"
    )
    answer: str
    source_doc_id: Optional[str] = None
    source_title: Optional[str] = None
    supporting_quote: Optional[str] = None
    abstention_reason: Optional[str] = None
    flagged_terms: List[str] = []
    retrieval_mode: str = "hybrid"  # 'hybrid' or 'bm25_only'
    retrieved_chunks: List[RetrievedChunkDetail] = []
    decision_details: Optional[DecisionDetail] = None
    latency_ms: Optional[float] = None


# Question Log Schemas
class QuestionLogResponse(BaseModel):
    id: int
    question: str
    status: str
    answer: Optional[str] = None
    source_doc_id: Optional[str] = None
    source_title: Optional[str] = None
    supporting_quote: Optional[str] = None
    abstention_reason: Optional[str] = None
    flagged_terms: Optional[str] = None
    retrieval_mode: str
    latency_ms: Optional[float] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# System Health Schema
class HealthResponse(BaseModel):
    status: str
    retrieval_mode: str  # 'hybrid' or 'bm25_only'
    database_connected: bool
    document_count: int
    chunk_count: int
    gemini_api_configured: bool
