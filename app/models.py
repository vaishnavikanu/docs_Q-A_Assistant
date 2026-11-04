from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Document(Base):
    """Document model representing documentation articles."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String(50), unique=True, index=True, nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)


class QuestionLog(Base):
    """Question log model recording queries, statuses, retrieval mode, latency, and abstentions."""
    __tablename__ = "question_log"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    status = Column(String(50), nullable=False)  # 'answered', 'not_covered', 'needs_clarification', 'degraded', 'abstain'
    answer = Column(Text, nullable=True)
    source_doc_id = Column(String(50), nullable=True)
    source_title = Column(String(255), nullable=True)
    supporting_quote = Column(Text, nullable=True)
    abstention_reason = Column(Text, nullable=True)
    flagged_terms = Column(Text, nullable=True)  # JSON-serialized list of flagged terms
    retrieval_mode = Column(String(50), nullable=False, default="hybrid")  # 'hybrid' or 'bm25_only'
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
