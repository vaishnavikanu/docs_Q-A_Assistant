import json
import logging
from pathlib import Path
from typing import List, Optional
from sqlalchemy.orm import Session

from app.database import Base, engine
from app.models import Document, QuestionLog
from app.schemas import DocumentCreate, DocumentUpdate

logger = logging.getLogger(__name__)


def init_db():
    """Create all database tables defined in models."""
    Base.metadata.create_all(bind=engine)


def seed_documents_if_empty(db: Session, docs_json_path: Path) -> int:
    """Seed documents from JSON file into the database if the table is empty.
    
    Returns the count of seeded documents.
    """
    existing_count = db.query(Document).count()
    if existing_count > 0:
        logger.info(f"Database already contains {existing_count} documents. Skipping seeding.")
        return 0

    if not docs_json_path.exists():
        logger.warning(f"Seed file not found at {docs_json_path}")
        return 0

    with open(docs_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_docs = data.get("docs", [])
    seeded = 0
    for item in raw_docs:
        doc = Document(
            doc_id=item["doc_id"],
            title=item["title"],
            content=item["content"]
        )
        db.add(doc)
        seeded += 1

    db.commit()
    logger.info(f"Successfully seeded {seeded} documents from {docs_json_path}.")
    return seeded


def get_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
    return db.query(Document).offset(skip).limit(limit).all()


def get_document_by_doc_id(db: Session, doc_id: str) -> Optional[Document]:
    return db.query(Document).filter(Document.doc_id == doc_id).first()


def create_document(db: Session, doc_in: DocumentCreate) -> Document:
    existing = get_document_by_doc_id(db, doc_in.doc_id)
    if existing:
        raise ValueError(f"Document with doc_id '{doc_in.doc_id}' already exists.")
    
    doc = Document(
        doc_id=doc_in.doc_id,
        title=doc_in.title,
        content=doc_in.content
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def update_document(db: Session, doc_id: str, doc_in: DocumentUpdate) -> Optional[Document]:
    doc = get_document_by_doc_id(db, doc_id)
    if not doc:
        return None
    
    if doc_in.title is not None:
        doc.title = doc_in.title
    if doc_in.content is not None:
        doc.content = doc_in.content
        
    db.commit()
    db.refresh(doc)
    return doc


def delete_document(db: Session, doc_id: str) -> bool:
    doc = get_document_by_doc_id(db, doc_id)
    if not doc:
        return False
    
    db.delete(doc)
    db.commit()
    return True


def log_question(
    db: Session,
    question: str,
    status: str,
    answer: Optional[str] = None,
    source_doc_id: Optional[str] = None,
    source_title: Optional[str] = None,
    supporting_quote: Optional[str] = None,
    abstention_reason: Optional[str] = None,
    flagged_terms: Optional[str] = None,
    retrieval_mode: str = "hybrid",
    latency_ms: Optional[float] = None,
) -> QuestionLog:
    entry = QuestionLog(
        question=question,
        status=status,
        answer=answer,
        source_doc_id=source_doc_id,
        source_title=source_title,
        supporting_quote=supporting_quote,
        abstention_reason=abstention_reason,
        flagged_terms=flagged_terms,
        retrieval_mode=retrieval_mode,
        latency_ms=latency_ms,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def get_unanswered_questions(db: Session, skip: int = 0, limit: int = 100) -> List[QuestionLog]:
    """Retrieve logged questions that could not be answered by the documentation."""
    return db.query(QuestionLog).filter(
        QuestionLog.status.in_(["not_covered", "needs_clarification", "abstain", "degraded"])
    ).order_by(QuestionLog.created_at.desc()).offset(skip).limit(limit).all()


def get_question_logs(db: Session, skip: int = 0, limit: int = 100) -> List[QuestionLog]:
    return db.query(QuestionLog).order_by(QuestionLog.created_at.desc()).offset(skip).limit(limit).all()
