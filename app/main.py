import logging
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, SessionLocal
from app.crud import (
    init_db,
    seed_documents_if_empty,
    get_documents,
    get_document_by_doc_id,
    create_document,
    update_document,
    delete_document,
    get_unanswered_questions,
    get_question_logs,
)
from app.schemas import (
    QuestionRequest,
    AnswerResponse,
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    QuestionLogResponse,
    HealthResponse,
)
from app.engine import engine_instance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize SQLite, seed data if empty, and build search indexes
    logger.info("Initializing database and search indexes...")
    init_db()
    db = SessionLocal()
    try:
        seeded = seed_documents_if_empty(db, settings.DEFAULT_DOCS_JSON_PATH)
        if seeded > 0:
            logger.info(f"Seeded {seeded} documents into database.")
        engine_instance.reload_indexes(db)
        logger.info(
            f"DocsQA Engine ready. Retrieval mode: {engine_instance.retrieval_mode}, "
            f"Chunks: {engine_instance.chunk_count}"
        )
    finally:
        db.close()
    yield
    # Shutdown
    logger.info("DocsQA Assistant shutting down.")


app = FastAPI(
    title="Solstice Documentation Q&A Assistant",
    description=(
        "Enterprise-grade Documentation Q&A Assistant with hybrid BM25 + Qdrant retrieval, "
        "Reciprocal Rank Fusion, grounded Gemini answer generation, and deterministic verification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Streamlit or web UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    """System health endpoint reporting retrieval mode, document count, and index status."""
    doc_count = len(get_documents(db))
    return HealthResponse(
        status="healthy",
        retrieval_mode=engine_instance.retrieval_mode,
        database_connected=True,
        document_count=doc_count,
        chunk_count=engine_instance.chunk_count,
        gemini_api_configured=bool(settings.GEMINI_API_KEY),
    )


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
def ask_question(request: QuestionRequest, db: Session = Depends(get_db)):
    """Ask a question against Solstice documentation.
    
    Validates input (1-500 characters), retrieves top passages with RRF,
    enforces the evidence gate, generates grounded answer, and runs deterministic verification.
    """
    cleaned_query = request.question.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or solely whitespace."
        )

    return engine_instance.answer_question(cleaned_query, db)


@app.get("/unanswered", response_model=List[QuestionLogResponse], tags=["Analytics"])
def get_unanswered(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve logged questions that the documentation could not answer."""
    return get_unanswered_questions(db, skip=skip, limit=limit)


@app.get("/logs", response_model=List[QuestionLogResponse], tags=["Analytics"])
def get_all_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve full question query history."""
    return get_question_logs(db, skip=skip, limit=limit)


# Document CRUD Endpoints
@app.get("/documents", response_model=List[DocumentResponse], tags=["Documents"])
def list_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all documentation articles."""
    return get_documents(db, skip=skip, limit=limit)


@app.get("/documents/{doc_id}", response_model=DocumentResponse, tags=["Documents"])
def get_document(doc_id: str, db: Session = Depends(get_db)):
    """Get a single document by doc_id."""
    doc = get_document_by_doc_id(db, doc_id)
    if not doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{doc_id}' not found.")
    return doc


@app.post("/documents", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, tags=["Documents"])
def add_document(doc_in: DocumentCreate, db: Session = Depends(get_db)):
    """Add a new document article and refresh search indexes."""
    try:
        created = create_document(db, doc_in)
        engine_instance.reload_indexes(db)
        return created
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.put("/documents/{doc_id}", response_model=DocumentResponse, tags=["Documents"])
def edit_document(doc_id: str, doc_in: DocumentUpdate, db: Session = Depends(get_db)):
    """Update a document article and refresh search indexes."""
    updated = update_document(db, doc_id, doc_in)
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{doc_id}' not found.")
    engine_instance.reload_indexes(db)
    return updated


@app.delete("/documents/{doc_id}", tags=["Documents"])
def remove_document(doc_id: str, db: Session = Depends(get_db)):
    """Delete a document article and refresh search indexes."""
    success = delete_document(db, doc_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Document '{doc_id}' not found.")
    engine_instance.reload_indexes(db)
    return {"message": f"Document '{doc_id}' deleted successfully"}
