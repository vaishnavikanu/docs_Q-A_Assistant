import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

from app.database import Base
from app.models import Document, QuestionLog
from app.schemas import DocumentCreate, DocumentUpdate
from app.crud import (
    seed_documents_if_empty,
    get_documents,
    get_document_by_doc_id,
    create_document,
    update_document,
    delete_document,
    log_question,
    get_unanswered_questions,
)

# Test in-memory SQLite database
TEST_DB_URL = "sqlite:///:memory:"


@pytest.fixture
def db_session():
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_seed_documents(db_session):
    docs_json_path = Path("starter_kit/data/docs.json")
    seeded = seed_documents_if_empty(db_session, docs_json_path)
    assert seeded == 6
    
    docs = get_documents(db_session)
    assert len(docs) == 6
    doc_ids = {d.doc_id for d in docs}
    assert "D-01" in doc_ids
    assert "D-06" in doc_ids

    # Second run should skip seeding
    seeded_again = seed_documents_if_empty(db_session, docs_json_path)
    assert seeded_again == 0


def test_document_crud(db_session):
    # Create
    new_doc = DocumentCreate(
        doc_id="D-99",
        title="Test Article",
        content="This is test content for unit testing."
    )
    created = create_document(db_session, new_doc)
    assert created.id is not None
    assert created.doc_id == "D-99"
    assert created.title == "Test Article"

    # Read
    fetched = get_document_by_doc_id(db_session, "D-99")
    assert fetched is not None
    assert fetched.content == "This is test content for unit testing."

    # Update
    update_data = DocumentUpdate(title="Updated Test Article")
    updated = update_document(db_session, "D-99", update_data)
    assert updated.title == "Updated Test Article"
    assert updated.content == "This is test content for unit testing."

    # Delete
    deleted = delete_document(db_session, "D-99")
    assert deleted is True
    assert get_document_by_doc_id(db_session, "D-99") is None


def test_question_log_and_unanswered(db_session):
    # Log answered question
    log1 = log_question(
        db=db_session,
        question="How do I reset my password?",
        status="answered",
        answer="Click Forgot password...",
        source_doc_id="D-01",
        source_title="Resetting Your Password",
        supporting_quote="Click 'Forgot password?' on the login screen",
        retrieval_mode="hybrid",
        latency_ms=120.5
    )
    assert log1.id is not None

    # Log unanswered question
    log2 = log_question(
        db=db_session,
        question="Can I connect to Salesforce?",
        status="not_covered",
        answer="I don't have an answer for that.",
        abstention_reason="Flagged unknown term 'Salesforce'",
        flagged_terms='["salesforce"]',
        retrieval_mode="hybrid",
        latency_ms=85.0
    )
    assert log2.id is not None

    unanswered = get_unanswered_questions(db_session)
    assert len(unanswered) == 1
    assert unanswered[0].question == "Can I connect to Salesforce?"
    assert unanswered[0].status == "not_covered"
