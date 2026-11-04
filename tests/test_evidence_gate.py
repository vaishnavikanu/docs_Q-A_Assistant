import pytest
from app.evidence_gate import detect_unknown_terms, evaluate_evidence_gate
from app.retrieval import build_corpus_vocabulary
from app.chunking import chunk_all_documents
import json
from pathlib import Path


@pytest.fixture
def sample_corpus_vocab():
    path = Path("starter_kit/data/docs.json")
    with open(path, "r", encoding="utf-8") as f:
        docs = json.load(f)["docs"]
    return build_corpus_vocabulary(docs)


def test_detect_unknown_terms_salesforce(sample_corpus_vocab):
    # Salesforce should be flagged
    q = "Can I connect Solstice to Salesforce?"
    flagged = detect_unknown_terms(q, sample_corpus_vocab)
    assert any("salesforce" in term.lower() for term in flagged)


def test_detect_unknown_terms_hubspot_pdf(sample_corpus_vocab):
    q1 = "Can I integrate with HubSpot?"
    assert any("hubspot" in t.lower() for t in detect_unknown_terms(q1, sample_corpus_vocab))

    q2 = "Can I export reports to PDF?"
    assert any("pdf" in t.lower() for t in detect_unknown_terms(q2, sample_corpus_vocab))


def test_detect_known_terms_not_flagged(sample_corpus_vocab):
    q = "What are the API rate limits for the Growth plan?"
    flagged = detect_unknown_terms(q, sample_corpus_vocab)
    assert len(flagged) == 0

    q2 = "What is the maximum number of rows I can export to CSV?"
    assert len(detect_unknown_terms(q2, sample_corpus_vocab)) == 0

    q3 = "How do I reset my password?"
    assert len(detect_unknown_terms(q3, sample_corpus_vocab)) == 0


def test_evidence_gate_score_floor(sample_corpus_vocab):
    # Weak scores below floor should fail gate
    result = evaluate_evidence_gate(
        question="What is the weather in Paris?",
        max_bm25_score=0.01,
        max_cosine_score=0.15,
        retrieval_mode="hybrid",
        corpus_vocab=sample_corpus_vocab,
        has_llm=True,
    )
    assert result.passed is False
    assert "Scores below floor threshold" in result.reason


def test_evidence_gate_bm25_only_mode(sample_corpus_vocab):
    # In BM25-only mode, unknown term causes immediate abstention
    result = evaluate_evidence_gate(
        question="Can I connect to Salesforce?",
        max_bm25_score=2.5,
        max_cosine_score=None,
        retrieval_mode="bm25_only",
        corpus_vocab=sample_corpus_vocab,
        has_llm=False,
    )
    assert result.passed is False
    assert "cannot verify unfamiliar terms" in result.reason
