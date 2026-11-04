import json
from pathlib import Path
from app.chunking import chunk_document, chunk_all_documents, split_into_sentences


def load_seed_docs():
    path = Path("starter_kit/data/docs.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["docs"]


def test_sentence_aware_chunking_count():
    docs = load_seed_docs()
    chunks = chunk_all_documents(docs, chunk_size=400)
    
    # Requirement from Solution Design: "The result is 10 chunks, each carrying its doc ID, title and chunk ID."
    assert len(chunks) == 10


def test_chunk_metadata_and_retention():
    docs = load_seed_docs()
    chunks = chunk_all_documents(docs, chunk_size=400)
    
    # All doc IDs must be represented
    doc_ids = {c["doc_id"] for c in chunks}
    assert doc_ids == {"D-01", "D-02", "D-03", "D-04", "D-05", "D-06"}
    
    for c in chunks:
        assert "chunk_id" in c
        assert "doc_id" in c
        assert "title" in c
        assert "chunk_index" in c
        assert "text" in c
        assert len(c["text"]) > 0
        assert c["chunk_id"].startswith(c["doc_id"])


def test_no_content_silently_discarded():
    docs = load_seed_docs()
    for doc in docs:
        chunks = chunk_document(doc["doc_id"], doc["title"], doc["content"], chunk_size=400)
        # Reconstructed content by joining chunk texts with space should contain all original words
        joined_chunks = " ".join([c["text"] for c in chunks])
        original_words = doc["content"].split()
        reconstructed_words = joined_chunks.split()
        assert original_words == reconstructed_words


def test_critical_facts_intact():
    docs = load_seed_docs()
    chunks = chunk_all_documents(docs, chunk_size=400)
    chunk_map = {c["chunk_id"]: c["text"] for c in chunks}
    
    # Q-2: 15 minutes expiration in D-01
    assert "15 minutes" in chunk_map["D-01-c1"]
    
    # Q-3: 10,000 rows in D-03
    assert "10,000 rows" in chunk_map["D-03-c1"]
    
    # Q-4: "60 requests per minute" intact in D-06
    assert "60 requests per minute" in chunk_map["D-06-c1"]
    
    # D-02 and D-05 are not dropped
    assert "D-02-c0" in chunk_map
    assert "D-05-c0" in chunk_map
