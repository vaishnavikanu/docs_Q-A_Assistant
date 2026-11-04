import re
from typing import List, Dict, Any
from app.config import settings


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly based on punctuation boundaries.
    
    Handles standard period, exclamation, and question mark endings followed by whitespace.
    """
    if not text:
        return []
    # Split on sentence boundaries (period, question mark, exclamation point followed by whitespace)
    # Also handle markdown/bullet boundaries if present
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text.strip()]


def chunk_text_sentence_aware(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE
) -> List[str]:
    """Split text into sentence-aware chunks of roughly `chunk_size` characters.
    
    Guarantees:
    1. Cuts only between sentences whenever possible.
    2. Keeps the remainder (never discards content).
    3. Preserves all sentences without truncation.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence)
        # Calculate new length if sentence were added (including separating space)
        potential_len = current_len + (1 if current_len > 0 else 0) + s_len
        
        # If adding this sentence would exceed chunk_size and we already have sentences accumulated:
        if current_sentences and potential_len > chunk_size:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_len = s_len
        else:
            current_sentences.append(sentence)
            current_len = potential_len

    # Never discard the remainder
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def chunk_document(
    doc_id: str,
    title: str,
    content: str,
    chunk_size: int = settings.CHUNK_SIZE
) -> List[Dict[str, Any]]:
    """Chunk a single document into chunk dictionaries with metadata."""
    text_chunks = chunk_text_sentence_aware(content, chunk_size=chunk_size)
    result = []
    for idx, text in enumerate(text_chunks):
        result.append({
            "chunk_id": f"{doc_id}-c{idx}",
            "doc_id": doc_id,
            "title": title,
            "chunk_index": idx,
            "text": text
        })
    return result


def chunk_all_documents(
    docs: List[Any],
    chunk_size: int = settings.CHUNK_SIZE
) -> List[Dict[str, Any]]:
    """Chunk a list of document objects or dicts into a flat list of chunk dicts."""
    all_chunks = []
    for doc in docs:
        if isinstance(doc, dict):
            doc_id = doc["doc_id"]
            title = doc["title"]
            content = doc["content"]
        else:
            doc_id = doc.doc_id
            title = doc.title
            content = doc.content
            
        chunks = chunk_document(doc_id, title, content, chunk_size=chunk_size)
        all_chunks.extend(chunks)
    return all_chunks
