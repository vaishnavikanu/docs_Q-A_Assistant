import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import settings
from app.chunking import chunk_all_documents

logger = logging.getLogger(__name__)

# Try importing Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai package not available. Semantic search will degrade to BM25-only.")


def normalize_tokens(text: str) -> List[str]:
    """Tokenize and normalize text for BM25 search."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [tok for tok in text.split() if tok]


def build_corpus_vocabulary(docs: List[Any]) -> Set[str]:
    """Extract all distinct word tokens from documentation articles."""
    vocab = set()
    for doc in docs:
        if isinstance(doc, dict):
            text = f"{doc.get('title', '')} {doc.get('content', '')}"
        else:
            text = f"{getattr(doc, 'title', '')} {getattr(doc, 'content', '')}"
        tokens = re.findall(r"\b[A-Za-z0-9_\-\.\+]+\b", text.lower())
        vocab.update(tokens)
    return vocab


class RetrievalIndex:
    """Manages hybrid retrieval combining BM25 lexical ranking and Qdrant semantic search."""

    COLLECTION_NAME = "solstice_chunks"

    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.qdrant: Optional[QdrantClient] = None
        self.genai_client: Optional[Any] = None
        self.retrieval_mode: str = "bm25_only"
        self.corpus_vocabulary: Set[str] = set()
        self.embedding_dimension: Optional[int] = None

    def build_index(self, docs: List[Any], api_key: Optional[str] = None) -> None:
        """Build or rebuild search indexes from the provided documents."""
        self.chunks = chunk_all_documents(docs, chunk_size=settings.CHUNK_SIZE)
        self.corpus_vocabulary = build_corpus_vocabulary(docs)
        
        if not self.chunks:
            logger.warning("No chunks generated from documents.")
            self.bm25 = None
            self.qdrant = None
            self.retrieval_mode = "bm25_only"
            return

        # 1. Build BM25 Index
        tokenized_corpus = [normalize_tokens(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(self.chunks)} chunks.")

        # 2. Attempt building Semantic / Qdrant Index
        resolved_key = api_key or settings.GEMINI_API_KEY
        if resolved_key and GENAI_AVAILABLE:
            try:
                self._build_semantic_index(resolved_key)
                self.retrieval_mode = "hybrid"
                logger.info(f"Hybrid retrieval index built successfully with {len(self.chunks)} chunks.")
                return
            except Exception as e:
                logger.warning(f"Failed to build semantic index with Gemini: {e}. Falling back to BM25-only.")
        else:
            logger.info("No Gemini API key configured or google-genai unavailable; operating in BM25-only mode.")

        self.retrieval_mode = "bm25_only"
        self.qdrant = None

    def _build_semantic_index(self, api_key: str) -> None:
        """Generate chunk embeddings via Gemini API and store them in in-memory Qdrant."""
        self.genai_client = genai.Client(api_key=api_key)
        self.qdrant = QdrantClient(":memory:")

        # Generate embeddings for each chunk
        texts = [c["text"] for c in self.chunks]
        logger.info(f"Requesting embeddings for {len(texts)} chunks using {settings.GEMINI_EMBEDDING_MODEL}...")

        # Embed content
        response = self.genai_client.models.embed_content(
            model=settings.GEMINI_EMBEDDING_MODEL,
            contents=texts,
        )

        embeddings: List[List[float]] = []
        if hasattr(response, "embeddings") and response.embeddings:
            for item in response.embeddings:
                if hasattr(item, "values"):
                    embeddings.append(list(item.values))
                elif isinstance(item, list):
                    embeddings.append(item)
        elif hasattr(response, "embedding"):
            embeddings = [list(response.embedding.values)]

        if not embeddings or len(embeddings) != len(self.chunks):
            raise ValueError(f"Expected {len(self.chunks)} embeddings, received {len(embeddings)}")

        self.embedding_dimension = len(embeddings[0])

        self.qdrant.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(size=self.embedding_dimension, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i],
                payload={
                    "chunk_id": self.chunks[i]["chunk_id"],
                    "doc_id": self.chunks[i]["doc_id"],
                    "title": self.chunks[i]["title"],
                    "text": self.chunks[i]["text"],
                    "chunk_index": self.chunks[i]["chunk_index"],
                }
            )
            for i in range(len(self.chunks))
        ]

        self.qdrant.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points
        )
        logger.info(f"Qdrant collection '{self.COLLECTION_NAME}' indexed with {len(points)} points (dim={self.embedding_dimension}).")

    def search_bm25(self, query: str) -> List[Tuple[Dict[str, Any], float, int]]:
        """Run BM25 search over all chunks.
        
        Returns:
            List of (chunk_dict, score, rank) sorted descending by score. Rank is 1-based.
        """
        if not self.bm25 or not self.chunks:
            return []

        tokens = normalize_tokens(query)
        scores = self.bm25.get_scores(tokens)

        scored = list(zip(self.chunks, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [(chunk, float(score), rank + 1) for rank, (chunk, score) in enumerate(scored)]

    def search_semantic(self, query: str) -> Optional[List[Tuple[Dict[str, Any], float, int]]]:
        """Run dense semantic search via Gemini embedding + Qdrant.
        
        Returns:
            List of (chunk_dict, cosine_score, rank) sorted descending by score. Rank is 1-based.
            Returns None if semantic search is unavailable or fails.
        """
        if not self.qdrant or not self.genai_client or self.retrieval_mode != "hybrid":
            return None

        try:
            embed_resp = self.genai_client.models.embed_content(
                model=settings.GEMINI_EMBEDDING_MODEL,
                contents=query,
            )
            query_vector: List[float] = []
            if hasattr(embed_resp, "embeddings") and embed_resp.embeddings:
                query_vector = list(embed_resp.embeddings[0].values)
            elif hasattr(embed_resp, "embedding"):
                query_vector = list(embed_resp.embedding.values)

            if not query_vector:
                return None

            results = self.qdrant.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vector,
                limit=len(self.chunks)
            )

            chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
            scored = []
            for rank, pt in enumerate(results.points):
                chunk_id = pt.payload["chunk_id"]
                score = float(pt.score)
                chunk_dict = chunk_by_id[chunk_id]
                scored.append((chunk_dict, score, rank + 1))

            return scored
        except Exception as e:
            logger.warning(f"Semantic search query error: {e}. Falling back to BM25.")
            return None

    def retrieve(
        self,
        query: str,
        top_k: int = settings.TOP_K_CHUNKS,
        rrf_k: int = settings.RRF_K
    ) -> Tuple[List[Dict[str, Any]], float, Optional[float], str]:
        """Execute hybrid retrieval with Reciprocal Rank Fusion (RRF).
        
        Formula:
            RRF(c) = 1 / (rrf_k + rank_bm25) + 1 / (rrf_k + rank_semantic)
            
        Returns:
            - top_chunks: List of chunk dictionaries augmented with scores and ranks.
            - max_bm25_score: Highest BM25 score among all chunks.
            - max_cosine_score: Highest cosine similarity score, or None in degraded mode.
            - effective_retrieval_mode: 'hybrid' or 'bm25_only'
        """
        if not self.chunks:
            return [], 0.0, None, "bm25_only"

        # 1. BM25 Search
        bm25_results = self.search_bm25(query)
        bm25_map = {c["chunk_id"]: (c, score, rank) for c, score, rank in bm25_results}
        max_bm25 = max([score for _, score, _ in bm25_results], default=0.0)

        # 2. Semantic Search
        semantic_results = self.search_semantic(query)
        effective_mode = "hybrid" if semantic_results is not None else "bm25_only"

        semantic_map = {}
        max_cosine = None
        if semantic_results is not None:
            semantic_map = {c["chunk_id"]: (c, score, rank) for c, score, rank in semantic_results}
            max_cosine = max([score for _, score, _ in semantic_results], default=0.0)

        # 3. Reciprocal Rank Fusion
        fused_scores: List[Dict[str, Any]] = []
        for chunk in self.chunks:
            cid = chunk["chunk_id"]
            
            # BM25 rank and score
            _, b_score, b_rank = bm25_map.get(cid, (chunk, 0.0, len(self.chunks)))
            
            if effective_mode == "hybrid" and cid in semantic_map:
                _, s_score, s_rank = semantic_map[cid]
                rrf_score = (1.0 / (rrf_k + b_rank)) + (1.0 / (rrf_k + s_rank))
            else:
                s_score = None
                s_rank = None
                rrf_score = 1.0 / (rrf_k + b_rank)

            fused_chunk = dict(chunk)
            fused_chunk["bm25_score"] = round(b_score, 4)
            fused_chunk["cosine_score"] = round(s_score, 4) if s_score is not None else None
            fused_chunk["rrf_score"] = round(rrf_score, 6)
            fused_chunk["bm25_rank"] = b_rank
            fused_chunk["semantic_rank"] = s_rank
            fused_scores.append(fused_chunk)

        # Sort by RRF score descending
        fused_scores.sort(key=lambda x: x["rrf_score"], reverse=True)

        # Assign final rank
        for idx, fc in enumerate(fused_scores[:top_k]):
            fc["rank"] = idx + 1

        return fused_scores[:top_k], max_bm25, max_cosine, effective_mode
