import time
import json
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.config import settings
from app.crud import get_documents, log_question
from app.retrieval import RetrievalIndex
from app.evidence_gate import evaluate_evidence_gate
from app.generator import generate_grounded_answer
from app.verifier import verify_grounded_answer
from app.schemas import AnswerResponse, RetrievedChunkDetail, DecisionDetail

logger = logging.getLogger(__name__)


class DocsQAEngine:
    """Core orchestration engine for documentation retrieval, evidence gating,
    grounded generation, verification, and question logging.
    """

    def __init__(self):
        self.index = RetrievalIndex()

    def reload_indexes(self, db: Session, api_key: Optional[str] = None) -> None:
        """Fetch current documents from SQLite and rebuild both BM25 and Qdrant indexes."""
        docs = get_documents(db)
        logger.info(f"Reloading DocsQAEngine indexes with {len(docs)} documents...")
        self.index.build_index(docs, api_key=api_key)

    @property
    def retrieval_mode(self) -> str:
        return self.index.retrieval_mode

    @property
    def chunk_count(self) -> int:
        return len(self.index.chunks)

    def answer_question(self, question: str, db: Session, api_key: Optional[str] = None) -> AnswerResponse:
        """Process a user question end-to-end through the 3-layer pipeline."""
        start_time = time.perf_counter()

        # Step 1: Retrieval
        retrieved_chunks, max_bm25, max_cosine, mode = self.index.retrieve(
            query=question,
            top_k=settings.TOP_K_CHUNKS,
            rrf_k=settings.RRF_K
        )

        chunk_details: List[RetrievedChunkDetail] = [
            RetrievedChunkDetail(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                title=c["title"],
                text=c["text"],
                bm25_score=c.get("bm25_score", 0.0),
                cosine_score=c.get("cosine_score"),
                rrf_score=c.get("rrf_score", 0.0),
                rank=c.get("rank", 0),
            )
            for c in retrieved_chunks
        ]

        has_llm = bool(api_key or settings.GEMINI_API_KEY)

        # Step 2: Evidence Gate (Layer 1)
        gate_res = evaluate_evidence_gate(
            question=question,
            max_bm25_score=max_bm25,
            max_cosine_score=max_cosine,
            retrieval_mode=mode,
            corpus_vocab=self.index.corpus_vocabulary,
            has_llm=has_llm,
        )

        decision = DecisionDetail(
            passed_evidence_gate=gate_res.passed,
            gate_reason=gate_res.reason,
            flagged_terms=gate_res.flagged_terms,
            verification_passed=None,
            verification_error=None,
        )

        # If evidence gate fails, abstain immediately without calling LLM
        if not gate_res.passed:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            abstain_status = "not_covered"
            abstain_reason = gate_res.reason

            # Record in question log
            log_question(
                db=db,
                question=question,
                status=abstain_status,
                answer=settings.FALLBACK_ANSWER,
                source_doc_id=None,
                source_title=None,
                supporting_quote=None,
                abstention_reason=abstain_reason,
                flagged_terms=json.dumps(gate_res.flagged_terms),
                retrieval_mode=mode,
                latency_ms=latency_ms,
            )

            return AnswerResponse(
                status=abstain_status,
                answer=settings.FALLBACK_ANSWER,
                source_doc_id=None,
                source_title=None,
                supporting_quote=None,
                abstention_reason=abstain_reason,
                flagged_terms=gate_res.flagged_terms,
                retrieval_mode=mode,
                retrieved_chunks=chunk_details,
                decision_details=decision,
                latency_ms=latency_ms,
            )

        # Degraded mode without LLM:
        if not has_llm:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            degraded_reason = "Gemini API key is not configured; running in BM25-only degraded mode."
            log_question(
                db=db,
                question=question,
                status="degraded",
                answer=settings.FALLBACK_ANSWER,
                source_doc_id=retrieved_chunks[0]["doc_id"] if retrieved_chunks else None,
                source_title=retrieved_chunks[0]["title"] if retrieved_chunks else None,
                supporting_quote=None,
                abstention_reason=degraded_reason,
                flagged_terms=json.dumps(gate_res.flagged_terms),
                retrieval_mode=mode,
                latency_ms=latency_ms,
            )
            return AnswerResponse(
                status="degraded",
                answer=settings.FALLBACK_ANSWER,
                source_doc_id=None,
                source_title=None,
                supporting_quote=None,
                abstention_reason=degraded_reason,
                flagged_terms=gate_res.flagged_terms,
                retrieval_mode=mode,
                retrieved_chunks=chunk_details,
                decision_details=decision,
                latency_ms=latency_ms,
            )

        # Step 3: Grounded Gemini Answer Generation (Layer 2)
        llm_payload = generate_grounded_answer(
            question=question,
            retrieved_chunks=retrieved_chunks,
            flagged_terms=gate_res.flagged_terms,
            api_key=api_key or settings.GEMINI_API_KEY,
        )

        # Step 4: Deterministic Verification (Layer 3)
        v_res = verify_grounded_answer(
            payload=llm_payload,
            retrieved_chunks=retrieved_chunks,
            flagged_terms=gate_res.flagged_terms,
        )

        decision.verification_passed = v_res.passed
        decision.verification_error = v_res.error_message

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        final_status = "answered" if (v_res.passed and v_res.final_answer != settings.FALLBACK_ANSWER) else "not_covered"
        final_answer = v_res.final_answer
        final_abstention_reason = v_res.error_message if final_status != "answered" else None

        # Record in question log
        log_question(
            db=db,
            question=question,
            status=final_status,
            answer=final_answer,
            source_doc_id=v_res.source_doc_id,
            source_title=v_res.source_title,
            supporting_quote=v_res.supporting_quote,
            abstention_reason=final_abstention_reason,
            flagged_terms=json.dumps(gate_res.flagged_terms),
            retrieval_mode=mode,
            latency_ms=latency_ms,
        )

        return AnswerResponse(
            status=final_status,
            answer=final_answer,
            source_doc_id=v_res.source_doc_id,
            source_title=v_res.source_title,
            supporting_quote=v_res.supporting_quote,
            abstention_reason=final_abstention_reason,
            flagged_terms=gate_res.flagged_terms,
            retrieval_mode=mode,
            retrieved_chunks=chunk_details,
            decision_details=decision,
            latency_ms=latency_ms,
        )


# Global singleton instance of DocsQAEngine
engine_instance = DocsQAEngine()
