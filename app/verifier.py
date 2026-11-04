import logging
import re
from typing import List, Dict, Any, Optional, Set
from app.config import settings
from app.generator import GroundedAnswerPayload

logger = logging.getLogger(__name__)


class VerificationResult:
    def __init__(
        self,
        passed: bool,
        error_message: Optional[str] = None,
        final_answer: str = settings.FALLBACK_ANSWER,
        source_doc_id: Optional[str] = None,
        source_title: Optional[str] = None,
        supporting_quote: Optional[str] = None,
    ):
        self.passed = passed
        self.error_message = error_message
        self.final_answer = final_answer
        self.source_doc_id = source_doc_id
        self.source_title = source_title
        self.supporting_quote = supporting_quote


def normalize_ws(text: str) -> str:
    """Normalize multiple spaces and newlines to a single space."""
    return re.sub(r"\s+", " ", text.strip())


def extract_numerical_units(text: str) -> List[str]:
    """Extract number + unit phrases (e.g. '15 minutes', '60 requests per minute', '10,000 rows')."""
    pattern = r"\b(\d+(?:[,\.]\d+)?\s+[a-zA-Z]+(?:\s+per\s+[a-zA-Z]+)?)\b"
    return [match.strip() for match in re.findall(pattern, text, re.IGNORECASE)]


def extract_standalone_numbers(text: str) -> Set[str]:
    """Extract all individual numbers (normalizing comma separators, e.g. '10,000' -> '10000')."""
    raw_nums = re.findall(r"\b\d+(?:[,\.]\d+)?\b", text)
    cleaned = set()
    for n in raw_nums:
        clean_n = n.replace(",", "")
        cleaned.add(clean_n)
    return cleaned


def verify_grounded_answer(
    payload: GroundedAnswerPayload,
    retrieved_chunks: List[Dict[str, Any]],
    flagged_terms: List[str],
) -> VerificationResult:
    """Perform deterministic verification on the generated answer.
    
    Verifies:
    1. If status is 'not_covered' or 'needs_clarification', gracefully return fallback.
    2. Answer must not mention any flagged unknown term.
    3. Every cited chunk ID must exist in retrieved_chunks.
    4. Every supporting quote must appear verbatim in the text of its cited chunk.
    5. Every numerical value and unit in the answer must be present and supported in cited text.
    
    Any failure produces the exact fallback answer: "I don't have an answer for that."
    """
    # 1. Status Check
    if payload.status != "answered":
        logger.info(f"Answer status is '{payload.status}'. Falling back.")
        return VerificationResult(
            passed=True,  # Abstention is expected and correct
            error_message=f"Status was '{payload.status}'",
            final_answer=settings.FALLBACK_ANSWER,
            source_doc_id=None,
            source_title=None,
            supporting_quote=None,
        )

    if not payload.answer or payload.answer.strip() == "":
        return VerificationResult(
            passed=False,
            error_message="Answer text is empty.",
            final_answer=settings.FALLBACK_ANSWER,
        )

    # 2. Flagged Unknown Terms Check
    lower_answer = payload.answer.lower()
    for term in flagged_terms:
        # Match as whole word or term
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, lower_answer):
            msg = f"Verifier rejected answer: contains flagged unknown term '{term}'."
            logger.warning(msg)
            return VerificationResult(
                passed=False,
                error_message=msg,
                final_answer=settings.FALLBACK_ANSWER,
            )

    # 3. Cited Chunk IDs Check
    retrieved_ids = {c["chunk_id"]: c for c in retrieved_chunks}
    if not payload.cited_chunk_ids:
        msg = "Verifier rejected answer: no chunk IDs cited for answered status."
        logger.warning(msg)
        return VerificationResult(
            passed=False,
            error_message=msg,
            final_answer=settings.FALLBACK_ANSWER,
        )

    for cid in payload.cited_chunk_ids:
        if cid not in retrieved_ids:
            msg = f"Verifier rejected answer: cited chunk '{cid}' was not among retrieved chunks."
            logger.warning(msg)
            return VerificationResult(
                passed=False,
                error_message=msg,
                final_answer=settings.FALLBACK_ANSWER,
            )

    # 4. Verbatim Quotation Matching Check
    if not payload.supporting_quotes:
        msg = "Verifier rejected answer: no supporting quotes provided for answered status."
        logger.warning(msg)
        return VerificationResult(
            passed=False,
            error_message=msg,
            final_answer=settings.FALLBACK_ANSWER,
        )

    # Combine text of cited chunks for quote matching
    cited_chunks_text = " ".join([retrieved_ids[cid]["text"] for cid in payload.cited_chunk_ids])
    norm_cited_text = normalize_ws(cited_chunks_text).lower()

    for quote in payload.supporting_quotes:
        norm_quote = normalize_ws(quote).lower()
        if norm_quote not in norm_cited_text:
            msg = f"Verifier rejected answer: quote '{quote}' does not appear verbatim in cited chunk text."
            logger.warning(msg)
            return VerificationResult(
                passed=False,
                error_message=msg,
                final_answer=settings.FALLBACK_ANSWER,
            )

    # 5. Numerical and Unit Accuracy Check
    answer_units = extract_numerical_units(payload.answer)
    for unit_phrase in answer_units:
        norm_unit = normalize_ws(unit_phrase).lower()
        if norm_unit not in norm_cited_text:
            msg = (
                f"Verifier rejected answer: numerical unit '{unit_phrase}' "
                f"is not supported by cited documentation text."
            )
            logger.warning(msg)
            return VerificationResult(
                passed=False,
                error_message=msg,
                final_answer=settings.FALLBACK_ANSWER,
            )

    answer_nums = extract_standalone_numbers(payload.answer)
    cited_nums = extract_standalone_numbers(cited_chunks_text)
    for num in answer_nums:
        if num not in cited_nums:
            msg = f"Verifier rejected answer: number '{num}' is not supported by cited documentation text."
            logger.warning(msg)
            return VerificationResult(
                passed=False,
                error_message=msg,
                final_answer=settings.FALLBACK_ANSWER,
            )

    # Determine primary source document and quote
    primary_chunk_id = payload.cited_chunk_ids[0]
    primary_chunk = retrieved_ids[primary_chunk_id]
    primary_quote = payload.supporting_quotes[0] if payload.supporting_quotes else None

    return VerificationResult(
        passed=True,
        error_message=None,
        final_answer=payload.answer,
        source_doc_id=primary_chunk["doc_id"],
        source_title=primary_chunk["title"],
        supporting_quote=primary_quote,
    )
