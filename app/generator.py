import json
import logging
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class GroundedAnswerPayload(BaseModel):
    status: Literal["answered", "not_covered", "needs_clarification"] = Field(
        ...,
        description="Status of the answer: answered (grounded in docs), not_covered (docs lack answer), needs_clarification (ambiguous)"
    )
    answer: str = Field(
        ...,
        description="The grounded answer text, or a statement that the docs do not cover the topic."
    )
    cited_chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs that directly support the answer, e.g. ['D-01-c0']"
    )
    supporting_quotes: List[str] = Field(
        default_factory=list,
        description="Exact verbatim sentences copied from the cited chunks providing evidence."
    )


def format_passages_for_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunk excerpts for inclusion in the prompt."""
    blocks = []
    for c in chunks:
        cid = c.get("chunk_id", "UNKNOWN")
        doc_id = c.get("doc_id", "UNKNOWN")
        title = c.get("title", "Untitled")
        text = c.get("text", "")
        blocks.append(f"--- CHUNK ID: {cid} | DOC: {doc_id} ({title}) ---\n{text}")
    return "\n\n".join(blocks)


def generate_grounded_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    flagged_terms: List[str],
    api_key: Optional[str] = None,
) -> GroundedAnswerPayload:
    """Generate a strictly grounded answer using Gemini LLM with structured output schema.
    
    Adheres to Elena Torres's constraint:
    If the documentation does not explicitly answer the question or depends on unfamiliar terms,
    the model must return status 'not_covered'.
    """
    resolved_key = api_key or settings.GEMINI_API_KEY
    if not resolved_key or not GENAI_AVAILABLE:
        logger.info("Gemini API key not configured or google-genai unavailable; returning degraded status.")
        return GroundedAnswerPayload(
            status="not_covered",
            answer=settings.FALLBACK_ANSWER,
            cited_chunk_ids=[],
            supporting_quotes=[]
        )

    passages_text = format_passages_for_prompt(retrieved_chunks)

    system_instruction = (
        "You are the official Solstice Learning Documentation Q&A Assistant.\n"
        "Your task is to answer admin questions based solely on the provided documentation excerpts below.\n\n"
        "CRITICAL GROUNDING RULES:\n"
        "1. Elena's Rule: If the documentation does not genuinely and explicitly answer the question, "
        "you MUST set status to 'not_covered'. Do NOT make anything up or extrapolate.\n"
        "2. Do NOT use outside knowledge. The documentation provided is your only source of truth.\n"
        "3. Unknown Terms: If the question asks about any term flagged as not existing in the documentation "
        "(e.g. unmentioned integrations or tools), set status to 'not_covered'.\n"
        "4. Exact Quotations: For status 'answered', you MUST provide verbatim sentences from the cited chunks "
        "in the 'supporting_quotes' list. Do not paraphrase the quotations.\n"
        "5. Cited Chunks: In 'cited_chunk_ids', list only the CHUNK IDs from which you extracted facts.\n"
        "6. If the question is ambiguous or lacks necessary context, set status to 'needs_clarification'."
    )

    flagged_warning = ""
    if flagged_terms:
        flagged_warning = (
            f"\n\nFLAGGED UNKNOWN TERMS IN QUESTION: {flagged_terms}\n"
            f"NOTE: The documentation never mentions these terms. If the question asks whether Solstice supports, "
            f"connects to, or has them, you MUST return status 'not_covered'."
        )

    prompt = (
        f"DOCUMENTATION PASSAGES:\n"
        f"{passages_text}\n"
        f"{flagged_warning}\n\n"
        f"USER QUESTION: {question}\n\n"
        f"Provide your response as structured JSON matching the requested schema."
    )

    client = genai.Client(api_key=resolved_key)
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=GroundedAnswerPayload,
    )

    # Candidate models in priority order (starting with user's configured model)
    primary_model = settings.GEMINI_GENERATION_MODEL
    candidate_models = [primary_model]
    for alt in ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]:
        if alt not in candidate_models:
            candidate_models.append(alt)

    for model_name in candidate_models:
        try:
            logger.info(f"Generating grounded answer using Gemini model '{model_name}'...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )

            raw_text = response.text
            if not raw_text:
                continue

            parsed = json.loads(raw_text)
            return GroundedAnswerPayload(**parsed)
        except Exception as e:
            err_msg = str(e)
            logger.warning(f"Generation with model '{model_name}' failed: {err_msg}")
            # If model is 404 (sunset) or 503 (high demand), try next candidate
            if "404" in err_msg or "503" in err_msg or "UNAVAILABLE" in err_msg:
                continue
            # For other errors, try next model

    return GroundedAnswerPayload(
        status="not_covered",
        answer=settings.FALLBACK_ANSWER,
        cited_chunk_ids=[],
        supporting_quotes=[]
    )
