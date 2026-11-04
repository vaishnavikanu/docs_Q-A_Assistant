import pytest
from app.verifier import verify_grounded_answer
from app.generator import GroundedAnswerPayload
from app.config import settings

SAMPLE_CHUNKS = [
    {
        "chunk_id": "D-01-c1",
        "doc_id": "D-01",
        "title": "Resetting Your Password",
        "text": "Note: password reset links expire after 15 minutes. If yours has expired, don't try to reuse the old email -- just go back to the login page and request a brand new reset link.",
    },
    {
        "chunk_id": "D-06-c1",
        "doc_id": "D-06",
        "title": "API Rate Limits",
        "text": "Rate limits: Growth plans are limited to 60 requests per minute, and Enterprise plans are limited to 300 requests per minute, per API key. Exceeding the limit returns an HTTP 429 response.",
    }
]


def test_verifier_passes_grounded_answer():
    payload = GroundedAnswerPayload(
        status="answered",
        answer="Password reset links expire after 15 minutes.",
        cited_chunk_ids=["D-01-c1"],
        supporting_quotes=["Note: password reset links expire after 15 minutes."],
    )
    res = verify_grounded_answer(payload, SAMPLE_CHUNKS, flagged_terms=[])
    assert res.passed is True
    assert res.source_doc_id == "D-01"
    assert res.final_answer == "Password reset links expire after 15 minutes."


def test_verifier_rejects_hallucinated_unit():
    # '15 hours' instead of '15 minutes'
    payload = GroundedAnswerPayload(
        status="answered",
        answer="Password reset links expire after 15 hours.",
        cited_chunk_ids=["D-01-c1"],
        supporting_quotes=["Note: password reset links expire after 15 minutes."],
    )
    res = verify_grounded_answer(payload, SAMPLE_CHUNKS, flagged_terms=[])
    assert res.passed is False
    assert res.final_answer == settings.FALLBACK_ANSWER
    assert "15 hours" in res.error_message


def test_verifier_rejects_hallucinated_rate_unit():
    # '60 requests per second' instead of '60 requests per minute'
    payload = GroundedAnswerPayload(
        status="answered",
        answer="Growth plans are limited to 60 requests per second.",
        cited_chunk_ids=["D-06-c1"],
        supporting_quotes=["Rate limits: Growth plans are limited to 60 requests per minute"],
    )
    res = verify_grounded_answer(payload, SAMPLE_CHUNKS, flagged_terms=[])
    assert res.passed is False
    assert res.final_answer == settings.FALLBACK_ANSWER
    assert "per second" in res.error_message


def test_verifier_rejects_flagged_unknown_term_in_answer():
    payload = GroundedAnswerPayload(
        status="answered",
        answer="You can easily connect Solstice to Salesforce by installing our plugin.",
        cited_chunk_ids=["D-01-c1"],
        supporting_quotes=["Note: password reset links expire after 15 minutes."],
    )
    res = verify_grounded_answer(payload, SAMPLE_CHUNKS, flagged_terms=["Salesforce"])
    assert res.passed is False
    assert res.final_answer == settings.FALLBACK_ANSWER
    assert "Salesforce" in res.error_message


def test_verifier_rejects_uncited_or_fake_chunk_id():
    payload = GroundedAnswerPayload(
        status="answered",
        answer="Password reset links expire after 15 minutes.",
        cited_chunk_ids=["D-99-c0"],  # Fake chunk
        supporting_quotes=["Note: password reset links expire after 15 minutes."],
    )
    res = verify_grounded_answer(payload, SAMPLE_CHUNKS, flagged_terms=[])
    assert res.passed is False
    assert res.final_answer == settings.FALLBACK_ANSWER
    assert "D-99-c0" in res.error_message


def test_verifier_rejects_fabricated_quote():
    payload = GroundedAnswerPayload(
        status="answered",
        answer="Password reset links expire after 15 minutes.",
        cited_chunk_ids=["D-01-c1"],
        supporting_quotes=["Links will expire exactly after 15 minutes of inactivity."],  # Non-verbatim
    )
    res = verify_grounded_answer(payload, SAMPLE_CHUNKS, flagged_terms=[])
    assert res.passed is False
    assert res.final_answer == settings.FALLBACK_ANSWER
    assert "does not appear verbatim" in res.error_message


def test_verifier_handles_not_covered_status():
    payload = GroundedAnswerPayload(
        status="not_covered",
        answer="The documentation does not cover this.",
        cited_chunk_ids=[],
        supporting_quotes=[],
    )
    res = verify_grounded_answer(payload, SAMPLE_CHUNKS, flagged_terms=["Salesforce"])
    assert res.passed is True
    assert res.final_answer == settings.FALLBACK_ANSWER
