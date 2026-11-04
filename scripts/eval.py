#!/usr/bin/env python3
"""Evaluation suite for Solstice Documentation Q&A Assistant.

Runs benchmarks against the running FastAPI service or local engine.
Tests all 5 official case study questions plus edge cases (paraphrased, irrelevant, unknown integrations).
"""

import sys
import json
import argparse
import requests
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

FALLBACK_ANSWER = "I don't have an answer for that."

BENCHMARK_CASES = [
    {
        "id": "Q-1",
        "question": "How do I reset my password?",
        "expected_doc_id": "D-01",
        "expected_keywords": ["forgot password", "reset link"],
        "should_abstain": False,
    },
    {
        "id": "Q-2",
        "question": "How long do I have to use a password reset link before it expires?",
        "expected_doc_id": "D-01",
        "expected_keywords": ["15 minutes"],
        "should_abstain": False,
    },
    {
        "id": "Q-3",
        "question": "What's the maximum number of rows I can export to CSV?",
        "expected_doc_id": "D-03",
        "expected_keywords": ["10,000", "rows"],
        "should_abstain": False,
    },
    {
        "id": "Q-4",
        "question": "What are the API rate limits for the Growth plan?",
        "expected_doc_id": "D-06",
        "expected_keywords": ["60 requests per minute"],
        "should_abstain": False,
    },
    {
        "id": "Q-5",
        "question": "Can I connect Solstice to Salesforce?",
        "expected_doc_id": None,
        "expected_keywords": [],
        "should_abstain": True,
    },
    # Edge Cases & Robustness
    {
        "id": "EDGE-1 (Paraphrased Q-2)",
        "question": "When does the password reset link expire?",
        "expected_doc_id": "D-01",
        "expected_keywords": ["15 minutes"],
        "should_abstain": False,
    },
    {
        "id": "EDGE-2 (Unknown Integration)",
        "question": "Can I integrate with HubSpot CRM?",
        "expected_doc_id": None,
        "expected_keywords": [],
        "should_abstain": True,
    },
    {
        "id": "EDGE-3 (Off-topic question)",
        "question": "What is the capital of Australia?",
        "expected_doc_id": None,
        "expected_keywords": [],
        "should_abstain": True,
    },
]


def evaluate_via_api(api_url: str):
    print("=" * 80)
    print(f"Running Solstice Docs Q&A Assistant Evaluation against: {api_url}")
    print("=" * 80)

    try:
        health_resp = requests.get(f"{api_url}/health", timeout=5)
        health_data = health_resp.json()
        print(f"Server Status: {health_data.get('status')}")
        print(f"Retrieval Mode: {health_data.get('retrieval_mode')}")
        print(f"Document Count: {health_data.get('document_count')}")
        print(f"Chunk Count: {health_data.get('chunk_count')}")
        print(f"Gemini Configured: {health_data.get('gemini_api_configured')}")
    except Exception as e:
        print(f"ERROR: Could not connect to API server at {api_url}: {e}")
        return False

    print("-" * 80)
    passed_count = 0
    total_count = len(BENCHMARK_CASES)

    for case in BENCHMARK_CASES:
        cid = case["id"]
        q = case["question"]
        expected_doc = case["expected_doc_id"]
        should_abstain = case["should_abstain"]

        try:
            resp = requests.post(f"{api_url}/ask", json={"question": q}, timeout=25)
            data = resp.json()
            answer = data.get("answer", "")
            source_doc = data.get("source_doc_id")
            latency = data.get("latency_ms", 0.0)

            # Check correctness
            if should_abstain:
                success = (answer == FALLBACK_ANSWER)
                details = f"Abstained correctly with: '{answer}'" if success else f"FAILED: Expected fallback, got: '{answer}'"
            else:
                # If Gemini is not configured, it will return fallback or degraded
                if not health_data.get("gemini_api_configured"):
                    # In degraded mode, top chunk check
                    top_chunk = data.get("retrieved_chunks", [{}])[0]
                    retrieved_doc = top_chunk.get("doc_id")
                    success = (retrieved_doc == expected_doc)
                    details = f"Degraded mode: Top retrieved doc {retrieved_doc} == {expected_doc}"
                else:
                    doc_ok = (source_doc == expected_doc)
                    keywords_ok = all(k.lower() in answer.lower() for k in case["expected_keywords"])
                    success = doc_ok and keywords_ok
                    details = f"Doc: {source_doc} (expected {expected_doc}) | Answer: {answer[:60]}..."

            status_symbol = "PASS" if success else "FAIL"
            if success:
                passed_count += 1

            print(f"[{status_symbol}] {cid}: '{q}'")
            print(f"         Result: {details} ({latency} ms)")

        except Exception as e:
            print(f"[FAIL] {cid}: Error during request: {e}")

    print("=" * 80)
    print(f"Evaluation Summary: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.1f}%)")
    print("=" * 80)
    return passed_count == total_count


def evaluate_local():
    print("=" * 80)
    print("Running Solstice Docs Q&A Assistant Evaluation in LOCAL ENGINE mode")
    print("=" * 80)

    from app.database import SessionLocal
    from app.crud import init_db, seed_documents_if_empty
    from app.config import settings
    from app.engine import engine_instance

    init_db()
    db = SessionLocal()
    seed_documents_if_empty(db, settings.DEFAULT_DOCS_JSON_PATH)
    engine_instance.reload_indexes(db)

    print(f"Retrieval Mode: {engine_instance.retrieval_mode}")
    print(f"Chunks Count: {engine_instance.chunk_count}")
    print(f"Gemini API Configured: {bool(settings.GEMINI_API_KEY)}")
    print("-" * 80)

    passed_count = 0
    total_count = len(BENCHMARK_CASES)

    for case in BENCHMARK_CASES:
        cid = case["id"]
        q = case["question"]
        expected_doc = case["expected_doc_id"]
        should_abstain = case["should_abstain"]

        ans_resp = engine_instance.answer_question(q, db)
        answer = ans_resp.answer
        source_doc = ans_resp.source_doc_id
        latency = ans_resp.latency_ms

        if should_abstain:
            success = (answer == FALLBACK_ANSWER)
            details = f"Abstained correctly with: '{answer}'" if success else f"FAILED: Expected fallback, got: '{answer}'"
        else:
            if not settings.GEMINI_API_KEY:
                # In degraded mode without Gemini key, verify top chunk retrieval
                top_chunk = ans_resp.retrieved_chunks[0] if ans_resp.retrieved_chunks else None
                retrieved_doc = top_chunk.doc_id if top_chunk else None
                success = (retrieved_doc == expected_doc)
                details = f"Degraded mode: Top retrieved doc {retrieved_doc} == {expected_doc}"
            else:
                doc_ok = (source_doc == expected_doc)
                keywords_ok = all(k.lower() in answer.lower() for k in case["expected_keywords"])
                success = doc_ok and keywords_ok
                details = f"Doc: {source_doc} (expected {expected_doc}) | Answer: {answer[:60]}..."

        status_symbol = "PASS" if success else "FAIL"
        if success:
            passed_count += 1

        print(f"[{status_symbol}] {cid}: '{q}'")
        print(f"         Result: {details} ({latency} ms)")

    db.close()
    print("=" * 80)
    print(f"Evaluation Summary: {passed_count}/{total_count} tests passed ({passed_count/total_count*100:.1f}%)")
    print("=" * 80)
    return passed_count == total_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Solstice Docs Q&A Assistant")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="FastAPI backend URL")
    parser.add_argument("--local", action="store_true", help="Run directly against local DocsQAEngine")
    args = parser.parse_args()

    if args.local:
        success = evaluate_local()
    else:
        try:
            requests.get(f"{args.api_url}/health", timeout=2)
            success = evaluate_via_api(args.api_url)
        except Exception:
            print("API server not detected. Falling back to direct local engine evaluation...")
            success = evaluate_local()

    sys.exit(0 if success else 1)
