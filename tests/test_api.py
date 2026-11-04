import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.engine import engine_instance


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True
    assert data["document_count"] >= 6
    assert data["chunk_count"] >= 10
    assert data["retrieval_mode"] in ["hybrid", "bm25_only"]


def test_ask_empty_question_fails(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_too_long_question_fails(client):
    long_q = "a" * 501
    response = client.post("/ask", json={"question": long_q})
    assert response.status_code == 422  # Pydantic validation error


def test_ask_salesforce_unsupported_abstains(client):
    # Q-5: Can I connect Solstice to Salesforce?
    response = client.post("/ask", json={"question": "Can I connect Solstice to Salesforce?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I don't have an answer for that."
    assert "salesforce" in [t.lower() for t in data["flagged_terms"]]


def test_document_crud_endpoints(client):
    initial_chunks = engine_instance.chunk_count

    # 1. List
    res = client.get("/documents")
    assert res.status_code == 200
    assert len(res.json()) >= 6

    # 2. Add
    new_doc = {
        "doc_id": "D-TEST-01",
        "title": "Custom Webhooks Guide",
        "content": "Solstice supports custom outgoing webhooks for alert triggers."
    }
    create_res = client.post("/documents", json=new_doc)
    assert create_res.status_code == 201
    assert create_res.json()["doc_id"] == "D-TEST-01"

    # Verify index updated
    assert engine_instance.chunk_count == initial_chunks + 1

    # 3. Get single
    get_res = client.get("/documents/D-TEST-01")
    assert get_res.status_code == 200
    assert get_res.json()["title"] == "Custom Webhooks Guide"

    # 4. Edit
    update_res = client.put("/documents/D-TEST-01", json={"title": "Updated Webhooks Guide"})
    assert update_res.status_code == 200
    assert update_res.json()["title"] == "Updated Webhooks Guide"

    # 5. Delete
    del_res = client.delete("/documents/D-TEST-01")
    assert del_res.status_code == 200
    assert engine_instance.chunk_count == initial_chunks


def test_unanswered_endpoint(client):
    client.post("/ask", json={"question": "Can I connect Solstice to Salesforce?"})
    
    unanswered_res = client.get("/unanswered")
    assert unanswered_res.status_code == 200
    unanswered_data = unanswered_res.json()
    assert len(unanswered_data) >= 1
    assert any("Salesforce" in item["question"] for item in unanswered_data)
