# Solution Design: Solstice Documentation Q&A Assistant

## 1. Problem Restatement & Elena's Core Constraint

Solstice Learning sells an online-course platform to corporate training teams. Elena Torres, Head of Support, faces a recurring issue: new customer admins repeatedly ask the support desk basic procedural questions (password resets, CSV export limits, billing cycles, API rate limits). These answers already exist in the documentation, but admins want to ask a question in plain English and get an immediate, direct answer without hunting across multiple help articles.

Elena has one non-negotiable rule:
> *"The one thing I really care about: if the docs genuinely don't cover something, I'd much rather it say 'I don't have an answer for that' than make something up."*

In enterprise support, hallucinations are unacceptable. Telling an admin they can connect to Salesforce when no such integration exists creates broken customer workflows and escalations. A confident refusal protects trust and routes unhandled queries to human support.

The goal is to build a reliable, maintainable Q&A system that accurately retrieves facts from the 6 documentation articles in `starter_kit/data/docs.json`, returns a clear answer with its source, and strictly abstains when the documentation does not contain the answer.

---

## 2. Scope & Boundaries for v1

### In Scope for v1
- **Document Ingestion & Storage (SQLite via SQLAlchemy)**:
  - The 6 articles from `starter_kit/data/docs.json` are auto-seeded into an SQLite database on first boot.
  - Articles can be added, edited or removed through CRUD endpoints (`/documents`). Every change re-indexes BM25 and Qdrant, so answers always reflect the current docs.
  - A second table, `question_log`, records every question with its status, abstention reason, flagged terms and latency. `GET /unanswered` lists the questions the docs couldn't answer.
  - On startup, the search indexes are rebuilt from the current SQLite documents; after a document is added, edited or deleted, both BM25 and Qdrant indexes are refreshed to prevent stale results.
  - Document CRUD operations are handled by FastAPI using SQLAlchemy. After a successful add, edit or delete operation, the DocsQA Engine reloads the current documents and rebuilds both BM25 and Qdrant indexes. If embedding generation fails, the system switches to BM25-only mode.
- **Sentence-Aware Chunking (starter kit approach, fixed)**: Articles are split into ~400-character chunks, the same `CHUNK_SIZE` as the starter kit's `docs_qa_utils.py`. Two fixes were needed:
  - **Keep the remainder.** The starter `chunk_text()` silently discards it, losing the answers to Q-2, Q-3 and Q-4 and dropping D-02 and D-05 entirely.
  - **Cut only between sentences.** This keeps facts like "Growth plans are limited to 60 requests per minute" in one chunk.

  The result is 10 chunks, each carrying its doc ID, title and chunk ID.
- **Hybrid Retrieval (Gemini Embeddings + BM25)**:
  - Dense semantic search using Gemini `gemini-embedding-001`, stored in **Qdrant in-memory mode**, handles synonym matches and conversational phrasing (e.g. "credentials" matching "password").
  - Lexical BM25 ranking preserves exact keyword matches for plan names ("Growth plan"), formats ("CSV") and numbers ("10,000 rows").
  - **Reciprocal Rank Fusion (RRF)** merges both rankings; the top-3 chunks move forward.
  - Graceful degradation: if the embedding API is unavailable, the system runs BM25-only and clearly reports "keyword-only mode" in `/health` and the UI.
- **Evidence Gate (Layer 1 of abstention)**:
  - **Score floor**: if both BM25 and semantic scores are weak, the question is treated as off-topic and the system returns `"I don't have an answer for that."` without calling the LLM.
  - When semantic retrieval is unavailable, the evidence gate uses the BM25 score threshold alone.
  - **Unknown-term signal**: product- or brand-like terms in the question that appear nowhere in the documentation (e.g. "Salesforce", "HubSpot", "PDF") are flagged. They are not rejected on the spot; the signal is carried forward and enforced in Layers 2 and 3.
  - A lightweight rule-based detector identifies candidate product, brand and integration names in the question and compares them against the documentation vocabulary. Matching terms are ignored; unfamiliar candidates are flagged for downstream verification. This heuristic is intentionally limited and may occasionally flag harmless words.
- **Grounded Answer Generation (Layer 2)**:
  - Gemini (`gemini-3.5-flash-lite` / `gemini-3.8-flash`, temperature 0, JSON schema output) receives only the retrieved passages.
  - It is told explicitly which flagged terms the docs never mention, and must return `not_covered` if the answer depends on them.
  - It returns a status (`answered` / `not_covered` / `needs_clarification`), the answer, chunk IDs and verbatim supporting quotes.
- **Deterministic Verification (Layer 3)**: Code checks that:
  - every cited chunk was retrieved;
  - every quote appears verbatim in its chunk;
  - every numerical value and its unit in the answer is supported by the cited text. "15 hours" fails when the doc says "15 minutes", and "60 requests per second" fails when it says "per minute";
  - the verifier rejects any answer that mentions a flagged unknown term.

  Any failure returns `"I don't have an answer for that."`.
- **Client Interfaces & Production Hosting**:
  - FastAPI backend exposing `POST /ask`, `GET /health`, `GET /unanswered` and `/documents` CRUD, with interactive Swagger docs (`/docs`).
  - Streamlit web portal for admin Q&A, showing answers, quoted sources, example questions and a retrieval-details panel.
  - Production deployment on AWS EC2 (`t3.small`, Ubuntu 24.04 LTS) in Sydney (`ap-southeast-2`) using Nginx on port 80 and isolated systemd background daemons.
  - Evaluation script (`scripts/eval.py`) that runs the benchmark questions against the API.

### Explicitly Excluded from v1
- User authentication and role-based permissions (RBAC).
- Multi-turn conversational chat memory (admins ask independent factual questions).
- Live ticketing sync (Zendesk / Jira ticket generation).
- Distributed vector databases and multiple external LLM providers.
- Complex Kubernetes/container orchestration (v1 uses native Linux systemd services and Nginx).

### Assumptions
- The 6 articles in `docs.json` are the initial knowledge base, seeded into SQLite on first boot. The 5 questions in `example_questions.json` are the primary evaluation set.
- Questions are in English, single-turn, and may need more than one article.
- A Gemini API key is available. If it isn't, the system degrades visibly and never fabricates.
- An answer counts as supported only when the documentation explicitly states it.
- Only Support staff use the document CRUD endpoints. They are unauthenticated in v1 because RBAC is out of scope.

---

## 3. System Architecture

### Internal Component Pipeline

```
[ Streamlit Web Portal ] (8501)       [ Eval Script (eval.py) ]
          │                                     │
          └───────────────────┬─────────────────┘
                              │ HTTP / REST (POST /ask, /documents, /unanswered)
                              ▼
                   [ FastAPI Server ] (8000)
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
        [ SQLAlchemy ORM ]          [ Qdrant In-Memory ]
        [ SQLite: documents,  ]     [ + Gemini Embeds  ]
        [ question_log        ]              │
               │                             │
               └──────────────┬──────────────┘
                              ▼
                      [ DocsQA Engine ]
                              │
               ┌──────────────┼──────────────────┐
               ▼              ▼                  ▼
           [ BM25 ]   [ Semantic Search ]   [ Corpus     ]
           [ Search ] [ (Qdrant)        ]   [ Vocabulary ]
               │              │                  │
               └──────┬───────┘                  │
                      ▼                          │
             [ RRF Fusion (top-3) ]              │
                      │                          │
                      ▼                          │
              [ Evidence Gate ] ◄────────────────┘
          (score floor + unknown-term signal)
                      │ pass                fail ──► "I don't have an answer for that."
                      ▼
       [ Grounded Gemini Answer (JSON) ]
                      │
                      ▼
       [ Citation, Quote, Number & Signal Verifier ] ── fail ──► "I don't have an answer for that."
                      │ pass
                      ▼
          [ Answer + Source Citations ] ──► logged to question_log
```

### Retrieval & Decision Flow
1. **Startup**:
   - If the SQLite database is empty, it is seeded from `docs.json`.
   - Articles are loaded from SQLite, chunked, and indexed into BM25 and Qdrant.
   - Any later CRUD change triggers a re-index.
2. **Query Input**: Admin enters a question in Streamlit, which sends `POST /ask`. FastAPI validates it (non-empty, ≤ 500 characters).
3. **Hybrid Scoring**:
   - Semantic score: cosine similarity between the question embedding and chunk embeddings in Qdrant (via the Gemini API).
   - Lexical score: Okapi BM25 over chunk tokens.
   - Fusion: Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank)`. Ranks are combined instead of raw scores, because BM25 and cosine are on different scales.
4. **Evidence Gate**:
   - If both scores fall below their floors, return `"I don't have an answer for that."`.
   - Unknown brand or product terms are flagged as a signal for the next steps.
5. **Grounded Synthesis**: The top-3 passages and any flagged terms are sent to Gemini with strict rules: answer only from the passages, cite chunk IDs, copy the supporting sentence verbatim, and return `not_covered` if the answer isn't explicitly stated or depends on a flagged term.
6. **Verification**: Code confirms the citations, quotes, numbers and units, and rejects any answer that mentions a flagged unknown term. Any failure, a `not_covered` status, or invalid JSON after one retry returns `"I don't have an answer for that."`.
   - If Gemini answer generation is unavailable, the API returns status degraded and the exact fallback answer "I don't have an answer for that.". Retrieved passages may be shown separately as diagnostic information, but they must not be presented as a generated answer.
   - With no LLM available, a flagged unknown term means abstain, because nothing else can judge whether the docs cover it.
7. **Response Delivery**: Returns status, answer, source document ID, title and the verified quote. The question and its outcome are written to `question_log`. A debug view shows retrieved chunks, their BM25/cosine/RRF scores, the gate result, flagged terms and the decision reason.

---

## 4. Production AWS EC2 Deployment Architecture

The application is deployed on an **AWS EC2 `t3.small`** instance running **Ubuntu Server 24.04 LTS** in the **Sydney (`ap-southeast-2`)** region:

```
                      Public Internet / User Browser
                                    │
                                    ▼ HTTP (Port 80)
                       [ Nginx Reverse Proxy ]
                       (Listens on 0.0.0.0:80)
                                    │
                      proxy_pass    ▼ (127.0.0.1:8501 - Loopback)
                 [ Streamlit Frontend Portal ]
                 (Managed by solstice-frontend.service)
                                    │
                         Internal Python requests
                       (API_URL=http://127.0.0.1:8000)
                                    │
                                    ▼ (127.0.0.1:8000 - Loopback)
                     [ FastAPI Backend (Uvicorn) ]
                     (Managed by solstice-backend.service)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
[ SQLite Database ]        [ Qdrant In-Memory ]        [ Google Gemini API ]
(solstice_docs.db)         (Dense Vector Index)        - gemini-embedding-001
- documents table          - 3072-dim embeddings       - gemini-3.5-flash-lite
- question_log table       - Cosine similarity         - Grounded JSON schema
```

### Production Access & Networking
- **Public URL**: `http://3.26.216.51` (Port 80 HTTP via Nginx)
- **Port Security**:
  - Only **Port 80** (HTTP) and **Port 22** (SSH) are open in the AWS Security Group.
  - Ports **8000** (FastAPI) and **8501** (Streamlit) are bound strictly to `127.0.0.1` (loopback only) and are not directly reachable from the public internet.
- **Reverse Proxy**: Nginx proxies incoming Port 80 traffic to Streamlit (`127.0.0.1:8501`) with WebSocket upgrade support (`Upgrade $http_upgrade`, `Connection "upgrade"`).
- **Process Management**:
  - `solstice-backend.service`: Runs `uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1` under systemd with automatic restart on failure.
  - `solstice-frontend.service`: Runs `streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true` under systemd with automatic restart on failure.
- **Service Management Commands**:
  - `sudo systemctl status solstice-backend solstice-frontend` (check health)
  - `sudo systemctl restart solstice-backend solstice-frontend` (restart both tiers)
  - `sudo systemctl stop solstice-backend solstice-frontend` (stop services)
  - `sudo systemctl start solstice-backend solstice-frontend` (start services)
  - `sudo journalctl -u solstice-backend -f` (view live backend logs)
- **Instance State & Cost Note**: The EC2 instance may be stopped when not in active use to minimize cloud hosting costs. When stopped, it must be restarted from the AWS Console before accessing the web interface.

---

## 5. Key Trade-offs & Decisions

| Decision | What We Chose | What We Skipped | Why |
| :--- | :--- | :--- | :--- |
| **Document Storage** | SQLite via SQLAlchemy, auto-seeded from `docs.json` | Read-only JSON file; PostgreSQL | SQLite is a single file with zero setup, yet supports editing docs and logging questions. SQLAlchemy means moving to PostgreSQL / AWS RDS later is a `DATABASE_URL` change. |
| **Chunking** | ~400-character sentence-aware chunks, remainder kept | Starter `chunk_text()` unchanged | Used unchanged, it discards the answers to Q-2 ("15 minutes"), Q-3 ("10,000 rows") and Q-4 ("60 requests per minute"), drops D-02 and D-05, and splits "Growth plans are / limited to 60" across chunks. |
| **Semantic Embeddings** | Cloud API (Gemini `gemini-embedding-001`) | Local PyTorch models (`sentence-transformers`) | Local embedding models require a ~1.5 GB PyTorch install and substantial memory. The Gemini API uses the same SDK and key as the LLM and keeps the setup light. |
| **Vector Storage** | Qdrant in-memory mode (`qdrant-client`) | Flat NumPy arrays; Qdrant server | Runs inside the backend with zero daemon setup. At 6 articles it returns the same results as NumPy, but moving to a Qdrant server later is a connection-string change (`QDRANT_URL`). |
| **Score Fusion** | Reciprocal Rank Fusion | Weighted blend (`0.6 × Semantic + 0.4 × BM25`) | BM25 is unbounded and cosine is model-specific; blending raw scores makes thresholds unstable. Ranks are always comparable. |
| **Unknown Terms** | Signal, enforced by the LLM prompt and the verifier | Hard rejection on any unfamiliar term | A hard rejection over-abstains on harmless extras ("reset my password on my iPhone"). Flagging unfamiliar terms instead of immediately rejecting the question avoids unnecessary abstentions. However, to maintain safety, the verifier rejects any generated answer that mentions a flagged unknown term. This is a conservative approach that may occasionally reject otherwise valid answers. |
| **Abstention Logic** | Score floor + LLM answerability + code verification | Score threshold alone; prompt-only guardrail | Similarity scores are high even for on-topic questions the docs don't answer ("Salesforce" resembles the API doc), and prompts alone can't be enforced. Code checks can. |
| **Answer Format** | Grounded LLM answer with verified quotes | Pure sentence extraction; unverified LLM generation | Extraction can return the wrong sentence and breaks multi-step answers (Q-1). Unverified generation can invent details. Verified quotes make every answer traceable. |
| **Client UI & Hosting** | Streamlit + FastAPI on AWS EC2 (`t3.small`) via Nginx | Single-page static HTML; Kubernetes | Lightweight pure-Python interface with Swagger documentation; Nginx on port 80 provides a single secure public entry point with loopback process isolation. |

---

## 6. Evaluation & Verification

The implementation is verified against the 5 questions in `starter_kit/data/example_questions.json`:

1. `Q-1`: *"How do I reset my password?"*
   - Target: `D-01` ("Resetting Your Password")
   - Expected: Click 'Forgot password?' on the login screen, enter the account email, use the reset link, choose a new password (at least 8 characters, with at least one number).
2. `Q-2`: *"How long do I have to use a password reset link before it expires?"*
   - Target: `D-01`
   - Expected: "15 minutes".
3. `Q-3`: *"What's the maximum number of rows I can export to CSV?"*
   - Target: `D-03` ("Exporting Reports to CSV")
   - Expected: "10,000 rows".
4. `Q-4`: *"What are the API rate limits for the Growth plan?"*
   - Target: `D-06` ("API Rate Limits")
   - Expected: "60 requests per minute".
5. `Q-5`: *"Can I connect Solstice to Salesforce?"*
   - Target: `null` (Unsupported)
   - Expected: Rejection with exact text `"I don't have an answer for that."`. "Salesforce" is flagged by the unknown-term signal, Gemini returns `not_covered`, and the verifier blocks any answer that mentions it.
