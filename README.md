# Solstice Documentation Q&A Assistant

> **HCLTech Hackathon Case Study Submission**  
> **Elena's Core Constraint**: *"The one thing I really care about: if the docs genuinely don't cover something, I'd much rather it say 'I don't have an answer for that' than make something up."*

An enterprise documentation Q&A system built for Solstice Learning support teams. It features **sentence-aware chunking**, **hybrid retrieval (BM25 + Qdrant in-memory vector database with Gemini embeddings)**, **Reciprocal Rank Fusion (RRF)**, **evidence gating**, **grounded Gemini answer generation**, and **deterministic multi-layer verification**.

---

## 1. Public Deployment & Access

The application is deployed on **AWS EC2** in the **Sydney (`ap-southeast-2`)** region and is publicly accessible:

* **Live Application URL**: [http://3.26.216.51](http://3.26.216.51)
* **Access Protocol**: HTTP on Port 80 (served via Nginx reverse proxy)
* **Instance Type**: AWS EC2 `t3.small` (2 vCPUs, 2 GB RAM, 8 GB gp3 root volume + 1.5 GB swapfile)
* **Operating System**: Ubuntu Server 24.04 LTS

> [!IMPORTANT]
> **EC2 Instance State Note**: To optimize cloud hosting costs, the EC2 instance may be stopped when not in active use or during demonstration evaluation intervals. If the live link does not immediately respond, the instance needs to be started from the AWS EC2 Console before accessing the web portal.

---

## 2. System & Deployment Architecture

### End-to-End Production Deployment Flow

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

### Network Security & Port Isolation

| Port | Service | Scope | Security Policy |
| :---: | :---: | :---: | :--- |
| **80** | Nginx Reverse Proxy | **Public** | Inbound allowed (`0.0.0.0/0`) in AWS Security Group. Proxies to Streamlit. |
| **22** | OpenSSH | **Administrative** | Inbound allowed for remote server administration via SSH. |
| **8501** | Streamlit Frontend | **Internal Only** | Bound strictly to `127.0.0.1:8501`. Blocked from external internet access. |
| **8000** | FastAPI Backend | **Internal Only** | Bound strictly to `127.0.0.1:8000`. Blocked from external internet access. |

---

## 3. Background Process Management (systemd)

On the EC2 instance, both application tiers run as isolated background daemon services managed by Ubuntu's `systemd` init system. This ensures automatic restarts upon failure, proper environment variable injection, and automated boot-up with the server.

### Systemd Service Configurations

1. **FastAPI Backend Service** (`/etc/systemd/system/solstice-backend.service`):
   ```ini
   [Unit]
   Description=Solstice Docs QA - FastAPI Backend (Internal)
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/docs_Q-A_Assistant
   EnvironmentFile=/home/ubuntu/docs_Q-A_Assistant/.env
   ExecStart=/home/ubuntu/docs_Q-A_Assistant/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

2. **Streamlit Frontend Service** (`/etc/systemd/system/solstice-frontend.service`):
   ```ini
   [Unit]
   Description=Solstice Docs QA - Streamlit Frontend (Internal)
   After=network.target solstice-backend.service

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/docs_Q-A_Assistant
   EnvironmentFile=/home/ubuntu/docs_Q-A_Assistant/.env
   Environment="API_URL=http://127.0.0.1:8000"
   ExecStart=/home/ubuntu/docs_Q-A_Assistant/venv/bin/streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   ```

### Service Management Commands

```bash
# Check service health and status
sudo systemctl status solstice-backend
sudo systemctl status solstice-frontend

# Restart services
sudo systemctl restart solstice-backend solstice-frontend

# Stop services
sudo systemctl stop solstice-backend solstice-frontend

# Start services
sudo systemctl start solstice-backend solstice-frontend

# View live real-time service logs
sudo journalctl -u solstice-backend -f
sudo journalctl -u solstice-frontend -f
```

---

## 4. Key Architectural Safeguards & Features

1. **Sentence-Aware Chunking (~400 chars)**:
   - Cuts strictly at sentence boundaries (`.!?`) to keep atomic facts intact (e.g. *"Growth plans are limited to 60 requests per minute"*).
   - Retains the remainder so no documentation content or short articles (`D-02`, `D-05`) are lost.
   - Seed documents produce exactly **10 chunks** with preserved metadata (`doc_id`, `title`, `chunk_id`, `text`).

2. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
   - **Lexical**: Okapi BM25 (`rank-bm25`) captures exact keyword matches (`Growth plan`, `CSV`, `10,000 rows`).
   - **Dense Semantic**: Gemini Embeddings (`gemini-embedding-001`, 3072 dimensions) stored in an in-memory **Qdrant** collection for cosine similarity search.
   - **Rank-Based Fusion**: $RRF(c) = \frac{1}{60 + \text{rank}_{BM25}} + \frac{1}{60 + \text{rank}_{semantic}}$ merges rankings without scale mismatch.
   - **Graceful Degradation**: If the Gemini API is unreachable or the key is omitted, the engine switches to BM25-only keyword mode and clearly reports the degraded status.

3. **Layer 1: Evidence Gate**:
   - Score floors (`BM25 >= 0.05`, `Cosine >= 0.40`). Questions failing the score floor trigger immediate abstention without calling the LLM.
   - Rule-based detection identifies candidate product/brand/integration names in the query that do not exist in the documentation corpus (e.g. *Salesforce*, *HubSpot*, *PDF*).

4. **Layer 2: Grounded Synthesis (Gemini API)**:
   - Prompt provides *only* the top-3 retrieved passages.
   - Strict system instruction: answer exclusively from the provided excerpts; if not explicitly stated, return status `not_covered`.
   - Structured JSON schema enforcing status, answer text, cited chunk IDs, and verbatim supporting sentences.
   - Resilient model fallback: prioritizes active Gemini Flash models (`gemini-3.5-flash-lite`, `gemini-3.8-flash`) to handle Google API model lifecycle transitions.

5. **Layer 3: Python Deterministic Verification**:
   - Rejects answers citing chunks that were not retrieved.
   - Confirms all supporting quotes exist verbatim in the cited documentation text.
   - **Numerical & Unit Checking**: Rejects answers where numbers or units are not in the cited text (e.g. *"15 hours"* fails when doc says *"15 minutes"*; *"60 requests per second"* fails when doc says *"60 requests per minute"*).
   - Blocks any generated answer that mentions a flagged unknown term.
   - Returns the exact required fallback: `"I don't have an answer for that."` on any verification failure.

6. **SQLite Storage & Document CRUD**:
   - Auto-seeds the 6 articles from `starter_kit/data/docs.json` into SQLite on first boot.
   - Adding, editing, or deleting documents via `/documents` endpoints immediately refreshes both BM25 and Qdrant search indexes.
   - `question_log` table logs every query, outcome, status, latency, and abstention reason.

---

## 5. Technology Stack

- **Cloud Infrastructure**: AWS EC2 (`t3.small`), Ubuntu 24.04 LTS, Nginx Reverse Proxy
- **Language & Runtime**: Python 3.12 / 3.13
- **Web API**: FastAPI, Uvicorn, Pydantic v2
- **Admin UI**: Streamlit
- **Database & ORM**: SQLite, SQLAlchemy 2.x
- **Vector Database**: Qdrant Client (In-Memory mode)
- **Keyword Search**: rank-bm25 (BM25Okapi)
- **LLM & Embeddings**: Google GenAI SDK (`gemini-3.5-flash-lite` / `gemini-3.8-flash`, `gemini-embedding-001`)
- **Process Management**: systemd
- **Testing**: Pytest, HTTPX

---

## 6. Repository Structure

```
├── app/
│   ├── __init__.py
│   ├── config.py           # Application settings and environment configuration
│   ├── database.py         # SQLAlchemy engine and session dependency
│   ├── models.py           # SQLite tables (Document, QuestionLog)
│   ├── schemas.py          # Pydantic models for API requests/responses
│   ├── crud.py             # Database CRUD and automatic JSON seeding
│   ├── chunking.py         # Sentence-aware chunker preserving all content
│   ├── retrieval.py        # BM25 + Qdrant in-memory vector hybrid search with RRF
│   ├── evidence_gate.py    # Score floor and rule-based unknown-term detector
│   ├── generator.py        # Gemini grounded answer generator (JSON schema)
│   ├── verifier.py         # Deterministic citation, quote, number & unit verifier
│   ├── engine.py           # DocsQAEngine orchestrating retrieval, gate & verifier
│   └── main.py             # FastAPI app with CORS, lifespan, and Swagger docs
├── frontend/
│   └── streamlit_app.py    # Streamlit admin portal with Q&A, CRUD, and analytics
├── scripts/
│   └── eval.py             # Benchmark evaluation script (official questions + edge cases)
├── tests/
│   ├── test_chunking.py    # Unit tests for sentence-aware chunking
│   ├── test_database.py    # Unit tests for SQLite seeding and document CRUD
│   ├── test_evidence_gate.py # Unit tests for score thresholds and unknown terms
│   ├── test_verifier.py    # Unit tests for deterministic quote and unit verification
│   └── test_api.py         # Integration tests for FastAPI endpoints
├── starter_kit/
│   ├── data/
│   │   ├── docs.json       # Seed documentation articles (D-01 to D-06)
│   │   └── example_questions.json # Official hackathon benchmark questions
│   ├── docs_qa_utils.py    # Original starter kit utility module
│   └── README.md
├── .env.example            # Environment variables template
├── requirements.txt        # Python package dependencies
├── SOLUTION_DESIGN.md      # Solution design document (single source of truth)
└── README.md               # Complete project documentation
```

---

## 7. Local Setup & Execution Instructions

### Step 1: Clone and Set Up Virtual Environment

```bash
git clone https://github.com/vaishnavikanu/docs_Q-A_Assistant.git
cd docs_Q-A_Assistant

python3 -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Populate `.env` with your configuration:

```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite
DATABASE_URL=sqlite:///./solstice_docs.db
CHUNK_SIZE=400
RRF_K=60
TOP_K_CHUNKS=3
BM25_SCORE_FLOOR=0.05
COSINE_SCORE_FLOOR=0.40
API_URL=http://127.0.0.1:8000
```

> [!NOTE]
> `.env` is ignored by `.gitignore`. Secret keys must never be committed to source control.

### Step 4: Run Locally

* **Start Backend**:
  ```bash
  uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
  ```
* **Start Admin Portal**:
  ```bash
  streamlit run frontend/streamlit_app.py
  ```

---

## 8. Evaluation & Verification Results

### Automated Unit Test Suite (`pytest tests/`)
```text
============================= test session starts =============================
platform win32 / linux -- Python 3.12/3.13, pytest-9.2.0, pluggy-1.7.0
collected 25 items

tests/test_api.py ......                                                 [ 24%]
tests/test_chunking.py ....                                              [ 40%]
tests/test_database.py ...                                               [ 52%]
tests/test_evidence_gate.py .....                                        [ 72%]
tests/test_verifier.py .......                                           [100%]

======================== 25 passed in 2.14s ========================
```

### Official Evaluation Suite (`python scripts/eval.py --local`)
```text
================================================================================
Running Solstice Docs Q&A Assistant Evaluation in LOCAL ENGINE mode
================================================================================
Retrieval Mode: HYBRID
Chunks Count: 10
Gemini API Configured: True
--------------------------------------------------------------------------------
[PASS] Q-1: 'How do I reset my password?'
         Result: Doc: D-01 (expected D-01) | Answer: To reset your password, click 'Forgot password?' on the login screen...
[PASS] Q-2: 'How long do I have to use a password reset link before it expires?'
         Result: Doc: D-01 (expected D-01) | Answer: Password reset links expire after 15 minutes.
[PASS] Q-3: 'What's the maximum number of rows I can export to CSV?'
         Result: Doc: D-03 (expected D-03) | Answer: CSV exports are limited to a maximum of 10,000 rows.
[PASS] Q-4: 'What are the API rate limits for the Growth plan?'
         Result: Doc: D-06 (expected D-06) | Answer: For Growth plans, the API rate limit is 60 requests per minute...
[PASS] Q-5: 'Can I connect Solstice to Salesforce?'
         Result: Abstained correctly with: 'I don't have an answer for that.'
[PASS] EDGE-1 (Paraphrased Q-2): 'When does the password reset link expire?'
         Result: Doc: D-01 (expected D-01) | Answer: Password reset links expire after 15 minutes.
[PASS] EDGE-2 (Unknown Integration): 'Can I integrate with HubSpot CRM?'
         Result: Abstained correctly with: 'I don't have an answer for that.'
[PASS] EDGE-3 (Off-topic question): 'What is the capital of Australia?'
         Result: Abstained correctly with: 'I don't have an answer for that.'
================================================================================
Evaluation Summary: 8/8 tests passed (100.0%)
================================================================================
```

---

## 9. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/ask` | Submit a question; returns verified answer, source citations, quotes, and scores |
| `GET` | `/health` | Check engine health, retrieval mode (`hybrid` or `bm25_only`), and chunk counts |
| `GET` | `/unanswered` | List questions the system refused to answer (Elena's support queue) |
| `GET` | `/logs` | Full query history with latency and gate results |
| `GET` | `/documents` | List all documentation articles |
| `POST` | `/documents` | Add a new article and trigger automatic re-indexing |
| `GET` | `/documents/{doc_id}` | Retrieve a specific article |
| `PUT` | `/documents/{doc_id}` | Update an article and refresh indexes |
| `DELETE` | `/documents/{doc_id}` | Delete an article and refresh indexes |

---

## 10. Known Limitations & Roadmap for v2

1. **Authentication & RBAC**: In v1, support admin CRUD operations are unauthenticated. v2 will integrate OAuth2/JWT and role-based permissions (Viewer, Editor, Admin).
2. **Conversational Memory**: v1 focuses on independent factual queries. v2 will support multi-turn query resolution while maintaining strict grounding.
3. **Live Ticketing Integration**: Direct webhook sync with Zendesk and Jira Service Management to automatically file tickets for queries in `/unanswered`.
4. **Persistent Vector Database**: Transition from in-memory Qdrant to a distributed Qdrant cluster by switching `QDRANT_URL`.
