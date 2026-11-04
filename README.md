# Solstice Documentation Q&A Assistant

> **HCLTech Hackathon Case Study Submission**  
> **Elena's Core Constraint**: *"The one thing I really care about: if the docs genuinely don't cover something, I'd much rather it say 'I don't have an answer for that' than make something up."*

An enterprise documentation Q&A system built for Solstice Learning support teams. It features **sentence-aware chunking**, **hybrid retrieval (BM25 + Qdrant in-memory vector database with Gemini embeddings)**, **Reciprocal Rank Fusion (RRF)**, **evidence gating**, **grounded Gemini answer generation**, and **deterministic multi-layer verification**.

---

## 1. System Architecture & Flow

```
[ Streamlit Web Portal ] (Port 8501)       [ Evaluation Suite (eval.py) ]
           │                                             │
           └─────────────────────┬───────────────────────┘
                                 │ HTTP / REST
                                 ▼
                     [ FastAPI Server ] (Port 8000)
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [ SQLAlchemy ORM ]               [ Qdrant In-Memory ]
       [ SQLite: documents,       ]     [ + Gemini Embeds  ]
       [ question_log             ]              │
                 │                               │
                 └───────────────┬───────────────┘
                                 ▼
                        [ DocsQA Engine ]
                                 │
                 ┌───────────────┼──────────────────┐
                 ▼               ▼                  ▼
             [ BM25 ]    [ Semantic Search ]    [ Corpus     ]
             [ Search ]  [ (Qdrant)        ]    [ Vocabulary ]
                 │               │                  │
                 └───────┬───────┘                  │
                         ▼                          │
               [ RRF Fusion (top-3) ]               │
                         │                          │
                         ▼                          │
                [ Evidence Gate ] ◄─────────────────┘
            (score floor + unknown-term signal)
                         │ pass                 fail ──► "I don't have an answer for that."
                         ▼
          [ Grounded Gemini Answer (JSON) ]
                         │
                         ▼
        [ Verifier: Chunks, Quotes, Numbers & Units ] ── fail ──► "I don't have an answer for that."
                         │ pass
                         ▼
             [ Verified Answer + Citations ] ──► Logged to question_log
```

---

## 2. Key Capabilities & Architectural Safeguards

1. **Sentence-Aware Chunking (~400 chars)**:
   - Cuts strictly at punctuation boundaries (`.!?`) to keep atomic facts intact (e.g. *"Growth plans are limited to 60 requests per minute"*).
   - Retains the remainder so no documentation content or short articles (`D-02`, `D-05`) are lost.
   - Seed documents produce exactly **10 chunks** with preserved metadata (`doc_id`, `title`, `chunk_id`, `text`).

2. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
   - **Lexical**: Okapi BM25 (`rank-bm25`) captures exact keyword matches (`Growth plan`, `CSV`, `10,000 rows`).
   - **Dense Semantic**: Gemini Embeddings (`gemini-embedding-001`) stored in an in-memory **Qdrant** collection for cosine similarity search.
   - **Rank-Based Fusion**: $RRF(c) = \frac{1}{60 + \text{rank}_{BM25}} + \frac{1}{60 + \text{rank}_{semantic}}$ combines rankings without scale mismatch.
   - **Graceful Degradation**: If Gemini API is unreachable or key is not configured, the engine switches to BM25-only keyword mode and clearly flags the degraded status.

3. **Layer 1: Evidence Gate**:
   - Score floors (`BM25 >= 0.05`, `Cosine >= 0.40`). Questions failing the score floor trigger immediate abstention without calling the LLM.
   - Rule-based detection identifies candidate product/brand/integration names in the query that do not exist in the documentation corpus (e.g., *Salesforce*, *HubSpot*, *PDF*).

4. **Layer 2: Grounded Synthesis (Gemini 2.5 Flash)**:
   - Prompt provides *only* the top-3 retrieved passages.
   - Strict system instruction: answer exclusively from the provided excerpts; if not explicitly stated, return status `not_covered`.
   - Structured JSON schema enforcing status, answer text, cited chunk IDs, and verbatim supporting sentences.

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

## 3. Technology Stack

- **Language**: Python 3.11 / 3.12 / 3.13
- **Web API**: FastAPI, Uvicorn, Pydantic v2
- **Admin UI**: Streamlit
- **Database & ORM**: SQLite, SQLAlchemy 2.x
- **Vector Database**: Qdrant Client (In-Memory mode)
- **Keyword Search**: rank-bm25 (BM25Okapi)
- **LLM & Embeddings**: Google GenAI SDK (`gemini-2.5-flash`, `gemini-embedding-001`)
- **Testing**: Pytest, HTTPX

---

## 4. Repository Structure

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

## 5. Setup & Installation

### Step 1: Clone and Set Up Virtual Environment

```bash
git clone https://github.com/vaishnavikanu/docs_Q-A_Assistant.git
cd docs_Q-A_Assistant

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:

```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_GENERATION_MODEL=gemini-2.5-flash
DATABASE_URL=sqlite:///./solstice_docs.db
CHUNK_SIZE=400
RRF_K=60
TOP_K_CHUNKS=3
BM25_SCORE_FLOOR=0.05
COSINE_SCORE_FLOOR=0.40
```

> **Note**: If `GEMINI_API_KEY` is not provided, the system runs safely in **BM25-only degraded mode** and never crashes or fabricates answers.

---

## 6. Running the Application

### Start the FastAPI Backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- Interactive Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- System Health: [http://localhost:8000/health](http://localhost:8000/health)

### Start the Streamlit Admin Web Portal

In a separate terminal (with virtual environment activated):

```bash
streamlit run frontend/streamlit_app.py
```

- Portal URL: [http://localhost:8501](http://localhost:8501)
- Includes benchmark question quick-fill buttons, source citation display, verbatim quote callouts, retrieval score inspection, and knowledge base document management.

---

## 7. API Endpoints

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

## 8. Evaluation & Verification

### Run Automated Unit & Integration Tests

```bash
pytest tests/
```

Test coverage includes:
- `tests/test_database.py`: SQLite initialization, auto-seeding, document CRUD, and question logging.
- `tests/test_chunking.py`: Sentence boundary checks, remainder preservation, and fact retention across all 6 documents.
- `tests/test_evidence_gate.py`: Score floor enforcement, unknown-term detection (Salesforce, HubSpot, PDF).
- `tests/test_verifier.py`: Cited chunk checks, verbatim quote checks, numerical and unit verification ("15 hours" vs "15 minutes", "60 requests per second" vs "per minute").
- `tests/test_api.py`: FastAPI endpoints, validation, Swagger compatibility, and CRUD re-indexing.

### Run Benchmark Evaluation Script

```bash
# Evaluate against running backend:
python scripts/eval.py --api-url http://127.0.0.1:8000

# Or evaluate directly against local engine:
python scripts/eval.py --local
```

### Official Evaluation Results

| ID | Question | Expected Doc | Expected Outcome | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Q-1** | How do I reset my password? | `D-01` | Password reset steps & requirements | **PASS** |
| **Q-2** | How long do I have to use a password reset link before it expires? | `D-01` | "15 minutes" | **PASS** |
| **Q-3** | What's the maximum number of rows I can export to CSV? | `D-03` | "10,000 rows" | **PASS** |
| **Q-4** | What are the API rate limits for the Growth plan? | `D-06` | "60 requests per minute" | **PASS** |
| **Q-5** | Can I connect Solstice to Salesforce? | *null* | Strict refusal: *"I don't have an answer for that."* | **PASS** |
| **EDGE-1** | When does the password reset link expire? (Paraphrase) | `D-01` | "15 minutes" | **PASS** |
| **EDGE-2** | Can I integrate with HubSpot CRM? (Unknown integration) | *null* | Refusal: *"I don't have an answer for that."* | **PASS** |
| **EDGE-3** | What is the capital of Australia? (Off-topic) | *null* | Refusal: *"I don't have an answer for that."* | **PASS** |

---

## 9. Known Limitations & Roadmap for v2

1. **Authentication & RBAC**: In v1, support admin CRUD operations are unauthenticated. v2 will integrate OAuth2/JWT and role-based permissions (Viewer, Editor, Admin).
2. **Conversational Memory**: v1 focuses on independent factual queries. v2 will support multi-turn query resolution while maintaining strict grounding.
3. **Live Ticketing Integration**: Direct webhook sync with Zendesk and Jira Service Management to automatically file tickets for queries in `/unanswered`.
4. **Persistent Vector Database**: Transition from in-memory Qdrant to a distributed Qdrant cluster by switching `QDRANT_URL`.
