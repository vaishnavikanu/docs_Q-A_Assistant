import os
import requests
import streamlit as st
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Solstice Docs Q&A Assistant",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Custom CSS for clean enterprise styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-badge-hybrid {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-degraded {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .answer-box {
        background-color: #F8FAFC;
        border-left: 5px solid #2563EB;
        padding: 18px 22px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .quote-box {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 14px 18px;
        border-radius: 6px;
        font-style: italic;
        margin-top: 10px;
    }
    .abstain-box {
        background-color: #FFF7ED;
        border-left: 5px solid #EA580C;
        padding: 18px 22px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


def fetch_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def fetch_documents():
    try:
        resp = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def fetch_unanswered():
    try:
        resp = requests.get(f"{API_BASE_URL}/unanswered", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def send_question(question_text):
    try:
        resp = requests.post(f"{API_BASE_URL}/ask", json={"question": question_text}, timeout=20)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return None, f"Failed to connect to API backend at {API_BASE_URL}: {str(e)}"


# Sidebar: System Status & Quick Examples
with st.sidebar:
    st.image("https://raw.githubusercontent.com/feathericons/feather/master/icons/sun.svg", width=40)
    st.title("Solstice Admin")
    st.markdown("Documentation Support Portal")
    
    st.divider()
    st.subheader("System Status")
    health = fetch_health()
    if health:
        st.success("🟢 API Connected")
        mode = health.get("retrieval_mode", "unknown")
        if mode == "hybrid":
            st.markdown('Retrieval: <span class="status-badge-hybrid">Hybrid (Gemini + BM25)</span>', unsafe_allow_html=True)
        else:
            st.markdown('Retrieval: <span class="status-badge-degraded">BM25-only (Degraded)</span>', unsafe_allow_html=True)
            st.caption("ℹ️ Configure GEMINI_API_KEY in .env to activate hybrid dense embeddings.")
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Documents", health.get("document_count", 0))
        with col_s2:
            st.metric("Chunks", health.get("chunk_count", 0))
    else:
        st.error("🔴 API Offline")
        st.caption(f"Cannot reach FastAPI backend at `{API_BASE_URL}`. Ensure backend is running (`uvicorn app.main:app --port 8000`).")

    st.divider()
    st.subheader("Benchmark Questions")
    benchmark_questions = [
        "How do I reset my password?",
        "How long do I have to use a password reset link before it expires?",
        "What's the maximum number of rows I can export to CSV?",
        "What are the API rate limits for the Growth plan?",
        "Can I connect Solstice to Salesforce?"
    ]

    for bq in benchmark_questions:
        if st.button(bq, key=f"btn_{bq}", use_container_width=True):
            st.session_state["query_input"] = bq


# Header
st.markdown('<div class="main-header">☀️ Solstice Documentation Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Internal admin support assistant grounded strictly in verified Solstice knowledge base articles.</div>', unsafe_allow_html=True)

# Tabs
tab_qa, tab_docs, tab_logs = st.tabs(["💬 Ask Documentation", "📚 Document Management", "📋 Unanswered Queries & History"])

# Tab 1: Q&A
with tab_qa:
    default_q = st.session_state.get("query_input", "")
    
    query = st.text_input(
        "Enter customer admin question:",
        value=default_q,
        placeholder="e.g. How long do I have to use a password reset link before it expires?",
        key="main_query_input"
    )

    col_btn, col_clear = st.columns([1, 6])
    with col_btn:
        ask_submitted = st.button("Submit Question", type="primary", use_container_width=True)

    if ask_submitted and query:
        with st.spinner("Retrieving passages and verifying answer..."):
            ans_data, err = send_question(query)

        if err:
            st.error(err)
        elif ans_data:
            status = ans_data.get("status")
            answer = ans_data.get("answer")
            source_id = ans_data.get("source_doc_id")
            source_title = ans_data.get("source_title")
            quote = ans_data.get("supporting_quote")
            flagged = ans_data.get("flagged_terms", [])
            latency = ans_data.get("latency_ms")

            # Answer Card
            if answer == "I don't have an answer for that." or status in ["not_covered", "needs_clarification"]:
                st.markdown(f"""
                <div class="abstain-box">
                    <h4 style="color: #9A3412; margin-top: 0;">⚠️ Elena's Abstention Policy Triggered</h4>
                    <p style="font-size: 1.15rem; font-weight: 600; color: #1F2937;">"{answer}"</p>
                    <p style="margin-bottom: 0; color: #6B7280; font-size: 0.9rem;">
                        <strong>Abstention Reason:</strong> {ans_data.get('abstention_reason') or 'Topic is not covered in the verified documentation.'}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                if flagged:
                    st.info(f"🚩 Unfamiliar candidate terms detected and blocked: **{', '.join(flagged)}**")
            else:
                st.markdown(f"""
                <div class="answer-box">
                    <h4 style="color: #1E3A8A; margin-top: 0;">Verified Answer</h4>
                    <p style="font-size: 1.15rem; color: #111827; font-weight: 500;">{answer}</p>
                    <p style="margin-bottom: 0; color: #4B5563; font-size: 0.9rem;">
                        <strong>Source Document:</strong> {source_id} — <em>{source_title}</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                if quote:
                    st.markdown(f"""
                    <div class="quote-box">
                        <strong>Verbatim Documentation Quote:</strong><br/>
                        "{quote}"
                    </div>
                    """, unsafe_allow_html=True)

            # Debug / Details Expander
            with st.expander("🔍 Inspection & Decision Details (RRF Scores, Gate, Verifier)", expanded=False):
                col_d1, col_d2, col_d3 = st.columns(3)
                with col_d1:
                    st.markdown(f"**Status:** `{status}`")
                    st.markdown(f"**Retrieval Mode:** `{ans_data.get('retrieval_mode')}`")
                with col_d2:
                    st.markdown(f"**Latency:** `{latency} ms`")
                    st.markdown(f"**Flagged Unknown Terms:** `{flagged}`")
                with col_d3:
                    decision = ans_data.get("decision_details") or {}
                    gate_passed = decision.get("passed_evidence_gate")
                    st.markdown(f"**Evidence Gate:** `{'Passed' if gate_passed else 'Failed'}`")
                    st.markdown(f"**Verification:** `{'Passed' if decision.get('verification_passed') else 'Abstained / Failed'}`")

                st.markdown("#### Retrieved Top-3 Chunks (Reciprocal Rank Fusion)")
                chunks = ans_data.get("retrieved_chunks", [])
                if chunks:
                    for c in chunks:
                        st.markdown(f"**Rank {c.get('rank')} — {c.get('chunk_id')} | {c.get('doc_id')}: {c.get('title')}**")
                        st.caption(f"BM25 Score: {c.get('bm25_score')} | Cosine Score: {c.get('cosine_score')} | RRF Score: {c.get('rrf_score')}")
                        st.markdown(f"> {c.get('text')}")
                        st.divider()

# Tab 2: Document Management
with tab_docs:
    st.subheader("Knowledge Base Documentation Articles")
    st.caption("All articles stored in SQLite and indexed into BM25 and Qdrant. Changes dynamically refresh search indexes.")
    
    docs = fetch_documents()
    if docs:
        for d in docs:
            with st.expander(f"📄 {d['doc_id']} — {d['title']}"):
                st.write(d["content"])
                st.caption(f"Last updated: {d.get('updated_at')}")

    st.divider()
    st.subheader("Add New Documentation Article")
    with st.form("add_doc_form", clear_on_submit=True):
        col_id, col_title = st.columns([1, 3])
        with col_id:
            new_id = st.text_input("Document ID", placeholder="e.g. D-07")
        with col_title:
            new_title = st.text_input("Title", placeholder="e.g. Webhook Notifications")
        new_content = st.text_area("Content", placeholder="Enter the full documentation article content...", height=150)
        submitted_doc = st.form_submit_button("Add Article & Re-Index")

        if submitted_doc:
            if not new_id or not new_title or not new_content:
                st.warning("Please fill out all fields.")
            else:
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/documents",
                        json={"doc_id": new_id, "title": new_title, "content": new_content},
                        timeout=10
                    )
                    if res.status_code == 201:
                        st.success(f"Article {new_id} added and search indexes refreshed!")
                        st.rerun()
                    else:
                        st.error(f"Failed to add document: {res.text}")
                except Exception as ex:
                    st.error(f"Error connecting to backend: {ex}")

# Tab 3: Unanswered Queries & History
with tab_logs:
    st.subheader("Unanswered Customer Admin Queries")
    st.caption("Elena's Support desk queue: queries where the assistant safely refused to hallucinate.")
    
    unanswered_list = fetch_unanswered()
    if unanswered_list:
        table_data = []
        for item in unanswered_list:
            table_data.append({
                "Date": item.get("created_at")[:19],
                "Question": item.get("question"),
                "Status": item.get("status"),
                "Flagged Terms": item.get("flagged_terms") or "[]",
                "Reason": item.get("abstention_reason") or "Not in documentation",
                "Latency (ms)": item.get("latency_ms"),
            })
        st.dataframe(table_data, use_container_width=True)
    else:
        st.info("No unanswered questions logged yet.")
