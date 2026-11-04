import logging
import re
from typing import List, Set, Optional, Tuple
from app.config import settings

logger = logging.getLogger(__name__)

# Standard common English words, question terms, prepositions, verbs, and pronouns
COMMON_WORDS = {
    # Question words and pronouns
    "what", "whats", "what's", "how", "why", "when", "where", "who", "whom", "which",
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs", "this", "that", "these", "those",
    # Verbs and auxiliaries
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "connect", "integrate", "link", "export", "import", "download",
    "upload", "reset", "change", "update", "invite", "add", "remove", "delete", "use",
    "set", "setup", "configure", "get", "find", "view", "see", "know", "tell", "show",
    "expire", "expired", "expires", "take", "support", "supported", "supports", "allow",
    # Prepositions, conjunctions, articles
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "don", "t", "s", "re", "ve", "ll", "d", "now", "also", "much", "many",
    # Common nouns and adjectives
    "time", "day", "days", "minute", "minutes", "hour", "hours", "week", "weeks",
    "month", "months", "year", "years", "number", "maximum", "minimum", "limit",
    "limits", "rate", "rates", "row", "rows", "file", "files", "account", "user",
    "users", "team", "member", "members", "admin", "admins", "role", "roles",
    "plan", "plans", "screen", "link", "links", "email", "emails", "password",
    "passwords", "report", "reports", "system", "service", "help", "doc", "docs",
    "documentation", "question", "questions", "answer", "answers"
}


def detect_unknown_terms(question: str, corpus_vocab: Set[str]) -> List[str]:
    """Identify candidate product, brand, integration or technical terms in the question
    that do not appear in the documentation corpus vocabulary.
    
    Heuristic rules:
    1. Capitalized words that do not start the sentence (e.g. 'Salesforce', 'HubSpot').
    2. CamelCase words (e.g. 'SalesForce', 'PayPal', 'iPhone').
    3. Acronyms of 2+ capital letters (e.g. 'PDF', 'CRM', 'ERP').
    4. Non-stopword words not found in common English vocabulary or the documentation.
    """
    if not question:
        return []

    # Clean tokens while keeping casing
    raw_tokens = re.findall(r"\b[A-Za-z0-9_\-\.]+\b", question)
    flagged: List[str] = []

    for idx, token in enumerate(raw_tokens):
        lower_token = token.lower()

        # If it's already in the docs vocabulary, it is known
        if lower_token in corpus_vocab:
            continue

        # If it's a known common English word or stopword, ignore
        if lower_token in COMMON_WORDS:
            continue

        # Rule 1: Acronym (all caps, length >= 2)
        is_acronym = token.isupper() and len(token) >= 2

        # Rule 2: Capitalized word not at the start of the query
        is_internal_cap = idx > 0 and token[0].isupper()

        # Rule 3: CamelCase / mixed case (e.g. 'iPhone', 'Salesforce')
        has_camel_case = bool(re.search(r"[a-z][A-Z]", token))

        # Rule 4: Lowercase question with unfamiliar noun/term
        is_unfamiliar_term = len(token) >= 3 and lower_token not in COMMON_WORDS and lower_token not in corpus_vocab

        if is_acronym or is_internal_cap or has_camel_case or is_unfamiliar_term:
            if token not in flagged and lower_token not in [f.lower() for f in flagged]:
                flagged.append(token)

    return flagged


class EvidenceGateResult:
    def __init__(
        self,
        passed: bool,
        reason: str,
        flagged_terms: List[str],
        bm25_score: float,
        cosine_score: Optional[float],
        retrieval_mode: str,
    ):
        self.passed = passed
        self.reason = reason
        self.flagged_terms = flagged_terms
        self.bm25_score = bm25_score
        self.cosine_score = cosine_score
        self.retrieval_mode = retrieval_mode


def evaluate_evidence_gate(
    question: str,
    max_bm25_score: float,
    max_cosine_score: Optional[float],
    retrieval_mode: str,
    corpus_vocab: Set[str],
    has_llm: bool = True,
) -> EvidenceGateResult:
    """Evaluate whether the retrieved evidence meets the gate requirements for answerability.
    
    Checks:
    1. Score floors:
       - Hybrid mode: at least one score must exceed its floor (BM25 >= BM25_SCORE_FLOOR or Cosine >= COSINE_SCORE_FLOOR).
       - BM25-only mode: BM25 score must meet BM25_SCORE_FLOOR.
    2. Unknown terms:
       - Detected and recorded.
       - In degraded mode without LLM, any flagged unknown term triggers immediate abstention.
    """
    flagged_terms = detect_unknown_terms(question, corpus_vocab)

    # 1. Score Floor Check
    bm25_floor = settings.BM25_SCORE_FLOOR
    cosine_floor = settings.COSINE_SCORE_FLOOR

    if retrieval_mode == "hybrid" and max_cosine_score is not None:
        bm25_pass = max_bm25_score >= bm25_floor
        cosine_pass = max_cosine_score >= cosine_floor
        if not bm25_pass and not cosine_pass:
            reason = (
                f"Scores below floor threshold. "
                f"BM25: {max_bm25_score:.3f} < {bm25_floor}, Cosine: {max_cosine_score:.3f} < {cosine_floor}"
            )
            return EvidenceGateResult(
                passed=False,
                reason=reason,
                flagged_terms=flagged_terms,
                bm25_score=max_bm25_score,
                cosine_score=max_cosine_score,
                retrieval_mode=retrieval_mode,
            )
    else:
        # BM25-only mode
        if max_bm25_score < bm25_floor:
            reason = f"BM25 score below floor threshold ({max_bm25_score:.3f} < {bm25_floor})"
            return EvidenceGateResult(
                passed=False,
                reason=reason,
                flagged_terms=flagged_terms,
                bm25_score=max_bm25_score,
                cosine_score=max_cosine_score,
                retrieval_mode=retrieval_mode,
            )

    # 2. Degraded mode without LLM check
    if not has_llm and flagged_terms:
        reason = f"BM25-only mode without LLM cannot verify unfamiliar terms: {flagged_terms}"
        return EvidenceGateResult(
            passed=False,
            reason=reason,
            flagged_terms=flagged_terms,
            bm25_score=max_bm25_score,
            cosine_score=max_cosine_score,
            retrieval_mode=retrieval_mode,
        )

    return EvidenceGateResult(
        passed=True,
        reason="Evidence gate passed",
        flagged_terms=flagged_terms,
        bm25_score=max_bm25_score,
        cosine_score=max_cosine_score,
        retrieval_mode=retrieval_mode,
    )
