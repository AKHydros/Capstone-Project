from __future__ import annotations

from dataclasses import dataclass
import re


# ---------------------------------------------------------------------------
# Blocked safety topics — absolute refusals regardless of domain
# ---------------------------------------------------------------------------
_BLOCKED_TOPICS = (
    "build a bomb",
    "credit card number",
    "steal password",
    "social insurance number",
    "social security number",
)

# ---------------------------------------------------------------------------
# Domain allow-list — topics that ARE within scope of this assistant.
# A query must contain at least one of these signals to pass the domain check.
# Kept broad enough to cover natural phrasing but tight enough to reject
# clearly off-topic requests (weather, cooking, sports, general knowledge, etc.)
# ---------------------------------------------------------------------------
_DOMAIN_KEYWORDS: tuple[str, ...] = (
    # Survey / research artefacts — clearly in-scope signals
    "survey", "wave", "question", "variable", "label", "value", "record",
    "pmg", "dictionary", "data dictionary", "codebook",
    # Question / item ID references — unambiguous survey context
    "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10",
    "q11", "q12", "q13", "q14", "q15", "q16", "q17", "q18", "q19", "q20",
    "_q", "_gam", "_wai", "_rob",
    # Research methodology language — survey-specific terms
    "measurement", "measurement level", "nominal", "ordinal", "scale",
    "respondent", "segment", "cohort", "panel", "demographic",
    # Financial-research terms — domain specific (present in PMG surveys)
    "trust", "sentiment", "provider", "advisor", "mortgage",
    "investment", "retirement", "income",
    # Data dictionary / lookup terms
    "allowable", "allowed value", "coded", "response option", "category",
    "topic label", "mapping", "lookup", "grounded", "retrieval", "filter",
    "lineage", "wave year", "value label",
    # Analytical intent — kept only when paired with unambiguous survey context
    # NOTE: generic single-word verbs (list, find, show, which, search, explain,
    # summarize, compare, trend, pattern, overview, breakdown, describe, analysis)
    # are intentionally removed.  They matched nearly any English sentence and
    # defeated the purpose of domain scoping.  Legitimate survey queries will
    # still pass via the survey-specific terms above.
)

# Explicit out-of-domain patterns — greet the user but redirect if matched.
# These are checked AFTER domain keywords to avoid false positives on e.g.
# "what is the label for q5a" — the phrase "what is" is in domain keywords.
_OUT_OF_DOMAIN_PATTERNS: tuple[str, ...] = (
    "weather", "recipe", "cook", "sport", "football", "basketball",
    "hockey", "soccer", "music", "movie", "film", "celebrity",
    "stock price", "cryptocurrency", "bitcoin", "nfl", "nba", "mlb",
    "travel", "flight", "hotel", "restaurant",
    "write a poem", "write a story", "tell me a joke",
    "who is the president", "capital of", "history of",
)

# Queries shorter than this character threshold bypass domain checking
# (they are likely greetings / clarifications like "yes", "ok", "thanks").
# Set to 6 so single-word off-topic queries like "weather" (7 chars) are
# still domain-checked, while true greetings ("hi", "ok") pass through.
_DOMAIN_CHECK_MIN_CHARS = 6


@dataclass(frozen=True)
class SafetyResult:
    """Immutable result from a safety or domain check."""

    allowed: bool
    reason: str


class SafetyService:
    """Provides query safety checks, domain scoping, and PII redaction.

    All methods are stateless and deterministic — no LLM calls are made.
    This keeps the safety layer fast and auditable.

    Methods
    -------
    check_user_query(query)
        Blocks absolute safety violations (unsafe phrases).
    check_domain(query)
        Rejects queries that fall outside the market-research dictionary domain.
    check_assistant_response(response_text)
        Ensures assistant output does not contain blocked phrases.
    redact_pii(text)
        Masks email addresses and phone numbers before persistence.
    """

    def check_user_query(self, query: str) -> SafetyResult:
        """Blocks known unsafe phrases in user input.

        Parameters
        ----------
        query:
            Raw user query string.

        Returns
        -------
        SafetyResult
            ``allowed=False`` with ``reason="blocked_topic"`` if a blocked
            phrase is detected; otherwise ``allowed=True``.
        """
        lowered = query.lower()
        for phrase in _BLOCKED_TOPICS:
            if phrase in lowered:
                return SafetyResult(allowed=False, reason="blocked_topic")
        return SafetyResult(allowed=True, reason="ok")

    def check_domain(self, query: str) -> SafetyResult:
        """Rejects queries that fall outside the market-research dictionary domain.

        The check uses two heuristics applied in order:

        1. **Short-query pass-through** — queries under ``_DOMAIN_CHECK_MIN_CHARS``
           characters (greetings, acknowledgements) are always allowed.
        2. **Domain keyword match** — the query must contain at least one keyword
           from ``_DOMAIN_KEYWORDS``.  If none match, the query is rejected unless
           no out-of-domain pattern is present either (ambiguous short queries get
           through).
        3. **Explicit out-of-domain pattern** — even if a domain keyword matches,
           a hard out-of-domain pattern (e.g. ``"weather"``, ``"recipe"``) causes
           rejection.

        Parameters
        ----------
        query:
            User query string, already translated to English.

        Returns
        -------
        SafetyResult
            ``allowed=False`` with ``reason="out_of_domain"`` when the query is
            clearly off-topic; otherwise ``allowed=True``.
        """
        stripped = query.strip()
        if len(stripped) < _DOMAIN_CHECK_MIN_CHARS:
            # Treat very short inputs as in-scope (greetings / confirmations).
            return SafetyResult(allowed=True, reason="ok")

        lowered = stripped.lower()

        # Hard out-of-domain patterns override any keyword match.
        for pattern in _OUT_OF_DOMAIN_PATTERNS:
            if pattern in lowered:
                return SafetyResult(allowed=False, reason="out_of_domain")

        # Require at least one domain keyword for longer queries.
        for keyword in _DOMAIN_KEYWORDS:
            if keyword in lowered:
                return SafetyResult(allowed=True, reason="ok")

        return SafetyResult(allowed=False, reason="out_of_domain")

    def check_assistant_response(self, response_text: str) -> SafetyResult:
        """Blocks known unsafe phrases in assistant output.

        Parameters
        ----------
        response_text:
            The full assistant response string before it is returned to the user.

        Returns
        -------
        SafetyResult
            ``allowed=False`` with ``reason="unsafe_response"`` if a blocked
            phrase is detected; otherwise ``allowed=True``.
        """
        lowered = response_text.lower()
        for phrase in _BLOCKED_TOPICS:
            if phrase in lowered:
                return SafetyResult(allowed=False, reason="unsafe_response")
        return SafetyResult(allowed=True, reason="ok")

    def redact_pii(self, text: str) -> str:
        """Redacts email addresses and phone numbers in text before persistence.

        Uses simple regex patterns.  Does not redact names or addresses.

        Parameters
        ----------
        text:
            Any string that may contain PII.

        Returns
        -------
        str
            Text with emails replaced by ``[redacted-email]`` and phone numbers
            replaced by ``[redacted-phone]``.
        """
        redacted = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[redacted-email]", text)
        redacted = re.sub(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[redacted-phone]", redacted)
        return redacted
