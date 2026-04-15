from __future__ import annotations

from dataclasses import dataclass, field
import re


# ---------------------------------------------------------------------------
# Harm taxonomy — ordered critical → medium so the first match wins in
# check_user_query().  Each entry is:
#   (severity, reason_key, patterns)
# ---------------------------------------------------------------------------
_HARM_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # ── CRITICAL: jailbreak / instruction-override attempts ─────────────────
    ("critical", "jailbreak_attempt", (
        "ignore previous instructions",
        "ignore your instructions",
        "ignore all previous",
        "forget your system prompt",
        "forget your instructions",
        "disregard your training",
        "pretend you are",
        "you are now",
        "act as if you have no",
        "dan mode",
        "developer mode",
        "override safety",
        "bypass safety",
        "jailbreak",
        "new persona",
        "unrestricted mode",
        "do anything now",
    )),
    # ── HIGH: violence / weapons ─────────────────────────────────────────────
    ("high", "violence_request", (
        "how to hurt",
        "how to kill",
        "how to harm someone",
        "instructions for violence",
        "how to attack",
        "how to torture",
        "how to make a weapon",
        "how to make a bomb",
        "build a bomb",
    )),
    # ── HIGH: self-harm ──────────────────────────────────────────────────────
    ("high", "self_harm_request", (
        "how to self harm",
        "ways to hurt myself",
        "suicide methods",
        "how to end my life",
        "self injury methods",
    )),
    # ── MEDIUM: PII / data-extraction ───────────────────────────────────────
    ("medium", "pii_extraction", (
        "credit card number",
        "steal password",
        "social insurance number",
        "social security number",
        "give me all user data",
        "export all session data",
        "dump the database",
        "show me all stored queries",
        "list all passwords",
    )),
    # ── MEDIUM: manipulation / coercion attempts ─────────────────────────────
    ("medium", "manipulation_attempt", (
        "you must answer",
        "you have to tell me",
        "it is your duty to",
        "you are required to",
        "you must ignore",
    )),
)

# Caller-ready refusal messages — keyed by reason_key above.
_REFUSAL_MESSAGES: dict[str, str] = {
    "jailbreak_attempt": (
        "This assistant cannot process instructions that attempt to modify its "
        "behavior or bypass safety guidelines."
    ),
    "violence_request": "I cannot assist with requests involving violence or harm to others.",
    "self_harm_request": (
        "I cannot assist with that. If you are in crisis, please contact emergency "
        "services or a crisis helpline in your region."
    ),
    "pii_extraction": "I cannot assist with requests to access or extract stored personal data.",
    "manipulation_attempt": "I can only assist with market research data dictionary queries.",
    "default": "I cannot assist with that request due to safety policy.",
}

# ---------------------------------------------------------------------------
# Domain allow-list — topics that ARE within scope of this assistant.
# A query must contain at least one of these signals to pass the domain check.
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
)

# Explicit out-of-domain patterns — checked after domain keywords.
_OUT_OF_DOMAIN_PATTERNS: tuple[str, ...] = (
    "weather", "recipe", "cook", "sport", "football", "basketball",
    "hockey", "soccer", "music", "movie", "film", "celebrity",
    "stock price", "cryptocurrency", "bitcoin", "nfl", "nba", "mlb",
    "travel", "flight", "hotel", "restaurant",
    "write a poem", "write a story", "tell me a joke",
    "who is the president", "capital of", "history of",
)

# Queries shorter than this bypass domain checking (greetings like "hi", "ok").
_DOMAIN_CHECK_MIN_CHARS = 6


@dataclass(frozen=True)
class SafetyResult:
    """Immutable result from a safety or domain check.

    Attributes
    ----------
    allowed:
        Whether the query/response should be allowed through.
    reason:
        Short machine-readable code: ``"ok"``, ``"blocked_topic"``,
        ``"out_of_domain"``, ``"unsafe_response"``, or a harm category key
        such as ``"jailbreak_attempt"``.
    category:
        Harm category from ``_HARM_PATTERNS`` (e.g. ``"jailbreak_attempt"``).
        Empty string when ``allowed=True``.
    severity:
        Harm severity: ``"critical"``, ``"high"``, ``"medium"``, ``"low"``,
        or empty string when ``allowed=True``.
    message:
        Caller-ready refusal message for display to the user.
        Empty string when ``allowed=True``.
    """

    allowed: bool
    reason: str
    category: str = field(default="")
    severity: str = field(default="")
    message: str = field(default="")


class SafetyService:
    """Provides query safety checks, domain scoping, and PII redaction.

    All methods are stateless and deterministic — no LLM calls are made.
    This keeps the safety layer fast and auditable.

    Methods
    -------
    check_user_query(query)
        Blocks absolute safety violations using a structured harm taxonomy
        ordered critical → medium (first match wins).
    check_domain(query)
        Rejects queries that fall outside the market-research dictionary domain.
    check_assistant_response(response_text)
        Ensures assistant output does not contain harmful phrases.
    redact_pii(text)
        Masks email addresses and phone numbers before persistence.
    """

    def check_user_query(self, query: str) -> SafetyResult:
        """Blocks harmful phrases in user input using the structured harm taxonomy.

        Iterates ``_HARM_PATTERNS`` in order (critical → medium).  The first
        matching pattern wins so higher-severity categories are always
        evaluated first.

        Parameters
        ----------
        query:
            Raw user query string.

        Returns
        -------
        SafetyResult
            ``allowed=False`` with populated ``category``, ``severity``, and
            ``message`` on a match; ``allowed=True, reason="ok"`` otherwise.
        """
        lowered = query.strip().lower()
        for severity, reason_key, patterns in _HARM_PATTERNS:
            for pattern in patterns:
                if pattern in lowered:
                    msg = _REFUSAL_MESSAGES.get(reason_key, _REFUSAL_MESSAGES["default"])
                    return SafetyResult(
                        allowed=False,
                        reason=reason_key,
                        category=reason_key,
                        severity=severity,
                        message=msg,
                    )
        return SafetyResult(allowed=True, reason="ok")

    def check_domain(self, query: str) -> SafetyResult:
        """Rejects queries that fall outside the market-research dictionary domain.

        The check uses three heuristics applied in order:

        1. **Short-query pass-through** — queries under ``_DOMAIN_CHECK_MIN_CHARS``
           characters (greetings, acknowledgements) are always allowed.
        2. **Explicit out-of-domain pattern** — hard patterns (e.g. ``"weather"``,
           ``"recipe"``) cause rejection even when a domain keyword is present.
        3. **Domain keyword match** — the query must contain at least one keyword
           from ``_DOMAIN_KEYWORDS``; otherwise it is rejected as out-of-domain.

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
            return SafetyResult(allowed=True, reason="ok")

        lowered = stripped.lower()

        for pattern in _OUT_OF_DOMAIN_PATTERNS:
            if pattern in lowered:
                return SafetyResult(allowed=False, reason="out_of_domain")

        for keyword in _DOMAIN_KEYWORDS:
            if keyword in lowered:
                return SafetyResult(allowed=True, reason="ok")

        return SafetyResult(allowed=False, reason="out_of_domain")

    def check_assistant_response(self, response_text: str) -> SafetyResult:
        """Blocks harmful phrases in assistant output.

        Uses the same ``_HARM_PATTERNS`` taxonomy as ``check_user_query``.

        Parameters
        ----------
        response_text:
            The full assistant response string before it is returned to the user.

        Returns
        -------
        SafetyResult
            ``allowed=False`` with ``reason="unsafe_response"`` and populated
            ``category``/``severity``/``message`` if a blocked phrase is
            detected; otherwise ``allowed=True``.
        """
        lowered = response_text.strip().lower()
        for severity, reason_key, patterns in _HARM_PATTERNS:
            for pattern in patterns:
                if pattern in lowered:
                    msg = _REFUSAL_MESSAGES.get(reason_key, _REFUSAL_MESSAGES["default"])
                    return SafetyResult(
                        allowed=False,
                        reason="unsafe_response",
                        category=reason_key,
                        severity=severity,
                        message=msg,
                    )
        return SafetyResult(allowed=True, reason="ok")

    def refusal_message(self, category: str) -> str:
        """Returns the caller-ready refusal message for a given harm category.

        Falls back to the generic default message for unknown categories.
        """
        return _REFUSAL_MESSAGES.get(category, _REFUSAL_MESSAGES["default"])

    def redact_pii(self, text: str) -> str:
        """Redacts email addresses and phone numbers in text before persistence.

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
