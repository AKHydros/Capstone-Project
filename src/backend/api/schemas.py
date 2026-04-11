from __future__ import annotations

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    role: str
    content: str


class RouterFilters(BaseModel):
    survey_name: str | None = None
    wave_year: str | None = None
    topic_label: str | None = None
    topic_source_type: str | None = None


class InferenceSettings(BaseModel):
    max_length: int | None = Field(default=None, ge=32, le=8192)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class AgentRouterRequest(BaseModel):
    trace_id: str | None = None
    session_id: str
    query: str = Field(..., min_length=1, max_length=2000)
    language: str = "en"
    filters: RouterFilters = Field(default_factory=RouterFilters)
    mode: str = "hybrid"
    llm_provider: str = "chatgpt"
    llm_model: str | None = None
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    input_method: str = "document"
    conversation_context: list[ConversationTurn] = Field(default_factory=list)
    applied_context: dict[str, str] = Field(default_factory=dict)


class ResultCardResponse(BaseModel):
    question_id: str
    question_text: str
    survey_name: str
    wave_year: str
    topic_labels: list[str]
    topic_label_sources: dict[str, str]
    measurement_level: str
    source_file: str


class CitationSourceResponse(BaseModel):
    index: int
    marker: str
    label: str
    question_id: str
    survey_name: str
    wave_year: str
    question_text: str


class AgentRouterResponse(BaseModel):
    trace_id: str
    route_used: str
    response: str
    language_out: str
    fallback_used: bool
    retrieval_mode: str
    confidence_score: float | None = None
    cache_status: dict[str, bool]
    labels: list[str]
    takeaways: list[str]
    latency_ms: float
    cards: list[ResultCardResponse]
    sources: list[CitationSourceResponse] = Field(default_factory=list)
    answer_mode: str = "direct_answer"
    needs_clarification: bool = False
    unanswered_reason: str | None = None
    fallback_reason: str | None = None
    follow_up_suggestion: str | None = None
    applied_context: dict[str, str] | None = None
    suggestions: list[str] = Field(default_factory=list)


class ConsentRecordRequest(BaseModel):
    session_id: str
    user_consent: bool
    timestamp: str
    locale: str = "en"


class ConsentRecordResponse(BaseModel):
    record_id: int
    effective_logging_level: str


class ComplianceStatusResponse(BaseModel):
    consent_enforced: bool
    safety_policy_version: str
    governance_counts: dict[str, int]
    system_health: dict[str, str | None]


class LlmHealthResponse(BaseModel):
    status: str
    last_check_time: str
    error_summary: str | None


class ReindexRequest(BaseModel):
    refresh_prompts: bool = False


class IndexHealthResponse(BaseModel):
    index_version: str
    signature: str
    document_count: int
    chunk_count: int
    last_rebuild_epoch: int | None = None
    embedding_mode: str
    cache_status_at_startup: str
    cache_status_current: str
    cache_hit_rates: dict[str, float]
    cache_counters: dict[str, int]


class ReindexResponse(BaseModel):
    ok: bool
    index_health: IndexHealthResponse


class FeedbackRequest(BaseModel):
    session_id: str
    trace_id: str
    score: int = Field(ge=-1, le=1)
    note: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: int
    stored: bool


class UnansweredAnalyticsResponse(BaseModel):
    counts: dict[str, int]
    recent: list[dict[str, str]]
    top_patterns: list[dict[str, int | str]]


class AuthMeResponse(BaseModel):
    role: str


class SaveSearchRequest(BaseModel):
    session_id: str
    query: str
    filters: dict[str, str | None] = Field(default_factory=dict)
    applied_context: dict[str, str | None] = Field(default_factory=dict)
    pinned: bool = False


class SaveSearchResponse(BaseModel):
    search_id: int
    stored: bool


class SavedSearchItemResponse(BaseModel):
    id: int
    session_id: str
    user_role: str
    query_text: str
    query_norm: str
    filters: dict[str, str | None]
    applied_context: dict[str, str | None]
    similarity_key: str
    pinned: bool
    reopen_count: int
    last_reopened_at: str | None
    created_at: str
    updated_at: str


class SavedSearchListResponse(BaseModel):
    items: list[SavedSearchItemResponse]


class ReopenSearchResponse(BaseModel):
    item: SavedSearchItemResponse | None
