from __future__ import annotations

from pydantic import BaseModel, Field


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
    query: str
    language: str = "en"
    filters: RouterFilters = Field(default_factory=RouterFilters)
    mode: str = "hybrid"
    llm_provider: str = "chatgpt"
    llm_model: str | None = None
    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    input_method: str = "document"


class ResultCardResponse(BaseModel):
    question_id: str
    question_text: str
    survey_name: str
    wave_year: str
    topic_labels: list[str]
    topic_label_sources: dict[str, str]
    measurement_level: str
    source_file: str


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
