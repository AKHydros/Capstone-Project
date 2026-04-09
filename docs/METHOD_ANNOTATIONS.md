# Method Annotations And Explanations

This document explains what each function and method does across the current repository's source code (`src/`).

Scope notes:
- Includes: backend API, services, retrieval, loaders, config, UI.
- Excludes: logs/data artifacts, `.gitignore`, and runtime-generated cache files.
- Repo detection: one repository at current workspace root.

## `src/backend/api/app.py`
| Symbol | What It Does |
|---|---|
| `_runtime()` | Lazily builds and returns a singleton `BootstrapArtifacts` runtime instance under a lock. |
| `_rebuild_runtime(refresh_prompts=False)` | Forces a full runtime rebuild (reindex + optional prompt refresh) and replaces the singleton instance. |
| `_auth_dependency(request, x_internal_token)` | Enforces internal token auth on API routes (except LLM health endpoint). |
| `warm_runtime()` | FastAPI startup hook that pre-warms runtime to avoid first-request cold-start latency. |
| `trace_middleware(request, call_next)` | Adds trace IDs, measures request latency, records API errors, and injects latency headers. |
| `get_llm_health()` | Returns LLM connectivity status from `LlmHealthService`. |
| `post_consent_record(payload)` | Persists consent decisions and session logging level. |
| `get_compliance_status()` | Returns consent enforcement, governance counts, and LLM health summary. |
| `post_agent_router(request, raw_request)` | Main chat route: converts API payload into router input and returns routed response payload. |
| `get_index_health()` | Returns index/cache diagnostics and cache hit-rate counters. |
| `post_index_rebuild(payload)` | Triggers forced index rebuild and returns rebuilt index health. |
| `get_metrics_sla()` | Returns SLA metrics (latency percentiles, rates, counts). |
| `get_metrics_monthly_snapshots()` | Creates a monthly snapshot and returns snapshots + session rollups. |
| `get_trace(trace_id)` | Retrieves a trace plus latest QA pair for that trace. |
| `list_governance_items(status, limit)` | Lists governance items with optional status filter and count summary. |
| `update_governance_item_status(item_id, status)` | Updates lifecycle status of a governance item. |
| `get_question_library()` | Returns top questions, generated questions, and approved library items. |
| `_index_health_payload(runtime)` | Builds normalized index-health response from runtime caches and counters. |
| `_index_health_payload._hit_rate(hits, misses)` | Nested helper that safely computes a percentage hit rate (0 when denominator is 0). |

## `src/backend/api/client.py`
| Symbol | What It Does |
|---|---|
| `ApiClient.__init__(config, artifacts)` | Initializes API client with remote base URL and local fallback artifacts. |
| `ApiClient.using_remote_api` | Indicates whether calls should go over HTTP (`API_BASE_URL` set). |
| `ApiClient.agent_router(...)` | Calls `/api/agent-router` remotely or local router fallback. |
| `ApiClient.compliance_status()` | Retrieves compliance status remotely or locally. |
| `ApiClient.consent_record(...)` | Records consent remotely or locally with current UTC timestamp. |
| `ApiClient.llm_health()` | Retrieves LLM health remotely or locally. |
| `ApiClient.metrics_sla()` | Retrieves SLA metrics remotely or from local observability store. |
| `ApiClient.monthly_snapshots()` | Retrieves monthly snapshots payload remotely or locally. |
| `ApiClient.get_trace(trace_id)` | Retrieves trace payload remotely or locally. |
| `ApiClient.governance_items(status, limit)` | Retrieves governance listing remotely or locally. |
| `ApiClient.update_governance_status(item_id, status)` | Updates governance item status remotely or locally. |
| `ApiClient.question_library()` | Retrieves question-library payload remotely or locally. |
| `ApiClient.index_health()` | Retrieves index-health diagnostics remotely or locally. |
| `ApiClient.rebuild_index(refresh_prompts=False)` | Requests reindex remotely or rebuilds local runtime artifacts. |
| `ApiClient._local_agent_router(...)` | Local adapter call into `AgentRouterService.handle`. |
| `ApiClient._local_compliance_status()` | Local compliance-status builder. |
| `ApiClient._local_consent_record(...)` | Local consent/session persistence helper. |
| `ApiClient._local_llm_health()` | Local LLM health payload helper. |
| `ApiClient._local_monthly_snapshots()` | Local monthly snapshot generation helper. |
| `ApiClient._local_update_governance_status(...)` | Local governance status update helper. |
| `ApiClient._local_question_library()` | Local question-library payload helper. |
| `ApiClient._local_index_health()` | Local index/cache diagnostics payload helper. |
| `ApiClient._local_index_health.hit_rate(hits, misses)` | Nested helper that computes cache hit-rate percentages safely. |
| `ApiClient._local_rebuild_index(refresh_prompts)` | Rebuilds local runtime and returns fresh index-health payload. |
| `ApiClient._remote_or_fallback(remote_fn, fallback_fn)` | Executes remote path when available; falls back on HTTP error. |
| `ApiClient._get_json(path, params, include_auth)` | Executes authenticated GET and returns JSON payload. |
| `ApiClient._post_json(path, payload, params)` | Executes authenticated POST and returns JSON payload. |
| `ApiClient._headers(include_auth)` | Builds request headers, including internal token when required. |

## `src/backend/api/gunicorn_conf.py`
| Symbol | What It Does |
|---|---|
| `_int_env(name, default, min_value, max_value)` | Safely parses and clamps integer env vars for gunicorn settings. |

## `src/backend/api/schemas.py`
These are Pydantic request/response models for API contracts.

| Symbol | What It Represents |
|---|---|
| `RouterFilters` | Optional filtering fields applied to retrieval. |
| `InferenceSettings` | Optional generation controls (`max_length`, `top_p`, `temperature`). |
| `AgentRouterRequest` | Input contract for routed chat requests. |
| `ResultCardResponse` | Card schema for retrieved grounded records. |
| `AgentRouterResponse` | Output contract for routed chat responses. |
| `ConsentRecordRequest` | Input contract for consent recording. |
| `ConsentRecordResponse` | Output contract for consent recording. |
| `ComplianceStatusResponse` | Compliance status response shape. |
| `LlmHealthResponse` | LLM health response shape. |
| `ReindexRequest` | Input contract for index rebuild requests. |
| `IndexHealthResponse` | Index/cache diagnostics response shape. |
| `ReindexResponse` | Rebuild operation response shape. |

## `src/backend/business_rules.py`
| Symbol | What It Does |
|---|---|
| `normalize_filter(value)` | Normalizes optional filter values by trimming and handling empty input. |
| `infer_wave_year(variable_name)` | Derives survey year from PMG variable naming convention. |
| `infer_survey_name(variable_name)` | Derives survey key prefix from variable ID. |
| `is_valid_question_text(text)` | Filters invalid/system rows from retrieval corpus. |
| `apply_filters(records, survey_name, wave_year, topic_label, topic_source_type)` | Applies deterministic filtering over candidate records. |
| `categorize_question_labels(question_text, value_labels)` | Maps question/value text into taxonomy labels and label-source metadata. |
| `build_grounded_context(records)` | Builds compact grounded context text used for LLM summarization. |

## `src/backend/cache/index_cache.py`
| Symbol | What It Does |
|---|---|
| `IndexCache.__init__(cache_dir)` | Initializes cache directory and cache file path. |
| `IndexCache.load(signature)` | Loads retriever bundle if cache exists and signature matches. |
| `IndexCache.inspect(signature)` | Returns cache status (`latent`, `stale`, `updated`) plus metadata. |
| `IndexCache.save(signature, retriever)` | Serializes retriever + metadata to cache file. |
| `IndexCache.clear()` | Deletes cache file if present. |
| `IndexCache._read_bundle()` | Reads and validates cached bundle, normalizing legacy metadata. |
| `IndexCache._normalize_metadata(metadata, retriever)` | Backfills/normalizes metadata fields for compatibility. |
| `build_signature(...)` | Builds deterministic cache signature from sources, model mode, and rules. |

## `src/backend/cache/ttl_cache.py`
| Symbol | What It Does |
|---|---|
| `TTLCache.__init__(ttl_seconds, max_size)` | Creates bounded in-memory TTL cache with thread lock and counters. |
| `TTLCache.__getstate__()` | Serializes cache state without non-pickleable lock. |
| `TTLCache.__setstate__(state)` | Restores serialized cache state and recreates lock. |
| `TTLCache.get(key)` | Reads non-expired value and updates hit/miss/expired counters. |
| `TTLCache.set(key, value)` | Inserts value with expiration and evicts oldest if full. |
| `TTLCache.clear()` | Removes all cached entries. |
| `TTLCache.stats()` | Returns cache statistics snapshot. |

## `src/backend/config.py`
| Symbol | What It Does |
|---|---|
| `_parse_csv_env(raw)` | Parses comma-separated env values into unique ordered tuple. |
| `_parse_bool_env(raw, default)` | Parses boolean-like env strings with default fallback. |
| `load_config()` | Loads all runtime configuration from environment variables (memoized). |
| `reset_config_cache()` | Clears memoized config to force re-read from environment. |

## `src/backend/llm/openai_client.py`
| Symbol | What It Does |
|---|---|
| `OpenAIChatClient.__init__()` | Initializes OpenAI client if key is configured and sets default model. |
| `OpenAIChatClient.enabled` | Indicates whether chat client is configured and available. |
| `OpenAIChatClient.summarize(...)` | Sends grounded chat prompts to OpenAI Responses API and returns text. |

## `src/backend/loaders/excel_repository.py`
| Symbol | What It Does |
|---|---|
| `ExcelRepository.load_records()` | Reads Excel sheets, builds value-label mappings, and returns normalized `QuestionRecord` list. |

## `src/backend/loaders/survey_prompt_loader.py`
| Symbol | What It Does |
|---|---|
| `SurveyPromptLoader.__init__(data_dir, cache_dir)` | Configures docx source and cache paths. |
| `SurveyPromptLoader.load_prompts(max_prompts, force_refresh)` | Loads/caches starter prompts extracted from docx files. |
| `SurveyPromptLoader.clear_cache()` | Removes prompt and question-hint cache files. |
| `SurveyPromptLoader.load_question_hints(force_refresh)` | Loads/caches survey question hints extracted from docx files. |
| `SurveyPromptLoader._extract_prompts(files, max_prompts)` | Extracts normalized, deduplicated prompt candidates from paragraphs. |
| `SurveyPromptLoader._extract_question_hints(files)` | Extracts question reference hints keyed by survey and question token. |
| `SurveyPromptLoader._iter_paragraphs(file_path)` | Iterates normalized paragraph text from a `.docx` XML document. |
| `SurveyPromptLoader._normalize_candidate(text)` | Filters and normalizes prompt text candidates. |
| `SurveyPromptLoader._build_signature(files)` | Computes prompt-cache signature from file names/sizes/mtimes. |
| `SurveyPromptLoader._read_cache()` | Reads and validates starter prompt cache payload. |
| `SurveyPromptLoader._write_cache(payload)` | Writes starter prompt cache payload to disk. |
| `SurveyPromptLoader._read_question_hint_cache()` | Reads and validates question-hint cache payload. |
| `SurveyPromptLoader._write_question_hint_cache(payload)` | Writes question-hint cache payload to disk. |

## `src/backend/models.py`
| Symbol | What It Does |
|---|---|
| `QuestionRecord.value_labels_text` | Returns compact joined string of value labels (up to 20). |
| `QuestionRecord.topic_labels_text` | Returns comma-separated topic labels. |
| `QuestionRecord.topic_sources_text` | Returns topic labels annotated with source attribution. |
| `QuestionRecord.document_text` | Returns normalized text payload used for indexing/retrieval. |
| `ChatResponse` | Dataclass response container for chat output plus retrieval metadata flags. |

## `src/backend/observability/db.py`
| Symbol | What It Does |
|---|---|
| `ObservabilityStore.__init__(db_path)` | Initializes DB path and ensures observability schema exists. |
| `ObservabilityStore._connect()` | Returns thread-local SQLite connection with performance PRAGMAs. |
| `ObservabilityStore.close()` | Closes and clears current thread-local DB connection. |
| `ObservabilityStore._init_schema()` | Creates required tables and indexes for sessions/events/traces/governance. |
| `ObservabilityStore.record_session_start(...)` | Inserts session if absent with locale/consent state. |
| `ObservabilityStore.ensure_session_and_get_consent(...)` | Ensures session row exists and returns current consent flag. |
| `ObservabilityStore.session_has_consent(session_id)` | Reads consent flag for session. |
| `ObservabilityStore.record_session_end(...)` | Marks session end time and completion status. |
| `ObservabilityStore.record_event(...)` | Inserts event record with payload/error details. |
| `ObservabilityStore.record_trace(...)` | Upserts trace record and latency/fallback metadata. |
| `ObservabilityStore.record_qa_pair(...)` | Inserts query/response pair for audit and analytics. |
| `ObservabilityStore.record_consent(...)` | Inserts consent record and updates session consent state. |
| `ObservabilityStore.upsert_governance_item(...)` | Upserts governance item by question text and returns item ID. |
| `ObservabilityStore.update_governance_status(item_id, status)` | Updates governance status and timestamp. |
| `ObservabilityStore.list_governance_items(status, limit)` | Lists governance items with deserialized labels/takeaways. |
| `ObservabilityStore.governance_counts()` | Returns aggregate governance counts by status. |
| `ObservabilityStore.get_trace(trace_id)` | Returns trace record and latest QA payload for that trace. |
| `ObservabilityStore.sla_metrics()` | Computes p50/p95 latency and operational rates/counters. |
| `ObservabilityStore.create_monthly_snapshot(month_key)` | Saves monthly KPI snapshot from current SLA metrics. |
| `ObservabilityStore.monthly_snapshots()` | Returns monthly snapshot history. |
| `ObservabilityStore.recent_qa_pairs(limit)` | Returns recent QA records with decoded JSON fields. |
| `ObservabilityStore.session_rollups(limit)` | Returns session-level rollups joined with event counts. |
| `_now()` | Returns current UTC ISO timestamp (seconds precision). |
| `_percentile(values, p)` | Returns percentile value from sorted latency list. |

## `src/backend/observability/json_logger.py`
| Symbol | What It Does |
|---|---|
| `JsonEventLogger.__post_init__()` | Ensures log directory exists and sets JSONL file path. |
| `JsonEventLogger.log(event)` | Appends timestamped event JSON line to log file. |

## `src/backend/retrieval/hybrid.py`
| Symbol | What It Does |
|---|---|
| `HybridRetriever.build(records)` | Builds chunks + lexical + semantic retrievers from records. |
| `HybridRetriever.search_with_details(...)` | Runs weighted lexical+semantic ranking, filtering, and returns diagnostics. |
| `HybridRetriever.search(...)` | Convenience wrapper returning only ranked records. |

## `src/backend/retrieval/lexical.py`
| Symbol | What It Does |
|---|---|
| `LexicalRetriever.build(chunks)` | Fits TF-IDF vectorizer and matrix over chunk corpus. |
| `LexicalRetriever.score(query)` | Returns lexical similarity scores per chunk. |

## `src/backend/retrieval/pipeline.py`
| Symbol | What It Does |
|---|---|
| `ingest_clean_chunk(records, chunk_size, overlap)` | Converts records into cleaned overlapping text chunks for indexing. |
| `_clean_text(text)` | Normalizes whitespace in source text. |
| `_chunk_text(text, chunk_size, overlap)` | Splits text into chunk windows with overlap. |

## `src/backend/retrieval/semantic.py`
| Symbol | What It Does |
|---|---|
| `SemanticRetriever.build(records, chunks)` | Builds semantic representation using OpenAI embeddings or local TF-IDF fallback. |
| `SemanticRetriever.score(query)` | Returns semantic scores only. |
| `SemanticRetriever.score_with_meta(query)` | Returns semantic scores plus embedding-cache-hit metadata. |
| `SemanticRetriever.embedding_cache_stats()` | Returns embedding query cache counters. |
| `_openai_client(api_key)` | Returns memoized OpenAI client for embedding calls. |
| `_normalize_query(query)` | Normalizes query text for embedding-cache keying. |

## `src/backend/services/agent_router_service.py`
| Symbol | What It Does |
|---|---|
| `AgentRouterService.__init__(...)` | Wires chatbot, safety, translation, governance, observability, and router cache. |
| `AgentRouterService.handle(request)` | Main routing pipeline: consent, safety, cache, route decision, answer generation, telemetry, governance persistence. |
| `AgentRouterService._record_qa(request, query_en, output)` | Persists redacted QA pair for traceability. |
| `AgentRouterService._record_event(request, output, payload, include_content)` | Writes structured event to DB and JSON logger. |
| `AgentRouterService._cache_key(...)` | Builds deterministic router-response cache key from request dimensions. |
| `AgentRouterService._decide_route(mode)` | Chooses deterministic vs LLM route based on mode and health. |
| `AgentRouterService._to_card(record)` | Maps `QuestionRecord` to API response card shape. |

## `src/backend/services/bootstrap_service.py`
| Symbol | What It Does |
|---|---|
| `BootstrapService.__init__(config)` | Stores app configuration for runtime assembly. |
| `BootstrapService.build(force_rebuild_cache, force_refresh_prompts)` | Constructs full runtime graph: data loading, retriever, caches, services, governance, observability. |
| `_infer_data_dir(excel_source_path)` | Determines data directory fallback when source parent is missing. |
| `_collect_excel_sources(primary_source, data_dir)` | Collects primary and uploaded Excel sources for ingestion. |

## `src/backend/services/chatbot_service.py`
| Symbol | What It Does |
|---|---|
| `ChatbotService.__post_init__()` | Initializes RAG behavior flags and answer cache from overrides/env. |
| `ChatbotService.chat(...)` | Main chat pipeline: exact-ID lookup, allowed-values path, hybrid retrieval, LLM/deterministic answer selection, answer caching. |
| `ChatbotService.answer_cache_stats()` | Returns answer cache usage counters. |
| `ChatbotService._should_use_llm(...)` | Applies RAG confidence/gap logic to decide if synthesis is needed. |
| `ChatbotService._confidence_score(diagnostics)` | Converts retrieval diagnostics top score into bounded confidence. |
| `ChatbotService._is_synthesis_intent(query)` | Detects explicit summarize/analysis intent in user query. |
| `ChatbotService._deterministic_answer(...)` | Builds grounded non-LLM fallback answer from top cards. |
| `ChatbotService._answer_cache_key(...)` | Creates deterministic key for answer-level cache. |
| `ChatbotService._exact_question_lookup_response(...)` | Executes deterministic survey/question variant lookup (including alias variants) before fuzzy search. |
| `ChatbotService._quick_allowed_values_response(...)` | Handles allowed-values intent with direct value-label answers. |
| `ChatbotService._is_allowed_values_intent(query)` | Detects dropdown/options-related intent keywords. |
| `ChatbotService._extract_survey_name(query)` | Extracts survey token like `PMG20_GAM` from free text. |
| `ChatbotService._extract_question_ref(query)` | Extracts question reference token (supports `q` and `qq` ID forms). |
| `ChatbotService._match_question_ref_records(records, question_ref)` | Filters records that match extracted question reference. |
| `ChatbotService._question_component(question_id)` | Extracts question number/suffix component from variable ID. |
| `ChatbotService._split_question_ref(question_ref)` | Splits normalized question ref into number/suffix tuple. |
| `ChatbotService._split_question_component(component)` | Splits extracted component into number/suffix tuple. |
| `ChatbotService._normalize_question_number(number)` | Normalizes numeric component (e.g., strips leading zeros). |
| `ChatbotService._build_variant_map(records, ref_number)` | Groups matching records into variant buckets (`11a`, `11b`, etc.). |
| `ChatbotService._variant_sort_key(key)` | Defines deterministic variant sort order. |
| `ChatbotService._record_sort_key(record)` | Prioritizes canonical IDs over alias IDs when selecting representative record. |
| `ChatbotService._best_variant_record(records)` | Chooses best representative record for a variant bucket. |
| `ChatbotService._build_specific_variant_response(...)` | Builds final answer for a specific variant, merging value-label aliases. |
| `ChatbotService._merge_value_labels(records)` | Deduplicates/merges formatted value labels across records. |
| `ChatbotService._hinted_question_ref_records(...)` | Uses docx-derived hints to improve question-reference retrieval fallback. |
| `ChatbotService._format_value_labels(value_labels)` | Normalizes value-label formatting (including numeric cleanup). |
| `_normalize_query(query)` | Module helper to normalize query strings for cache keys. |
| `_env_bool(name, default)` | Reads boolean env variable with fallback. |
| `_env_float(name, default)` | Reads float env variable with fallback. |
| `_env_int(name, default)` | Reads integer env variable with fallback. |

## `src/backend/services/governance_service.py`
| Symbol | What It Does |
|---|---|
| `GovernanceService.__init__(store)` | Stores observability/governance persistence dependency. |
| `GovernanceService.generate_labels_takeaways(...)` | Derives governance labels and key takeaways from ranked results. |
| `GovernanceService.save_qa_item(...)` | Saves or updates governance item with validated status. |
| `GovernanceService.set_status(item_id, status)` | Validates and updates governance item status. |
| `GovernanceService.approved_library(limit)` | Returns approved governance library items. |

## `src/backend/services/llm_health_service.py`
| Symbol | What It Does |
|---|---|
| `LlmHealthService.status()` | Reports `Connected` when API key exists, otherwise `Degraded` with reason. |

## `src/backend/services/question_library_service.py`
| Symbol | What It Does |
|---|---|
| `QuestionLibraryService.__init__(governance_service)` | Stores governance dependency for preloaded question persistence. |
| `QuestionLibraryService.preload(records, iterations)` | Builds top/generated question lists and persists them into governance store. |
| `QuestionLibraryService._derive_top_questions(records, limit)` | Produces deterministic top-question list from ordered records. |
| `QuestionLibraryService._generate_questions(records, iterations)` | Generates heuristic question prompts from dominant topic labels. |

## `src/backend/services/safety_service.py`
| Symbol | What It Does |
|---|---|
| `SafetyService.check_user_query(query)` | Blocks known unsafe phrases in user input. |
| `SafetyService.check_assistant_response(response_text)` | Blocks known unsafe phrases in assistant output. |
| `SafetyService.redact_pii(text)` | Redacts email addresses and phone numbers in text. |

## `src/backend/services/translation_service.py`
| Symbol | What It Does |
|---|---|
| `TranslationService.__init__(ttl_seconds)` | Initializes translation cache. |
| `TranslationService.normalize_language(language)` | Normalizes language code to supported set (`en`, `fr`). |
| `TranslationService.translate(text, source_language, target_language)` | Performs cached dictionary translation between supported languages. |
| `TranslationService.stats()` | Returns translation cache stats. |
| `TranslationService._translate_with_dictionary(text, source_language, target_language)` | Applies phrase-level mapping rules for simple translation fallback. |

## `src/ui/app.py`
| Symbol | What It Does |
|---|---|
| `build_runtime(force_rebuild_cache, force_refresh_prompts)` | Builds UI runtime artifacts (cached) via backend bootstrap. |
| `_init_state()` | Initializes all Streamlit session state defaults. |
| `_save_uploaded_files(uploads_dir, uploaded_files)` | Persists uploaded files and returns count saved. |
| `_cache_status_text(artifacts)` | Formats cache status summary values for UI display. |
| `_request_cache_action(...)` | Schedules cache/prompt refresh action and triggers Streamlit rerun. |
| `_ensure_chat_session_enabled(client)` | Best-effort auto-consent call for chat session activation. |
| `_combine_starter_prompts(starter_prompts, top_questions)` | Merges and deduplicates starter prompts for UX. |
| `_chat_history_text(history)` | Serializes chat history into plain-text transcript format. |
| `_render_admin_dashboard(client)` | Renders health, index, SLA, trace, and governance admin panels. |

## `src/ui/style.py`
| Symbol | What It Does |
|---|---|
| `apply_pmg_theme()` | Injects custom CSS style overrides into Streamlit app. |
