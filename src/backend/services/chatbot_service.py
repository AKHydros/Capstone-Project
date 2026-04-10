from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
import re

from ..business_rules import CHAT_RULES, apply_filters, build_grounded_context
from ..cache.ttl_cache import CacheStats, TTLCache
from ..llm.openai_client import OpenAIChatClient
from ..models import ChatResponse, QuestionRecord
from ..retrieval.hybrid import HybridRetriever, RetrievalDiagnostics


_CONTEXT_FIELD_DISPLAY: dict[str, str] = {
    "variable": "Variable",
    "position": "Position",
    "label": "Label",
    "measurement_level": "Measurement Level",
    "role": "Role",
    "column_width": "Column Width",
    "alignment": "Alignment",
    "print_format": "Print Format",
    "write_format": "Write Format",
    "missing_values": "Missing Values",
}
_DEFAULT_CONTEXT_FIELDS: tuple[str, ...] = ("position", "label", "measurement_level", "role")


@dataclass
class ChatbotService:
    retriever: HybridRetriever
    llm_client: OpenAIChatClient
    doc_question_hints: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    rag_enabled_override: bool | None = None
    rag_confidence_threshold_override: float | None = None
    rag_score_gap_threshold_override: float | None = None
    rag_answer_cache_ttl_override: int | None = None
    answer_cache: TTLCache[dict[str, object]] = field(init=False)
    record_index: dict[str, QuestionRecord] = field(init=False, repr=False)
    rag_enabled: bool = field(init=False)
    rag_confidence_threshold: float = field(init=False)
    rag_score_gap_threshold: float = field(init=False)

    def __post_init__(self) -> None:
        """Initializes RAG behavior flags and answer cache from overrides/env."""
        self.rag_enabled = (
            self.rag_enabled_override if self.rag_enabled_override is not None else _env_bool("RAG_ENABLED", default=True)
        )
        self.rag_confidence_threshold = (
            self.rag_confidence_threshold_override
            if self.rag_confidence_threshold_override is not None
            else _env_float("RAG_CONFIDENCE_THRESHOLD", default=0.72)
        )
        self.rag_score_gap_threshold = (
            self.rag_score_gap_threshold_override
            if self.rag_score_gap_threshold_override is not None
            else _env_float("RAG_SCORE_GAP_THRESHOLD", default=0.07)
        )
        answer_cache_ttl = (
            self.rag_answer_cache_ttl_override
            if self.rag_answer_cache_ttl_override is not None
            else _env_int("RAG_ANSWER_CACHE_TTL", default=600)
        )
        self.answer_cache = TTLCache(ttl_seconds=max(answer_cache_ttl, 30), max_size=4096)
        self.record_index = {record.question_id: record for record in self.retriever.records}

    def chat(
        self,
        query: str,
        survey_name: str | None = None,
        wave_year: str | None = None,
        topic_label: str | None = None,
        topic_source_type: str | None = None,
        use_llm: bool | None = None,
        llm_provider: str = "chatgpt",
        llm_model: str | None = None,
        inference: dict[str, int | float] | None = None,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> ChatResponse:
        """Main chat pipeline: exact-ID lookup, allowed-values path, hybrid retrieval, LLM/deterministic answer selection, answer caching."""
        exact_cache_key = self._exact_lookup_cache_key(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        cached_exact_lookup = self.answer_cache.get(exact_cache_key)
        if cached_exact_lookup is not None and str(cached_exact_lookup.get("cache_type", "")) == "exact_lookup":
            cached_response = self._cached_chat_response(cached_exact_lookup)
            if cached_response is not None:
                return cached_response

        exact_response = self._exact_question_lookup_response(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        if exact_response is not None:
            self.answer_cache.set(exact_cache_key, self._serialize_chat_response(exact_response, cache_type="exact_lookup"))
            return exact_response

        quick_response = self._quick_allowed_values_response(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        if quick_response is not None:
            self.answer_cache.set(exact_cache_key, self._serialize_chat_response(quick_response, cache_type="exact_lookup"))
            return quick_response

        if self._is_comparison_intent(query):
            pair = self._extract_two_surveys(query)
            if pair:
                return self._comparison_response(
                    query=query,
                    survey_a=pair[0],
                    survey_b=pair[1],
                    wave_year=wave_year,
                    topic_label=topic_label,
                    topic_source_type=topic_source_type,
                )

        lineage_response = self._lineage_response(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        if lineage_response is not None:
            self.answer_cache.set(exact_cache_key, self._serialize_chat_response(lineage_response, cache_type="exact_lookup"))
            return lineage_response

        search = self.retriever.search_with_details(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        cards = [item.record for item in search.scored_results[: CHAT_RULES.max_cards_display]]

        confidence = self._confidence_score(search.diagnostics)
        embedding_cache_hit = search.diagnostics.embedding_cache_hit

        if not cards:
            suggestions = self._build_did_you_mean(query)
            return ChatResponse(
                answer=_sanitize_answer(
                    "I could not find grounded matches in the current data dictionary. "
                    "Try broader wording, remove filters, or check the question ID format."
                ),
                ranked_results=[],
                retrieval_mode="deterministic",
                answer_mode="direct_answer",
                confidence_score=confidence,
                embedding_cache_hit=embedding_cache_hit,
                answer_cache_hit=False,
                suggestions=suggestions,
            )

        provider_is_supported = llm_provider.lower() in {"chatgpt", "openai"}
        llm_requested = (
            self.llm_client.enabled
            if use_llm is None
            else (use_llm and self.llm_client.enabled and provider_is_supported)
        )

        should_use_llm = self._should_use_llm(
            llm_requested=llm_requested,
            query=query,
            diagnostics=search.diagnostics,
            confidence=confidence,
        )
        retrieval_mode = "llm_synthesized" if should_use_llm else "deterministic"

        answer_cache_key = self._answer_cache_key(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
            cards=cards,
            retrieval_mode=retrieval_mode,
            llm_model=llm_model,
            conversation_context=conversation_context or [],
        )
        cached_answer = self.answer_cache.get(answer_cache_key)
        if cached_answer is not None and str(cached_answer.get("cache_type", "")) == "hybrid_answer":
            raw_follow_up = cached_answer.get("follow_up_suggestion")
            return ChatResponse(
                answer=str(cached_answer.get("answer", "")),
                ranked_results=cards,
                lookup_mode="hybrid_fallback",
                variant_count=0,
                retrieval_mode=str(cached_answer.get("retrieval_mode", retrieval_mode)),
                answer_mode=str(cached_answer.get("answer_mode", "direct_answer")),
                needs_clarification=bool(cached_answer.get("needs_clarification", False)),
                confidence_score=confidence,
                embedding_cache_hit=embedding_cache_hit,
                answer_cache_hit=True,
                follow_up_suggestion=str(raw_follow_up) if raw_follow_up else None,
            )

        if should_use_llm:
            context = build_grounded_context(cards)
            conversation_block = self._conversation_context_block(conversation_context or [])
            low_confidence_notice = (
                "Note: retrieval confidence is below 0.50, meaning the matched records "
                "may not be a perfect fit. Acknowledge this briefly at the start of your answer.\n\n"
                if confidence < 0.50
                else ""
            )
            system_prompt = (
                "You are a grounded research data dictionary assistant for market research surveys. "
                "Be concise, direct, and easy to read. "
                "Only reference retrieved records — never invent question text, IDs, or survey names. "
                "If records do not clearly match the query, say so explicitly.\n\n"
                "Always structure your response EXACTLY as follows (use these exact headings):\n\n"
                "**Answer:** [Direct answer in 1–2 sentences]\n\n"
                "**Key Details:**\n"
                "- [Supporting detail 1]\n"
                "- [Supporting detail 2]\n"
                "- [Supporting detail 3, if applicable]\n\n"
                "**Suggested Follow-up:** [One follow-up question the user might want to ask next]\n\n"
                "If the user asks for allowable values or coded options, list them as a bullet list "
                "under Key Details."
            )
            user_prompt = (
                f"{low_confidence_notice}"
                f"{conversation_block}"
                f"User query: {query}\n\n"
                f"Retrieved records:\n{context}"
            )
            max_length = int(inference["max_length"]) if inference and "max_length" in inference else None
            top_p = float(inference["top_p"]) if inference and "top_p" in inference else None
            temperature = float(inference["temperature"]) if inference and "temperature" in inference else None
            raw_answer = self.llm_client.summarize(
                system_prompt,
                user_prompt,
                model=llm_model,
                max_output_tokens=max_length,
                top_p=top_p,
                temperature=temperature,
            )
            follow_up = self._parse_follow_up_suggestion(raw_answer)
            answer = _sanitize_answer(raw_answer)
        else:
            answer = _sanitize_answer(self._deterministic_answer(query=query, cards=cards, llm_provider=llm_provider))
            follow_up = None

        self.answer_cache.set(
            answer_cache_key,
            {
                "cache_type": "hybrid_answer",
                "answer": answer,
                "retrieval_mode": retrieval_mode,
                "answer_mode": "summary" if should_use_llm else "direct_answer",
                "needs_clarification": False,
                "follow_up_suggestion": follow_up,
            },
        )

        return ChatResponse(
            answer=answer,
            ranked_results=cards,
            lookup_mode="hybrid_fallback",
            variant_count=0,
            retrieval_mode=retrieval_mode,
            answer_mode="summary" if should_use_llm else "direct_answer",
            needs_clarification=False,
            confidence_score=confidence,
            embedding_cache_hit=embedding_cache_hit,
            answer_cache_hit=False,
            follow_up_suggestion=follow_up,
        )

    def answer_cache_stats(self) -> CacheStats:
        """Returns answer cache usage counters."""
        return self.answer_cache.stats()

    def _exact_lookup_cache_key(
        self,
        *,
        query: str,
        survey_name: str | None,
        wave_year: str | None,
        topic_label: str | None,
        topic_source_type: str | None,
    ) -> str:
        """Creates deterministic cache key for exact lookup responses."""
        payload = {
            "q": _normalize_query(query),
            "filters": {
                "survey_name": survey_name,
                "wave_year": wave_year,
                "topic_label": topic_label,
                "topic_source_type": topic_source_type,
            },
            "lookup_mode": "exact_lookup",
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _serialize_chat_response(self, response: ChatResponse, *, cache_type: str) -> dict[str, object]:
        """Serializes a `ChatResponse` into a cache-safe payload."""
        return {
            "cache_type": cache_type,
            "answer": response.answer,
            "ranked_question_ids": [record.question_id for record in response.ranked_results],
            "lookup_mode": response.lookup_mode,
            "variant_count": response.variant_count,
            "retrieval_mode": response.retrieval_mode,
            "answer_mode": response.answer_mode,
            "needs_clarification": response.needs_clarification,
            "confidence_score": response.confidence_score,
            "embedding_cache_hit": response.embedding_cache_hit,
            "follow_up_suggestion": response.follow_up_suggestion,
        }

    def _cached_chat_response(self, payload: dict[str, object]) -> ChatResponse | None:
        """Restores cached chat payload into a `ChatResponse` with resolved ranked records."""
        ranked_ids = payload.get("ranked_question_ids")
        if not isinstance(ranked_ids, list):
            return None

        ranked_results: list[QuestionRecord] = []
        for raw_id in ranked_ids:
            if not isinstance(raw_id, str):
                continue
            record = self.record_index.get(raw_id)
            if record is not None:
                ranked_results.append(record)

        confidence_score = payload.get("confidence_score")
        if isinstance(confidence_score, (int, float)):
            parsed_confidence = float(confidence_score)
        else:
            parsed_confidence = None

        raw_variant_count = payload.get("variant_count", 0)
        try:
            parsed_variant_count = int(raw_variant_count)
        except (TypeError, ValueError):
            parsed_variant_count = 0

        raw_follow_up = payload.get("follow_up_suggestion")
        return ChatResponse(
            answer=str(payload.get("answer", "")),
            ranked_results=ranked_results,
            lookup_mode=str(payload.get("lookup_mode", "hybrid_fallback")),
            variant_count=parsed_variant_count,
            retrieval_mode=str(payload.get("retrieval_mode", "deterministic")),
            answer_mode=str(payload.get("answer_mode", "direct_answer")),
            needs_clarification=bool(payload.get("needs_clarification", False)),
            confidence_score=parsed_confidence,
            embedding_cache_hit=bool(payload.get("embedding_cache_hit", False)),
            answer_cache_hit=True,
            follow_up_suggestion=str(raw_follow_up) if raw_follow_up else None,
        )

    def _should_use_llm(
        self,
        *,
        llm_requested: bool,
        query: str,
        diagnostics: RetrievalDiagnostics,
        confidence: float,
    ) -> bool:
        """Applies RAG confidence/gap logic to decide if synthesis is needed.

        A confidence floor of 0.25 prevents sending noisy/irrelevant records to
        the LLM.  Below that threshold the retrieved context is too weak for
        meaningful synthesis and a deterministic answer (with a low-confidence
        notice) is safer and more honest than an LLM-generated response.
        """
        if not llm_requested:
            return False
        if not self.rag_enabled:
            return True

        explicit_synthesis = self._is_synthesis_intent(query)
        ambiguous = diagnostics.score_gap < self.rag_score_gap_threshold
        high_confidence = confidence >= self.rag_confidence_threshold and not ambiguous

        # Confidence floor: never synthesize when retrieval quality is too low.
        # The LLM cannot produce a trustworthy answer from weakly-matched records.
        _CONFIDENCE_FLOOR = 0.25
        if confidence < _CONFIDENCE_FLOOR and not explicit_synthesis:
            return False

        if explicit_synthesis:
            return True
        return not high_confidence

    def _confidence_score(self, diagnostics: RetrievalDiagnostics) -> float:
        """Converts retrieval diagnostics top score into bounded confidence."""
        return max(0.0, min(1.0, diagnostics.top_score))

    def _is_synthesis_intent(self, query: str) -> bool:
        """Detects explicit summarize/analysis intent in user query."""
        lowered = query.lower()
        keywords = (
            "summarize",
            "summary",
            "explain",
            "insight",
            "pattern",
            "compare",
            "trend",
            "why",
            "analysis",
            "what are",
            "tell me about",
            "describe",
            "overview",
            "breakdown",
            "how many",
            "which surveys",
            "across surveys",
            "over time",
            "difference between",
        )
        return any(keyword in lowered for keyword in keywords)

    def _parse_follow_up_suggestion(self, llm_text: str) -> str | None:
        """Extracts the 'Suggested Follow-up:' line from a structured LLM response.

        Looks for the heading injected by the system prompt.  Returns None when
        the heading is absent (e.g. the model deviated from the format).
        """
        for line in llm_text.splitlines():
            stripped = line.strip()
            # Match bold heading variants the model might produce
            for prefix in (
                "**Suggested Follow-up:**",
                "**Suggested Follow-up**:",
                "Suggested Follow-up:",
            ):
                if stripped.startswith(prefix):
                    suggestion = stripped[len(prefix):].strip()
                    # Strip surrounding quotes if the model added them
                    suggestion = suggestion.strip('"').strip("'").strip()
                    return suggestion if suggestion else None
        return None

    def _deterministic_answer(
        self,
        *,
        query: str,
        cards: list[QuestionRecord],
        llm_provider: str,
    ) -> str:
        """Builds grounded non-LLM fallback answer from top cards."""
        top = cards[0]
        lines: list[str] = [
            f"**Answer:** {top.question_id} — {top.question_text}",
            f"Survey: {top.survey_name} | Wave: {top.wave_year}",
        ]

        if len(cards) > 1:
            lines.append("\n**Key Details:**")
            for item in cards[1:4]:
                lines.append(f"- {item.question_id} — {item.question_text}")

        if llm_provider.lower() not in {"chatgpt", "openai"}:
            lines.append(f"\n_{llm_provider.title()} is not yet supported; showing deterministic results._")

        if "allowable" in query.lower() or "options" in query.lower():
            formatted_values = self._format_value_labels(top.value_labels)
            if formatted_values:
                lines.append("\n**Allowed Values:**")
                lines.extend([f"- {value}" for value in formatted_values])

        return "\n".join(lines)

    def _conversation_context_block(self, conversation_context: list[dict[str, str]]) -> str:
        """Formats recent conversation turns into a compact prompt context block."""
        if not conversation_context:
            return ""
        lines: list[str] = []
        for turn in conversation_context[-20:]:
            role = str(turn.get("role", "")).strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
        if not lines:
            return ""
        return "Recent conversation context:\n" + "\n".join(lines) + "\n\n"

    def _answer_cache_key(
        self,
        *,
        query: str,
        survey_name: str | None,
        wave_year: str | None,
        topic_label: str | None,
        topic_source_type: str | None,
        cards: list[QuestionRecord],
        retrieval_mode: str,
        llm_model: str | None,
        conversation_context: list[dict[str, str]],  # kept for signature compatibility
    ) -> str:
        """Creates deterministic key for answer-level cache.

        Conversation context is intentionally excluded: answers are grounded
        in retrieved records, not conversation history, so the same records +
        query + model should return the same cached answer regardless of prior
        turns.  Including context_fp caused a 100% cache-miss rate.
        """
        top_ids = [item.question_id for item in cards[:8]]
        top_ids_fingerprint = hashlib.sha256("|".join(top_ids).encode("utf-8")).hexdigest()
        payload = {
            "q": _normalize_query(query),
            "filters": {
                "survey_name": survey_name,
                "wave_year": wave_year,
                "topic_label": topic_label,
                "topic_source_type": topic_source_type,
            },
            "top_ids_fp": top_ids_fingerprint,
            "retrieval_mode": retrieval_mode,
            "llm_model": llm_model or "",
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _exact_question_lookup_response(
        self,
        *,
        query: str,
        survey_name: str | None,
        wave_year: str | None,
        topic_label: str | None,
        topic_source_type: str | None,
    ) -> ChatResponse | None:
        """Executes deterministic survey/question variant lookup (including alias variants) before fuzzy search."""
        question_ref = self._extract_question_ref(query)
        if question_ref is None:
            return None

        query_survey = self._extract_survey_name(query)
        resolved_survey = (survey_name or query_survey or "").strip().upper()
        if not resolved_survey:
            return None

        parsed_ref = self._split_question_ref(question_ref)
        if parsed_ref is None:
            return None
        ref_number, ref_suffix = parsed_ref

        filtered_records = apply_filters(
            self.retriever.records,
            survey_name=resolved_survey,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        variant_map = self._build_variant_map(filtered_records, ref_number)
        if not variant_map:
            return None

        if not ref_suffix and len(variant_map) > 1:
            variant_keys = list(variant_map.keys())
            options_text = ", ".join(f"`{variant}`" for variant in variant_keys)
            suggested_variant = next(
                (variant for variant in variant_keys if len(variant) > len(ref_number)),
                variant_keys[0],
            )
            detail_target = "exact context fields" if self._extract_requested_context_fields(query) else "exact dropdown values"
            answer = (
                f"I found multiple variants for **{resolved_survey} question {ref_number}**: {options_text}. "
                f"Please choose one variant (for example, `{suggested_variant}`) and I will return the {detail_target}."
            )
            ranked_results = [self._best_variant_record(variant_map[key]) for key in variant_keys]
            return ChatResponse(
                answer=answer,
                ranked_results=ranked_results[: CHAT_RULES.max_cards_display],
                lookup_mode="exact_id",
                variant_count=len(variant_keys),
                retrieval_mode="deterministic",
                answer_mode="clarifier",
                needs_clarification=True,
            )

        if not ref_suffix and len(variant_map) == 1:
            selected_key = next(iter(variant_map.keys()))
            selected_records = variant_map[selected_key]
        else:
            selected_key = f"{ref_number}{ref_suffix}"
            selected_records = variant_map.get(selected_key)
        if not selected_records:
            return None

        return self._build_specific_variant_response(
            query=query,
            survey_name=resolved_survey,
            variant_key=selected_key,
            records=selected_records,
        )

    def _quick_allowed_values_response(
        self,
        *,
        query: str,
        survey_name: str | None,
        wave_year: str | None,
        topic_label: str | None,
        topic_source_type: str | None,
    ) -> ChatResponse | None:
        """Handles allowed-values intent with direct value-label answers."""
        if not self._is_allowed_values_intent(query):
            return None

        query_survey = self._extract_survey_name(query)
        question_ref = self._extract_question_ref(query)
        resolved_survey = survey_name or query_survey

        filtered_records = apply_filters(
            self.retriever.records,
            survey_name=resolved_survey,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        question_matches = self._match_question_ref_records(filtered_records, question_ref)

        if not question_matches and question_ref and resolved_survey:
            hinted_matches = self._hinted_question_ref_records(
                survey_name=resolved_survey,
                question_ref=question_ref,
                wave_year=wave_year,
                topic_label=topic_label,
                topic_source_type=topic_source_type,
                query=query,
            )
            question_matches = hinted_matches

        ranked_results = question_matches[: CHAT_RULES.max_cards_display]
        if not ranked_results:
            return None

        records_with_values = [record for record in ranked_results if record.value_labels]
        if not records_with_values:
            first = ranked_results[0]
            answer = _sanitize_answer(
                f"I found {first.question_id}, but it does not have coded dropdown/allowed values in the dictionary. "
                "Try asking for a question that has response options."
            )
            return ChatResponse(
                answer=answer,
                ranked_results=ranked_results,
                lookup_mode="exact_id",
                variant_count=max(1, len(ranked_results)),
                retrieval_mode="deterministic",
                answer_mode="allowed_values",
                needs_clarification=False,
            )

        if len(records_with_values) == 1:
            record = records_with_values[0]
            formatted_values = "\n".join(f"- {item}" for item in self._format_value_labels(record.value_labels))
            answer = _sanitize_answer(
                f"For **{record.question_id}** ({record.question_text}), the allowable response options are:\n"
                f"{formatted_values}"
            )
            return ChatResponse(
                answer=answer,
                ranked_results=ranked_results,
                lookup_mode="exact_id",
                variant_count=1,
                retrieval_mode="deterministic",
                answer_mode="allowed_values",
                needs_clarification=False,
            )

        sections: list[str] = []
        for record in records_with_values[:4]:
            formatted_values = "\n".join(f"  - {item}" for item in self._format_value_labels(record.value_labels))
            sections.append(
                f"- **{record.question_id}** ({record.question_text})\n{formatted_values}"
            )

        survey_note = f" in **{resolved_survey}**" if resolved_survey else ""
        answer = _sanitize_answer(
            f"I found multiple Question {question_ref[1:] if question_ref else ''} variants{survey_note}. "
            "Here are the allowable options for each:\n"
            + "\n".join(sections)
            + "\n\nAsk for one specifically (for example, `q5a` or `q5b`) to get a focused answer."
        )
        return ChatResponse(
            answer=answer,
            ranked_results=ranked_results,
            lookup_mode="exact_id",
            variant_count=max(1, len(ranked_results)),
            retrieval_mode="deterministic",
            answer_mode="clarifier",
            needs_clarification=True,
        )

    def _is_allowed_values_intent(self, query: str) -> bool:
        """Detects dropdown/options-related intent keywords."""
        lowered = query.lower()
        keywords = (
            "dropdown",
            "drop down",
            "allowable",
            "alloweable",
            "allowed",
            "options",
            "choices",
            "valid responses",
            "response options",
            "what can i select",
            "fields allowable",
        )
        return any(keyword in lowered for keyword in keywords)

    def _build_did_you_mean(self, query: str) -> list[str]:
        """Run a broad (unfiltered) search and return up to 3 question texts as suggestions."""
        try:
            broad = self.retriever.search_with_details(query=query)
            return [
                item.record.question_text
                for item in broad.scored_results[:3]
                if item.record.question_text
            ]
        except Exception:  # noqa: BLE001
            return []

    def _is_comparison_intent(self, query: str) -> bool:
        """Detects cross-survey comparison intent (e.g. 'compare PMG20 vs PMG22')."""
        lowered = query.lower()
        return bool(
            re.search(r"\bvs\.?\b|\bversus\b|\bcompare\b.*\bvs\b|\bcompare\b.*\band\b", lowered)
        )

    def _extract_two_surveys(self, query: str) -> tuple[str, str] | None:
        """Returns the first two distinct survey tokens found in a query, or None."""
        tokens = re.findall(r"\b[A-Za-z]{3}\d{2}_[A-Za-z]{3}\b", query, re.IGNORECASE)
        unique = list(dict.fromkeys(t.upper() for t in tokens))
        if len(unique) >= 2:
            return unique[0], unique[1]
        return None

    def _comparison_response(
        self,
        query: str,
        survey_a: str,
        survey_b: str,
        wave_year: str | None,
        topic_label: str | None,
        topic_source_type: str | None,
    ) -> ChatResponse:
        """Run two searches and format a side-by-side comparison table."""
        def _top_text(survey: str) -> str:
            search = self.retriever.search_with_details(
                query=query,
                survey_name=survey,
                wave_year=wave_year,
                topic_label=topic_label,
                topic_source_type=topic_source_type,
            )
            results = search.scored_results[:3]
            if not results:
                return "_No matching records_"
            lines = []
            for item in results:
                r = item.record
                lines.append(f"**{r.question_id}** — {r.question_text}")
                if r.value_labels:
                    lines.append("  Options: " + " | ".join(r.value_labels[:5]))
            return "\n".join(lines)

        text_a = _top_text(survey_a)
        text_b = _top_text(survey_b)
        answer = _sanitize_answer(
            f"**Cross-Survey Comparison: {survey_a} vs {survey_b}**\n\n"
            f"**{survey_a}**\n{text_a}\n\n"
            f"**{survey_b}**\n{text_b}"
        )
        all_records = []
        for survey in (survey_a, survey_b):
            s = self.retriever.search_with_details(query=query, survey_name=survey)
            all_records.extend(r.record for r in s.scored_results[:3])
        return ChatResponse(
            answer=answer,
            ranked_results=all_records,
            retrieval_mode="deterministic",
            answer_mode="comparison",
            confidence_score=None,
            embedding_cache_hit=False,
            answer_cache_hit=False,
        )

    def _is_lineage_intent(self, query: str) -> bool:
        """Detects lineage/evolution intent for question changes over waves/projects."""
        lowered = query.lower()
        keywords = (
            "lineage",
            "changed over time",
            "change over time",
            "across waves",
            "across projects",
            "historical",
            "evolution",
            "how has",
            "over time",
        )
        return any(keyword in lowered for keyword in keywords)

    def _lineage_response(
        self,
        *,
        query: str,
        survey_name: str | None,
        wave_year: str | None,
        topic_label: str | None,
        topic_source_type: str | None,
    ) -> ChatResponse | None:
        """Builds deterministic timeline showing question wording/value-label evolution."""
        if not self._is_lineage_intent(query):
            return None

        question_ref = self._extract_question_ref(query)
        if question_ref is None:
            return ChatResponse(
                answer=(
                    "I can show lineage once you specify a question reference "
                    "(for example, `question 5`, `q5a`, or `PMG19_GAM_q5`)."
                ),
                ranked_results=[],
                lookup_mode="exact_id",
                variant_count=0,
                retrieval_mode="deterministic",
                answer_mode="clarifier",
                needs_clarification=True,
            )

        query_survey = self._extract_survey_name(query)
        resolved_survey = survey_name or query_survey
        filtered_records = apply_filters(
            self.retriever.records,
            survey_name=resolved_survey,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        matches = self._match_question_ref_records(filtered_records, question_ref)
        if not matches and question_ref and resolved_survey:
            matches = self._hinted_question_ref_records(
                survey_name=resolved_survey,
                question_ref=question_ref,
                wave_year=wave_year,
                topic_label=topic_label,
                topic_source_type=topic_source_type,
                query=query,
            )
        if not matches:
            return ChatResponse(
                answer=(
                    f"I could not find lineage records for **{question_ref}** with the current filters. "
                    "Try clearing filters or specifying a survey token (for example, `PMG19_GAM`)."
                ),
                ranked_results=[],
                lookup_mode="exact_id",
                variant_count=0,
                retrieval_mode="deterministic",
                answer_mode="lineage",
                needs_clarification=False,
            )

        groups: dict[tuple[str, str], list[QuestionRecord]] = {}
        for record in matches:
            groups.setdefault((record.survey_name, record.wave_year), []).append(record)

        timeline = sorted(
            (
                (survey, wave, self._best_variant_record(group_records), group_records)
                for (survey, wave), group_records in groups.items()
            ),
            key=lambda item: (item[0], self._wave_sort_key(item[1])),
        )
        if not timeline:
            return None

        parsed_ref = self._split_question_ref(question_ref)
        suffix = parsed_ref[1] if parsed_ref is not None else ""
        if not suffix:
            variants = sorted(
                {
                    self._question_component(record.question_id) or ""
                    for record in matches
                    if self._question_component(record.question_id)
                },
                key=self._variant_sort_key,
            )
        else:
            variants = []

        lines: list[str] = [
            f"Lineage for **{question_ref}** across {len(timeline)} survey-wave point(s):",
        ]
        if variants:
            variant_labels = ", ".join(f"`q{variant}`" for variant in variants[:8])
            lines.append(f"Included variants: {variant_labels}")

        previous_record: QuestionRecord | None = None
        for survey, wave, record, _group_records in timeline[:12]:
            line = f"- {survey} {wave} (`{record.question_id}`): {record.question_text}"
            changes: list[str] = []
            if previous_record is not None:
                if record.question_text.strip().lower() != previous_record.question_text.strip().lower():
                    changes.append("wording changed")
                previous_values = set(self._format_value_labels(previous_record.value_labels))
                current_values = set(self._format_value_labels(record.value_labels))
                added = sorted(current_values - previous_values)
                removed = sorted(previous_values - current_values)
                if added:
                    changes.append(f"+{len(added)} response option(s)")
                if removed:
                    changes.append(f"-{len(removed)} response option(s)")
            if changes:
                line += f" ({'; '.join(changes)})"
            lines.append(line)
            previous_record = record

        if len(timeline) > 12:
            lines.append(f"... plus {len(timeline) - 12} more records. Refine filters to narrow lineage view.")

        ranked_results = [item[2] for item in timeline[: CHAT_RULES.max_cards_display]]
        return ChatResponse(
            answer="\n".join(lines),
            ranked_results=ranked_results,
            lookup_mode="exact_id",
            variant_count=max(1, len(variants) if variants else 1),
            retrieval_mode="deterministic",
            answer_mode="lineage",
            needs_clarification=False,
        )

    def _extract_survey_name(self, query: str) -> str | None:
        """Extracts survey token like `PMG20_GAM` from free text."""
        id_match = re.search(r"\b([A-Za-z]{3}\d{2}_[A-Za-z]{3})(?=_q{1,2})", query, flags=re.IGNORECASE)
        if id_match:
            return id_match.group(1).upper()

        match = re.search(r"\b([A-Za-z]{3}\d{2}_[A-Za-z]{3})\b", query)
        if match:
            return match.group(1).upper()

        relaxed_match = re.search(r"\b([A-Za-z]{3})[\s_-]?(\d{2})[\s_-]?([A-Za-z]{3})\b", query)
        if not relaxed_match:
            return None
        return f"{relaxed_match.group(1).upper()}{relaxed_match.group(2)}_{relaxed_match.group(3).upper()}"

    def _wave_sort_key(self, wave_year: str) -> tuple[int, str]:
        """Sorts wave labels chronologically when numeric year is present."""
        normalized = wave_year.strip()
        match = re.search(r"(19|20)\d{2}", normalized)
        if match:
            return int(match.group(0)), normalized
        return 10_000, normalized

    def _extract_question_ref(self, query: str) -> str | None:
        """Extracts question reference token (supports `q` and `qq` ID forms)."""
        id_match = re.search(r"\b[A-Za-z]{3}\d{2}_[A-Za-z]{3}_q{1,2}(\d+(?:\.\d+)?[a-z]?)\b", query, flags=re.IGNORECASE)
        if id_match:
            return f"q{id_match.group(1).lower()}"

        q_match = re.search(r"\bq(?:uestion)?\s*(\d+(?:\.\d+)?[a-z]?)\b", query, flags=re.IGNORECASE)
        if not q_match:
            return None
        return f"q{q_match.group(1).lower()}"

    def _match_question_ref_records(self, records: list[QuestionRecord], question_ref: str | None) -> list[QuestionRecord]:
        """Filters records that match extracted question reference."""
        if not question_ref:
            return []

        parsed_ref = self._split_question_ref(question_ref)
        if parsed_ref is None:
            return []
        ref_num, ref_suffix = parsed_ref

        matches: list[QuestionRecord] = []
        for record in records:
            q_component = self._question_component(record.question_id)
            if not q_component:
                continue
            parsed_component = self._split_question_component(q_component)
            if parsed_component is None:
                continue
            comp_num, comp_suffix = parsed_component
            if comp_num != ref_num:
                continue
            if ref_suffix and comp_suffix != ref_suffix:
                continue
            matches.append(record)

        return sorted(matches, key=lambda item: item.question_id)

    def _question_component(self, question_id: str) -> str | None:
        """Extracts question number/suffix component from variable ID."""
        match = re.search(r"_q{1,2}(\d+(?:\.\d+)?[a-z]?)\b", question_id.lower())
        if not match:
            return None
        return match.group(1)

    def _split_question_ref(self, question_ref: str) -> tuple[str, str] | None:
        """Splits normalized question ref into number/suffix tuple."""
        match = re.match(r"^q(\d+(?:\.\d+)?)([a-z]?)$", question_ref.strip().lower())
        if not match:
            return None
        return self._normalize_question_number(match.group(1)), match.group(2)

    def _split_question_component(self, component: str) -> tuple[str, str] | None:
        """Splits extracted component into number/suffix tuple."""
        match = re.match(r"^(\d+(?:\.\d+)?)([a-z]?)$", component.strip().lower())
        if not match:
            return None
        return self._normalize_question_number(match.group(1)), match.group(2)

    def _normalize_question_number(self, number: str) -> str:
        """Normalizes numeric component (e.g., strips leading zeros)."""
        segments = [segment for segment in number.strip().split(".") if segment != ""]
        if not segments:
            return "0"
        normalized_segments = [(segment.lstrip("0") or "0") for segment in segments]
        return ".".join(normalized_segments)

    def _build_variant_map(self, records: list[QuestionRecord], ref_number: str) -> dict[str, list[QuestionRecord]]:
        """Groups matching records into variant buckets (`11a`, `11b`, etc.)."""
        variant_map: dict[str, list[QuestionRecord]] = {}
        for record in records:
            component = self._question_component(record.question_id)
            if component is None:
                continue
            parsed_component = self._split_question_component(component)
            if parsed_component is None:
                continue
            comp_number, comp_suffix = parsed_component
            if comp_number != ref_number:
                continue
            variant_key = f"{comp_number}{comp_suffix}"
            variant_map.setdefault(variant_key, []).append(record)

        ordered_keys = sorted(variant_map.keys(), key=self._variant_sort_key)
        return {key: sorted(variant_map[key], key=self._record_sort_key) for key in ordered_keys}

    def _variant_sort_key(self, key: str) -> tuple[tuple[int, ...], str]:
        """Defines deterministic variant sort order."""
        match = re.match(r"^(\d+(?:\.\d+)?)([a-z]?)$", key.lower())
        if not match:
            return ((10_000,), key.lower())
        numeric_parts = tuple(int(part) for part in match.group(1).split("."))
        return (numeric_parts, match.group(2))

    def _record_sort_key(self, record: QuestionRecord) -> tuple[int, str]:
        """Prioritizes canonical IDs over alias IDs when selecting representative record."""
        alias_penalty = 1 if "_qq" in record.question_id.lower() else 0
        return (alias_penalty, record.question_id.lower())

    def _best_variant_record(self, records: list[QuestionRecord]) -> QuestionRecord:
        """Chooses best representative record for a variant bucket."""
        records_with_values = [record for record in records if record.value_labels]
        candidate_pool = records_with_values or records
        return sorted(candidate_pool, key=self._record_sort_key)[0]

    def _build_specific_variant_response(
        self,
        *,
        query: str,
        survey_name: str,
        variant_key: str,
        records: list[QuestionRecord],
    ) -> ChatResponse:
        """Builds final answer for a specific variant, merging value-label aliases."""
        requested_context_fields = self._extract_requested_context_fields(query)
        if requested_context_fields:
            return self._build_context_field_response(
                survey_name=survey_name,
                variant_key=variant_key,
                records=records,
                requested_fields=requested_context_fields,
            )

        records_with_values = [record for record in records if record.value_labels]
        candidate_pool = records_with_values or records
        primary = sorted(candidate_pool, key=self._record_sort_key)[0]
        merged_values = self._merge_value_labels(records_with_values)

        question_label = f"{survey_name} question {variant_key}"
        if merged_values:
            values_text = "\n".join(f"- {item}" for item in merged_values)
            raw = (
                f"For **{question_label}** ({primary.question_text}), the allowable response options are:\n"
                f"{values_text}"
            )
        else:
            raw = (
                f"I found **{question_label}** ({primary.question_text}), but there are no coded dropdown/allowed values "
                "for this variant in the current dictionary."
            )

        matched_ids = sorted({record.question_id for record in records})
        if len(matched_ids) > 1:
            ids_text = ", ".join(f"`{question_id}`" for question_id in matched_ids)
            raw += f"\n\nMatched variable IDs: {ids_text}"
        answer = _sanitize_answer(raw)

        ranked_results = sorted(candidate_pool, key=self._record_sort_key)[: CHAT_RULES.max_cards_display]
        return ChatResponse(
            answer=answer,
            ranked_results=ranked_results,
            lookup_mode="exact_id",
            variant_count=1,
            retrieval_mode="deterministic",
            answer_mode="allowed_values",
            needs_clarification=False,
        )

    def _extract_requested_context_fields(self, query: str) -> list[str]:
        """Extracts requested metadata fields (for example `position`, `label`) from query text."""
        lowered = query.lower()
        requested: list[str] = []

        def add(field_name: str) -> None:
            """Append a field name once while preserving query-order intent."""
            if field_name not in requested:
                requested.append(field_name)

        if "measurement level" in lowered or "measurement label" in lowered:
            add("measurement_level")
        elif re.search(r"\bmeasurement\b", lowered):
            add("measurement_level")

        if re.search(r"\bposition\b", lowered):
            add("position")
        if re.search(r"\brole\b", lowered):
            add("role")
        if re.search(r"\bcolumn\s+width\b", lowered):
            add("column_width")
        if re.search(r"\balignment\b", lowered):
            add("alignment")
        if re.search(r"\bprint\s+format\b", lowered):
            add("print_format")
        if re.search(r"\bwrite\s+format\b", lowered):
            add("write_format")
        if re.search(r"\bmissing\s+values?\b", lowered):
            add("missing_values")
        if re.search(r"\b(variable|question id)\b", lowered):
            add("variable")
        if re.search(r"\blabel\b", lowered) and "measurement label" not in lowered:
            add("label")

        if not requested and "context" in lowered:
            requested.extend(_DEFAULT_CONTEXT_FIELDS)
        return requested

    def _build_context_field_response(
        self,
        *,
        survey_name: str,
        variant_key: str,
        records: list[QuestionRecord],
        requested_fields: list[str],
    ) -> ChatResponse:
        """Builds exact context-field response for a resolved survey question variant."""
        primary = sorted(records, key=self._record_sort_key)[0]
        question_label = f"{survey_name} question {variant_key}"
        lines = [f"For **{question_label}** (`{primary.question_id}`), the exact dictionary context is:"]

        for field_name in requested_fields:
            display = _CONTEXT_FIELD_DISPLAY.get(field_name, field_name.replace("_", " ").title())
            value = self._resolve_context_field_value(records, field_name)
            if value:
                lines.append(f"- {display}: {value}")
            else:
                lines.append(f"- {display}: Not available in the current dictionary")

        matched_ids = sorted({record.question_id for record in records})
        if len(matched_ids) > 1:
            ids_text = ", ".join(f"`{question_id}`" for question_id in matched_ids)
            lines.append(f"Matched variable IDs: {ids_text}")

        ranked_results = sorted(records, key=self._record_sort_key)[: CHAT_RULES.max_cards_display]
        return ChatResponse(
            answer="\n".join(lines),
            ranked_results=ranked_results,
            lookup_mode="exact_id",
            variant_count=1,
            retrieval_mode="deterministic",
            answer_mode="metadata_lookup",
            needs_clarification=False,
        )

    def _resolve_context_field_value(self, records: list[QuestionRecord], field_name: str) -> str:
        """Returns stable context-field value across aliases for a specific question variant."""
        values: list[str] = []
        seen: set[str] = set()
        for record in sorted(records, key=self._record_sort_key):
            if field_name == "variable":
                value = record.question_id.strip()
            elif field_name == "label":
                value = record.question_text.strip()
            elif field_name == "measurement_level":
                value = record.measurement_level.strip()
            elif field_name == "role":
                value = record.role.strip()
            else:
                context_fields = getattr(record, "context_fields", {}) or {}
                if not isinstance(context_fields, dict):
                    context_fields = {}
                raw_value = context_fields.get(field_name, "")
                value = str(raw_value).strip() if raw_value is not None else ""
            if not value:
                continue
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            values.append(value)

        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        return " | ".join(values)

    def _merge_value_labels(self, records: list[QuestionRecord]) -> list[str]:
        """Deduplicates/merges formatted value labels across records."""
        merged: list[str] = []
        seen: set[str] = set()
        for record in records:
            for item in self._format_value_labels(record.value_labels):
                key = item.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    def _hinted_question_ref_records(
        self,
        *,
        survey_name: str,
        question_ref: str,
        wave_year: str | None,
        topic_label: str | None,
        topic_source_type: str | None,
        query: str,
    ) -> list[QuestionRecord]:
        """Uses docx-derived hints to improve question-reference retrieval fallback."""
        survey_hints = self.doc_question_hints.get(survey_name.upper(), {})
        hint_lines = list(survey_hints.get(question_ref, []))
        if not hint_lines and question_ref:
            hint_lines = [line for ref, lines in survey_hints.items() if ref.startswith(question_ref) for line in lines]
        if not hint_lines:
            return []

        hint_context = " ".join(hint_lines[:3])
        hinted_ranked = self.retriever.search(
            query=f"{query}\n{hint_context}",
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
            top_k=24,
        )
        return self._match_question_ref_records(hinted_ranked, question_ref)

    def _format_value_labels(self, value_labels: list[str]) -> list[str]:
        """Normalizes value-label formatting (including numeric cleanup)."""
        out: list[str] = []
        for item in value_labels:
            cleaned = item.strip()
            if ":" not in cleaned:
                out.append(cleaned)
                continue
            code, label = cleaned.split(":", 1)
            code = code.strip()
            label = label.strip()
            if code.endswith(".0"):
                code = code[:-2]
            out.append(f"{code}: {label}")
        return out


def _sanitize_answer(text: str) -> str:
    """Cleans answer text before it is stored or shown to users.

    Removes or normalises common artefacts that appear in raw LLM output or
    templated deterministic strings:

    * Strips leading/trailing whitespace.
    * Collapses runs of 3+ blank lines to a maximum of 2 (one blank line
      separating paragraphs is fine; more creates excessive padding in the UI).
    * Removes stray literal ``\\n`` sequences that survive JSON round-trips or
      string concatenation errors (e.g. a line that reads ``"text \\n more"``).
    * Normalises Windows-style ``\\r\\n`` line endings to ``\\n``.
    * Strips trailing whitespace from every line.

    Parameters
    ----------
    text:
        Raw answer string from LLM synthesis or a deterministic builder.

    Returns
    -------
    str
        Cleaned answer string safe for Markdown rendering.
    """
    if not text:
        return text
    # Normalise CRLF → LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove literal backslash-n escape sequences that sometimes leak through
    # JSON serialisation or string interpolation bugs (e.g. "text \\n more").
    text = re.sub(r"(?<!\\)\\n", "\n", text)
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    # Collapse 3+ consecutive blank lines → 2
    cleaned: list[str] = []
    blank_run = 0
    for line in lines:
        if line == "":
            blank_run += 1
            if blank_run <= 2:
                cleaned.append(line)
        else:
            blank_run = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def _normalize_query(query: str) -> str:
    """Module helper to normalize query strings for cache keys."""
    return " ".join(query.lower().split())


def _env_bool(name: str, *, default: bool) -> bool:
    """Reads boolean env variable with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, *, default: float) -> float:
    """Reads float env variable with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, *, default: int) -> int:
    """Reads integer env variable with fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
