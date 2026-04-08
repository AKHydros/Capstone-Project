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


@dataclass
class ChatbotService:
    retriever: HybridRetriever
    llm_client: OpenAIChatClient
    doc_question_hints: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    rag_enabled_override: bool | None = None
    rag_confidence_threshold_override: float | None = None
    rag_score_gap_threshold_override: float | None = None
    rag_answer_cache_ttl_override: int | None = None
    answer_cache: TTLCache[dict[str, str]] = field(init=False)
    rag_enabled: bool = field(init=False)
    rag_confidence_threshold: float = field(init=False)
    rag_score_gap_threshold: float = field(init=False)

    def __post_init__(self) -> None:
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
    ) -> ChatResponse:
        exact_response = self._exact_question_lookup_response(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        if exact_response is not None:
            return exact_response

        quick_response = self._quick_allowed_values_response(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        if quick_response is not None:
            return quick_response

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
            return ChatResponse(
                answer="I could not find grounded matches in the current Excel dictionary. Try broader wording or relax filters.",
                ranked_results=[],
                retrieval_mode="deterministic",
                confidence_score=confidence,
                embedding_cache_hit=embedding_cache_hit,
                answer_cache_hit=False,
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
        )
        cached_answer = self.answer_cache.get(answer_cache_key)
        if cached_answer is not None:
            return ChatResponse(
                answer=str(cached_answer.get("answer", "")),
                ranked_results=cards,
                lookup_mode="hybrid_fallback",
                variant_count=0,
                retrieval_mode=str(cached_answer.get("retrieval_mode", retrieval_mode)),
                confidence_score=confidence,
                embedding_cache_hit=embedding_cache_hit,
                answer_cache_hit=True,
            )

        if should_use_llm:
            context = build_grounded_context(cards)
            system_prompt = (
                "You are a grounded research dictionary assistant. "
                "Be conversational, direct, and quick to read. "
                "Lead with a short answer, then add concise supporting details. "
                "Only summarize retrieved records. Never invent question text, IDs, or surveys. "
                "If uncertain, say so plainly."
            )
            user_prompt = (
                f"User query: {query}\n\n"
                f"Retrieved records:\n{context}\n\n"
                "Return: (1) direct answer, (2) notable patterns, (3) 1 suggested follow-up query. "
                "If the user asks for response options or allowable values, list the coded values clearly."
            )
            max_length = int(inference["max_length"]) if inference and "max_length" in inference else None
            top_p = float(inference["top_p"]) if inference and "top_p" in inference else None
            temperature = float(inference["temperature"]) if inference and "temperature" in inference else None
            answer = self.llm_client.summarize(
                system_prompt,
                user_prompt,
                model=llm_model,
                max_output_tokens=max_length,
                top_p=top_p,
                temperature=temperature,
            )
        else:
            answer = self._deterministic_answer(query=query, cards=cards, llm_provider=llm_provider)

        self.answer_cache.set(
            answer_cache_key,
            {
                "answer": answer,
                "retrieval_mode": retrieval_mode,
            },
        )

        return ChatResponse(
            answer=answer,
            ranked_results=cards,
            lookup_mode="hybrid_fallback",
            variant_count=0,
            retrieval_mode=retrieval_mode,
            confidence_score=confidence,
            embedding_cache_hit=embedding_cache_hit,
            answer_cache_hit=False,
        )

    def answer_cache_stats(self) -> CacheStats:
        return self.answer_cache.stats()

    def _should_use_llm(
        self,
        *,
        llm_requested: bool,
        query: str,
        diagnostics: RetrievalDiagnostics,
        confidence: float,
    ) -> bool:
        if not llm_requested:
            return False
        if not self.rag_enabled:
            return True

        explicit_synthesis = self._is_synthesis_intent(query)
        ambiguous = diagnostics.score_gap < self.rag_score_gap_threshold
        high_confidence = confidence >= self.rag_confidence_threshold and not ambiguous

        if explicit_synthesis:
            return True
        return not high_confidence

    def _confidence_score(self, diagnostics: RetrievalDiagnostics) -> float:
        return max(0.0, min(1.0, diagnostics.top_score))

    def _is_synthesis_intent(self, query: str) -> bool:
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
        )
        return any(keyword in lowered for keyword in keywords)

    def _deterministic_answer(
        self,
        *,
        query: str,
        cards: list[QuestionRecord],
        llm_provider: str,
    ) -> str:
        top = cards[0]
        lines: list[str] = [
            f"Top grounded match: {top.question_id} - {top.question_text}.",
            f"Survey: {top.survey_name} | Wave: {top.wave_year}.",
        ]

        if len(cards) > 1:
            lines.append("Other relevant grounded matches:")
            for item in cards[1:4]:
                lines.append(f"- {item.question_id} - {item.question_text}")

        if llm_provider.lower() not in {"chatgpt", "openai"}:
            lines.append(f"{llm_provider.title()} is not implemented yet; using deterministic summary.")

        if "allowable" in query.lower() or "options" in query.lower():
            formatted_values = self._format_value_labels(top.value_labels)
            if formatted_values:
                lines.append("Allowed values for the top match:")
                lines.extend([f"- {value}" for value in formatted_values])

        return "\n".join(lines)

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
    ) -> str:
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
            answer = (
                f"I found multiple variants for **{resolved_survey} question {ref_number}**: {options_text}. "
                f"Please choose one variant (for example, `{suggested_variant}`) and I will return the exact dropdown values."
            )
            ranked_results = [self._best_variant_record(variant_map[key]) for key in variant_keys]
            return ChatResponse(
                answer=answer,
                ranked_results=ranked_results[: CHAT_RULES.max_cards_display],
                lookup_mode="exact_id",
                variant_count=len(variant_keys),
                retrieval_mode="deterministic",
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
            answer = (
                f"I found {first.question_id}, but it does not have coded dropdown/allowed values in the dictionary. "
                "Try asking for a question that has response options."
            )
            return ChatResponse(
                answer=answer,
                ranked_results=ranked_results,
                lookup_mode="exact_id",
                variant_count=max(1, len(ranked_results)),
                retrieval_mode="deterministic",
            )

        if len(records_with_values) == 1:
            record = records_with_values[0]
            formatted_values = "\n".join(f"- {item}" for item in self._format_value_labels(record.value_labels))
            answer = (
                f"Absolutely. For **{record.question_id}** ({record.question_text}), the allowable response options are:\n"
                f"{formatted_values}"
            )
            return ChatResponse(
                answer=answer,
                ranked_results=ranked_results,
                lookup_mode="exact_id",
                variant_count=1,
                retrieval_mode="deterministic",
            )

        sections: list[str] = []
        for record in records_with_values[:4]:
            formatted_values = "\n".join(f"  - {item}" for item in self._format_value_labels(record.value_labels))
            sections.append(
                f"- **{record.question_id}** ({record.question_text})\n{formatted_values}"
            )

        survey_note = f" in **{resolved_survey}**" if resolved_survey else ""
        answer = (
            f"I found multiple Question {question_ref[1:] if question_ref else ''} variants{survey_note}. "
            "Here are the allowable options for each:\n"
            + "\n".join(sections)
            + "\n\nIf you want, ask for one specifically (for example, `q5a` or `q5b`) and I’ll give a focused answer."
        )
        return ChatResponse(
            answer=answer,
            ranked_results=ranked_results,
            lookup_mode="exact_id",
            variant_count=max(1, len(ranked_results)),
            retrieval_mode="deterministic",
        )

    def _is_allowed_values_intent(self, query: str) -> bool:
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

    def _extract_survey_name(self, query: str) -> str | None:
        match = re.search(r"\b([A-Za-z]{3}\d{2}_[A-Za-z]{3})\b", query)
        if not match:
            return None
        return match.group(1).upper()

    def _extract_question_ref(self, query: str) -> str | None:
        id_match = re.search(r"\b[A-Za-z]{3}\d{2}_[A-Za-z]{3}_q{1,2}(\d{1,3}[a-z]?)\b", query, flags=re.IGNORECASE)
        if id_match:
            return f"q{id_match.group(1).lower()}"

        q_match = re.search(r"\bq(?:uestion)?\s*(\d{1,3}[a-z]?)\b", query, flags=re.IGNORECASE)
        if not q_match:
            return None
        return f"q{q_match.group(1).lower()}"

    def _match_question_ref_records(self, records: list[QuestionRecord], question_ref: str | None) -> list[QuestionRecord]:
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
        match = re.search(r"_q{1,2}(\d+[a-z]?)\b", question_id.lower())
        if not match:
            return None
        return match.group(1)

    def _split_question_ref(self, question_ref: str) -> tuple[str, str] | None:
        match = re.match(r"^q(\d+)([a-z]?)$", question_ref.strip().lower())
        if not match:
            return None
        return self._normalize_question_number(match.group(1)), match.group(2)

    def _split_question_component(self, component: str) -> tuple[str, str] | None:
        match = re.match(r"^(\d+)([a-z]?)$", component.strip().lower())
        if not match:
            return None
        return self._normalize_question_number(match.group(1)), match.group(2)

    def _normalize_question_number(self, number: str) -> str:
        normalized = number.strip().lstrip("0")
        return normalized or "0"

    def _build_variant_map(self, records: list[QuestionRecord], ref_number: str) -> dict[str, list[QuestionRecord]]:
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

    def _variant_sort_key(self, key: str) -> tuple[int, str]:
        match = re.match(r"^(\d+)([a-z]?)$", key.lower())
        if not match:
            return (10_000, key.lower())
        return (int(match.group(1)), match.group(2))

    def _record_sort_key(self, record: QuestionRecord) -> tuple[int, str]:
        alias_penalty = 1 if "_qq" in record.question_id.lower() else 0
        return (alias_penalty, record.question_id.lower())

    def _best_variant_record(self, records: list[QuestionRecord]) -> QuestionRecord:
        records_with_values = [record for record in records if record.value_labels]
        candidate_pool = records_with_values or records
        return sorted(candidate_pool, key=self._record_sort_key)[0]

    def _build_specific_variant_response(
        self,
        *,
        survey_name: str,
        variant_key: str,
        records: list[QuestionRecord],
    ) -> ChatResponse:
        records_with_values = [record for record in records if record.value_labels]
        candidate_pool = records_with_values or records
        primary = sorted(candidate_pool, key=self._record_sort_key)[0]
        merged_values = self._merge_value_labels(records_with_values)

        question_label = f"{survey_name} question {variant_key}"
        if merged_values:
            values_text = "\n".join(f"- {item}" for item in merged_values)
            answer = (
                f"For **{question_label}** ({primary.question_text}), the allowable response options are:\n"
                f"{values_text}"
            )
        else:
            answer = (
                f"I found **{question_label}** ({primary.question_text}), but there are no coded dropdown/allowed values "
                "for this variant in the current dictionary."
            )

        matched_ids = sorted({record.question_id for record in records})
        if len(matched_ids) > 1:
            ids_text = ", ".join(f"`{question_id}`" for question_id in matched_ids)
            answer += f"\n\nMatched variable IDs: {ids_text}"

        ranked_results = sorted(candidate_pool, key=self._record_sort_key)[: CHAT_RULES.max_cards_display]
        return ChatResponse(
            answer=answer,
            ranked_results=ranked_results,
            lookup_mode="exact_id",
            variant_count=1,
            retrieval_mode="deterministic",
        )

    def _merge_value_labels(self, records: list[QuestionRecord]) -> list[str]:
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


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
