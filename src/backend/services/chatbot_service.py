from __future__ import annotations

from dataclasses import dataclass

from ..business_rules import CHAT_RULES, build_grounded_context
from ..llm.openai_client import OpenAIChatClient
from ..models import ChatResponse
from ..retrieval.hybrid import HybridRetriever


@dataclass
class ChatbotService:
    retriever: HybridRetriever
    llm_client: OpenAIChatClient

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
        ranked = self.retriever.search(
            query=query,
            survey_name=survey_name,
            wave_year=wave_year,
            topic_label=topic_label,
            topic_source_type=topic_source_type,
        )
        cards = ranked[: CHAT_RULES.max_cards_display]

        if not cards:
            return ChatResponse(
                answer="I could not find grounded matches in the current Excel dictionary. Try broader wording or relax filters.",
                ranked_results=[],
            )

        provider_is_supported = llm_provider.lower() in {"chatgpt", "openai"}
        should_use_llm = (
            self.llm_client.enabled
            if use_llm is None
            else (use_llm and self.llm_client.enabled and provider_is_supported)
        )
        if should_use_llm:
            context = build_grounded_context(cards)
            system_prompt = (
                "You are a grounded research dictionary assistant. "
                "Only summarize retrieved records. Never invent question text, IDs, or surveys. "
                "If uncertain, say so plainly."
            )
            user_prompt = (
                f"User query: {query}\n\n"
                f"Retrieved records:\n{context}\n\n"
                "Return: (1) concise summary, (2) notable patterns, (3) 1 suggested follow-up query."
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
            first = cards[0]
            provider_note = (
                f" {llm_provider.title()} is not implemented yet; using deterministic summary."
                if llm_provider.lower() not in {"chatgpt", "openai"}
                else ""
            )
            answer = (
                f"Top grounded match: {first.question_id} - {first.question_text}. "
                f"Found {len(cards)} relevant questions. "
                f"Configure OPENAI_API_KEY for richer conversational summaries.{provider_note}"
            )

        return ChatResponse(answer=answer, ranked_results=cards)
