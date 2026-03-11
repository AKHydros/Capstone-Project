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
    ) -> ChatResponse:
        ranked = self.retriever.search(query=query, survey_name=survey_name, wave_year=wave_year)
        cards = ranked[: CHAT_RULES.max_cards_display]

        if not cards:
            return ChatResponse(
                answer="I could not find grounded matches in the current Excel dictionary. Try broader wording or relax filters.",
                ranked_results=[],
            )

        if self.llm_client.enabled:
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
            answer = self.llm_client.summarize(system_prompt, user_prompt)
        else:
            first = cards[0]
            answer = (
                f"Top grounded match: {first.question_id} - {first.question_text}. "
                f"Found {len(cards)} relevant questions. Configure OPENAI_API_KEY for richer conversational summaries."
            )

        return ChatResponse(answer=answer, ranked_results=cards)
