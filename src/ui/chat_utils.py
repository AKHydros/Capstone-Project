from __future__ import annotations

from datetime import datetime, timezone


def build_conversation_context(history: list[dict[str, object]], *, max_turns: int = 20) -> list[dict[str, str]]:
    """Builds bounded user/assistant turn context for router requests."""
    context: list[dict[str, str]] = []
    for message in history:
        role = str(message.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        if role == "assistant" and message.get("answer"):
            content = str(message.get("answer", "")).strip()
        else:
            content = str(message.get("content", "")).strip()
        if not content:
            continue
        context.append({"role": role, "content": content})
    return context[-max_turns:]


def chat_history_markdown(history: list[dict[str, object]]) -> str:
    """Renders chat history as a markdown transcript with metadata and citations."""
    header = f"# Chat Export\n\nGenerated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n\n"
    if not history:
        return header + "_No chat messages yet._\n"

    lines: list[str] = [header]
    for idx, msg in enumerate(history, start=1):
        role = str(msg.get("role", "unknown")).strip().title()
        if role == "Assistant" and msg.get("answer"):
            body = str(msg.get("answer", "")).strip()
        else:
            body = str(msg.get("content", "")).strip()
        if not body:
            continue
        lines.append(f"## {idx}. {role}\n")
        lines.append(f"{body}\n")

        meta = str(msg.get("meta", "")).strip()
        if meta:
            lines.append(f"- Meta: `{meta}`\n")
        sources = msg.get("sources", [])
        if isinstance(sources, list) and sources:
            lines.append("- Sources:\n")
            for source in sources:
                if not isinstance(source, dict):
                    continue
                marker = str(source.get("marker", ""))
                label = str(source.get("label", ""))
                if marker or label:
                    lines.append(f"  - {marker} {label}".rstrip() + "\n")
        lines.append("\n")
    return "".join(lines)


def citation_markers(sources: list[dict[str, object]], *, max_markers: int = 4) -> str:
    """Returns compact citation markers string like `[1] [2]`."""
    if not sources:
        return ""
    markers: list[str] = []
    for source in sources[:max_markers]:
        marker = str(source.get("marker", "")).strip()
        if marker:
            markers.append(marker)
    return " ".join(markers)


def confidence_badge(confidence_score: float | None) -> str:
    """Returns user-facing confidence label with percentage and plain-language description."""
    if confidence_score is None:
        return "Match quality: exact lookup"
    pct = f"{confidence_score:.0%}"
    if confidence_score >= 0.75:
        return f"Match quality: High ({pct}) — strong retrieval match"
    if confidence_score >= 0.45:
        return f"Match quality: Medium ({pct}) — good retrieval match"
    if confidence_score >= 0.25:
        return f"Match quality: Low ({pct}) — partial match, review results"
    return f"Match quality: Very low ({pct}) — consider rephrasing your query"
