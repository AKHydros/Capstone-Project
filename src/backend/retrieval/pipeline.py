from __future__ import annotations

from dataclasses import dataclass
import re

from ..models import QuestionRecord


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    record_index: int
    question_id: str
    text: str


def ingest_clean_chunk(
    records: list[QuestionRecord],
    *,
    chunk_size: int = 720,
    overlap: int = 120,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for record_index, record in enumerate(records):
        cleaned = _clean_text(record.document_text)
        pieces = _chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)
        if not pieces:
            continue
        for idx, piece in enumerate(pieces):
            chunks.append(
                TextChunk(
                    chunk_id=f"{record.question_id}:{idx}",
                    record_index=record_index,
                    question_id=record.question_id,
                    text=piece,
                )
            )
    return chunks


def _clean_text(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    safe_overlap = max(0, min(overlap, chunk_size // 3))
    step = max(1, chunk_size - safe_overlap)

    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        slice_text = text[start:end].strip()
        if slice_text:
            out.append(slice_text)
        if end >= len(text):
            break
        start += step
    return out
