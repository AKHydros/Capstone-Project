from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET


DOCX_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


@dataclass(frozen=True)
class PromptCachePayload:
    signature: str
    prompts: list[str]


@dataclass(frozen=True)
class QuestionHintCachePayload:
    signature: str
    hints: dict[str, dict[str, list[str]]]


class SurveyPromptLoader:
    def __init__(self, data_dir: Path, cache_dir: Path) -> None:
        """Configures docx source and cache paths."""
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "starter_prompts.json"
        self.question_hint_cache_path = self.cache_dir / "doc_question_hints.json"

    def load_prompts(self, max_prompts: int = 20, force_refresh: bool = False) -> list[str]:
        """Loads/caches starter prompts extracted from docx files."""
        docx_files = sorted(self.data_dir.rglob("*.docx"))
        signature = self._build_signature(docx_files)

        cached = self._read_cache()
        if (not force_refresh) and cached and cached.signature == signature:
            return cached.prompts[:max_prompts]

        prompts = self._extract_prompts(docx_files, max_prompts=max_prompts)
        self._write_cache(PromptCachePayload(signature=signature, prompts=prompts))
        return prompts

    def clear_cache(self) -> None:
        """Removes prompt and question-hint cache files."""
        if self.cache_path.exists():
            self.cache_path.unlink()
        if self.question_hint_cache_path.exists():
            self.question_hint_cache_path.unlink()

    def load_question_hints(self, force_refresh: bool = False) -> dict[str, dict[str, list[str]]]:
        """Loads/caches survey question hints extracted from docx files."""
        docx_files = sorted(self.data_dir.rglob("*.docx"))
        signature = f"hints-v1|{self._build_signature(docx_files)}"

        cached = self._read_question_hint_cache()
        if (not force_refresh) and cached and cached.signature == signature:
            return cached.hints

        hints = self._extract_question_hints(docx_files)
        self._write_question_hint_cache(QuestionHintCachePayload(signature=signature, hints=hints))
        return hints

    def _extract_prompts(self, files: list[Path], max_prompts: int) -> list[str]:
        """Extracts normalized, deduplicated prompt candidates from paragraphs."""
        seen: set[str] = set()
        out: list[str] = []
        for file_path in files:
            for text in self._iter_paragraphs(file_path):
                candidate = self._normalize_candidate(text)
                if not candidate:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(candidate)
                if len(out) >= max_prompts:
                    return out
        return out

    def _extract_question_hints(self, files: list[Path]) -> dict[str, dict[str, list[str]]]:
        """Extracts question reference hints keyed by survey and question token."""
        out: dict[str, dict[str, list[str]]] = {}
        for file_path in files:
            survey_name = file_path.stem.upper()
            if not survey_name:
                continue

            survey_hints = out.setdefault(survey_name, {})
            for text in self._iter_paragraphs(file_path):
                match = re.match(r"^\s*(\d{1,3}[a-z]?)\s*[\)\.\:\-]?\s+(.+)$", text, flags=re.IGNORECASE)
                if not match:
                    continue
                number_token = match.group(1).lower()
                question_ref = f"q{number_token}"
                question_text = text.strip()
                if len(question_text) < 12:
                    continue

                bucket = survey_hints.setdefault(question_ref, [])
                if question_text not in bucket:
                    bucket.append(question_text)
        return out

    def _iter_paragraphs(self, file_path: Path):
        """Iterates normalized paragraph text from a `.docx` XML document."""
        try:
            with zipfile.ZipFile(file_path) as archive:
                root = ET.fromstring(archive.read("word/document.xml"))
        except (zipfile.BadZipFile, KeyError, ET.ParseError):
            return
        for para in root.findall(".//w:body/w:p", DOCX_NS):
            text = "".join((node.text or "") for node in para.findall(".//w:t", DOCX_NS))
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                yield text

    def _normalize_candidate(self, text: str) -> str | None:
        """Filters and normalizes prompt text candidates."""
        if "?" not in text:
            return None
        text = text.strip()
        if len(text) < 24 or len(text) > 220:
            return None
        lower = text.lower()
        if lower.startswith("hello, my name"):
            return None
        if "would you like to take part" in lower:
            return None
        if "thank you" in lower:
            return None
        text = re.sub(r"\s+", " ", text)
        return text

    def _build_signature(self, files: list[Path]) -> str:
        """Computes prompt-cache signature from file names/sizes/mtimes."""
        parts: list[str] = []
        for p in files:
            stat = p.stat()
            parts.append(f"{p.name}:{stat.st_size}:{stat.st_mtime_ns}")
        return "|".join(parts)

    def _read_cache(self) -> PromptCachePayload | None:
        """Reads and validates starter prompt cache payload."""
        if not self.cache_path.exists():
            return None
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        signature = raw.get("signature")
        prompts = raw.get("prompts")
        if not isinstance(signature, str) or not isinstance(prompts, list):
            return None
        cleaned = [str(p) for p in prompts if isinstance(p, str)]
        return PromptCachePayload(signature=signature, prompts=cleaned)

    def _write_cache(self, payload: PromptCachePayload) -> None:
        """Writes starter prompt cache payload to disk."""
        self.cache_path.write_text(
            json.dumps({"signature": payload.signature, "prompts": payload.prompts}, indent=2),
            encoding="utf-8",
        )

    def _read_question_hint_cache(self) -> QuestionHintCachePayload | None:
        """Reads and validates question-hint cache payload."""
        if not self.question_hint_cache_path.exists():
            return None
        try:
            raw = json.loads(self.question_hint_cache_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        signature = raw.get("signature")
        hints = raw.get("hints")
        if not isinstance(signature, str) or not isinstance(hints, dict):
            return None
        cleaned: dict[str, dict[str, list[str]]] = {}
        for survey_name, refs in hints.items():
            if not isinstance(survey_name, str) or not isinstance(refs, dict):
                continue
            cleaned_refs: dict[str, list[str]] = {}
            for ref, texts in refs.items():
                if not isinstance(ref, str) or not isinstance(texts, list):
                    continue
                cleaned_texts = [str(t) for t in texts if isinstance(t, str)]
                if cleaned_texts:
                    cleaned_refs[ref] = cleaned_texts
            if cleaned_refs:
                cleaned[survey_name] = cleaned_refs
        return QuestionHintCachePayload(signature=signature, hints=cleaned)

    def _write_question_hint_cache(self, payload: QuestionHintCachePayload) -> None:
        """Writes question-hint cache payload to disk."""
        self.question_hint_cache_path.write_text(
            json.dumps({"signature": payload.signature, "hints": payload.hints}, indent=2),
            encoding="utf-8",
        )
