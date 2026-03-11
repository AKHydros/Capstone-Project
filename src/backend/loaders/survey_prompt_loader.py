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


class SurveyPromptLoader:
    def __init__(self, data_dir: Path, cache_dir: Path) -> None:
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "starter_prompts.json"

    def load_prompts(self, max_prompts: int = 20, force_refresh: bool = False) -> list[str]:
        docx_files = sorted(self.data_dir.rglob("*.docx"))
        signature = self._build_signature(docx_files)

        cached = self._read_cache()
        if (not force_refresh) and cached and cached.signature == signature:
            return cached.prompts[:max_prompts]

        prompts = self._extract_prompts(docx_files, max_prompts=max_prompts)
        self._write_cache(PromptCachePayload(signature=signature, prompts=prompts))
        return prompts

    def clear_cache(self) -> None:
        if self.cache_path.exists():
            self.cache_path.unlink()

    def _extract_prompts(self, files: list[Path], max_prompts: int) -> list[str]:
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

    def _iter_paragraphs(self, file_path: Path):
        with zipfile.ZipFile(file_path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        for para in root.findall(".//w:body/w:p", DOCX_NS):
            text = "".join((node.text or "") for node in para.findall(".//w:t", DOCX_NS))
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                yield text

    def _normalize_candidate(self, text: str) -> str | None:
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
        parts: list[str] = []
        for p in files:
            stat = p.stat()
            parts.append(f"{p.name}:{stat.st_size}:{stat.st_mtime_ns}")
        return "|".join(parts)

    def _read_cache(self) -> PromptCachePayload | None:
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
        self.cache_path.write_text(
            json.dumps({"signature": payload.signature, "prompts": payload.prompts}, indent=2),
            encoding="utf-8",
        )
