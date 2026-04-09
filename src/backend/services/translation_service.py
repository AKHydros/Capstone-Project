from __future__ import annotations

from dataclasses import dataclass

from ..cache.ttl_cache import TTLCache


_SUPPORTED = {"en", "fr"}

_TRANSLATION_MAP_EN_TO_FR = {
    "question": "question",
    "questions": "questions",
    "survey": "sondage",
    "label": "etiquette",
    "labels": "etiquettes",
    "results": "resultats",
    "summary": "resume",
    "found": "trouve",
    "could not": "n'a pas pu",
    "filter": "filtre",
    "topic": "sujet",
}
_TRANSLATION_MAP_FR_TO_EN = {v: k for k, v in _TRANSLATION_MAP_EN_TO_FR.items()}


@dataclass(frozen=True)
class TranslationResult:
    text: str
    language: str
    translated: bool


class TranslationService:
    def __init__(self, ttl_seconds: int = 600) -> None:
        """Initializes translation cache."""
        self._cache: TTLCache[TranslationResult] = TTLCache(ttl_seconds=ttl_seconds, max_size=2048)

    def normalize_language(self, language: str | None) -> str:
        """Normalizes language code to supported set (`en`, `fr`)."""
        if not language:
            return "en"
        lang = language.lower().strip()
        return lang if lang in _SUPPORTED else "en"

    def translate(self, text: str, source_language: str, target_language: str) -> TranslationResult:
        """Performs cached dictionary translation between supported languages."""
        src = self.normalize_language(source_language)
        dst = self.normalize_language(target_language)
        if src == dst:
            return TranslationResult(text=text, language=dst, translated=False)

        cache_key = f"{src}:{dst}:{text}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        translated = self._translate_with_dictionary(text, src, dst)
        payload = TranslationResult(text=translated, language=dst, translated=True)
        self._cache.set(cache_key, payload)
        return payload

    def stats(self) -> dict[str, int]:
        """Returns translation cache stats."""
        stats = self._cache.stats()
        return {
            "hits": stats.hits,
            "misses": stats.misses,
            "expired": stats.expired,
            "size": stats.size,
        }

    def _translate_with_dictionary(self, text: str, source_language: str, target_language: str) -> str:
        """Applies phrase-level mapping rules for simple translation fallback."""
        mapping = _TRANSLATION_MAP_EN_TO_FR if source_language == "en" and target_language == "fr" else _TRANSLATION_MAP_FR_TO_EN
        output = text
        for src, dst in mapping.items():
            output = output.replace(src, dst)
            output = output.replace(src.title(), dst.title())
            output = output.replace(src.upper(), dst.upper())
        return output
