from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..cache.ttl_cache import CacheStats, TTLCache
from ..models import QuestionRecord
from .pipeline import TextChunk


@dataclass
class SemanticRetriever:
    records: list[QuestionRecord]
    chunks: list[TextChunk]
    matrix: np.ndarray
    mode: str
    vectorizer: TfidfVectorizer | None = None
    embedding_model: str | None = None
    query_embedding_cache: TTLCache[list[float]] | None = None

    @classmethod
    def build(cls, records: list[QuestionRecord], chunks: list[TextChunk]) -> "SemanticRetriever":
        texts = [c.text for c in chunks]
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if api_key:
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            embed_cache_ttl = int(os.getenv("RAG_EMBED_CACHE_TTL", "1800"))
            embeddings: list[list[float]] = []
            for i in range(0, len(texts), 200):
                batch = texts[i : i + 200]
                response = _openai_client(api_key).embeddings.create(model=model, input=batch)
                embeddings.extend([d.embedding for d in response.data])
            matrix = np.array(embeddings, dtype=np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
            return cls(
                records=records,
                chunks=chunks,
                matrix=matrix,
                mode="openai",
                embedding_model=model,
                query_embedding_cache=TTLCache(ttl_seconds=max(embed_cache_ttl, 30), max_size=4096),
            )

        # Local fallback for development without API key.
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        return cls(records=records, chunks=chunks, matrix=matrix, mode="local", vectorizer=vectorizer)

    def score(self, query: str) -> list[float]:
        sims, _ = self.score_with_meta(query)
        return sims

    def score_with_meta(self, query: str) -> tuple[list[float], bool]:
        if self.mode == "openai":
            assert self.embedding_model is not None
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            query_key = _normalize_query(query)
            emb: list[float] | None = None
            cache_hit = False
            if self.query_embedding_cache:
                emb = self.query_embedding_cache.get(query_key)
                cache_hit = emb is not None
            if emb is None:
                emb = _openai_client(api_key).embeddings.create(
                    model=self.embedding_model, input=[query]
                ).data[0].embedding
                if self.query_embedding_cache:
                    self.query_embedding_cache.set(query_key, emb)
            q = np.array(emb, dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-12
            sims = self.matrix @ q
            return sims.tolist(), cache_hit

        assert self.vectorizer is not None
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(self.matrix, q).ravel()
        return sims.tolist(), False

    def embedding_cache_stats(self) -> CacheStats:
        if self.query_embedding_cache is None:
            return CacheStats(hits=0, misses=0, expired=0, size=0)
        return self.query_embedding_cache.stats()


@lru_cache(maxsize=1)
def _openai_client(api_key: str) -> OpenAI:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for OpenAI semantic mode")
    return OpenAI(api_key=api_key)


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().split())
