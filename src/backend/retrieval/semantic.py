from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..models import QuestionRecord


@dataclass
class SemanticRetriever:
    records: list[QuestionRecord]
    matrix: np.ndarray
    mode: str
    vectorizer: TfidfVectorizer | None = None
    embedding_model: str | None = None

    @classmethod
    def build(cls, records: list[QuestionRecord]) -> "SemanticRetriever":
        texts = [r.document_text for r in records]
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if api_key:
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            embeddings: list[list[float]] = []
            for i in range(0, len(texts), 200):
                batch = texts[i : i + 200]
                response = _openai_client(api_key).embeddings.create(model=model, input=batch)
                embeddings.extend([d.embedding for d in response.data])
            matrix = np.array(embeddings, dtype=np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
            return cls(records=records, matrix=matrix, mode="openai", embedding_model=model)

        # Local fallback for development without API key.
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        return cls(records=records, matrix=matrix, mode="local", vectorizer=vectorizer)

    def score(self, query: str) -> list[float]:
        if self.mode == "openai":
            assert self.embedding_model is not None
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            emb = _openai_client(api_key).embeddings.create(
                model=self.embedding_model, input=[query]
            ).data[0].embedding
            q = np.array(emb, dtype=np.float32)
            q /= np.linalg.norm(q) + 1e-12
            sims = self.matrix @ q
            return sims.tolist()

        assert self.vectorizer is not None
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(self.matrix, q).ravel()
        return sims.tolist()


@lru_cache(maxsize=1)
def _openai_client(api_key: str) -> OpenAI:
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for OpenAI semantic mode")
    return OpenAI(api_key=api_key)
