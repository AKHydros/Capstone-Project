from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

from .pipeline import TextChunk


@dataclass
class LexicalRetriever:
    vectorizer: TfidfVectorizer
    matrix: object
    chunks: list[TextChunk]

    @classmethod
    def build(cls, chunks: list[TextChunk]) -> "LexicalRetriever":
        corpus = [chunk.text for chunk in chunks]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        return cls(vectorizer=vectorizer, matrix=matrix, chunks=chunks)

    def score(self, query: str) -> list[float]:
        q = self.vectorizer.transform([query])
        sims = (self.matrix @ q.T).toarray().ravel()
        return sims.tolist()
