from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer

from ..models import QuestionRecord


@dataclass
class LexicalRetriever:
    vectorizer: TfidfVectorizer
    matrix: object
    records: list[QuestionRecord]

    @classmethod
    def build(cls, records: list[QuestionRecord]) -> "LexicalRetriever":
        corpus = [r.document_text for r in records]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)
        matrix = vectorizer.fit_transform(corpus)
        return cls(vectorizer=vectorizer, matrix=matrix, records=records)

    def score(self, query: str) -> list[float]:
        q = self.vectorizer.transform([query])
        sims = (self.matrix @ q.T).toarray().ravel()
        return sims.tolist()
