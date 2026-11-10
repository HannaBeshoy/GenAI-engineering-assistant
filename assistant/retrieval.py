"""Finds the datasheet pages that best match a question (TF-IDF + cosine similarity, from
scikit-learn as in the course), so the question-answering step only sees relevant text."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def top_pages(pages: list, question: str, k: int = 3, max_chars: int = 2500) -> list:
    texts = [p.text for p in pages]
    if not any(texts):
        return []
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b[\w.]+\b")
    m = vec.fit_transform(texts + [question])
    scores = cosine_similarity(m[-1], m[:-1]).ravel()
    order = scores.argsort()[::-1][:k]
    return [(pages[i].number, pages[i].text[:max_chars], float(scores[i])) for i in order if scores[i] > 0]
