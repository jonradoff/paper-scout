"""Embedding generation and cosine similarity scoring."""

import os

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from paper_scout.config import settings

log = structlog.get_logger()

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once, caching to disk."""
    global _model
    if _model is None:
        cache_dir = settings.model_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        log.info("loading_embedding_model", model=settings.embedding_model_name, cache=cache_dir)
        _model = SentenceTransformer(
            settings.embedding_model_name,
            cache_folder=cache_dir,
        )
        log.info("embedding_model_loaded")
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    if not texts:
        return []
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text."""
    return embed_texts([text])[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def score_embedding_similarity(
    paper_text: str,
    profile_embeddings: list[list[float]],
) -> float:
    """Score a paper's embedding similarity against the interest profile.

    Returns the max cosine similarity across all profile topic embeddings,
    normalized to 0-1 range (cosine similarity is already -1 to 1, but
    for text embeddings from the same model it's typically 0 to 1).
    """
    if not profile_embeddings:
        return 0.0

    paper_embedding = embed_text(paper_text)

    max_sim = 0.0
    for profile_emb in profile_embeddings:
        sim = cosine_similarity(paper_embedding, profile_emb)
        max_sim = max(max_sim, sim)

    # Clamp to 0-1
    return max(0.0, min(1.0, max_sim))


def score_keyword_match(
    title: str,
    abstract: str,
    keywords: dict[str, list[str]],
) -> float:
    """Score keyword matches in title and abstract.

    Returns a score normalized to 0-1.
    """
    title_lower = title.lower()
    abstract_lower = abstract.lower()

    weights = {
        "high_weight": settings.keyword_high_weight,
        "medium_weight": settings.keyword_medium_weight,
        "low_weight": settings.keyword_low_weight,
    }

    total_score = 0.0
    max_possible = 0.0

    for weight_key, weight in weights.items():
        kw_list = keywords.get(weight_key, [])
        for kw in kw_list:
            kw_lower = kw.lower()
            max_possible += weight * settings.keyword_title_boost

            if kw_lower in title_lower:
                total_score += weight * settings.keyword_title_boost
            elif kw_lower in abstract_lower:
                total_score += weight

    if max_possible == 0:
        return 0.0

    return min(1.0, total_score / max_possible)
