"""
Extractive summarization using the TextRank algorithm.

TextRank treats sentences like web pages in Google's PageRank — sentences
that are "voted for" by many similar sentences rank higher and get selected.
No model weights; purely graph-based math.
"""

import re
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_N: int = 3          # Sentences to return by default
DAMPING_FACTOR: float = 0.85    # Standard PageRank damping (teleportation prob)
MAX_ITERATIONS: int = 100       # Power-iteration convergence cap
CONVERGENCE_THRESHOLD: float = 1e-4


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """
    Split article text into individual sentences.

    Parameters
    ----------
    text : str
        Raw article text.

    Returns
    -------
    list[str]
        List of non-empty sentence strings.
    """
    # Split on period/exclamation/question followed by whitespace
    raw: list[str] = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_similarity_matrix(sentences: list[str]) -> np.ndarray:
    """
    Build an N×N cosine-similarity matrix over TF-IDF sentence vectors.

    Each cell [i][j] holds how similar sentence i is to sentence j.
    Self-similarity (diagonal) is zeroed out so sentences don't vote for
    themselves.

    Parameters
    ----------
    sentences : list[str]

    Returns
    -------
    np.ndarray
        Normalised similarity matrix of shape (N, N).
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sim_matrix: np.ndarray = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0.0)

    # Row-normalise so each row sums to 1 (stochastic matrix for PageRank)
    row_sums = sim_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated sentences
    row_sums[row_sums == 0] = 1.0
    return sim_matrix / row_sums


# ---------------------------------------------------------------------------
# PageRank / TextRank
# ---------------------------------------------------------------------------

def _textrank_scores(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Run power-iteration PageRank on the similarity matrix.

    Parameters
    ----------
    sim_matrix : np.ndarray
        Row-normalised N×N similarity matrix.

    Returns
    -------
    np.ndarray
        Score vector of length N.
    """
    n: int = sim_matrix.shape[0]
    scores: np.ndarray = np.ones(n) / n  # Uniform initialisation

    for _ in range(MAX_ITERATIONS):
        prev_scores = scores.copy()
        scores = (1 - DAMPING_FACTOR) / n + DAMPING_FACTOR * sim_matrix.T @ scores
        delta = np.abs(scores - prev_scores).sum()
        if delta < CONVERGENCE_THRESHOLD:
            break

    return scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_summary(text: str, top_n: int = DEFAULT_TOP_N) -> str:
    """
    Generate an extractive summary by selecting the top-ranked sentences.

    Sentence order in the output mirrors the original article order
    (reads naturally rather than by rank).

    Parameters
    ----------
    text : str
        Raw article text.
    top_n : int
        Number of sentences to include in the summary.

    Returns
    -------
    str
        Extractive summary as a single string.
    """
    sentences: list[str] = _split_sentences(text)

    if len(sentences) <= top_n:
        # Article is already short — return as-is
        return " ".join(sentences)

    sim_matrix: np.ndarray = _build_similarity_matrix(sentences)
    scores: np.ndarray = _textrank_scores(sim_matrix)

    # Pick indices of top-N scoring sentences, preserve original order
    top_indices: list[int] = sorted(
        np.argsort(scores)[-top_n:].tolist()
    )

    summary: str = " ".join(sentences[i] for i in top_indices)
    return summary
