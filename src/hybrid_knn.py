"""
hybrid_knn.py

Hybrid (lyrics + audio) nearest neighbor utilities for the
Spectral Neighbor Plus prototype.

This module provides:
- cosine-based hybrid similarity over lyric + audio embeddings
- a get_hybrid_neighbors(...) function for top-K indices + scores
- a recommend(...) helper that returns seed + formatted recommendations
"""

from typing import Tuple, List, Optional

import numpy as np
import pandas as pd


def _norm01(x: np.ndarray) -> np.ndarray:
    """
    Normalize a score vector to the [0, 1] range.
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + 1e-9)


def _compute_cosine_scores(seed_idx: int, matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity scores between one seed vector and all others,
    assuming rows of `matrix` are already L2-normalized.

    Args:
        seed_idx: Row index of the seed vector in `matrix`.
        matrix: 2D array of shape (N, D) with normalized embeddings.

    Returns:
        1D array of cosine scores of length N.
    """
    query = matrix[seed_idx : seed_idx + 1]  # shape (1, D)
    scores = matrix @ query.T                # (N, D) @ (D, 1) -> (N, 1)
    return scores.ravel()


def get_hybrid_neighbors(
    seed_idx: int,
    lyrics_emb: np.ndarray,
    audio_emb: np.ndarray,
    alpha: float = 0.5,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve top-K hybrid (lyrics + audio) neighbors for a given track.

    Assumes:
        - `lyrics_emb` and `audio_emb` are (N, D_*) matrices with L2-normalized rows.
        - Both have the same number of rows (same track ordering).

    Hybrid score:
        s_hybrid = alpha * s_audio + (1 - alpha) * s_lyrics

    Args:
        seed_idx: Row index of the seed track.
        lyrics_emb: Lyric embedding matrix, shape (N, D_lyrics).
        audio_emb: Audio embedding matrix, shape (N, D_audio).
        alpha: Weight on audio similarity (0 = lyrics-only, 1 = audio-only).
        k: Number of neighbors to return (excluding the seed itself).

    Returns:
        top_idxs: 1D array of neighbor indices (int), length <= k.
        top_scores: 1D array of hybrid similarity scores aligned with top_idxs.
    """
    # cosine similarities via dot product
    lyr_scores = _compute_cosine_scores(seed_idx, lyrics_emb)
    aud_scores = _compute_cosine_scores(seed_idx, audio_emb)

    # normalize each score vector to [0, 1] for balance
    lyr_scores_n = _norm01(lyr_scores)
    aud_scores_n = _norm01(aud_scores)

    # hybrid mixture
    hybrid_scores = alpha * aud_scores_n + (1.0 - alpha) * lyr_scores_n

    # rank in descending order, remove the seed
    idxs = np.argsort(-hybrid_scores)
    idxs = idxs[idxs != seed_idx]

    top_idxs = idxs[:k]
    top_scores = hybrid_scores[top_idxs]
    return top_idxs, top_scores


def recommend(
    seed_idx: int,
    df: pd.DataFrame,
    lyrics_emb: np.ndarray,
    audio_emb: np.ndarray,
    alpha: float = 0.5,
    k: int = 10,
    context_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a hybrid recommendation for a given seed track and return
    the seed row plus a small recommendation table.

    Args:
        seed_idx: Row index of the seed track in `df` and embedding matrices.
        df: DataFrame containing at least the lyrics + metadata columns.
        lyrics_emb: Lyric embedding matrix aligned with df rows.
        audio_emb: Audio embedding matrix aligned with df rows.
        alpha: Weight on audio similarity (0 = lyrics-only, 1 = audio-only).
        k: Number of neighbors to return.
        context_cols: Optional list of metadata columns to show in the recs
                      (e.g. ["Artist(s)", "song", "Genre", "Energy"]).

    Returns:
        seed_df: DataFrame with a single row for the seed track.
        recs_df: DataFrame with top-K neighbors and a `hybrid_score` column.
    """
    if context_cols is None:
        # sensible default set; adjust to your schema as needed
        context_cols = ["Artist(s)", "song", "text", "Genre", "Energy", "Positiveness"]

    neighbors, scores = get_hybrid_neighbors(
        seed_idx=seed_idx,
        lyrics_emb=lyrics_emb,
        audio_emb=audio_emb,
        alpha=alpha,
        k=k,
    )

    # guard against missing columns
    cols_existing = [c for c in context_cols if c in df.columns]

    seed_df = df.iloc[[seed_idx]][cols_existing]
    recs_df = df.iloc[neighbors][cols_existing].copy()
    recs_df = recs_df.assign(hybrid_score=scores)

    return seed_df, recs_df
