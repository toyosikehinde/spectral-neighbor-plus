"""
metrics.py

Evaluation utilities for the Spectral Neighbor Plus prototype.

Provides:
- generic label Purity@K (genre, emotion, theme, etc.)
- artist diversity@K
- per-seed evaluation wrapper for lyrics/audio/hybrid retrieval
"""

from typing import Dict, List

import numpy as np
import pandas as pd

# Core metrics

def purity_at_k(
    seed_idx: int,
    neighbor_idxs: np.ndarray,
    df: pd.DataFrame,
    label_col: str,
) -> float:
    """
    Proportion of neighbors that share the same label as the seed.
    Works for genre, emotion, theme, or any categorical field.
    """
    if label_col not in df.columns:
        return np.nan

    seed_label = df.loc[seed_idx, label_col]
    if pd.isna(seed_label):
        return np.nan

    neighbor_labels = df.loc[neighbor_idxs, label_col]
    valid = neighbor_labels.notna()
    if valid.sum() == 0:
        return np.nan

    return (neighbor_labels[valid] == seed_label).mean()


def artist_diversity_at_k(
    neighbor_idxs: np.ndarray,
    df: pd.DataFrame,
    artist_col: str = "artist",
) -> float:
    """
    Fraction of unique artists among the top-K neighbors.
    """
    if artist_col not in df.columns:
        return np.nan

    artists = df.loc[neighbor_idxs, artist_col].dropna()
    if len(artists) == 0:
        return np.nan

    return artists.nunique() / len(artists)

# Per-seed evaluation wrapper

def evaluate_seed(
    seed_idx: int,
    df: pd.DataFrame,
    mode: str,
    k: int,
    label_cols: List[str],
    get_lyrics_neighbors,
    get_audio_neighbors,
    get_hybrid_neighbors,
    lyrics_emb=None,
    audio_emb=None,
    lyrics_index=None,
    audio_index=None,
    alpha: float = 0.5,
    artist_col: str = "artist",
) -> Dict[str, float]:
    """
    Compute metrics for a single seed track under a given similarity mode.

    Supports purity for any label column provided in `label_cols`,
    including genre, emotion, and theme.
    """
    if mode == "lyrics":
        neighbor_idxs, _ = get_lyrics_neighbors(seed_idx, lyrics_emb, lyrics_index, k)
    elif mode == "audio":
        neighbor_idxs, _ = get_audio_neighbors(seed_idx, audio_emb, audio_index, k)
    else:
        neighbor_idxs, _ = get_hybrid_neighbors(seed_idx, lyrics_emb, audio_emb, alpha, k)

    metrics = {}
    for col in label_cols:
        metrics[f"{col}_purity@{k}"] = purity_at_k(seed_idx, neighbor_idxs, df, col)

    metrics[f"artist_diversity@{k}"] = artist_diversity_at_k(neighbor_idxs, df, artist_col)

    metrics["seed_idx"] = seed_idx
    metrics["mode"] = mode

    return metrics
