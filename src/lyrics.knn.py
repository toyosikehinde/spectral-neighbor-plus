"""
lyrics_knn.py

Lyric-space nearest neighbor utilities for the Spectral Neighbor Plus prototype.

This module provides:
- a helper to load a precomputed FAISS index for lyric embeddings
- a get_lyrics_neighbors(...) function that returns top-K lyric-similar tracks
"""

from pathlib import Path
from typing import Tuple

import faiss
import numpy as np


def load_lyrics_index(index_path: str) -> faiss.Index:
    """
    Load a precomputed FAISS index for lyric embeddings.

    Args:
        index_path: Path to the .faiss index file.

    Returns:
        A FAISS Index object ready for similarity search.
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS lyrics index not found at: {index_path}")
    return faiss.read_index(str(index_path))


def get_lyrics_neighbors(
    seed_idx: int,
    embeddings: np.ndarray,
    index: faiss.Index,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retrieve top-K lyric-space nearest neighbors for a given track.

    Assumes:
        - `embeddings` is an array of shape (N, D) with L2-normalized rows.
        - `index` is a FAISS inner-product index built on the same embeddings.

    Args:
        seed_idx: Row index of the seed track in `embeddings`.
        embeddings: 2D array of lyric embeddings (N x D), already normalized.
        index: FAISS IndexFlatIP (or compatible) built on `embeddings`.
        k: Number of neighbors to return (excluding the seed itself).

    Returns:
        neighbor_idxs: 1D array of neighbor indices (int) of length <= k.
        neighbor_scores: 1D array of similarity scores (float) aligned with neighbor_idxs.
    """
    # Extract query vector and ensure it has shape (1, D)
    query = embeddings[seed_idx : seed_idx + 1]

    # Search returns the seed itself as the closest match, so we request k+1
    scores, idxs = index.search(query, k + 1)
    scores = scores[0]
    idxs = idxs[0]

    # Remove the seed track from the results if present
    mask = idxs != seed_idx
    neighbor_idxs = idxs[mask][:k]
    neighbor_scores = scores[mask][:k]

    return neighbor_idxs, neighbor_scores
