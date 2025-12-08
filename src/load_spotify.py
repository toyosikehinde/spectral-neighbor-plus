"""
load_spotify.py

Utility functions for loading the cleaned Spotify subset used in the
Spectral Neighbor Plus hybrid recommender.

This module centralizes:
- reading the processed CSV containing track_id
- applying canonical column renaming
- selecting lyrics, audio, and metadata columns based on schema.py
- returning a clean DataFrame ready for embedding, FAISS indexing, or evaluation
"""
# load_spotify.py

import pandas as pd
from pathlib import Path
from schema import (
    ID_COL,
    LYRICS_COL,
    AUDIO_FEATURE_COLS,
    CONTEXT_COLS,
    CANONICAL_RENAMES,
    ALL_USED_COLS,
)

def load_spotify(path: str = "data/processed/spotify_900k_sample.csv") -> pd.DataFrame:
    """
    Load the processed Spotify dataset with track_id and apply schema-based cleanup.

    Returns:
        pd.DataFrame with canonical column names and only the columns used
        by the hybrid recommendation pipeline.
    """
    path = Path(path)
    df = pd.read_csv(path)

    # apply canonical names (Artist(s) -> artist, Positiveness -> valence, etc.)
    df = df.rename(columns=CANONICAL_RENAMES)

    # ensure track_id exists (fallback if missing)
    if ID_COL not in df.columns:
        df[ID_COL] = range(len(df))

    # select all relevant columns that exist in this dataset
    keep_cols = [col for col in ALL_USED_COLS if col in df.columns]
    df = df[keep_cols].copy()

    return df
