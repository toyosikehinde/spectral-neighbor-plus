"""
schema.py

Central schema definition for the Spectral Neighbor Plus prototype.
Defines which columns are used for lyrics, audio features, and
context/metadata, plus a canonical renaming map.
"""

from typing import List, Dict

# -------------------------------------------------------------------
# Core column names
# -------------------------------------------------------------------

ID_COL: str = "track_id"          
LYRICS_COL: str = "text"          # full lyrics field

# Label columns (for purity@K metrics)
GENRE_COL: str = "Genre"
EMOTION_COL: str = "emotion"

LABEL_COLS: List[str] = [GENRE_COL, EMOTION_COL]

# -------------------------------------------------------------------
# Audio feature columns (from your Spotify 900k subset)
# -------------------------------------------------------------------

AUDIO_FEATURE_COLS: List[str] = [
    "Energy",
    "Danceability",
    "Positiveness",      # acts as valence
    "Acousticness",
    "Instrumentalness",
    "Speechiness",
    "Liveness",
    "Tempo",
    "Loudness (db)",
]

# -------------------------------------------------------------------
# Context / metadata columns
# -------------------------------------------------------------------

CONTEXT_COLS: List[str] = [
    "Artist(s)",
    "song",
    "Album",
    GENRE_COL,
    EMOTION_COL,
]

# -------------------------------------------------------------------
# Canonical renames used internally (optional but handy)
# -------------------------------------------------------------------

CANONICAL_RENAMES: Dict[str, str] = {
    "Artist(s)": "artist",
    "song": "title",
    "Album": "album",
    "Genre": "genre",
    "Energy": "energy",
    "Danceability": "danceability",
    "Positiveness": "valence",
    "Loudness (db)": "loudness_db",
}

# Convenience: all columns expected to keep in the cleaned subset
ALL_USED_COLS: List[str] = [
    ID_COL,
    LYRICS_COL,
    *AUDIO_FEATURE_COLS,
    *CONTEXT_COLS,
]

