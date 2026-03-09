from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

def _cosine_scores(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosine similarity scores of every row in x against vector q."""
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    return x_norm @ q_norm

@lru_cache(maxsize=16)
def load_feature_matrix(
    engine: Engine,
    season: int,
    feat_set: str
) -> Tuple[List[str], np.ndarray]:
    """Load precomputed z-score feature vectors from Postgres.

    Returns:
      - players: list of player names (same order as matrix rows)
      - matrix: 2D numpy array [n_players, n_features]

    Cached per (engine object identity, season, feature_set). If you restart the app,
    cache resets.
    """
    sql = text(
        """
        SELECT ps.player AS player, f.z_vector AS z_vector
        FROM player_season_features f
        JOIN player_seasons ps
            ON ps.player_key = f.player_key
            AND ps.season = f.season
        WHERE f.season = :season
            AND f.feature_set = :feature_set
        ORDER BY ps.player_name ASC
        """
    )
    df = pd.read_sql(sql, engine, params={'season': season, 'feature_set': feat_set})
    if df.empty:
        return [], np.zeros((0,0), dtype=float)
    players = df['players'].astype(str).tolist()
    mat = np.array(df['z_vector'].to_list(), dtype=float)
    return players, mat

def top_k_similar(
    engine: Engine,
    season: int,
    feat_set: str,
    player_name: str,
    k: int
) -> List[dict]:
    players, mat = load_feature_matrix(engine, season, feat_set)
    if not players:
        raise KeyError('No feature rows found. Run the ETL feature stage first')
    try:
        idx = players.index(player_name)
    except ValueError as e:
        raise KeyError('Player not found in featue table') from e
    q = mat[idx]
    scores = _cosine_scores(mat, q)
    order = np.argsort(-scores)
    out = []
    for j in order:
        if int(j) == idx:
            continue
        out.append({'player': players[int(j)], 'score': float(scores[int(j)])})
        if len(out) >= k:
            break
    return out