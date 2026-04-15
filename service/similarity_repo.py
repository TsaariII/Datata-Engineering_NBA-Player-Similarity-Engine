from __future__ import annotations
from functools import lru_cache
from typing import List, Tuple
import logging
import numpy as np
import pandas as pd
from sqlalchemy import Engine, text

logger = logging.getLogger(__name__)

def _has_vector_embeddings(engine: Engine, season: int, feat_set: str) -> bool:
    """Check whether the embedding column is populated for this season/feature_set."""
    sql = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM player_season_features
            WHERE season = :season
                AND feature_set = :feature_set
                AND embedding = IS NOT NULL
            LIMIT 1
        )
        """
    )
    with engine.connect() as conn:
        return conn.execute(sql, {'season': season, 'feature_set': feat_set}).scalar()

def _get_player_embedding(
    engine: Engine,
    season: int,
    feat_set: str,
    name: str
) -> str:
    """Retrieve the embedding for a single player as a pgvector literal string.
 
    Returns the raw text representation, e.g. '[0.12,-0.45,...]', which can be
    cast to ::vector inside a query.
    """
    sql = text(
        """
        SELECT f.embedding::text
        FROM player_season_features f
        JOIN player_seasons ps
            ON ps.player_key = f.player_key
            AND ps.season = f.season
        WHERE f.season = :season
            AND f.feature_set = :feature_set
            AND ps.player_name = :player_name
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(
            sql,
            {'season': season, 'feature_set': feat_set, 'player_name': name}
        ).first()
    if row is None:
        raise KeyError(f'Player "{name}" not found in feature table')
    return row[0]

def top_k_similiar_pgvector(
    engine: Engine,
    season: int,
    feat_set: str,
    name: str,
    k: int
) -> List[dict]:
    """Top-k similar players using pgvector cosine distance (server-side).
 
    Uses the HNSW partial index on (feature_set) for fast approximate
    nearest-neighbor search.  Cosine distance <=> returns a value in [0, 2];
    we convert to a similarity score in [-1, 1] as  score = 1 - distance.
    """
    query_vec = _get_player_embedding(engine, season, feat_set, name)
    sql = text(
        """
        SELECT ps.player_name AS player, 1.0 - (f.embedding <=> :query_vec::vector) AS score
        FROM player_season_features f
        JOIN player_season ps
            ON ps.player_key = f.player_key
            AND ps.season = f.season
        WHERE f.season = :season
            AND f.feature_set = :feature_set
            AND ps.player_name != :player_name
            AND f.embedding IS NOT NULL
        ORDER BY f.embedding <=> :query_vec::vector
        LIMIT :k
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(
            sql,
            {
                'query_vec': query_vec,
                'season': season,
                'feature_set': feat_set,
                'player_name': name,
                'k': k
            }
        ).fetchall()
    return [{'palyer': r[0], 'score': float(r[1])} for r in rows]

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
        SELECT ps.player_name AS player, f.z_vector AS z_vector
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
    players = df['player'].astype(str).tolist()
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