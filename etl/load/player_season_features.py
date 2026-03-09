from typing import Any, Dict, Iterable, List
from sqlalchemy import (
    ARRAY,
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    create_engine,
    func
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert

def _normalize_database_url(url: str) -> str:
    url = url.strip()
    if url.startswith("postgres://") and "+" not in url.split("://", 1)[0]:
        url = "postgres+psycopg://" + url[len("postgres://") :]
    return url

def _tables(metadata: MetaData) -> tuple[Table, Table]:
    if "player_seasons" in metadata.tables:
        return metadata.tables["player_seasons"], metadata.tables["player_season_features"]
    player_seasons = Table(
        "player_seasons",
        metadata,
        Column("player_key", Text, primary_key=True),
        Column("player_name", Text, nullable=False),
        Column("season", Integer, primary_key=True),
        Column("team", Text),
        Column("pos", Text),
        Column("g", Integer),
        Column("mp", Numeric),
        Column("mpg", Numeric),
        Column("total_minutes", Numeric),
        Column("stats", JSONB, nullable=False),
        Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    )
    player_season_features = Table(
        "player_season_features",
        metadata,
        Column("player_key", Text, primary_key=True),
        Column("season", Integer, primary_key=True),
        Column("feature_set", Text, primary_key=True),
        Column("feature_names", ARRAY(Text), nullable=False),
        Column("z_scores", JSONB, nullable=False),
        Column("z_vector", ARRAY(Float), nullable=False),
        Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    )
    return player_seasons, player_season_features

def ensure_tables(engine_url: str) -> None:
    """Creates tables if they don't exist.

    You can also rely on docker-entrypoint-initdb.d SQL files; this is a safety net.
    """
    engine = create_engine(_normalize_database_url(engine_url), pool_pre_ping=True, future=True)
    md = MetaData()
    _tables(md)
    md.create_all(engine, checkfirst=True)

def upsert_player_seasons(engine_url: str, rows: List[Dict[str, Any]]) -> int:
    engine = create_engine(_normalize_database_url(engine_url), pool_pre_ping=True, future=True)
    md = MetaData()
    player_seasons, _ = _tables(md)
    md.create_all(engine, checkfirst=True)
    if not rows:
        return 0
    stmt = pg_insert(player_seasons).values(rows)
    update_cols = {
        c.name: getattr(stmt.excluded, c.name)
        for c in player_seasons.columns
        if c.name not in {'playerr_key', 'season'}
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=[player_seasons.c.player_key, player_seasons.c.season],
        set_=update_cols
    )
    with engine.begin() as conn:
        res = conn.execute(stmt)
        return res.rowcount or 0

def upsert_player_season_features(engine_url: str, rows: List[Dict[str, Any]]) -> int:
    engine = create_engine(_normalize_database_url(engine_url), pool_pre_ping=True, future=True)
    md = MetaData()
    _, player_season_features = _tables(md)
    md.create_all(engine, checkfirst=True)
    stmt = pg_insert(player_season_features).values(rows)
    update_cols = {
        c.name: getattr(stmt.excluded, c.name)
        for c in player_season_features.columns
        if c.name not in {'player_key', 'season'}
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=[
            player_season_features.c.player_key,
            player_season_features.c.season,
            player_season_features.c.feature_set
        ],
        set_=update_cols
    )
    with engine.begin() as conn:
        res = conn.execute(stmt)
        return res.rowcount or 0