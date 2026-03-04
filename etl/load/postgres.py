from __future__ import annotations
from sqlalchemy import create_engine, Text
import pandas as pd
from typing import Any

def _normalize_database_url(url: str) -> str:
    url = url.strip()
    if url.startswith('postgres://'):
        url = 'postgres://' + url[len('postgres://') :]
    if url.startswith('postgres://') and '+' not in url.split('://', 1)[0]:
        url = 'postgres+psycopg://' + url[len('postgres://') :]
    return url

def write_per_game(engine_url: str, df: pd.DataFrame, year: int) -> tuple[str, str]:
    stage = f"per_game_{year}_raw"
    final = f"per_game_{year}"
    engine = create_engine(_normalize_database_url(engine_url), pool_pre_ping=True, future=True)
    dtype: dict[str, Any] = {col: Text() for col in df.columns}
    df.to_sql(
        stage,
        engine,
        if_exists='replace',
        index=False,
        dtype=dtype,
        method='multi',
        chunksize=1000
    )
    with engine.begin() as conn:
        conn.exec_driver_sql(f"""
            DROP TABLE IF EXISTS {final};
            CREATE TABLE {final} AS
            SELECT
                "Player"::varchar AS player,
                NULLIF("Age",'')::int AS age,
                "Team"::varchar AS team,
                "Pos"::varchar AS pos,
                NULLIF("G",'')::int AS g,
                NULLIF("GS",'')::int AS gs,
                NULLIF("MP",'')::numeric AS mp,
                NULLIF("FG",'')::numeric AS fg,
                NULLIF("FGA",'')::numeric AS fga,
                NULLIF("fg_percent",'')::numeric AS fg_pct,
                NULLIF("3P",'')::numeric AS p3,
                NULLIF("3PA",'')::numeric AS p3a,
                NULLIF("3P_percent",'')::numeric AS p3_pct,
                NULLIF("2P",'')::numeric AS p2,
                NULLIF("2PA",'')::numeric AS p2a,
                NULLIF("2P_percent",'')::numeric AS p2_pct,
                NULLIF("eFG_percent", '')::numeric AS efg_pct,
                NULLIF("FT",'')::numeric AS ft,
                NULLIF("FTA",'')::numeric AS fta,
                NULLIF("FT_percent",'')::numeric AS ft_pct,
                NULLIF("ORB",'')::numeric AS orb,
                NULLIF("DRB",'')::numeric AS drb,
                NULLIF("TRB",'')::numeric AS trb,
                NULLIF("AST",'')::numeric AS ast,
                NULLIF("STL",'')::numeric AS stl,
                NULLIF("BLK",'')::numeric AS blk,
                NULLIF("TOV",'')::numeric AS tov,
                NULLIF("PF",'')::numeric AS pf,
                NULLIF("PTS",'')::numeric AS pts,
                COALESCE("Awards",'')::varchar AS awards
            FROM {stage}
            WHERE "Player" IS NOT NULL AND "Player" <> 'League Avarage';
        """)
    return stage, final