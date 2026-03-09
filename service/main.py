from __future__ import annotations
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import create_engine, text
from service.similarity_repo import top_k_similar
from etl.features.per_game_v1 import FEATURE_SET

app = FastAPI(title="NBA Player Similarity Engine")

def _normalize_database_url(url: str) -> str:
    """Normalize common Postgres URL variants.
    - docker-compose examples often use postgres://
    - SQLAlchemy + psycopg (v3) wants postgresql+psycopg://
    """
    url = url.strip()
    if url.startswith("postgres://") and "+" not in url.split("://", 1)[0]:
        url = "postgres+psycopg://" + url[len("postgres://"):]
    return url


@lru_cache(maxsize=1)
def engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(_normalize_database_url(db_url), pool_pre_ping=True, future=True)


def _table_name_for_year(year: int) -> str:
    if year < 1947 or year > 2100:
        raise HTTPException(status_code=400, detail="Year seems wrong")
    return f"per_game_{year}"


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        with engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/players")
def players(
    year: int = Query(2024, ge=1947, le=2100),
    limit: int = Query(50, ge=1, le=500),
    q: Optional[str] = Query(None, description="case-insensitive substring match"),
) -> Dict[str, Any]:
    tbl = _table_name_for_year(year)
    sql = f"SELECT player FROM {tbl}"
    params: Dict[str, Any] = {}
    if q:
        sql += " WHERE player ILIKE :q"
        params["q"] = f"%{q}%"
    sql += " ORDER BY player ASC LIMIT :limit"
    params["limit"] = limit
    try:
        with engine().connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
            return {"year": year, "players": [r[0] for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/player/{name}")
def player_stats(name: str, year: int = Query(2024, ge=1947, le=2100)) -> Dict[str, Any]:
    tbl = _table_name_for_year(year)
    try:
        with engine().connect() as conn:
            row = conn.execute(
                text(f"SELECT * FROM {tbl} WHERE player = :p"),
                {"p": name},
            ).mappings().first()
        if not row:
            raise HTTPException(status_code=404, detail="Player not found")
        return {"year": year, "player": name, "stats": dict(row)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar/{name}")
def similar_players(
    name: str,
    year: int = Query(2024, ge=1947, le=2100),
    k: int = Query(10, ge=1, le=50),
    feature_set: str = Query(FEATURE_SET, description="Feature set version to use"),
) -> Dict[str, Any]:
    """Return top-k similar players based on precomputed z-score feature vectors.

    Reads from player_season_features — no per-request normalization.
    Run `build_features --year {year}` first to populate the feature table.
    """
    try:
        results = top_k_similar(
            engine=engine(),
            season=year,
            feat_set=feature_set,
            player_name=name,
            k=k,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"year": year, "player": name, "feature_set": feature_set, "top_k": results}