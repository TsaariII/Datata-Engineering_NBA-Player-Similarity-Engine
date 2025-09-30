import os
import time
import logging
from typing import Tuple, Optional, Dict

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, Integer, Float, Text
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; nba-similarity-etl/1.0; +https://example.local)"
}

def per_game_url(year: int) -> str:
    return f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"

def fetch_html(url: str, headers: Optional[Dict[str, str]] = None, retries: int = 3, backoff: float = 1.5) -> str:
    headers = {**DEFAULT_HEADERS, **(headers or {})}
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:
            last_exc = exc
            sleep = backoff ** attempt
            logger.warning("GET %s failed (attempt %d/%d): %s; retrying in %.1fs", url, attempt, retries, exc, sleep)
            time.sleep(sleep)
    raise RuntimeError(f"Failed to fetch {url}") from last_exc

def parse_per_game_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"id": "per_game_stats"})
    if table is None:
        raise ValueError("Could not find per_game_stats table")

    headers = [th.get_text() for th in table.find("thead").find_all("th")][1:]  # skip rank
    rows = table.find("tbody").find_all("tr")

    player_stats = []
    for row in rows:
        if row.find("th", {"scope": "row"}) is None:
            continue
        stats = [td.get_text() for td in row.find_all("td")]
        if stats:
            player_stats.append(stats)

    df = pd.DataFrame(player_stats, columns=headers)

    # Normalize column names
    rename_map = {
        "FG%": "FG_percent",
        "3P%": "3P_percent",
        "2P%": "2P_percent",
        "eFG%": "eFG_percent",
        "FT%": "FT_percent",
        "3P": "3P",
        "3PA": "3PA",
        "2P": "2P",
        "2PA": "2PA",
    }
    df = df.rename(columns=rename_map)
    return df

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # Match your previous explicit CASTs, but done in pandas
    int_cols = ["Age", "G", "GS"]
    float_cols = [
        "MP","FG","FGA","FG_percent","3P","3PA","3P_percent","2P","2PA",
        "2P_percent","eFG_percent","FT","FTA","FT_percent","ORB","DRB","TRB",
        "AST","STL","BLK","TOV","PF","PTS"
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Some tables use "Tm" not "Team". Align to "Team".
    if "Tm" in df.columns and "Team" not in df.columns:
        df = df.rename(columns={"Tm": "Team"})
    return df

def save_csv(df: pd.DataFrame, csv_dir: str, year: int) -> str:
    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, f"nba_per_game_{year}.csv")
    df.to_csv(path, index=False)
    logger.info("Saved CSV to %s", path)
    return path

def save_sql(df: pd.DataFrame, db_url: str, year: int):
    engine = create_engine(db_url, pool_pre_ping=True, future=True)

    base_table = f"player_per_game_{year}"
    clean_table = f"player_per_game_{year}_cleaned"

    # dtype hints so Postgres doesn’t guess weird types
    dtype_map = {
        "Player": Text(),
        "Pos": Text(),
        "Team": Text(),
        "Age": Float(),     # nullable in pandas; use Float to avoid cast issues
        "G": Float(),
        "GS": Float(),
        # numeric columns default to Float if present
    }
    # widen map for any column that looks numeric
    for c in df.columns:
        if c not in dtype_map:
            dtype_map[c] = Float()

    # raw
    df.to_sql(
        base_table, engine, if_exists="replace", index=False,
        dtype=dtype_map, method="multi", chunksize=1000
    )

    cleaned = coerce_types(df.copy())

    # Ensure floats are regular float64 so psycopg doesn’t choke on pandas NA
    for c in cleaned.columns:
        if str(cleaned[c].dtype).startswith(("Float", "float", "Int")):
            cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce").astype("float64")

    cleaned.to_sql(
        clean_table, engine, if_exists="replace", index=False,
        dtype=dtype_map, method="multi", chunksize=1000
    )

    return base_table, clean_table

def run_etl(year: int, db_url: str = "sqlite:///data/nba_stats.db", csv_dir: str = "data") -> Dict[str, str]:
    url = per_game_url(year)
    html = fetch_html(url)
    df = parse_per_game_table(html)
    csv_path = save_csv(df, csv_dir, year)
    base_table, clean_table = save_sql(df, db_url, year)
    return {
        "csv_path": csv_path,
        "base_table": base_table,
        "clean_table": clean_table,
        "db_url": db_url,
    }
