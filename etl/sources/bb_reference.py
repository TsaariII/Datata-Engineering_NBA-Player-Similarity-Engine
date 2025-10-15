import os
import time
import logging
from typing import Optional, Dict

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, Integer, Float, Text

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
    return df

def fetch_per_game_df(year: int) -> pd.DataFrame:
    html = fetch_html(per_game_url(year))
    return parse_per_game_table(html)