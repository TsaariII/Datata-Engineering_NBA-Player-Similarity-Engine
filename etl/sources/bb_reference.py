import os
import time
import logging
import random
import subprocess
from typing import Optional, Dict

import requests
from bs4 import BeautifulSoup
import pandas as pd
from sqlalchemy import create_engine, Integer, Float, Text

logger = logging.getLogger(__name__)

BASE_URL = 'https://www.basketball-reference.com/'
DEFAULT_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        '(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml:q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Request': '1',
    'Referer': BASE_URL
}

def per_game_url(year: int) -> str:
    return f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"

def bot_block(html: str) -> bool:
    """Heuristics for "you need JS/cookies" pages."""
    h = (html or '').lower()
    needles = [
        'enable javascript',
        'cookies to continue',
        'access denied',
        'request blocked',
        'forbidden'
    ]
    return any(n in h for n in needles)

def _fetch_html_via_curl(url: str, headers: Dict[str, str]) -> str:
    """Fallback that uses curl (installed in the Docker image) to fetch HTML.

    Why: some anti-bot systems key off the TLS/client fingerprint used by
    python-requests. curl often passes where requests fails.
    """
    cmd = ['curl', '-L', '--compressed', '-sS', '--fail']
    for k, v in headers.items():
        cmd += ['-H', f"{k}: {v}"]
    cmd += [url]
    return subprocess.check_output(cmd, text=True)

def fetch_html(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 3,
    backoff: float = 1.5
) -> str:
    headers = {**DEFAULT_HEADERS, **(headers or {})}
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            try:
                s = requests.Session()
                try:
                    s.get(BASE_URL, timeout=20)
                except Exception:
                    pass
                resp = s.get(url, timeout=30)
            finally:
                s.close()
            if resp.status_code == 403 or bot_block(resp.text):
                raise requests.HTTPError(
                    f"Blocked by Basketball-Reference (status={resp.status_code})."
                )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            last_exc = e
            sleep = (backoff ** attempt) + random.random() * 0.5
            logger.warning(
                'GET %s failed (attempt %d/%d): %s; retrying in %.1fs',
                url,
                attempt,
                retries,
                e,
                sleep
            )
            time.sleep(sleep)
    try:
        logger.warning('Falling back to curl for %s', url)
        return _fetch_html_via_curl(url, headers)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch {url}") from (e or last_exc)

def parse_per_game_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'id': 'per_game_stats'})
    if table is None:
        raise ValueError('Could not find table')
    headers = [th.get_text() for th in table.find('thead').find_all('th')][1:]
    rows = table.find('tbody').find_all('tr')
    player_stats = []
    for row in rows:
        if row.find('th', {'scope': 'row'}) is None:
            continue
        stats = [td.get_text() for td in row.find_all('td')]
        if stats:
            player_stats.append(stats)
    df = pd.DataFrame(player_stats, columns=headers)
    return df

def fetch_per_game_df(year: int) -> pd.DataFrame:
    disable_cache = os.getenv('NBA_PER_GAME_DISABLE_CACHE', '').strip().lower() in {
        '1', 'true', 'yes', 'y'
    }
    env_csv = os.getenv('NBA_PER_GAME_CSV')
    def_csv = os.path.join('data', f"NBA_per_game_{year}.csv")
    if not disable_cache:
        if env_csv:
            if os.path.exists(env_csv):
                logger.info('Loading per-game stats from %s', env_csv)
                return pd.read_csv(env_csv)
            logger.warning('NBA_PER_GAME_CSV is set but file does not exist: %s', env_csv)
        if os.path.exists(def_csv):
                logger.info('Loading per-game stats from %s', env_csv)
                return pd.read_csv(def_csv)
    html = fetch_html(per_game_url(year))
    return parse_per_game_table(html)