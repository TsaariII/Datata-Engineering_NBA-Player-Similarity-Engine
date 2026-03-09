from __future__ import annotations
import re
from typing import Iterable
import pandas as pd

_slug_re = re.compile(r"[^a-z0-9]+")
def player_key(name: str) -> str:
    """Deterministic, human-readable player id.
    """
    s = (name or '').strip().lower()
    s = _slug_re.sub('_', s).strip('_')
    return s

def _canonical_row(group: pd.DataFrame) -> pd.Series:
    """Pick one row for a player-season.

    Basketball-Reference per-game tables contain duplicates for traded players.
    Usually there is a TOT row that already aggregates the season.

    Priority:
    1) Team == 'TOT'
    2) Otherwise: row with max games played
    3) Otherwise: first row
    """
    if 'team' in group.columns:
        tot = group[group['team'].astype(str) == 'TOT']
        if not tot.empty:
            return tot.iloc[0]
    if 'g' in group.columns:
        idx = group['g'].fillna(0).astype(float).values.argmax()
        return group.iloc[idx]
    return group.iloc[0]

def build_player_seasons(per_game_df: pd.DataFrame, season: int) -> pd.DataFrame:
    df = per_game_df.copy()
    if "player" not in df.columns:
        raise ValueError('Expected column "player" in per_game dataframe')
    df["player"] = df["player"].astype(str)
    if "team" in df.columns:
        tot_rows = df[df["team"].astype(str) == "TOT"]
        non_tot = df[df["team"].astype(str) != "TOT"]
    else:
        tot_rows = pd.DataFrame(columns=df.columns)
        non_tot = df
    if "g" in non_tot.columns:
        non_tot = (
            non_tot
            .sort_values("g", ascending=False)
            .groupby("player", sort=False)
            .first()
            .reset_index()
        )
    else:
        non_tot = (
            non_tot
            .groupby("player", sort=False)
            .first()
            .reset_index()
        )
    if not tot_rows.empty:
        tot_players = set(tot_rows["player"].astype(str))
        non_tot = non_tot[~non_tot["player"].isin(tot_players)]
        canon = pd.concat([tot_rows, non_tot], ignore_index=True)
    else:
        canon = non_tot.reset_index(drop=True)
    canon["player_key"] = canon["player"].map(player_key)
    canon["player_name"] = canon["player"]
    canon["season"] = int(season)
    if "mp" in canon.columns:
        canon["mpg"] = canon["mp"]
    else:
        canon["mpg"] = pd.NA
    if "g" in canon.columns and "mp" in canon.columns:
        canon["total_minutes"] = canon["g"].astype(float) * canon["mp"].astype(float)
    else:
        canon["total_minutes"] = pd.NA
    keep_cols = [
        "player_key", "player_name", "season",
        "team", "pos", "g", "mp", "mpg", "total_minutes",
    ]
    keep_cols = [c for c in keep_cols if c in canon.columns]
    return canon[keep_cols + [c for c in canon.columns if c not in keep_cols]]