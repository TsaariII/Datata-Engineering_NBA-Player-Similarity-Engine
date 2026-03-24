from __future__ import annotations
import numpy as np
import pandas as pd

# Minimum FGA threshold to compute shooting ratios.
# Below this the ratio is set to NaN so the player isn't penalised for noise.
_MIN_FGA = 1.0

# Minimum minutes per game to compute per-minute proxies.
_MIN_MP = 1.0

def _safe_div(num: pd.Series, denom: pd.Series, min_denom: float = 0.0) -> pd.Series:
    """Element-wise division; returns NaN wherever denom <= min_denom."""
    mask = denom > min_denom
    result = pd.Series(np.nan, index=num.index, dtype=float)
    result[mask] = num[mask].astype(float) / denom[mask].astype(float)
    return result

def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived advanced-stat columns to a player-seasons dataframe.
 
    Input columns used (all present after etl/load/postgres.py):
        pts, fga, fta, ft, p3, p3a, p2, p2a,
        ast, tov, orb, drb, trb, stl, blk, mp, g
 
    New columns added:
        ts_pct          True Shooting %
        p3a_rate        3-Point Attempt Rate
        ftr             Free Throw Rate
        ast_pct_proxy   Assist % proxy
        tov_pct         Turnover %
        orb_pct_proxy   Offensive Rebound % proxy
        drb_pct_proxy   Defensive Rebound % proxy
        usg_proxy       Usage Rate proxy (per-minute)
    """
    df = df.copy()
    def col(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors='coerce').astype(float)
        return pd.Series(np,nan, index=df.index, dtype=float)
    
    pts = col('pts')
    fga  = col('fga')
    fta  = col('fta')
    ft   = col('ft')
    p3   = col('p3')
    p3a  = col('p3a')
    ast  = col('ast')
    tov  = col('tov')
    orb  = col('orb')
    drb  = col('drb')
    trb  = col('trb')
    mp   = col('mp')

    # ------------------------------------------------------------------
    # True Shooting %
    #
    # TS% = PTS / (2 * (FGA + 0.44 * FTA))
    #
    # Measures how efficiently a player converts possessions into points.
    # Weights 3-pointers and free throws correctly unlike FG%.
    # ------------------------------------------------------------------
    ts_denom = 2.0 * (fga + 0.44 * fta)
    df['ts_pct'] = _safe_div(pts, ts_denom, min_denom=_MIN_FGA)
    # ------------------------------------------------------------------
    # 3-Point Attempt Rate
    #
    # 3PAr = 3PA / FGA
    #
    # Captures shooting *style* — high values = perimeter shooter,
    # low values = paint/mid-range player.
    # ------------------------------------------------------------------
    df['p3a_rate'] = _safe_div(p3a, fga, min_denom=_MIN_FGA)

    # ------------------------------------------------------------------
    # Free Throw Rate
    #
    # FTr = FTA / FGA
    #
    # High FTr = player draws contact and gets to the line (paint aggression
    # or foul-drawing guards). Low FTr = jump shooter.
    # ------------------------------------------------------------------
    df['ftr'] = _safe_div(fta, fga, min_denom=_MIN_FGA)
 
    # ------------------------------------------------------------------
    # AST% proxy
    #
    # Full formula needs team FGM while player is on court (unavailable here).
    # Proxy: AST / (FGA + 0.44*FTA + AST + TOV)
    #
    # This is the player's share of their own "offensive actions" that
    # are assists — a reasonable playmaking intensity measure.
    # ------------------------------------------------------------------
    ast_denom = fga + 0.44 * fta + ast + tov
    df['ast_pct_proxy'] = _safe_div(ast, ast_denom, min_denom=_MIN_FGA)
 
    # ------------------------------------------------------------------
    # Turnover %
    #
    # TOV% = TOV / (FGA + 0.44*FTA + TOV)
    #
    # Share of offensive possessions that end in a turnover.
    # ------------------------------------------------------------------
    tov_denom = fga + 0.44 * fta + tov
    df['tov_pct'] = _safe_div(tov, tov_denom, min_denom=_MIN_FGA)
 
    # ------------------------------------------------------------------
    # ORB% proxy
    #
    # Full formula needs team and opponent rebound totals (unavailable here).
    # Proxy: ORB / TRB (offensive share of own total rebounds)
    #
    # Distinguishes offensive rebounders from defensive rebounders.
    # ------------------------------------------------------------------
    df['orb_pct_proxy'] = _safe_div(orb, trb, min_denom=0.5)
 
    # ------------------------------------------------------------------
    # DRB% proxy
    #
    # Proxy: DRB / TRB (defensive share of own total rebounds)
    # ------------------------------------------------------------------
    df['drb_pct_proxy'] = _safe_div(drb, trb, min_denom=0.5)
 
    # ------------------------------------------------------------------
    # Usage proxy
    #
    # Full USG% = (FGA + 0.44*FTA + TOV) / (MP/TeamMP * TeamPoss)
    # which needs team possessions.
    #
    # Proxy: (FGA + 0.44*FTA + TOV) / MP
    #
    # Measures how many "possessions used" per minute — a scale-free
    # usage indicator that doesn't depend on team data.
    # ------------------------------------------------------------------
    usg_num = fga + 0.44 * fta + tov
    df['usg_proxy'] = _safe_div(usg_num, mp, min_denom=_MIN_MP)

    return df