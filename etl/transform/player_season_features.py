from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeaturePack:
    feature_set: str
    feature_names: List[str]


def zscore_features(
    player_seasons_df: pd.DataFrame,
    pack: FeaturePack,
) -> pd.DataFrame:
    """Compute per-season z-scores for a curated feature list.

    - z = (x - mean) / std, computed across players for that season
    - std==0 gets coerced to 1 so you don't divide by 0
    - NaNs become 0 after z-scoring (meaning "league average")

    Returns a dataframe ready to upsert into player_season_features.
    """
    df = player_seasons_df.copy()
    required = {"player_key", "season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"player_seasons_df missing columns: {sorted(missing)}")
    # Pull numeric features.
    feat_df = df.reindex(columns=pack.feature_names)
    # Coerce to floats.
    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")
    mu = feat_df.mean(axis=0, skipna=True)
    sigma = feat_df.std(axis=0, skipna=True).replace(0, 1)
    z = (feat_df - mu) / sigma
    z = z.fillna(0.0)
    # Build outputs.
    out = pd.DataFrame(
        {
            "player_key": df["player_key"].astype(str),
            "season": df["season"].astype(int),
            "feature_set": pack.feature_set,
            "feature_names": [pack.feature_names] * len(df),
            "z_scores": z.apply(lambda r: {k: float(r[k]) for k in pack.feature_names}, axis=1),
            "z_vector": z.apply(lambda r: [float(r[k]) for k in pack.feature_names], axis=1),
        }
    )

    return out