"""Per-game similarity feature set (v1).

Keep this list stable. If you change it, copy this file to per_game_v2.py,
update FEATURE_SET, and keep v1 intact for reproducibility.
"""

FEATURE_SET = "per_game_v1"

# Curated per-game features. These are columns coming from per_game_{year}.
# Notes:
# - We keep attempt/volume stats because they matter for role similarity.
# - We include percentages but they can be noisy for low-minute players.
FEATURES = [
    "mp",      # minutes per game
    "pts",
    "ast",
    "trb",
    "stl",
    "blk",
    "tov",
    "pf",
    "fga",
    "p3a",
    "fta",
    "fg_pct",
    "p3_pct",
    "ft_pct",
    "efg_pct",
]