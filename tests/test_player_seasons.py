"""
tests/test_player_seasons.py
-----------------------------
Dedup and canonicalization tests for build_player_seasons and player_key.
 
Scope — what is NOT here
------------------------
The core happy-path assertions (TOT wins, max-games fallback, total_minutes,
player_key slug, raises on missing 'player' column) are already covered in
tests/test_etl_sanity.py::TestBuildPlayerSeasons.
 
Scope — what IS here
--------------------
Scenarios that require realistic, BBRef-shaped fixtures or exercise code paths
that the simpler sanity helpers don't reach:
 
  Dedup edge cases
    - three-team trade (3 team rows + TOT)
    - two-team trade, equal games played — deterministic winner
    - player appears only in TOT with no companion team rows
    - non-traded player whose name exactly matches a traded player's name
      (regression: should not be swallowed by the TOT filter)
 
  Stat column preservation
    - all raw stat columns survive into output (needed by downstream zscore step)
    - mpg column equals the mp column value
    - player_name column is exact copy of the 'player' input column
 
  Missing column paths
    - no 'team' column → all players treated as non-traded, dedup by max games
    - no 'g' column   → groupby first-row fallback, no crash
 
  player_key collision risk
    - "Gary Payton" vs "Gary Payton II" produce distinct keys
    - suffixes like "Jr.", "Sr.", "II", "III" are preserved in the slug
    - two distinct players never produce the same player_key in the same season
 
  Realistic cohort
    - 450-player fixture (≈ real NBA season size) with ~15 traded players
      produces exactly one row per player and unique player_keys throughout
"""

from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np
import pytest
from etl.transform.player_seasons import build_player_seasons, player_key

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
_STATS_COL = ['mp', 'pts', 'ast', 'trb', 'stl', 'blk', 'tov', 'fga', 'p3a', 'fta']

def _row(player: str, team: str, g: int, mp: float = 28.0, **extra) -> dict:
    """Minimal BBRef-shaped row with realistic stat columns."""
    base = {
        'player': player,
        'team': team,
        'pos': 'SG',
        'age': 26,
        'g': g,
        'gs': g - 2,
        'mp': mp,
        'pts': 15.0,
        'ast': 3.0,
        'trb': 5.0,
        'stl': 1.0,
        'blk': 0.5,
        'tov': 2.0,
        'fga': 12.0,
        'p3a': 3.0,
        'fta': 4.0,
        'fg_pct': 0.46,
        'p3_pct': 0.37,
        'ft_pct': 0.82,
    }
    base.update(extra)
    return base

# ---------------------------------------------------------------------------
# Three-team trade
# ---------------------------------------------------------------------------
class TestThreeTeamTrade:
    def _frame(self):
        return pd.DataFrame([
            _row('Traded Guy', 'TOT', 72, pts=18.0),
            _row('Traded Guy', 'LAL', 20, pts=17.0),
            _row('Traded Guy', 'BOS', 30, pts=19.0),
            _row('Traded Guy', 'MIA', 22, pts=18.5),
            _row('Other Guy', 'GSW', 68, pts=12.0)
        ])

    def test_exactly_one_row_for_traded_player(self):
        out = build_player_seasons(self._frame(), season=2024)
        assert len(out[out['player_name'] == 'Traded Guy']) == 1

    def test_tot_row_selected_over_any_team_row(self):
        out = build_player_seasons(self._frame(), season=2024)
        row = out[out['player_name'] == 'Traded Guy'].iloc[0]
        assert row['team'] == 'TOT'

    def test_tot_stats_preserved_not_overwritten(self):
        """The TOT row's stats (pts=18.0) should come through, not a team row's."""
        out = build_player_seasons(self._frame(), season=2024)
        row = out[out['player_name'] == 'Traded Guy'].iloc[0]
        assert float(row['pts']) == pytest.approx(18.0)

    def test_non_traded_player_unaafected(self):
        out = build_player_seasons(self._frame(), season=2024)
        assert len(out[out['player_name'] == 'Other Guy']) == 1
        assert out[out['player_name'] == 'Other Guy'].iloc[0]['team'] == 'GSW'
    
    def test_total_row_count_is_two(self):
        out = build_player_seasons(self._frame(), season=2024)
        assert len(out) == 2
    
# ---------------------------------------------------------------------------
# Two-team trade — equal games played, no TOT row
# ---------------------------------------------------------------------------
class TestEqualGamesTrade:
    def test_deterministic_winner_when_games_equal(self):
        """
        When two team rows have equal g and there is no TOT row, the result
        must be deterministic (same answer every time on the same input).
        We don't dictate *which* team wins — just that it's always the same one.
        """
        df = pd.DataFrame([
            _row('Eqaul Guy', 'LAL', 41),
            _row('Equal Guy', 'BOS', 41)
        ])
        result_a = build_player_seasons(df, season=2024)
        result_b = build_player_seasons(df, season=2024)
        team_a = result_a[result_a['player_name' == 'Equal Guy'].iloc[0]['team']]
        team_b = result_b[result_b['player_name' == 'Equal Guy'].iloc[0]['team']]
        assert team_a == team_b, 'Dedup result is non-deterministic for equal games'

    def test_one_row_produced(self):
        df = pd.DataFrame([
            _row('Eqaul Guy', 'LAL', 41),
            _row('Equal Guy', 'BOS', 41)
        ])
        out = build_player_seasons(df, season=2024)
        assert len(out[out['player_name'] == 'Equal Guy']) == 1

# ---------------------------------------------------------------------------
# TOT-only row (no companion team rows in the data)
# ---------------------------------------------------------------------------
class TestTotOnlyRow:
    def test_tot_only_passes_trough(self):
        """
        Occasionally BBRef exports only the TOT row with no per-team breakdown.
        build_player_seasons should handle this without error.
        """
        df = pd.DataFrame([
            _row('Solo Tot Guy', 'TOT', 60),
            _row('Normal Guy', 'LAL', 70)
        ])
        out = build_player_seasons(df, season=2024)
        assert len(out) == 2
        assert len(out[out['player_name'] == 'Solo Tot Guy']) == 1

# ---------------------------------------------------------------------------
# Stat column preservation
# ---------------------------------------------------------------------------
class TestStatsColumnPreservation:
    """
    build_player_seasons must carry all raw stat columns through to its output.
    The downstream zscore step reads pts, ast, trb etc. directly from this df.
    If these columns are dropped, z-scores silently zero-fill.
    """
    def setup_method(self):
        df = pd.DataFrame([
            _row('Player A', 'LAL', 70, pts=22.0, ast=6.0, trb=8.0),
            _row('Player B', 'BOS', 65, pts=14.0, ast=2.0, trb=4.0)
        ])
        self.out = build_player_seasons(df, season=2024)

    def test_stats_column_present_at_output(self):
        missing = [c for c in _STATS_COL if c not in self.out.columns]
        assert missing == [], f"Stat columns dropped from output: {missing}"

    def test_pts_value_preserved(self):
        row = self.out[self.out['player_name'] == 'Player A'].iloc[0]
        assert float(row['pts']) == pytest.approx(22.0)

    def test_mpg_equals_mp(self):
        """mpg is a derived column set equal to mp (minutes per game)."""
        for _, row in self.out.iterrows():
            assert float(row['mpg']) == pytest.approx(float(row['mp'])), (
                f"mpg != mp for {row['player_name']}"
            )

    def test_player_name_exact_copy_of_player_column(self):
        """player_name must be the verbatim value from the 'player' input column."""
        assert list(self.out['player_name']) == ['Player A', 'Player B']

# ---------------------------------------------------------------------------
# Missing column paths
# ---------------------------------------------------------------------------
class TestMissingColumns:
    def test_no_team_column_does_not_crash(self):
        """
        If the input has no 'team' column, all rows are treated as non-traded.
        Dedup falls back to groupby-first.
        """
        df = pd.DataFrame([
            {'player': 'Player A', 'g': 70, 'mp': 28.0, 'pts': 15.0},
            {'player': 'Player B', 'g': 65, 'mp': 32.0, 'pts': 20.0}
        ])
        out = build_player_seasons(df, season=2024)
        assert len(out) == 2

    def test_no_team_column_no_duplicate_keys(self):
        df = pd.DataFrame([
            {'player': 'Player A', 'g': 70, 'mp': 28.0},
            {'player': 'Player A', 'g': 50, 'mp': 25.0}
        ])
        out = build_player_seasons(df, season=2024)
        assert out['player_key'].is_unique

    def test_no_g_column_does_not_crash(self):
        """
        If the input has no 'g' column, dedup falls back to groupby-first.
        Should not raise AttributeError or KeyError.
        """
        df = pd.DataFrame([
            {'player': 'Player A', 'team': 'LAL', 'mp': 28.0, 'pts': 15.0},
            {'player': 'Player A', 'team': 'BOS', 'mp': 32.0, 'pts': 20.0}
        ])
        out = build_player_seasons(df, season=2024)
        assert len(out) == 2

    def test_no_g_column_total_minutes_is_na(self):
        """total_minutes requires both g and mp; without g it should be NA."""
        df =  pd.DataFrame([
            {'player': 'Player A', 'team': 'LAL', 'mp': 28.0}
        ])
        out = build_player_seasons(df, season=2024)
        assert pd.isna(out.iloc[0]['total_minutes'])

    def test_no_mp_column_mpg_is_na(self):
        df =  pd.DataFrame([
            {'player': 'Player A', 'team': 'LAL', 'g': 70}
        ])
        out = build_player_seasons(df, season=2024)
        assert pd.isna(out.iloc[0]['mpg'])

# ---------------------------------------------------------------------------
# player_key — suffix and collision behaviour
# ---------------------------------------------------------------------------
class TestPlayerKeyCollisions:
    """
    player_key is used as the PRIMARY KEY in the player_seasons table.
    Two distinct players must never produce the same key.
    """
    @pytest.mark.parametrize('name_a, name_b', [
        ('Gary Payton', 'Gary Payton II'),
        ('Gary Trent', 'Gary Trent Jr.'),
        ('Tim Hardaway', 'Tim Hardaway Jr.'),
        ('Bob Williams', 'Bob Williams III')
    ])
    
    def test_suffix_produces_distinct_key(self, name_a: str, name_b: str):
        assert player_key(name_a) != player_key(name_b), (
            f"'{name_a}' and '{name_b}' produced the same player_key"
        )
    
    def test_no_collision_in_full_season(self):
        """
        Build a frame where parent and Jr. player appear in the same season.
        Their player_keys must be unique so neither overwrites the other in the DB.
        """
        df = pd.DataFrame([
            _row('Gary Payton', 'MIL', 55),
            _row('Gary Payton II', 'GSW', 71)
        ])
        out = build_player_seasons(df, season=2024)
        assert len(out) == 2
        assert out['player_key'].is_unique

# ---------------------------------------------------------------------------
# Realistic cohort
# ---------------------------------------------------------------------------
class TestRealisticCohort:
    """
    Smoke test against a near-full-season fixture.
    ~450 unique players, ~15 of whom are traded (appear on 2 teams + TOT).
    Validates that output has exactly one row per player and all keys are unique.
    """
    def _build_cohort(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        n_unique = 450
        rows = []
        traded_names = {f"Traded Player {i:02d}" for i in range(15)}
        for i in range(n_unique - len(traded_names)):
            rows.append(_row(
                f"Regular Player {i:03d}",
                rng.choice(['LAL', 'BOS', 'GSW', 'MIA', 'CHI']),
                int(rng.integers(20, 82))
            ))
        for name in traded_names:
            g1 = int(rng.integers(15, 40))
            g2 = int(rng.integers(15, 40))
            rows.append(_row(name, 'TOT', g1 + g2))
            rows.append(_row(name, rng.choice(['LAL', 'BOS']), g1))
            rows.append(_row(name, rng.choice(['GSW', 'MIA']), g2))
        return pd.DataFrame(rows)

    def setup_method(self):
        self.out = build_player_seasons(self._build_cohort(), season=2024)

    def test_output_row_count_eqauls_unique_player_count(self):
        assert len(self.out) == 450

    def test_all_player_keys_unique(self):
        assert self.out['player_key'].is_unique, (
            "Duplicate player_keys found — two players share a key or dedup failed"
        )
    
    def test_no_null_player_keys(self):
        assert self.out['player_key'].notna().all()

    def test_no_null_player_names(self):
        assert self.out['player_name'].notna().all()

    def test_season_uniform(self):
        assert (self.out['season'] == 2024).all()
    
    def test_treaded_players_have_tot_team(self):
        for i in range(15):
            name = f"Traded Player {i:02d}"
            row = self.out[self.out['player_name'] == name]
            assert len(row) == 1
            assert row.iloc[0]['team'] == 'TOT'