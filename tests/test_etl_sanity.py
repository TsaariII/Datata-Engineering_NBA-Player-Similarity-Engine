"""
tests/test_etl_sanity.py
------------------------
Fast, no-DB sanity checks for the ETL transform layer.
 
All tests here run against pure Python/pandas — no database, no Docker,
no network.  They should pass in under a second on any machine.
 
Coverage
--------
zscore_features:
  - output schema (required columns present)
  - z_vector length matches feature count
  - z_scores dict keys match feature list
  - column means are near zero (z-score invariant)
  - no NaN or Inf values in z_vector
  - zero-variance column coerced to 1 (no ZeroDivisionError)
  - NaN inputs become 0.0 (league-average fill)
  - missing feature columns zero-fill rather than crash
  - single-player edge case (std = 0 everywhere)
  - player_key and season survive unchanged
 
derive_features:
  - all expected output columns are present
  - column names match what advanced_v1.FEATURES references   ← catches name mismatch bug
  - ts_pct is in (0, 1] for normal inputs
  - p3a_rate is in [0, 1] for normal inputs
  - ftr is >= 0 for normal inputs
  - players below _MIN_FGA threshold get NaN (not garbage values)
  - no crash on all-zero or all-NaN input rows
 
build_player_seasons / player_key:
  - player_key is deterministic and slug-safe
  - player_key strips special characters
  - traded player (TOT row present) → one canonical row
  - traded player (no TOT row) → row with most games wins
  - non-traded player → passes through unchanged
  - season column is stamped correctly
  - total_minutes = g * mp
"""

from __future__ import annotations
import math
from typing import List
import numpy as np
import pandas as pd
import pytest
from etl.features.advanced_v1 import FEATURE_SET as ADV_FEAT_SET
from etl.features.advanced_v1 import FEATURES as ADV_FEAT
from etl.features.per_game_v1 import FEATURE_SET as PG_FEAT_SET
from etl.features.per_game_v1 import FEATURES as PG_FEAT
from etl.transform.derive_features import derive_features
from etl.transform.player_season_features import FeaturePack, zscore_features
from etl.transform.player_seasons import build_player_seasons, player_key

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------
def _make_seasons(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Build a minimal fake player-seasons DataFrame with realistic column names.
    Covers all raw columns consumed by both derive_features and zscore_features.
    """
    rng = np.random.default_rng(seed)
    fga = rng.uniform(5, 20, n)
    fta = rng.uniform(1, 6, n)
    orb = rng.uniform(0, 3, n)
    drb = rng.uniform(1, 7, n)
    return pd.DataFrame(
        {
        'player_key': [f"player_{i:03d}" for i in range(n)],
        'player_name': [f"Player {i}" for i in range(n)],
        'season': [2024] * n,
        'mp': rng.uniform(10, 36, n),
        'pts': fga * rng.uniform(0.8, 1.8, n) + fta * rng.uniform(0.6, 0.9, n),
        'fga': fga,
        'fta': fta,
        'ft': fta * rng.uniform(0.6, 0.9, n),
        'p3a': rng.uniform(0, fga * 0.6),
        'p3': rng.uniform(0, 4, n),
        'ast': rng.uniform(0, 10, n),
        'tov': rng.uniform(0.5, 5, n),
        'orb': orb,
        'drb': drb,
        'trb': orb + drb,
        'stl': rng.uniform(0, 3, n),
        'blk': rng.uniform(0, 3, n),
        'g': rng.uniform(20, 82, n)
        }
    )

def _make_per_game_pack(feats: List[str] = PG_FEAT) -> FeaturePack:
    return FeaturePack(feature_set=PG_FEAT_SET, feature_names=feats)

def _make_adv_pack(feats: List[str] = ADV_FEAT) -> FeaturePack:
    return FeaturePack(feature_set=ADV_FEAT_SET, feature_names=feats)

# ---------------------------------------------------------------------------
# zscore_features — output schema
# ---------------------------------------------------------------------------
class TestZscoreOutputSchame:
    def test_required_columns(self):
        df = _make_seasons()
        out = zscore_features(df, _make_per_game_pack())
        for col in ('player_key', 'season', 'feature_set', 'feature_names', 'z_scores', 'z_vector'):
            assert col in out.columns, f"Missing column: {col}"

    def test_row_count_mtches_input(self):
        df = _make_seasons(n=50)
        out = zscore_features(df, _make_per_game_pack())
        assert len(out) == 50
    
    def test_features_set_label_stored(self):
        df = _make_seasons()
        out = zscore_features(df, _make_per_game_pack())
        assert (out['feature_set'] == PG_FEAT_SET).all()
    
    def test_feature_names_list_stored(self):
        df = _make_seasons()
        out = zscore_features(df, _make_per_game_pack())
        for names in out['feature_names']:
            assert names == PG_FEAT

    def test_z_vector_length_matches_feature_count(self):
        df = _make_seasons()
        pack = _make_per_game_pack()
        out = zscore_features(df, pack)
        excepted_len = len(pack.feature_names)
        for vec in out['z_scores']:
            assert len(vec) == excepted_len, (
                f"z_vector length {len(vec)} != feature count {excepted_len}"
            )

    def test_z_scores_dict_keays_match_features(self):
        df = _make_seasons()
        pack = _make_per_game_pack()
        out = zscore_features(df, pack)
        for d in out['z_scores']:
            assert set(d.keys()) == set(pack.feature_names)
    
    def test_player_key_passtrough(self):
        df = _make_seasons(n=10)
        out = zscore_features(df, _make_per_game_pack())
        assert list(out['player_key']) == list(df['player_key'].astype(str))

    def test_season_passtrough(self):
        df = _make_seasons(n=10)
        out = zscore_features(df, _make_per_game_pack())
        assert (out['season'] == 2024).all()

    def test_raises_on_missing_player_key(self):
        df = _make_seasons().drop(columns=['player_key'])
        with pytest.raises(ValueError, match='missing columns'):
            zscore_features(df, _make_per_game_pack())
    
    def test_raises_on_missing_season(self):
        df = _make_seasons().drop(columns=['season'])
        with pytest.raises(ValueError, match='missing columns'):
            zscore_features(df, _make_per_game_pack())


# ---------------------------------------------------------------------------
# zscore_features — numeric invariants
# ---------------------------------------------------------------------------
class TestZscoreNumerics:
    def test_column_means_near_zero(self):
        """z-score invariant: mean of each z-scored column should be ~0."""
        df = _make_seasons(n=100)
        out = zscore_features(df, _make_per_game_pack())
        matrix = np.array(out['z_vector'].tolist())
        col_means = matrix.mean(axis=0)
        for i, feat in enumerate(PG_FEAT):
            assert abs(col_means[i]) < 0.1, (
                f"Column mean for '{feat}' is {col_means[i]:.4f}, expected ~0"
            )

    def test_no_nan_in_z_vector(self):
        df = _make_seasons(n=50)
        out = zscore_features(df, _make_per_game_pack())
        for i, vec in enumerate(out['z_vector']):
            for j, v in enumerate(vec):
                assert not math.isnan(v), (
                    f"NaN in z_vector at row {i}, feature index {j} "
                    f"({PG_FEAT[j]})"
                )

    def test_no_inf_in_z_vectors(self):
        df = _make_seasons(n=50)
        out = zscore_features(df, _make_per_game_pack())
        for vec in out['z_vector']:
            assert not any(math.isinf(v) for v in vec)
    
    def test_no_nan_in_z_scores_dict(self):
        df = _make_seasons(n=50)
        out = zscore_features(df, _make_per_game_pack())
        for d in out['z_scores']:
            for k, v in d.items():
                assert not math.isnan(v), f"NaN in z_scores[{k}]"

# ---------------------------------------------------------------------------
# zscore_features — edge cases
# ---------------------------------------------------------------------------
class TestZscoreEdgeCases:
    def test_zero_variance_column_does_not_crash(self):
        """If all players have the same value, std=0 → coerced to 1, z=0 for all."""
        df = _make_seasons(n=20)
        df['mp'] = 30.0
        out = zscore_features(df, _make_per_game_pack())
        mp_idx = PG_FEAT.index('mp')
        for vec in out['z_vector']:
            assert vec[mp_idx] == pytest.approx(0.0), (
                'Constant feature should produce z=0, not error or garbage'
            )
    
    def test_nan_inputs_become_zero(self):
        """NaN inputs should be filled with 0.0 (league-average convention)."""
        df = _make_seasons(n=20)
        df.loc[0, 'pts'] = float('nan')
        out = zscore_features(df, _make_per_game_pack())
        pts_idx = PG_FEAT.index('pts')
        first_vec = out['z_vector'].iloc[0]
        assert not math.isnan(first_vec[pts_idx]), (
            'NaN input should be filled to 0.0 in z_vector, not stay NaN'
        )
    
    def test_missing_feature_column_zero_fills(self):
        """
        If a feature in the pack is not present in the dataframe,
        zscore_features should zero-fill rather than crash.
        This mirrors the build_features.py warning path.
        """
        df = _make_seasons(n=20)
        df = df.drop(columns=['p3a'], errors='ignore')
        pack = _make_per_game_pack()
        out = zscore_features(df, pack)
        p3a_idx = PG_FEAT.index('p3a')
        for vec in out['z_vector']:
            assert vec[p3a_idx] == pytest.approx(0.0)
    
    def test_single_player_does_not_crash(self):
        """
        With only one player, std=0 for all columns.
        Should produce a zero vector, not a ZeroDivisionError.
        """
        df = _make_seasons(n=1)
        out = zscore_features(df, _make_per_game_pack())
        assert len(out) == 1
        for vec in out['z_vector']:
            for v in vec:
                assert not math.isnan(v) and not math.isinf(v)

    def test_two_players_opposite_z_scores(self):
        """
        Two players with very different stats should produce opposite-sign z-scores.
        """
        df = pd.DataFrame(
            {
                'player_key': ['low_scorer', 'high_scorer'],
                'season': [2024, 2024],
                'mp': [10.0, 35.0],
                'pts': [10.0, 35.0],
                'ast': [1.0, 8.0],
                'trb': [2.0, 10.0],
                'stl': [0.3, 2.0],
                'blk': [0.1,  2.0],
                'tov': [0.5,  4.0],
                'pf':  [1.0,  3.0],
                'fga': [3.0,  18.0],
                'p3a': [0.5,  7.0],
                'fta': [0.5,  6.0],
                'fg_pct':  [0.40, 0.52],
                'p3_pct':  [0.30, 0.38],
                'ft_pct':  [0.65, 0.80],
                'efg_pct': [0.42, 0.55]
            }
        )
        out = zscore_features(df, _make_per_game_pack())
        vecs = out['z_vector'].tolist()
        pts_idx = PG_FEAT.index('pts')
        assert vecs[1][pts_idx] > 0
        assert vecs[0][pts_idx] < 0

# ---------------------------------------------------------------------------
# derive_features — output column names
# ---------------------------------------------------------------------------
class TestDeriveFeatureColumnNames:
    """
    Validates that derive_features produces the exact column names that
    advanced_v1.FEATURES references.
    """
    def setup_method(self):
        self.df = _make_seasons(n=30)
        self.derived = derive_features(self.df)

    def test_all_derived_columns_present(self):
        expected = ['mp', 'usg_proxy', 'ts_pct', 'p3a_rate', 'ftr',
                    'ast_pct_proxy', 'tov_pct', 'orb_pct_proxy', 'drb_pct_proxy',
                    'stl', 'blk'
        ]
        missing = [c for c in expected if c not in self.derived.columns]
        assert missing == [], f"Missing columns: {missing}"
    
    def test_all_advanced_v1_features_resolvable_after_derive(self):
        """
        All features listed in advanced_v1.FEATURES must be present as columns
        after derive_features runs.  A missing column means the feature set
        references a name that doesn't exist, so z-scores will silently be 0.
        """
        missing = [f for f in ADV_FEAT if f not in self.derived.columns]
        assert missing == [], (
            f"advanced_v1.FEATURES references columns not produced by "
            f"derive_features: {missing}"
        )

# ---------------------------------------------------------------------------
# derive_features — numeric sanity
# ---------------------------------------------------------------------------
class TestDeriveFeatureNumerics:
    def setup_method(self):
        self.df = _make_seasons(n=50)
        self.derived = derive_features(self.df)

    def test_ts_pct_in_valid_range(self):
        """TS% should be in (0, 1] for players with meaningful shot attempts."""
        valid = self.derived['ts_pct'].dropna()
        assert (valid > 0).all(), 'ts_pct should be positive'
        assert (valid <= 1.05).all(), 'ts_pct should not exceed 1.05 for real players'

    def test_p3a_rate_in_zero_to_one(self):
        valid = self.derived['p3a_rate'].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.0).all()

    def test_ftr_non_negative(self):
        valid = self.derived['ftr'].dropna()
        assert (valid >= 0).all()

    def test_orb_pct_proxy_in_zero_to_one(self):
        valid = self.derived['orb_pct_proxy'].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1.0).all()
    
    def test_drb_pct_proxy_in_zero_to_one(self):
        valid = self.derived['drb_pct_proxy'].dropna()
        assert (valid >= 0).all()
        assert (valid <= 1).all()
    
    def test_low_fga_player_gets_nan_not_garbage(self):
        """
        Players below _MIN_FGA should get NaN for shooting ratios,
        not divide-by-near-zero garbage values.
        """
        df = _make_seasons(n=5)
        df.loc[0, 'fga'] = 0.5
        df.loc[0, 'fta'] = 0.0
        derived = derive_features(df)
        assert pd.isna(derived.loc[0, 'ts_pct']), (
            'Player with fga=0.5 should have NaN ts_pct, not a computed value'
        )
    
    def test_all_zero_row_does_not_crash(self):
        """A row of all zeros (rare but possible) should not raise."""
        df = _make_seasons(n=5)
        for col in ['pts', 'fga', 'fta', 'p3a', 'ast', 'tov', 'orb', 'drb', 'trb', 'mp']:
            if col in df.columns:
                df.loc[0, col] = 0.0
        derive_features(df)

    def test_all_nan_row_does_not_crash(self):
        """A row of all NaN values should not raise."""
        df = _make_seasons(n=5)
        for col in ['pts', 'fga', 'fta', 'p3a', 'ast', 'tov', 'orb', 'drb', 'trb', 'mp']:
            if col in df.columns:
                df.loc[0, col] = float('nan')
        derive_features(df)

# ---------------------------------------------------------------------------
# player_key
# ---------------------------------------------------------------------------
class TestPlayerKey:
    def test_basic_name(self):
        assert player_key('Lebron James') == 'lebron_james'

    def test_lowercase(self):
        assert player_key('STEPHEN CURRY') == 'stephen_curry'
    
    def test_special_characters_stripped(self):
        result = player_key('Luka Dončić')
        assert result == result.replace(' ', '_')
        assert all(c.isalnum() or c == '_' for c in result)

    def test_multiple_spaces_collapsed(self):
        assert '_' not in player_key('A  B').replace('a_b', '')
    
    def test_deterministic(self):
        assert player_key('Kevin Durant') == player_key('Kevin Durant')
    
    def test_empty_string(self):
        result = player_key('')
        assert isinstance(result, str)

    def test_no_leading_trailing_underscores(self):
        result = player_key('  LeBron James  ')
        assert not result.startswith('_')
        assert not result.endswith('_')

# ---------------------------------------------------------------------------
# build_player_seasons
# ---------------------------------------------------------------------------
class TestBuildSeasons:
    def _base_row(self, player: str, team: str, g: int) -> dict:
        return {
            'player': player,
            'team': team,
            'g': g,
            'mp': 28.0,
            'pts': 15.0,
            'ast': 3.0,
            'trb': 5.0,
            'stl': 1.0,
            'blk': 0.5,
            'tov': 2.0,
            'fga': 12.0,
            'p3a': 3.0,
            'fta': 4.0,
        }
    
    def test_non_traded_player_passes_trough(self):
        df = pd.DataFrame([self._base_row('Player A', 'DAL', 70)])
        out = build_player_seasons(df, season=2024)
        assert len(out) == 1
        assert out.iloc[0]['player_name'] == 'Player A'

    def test_season_stamped_correctly(self):
        df = pd.DataFrame([self._base_row('Player A', 'DAL', 70)])
        out = build_player_seasons(df, season=2022)
        assert out.iloc[0]['season'] == 2022

    def test_player_key_is_slug(self):
        df = pd.DataFrame([self._base_row('Lebron James', 'LAL', 70)])
        out = build_player_seasons(df, season=2024)
        assert out.iloc[0]['player_key'] == 'lebron_james'

    def test_traded_player_tot_row_selected(self):
        """When TOT row exists, it should be the canonical row, not team rows."""
        rows = [
            self._base_row('Traded Guy', 'TOT', 65),
            self._base_row('Traded Guy', 'LAL', 30),
            self._base_row('Traded Guy', 'BOS', 35)
        ]
        df = pd.DataFrame(rows)
        out = build_player_seasons(df, season=2024)
        assert len(out[out['player_name'] == 'Traded Guy']) == 1
        assert out[out['player_name'] == 'Traded Guy'].iloc[0]['team'] == 'TOT'

    def test_traded_player_not_tot_uses_max_games(self):
        """Without a TOT row, the row with most games played wins."""
        rows = [
            self._base_row('Traded Guy', 'LAL', 30),
            self._base_row('Traded Guy', 'BOS', 45)
        ]
        df = pd.DataFrame(rows)
        out = build_player_seasons(df, season=2024)
        assert len(out[out['player_name'] == 'Traded Guy']) == 1
        assert out[out['player_name'] == 'Traded Guy'].iloc[0]['team'] == 'BOS'
    
    def test_multiple_players_all_preserved(self):
        rows = [
            self._base_row("Player A", "LAL", 70),
            self._base_row("Player B", "GSW", 65),
            self._base_row("Player C", "BOS", 60)
        ]
        df = pd.DataFrame(rows)
        out = build_player_seasons(df, season=2024)
        assert len(out) == 3

    def test_total_minutes_computed(self):
        df = pd.DataFrame([self._base_row('Player A', 'LAL', 70)])
        out = build_player_seasons(df, season=2024)
        row = out.iloc[0]
        assert float(row['total_minutes']) == pytest.approx(70 * 28.0)

    def test_raises_without_player_column(self):
        df = pd.DataFrame([{'team': 'LAL', 'g': 70}])
        with pytest.raises(ValueError, match='player'):
            build_player_seasons(df, season=2024)

    def test_no_duplicate_player_keys(self):
        """player_key must be unique per season after dedup."""
        rows = [
            self._base_row('Traded Guy', 'TOT', 65),
            self._base_row('Traded Guy', 'LAL', 30),
            self._base_row('Traded Guy', 'BOS', 35),
            self._base_row('Other Guy', 'MIA', 70)
        ]
        df = pd.DataFrame(rows)
        out = build_player_seasons(df, season=2024)
        assert out['player_key'].is_unique, 'Duplicate player_keys found after dedup'