"""
tests/test_feature_sets.py
--------------------------
Version stability guards for feature set definitions.
 
Purpose
-------
Feature set files (per_game_v1.py, advanced_v1.py) are versioned artifacts.
Once a feature set is in production and has rows in player_season_features,
changing it silently breaks all precomputed similarity vectors stored under
that version label — without any DB error or warning.
 
These tests act as a snapshot guard: they fail the moment a versioned file
is edited, forcing the developer to either:
  a) confirm the change is intentional and update the snapshot here, or
  b) realise it was accidental and revert.
 
If you are intentionally adding a new feature set, the correct workflow is:
  1. Copy per_game_v1.py → per_game_v2.py (new file, new FEATURE_SET label).
  2. Add the new module to _SUPPORTED_FEATURE_SETS in build_features.py.
  3. Add a corresponding snapshot class below.
  4. Do NOT edit the v1 file — leave it frozen.
 
Scope
-----
These tests cover:
  - Exact feature list contents and order (order matters for z_vector indexing)
  - Feature count (catches additions and removals)
  - FEATURE_SET label strings (these are stored as PK values in the DB)
  - No duplicate feature names within a list
  - No feature name is empty or whitespace
  - build_features._SUPPORTED_FEATURE_SETS registry is consistent with modules
  - Default feature set points at a registered feature set
  - All feature set labels in the registry are unique
  - Feature set label matches the module-level FEATURE_SET constant
"""

from __future__ import annotations
import importlib
import pytest
from etl.features.advanced_v1 import FEATURE_SET as ADV_FEAT_SET
from etl.features.advanced_v1 import FEATURES as ADV_FEAT
from etl.features.per_game_v1 import FEATURE_SET as PG_FEAT_SET
from etl.features.per_game_v1 import FEATURES as PG_FEAT

# ---------------------------------------------------------------------------
# Snapshots — per_game_v1
# ---------------------------------------------------------------------------
class TestPerGameSnapshot:
    def test_label(self):
        assert PG_FEAT_SET == 'per_game_v1'
    
    def test_feature_count(self):
        assert len(PG_FEAT) == 15, (
            f"per_game_v1 has {len(PG_FEAT)} features, expected 15. "
            'If this is intentional, create per_game_v2.py instead of editing v1.'
        )
    
    def test_exact_features_list_and_order(self):
        """
        Order matters: z_vector is a positional list.
        Changing order silently invalidates all stored vectors.
        """
        expected = [
            'mp',
            'pts',
            'ast',
            'trb',
            'stl',
            'blk',
            'tov',
            'pf',
            'fga',
            'p3a',
            'fta',
            'fg_pct',
            'p3_pct',
            'ft_pct',
            'efg_pct'
        ]
        assert PG_FEAT == expected, (
            'per_game_v1.FEATURES has changed. '
            'Create per_game_v2.py instead of editing a versioned file.'
        )

# ---------------------------------------------------------------------------
# Snapshots — advanced_v1
# ---------------------------------------------------------------------------
class TestAdvancedV1Snapshot:
    def test_label(self):
        assert ADV_FEAT_SET == 'advanced_v1'
    
    def test_feature_count(self):
        assert len(ADV_FEAT) == 11, (
            f"per_game_v1 has {len(ADV_FEAT)} features, expected 11. "
            'If this is intentional, create advanced_v2.py instead of editing v1.'
        )
    
    def test_exact_features_list_and_order(self):
        expected = [
            'mp',
            'usage_proxy',
            'ts_pct',
            'p3a_rate',
            'ftr',
            'assist_pct_proxy',
            'turover_pct',
            'offrb_pct_proxy',
            'defrb_pct_proxy',
            'steals',
            'blocks'
        ]
        assert ADV_FEAT == expected, (
            'advanced_v1.FEATURES has changed. '
            'Create advanced_v2.py instead of editing a versioned file.'
        )

# ---------------------------------------------------------------------------
# Internal consistency — applies to every feature set
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('label, feats', [
    (PG_FEAT_SET, PG_FEAT),
    (ADV_FEAT_SET, ADV_FEAT)
])
class TestFeatureSetInternalConsistency:
    def test_no_duolicate_feature_names(self, label, feats):
        dupes = [f for f in feats if feats.count(f) > 1]
        assert dupes == [], (
            f"{label} contains duplicate feature names: {set(dupes)}"
        )
    
    def test_no_empty_or_whitespace_names(self, label, feats):
        bad = [f for f in feats if not f or not f.strip()]
        assert bad == [], f"{label} contains empty/whitespace feature names"

    def test_all_names_are_strings(self, label, feats):
        non_str = [f for f in feats if not isinstance(f, str)]
        assert non_str == [], f"{label} contains non-string feature names: {non_str}"
    
    def test_all_names_are_lowercase(self, label, feats):
        """Convention: feature names are lowercase snake_case, matching DB column names."""
        upper = [f for f in feats if f != f.lower()]
        assert upper == [], (
            f"{label} contains non-lowercase feature names: {upper}. "
            "Feature names must match DB column names exactly."
        )
    
    def test_label_is_nonempty_string(self, label, feats):
        assert isinstance(label, str) and label.strip()

    def test_features_is_a_list(self, label, feats):
        """Must be a list, not a tuple or set — order must be preserved."""
        assert isinstance(feats, list), (
            f"{label}.FEATURES must be a list (order-preserving), got {type(feats)}"
        )

# ---------------------------------------------------------------------------
# Registry consistency — build_features._SUPPORTED_FEATURE_SETS
# ---------------------------------------------------------------------------
class TestBuildFeaturesRegistry:
    """
    build_features.py maintains a _SUPPORTED_FEATURE_SETS dict that maps
    label → feature list.  If a module is added but not registered (or vice
    versa), the CLI silently ignores it or crashes at runtime.
    """
    def setup_method(self):
        mod = importlib.import_module('etl.scripts.build_features')
        self._registery = mod._SUPPORTED_FEATURE_SETS
        self._default = mod._DEFAULT_FEATURE_SET

    def test_registery_contains_per_game_v1(self):
        assert PG_FEAT_SET in self._registery, (
            f"'{PG_FEAT_SET}' is missing from build_features._SUPPORTED_FEATURE_SETS"
        )
    
    def test_registery_contains_advanced_v1(self):
        assert ADV_FEAT_SET in self._registery, (
            f"'{ADV_FEAT_SET}' is missing from build_features._SUPPORTED_FEATURE_SETS"
        )
    
    def test_registery_features_match_modules(self):
        """
        The feature list in the registry must be the same object (or equal list)
        as the one exported from the module.  A copy-pasted list that drifts is
        a silent bug.
        """
        assert self._registery[PG_FEAT_SET] == PG_FEAT, (
            'Registry entry for per_game_v1 differs from per_game_v1.FEATURES'
        )
        assert self._registery[ADV_FEAT_SET] == ADV_FEAT, (
            'Registry entry for advanced_v1 differs from advanced_v1.FEATURES'
        )
    
    def test_default_feature_set_is_registered(self):
        assert self._default in self._registery, (
            f"Default feature set '{self._default}' is not in _SUPPORTED_FEATURE_SETS"
        )
    
    def test_all_registery_labels_are_unique(self):
        labels = list(self._registery.keys())
        assert len(labels) == len(set(labels)), 'Registry contains duplicate labels'
    
    def test_registery_values_are_nonempty_lists(self):
        for label, feats in self._registery.items():
            assert isinstance(feats, list) and len(feats) > 0, (
                f"Registry entry for '{label}' is empty or not a list"
            )