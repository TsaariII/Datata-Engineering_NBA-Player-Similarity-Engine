"""
tests/test_golden_comps.py
--------------------------
Golden comparison regression tests.
 
These tests call top_k_similar directly against a live database populated by
the ETL pipeline.  They are the final layer of the evaluation suite — they
verify that the full ETL → DB → similarity path produces sensible results for
known archetypal players.
 
Prerequisites
-------------
1. A running PostgreSQL instance (DATABASE_URL set in environment).
2. ETL has been run for every season referenced in golden_comps.json:
       uv run python -m etl.scripts.run_etl --year <season>
       uv run python -m etl.scripts.build_features --year <season>
3. player_season_features is populated (checked by require_feature_table).
 
Running
-------
    # Integration tests only:
    DATABASE_URL=postgresql+psycopg://... uv run pytest tests/test_golden_comps.py -m integration -v
 
    # Or via docker compose after ETL completes:
    docker compose run --rm api uv run pytest tests/test_golden_comps.py -m integration -v
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import pytest
from sqlalchemy.engine import Engine
from etl.features.advanced_v1 import FEATURE_SET as ADV_FEAT_SET
from etl.features.per_game_v1 import FEATURE_SET as PG_FEAT_SET
from service.similarity_repo import load_feature_matrix, top_k_similar

# ---------------------------------------------------------------------------
# Load golden fixture
# ---------------------------------------------------------------------------
_FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'golden_comps.json'

def _load_golden_comps() -> List[Dict[str, Any]]:
    if not _FIXTURE_PATH.exists():
        return []
    return json.loads(_FIXTURE_PATH.read_text())

_GOLDEN = _load_golden_comps()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _result_names(results: List[dict]) -> List[str]:
    """Extract player names from top_k_similar output."""
    return [r['player'] for r in results]

def _clear_feature_matrix_cache() -> None:
    """
    load_feature_matrix uses lru_cache. This clears it between tests so one test's
    DB state doesn't leak into the next — especially important if tests run
    across multiple seasons or feature sets.
    """
    load_feature_matrix.cache_clear()

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixtures(autouse=True)
def clear_cache():
    """Clear the feature matrix cache before and after every test in this module."""
    _clear_feature_matrix_cache()
    yield
    _clear_feature_matrix_cache()

# ---------------------------------------------------------------------------
# Parametrized golden comparison tests
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.parametrize('case', _GOLDEN, ids=[c['_id'] for c in _GOLDEN])
def test_golden_comp(case: Dict[str, Any], engine: Engine, required_feats_table: None):
    """
    For each golden case, assert that:
      - must_appear_in_top_k players are all present in the top-k results
      - must_not_appear players are absent from the top-k results
 
    A failure here means one of:
      a) The ETL feature derivation has a bug (wrong column names, broken math)
      b) The z-score normalization collapsed (e.g. std=0 everywhere)
      c) The cosine similarity is computing nonsense (check z_vector contents)
      d) A player name in the fixture doesn't match the DB exactly
    """
    query = case['query']
    k = case.get('k', 10)
    feat_set = case.get('feat_set', ADV_FEAT_SET)
    players, _ = load_feature_matrix(engine, query['season'], feat_set)
    if query['player'] not in players:
        pytest.skip(
            f"Query player '{query['player']}' not found in player_season_features "
            f"for season={query['season']}, feature_set={feat_set}. "
            f"Check player name matches exactly what BBRef stores in the DB."
        )
    results = top_k_similar(
        engine=engine,
        season=query['seson'],
        feat_set=feat_set,
        player_name=query['name'],
        k=k
    )
    names = _result_names(results)
    missing = [p for p in case.get('must_appear_in_top_k', []) if p not in names]
    assert missing == [], (
        f"[{case['_id']}] Expected these players in top-{k} for "
        f"'{query['player']}' ({query['season']}): {missing}\n"
        f"Got: {names}\n"
        f"Rationale: {case.get('rationale', {}).get('why_matches', '')}"
    )
    wrong = [p for p in case.get('must_not_appear', []) if p in names]
    assert wrong == (
        f"[{case['_id']}] These players should NOT appear in top-{k} for "
        f"'{query['player']}' ({query['season']}): {wrong}\n"
        f"Got: {names}\n"
        f"Rationale: {case.get('rationale', {}).get('why_exclusions', '')}"
    )

# ---------------------------------------------------------------------------
# Structural sanity — applies to every populated season/feature_set combo
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.parametrize('feat_set,season', [
    (ADV_FEAT_SET, 2024),
    (PG_FEAT_SET, 2024)
])
def test_feature_matrix_is_populated(feat_set: str, season: int, engine: Engine):
    """
    The feature matrix for each supported feature set must be non-empty
    and have the right vector shape.  This catches:
      - ETL ran but wrote zero rows (silent load failure)
      - z_vector stored as NULL or empty array
      - Wrong feature_set label written to DB
    """
    players, mat = load_feature_matrix(engine, season, feat_set)
    assert len(players) > 0, (
        f"No players found in player_season_features for "
        f"season={season}, feature_set={feat_set}. Run build_features first."
    )
    assert mat.ndim == 2, 'Feature matrix must be 2D'
    assert mat.shape[0] == len(players), 'Row count must match player count'
    assert mat.shape[1] > 0, 'Feature vector must have at least one dimension'

@pytest.mark.integration
@pytest.mark.parametrize('feat_set,season', [
    (ADV_FEAT_SET, 2024),
    (PG_FEAT_SET, 2024)
])
def test_feature_matrix_has_no_all_zero_rows(feat_set: str, season: int, engine: Engine):
    """
    An all-zero z-vector means the player's stats were entirely missing or
    every feature was NaN-filled.  A handful of these is expected (injured
    players with very few games), but if more than 10% are all-zero it
    indicates a systematic ETL problem such as column name mismatches.
    """
    import numpy as np
    players, mat = load_feature_matrix(engine, season, feat_set)
    if len(players) == 0:
        pytest.skip(f"No data for season={season}, feature_set={feat_set}")
    all_zero_mask = np.all(mat == 0.0, axis=1)
    all_zero_count = int(all_zero_mask.sum)
    all_zero_pct = all_zero_count / len(players)
    assert all_zero_pct < 0.10, (
        f"{all_zero_count}/{len(players)} players ({all_zero_pct:.1%}) have "
        f"all-zero z-vectors for {feat_set} season={season}. "
        f"This likely means feature columns are missing from the ETL output. "
        f"Check for name mismatches between advanced_v1.FEATURES and "
        f"derive_features output columns."
    )

@pytest.mark.integration
@pytest.mark.parametrize('feat_set,season', [
    (ADV_FEAT_SET, 2024),
    (PG_FEAT_SET, 2024)
])
def test_simililarity_scores_are_in_valid_range(feat_set: str, season: int, engine: Engine):
    """
    Cosine similarity scores must be in [-1, 1].
    A score outside this range means the z_vector contains NaN or Inf values
    that survived the ETL pipeline, or the cosine math itself is broken.
    """
    players, mat = load_feature_matrix(engine, season, feat_set)
    if not players:
        pytest.skip(f"No data for season={season}, feature_set={feat_set}")
    test = players[0]
    results = top_k_similar(
        engine=engine,
        season=season,
        feat_set=feat_set,
        player_name=test,
        k=20
    )
    for r in results:
        score = r['score']
        assert -1.0 <= score <= 1.0, (
            f"Score {score:.4f} for '{r['player']}' is outside [-1, 1]. "
            f"Check z_vector for NaN/Inf values in player_season_features."
        )

@pytest.mark.integration
def test_self_similiarity_not_in_results(engine: Engine, feat_table: None):
    """
    A player must never appear in their own similarity results.
    top_k_similar explicitly skips the query player's index, but this test
    verifies that logic holds end-to-end against real DB data.
    """
    players, mat = load_feature_matrix(engine, 2024, ADV_FEAT_SET)
    if not players:
        pytest.skip('No data for season=2024')
    test = players[0]
    results = top_k_similar(
        engine=engine,
        season=2024,
        feat_set=ADV_FEAT_SET,
        player_name=test,
        k=10
    )
    names = _result_names(results)
    assert test not in names, (
       f"Player '{test}' appeared in their own similarity results" 
    )

@pytest.mark.integration
def test_unknown_player_raises_key_error(engine: Engine, feat_table: None):
    """
    Querying for a player not in the feature table must raise KeyError,
    not return empty results silently.  The API depends on this to return 404.
    """
    with pytest.raises(KeyError):
        top_k_similar(
        engine=engine,
        season=2024,
        feat_set=ADV_FEAT_SET,
        player_name='Not a player name',
        k=10
    )