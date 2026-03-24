import argparse
import logging
import os
import pandas as pd
from sqlalchemy import create_engine, text
from etl.features.per_game_v1 import FEATURE_SET as PG_V1_FEATURE_SET
from etl.features.per_game_v1 import FEATURES as PG_V1_FEATURES
from etl.features.advanced_v1 import FEATURE_SET as ADV_V1_FEATURE_SET
from etl.features.advanced_v1 import FEATURES as ADV_V1_FEATURES
from etl.load.player_season_features import (
    ensure_tables,
    upsert_player_seasons,
    upsert_player_season_features
)
from etl.transform.player_seasons import build_player_seasons
from etl.transform.derive_features import derive_features
from etl.transform.player_season_features import FeaturePack, zscore_features

logger = logging.getLogger(__name__)

_SUPPORTED_FEATURE_SETS = {
    PG_V1_FEATURE_SET: PG_V1_FEATURES,
    ADV_V1_FEATURE_SET: ADV_V1_FEATURES
}

# Default feature set used when --feature-set is not specified
_DEFAULT_FEATURE_SET = ADV_V1_FEATURE_SET

def _normalize_database_url(url: str) -> str:
    url = url.strip()
    if url.startswith("postgres://") and "+" not in url.split("://", 1)[0]:
        url = "postgres+psycopg://" + url[len("postgres://") :]
    return url

def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def _safe_str(val) -> str | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return None if s == "" else s


def _safe_int(val) -> int | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Build player_seasons and player_season_features from a raw per-game table.'
    )
    ap.add_argument('--year', type=int, required=True, help='NBA season')
    ap.add_argument(
        '--db-url',
        default=os.getenv('DATABASE_URL'),
        help='SQLAlchemy database url. If omitted, uses DATABASE_URL env var'
    )
    ap.add_argument(
        '--feature-set',
        default=_DEFAULT_FEATURE_SET,
        choices=list(_SUPPORTED_FEATURE_SETS),
        help=(f"Feature set to compute. "
            f"Choices: {list(_SUPPORTED_FEATURE_SETS.keys())}. "
            f"Default: {_DEFAULT_FEATURE_SET}")
    )
    args = ap.parse_args()
    if not args.db_url:
        raise SystemExit('Missing --db-url (not set DATABASE_URL)')
    logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
    feat_set = args.feature_set
    feats = _SUPPORTED_FEATURE_SETS[feat_set]
    logger.info('Building feature set=%s for season=%d', feat_set, args.year)
    # 1. Load raw per-game table from DB
    engine = create_engine(_normalize_database_url(args.db_url), pool_pre_ping=True, future=True)
    per_game_tbl = f"per_game_{args.year}"
    with engine.connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {per_game_tbl}"), conn)
    logger.info('Loaded %d rows from %s', len(df), per_game_tbl)
    # 2. Build canonical player-seasons (dedup traded players, add keys)
    seasons_df = build_player_seasons(df, season=args.year)
    logger.info('Canonical player-seasons: %d rows', len(seasons_df))
    # 3. Derive advanced features (only needed for advanced_v1+)
    #    per_game_v1 uses raw columns only, so we skip derivation for it.
    if feat_set == ADV_V1_FEATURE_SET:
        seasons_df = derive_features(seasons_df)
        logger.info('Derived advanced features: %s', ADV_V1_FEATURES)
    # 4. Build player_seasons upsert payload
    #    Always uses the raw/base columns regardless of feature set.
    stats_cols = [c for c in PG_V1_FEATURES if c in seasons_df.columns]
    seasons_payload = []
    has_team = 'team' in seasons_df.columns
    has_pos = 'pos' in seasons_df.columns
    has_g = 'g' in seasons_df.columns
    has_mp = 'mp' in seasons_df.columns
    has_mpg = 'mpg' in seasons_df.columns
    has_total_minutes = 'total_minutes' in seasons_df.columns
    for _, row in seasons_df.iterrows():
        stats = {k: _safe_float(row.get(k)) for k in stats_cols}
        seasons_payload.append(
            {
                'player_key': str(row['player_key']),
                'player_name': str(row['player_name']),
                'season': int(row['season']),
                'team': _safe_str(row.get('team')) if has_team else None,
                'pos': _safe_str(row.get('pos')) if has_pos else None,
                'g': _safe_int(row.get('g')) if has_g else None,
                'mp': _safe_float(row.get('mp')) if has_mp else None,
                'mpg': _safe_float(row.get('mpg')) if has_mpg else None,
                'total_minutes': _safe_float(row.get('total_minutes')) if has_total_minutes else None,
                'stats': stats
            }
        )
    # 5. Z-score the chosen feature set
    pack = FeaturePack(feature_set=feat_set, feature_names=feats)
    feat_df = zscore_features(seasons_df, pack)
    feat_payload = feat_df.to_dict(orient='records')
    missing_feats = [f for f in feats if f not in seasons_df.columns]
    if missing_feats:
        logger.warning(
            'These features were not found in seasons_df and will be zero-filled: %s',
            missing_feats,
        )
    # 6. Upsert into DB
    ensure_tables(args.db_url)
    n1 = upsert_player_seasons(args.db_url, seasons_payload)
    n2 = upsert_player_season_features(args.db_url, feat_payload)
    print({
        'season': args.year,
        'player_seasons_upserted': n1,
        'player_season_features_upserted': n2,
        'feature_set': feat_set,
        'feature_count': len(feats),
    })

if __name__ == '__main__':
    main()