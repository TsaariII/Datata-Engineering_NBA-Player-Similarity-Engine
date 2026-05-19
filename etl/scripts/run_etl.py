import argparse
import logging
import os
from etl.sources.bb_reference import fetch_per_game_df
from etl.transform.per_game_clean import clean_per_game
from etl.load.postgres import write_per_game
from etl.utils.files import save_csv

def _parse_years(args: argparse.Namespace) -> list[int]:
    """
    Resolve the final list of seasons to process.
 
    Priority:
      1. --years 2020 2021 2022   (explicit multi-year list)
      2. --year 2024               (single-year legacy flag)
 
    Duplicates are removed and the list is sorted ascending.
    """
    if args.years:
        years = args.years
    elif args.year is not None:
        years = [args.year]
    else:
        raise SystemExit('Provide --year <N> or --years <N> [<N> ...]')
    seen: set[int] = set()
    deduped: list[int] = []
    for y in sorted(years):
        if y not in seen:
            seen.add(y)
            deduped.append(y)
    return deduped

def run_one_year(year: int, db_url: str, csv_dir: str) -> dict:
    """Run the full extraction + load pipeline for a single season."""
    raw = fetch_per_game_df(year)
    save_csv(raw, csv_dir, f"nba_per_game_{year}_raw")
    tidy = clean_per_game(raw)
    save_csv(tidy, csv_dir, f"NBA_per_game_{year}")
    stage, final = write_per_game(db_url, tidy, year)
    return {'year': year, 'stage': stage, 'final': final}

def main():
    ap = argparse.ArgumentParser(
        description='Extract and load NBA per-game stats into Postgres.'
    )
    year_group = ap.add_mutually_exclusive_group(required=True)
    year_group.add_argument(
        '--year',
        type=int,
        metavar='YEAR',
        help='Single NBA season end-year (e.g. 2024). Kept for backward compatibility.'
    )
    year_group.add_argument(
        '--years',
        type=int,
        nargs='+',
        metavar='YEARS',
        help=(
            'One or more NBA season end-years to process in sequence '
            '(e.g. --years 2020 2021 2022 2023 2024). '
            'Duplicates are silently ignored.'
        )
    )
    ap.add_argument(
        '--db-url',
        default=os.getenv('DATABASE_URL'),
        help='SQLAlchemy database URL. If omitted, uses DATABASE_URL env var'
    )
    ap.add_argument('--csv-dir', default='data')
    args = ap.parse_args()
    if not args.db_url:
        raise SystemExit('Missing --db-url (not set DATABASE_URL)')
    logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
    years = _parse_years(args)
    logger = logging.getLogger(__name__)
    logger.info('Processing %d season(s): %s', len(years), years)
    errors = []
    for year in years:
        try:
            result = run_one_year(year, args.db_url, args.csv_dir)
            print(result)
        except Exception as e:
            logger.error('Season %d failed: %s', year, e, exc_info=True)
            errors.append({'year': year, 'error': str(e)})
    if errors:
        logger.error('%d season(s) failed: %s', len(errors), [e['year'] for e in errors])
        raise SystemExit(1)

if __name__ == '__main__':
    main()