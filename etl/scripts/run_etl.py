import argparse
import logging
from etl.sources.basketball_reference import run_etl

def main():
    parser = argparse.ArgumentParser(description="Run Basketball-Reference per-game ETL")
    parser.add_argument("--year", type=int, required=True, help="Season year, e.g. 2024")
    parser.add_argument("--db-url postgresql+psycopg://user:pass@host/dbname")
    parser.add_argument("--csv-dir", default="data", help="Directory to write CSV")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_etl(args.year, db_url=args.db_url, csv_dir=args.csv_dir)
    print(result)

if __name__ == "__main__":
    main()
