import argparse
import logging
from etl.sources.bb_reference import fetch_per_game_df
from etl.transform.per_game_clean import clean_per_game
from etl.load.postgres import write_per_game
from etl.utils.files import save_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--db-url", required=True)
    ap.add_argument("--csv-dir", default="data")
    args = ap.parse_args()
    raw = fetch_per_game_df(args.year)
    save_csv(raw, args.csv_dir, f"nba_per_game_{args.year}_raw")
    tidy = clean_per_game(raw)
    save_csv(tidy, args.csv_dir, f"nba_per_game_{args.year}")
    stage, final = write_per_game(args.db_url, tidy, args.year)
    print({"stage": stage, "final": final})

if __name__ == "__main__":
    main()
