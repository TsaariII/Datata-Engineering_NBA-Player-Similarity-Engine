from sqlalchemy import create_engine, Text
import pandas as pd

def write_per_game(engine_url: str, df: pd.DataFrame, year: int) -> tuple[str, str]:
    stage = f"per_game_{year}_raw"
    final = f"per_game_{year}"
    engine = create_engine(engine_url, pool_pre_ping=True, future=True)
    df.to_sql(stage, engine, if_exists='replace', index=False, dtype={col: Text() for col in df.columns0}, method="multi", chunksize=1000)
    with engine.begin() as con:
        con.exec_driver_sql(f"""
            DROP TABLE IF EXISTS {final};
            CREATE TABLE {final} AS
            SELECT
                "Player"::varchar AS player,
                NULLIF("Age",'')::int AS age,
                COALESCE("Team","Tm")::varchar AS team,
                "Pos"::varchar AS pos,
                NULLIF("G",'')::int AS g,
                NULLIF("GS",'')::int AS gs,
                NULLIF("MP",'')::numeric AS mp,
                NULLIF("FG",'')::numeric AS fg,
                NULLIF("FGA",'')::numeric AS fga,
                NULLIF(COALESCE("FG_percent","FG%"),'')::numeric AS fg_pct,
                NULLIF("3P",'')::numeric AS p3,
                NULLIF("3PA",'')::numeric AS p3a,
                NULLIF(COALESCE("3P_percent","3P%"),'')::numeric AS p3_pct,
                NULLIF("2P",'')::numeric AS p2,
                NULLIF("2PA",'')::numeric AS p2a,
                NULLIF(COALESCE("2P_percent","2P%"),'')::numeric AS p2_pct,
                NULLIF(COALESCE("eFG_percent","eFG%"),'')::numeric AS efg_pct,
                NULLIF("FT",'')::numeric AS ft,
                NULLIF("FTA",'')::numeric AS fta,
                NULLIF(COALESCE("FT_percent","FT%"),'')::numeric AS ft_pct,
                NULLIF("ORB",'')::numeric AS orb,
                NULLIF("DRB",'')::numeric AS drb,
                NULLIF("TRB",'')::numeric AS trb,
                NULLIF("AST",'')::numeric AS ast,
                NULLIF("STL",'')::numeric AS stl,
                NULLIF("BLK",'')::numeric AS blk,
                NULLIF("TOV",'')::numeric AS tov,
                NULLIF("PF",'')::numeric AS pf,
                NULLIF("PTS",'')::numeric AS pts,
                COALESCE("Awards",'')::varchar AS awards
            FROM {stage}
            WHERE "Player" IS NOT NULL AND "Player" <> 'League Avarage';
        """)
    return stage, final