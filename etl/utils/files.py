import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def save_csv(df: pd.DataFrame, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(pat, index=False)
    logger.info("Saved CSV to %s", path)
    return path