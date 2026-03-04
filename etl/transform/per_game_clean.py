import numpy as np
import pandas as pd

_RENAMES = {
    'FG%': 'fg_percent',
    '3P%': '3P_percent',
    '2P%': '2P_percent',
    'eFG%': 'eFG_percent',
    'FT%': 'FT_percent',
}

_NUMERIC_INT = ['Age', 'G', 'GS']
_NUMERIC_FLOAT = [
    'MP','FG','FGA','FG_percent','3P','3PA','3P_percent','2P','2PA',
    '2P_percent','eFG_percent','FT','FTA','FT_percent','ORB','DRB','TRB',
    'AST','STL','BLK','TOV','PF','PTS'
]

def clean_per_game(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Tm' in df.columns and 'Team' not in df.columns:
        df.rename(columns={'Tm': 'Team'}, inplace=True)
    df.rename(columns=_RENAMES, inplace=True)
    if 'Player' in df.columns:
        df = df[df['Player'].notna()]
        df = df[df['Player'] != 'Player']
        df =  df[~df['Player'].str.contains('League Avarage', case=False, na=False)]
    for key in ('Team', 'Pos'):
        if key in df.columns:
            df = df[df[key] != key]
    df.replace(['', ' ', '_', '-'], np.nan, inplace=True)
    for c in _NUMERIC_INT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
    for c in _NUMERIC_FLOAT:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df