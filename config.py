from __future__ import annotations
import os
from dataclasses import dataclass

def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

@dataclass(frozen=True)
class Settings:
    HOST: str = os.getenv('HOST', '0.0.0.0')
    PORT: int = _int('PORT', 8555)
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    DATABASE_URL: str | None = os.getenv('DATABASE_URL')
    ETL_YEAR: int = _int('ETL_YEAR', 2024)
    NBA_PER_GAME_CSV: str | None = os.getenv('NBA_PER_GAME_CSV')

settings = Settings()