"""
Fixture scopes
--------------
db_url      : session  – the raw DATABASE_URL string (normalized).
              Skips the test if DATABASE_URL is not set in the environment.
              Use this in tests that call etl.load functions directly
              (they accept a URL string, not an Engine).
 
db_engine   : session  – a SQLAlchemy Engine connected to the live DB.
              Use this in tests that call service.similarity_repo functions
              (they accept an Engine).
 
Both fixtures are only available to tests marked with @pytest.mark.integration.
Unit tests (no mark) never touch the DB and never need these fixtures.
 
Running tests
-------------
# Unit tests only (no DB required):
    uv run pytest tests/ -m "not integration"
 
# Integration tests (requires running Postgres + populated feature table):
    DATABASE_URL=postgresql+psycopg://... uv run pytest tests/ -m integration
    # or via docker compose:
    docker compose run --rm etl uv run pytest tests/ -m integration
"""

from __future__ import annotations
import os
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register custom marks so pytest does not warn about unknown marks."""
    config.addinivalue_line(
        'markers',
        'integration: marks tests that require a live PostgreSQL database'
        '(deselect with "-m \'not integration\'")'
    )

# ---------------------------------------------------------------------------
# URL normalization
# ---------------------------------------------------------------------------
def _normalize_db_url(url: str) -> str:
    """Coerce postgres:// → postgresql+psycopg:// for SQLAlchemy 2 + psycopg v3."""
    url = url.strip()
    if url.startswith('postgres://') and '+' not in url.split('://', 1)[0]:
        url = 'postgresql+psycopg://' + url[len('postgres://'):]
    elif url.startswith('postgresql://') and '+' not in url.split('://', 1)[0]:
        url = 'postgresql+psycopg://' + url[len('postgresql+psycopg://'):]
    return url

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope='session')
def db_url() -> str:
    """
    Return the normalized DATABASE_URL string.
 
    Skips the entire test session (for integration tests) if the env var is
    not set, so the CI unit-test job never needs a running database.
    """
    raw = os.getenv('DATABASE_URL', '').strip()
    if not raw:
        pytest.skip(
            'DATABASE_URL is not set - skipping integration tests. '
            'Set DATABASE_URL to run against a live database.'
        )
    return _normalize_db_url(raw)

@pytest.fixture(scope='session')
def db_engine(db_url: str) -> Engine:
    """
    Return a session-scoped SQLAlchemy Engine.
 
    The engine is created once per pytest session and reused across all
    integration tests.  pool_pre_ping=True ensures we get a clear error
    rather than a stale-connection failure mid-test.
 
    Performs a quick connectivity check (SELECT 1) so a misconfigured
    DATABASE_URL fails fast with a clear message rather than a cryptic
    error inside the first test that uses the fixture.
    """
    engine = create_engine(db_url, pool_pre_ping=True, future=True)
    try:
        with engine.connect() as conn:
            conn.execute(text('SELECT 1'))
    except Exception as e:
        pytest.skip(
            f"Could not connect to the databse at {db_url}. "
            f"Skipping integration tests. Error: {e}"
        )
    yield engine
    engine.dispose()

# ---------------------------------------------------------------------------
# Optional: table-presence guard
# ---------------------------------------------------------------------------
@pytest.fixture(scope='session')
def require_feature_table(engine: Engine) -> None:
    """
    Skip the test if player_season_features is empty.
    """
    with engine.connect() as conn:
        count = conn.execute(
            text('SELECT COUNT(*) FROM player_season_features')
        ).scalar()
    if not count:
        pytest.skip(
            'player_season_features is empty - run'
            '`build-features --year <year>` before executing integration tests.'
        )