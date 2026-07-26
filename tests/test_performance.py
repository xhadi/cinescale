import os
import pytest

from app import db_queries


def _db_available() -> bool:
    """Return True if the PostgreSQL DB is reachable."""
    try:
        with db_queries.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except Exception:
        return False


@pytest.fixture
def db_available():
    if not _db_available():
        pytest.skip("Postgres is not reachable")
    return True


@pytest.fixture
def any_user_id(db_available):
    users = db_queries.get_all_users(limit=10)
    if not users:
        pytest.skip("No users found in the database")
    return users[0]["user_id"]


def test_system_health(db_available):
    """Verify the DB has loaded data."""
    health = db_queries.get_system_health()
    assert health["total_users"] > 0
    assert health["total_movies"] > 0
    assert health["total_ratings"] > 0


def test_benchmark_recommendation_latency(db_available, any_user_id):
    """Verify recommendation latency is within a reasonable bound."""
    result = db_queries.benchmark_recommendation(any_user_id, runs=10)
    assert "p50_ms" in result
    assert "p95_ms" in result
    assert "p99_ms" in result
    assert "avg_ms" in result
    assert result["runs"] == 10
    # Loose thresholds for CI / small datasets; tune for 32M production runs
    assert result["p95_ms"] < 100.0
    assert result["p99_ms"] < 200.0
