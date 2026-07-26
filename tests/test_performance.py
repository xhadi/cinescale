import os
import pytest
import psycopg2

from app import db_queries


P95_THRESHOLD_MS = float(os.getenv("CINESCALE_P95_THRESHOLD_MS", "100.0"))
P99_THRESHOLD_MS = float(os.getenv("CINESCALE_P99_THRESHOLD_MS", "200.0"))


def _db_available() -> bool:
    """Return True if the PostgreSQL DB is reachable."""
    try:
        with db_queries.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return True
    except psycopg2.OperationalError:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(),
    reason="Postgres is not reachable",
)


@pytest.fixture
def any_user_id():
    users = db_queries.get_all_users(limit=10)
    if not users:
        pytest.skip("No users found in the database")
    return users[0]["user_id"]


def test_system_health():
    """Verify the DB has loaded data and the expected schema keys."""
    health = db_queries.get_system_health()
    assert "total_users" in health
    assert "total_movies" in health
    assert "total_ratings" in health
    assert health["total_users"] > 0
    assert health["total_movies"] > 0
    assert health["total_ratings"] > 0


def test_benchmark_recommendation_latency(any_user_id):
    """Verify recommendation latency is within a reasonable bound."""
    result = db_queries.benchmark_recommendation(any_user_id, runs=10)
    assert "p50_ms" in result
    assert "p95_ms" in result
    assert "p99_ms" in result
    assert "avg_ms" in result
    assert result["runs"] == 10
    # Thresholds are configurable via env vars for CI / different dataset sizes
    assert result["p95_ms"] < P95_THRESHOLD_MS
    assert result["p99_ms"] < P99_THRESHOLD_MS
