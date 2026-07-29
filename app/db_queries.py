"""
db_queries.py — CineScale Backend Query Module (Module 4, Task 4.1)

Handles all PostgreSQL + pgvector queries for the Streamlit frontend.
Uses psycopg2 with context-managed connections.

All queries are designed to work against both synthetic and real data
without code changes, per the Interface Contract (rank=50, INTEGER IDs,
Parquet hand-off, cosine similarity via <=> operator).
"""

import os
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor


# ---------------------------------------------------------------------------
# Connection Management
# ---------------------------------------------------------------------------

@contextmanager
def get_conn():
    """
    Yields a psycopg2 connection from environment variables.
    Falls back to sensible defaults for local Docker Compose setup.

    Expected env vars (or defaults):
        POSTGRES_HOST=localhost, POSTGRES_PORT=5433,
        POSTGRES_DB=cinescale, POSTGRES_USER=cinescale_admin,
        POSTGRES_PASSWORD=cinescale_password
    """
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        database=os.getenv("POSTGRES_DB", "cinescale"),
        user=os.getenv("POSTGRES_USER", "cinescale_admin"),
        password=os.getenv(
            "POSTGRES_PASSWORD", "cinescale_password"
        ),
    )
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Task 4.1 — Core Recommendation Query (HNSW cosine similarity)
# ---------------------------------------------------------------------------

def get_recommendations(user_id: int, top_n: int = 10) -> List[Dict]:
    """
    Return top-N movie recommendations for a user using pgvector's <=> operator.

    The <=> operator returns cosine DISTANCE (0 = identical, 2 = opposite).
    Similarity score is computed as 1 - distance for UI display (0.0 to 1.0).

    Uses CROSS JOIN LATERAL to let the HNSW index serve the ORDER BY ... LIMIT
    efficiently inside the subquery before joining metadata.

    Args:
        user_id: MovieLens user ID (INTEGER, per Interface Contract)
        top_n: Number of recommendations to return (default 10)

    Returns:
        List of dicts with keys: movie_id, title, genres, similarity
    """
    sql = """
        SELECT 
            m.movie_id,
            m.title,
            m.genres,
            1 - (mf.features <=> uf.features) AS similarity
        FROM cinescale.user_factors uf
        CROSS JOIN LATERAL (
            SELECT 
                mf.movie_id,
                mf.features
            FROM cinescale.movie_factors mf
            ORDER BY mf.features <=> uf.features
            LIMIT %s
        ) mf
        JOIN cinescale.movies m ON m.movie_id = mf.movie_id
        WHERE uf.user_id = %s;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (top_n, user_id))
            rows = cur.fetchall()
            # RealDictCursor returns RealDictRow; cast to plain dict for safety
            return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# User Profile & History Queries (Left Panel)
# ---------------------------------------------------------------------------

def get_user_history(user_id: int, limit: int = 50) -> List[Dict]:
    """
    Return a user's rated movies, newest first, for the "Recent History" panel.

    Returns:
        List of dicts with keys: movie_id, title, genres, rating, rated_at
    """
    sql = """
        SELECT 
            m.movie_id,
            m.title,
            m.genres,
            r.rating,
            r.rated_at
        FROM cinescale.ratings r
        JOIN cinescale.movies m ON m.movie_id = r.movie_id
        WHERE r.user_id = %s
        ORDER BY r.rated_at DESC NULLS LAST
        LIMIT %s;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id, limit))
            return [dict(r) for r in cur.fetchall()]


def get_user_stats(user_id: int) -> Dict:
    """
    Return aggregate stats for the User Profile panel:
      - ratings_count
      - avg_score  (rounded to 1 decimal, matching NUMERIC(2,1))
      - top_affinity (most common genre among rated movies)

    Returns a single dict. If the user has no ratings, returns zeros/defaults.
    """
    sql = """
        WITH user_ratings AS (
            SELECT r.rating, m.genres
            FROM cinescale.ratings r
            JOIN cinescale.movies m ON m.movie_id = r.movie_id
            WHERE r.user_id = %s
        ),
        genre_counts AS (
            SELECT 
                UNNEST(STRING_TO_ARRAY(genres, '|')) AS genre,
                COUNT(*) AS cnt
            FROM user_ratings
            GROUP BY genre
            ORDER BY cnt DESC
            LIMIT 1
        )
        SELECT 
            (SELECT COUNT(*) FROM user_ratings) AS ratings_count,
            (SELECT ROUND(AVG(rating)::numeric, 1) FROM user_ratings) AS avg_score,
            (SELECT genre FROM genre_counts) AS top_affinity;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id,))
            row = cur.fetchone()
            if row is None:
                return {"ratings_count": 0, "avg_score": None, "top_affinity": None}
            result = dict(row)
            # Handle NULLs gracefully for cold-start users
            if result["ratings_count"] == 0:
                result["avg_score"] = None
                result["top_affinity"] = None
            return result


def get_rating_count(user_id: int) -> int:
    """
    Cold-start guard (Task 3.5). 
    Returns the number of ratings a user has submitted.
    If < 5, the frontend should show the "insufficient history" message.
    """
    sql = "SELECT COUNT(*) FROM cinescale.ratings WHERE user_id = %s;"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id,))
            return cur.fetchone()[0]


def get_genre_distribution(user_id: int) -> List[Dict]:
    """
    Return genre affinity percentages for the "Genre Distribution" bars.

    Computes what % of the user's rated movies contain each genre.
    Returns top 5 genres sorted by percentage desc.

    Returns:
        List of dicts with keys: genre, percentage (0-100)
    """
    sql = """
        WITH user_ratings AS (
            SELECT m.genres
            FROM cinescale.ratings r
            JOIN cinescale.movies m ON m.movie_id = r.movie_id
            WHERE r.user_id = %s
        ),
        total AS (
            SELECT COUNT(*)::float AS cnt FROM user_ratings
        ),
        genre_counts AS (
            SELECT 
                UNNEST(STRING_TO_ARRAY(genres, '|')) AS genre,
                COUNT(*) AS genre_cnt
            FROM user_ratings
            GROUP BY genre
        )
        SELECT 
            genre,
            ROUND((genre_cnt / NULLIF((SELECT cnt FROM total), 0) * 100)::numeric, 0) AS percentage
        FROM genre_counts
        ORDER BY genre_cnt DESC
        LIMIT 5;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (user_id,))
            return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# Cold-Start & Discovery Queries (Right Panel / Fallback)
# ---------------------------------------------------------------------------

def get_trending_movies(limit: int = 10, min_ratings: int = 100) -> List[Dict]:
    """
    Return globally trending movies for cold-start users or the "Trending Now" sidebar.

    Movies are ranked by average rating (desc), then by rating count (desc).
    Only includes movies with at least `min_ratings` reviews to avoid outliers.

    Returns:
        List of dicts with keys: movie_id, title, genres, avg_rating, rating_count
    """
    sql = """
        SELECT 
            m.movie_id,
            m.title,
            m.genres,
            ROUND(AVG(r.rating)::numeric, 1) AS avg_rating,
            COUNT(r.rating) AS rating_count
        FROM cinescale.movies m
        JOIN cinescale.ratings r ON r.movie_id = m.movie_id
        GROUP BY m.movie_id, m.title, m.genres
        HAVING COUNT(r.rating) >= %s
        ORDER BY avg_rating DESC, rating_count DESC
        LIMIT %s;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (min_ratings, limit))
            return [dict(r) for r in cur.fetchall()]


def search_movies(query: str, limit: int = 20, genre: str | None = None) -> List[Dict]:
    """
    Fuzzy title search for the Explore panel search bar, with optional genre filter.

    Uses ILIKE for case-insensitive prefix matching. For production,
    consider pg_trgm or a dedicated search index.

    Returns:
        List of dicts with keys: movie_id, title, genres
    """
    conditions = []
    params: list = []

    if query:
        conditions.append("title ILIKE %s")
        params.append(f"%{query}%")
    if genre:
        conditions.append("genres ILIKE %s")
        params.append(f"%{genre}%")

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sql = f"""
        SELECT 
            movie_id,
            title,
            genres
        FROM cinescale.movies
        {where}
        ORDER BY title
        LIMIT %s;
    """
    params.append(limit)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]


def get_all_genres() -> List[str]:
    """
    Return the distinct set of all genres across the catalog.
    Used to populate the genre filter chips dynamically.
    """
    sql = """
        SELECT DISTINCT UNNEST(STRING_TO_ARRAY(genres, '|')) AS genre
        FROM cinescale.movies
        ORDER BY genre;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [row[0] for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Movie Detail & Explanation Queries
# ---------------------------------------------------------------------------

def get_movie_details(movie_id: int) -> Optional[Dict]:
    """
    Return full metadata for a single movie (used by the "Details" button).

    Returns:
        Dict with keys: movie_id, title, genres
        or None if not found.
    """
    sql = """
        SELECT movie_id, title, genres
        FROM cinescale.movies
        WHERE movie_id = %s;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (movie_id,))
            row = cur.fetchone()
            return dict(row) if row else None


def get_user_rating_for_movie(user_id: int, movie_id: int) -> Optional[float]:
    """
    Check if a user has already rated a specific movie.
    Returns the rating value or None.
    """
    sql = """
        SELECT rating FROM cinescale.ratings
        WHERE user_id = %s AND movie_id = %s;
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (user_id, movie_id))
            row = cur.fetchone()
            return float(row[0]) if row else None


def get_explanation(user_id: int, movie_id: int, top_n: int = 3) -> List[Dict]:
    """
    Lightweight "Why This?" explanation for the Details panel.

    Returns the user's top-rated movies that share at least one genre
    with the recommended movie. This is "faux-explainability" — cheap to
    compute, high user trust. No need to introspect the ALS embedding.

    Returns:
        List of dicts with keys: title, rating, shared_genres
    """
    sql = """
        WITH target_genres AS (
            SELECT STRING_TO_ARRAY(genres, '|') AS g_arr
            FROM cinescale.movies
            WHERE movie_id = %s
        )
        SELECT 
            m.title,
            r.rating,
            ARRAY(
                SELECT UNNEST(STRING_TO_ARRAY(m.genres, '|'))
                INTERSECT
                SELECT UNNEST((SELECT g_arr FROM target_genres))
            ) AS shared_genres
        FROM cinescale.ratings r
        JOIN cinescale.movies m ON m.movie_id = r.movie_id
        WHERE r.user_id = %s
          AND r.rating >= 4.0
        ORDER BY r.rating DESC, r.rated_at DESC NULLS LAST
        LIMIT %s;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (movie_id, user_id, top_n))
            return [dict(r) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# System Health / V&V Queries (Task 4.3)
# ---------------------------------------------------------------------------

def get_system_health() -> Dict:
    """
    Return high-level database stats for the System Health tab.

    Returns:
        Dict with keys: total_users, total_movies, total_ratings, index_status
    """
    sql = """
        SELECT 
            (SELECT COUNT(*) FROM cinescale.user_factors) AS total_users,
            (SELECT COUNT(*) FROM cinescale.movie_factors) AS total_movies,
            (SELECT COUNT(*) FROM cinescale.ratings) AS total_ratings,
            (SELECT COUNT(*) FROM pg_indexes
             WHERE schemaname = 'cinescale'
               AND indexname = 'idx_movie_factors_hnsw_cosine') AS index_present;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            result = dict(cur.fetchone())
            result["index_status"] = "present" if result["index_present"] else "missing"
            del result["index_present"]
            return result


def benchmark_recommendation(user_id: int, runs: int = 100, return_raw_times: bool = False) -> Dict:
    """
    Benchmark the HNSW recommendation query latency.

    Runs the core similarity query `runs` times and reports p50/p95/p99.
    This is a raw benchmark — in production, use proper load-testing tools.

    Args:
        user_id: MovieLens user ID
        runs: Number of query executions
        return_raw_times: If True, include raw per-run latencies in the result

    Returns:
        Dict with keys: p50_ms, p95_ms, p99_ms, avg_ms, runs
        (and raw_times_ms when return_raw_times is True)
    """
    import time

    sql = """
        SELECT 1 - (mf.features <=> uf.features) AS similarity
        FROM cinescale.user_factors uf
        CROSS JOIN LATERAL (
            SELECT features FROM cinescale.movie_factors mf
            ORDER BY mf.features <=> uf.features
            LIMIT 10
        ) mf
        WHERE uf.user_id = %s;
    """
    times = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            for _ in range(runs):
                t0 = time.perf_counter()
                cur.execute(sql, (user_id,))
                cur.fetchall()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms

    times.sort()
    n = len(times)
    result = {
        "runs": runs,
        "avg_ms": round(sum(times) / n, 2),
        "p50_ms": round(times[int(n * 0.50)], 2),
        "p95_ms": round(times[int(n * 0.95)], 2),
        "p99_ms": round(times[int(n * 0.99)], 2),
    }
    if return_raw_times:
        result["raw_times_ms"] = times
    return result


# ---------------------------------------------------------------------------
# User Dropdown Helper
# ---------------------------------------------------------------------------

def get_all_users(limit: int = 500) -> List[Dict]:
    """
    Return all users who have embeddings loaded, for the frontend dropdown.
    Includes rating count so the UI can flag cold-start users.

    Returns:
        List of dicts with keys: user_id, rating_count
    """
    sql = """
        SELECT 
            uf.user_id,
            COUNT(r.user_id) AS rating_count
        FROM cinescale.user_factors uf
        LEFT JOIN cinescale.ratings r ON r.user_id = uf.user_id
        GROUP BY uf.user_id
        ORDER BY uf.user_id
        LIMIT %s;
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, (limit,))
            return [dict(r) for r in cur.fetchall()]