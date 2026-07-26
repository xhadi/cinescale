from pathlib import Path
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

MOVIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ml-latest-small"
    / "movies.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "powerbi" / "data"
OUTPUT_PATH = OUTPUT_DIR / "recommendations.csv"


def get_database_connection():
    """Create a PostgreSQL database connection."""

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        database=os.getenv("POSTGRES_DB", "cinescale"),
        user=os.getenv(
            "POSTGRES_USER",
            "cinescale_admin",
        ),
        password=os.getenv(
            "POSTGRES_PASSWORD",
            "cinescale_password",
        ),
    )


def export_recommendations(top_n: int = 10) -> None:
    """Export top-N recommendations for every available user."""

    if not MOVIES_PATH.exists():
        raise FileNotFoundError(
            f"movies.csv not found: {MOVIES_PATH}"
        )

    movies = pd.read_csv(MOVIES_PATH)

    connection = get_database_connection()

    try:
        query = """
            SELECT
                uf.user_id,
                mf.movie_id,
                1 - (
                    mf.features <=> uf.features
                ) AS similarity
            FROM cinescale.user_factors uf
            CROSS JOIN LATERAL (
                SELECT
                    movie_id,
                    features
                FROM cinescale.movie_factors
                ORDER BY features <=> uf.features
                LIMIT %s
            ) mf
            ORDER BY
                uf.user_id,
                similarity DESC;
        """

        recommendations = pd.read_sql_query(
            query,
            connection,
            params=(top_n,),
        )

    finally:
        connection.close()

    recommendations["rank"] = (
        recommendations
        .groupby("user_id")
        .cumcount()
        + 1
    )

    movies = movies.rename(
        columns={"movieId": "movie_id"}
    )

    recommendations = recommendations.merge(
        movies[
            [
                "movie_id",
                "title",
                "genres",
            ]
        ],
        on="movie_id",
        how="left",
    )

    recommendations = recommendations[
        [
            "user_id",
            "rank",
            "movie_id",
            "title",
            "genres",
            "similarity",
        ]
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"Exported {len(recommendations)} recommendations."
    )

    print(
        f"Saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    export_recommendations(top_n=10)