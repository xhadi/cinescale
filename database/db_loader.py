from datetime import datetime, timezone
from pathlib import Path
import os
import sys

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.schema import config


load_dotenv(PROJECT_ROOT / ".env")


def resolve_factor_path(
    processed_path: Path,
    synthetic_path: Path,
) -> Path:
    """Return processed factors if present, otherwise synthetic."""

    if processed_path.exists():
        return processed_path
    return synthetic_path


USER_FACTORS_PATH = resolve_factor_path(
    Path(config.PROCESSED_DATA_DIR) / config.USER_FACTORS_FILENAME,
    PROJECT_ROOT / "data" / "synthetic" / "user_factors.parquet",
)

MOVIE_FACTORS_PATH = resolve_factor_path(
    Path(config.PROCESSED_DATA_DIR) / config.MOVIE_FACTORS_FILENAME,
    PROJECT_ROOT / "data" / "synthetic" / "movie_factors.parquet",
)

MOVIES_PATH = Path(config.PROCESSED_DATA_DIR) / "movies_clean.parquet"

RATINGS_PATH = Path(config.PROCESSED_DATA_DIR) / "ratings_clean.parquet"

# Commit factor loads in batches to keep transactions bounded.
FACTOR_BATCH_SIZE = 10_000


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


def drop_hnsw_index(cursor) -> None:
    """Drop the HNSW cosine index before bulk-loading embeddings."""

    cursor.execute(
        "DROP INDEX IF EXISTS cinescale.idx_movie_factors_hnsw_cosine;"
    )


def create_hnsw_index(cursor) -> None:
    """Recreate the HNSW cosine index after bulk-loading embeddings."""

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_movie_factors_hnsw_cosine
        ON cinescale.movie_factors
        USING hnsw (features vector_cosine_ops)
        WITH (
            m = 16,
            ef_construction = 64
        );
        """
    )


def convert_vector_to_pgvector(vector) -> str:
    """Convert a vector into PostgreSQL pgvector format."""

    if vector is None:
        raise ValueError("Vector cannot be None.")

    if len(vector) != config.EMBEDDING_DIM:
        raise ValueError(
            f"Expected vector dimension "
            f"{config.EMBEDDING_DIM}, "
            f"but found {len(vector)}."
        )

    try:
        values = [float(value) for value in vector]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Vector must contain numeric values only."
        ) from error

    return "[" + ",".join(
        str(value) for value in values
    ) + "]"


def convert_timestamp_to_datetime(value) -> datetime | None:
    """Convert an epoch-seconds value to a timezone-aware datetime."""

    if value is None or pd.isna(value):
        return None

    return datetime.fromtimestamp(
        int(value), tz=timezone.utc
    )


def read_factors_file(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Read and validate a factors Parquet file.

    Accepts both the synthetic layout (``id``, ``features``) and the
    processed/exported ALS layout (``userId``/``movieId``, ``features``).
    The ID column is normalized to ``id`` before validation.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file not found: {file_path}"
        )

    dataframe = pd.read_parquet(
        file_path,
        engine="pyarrow",
    )

    # Normalize ALS-exported column names to the synthetic layout.
    # Processed factors may use ``userId`` / ``movieId`` and ``embedding``
    # while synthetic factors use ``id`` and ``features``.
    if "userId" in dataframe.columns:
        dataframe = dataframe.rename(columns={"userId": "id"})
    elif "movieId" in dataframe.columns:
        dataframe = dataframe.rename(columns={"movieId": "id"})

    if "embedding" in dataframe.columns:
        dataframe = dataframe.rename(columns={"embedding": "features"})

    required_columns = {"id", "features"}

    if not required_columns.issubset(
        dataframe.columns
    ):
        raise ValueError(
            f"{dataset_name} file must contain "
            "'id' and 'features' columns."
        )

    if dataframe.empty:
        raise ValueError(
            f"{dataset_name} file contains no records."
        )

    if dataframe["id"].isnull().any():
        raise ValueError(
            f"{dataset_name} file contains missing IDs."
        )

    if dataframe["features"].isnull().any():
        raise ValueError(
            f"{dataset_name} file contains "
            "missing feature vectors."
        )

    if dataframe["id"].duplicated().any():
        raise ValueError(
            f"{dataset_name} file contains duplicate IDs."
        )

    return dataframe


def read_parquet_file(
    file_path: Path,
    dataset_name: str,
    required_columns: set,
    id_columns: str | list[str],
) -> pd.DataFrame:
    """Read and validate a generic Parquet file.

    ``id_columns`` may be a single column name or a list of column names
    (for composite keys). Missing or duplicate IDs are rejected.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file not found: {file_path}"
        )

    dataframe = pd.read_parquet(
        file_path,
        engine="pyarrow",
    )

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            f"{dataset_name} file must contain "
            f"{sorted(required_columns)} columns."
        )

    if dataframe.empty:
        raise ValueError(
            f"{dataset_name} file contains no records."
        )

    id_columns_list = (
        [id_columns]
        if isinstance(id_columns, str)
        else id_columns
    )

    for column in id_columns_list:
        if dataframe[column].isnull().any():
            raise ValueError(
                f"{dataset_name} file contains missing IDs "
                f"in column '{column}'."
            )

    if dataframe.duplicated(subset=id_columns_list).any():
        raise ValueError(
            f"{dataset_name} file contains duplicate IDs."
        )

    return dataframe


def load_movies(connection) -> None:
    """Load movie metadata into PostgreSQL."""

    dataframe = read_parquet_file(
        MOVIES_PATH,
        "Movies",
        {"movieId", "title", "genres"},
        "movieId",
    )

    query = """
        INSERT INTO cinescale.movies (
            movie_id,
            title,
            genres
        )
        VALUES %s
        ON CONFLICT (movie_id)
        DO UPDATE SET
            title = EXCLUDED.title,
            genres = EXCLUDED.genres;
    """

    records = [
        (
            int(row.movieId),
            str(row.title) if row.title is not None else "Unknown",
            str(row.genres) if row.genres is not None else "Unknown",
        )
        for row in dataframe.itertuples(index=False)
    ]

    with connection.cursor() as cursor:
        execute_values(cursor, query, records, page_size=5000)

    print(f"Loaded {len(records)} movies.")


def load_ratings(connection) -> None:
    """Load user ratings into PostgreSQL."""

    dataframe = read_parquet_file(
        RATINGS_PATH,
        "Ratings",
        {"userId", "movieId", "rating", "timestamp"},
        ["userId", "movieId"],
    )

    query = """
        INSERT INTO cinescale.ratings (
            user_id,
            movie_id,
            rating,
            rated_at
        )
        VALUES %s
        ON CONFLICT (user_id, movie_id)
        DO UPDATE SET
            rating = EXCLUDED.rating,
            rated_at = EXCLUDED.rated_at;
    """

    records = [
        (
            int(row.userId),
            int(row.movieId),
            float(row.rating),
            convert_timestamp_to_datetime(row.timestamp),
        )
        for row in dataframe.itertuples(index=False)
    ]

    with connection.cursor() as cursor:
        execute_values(cursor, query, records, page_size=5000)

    print(f"Loaded {len(records)} ratings.")


def load_user_factors(connection) -> None:
    """Load user embeddings into PostgreSQL."""

    dataframe = read_factors_file(
        USER_FACTORS_PATH,
        "User factors",
    )

    query = """
        INSERT INTO cinescale.user_factors (
            user_id,
            features
        )
        VALUES %s
        ON CONFLICT (user_id)
        DO UPDATE SET
            features = EXCLUDED.features;
    """

    records = [
        (
            int(row.id),
            convert_vector_to_pgvector(
                row.features
            ),
        )
        for row in dataframe.itertuples(
            index=False
        )
    ]

    with connection.cursor() as cursor:
        for batch_start in range(0, len(records), FACTOR_BATCH_SIZE):
            batch = records[batch_start : batch_start + FACTOR_BATCH_SIZE]
            execute_values(
                cursor,
                query,
                batch,
                page_size=5000,
            )
            connection.commit()

    print(
        f"Loaded {len(records)} "
        "user embeddings."
    )


def load_movie_factors(connection) -> None:
    """Load movie embeddings into PostgreSQL."""

    dataframe = read_factors_file(
        MOVIE_FACTORS_PATH,
        "Movie factors",
    )

    query = """
        INSERT INTO cinescale.movie_factors (
            movie_id,
            features
        )
        VALUES %s
        ON CONFLICT (movie_id)
        DO UPDATE SET
            features = EXCLUDED.features;
    """

    records = [
        (
            int(row.id),
            convert_vector_to_pgvector(
                row.features
            ),
        )
        for row in dataframe.itertuples(
            index=False
        )
    ]

    with connection.cursor() as cursor:
        for batch_start in range(0, len(records), FACTOR_BATCH_SIZE):
            batch = records[batch_start : batch_start + FACTOR_BATCH_SIZE]
            execute_values(
                cursor,
                query,
                batch,
                page_size=5000,
            )
            connection.commit()

    print(
        f"Loaded {len(records)} "
        "movie embeddings."
    )


def main() -> None:
    """Run the complete database loading process."""

    connection = None

    try:
        print("Connecting to PostgreSQL...")

        connection = get_database_connection()

        print(
            "Database connection established "
            "successfully."
        )

        print("Dropping HNSW index for fast bulk loading...")
        with connection.cursor() as cursor:
            drop_hnsw_index(cursor)
        connection.commit()
        print("HNSW index dropped.")

        load_movies(connection)
        load_ratings(connection)
        load_user_factors(connection)
        load_movie_factors(connection)

        connection.commit()

        print("Data loaded. Recreating HNSW index...")
        with connection.cursor() as cursor:
            create_hnsw_index(cursor)
        connection.commit()
        print("HNSW index recreated.")

        print(
            "Database loading completed "
            "successfully."
        )

    except Exception as error:
        if connection is not None:
            connection.rollback()

        print(
            f"Database loading failed: {error}"
        )
        raise

    finally:
        if connection is not None:
            connection.close()
            print("Database connection closed.")


if __name__ == "__main__":
    main()