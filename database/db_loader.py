from pathlib import Path
import os
import sys

import pandas as pd
import psycopg2
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.schema import config


load_dotenv(PROJECT_ROOT / ".env")


USER_FACTORS_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "user_factors.parquet"
)

MOVIE_FACTORS_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "movie_factors.parquet"
)


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


def read_factors_file(
    file_path: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Read and validate a factors Parquet file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file not found: {file_path}"
        )

    dataframe = pd.read_parquet(
        file_path,
        engine="pyarrow",
    )

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
        VALUES (%s, %s::vector)
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
        cursor.executemany(query, records)

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
        VALUES (%s, %s::vector)
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
        cursor.executemany(query, records)

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

        load_user_factors(connection)
        load_movie_factors(connection)

        connection.commit()

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