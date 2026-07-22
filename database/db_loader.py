from pathlib import Path
import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv


# Load environment variables from the .env file if available
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

USER_FACTORS_PATH = (
    PROJECT_ROOT / "data" / "synthetic" / "user_factors.parquet"
)

MOVIE_FACTORS_PATH = (
    PROJECT_ROOT / "data" / "synthetic" / "movie_factors.parquet"
)


def get_database_connection():
    """
    Create a connection to the PostgreSQL database.
    """

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        database=os.getenv("POSTGRES_DB", "cinescale"),
        user=os.getenv("POSTGRES_USER", "cinescale_admin"),
        password=os.getenv("POSTGRES_PASSWORD", "cinescale_password"),
    )


def convert_vector_to_pgvector(vector):
    """
    Convert a Python list into pgvector format.

    Example:
    [0.1, 0.2, 0.3]
    """

    return "[" + ",".join(str(float(value)) for value in vector) + "]"


def load_user_factors(connection):
    """
    Load user embeddings from the Parquet file into PostgreSQL.
    """

    if not USER_FACTORS_PATH.exists():
        raise FileNotFoundError(
            f"User factors file not found: {USER_FACTORS_PATH}"
        )

    dataframe = pd.read_parquet(USER_FACTORS_PATH)

    required_columns = {"id", "features"}

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "The user_factors.parquet file must contain "
            "'id' and 'features' columns."
        )

    query = """
        INSERT INTO cinescale.user_factors (user_id, features)
        VALUES (%s, %s::vector)
        ON CONFLICT (user_id)
        DO UPDATE SET features = EXCLUDED.features;
    """

    records = [
        (
            int(row["id"]),
            convert_vector_to_pgvector(row["features"]),
        )
        for _, row in dataframe.iterrows()
    ]

    with connection.cursor() as cursor:
        cursor.executemany(query, records)

    print(f"Loaded {len(records)} user embeddings.")


def load_movie_factors(connection):
    """
    Load movie embeddings from the Parquet file into PostgreSQL.
    """

    if not MOVIE_FACTORS_PATH.exists():
        raise FileNotFoundError(
            f"Movie factors file not found: {MOVIE_FACTORS_PATH}"
        )

    dataframe = pd.read_parquet(MOVIE_FACTORS_PATH)

    required_columns = {"id", "features"}

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "The movie_factors.parquet file must contain "
            "'id' and 'features' columns."
        )

    query = """
        INSERT INTO cinescale.movie_factors (movie_id, features)
        VALUES (%s, %s::vector)
        ON CONFLICT (movie_id)
        DO UPDATE SET features = EXCLUDED.features;
    """

    records = [
        (
            int(row["id"]),
            convert_vector_to_pgvector(row["features"]),
        )
        for _, row in dataframe.iterrows()
    ]

    with connection.cursor() as cursor:
        cursor.executemany(query, records)

    print(f"Loaded {len(records)} movie embeddings.")


def main():
    """
    Execute the complete database loading process.
    """

    connection = None

    try:
        print("Connecting to PostgreSQL...")

        connection = get_database_connection()

        print("Database connection established successfully.")

        load_user_factors(connection)
        load_movie_factors(connection)

        connection.commit()

        print("Database loading completed successfully.")

    except Exception as error:
        if connection is not None:
            connection.rollback()

        print(f"Database loading failed: {error}")
        raise

    finally:
        if connection is not None:
            connection.close()
            print("Database connection closed.")


if __name__ == "__main__":
    main()