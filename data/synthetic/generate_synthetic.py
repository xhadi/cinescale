import os
import sys

import numpy as np
import pandas as pd


# File location:
# cinescale/data/synthetic/generate_synthetic.py
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.schema import config


def generate_factors(
    num_records: int,
    dim: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate reproducible synthetic factor embeddings."""

    if num_records <= 0:
        raise ValueError("num_records must be greater than zero")

    if dim <= 0:
        raise ValueError("dim must be greater than zero")

    rng = np.random.default_rng(seed)

    ids = np.arange(
        1,
        num_records + 1,
        dtype=np.int32,
    )

    embeddings = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(num_records, dim),
    ).astype(np.float32)

    return pd.DataFrame(
        {
            "id": ids,
            "features": embeddings.tolist(),
        }
    )


def validate_factors(
    dataframe: pd.DataFrame,
    expected_records: int,
    expected_dim: int,
    dataset_name: str,
) -> None:
    """Validate the generated factors DataFrame."""

    expected_columns = ["id", "features"]

    assert len(dataframe) == expected_records, (
        f"{dataset_name}: expected {expected_records} records, "
        f"but found {len(dataframe)}"
    )

    assert list(dataframe.columns) == expected_columns, (
        f"{dataset_name}: expected columns {expected_columns}, "
        f"but found {list(dataframe.columns)}"
    )

    assert dataframe["id"].is_unique, (
        f"{dataset_name}: IDs must be unique"
    )

    assert dataframe["id"].tolist() == list(
        range(1, expected_records + 1)
    ), (
        f"{dataset_name}: IDs must start from 1 and be sequential"
    )

    for row_number, features in enumerate(
        dataframe["features"],
        start=1,
    ):
        assert features is not None, (
            f"{dataset_name}: features are missing at row {row_number}"
        )

        assert len(features) == expected_dim, (
            f"{dataset_name}: expected embedding dimension "
            f"{expected_dim} at row {row_number}, "
            f"but found {len(features)}"
        )

        assert all(
            isinstance(value, (float, np.floating))
            for value in features
        ), (
            f"{dataset_name}: embedding at row {row_number} "
            "contains non-floating-point values"
        )

        assert all(
            -1.0 <= float(value) <= 1.0
            for value in features
        ), (
            f"{dataset_name}: embedding at row {row_number} "
            "contains values outside the range [-1.0, 1.0]"
        )


def main() -> None:
    """Generate, save, reload, and validate synthetic datasets."""

    print("Generating synthetic dataset...")

    os.makedirs(
        config.SYNTHETIC_DATA_DIR,
        exist_ok=True,
    )

    num_users = 50
    num_movies = 50
    embedding_dim = config.EMBEDDING_DIM

    user_factors = generate_factors(
        num_records=num_users,
        dim=embedding_dim,
        seed=42,
    )

    movie_factors = generate_factors(
        num_records=num_movies,
        dim=embedding_dim,
        seed=43,
    )

    user_path = config.SYNTHETIC_USER_FACTORS_PATH
    movie_path = config.SYNTHETIC_MOVIE_FACTORS_PATH

    user_factors.to_parquet(
        user_path,
        index=False,
        engine="pyarrow",
    )

    movie_factors.to_parquet(
        movie_path,
        index=False,
        engine="pyarrow",
    )

    print(
        f"Successfully wrote synthetic user factors to {user_path}"
    )

    print(
        f"Successfully wrote synthetic movie factors to {movie_path}"
    )

    loaded_users = pd.read_parquet(
        user_path,
        engine="pyarrow",
    )

    loaded_movies = pd.read_parquet(
        movie_path,
        engine="pyarrow",
    )

    validate_factors(
        dataframe=loaded_users,
        expected_records=num_users,
        expected_dim=embedding_dim,
        dataset_name="User factors",
    )

    validate_factors(
        dataframe=loaded_movies,
        expected_records=num_movies,
        expected_dim=embedding_dim,
        dataset_name="Movie factors",
    )

    print("Verification passed successfully!")


if __name__ == "__main__":
    main()