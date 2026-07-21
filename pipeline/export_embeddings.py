import os
import sys
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession

# Ensure project root is in sys.path for direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.schema import config
from pipeline.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def export_embeddings(
    model: ALSModel, processed_dir: str
) -> Tuple[str, str]:
    """Export user and movie factor matrices to Parquet via Pandas/PyArrow.

    Avoids PySpark's native Parquet writer which requires winutils on Windows.

    Args:
        model: Fitted ALS model with userFactors and itemFactors
        processed_dir: Output directory for Parquet files

    Returns:
        Tuple of (user_factors_path, movie_factors_path)
    """
    user_factors = model.userFactors
    movie_factors = model.itemFactors

    user_pdf = user_factors.toPandas()
    movie_pdf = movie_factors.toPandas()

    # Rename id -> userId/movieId to match database schema
    user_pdf = user_pdf.rename(columns={"id": "userId"})
    movie_pdf = movie_pdf.rename(columns={"id": "movieId"})

    user_pdf["features"] = user_pdf["features"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )
    movie_pdf["features"] = movie_pdf["features"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )

    # Validate Interface Contract: rank=50, integer IDs
    expected_dim = config.EMBEDDING_DIM
    for label, pdf, id_col in [("user", user_pdf, "userId"), ("movie", movie_pdf, "movieId")]:
        dims = pdf["features"].apply(len)
        if not dims.eq(expected_dim).all():
            raise ValueError(
                f"{label} factors have inconsistent dimensions: {dims.unique().tolist()}"
            )
        if pdf[id_col].dtype not in (np.int32, np.int64):
            raise TypeError(f"{label} factor IDs must be integer, got {pdf[id_col].dtype}")

    os.makedirs(processed_dir, exist_ok=True)
    user_path = os.path.join(processed_dir, config.USER_FACTORS_FILENAME)
    movie_path = os.path.join(processed_dir, config.MOVIE_FACTORS_FILENAME)

    user_pdf.to_parquet(user_path, index=False)
    movie_pdf.to_parquet(movie_path, index=False)

    return user_path, movie_path


def export_from_parquet(
    user_factors_parquet: str,
    movie_factors_parquet: str,
    processed_dir: str,
) -> Tuple[str, str]:
    """Load ALS model factors from Parquet and re-export to processed dir.

    Used when downloading model factors from Databricks.

    Args:
        user_factors_parquet: Path to downloaded user_factors.parquet
        movie_factors_parquet: Path to downloaded movie_factors.parquet
        processed_dir: Output directory for final processed files

    Returns:
        Tuple of (user_factors_path, movie_factors_path)
    """
    user_pdf = pd.read_parquet(user_factors_parquet)
    movie_pdf = pd.read_parquet(movie_factors_parquet)

    # Rename id -> userId/movieId if present
    if "id" in user_pdf.columns:
        user_pdf = user_pdf.rename(columns={"id": "userId"})
    if "id" in movie_pdf.columns:
        movie_pdf = movie_pdf.rename(columns={"id": "movieId"})

    user_pdf["features"] = user_pdf["features"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )
    movie_pdf["features"] = movie_pdf["features"].apply(
        lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
    )

    # Validate Interface Contract
    expected_dim = config.EMBEDDING_DIM
    for label, pdf, id_col in [("user", user_pdf, "userId"), ("movie", movie_pdf, "movieId")]:
        dims = pdf["features"].apply(len)
        if not dims.eq(expected_dim).all():
            raise ValueError(
                f"{label} factors have inconsistent dimensions: {dims.unique().tolist()}"
            )
        if pdf[id_col].dtype not in (np.int32, np.int64):
            raise TypeError(f"{label} factor IDs must be integer, got {pdf[id_col].dtype}")

    os.makedirs(processed_dir, exist_ok=True)
    user_path = os.path.join(processed_dir, config.USER_FACTORS_FILENAME)
    movie_path = os.path.join(processed_dir, config.MOVIE_FACTORS_FILENAME)

    user_pdf.to_parquet(user_path, index=False)
    movie_pdf.to_parquet(movie_path, index=False)

    logger.info(f"Exported user factors to {user_path}")
    logger.info(f"Exported movie factors to {movie_path}")

    return user_path, movie_path


def main():
    """CLI entry point to export embeddings from downloaded Databricks Parquet files.

    Usage:
        python -m pipeline.export_embeddings \
            --user-factors /path/to/user_factors.parquet \
            --movie-factors /path/to/movie_factors.parquet \
            --output-dir data/processed
    """
    import argparse

    parser = argparse.ArgumentParser(description="Export ALS embeddings from Databricks Parquet")
    parser.add_argument("--user-factors", required=True, help="Path to user_factors.parquet from Databricks")
    parser.add_argument("--movie-factors", required=True, help="Path to movie_factors.parquet from Databricks")
    parser.add_argument("--output-dir", default=config.PROCESSED_DATA_DIR, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    export_from_parquet(args.user_factors, args.movie_factors, args.output_dir)


if __name__ == "__main__":
    main()