import os
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import IntegerType, FloatType, StructType, StructField

from config.schema import config
from pipeline.spark_session import get_spark_session
from pipeline.train_als import (
    load_and_prepare_ratings,
    split_data,
    train_als_model,
    compute_rmse,
    compute_precision_at_k,
)
from pipeline.export_embeddings import export_embeddings
from pipeline.export_embeddings import export_embeddings


@pytest.fixture(scope="module")
def spark() -> Generator[SparkSession, None, None]:
    """Fixture to initialize and stop the SparkSession for model testing."""
    spark_sess = get_spark_session("ModelTest")
    yield spark_sess
    spark_sess.stop()


@pytest.fixture(scope="module")
def sample_ratings(spark: SparkSession) -> DataFrame:
    """Create a small synthetic ratings DataFrame for unit tests."""
    schema = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", IntegerType(), True),
    ])
    data = [
        (1, 1, 4.0, 100),
        (1, 2, 3.0, 101),
        (1, 3, 5.0, 102),
        (2, 1, 2.0, 103),
        (2, 2, 4.0, 104),
        (2, 3, 3.0, 105),
        (3, 1, 5.0, 106),
        (3, 2, 1.0, 107),
        (3, 3, 4.0, 108),
        (4, 1, 3.0, 109),
        (4, 2, 5.0, 110),
        (4, 3, 2.0, 111),
        (5, 1, 4.0, 112),
        (5, 2, 4.0, 113),
        (5, 3, 5.0, 114),
        (6, 1, 1.0, 115),
        (6, 2, 2.0, 116),
        (6, 3, 3.0, 117),
        (7, 1, 5.0, 118),
        (7, 2, 4.0, 119),
        (7, 3, 1.0, 120),
        (8, 1, 2.0, 121),
        (8, 2, 3.0, 122),
        (8, 3, 4.0, 123),
        (9, 1, 4.0, 124),
        (9, 2, 5.0, 125),
        (9, 3, 2.0, 126),
        (10, 1, 3.0, 127),
        (10, 2, 4.0, 128),
        (10, 3, 5.0, 129),
    ]
    return spark.createDataFrame(data, schema)


def test_split_data_preserves_counts(sample_ratings: DataFrame) -> None:
    """Test that split_data preserves total row count."""
    train_df, test_df = split_data(sample_ratings)
    total = sample_ratings.count()
    train_count = train_df.count()
    test_count = test_df.count()
    assert train_count + test_count == total


def test_split_data_no_overlap(sample_ratings: DataFrame) -> None:
    """Test that train and test sets have no overlapping rows."""
    train_df, test_df = split_data(sample_ratings)
    train_rows = set((r.userId, r.movieId) for r in train_df.collect())
    test_rows = set((r.userId, r.movieId) for r in test_df.collect())
    assert len(train_rows & test_rows) == 0


def test_split_data_reproducible(sample_ratings: DataFrame) -> None:
    """Test that split_data produces identical results with the same seed."""
    train1, test1 = split_data(sample_ratings, seed=42)
    train2, test2 = split_data(sample_ratings, seed=42)
    train_rows1 = set((r.userId, r.movieId) for r in train1.collect())
    train_rows2 = set((r.userId, r.movieId) for r in train2.collect())
    test_rows1 = set((r.userId, r.movieId) for r in test1.collect())
    test_rows2 = set((r.userId, r.movieId) for r in test2.collect())
    assert train_rows1 == train_rows2
    assert test_rows1 == test_rows2


def test_train_als_model_returns_fitted_model(sample_ratings: DataFrame) -> None:
    """Test that train_als_model returns a valid ALS model."""
    train_df, _ = split_data(sample_ratings)
    model = train_als_model(train_df, rank=5, max_iter=3, reg_param=0.1)
    assert model is not None
    assert hasattr(model, "userFactors")
    assert hasattr(model, "itemFactors")
    assert model.userFactors.count() > 0
    assert model.itemFactors.count() > 0


def test_compute_rmse_returns_positive_float(sample_ratings: DataFrame) -> None:
    """Test that compute_rmse returns a positive float value."""
    train_df, test_df = split_data(sample_ratings)
    model = train_als_model(train_df, rank=5, max_iter=3, reg_param=0.1)
    rmse = compute_rmse(model, test_df)
    assert isinstance(rmse, float)
    assert rmse >= 0.0


def test_compute_precision_at_k(spark: SparkSession, sample_ratings: DataFrame) -> None:
    """Test that compute_precision_at_k returns a dict with precision, hit_rate, recall."""
    train_df, test_df = split_data(sample_ratings)
    model = train_als_model(train_df, rank=5, max_iter=3, reg_param=0.1)
    metrics = compute_precision_at_k(model, train_df, test_df, spark, k=10, rating_threshold=4.0)
    assert isinstance(metrics, dict)
    assert "precision" in metrics
    assert "hit_rate" in metrics
    assert "recall" in metrics
    for key in ("precision", "hit_rate", "recall"):
        assert isinstance(metrics[key], float)
        assert 0.0 <= metrics[key] <= 1.0


def test_export_embeddings(tmp_path: Path) -> None:
    """Test that export_embeddings writes valid Parquet files."""
    mock_model = MagicMock()
    mock_model.userFactors = MagicMock()
    mock_model.userFactors.toPandas.return_value = pd.DataFrame({
        "id": [1, 2],
        "features": [[0.1] * 50, [0.2] * 50],
    })
    mock_model.itemFactors = MagicMock()
    mock_model.itemFactors.toPandas.return_value = pd.DataFrame({
        "id": [1, 2, 3],
        "features": [[0.3] * 50, [0.4] * 50, [0.5] * 50],
    })

    processed_dir = str(tmp_path / "processed")

    user_path, movie_path = export_embeddings(mock_model, processed_dir)

    assert os.path.exists(user_path)
    assert os.path.exists(movie_path)

    user_pdf = pd.read_parquet(user_path)
    movie_pdf = pd.read_parquet(movie_path)

    assert len(user_pdf) == 2
    assert len(movie_pdf) == 3
    assert list(user_pdf.columns) == ["id", "features"]
    assert list(movie_pdf.columns) == ["id", "features"]
    assert len(user_pdf.iloc[0]["features"]) == 50
    assert user_pdf["id"].dtype in (pd.Int32Dtype(), pd.Int64Dtype(), "int32", "int64")
    assert movie_pdf["id"].dtype in (pd.Int32Dtype(), pd.Int64Dtype(), "int32", "int64")


def test_model_e2e_real_data(spark: SparkSession, tmp_path: Path) -> None:
    """Integration test: full training pipeline on real MovieLens Small data."""
    raw_dir = config.RAW_DATA_DIR
    if not os.path.exists(raw_dir):
        pytest.skip(f"Local raw directory {raw_dir} does not exist. Skipping integration test.")

    ratings_df = load_and_prepare_ratings(spark, raw_dir)
    assert ratings_df.count() > 0

    train_df, test_df = split_data(ratings_df)
    assert train_df.count() > 0
    assert test_df.count() > 0

    model = train_als_model(train_df, rank=config.EMBEDDING_DIM, max_iter=5, reg_param=0.1)
    assert model.userFactors.count() > 0
    assert model.itemFactors.count() > 0

    rmse = compute_rmse(model, test_df)
    assert isinstance(rmse, float)
    assert rmse >= 0.0

    metrics = compute_precision_at_k(model, train_df, test_df, spark, k=10, rating_threshold=4.0)
    assert isinstance(metrics, dict)
    assert "precision" in metrics
    assert "hit_rate" in metrics
    assert "recall" in metrics
    for key in ("precision", "hit_rate", "recall"):
        assert isinstance(metrics[key], float)
        assert 0.0 <= metrics[key] <= 1.0
