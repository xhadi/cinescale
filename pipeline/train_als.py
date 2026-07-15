import os
import sys
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, explode
from pyspark.sql.types import StructType, StructField

from config.schema import config
from pipeline.spark_session import get_spark_session
from pipeline.etl import extract_data, transform_movies, transform_ratings

logger = logging.getLogger(__name__)


def load_and_prepare_ratings(spark: SparkSession, raw_dir: str) -> DataFrame:
    """Extract, transform, and return clean ratings DataFrame."""
    _, raw_ratings_df = extract_data(spark, raw_dir)
    clean_ratings = transform_ratings(raw_ratings_df)
    return clean_ratings


def split_data(
    ratings_df: DataFrame, train_ratio: float = 0.8, seed: int = 42
) -> Tuple[DataFrame, DataFrame]:
    """Split ratings into train and test sets using random assignment."""
    train_df, test_df = ratings_df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)
    return train_df, test_df


def train_als_model(
    train_df: DataFrame,
    rank: int = 50,
    max_iter: int = 10,
    reg_param: float = 0.1,
) -> ALS:
    """Train ALS model and return the fitted model."""
    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
    )
    model = als.fit(train_df)
    return model


def compute_rmse(
    model: ALS, test_df: DataFrame
) -> float:
    """Compute RMSE on the test set."""
    predictions = model.transform(test_df)
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction",
    )
    rmse = evaluator.evaluate(predictions)
    return rmse


def compute_precision_at_k(
    model: ALS, test_df: DataFrame, k: int = 10, rating_threshold: float = 4.0
) -> float:
    """Compute Precision@K averaged over all users in the test set.

    For each user, we:
    1. Get their top-K recommended item IDs from the model
    2. Identify items they actually rated >= rating_threshold in the test set
    3. Precision = |recommended ∩ relevant| / K
    """
    # Get distinct users in the test set
    test_users = test_df.select("userId").distinct()

    # Generate top-K recommendations for each test user
    user_recs = model.recommendForUserSubset(test_users, k)

    # Explode recommendations into (userId, recommended_movieId)
    recs_exploded = user_recs.select(
        col("userId"),
        explode(col("recommendations")).alias("rec"),
    ).select(
        col("userId"),
        col("rec.movieId").alias("recommended_movieId"),
    )

    # Get relevant items: items each user rated >= threshold in the test set
    relevant_items = (
        test_df.filter(col("rating") >= rating_threshold)
        .select("userId", "movieId")
        .withColumnRenamed("movieId", "relevant_movieId")
    )

    # Join recommended with relevant to find hits
    hits = recs_exploded.join(
        relevant_items,
        (recs_exploded.userId == relevant_items.userId)
        & (recs_exploded.recommended_movieId == relevant_items.relevant_movieId),
        "inner",
    ).select(recs_exploded.userId, recs_exploded.recommended_movieId)

    # Count hits per user
    hits_per_user = hits.groupBy("userId").count().withColumnRenamed("count", "hits")

    # Count total recommendations per user (should be K)
    recs_per_user = recs_exploded.groupBy("userId").count().withColumnRenamed("count", "total_recs")

    # Join and compute precision per user
    precision_per_user = hits_per_user.join(
        recs_per_user, "userId", "outer"
    ).fillna(0, subset=["hits"]).withColumn(
        "precision", col("hits") / col("total_recs")
    )

    # Average precision across all users
    avg_precision = precision_per_user.agg({"precision": "mean"}).collect()[0][0]
    return float(avg_precision) if avg_precision is not None else 0.0


def export_embeddings(
    model: ALS, processed_dir: str
) -> Tuple[str, str]:
    """Export user and movie factor matrices to Parquet via Pandas/PyArrow.

    Avoids PySpark's native Parquet writer which requires winutils on Windows.
    """
    user_factors = model.userFactors
    movie_factors = model.itemFactors

    user_pdf = user_factors.toPandas()
    movie_pdf = movie_factors.toPandas()

    user_pdf["features"] = user_pdf["features"].apply(lambda x: x.tolist() if hasattr(x, "tolist") else list(x))
    movie_pdf["features"] = movie_pdf["features"].apply(lambda x: x.tolist() if hasattr(x, "tolist") else list(x))

    os.makedirs(processed_dir, exist_ok=True)
    user_path = os.path.join(processed_dir, config.USER_FACTORS_FILENAME)
    movie_path = os.path.join(processed_dir, config.MOVIE_FACTORS_FILENAME)

    user_pdf.to_parquet(user_path, index=False)
    movie_pdf.to_parquet(movie_path, index=False)

    return user_path, movie_path


def run_training() -> dict:
    """Main training pipeline: load, split, train, evaluate, export.

    Returns:
        dict with keys: rmse, precision_at_10, user_factors_path, movie_factors_path
    """
    spark = None
    try:
        spark = get_spark_session("CineScaleALS")

        logger.info("Loading and preparing ratings data...")
        ratings_df = load_and_prepare_ratings(spark, config.RAW_DATA_DIR)
        total_count = ratings_df.count()
        logger.info(f"Total ratings loaded: {total_count}")

        logger.info("Splitting data into train/test (80/20)...")
        train_df, test_df = split_data(ratings_df)
        train_count = train_df.count()
        test_count = test_df.count()
        logger.info(f"Train ratings: {train_count}, Test ratings: {test_count}")

        logger.info(f"Training ALS model (rank={config.EMBEDDING_DIM}, maxIter=10, regParam=0.1)...")
        model = train_als_model(
            train_df,
            rank=config.EMBEDDING_DIM,
            max_iter=10,
            reg_param=0.1,
        )

        logger.info("Computing RMSE on test set...")
        rmse = compute_rmse(model, test_df)
        logger.info(f"RMSE: {rmse:.4f}")

        logger.info("Computing Precision@10 on test set...")
        precision_at_10 = compute_precision_at_k(model, test_df, k=10, rating_threshold=4.0)
        logger.info(f"Precision@10: {precision_at_10:.4f}")

        logger.info("Exporting embeddings to Parquet...")
        user_path, movie_path = export_embeddings(model, config.PROCESSED_DATA_DIR)
        logger.info(f"User factors: {user_path}")
        logger.info(f"Movie factors: {movie_path}")

        return {
            "rmse": rmse,
            "precision_at_10": precision_at_10,
            "user_factors_path": user_path,
            "movie_factors_path": movie_path,
        }

    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    logging.basicConfig(level=logging.INFO)
    results = run_training()
    print(f"\n{'='*50}")
    print(f"Training Results:")
    print(f"  RMSE:          {results['rmse']:.4f}")
    print(f"  Precision@10:  {results['precision_at_10']:.4f}")
    print(f"  User factors:  {results['user_factors_path']}")
    print(f"  Movie factors: {results['movie_factors_path']}")
    print(f"{'='*50}")
