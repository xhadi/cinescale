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

# Ensure project root is in sys.path for direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.schema import config
from pipeline.spark_session import get_spark_session
from pipeline.etl import extract_data, transform_ratings

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
    max_iter: int = 25,
    reg_param: float = 0.05,
) -> ALS:
    """Train ALS model with implicit preferences and return the fitted model."""
    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        implicitPrefs=True,
        alpha=40,
        coldStartStrategy="drop",
        checkpointInterval=5,
        seed=42,
    )
    model = als.fit(train_df)
    model.userFactors.cache()
    model.itemFactors.cache()
    model.userFactors.count()
    model.itemFactors.count()
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
    model: ALS, train_df: DataFrame, test_df: DataFrame, k: int = 10, rating_threshold: float = 4.0
) -> float:
    """Compute Precision@K averaged over all users in the test set.

    For each user, we:
    1. Get their top items from the model
    2. Filter out items they already rated in the training set
    3. Take the top K items
    4. Identify items they actually rated >= rating_threshold in the test set
    5. Precision = |recommended ∩ relevant| / K
    """
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number

    test_users = test_df.select("userId").distinct()

    # Request more recommendations to ensure we have K after filtering
    user_recs = model.recommendForUserSubset(test_users, k + 500)

    recs_exploded = user_recs.select(
        col("userId"),
        explode(col("recommendations")).alias("rec"),
    ).select(
        col("userId"),
        col("rec.movieId").alias("recommended_movieId"),
        col("rec.rating").alias("predicted_rating"),
    )

    # Exclude items the user has already rated in train_df
    train_items = train_df.select(
        col("userId"), col("movieId").alias("train_movieId")
    )
    
    recs_filtered = recs_exploded.join(
        train_items,
        (recs_exploded.userId == train_items.userId)
        & (recs_exploded.recommended_movieId == train_items.train_movieId),
        "left_anti",
    )

    # Take top K for each user
    windowSpec = Window.partitionBy("userId").orderBy(col("predicted_rating").desc())
    top_k_recs = recs_filtered.withColumn("rank", row_number().over(windowSpec)).filter(col("rank") <= k)

    relevant_items = (
        test_df.filter(col("rating") >= rating_threshold)
        .select("userId", "movieId")
        .withColumnRenamed("movieId", "relevant_movieId")
    )

    hits = top_k_recs.join(
        relevant_items,
        (top_k_recs.userId == relevant_items.userId)
        & (top_k_recs.recommended_movieId == relevant_items.relevant_movieId),
        "inner",
    ).select(top_k_recs.userId, top_k_recs.recommended_movieId)

    hits_per_user = hits.groupBy("userId").count().withColumnRenamed("count", "hits")

    recs_per_user = top_k_recs.groupBy("userId").count().withColumnRenamed("count", "total_recs")

    precision_per_user = hits_per_user.join(
        recs_per_user, "userId", "outer"
    ).fillna(0, subset=["hits"]).withColumn(
        "precision", col("hits") / col("total_recs")
    )

    avg_precision = precision_per_user.agg({"precision": "mean"}).collect()[0][0]
    return float(avg_precision) if avg_precision is not None else 0.0


def run_training() -> dict:
    """Main training pipeline: load, split, train, evaluate.

    Returns:
        dict with keys: rmse, precision_at_10, model
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
        train_df.cache()
        test_df.cache()
        train_count = train_df.count()
        test_count = test_df.count()
        logger.info(f"Train ratings: {train_count}, Test ratings: {test_count}")

        logger.info(f"Training ALS model (rank={config.EMBEDDING_DIM}, maxIter=25, regParam=0.05, implicitPrefs=True)...")
        model = train_als_model(
            train_df,
            rank=config.EMBEDDING_DIM,
            max_iter=25,
            reg_param=0.05,
        )

        logger.info("Computing RMSE on test set...")
        rmse = compute_rmse(model, test_df)
        logger.info(f"RMSE: {rmse:.4f}")

        logger.info("Computing Precision@10 on test set...")
        precision_at_10 = compute_precision_at_k(model, train_df, test_df, k=10, rating_threshold=4.0)
        logger.info(f"Precision@10: {precision_at_10:.4f}")

        return {
            "rmse": rmse,
            "precision_at_10": precision_at_10,
            "model": model,
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
    print(f"{'='*50}")
