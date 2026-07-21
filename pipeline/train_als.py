import os
import sys
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
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
    model: ALS,
    train_df: DataFrame,
    test_df: DataFrame,
    spark: SparkSession,
    k: int = 10,
    rating_threshold: float = 4.0,
    user_batch_size: int = 2000,
    arrow_batch_size: int = 500,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
) -> dict:
    """Compute Precision@K, Hit Rate, and Recall using batch-based approach.

    Memory-efficient for large datasets by processing users in batches
    with pandas_udf and broadcast variables.

    Returns:
        dict with keys: precision, hit_rate, recall
    """
    import gc
    from pyspark.sql import functions as F
    from pyspark.sql.functions import pandas_udf, col
    from pyspark.sql.types import ArrayType, IntegerType

    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", str(arrow_batch_size))

    print("1. Collecting Item Factors to driver...")
    item_factors_pdf = model.itemFactors.toPandas()
    item_ids = item_factors_pdf["id"].values.astype(np.int32)
    item_matrix = np.vstack(item_factors_pdf["features"].values).astype(np.float32)
    print(f"   Items: {len(item_ids)}, item_matrix memory: {item_matrix.nbytes / (1024**2):.2f} MB")
    del item_factors_pdf
    gc.collect()

    print("2. Broadcasting item data...")
    item_ids_bc = spark.sparkContext.broadcast(item_ids)
    item_matrix_bc = spark.sparkContext.broadcast(item_matrix)

    K_FETCH_BUFFER = 5000

    @pandas_udf(ArrayType(IntegerType()))
    def get_top_k_recs(user_features_series: pd.Series) -> pd.Series:
        items = item_ids_bc.value
        i_matrix = item_matrix_bc.value
        user_matrix = np.vstack(user_features_series.values).astype(i_matrix.dtype)
        all_scores = user_matrix @ i_matrix.T

        K_fetch = min(k + K_FETCH_BUFFER, len(items))
        result = []
        for scores in all_scores:
            top_idx = np.argpartition(scores, -K_fetch)[-K_fetch:]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            result.append(items[top_idx].tolist())

        del all_scores, user_matrix
        return pd.Series(result)

    print("3. Collecting distinct test users...")
    test_users = [r[user_col] for r in test_df.select(user_col).distinct().collect()]
    print(f"   {len(test_users)} distinct test users -> batching by {user_batch_size}")

    print("3a. Preparing relevant items per user (test set, rating >= threshold)...")
    model_item_ids_df = model.itemFactors.select(col("id").alias(item_col)).distinct()

    relevant_items_df = (
        test_df.filter(col(rating_col) >= rating_threshold)
        .join(model_item_ids_df, on=item_col, how="inner")
        .groupBy(user_col)
        .agg(F.collect_set(item_col).alias("relevant_items"))
        .filter(F.size("relevant_items") > 0)
    )
    relevant_items_df.cache()
    n_relevant_users = relevant_items_df.count()
    print(f"   Number of users with model-known relevant items: {n_relevant_users}")

    print("3b. Preparing each user's training-set items for exclusion...")
    train_items_df = (
        train_df.groupBy(user_col)
        .agg(F.collect_set(item_col).alias("train_items"))
    )
    train_items_df.cache()
    train_items_df.count()

    total_precision_sum = 0.0
    total_hits_sum = 0.0
    total_recall_sum = 0.0
    total_users_counted = 0
    n_batches = (len(test_users) + user_batch_size - 1) // user_batch_size

    for b in range(n_batches):
        batch_users = test_users[b * user_batch_size : (b + 1) * user_batch_size]
        print(f"   Batch {b + 1}/{n_batches} - {len(batch_users)} users")

        user_batch_df = (
            model.userFactors.filter(col("id").isin(batch_users))
            .withColumnRenamed("id", user_col)
        )

        recs_df = user_batch_df.withColumn(
            "recommendations", get_top_k_recs(col("features"))
        ).select(user_col, "recommendations")

        joined = (
            recs_df.join(relevant_items_df, on=user_col, how="inner")
            .join(train_items_df, on=user_col, how="left")
        )

        metrics_row = joined.select(
            F.expr(
                f"size(array_intersect("
                f"slice(array_except(recommendations, coalesce(train_items, array())), 1, {k}), "
                f"relevant_items)) / {k}"
            ).alias("precision"),
            F.expr(
                f"IF(size(array_intersect("
                f"slice(array_except(recommendations, coalesce(train_items, array())), 1, {k}), "
                f"relevant_items)) > 0, 1.0, 0.0)"
            ).alias("hit"),
            F.expr(
                f"size(array_intersect("
                f"slice(array_except(recommendations, coalesce(train_items, array())), 1, {k}), "
                f"relevant_items)) / size(relevant_items)"
            ).alias("recall"),
        )

        agg = metrics_row.agg(
            F.sum("precision").alias("sum_p"),
            F.sum("hit").alias("sum_hit"),
            F.sum("recall").alias("sum_r"),
            F.count("*").alias("cnt"),
        ).collect()[0]

        if agg["cnt"]:
            total_precision_sum += float(agg["sum_p"])
            total_hits_sum += float(agg["sum_hit"])
            total_recall_sum += float(agg["sum_r"])
            total_users_counted += agg["cnt"]

        user_batch_df.unpersist()
        recs_df.unpersist()
        del user_batch_df, recs_df, joined, metrics_row
        gc.collect()

    item_ids_bc.unpersist()
    item_matrix_bc.unpersist()
    relevant_items_df.unpersist()
    train_items_df.unpersist()

    avg_precision = total_precision_sum / total_users_counted if total_users_counted else 0.0
    avg_hit_rate = total_hits_sum / total_users_counted if total_users_counted else 0.0
    avg_recall = total_recall_sum / total_users_counted if total_users_counted else 0.0

    print(f"Precision@{k}: {avg_precision:.4f}")
    print(f"Hit Rate@{k}: {avg_hit_rate:.4f}")
    print(f"Recall@{k}: {avg_recall:.4f}")
    print(f"(over {total_users_counted} users)")

    return {"precision": avg_precision, "hit_rate": avg_hit_rate, "recall": avg_recall}


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
        metrics = compute_precision_at_k(
            model, train_df, test_df, spark, k=10, rating_threshold=4.0
        )
        precision_at_10 = metrics["precision"]
        hit_rate_at_10 = metrics["hit_rate"]
        recall_at_10 = metrics["recall"]
        logger.info(f"Precision@10: {precision_at_10:.4f}")
        logger.info(f"Hit Rate@10: {hit_rate_at_10:.4f}")
        logger.info(f"Recall@10: {recall_at_10:.4f}")

        return {
            "rmse": rmse,
            "precision_at_10": precision_at_10,
            "hit_rate_at_10": hit_rate_at_10,
            "recall_at_10": recall_at_10,
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
    print(f"  Hit Rate@10:   {results['hit_rate_at_10']:.4f}")
    print(f"  Recall@10:     {results['recall_at_10']:.4f}")
    print(f"{'='*50}")
