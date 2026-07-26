import os
import sys
import logging
import math
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, DoubleType

# Ensure project root is in sys.path for direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.schema import config
from pipeline.spark_session import get_spark_session
from pipeline.etl import extract_data, transform_ratings, filter_low_support_items
from pipeline.export_embeddings import export_embeddings

logger = logging.getLogger(__name__)


def _get_top_k_recs_pandas_udf(item_ids_bc, item_matrix_bc, k_fetch: int):
    """Return a pandas_udf that computes top-k recommendations for a batch of users.

    Parameters
    ----------
    item_ids_bc
        Broadcast variable containing item id values.
    item_matrix_bc
        Broadcast variable containing item factor matrix.
    k_fetch
        Number of top-scoring items to fetch per user.
    """
    import pandas as pd
    import numpy as np
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import ArrayType, IntegerType

    @pandas_udf(ArrayType(IntegerType()))
    def _recs_udf(user_features_series: pd.Series) -> pd.Series:
        items = item_ids_bc.value
        i_matrix = item_matrix_bc.value
        user_matrix = np.vstack(user_features_series.values).astype(i_matrix.dtype)
        all_scores = user_matrix @ i_matrix.T
        result = []
        for scores in all_scores:
            top_idx = np.argpartition(scores, -k_fetch)[-k_fetch:]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            result.append(items[top_idx].tolist())
        del all_scores, user_matrix
        return pd.Series(result)

    return _recs_udf


def _ndcg_udf(recommendations, relevant_items, train_items, k: int = 10):
    """Compute binary-relevance NDCG@k for a single user.

    Parameters
    ----------
    recommendations
        Ranked list of recommended item ids.
    relevant_items
        Set of item ids considered relevant for the user.
    train_items
        Set of item ids to exclude from recommendations.
    k
        Number of recommendations to consider.
    """
    train_set = set(train_items) if train_items else set()
    relevant_set = set(relevant_items)

    ranked = []
    for item in recommendations:
        if item in train_set:
            continue
        ranked.append(item)
        if len(ranked) == k:
            break

    dcg = 0.0
    for i, item in enumerate(ranked, start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    ideal_len = min(k, len(relevant_set))
    idcg = 0.0
    for i in range(1, ideal_len + 1):
        idcg += 1.0 / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def _prepare_eval_data(
    model: ALS,
    train_df: DataFrame,
    test_df: DataFrame,
    spark: SparkSession,
    rating_threshold: float,
    user_col: str,
    item_col: str,
    rating_col: str,
    user_batch_size: int,
    arrow_batch_size: int,
    K_FETCH_BUFFER: int = 5000,
) -> Tuple[Any, Any, List[int], DataFrame, DataFrame]:
    """Collect item factors, broadcast them, and prepare evaluation DataFrames.

    Parameters
    ----------
    model
        Fitted ALS model.
    train_df
        Training ratings DataFrame.
    test_df
        Test ratings DataFrame.
    spark
        Active SparkSession.
    rating_threshold
        Minimum rating considered relevant.
    user_col
        Name of the user id column.
    item_col
        Name of the item id column.
    rating_col
        Name of the rating column.
    user_batch_size
        Number of users to evaluate in a single batch.
    arrow_batch_size
        Max Arrow records per batch for pandas UDF.
    K_FETCH_BUFFER
        Unused in this helper; kept for API compatibility with _batch_eval_data.

    Returns
    -------
    Tuple of (item_ids_bc, item_matrix_bc, test_users, relevant_items_df,
    train_items_df).
    """
    import gc
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col

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

    return item_ids_bc, item_matrix_bc, test_users, relevant_items_df, train_items_df


def _batch_eval_data(
    model: ALS,
    item_ids_bc,
    item_matrix_bc,
    test_users: List[int],
    relevant_items_df: DataFrame,
    train_items_df: DataFrame,
    k: int,
    user_batch_size: int,
    user_col: str,
    K_FETCH_BUFFER: int = 5000,
):
    """Yield joined recommendation DataFrames for each user batch.

    The caller is responsible for triggering the action on the yielded
    DataFrame (e.g., calling ``collect()`` or ``count()``). Cleanup of the
    intermediate ``user_batch_df`` and ``recs_df`` happens after each yield.
    """
    import gc
    from pyspark.sql.functions import col

    k_fetch = min(k + K_FETCH_BUFFER, len(item_ids_bc.value))
    get_top_k_recs = _get_top_k_recs_pandas_udf(item_ids_bc, item_matrix_bc, k_fetch)

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

        yield joined

        user_batch_df.unpersist()
        recs_df.unpersist()
        del user_batch_df, recs_df
        gc.collect()


def load_and_prepare_ratings(spark: SparkSession, raw_dir: str) -> DataFrame:
    """Extract, transform, filter low-support items, and return clean ratings DataFrame."""
    _, raw_ratings_df = extract_data(spark, raw_dir)
    clean_ratings = transform_ratings(raw_ratings_df)
    clean_ratings = filter_low_support_items(clean_ratings, min_ratings=10)
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

    item_ids_bc, item_matrix_bc, test_users, relevant_items_df, train_items_df = _prepare_eval_data(
        model,
        train_df,
        test_df,
        spark,
        rating_threshold,
        user_col,
        item_col,
        rating_col,
        user_batch_size,
        arrow_batch_size,
    )

    total_precision_sum = 0.0
    total_hits_sum = 0.0
    total_recall_sum = 0.0
    total_users_counted = 0

    for joined in _batch_eval_data(
        model,
        item_ids_bc,
        item_matrix_bc,
        test_users,
        relevant_items_df,
        train_items_df,
        k,
        user_batch_size,
        user_col,
    ):
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

        del metrics_row, agg
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


def compute_ndcg_at_k(
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
    """Compute NDCG@K using binary relevance (rating >= threshold).

    Memory-efficient batch-based approach that mirrors compute_precision_at_k.
    Averages NDCG only over users that have at least one relevant item.

    Returns:
        dict with key: ndcg
    """
    import gc
    from pyspark.sql import functions as F

    item_ids_bc, item_matrix_bc, test_users, relevant_items_df, train_items_df = _prepare_eval_data(
        model,
        train_df,
        test_df,
        spark,
        rating_threshold,
        user_col,
        item_col,
        rating_col,
        user_batch_size,
        arrow_batch_size,
    )

    ndcg_udf = udf(lambda recs, rel, train: _ndcg_udf(recs, rel, train, k=k), DoubleType())

    total_ndcg_sum = 0.0
    total_users_counted = 0

    for joined in _batch_eval_data(
        model,
        item_ids_bc,
        item_matrix_bc,
        test_users,
        relevant_items_df,
        train_items_df,
        k,
        user_batch_size,
        user_col,
    ):
        batch_ndcg = joined.withColumn(
            "ndcg",
            ndcg_udf(col("recommendations"), col("relevant_items"), col("train_items")),
        )

        agg = batch_ndcg.agg(
            F.sum("ndcg").alias("sum_ndcg"),
            F.count("*").alias("cnt"),
        ).collect()[0]

        if agg["cnt"]:
            total_ndcg_sum += float(agg["sum_ndcg"])
            total_users_counted += agg["cnt"]

        del batch_ndcg, agg
        gc.collect()

    item_ids_bc.unpersist()
    item_matrix_bc.unpersist()
    relevant_items_df.unpersist()
    train_items_df.unpersist()

    avg_ndcg = total_ndcg_sum / total_users_counted if total_users_counted else 0.0
    print(f"NDCG@{k}: {avg_ndcg:.4f}")
    print(f"(over {total_users_counted} users)")

    return {"ndcg": avg_ndcg}


def run_training() -> dict:
    """Main training pipeline: load, split, train, evaluate.

    Returns:
        dict with keys: rmse, precision_at_10, hit_rate_at_10, recall_at_10,
        ndcg_at_10, model
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

        logger.info("Computing NDCG@10 on test set...")
        ndcg_metrics = compute_ndcg_at_k(
            model, train_df, test_df, spark, k=10, rating_threshold=4.0
        )
        ndcg_at_10 = ndcg_metrics["ndcg"]
        logger.info(f"NDCG@10: {ndcg_at_10:.4f}")

        return {
            "rmse": rmse,
            "precision_at_10": precision_at_10,
            "hit_rate_at_10": hit_rate_at_10,
            "recall_at_10": recall_at_10,
            "ndcg_at_10": ndcg_at_10,
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
    print(f"  NDCG@10:       {results['ndcg_at_10']:.4f}")
    print(f"{'='*50}")
    
    user_factors_path, movie_factors_path = export_embeddings(
        results["model"], config.PROCESSED_DATA_DIR
    )
    print(f"User factors exported to: {user_factors_path}")
    print(f"Movie factors exported to: {movie_factors_path}")