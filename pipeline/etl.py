import os
import sys
import zipfile
import logging
from typing import Tuple

# Ensure project root is in sys.path for direct script execution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, trim
from pyspark.sql.types import IntegerType, StringType, FloatType, LongType

from config.schema import config
from pipeline.spark_session import get_spark_session

logger = logging.getLogger(__name__)


def extract_data(spark: SparkSession, raw_dir: str) -> Tuple[DataFrame, DataFrame]:
    """Extracts ratings.csv and movies.csv from raw_dir.

    If raw_dir does not exist but a corresponding zip file exists in data/raw/,
    extracts it first.

    Args:
        spark (SparkSession): The active Spark Session.
        raw_dir (str): Path to the raw data directory.

    Returns:
        Tuple[DataFrame, DataFrame]: A tuple containing the movies DataFrame and ratings DataFrame.

    Raises:
        FileNotFoundError: If the raw data directory or CSV files/zip do not exist.
        IOError: If zip file extraction fails.
    """
    raw_dir = os.path.normpath(raw_dir)
    if not os.path.exists(raw_dir):
        parent_dir = os.path.dirname(raw_dir)
        zip_name = os.path.basename(raw_dir) + ".zip"
        zip_path = os.path.join(parent_dir, zip_name)
        if os.path.exists(zip_path):
            logger.info(f"Extracting {zip_path} to {parent_dir}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Zip Slip Validation: check each member's resolved path to ensure it starts with parent_dir
                    resolved_parent = os.path.abspath(parent_dir)
                    for member in zip_ref.namelist():
                        member_path = os.path.abspath(os.path.join(resolved_parent, member))
                        if not member_path.startswith(resolved_parent + os.sep):
                            raise PermissionError(f"Attempted directory traversal in zip file: {member}")
                    
                    zip_ref.extractall(parent_dir)
            except (zipfile.BadZipFile, PermissionError) as e:
                raise IOError(f"Failed to extract zip file {zip_path}: {e}") from e
        else:
            raise FileNotFoundError(f"Raw data directory {raw_dir} and zip file {zip_path} not found.")
            
    movies_path = os.path.join(raw_dir, "movies.csv")
    ratings_path = os.path.join(raw_dir, "ratings.csv")
    
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"movies.csv not found at {movies_path}")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.csv not found at {ratings_path}")
        
    logger.info(f"Loading raw movies from {movies_path}")
    movies_df = spark.read.option("header", "true").csv(movies_path)
    
    logger.info(f"Loading raw ratings from {ratings_path}")
    ratings_df = spark.read.option("header", "true").csv(ratings_path)
    
    return movies_df, ratings_df


def transform_movies(movies_df: DataFrame) -> DataFrame:
    """
    Transforms the raw movies DataFrame by filtering null movieIds,
    casting columns, and filling remaining nulls in title/genres.

    Args:
        movies_df (DataFrame): The raw movies DataFrame.

    Returns:
        DataFrame: The cleaned and transformed movies DataFrame.
    """
    # Filter out rows where movieId is null, empty, or malformed
    filtered_df = movies_df.filter(col("movieId").isNotNull())
    filtered_df = filtered_df.filter(trim(col("movieId")).rlike(r"^\d+$"))

    # Cast columns: trimmed movieId to IntegerType, title and genres to StringType
    casted_df = filtered_df \
        .withColumn("movieId", trim(col("movieId")).cast(IntegerType())) \
        .withColumn("title", col("title").cast(StringType())) \
        .withColumn("genres", col("genres").cast(StringType()))

    # Fill remaining null values in title or genres with "Unknown"
    transformed_df = casted_df.fillna("Unknown", subset=["title", "genres"])

    return transformed_df


def transform_ratings(ratings_df: DataFrame) -> DataFrame:
    """
    Transforms the raw ratings DataFrame by filtering out null, empty,
    or malformed values in critical columns (userId, movieId, rating),
    and casting columns to their appropriate types.

    Note: Timestamp is non-critical and may be null.

    Args:
        ratings_df (DataFrame): The raw ratings DataFrame.

    Returns:
        DataFrame: The cleaned and transformed ratings DataFrame.
    """
    # Filter out rows where critical columns (userId, movieId, rating) are null
    filtered_df = ratings_df.filter(
        col("userId").isNotNull() &
        col("movieId").isNotNull() &
        col("rating").isNotNull()
    )

    # Trim and verify critical columns using regular expressions
    filtered_df = filtered_df.filter(
        trim(col("userId")).rlike(r"^\d+$") &
        trim(col("movieId")).rlike(r"^\d+$") &
        trim(col("rating")).rlike(r"^\d+(\.\d+)?$")
    )

    # Cast columns to their appropriate types
    transformed_df = filtered_df \
        .withColumn("userId", trim(col("userId")).cast(IntegerType())) \
        .withColumn("movieId", trim(col("movieId")).cast(IntegerType())) \
        .withColumn("rating", trim(col("rating")).cast(FloatType())) \
        .withColumn("timestamp", trim(col("timestamp")).cast(LongType()))

    return transformed_df


def filter_low_support_items(ratings_df: DataFrame, min_ratings: int = 10) -> DataFrame:
    """Filter out movies with fewer than min_ratings ratings.

    Args:
        ratings_df: The ratings DataFrame.
        min_ratings: Minimum number of ratings required per movie.

    Returns:
        Filtered ratings DataFrame.
    """
    from pyspark.sql.functions import count as spark_count

    item_counts = ratings_df.groupBy("movieId").agg(spark_count("*").alias("count"))
    valid_items = item_counts.filter(col("count") >= min_ratings).select("movieId")
    filtered_df = ratings_df.join(valid_items, on="movieId", how="inner")

    original_count = ratings_df.count()
    filtered_count = filtered_df.count()
    logger.info(f"Ratings before: {original_count}, after min_ratings={min_ratings} filter: {filtered_count}")

    return filtered_df


def load_data(movies_df: DataFrame, ratings_df: DataFrame, processed_dir: str) -> Tuple[str, str]:
    """
    Writes the cleaned movies and ratings DataFrames to processed_dir in Parquet format.
    Uses Pandas/PyArrow to avoid Hadoop winutils dependency on Windows.

    Args:
        movies_df (DataFrame): The cleaned movies DataFrame.
        ratings_df (DataFrame): The cleaned ratings DataFrame.
        processed_dir (str): The destination directory path.

    Returns:
        Tuple[str, str]: A tuple containing the resolved movies output path and ratings output path.
    """
    import pandas as pd

    movies_path = os.path.join(processed_dir, "movies_clean.parquet")
    ratings_path = os.path.join(processed_dir, "ratings_clean.parquet")

    os.makedirs(processed_dir, exist_ok=True)

    movies_df.toPandas().to_parquet(movies_path, index=False)
    ratings_df.toPandas().to_parquet(ratings_path, index=False)

    return movies_path, ratings_path


def run_etl() -> None:
    """
    Main ETL execution pipeline:
    1. Initializes Spark Session.
    2. Extracts raw CSV data.
    3. Transforms movies and ratings.
    4. Saves outputs as Parquet.
    5. Stops the Spark Session.
    """
    spark = None
    try:
        spark = get_spark_session("CineScaleETL")
        
        # Extract raw DataFrames
        logger.info("Extracting raw data...")
        raw_movies_df, raw_ratings_df = extract_data(spark, config.RAW_DATA_DIR)
        
        raw_movies_count = raw_movies_df.count()
        raw_ratings_count = raw_ratings_df.count()
        logger.info(f"Raw movies row count: {raw_movies_count}")
        logger.info(f"Raw ratings row count: {raw_ratings_count}")
        
        # Transform DataFrames
        logger.info("Transforming movies and ratings...")
        clean_movies_df = transform_movies(raw_movies_df)
        clean_ratings_df = transform_ratings(raw_ratings_df)
        
        # Filter low-support items
        clean_ratings_df = filter_low_support_items(clean_ratings_df, min_ratings=10)
        
        # Cache for downstream use
        clean_ratings_df.cache()
        
        clean_movies_count = clean_movies_df.count()
        clean_ratings_count = clean_ratings_df.count()
        logger.info(f"Clean movies row count: {clean_movies_count}")
        logger.info(f"Clean ratings row count: {clean_ratings_count}")
        
        # Log dropped row counts
        dropped_movies = raw_movies_count - clean_movies_count
        dropped_ratings = raw_ratings_count - clean_ratings_count
        logger.info(f"Dropped invalid movies: {dropped_movies}")
        logger.info(f"Dropped invalid ratings: {dropped_ratings}")
        
        # Load (save) data
        logger.info("Loading cleaned data to Parquet...")
        movies_path, ratings_path = load_data(clean_movies_df, clean_ratings_df, config.PROCESSED_DATA_DIR)
        logger.info(f"Cleaned movies output path: {movies_path}")
        logger.info(f"Cleaned ratings output path: {ratings_path}")
        
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_etl()


