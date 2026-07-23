import os
import sys
from pyspark.sql import SparkSession
from config.schema import config


def get_spark_session(app_name: str = "CineScale") -> SparkSession:
    """Creates or retrieves a SparkSession configured for CineScale.

    Uses local mode by default, or a custom master via SPARK_MASTER env var.
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    master_url = os.environ.get("SPARK_MASTER", "local[*]")
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master_url)
        .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", config.SPARK_EXECUTOR_MEMORY)
        .config("spark.sql.shuffle.partitions", str(config.SPARK_SHUFFLE_PARTITIONS))
        .config("spark.default.parallelism", str(config.SPARK_DEFAULT_PARALLELISM))
    )

    spark = builder.getOrCreate()
    try:
        spark.sparkContext.setCheckpointDir(
            os.path.join(config.PROCESSED_DATA_DIR, "checkpoints")
        )
    except Exception:
        pass
    return spark
