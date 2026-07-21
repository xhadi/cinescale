import os
import sys
from pyspark.sql import SparkSession
from config.schema import config
from config.databricks_config import databricks_config


def get_spark_session(app_name: str = "CineScale") -> SparkSession:
    """
    Creates or retrieves a SparkSession configured for CineScale.
    Supports three modes:
    - Databricks Connect (if DATABRICKS_CONNECT_ENABLED=true)
    - Local/Cluster via SPARK_MASTER env var (default: local[*])
    """
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    driver_memory = config.SPARK_DRIVER_MEMORY
    shuffle_partitions = config.SPARK_SHUFFLE_PARTITIONS

    if databricks_config.ENABLE_CONNECT:
        if not all([databricks_config.WORKSPACE_URL, databricks_config.TOKEN, databricks_config.CLUSTER_ID]):
            raise ValueError(
                "Databricks Connect enabled but missing required env vars: "
                "DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_CLUSTER_ID"
            )
        builder = (
            SparkSession.builder
            .appName(app_name)
            .config("spark.databricks.service.client.enabled", "true")
            .config("spark.databricks.service.address", databricks_config.WORKSPACE_URL)
            .config("spark.databricks.service.token", databricks_config.TOKEN)
            .config("spark.databricks.service.clusterId", databricks_config.CLUSTER_ID)
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        )
    else:
        master_url = os.environ.get("SPARK_MASTER", "local[*]")
        builder = (
            SparkSession.builder
            .appName(app_name)
            .master(master_url)
            .config("spark.driver.memory", driver_memory)
            .config("spark.executor.memory", config.SPARK_EXECUTOR_MEMORY)
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
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
