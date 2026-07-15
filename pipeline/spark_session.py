import os
import sys
from pyspark.sql import SparkSession
from config.schema import config

def get_spark_session(app_name: str = "CineScale") -> SparkSession:
    """
    Creates or retrieves a SparkSession configured for CineScale.
    Determines execution mode (local vs cluster) using the SPARK_MASTER environment variable.
    """
    # Configure Python worker executable to match the current running python environment (e.g. virtual environment)
    # This avoids issues on Windows/macOS where the system python or a store stub is invoked instead.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    # Load settings from schema configuration
    driver_memory = config.SPARK_DRIVER_MEMORY
    shuffle_partitions = config.SPARK_SHUFFLE_PARTITIONS
    
    # Determine Master URL (default to local[*] if SPARK_MASTER is not set)
    master_url = os.environ.get("SPARK_MASTER", "local[*]")
    
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master_url)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    )
    
    return builder.getOrCreate()
