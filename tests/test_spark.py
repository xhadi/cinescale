import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.schema import config
from pipeline.spark_session import get_spark_session

def test_spark_local_session(monkeypatch):
    """Verify that get_spark_session configures and initializes a local SparkSession."""
    monkeypatch.delenv("SPARK_MASTER", raising=False)

    spark = get_spark_session("TestLocalSpark")

    try:
        assert spark.conf.get("spark.master") == "local[*]"
        assert spark.conf.get("spark.app.name") == "TestLocalSpark"

        df = spark.createDataFrame([(1, "test")], ["id", "val"])
        assert df.count() == 1
    finally:
        spark.stop()

def test_spark_config_loading(monkeypatch):
    """Verify SparkSession builder sets configs when SPARK_MASTER environment is overridden."""
    monkeypatch.setenv("SPARK_MASTER", "spark://mock-master:7077")
    
    mock_builder = MagicMock()
    mock_builder.appName.return_value = mock_builder
    mock_builder.master.return_value = mock_builder
    mock_builder.config.return_value = mock_builder
    mock_builder.getOrCreate.return_value = MagicMock()
    
    with patch("pyspark.sql.SparkSession.builder", mock_builder):
        get_spark_session("MockApp")
        
        mock_builder.appName.assert_called_with("MockApp")
        mock_builder.master.assert_called_with("spark://mock-master:7077")
        
        mock_builder.config.assert_any_call("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
        mock_builder.config.assert_any_call("spark.sql.shuffle.partitions", str(config.SPARK_SHUFFLE_PARTITIONS))
        mock_builder.config.assert_any_call("spark.executor.memory", config.SPARK_EXECUTOR_MEMORY)
        mock_builder.config.assert_any_call("spark.default.parallelism", str(config.SPARK_DEFAULT_PARALLELISM))
