import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.schema import config
from data.synthetic.generate_synthetic import generate_factors, main as run_generator

def test_config_paths():
    """Verify that configuration paths are absolute and resolved correctly."""
    assert os.path.isabs(config.PROJECT_ROOT)
    assert os.path.isabs(config.RAW_DATA_DIR)
    assert os.path.isabs(config.PROCESSED_DATA_DIR)
    assert os.path.isabs(config.SYNTHETIC_DATA_DIR)
    
    assert config.EMBEDDING_DIM == 50
    assert config.USER_FACTORS_FILENAME == "user_factors.parquet"
    assert config.MOVIE_FACTORS_FILENAME == "movie_factors.parquet"

def test_generate_factors():
    """Verify generate_factors function returns expected schema and shapes."""
    num_records = 15
    dim = 50
    df = generate_factors(num_records, dim, seed=123)
    
    assert len(df) == num_records
    assert list(df.columns) == ["id", "features"]
    assert df["id"].dtype in (np.int32, np.int64)
    
    # Check features dimension and types
    first_row = df.iloc[0]
    assert len(first_row["features"]) == dim
    assert isinstance(first_row["features"][0], float)

def test_synthetic_files_creation(tmp_path, monkeypatch):
    """Verify the main entrypoint generates the files and they pass asserts."""
    # Mock config paths to use a temporary directory for tests
    synthetic_dir = tmp_path / "synthetic"
    
    from config.schema import AppConfig
    # We patch the property values on the AppConfig class
    monkeypatch.setattr(AppConfig, "SYNTHETIC_DATA_DIR", property(lambda self: str(synthetic_dir)))
    monkeypatch.setattr(AppConfig, "SYNTHETIC_USER_FACTORS_PATH", property(lambda self: str(synthetic_dir / "user_factors.parquet")))
    monkeypatch.setattr(AppConfig, "SYNTHETIC_MOVIE_FACTORS_PATH", property(lambda self: str(synthetic_dir / "movie_factors.parquet")))
    
    # Run the generator main function
    run_generator()
    
    # Check if files exist
    user_file = synthetic_dir / "user_factors.parquet"
    movie_file = synthetic_dir / "movie_factors.parquet"
    assert user_file.exists()
    assert movie_file.exists()
    
    # Read and verify content
    user_df = pd.read_parquet(user_file)
    movie_df = pd.read_parquet(movie_file)
    
    assert len(user_df) == 50
    assert len(movie_df) == 50
    assert len(user_df.iloc[0]["features"]) == 50
