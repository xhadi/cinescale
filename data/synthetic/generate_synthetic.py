import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.schema import config

def generate_factors(num_records: int, dim: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic factor embeddings."""
    np.random.seed(seed)
    
    # Generate unique integer IDs starting from 1
    ids = np.arange(1, num_records + 1, dtype=np.int32)
    
    # Generate random floats in [-1.0, 1.0] for the embeddings
    embeddings = np.random.uniform(-1.0, 1.0, size=(num_records, dim)).astype(np.float32)
    
    # Convert numpy array to list of floats for each row
    features_list = [emb.tolist() for emb in embeddings]
    
    # Construct DataFrame
    df = pd.DataFrame({
        "id": ids,
        "features": features_list
    })
    
    return df

def main():
    print("Generating synthetic dataset...")
    
    # Create the output directory if it doesn't exist
    os.makedirs(config.SYNTHETIC_DATA_DIR, exist_ok=True)
    
    num_users = 50
    num_movies = 50
    dim = config.EMBEDDING_DIM
    
    user_df = generate_factors(num_users, dim, seed=42)
    movie_df = generate_factors(num_movies, dim, seed=43)
    
    user_path = config.SYNTHETIC_USER_FACTORS_PATH
    movie_path = config.SYNTHETIC_MOVIE_FACTORS_PATH
    
    # Write to Parquet using PyArrow engine
    user_df.to_parquet(user_path, index=False, engine="pyarrow")
    movie_df.to_parquet(movie_path, index=False, engine="pyarrow")
    
    print(f"Successfully wrote synthetic user factors to {user_path}")
    print(f"Successfully wrote synthetic movie factors to {movie_path}")
    
    # Quick sanity checks
    # Load back and verify
    u_loaded = pd.read_parquet(user_path)
    m_loaded = pd.read_parquet(movie_path)
    
    assert len(u_loaded) == num_users, f"Expected {num_users} users, got {len(u_loaded)}"
    assert len(m_loaded) == num_movies, f"Expected {num_movies} movies, got {len(m_loaded)}"
    
    assert list(u_loaded.columns) == ["id", "features"], f"Expected columns ['id', 'features'], got {list(u_loaded.columns)}"
    assert list(m_loaded.columns) == ["id", "features"], f"Expected columns ['id', 'features'], got {list(m_loaded.columns)}"
    
    # Check shape of features
    first_feat = u_loaded["features"].iloc[0]
    assert len(first_feat) == dim, f"Expected embedding size {dim}, got {len(first_feat)}"
    assert isinstance(first_feat[0], (float, np.float32, float)), f"Expected float, got {type(first_feat[0])}"
    
    print("Verification passed successfully!")

if __name__ == "__main__":
    main()
