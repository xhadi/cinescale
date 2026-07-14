# CineScale Section 1 Design — Schema Configuration & Synthetic Data

## 1. Goal Description
The goal is to define the project's interface contract as code and generate a tiny synthetic dataset that matches the schema exactly. This unblocks downstream development (database loading, streamlit frontend, backend queries) by providing a realistic data format before the full PySpark ML training pipeline is complete.

## 2. Component Details

### A. Configuration Schema (`config/schema.py`)
This file will serve as the central source of truth for configuration variables. We will use Python's built-in `dataclasses` and `os` module combined with `python-dotenv` to load settings from environment variables.

Key constants defined:
*   `EMBEDDING_DIM` = 50 (The dimension size `rank` of the latent factor vectors).
*   `DATASET_SIZE`: Either `'small'` or `'32m'`.
*   Input/Output paths resolved dynamically relative to project root:
    *   `RAW_DATA_DIR`: Path to the raw CSV files (`data/raw/ml-latest-small` or `data/raw/ml-32m`).
    *   `PROCESSED_DATA_DIR`: Path to output directory (`data/processed`).
    *   `SYNTHETIC_DATA_DIR`: Path to synthetic directory (`data/synthetic`).
    *   `USER_FACTORS_FILENAME`: `'user_factors.parquet'`
    *   `MOVIE_FACTORS_FILENAME`: `'movie_factors.parquet'`
*   Spark configuration keys:
    *   `SPARK_DRIVER_MEMORY`: Driver memory limit (e.g. `'4g'` for small, `'8g'` for 32M).
    *   `SPARK_SHUFFLE_PARTITIONS`: Spark partition count (e.g. `8` for small, `200` for 32M).

### B. Synthetic Data Generation (`data/synthetic/generate_synthetic.py`)
This script will programmatically generate mock latent factor vectors to simulate the output of the ALS algorithm.
*   **Methodology:**
    *   Generate 50 synthetic user IDs (range 1 to 50).
    *   Generate 50 synthetic movie IDs (range 1 to 50).
    *   For each ID, generate a random 50-dimensional vector of floats (e.g., normally distributed or uniformly distributed floats in `[-1.0, 1.0]`).
    *   Construct Pandas DataFrames with columns:
        *   `id` (integer)
        *   `features` (list of floats)
    *   Write the DataFrames to `user_factors.parquet` and `movie_factors.parquet` using PyArrow in `data/synthetic/`.

## 3. Verification Plan

### Automated Verification
*   Verify that `config/schema.py` loads successfully and displays paths correctly.
*   Run the synthetic data generator and check that the resulting parquet files exist.
*   Verify the schema of generated parquet files by reading them back via Pandas:
    *   Assert output file types are Parquet.
    *   Assert table columns are precisely `['id', 'features']`.
    *   Assert `features` column contains lists/arrays of length 50.
    *   Assert `id` data type is integer.
