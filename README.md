# CineScale

Movie recommendation system using PySpark ALS collaborative filtering, PostgreSQL with pgvector for similarity search, and a Streamlit frontend.

## Architecture

```
Raw CSV/Zip → PySpark ETL → Parquet → ALS Training → Embeddings → PostgreSQL/pgvector → Streamlit UI
```

| Layer | Key file | Purpose |
|-------|----------|---------|
| Config | `config/schema.py` | Singleton config from `.env`; paths, Spark memory, embedding dim |
| ETL | `pipeline/etl.py` | Clean raw MovieLens data, filter low-support items, write Parquet |
| Training | `pipeline/train_als.py` | Train ALS (rank=50, implicit), evaluate RMSE/Precision@10/NDCG@10, export factors |
| Database | `database/db_loader.py` | Bulk-load Parquet into PostgreSQL with HNSW cosine index |
| Backend | `app/db_queries.py` | Recommendations, user history, search, trending, benchmark |
| Frontend | `app/app.py` | Streamlit UI with sidebar profile, recommendations, browse, health |

## Prerequisites

- Python 3.10+
- Java 21 (for PySpark)
- Docker (for PostgreSQL)

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start PostgreSQL with pgvector
docker compose up -d postgres
```

Set environment variables (or create `.env`):

```bash
DATASET_SIZE=small          # "small" or "32m"
JAVA_HOME=/path/to/jdk21   # required for PySpark
```

## Usage

```bash
# 1. Run ETL (clean raw data → Parquet)
DATASET_SIZE=small python3 pipeline/etl.py

# 2. Train ALS model and export embeddings
DATASET_SIZE=small python3 pipeline/train_als.py

# 3. Load data into PostgreSQL
python3 database/db_loader.py

# 4. Launch the Streamlit frontend
streamlit run app/app.py
```

## Dataset sizes

| `DATASET_SIZE` | Source | Location |
|----------------|--------|----------|
| `small` | MovieLens Latest-Small (~100k ratings) | `data/raw/ml-latest-small/` |
| `32m` | MovieLens 32M (~32M ratings) | `data/raw/ml-32m/` |

Raw data directories are gitignored. Place the MovieLens CSVs (or zip) in the appropriate directory before running ETL.

## Testing

```bash
# Unit + integration tests (skip long-running e2e tests)
python3 -m pytest tests/ -v -k 'not e2e'

# All tests (requires raw MovieLens small dataset)
python3 -m pytest tests/ -v
```

`test_performance.py` requires a running PostgreSQL instance and skips automatically if unreachable.

## Project structure

```
CineScale/
├── app/
│   ├── app.py              # Streamlit frontend
│   ├── db_queries.py       # PostgreSQL/pgvector query module
│   └── styles.py           # CSS theme
├── config/
│   └── schema.py           # AppConfig singleton
├── database/
│   ├── db_loader.py        # Bulk loader (drops/recreates HNSW index)
│   └── init.sql            # pgvector schema + HNSW index
├── pipeline/
│   ├── etl.py              # Extract, transform, load to Parquet
│   ├── train_als.py        # ALS training + evaluation metrics
│   ├── export_embeddings.py # Export ALS factors to Parquet
│   └── spark_session.py    # SparkSession builder
├── scripts/
│   └── benchmark_stress.py # Recommendation latency benchmark
├── tests/                   # pytest test suite
├── data/                    # Raw + processed data (gitignored)
├── docker-compose.yml       # PostgreSQL with pgvector
└── requirements.txt
```

## Key technical details

- **Embedding dimension**: 50 (ALS rank)
- **Similarity**: cosine distance via pgvector `<=>` operator (0 = identical, 2 = opposite)
- **Index**: HNSW (`m=16`, `ef_construction=64`) on `movie_factors.features`
- **Cold-start guard**: users with < 5 ratings get an "insufficient history" message
- **IDs**: raw MovieLens integer IDs, no remapping
- **Parquet I/O**: uses Pandas/PyArrow (not Hadoop) to avoid winutils dependency on Windows
