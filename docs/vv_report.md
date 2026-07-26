# CineScale — Verification & Validation Report

## 1. Objective

Validate the accuracy and serving performance of the CineScale ALS-based recommendation pipeline.

## 2. Datasets

### 2.1 MovieLens Small (accuracy metrics)

| Property | Value |
|---|---|
| Raw ratings | 100,836 |
| Raw movies | 9,742 |
| Train / test split | 80 / 20 |
| Items after filtering | 2,269 |
| Distinct test users | 609 |

### 2.2 MovieLens 32M (serving latency)

The production database was loaded with the full MovieLens 32M dataset:

| Property | Value |
|---|---|
| Total users | 200,948 |
| Total movies | 31,961 |
| Total ratings | 31,842,705 |

## 3. Model

| Property | Value |
|---|---|
| Algorithm | Spark MLlib ALS |
| Implicit preferences | True |
| Rank | 50 |
| Max iterations | 25 |
| Regularization parameter | 0.05 |
| Alpha | 40 |
| Cold-start strategy | drop |
| Seed | 42 |

## 4. Accuracy Metrics

Accuracy metrics were measured on the MovieLens Small dataset.

| Metric | Value |
|---|---|
| RMSE | 3.0966 |
| Precision@10 | 0.1080 |
| NDCG@10 | 0.1488 |
| Hit Rate@10 | 0.5966 |
| Recall@10 | 0.1546 |

## 5. Serving Latency

HNSW index: `vector_cosine_ops` with `m=16`, `ef_construction=64`.

| Percentile | Latency (ms) |
|---|---|
| p50 | 0.34 |
| p95 | 0.66 |
| p99 | 1.50 |
| avg | 0.44 |

Test command: `DATASET_SIZE=small python3 scripts/benchmark_stress.py --users 20 --runs 100`

## 6. Reproduction

### 6.1 Train and evaluate on MovieLens Small

```bash
# Ensure raw data is available at data/raw/ml-latest-small/
DATASET_SIZE=small JAVA_HOME=/usr/lib/jvm/java-21-openjdk python3 pipeline/train_als.py
```

The script prints RMSE, Precision@10, Hit Rate@10, Recall@10, and NDCG@10.

### 6.2 Start the database

```bash
DATASET_SIZE=small docker compose up -d postgres
```

### 6.3 Load the database

```bash
DATASET_SIZE=small python3 database/db_loader.py
```

### 6.4 Run the stress harness

```bash
DATASET_SIZE=small python3 scripts/benchmark_stress.py --users 20 --runs 100 --output docs/stress_results.json
```

## 7. 32M Dataset Notes

Run the same commands with `DATASET_SIZE=32m` (and `data/raw/ml-32m/` available) to collect full-scale accuracy metrics. Update the accuracy table above with the resulting values.
