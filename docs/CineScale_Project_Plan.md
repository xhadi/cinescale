# CineScale — Project Plan
**14-Day Sprint | Movie & Series Recommendation Engine**

---

## 1. Project Overview

CineScale is a high-performance movie and series recommendation engine built to handle large-scale user interaction data. It combines Big Data processing, Machine Learning, and Vector Storage to serve real-time, personalized recommendations without the bottlenecks of traditional single-node tools (like Pandas) or runtime ML inference.

**The core architecture principle:** separate heavy analytical training from fast real-time serving.

- **Big Data & ML (offline):** A PySpark pipeline ingests the MovieLens 32M dataset and trains a Collaborative Filtering model (ALS matrix factorization) to compress user histories into dense latent factor vectors (embeddings).
- **Vector Database:** Embeddings are stored in PostgreSQL with the `pgvector` extension, indexed with HNSW for sub-millisecond similarity search.
- **Interactive App (online):** A Streamlit app serves recommendations by running native vector cosine similarity queries directly in the database — no ML computation at request time.

---

## 2. Interface Contract — Agree on This Before Writing Any Code

This is the most important section in the plan. Locking these decisions on Day 1 prevents rework later and lets all four workstreams build in parallel from day one.

| Decision | Value |
|---|---|
| Embedding dimensionality | `rank = 50` (ALS latent factors) → Postgres columns are `vector(50)` |
| User ID / Movie ID types | `INTEGER`, matching raw MovieLens IDs (no re-mapping) |
| Exchange format | Parquet (not CSV) — preserves float precision, faster to load |
| Similarity metric | Cosine similarity (`vector_cosine_ops` index, `<=>` query operator) — consistent across index build and queries |
| File naming | `user_factors.parquet`, `movie_factors.parquet`, standard paths in `data/processed/` |

**Action:** All four members confirm this in a 15-minute sync before starting Day 1 work.

---

## 3. Day 1 Unblock — Synthetic Data Contract

**Owner: Member 1**

Before real ALS training finishes, Member 1 generates a tiny synthetic dataset (random 50-dim vectors, ~50 fake users/movies) matching the Interface Contract exactly, and commits it to the repo by end of Day 1.

This means Members 2, 3, and 4 are never blocked waiting on real training — they build and test their full slice against realistic-shaped data immediately, and swap in real embeddings on Day 7 with zero interface changes.

---

## 4. Repository Structure

```
cinescale/
├── .gitignore
├── README.md                  # Project overview, setup guide, run instructions
├── requirements.txt           # pyspark, psycopg2-binary, streamlit, etc.
├── docker-compose.yml         # One-command Postgres + pgvector setup
│
├── config/
│   └── schema.py               # The Interface Contract as code: vector dim, ID types, paths
│
├── data/                       # Git-ignored except synthetic/
│   ├── synthetic/               # Committed tiny fake dataset — unblocks Day 1 parallel work
│   ├── raw/                     # Real MovieLens CSVs (start Small, scale to 32M)
│   └── processed/               # Exported ALS factors/embeddings
│
├── database/
│   ├── init.sql                 # Enables pgvector, creates schemas + indexes
│   └── db_loader.py             # Batch-inserts metadata + embeddings via psycopg2
│
├── pipeline/
│   ├── spark_session.py         # Shared PySpark session config
│   ├── etl.py                   # Ingest/clean/cast ratings.csv, movies.csv
│   └── train_als.py             # Spark MLlib ALS training, outputs latent factors
│
├── app/
│   ├── app.py                   # Streamlit frontend
│   └── db_queries.py            # SQL helper functions (cosine similarity search)
│
├── tests/
│   ├── test_model.py            # RMSE + Precision@K / NDCG@K evaluation
│   └── test_performance.py      # Query latency under simulated load
│
└── docs/
    ├── architecture.md          # Final architecture diagram
    ├── vv_report.md             # V&V report (accuracy + performance results)
    └── reflection.md            # Team reflection summary
```

---

## 5. Team Assignments

### 🧠 Module 1 — ML & Big Data Pipeline
**Owner: Member 1 (ML & Pipeline Engineer)**

| # | Task | Definition of Done |
|---|---|---|
| 1.0 | Draft & confirm Interface Contract | All members sign off before Day 1 coding starts |
| 1.0b | Generate synthetic dataset | Committed to `data/synthetic/`, matches schema exactly, by end of Day 1 |
| 1.1 | Configure local + cluster PySpark environments | Both run a smoke-test job successfully |
| 1.2 | Build ETL pipeline (`pipeline/etl.py`) for ratings.csv / movies.csv | Nulls handled, types cast, row counts logged and match source files |
| 1.3 | Train Spark MLlib ALS model | RMSE **and** Precision@10 computed on a held-out split |
| 1.4 | Export embeddings to Parquet | Matches Interface Contract exactly; validated by reloading and checking dtypes |

### 🗄️ Module 2 — Database Infrastructure & Loading
**Owner: Member 2 (Database & Integration Engineer)**

| # | Task | Definition of Done |
|---|---|---|
| 2.1 | Deploy PostgreSQL via Docker (pgvector image) | `docker-compose up` works with one command, documented in README |
| 2.2 | Draft `init.sql` — extensions, schemas, keys | Includes a movies **metadata** table (title, genres, year) joinable to vector tables, not just raw vectors |
| 2.3 | Write `db_loader.py` (psycopg2 batch insert) | Idempotent — safe to re-run without duplicate rows; works unchanged on synthetic and real data |
| 2.4 | Build & test HNSW index (`vector_cosine_ops`) | Build time and `m` / `ef_construction` params logged for Member 4's latency report |

### 🎨 Module 3 — Frontend UI
**Owner: Member 3 (Frontend UI Developer)**

| # | Task | Definition of Done |
|---|---|---|
| 3.1 | Design Streamlit dashboard (`app.py`) | Runs end-to-end against synthetic data by Day 3 |
| 3.2 | Build user-profile selector | Dropdown works against both synthetic and real data without code change |
| 3.3 | Visualize historical ratings + recommendations | Clear side-by-side layout, movie metadata (title/genre) rendered, not just IDs |
| 3.4 | Connect frontend to backend query functions | Uses Member 4's `db_queries.py` directly — no duplicate query logic in frontend |
| 3.5 | Handle cold-start users | If a selected user has <5 ratings, show an "insufficient history" message rather than erroring |

### 🛠️ Module 4 — Backend Queries & V&V
**Owner: Member 4 (V&V & Backend Query Specialist)**

| # | Task | Definition of Done |
|---|---|---|
| 4.1 | Build `db_queries.py` — top-10 via `<=>` cosine operator | Unit-tested against synthetic data first, then real |
| 4.2 | Lead V&V — accuracy metrics | **Both** RMSE (rating prediction) and Precision@10 / NDCG@10 (ranking quality) reported |
| 4.3 | Performance & stress testing | Tested at Small and Full (32M) scale using Member 2's HNSW params; report includes p50/p95/p99 latency |

---

## 6. Sprint Calendar

| Day | Milestone |
|---|---|
| **Day 1** | Interface Contract locked. Synthetic embeddings committed. All four members unblocked and building in parallel. |
| **Day 3** | Real PySpark ETL working on MovieLens Small. Postgres + schema live via Docker. Streamlit UI running end-to-end against synthetic data. |
| **Day 5** | **Integration sync (live, 30 min).** First real ALS embeddings loaded and queried end-to-end. Catch schema/type mismatches here, not on Day 7. |
| **Day 7** | Full end-to-end pipeline confirmed on MovieLens Small with **real** data throughout. |
| **Day 9** | Buffer day: kick off ALS retraining on the full 32M dataset (long-running job started early, not on Day 10). |
| **Day 11** | Full 32M dataset loaded into production DB; HNSW index rebuilt at scale. |
| **Day 12** | V&V complete: RMSE + Precision@K recorded, latency benchmarks done, repo locked. |
| **Day 14** | Architecture diagram, V&V report, and reflection doc submitted. |

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ALS training on 32M dataset takes longer than expected | High | High | Start Day 9, not Day 10; keep a Small-dataset fallback ready for the demo if it slips |
| HNSW index build is slow/memory-heavy at full scale | Medium | High | Test build time on Day 5 with a mid-size subset to extrapolate timing |
| Schema drift between Member 1's export and Member 2's loader | Medium | Medium | Interface Contract + Day 1 synthetic data catches this immediately, not on Day 7 |
| RMSE looks fine but recommendations feel wrong in the UI | Medium | Medium | Precision@K / NDCG@K added specifically to catch ranking-quality issues RMSE misses |
| Docker/pgvector environment differs across members' machines | Low | Medium | `docker-compose.yml` committed Day 1; everyone validates locally before Day 3 |

---

## 8. Communication Cadence

- **Daily:** Async standup — what's done, blocked, next (Slack thread is fine, no call needed)
- **Day 5:** Mandatory live integration sync (30 min)
- **Day 9:** Live check-in before the 32M scale-up begins — confirm everyone's ready for the production cutover
- **Day 12:** Live wrap-up to divide final documentation:
  - Member 4 leads the V&V report (owns the metrics)
  - Member 3 leads the architecture diagram (owns the full user-facing flow)
  - Members 1 & 2 co-write the pipeline/infra sections

---

## 9. Explicitly Out of Scope (v1)

Documented now to avoid scope creep and to give clean talking points for the reflection doc:

- Cold-start recommendations for brand-new users (handled via a UI message only, see Task 3.5)
- Real-time model retraining (embeddings are batch-refreshed, not streamed)
- A/B testing or online evaluation — V&V is offline metrics only

---

## 10. Final Deliverables Checklist (Day 14)

- [ ] Working end-to-end app (PySpark → ALS → Postgres/pgvector → Streamlit)
- [ ] Codebase following the repo structure above, committed and locked
- [ ] `docs/architecture.md` — system architecture diagram
- [ ] `docs/vv_report.md` — RMSE, Precision@K/NDCG@K, and latency benchmarks
- [ ] `docs/reflection.md` — team reflection summary
- [ ] README with setup and run instructions verified by someone outside the original author
