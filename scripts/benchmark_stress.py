#!/usr/bin/env python3
"""Standalone stress harness for CineScale recommendation latency.

Run against a loaded Postgres DB to collect p50/p95/p99 latency numbers.
"""

import argparse
import json
import os
import random
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import db_queries


def percentile(values, pct):
    values = sorted(values)
    idx = int(len(values) * pct)
    idx = min(idx, len(values) - 1)
    return values[idx]


def main():
    parser = argparse.ArgumentParser(description="Benchmark recommendation latency")
    parser.add_argument("--users", type=int, default=50, help="Number of random users to sample")
    parser.add_argument("--runs", type=int, default=100, help="Runs per user")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON file to write results")
    parser.add_argument("--user-limit", type=int, default=10000, help="Max users to fetch for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    users = db_queries.get_all_users(limit=args.user_limit)
    if not users:
        print("ERROR: No users found in user_factors table.")
        return

    user_ids = [u["user_id"] for u in users]
    random.seed(args.seed)
    sampled = random.sample(user_ids, min(args.users, len(user_ids)))

    per_user_p50 = []
    per_user_p95 = []
    per_user_p99 = []
    per_user_avg = []

    print(f"Benchmarking {len(sampled)} users with {args.runs} runs each...")
    for i, user_id in enumerate(sampled, 1):
        result = db_queries.benchmark_recommendation(user_id, runs=args.runs)
        per_user_p50.append(result["p50_ms"])
        per_user_p95.append(result["p95_ms"])
        per_user_p99.append(result["p99_ms"])
        per_user_avg.append(result["avg_ms"])
        print(
            f"  {i}/{len(sampled)} user {user_id}: "
            f"p50={result['p50_ms']:.2f}ms p95={result['p95_ms']:.2f}ms "
            f"p99={result['p99_ms']:.2f}ms avg={result['avg_ms']:.2f}ms"
        )

    report = {
        "dataset_size": os.getenv("DATASET_SIZE", "unknown"),
        "users_sampled": len(sampled),
        "runs_per_user": args.runs,
        "total_runs": len(sampled) * args.runs,
        "p50_ms": percentile(per_user_p50, 0.50),
        "p95_ms": percentile(per_user_p95, 0.95),
        "p99_ms": percentile(per_user_p99, 0.99),
        "avg_ms": sum(per_user_avg) / len(per_user_avg),
    }

    print("\nAggregate latency report:")
    print(f"  p50: {report['p50_ms']:.2f} ms")
    print(f"  p95: {report['p95_ms']:.2f} ms")
    print(f"  p99: {report['p99_ms']:.2f} ms")
    print(f"  avg: {report['avg_ms']:.2f} ms")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote results to {args.output}")


if __name__ == "__main__":
    main()
