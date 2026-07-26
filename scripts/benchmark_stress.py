#!/usr/bin/env python3
"""Standalone stress harness for CineScale recommendation latency.

Run against a loaded Postgres DB to collect p50/p95/p99 latency numbers.
"""

import argparse
import json
import math
import os
import random
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app import db_queries


def percentile(values, pct):
    """Nearest-rank percentile for a list of raw latency measurements."""
    if not values:
        raise ValueError("percentile requires a non-empty list")
    if not 0.0 <= pct <= 1.0:
        raise ValueError("pct must be between 0 and 1")
    sorted_vals = sorted(values)
    rank = int(math.ceil(pct * len(sorted_vals))) - 1
    rank = max(0, min(rank, len(sorted_vals) - 1))
    return sorted_vals[rank]


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

    all_times = []
    per_user_summaries = []

    print(f"Benchmarking {len(sampled)} users with {args.runs} runs each...")
    for i, user_id in enumerate(sampled, 1):
        result = db_queries.benchmark_recommendation(
            user_id, runs=args.runs, return_raw_times=True
        )
        all_times.extend(result["raw_times_ms"])
        per_user_summaries.append(
            {
                "user_id": user_id,
                "p50_ms": result["p50_ms"],
                "p95_ms": result["p95_ms"],
                "p99_ms": result["p99_ms"],
                "avg_ms": result["avg_ms"],
            }
        )
        print(
            f"  {i}/{len(sampled)} user {user_id}: "
            f"p50={result['p50_ms']:.2f}ms p95={result['p95_ms']:.2f}ms "
            f"p99={result['p99_ms']:.2f}ms avg={result['avg_ms']:.2f}ms"
        )

    report = {
        "dataset_size": os.getenv("DATASET_SIZE", "unknown"),
        "users_sampled": len(sampled),
        "runs_per_user": args.runs,
        "total_runs": len(all_times),
        "p50_ms": percentile(all_times, 0.50),
        "p95_ms": percentile(all_times, 0.95),
        "p99_ms": percentile(all_times, 0.99),
        "avg_ms": sum(all_times) / len(all_times),
    }

    print("\nAggregate latency report (over all raw runs):")
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
