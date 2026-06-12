"""
Aggregate per-run JSON results into a summary table.

Computes mean ± std across the 5 seeds for each (architecture, imbalance,
dataset) configuration and produces a CSV that can be pasted into Chapter 5.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --results-dir results --output summary.csv
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


METRICS_OF_INTEREST = [
    "auc_pr",
    "auc_roc",
    "f1_at_threshold",
    "recall_at_50",
    "recall_at_100",
    "recall_at_200",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/_summary_aggregated.csv")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    json_files = [f for f in results_dir.glob("*.json") if not f.name.startswith("_")]
    if not json_files:
        print(f"No result JSON files found in {results_dir}")
        return

    # Group by (dataset, architecture, imbalance_strategy)
    grouped = defaultdict(list)
    for jf in json_files:
        with open(jf) as f:
            r = json.load(f)
        if "error" in r:
            continue
        key = (
            r["config"]["dataset"],
            r["config"]["architecture"],
            r["config"]["imbalance_strategy"],
        )
        grouped[key].append(r["test_metrics"])

    rows = []
    for (dataset, arch, imb), runs in sorted(grouped.items()):
        row = {
            "dataset": dataset,
            "architecture": arch,
            "imbalance_strategy": imb,
            "n_seeds": len(runs),
        }
        for metric in METRICS_OF_INTEREST:
            values = [r.get(metric) for r in runs if r.get(metric) is not None]
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values))
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_std"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    # Order columns sensibly
    base_cols = ["dataset", "architecture", "imbalance_strategy", "n_seeds"]
    metric_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + sorted(metric_cols)]
    df = df.sort_values(["dataset", "architecture", "imbalance_strategy"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Aggregated summary written to: {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
