"""
Experiment configurations for the primary 90 detection runs.

Cartesian product:
    architectures × imbalance_strategies × datasets × seeds = 3 × 3 × 2 × 5 = 90.

Edit `ALL_DATASETS` to ["elliptic"] if you want to start with just Elliptic
while IT-AML is being prepared.
"""
from itertools import product
from typing import List, Dict

ALL_ARCHITECTURES = ["gcn", "gat", "graphsage"]
ALL_IMBALANCE = ["weighted_ce", "focal", "graphsmote"]
ALL_DATASETS = ["elliptic", "itaml"]
ALL_SEEDS = [0, 1, 2, 3, 4]


def build_all_configs() -> List[Dict]:
    """Return a list of 90 configuration dicts."""
    configs = []
    for arch, imb, dataset, seed in product(
        ALL_ARCHITECTURES, ALL_IMBALANCE, ALL_DATASETS, ALL_SEEDS
    ):
        run_name = f"{dataset}__{arch}__{imb}__seed{seed}"
        configs.append({
            "run_name": run_name,
            "architecture": arch,
            "imbalance_strategy": imb,
            "dataset": dataset,
            "seed": seed,
            # Training hyperparameters (tunable via Optuna later)
            "hidden_dim": 64,
            "dropout": 0.5,
            "heads": 4,
            "learning_rate": 1e-3,
            "weight_decay": 5e-4,
            "max_epochs": 200,
            "patience": 30,
            # Loss hyperparameters
            "focal_gamma": 2.0,
            # GraphSMOTE hyperparameters
            "graphsmote_oversample_ratio": 1.0,
            "graphsmote_k_neighbors": 5,
        })
    return configs


def filter_configs(configs: List[Dict], **filters) -> List[Dict]:
    """Subset configs by exact-match filters.

    Example:
        filter_configs(configs, dataset="elliptic", architecture="gcn")
    """
    out = []
    for cfg in configs:
        if all(cfg.get(k) == v for k, v in filters.items()):
            out.append(cfg)
    return out


if __name__ == "__main__":
    cfgs = build_all_configs()
    print(f"Total configurations: {len(cfgs)}")
    print(f"Example: {cfgs[0]}")
    print(f"\nElliptic-only subset: {len(filter_configs(cfgs, dataset='elliptic'))} runs")
    print(f"GCN-only subset:      {len(filter_configs(cfgs, architecture='gcn'))} runs")
