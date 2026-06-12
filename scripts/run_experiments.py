"""
Main experiment runner.

Usage:
    # Run all 90 configurations
    python scripts/run_experiments.py

    # Run a subset (e.g. just Elliptic + GCN to validate the pipeline)
    python scripts/run_experiments.py --dataset elliptic --architecture gcn

    # Limit number of runs (for smoke testing)
    python scripts/run_experiments.py --max-runs 1

    # Disable wandb (default is disabled unless --wandb is passed)
    python scripts/run_experiments.py
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

# Add project root to path so we can import `src` and `configs`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.utils import set_seed, get_device, setup_logger, save_results, count_parameters
from src.data.elliptic_loader import load_elliptic
from src.data.itaml_loader import load_itaml
from src.models.gnn_models import build_model
from src.losses.losses import build_loss
from src.sampling.graphsmote import apply_graphsmote
from src.training.trainer import Trainer, TrainerConfig

from configs.experiments import build_all_configs, filter_configs


def load_dataset(dataset_name: str, logger):
    """Dispatch to the appropriate loader."""
    if dataset_name == "elliptic":
        logger.info("Loading Elliptic dataset...")
        data, info = load_elliptic(root="data/elliptic")
    elif dataset_name == "itaml":
        logger.info("Loading IT-AML dataset...")
        data, info = load_itaml(
            csv_path="data/it_aml/HI-Small_Trans.csv",
            patterns_path="data/it_aml/HI-Small_Patterns.txt",
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return data, info


def run_single_experiment(config: Dict, logger, wandb_enabled: bool = False) -> Dict:
    """Run a single training configuration and return results."""
    set_seed(config["seed"])
    device = get_device()
    logger.info(
        f"=== {config['run_name']} | device={device} ==="
    )

    # 1. Load dataset (cached load is fine but we keep it simple here)
    data, info = load_dataset(config["dataset"], logger)
    logger.info(f"Dataset info: {info}")

    # 2. Apply class-imbalance strategy at the data level (GraphSMOTE only)
    if config["imbalance_strategy"] == "graphsmote":
        logger.info("Applying GraphSMOTE oversampling...")
        data = apply_graphsmote(
            data,
            minority_label=1,
            oversample_ratio=config["graphsmote_oversample_ratio"],
            k_neighbors=config["graphsmote_k_neighbors"],
            seed=config["seed"],
        )
        logger.info(
            f"Post-augmentation: {data.num_nodes} nodes, "
            f"{int(data.train_mask.sum())} training nodes, "
            f"{int(data.y[data.train_mask].sum())} minority training nodes"
        )

    # 3. Build the loss function
    train_y = data.y[data.train_mask]
    if config["imbalance_strategy"] == "graphsmote":
        # After oversampling, the data are balanced -> use plain CE
        loss_fn = build_loss("ce", train_y)
    elif config["imbalance_strategy"] == "weighted_ce":
        loss_fn = build_loss("weighted_ce", train_y)
    elif config["imbalance_strategy"] == "focal":
        loss_fn = build_loss(
            "focal", train_y, focal_gamma=config["focal_gamma"]
        )
    else:
        raise ValueError(f"Unknown imbalance strategy: {config['imbalance_strategy']}")

    # 4. Build the model
    model = build_model(
        architecture=config["architecture"],
        in_dim=data.num_features,
        hidden_dim=config["hidden_dim"],
        out_dim=2,
        dropout=config["dropout"],
        heads=config["heads"],
    )
    logger.info(f"Model: {config['architecture']} | params={count_parameters(model):,}")

    # 5. Train
    trainer_cfg = TrainerConfig(
        max_epochs=config["max_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        patience=config["patience"],
    )
    trainer = Trainer(model, data, loss_fn, device, trainer_cfg, logger=logger)
    result = trainer.fit()

    # 6. Package final result
    final = {
        "run_name": config["run_name"],
        "config": config,
        "dataset_info": info,
        "model_params": count_parameters(model),
        "best_epoch": result.best_epoch,
        "elapsed_seconds": result.elapsed_seconds,
        "best_val_metrics": result.best_val_metrics,
        "test_metrics": result.test_metrics,
        "threshold_from_val": result.threshold_from_val,
    }
    return final


def main():
    parser = argparse.ArgumentParser(description="Run AML-GNN experiments")
    parser.add_argument("--dataset", choices=["elliptic", "itaml"], default=None,
                        help="Restrict to one dataset")
    parser.add_argument("--architecture", choices=["gcn", "gat", "graphsage"],
                        default=None, help="Restrict to one architecture")
    parser.add_argument("--imbalance_strategy",
                        choices=["weighted_ce", "focal", "graphsmote"], default=None,
                        help="Restrict to one class-imbalance strategy")
    parser.add_argument("--seed", type=int, default=None,
                        help="Restrict to one random seed")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Cap the number of runs (for smoke testing)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to save per-run JSON results")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    args = parser.parse_args()

    logger = setup_logger("run_experiments")
    logger.info(f"Project root: {PROJECT_ROOT}")

    # Build & filter configurations
    all_configs = build_all_configs()
    filters = {}
    if args.dataset is not None:
        filters["dataset"] = args.dataset
    if args.architecture is not None:
        filters["architecture"] = args.architecture
    if args.imbalance_strategy is not None:
        filters["imbalance_strategy"] = args.imbalance_strategy
    if args.seed is not None:
        filters["seed"] = args.seed
    configs = filter_configs(all_configs, **filters) if filters else all_configs

    if args.max_runs is not None:
        configs = configs[: args.max_runs]
    logger.info(f"Will execute {len(configs)} configurations.")
    logger.info(f"First run: {configs[0]['run_name']}")
    logger.info(f"Last run:  {configs[-1]['run_name']}")

    # Sweep
    overall_start = time.time()
    all_results = []
    for i, cfg in enumerate(configs, start=1):
        logger.info(f"\n=== Run {i}/{len(configs)} ===")
        try:
            res = run_single_experiment(cfg, logger, wandb_enabled=args.wandb)
            all_results.append(res)
            save_results(res, args.results_dir, cfg["run_name"])
        except Exception as e:
            logger.exception(f"Run {cfg['run_name']} failed: {e}")
            all_results.append({"run_name": cfg["run_name"], "error": str(e)})

    # Aggregate summary
    summary_path = Path(args.results_dir) / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results saved to {args.results_dir}/")
    logger.info(f"Total wall time: {(time.time() - overall_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
