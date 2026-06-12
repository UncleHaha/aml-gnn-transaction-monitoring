"""
Shared utilities: seeding, device, logging.
"""
import os
import random
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # The following two lines slow training but make results deterministic.
    # Comment out if you need maximum speed and only want approximate reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """Set up a logger that writes to both file and console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers in case logger already exists
    logger.handlers = []

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    )

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def save_results(results: dict, save_dir: str, run_name: str) -> None:
    """Save a results dict as JSON."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(save_dir) / f"{run_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
