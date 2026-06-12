"""
Training loop with early stopping, best-checkpoint tracking, and metric logging.
"""
import copy
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import Data

from src.training.metrics import compute_all_metrics, get_probabilities


@dataclass
class TrainerConfig:
    max_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    patience: int = 30  # early stopping patience (epochs without improvement)
    scheduler_patience: int = 10  # LR scheduler patience
    scheduler_factor: float = 0.1  # LR reduction factor
    min_lr: float = 1e-6
    verbose: bool = True
    log_every: int = 10


@dataclass
class TrainerResult:
    best_val_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    best_epoch: int = -1
    train_history: list = field(default_factory=list)  # list of dicts
    elapsed_seconds: float = 0.0
    threshold_from_val: float | None = None


class Trainer:
    """
    Standard full-batch GNN trainer for node classification.

    For each epoch:
        1. Forward pass on the full graph.
        2. Compute loss on training nodes only.
        3. Backward pass and update.
        4. Evaluate on validation set; track best (by AUC-PR).

    At the end, restore the best checkpoint and evaluate on test.
    """

    def __init__(
        self,
        model: nn.Module,
        data: Data,
        loss_fn: nn.Module,
        device: torch.device,
        config: Optional[TrainerConfig] = None,
        logger=None,
    ):
        self.model = model.to(device)
        self.data = data.to(device)
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.config = config or TrainerConfig()
        self.logger = logger

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
            min_lr=self.config.min_lr,
        )

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)
        elif self.config.verbose:
            print(msg)

    def _train_one_epoch(self) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(self.data.x, self.data.edge_index)
        loss = self.loss_fn(
            logits[self.data.train_mask],
            self.data.y[self.data.train_mask],
        )
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    @torch.no_grad()
    def _evaluate(self, mask: torch.Tensor, threshold: float | None = None) -> dict:
        self.model.eval()
        logits = self.model(self.data.x, self.data.edge_index)
        y_prob = get_probabilities(logits[mask])
        y_true = self.data.y[mask].cpu().numpy()
        return compute_all_metrics(y_true, y_prob, threshold=threshold)

    def fit(self) -> TrainerResult:
        """Train until early stopping; return best results."""
        result = TrainerResult()
        best_val_auc_pr = -1.0
        epochs_since_improvement = 0
        best_state_dict = None
        start_time = time.time()

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self._train_one_epoch()
            val_metrics = self._evaluate(self.data.val_mask)
            current_auc_pr = val_metrics["auc_pr"]

            self.scheduler.step(current_auc_pr)

            improved = current_auc_pr > best_val_auc_pr
            if improved:
                best_val_auc_pr = current_auc_pr
                epochs_since_improvement = 0
                best_state_dict = copy.deepcopy(self.model.state_dict())
                result.best_val_metrics = val_metrics
                result.best_epoch = epoch
                result.threshold_from_val = val_metrics.get("threshold_best")
            else:
                epochs_since_improvement += 1

            result.train_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()
                   if isinstance(v, (int, float))},
            })

            if epoch % self.config.log_every == 0 or epoch == 1:
                self._log(
                    f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
                    f"val_AUC-PR={current_auc_pr:.4f} | val_F1={val_metrics.get('f1_best', 0):.4f} | "
                    f"best_epoch={result.best_epoch}"
                )

            if epochs_since_improvement >= self.config.patience:
                self._log(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.config.patience} epochs)."
                )
                break

        # Restore best checkpoint and evaluate on test
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        # On test, use the validation-selected threshold (avoid leakage)
        test_metrics = self._evaluate(
            self.data.test_mask, threshold=result.threshold_from_val
        )
        result.test_metrics = test_metrics
        result.elapsed_seconds = time.time() - start_time

        self._log(
            f"DONE | best_epoch={result.best_epoch} | "
            f"test_AUC-PR={test_metrics['auc_pr']:.4f} | "
            f"test_F1={test_metrics.get('f1_at_threshold', 0):.4f} | "
            f"test_AUC-ROC={test_metrics['auc_roc']:.4f} | "
            f"time={result.elapsed_seconds:.1f}s"
        )
        return result
