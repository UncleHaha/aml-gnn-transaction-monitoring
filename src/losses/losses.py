"""
Loss functions for imbalanced node classification.

Implements:
    - WeightedCrossEntropyLoss: standard cross-entropy with class-weighting
      where weights are inversely proportional to class frequency.
    - FocalLoss: Lin et al. (2017) focal loss, which down-weights well-classified
      examples to focus learning on hard minority examples.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(y: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Inverse-frequency class weights, normalised to sum to num_classes.

    Parameters
    ----------
    y : (N,) tensor of class labels
    num_classes : int

    Returns
    -------
    (num_classes,) tensor of weights
    """
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp(min=1.0) * num_classes)
    return weights


class WeightedCrossEntropyLoss(nn.Module):
    """Standard cross-entropy with explicit class weights."""

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.class_weights)


class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced classification (Lin et al., 2017).

    L = - alpha * (1 - p_t)^gamma * log(p_t)

    where p_t is the predicted probability of the true class.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # scalar weight for the minority class (optional)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_term = (1.0 - pt) ** self.gamma

        loss = -focal_term * log_pt

        if self.class_weights is not None:
            w = self.class_weights[target]
            loss = w * loss
        elif self.alpha is not None:
            # Apply alpha to minority class only (assumed class 1)
            alpha_t = torch.where(
                target == 1,
                torch.tensor(self.alpha, device=logits.device),
                torch.tensor(1.0 - self.alpha, device=logits.device),
            )
            loss = alpha_t * loss

        return loss.mean()


def build_loss(
    loss_name: str,
    y_train: torch.Tensor,
    num_classes: int = 2,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Factory function: build a loss module by name."""
    name = loss_name.lower()
    weights = compute_class_weights(y_train, num_classes)

    if name in ("weighted_ce", "wce"):
        return WeightedCrossEntropyLoss(weights)
    if name == "focal":
        return FocalLoss(gamma=focal_gamma, class_weights=weights)
    if name in ("ce", "cross_entropy"):
        return nn.CrossEntropyLoss()
    raise ValueError(
        f"Unknown loss: {loss_name}. Expected: weighted_ce, focal, ce."
    )
