"""Class-imbalance helpers for gaze-crop classifiers.

Problem we fix
--------------
Inverse-frequency loss + focal loss treats every *class* equally in total
gradient mass. Easy majority crops (trail_ground) get near-zero focal
gradient, so the model stops predicting trail_ground entirely.

Approach
--------
1. ResNet: WeightedRandomSampler (balanced minibatches) + *unweighted* CE,
   or mild sqrt-frequency weights with a cap on rare-class boost.
2. CLIP head: sqrt-frequency CE (no aggressive focal on full-batch training).
3. Checkpointing: pick best epoch by *balanced* validation accuracy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import WeightedRandomSampler


def class_counts(y, n_classes: int) -> np.ndarray:
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=n_classes).astype(
        np.float32
    )


def compute_class_weights(
    y,
    n_classes: int,
    *,
    method: str = "sqrt_inv",
    max_boost: float = 6.0,
) -> np.ndarray:
    """Per-class loss weights (mean ≈ 1).

    Parameters
    ----------
    method
        ``sqrt_inv``: sqrt(N / n_i), capped — boosts rare classes without
        crushing trail_ground.
        ``none``: uniform weights.
    max_boost
        Max ratio of highest / lowest class weight (after sqrt).
    """
    counts = class_counts(y, n_classes)
    counts = np.maximum(counts, 1.0)

    if method == "none":
        return np.ones(n_classes, dtype=np.float32)

    if method != "sqrt_inv":
        raise ValueError(f"Unknown method: {method}")

    total = counts.sum()
    weights = np.sqrt(total / counts)
    weights = weights / weights.min()
    if max_boost > 1.0:
        weights = np.clip(weights, 1.0, max_boost)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def make_weighted_sampler(y, n_classes: int) -> WeightedRandomSampler:
    """Oversample rare classes in minibatches (standard inverse-count)."""
    counts = class_counts(y, n_classes)
    counts = np.maximum(counts, 1.0)
    y_arr = np.asarray(y, dtype=np.int64)
    sample_weights = (1.0 / counts)[y_arr]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(y_arr),
        replacement=True,
    )


def make_classification_loss(
    y,
    n_classes: int,
    *,
    use_sampler: bool = False,
    loss_type: str = "ce",
    focal_gamma: float = 1.0,
    max_boost: float = 6.0,
) -> nn.Module:
    """Build a loss suited for imbalanced gaze crops.

    When ``use_sampler`` is True (ResNet path), default is unweighted CE
    because batches are already class-balanced.
    """
    if use_sampler:
        return nn.CrossEntropyLoss()

    weights = compute_class_weights(y, n_classes, max_boost=max_boost)
    weight_t = torch.from_numpy(weights)

    if loss_type == "ce":
        return nn.CrossEntropyLoss(weight=weight_t)
    if loss_type == "focal":
        return FocalLoss(alpha=weight_t, gamma=focal_gamma)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def balanced_val_accuracy(y_true, y_pred) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


class FocalLoss(nn.Module):
    """Focal loss with optional per-class alpha (use mild gamma ≤ 1.5)."""

    def __init__(self, alpha=None, gamma: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none"
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal
