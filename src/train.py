"""
PSY197B — Model Training
========================
Implements the full ML strategy from scalar baselines through multimodal DL.

Usage:
    python src/train.py                       # all phases, latest run
    python src/train.py --phase 1             # scalar baselines only
    python src/train.py --phase 2             # EEGNet only
    python src/train.py --phase 3             # multimodal EEG+ET
    python src/train.py --phase 4             # outcome prediction
    python src/train.py --phase 5             # vision integration
    python src/train.py --phase 8             # LOSO cross-validation
    python src/train.py --phase 9             # gaze comparison (walking)
    python src/train.py --phase 10            # cross-condition transfer
    python src/train.py --run 2026-04-22_1430_sj04_walk_pilot
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    classification_report, confusion_matrix,
    mean_squared_error, r2_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon as wilcoxon_test
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")


def load_model_config():
    """Load configs/config.yaml for model architecture and training params."""
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


def _eegnet_kwargs(cfg):
    """Extract EEGNet hyperparams from config dict."""
    ec = cfg.get("model", {}).get("eegnet", {})
    return {
        "F1": ec.get("F1", 8),
        "D": ec.get("D", 2),
        "F2": ec.get("F2", 16),
        "dropout": ec.get("dropout", 0.25),
        "kernel_length": ec.get("kernel_length", 64),
        "sep_kernel_length": ec.get("sep_kernel_length", 16),
        "use_transformer": ec.get("use_transformer", False),
        "n_heads": ec.get("transformer_heads", 2),
        "transformer_dropout": ec.get("transformer_dropout", 0.1),
    }


def _training_kwargs(cfg):
    """Extract training hyperparams from config dict."""
    tc = cfg.get("training", {})
    return {
        "n_epochs": tc.get("epochs", 100),
        "batch_size": tc.get("batch_size", 32),
        "lr": tc.get("learning_rate", 1e-3),
        "patience": tc.get("early_stopping_patience", 10),
    }


# ═══════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════

def find_latest_run():
    runs = sorted(
        [d for d in os.listdir(RUNS_ROOT)
         if os.path.isdir(os.path.join(RUNS_ROOT, d))],
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError("No runs found in runs/")
    return runs[0]


def load_tensors(run_name, condition):
    """Load pre-built tensors and metadata for a single condition.

    Returns None if any required file (EEG tensors, labels, metadata) is
    missing — callers must handle the None case.
    """
    tensor_dir = os.path.join(RUNS_ROOT, run_name, "data", "dl_tensors")
    prefix = condition

    if not os.path.isdir(tensor_dir):
        print(f"  [load_tensors] Missing tensor directory: {tensor_dir}")
        return None

    required_files = []
    for split in ("train", "val"):
        required_files.append(f"{prefix}_X_eeg_{split}.npy")
        required_files.append(f"{prefix}_y_{split}.npy")
        required_files.append(f"{prefix}_meta_{split}.csv")

    for fname in required_files:
        fpath = os.path.join(tensor_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [load_tensors] Missing required file: {fname}")
            return None

    data = {}
    for split in ("train", "val"):
        data[f"X_eeg_{split}"] = np.load(
            os.path.join(tensor_dir, f"{prefix}_X_eeg_{split}.npy"))
        data[f"y_{split}"] = np.load(
            os.path.join(tensor_dir, f"{prefix}_y_{split}.npy"))
        data[f"meta_{split}"] = pd.read_csv(
            os.path.join(tensor_dir, f"{prefix}_meta_{split}.csv"))

        et_path = os.path.join(tensor_dir, f"{prefix}_X_et_{split}.npy")
        if os.path.exists(et_path):
            data[f"X_et_{split}"] = np.load(et_path)

    return data


def discover_conditions(run_name):
    """Find all condition prefixes in a run's dl_tensors dir."""
    tensor_dir = os.path.join(RUNS_ROOT, run_name, "data", "dl_tensors")
    if not os.path.isdir(tensor_dir):
        print(f"  [discover_conditions] Tensor directory not found: {tensor_dir}")
        return []
    prefixes = set()
    for f in os.listdir(tensor_dir):
        if f.endswith("_X_eeg_train.npy"):
            prefixes.add(f.replace("_X_eeg_train.npy", ""))
    return sorted(prefixes)


def pool_conditions(run_name, conditions):
    """Stack data across conditions, adding a condition column to metadata.

    Skips conditions whose tensors are missing. Returns an empty dict if no
    conditions have valid data.
    """
    pooled = {}
    for i, cond in enumerate(conditions):
        d = load_tensors(run_name, cond)
        if d is None:
            print(f"  [pool_conditions] Skipping {cond} — missing tensors")
            continue
        for split in ("train", "val"):
            d[f"meta_{split}"]["_condition"] = cond
            d[f"meta_{split}"]["_cond_idx"] = i
            for key in ("X_eeg", "X_et", "y", "meta"):
                full_key = f"{key}_{split}"
                if full_key in d:
                    pooled.setdefault(full_key, []).append(d[full_key])

    if not pooled:
        return {}

    result = {}
    for key, parts in pooled.items():
        if isinstance(parts[0], pd.DataFrame):
            result[key] = pd.concat(parts, ignore_index=True)
        else:
            result[key] = np.concatenate(parts, axis=0)
    return result


# ═══════════════════════════════════════════════════════════
#  PHASE 1 — SCALAR BASELINES
# ═══════════════════════════════════════════════════════════

SCALAR_FEATURES = [
    "P300_cluster_uV", "N200_cluster_uV",
    "alpha_frontal_uV2", "alpha_parietal_uV2", "alpha_occipital_uV2",
    "gaze_mean_x_px", "gaze_mean_y_px",
]


def _extract_scalar_Xy(meta_train, meta_val, feature_cols, y_train, y_val):
    """Pull scalar feature columns, handle NaN, scale."""
    available = [c for c in feature_cols if c in meta_train.columns]
    if not available:
        return None, None, None, None, []

    X_tr = meta_train[available].values.astype(np.float64)
    X_va = meta_val[available].values.astype(np.float64)

    nan_mask_tr = np.isnan(X_tr).any(axis=1)
    nan_mask_va = np.isnan(X_va).any(axis=1)
    X_tr, y_tr = X_tr[~nan_mask_tr], y_train[~nan_mask_tr]
    X_va, y_va = X_va[~nan_mask_va], y_val[~nan_mask_va]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    return X_tr, X_va, y_tr, y_va, available


def run_scalar_baseline(data, task_name, label_fn=None):
    """Train LogReg, SVM, LDA on scalar features."""
    meta_tr = data["meta_train"].copy()
    meta_va = data["meta_val"].copy()

    if label_fn is not None:
        y_tr, mask_tr = label_fn(meta_tr)
        y_va, mask_va = label_fn(meta_va)
        meta_tr = meta_tr[mask_tr].reset_index(drop=True)
        meta_va = meta_va[mask_va].reset_index(drop=True)
        y_tr = y_tr[mask_tr].values if hasattr(y_tr, 'values') else y_tr[mask_tr]
        y_va = y_va[mask_va].values if hasattr(y_va, 'values') else y_va[mask_va]
    else:
        y_tr = data["y_train"]
        y_va = data["y_val"]

    X_tr, X_va, y_tr, y_va, used_cols = _extract_scalar_Xy(
        meta_tr, meta_va, SCALAR_FEATURES, y_tr, y_va)

    if X_tr is None or len(np.unique(y_tr)) < 2:
        print(f"  [{task_name}] Skipped — insufficient features or classes")
        return {}

    print(f"\n  ── {task_name} ──")
    print(f"  Features: {used_cols}")
    print(f"  Train: {X_tr.shape[0]}  Val: {X_va.shape[0]}")
    print(f"  Class distribution (train): {dict(zip(*np.unique(y_tr, return_counts=True)))}")

    results = {}
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42),
        "SVM": SVC(
            kernel="rbf", class_weight="balanced", random_state=42),
        "LDA": LinearDiscriminantAnalysis(),
    }

    for name, model in models.items():
        try:
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            acc = accuracy_score(y_va, y_pred)
            f1 = f1_score(y_va, y_pred, average="weighted", zero_division=0)
            prec = precision_score(y_va, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_va, y_pred, average="weighted", zero_division=0)
            results[name] = {
                "accuracy": acc, "f1_weighted": f1,
                "precision": prec, "recall": rec,
            }
            print(f"    {name:25s}  acc={acc:.3f}  F1={f1:.3f}  "
                  f"P={prec:.3f}  R={rec:.3f}")
        except Exception as e:
            print(f"    {name:25s}  FAILED — {e}")

    best = max(results, key=lambda k: results[k]["f1_weighted"]) if results else None
    if best:
        model = models[best]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        print(f"\n  Best: {best}")
        labels = sorted(np.unique(np.concatenate([y_tr, y_va])))
        print(classification_report(y_va, y_pred, labels=labels, zero_division=0))

    return results


def phase1(run_name):
    print("\n" + "=" * 60)
    print("  PHASE 1 — SCALAR BASELINES")
    print("=" * 60)

    conditions = discover_conditions(run_name)
    if not conditions:
        print("  No conditions found — skipping phase 1")
        return {}

    all_results = {}

    for cond in conditions:
        print(f"\n{'─' * 40}")
        print(f"  Condition: {cond}")
        print(f"{'─' * 40}")
        data = load_tensors(run_name, cond)
        if data is None:
            print(f"  Skipping {cond} — missing tensors")
            continue

        all_results[f"{cond}_gonogo"] = run_scalar_baseline(
            data, f"Go vs NoGo ({cond})")

    data_pooled = pool_conditions(run_name, conditions)
    if not data_pooled:
        print("  No valid conditions for pooling — skipping pooled baseline")
        return all_results

    print(f"\n{'─' * 40}")
    print(f"  Pooled ({len(conditions)} conditions)")
    print(f"{'─' * 40}")
    all_results["pooled_gonogo"] = run_scalar_baseline(
        data_pooled, "Go vs NoGo (pooled)")

    return all_results


# ═══════════════════════════════════════════════════════════
#  TEMPORAL ATTENTION BLOCK (optional, for EEG-Conformer-style models)
# ═══════════════════════════════════════════════════════════

class TemporalAttentionBlock(nn.Module):
    """Lightweight multi-head self-attention over the time dimension.

    Inserted between EEGNet block1 and block2. Input shape from block1:
    (B, F1*D, 1, T_reduced) -> reshape to (B, T, F1*D) for attention.
    """

    def __init__(self, embed_dim, n_heads=2, ff_dim=None, dropout=0.1):
        super().__init__()
        if ff_dim is None:
            ff_dim = embed_dim * 4
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        B, C, _, T = x.shape
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T, C)
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.ff(self.norm2(x))
        return x.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, T)


# ═══════════════════════════════════════════════════════════
#  EEGNET MODEL
# ═══════════════════════════════════════════════════════════

class EEGNet(nn.Module):
    """Compact CNN for EEG decoding (Lawhern et al., 2018).

    Extended with configurable kernel sizes and optional transformer layer.
    """

    def __init__(self, n_channels, n_times, n_classes,
                 F1=8, D=2, F2=16, dropout=0.25,
                 kernel_length=64, sep_kernel_length=16,
                 use_transformer=False, n_heads=2,
                 transformer_dropout=0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        if use_transformer:
            self.transformer = TemporalAttentionBlock(
                embed_dim=F1 * D, n_heads=n_heads,
                ff_dim=F1 * D * 4, dropout=transformer_dropout)
        else:
            self.transformer = None

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, sep_kernel_length),
                      padding=(0, sep_kernel_length // 2), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        dummy = torch.zeros(1, 1, n_channels, n_times)
        with torch.no_grad():
            out = self.block1(dummy)
            if self.transformer is not None:
                out = self.transformer(out)
            out = self.block2(out)
        self.flat_size = out.numel()
        self.classifier = nn.Linear(self.flat_size, n_classes)

    def _features(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        if self.transformer is not None:
            x = self.transformer(x)
        x = self.block2(x)
        return x.flatten(1)

    def forward(self, x):
        return self.classifier(self._features(x))

    def embed(self, x):
        """Return the pre-classifier embedding."""
        return self._features(x)


# ═══════════════════════════════════════════════════════════
#  MULTIMODAL MODEL (Phase 3)
# ═══════════════════════════════════════════════════════════

class MultimodalNet(nn.Module):
    """Dual-branch: EEGNet for EEG + small CNN for ET, late fusion."""

    def __init__(self, n_eeg_ch, n_et_ch, n_times, n_classes,
                 F1=8, D=2, F2=16, dropout=0.25, **eeg_kwargs):
        super().__init__()
        self.eeg_branch = EEGNet(
            n_eeg_ch, n_times, n_classes, F1=F1, D=D, F2=F2, dropout=dropout,
            **{k: v for k, v in eeg_kwargs.items()
               if k in ("kernel_length", "sep_kernel_length",
                         "use_transformer", "n_heads", "transformer_dropout")})
        self.eeg_branch.classifier = nn.Identity()

        self.et_branch = nn.Sequential(
            nn.Conv2d(1, 8, (1, 32), padding=(0, 16), bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (n_et_ch, 1), groups=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        dummy_eeg = torch.zeros(1, 1, n_eeg_ch, n_times)
        dummy_et = torch.zeros(1, 1, n_et_ch, n_times)
        with torch.no_grad():
            eeg_out = self.eeg_branch._features(dummy_eeg)
            et_out = self.et_branch(dummy_et)
        self.eeg_flat = eeg_out.shape[1]
        self.et_flat = et_out.numel()

        self.classifier = nn.Sequential(
            nn.Linear(self.eeg_flat + self.et_flat, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x_eeg, x_et):
        if x_et.dim() == 3:
            x_et = x_et.unsqueeze(1)

        eeg_feat = self.eeg_branch._features(x_eeg)
        et_feat = self.et_branch(x_et).flatten(1)
        fused = torch.cat([eeg_feat, et_feat], dim=1)
        return self.classifier(fused)


# ═══════════════════════════════════════════════════════════
#  TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_class_weights(y):
    """Inverse-frequency class weights for imbalanced data."""
    classes, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(classes) * counts)
    w = torch.zeros(int(classes.max()) + 1, dtype=torch.float32)
    for c, wt in zip(classes, weights):
        w[int(c)] = wt
    return w


def train_epoch(model, loader, optimizer, criterion, device, multimodal=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        if multimodal:
            x_eeg, x_et, y = batch
            x_eeg, x_et, y = x_eeg.to(device), x_et.to(device), y.to(device)
            logits = model(x_eeg, x_et)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, multimodal=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for batch in loader:
        if multimodal:
            x_eeg, x_et, y = batch
            x_eeg, x_et, y = x_eeg.to(device), x_et.to(device), y.to(device)
            logits = model(x_eeg, x_et)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_targets.extend(y.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_targets)


def train_dl_model(model, X_train, X_val, y_train, y_val,
                   n_epochs=80, batch_size=64, lr=1e-3,
                   X_et_train=None, X_et_val=None,
                   task_name="", save_dir=None):
    """Generic DL training loop for single or multimodal models."""
    device = get_device()
    model = model.to(device)
    multimodal = X_et_train is not None

    weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    y_tr_t = torch.from_numpy(y_train).long()
    y_va_t = torch.from_numpy(y_val).long()

    if multimodal:
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(X_et_train).float(),
            y_tr_t)
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(X_et_val).float(),
            y_va_t)
    else:
        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(), y_tr_t)
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(), y_va_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_acc = 0.0
    best_state = None

    print(f"\n  Training {task_name} on {device} "
          f"({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  "
          f"Epochs: {n_epochs}  Batch: {batch_size}")

    for epoch in range(n_epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, multimodal)
        va_loss, va_acc, va_preds, va_targets = eval_epoch(
            model, val_loader, criterion, device, multimodal)
        scheduler.step()

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = va_preds
            best_targets = va_targets

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:3d}/{n_epochs}  "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}  "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\n  Best val acc: {best_val_acc:.3f}")
    f1 = f1_score(best_targets, best_preds, average="weighted", zero_division=0)
    prec = precision_score(best_targets, best_preds, average="weighted", zero_division=0)
    rec = recall_score(best_targets, best_preds, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(best_targets, best_preds)
    print(f"  Best val F1 (weighted): {f1:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}")
    labels = sorted(np.unique(np.concatenate([y_train, y_val])))
    print(classification_report(
        best_targets, best_preds, labels=labels, zero_division=0))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = task_name.replace(" ", "_").replace("/", "_")
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f"{safe_name}.pt"))
        print(f"  Saved: {save_dir}/{safe_name}.pt")

    return model, {
        "accuracy": best_val_acc, "balanced_accuracy": bal_acc,
        "f1_weighted": f1, "precision": prec, "recall": rec,
    }


# ═══════════════════════════════════════════════════════════
#  PHASE 2 — EEGNET
# ═══════════════════════════════════════════════════════════

def phase2(run_name, cfg=None):
    print("\n" + "=" * 60)
    print("  PHASE 2 — EEGNet")
    print("=" * 60)

    cfg = cfg or {}
    ekw = _eegnet_kwargs(cfg)
    tkw = _training_kwargs(cfg)
    conditions = discover_conditions(run_name)
    if not conditions:
        print("  No conditions found — skipping phase 2")
        return {}

    results = {}
    save_dir = os.path.join(RUNS_ROOT, run_name, "models")

    for cond in conditions:
        print(f"\n{'─' * 40}")
        print(f"  Condition: {cond}")
        print(f"{'─' * 40}")
        data = load_tensors(run_name, cond)
        if data is None:
            print(f"  Skipping {cond} — missing tensors")
            continue

        n_ch = data["X_eeg_train"].shape[1]
        n_times = data["X_eeg_train"].shape[2]
        n_classes = len(np.unique(data["y_train"]))

        model = EEGNet(n_ch, n_times, n_classes, **ekw)
        _, res = train_dl_model(
            model,
            data["X_eeg_train"], data["X_eeg_val"],
            data["y_train"], data["y_val"],
            n_epochs=tkw["n_epochs"], batch_size=tkw["batch_size"],
            lr=tkw["lr"],
            task_name=f"EEGNet_{cond}",
            save_dir=save_dir,
        )
        results[cond] = res

    data_pooled = pool_conditions(run_name, conditions)
    if not data_pooled:
        print("  No valid conditions for pooling — skipping pooled EEGNet")
        return results

    print(f"\n{'─' * 40}")
    print(f"  Pooled ({len(conditions)} conditions)")
    print(f"{'─' * 40}")

    n_ch = data_pooled["X_eeg_train"].shape[1]
    n_times = data_pooled["X_eeg_train"].shape[2]
    n_classes = len(np.unique(data_pooled["y_train"]))

    model = EEGNet(n_ch, n_times, n_classes, **ekw)
    _, res = train_dl_model(
        model,
        data_pooled["X_eeg_train"], data_pooled["X_eeg_val"],
        data_pooled["y_train"], data_pooled["y_val"],
        n_epochs=tkw["n_epochs"], batch_size=tkw["batch_size"],
        lr=tkw["lr"],
        task_name="EEGNet_pooled",
        save_dir=save_dir,
    )
    results["pooled"] = res

    return results


# ═══════════════════════════════════════════════════════════
#  PHASE 3 — MULTIMODAL (EEG + ET)
# ═══════════════════════════════════════════════════════════

def phase3(run_name, cfg=None):
    print("\n" + "=" * 60)
    print("  PHASE 3 — MULTIMODAL (EEG + ET)")
    print("=" * 60)

    cfg = cfg or {}
    ekw = _eegnet_kwargs(cfg)
    tkw = _training_kwargs(cfg)
    conditions = discover_conditions(run_name)
    if not conditions:
        print("  No conditions found — skipping phase 3")
        return {}

    results = {}
    save_dir = os.path.join(RUNS_ROOT, run_name, "models")

    data_pooled = pool_conditions(run_name, conditions)

    if not data_pooled or "X_et_train" not in data_pooled:
        print("  No ET tensors found — skipping multimodal phase")
        return results

    n_eeg_ch = data_pooled["X_eeg_train"].shape[1]
    n_et_ch = data_pooled["X_et_train"].shape[1]
    n_times = data_pooled["X_eeg_train"].shape[2]
    n_classes = len(np.unique(data_pooled["y_train"]))

    print(f"  EEG: {n_eeg_ch} ch × {n_times} t")
    print(f"  ET:  {n_et_ch} ch × {n_times} t")

    model = MultimodalNet(n_eeg_ch, n_et_ch, n_times, n_classes, **ekw)
    _, res = train_dl_model(
        model,
        data_pooled["X_eeg_train"], data_pooled["X_eeg_val"],
        data_pooled["y_train"], data_pooled["y_val"],
        X_et_train=data_pooled["X_et_train"],
        X_et_val=data_pooled["X_et_val"],
        n_epochs=tkw["n_epochs"], batch_size=tkw["batch_size"],
        lr=tkw["lr"],
        task_name="Multimodal_pooled",
        save_dir=save_dir,
    )
    results["multimodal_pooled"] = res

    for cond in conditions:
        data = load_tensors(run_name, cond)
        if data is None or "X_et_train" not in data:
            continue
        print(f"\n{'─' * 40}")
        print(f"  Condition: {cond}")
        print(f"{'─' * 40}")
        model = MultimodalNet(n_eeg_ch, n_et_ch, n_times, n_classes, **ekw)
        _, res = train_dl_model(
            model,
            data["X_eeg_train"], data["X_eeg_val"],
            data["y_train"], data["y_val"],
            X_et_train=data["X_et_train"],
            X_et_val=data["X_et_val"],
            n_epochs=tkw["n_epochs"], batch_size=tkw["batch_size"],
            lr=tkw["lr"],
            task_name=f"Multimodal_{cond}",
            save_dir=save_dir,
        )
        results[cond] = res

    return results


# ═══════════════════════════════════════════════════════════
#  PHASE 4 — OUTCOME PREDICTION
# ═══════════════════════════════════════════════════════════

def _label_correct_vs_error(meta):
    """Correct (HIT + CORRECT_REJECTION) = 0, Error (MISS + CE) = 1."""
    outcome = meta["outcome"].astype(str)
    mask = outcome.isin(["HIT", "CORRECT_REJECTION", "MISS", "COMMISSION_ERROR"])
    y = np.where(outcome.isin(["HIT", "CORRECT_REJECTION"]), 0, 1)
    return pd.Series(y, index=meta.index), mask


def _label_hit_vs_miss(meta):
    """Within Go trials only: HIT=0, MISS=1."""
    outcome = meta["outcome"].astype(str)
    mask = outcome.isin(["HIT", "MISS"])
    y = np.where(outcome == "HIT", 0, 1)
    return pd.Series(y, index=meta.index), mask


def phase4(run_name, cfg=None):
    print("\n" + "=" * 60)
    print("  PHASE 4 — OUTCOME PREDICTION")
    print("=" * 60)

    cfg = cfg or {}
    ekw = _eegnet_kwargs(cfg)
    tkw = _training_kwargs(cfg)
    conditions = discover_conditions(run_name)
    if not conditions:
        print("  No conditions found — skipping phase 4")
        return {}

    data_pooled = pool_conditions(run_name, conditions)
    if not data_pooled:
        print("  No valid conditions for pooling — skipping phase 4")
        return {}

    results = {}
    save_dir = os.path.join(RUNS_ROOT, run_name, "models")

    # 4A: Scalar baselines for outcome tasks
    print("\n  ── 4A: Scalar baselines (Correct vs Error) ──")
    results["scalar_correct_error"] = run_scalar_baseline(
        data_pooled, "Correct vs Error (pooled)", _label_correct_vs_error)

    print("\n  ── 4B: Scalar baselines (HIT vs MISS, Go trials) ──")
    results["scalar_hit_miss"] = run_scalar_baseline(
        data_pooled, "HIT vs MISS (pooled)", _label_hit_vs_miss)

    # 4C: EEGNet for Correct vs Error
    print(f"\n{'─' * 40}")
    print("  4C: EEGNet — Correct vs Error (pooled)")
    print(f"{'─' * 40}")

    meta_tr = data_pooled["meta_train"]
    meta_va = data_pooled["meta_val"]

    y_tr_ce, mask_tr = _label_correct_vs_error(meta_tr)
    y_va_ce, mask_va = _label_correct_vs_error(meta_va)

    X_tr = data_pooled["X_eeg_train"][mask_tr.values]
    X_va = data_pooled["X_eeg_val"][mask_va.values]
    y_tr = y_tr_ce[mask_tr].values
    y_va = y_va_ce[mask_va].values

    if len(np.unique(y_tr)) >= 2 and len(np.unique(y_va)) >= 2:
        n_ch, n_times = X_tr.shape[1], X_tr.shape[2]
        model = EEGNet(n_ch, n_times, 2, **ekw)
        _, res = train_dl_model(
            model, X_tr, X_va, y_tr, y_va,
            n_epochs=tkw["n_epochs"], batch_size=tkw["batch_size"],
            lr=tkw["lr"],
            task_name="EEGNet_correct_vs_error",
            save_dir=save_dir,
        )
        results["eegnet_correct_error"] = res
    else:
        print("  Skipped — not enough classes in both splits")

    # 4D: EEGNet for HIT vs MISS
    print(f"\n{'─' * 40}")
    print("  4D: EEGNet — HIT vs MISS (pooled)")
    print(f"{'─' * 40}")

    y_tr_hm, mask_tr = _label_hit_vs_miss(meta_tr)
    y_va_hm, mask_va = _label_hit_vs_miss(meta_va)

    X_tr = data_pooled["X_eeg_train"][mask_tr.values]
    X_va = data_pooled["X_eeg_val"][mask_va.values]
    y_tr = y_tr_hm[mask_tr].values
    y_va = y_va_hm[mask_va].values

    if len(np.unique(y_tr)) >= 2 and len(np.unique(y_va)) >= 2:
        model = EEGNet(n_ch, n_times, 2, **ekw)
        _, res = train_dl_model(
            model, X_tr, X_va, y_tr, y_va,
            n_epochs=tkw["n_epochs"], batch_size=tkw["batch_size"],
            lr=tkw["lr"],
            task_name="EEGNet_hit_vs_miss",
            save_dir=save_dir,
        )
        results["eegnet_hit_miss"] = res
    else:
        print("  Skipped — not enough classes in both splits")

    # 4E: Multimodal for Correct vs Error (if ET available)
    if "X_et_train" in data_pooled:
        print(f"\n{'─' * 40}")
        print("  4E: Multimodal — Correct vs Error (pooled)")
        print(f"{'─' * 40}")

        y_tr_ce, mask_tr = _label_correct_vs_error(data_pooled["meta_train"])
        y_va_ce, mask_va = _label_correct_vs_error(data_pooled["meta_val"])

        X_eeg_tr = data_pooled["X_eeg_train"][mask_tr.values]
        X_eeg_va = data_pooled["X_eeg_val"][mask_va.values]
        X_et_tr = data_pooled["X_et_train"][mask_tr.values]
        X_et_va = data_pooled["X_et_val"][mask_va.values]
        y_tr = y_tr_ce[mask_tr].values
        y_va = y_va_ce[mask_va].values

        if len(np.unique(y_tr)) >= 2 and len(np.unique(y_va)) >= 2:
            n_eeg_ch = X_eeg_tr.shape[1]
            n_et_ch = X_et_tr.shape[1]
            n_times = X_eeg_tr.shape[2]

            model = MultimodalNet(n_eeg_ch, n_et_ch, n_times, 2, **ekw)
            _, res = train_dl_model(
                model, X_eeg_tr, X_eeg_va, y_tr, y_va,
                X_et_train=X_et_tr, X_et_val=X_et_va,
                n_epochs=tkw["n_epochs"], batch_size=tkw["batch_size"],
                lr=tkw["lr"],
                task_name="Multimodal_correct_vs_error",
                save_dir=save_dir,
            )
            results["multimodal_correct_error"] = res

    return results


# ═══════════════════════════════════════════════════════════
#  PHASE 5 — VISION INTEGRATION
# ═══════════════════════════════════════════════════════════

VISION_SCALAR_FEATURES = [
    "dominant_cluster", "cluster_entropy",
    "n_fixations_in_window", "emb_spread",
]


def phase5(run_name):
    print("\n" + "=" * 60)
    print("  PHASE 5 — VISION INTEGRATION")
    print("=" * 60)

    conditions = discover_conditions(run_name)
    if not conditions:
        print("  No conditions found — skipping phase 5")
        return {}

    data_pooled = pool_conditions(run_name, conditions)
    if not data_pooled:
        print("  No valid conditions for pooling — skipping phase 5")
        return {}

    meta_tr = data_pooled["meta_train"]
    vis_cols = [c for c in VISION_SCALAR_FEATURES if c in meta_tr.columns]
    vis_cluster_cols = [c for c in meta_tr.columns if c.startswith("vis_cluster_")]
    all_vis = vis_cols + vis_cluster_cols

    if not all_vis:
        print("  No vision features found in metadata.")
        print("  Run the vision pipeline first, then re-run fusion + features + dl_prep.")
        print("  Then re-run: python src/train.py --phase 5")
        return {}

    print(f"  Vision features found: {all_vis}")

    combined_features = SCALAR_FEATURES + all_vis
    results = {}

    results["gonogo_eeg_et_vis"] = run_scalar_baseline(
        data_pooled, "Go/NoGo (EEG+ET+Vision features)")

    results["correct_error_eeg_et_vis"] = run_scalar_baseline(
        data_pooled, "Correct vs Error (EEG+ET+Vision)",
        _label_correct_vs_error)

    return results


# ═══════════════════════════════════════════════════════════
#  GAZE SEQUENCE ENCODER (for Phase 7)
# ═══════════════════════════════════════════════════════════

class GazeSequenceEncoder(nn.Module):
    """Encodes CLIP fixation category sequences via LSTM or 1D CNN."""

    def __init__(self, n_categories, embedding_dim=32, hidden_dim=64,
                 encoder_type="lstm", dropout=0.3):
        super().__init__()
        self.encoder_type = encoder_type
        self.category_embed = nn.Linear(n_categories, embedding_dim)
        self.drop = nn.Dropout(dropout)

        if encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=embedding_dim, hidden_size=hidden_dim,
                num_layers=1, batch_first=True, bidirectional=True,
            )
            self.output_dim = hidden_dim * 2
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim), nn.ELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim), nn.ELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.output_dim = hidden_dim

    def forward(self, x):
        emb = self.drop(self.category_embed(x))
        if self.encoder_type == "lstm":
            _, (h_n, _) = self.encoder(emb)
            return torch.cat([h_n[0], h_n[1]], dim=1)
        return self.encoder(emb.transpose(1, 2)).squeeze(-1)


# ═══════════════════════════════════════════════════════════
#  NO-GO FUSION MODEL (Phase 7)
# ═══════════════════════════════════════════════════════════

class NoGoFusionNet(nn.Module):
    """EEGNet + GazeSequenceEncoder → late fusion for no-go CR vs FA."""

    def __init__(self, n_eeg_ch, n_times, n_categories,
                 gaze_embed_dim=32, gaze_hidden=64, gaze_type="lstm",
                 fusion_dropout=0.3, **eeg_kwargs):
        super().__init__()
        self.eeg_branch = EEGNet(n_eeg_ch, n_times, 2, **eeg_kwargs)
        eeg_embed_dim = self.eeg_branch.flat_size
        self.eeg_branch.classifier = nn.Identity()

        self.gaze_branch = GazeSequenceEncoder(
            n_categories, gaze_embed_dim, gaze_hidden, gaze_type, fusion_dropout)
        gaze_out_dim = self.gaze_branch.output_dim

        self.fusion_head = nn.Sequential(
            nn.Linear(eeg_embed_dim + gaze_out_dim, 64),
            nn.ELU(), nn.Dropout(fusion_dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x_eeg, x_gaze):
        eeg_feat = self.eeg_branch.embed(x_eeg)
        gaze_feat = self.gaze_branch(x_gaze)
        return self.fusion_head(torch.cat([eeg_feat, gaze_feat], dim=1))

    def embed(self, x_eeg, x_gaze):
        eeg_feat = self.eeg_branch.embed(x_eeg)
        gaze_feat = self.gaze_branch(x_gaze)
        return torch.cat([eeg_feat, gaze_feat], dim=1)

    def load_eegnet_weights(self, state_dict):
        filtered = {k: v for k, v in state_dict.items()
                    if not k.startswith("classifier")}
        self.eeg_branch.load_state_dict(filtered, strict=False)


# ═══════════════════════════════════════════════════════════
#  NO-GO DATA UTILITIES
# ═══════════════════════════════════════════════════════════

CLIP_CATEGORIES = [
    "sky", "ocean", "water", "people",
    "vegetation", "trail_ground", "other",
]


def _pool_and_filter_nogo(run_name):
    """Pool all conditions, merge train+val, filter to no-go CR vs FA."""
    conditions = discover_conditions(run_name)
    if not conditions:
        print("  No conditions found")
        return None

    X_all, y_all, meta_all = [], [], []
    for cond in conditions:
        data = load_tensors(run_name, cond)
        if data is None:
            print(f"  Skipping {cond} — missing tensors")
            continue
        for split in ("train", "val"):
            X_all.append(data[f"X_eeg_{split}"])
            y_all.append(data[f"y_{split}"])
            meta_all.append(data[f"meta_{split}"])

    if not X_all:
        print("  No valid conditions loaded")
        return None

    X_eeg = np.concatenate(X_all)
    y_gonogo = np.concatenate(y_all)
    meta = pd.concat(meta_all, ignore_index=True)

    if "outcome" not in meta.columns:
        print("  No 'outcome' column — cannot distinguish CR from FA")
        return None

    nogo_mask = y_gonogo == 1
    if "trialType" in meta.columns:
        nogo_mask = meta["trialType"].values == 20

    outcome = meta["outcome"].astype(str).str.upper()
    cr_fa_mask = nogo_mask & outcome.isin(["CORRECT_REJECTION", "COMMISSION_ERROR"])
    indices = np.where(cr_fa_mask)[0]

    if len(indices) == 0:
        print("  No valid no-go trials (CR or FA) found")
        return None

    X = X_eeg[indices]
    filtered_meta = meta.iloc[indices].reset_index(drop=True)
    labels = (filtered_meta["outcome"].str.upper() == "CORRECT_REJECTION").astype(int).values

    n_cr, n_fa = int(labels.sum()), int(len(labels) - labels.sum())
    print(f"  No-go trials: {len(labels)} (CR={n_cr}, FA={n_fa})")

    if n_fa == 0:
        print("  WARNING: No false alarms found — cannot train binary classifier")
        return None

    return {"X_eeg": X, "labels": labels, "meta": filtered_meta,
            "n_cr": n_cr, "n_fa": n_fa, "conditions": conditions}


def _load_clip_gaze_sequences(run_name, meta, conditions, pre_stim_ms=2000):
    """Load CLIP fixation categories per trial from vision results."""
    vision_root = os.path.join(RUNS_ROOT, run_name, "vision")

    # Also check other runs if current run has no vision data
    if not os.path.isdir(vision_root):
        for rn in sorted(os.listdir(RUNS_ROOT), reverse=True):
            candidate = os.path.join(RUNS_ROOT, rn, "vision")
            if os.path.isdir(candidate):
                vision_root = candidate
                print(f"  Using vision data from run: {rn}")
                break

    if not os.path.isdir(vision_root):
        return None

    vision_dfs = {}
    for vdir in os.listdir(vision_root):
        vpath = os.path.join(vision_root, vdir, f"{vdir}_vision_results.csv")
        if os.path.exists(vpath):
            vision_dfs[vdir] = pd.read_csv(vpath)

    if not vision_dfs or "trigger_time" not in meta.columns:
        return None

    pre_s = pre_stim_ms / 1000.0
    sequences = []

    for _, trial in meta.iterrows():
        tt = trial.get("trigger_time")
        if tt is None or (isinstance(tt, float) and np.isnan(tt)):
            sequences.append([])
            continue

        cats = []
        for _, vdf in vision_dfs.items():
            # Prefer absolute timestamps if available; otherwise keep legacy behavior.
            if "timestamp_ns" in vdf.columns:
                ts = vdf["timestamp_ns"].astype(np.float64) / 1e9
            elif "timestamp_s" in vdf.columns:
                ts = vdf["timestamp_s"].astype(np.float64)
            else:
                continue

            if "gaze_target_category" in vdf.columns:
                lab_col = "gaze_target_category"
            elif "label" in vdf.columns:
                lab_col = "label"
            else:
                continue

            mask = (ts >= tt - pre_s) & (ts < tt)
            if np.any(mask):
                vals = vdf.loc[mask, lab_col].dropna()
                cats.extend(vals.astype(str).tolist())
        sequences.append(cats)

    n_with = sum(1 for s in sequences if len(s) > 0)
    print(f"  Gaze sequences: {n_with}/{len(sequences)} trials have CLIP data")
    return sequences if n_with > 0 else None


def _encode_gaze_onehot(sequences, categories=None, max_len=20):
    """One-hot encode fixation category sequences."""
    if categories is None:
        categories = CLIP_CATEGORIES
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)
    encoded = np.zeros((len(sequences), max_len, n_cats), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, cat in enumerate(seq[:max_len]):
            key = cat.lower().strip() if isinstance(cat, str) else ""
            if key in cat_to_idx:
                encoded[i, j, cat_to_idx[key]] = 1.0
    return encoded


def _nogo_kfold(model_factory, X_eeg, labels, n_folds=5, n_epochs=100,
                batch_size=32, lr=1e-3, patience=10,
                X_gaze=None, is_fusion=False, save_embeds=True):
    """Stratified k-fold CV for no-go CR vs FA classification."""
    device = get_device()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    all_embeddings, all_emb_labels, all_emb_folds = [], [], []

    for fi, (tr_idx, te_idx) in enumerate(skf.split(X_eeg, labels)):
        print(f"\n    Fold {fi+1}/{n_folds} "
              f"(train={len(tr_idx)}, test={len(te_idx)})")

        X_tr, X_te = X_eeg[tr_idx].copy(), X_eeg[te_idx].copy()
        y_tr, y_te = labels[tr_idx], labels[te_idx]

        for ch in range(X_tr.shape[1]):
            mu, sd = X_tr[:, ch].mean(), X_tr[:, ch].std() + 1e-8
            X_tr[:, ch] = (X_tr[:, ch] - mu) / sd
            X_te[:, ch] = (X_te[:, ch] - mu) / sd

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            print(f"      Skipped — single class in a split")
            continue

        tr_t = [torch.from_numpy(X_tr).float()]
        te_t = [torch.from_numpy(X_te).float()]
        if is_fusion and X_gaze is not None:
            tr_t.append(torch.from_numpy(X_gaze[tr_idx]).float())
            te_t.append(torch.from_numpy(X_gaze[te_idx]).float())
        tr_t.append(torch.from_numpy(y_tr).long())
        te_t.append(torch.from_numpy(y_te).long())

        train_ld = DataLoader(TensorDataset(*tr_t), batch_size=batch_size, shuffle=True)
        test_ld = DataLoader(TensorDataset(*te_t), batch_size=batch_size)

        weights = compute_class_weights(y_tr).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        model = model_factory().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

        best_loss, best_state, no_imp = float("inf"), None, 0

        for epoch in range(n_epochs):
            model.train()
            for batch in train_ld:
                if is_fusion:
                    xe, xg, yb = [b.to(device) for b in batch]
                    logits = model(xe, xg)
                else:
                    xb, yb = batch[0].to(device), batch[-1].to(device)
                    logits = model(xb)
                loss = criterion(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

            model.eval()
            vloss, vn = 0.0, 0
            with torch.no_grad():
                for batch in test_ld:
                    if is_fusion:
                        xe, xg, yb = [b.to(device) for b in batch]
                        logits = model(xe, xg)
                    else:
                        xb, yb = batch[0].to(device), batch[-1].to(device)
                        logits = model(xb)
                    vloss += criterion(logits, yb).item() * yb.size(0)
                    vn += yb.size(0)
            vloss /= vn

            if vloss < best_loss:
                best_loss = vloss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= patience:
                break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        probs_l, preds_l, tgts_l, embs_l = [], [], [], []
        with torch.no_grad():
            for batch in test_ld:
                if is_fusion:
                    xe, xg, yb = [b.to(device) for b in batch]
                    logits = model(xe, xg)
                    emb = model.embed(xe, xg)
                else:
                    xb, yb = batch[0].to(device), batch[-1].to(device)
                    logits = model(xb)
                    emb = model.embed(xb)
                probs_l.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                preds_l.extend(logits.argmax(1).cpu().numpy())
                tgts_l.extend(yb.cpu().numpy())
                embs_l.append(emb.cpu().numpy())

        probs_a, preds_a, tgts_a = np.array(probs_l), np.array(preds_l), np.array(tgts_l)
        bal_acc = balanced_accuracy_score(tgts_a, preds_a)
        try:
            auc = roc_auc_score(tgts_a, probs_a)
        except ValueError:
            auc = 0.5
        f1 = f1_score(tgts_a, preds_a, average="binary", zero_division=0)
        prec = precision_score(tgts_a, preds_a, average="binary", zero_division=0)
        rec = recall_score(tgts_a, preds_a, average="binary", zero_division=0)

        print(f"      bal_acc={bal_acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}")

        fold_results.append({
            "fold": fi, "balanced_accuracy": float(bal_acc),
            "auc_roc": float(auc), "f1": float(f1),
            "precision": float(prec), "recall": float(rec),
            "n_test": len(te_idx),
            "confusion_matrix": confusion_matrix(tgts_a, preds_a, labels=[0, 1]).tolist(),
        })

        if save_embeds:
            all_embeddings.append(np.concatenate(embs_l))
            all_emb_labels.append(y_te)
            all_emb_folds.append(np.full(len(y_te), fi))

        if fi == n_folds - 1 and best_state:
            fold_results[-1]["_best_state"] = best_state

    if not fold_results:
        return {"error": "No valid folds completed"}

    summary = {}
    for key in ["balanced_accuracy", "auc_roc", "f1", "precision", "recall"]:
        vals = [r[key] for r in fold_results]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
        summary[f"{key}_values"] = vals
    cm_list = [np.array(r["confusion_matrix"]) for r in fold_results]
    summary["confusion_matrix_sum"] = np.sum(cm_list, axis=0).tolist()
    summary["n_folds"] = len(fold_results)

    out = {"fold_results": fold_results, "summary": summary}
    if save_embeds and all_embeddings:
        out["embeddings"] = np.concatenate(all_embeddings)
        out["embedding_labels"] = np.concatenate(all_emb_labels)
        out["embedding_folds"] = np.concatenate(all_emb_folds)
    return out


# ═══════════════════════════════════════════════════════════
#  PHASE 6 — NO-GO INHIBITORY CONTROL (EEG-Only)
# ═══════════════════════════════════════════════════════════

def phase6(run_name, cfg=None):
    """Model A: EEGNet on no-go trials (CR vs FA), stratified k-fold."""
    print("\n" + "=" * 60)
    print("  PHASE 6 — NO-GO INHIBITORY CONTROL (EEG-Only)")
    print("=" * 60)

    if cfg is None:
        cfg = {}
    ekw = _eegnet_kwargs(cfg)
    tkw = _training_kwargs(cfg)

    nogo = _pool_and_filter_nogo(run_name)
    if nogo is None:
        return {"error": "insufficient_data"}

    X, labels = nogo["X_eeg"], nogo["labels"]
    n_ch, n_t = X.shape[1], X.shape[2]

    results = _nogo_kfold(
        lambda: EEGNet(n_ch, n_t, 2, **ekw),
        X, labels, **tkw,
    )
    if "error" in results:
        return results

    s = results["summary"]
    print(f"\n  Phase 6 Summary:")
    print(f"    Balanced Accuracy: {s['balanced_accuracy_mean']:.3f} "
          f"± {s['balanced_accuracy_std']:.3f}")
    print(f"    AUC-ROC:           {s['auc_roc_mean']:.3f} "
          f"± {s['auc_roc_std']:.3f}")
    print(f"    F1:                {s['f1_mean']:.3f} "
          f"± {s['f1_std']:.3f}")
    print(f"    Precision:         {s['precision_mean']:.3f} "
          f"± {s['precision_std']:.3f}")
    print(f"    Recall:            {s['recall_mean']:.3f} "
          f"± {s['recall_std']:.3f}")

    save_dir = os.path.join(RUNS_ROOT, run_name, "models")
    os.makedirs(save_dir, exist_ok=True)
    if "embeddings" in results:
        np.save(os.path.join(save_dir, "nogo_eeg_embeddings.npy"),
                results["embeddings"])
        np.save(os.path.join(save_dir, "nogo_eeg_embedding_labels.npy"),
                results["embedding_labels"])
        np.save(os.path.join(save_dir, "nogo_eeg_embedding_folds.npy"),
                results["embedding_folds"])

    last = results["fold_results"][-1]
    if "_best_state" in last:
        torch.save(last["_best_state"],
                   os.path.join(save_dir, "nogo_eeg_best.pt"))

        # Generate filter visualizations from best model
        try:
            from interpret import plot_spatial_filters, plot_temporal_filters, plot_saliency_topomap
            viz_model = EEGNet(n_ch, n_t, 2, **ekw)
            viz_model.load_state_dict(last["_best_state"])
            viz_model.eval()

            filter_dir = os.path.join(RUNS_ROOT, run_name, "plots", "filters")
            conditions = discover_conditions(run_name)
            fif_path = None
            for c in conditions:
                candidate = os.path.join(
                    RUNS_ROOT, run_name, "data",
                    f"sj*_{c}_Features-epo.fif")
                import glob
                matches = glob.glob(candidate)
                if matches:
                    fif_path = matches[0]
                    break

            if fif_path:
                plot_spatial_filters(viz_model, fif_path, filter_dir)
                plot_temporal_filters(viz_model, 250, filter_dir)
                plot_saliency_topomap(viz_model, X[:min(50, len(X))],
                                      fif_path, filter_dir)
        except Exception as e:
            print(f"  Filter visualization failed: {e}")

    detail = {
        "summary": s,
        "fold_results": [{k: v for k, v in r.items() if k != "_best_state"}
                         for r in results["fold_results"]],
        "n_cr": nogo["n_cr"], "n_fa": nogo["n_fa"],
    }
    import json as _json
    with open(os.path.join(RUNS_ROOT, run_name, "nogo_results.json"), "w") as f:
        _json.dump({"phase6": detail}, f, indent=2)

    return {
        "nogo_eeg_kfold": {
            "balanced_accuracy": s["balanced_accuracy_mean"],
            "auc_roc": s["auc_roc_mean"],
            "f1": s["f1_mean"],
            "precision": s["precision_mean"],
            "recall": s["recall_mean"],
            "n_folds": s["n_folds"],
            "n_cr": nogo["n_cr"],
            "n_fa": nogo["n_fa"],
        },
        "_internal": results,
    }


# ═══════════════════════════════════════════════════════════
#  PHASE 7 — NO-GO INHIBITORY CONTROL (EEG + Gaze Fusion)
# ═══════════════════════════════════════════════════════════

def phase7(run_name, phase6_results=None, cfg=None):
    """Model B: EEGNet + CLIP gaze fusion on no-go trials."""
    print("\n" + "=" * 60)
    print("  PHASE 7 — NO-GO INHIBITORY CONTROL (EEG + Gaze Fusion)")
    print("=" * 60)

    if cfg is None:
        cfg = {}
    ekw = _eegnet_kwargs(cfg)
    tkw = _training_kwargs(cfg)

    nogo = _pool_and_filter_nogo(run_name)
    if nogo is None:
        return {"error": "insufficient_data"}

    X, labels, meta = nogo["X_eeg"], nogo["labels"], nogo["meta"]
    n_ch, n_t = X.shape[1], X.shape[2]
    n_cats = len(CLIP_CATEGORIES)

    gaze_seqs = _load_clip_gaze_sequences(run_name, meta, nogo["conditions"])
    if gaze_seqs is None:
        print("\n  No CLIP gaze data available.")
        print("  Run the vision pipeline first, then re-run: "
              "python src/train.py --phase 7")
        return {"skipped": True, "reason": "no_clip_data"}

    X_gaze = _encode_gaze_onehot(gaze_seqs)
    print(f"  Gaze tensor: {X_gaze.shape}")

    p6_state = None
    if phase6_results and "_internal" in phase6_results:
        p6_folds = phase6_results["_internal"].get("fold_results", [])
        if p6_folds:
            p6_state = p6_folds[-1].get("_best_state")

    def factory():
        m = NoGoFusionNet(n_ch, n_t, n_cats,
                          gaze_embed_dim=32, gaze_hidden=64,
                          gaze_type="lstm", fusion_dropout=0.3,
                          **ekw)
        if p6_state:
            try:
                m.load_eegnet_weights(p6_state)
            except Exception:
                pass
        return m

    results = _nogo_kfold(
        factory, X, labels, **tkw,
        X_gaze=X_gaze, is_fusion=True,
    )
    if "error" in results:
        return results

    s = results["summary"]
    print(f"\n  Phase 7 Summary:")
    print(f"    Balanced Accuracy: {s['balanced_accuracy_mean']:.3f} "
          f"± {s['balanced_accuracy_std']:.3f}")
    print(f"    AUC-ROC:           {s['auc_roc_mean']:.3f} "
          f"± {s['auc_roc_std']:.3f}")
    print(f"    F1:                {s['f1_mean']:.3f} "
          f"± {s['f1_std']:.3f}")
    print(f"    Precision:         {s['precision_mean']:.3f} "
          f"± {s['precision_std']:.3f}")
    print(f"    Recall:            {s['recall_mean']:.3f} "
          f"± {s['recall_std']:.3f}")

    save_dir = os.path.join(RUNS_ROOT, run_name, "models")
    if "embeddings" in results:
        np.save(os.path.join(save_dir, "nogo_fusion_embeddings.npy"),
                results["embeddings"])
        np.save(os.path.join(save_dir, "nogo_fusion_embedding_labels.npy"),
                results["embedding_labels"])

    # Wilcoxon comparison with Phase 6
    comparison = {}
    if phase6_results and "_internal" in phase6_results:
        p6_s = phase6_results["_internal"].get("summary", {})
        p6_auc = p6_s.get("auc_roc_values", [])
        p7_auc = s.get("auc_roc_values", [])
        n = min(len(p6_auc), len(p7_auc))
        if n >= 2:
            a, b = np.array(p6_auc[:n]), np.array(p7_auc[:n])
            diff = b - a
            comparison = {
                "metric": "auc_roc", "n_pairs": n,
                "model_a_mean": float(np.mean(a)),
                "model_b_mean": float(np.mean(b)),
                "mean_difference": float(np.mean(diff)),
            }
            if not np.all(diff == 0):
                try:
                    stat, p = wilcoxon_test(a, b, alternative="two-sided")
                    comparison["p_value"] = float(p)
                    comparison["significant"] = p < 0.05
                    comparison["cohens_d"] = float(
                        np.mean(diff) / (np.std(diff) + 1e-10))
                except Exception as e:
                    comparison["error"] = str(e)
            else:
                comparison["p_value"] = 1.0
                comparison["significant"] = False

            print(f"\n  Model Comparison (A vs B, AUC-ROC):")
            print(f"    A: {comparison['model_a_mean']:.3f}  "
                  f"B: {comparison['model_b_mean']:.3f}  "
                  f"Diff: {comparison['mean_difference']:.3f}")
            if "p_value" in comparison:
                print(f"    Wilcoxon p={comparison['p_value']:.4f}  "
                      f"Significant: {comparison.get('significant')}")

    # Save detailed results for dashboard
    detail = {
        "summary": s,
        "fold_results": [{k: v for k, v in r.items() if k != "_best_state"}
                         for r in results["fold_results"]],
        "comparison": comparison,
        "n_cr": nogo["n_cr"], "n_fa": nogo["n_fa"],
    }
    import json as _json
    nogo_path = os.path.join(RUNS_ROOT, run_name, "nogo_results.json")
    existing = {}
    if os.path.exists(nogo_path):
        with open(nogo_path) as f:
            existing = _json.load(f)
    existing["phase7"] = detail
    with open(nogo_path, "w") as f:
        _json.dump(existing, f, indent=2)

    return {
        "nogo_fusion_kfold": {
            "balanced_accuracy": s["balanced_accuracy_mean"],
            "auc_roc": s["auc_roc_mean"],
            "f1": s["f1_mean"],
            "precision": s["precision_mean"],
            "recall": s["recall_mean"],
            "n_folds": s["n_folds"],
        },
        "nogo_comparison": comparison,
    }


# ═══════════════════════════════════════════════════════════
#  PHASE 8 — LOSO CROSS-VALIDATION (Primary RQ)
# ═══════════════════════════════════════════════════════════

def _load_subject_data(run_dirs):
    """Load nogo EEG data per subject from their run directories.

    Parameters
    ----------
    run_dirs : dict
        {subject_id: run_name} mapping.

    Returns
    -------
    X_all, labels_all, subject_ids_all, meta_all
    """
    X_parts, label_parts, sid_parts, meta_parts = [], [], [], []
    for sj, rn in run_dirs.items():
        nogo = _pool_and_filter_nogo(rn)
        if nogo is None:
            print(f"  Subject {sj}: no valid nogo data in {rn}")
            continue
        n = len(nogo["labels"])
        X_parts.append(nogo["X_eeg"])
        label_parts.append(nogo["labels"])
        sid_parts.append(np.full(n, sj, dtype=int))
        meta_parts.append(nogo["meta"].assign(subject_id=sj))
        print(f"  Subject {sj}: {n} nogo trials (CR={nogo['n_cr']}, FA={nogo['n_fa']})")
    if not X_parts:
        return None, None, None, None
    return (np.concatenate(X_parts), np.concatenate(label_parts),
            np.concatenate(sid_parts), pd.concat(meta_parts, ignore_index=True))


def _normalize_cross_subject(X_train, X_test):
    """Channel-wise z-score: fit on train subjects, apply to both."""
    for ch in range(X_train.shape[1]):
        mu = X_train[:, ch].mean()
        sd = X_train[:, ch].std() + 1e-8
        X_train[:, ch] = (X_train[:, ch] - mu) / sd
        X_test[:, ch] = (X_test[:, ch] - mu) / sd
    return X_train, X_test


def _loso_train_eval(model_factory, X_train, y_train, X_test, y_test,
                     n_epochs=100, batch_size=32, lr=1e-3, patience=10,
                     X_gaze_train=None, X_gaze_test=None, is_fusion=False):
    """Train and evaluate a single LOSO fold."""
    device = get_device()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    tr_tensors = [torch.from_numpy(X_train).float()]
    te_tensors = [torch.from_numpy(X_test).float()]
    if is_fusion and X_gaze_train is not None:
        tr_tensors.append(torch.from_numpy(X_gaze_train).float())
        te_tensors.append(torch.from_numpy(X_gaze_test).float())
    tr_tensors.append(torch.from_numpy(y_train).long())
    te_tensors.append(torch.from_numpy(y_test).long())

    train_ld = DataLoader(TensorDataset(*tr_tensors), batch_size=batch_size, shuffle=True)
    test_ld = DataLoader(TensorDataset(*te_tensors), batch_size=batch_size)

    weights = compute_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    model = model_factory().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    best_loss, best_state, no_imp = float("inf"), None, 0
    for epoch in range(n_epochs):
        model.train()
        for batch in train_ld:
            if is_fusion:
                xe, xg, yb = [b.to(device) for b in batch]
                logits = model(xe, xg)
            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for batch in test_ld:
                if is_fusion:
                    xe, xg, yb = [b.to(device) for b in batch]
                    logits = model(xe, xg)
                else:
                    xb, yb = batch[0].to(device), batch[-1].to(device)
                    logits = model(xb)
                vloss += criterion(logits, yb).item() * yb.size(0)
                vn += yb.size(0)
        vloss /= vn
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    probs_l, preds_l, tgts_l = [], [], []
    with torch.no_grad():
        for batch in test_ld:
            if is_fusion:
                xe, xg, yb = [b.to(device) for b in batch]
                logits = model(xe, xg)
            else:
                xb, yb = batch[0].to(device), batch[-1].to(device)
                logits = model(xb)
            probs_l.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            preds_l.extend(logits.argmax(1).cpu().numpy())
            tgts_l.extend(yb.cpu().numpy())

    probs_a, preds_a, tgts_a = np.array(probs_l), np.array(preds_l), np.array(tgts_l)
    bal_acc = balanced_accuracy_score(tgts_a, preds_a)
    try:
        auc = roc_auc_score(tgts_a, probs_a)
    except ValueError:
        auc = 0.5
    return {
        "balanced_accuracy": float(bal_acc),
        "auc_roc": float(auc),
        "f1": float(f1_score(tgts_a, preds_a, average="binary", zero_division=0)),
        "precision": float(precision_score(tgts_a, preds_a, average="binary", zero_division=0)),
        "recall": float(recall_score(tgts_a, preds_a, average="binary", zero_division=0)),
        "n_test": len(tgts_a),
        "confusion_matrix": confusion_matrix(tgts_a, preds_a, labels=[0, 1]).tolist(),
        "_best_state": best_state,
    }


def phase8_loso(run_name, cfg=None):
    """Primary RQ: LOSO cross-validation for EEG-based inhibitory control prediction."""
    print("\n" + "=" * 60)
    print("  PHASE 8 — LOSO CROSS-VALIDATION")
    print("=" * 60)

    if cfg is None:
        cfg = {}
    ekw = _eegnet_kwargs(cfg)
    tkw = _training_kwargs(cfg)

    loso_cfg = cfg.get("loso", {})
    run_dirs = loso_cfg.get("run_dirs", {})
    if not run_dirs:
        print("  No LOSO run_dirs configured in configs/config.yaml")
        print("  Add loso.run_dirs mapping subject IDs to run names")
        return {"error": "no_loso_config"}

    run_dirs = {int(k): v for k, v in run_dirs.items()}
    print(f"  Subjects: {sorted(run_dirs.keys())}")

    X, labels, sids, meta = _load_subject_data(run_dirs)
    if X is None:
        return {"error": "no_data"}

    n_ch, n_t = X.shape[1], X.shape[2]
    subjects = sorted(np.unique(sids))
    print(f"\n  Total: {len(labels)} trials across {len(subjects)} subjects")

    fold_results = []
    for held_out in subjects:
        print(f"\n  --- Held out: subject {held_out} ---")
        train_mask = sids != held_out
        test_mask = sids == held_out

        X_tr, X_te = X[train_mask].copy(), X[test_mask].copy()
        y_tr, y_te = labels[train_mask], labels[test_mask]

        X_tr, X_te = _normalize_cross_subject(X_tr, X_te)

        result = _loso_train_eval(
            lambda: EEGNet(n_ch, n_t, 2, **ekw),
            X_tr, y_tr, X_te, y_te, **tkw,
        )
        if result is None:
            print(f"    Skipped — single class in split")
            continue

        result["held_out_subject"] = int(held_out)
        result.pop("_best_state", None)
        fold_results.append(result)
        print(f"    bal_acc={result['balanced_accuracy']:.3f}  "
              f"AUC={result['auc_roc']:.3f}  F1={result['f1']:.3f}")

    if not fold_results:
        return {"error": "no_valid_folds"}

    summary = {"n_folds": len(fold_results)}
    for key in ["balanced_accuracy", "auc_roc", "f1", "precision", "recall"]:
        vals = [r[key] for r in fold_results]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
        summary[f"{key}_values"] = vals
    cm_list = [np.array(r["confusion_matrix"]) for r in fold_results]
    summary["confusion_matrix_sum"] = np.sum(cm_list, axis=0).tolist()

    print(f"\n  LOSO Summary ({len(fold_results)} folds):")
    print(f"    Balanced Accuracy: {summary['balanced_accuracy_mean']:.3f} "
          f"± {summary['balanced_accuracy_std']:.3f}")
    print(f"    AUC-ROC:           {summary['auc_roc_mean']:.3f} "
          f"± {summary['auc_roc_std']:.3f}")
    print(f"    F1:                {summary['f1_mean']:.3f} "
          f"± {summary['f1_std']:.3f}")

    # Moderator analysis: attend vs unattend
    moderator = {}
    for cond_filter, label in [("attend", "attend"), ("unattend", "unattend")]:
        cond_folds = []
        for held_out in subjects:
            train_mask = sids != held_out
            test_mask = sids == held_out

            if meta is not None and "condition" in meta.columns:
                cond_mask = meta["condition"].str.contains(cond_filter, case=False, na=False)
                tr_mask = train_mask & cond_mask.values
                te_mask = test_mask & cond_mask.values
            else:
                continue

            if tr_mask.sum() < 4 or te_mask.sum() < 2:
                continue

            X_tr, X_te = X[tr_mask].copy(), X[te_mask].copy()
            y_tr, y_te = labels[tr_mask], labels[te_mask]
            X_tr, X_te = _normalize_cross_subject(X_tr, X_te)

            result = _loso_train_eval(
                lambda: EEGNet(n_ch, n_t, 2, **ekw),
                X_tr, y_tr, X_te, y_te, **tkw,
            )
            if result:
                cond_folds.append(result)

        if cond_folds:
            aucs = [r["auc_roc"] for r in cond_folds]
            moderator[label] = {
                "auc_roc_mean": float(np.mean(aucs)),
                "auc_roc_std": float(np.std(aucs)),
                "n_folds": len(cond_folds),
            }
            print(f"\n    {label}: AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f} "
                  f"({len(cond_folds)} folds)")

    if len(moderator) == 2:
        a_vals = [r["auc_roc"] for r in cond_folds]  # last cond_folds
        att_aucs = moderator.get("attend", {})
        unatt_aucs = moderator.get("unattend", {})
        moderator["comparison"] = {
            "attend_mean": att_aucs.get("auc_roc_mean", 0),
            "unattend_mean": unatt_aucs.get("auc_roc_mean", 0),
            "note": "N<6: descriptive only, no significance test" if len(subjects) < 6
                    else "Wilcoxon paired test recommended with N≥6",
        }

    out_path = os.path.join(RUNS_ROOT, run_name, "loso_results.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "fold_results": fold_results,
                   "moderator": moderator}, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return {"loso": summary, "moderator": moderator}


# ═══════════════════════════════════════════════════════════
#  PHASE 9 — GAZE COMPARISON IN WALKING (Secondary RQ)
# ═══════════════════════════════════════════════════════════

def phase9_gaze_walking(run_name, cfg=None):
    """Secondary RQ: Does adding semantic gaze improve prediction during walking?"""
    print("\n" + "=" * 60)
    print("  PHASE 9 — GAZE COMPARISON (WALKING)")
    print("=" * 60)

    if cfg is None:
        cfg = {}
    ekw = _eegnet_kwargs(cfg)
    tkw = _training_kwargs(cfg)

    loso_cfg = cfg.get("loso", {})
    run_dirs = loso_cfg.get("run_dirs", {})
    if not run_dirs:
        print("  No LOSO run_dirs configured")
        return {"error": "no_loso_config"}

    run_dirs = {int(k): v for k, v in run_dirs.items()}

    # Load only walking conditions
    X_parts, label_parts, sid_parts, meta_parts = [], [], [], []
    gaze_parts = []
    has_gaze = True
    n_cats = len(CLIP_CATEGORIES)

    for sj, rn in run_dirs.items():
        conditions = discover_conditions(rn)
        walk_conds = [c for c in conditions if "walk" in c.lower()]
        if not walk_conds:
            continue

        for cond in walk_conds:
            data = load_tensors(rn, cond)
            for split in ("train", "val"):
                X_parts.append(data[f"X_eeg_{split}"])
                label_parts.append(data[f"y_{split}"])
                n = len(data[f"y_{split}"])
                sid_parts.append(np.full(n, sj, dtype=int))
                m = data[f"meta_{split}"].copy()
                m["subject_id"] = sj
                m["condition"] = cond
                meta_parts.append(m)

    if not X_parts:
        print("  No walking condition data found")
        return {"error": "no_walk_data"}

    X_all = np.concatenate(X_parts)
    y_all = np.concatenate(label_parts)
    sids = np.concatenate(sid_parts)
    meta_all = pd.concat(meta_parts, ignore_index=True)

    # Filter to nogo CR vs FA
    if "outcome" not in meta_all.columns:
        print("  No outcome column")
        return {"error": "no_outcome"}

    nogo_mask = meta_all.get("trialType", pd.Series(dtype=float)).values == 20
    if not nogo_mask.any():
        nogo_mask = y_all == 1
    outcome = meta_all["outcome"].astype(str).str.upper()
    cr_fa = nogo_mask & outcome.isin(["CORRECT_REJECTION", "COMMISSION_ERROR"])
    idx = np.where(cr_fa)[0]
    if len(idx) < 10:
        print(f"  Only {len(idx)} nogo walking trials — too few")
        return {"error": "insufficient_walk_nogo"}

    X = X_all[idx]
    labels = (meta_all.iloc[idx]["outcome"].str.upper() == "CORRECT_REJECTION").astype(int).values
    sids_f = sids[idx]
    meta_f = meta_all.iloc[idx].reset_index(drop=True)
    n_ch, n_t = X.shape[1], X.shape[2]
    subjects = sorted(np.unique(sids_f))

    print(f"  Walking nogo trials: {len(labels)} across {len(subjects)} subjects")

    # Model A: EEGNet only
    model_a_folds = []
    for held_out in subjects:
        tr = sids_f != held_out; te = sids_f == held_out
        X_tr, X_te = X[tr].copy(), X[te].copy()
        X_tr, X_te = _normalize_cross_subject(X_tr, X_te)
        r = _loso_train_eval(lambda: EEGNet(n_ch, n_t, 2, **ekw),
                             X_tr, labels[tr], X_te, labels[te], **tkw)
        if r:
            r["held_out"] = int(held_out)
            model_a_folds.append(r)

    # Model B: NoGoFusionNet (if gaze data available)
    model_b_folds = []
    gaze_seqs = _load_clip_gaze_sequences(
        list(run_dirs.values())[0], meta_f, [])
    if gaze_seqs is not None:
        X_gaze = _encode_gaze_onehot(gaze_seqs)
        for held_out in subjects:
            tr = sids_f != held_out; te = sids_f == held_out
            X_tr, X_te = X[tr].copy(), X[te].copy()
            X_tr, X_te = _normalize_cross_subject(X_tr, X_te)
            r = _loso_train_eval(
                lambda: NoGoFusionNet(n_ch, n_t, n_cats,
                                      gaze_embed_dim=32, gaze_hidden=64,
                                      gaze_type="lstm", fusion_dropout=0.3, **ekw),
                X_tr, labels[tr], X_te, labels[te], **tkw,
                X_gaze_train=X_gaze[tr], X_gaze_test=X_gaze[te],
                is_fusion=True,
            )
            if r:
                r["held_out"] = int(held_out)
                model_b_folds.append(r)

    result = {}
    for name, folds in [("model_a_eeg", model_a_folds), ("model_b_fusion", model_b_folds)]:
        if folds:
            aucs = [r["auc_roc"] for r in folds]
            result[name] = {
                "auc_roc_mean": float(np.mean(aucs)),
                "auc_roc_std": float(np.std(aucs)),
                "n_folds": len(folds),
                "fold_results": [{k: v for k, v in r.items() if k != "_best_state"}
                                 for r in folds],
            }
            print(f"\n  {name}: AUC={np.mean(aucs):.3f} ± {np.std(aucs):.3f}")

    # Paired comparison
    if model_a_folds and model_b_folds:
        n = min(len(model_a_folds), len(model_b_folds))
        a_aucs = np.array([r["auc_roc"] for r in model_a_folds[:n]])
        b_aucs = np.array([r["auc_roc"] for r in model_b_folds[:n]])
        diff = b_aucs - a_aucs
        comparison = {
            "n_pairs": n,
            "a_mean": float(np.mean(a_aucs)),
            "b_mean": float(np.mean(b_aucs)),
            "mean_diff": float(np.mean(diff)),
            "cohens_d": float(np.mean(diff) / (np.std(diff) + 1e-10)),
        }
        if n >= 6 and not np.all(diff == 0):
            try:
                stat, p = wilcoxon_test(a_aucs, b_aucs)
                comparison["p_value"] = float(p)
                comparison["significant"] = p < 0.05
            except Exception:
                pass
        result["comparison"] = comparison
        print(f"\n  Comparison: A={comparison['a_mean']:.3f} B={comparison['b_mean']:.3f} "
              f"d={comparison['cohens_d']:.3f}")

    out_path = os.path.join(RUNS_ROOT, run_name, "gaze_comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {out_path}")

    return result


# ═══════════════════════════════════════════════════════════
#  PHASE 10 — CROSS-CONDITION TRANSFER (Tertiary RQ)
# ═══════════════════════════════════════════════════════════

def phase10_cross_condition(run_name, cfg=None):
    """Tertiary RQ: Do lab-learned (sit) signatures generalize to walking?"""
    print("\n" + "=" * 60)
    print("  PHASE 10 — CROSS-CONDITION TRANSFER (Sit → Walk)")
    print("=" * 60)

    if cfg is None:
        cfg = {}
    ekw = _eegnet_kwargs(cfg)
    tkw = _training_kwargs(cfg)

    loso_cfg = cfg.get("loso", {})
    run_dirs = loso_cfg.get("run_dirs", {})
    if not run_dirs:
        print("  No LOSO run_dirs configured")
        return {"error": "no_loso_config"}
    run_dirs = {int(k): v for k, v in run_dirs.items()}

    def _load_by_movement(movement):
        X_parts, y_parts, sid_parts, meta_parts = [], [], [], []
        for sj, rn in run_dirs.items():
            conditions = discover_conditions(rn)
            conds = [c for c in conditions if movement in c.lower()]
            for cond in conds:
                data = load_tensors(rn, cond)
                for split in ("train", "val"):
                    X_parts.append(data[f"X_eeg_{split}"])
                    y_parts.append(data[f"y_{split}"])
                    n = len(data[f"y_{split}"])
                    sid_parts.append(np.full(n, sj, dtype=int))
                    m = data[f"meta_{split}"].copy()
                    m["subject_id"] = sj
                    meta_parts.append(m)
        if not X_parts:
            return None, None, None, None
        return (np.concatenate(X_parts), np.concatenate(y_parts),
                np.concatenate(sid_parts),
                pd.concat(meta_parts, ignore_index=True))

    def _filter_nogo(X, y, meta):
        if "outcome" not in meta.columns:
            return None, None
        nogo_mask = meta.get("trialType", pd.Series(dtype=float)).values == 20
        if not nogo_mask.any():
            nogo_mask = y == 1
        outcome = meta["outcome"].astype(str).str.upper()
        cr_fa = nogo_mask & outcome.isin(["CORRECT_REJECTION", "COMMISSION_ERROR"])
        idx = np.where(cr_fa)[0]
        if len(idx) < 4:
            return None, None
        labels = (meta.iloc[idx]["outcome"].str.upper() == "CORRECT_REJECTION").astype(int).values
        return X[idx], labels

    results = {}
    for train_cond, test_cond, name in [("sit", "walk", "sit_to_walk"),
                                         ("walk", "sit", "walk_to_sit")]:
        print(f"\n  --- Train: {train_cond} → Test: {test_cond} ---")
        X_train_raw, y_train_raw, _, meta_train = _load_by_movement(train_cond)
        X_test_raw, y_test_raw, _, meta_test = _load_by_movement(test_cond)

        if X_train_raw is None or X_test_raw is None:
            print(f"    No data for {train_cond} or {test_cond}")
            continue

        X_tr, y_tr = _filter_nogo(X_train_raw, y_train_raw, meta_train)
        X_te, y_te = _filter_nogo(X_test_raw, y_test_raw, meta_test)

        if X_tr is None or X_te is None:
            print(f"    Insufficient nogo trials")
            continue

        print(f"    Train: {len(y_tr)} trials  Test: {len(y_te)} trials")

        X_tr, X_te = X_tr.copy(), X_te.copy()
        X_tr, X_te = _normalize_cross_subject(X_tr, X_te)
        n_ch, n_t = X_tr.shape[1], X_tr.shape[2]

        r = _loso_train_eval(
            lambda: EEGNet(n_ch, n_t, 2, **ekw),
            X_tr, y_tr, X_te, y_te, **tkw,
        )
        if r:
            r.pop("_best_state", None)
            results[name] = r
            print(f"    bal_acc={r['balanced_accuracy']:.3f}  "
                  f"AUC={r['auc_roc']:.3f}  F1={r['f1']:.3f}")

    if results:
        out_path = os.path.join(RUNS_ROOT, run_name, "cross_condition_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {out_path}")

    return results


# ═══════════════════════════════════════════════════════════
#  SUMMARY & MAIN
# ═══════════════════════════════════════════════════════════

def save_summary(run_name, all_results):
    """Save a JSON summary of all results, merging with previous runs."""
    out = os.path.join(RUNS_ROOT, run_name, "ml_results.json")

    existing = {}
    if os.path.exists(out):
        try:
            with open(out) as f:
                existing = json.load(f)
        except Exception:
            pass

    for phase_name, phase_results in all_results.items():
        existing[phase_name] = {}
        for k, v in phase_results.items():
            if isinstance(v, dict):
                existing[phase_name][k] = {
                    kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                    for kk, vv in v.items()
                }
            else:
                existing[phase_name][k] = str(v)

    with open(out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nResults saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="PSY197B Model Training")
    parser.add_argument("--run", type=str, default=None,
                        help="Run directory name (default: latest)")
    parser.add_argument("--phase", type=int, nargs="+", default=None,
                        help="Phase(s) to run: 1-10 (default: 1-7)")
    args = parser.parse_args()

    run_name = args.run or find_latest_run()
    phases = args.phase or [1, 2, 3, 4, 5, 6, 7]
    cfg = load_model_config()

    print(f"\nRun: {run_name}")
    print(f"Phases: {phases}")
    print(f"Device: {get_device()}")

    all_results = {}

    if 1 in phases:
        all_results["phase1_scalar"] = phase1(run_name)
    if 2 in phases:
        all_results["phase2_eegnet"] = phase2(run_name, cfg=cfg)
    if 3 in phases:
        all_results["phase3_multimodal"] = phase3(run_name, cfg=cfg)
    if 4 in phases:
        all_results["phase4_outcome"] = phase4(run_name, cfg=cfg)
    if 5 in phases:
        all_results["phase5_vision"] = phase5(run_name)
    if 6 in phases:
        p6 = phase6(run_name, cfg=cfg)
        all_results["phase6_nogo_eeg"] = {
            k: v for k, v in p6.items() if not k.startswith("_")}
        _p6_ref = p6
    else:
        _p6_ref = None
    if 7 in phases:
        p7 = phase7(run_name, _p6_ref, cfg=cfg)
        all_results["phase7_nogo_fusion"] = {
            k: v for k, v in p7.items() if not k.startswith("_")}
    if 8 in phases:
        all_results["phase8_loso"] = phase8_loso(run_name, cfg=cfg)
    if 9 in phases:
        all_results["phase9_gaze_walking"] = phase9_gaze_walking(run_name, cfg=cfg)
    if 10 in phases:
        all_results["phase10_cross_condition"] = phase10_cross_condition(run_name, cfg=cfg)

    save_summary(run_name, all_results)

    print("\n" + "=" * 60)
    print("  ALL PHASES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
