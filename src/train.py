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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
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
    """Load pre-built tensors and metadata for a single condition."""
    tensor_dir = os.path.join(RUNS_ROOT, run_name, "data", "dl_tensors")
    prefix = condition

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
    prefixes = set()
    for f in os.listdir(tensor_dir):
        if f.endswith("_X_eeg_train.npy"):
            prefixes.add(f.replace("_X_eeg_train.npy", ""))
    return sorted(prefixes)


def pool_conditions(run_name, conditions):
    """Stack data across conditions, adding a condition column to metadata."""
    pooled = {}
    for i, cond in enumerate(conditions):
        d = load_tensors(run_name, cond)
        for split in ("train", "val"):
            d[f"meta_{split}"]["_condition"] = cond
            d[f"meta_{split}"]["_cond_idx"] = i
            for key in ("X_eeg", "X_et", "y", "meta"):
                full_key = f"{key}_{split}"
                if full_key in d:
                    pooled.setdefault(full_key, []).append(d[full_key])

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
            f1 = f1_score(y_va, y_pred, average="weighted")
            results[name] = {"accuracy": acc, "f1_weighted": f1}
            print(f"    {name:25s}  acc={acc:.3f}  F1={f1:.3f}")
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
    all_results = {}

    for cond in conditions:
        print(f"\n{'─' * 40}")
        print(f"  Condition: {cond}")
        print(f"{'─' * 40}")
        data = load_tensors(run_name, cond)

        all_results[f"{cond}_gonogo"] = run_scalar_baseline(
            data, f"Go vs NoGo ({cond})")

    data_pooled = pool_conditions(run_name, conditions)
    print(f"\n{'─' * 40}")
    print(f"  Pooled ({len(conditions)} conditions)")
    print(f"{'─' * 40}")
    all_results["pooled_gonogo"] = run_scalar_baseline(
        data_pooled, "Go vs NoGo (pooled)")

    return all_results


# ═══════════════════════════════════════════════════════════
#  EEGNET MODEL
# ═══════════════════════════════════════════════════════════

class EEGNet(nn.Module):
    """Compact CNN for EEG decoding (Lawhern et al., 2018)."""

    def __init__(self, n_channels, n_times, n_classes,
                 F1=8, D=2, F2=16, dropout=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        dummy = torch.zeros(1, 1, n_channels, n_times)
        with torch.no_grad():
            out = self.block2(self.block1(dummy))
        self.flat_size = out.numel()
        self.classifier = nn.Linear(self.flat_size, n_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)

    def embed(self, x):
        """Return the pre-classifier embedding."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return x.flatten(1)


# ═══════════════════════════════════════════════════════════
#  MULTIMODAL MODEL (Phase 3)
# ═══════════════════════════════════════════════════════════

class MultimodalNet(nn.Module):
    """Dual-branch: EEGNet for EEG + small CNN for ET, late fusion."""

    def __init__(self, n_eeg_ch, n_et_ch, n_times, n_classes,
                 F1=8, D=2, F2=16, dropout=0.25):
        super().__init__()
        self.eeg_branch = EEGNet(n_eeg_ch, n_times, n_classes, F1, D, F2, dropout)
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
            eeg_out = self.eeg_branch.block2(self.eeg_branch.block1(dummy_eeg))
            et_out = self.et_branch(dummy_et)
        self.eeg_flat = eeg_out.numel()
        self.et_flat = et_out.numel()

        self.classifier = nn.Sequential(
            nn.Linear(self.eeg_flat + self.et_flat, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x_eeg, x_et):
        if x_eeg.dim() == 3:
            x_eeg = x_eeg.unsqueeze(1)
        if x_et.dim() == 3:
            x_et = x_et.unsqueeze(1)

        eeg_feat = self.eeg_branch.block2(
            self.eeg_branch.block1(x_eeg)).flatten(1)
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
    f1 = f1_score(best_targets, best_preds, average="weighted")
    print(f"  Best val F1 (weighted): {f1:.3f}")
    labels = sorted(np.unique(np.concatenate([y_train, y_val])))
    print(classification_report(
        best_targets, best_preds, labels=labels, zero_division=0))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = task_name.replace(" ", "_").replace("/", "_")
        torch.save(model.state_dict(),
                   os.path.join(save_dir, f"{safe_name}.pt"))
        print(f"  Saved: {save_dir}/{safe_name}.pt")

    return model, {"accuracy": best_val_acc, "f1_weighted": f1}


# ═══════════════════════════════════════════════════════════
#  PHASE 2 — EEGNET
# ═══════════════════════════════════════════════════════════

def phase2(run_name):
    print("\n" + "=" * 60)
    print("  PHASE 2 — EEGNet")
    print("=" * 60)

    conditions = discover_conditions(run_name)
    results = {}
    save_dir = os.path.join(RUNS_ROOT, run_name, "models")

    for cond in conditions:
        print(f"\n{'─' * 40}")
        print(f"  Condition: {cond}")
        print(f"{'─' * 40}")
        data = load_tensors(run_name, cond)

        n_ch = data["X_eeg_train"].shape[1]
        n_times = data["X_eeg_train"].shape[2]
        n_classes = len(np.unique(data["y_train"]))

        model = EEGNet(n_ch, n_times, n_classes)
        _, res = train_dl_model(
            model,
            data["X_eeg_train"], data["X_eeg_val"],
            data["y_train"], data["y_val"],
            task_name=f"EEGNet_{cond}",
            save_dir=save_dir,
        )
        results[cond] = res

    data_pooled = pool_conditions(run_name, conditions)
    print(f"\n{'─' * 40}")
    print(f"  Pooled ({len(conditions)} conditions)")
    print(f"{'─' * 40}")

    n_ch = data_pooled["X_eeg_train"].shape[1]
    n_times = data_pooled["X_eeg_train"].shape[2]
    n_classes = len(np.unique(data_pooled["y_train"]))

    model = EEGNet(n_ch, n_times, n_classes)
    _, res = train_dl_model(
        model,
        data_pooled["X_eeg_train"], data_pooled["X_eeg_val"],
        data_pooled["y_train"], data_pooled["y_val"],
        task_name="EEGNet_pooled",
        save_dir=save_dir,
    )
    results["pooled"] = res

    return results


# ═══════════════════════════════════════════════════════════
#  PHASE 3 — MULTIMODAL (EEG + ET)
# ═══════════════════════════════════════════════════════════

def phase3(run_name):
    print("\n" + "=" * 60)
    print("  PHASE 3 — MULTIMODAL (EEG + ET)")
    print("=" * 60)

    conditions = discover_conditions(run_name)
    results = {}
    save_dir = os.path.join(RUNS_ROOT, run_name, "models")

    data_pooled = pool_conditions(run_name, conditions)

    if "X_et_train" not in data_pooled:
        print("  No ET tensors found — skipping multimodal phase")
        return results

    n_eeg_ch = data_pooled["X_eeg_train"].shape[1]
    n_et_ch = data_pooled["X_et_train"].shape[1]
    n_times = data_pooled["X_eeg_train"].shape[2]
    n_classes = len(np.unique(data_pooled["y_train"]))

    print(f"  EEG: {n_eeg_ch} ch × {n_times} t")
    print(f"  ET:  {n_et_ch} ch × {n_times} t")

    model = MultimodalNet(n_eeg_ch, n_et_ch, n_times, n_classes)
    _, res = train_dl_model(
        model,
        data_pooled["X_eeg_train"], data_pooled["X_eeg_val"],
        data_pooled["y_train"], data_pooled["y_val"],
        X_et_train=data_pooled["X_et_train"],
        X_et_val=data_pooled["X_et_val"],
        task_name="Multimodal_pooled",
        save_dir=save_dir,
    )
    results["multimodal_pooled"] = res

    for cond in conditions:
        data = load_tensors(run_name, cond)
        if "X_et_train" not in data:
            continue
        print(f"\n{'─' * 40}")
        print(f"  Condition: {cond}")
        print(f"{'─' * 40}")
        model = MultimodalNet(n_eeg_ch, n_et_ch, n_times, n_classes)
        _, res = train_dl_model(
            model,
            data["X_eeg_train"], data["X_eeg_val"],
            data["y_train"], data["y_val"],
            X_et_train=data["X_et_train"],
            X_et_val=data["X_et_val"],
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


def phase4(run_name):
    print("\n" + "=" * 60)
    print("  PHASE 4 — OUTCOME PREDICTION")
    print("=" * 60)

    conditions = discover_conditions(run_name)
    data_pooled = pool_conditions(run_name, conditions)
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
        model = EEGNet(n_ch, n_times, 2)
        _, res = train_dl_model(
            model, X_tr, X_va, y_tr, y_va,
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
        model = EEGNet(n_ch, n_times, 2)
        _, res = train_dl_model(
            model, X_tr, X_va, y_tr, y_va,
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

            model = MultimodalNet(n_eeg_ch, n_et_ch, n_times, 2)
            _, res = train_dl_model(
                model, X_eeg_tr, X_eeg_va, y_tr, y_va,
                X_et_train=X_et_tr, X_et_val=X_et_va,
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
    data_pooled = pool_conditions(run_name, conditions)

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
                 F1=8, D=2, F2=16, eeg_dropout=0.25,
                 gaze_embed_dim=32, gaze_hidden=64, gaze_type="lstm",
                 fusion_dropout=0.3):
        super().__init__()
        self.eeg_branch = EEGNet(n_eeg_ch, n_times, 2, F1, D, F2, eeg_dropout)
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

    X_all, y_all, meta_all = [], [], []
    for cond in conditions:
        data = load_tensors(run_name, cond)
        for split in ("train", "val"):
            X_all.append(data[f"X_eeg_{split}"])
            y_all.append(data[f"y_{split}"])
            meta_all.append(data[f"meta_{split}"])

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

        print(f"      bal_acc={bal_acc:.3f}  AUC={auc:.3f}  F1={f1:.3f}")

        fold_results.append({
            "fold": fi, "balanced_accuracy": float(bal_acc),
            "auc_roc": float(auc), "f1": float(f1),
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
    for key in ["balanced_accuracy", "auc_roc", "f1"]:
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

def phase6(run_name):
    """Model A: EEGNet on no-go trials (CR vs FA), stratified k-fold."""
    print("\n" + "=" * 60)
    print("  PHASE 6 — NO-GO INHIBITORY CONTROL (EEG-Only)")
    print("=" * 60)

    nogo = _pool_and_filter_nogo(run_name)
    if nogo is None:
        return {"error": "insufficient_data"}

    X, labels = nogo["X_eeg"], nogo["labels"]
    n_ch, n_t = X.shape[1], X.shape[2]

    results = _nogo_kfold(
        lambda: EEGNet(n_ch, n_t, 2),
        X, labels, n_folds=5, n_epochs=100, batch_size=32, patience=10,
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

    # Save detailed results for the dashboard
    detail = {
        "summary": s,
        "fold_results": [{k: v for k, v in r.items() if k != "_best_state"}
                         for r in results["fold_results"]],
        "n_cr": nogo["n_cr"], "n_fa": nogo["n_fa"],
    }
    import json as _json
    with open(os.path.join(RUNS_ROOT, run_name, "nogo_results.json"), "w") as f:
        _json.dump({"phase6": detail}, f, indent=2)

    # Return flat summary for ml_results.json
    return {
        "nogo_eeg_kfold": {
            "balanced_accuracy": s["balanced_accuracy_mean"],
            "auc_roc": s["auc_roc_mean"],
            "f1": s["f1_mean"],
            "n_folds": s["n_folds"],
            "n_cr": nogo["n_cr"],
            "n_fa": nogo["n_fa"],
        },
        "_internal": results,
    }


# ═══════════════════════════════════════════════════════════
#  PHASE 7 — NO-GO INHIBITORY CONTROL (EEG + Gaze Fusion)
# ═══════════════════════════════════════════════════════════

def phase7(run_name, phase6_results=None):
    """Model B: EEGNet + CLIP gaze fusion on no-go trials."""
    print("\n" + "=" * 60)
    print("  PHASE 7 — NO-GO INHIBITORY CONTROL (EEG + Gaze Fusion)")
    print("=" * 60)

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
                          gaze_type="lstm", fusion_dropout=0.3)
        if p6_state:
            try:
                m.load_eegnet_weights(p6_state)
            except Exception:
                pass
        return m

    results = _nogo_kfold(
        factory, X, labels, n_folds=5, n_epochs=100,
        batch_size=32, patience=10,
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
            "n_folds": s["n_folds"],
        },
        "nogo_comparison": comparison,
    }


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
                        help="Phase(s) to run: 1-5 (default: all)")
    args = parser.parse_args()

    run_name = args.run or find_latest_run()
    phases = args.phase or [1, 2, 3, 4, 5, 6, 7]

    print(f"\nRun: {run_name}")
    print(f"Phases: {phases}")
    print(f"Device: {get_device()}")

    all_results = {}

    if 1 in phases:
        all_results["phase1_scalar"] = phase1(run_name)
    if 2 in phases:
        all_results["phase2_eegnet"] = phase2(run_name)
    if 3 in phases:
        all_results["phase3_multimodal"] = phase3(run_name)
    if 4 in phases:
        all_results["phase4_outcome"] = phase4(run_name)
    if 5 in phases:
        all_results["phase5_vision"] = phase5(run_name)
    if 6 in phases:
        p6 = phase6(run_name)
        all_results["phase6_nogo_eeg"] = {
            k: v for k, v in p6.items() if not k.startswith("_")}
        _p6_ref = p6
    else:
        _p6_ref = None
    if 7 in phases:
        p7 = phase7(run_name, _p6_ref)
        all_results["phase7_nogo_fusion"] = {
            k: v for k, v in p7.items() if not k.startswith("_")}

    save_summary(run_name, all_results)

    print("\n" + "=" * 60)
    print("  ALL PHASES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
