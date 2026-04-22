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
    accuracy_score, f1_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
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
    phases = args.phase or [1, 2, 3, 4, 5]

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

    save_summary(run_name, all_results)

    print("\n" + "=" * 60)
    print("  ALL PHASES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
