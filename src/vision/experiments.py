"""Vision model experiments: class balance, learning curves, CLIP fine-tuning, compute.

Portable script — auto-detects available label count and adapts sample
points for the learning curve.  Run on this machine or the remote one
with 8k+ labels.

Usage:
    python src/vision/experiments.py --experiment 0        # balance audit
    python src/vision/experiments.py --experiment 1        # learning curve
    python src/vision/experiments.py --experiment 2        # CLIP backbone fine-tune
    python src/vision/experiments.py --experiment 3        # compute benchmark
    python src/vision/experiments.py --all                 # everything
    python src/vision/experiments.py --experiment 1 --seeds 5 --epochs 300
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedShuffleSplit

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.config import CATEGORIES, LABEL_MERGE_MAP, CLIP_MODEL, CATEGORY_COLORS
from vision.label_store import load_trainable_labels, get_crop_path, PROJECT_ROOT
from vision.train_head import LinearHead, split_within_subject
from vision.resnet_head import (
    build_resnet50,
    train_resnet,
    _build_records,
    CropDataset,
    _get_transforms,
    _HAS_TORCHVISION,
)

EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "models", "experiments")
LABEL_NAMES = list(CATEGORIES.keys())
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}
N_CLASSES = len(LABEL_NAMES)


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {path}")


# ── Data loading ─────────────────────────────────────────────────

def _load_all_data():
    """Load all trainable labels and build arrays for experiments.

    Returns (labels_df, y_array) where y_array has class indices.
    Labels not in CATEGORIES are dropped.
    """
    df = load_trainable_labels()
    if df.empty:
        return df, np.array([])
    df = df[df["human_label"].isin(LABEL_TO_IDX)].reset_index(drop=True)
    y = np.array([LABEL_TO_IDX[l] for l in df["human_label"]])
    return df, y


def _load_clip_embeddings(labels_df):
    """Try to find CLIP embeddings that match the labels.

    Looks in run directories for embeddings files, matching by
    subject + condition. Returns (X_emb, matched_indices) where
    X_emb is (N_matched, 512) and matched_indices are row indices
    into labels_df.
    """
    runs_dir = os.path.join(PROJECT_ROOT, "runs")
    if not os.path.isdir(runs_dir):
        return None, None

    run_dirs = sorted(os.listdir(runs_dir), reverse=True)

    fid_to_row = {}
    for idx, row in labels_df.iterrows():
        key = (int(row["subject_id"]), row["condition"], int(row["fixation_id"]))
        fid_to_row[key] = idx

    X_list = []
    matched_idx = []

    for sj_cond, group in labels_df.groupby(["subject_id", "condition"]):
        sj_num, condition = int(sj_cond[0]), sj_cond[1]
        prefix = f"sj{sj_num:02d}_{condition}"

        emb_npy = None
        emb_ids = None
        for rd in run_dirs:
            vis_dir = os.path.join(runs_dir, rd, "vision", prefix)
            candidate_npy = os.path.join(vis_dir, f"{prefix}_embeddings.npy")
            candidate_ids = os.path.join(vis_dir, f"{prefix}_embeddings_ids.csv")
            if os.path.exists(candidate_npy) and os.path.exists(candidate_ids):
                emb_npy = candidate_npy
                emb_ids = candidate_ids
                break

        if emb_npy is None:
            continue

        embs = np.load(emb_npy)
        ids_df = pd.read_csv(emb_ids)
        fid_to_emb_idx = {int(fid): i for i, fid in enumerate(ids_df["fixation_id"])}

        for _, row in group.iterrows():
            fid = int(row["fixation_id"])
            if fid in fid_to_emb_idx:
                row_idx = fid_to_row.get((sj_num, condition, fid))
                if row_idx is not None:
                    X_list.append(embs[fid_to_emb_idx[fid]])
                    matched_idx.append(row_idx)

    if not X_list:
        return None, None

    return np.stack(X_list).astype(np.float32), np.array(matched_idx)


def _stratified_subsample(y, n_samples, seed):
    """Return indices for a stratified subsample of size n_samples."""
    if n_samples >= len(y):
        return np.arange(len(y))
    y_strat = _merge_rare_classes(y)
    frac = n_samples / len(y)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
        idx, _ = next(sss.split(np.zeros(len(y)), y_strat))
    except ValueError:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y), size=n_samples, replace=False)
    return idx


def _merge_rare_classes(y, min_count=5):
    """Merge classes with < min_count samples into 'other' (index 5).

    Returns a copy of y with rare classes remapped.
    """
    y_out = y.copy()
    other_idx = LABEL_TO_IDX["other"]
    counts = np.bincount(y_out, minlength=N_CLASSES)
    for cls_idx in range(N_CLASSES):
        if cls_idx != other_idx and counts[cls_idx] < min_count:
            y_out[y_out == cls_idx] = other_idx
    return y_out


def _train_val_test_split(y, seed):
    """Stratified 70/15/15 split. Returns (train_idx, val_idx, test_idx).

    Rare classes are temporarily merged for stratification, but original
    labels are preserved in the returned indices.
    """
    y_strat = _merge_rare_classes(y)

    unique, counts = np.unique(y_strat, return_counts=True)
    if len(unique) < 2 or counts.min() < 2:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(y))
        n_test = max(1, int(0.15 * len(y)))
        n_val = max(1, int(0.15 * len(y)))
        return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(np.zeros(len(y)), y_strat))

    val_frac = 0.15 / 0.85
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    rel_train, rel_val = next(sss_val.split(
        np.zeros(len(train_val_idx)), y_strat[train_val_idx]
    ))
    train_idx = train_val_idx[rel_train]
    val_idx = train_val_idx[rel_val]
    return train_idx, val_idx, test_idx


# ── CLIP linear head training (for experiments) ─────────────────

def _train_clip_head(X_train, y_train, X_val, y_val, n_epochs=200, lr=0.01, device=None):
    """Train a CLIP linear head. Returns (model_cpu, best_val_acc)."""
    if device is None:
        device = _get_device()

    model = LinearHead(N_CLASSES).to(device)
    X_tr = torch.from_numpy(X_train).to(device)
    y_tr = torch.from_numpy(y_train).long().to(device)
    X_v = torch.from_numpy(X_val).to(device)
    y_v = torch.from_numpy(y_val).long().to(device)

    class_counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * N_CLASSES
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_acc = 0.0
    best_state = None

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_acc = (model(X_v).argmax(1) == y_v).float().mean().item()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()

    if best_state:
        model.load_state_dict(best_state)
    return model.cpu().eval(), best_val_acc


def _eval_model_metrics(model, X_test, y_test, device=None):
    """Evaluate a CLIP head model. Returns metrics dict."""
    if device is None:
        device = _get_device()
    model = model.to(device).eval()
    X_t = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        preds = model(X_t).argmax(1).cpu().numpy()
    model.cpu()

    present = sorted(set(y_test.tolist()))
    present_names = [LABEL_NAMES[c] for c in present]

    bal_acc = balanced_accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds, labels=present, target_names=present_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=present).tolist()

    return {
        "balanced_accuracy": round(float(bal_acc), 4),
        "accuracy": round(float((preds == y_test).mean()), 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "label_order": present_names,
    }


# ── ResNet training (for experiments) ────────────────────────────

def _train_resnet_subset(labels_df, train_idx, val_idx, test_idx,
                         n_epochs=30, lr=1e-4, batch_size=32, progress=True):
    """Train ResNet-50 on a subset. Returns metrics dict."""
    if not _HAS_TORCHVISION:
        return {"error": "torchvision not installed"}

    train_records = _build_records(labels_df.iloc[train_idx], LABEL_TO_IDX)
    val_records = _build_records(labels_df.iloc[val_idx], LABEL_TO_IDX)
    test_records = _build_records(labels_df.iloc[test_idx], LABEL_TO_IDX)

    if len(train_records) < 10 or len(val_records) < 5:
        return {"error": "not enough data for ResNet"}

    cb = None
    if progress:
        def cb(epoch, n_ep, metrics):
            if epoch % 5 == 0 or epoch == n_ep:
                print(f"    ResNet epoch {epoch}/{n_ep} "
                      f"val_acc={metrics['val_acc']:.3f}")

    model, stats = train_resnet(
        train_records, val_records, LABEL_NAMES,
        n_epochs=n_epochs, lr=lr, batch_size=batch_size,
        progress_cb=cb,
    )

    if not test_records:
        return {
            "balanced_accuracy": None,
            "accuracy": stats.get("best_val_acc"),
            "note": "no test records",
        }

    device = _get_device()
    _, val_tfm = _get_transforms()
    test_ds = CropDataset(test_records, transform=val_tfm)
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device).eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(labels.numpy())
    model.cpu()

    test_true = np.array(test_true)
    test_preds = np.array(test_preds)
    present = sorted(set(test_true.tolist()))
    present_names = [LABEL_NAMES[c] for c in present]

    bal_acc = balanced_accuracy_score(test_true, test_preds)
    report = classification_report(
        test_true, test_preds, labels=present, target_names=present_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(test_true, test_preds, labels=present).tolist()

    return {
        "balanced_accuracy": round(float(bal_acc), 4),
        "accuracy": round(float((test_preds == test_true).mean()), 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "label_order": present_names,
    }


# ═══════════════════════════════════════════════════════════════
# Experiment 0: Class Balance Audit
# ═══════════════════════════════════════════════════════════════

def experiment_0_balance_audit(labels_df, y, X_emb, emb_idx, n_epochs_clip=200, n_epochs_resnet=30):
    print("\n" + "=" * 60)
    print("EXPERIMENT 0: Class Balance Audit")
    print("=" * 60)

    counts = pd.Series(y).value_counts().sort_index()
    total = len(y)
    print(f"\nTotal labels: {total}")
    print(f"\nClass distribution:")
    for idx, count in counts.items():
        name = LABEL_NAMES[idx]
        pct = count / total * 100
        print(f"  {name:15s}  {count:5d}  ({pct:5.1f}%)")

    imbalance_ratio = counts.max() / max(counts.min(), 1)
    print(f"\nImbalance ratio (max/min): {imbalance_ratio:.1f}x")

    results = {
        "total_labels": total,
        "class_distribution": {LABEL_NAMES[i]: int(c) for i, c in counts.items()},
        "imbalance_ratio": round(float(imbalance_ratio), 1),
        "models": {},
    }

    seed = 42

    # CLIP linear head evaluation
    if X_emb is not None:
        print(f"\n--- CLIP Linear Head (on {len(emb_idx)} matched embeddings) ---")
        y_emb = y[emb_idx]
        train_idx, val_idx, test_idx = _train_val_test_split(y_emb, seed)
        model, val_acc = _train_clip_head(
            X_emb[train_idx], y_emb[train_idx],
            X_emb[val_idx], y_emb[val_idx],
            n_epochs=n_epochs_clip,
        )
        clip_metrics = _eval_model_metrics(model, X_emb[test_idx], y_emb[test_idx])
        clip_metrics["n_train"] = len(train_idx)
        clip_metrics["n_val"] = len(val_idx)
        clip_metrics["n_test"] = len(test_idx)
        clip_metrics["best_val_acc"] = round(val_acc, 4)
        results["models"]["clip_linear_head"] = clip_metrics
        print(f"  Balanced accuracy: {clip_metrics['balanced_accuracy']:.4f}")
        print(f"  Raw accuracy:      {clip_metrics['accuracy']:.4f}")
        _print_per_class(clip_metrics)
    else:
        print("\n  No CLIP embeddings found — skipping CLIP head evaluation")

    # ResNet-50 evaluation
    if _HAS_TORCHVISION and total >= 50:
        print(f"\n--- ResNet-50 (on {total} labeled crops) ---")
        train_idx, val_idx, test_idx = _train_val_test_split(y, seed)
        resnet_metrics = _train_resnet_subset(
            labels_df, train_idx, val_idx, test_idx,
            n_epochs=n_epochs_resnet,
        )
        results["models"]["resnet50"] = resnet_metrics
        if "balanced_accuracy" in resnet_metrics and resnet_metrics["balanced_accuracy"]:
            print(f"  Balanced accuracy: {resnet_metrics['balanced_accuracy']:.4f}")
            print(f"  Raw accuracy:      {resnet_metrics['accuracy']:.4f}")
            _print_per_class(resnet_metrics)
    else:
        print("\n  Skipping ResNet (torchvision missing or too few labels)")

    # Save results + confusion matrix plot
    _save_json(results, os.path.join(EXPERIMENTS_DIR, "balance_audit.json"))
    _plot_confusion_matrices(results)

    return results


def _print_per_class(metrics):
    report = metrics.get("classification_report", {})
    print(f"  {'class':15s} {'precision':>9s} {'recall':>9s} {'f1':>9s} {'support':>9s}")
    for cls_name in LABEL_NAMES:
        if cls_name in report:
            r = report[cls_name]
            print(f"  {cls_name:15s} {r['precision']:9.3f} {r['recall']:9.3f} "
                  f"{r['f1-score']:9.3f} {r['support']:9.0f}")


def _plot_confusion_matrices(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    models = results.get("models", {})
    n_models = len(models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, metrics) in zip(axes, models.items()):
        cm = np.array(metrics.get("confusion_matrix", []))
        labels = metrics.get("label_order", [])
        if cm.size == 0:
            continue
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        bal_acc = metrics.get("balanced_accuracy", "?")
        ax.set_title(f"{model_name}\nBalanced Acc: {bal_acc}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=color, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(EXPERIMENTS_DIR, "balance_confusion.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# Experiment 1: Learning Curve
# ═══════════════════════════════════════════════════════════════

def experiment_1_learning_curve(labels_df, y, X_emb, emb_idx,
                                n_seeds=3, n_epochs_clip=200, n_epochs_resnet=30):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Learning Curve")
    print("=" * 60)

    N = len(y)
    N_emb = len(emb_idx) if emb_idx is not None else 0

    if N >= 8000:
        sample_points = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    elif N >= 500:
        step = max(N // 6, 100)
        sample_points = list(range(step, N + 1, step))
        if sample_points[-1] != N:
            sample_points.append(N)
    else:
        print(f"  Only {N} labels — need at least 500 for a meaningful learning curve")
        return None

    print(f"  Total labels: {N} (CLIP embeddings: {N_emb})")
    print(f"  Sample points: {sample_points}")
    print(f"  Seeds per point: {n_seeds}")

    all_results = []
    device = _get_device()

    for n_samples in sample_points:
        print(f"\n--- n={n_samples} ---")
        for seed in range(n_seeds):
            print(f"  seed {seed}:")
            result_row = {"n_samples": n_samples, "seed": seed}

            # CLIP linear head
            if X_emb is not None:
                clip_n = min(n_samples, N_emb)
                sub_idx = _stratified_subsample(y[emb_idx], clip_n, seed=seed * 1000 + n_samples)
                y_sub = y[emb_idx][sub_idx]
                X_sub = X_emb[sub_idx]

                if len(np.unique(y_sub)) < 2:
                    result_row["clip_balanced_acc"] = None
                    result_row["clip_accuracy"] = None
                else:
                    train_i, val_i, test_i = _train_val_test_split(y_sub, seed)
                    model, _ = _train_clip_head(
                        X_sub[train_i], y_sub[train_i],
                        X_sub[val_i], y_sub[val_i],
                        n_epochs=n_epochs_clip, device=device,
                    )
                    metrics = _eval_model_metrics(model, X_sub[test_i], y_sub[test_i], device)
                    result_row["clip_balanced_acc"] = metrics["balanced_accuracy"]
                    result_row["clip_accuracy"] = metrics["accuracy"]
                    result_row["clip_report"] = metrics["classification_report"]
                    print(f"    CLIP  bal_acc={metrics['balanced_accuracy']:.3f}  "
                          f"acc={metrics['accuracy']:.3f}")

            # ResNet-50
            if _HAS_TORCHVISION and n_samples >= 50:
                resnet_n = min(n_samples, N)
                sub_idx_r = _stratified_subsample(y, resnet_n, seed=seed * 1000 + n_samples + 1)
                y_sub_r = y[sub_idx_r]

                if len(np.unique(y_sub_r)) < 2:
                    result_row["resnet_balanced_acc"] = None
                    result_row["resnet_accuracy"] = None
                else:
                    train_i, val_i, test_i = _train_val_test_split(y_sub_r, seed)
                    resnet_metrics = _train_resnet_subset(
                        labels_df.iloc[sub_idx_r].reset_index(drop=True),
                        train_i, val_i, test_i,
                        n_epochs=n_epochs_resnet, progress=False,
                    )
                    result_row["resnet_balanced_acc"] = resnet_metrics.get("balanced_accuracy")
                    result_row["resnet_accuracy"] = resnet_metrics.get("accuracy")
                    result_row["resnet_report"] = resnet_metrics.get("classification_report")
                    if resnet_metrics.get("balanced_accuracy"):
                        print(f"    RN50  bal_acc={resnet_metrics['balanced_accuracy']:.3f}  "
                              f"acc={resnet_metrics['accuracy']:.3f}")

            all_results.append(result_row)

    _save_json(all_results, os.path.join(EXPERIMENTS_DIR, "learning_curve.json"))
    _plot_learning_curve(all_results)

    return all_results


def _plot_learning_curve(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot")
        return

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric, title in [
        (axes[0], "balanced_acc", "Balanced Accuracy"),
        (axes[1], "accuracy", "Raw Accuracy"),
    ]:
        for model, color, marker in [
            ("clip", "#2196F3", "o"),
            ("resnet", "#FF5722", "s"),
        ]:
            col = f"{model}_{metric}"
            if col not in df.columns:
                continue
            valid = df[df[col].notna()]
            if valid.empty:
                continue
            grouped = valid.groupby("n_samples")[col]
            means = grouped.mean()
            stds = grouped.std().fillna(0)

            ax.errorbar(
                means.index, means.values, yerr=stds.values,
                label=model.upper().replace("CLIP", "CLIP Linear").replace("RESNET", "ResNet-50"),
                color=color, marker=marker, capsize=4, linewidth=2, markersize=6,
            )

        ax.set_xlabel("Number of Training Labels")
        ax.set_ylabel(title)
        ax.set_title(f"Learning Curve: {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(EXPERIMENTS_DIR, "learning_curve.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# Experiment 2: CLIP Backbone Fine-Tuning
# ═══════════════════════════════════════════════════════════════

class MLPHead(nn.Module):
    """2-layer MLP: 512 -> 256 -> N with dropout + ReLU."""
    def __init__(self, n_classes, embed_dim=512, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CLIPFineTuner(nn.Module):
    """CLIP ViT-B/32 with optionally unfrozen transformer blocks + classification head."""

    def __init__(self, n_classes, unfreeze_blocks=0, head_type="linear"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(CLIP_MODEL, device="cpu")
        self.clip_model = self.clip_model.float()

        for param in self.clip_model.parameters():
            param.requires_grad = False

        if unfreeze_blocks > 0:
            visual = self.clip_model.visual
            if hasattr(visual, "transformer"):
                blocks = visual.transformer.resblocks
                for block in list(blocks)[-unfreeze_blocks:]:
                    for param in block.parameters():
                        param.requires_grad = True
            if hasattr(visual, "ln_post"):
                for param in visual.ln_post.parameters():
                    param.requires_grad = True
            if hasattr(visual, "proj") and visual.proj is not None:
                visual.proj.requires_grad = True

        embed_dim = 512
        if head_type == "mlp":
            self.head = MLPHead(n_classes, embed_dim)
        else:
            self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, images):
        with torch.set_grad_enabled(self.training):
            features = self.clip_model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)
        return self.head(features.float())

    def backbone_params(self):
        return [p for p in self.clip_model.parameters() if p.requires_grad]

    def head_params(self):
        return list(self.head.parameters())


def _train_clip_finetuner(model, labels_df, train_idx, val_idx,
                          n_epochs=15, backbone_lr=1e-5, head_lr=1e-3,
                          batch_size=32, device=None):
    """Train a CLIPFineTuner on crop images. Returns (model_cpu, best_val_acc)."""
    if device is None:
        device = _get_device()

    from torch.utils.data import DataLoader

    train_records = _build_records(labels_df.iloc[train_idx], LABEL_TO_IDX)
    val_records = _build_records(labels_df.iloc[val_idx], LABEL_TO_IDX)

    class _CLIPCropDataset(torch.utils.data.Dataset):
        def __init__(self, records, preprocess):
            self.records = records
            self.preprocess = preprocess

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            from PIL import Image
            r = self.records[idx]
            path = get_crop_path(r["sj_num"], r["condition"], r["filename"])
            try:
                img = Image.open(path).convert("RGB")
            except (OSError, FileNotFoundError):
                img = Image.new("RGB", (224, 224), (128, 128, 128))
            return self.preprocess(img), r["label_idx"]

    train_ds = _CLIPCropDataset(train_records, model.preprocess)
    val_ds = _CLIPCropDataset(val_records, model.preprocess)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)

    param_groups = []
    bb_params = model.backbone_params()
    if bb_params:
        param_groups.append({"params": bb_params, "lr": backbone_lr})
    param_groups.append({"params": model.head_params(), "lr": head_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    y_train = np.array([r["label_idx"] for r in train_records])
    class_counts = np.bincount(y_train, minlength=N_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * N_CLASSES
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    best_val_acc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total_val = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(1)
                correct += (preds == labels).sum().item()
                total_val += len(labels)
        val_acc = correct / max(total_val, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            print(f"    epoch {epoch+1}/{n_epochs}  val_acc={val_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    return model.cpu().eval(), best_val_acc


def experiment_2_clip_finetune(labels_df, y, X_emb, emb_idx,
                               n_seeds=3, n_epochs_frozen=200, n_epochs_finetune=15):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: CLIP Backbone Fine-Tuning Comparison")
    print("=" * 60)

    N = len(y)
    if N < 50:
        print(f"  Only {N} labels — need at least 50")
        return None

    conditions = [
        {"name": "frozen_linear", "unfreeze_blocks": 0, "head_type": "linear"},
        {"name": "frozen_mlp", "unfreeze_blocks": 0, "head_type": "mlp"},
        {"name": "finetune_2blocks", "unfreeze_blocks": 2, "head_type": "linear"},
    ]

    all_results = []

    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        train_idx, val_idx, test_idx = _train_val_test_split(y, seed)

        for cond in conditions:
            print(f"\n  Condition: {cond['name']}")

            if cond["name"] == "frozen_linear" and X_emb is not None:
                emb_train = [i for i in range(len(emb_idx)) if emb_idx[i] in train_idx]
                emb_val = [i for i in range(len(emb_idx)) if emb_idx[i] in val_idx]
                emb_test = [i for i in range(len(emb_idx)) if emb_idx[i] in test_idx]

                if len(emb_train) > 10 and len(emb_val) > 5 and len(emb_test) > 5:
                    model, _ = _train_clip_head(
                        X_emb[emb_train], y[emb_idx[emb_train]],
                        X_emb[emb_val], y[emb_idx[emb_val]],
                        n_epochs=n_epochs_frozen,
                    )
                    metrics = _eval_model_metrics(
                        model, X_emb[emb_test], y[emb_idx[emb_test]]
                    )
                    all_results.append({
                        "condition": cond["name"],
                        "seed": seed,
                        **metrics,
                    })
                    print(f"    bal_acc={metrics['balanced_accuracy']:.3f}")
                    continue

            if cond["name"] == "frozen_mlp" and X_emb is not None:
                emb_train = [i for i in range(len(emb_idx)) if emb_idx[i] in train_idx]
                emb_val = [i for i in range(len(emb_idx)) if emb_idx[i] in val_idx]
                emb_test = [i for i in range(len(emb_idx)) if emb_idx[i] in test_idx]

                if len(emb_train) > 10 and len(emb_val) > 5 and len(emb_test) > 5:
                    device = _get_device()
                    mlp = MLPHead(N_CLASSES).to(device)
                    X_tr = torch.from_numpy(X_emb[emb_train]).to(device)
                    y_tr = torch.from_numpy(y[emb_idx[emb_train]]).long().to(device)
                    X_v = torch.from_numpy(X_emb[emb_val]).to(device)
                    y_v = torch.from_numpy(y[emb_idx[emb_val]]).long().to(device)

                    class_counts = np.bincount(y[emb_idx[emb_train]], minlength=N_CLASSES).astype(np.float32)
                    class_counts = np.maximum(class_counts, 1.0)
                    w = 1.0 / class_counts
                    w = w / w.sum() * N_CLASSES
                    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(device))
                    opt = torch.optim.AdamW(mlp.parameters(), lr=0.01, weight_decay=1e-3)

                    best_va, best_st = 0.0, None
                    mlp.train()
                    for ep in range(n_epochs_frozen):
                        opt.zero_grad()
                        loss_fn(mlp(X_tr), y_tr).backward()
                        opt.step()
                        if (ep + 1) % 10 == 0:
                            mlp.eval()
                            with torch.no_grad():
                                va = (mlp(X_v).argmax(1) == y_v).float().mean().item()
                            if va > best_va:
                                best_va = va
                                best_st = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
                            mlp.train()

                    if best_st:
                        mlp.load_state_dict(best_st)
                    mlp = mlp.cpu().eval()

                    X_te = torch.from_numpy(X_emb[emb_test])
                    with torch.no_grad():
                        preds = mlp(X_te).argmax(1).numpy()
                    y_te = y[emb_idx[emb_test]]
                    present = sorted(set(y_te.tolist()))
                    present_names = [LABEL_NAMES[c] for c in present]
                    bal_acc = balanced_accuracy_score(y_te, preds)
                    report = classification_report(
                        y_te, preds, labels=present, target_names=present_names,
                        output_dict=True, zero_division=0,
                    )
                    all_results.append({
                        "condition": "frozen_mlp",
                        "seed": seed,
                        "balanced_accuracy": round(float(bal_acc), 4),
                        "accuracy": round(float((preds == y_te).mean()), 4),
                        "classification_report": report,
                    })
                    print(f"    bal_acc={bal_acc:.3f}")
                    continue

            # Fine-tune backbone (trains on raw crop images)
            try:
                import clip as _clip_check
            except ImportError:
                print("    clip package not installed — skipping")
                continue

            finetuner = CLIPFineTuner(
                N_CLASSES,
                unfreeze_blocks=cond["unfreeze_blocks"],
                head_type=cond["head_type"],
            )
            labels_subset = labels_df  # use full dataset with the split indices
            model, best_va = _train_clip_finetuner(
                finetuner, labels_subset, train_idx, val_idx,
                n_epochs=n_epochs_finetune,
            )

            # Evaluate on test set
            test_records = _build_records(labels_df.iloc[test_idx], LABEL_TO_IDX)
            if test_records:
                device = _get_device()

                class _CLIPTestDS(torch.utils.data.Dataset):
                    def __init__(self, records, preprocess):
                        self.records = records
                        self.preprocess = preprocess
                    def __len__(self):
                        return len(self.records)
                    def __getitem__(self, idx):
                        from PIL import Image
                        r = self.records[idx]
                        path = get_crop_path(r["sj_num"], r["condition"], r["filename"])
                        try:
                            img = Image.open(path).convert("RGB")
                        except (OSError, FileNotFoundError):
                            img = Image.new("RGB", (224, 224), (128, 128, 128))
                        return self.preprocess(img), r["label_idx"]

                from torch.utils.data import DataLoader
                test_ds = _CLIPTestDS(test_records, model.preprocess)
                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

                model = model.to(device).eval()
                test_preds, test_true = [], []
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs = imgs.to(device)
                        preds = model(imgs).argmax(1).cpu().numpy()
                        test_preds.extend(preds)
                        test_true.extend(labels.numpy())
                model.cpu()

                test_true = np.array(test_true)
                test_preds = np.array(test_preds)
                present = sorted(set(test_true.tolist()))
                present_names = [LABEL_NAMES[c] for c in present]
                bal_acc = balanced_accuracy_score(test_true, test_preds)
                report = classification_report(
                    test_true, test_preds, labels=present, target_names=present_names,
                    output_dict=True, zero_division=0,
                )
                all_results.append({
                    "condition": cond["name"],
                    "seed": seed,
                    "balanced_accuracy": round(float(bal_acc), 4),
                    "accuracy": round(float((test_preds == test_true).mean()), 4),
                    "classification_report": report,
                })
                print(f"    bal_acc={bal_acc:.3f}")

    _save_json(all_results, os.path.join(EXPERIMENTS_DIR, "finetune_comparison.json"))
    _plot_finetune_comparison(all_results)

    return all_results


def _plot_finetune_comparison(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df = pd.DataFrame(results)
    if df.empty or "balanced_accuracy" not in df.columns:
        return

    grouped = df.groupby("condition")["balanced_accuracy"]
    means = grouped.mean()
    stds = grouped.std().fillna(0)

    order = ["frozen_linear", "frozen_mlp", "finetune_2blocks"]
    order = [o for o in order if o in means.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2196F3", "#4CAF50", "#FF5722"]
    x = range(len(order))
    bars = ax.bar(x, [means[o] for o in order],
                  yerr=[stds[o] for o in order],
                  color=colors[:len(order)], capsize=6, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", "\n") for o in order])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("CLIP Fine-Tuning Comparison")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar, o in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{means[o]:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(EXPERIMENTS_DIR, "finetune_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# Experiment 3: Compute Benchmark
# ═══════════════════════════════════════════════════════════════

def experiment_3_compute_benchmark(batch_size=32, n_warmup=3, n_runs=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Compute Benchmark")
    print("=" * 60)

    device = _get_device()
    print(f"  Device: {device}")
    results = {"device": str(device), "batch_size": batch_size, "models": {}}

    dummy_input_clip = torch.randn(batch_size, 3, 224, 224).to(device)

    # CLIP ViT-B/32
    try:
        import clip
        clip_model, _ = clip.load(CLIP_MODEL, device=device)
        clip_model = clip_model.float().eval()

        for _ in range(n_warmup):
            with torch.no_grad():
                clip_model.encode_image(dummy_input_clip)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                clip_model.encode_image(dummy_input_clip)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        avg_time = np.mean(times)
        results["models"]["clip_vit_b32"] = {
            "avg_forward_ms": round(avg_time * 1000, 2),
            "throughput_img_per_sec": round(batch_size / avg_time, 1),
            "times_ms": [round(t * 1000, 2) for t in times],
        }
        print(f"  CLIP ViT-B/32:  {avg_time*1000:.1f} ms/batch  "
              f"({batch_size / avg_time:.0f} img/s)")
        del clip_model
    except Exception as e:
        print(f"  CLIP benchmark failed: {e}")

    # ResNet-50
    if _HAS_TORCHVISION:
        try:
            from torchvision import models
            resnet = models.resnet50(weights=None).to(device).eval()
            dummy_input_rn = torch.randn(batch_size, 3, 224, 224).to(device)

            for _ in range(n_warmup):
                with torch.no_grad():
                    resnet(dummy_input_rn)
            if device.type == "cuda":
                torch.cuda.synchronize()

            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                with torch.no_grad():
                    resnet(dummy_input_rn)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            avg_time = np.mean(times)
            results["models"]["resnet50"] = {
                "avg_forward_ms": round(avg_time * 1000, 2),
                "throughput_img_per_sec": round(batch_size / avg_time, 1),
                "times_ms": [round(t * 1000, 2) for t in times],
            }
            print(f"  ResNet-50:      {avg_time*1000:.1f} ms/batch  "
                  f"({batch_size / avg_time:.0f} img/s)")
            del resnet
        except Exception as e:
            print(f"  ResNet benchmark failed: {e}")

    # Approximate FLOPs
    results["approximate_flops"] = {
        "clip_vit_b32": "~4.4 GFLOPs (ViT-B/32, 224x224)",
        "resnet50": "~4.1 GFLOPs (ResNet-50, 224x224)",
        "note": "CLIP is ~1.07x ResNet-50 in FLOPs but uses attention (memory-heavier)",
    }

    _save_json(results, os.path.join(EXPERIMENTS_DIR, "compute_benchmark.json"))
    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Vision model experiments: balance, learning curve, fine-tune, benchmark"
    )
    parser.add_argument("--experiment", type=int, nargs="+", default=None,
                        help="Experiment number(s) to run: 0=balance, 1=learning_curve, "
                             "2=clip_finetune, 3=compute_benchmark")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds per sample point (default: 3)")
    parser.add_argument("--epochs-clip", type=int, default=200, help="Epochs for CLIP head (default: 200)")
    parser.add_argument("--epochs-resnet", type=int, default=30, help="Epochs for ResNet (default: 30)")
    parser.add_argument("--epochs-finetune", type=int, default=15, help="Epochs for CLIP fine-tune (default: 15)")
    args = parser.parse_args()

    if args.experiment is None and not args.all:
        parser.print_help()
        print("\nExample: python src/vision/experiments.py --experiment 0")
        return

    experiments = set(args.experiment or [])
    if args.all:
        experiments = {0, 1, 2, 3}

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    print(f"Experiments to run: {sorted(experiments)}")
    print(f"Output directory: {EXPERIMENTS_DIR}")

    labels_df, y = _load_all_data()
    if len(y) == 0:
        print("ERROR: No trainable labels found in human_labels.csv")
        return

    print(f"Loaded {len(y)} labels across {len(labels_df['subject_id'].unique())} subject(s)")

    X_emb, emb_idx = None, None
    if 0 in experiments or 1 in experiments or 2 in experiments:
        print("Loading CLIP embeddings...")
        X_emb, emb_idx = _load_clip_embeddings(labels_df)
        if X_emb is not None:
            print(f"  Matched {len(emb_idx)} labels to CLIP embeddings")
        else:
            print("  No CLIP embeddings found — CLIP head experiments will be skipped")

    start = time.time()

    if 0 in experiments:
        experiment_0_balance_audit(labels_df, y, X_emb, emb_idx,
                                   n_epochs_clip=args.epochs_clip,
                                   n_epochs_resnet=args.epochs_resnet)

    if 1 in experiments:
        experiment_1_learning_curve(labels_df, y, X_emb, emb_idx,
                                    n_seeds=args.seeds,
                                    n_epochs_clip=args.epochs_clip,
                                    n_epochs_resnet=args.epochs_resnet)

    if 2 in experiments:
        experiment_2_clip_finetune(labels_df, y, X_emb, emb_idx,
                                   n_seeds=args.seeds,
                                   n_epochs_frozen=args.epochs_clip,
                                   n_epochs_finetune=args.epochs_finetune)

    if 3 in experiments:
        experiment_3_compute_benchmark()

    elapsed = time.time() - start
    print(f"\nAll experiments completed in {elapsed / 60:.1f} minutes")
    print(f"Results saved to {EXPERIMENTS_DIR}/")


if __name__ == "__main__":
    main()
