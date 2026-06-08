"""Vision model experiments: class balance audit, learning curves, compute benchmark.

Usage:
    python src/vision/experiments.py --experiment 0        # balance audit
    python src/vision/experiments.py --experiment 1        # learning curve
    python src/vision/experiments.py --experiment 2        # compute benchmark
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

from vision.config import CATEGORIES, LABEL_MERGE_MAP, CATEGORY_COLORS
from vision.label_store import load_trainable_labels, get_crop_path, PROJECT_ROOT
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


def _load_all_data():
    """Load all trainable labels and build arrays for experiments."""
    df = load_trainable_labels()
    if df.empty:
        return df, np.array([])
    df = df[df["human_label"].isin(LABEL_TO_IDX)].reset_index(drop=True)
    y = np.array([LABEL_TO_IDX[l] for l in df["human_label"]])
    return df, y


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
    """Merge classes with < min_count samples into 'other'."""
    y_out = y.copy()
    other_idx = LABEL_TO_IDX["other"]
    counts = np.bincount(y_out, minlength=N_CLASSES)
    for cls_idx in range(N_CLASSES):
        if cls_idx != other_idx and counts[cls_idx] < min_count:
            y_out[y_out == cls_idx] = other_idx
    return y_out


def _train_val_test_split(y, seed):
    """Stratified 70/15/15 split. Returns (train_idx, val_idx, test_idx)."""
    y_strat = _merge_rare_classes(y)

    unique, counts = np.unique(y_strat, return_counts=True)
    if len(unique) < 2 or any(c < 3 for c in counts):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(y))
        n = len(y)
        n_train = int(n * 0.7)
        n_val = int(n * 0.15)
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(len(y)), y_strat))

    y_temp = y_strat[temp_idx]
    unique_temp, counts_temp = np.unique(y_temp, return_counts=True)
    if len(unique_temp) < 2 or any(c < 2 for c in counts_temp):
        mid = len(temp_idx) // 2
        return train_idx, temp_idx[:mid], temp_idx[mid:]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(sss2.split(np.zeros(len(temp_idx)), y_temp))
    return train_idx, temp_idx[val_rel], temp_idx[test_rel]


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

def experiment_0_balance_audit(labels_df, y, n_epochs_resnet=30):
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
        cm = metrics.get("confusion_matrix")
        labels = metrics.get("label_order", LABEL_NAMES)
        if cm is None:
            ax.text(0.5, 0.5, f"{model_name}\n(no confusion matrix)",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        cm_arr = np.array(cm)
        im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm_arr[i, j]),
                        ha="center", va="center", fontsize=8,
                        color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")

    plt.tight_layout()
    path = os.path.join(EXPERIMENTS_DIR, "balance_confusion.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ═══════════════════════════════════════════════════════════════
# Experiment 1: Learning Curve
# ═══════════════════════════════════════════════════════════════

def experiment_1_learning_curve(labels_df, y, n_seeds=3, n_epochs_resnet=30):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Learning Curve (ResNet-50)")
    print("=" * 60)

    N = len(y)

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

    print(f"  Total labels: {N}")
    print(f"  Sample points: {sample_points}")
    print(f"  Seeds per point: {n_seeds}")

    all_results = []

    for n_samples in sample_points:
        print(f"\n--- n={n_samples} ---")
        for seed in range(n_seeds):
            print(f"  seed {seed}:")
            result_row = {"n_samples": n_samples, "seed": seed}

            if _HAS_TORCHVISION and n_samples >= 50:
                sub_idx = _stratified_subsample(y, n_samples, seed=seed * 1000 + n_samples + 1)
                y_sub = y[sub_idx]

                if len(np.unique(y_sub)) < 2:
                    result_row["resnet_balanced_acc"] = None
                    result_row["resnet_accuracy"] = None
                else:
                    train_i, val_i, test_i = _train_val_test_split(y_sub, seed)
                    resnet_metrics = _train_resnet_subset(
                        labels_df.iloc[sub_idx].reset_index(drop=True),
                        train_i, val_i, test_i,
                        n_epochs=n_epochs_resnet, progress=False,
                    )
                    result_row["resnet_balanced_acc"] = resnet_metrics.get("balanced_accuracy")
                    result_row["resnet_accuracy"] = resnet_metrics.get("accuracy")
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
        col = f"resnet_{metric}"
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
            label="ResNet-50",
            color="#FF5722", marker="s", capsize=4, linewidth=2, markersize=6,
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
# Experiment 2: Compute Benchmark
# ═══════════════════════════════════════════════════════════════

def experiment_2_compute_benchmark(batch_size=32, n_warmup=3, n_runs=10):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Compute Benchmark (ResNet-50)")
    print("=" * 60)

    device = _get_device()
    print(f"  Device: {device}")

    results = {"device": str(device), "batch_size": batch_size, "models": {}}
    dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)

    if _HAS_TORCHVISION:
        from torchvision.models import resnet50
        resnet_model = resnet50(num_classes=N_CLASSES).to(device).eval()

        for _ in range(n_warmup):
            with torch.no_grad():
                resnet_model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                resnet_model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_time = np.mean(times)
        results["models"]["resnet50"] = {
            "avg_ms_per_batch": round(avg_time * 1000, 1),
            "throughput_img_per_sec": round(batch_size / avg_time, 0),
        }
        print(f"  ResNet-50:  {avg_time*1000:.1f} ms/batch  "
              f"({batch_size / avg_time:.0f} img/s)")
        del resnet_model

    _save_json(results, os.path.join(EXPERIMENTS_DIR, "compute_benchmark.json"))
    return results


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Vision model experiments: balance, learning curve, benchmark"
    )
    parser.add_argument("--experiment", type=int, nargs="+", default=None,
                        help="Which experiments to run: 0=balance, "
                             "1=learning_curve, 2=compute_benchmark")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--seeds", type=int, default=3, help="Seeds for learning curve")
    parser.add_argument("--epochs-resnet", type=int, default=30, help="Epochs for ResNet")
    args = parser.parse_args()

    if args.experiment is None and not args.all:
        parser.print_help()
        print("\nExample: python src/vision/experiments.py --experiment 0")
        return

    experiments = set(args.experiment or [])
    if args.all:
        experiments = {0, 1, 2}

    t0 = time.time()
    print(f"Experiments to run: {sorted(experiments)}")

    labels_df, y = _load_all_data()
    if labels_df.empty:
        print("No trainable labels found — cannot run experiments.")
        return
    print(f"Loaded {len(labels_df)} labels ({N_CLASSES} classes)")

    if 0 in experiments:
        experiment_0_balance_audit(labels_df, y, n_epochs_resnet=args.epochs_resnet)

    if 1 in experiments:
        experiment_1_learning_curve(labels_df, y,
                                    n_seeds=args.seeds,
                                    n_epochs_resnet=args.epochs_resnet)

    if 2 in experiments:
        experiment_2_compute_benchmark()

    elapsed = time.time() - t0
    print(f"\nAll experiments completed in {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
