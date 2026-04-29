"""Train a linear classification head on frozen CLIP embeddings.

Takes human-labeled crops + pre-extracted CLIP embeddings and trains a
512 → N_CATEGORIES linear layer.  Saves the trained weights as a .pt
file that GazeClassifier can load for inference.

Split strategies
----------------
within_subject  : stratified 70/15/15 random split (default)
cross_subject   : train on all-but-one subject, test on held-out subject
                  (requires labels from ≥2 subjects)

Can run standalone:  python src/vision/train_head.py
Or called from vision_main.py as part of the pipeline.
"""

import os
import sys
from datetime import datetime
from typing import Optional, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.config import CATEGORIES
from vision.label_store import load_trainable_labels, load_labels_for, PROJECT_ROOT

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


# ── Data loading ──────────────────────────────────────────────

def _load_labeled_embeddings(labels_csv, embeddings_npy, embeddings_ids_csv):
    """Match human labels to their CLIP embedding vectors.

    Returns (embeddings array [N, 512], labels list [N], label_names list).
    """
    labels_df = pd.read_csv(labels_csv)
    embs = np.load(embeddings_npy)
    ids_df = pd.read_csv(embeddings_ids_csv)

    fid_to_emb_idx = {int(fid): i for i, fid in
                       enumerate(ids_df["fixation_id"].values)}

    label_names = list(CATEGORIES.keys())
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    X_list, y_list = [], []
    n_skipped = 0
    for _, row in labels_df.iterrows():
        fid = int(row["fixation_id"])
        lbl = row["human_label"]
        if fid not in fid_to_emb_idx or lbl not in label_to_idx:
            n_skipped += 1
            continue
        X_list.append(embs[fid_to_emb_idx[fid]])
        y_list.append(label_to_idx[lbl])

    if n_skipped:
        print(f"    Skipped {n_skipped} labels (missing embedding or unknown category)")

    X = np.stack(X_list).astype(np.float32)
    return X, y_list, label_names


# ── Model ─────────────────────────────────────────────────────

class LinearHead(nn.Module):
    """Single linear layer: 512-dim CLIP embedding → N categories."""

    def __init__(self, n_classes, embed_dim=512):
        super().__init__()
        self.linear = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


# ── Splits ────────────────────────────────────────────────────

def split_within_subject(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Stratified random 70/15/15 split.

    Returns dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
    train_idx, val_idx, test_idx
    """
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=random_state)
    train_val_idx, test_idx = next(sss_test.split(X, y))

    val_frac = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac,
                                     random_state=random_state)
    rel_train_idx, rel_val_idx = next(
        sss_val.split(X[train_val_idx], y[train_val_idx])
    )
    train_idx = train_val_idx[rel_train_idx]
    val_idx = train_val_idx[rel_val_idx]

    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val": X[val_idx], "y_val": y[val_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "strategy": "within_subject",
    }


def split_cross_subject(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    test_subject: Optional[int] = None,
) -> dict:
    """Leave-one-subject-out split.

    If test_subject is None, the last subject (by ID) is used as test.
    Returns same keys as split_within_subject, plus 'test_subject'.
    """
    unique_subjects = sorted(np.unique(subject_ids))
    if len(unique_subjects) < 2:
        raise ValueError(
            f"Cross-subject split requires labels from ≥2 subjects "
            f"(found {len(unique_subjects)})"
        )

    if test_subject is None:
        test_subject = unique_subjects[-1]

    test_mask = subject_ids == test_subject
    train_val_mask = ~test_mask

    train_val_idx = np.where(train_val_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Further split train_val into train / val (80/20 of training portion)
    tv_y = y[train_val_idx]
    if len(np.unique(tv_y)) >= 2 and len(tv_y) >= 10:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        rel_train, rel_val = next(sss.split(X[train_val_idx], tv_y))
        train_idx = train_val_idx[rel_train]
        val_idx = train_val_idx[rel_val]
    else:
        train_idx = train_val_idx
        val_idx = train_val_idx[:max(1, len(train_val_idx) // 5)]

    return {
        "X_train": X[train_idx], "y_train": y[train_idx],
        "X_val": X[val_idx], "y_val": y[val_idx],
        "X_test": X[test_idx], "y_test": y[test_idx],
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "strategy": "cross_subject",
        "test_subject": int(test_subject),
        "train_subjects": [int(s) for s in unique_subjects if s != test_subject],
    }


# ── Core training ─────────────────────────────────────────────

def train_head(
    X, y, label_names,
    n_epochs=200, lr=0.01, weight_decay=1e-3,
    device=None, verbose=True,
    progress_cb: Optional[Callable] = None,
):
    """Train the linear head with cross-validation reporting.

    Args:
        X: numpy array (N, 512) — CLIP embeddings
        y: list of int — class indices
        label_names: list of str — category names in order
        progress_cb: called each epoch as progress_cb(epoch, n_epochs, metrics)

    Returns:
        trained LinearHead module (on CPU), training stats dict
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    n_classes = len(label_names)
    y_arr = np.array(y)

    if verbose:
        print(f"    Training linear head: {len(X)} samples, {n_classes} classes, device={device}")
        _cross_val_report(X, y_arr, label_names, n_classes, device, n_epochs,
                          lr, weight_decay)

    # Final model trained on all data
    model = LinearHead(n_classes).to(device)
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y_arr).long().to(device)

    class_counts = np.bincount(y_arr, minlength=n_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = []
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y_t).float().mean().item()
            metrics = {"train_loss": round(loss.item(), 4), "train_acc": round(acc, 4)}
            history.append({"epoch": epoch + 1, **metrics})

            if progress_cb is not None:
                progress_cb(epoch + 1, n_epochs, metrics)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"      epoch {epoch+1:4d}  loss={loss.item():.4f}  acc={acc:.3f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    train_acc = (preds == y_arr).mean()

    model = model.cpu()
    stats = {
        "model": "clip_linear_head",
        "n_samples": len(X),
        "n_classes": n_classes,
        "train_acc": float(train_acc),
        "n_epochs": n_epochs,
        "label_names": label_names,
        "history": history,
    }

    if verbose:
        print(f"    Final train accuracy: {train_acc:.3f}")

    return model, stats


def _cross_val_report(X, y_arr, label_names, n_classes, device,
                      n_epochs, lr, weight_decay):
    """Stratified CV to estimate generalization (adapts fold count to data)."""
    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        print("    Only 1 class labeled — skipping CV")
        return

    min_class_count = min(np.bincount(y_arr, minlength=n_classes)[c]
                          for c in unique_classes)
    n_folds = min(3, min_class_count, len(unique_classes))
    if n_folds < 2:
        print(f"    Smallest class has {min_class_count} sample(s) — "
              f"skipping CV (will still train on all data)")
        return

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_preds, all_true = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_arr)):
        model = LinearHead(n_classes).to(device)
        X_tr = torch.from_numpy(X[train_idx]).to(device)
        y_tr = torch.from_numpy(y_arr[train_idx]).long().to(device)
        X_val = torch.from_numpy(X[val_idx]).to(device)

        class_counts = np.bincount(y_arr[train_idx], minlength=n_classes).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * n_classes
        loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                       weight_decay=weight_decay)

        model.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            loss = loss_fn(model(X_tr), y_tr)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(y_arr[val_idx])

    present_names = [label_names[c] for c in sorted(unique_classes)]
    print(f"\n    {n_folds}-fold CV classification report:")
    print(classification_report(
        all_true, all_preds,
        target_names=present_names, labels=sorted(unique_classes),
        zero_division=0,
    ))


# ── Full training with holdout split ─────────────────────────

def train_with_holdout(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    subject_ids: Optional[np.ndarray] = None,
    strategy: str = "within_subject",
    test_subject: Optional[int] = None,
    n_epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 1e-3,
    device=None,
    progress_cb: Optional[Callable] = None,
) -> tuple:
    """Train linear head with a proper held-out test set.

    Args:
        strategy: "within_subject" or "cross_subject"
        subject_ids: required when strategy="cross_subject"
        test_subject: which subject to hold out (cross_subject only; default=last)

    Returns:
        (model, stats, split_info)
        stats includes val_acc, test_acc, test_classification_report
        split_info has train/val/test indices for reproducibility
    """
    y_arr = np.array(y)

    if strategy == "cross_subject":
        if subject_ids is None:
            raise ValueError("subject_ids required for cross_subject strategy")
        split = split_cross_subject(X, y_arr, np.array(subject_ids), test_subject)
    else:
        split = split_within_subject(X, y_arr)

    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]

    print(f"  Split ({strategy}): train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    n_classes = len(label_names)
    model = LinearHead(n_classes).to(device)
    X_tr_t = torch.from_numpy(X_train).to(device)
    y_tr_t = torch.from_numpy(y_train).long().to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_acc = 0.0
    best_state = None
    history = []

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss = loss_fn(logits, y_tr_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val_t).argmax(1)
                val_acc = (val_preds == y_val_t).float().mean().item()
                train_loss = loss.item()
            model.train()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            metrics = {
                "train_loss": round(train_loss, 4),
                "val_acc": round(val_acc, 4),
            }
            history.append({"epoch": epoch + 1, **metrics})

            if progress_cb is not None:
                progress_cb(epoch + 1, n_epochs, metrics)

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.cpu()
    model.eval()

    # Test set evaluation
    X_test_t = torch.from_numpy(X_test)
    with torch.no_grad():
        test_preds = model(X_test_t).argmax(1).numpy()
    test_acc = (test_preds == y_test).mean()

    present = sorted(set(y_test.tolist()))
    present_names = [label_names[c] for c in present]
    test_report = classification_report(
        y_test, test_preds, labels=present, target_names=present_names,
        output_dict=True, zero_division=0,
    )
    test_cm = confusion_matrix(y_test, test_preds, labels=present).tolist()

    stats = {
        "model": "clip_linear_head",
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_classes": n_classes,
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(float(test_acc), 4),
        "test_classification_report": test_report,
        "test_confusion_matrix": test_cm,
        "test_label_order": present_names,
        "n_epochs": n_epochs,
        "label_names": label_names,
        "history": history,
        "split_strategy": split.get("strategy", strategy),
    }
    if "test_subject" in split:
        stats["test_subject"] = split["test_subject"]
        stats["train_subjects"] = split["train_subjects"]

    return model, stats, split


# ── Checkpoint I/O ────────────────────────────────────────────

def save_head(model, stats, out_path):
    """Save trained linear head weights + metadata."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "stats": stats,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }, out_path)
    print(f"    Saved trained head → {out_path}")


def save_head_versioned(
    model,
    stats: dict,
    models_dir: str = MODELS_DIR,
    prefix: str = "clip_head",
    test_filenames: Optional[list] = None,
    test_labels: Optional[list] = None,
) -> str:
    """Save with a timestamped filename. Returns the saved path."""
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.pt"
    path = os.path.join(models_dir, filename)
    payload = {
        "state_dict": model.state_dict(),
        "stats": stats,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if test_filenames is not None:
        payload["test_filenames"] = test_filenames
    if test_labels is not None:
        payload["test_labels"] = test_labels
    torch.save(payload, path)
    print(f"    Saved versioned head → {path}")
    return path


def load_head(path, n_classes=None, embed_dim=512):
    """Load a trained linear head from .pt file.

    Returns (LinearHead model, stats dict).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stats = checkpoint["stats"]
    nc = n_classes or stats.get("n_classes", len(CATEGORIES))
    model = LinearHead(nc, embed_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, stats


def list_saved_models(models_dir: str = MODELS_DIR) -> list[dict]:
    """Return metadata for all saved model checkpoints in models_dir.

    Returns list of dicts sorted by save time (newest first):
    {path, filename, model_type, saved_at, n_samples, val_acc, test_acc}
    """
    if not os.path.isdir(models_dir):
        return []
    entries = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(models_dir, fname)
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            stats = ckpt.get("stats", {})
            entries.append({
                "path": path,
                "filename": fname,
                "model_type": stats.get("model", "unknown"),
                "saved_at": ckpt.get("saved_at", ""),
                "n_samples": stats.get("n_samples") or stats.get("n_train", "?"),
                "val_acc": stats.get("best_val_acc") or stats.get("train_acc", None),
                "test_acc": stats.get("test_acc", None),
                "label_names": stats.get("label_names", []),
                "split_strategy": stats.get("split_strategy", "unknown"),
            })
        except Exception:
            pass
    return sorted(entries, key=lambda e: e["saved_at"], reverse=True)


# ── High-level entry point ────────────────────────────────────

def train_from_files(labels_csv, embeddings_npy, embeddings_ids_csv,
                     out_model_path, **kwargs):
    """End-to-end: load data → train → save.  Returns stats dict."""
    print(f"  Training CLIP linear head from {labels_csv}")
    X, y, label_names = _load_labeled_embeddings(
        labels_csv, embeddings_npy, embeddings_ids_csv
    )

    if len(X) < 10:
        print(f"    Only {len(X)} labeled samples — need at least 10 to train")
        return None

    model, stats = train_head(X, y, label_names, **kwargs)
    save_head(model, stats, out_model_path)
    return stats


def train_from_label_store(
    sj_num,
    condition,
    embeddings_base,
    out_model_path,
    strategy: str = "within_subject",
    subject_ids_array: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable] = None,
    **kwargs,
):
    """Train from the central label store with a proper holdout split.

    Args:
        sj_num: subject number (int), or None to pool all subjects.
        condition: condition string, or None to pool all conditions.
        embeddings_base: path prefix for <base>.npy and <base>_ids.csv.
        strategy: "within_subject" or "cross_subject"
        subject_ids_array: array of subject IDs parallel to X (cross_subject only)
        progress_cb: called each epoch as progress_cb(epoch, n_epochs, metrics)

    Returns stats dict, or None if not enough data.
    """
    if sj_num is not None and condition is not None:
        labels_df = load_labels_for(int(sj_num), condition)
        scope = f"sj{sj_num:02d} {condition}"
    else:
        labels_df = load_trainable_labels()
        scope = "all subjects pooled"

    if labels_df.empty:
        print(f"  No labels found in central store ({scope})")
        return None

    print(f"  Training CLIP linear head from central store ({scope}, "
          f"{len(labels_df)} labels)")

    emb_npy = f"{embeddings_base}.npy"
    emb_ids = f"{embeddings_base}_ids.csv"
    if not os.path.exists(emb_npy) or not os.path.exists(emb_ids):
        print(f"    Embeddings not found at {embeddings_base}")
        return None

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp_path = tmp.name
        labels_df[["fixation_id", "human_label"]].to_csv(tmp_path, index=False)

    try:
        X, y, label_names = _load_labeled_embeddings(tmp_path, emb_npy, emb_ids)
    finally:
        os.unlink(tmp_path)

    if len(X) < 10:
        print(f"    Only {len(X)} matched samples — need at least 10")
        return None

    model, stats, split = train_with_holdout(
        X, np.array(y), label_names,
        subject_ids=subject_ids_array,
        strategy=strategy,
        progress_cb=progress_cb,
        **kwargs,
    )

    # Save test filenames for reproduce-able evaluation
    test_filenames = labels_df.iloc[split["test_idx"]]["filename"].tolist() \
        if len(split["test_idx"]) <= len(labels_df) else []
    test_true_labels = labels_df.iloc[split["test_idx"]]["human_label"].tolist() \
        if test_filenames else []

    save_head_versioned(
        model, stats,
        models_dir=os.path.dirname(out_model_path) or MODELS_DIR,
        prefix="clip_head",
        test_filenames=test_filenames,
        test_labels=test_true_labels,
    )
    # Also save to the legacy fixed path for backward compat
    save_head(model, stats, out_model_path)
    return stats


# ── Standalone entry point ────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CLIP linear head")
    parser.add_argument("--labels", required=True, help="Path to human_labels.csv")
    parser.add_argument("--embeddings", required=True,
                        help="Path to embeddings .npy (without extension)")
    parser.add_argument("--output", required=True, help="Path for output .pt model file")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--strategy", default="within_subject",
                        choices=["within_subject", "cross_subject"])
    args = parser.parse_args()

    X, y, label_names = _load_labeled_embeddings(
        args.labels, f"{args.embeddings}.npy", f"{args.embeddings}_ids.csv"
    )
    if len(X) < 10:
        print(f"Only {len(X)} samples — need ≥10")
        sys.exit(1)

    model, stats, split = train_with_holdout(
        X, np.array(y), label_names,
        strategy=args.strategy,
        n_epochs=args.epochs,
        lr=args.lr,
    )
    save_head_versioned(model, stats, models_dir=os.path.dirname(args.output) or MODELS_DIR)
    save_head(model, stats, args.output)
    print(f"\nDone.  Val acc: {stats['best_val_acc']:.3f}  Test acc: {stats['test_acc']:.3f}")
