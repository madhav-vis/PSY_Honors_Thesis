"""Train a linear classification head on frozen CLIP embeddings.

Takes human-labeled crops + pre-extracted CLIP embeddings and trains a
512 → N_CATEGORIES linear layer.  Saves the trained weights as a .pt
file that GazeClassifier can load for inference.

Can run standalone:  python src/vision/train_head.py
Or called from vision_main.py as part of the pipeline.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.config import CATEGORIES


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


class LinearHead(nn.Module):
    """Single linear layer: 512-dim CLIP embedding → N categories."""

    def __init__(self, n_classes, embed_dim=512):
        super().__init__()
        self.linear = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


def train_head(X, y, label_names, n_epochs=200, lr=0.01, weight_decay=1e-3,
               device=None, verbose=True):
    """Train the linear head with cross-validation reporting.

    Args:
        X: numpy array (N, 512) — CLIP embeddings
        y: list of int — class indices
        label_names: list of str — category names in order
        n_epochs: training epochs
        lr: learning rate
        weight_decay: L2 regularization
        device: torch device (auto-detected if None)
        verbose: print training progress

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

    # Cross-validation for reporting (not model selection — we train on all data)
    if verbose:
        print(f"    Training linear head: {len(X)} samples, {n_classes} classes, device={device}")
        _cross_val_report(X, y_arr, label_names, n_classes, device, n_epochs,
                          lr, weight_decay)

    # Final model trained on all data
    model = LinearHead(n_classes).to(device)
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y_arr).long().to(device)

    # Class-balanced loss weights
    class_counts = np.bincount(y_arr, minlength=n_classes).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if verbose and (epoch + 1) % 50 == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == y_t).float().mean().item()
            print(f"      epoch {epoch+1:4d}  loss={loss.item():.4f}  acc={acc:.3f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).cpu().numpy()
    train_acc = (preds == y_arr).mean()

    model = model.cpu()
    stats = {
        "n_samples": len(X),
        "n_classes": n_classes,
        "train_acc": float(train_acc),
        "n_epochs": n_epochs,
        "label_names": label_names,
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


def save_head(model, stats, out_path):
    """Save trained linear head weights + metadata."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "stats": stats,
    }, out_path)
    print(f"    Saved trained head → {out_path}")


def load_head(path, n_classes=None, embed_dim=512):
    """Load a trained linear head from .pt file.

    Returns (LinearHead model, stats dict).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    stats = checkpoint["stats"]
    nc = n_classes or stats.get("n_classes", len(CATEGORIES))
    model = LinearHead(nc, embed_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, stats


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


def train_from_label_store(sj_num, condition, embeddings_base,
                           out_model_path, **kwargs):
    """Train from the central label store.

    Args:
        sj_num: subject number (int), or None to pool all subjects.
        condition: condition string, or None to pool all conditions.
        embeddings_base: path prefix for <base>.npy and <base>_ids.csv —
            used only when sj_num/condition are specified. Pooled training
            requires embeddings to exist per-subject (uses first available).
        out_model_path: where to save the .pt file.

    Returns stats dict, or None if not enough data.
    """
    import sys, os
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from vision.label_store import load_labels, load_labels_for

    if sj_num is not None and condition is not None:
        labels_df = load_labels_for(int(sj_num), condition)
        scope = f"sj{sj_num:02d} {condition}"
    else:
        labels_df = load_labels()
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

    # Write a temp CSV compatible with _load_labeled_embeddings
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as tmp:
        tmp_path = tmp.name
        labels_df[["fixation_id", "human_label"]].to_csv(tmp_path, index=False)

    try:
        return train_from_files(
            labels_csv=tmp_path,
            embeddings_npy=emb_npy,
            embeddings_ids_csv=emb_ids,
            out_model_path=out_model_path,
            **kwargs,
        )
    finally:
        os.unlink(tmp_path)


# ── Standalone entry point ────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train CLIP linear head")
    parser.add_argument("--labels", required=True,
                        help="Path to human_labels.csv")
    parser.add_argument("--embeddings", required=True,
                        help="Path to embeddings .npy (without extension)")
    parser.add_argument("--output", required=True,
                        help="Path for output .pt model file")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    stats = train_from_files(
        labels_csv=args.labels,
        embeddings_npy=f"{args.embeddings}.npy",
        embeddings_ids_csv=f"{args.embeddings}_ids.csv",
        out_model_path=args.output,
        n_epochs=args.epochs,
        lr=args.lr,
    )
    if stats:
        print(f"\n  Done. Train acc: {stats['train_acc']:.3f}")
