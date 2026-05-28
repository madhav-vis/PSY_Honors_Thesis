"""Fine-tune ResNet-50 on raw gaze crop images.

Trains end-to-end on 224×224 PNG crops read directly from stable storage.
Requires raw pixels — does NOT use CLIP embeddings.

Usage:
    from vision.resnet_head import train_from_label_store
    stats = train_from_label_store(out_path="models/resnet50_20260429.pt")

Or standalone:
    python src/vision/resnet_head.py --output models/resnet50.pt
"""

import os
import sys
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.config import CATEGORIES
from vision.label_store import (
    FLAG_LABEL,
    get_crop_path,
    load_trainable_labels,
    load_labels_for,
    PROJECT_ROOT,
)

try:
    from torchvision import models, transforms
    from PIL import Image
    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

_TRAIN_TRANSFORMS = None
_VAL_TRANSFORMS = None


def _get_transforms():
    global _TRAIN_TRANSFORMS, _VAL_TRANSFORMS
    if _TRAIN_TRANSFORMS is None:
        _TRAIN_TRANSFORMS = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
        _VAL_TRANSFORMS = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return _TRAIN_TRANSFORMS, _VAL_TRANSFORMS


# ── Dataset ───────────────────────────────────────────────────

class CropDataset(Dataset):
    """PyTorch Dataset loading 224×224 PNG crops from stable storage."""

    def __init__(self, records: list[dict], transform=None):
        """
        Args:
            records: list of dicts with keys: sj_num, condition, filename, label_idx
            transform: torchvision transform to apply
        """
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        path = get_crop_path(r["sj_num"], r["condition"], r["filename"])
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, FileNotFoundError):
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, r["label_idx"]


def _build_records(labels_df: pd.DataFrame, label_to_idx: dict) -> list[dict]:
    """Convert labels DataFrame rows into record dicts for CropDataset."""
    records = []
    for _, row in labels_df.iterrows():
        lbl = str(row["human_label"])
        if lbl not in label_to_idx:
            continue
        records.append({
            "sj_num": int(row["subject_id"]),
            "condition": str(row["condition"]),
            "filename": str(row["filename"]),
            "label_idx": label_to_idx[lbl],
        })
    return records


# ── Model ─────────────────────────────────────────────────────

def build_resnet50(n_classes: int) -> nn.Module:
    """ImageNet-pretrained ResNet-50 with a dropout + linear classification head."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, n_classes),
    )
    return model


# ── Training ──────────────────────────────────────────────────

def train_resnet(
    train_records: list[dict],
    val_records: list[dict],
    label_names: list[str],
    n_epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    progress_cb: Optional[Callable] = None,
    loss_fn: Optional[nn.Module] = None,
):
    """Train ResNet-50 end-to-end on crop images.

    Args:
        train_records / val_records: list of dicts from _build_records()
        label_names: ordered list of class names
        progress_cb: called each epoch as progress_cb(epoch, n_epochs, metrics_dict)
                     where metrics_dict has keys: train_loss, val_loss, val_acc

    Returns:
        (trained model on CPU, stats dict)
    """
    if not _HAS_TORCHVISION:
        raise ImportError("torchvision and Pillow are required for ResNet training")

    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    train_tfm, val_tfm = _get_transforms()
    n_classes = len(label_names)
    label_to_idx = {n: i for i, n in enumerate(label_names)}

    train_ds = CropDataset(train_records, transform=train_tfm)
    val_ds = CropDataset(val_records, transform=val_tfm)

    # Balanced minibatches: rare classes seen as often as trail_ground per epoch.
    y_train = [r["label_idx"] for r in train_records]
    from vision.class_balance import make_weighted_sampler

    sampler = make_weighted_sampler(y_train, n_classes)

    _pin = device.type == "cuda"
    from device_utils import (
        dataloader_workers as _dl_workers,
        persistent_workers as _persistent,
        prefetch_factor as _prefetch,
        use_amp as _use_amp,
        use_channels_last as _use_channels_last,
        enable_cudnn_benchmark,
        make_autocast, make_grad_scaler,
    )
    enable_cudnn_benchmark()
    _nw = _dl_workers()
    _pw = _persistent() and _nw > 0
    _pf_kwargs = {"prefetch_factor": _prefetch()} if _nw > 0 else {}
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=_nw, pin_memory=_pin,
                              persistent_workers=_pw, **_pf_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=_nw, pin_memory=_pin,
                            persistent_workers=_pw, **_pf_kwargs)

    model = build_resnet50(n_classes).to(device)
    _channels_last = _use_channels_last()
    if _channels_last:
        model = model.to(memory_format=torch.channels_last)

    if loss_fn is None:
        from vision.class_balance import make_classification_loss

        loss_fn = make_classification_loss(
            y_train, n_classes, use_sampler=True
        )
    loss_fn = loss_fn.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    amp_on = _use_amp()
    scaler = make_grad_scaler(enabled=amp_on)
    _perf_msg = []
    if amp_on:
        _perf_msg.append("AMP fp16")
    if _channels_last:
        _perf_msg.append("channels_last")
    if _nw > 0:
        _perf_msg.append(f"{_nw} workers x prefetch {_prefetch()}")
    if _perf_msg:
        print(f"  ResNet perf: {', '.join(_perf_msg)} on {device}")

    from vision.class_balance import balanced_val_accuracy

    history = []
    best_val_acc = 0.0
    best_val_bal_acc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=_pin)
            if _channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=_pin)
            optimizer.zero_grad(set_to_none=True)
            with make_autocast(device, enabled=amp_on):
                logits = model(imgs)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(imgs)
        train_loss /= max(len(train_ds), 1)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=_pin)
                if _channels_last:
                    imgs = imgs.to(memory_format=torch.channels_last)
                labels = labels.to(device, non_blocking=_pin)
                with make_autocast(device, enabled=amp_on):
                    logits = model(imgs)
                    batch_loss = loss_fn(logits, labels)
                val_loss += batch_loss.item() * len(imgs)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        val_loss /= max(len(val_ds), 1)
        val_acc = float(np.mean(np.asarray(val_preds) == np.asarray(val_true)))
        val_bal_acc = balanced_val_accuracy(val_true, val_preds)

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        metrics = {
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_bal_acc": round(val_bal_acc, 4),
        }
        history.append({"epoch": epoch + 1, **metrics})

        if progress_cb is not None:
            progress_cb(epoch + 1, n_epochs, metrics)

    # Restore best checkpoint (keep on device for the eval pass; move to CPU at the end)
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()

    # Final classification report on validation set
    val_preds, val_true = [], []
    val_ds_plain = CropDataset(val_records, transform=val_tfm)
    val_loader_plain = DataLoader(val_ds_plain, batch_size=batch_size, shuffle=False,
                                  num_workers=_nw, pin_memory=_pin,
                                  persistent_workers=_pw, **_pf_kwargs)
    with torch.no_grad():
        for imgs, labels in val_loader_plain:
            imgs = imgs.to(device, non_blocking=_pin)
            if _channels_last:
                imgs = imgs.to(memory_format=torch.channels_last)
            with make_autocast(device, enabled=amp_on):
                logits = model(imgs)
            val_preds.extend(logits.argmax(1).cpu().numpy())
            val_true.extend(labels.numpy())

    model = model.cpu()

    present = sorted(set(val_true))
    present_names = [label_names[c] for c in present]
    report = classification_report(
        val_true, val_preds, labels=present, target_names=present_names,
        output_dict=True, zero_division=0,
    )

    stats = {
        "model": "resnet50",
        "n_train": len(train_records),
        "n_val": len(val_records),
        "n_classes": n_classes,
        "label_names": label_names,
        "best_val_acc": round(best_val_acc, 4),
        "best_val_bal_acc": round(best_val_bal_acc, 4),
        "n_epochs": n_epochs,
        "history": history,
        "classification_report": report,
    }
    return model, stats


# ── Checkpoint I/O ────────────────────────────────────────────

def save_resnet(
    model: nn.Module,
    stats: dict,
    out_path: str,
    test_filenames: Optional[list] = None,
    test_labels: Optional[list] = None,
) -> str:
    """Save ResNet checkpoint with metadata."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "stats": stats,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if test_filenames is not None:
        payload["test_filenames"] = test_filenames
    if test_labels is not None:
        payload["test_labels"] = test_labels
    torch.save(payload, out_path)
    return out_path


def load_resnet(path: str, n_classes: Optional[int] = None) -> tuple:
    """Load ResNet-50 checkpoint.  Returns (model, stats)."""
    if not _HAS_TORCHVISION:
        raise ImportError("torchvision is required to load ResNet models")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    nc = n_classes or stats.get("n_classes", len(CATEGORIES))
    model = build_resnet50(nc)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, stats


# ── High-level entry point ────────────────────────────────────

def train_from_label_store(
    out_path: str,
    sj_num: Optional[int] = None,
    condition: Optional[str] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    n_epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 32,
    progress_cb: Optional[Callable] = None,
) -> Optional[dict]:
    """Train ResNet-50 from the central label store.

    Args:
        out_path: where to save the .pt checkpoint
        sj_num / condition: filter to one subject/condition, or None to pool all
        test_size / val_size: fractions for held-out test and validation splits
        progress_cb: called each epoch as progress_cb(epoch, n_epochs, metrics)

    Returns:
        stats dict (includes history, val_acc, test_acc), or None if not enough data
    """
    if not _HAS_TORCHVISION:
        raise ImportError("torchvision and Pillow are required for ResNet training")

    if sj_num is not None and condition is not None:
        labels_df = load_labels_for(int(sj_num), condition)
    else:
        labels_df = load_trainable_labels()

    if labels_df.empty or len(labels_df) < 20:
        print(f"  Not enough labels ({len(labels_df)}) to train ResNet")
        return None

    label_names = list(CATEGORIES.keys())
    label_to_idx = {n: i for i, n in enumerate(label_names)}

    # Filter to known categories
    labels_df = labels_df[labels_df["human_label"].isin(label_to_idx)].reset_index(drop=True)
    if len(labels_df) < 20:
        return None

    y = np.array([label_to_idx[l] for l in labels_df["human_label"]])

    # Stratified train / val / test split
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_val_idx, test_idx = next(sss_test.split(labels_df, y))

    val_frac = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=42)
    tv_labels = labels_df.iloc[train_val_idx]
    tv_y = y[train_val_idx]
    rel_train_idx, rel_val_idx = next(sss_val.split(tv_labels, tv_y))
    train_idx = train_val_idx[rel_train_idx]
    val_idx = train_val_idx[rel_val_idx]

    train_records = _build_records(labels_df.iloc[train_idx], label_to_idx)
    val_records = _build_records(labels_df.iloc[val_idx], label_to_idx)
    test_records = _build_records(labels_df.iloc[test_idx], label_to_idx)

    print(f"  ResNet training: {len(train_records)} train, {len(val_records)} val, "
          f"{len(test_records)} test")

    model, stats = train_resnet(
        train_records, val_records, label_names,
        n_epochs=n_epochs, lr=lr, batch_size=batch_size,
        progress_cb=progress_cb,
    )

    # Save the model FIRST so a crash in the test-eval step below can't
    # destroy a successful training run. We save again after test eval
    # (with test_acc etc. populated) if it completes.
    test_filenames = labels_df.iloc[test_idx]["filename"].tolist()
    test_true_labels = labels_df.iloc[test_idx]["human_label"].tolist()
    save_resnet(model, stats, out_path,
                test_filenames=test_filenames, test_labels=test_true_labels)
    print(f"  Saved ResNet checkpoint (pre-test) -> {out_path}")

    # Evaluate on test set. We deliberately use num_workers=0 here:
    # - Training workers from train_resnet are gone, but launching 8 fresh
    #   spawn-workers each re-imports torch/scipy/sklearn from scratch,
    #   which can blow Windows commit-charge on large stacks.
    # - The test pass is a single shot over a few hundred batches; CPU
    #   image-decode is not the bottleneck once the model is on GPU.
    if test_records:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from device_utils import (
            get_device as _get_device,
            use_amp as _use_amp_t,
            use_channels_last as _channels_last_t,
            make_autocast as _autocast_t,
        )
        _eval_device = _get_device()
        _pin_eval = _eval_device.type == "cuda"
        _amp_t = _use_amp_t()
        _cl_t = _channels_last_t()
        _, val_tfm = _get_transforms()
        test_ds = CropDataset(test_records, transform=val_tfm)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=_pin_eval)
        model_eval = model.to(_eval_device)
        if _cl_t:
            model_eval = model_eval.to(memory_format=torch.channels_last)
        model_eval.eval()
        test_preds, test_true = [], []
        try:
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.to(_eval_device, non_blocking=_pin_eval)
                    if _cl_t:
                        imgs = imgs.to(memory_format=torch.channels_last)
                    with _autocast_t(_eval_device, enabled=_amp_t):
                        logits = model_eval(imgs)
                    test_preds.extend(logits.argmax(1).cpu().numpy())
                    test_true.extend(labels.numpy())
            model = model_eval.cpu()
            test_acc = sum(p == t for p, t in zip(test_preds, test_true)) / len(test_true)
            present = sorted(set(test_true))
            present_names = [label_names[c] for c in present]
            stats["test_acc"] = round(test_acc, 4)
            stats["test_classification_report"] = classification_report(
                test_true, test_preds, labels=present, target_names=present_names,
                output_dict=True, zero_division=0,
            )
            stats["test_confusion_matrix"] = confusion_matrix(
                test_true, test_preds, labels=present
            ).tolist()
            stats["test_label_order"] = present_names

            # Re-save with full test metrics. Training metrics are already
            # safe on disk from the pre-test save.
            save_resnet(model, stats, out_path,
                        test_filenames=test_filenames, test_labels=test_true_labels)
            print(f"  Updated ResNet checkpoint with test metrics -> {out_path}")
        except Exception as e:
            print(f"  WARNING: test-set evaluation failed ({type(e).__name__}: {e})")
            print(f"  The trained model is still safely saved at {out_path}.")
            print(f"  You can re-run test eval later from the Evaluate tab.")

    return stats


# ── Standalone CLI ────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ResNet-50 on gaze crops")
    parser.add_argument(
        "--output", default=None,
        help="Output .pt path. If omitted, auto-saves to "
             "models/resnet50_<timestamp>.pt (same naming as the Streamlit "
             "Train tab). Use --versioned to force the timestamp suffix "
             "even when --output is provided.",
    )
    parser.add_argument(
        "--versioned", action="store_true",
        help="Append a _YYYYMMDD_HHMMSS timestamp to the output filename so "
             "successive runs don't overwrite each other.",
    )
    parser.add_argument("--sj", type=int, default=None, help="Subject number (omit to pool all)")
    parser.add_argument("--condition", default=None, help="Condition (omit to pool all)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Resolve output path: default-or-versioned both add a timestamp suffix
    _stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        out_path = os.path.join(MODELS_DIR, f"resnet50_{_stamp}.pt")
    elif args.versioned:
        base, ext = os.path.splitext(args.output)
        out_path = f"{base}_{_stamp}{ext or '.pt'}"
    else:
        out_path = args.output
    print(f"  Output checkpoint: {out_path}")

    def _cli_cb(epoch, n_epochs, metrics):
        print(f"  epoch {epoch:3d}/{n_epochs}  "
              f"train_loss={metrics['train_loss']:.4f}  "
              f"val_loss={metrics['val_loss']:.4f}  "
              f"val_acc={metrics['val_acc']:.3f}")

    stats = train_from_label_store(
        out_path=out_path,
        sj_num=args.sj,
        condition=args.condition,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        progress_cb=_cli_cb,
    )
    if stats:
        print(f"\nDone. Best val acc: {stats['best_val_acc']:.3f}  "
              f"Test acc: {stats.get('test_acc', 'N/A')}")
        print(f"Open the Streamlit annotator's Evaluate tab to view this run "
              f"({os.path.basename(out_path)}).")
