"""Train ResNet-50 gaze crop classifier from the central label store.

Saves checkpoints:
  models/resnet50.pt  (canonical)
  models/resnet50_<timestamp>.pt  (versioned backup)

Usage (from repo root, venv active):
    python src/vision/train_models.py
    python src/vision/train_models.py --resnet-epochs 40
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.label_store import PROJECT_ROOT, load_trainable_labels

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESNET_PATH = os.path.join(MODELS_DIR, "resnet50.pt")


def train_resnet(
    *,
    n_epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 32,
) -> str | None:
    try:
        from torchvision import models as _tv  # noqa: F401
    except ImportError:
        print("  ResNet: torchvision not installed (`pip install torchvision Pillow`)")
        return None

    from vision.resnet_head import train_from_label_store

    trainable = load_trainable_labels()
    if len(trainable) < 20:
        print(f"  ResNet: only {len(trainable)} labels — need >=20")
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned = os.path.join(MODELS_DIR, f"resnet50_{stamp}.pt")
    os.makedirs(MODELS_DIR, exist_ok=True)

    def _cb(epoch, n_epochs, metrics):
        bal = metrics.get("val_bal_acc", metrics["val_acc"])
        print(
            f"    epoch {epoch:3d}/{n_epochs}  "
            f"train_loss={metrics['train_loss']:.4f}  "
            f"val_acc={metrics['val_acc']:.3f}  val_bal_acc={bal:.3f}"
        )

    print("  ResNet: training on all labeled crops in data/crops/")
    stats = train_from_label_store(
        out_path=versioned,
        sj_num=None,
        condition=None,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        progress_cb=_cb,
    )
    if not stats:
        print("  ResNet: training failed (not enough data?)")
        return None

    shutil.copy2(versioned, RESNET_PATH)
    test_acc = stats.get("test_acc")
    test_str = f"{test_acc:.3f}" if isinstance(test_acc, (int, float)) else str(test_acc)
    print(
        f"  ResNet done — best val acc {stats['best_val_acc']:.3f}, "
        f"val bal acc {stats.get('best_val_bal_acc', 0):.3f}, "
        f"test acc {test_str}"
    )
    print(f"  -> {RESNET_PATH}")
    print(f"  -> {versioned}")
    return RESNET_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 gaze crop classifier"
    )
    parser.add_argument("--resnet-epochs", type=int, default=30)
    parser.add_argument("--resnet-batch-size", type=int, default=32)
    parser.add_argument("--lr-resnet", type=float, default=1e-4)
    args = parser.parse_args()

    print("=" * 60)
    print("  TRAIN RESNET-50 (label store -> models/)")
    print(f"  Labels: {PROJECT_ROOT}/data/human_labels.csv")
    print("=" * 60)

    print("\n--- ResNet-50 ---")
    p = train_resnet(
        n_epochs=args.resnet_epochs,
        lr=args.lr_resnet,
        batch_size=args.resnet_batch_size,
    )

    print("\n" + "=" * 60)
    if p:
        print(f"  Saved: {p}")
        print("\n  Next: run evaluation")
        print("    python src/evaluate.py --vision-only")
    else:
        print("  No model saved — check errors above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
