"""Train CLIP linear head + ResNet-50 from the central label store.

Saves checkpoints for the Evaluate / Deploy tabs:
  models/clip_head.pt
  models/resnet50.pt
(+ timestamped copies in models/)

Usage (from repo root, venv active):
    python src/vision/train_models.py
    python src/vision/train_models.py --clip-only
    python src/vision/train_models.py --resnet-only --resnet-epochs 40

CLIP head: uses runs/*/vision/*_embeddings.npy if present; otherwise encodes
fresh CLIP vectors from data/crops/ (same as the vision pipeline).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.config import CATEGORIES
from vision.label_store import PROJECT_ROOT, get_crop_path, load_trainable_labels

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CLIP_HEAD_PATH = os.path.join(MODELS_DIR, "clip_head.pt")
RESNET_PATH = os.path.join(MODELS_DIR, "resnet50.pt")
CLIP_EMB_CACHE = os.path.join(MODELS_DIR, "labeled_clip_embeddings.npz")


def _all_runs_roots() -> list[str]:
    """Search project runs/ and optional PSY197B_RUNS_DIR."""
    roots = [os.path.join(PROJECT_ROOT, "runs")]
    extra = os.environ.get("PSY197B_RUNS_DIR")
    if extra:
        extra = os.path.abspath(extra)
        if extra not in roots:
            roots.append(extra)
    return [r for r in roots if os.path.isdir(r)]


def find_all_embeddings() -> dict[tuple[int, str], str]:
    """Map (subject_id, condition) → embeddings path prefix (no extension)."""
    result: dict[tuple[int, str], str] = {}
    for runs_root in _all_runs_roots():
        for run_name in sorted(os.listdir(runs_root), reverse=True):
            vision_dir = os.path.join(runs_root, run_name, "vision")
            if not os.path.isdir(vision_dir):
                continue
            for sj_cond in sorted(os.listdir(vision_dir)):
                parts = sj_cond.split("_", 1)
                if len(parts) != 2 or not parts[0].startswith("sj"):
                    continue
                try:
                    sj_num = int(parts[0][2:])
                except ValueError:
                    continue
                cond = parts[1]
                base = os.path.join(vision_dir, sj_cond, f"{sj_cond}_embeddings")
                if os.path.exists(f"{base}.npy") and os.path.exists(f"{base}_ids.csv"):
                    if (sj_num, cond) not in result:
                        result[(sj_num, cond)] = base
    return result


def _count_runs_without_vision() -> int:
    n = 0
    for runs_root in _all_runs_roots():
        for run_name in os.listdir(runs_root):
            rd = os.path.join(runs_root, run_name)
            if os.path.isdir(rd) and not os.path.isdir(os.path.join(rd, "vision")):
                n += 1
    return n


def pool_clip_from_runs(trainable_df, emb_lookup: dict):
    """Stack precomputed embeddings from runs/*/vision/."""
    from vision.train_head import _load_labeled_embeddings

    label_names = list(CATEGORIES.keys())
    label_to_idx = {n: i for i, n in enumerate(label_names)}

    X_all, y_all, sj_ids = [], [], []

    for (sj, cond), emb_base in sorted(emb_lookup.items()):
        subset = trainable_df[
            (trainable_df["subject_id"] == sj)
            & (trainable_df["condition"] == cond)
        ]
        if subset.empty:
            continue
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name
            subset[["fixation_id", "human_label"]].to_csv(tmp_path, index=False)
        try:
            X_sub, y_sub, _ = _load_labeled_embeddings(
                tmp_path, f"{emb_base}.npy", f"{emb_base}_ids.csv"
            )
        finally:
            os.unlink(tmp_path)
        if len(X_sub) == 0:
            continue
        X_all.append(X_sub)
        y_all.extend(y_sub)
        sj_ids.extend([sj] * len(X_sub))
        print(f"    sj{sj:02d} {cond}: {len(X_sub)} from {emb_base}")

    if not X_all:
        return None, None, None, label_names, trainable_df.iloc[0:0]

    return np.vstack(X_all), np.array(y_all), np.array(sj_ids), label_names, trainable_df


def encode_clip_from_crops(trainable_df, batch_size: int = 64, use_cache: bool = True):
    """Encode CLIP embeddings directly from data/crops/ PNGs."""
    import cv2
    from vision.classifier import GazeClassifier

    label_names = list(CATEGORIES.keys())
    label_to_idx = {n: i for i, n in enumerate(label_names)}

    df = trainable_df[trainable_df["human_label"].isin(label_to_idx)].copy()
    if use_cache and os.path.isfile(CLIP_EMB_CACHE):
        cached = np.load(CLIP_EMB_CACHE, allow_pickle=True)
        n_cached = len(cached["y"])
        if n_cached == len(df):
            print(f"  CLIP: loaded cached embeddings ({n_cached}) → {CLIP_EMB_CACHE}")
            return (
                cached["X"],
                cached["y"],
                cached["subject_id"],
                label_names,
                df,
            )
        print(f"  CLIP: cache stale ({n_cached} vs {len(df)} labels) — re-encoding")

    crops_rgb, y_list, sj_list, kept_rows = [], [], [], []
    missing = 0
    for _, row in df.iterrows():
        path = get_crop_path(
            int(row["subject_id"]), str(row["condition"]), str(row["filename"])
        )
        if not os.path.isfile(path):
            missing += 1
            continue
        img = cv2.imread(path)
        if img is None:
            missing += 1
            continue
        crops_rgb.append(img[:, :, ::-1].copy())
        y_list.append(label_to_idx[row["human_label"]])
        sj_list.append(int(row["subject_id"]))
        kept_rows.append(row)

    if len(crops_rgb) < 20:
        print(f"  CLIP: only {len(crops_rgb)} readable crops (missing {missing})")
        return None, None, None, label_names, df.iloc[0:0]

    print(
        f"  CLIP: encoding {len(crops_rgb)} crops with CLIP ViT-B/32 "
        f"(batch={batch_size})…"
    )
    if missing:
        print(f"    ({missing} labeled crops missing PNG files)")

    classifier = GazeClassifier()
    X = classifier.extract_embeddings_batch(crops_rgb, batch_size=batch_size)

    y_arr = np.array(y_list, dtype=np.int64)
    sj_arr = np.array(sj_list, dtype=np.int64)

    os.makedirs(MODELS_DIR, exist_ok=True)
    np.savez(
        CLIP_EMB_CACHE,
        X=X,
        y=y_arr,
        subject_id=sj_arr,
    )
    print(f"  CLIP: cached embeddings → {CLIP_EMB_CACHE}")

    filtered_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    return X, y_arr, sj_arr, label_names, filtered_df


def load_clip_training_data(trainable_df, *, batch_size: int = 64, use_cache: bool = True):
    """Prefer runs/*/vision embeddings; fall back to encoding crops."""
    emb_lookup = find_all_embeddings()
    if emb_lookup:
        print(f"  CLIP: found precomputed embeddings for {len(emb_lookup)} pair(s)")
        pooled = pool_clip_from_runs(trainable_df, emb_lookup)
        if pooled[0] is not None:
            return (*pooled, trainable_df)

    no_vis = _count_runs_without_vision()
    if no_vis:
        print(
            f"  CLIP: {no_vis} run folder(s) under runs/ have no vision/ subfolder "
            "(EEG-only runs). Old embedding .npy files may have been deleted."
        )
    print("  CLIP: encoding from data/crops/ instead (no *_embeddings.npy found)")
    return encode_clip_from_crops(
        trainable_df, batch_size=batch_size, use_cache=use_cache
    )


def train_clip_head(
    *,
    strategy: str = "within_subject",
    test_subject: int | None = None,
    n_epochs: int = 200,
    lr: float = 0.01,
    clip_batch_size: int = 64,
    use_cache: bool = True,
) -> str | None:
    from vision.train_head import (
        save_head,
        save_head_versioned,
        train_with_holdout,
    )

    trainable = load_trainable_labels()
    if len(trainable) < 20:
        print(f"  CLIP: only {len(trainable)} labels — need ≥20")
        return None

    X, y, sj_arr, label_names, _df = load_clip_training_data(
        trainable, batch_size=clip_batch_size, use_cache=use_cache
    )
    if X is None:
        return None

    print(f"  CLIP: training linear head on {len(X)} samples")

    subject_ids = sj_arr if strategy == "cross_subject" else None
    model, stats, _split = train_with_holdout(
        X, y, label_names,
        subject_ids=subject_ids,
        strategy=strategy,
        test_subject=test_subject,
        n_epochs=n_epochs,
        lr=lr,
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    versioned = save_head_versioned(model, stats, MODELS_DIR, prefix="clip_head")
    save_head(model, stats, CLIP_HEAD_PATH)
    print(
        f"  CLIP done — val acc {stats['best_val_acc']:.3f}, "
        f"test acc {stats['test_acc']:.3f}"
    )
    print(f"  → {CLIP_HEAD_PATH}")
    print(f"  → {versioned}")
    return CLIP_HEAD_PATH


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
        print(f"  ResNet: only {len(trainable)} labels — need ≥20")
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
    print(f"  → {RESNET_PATH}")
    print(f"  → {versioned}")
    return RESNET_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Train CLIP head + ResNet-50 for model comparison"
    )
    parser.add_argument("--clip-only", action="store_true")
    parser.add_argument("--resnet-only", action="store_true")
    parser.add_argument("--clip-epochs", type=int, default=200)
    parser.add_argument("--resnet-epochs", type=int, default=30)
    parser.add_argument("--resnet-batch-size", type=int, default=32)
    parser.add_argument("--clip-batch-size", type=int, default=64,
                        help="Batch size when encoding CLIP from crops")
    parser.add_argument("--no-emb-cache", action="store_true",
                        help="Re-encode CLIP even if models/labeled_clip_embeddings.npz exists")
    parser.add_argument("--lr-clip", type=float, default=0.01)
    parser.add_argument("--lr-resnet", type=float, default=1e-4)
    parser.add_argument(
        "--strategy", default="within_subject",
        choices=["within_subject", "cross_subject"],
    )
    parser.add_argument("--test-subject", type=int, default=None,
                        help="Hold-out subject for cross_subject CLIP split")
    args = parser.parse_args()

    do_clip = not args.resnet_only
    do_resnet = not args.clip_only

    print("=" * 60)
    print("  TRAIN VISION MODELS (label store → models/)")
    print(f"  Runs search: {_all_runs_roots()}")
    print(f"  Labels:      {PROJECT_ROOT}/data/human_labels.csv")
    print("=" * 60)

    paths = []
    if do_clip:
        print("\n--- CLIP linear head ---")
        p = train_clip_head(
            strategy=args.strategy,
            test_subject=args.test_subject,
            n_epochs=args.clip_epochs,
            lr=args.lr_clip,
            clip_batch_size=args.clip_batch_size,
            use_cache=not args.no_emb_cache,
        )
        if p:
            paths.append(p)

    if do_resnet:
        print("\n--- ResNet-50 ---")
        p = train_resnet(
            n_epochs=args.resnet_epochs,
            lr=args.lr_resnet,
            batch_size=args.resnet_batch_size,
        )
        if p:
            paths.append(p)

    print("\n" + "=" * 60)
    if paths:
        print("  Saved:")
        for p in paths:
            print(f"    {p}")
        print("\n  Next: run model comparison")
        print("    run_eval_local.bat")
        print("  Or open Streamlit → Evaluate tab")
    else:
        print("  No models saved — check errors above.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
