"""
PSY197B — Unified Evaluation Pipeline
======================================
Thesis-ready comparison tables and confusion matrix heatmaps.

Vision:  Fine-tuned ResNet-50 evaluation
EEG:     Leave-One-Subject-Out (LOSO) — EEGNet vs RawGazeFusionNet

Usage:
    python src/evaluate.py                    # full evaluation
    python src/evaluate.py --vision-only      # vision comparison only
    python src/evaluate.py --eeg-only         # EEG LOSO only
    python src/evaluate.py --n-repeats 3      # fewer CV repeats (faster)

"""

import argparse
import gc
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

RESULTS_DIR = os.environ.get("PSY197B_RESULTS_DIR") or os.path.join(PROJECT_ROOT, "results")
RESULTS_DIR = os.path.abspath(RESULTS_DIR)
RUNS_ROOT = os.environ.get("PSY197B_RUNS_DIR") or os.path.join(PROJECT_ROOT, "runs")
RUNS_ROOT = os.path.abspath(RUNS_ROOT)


# ═══════════════════════════════════════════════════════════
#  FOCAL LOSS
# ═══════════════════════════════════════════════════════════

def _make_train_loss(y, n_classes, *, use_sampler: bool = False):
    """Mild imbalance handling — avoids suppressing trail_ground."""
    from vision.class_balance import make_classification_loss

    return make_classification_loss(
        y, n_classes, use_sampler=use_sampler, loss_type="ce", max_boost=6.0
    )


# ═══════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_vision_data():
    """Load labeled vision data: crop paths + labels for ResNet evaluation.

    Returns dict with keys: y, crop_records, label_names, label_dist
    or None if insufficient data.
    """
    from vision.label_store import load_trainable_labels, get_crop_path
    from vision.config import CATEGORIES

    labels_df = load_trainable_labels()
    label_names = list(CATEGORIES.keys())
    label_to_idx = {n: i for i, n in enumerate(label_names)}

    labels_df = labels_df[
        labels_df["human_label"].isin(label_to_idx)
    ].reset_index(drop=True)

    if len(labels_df) < 20:
        print(f"  Only {len(labels_df)} usable labels — need at least 20")
        return None

    y = np.array([label_to_idx[l] for l in labels_df["human_label"]])
    label_dist = {label_names[i]: int(c)
                  for i, c in enumerate(np.bincount(y, minlength=len(label_names)))}
    print(f"  Labels loaded: {len(y)} samples across {len(label_names)} classes")
    for name, count in label_dist.items():
        flag = " (WARNING: <5)" if count < 5 else ""
        print(f"    {name:15s} {count:5d}{flag}")

    crop_records = []
    for _, row in labels_df.iterrows():
        crop_records.append({
            "sj_num": int(row["subject_id"]),
            "condition": str(row["condition"]),
            "filename": str(row["filename"]),
            "label_idx": label_to_idx[row["human_label"]],
        })

    return {
        "y": y,
        "crop_records": crop_records,
        "label_names": label_names,
        "label_dist": label_dist,
        "labels_df": labels_df,
    }


# ═══════════════════════════════════════════════════════════
#  VISION EVALUATION
# ═══════════════════════════════════════════════════════════

def _stratified_split(y, test_size=0.15, val_size=0.15, seed=42):
    """Stratified train/val/test split. Returns index arrays."""
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(np.zeros(len(y)), y))
    val_frac = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_frac,
                                    random_state=seed)
    rel_train, rel_val = next(
        sss_val.split(np.zeros(len(train_val_idx)), y[train_val_idx])
    )
    return train_val_idx[rel_train], train_val_idx[rel_val], test_idx


def _eval_metrics(y_true, y_pred, n_classes, label_names):
    """Compute full metric suite."""
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_names = [label_names[c] for c in present]
    report = classification_report(
        y_true, y_pred, labels=present, target_names=present_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(n_classes)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred,
                                                 average="macro",
                                                 zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro",
                                           zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }


def _eval_resnet(crop_records, y, train_idx, val_idx, test_idx,
                 label_names, n_epochs=30, lr=1e-4, batch_size=32):
    """Train ResNet-50 and evaluate on test set."""
    from vision.resnet_head import train_resnet, CropDataset, _get_transforms

    n_classes = len(label_names)
    train_records = [crop_records[i] for i in train_idx]
    val_records = [crop_records[i] for i in val_idx]
    test_records = [crop_records[i] for i in test_idx]

    model, stats = train_resnet(
        train_records, val_records, label_names,
        n_epochs=n_epochs, lr=lr, batch_size=batch_size,
        loss_fn=None,
    )

    _eval_device = _get_eval_device()
    _pin = _eval_device.type == "cuda"
    from device_utils import (
        dataloader_workers as _dl_workers,
        persistent_workers as _persistent,
        prefetch_factor as _prefetch,
        use_amp as _use_amp,
        use_channels_last as _channels_last,
        make_autocast,
    )
    _nw = _dl_workers()
    _pw = _persistent() and _nw > 0
    _pf = {"prefetch_factor": _prefetch()} if _nw > 0 else {}
    _amp = _use_amp()
    _cl = _channels_last()
    _, val_tfm = _get_transforms()
    test_ds = CropDataset(test_records, transform=val_tfm)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=_nw, pin_memory=_pin,
                             persistent_workers=_pw, **_pf)
    model = model.to(_eval_device)
    if _cl:
        model = model.to(memory_format=torch.channels_last)
    model.eval()
    preds_list = []
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(_eval_device, non_blocking=_pin)
            if _cl:
                imgs = imgs.to(memory_format=torch.channels_last)
            with make_autocast(_eval_device, enabled=_amp):
                logits = model(imgs)
            preds_list.extend(logits.argmax(1).cpu().numpy())

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return np.array(preds_list)


def _get_eval_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_vision_models(
    n_repeats=5, seed=42, progress_cb=None,
    resnet_epochs=30, resnet_batch_size=64,
):
    """Evaluate ResNet-50 gaze crop classifier.

    Returns dict with per-model results, comparison table data, and
    confusion matrices.
    """
    from device_utils import dataloader_workers, device_summary, print_device_banner

    print_device_banner(prefix="  ")
    _nw = dataloader_workers()
    print(
        f"  ResNet eval: batch_size={resnet_batch_size}, epochs={resnet_epochs}, "
        f"DataLoader workers={_nw} (RAM-capped on Windows; set PSY_DATALOADER_WORKERS to override)"
    )

    data = load_vision_data()
    if data is None:
        return {"error": "insufficient_data"}

    y = data["y"]
    n_classes = len(data["label_names"])
    label_names = data["label_names"]
    crop_records = data["crop_records"]

    model_results = {
        "resnet50": [],
    }
    aggregate_cms = {name: np.zeros((n_classes, n_classes), dtype=int)
                     for name in model_results}

    total_steps = n_repeats
    step = 0

    for rep in range(n_repeats):
        rep_seed = seed + rep
        print(f"\n  Repeat {rep + 1}/{n_repeats} (seed={rep_seed})")

        try:
            train_idx, val_idx, test_idx = _stratified_split(
                y, seed=rep_seed)
        except ValueError as e:
            print(f"    Split failed: {e} — skipping repeat")
            continue

        test_y = y[test_idx]

        # ResNet-50
        if progress_cb:
            step += 1
            progress_cb(step, total_steps, "ResNet-50")
        print("    ResNet-50...")
        rn_preds = _eval_resnet(
            crop_records, y, train_idx, val_idx, test_idx, label_names,
            n_epochs=resnet_epochs, batch_size=resnet_batch_size,
        )
        rn_metrics = _eval_metrics(test_y, rn_preds, n_classes, label_names)
        model_results["resnet50"].append(rn_metrics)
        aggregate_cms["resnet50"] += np.array(rn_metrics["confusion_matrix"])
        print(f"      macro F1={rn_metrics['macro_f1']:.3f}  "
              f"acc={rn_metrics['accuracy']:.3f}")

    summary = {}
    for model_name, repeats in model_results.items():
        if not repeats:
            continue
        metrics_keys = ["accuracy", "macro_f1", "macro_precision",
                        "macro_recall", "weighted_f1"]
        s = {}
        for key in metrics_keys:
            vals = [r[key] for r in repeats]
            s[f"{key}_mean"] = float(np.mean(vals))
            s[f"{key}_std"] = float(np.std(vals))
        s["confusion_matrix"] = aggregate_cms[model_name].tolist()
        s["n_repeats"] = len(repeats)
        summary[model_name] = s

    best_model = max(
        (k for k in summary if summary[k].get("n_repeats", 0) > 0),
        key=lambda k: summary[k]["macro_f1_mean"],
        default=None,
    )
    if best_model:
        summary["best_model"] = best_model
        print(f"\n  Best vision model: {best_model} "
              f"(macro F1 = {summary[best_model]['macro_f1_mean']:.3f})")

    result = {
        "summary": summary,
        "label_names": label_names,
        "label_dist": data["label_dist"],
        "n_repeats": n_repeats,
        "seed": seed,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "vision_comparison.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    _save_vision_tables(result)
    _save_vision_plots(result)

    return result


# ═══════════════════════════════════════════════════════════
#  EEG LOSO EVALUATION
# ═══════════════════════════════════════════════════════════

def _load_loso_multimodal(run_dirs, walk_only=True):
    """Load nogo EEG + ET data per subject for LOSO.

    Returns (X_eeg, X_et, labels, sids) or (None,)*4.
    """
    from train import (discover_conditions, load_tensors)

    X_eeg_parts, X_et_parts, label_parts, sid_parts = [], [], [], []

    for sj, rn in run_dirs.items():
        conditions = discover_conditions(rn)
        if walk_only:
            conditions = [c for c in conditions if "walk" in c.lower()]
        if not conditions:
            print(f"  Subject {sj}: no matching conditions in {rn}")
            continue

        sj_X_eeg, sj_X_et, sj_y, sj_meta = [], [], [], []
        for cond in conditions:
            data = load_tensors(rn, cond)
            if data is None:
                continue
            for split in ("train", "val"):
                sj_X_eeg.append(data[f"X_eeg_{split}"])
                sj_y.append(data[f"y_{split}"])
                if f"X_et_{split}" in data:
                    sj_X_et.append(data[f"X_et_{split}"])
                meta = data[f"meta_{split}"]
                meta = meta.copy()
                meta["_condition"] = cond
                sj_meta.append(meta)

        if not sj_X_eeg:
            continue

        X_eeg_cat = np.concatenate(sj_X_eeg)
        y_cat = np.concatenate(sj_y)
        meta_cat = pd.concat(sj_meta, ignore_index=True)

        has_et = len(sj_X_et) == len(sj_X_eeg)
        if has_et:
            X_et_cat = np.concatenate(sj_X_et)
        else:
            X_et_cat = None

        if "outcome" not in meta_cat.columns:
            print(f"  Subject {sj}: no outcome column")
            continue

        nogo_mask = meta_cat.get("trialType",
                                 pd.Series(dtype=float)).values == 20
        if not nogo_mask.any():
            nogo_mask = y_cat == 1
        outcome = meta_cat["outcome"].astype(str).str.upper()
        cr_fa = nogo_mask & outcome.isin(
            ["CORRECT_REJECTION", "COMMISSION_ERROR"])
        idx = np.where(cr_fa)[0]

        if len(idx) < 4:
            print(f"  Subject {sj}: only {len(idx)} nogo trials — skipping")
            continue

        labels = (meta_cat.iloc[idx]["outcome"].str.upper()
                  == "CORRECT_REJECTION").astype(int).values
        n_cr = int(labels.sum())
        n_fa = int(len(labels) - labels.sum())

        X_eeg_parts.append(X_eeg_cat[idx])
        if has_et:
            X_et_parts.append(X_et_cat[idx])
        label_parts.append(labels)
        sid_parts.append(np.full(len(idx), sj, dtype=int))
        print(f"  Subject {sj}: {len(idx)} nogo trials "
              f"(CR={n_cr}, FA={n_fa})")

    if not X_eeg_parts:
        return None, None, None, None

    X_eeg = np.concatenate(X_eeg_parts)
    labels = np.concatenate(label_parts)
    sids = np.concatenate(sid_parts)

    X_et = None
    if X_et_parts and len(X_et_parts) == len(X_eeg_parts):
        et_shapes = set(x.shape[1] for x in X_et_parts)
        if len(et_shapes) == 1:
            X_et = np.concatenate(X_et_parts)
        else:
            min_ch = min(et_shapes)
            X_et = np.concatenate([x[:, :min_ch] for x in X_et_parts])
            print(f"  ET channel mismatch {et_shapes} — truncated to {min_ch}")

    return X_eeg, X_et, labels, sids


def evaluate_eeg_loso(run_dirs=None, progress_cb=None):
    """LOSO cross-validation: EEGNet vs RawGazeFusionNet.

    Returns dict with per-fold and summary results.
    """
    from train import (
        EEGNet, RawGazeFusionNet, _eegnet_kwargs, _training_kwargs,
        load_model_config, _loso_train_eval, _normalize_cross_subject,
    )

    cfg = load_model_config()
    ekw = _eegnet_kwargs(cfg)
    tkw = _training_kwargs(cfg)

    if run_dirs is None:
        loso_cfg = cfg.get("loso", {})
        run_dirs = loso_cfg.get("run_dirs", {})
        run_dirs = {int(k): v for k, v in run_dirs.items()}

    if not run_dirs:
        print("  No LOSO run_dirs configured")
        return {"error": "no_loso_config"}

    print(f"  Loading data for subjects: {sorted(run_dirs.keys())}")
    X_eeg, X_et, labels, sids = _load_loso_multimodal(run_dirs)
    if X_eeg is None:
        return {"error": "no_data"}

    has_et = X_et is not None
    n_ch, n_t = X_eeg.shape[1], X_eeg.shape[2]
    n_et_ch = X_et.shape[1] if has_et else 0
    subjects = sorted(np.unique(sids))

    print(f"\n  Total: {len(labels)} nogo trials across {len(subjects)} subjects")
    print(f"  EEG: {n_ch} ch × {n_t} t")
    if has_et:
        print(f"  ET:  {n_et_ch} ch × {n_t} t")
    else:
        print("  ET:  not available — multimodal comparison will be skipped")

    models_to_run = {"eeg_only": True, "multimodal": has_et}
    fold_results = {"eeg_only": [], "multimodal": []}

    total_steps = len(subjects) * sum(models_to_run.values())
    step = 0

    for held_out in subjects:
        print(f"\n  --- Held out: subject {held_out} ---")
        train_mask = sids != held_out
        test_mask = sids == held_out
        y_tr, y_te = labels[train_mask], labels[test_mask]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            print(f"    Skipped — single class in split")
            continue

        X_eeg_tr = X_eeg[train_mask].copy()
        X_eeg_te = X_eeg[test_mask].copy()
        X_eeg_tr, X_eeg_te = _normalize_cross_subject(X_eeg_tr, X_eeg_te)

        # Model A: EEGNet (EEG-only)
        if progress_cb:
            step += 1
            progress_cb(step, total_steps, f"EEGNet — held out sj{held_out}")
        print(f"    EEGNet (EEG-only)...")
        r_eeg = _loso_train_eval(
            lambda: EEGNet(n_ch, n_t, 2, **ekw),
            X_eeg_tr, y_tr, X_eeg_te, y_te, **tkw,
        )
        if r_eeg:
            r_eeg["held_out_subject"] = int(held_out)
            r_eeg.pop("_best_state", None)
            fold_results["eeg_only"].append(r_eeg)
            print(f"      bal_acc={r_eeg['balanced_accuracy']:.3f}  "
                  f"AUC={r_eeg['auc_roc']:.3f}  F1={r_eeg['f1']:.3f}")

        # Model B: RawGazeFusionNet (EEG+ET)
        if has_et:
            if progress_cb:
                step += 1
                progress_cb(step, total_steps,
                            f"RawGazeFusionNet — held out sj{held_out}")
            print(f"    RawGazeFusionNet (EEG+ET)...")
            X_et_tr = X_et[train_mask].copy()
            X_et_te = X_et[test_mask].copy()
            X_et_tr, X_et_te = _normalize_cross_subject(X_et_tr, X_et_te)

            r_mm = _loso_train_eval(
                lambda: RawGazeFusionNet(n_ch, n_et_ch, n_t, 2, **ekw),
                X_eeg_tr, y_tr, X_eeg_te, y_te,
                X_gaze_train=X_et_tr, X_gaze_test=X_et_te,
                is_fusion=True, **tkw,
            )
            if r_mm:
                r_mm["held_out_subject"] = int(held_out)
                r_mm.pop("_best_state", None)
                fold_results["multimodal"].append(r_mm)
                print(f"      bal_acc={r_mm['balanced_accuracy']:.3f}  "
                      f"AUC={r_mm['auc_roc']:.3f}  F1={r_mm['f1']:.3f}")

        del X_eeg_tr, X_eeg_te
        gc.collect()

    summary = {}
    for model_name, folds in fold_results.items():
        if not folds:
            continue
        s = {"n_folds": len(folds)}
        for key in ["balanced_accuracy", "auc_roc", "f1",
                     "precision", "recall"]:
            vals = [r[key] for r in folds]
            s[f"{key}_mean"] = float(np.mean(vals))
            s[f"{key}_std"] = float(np.std(vals))
            s[f"{key}_values"] = vals
        cms = [np.array(r["confusion_matrix"]) for r in folds]
        s["confusion_matrix_sum"] = np.sum(cms, axis=0).tolist()
        summary[model_name] = s

    print(f"\n  LOSO Summary:")
    for model_name, s in summary.items():
        print(f"    {model_name:15s}  bal_acc={s['balanced_accuracy_mean']:.3f} "
              f"± {s['balanced_accuracy_std']:.3f}  "
              f"AUC={s['auc_roc_mean']:.3f} ± {s['auc_roc_std']:.3f}  "
              f"F1={s['f1_mean']:.3f} ± {s['f1_std']:.3f}")

    result = {
        "summary": summary,
        "fold_results": {k: [{kk: vv for kk, vv in r.items()
                              if kk != "_best_state"}
                             for r in v]
                         for k, v in fold_results.items()},
        "subjects": [int(s) for s in subjects],
        "n_eeg_ch": n_ch,
        "n_et_ch": n_et_ch,
        "n_times": n_t,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "eeg_loso_comparison.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    _save_eeg_tables(result)
    _save_eeg_plots(result)

    return result


# ═══════════════════════════════════════════════════════════
#  RELABEL CROPS WITH BEST MODEL
# ═══════════════════════════════════════════════════════════

def _human_label_rows_for_condition(sj_num, condition, label_names):
    """Fixation-level rows from human_labels.csv (confidence=1)."""
    from vision.label_store import load_trainable_labels

    df = load_trainable_labels()
    if df.empty:
        return []
    sub = df[
        (df["subject_id"] == int(sj_num)) & (df["condition"] == condition)
    ]
    records = []
    for row in sub.itertuples(index=False):
        label = str(row.human_label)
        if label not in label_names:
            continue
        scores = {n: 0.0 for n in label_names}
        scores[label] = 1.0
        records.append({
            "fixation_id": int(row.fixation_id),
            "timestamp_ns": int(row.timestamp_ns) if row.timestamp_ns else 0,
            "gaze_target_category": label,
            "confidence": 1.0,
            **{f"score_{n}": scores[n] for n in label_names},
        })
    return records


def _merge_vision_results(existing_df, new_df, human_df, label_names):
    """Combine prior CSV, new model preds, and human labels (human wins)."""
    parts = []
    if existing_df is not None and not existing_df.empty:
        parts.append(existing_df.copy())
    if new_df is not None and not new_df.empty:
        parts.append(new_df.copy())
    if human_df is not None and not human_df.empty:
        parts.append(human_df.copy())

    if not parts:
        return pd.DataFrame()

    merged = pd.concat(parts, ignore_index=True)
    if "fixation_id" not in merged.columns:
        return merged

    # Last row wins per fixation: human rows appended last in caller order.
    merged = merged.drop_duplicates(subset=["fixation_id"], keep="last")
    return merged.sort_values("fixation_id").reset_index(drop=True)


def relabel_crops_with_best(best_model_name, run_name=None,
                            progress_cb=None, output_dir=None,
                            only_unlabeled=True):
    """Classify gaze crops with the best vision model and regenerate fusion features.

    Args:
        best_model_name: "resnet50"
        run_name: target run directory (default: latest). Ignored if
            output_dir is provided.
        output_dir: standalone output directory for results and features.
            If provided, writes all CSVs here instead of into a run.
        only_unlabeled: if True (default), skip crops that already have a
            trainable human label in data/human_labels.csv. Human labels
            are always written into the results CSV for trial aggregation.

    Returns dict with relabeling summary.
    """
    from vision.config import CATEGORIES
    from vision.label_store import CROPS_BASE, trainable_labeled_crop_keys

    label_names = list(CATEGORIES.keys())

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        run_path = None
        vision_root = None
        run_name = None
    else:
        if run_name is None:
            runs = sorted(
                [d for d in os.listdir(RUNS_ROOT)
                 if os.path.isdir(os.path.join(RUNS_ROOT, d))],
                reverse=True,
            )
            if not runs:
                return {"error": "no_runs"}
            run_name = runs[0]

        run_path = os.path.join(RUNS_ROOT, run_name)
        vision_root = os.path.join(run_path, "vision")

    if best_model_name == "resnet50":
        from vision.resnet_head import load_resnet, _get_transforms
        resnet_path = os.path.join(PROJECT_ROOT, "models", "resnet50_best.pt")
        if not os.path.exists(resnet_path):
            ckpts = sorted(
                [f for f in os.listdir(os.path.join(PROJECT_ROOT, "models"))
                 if f.startswith("resnet") and f.endswith(".pt")],
                reverse=True,
            )
            if ckpts:
                resnet_path = os.path.join(PROJECT_ROOT, "models", ckpts[0])
            else:
                return {"error": "no_resnet_checkpoint"}
        model, _ = load_resnet(resnet_path)
        model.eval()
        _, val_tfm = _get_transforms()
        use_resnet = True
    else:
        return {"error": f"unknown_model: {best_model_name}"}

    import cv2
    from scipy.stats import entropy as sp_entropy

    labeled_keys = trainable_labeled_crop_keys() if only_unlabeled else set()
    total_relabeled = 0
    total_skipped_human = 0
    condition_summaries = []

    crop_dirs = []
    if os.path.isdir(CROPS_BASE):
        for d in sorted(os.listdir(CROPS_BASE)):
            if os.path.isdir(os.path.join(CROPS_BASE, d)):
                crop_dirs.append(d)

    for ci, crop_dir_name in enumerate(crop_dirs):
        if progress_cb:
            progress_cb(ci + 1, len(crop_dirs),
                        f"Relabeling {crop_dir_name}")

        parts = crop_dir_name.split("_", 1)
        if len(parts) < 2:
            continue
        sj_str, condition = parts[0], parts[1]
        try:
            sj_num = int(sj_str.replace("sj", ""))
        except ValueError:
            continue

        crop_path = os.path.join(CROPS_BASE, crop_dir_name)
        all_crop_files = sorted(f for f in os.listdir(crop_path)
                                if f.endswith(".png"))
        if not all_crop_files:
            continue

        if only_unlabeled:
            crop_files = [
                cf for cf in all_crop_files
                if (sj_num, condition, cf) not in labeled_keys
            ]
            n_skip = len(all_crop_files) - len(crop_files)
            total_skipped_human += n_skip
            if n_skip:
                print(f"    {crop_dir_name}: skip {n_skip} human-labeled crop(s)")
        else:
            crop_files = all_crop_files

        crops_rgb = []
        crop_meta = []
        for cf in crop_files:
            img = cv2.imread(os.path.join(crop_path, cf))
            if img is None:
                continue
            crops_rgb.append(img[:, :, ::-1].copy())
            fparts = cf.replace(".png", "").split("_")
            crop_meta.append({
                "fixation_id": int(fparts[0]),
                "timestamp_ns": int(fparts[1]),
            })

        if use_resnet and crops_rgb:
            from PIL import Image as PILImage
            preds = []
            for crop in crops_rgb:
                pil = PILImage.fromarray(crop)
                inp = val_tfm(pil).unsqueeze(0)
                with torch.no_grad():
                    logits = model(inp)
                pred_idx = logits.argmax(1).item()
                probs = torch.softmax(logits, 1).squeeze(0).numpy()
                preds.append({
                    "label": label_names[pred_idx],
                    "confidence": float(probs[pred_idx]),
                    "all_scores": {n: float(p)
                                   for n, p in zip(label_names, probs)},
                })
        elif crops_rgb:
            batch_results = classifier.classify_batch(crops_rgb, batch_size=64)
            preds = batch_results
        else:
            preds = []

        records = []
        for meta_row, res in zip(crop_meta, preds):
            rec = {
                "fixation_id": meta_row["fixation_id"],
                "timestamp_ns": meta_row["timestamp_ns"],
                "gaze_target_category": res["label"],
                "confidence": res["confidence"],
            }
            for cat_label, score in res["all_scores"].items():
                rec[f"score_{cat_label}"] = score
            records.append(rec)

        new_results_df = pd.DataFrame(records)
        total_relabeled += len(new_results_df)

        if output_dir:
            csv_path = os.path.join(
                output_dir,
                f"sj{sj_num:02d}_{condition}_vision_results.csv")
            feat_path = os.path.join(
                output_dir,
                f"sj{sj_num:02d}_{condition}_vision_trial_features.csv")
            et_path = None
            existing_df = None
            if os.path.exists(csv_path):
                existing_df = pd.read_csv(csv_path)
        else:
            vis_dir = os.path.join(vision_root,
                                   f"sj{sj_num:02d}_{condition}")
            csv_path = None
            feat_path = None
            et_path = None
            existing_df = None
            if os.path.isdir(vis_dir):
                csv_path = os.path.join(
                    vis_dir,
                    f"sj{sj_num:02d}_{condition}_vision_results.csv")
                feat_path = os.path.join(
                    run_path, "data",
                    f"sj{sj_num:02d}_{condition}_vision_trial_features.csv")
                if os.path.exists(csv_path):
                    existing_df = pd.read_csv(csv_path)
            data_dir = os.path.join(run_path, "data")
            et_path = os.path.join(
                data_dir, f"sj{sj_num:02d}_{condition}_ET_Prepro1.csv")

        human_records = _human_label_rows_for_condition(
            sj_num, condition, label_names)
        human_df = pd.DataFrame(human_records)

        results_df = _merge_vision_results(
            existing_df, new_results_df, human_df, label_names)

        if not results_df.empty and (
            "timestamp_ns" in results_df.columns
            and "timestamp_s" not in results_df.columns
        ):
            try:
                from vision.config import get_eye_dir, ET_FOLDER_MAP
                from vision.label_store import data_root_from_config
                _dr = data_root_from_config()
                if _dr and condition in ET_FOLDER_MAP:
                    _gaze_csv = os.path.join(
                        get_eye_dir(_dr, sj_num, condition),
                        "gaze_positions.csv")
                    if os.path.exists(_gaze_csv):
                        _gaze_t0 = pd.read_csv(
                            _gaze_csv, usecols=["timestamp [ns]"], nrows=1
                        )["timestamp [ns]"].iloc[0]
                        results_df["timestamp_s"] = (
                            results_df["timestamp_ns"] - _gaze_t0
                        ) / 1e9
            except Exception:
                pass

        if csv_path and not results_df.empty:
            results_df.to_csv(csv_path, index=False)
        elif csv_path and results_df.empty and os.path.exists(csv_path):
            results_df = pd.read_csv(csv_path)

        if et_path is None and output_dir and os.path.isdir(RUNS_ROOT):
            for _run in sorted(
                (d for d in os.listdir(RUNS_ROOT)
                 if os.path.isdir(os.path.join(RUNS_ROOT, d))),
                reverse=True,
            ):
                _cand = os.path.join(
                    RUNS_ROOT, _run, "data",
                    f"sj{sj_num:02d}_{condition}_ET_Prepro1.csv")
                if os.path.exists(_cand):
                    et_path = _cand
                    break

        if et_path and os.path.exists(et_path) and "timestamp_s" in results_df.columns:
            et_df = pd.read_csv(et_path)
            cat_labels = label_names
            trial_records = []
            for _, trial_row in et_df.iterrows():
                t_idx = trial_row["trialIdx"]
                t_time = trial_row["trigger_time"]
                window = results_df[
                    (results_df["timestamp_s"] >= t_time - 1.0)
                    & (results_df["timestamp_s"] <= t_time + 1.0)
                ]
                rec = {"trialIdx": t_idx}
                if window.empty:
                    rec["most_common_category"] = np.nan
                    rec["mean_confidence"] = np.nan
                    rec["category_entropy"] = np.nan
                    for c in cat_labels:
                        rec[f"vis_prop_{c}"] = np.nan
                else:
                    counts = window["gaze_target_category"].value_counts()
                    rec["most_common_category"] = counts.index[0]
                    rec["mean_confidence"] = float(
                        window["confidence"].mean())
                    count_arr = np.array(
                        [counts.get(c, 0) for c in cat_labels], dtype=float)
                    total = count_arr.sum()
                    if total > 0:
                        probs = count_arr / total
                        rec["category_entropy"] = float(
                            sp_entropy(probs, base=2))
                    else:
                        rec["category_entropy"] = np.nan
                    for c in cat_labels:
                        rec[f"vis_prop_{c}"] = (
                            counts.get(c, 0) / total if total > 0
                            else np.nan)
                trial_records.append(rec)
            fusion_df = pd.DataFrame(trial_records)
            fusion_df.to_csv(feat_path, index=False)

        if not results_df.empty:
            dist = results_df["gaze_target_category"].value_counts().to_dict()
        else:
            dist = {}
        condition_summaries.append({
            "sj_num": sj_num, "condition": condition,
            "n_crops": len(results_df),
            "n_model_classified": len(new_results_df),
            "n_human_labeled": len(human_df),
            "distribution": dist,
        })

    if not use_resnet:
        del classifier
    else:
        del model
    gc.collect()

    return {
        "model": best_model_name,
        "total_relabeled": total_relabeled,
        "total_skipped_human_labeled": total_skipped_human,
        "only_unlabeled": only_unlabeled,
        "conditions": condition_summaries,
        "run_name": run_name,
    }


# ═══════════════════════════════════════════════════════════
#  CONFUSION MATRIX PLOTS
# ═══════════════════════════════════════════════════════════

def make_cm_figure(cm, labels, title="", figsize=None):
    """Create a thesis-quality confusion matrix heatmap.

    Returns matplotlib Figure object (for st.pyplot or savefig).
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n = len(labels)
    cm_arr = np.array(cm, dtype=float)

    row_sums = cm_arr.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm_arr / row_sums

    col_sums = cm_arr.sum(axis=0)
    col_sums_safe = np.where(col_sums == 0, 1, col_sums)
    precision = np.diag(cm_arr) / col_sums_safe
    recall = np.diag(cm_arr) / row_sums.flatten()
    total = cm_arr.sum()
    accuracy = np.diag(cm_arr).sum() / total if total > 0 else 0

    if figsize is None:
        figsize = (max(5, n * 0.9 + 2), max(4, n * 0.8 + 2))
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ext_norm = np.zeros((n + 1, n + 1))
    ext_norm[:n, :n] = cm_norm
    for i in range(n):
        ext_norm[n, i] = precision[i]
        ext_norm[i, n] = recall[i]
    ext_norm[n, n] = accuracy

    colors_low = np.array([0.98, 0.92, 0.90])
    colors_mid = np.array([1.0, 1.0, 1.0])
    colors_high = np.array([0.75, 0.92, 0.78])
    n_steps = 128
    cmap_colors = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        if t < 0.5:
            c = colors_low + (colors_mid - colors_low) * (t / 0.5)
        else:
            c = colors_mid + (colors_high - colors_mid) * ((t - 0.5) / 0.5)
        cmap_colors.append(c)
    cmap = mcolors.ListedColormap(cmap_colors)

    ax.imshow(ext_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for i in range(n + 1):
        for j in range(n + 1):
            val = ext_norm[i, j]
            text = f"{val * 100:.1f}%"
            color = "black"
            fontweight = "bold" if i == j and i < n else "normal"
            if i == n or j == n:
                fontweight = "bold"
                if i == n and j == n:
                    color = "#333333"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=max(7, 11 - n * 0.5),
                    fontweight=fontweight, color=color)

    extended_labels = list(labels) + ["precision"]
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(extended_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n + 1))
    ylabels = list(labels) + ["precision"]
    ylabels[-1] = ""
    ax.set_yticklabels(ylabels, fontsize=9)

    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)

    recall_label_x = n
    for i in range(n):
        if i == 0:
            ax.text(recall_label_x, -0.7, "recall", ha="center",
                    va="bottom", fontsize=8, fontstyle="italic")

    accuracy_text = f"accuracy"
    ax.text(n, n + 0.45, accuracy_text, ha="center", va="top",
            fontsize=7, fontstyle="italic")

    ax.axhline(n - 0.5, color="#999999", linewidth=1.5)
    ax.axvline(n - 0.5, color="#999999", linewidth=1.5)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    return fig


def _save_vision_plots(result):
    """Save vision confusion matrix plots."""
    import matplotlib.pyplot as plt

    labels = result["label_names"]
    summary = result["summary"]

    model_titles = {
        "resnet50": "Fine-tuned ResNet-50",
    }

    for model_name, s in summary.items():
        if model_name == "best_model" or "confusion_matrix" not in s:
            continue
        fig = make_cm_figure(s["confusion_matrix"], labels,
                             title=model_titles.get(model_name, model_name))
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(RESULTS_DIR, f"cm_{model_name}.{ext}"),
                        dpi=300, bbox_inches="tight")
        plt.close(fig)

    models_with_cm = [k for k in summary
                      if k != "best_model" and "confusion_matrix" in summary[k]]
    if len(models_with_cm) >= 2:
        n_panels = len(models_with_cm)
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(5 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]
        for ax, model_name in zip(axes, models_with_cm):
            cm = np.array(summary[model_name]["confusion_matrix"])
            n = len(labels)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_norm = cm / row_sums

            import matplotlib.colors as mcolors
            colors_low = np.array([0.98, 0.92, 0.90])
            colors_mid = np.array([1.0, 1.0, 1.0])
            colors_high = np.array([0.75, 0.92, 0.78])
            n_steps = 128
            cmap_colors = []
            for i in range(n_steps):
                t = i / (n_steps - 1)
                if t < 0.5:
                    c = colors_low + (colors_mid - colors_low) * (t / 0.5)
                else:
                    c = colors_mid + (colors_high - colors_mid) * (
                        (t - 0.5) / 0.5)
                cmap_colors.append(c)
            cmap = mcolors.ListedColormap(cmap_colors)

            ax.imshow(cm_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{cm_norm[i, j] * 100:.0f}%",
                            ha="center", va="center",
                            fontsize=max(6, 9 - n * 0.3),
                            fontweight="bold" if i == j else "normal")
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=7)
            f1 = summary[model_name].get("macro_f1_mean", 0)
            ax.set_title(
                f"{model_titles.get(model_name, model_name)}\n"
                f"Macro F1={f1:.3f}",
                fontsize=10, fontweight="bold")
        fig.suptitle("Vision Model Comparison", fontsize=13,
                     fontweight="bold", y=1.02)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(
                os.path.join(RESULTS_DIR, f"cm_vision_comparison.{ext}"),
                dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  Vision plots saved to {RESULTS_DIR}/")


def _save_eeg_plots(result):
    """Save EEG LOSO confusion matrix plots."""
    import matplotlib.pyplot as plt

    labels = ["FA", "CR"]
    summary = result["summary"]

    model_titles = {
        "eeg_only": "EEGNet (EEG-only)",
        "multimodal": "RawGazeFusionNet (EEG+ET)",
    }

    for model_name, s in summary.items():
        if "confusion_matrix_sum" not in s:
            continue
        fig = make_cm_figure(s["confusion_matrix_sum"], labels,
                             title=model_titles.get(model_name, model_name))
        for ext in ("png", "pdf"):
            fig.savefig(
                os.path.join(RESULTS_DIR, f"cm_{model_name}_loso.{ext}"),
                dpi=300, bbox_inches="tight")
        plt.close(fig)

    models_with_cm = [k for k in summary if "confusion_matrix_sum" in summary[k]]
    if len(models_with_cm) >= 2:
        n_panels = len(models_with_cm)
        fig, axes = plt.subplots(1, n_panels,
                                 figsize=(4.5 * n_panels, 4))
        if n_panels == 1:
            axes = [axes]
        for ax, model_name in zip(axes, models_with_cm):
            cm = np.array(summary[model_name]["confusion_matrix_sum"])
            n = len(labels)
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            cm_norm = cm / row_sums

            import matplotlib.colors as mcolors
            colors_low = np.array([0.98, 0.92, 0.90])
            colors_mid = np.array([1.0, 1.0, 1.0])
            colors_high = np.array([0.75, 0.92, 0.78])
            n_steps = 128
            cmap_colors = []
            for i in range(n_steps):
                t = i / (n_steps - 1)
                if t < 0.5:
                    c = colors_low + (colors_mid - colors_low) * (t / 0.5)
                else:
                    c = colors_mid + (colors_high - colors_mid) * (
                        (t - 0.5) / 0.5)
                cmap_colors.append(c)
            cmap = mcolors.ListedColormap(cmap_colors)

            ax.imshow(cm_norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{cm_norm[i, j] * 100:.1f}%",
                            ha="center", va="center",
                            fontsize=12, fontweight="bold" if i == j
                            else "normal")
            ax.set_xticks(range(n))
            ax.set_xticklabels(labels, fontsize=10)
            ax.set_yticks(range(n))
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("True", fontsize=10)
            f1 = summary[model_name].get("f1_mean", 0)
            ba = summary[model_name].get("balanced_accuracy_mean", 0)
            ax.set_title(
                f"{model_titles.get(model_name, model_name)}\n"
                f"Bal.Acc={ba:.3f}  F1={f1:.3f}",
                fontsize=10, fontweight="bold")
        fig.suptitle("EEG LOSO: EEG-only vs Multimodal", fontsize=13,
                     fontweight="bold", y=1.02)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(
                os.path.join(RESULTS_DIR, f"cm_eeg_comparison.{ext}"),
                dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  EEG plots saved to {RESULTS_DIR}/")


# ═══════════════════════════════════════════════════════════
#  TABLE GENERATION
# ═══════════════════════════════════════════════════════════

def generate_vision_table(result):
    """Generate vision comparison table as (LaTeX string, DataFrame)."""
    summary = result["summary"]
    model_names = {
        "resnet50": "Fine-tuned ResNet-50",
    }
    rows = []
    for key in ["resnet50"]:
        s = summary.get(key)
        if not s:
            continue
        rows.append({
            "Method": model_names.get(key, key),
            "Macro R": f"{s['macro_recall_mean']:.2f} ± {s['macro_recall_std']:.2f}",
            "Macro P": f"{s['macro_precision_mean']:.2f} ± {s['macro_precision_std']:.2f}",
            "Macro F1": f"{s['macro_f1_mean']:.2f} ± {s['macro_f1_std']:.2f}",
            "Weighted F1": f"{s['weighted_f1_mean']:.2f} ± {s['weighted_f1_std']:.2f}",
            "Accuracy": f"{s['accuracy_mean']:.2f} ± {s['accuracy_std']:.2f}",
        })

    df = pd.DataFrame(rows)

    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Vision pipeline comparison on gaze crop classification.}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Macro R & Macro P & Macro F1 & Weighted F1 & Accuracy \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        latex_lines.append(
            f"{row['Method']} & {row['Macro R']} & {row['Macro P']} & "
            f"{row['Macro F1']} & {row['Weighted F1']} & {row['Accuracy']} \\\\"
        )
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:vision_comparison}",
        r"\end{table}",
    ])
    latex = "\n".join(latex_lines)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "table_vision_comparison.tex"), "w") as f:
        f.write(latex)
    df.to_csv(os.path.join(RESULTS_DIR, "table_vision_comparison.csv"),
              index=False)

    return latex, df


def generate_eeg_table(result):
    """Generate EEG LOSO comparison table as (LaTeX string, DataFrame)."""
    summary = result["summary"]
    model_names = {
        "eeg_only": "EEGNet (EEG-only)",
        "multimodal": "RawGazeFusionNet (EEG+ET)",
    }
    rows = []
    for key in ["eeg_only", "multimodal"]:
        s = summary.get(key)
        if not s:
            continue
        rows.append({
            "Method": model_names.get(key, key),
            "Bal. Acc": f"{s['balanced_accuracy_mean']:.3f} ± {s['balanced_accuracy_std']:.3f}",
            "AUC-ROC": f"{s['auc_roc_mean']:.3f} ± {s['auc_roc_std']:.3f}",
            "F1": f"{s['f1_mean']:.3f} ± {s['f1_std']:.3f}",
            "Precision": f"{s['precision_mean']:.3f} ± {s['precision_std']:.3f}",
            "Recall": f"{s['recall_mean']:.3f} ± {s['recall_std']:.3f}",
        })

    df = pd.DataFrame(rows)

    latex_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{EEG LOSO cross-validation: EEG-only vs Multimodal.}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Method & Bal.\ Acc & AUC-ROC & F1 & Precision & Recall \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        latex_lines.append(
            f"{row['Method']} & {row['Bal. Acc']} & {row['AUC-ROC']} & "
            f"{row['F1']} & {row['Precision']} & {row['Recall']} \\\\"
        )
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:eeg_loso}",
        r"\end{table}",
    ])
    latex = "\n".join(latex_lines)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "table_eeg_comparison.tex"), "w") as f:
        f.write(latex)
    df.to_csv(os.path.join(RESULTS_DIR, "table_eeg_comparison.csv"),
              index=False)

    fold_rows = []
    for model_key in ["eeg_only", "multimodal"]:
        folds = result.get("fold_results", {}).get(model_key, [])
        for fold in folds:
            fold_rows.append({
                "Method": model_names.get(model_key, model_key),
                "Held-out Subject": fold.get("held_out_subject", "?"),
                "Bal. Acc": f"{fold['balanced_accuracy']:.3f}",
                "AUC-ROC": f"{fold['auc_roc']:.3f}",
                "F1": f"{fold['f1']:.3f}",
            })
    if fold_rows:
        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(os.path.join(RESULTS_DIR,
                                     "table_eeg_loso_folds.csv"),
                       index=False)

    return latex, df


def _save_vision_tables(result):
    generate_vision_table(result)
    print(f"  Vision tables saved to {RESULTS_DIR}/")


def _save_eeg_tables(result):
    generate_eeg_table(result)
    print(f"  EEG tables saved to {RESULTS_DIR}/")


# ═══════════════════════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════════════════════

def run_full_evaluation(
    vision=True, eeg=True, n_repeats=5, seed=42, progress_cb=None,
    resnet_epochs=30, resnet_batch_size=64,
):
    """Run the complete evaluation pipeline.

    Args:
        vision: run vision model comparison
        eeg: run EEG LOSO comparison
        n_repeats: number of CV repeats for vision
        seed: random seed
        progress_cb: optional callback(step, total, message)

    Returns dict with all results.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    if vision:
        print("\n" + "=" * 60)
        print("  VISION MODEL COMPARISON")
        print("=" * 60)
        results["vision"] = evaluate_vision_models(
            n_repeats=n_repeats, seed=seed, progress_cb=progress_cb,
            resnet_epochs=resnet_epochs, resnet_batch_size=resnet_batch_size,
        )

    if eeg:
        print("\n" + "=" * 60)
        print("  EEG LOSO COMPARISON")
        print("=" * 60)
        results["eeg"] = evaluate_eeg_loso(progress_cb=progress_cb)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print(f"  Results saved to {RESULTS_DIR}/")
    print("=" * 60)

    return results


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PSY197B Evaluation Pipeline")
    parser.add_argument("--vision-only", action="store_true",
                        help="Run vision comparison only")
    parser.add_argument("--eeg-only", action="store_true",
                        help="Run EEG LOSO comparison only")
    parser.add_argument("--n-repeats", type=int, default=5,
                        help="Number of CV repeats for vision (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--resnet-epochs", type=int, default=30,
                        help="Epochs per ResNet CV fold (default: 30)")
    parser.add_argument("--resnet-batch-size", type=int, default=64,
                        help="ResNet batch size — raise to feed GPU (default: 64)")
    parser.add_argument(
        "--dataloader-workers", type=int, default=None,
        help="Override PSY_DATALOADER_WORKERS (Windows: 2-4 is a good balance)",
    )
    parser.add_argument(
        "--deploy", action="store_true",
        help="Deploy model on unlabeled crops → data/vision_features/",
    )
    parser.add_argument(
        "--deploy-model", choices=["resnet50"],
        default="resnet50",
        help="Model for --deploy (default: resnet50)",
    )
    parser.add_argument(
        "--deploy-output-dir", default=None,
        help="Output dir for --deploy (default: <project>/data/vision_features)",
    )
    parser.add_argument(
        "--deploy-all-crops", action="store_true",
        help="With --deploy, reclassify human-labeled crops too (not recommended)",
    )
    args = parser.parse_args()

    if args.deploy:
        deploy_out = args.deploy_output_dir or os.path.join(
            PROJECT_ROOT, "data", "vision_features")
        os.makedirs(deploy_out, exist_ok=True)
        print(f"\n  Deploying {args.deploy_model} → {deploy_out}")
        print(f"  only_unlabeled={not args.deploy_all_crops}\n")
        result = relabel_crops_with_best(
            args.deploy_model,
            output_dir=deploy_out,
            only_unlabeled=not args.deploy_all_crops,
        )
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
            raise SystemExit(1)
        print(f"\n  Model classified: {result.get('total_relabeled', 0)}")
        print(f"  Human-labeled skipped: {result.get('total_skipped_human_labeled', 0)}")
        return

    if args.dataloader_workers is not None:
        os.environ["PSY_DATALOADER_WORKERS"] = str(max(0, args.dataloader_workers))
        global _COMPUTE_CACHE
        try:
            import device_utils
            device_utils._COMPUTE_CACHE = None
        except Exception:
            pass

    vision = not args.eeg_only
    eeg = not args.vision_only

    run_full_evaluation(
        vision=vision, eeg=eeg,
        n_repeats=args.n_repeats, seed=args.seed,
        resnet_epochs=args.resnet_epochs,
        resnet_batch_size=args.resnet_batch_size,
    )


if __name__ == "__main__":
    main()
