"""Orchestrator for the gaze-contingent scene classification pipeline.

Run standalone:  python src/vision/vision_main.py
"""

import os
import shutil
import sys
import yaml

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy

# Ensure project src/ is importable (for standalone execution)
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.config import (
    CATEGORIES,
    CROP_SIZE,
    MIN_FIXATION_MS,
    get_eye_dir,
    get_vision_out_dir,
    get_world_video_path,
)
from vision.label_store import (
    load_labels_for,
    mirror_crops_dir,
)
from vision.frame_extractor import extract_frames_at_timestamps
from vision.gaze_crop import crop_gaze_region, get_fixation_gaze_center
from vision.classifier import GazeClassifier
from vision.annotator import run_annotator
from vision.train_head import train_from_files
from vision.visualizer import (
    plot_labeled_frame_grid,
    plot_category_timeline,
    plot_clip_vs_human,
    save_debug_frames,
    plot_embedding_clusters,
    plot_optimal_k,
    plot_cluster_timeline as plot_cluster_timeline_viz,
)
from vision import embeddings as emb_module

# ── Configuration ─────────────────────────────────────────────
# Project root is derived from this file's location — never hardcoded.
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

DEFAULT_RUN_ID = None
SUBJECTS = [3]        # fallback only; overridden by run_config.yaml data.subjects
CONDITIONS = None
N_LABEL_SAMPLES = 100
RUN_ANNOTATOR_FLAG = False
SAVE_DEBUG_FRAMES = True
N_DEBUG_FRAMES = 10
MAX_FIXATIONS = None
N_CLUSTERS = 7

# Shared trained classification head (run-independent).
TRAINED_HEAD_PATH = os.path.join(_PROJECT_ROOT, "models", "clip_head.pt")


def _load_run_config(run_dir):
    """Load run config — prefers run snapshot, falls back to global src config."""
    candidates = [
        os.path.join(run_dir, "run_config_snapshot.yaml"),
        os.path.join(_SRC_DIR, "run_config.yaml"),
    ]
    for cfg_path in candidates:
        if not os.path.exists(cfg_path):
            continue
        try:
            with open(cfg_path, "r") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            continue
    return {}


def _conditions_from_run_dir(run_dir):
    """Read condition labels from run config."""
    cfg = _load_run_config(run_dir)
    conds = cfg.get("conditions", [])
    labels = [c["eeg_label"] for c in conds
              if isinstance(c, dict) and c.get("eeg_label")]
    return labels or ["walk_attend", "walk_unattend"]


def _subjects_from_run_dir(run_dir):
    """Read subject list from run config (fallback: global SUBJECTS constant)."""
    cfg = _load_run_config(run_dir)
    sjs = cfg.get("data", {}).get("subjects", None)
    if sjs and isinstance(sjs, list) and all(isinstance(s, int) for s in sjs):
        return sorted(sjs)
    return SUBJECTS


def _world_video_dir_from_run_dir(run_dir):
    """Read world_video_dir from run config; returns None if not set."""
    cfg = _load_run_config(run_dir)
    return cfg.get("data", {}).get("world_video_dir", None)


def _process_condition(sj_num, condition, run_dir, classifier, world_video_dir=None):
    """Run all vision pipeline phases for one subject × condition."""
    label = condition
    eye_dir = get_eye_dir(_PROJECT_ROOT, sj_num, label)
    video_path = get_world_video_path(sj_num, label, world_video_dir)
    vision_dir = get_vision_out_dir(run_dir, sj_num, label)

    # Clean slate: wipe any previous vision output for this condition
    if os.path.isdir(vision_dir):
        shutil.rmtree(vision_dir)
        print(f"    Cleared previous vision cache: {vision_dir}")
    frames_dir = os.path.join(vision_dir, "frames")
    crops_dir = os.path.join(vision_dir, "crops")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)

    results_csv = os.path.join(vision_dir, f"sj{sj_num:02d}_{label}_vision_results.csv")

    # ── PHASE 1: Load Data ──
    fix_path = os.path.join(eye_dir, "fixations.csv")
    gaze_path = os.path.join(eye_dir, "gaze_positions.csv")

    if not os.path.exists(fix_path):
        print(f"    Missing fixations: {fix_path}")
        return None
    if not os.path.exists(gaze_path):
        print(f"    Missing gaze: {gaze_path}")
        return None
    if not video_path:
        print(f"    No world-video mapping for condition: {label}")
        return None
    if not os.path.exists(video_path):
        print(f"    Missing world video: {video_path}")
        return None

    fixations = pd.read_csv(fix_path)
    gaze_df = pd.read_csv(gaze_path)

    fixations = fixations[fixations["duration [ms]"] >= MIN_FIXATION_MS].reset_index(drop=True)
    fixations["mid_ns"] = (
        (fixations["start timestamp [ns]"] + fixations["end timestamp [ns]"]) // 2
    ).astype(np.int64)
    print(f"    Loaded {len(fixations)} fixations after duration filter (>={MIN_FIXATION_MS} ms)")

    if MAX_FIXATIONS is not None:
        fixations = fixations.head(MAX_FIXATIONS).reset_index(drop=True)
        print(f"    Capped to first {MAX_FIXATIONS} fixations (TEST mode)")

    # Build arrays once to avoid iterrows float64 precision loss
    fix_ids = fixations["fixation id"].values
    mid_ns_arr = fixations["mid_ns"].values
    start_ns_arr = fixations["start timestamp [ns]"].values
    end_ns_arr = fixations["end timestamp [ns]"].values

    # Map mid_ns → fixation index for fast lookup after frame extraction
    midns_to_idx = {int(mid_ns_arr[i]): i for i in range(len(mid_ns_arr))}

    # ── PHASE 2: Extract Frames ──
    print("    Extracting frames from world video...")
    ts_frames = extract_frames_at_timestamps(
        video_path, gaze_df, mid_ns_arr
    )
    frames_by_fid = {}
    for ts_ns, frame in ts_frames.items():
        idx = midns_to_idx.get(ts_ns)
        if idx is None:
            continue
        fid = int(fix_ids[idx])
        frames_by_fid[fid] = frame
        cv2.imwrite(os.path.join(frames_dir, f"{fid}_{ts_ns}.jpg"), frame)
    print(f"    Extracted {len(frames_by_fid)} frames")

    # ── PHASE 3: Extract Gaze Crops ──
    print("    Extracting gaze crops...")
    n_saved = 0
    n_skipped = 0
    for i in range(len(fixations)):
        fid = int(fix_ids[i])
        ts_ns = int(mid_ns_arr[i])

        frame = frames_by_fid.get(fid)
        if frame is None:
            n_skipped += 1
            continue

        center = get_fixation_gaze_center(
            gaze_df, int(start_ns_arr[i]), int(end_ns_arr[i])
        )
        if center is None:
            n_skipped += 1
            continue

        gx, gy = center
        crop = crop_gaze_region(frame, gx, gy, crop_size=CROP_SIZE)
        if crop is None:
            n_skipped += 1
            continue

        cv2.imwrite(
            os.path.join(crops_dir, f"{fid}_{ts_ns}.png"),
            crop[:, :, ::-1],
        )
        n_saved += 1

    print(f"    Saved {n_saved} crops ({n_skipped} skipped — gaze outside frame or no samples)")

    # Mirror crops to stable data/crops/ so labels survive pipeline re-runs
    n_mirrored = mirror_crops_dir(crops_dir, sj_num, label)
    if n_mirrored:
        print(f"    Mirrored {n_mirrored} new crops → data/crops/sj{sj_num:02d}_{label}/")

    # ── PHASE 4: CLIP Classification ──
    crop_files = sorted(f for f in os.listdir(crops_dir) if f.endswith(".png"))
    if not crop_files:
        print("    No crops to classify")
        return None

    crops_rgb = []
    crop_meta = []
    for cf in crop_files:
        img = cv2.imread(os.path.join(crops_dir, cf))
        if img is None:
            continue
        rgb = img[:, :, ::-1].copy()
        crops_rgb.append(rgb)

        parts = cf.replace(".png", "").split("_")
        crop_meta.append({"fixation_id": int(parts[0]), "timestamp_ns": int(parts[1])})

    print(f"    Classifying {len(crops_rgb)} crops...")
    batch_results = classifier.classify_batch(crops_rgb)

    records = []
    for meta_row, res in zip(crop_meta, batch_results):
        fid = meta_row["fixation_id"]
        ts_ns = meta_row["timestamp_ns"]

        fix_match = fixations[fixations["fixation id"] == fid]
        if fix_match.empty:
            dur_ms = np.nan
        else:
            dur_ms = float(fix_match["duration [ms]"].iloc[0])

        center = get_fixation_gaze_center(
            gaze_df,
            int(fix_match["start timestamp [ns]"].iloc[0]) if not fix_match.empty else ts_ns,
            int(fix_match["end timestamp [ns]"].iloc[0]) if not fix_match.empty else ts_ns,
        )
        gx, gy = center if center else (np.nan, np.nan)

        gaze_start_ns = int(gaze_df["timestamp [ns]"].iloc[0])
        ts_s = (ts_ns - gaze_start_ns) / 1e9

        rec = {
            "fixation_id": fid,
            "timestamp_ns": ts_ns,
            "timestamp_s": ts_s,
            "duration_ms": dur_ms,
            "gaze_x_px": gx,
            "gaze_y_px": gy,
            "gaze_target_category": res["label"],
            "confidence": res["confidence"],
        }
        for cat_label, score in res["all_scores"].items():
            rec[f"score_{cat_label}"] = score
        records.append(rec)

    results_df = pd.DataFrame(records)
    results_df.to_csv(results_csv, index=False)
    print(f"    Classified {len(results_df)} fixations")
    print(f"    Category distribution:")
    print(results_df["gaze_target_category"].value_counts().to_string(header=False))

    # ── PHASES 4B–4E: Embedding Pipeline ──
    _run_embedding_phases(sj_num, label, results_df, vision_dir,
                          crops_dir, run_dir, classifier)

    # Reload results_df in case cluster_id was added
    if os.path.exists(results_csv):
        results_df = pd.read_csv(results_csv)

    # ── PHASE 5: Hand Labeling ──
    human_labels_df = None
    human_csv = os.path.join(vision_dir, f"sj{sj_num:02d}_{label}_human_labels.csv")
    if RUN_ANNOTATOR_FLAG:
        run_annotator(crops_dir, human_csv, n_samples=N_LABEL_SAMPLES)
    if os.path.exists(human_csv):
        human_labels_df = pd.read_csv(human_csv)

    # ── PHASE 5B: Train Fine-Tuned Head ──
    # Retrain if: no head yet, OR central label count grew since last training.
    emb_base = os.path.join(vision_dir, f"sj{sj_num:02d}_{label}_embeddings")
    central_labels = load_labels_for(sj_num, label)
    n_central = len(central_labels)

    head_is_stale = False
    if os.path.exists(TRAINED_HEAD_PATH):
        import torch as _torch
        try:
            _ckpt = _torch.load(TRAINED_HEAD_PATH, map_location="cpu", weights_only=True)
            head_is_stale = n_central > _ckpt.get("stats", {}).get("n_samples", 0)
        except Exception:
            head_is_stale = True

    should_train = (
        n_central >= 10
        and os.path.exists(f"{emb_base}.npy")
        and (not os.path.exists(TRAINED_HEAD_PATH) or head_is_stale)
    )

    if should_train:
        print("  Phase 5B — Training linear classification head from central store...")
        from vision.train_head import train_from_label_store
        train_from_label_store(
            sj_num=sj_num,
            condition=label,
            embeddings_base=emb_base,
            out_model_path=TRAINED_HEAD_PATH,
        )
        if os.path.exists(TRAINED_HEAD_PATH):
            classifier.load_head(TRAINED_HEAD_PATH)
    elif not classifier.has_head and os.path.exists(TRAINED_HEAD_PATH):
        classifier.load_head(TRAINED_HEAD_PATH)

    if classifier.has_head:
            print("  Phase 5B — Reclassifying with newly trained head...")
            batch_results = classifier.classify_batch(crops_rgb)
            for i, res in enumerate(batch_results):
                results_df.loc[i, "gaze_target_category"] = res["label"]
                results_df.loc[i, "confidence"] = res["confidence"]
                for cat_label, score in res["all_scores"].items():
                    results_df.loc[i, f"score_{cat_label}"] = score
            results_df.to_csv(results_csv, index=False)
            print(f"    Reclassified {len(results_df)} fixations")

    # ── PHASE 6: Visualizations ──
    _run_visualizations(sj_num, label, results_df, human_labels_df,
                        run_dir, vision_dir, eye_dir, video_path)

    # ── PHASE 7: Fusion CSV ──
    _build_fusion_csv(sj_num, label, results_df, run_dir)

    return results_df


def _run_embedding_phases(sj_num, label, results_df, vision_dir,
                          crops_dir, run_dir, classifier):
    """Phases 4B–4E: embedding extraction, optimal k, clustering, trial features."""
    emb_base = os.path.join(vision_dir, f"sj{sj_num:02d}_{label}_embeddings")
    results_csv = os.path.join(vision_dir,
                               f"sj{sj_num:02d}_{label}_vision_results.csv")

    # ── PHASE 4B: Extract CLIP Embeddings ──
    print("  Phase 4B — Extracting CLIP embeddings...")
    crop_files = sorted(f for f in os.listdir(crops_dir)
                        if f.endswith(".png"))
    crops_rgb = []
    fid_list = []
    for cf in crop_files:
        img = cv2.imread(os.path.join(crops_dir, cf))
        if img is None:
            continue
        crops_rgb.append(img[:, :, ::-1].copy())
        fid_list.append(int(cf.split("_")[0]))
    embs = classifier.extract_embeddings_batch(crops_rgb)
    emb_module.save_embeddings(embs, fid_list, emb_base)
    print(f"    Embeddings shape: {embs.shape}")

    # ── PHASE 4C: Find Optimal K ──
    print("  Phase 4C — Finding optimal k...")
    k_csv = os.path.join(vision_dir, "k_analysis.csv")
    k_analysis = emb_module.find_optimal_k(embs, k_range=range(3, 12))
    pd.DataFrame(k_analysis).to_csv(k_csv, index=False)

    plots_dir = os.path.join(run_dir, "plots", "vision")
    os.makedirs(plots_dir, exist_ok=True)
    v4_path = os.path.join(plots_dir,
                           f"sj{sj_num:02d}_{label}_V4_optimal_k.png")
    plot_optimal_k(k_analysis, v4_path)

    best_k = k_analysis["k_values"][
        int(np.argmax(k_analysis["silhouettes"]))
    ]
    print(f"    Recommended n_clusters={best_k} based on silhouette score")

    # ── PHASE 4D: Cluster Embeddings ──
    print(f"  Phase 4D — Clustering with K={N_CLUSTERS}...")
    clusters_csv = os.path.join(vision_dir,
                                f"sj{sj_num:02d}_{label}_clusters.csv")
    cluster_labels, _ = emb_module.cluster_embeddings(embs, N_CLUSTERS)
    cl_df = pd.DataFrame({
        "fixation_id": fid_list,
        "cluster_id": cluster_labels,
    })
    fid_to_ts = dict(zip(results_df["fixation_id"],
                          results_df["timestamp_ns"]))
    cl_df["timestamp_ns"] = cl_df["fixation_id"].map(fid_to_ts)
    cl_df.to_csv(clusters_csv, index=False)

    # Add cluster_id to results_df and overwrite CSV
    fid_to_cluster = dict(zip(cl_df["fixation_id"], cl_df["cluster_id"]))
    results_df["cluster_id"] = results_df["fixation_id"].map(
        fid_to_cluster
    ).values
    results_df.to_csv(results_csv, index=False)
    print(f"    Updated results CSV with cluster_id column")

    # ── PHASE 4E: Compute Trial Embedding Features ──
    data_dir = os.path.join(run_dir, "data")
    et_path = os.path.join(data_dir,
                           f"sj{sj_num:02d}_{label}_ET_Prepro1.csv")
    if os.path.exists(et_path):
        print("  Phase 4E — Building trial-level embedding features...")
        et_df = pd.read_csv(et_path)
        trial_feats = emb_module.compute_trial_embedding_features(
            results_df, et_df, embs
        )
        feat_path = os.path.join(
            data_dir,
            f"sj{sj_num:02d}_{label}_vision_trial_features.csv",
        )
        trial_feats.to_csv(feat_path, index=False)
        print(f"    Saved trial features → {feat_path}")
    else:
        print(f"    No ET_Prepro1.csv — skipping trial features")


def _run_visualizations(sj_num, label, results_df, human_labels_df,
                        run_dir, vision_dir, eye_dir, video_path):
    """Generate all visualization plots."""
    plots_dir = os.path.join(run_dir, "plots", "vision")
    os.makedirs(plots_dir, exist_ok=True)

    # V1: Labeled frame grid — extract frames for ALL results so the
    # grid's category-diverse sampling always finds its frames.
    gaze_path = os.path.join(eye_dir, "gaze_positions.csv")
    if os.path.exists(gaze_path):
        gaze_df = pd.read_csv(gaze_path)
        all_ts = results_df["timestamp_ns"].values.astype(np.int64)
        frames_for_viz = extract_frames_at_timestamps(
            video_path, gaze_df, all_ts
        )
        crops_dir = os.path.join(vision_dir, "crops")
        v1_path = os.path.join(plots_dir, f"sj{sj_num:02d}_{label}_V1_labeled_frames.png")
        plot_labeled_frame_grid(results_df, frames_for_viz, gaze_df, v1_path,
                                n=12, crops_dir=crops_dir)

    # V2: Category timeline
    v2_path = os.path.join(plots_dir, f"sj{sj_num:02d}_{label}_V2_category_timeline.png")
    plot_category_timeline(results_df, v2_path)

    # V3: CLIP vs Human
    if human_labels_df is not None:
        v3_path = os.path.join(plots_dir, f"sj{sj_num:02d}_{label}_V3_clip_vs_human.png")
        plot_clip_vs_human(results_df, human_labels_df, v3_path)
    else:
        print("    No human labels — skipping V3 plot")

    # V5/V6: Embedding visualizations
    if "cluster_id" in results_df.columns:
        emb_base = os.path.join(vision_dir,
                                f"sj{sj_num:02d}_{label}_embeddings")
        if os.path.exists(f"{emb_base}.npy"):
            embs, _ = emb_module.load_embeddings(emb_base)
            crops_dir_viz = os.path.join(vision_dir, "crops")

            v5_path = os.path.join(
                plots_dir,
                f"sj{sj_num:02d}_{label}_V5_embedding_clusters.png",
            )
            plot_embedding_clusters(
                embs, results_df["cluster_id"].values, results_df,
                crops_dir_viz, v5_path,
            )

            v6_path = os.path.join(
                plots_dir,
                f"sj{sj_num:02d}_{label}_V6_cluster_timeline.png",
            )
            plot_cluster_timeline_viz(results_df, v6_path)
            print("    Saved embedding visualizations")

    # Debug: annotated sample frames
    if SAVE_DEBUG_FRAMES:
        debug_dir = os.path.join(plots_dir, "debug_frames", f"sj{sj_num:02d}_{label}")
        gaze_path = os.path.join(eye_dir, "gaze_positions.csv")
        if os.path.exists(gaze_path):
            gaze_df_dbg = pd.read_csv(gaze_path)
            debug_ts = results_df.sample(
                min(N_DEBUG_FRAMES, len(results_df)), random_state=42
            )["timestamp_ns"].values.astype(np.int64)
            debug_frames = extract_frames_at_timestamps(
                video_path, gaze_df_dbg, debug_ts
            )
            save_debug_frames(results_df, debug_frames, debug_dir, n=N_DEBUG_FRAMES)


def _build_fusion_csv(sj_num, label, results_df, run_dir):
    """Aggregate vision results to trial level for EEG fusion.

    Skipped if embedding-based trial features already exist (Phase 4E).
    """
    data_dir = os.path.join(run_dir, "data")
    out_path = os.path.join(data_dir,
                            f"sj{sj_num:02d}_{label}_vision_trial_features.csv")

    # If embedding-based features already written by Phase 4E, don't overwrite
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, nrows=0)
        if "dominant_cluster" in existing.columns:
            print(f"    Embedding-based trial features already exist — "
                  f"skipping category-based fusion CSV")
            return

    et_path = os.path.join(data_dir, f"sj{sj_num:02d}_{label}_ET_Prepro1.csv")
    if not os.path.exists(et_path):
        print(f"    No ET_Prepro1.csv found — skipping fusion CSV")
        return

    et_df = pd.read_csv(et_path)
    cat_labels = list(CATEGORIES.keys())

    trial_records = []
    for _, trial_row in et_df.iterrows():
        t_idx = trial_row["trialIdx"]
        t_time = trial_row["trigger_time"]

        ts_s_min = t_time - 1.0
        ts_s_max = t_time + 1.0

        window = results_df[
            (results_df["timestamp_s"] >= ts_s_min)
            & (results_df["timestamp_s"] <= ts_s_max)
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
            rec["mean_confidence"] = float(window["confidence"].mean())

            count_arr = np.array([counts.get(c, 0) for c in cat_labels], dtype=float)
            total = count_arr.sum()
            if total > 0:
                probs = count_arr / total
                rec["category_entropy"] = float(sp_entropy(probs, base=2))
            else:
                rec["category_entropy"] = np.nan

            for c in cat_labels:
                rec[f"vis_prop_{c}"] = counts.get(c, 0) / total if total > 0 else np.nan

        trial_records.append(rec)

    fusion_df = pd.DataFrame(trial_records)
    fusion_df.to_csv(out_path, index=False)
    print(f"    Built trial-level features for {len(fusion_df)} trials → {out_path}")


def run(run_dir_override=None):
    """Run the vision pipeline.

    Args:
        run_dir_override: Absolute path to the run directory.
            If None, uses the most recent run in _PROJECT_ROOT/runs/ (or
            _PROJECT_ROOT/runs/DEFAULT_RUN_ID if DEFAULT_RUN_ID is set).
    """
    if run_dir_override:
        the_run_dir = run_dir_override
    elif DEFAULT_RUN_ID is None:
        runs_root = os.path.join(_PROJECT_ROOT, "runs")
        run_dirs = sorted(
            [
                os.path.join(runs_root, d)
                for d in os.listdir(runs_root)
                if os.path.isdir(os.path.join(runs_root, d))
            ],
            reverse=True,
        )
        if not run_dirs:
            raise FileNotFoundError("No run directories found in runs/")
        the_run_dir = run_dirs[0]
    else:
        the_run_dir = os.path.join(_PROJECT_ROOT, "runs", DEFAULT_RUN_ID)
    os.makedirs(the_run_dir, exist_ok=True)

    print(f"Vision run dir: {the_run_dir}")
    print("Loading CLIP model...")
    head = TRAINED_HEAD_PATH if os.path.exists(TRAINED_HEAD_PATH) else None
    classifier = GazeClassifier(head_path=head)

    summary = []

    run_conditions = CONDITIONS or _conditions_from_run_dir(the_run_dir)
    run_subjects = _subjects_from_run_dir(the_run_dir)
    world_video_dir = _world_video_dir_from_run_dir(the_run_dir)
    print(f"Subjects:        {run_subjects}")
    print(f"Conditions:      {run_conditions}")
    print(f"World video dir: {world_video_dir or '(not set — video-dependent phases will be skipped)'}")

    for sj_num in run_subjects:
        for condition in run_conditions:
            print(f"\n{'='*60}")
            print(f"  Vision Pipeline — sj{sj_num:02d} {condition}")
            print(f"{'='*60}")

            results_df = _process_condition(sj_num, condition, the_run_dir, classifier,
                                            world_video_dir=world_video_dir)

            if results_df is not None and not results_df.empty:
                top3 = results_df["gaze_target_category"].value_counts().head(3)
                top3_str = ", ".join(f"{k}({v})" for k, v in top3.items())

                data_dir = os.path.join(the_run_dir, "data")
                fusion_path = os.path.join(data_dir, f"sj{sj_num:02d}_{condition}_vision_trial_features.csv")
                n_trials = 0
                if os.path.exists(fusion_path):
                    n_trials = len(pd.read_csv(fusion_path))

                summary.append({
                    "condition": condition,
                    "n_fixations": len(results_df),
                    "top_3": top3_str,
                    "mean_conf": f"{results_df['confidence'].mean():.3f}",
                    "n_trials_with_vision": n_trials,
                })

    if summary:
        print(f"\n{'='*60}")
        print("  VISION PIPELINE SUMMARY")
        print(f"{'='*60}")
        for s in summary:
            print(f"  {s['condition']:20s}  fixations={s['n_fixations']:5d}  "
                  f"top3=[{s['top_3']}]  mean_conf={s['mean_conf']}  "
                  f"trials={s['n_trials_with_vision']}")

    print("\nVision pipeline complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gaze-contingent scene classification")
    parser.add_argument("--run-dir", default=None,
                        help="Absolute path to the run directory")
    args = parser.parse_args()
    run(run_dir_override=args.run_dir)
