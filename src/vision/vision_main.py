"""Orchestrator for the gaze-contingent scene classification pipeline.

Run standalone:  python src/vision/vision_main.py
"""

import os
import shutil
import sys
import yaml

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.config import (
    CATEGORIES,
    CROP_SIZE,
    MIN_FIXATION_MS,
    VISION_CONDITIONS,
    get_eye_dir,
    get_vision_out_dir,
    get_world_video_path,
)
from vision.label_store import (
    crops_exist,
    data_root_from_config,
    get_crop_dir,
    load_labels_for,
    mirror_crops_dir,
)
from vision.frame_extractor import extract_frames_at_timestamps
from vision.gaze_crop import crop_gaze_region, get_fixation_gaze_center
from vision.classifier import ResNetGazeClassifier
from vision.annotator import run_annotator
from vision.visualizer import (
    plot_labeled_frame_grid,
    plot_category_timeline,
    save_debug_frames,
)

_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

DEFAULT_RUN_ID = None
CONDITIONS = None
N_LABEL_SAMPLES = 100
RUN_ANNOTATOR_FLAG = False
SAVE_DEBUG_FRAMES = True
N_DEBUG_FRAMES = 10
MAX_FIXATIONS = None

RESNET_HEAD_PATH = os.path.join(_PROJECT_ROOT, "models", "resnet50.pt")


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
    """Read vision condition labels from run config (walk-only by default)."""
    cfg = _load_run_config(run_dir)
    vis_conds = cfg.get("vision_conditions")
    if isinstance(vis_conds, list) and vis_conds:
        return vis_conds
    return VISION_CONDITIONS


def _subjects_from_run_dir(run_dir):
    """Read subject list from run config; fallback scans data/crops/ for available subjects."""
    cfg = _load_run_config(run_dir)
    sjs = cfg.get("data", {}).get("subjects", None)
    if sjs and isinstance(sjs, list) and all(isinstance(s, int) for s in sjs):
        return sorted(sjs)
    from vision.label_store import available_subjects_conditions
    found = sorted({sj for sj, _ in available_subjects_conditions()})
    if found:
        return found
    raise RuntimeError(
        "No subjects found in run config or data/crops/. "
        "Check run_config.yaml data.subjects or run the vision pipeline first."
    )


def _load_classifier(label_names):
    """Load the ResNet-50 classifier."""
    if not os.path.exists(RESNET_HEAD_PATH):
        raise FileNotFoundError(
            f"ResNet model not found at {RESNET_HEAD_PATH}. "
            "Train one first: python src/vision/train_models.py"
        )
    return ResNetGazeClassifier(RESNET_HEAD_PATH, label_names)


def _world_video_dir_from_run_dir(run_dir):
    """Read world_video_dir from run config; returns None if not set."""
    cfg = _load_run_config(run_dir)
    return cfg.get("data", {}).get("world_video_dir", None)


def generate_crops_for_condition(
    sj_num,
    condition,
    world_video_dir=None,
    force=False,
    progress_cb=None,
):
    """Generate gaze crops and save to data/crops/. Returns (crop_dir, n_crops).

    Skips if crops already exist in stable storage unless force=True.
    progress_cb(phase_name, current, total, message) for UI updates.
    """
    label = condition
    stable_dir = get_crop_dir(sj_num, label)

    n_existing = crops_exist(sj_num, label)
    if n_existing > 0 and not force:
        msg = f"Using {n_existing} cached crops from data/crops/sj{sj_num:02d}_{label}/"
        print(f"    {msg}")
        if progress_cb:
            progress_cb("Crops", 1, 1, msg)
        return stable_dir, n_existing

    eye_dir = get_eye_dir(data_root_from_config(), sj_num, label)
    video_path = get_world_video_path(sj_num, label, world_video_dir)

    fix_path = os.path.join(eye_dir, "fixations.csv")
    gaze_path = os.path.join(eye_dir, "gaze_positions.csv")

    if not os.path.exists(fix_path):
        print(f"    Missing fixations: {fix_path}")
        return stable_dir, 0
    if not os.path.exists(gaze_path):
        print(f"    Missing gaze: {gaze_path}")
        return stable_dir, 0
    if not video_path:
        print(f"    No world-video mapping for condition: {label}")
        return stable_dir, 0
    if not os.path.exists(video_path):
        print(f"    Missing world video: {video_path}")
        return stable_dir, 0

    if progress_cb:
        progress_cb("Load data", 0, 3, "Reading fixations and gaze data...")

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

    fix_ids = fixations["fixation id"].values
    mid_ns_arr = fixations["mid_ns"].values
    start_ns_arr = fixations["start timestamp [ns]"].values
    end_ns_arr = fixations["end timestamp [ns]"].values
    midns_to_idx = {int(mid_ns_arr[i]): i for i in range(len(mid_ns_arr))}

    if progress_cb:
        progress_cb("Extract frames", 1, 3, f"Extracting frames from world video ({len(fixations)} fixations)...")

    print("    Extracting frames from world video...")
    ts_frames = extract_frames_at_timestamps(video_path, gaze_df, mid_ns_arr)
    frames_by_fid = {}
    for ts_ns, frame in ts_frames.items():
        idx = midns_to_idx.get(ts_ns)
        if idx is None:
            continue
        fid = int(fix_ids[idx])
        frames_by_fid[fid] = frame
    print(f"    Extracted {len(frames_by_fid)} frames")

    if progress_cb:
        progress_cb("Gaze crops", 2, 3, f"Cropping {len(frames_by_fid)} gaze regions...")

    os.makedirs(stable_dir, exist_ok=True)
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
            os.path.join(stable_dir, f"{fid}_{ts_ns}.png"),
            crop[:, :, ::-1],
        )
        n_saved += 1

    print(f"    Saved {n_saved} crops ({n_skipped} skipped — gaze outside frame or no samples)")

    if progress_cb:
        progress_cb("Gaze crops", 3, 3, f"Done — {n_saved} crops saved to data/crops/")

    return stable_dir, n_saved


def _process_condition(sj_num, condition, run_dir, classifier, world_video_dir=None):
    """Run all vision pipeline phases for one subject x condition."""
    label = condition
    eye_dir = get_eye_dir(data_root_from_config(), sj_num, label)
    video_path = get_world_video_path(sj_num, label, world_video_dir)
    vision_dir = get_vision_out_dir(run_dir, sj_num, label)

    if os.path.isdir(vision_dir):
        for item in os.listdir(vision_dir):
            item_path = os.path.join(vision_dir, item)
            if item == "crops":
                if os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    frames_dir = os.path.join(vision_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    stable_crop_dir, n_crops = generate_crops_for_condition(
        sj_num, condition, world_video_dir=world_video_dir
    )

    if n_crops == 0:
        print("    No crops available — skipping remaining phases")
        return None

    crops_dir = os.path.join(vision_dir, "crops")
    if not os.path.exists(crops_dir):
        os.symlink(stable_crop_dir, crops_dir)

    results_csv = os.path.join(vision_dir, f"sj{sj_num:02d}_{label}_vision_results.csv")

    fix_path = os.path.join(eye_dir, "fixations.csv")
    gaze_path = os.path.join(eye_dir, "gaze_positions.csv")

    if not os.path.exists(fix_path) or not os.path.exists(gaze_path):
        print(f"    Missing fixation/gaze data for classification metadata")
        return None

    fixations = pd.read_csv(fix_path)
    gaze_df = pd.read_csv(gaze_path)
    fixations = fixations[fixations["duration [ms]"] >= MIN_FIXATION_MS].reset_index(drop=True)
    fixations["mid_ns"] = (
        (fixations["start timestamp [ns]"] + fixations["end timestamp [ns]"]) // 2
    ).astype(np.int64)

    if MAX_FIXATIONS is not None:
        fixations = fixations.head(MAX_FIXATIONS).reset_index(drop=True)

    # ── Classification ──
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

    print(f"    Classifying {len(crops_rgb)} crops with ResNet...")
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

    # ── Hand Labeling ──
    human_labels_df = None
    human_csv = os.path.join(vision_dir, f"sj{sj_num:02d}_{label}_human_labels.csv")
    if RUN_ANNOTATOR_FLAG:
        run_annotator(crops_dir, human_csv, n_samples=N_LABEL_SAMPLES)
    if os.path.exists(human_csv):
        human_labels_df = pd.read_csv(human_csv)

    # ── Visualizations ──
    _run_visualizations(sj_num, label, results_df, human_labels_df,
                        run_dir, vision_dir, eye_dir, video_path)

    # ── Fusion CSV ──
    _build_fusion_csv(sj_num, label, results_df, run_dir)

    return results_df


def _run_visualizations(sj_num, label, results_df, human_labels_df,
                        run_dir, vision_dir, eye_dir, video_path):
    """Generate visualization plots."""
    plots_dir = os.path.join(run_dir, "plots", "vision")
    os.makedirs(plots_dir, exist_ok=True)

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

    v2_path = os.path.join(plots_dir, f"sj{sj_num:02d}_{label}_V2_category_timeline.png")
    plot_category_timeline(results_df, v2_path)

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
    """Aggregate vision results to trial level for EEG fusion."""
    data_dir = os.path.join(run_dir, "data")
    out_path = os.path.join(data_dir,
                            f"sj{sj_num:02d}_{label}_vision_trial_features.csv")

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
    print(f"    Built trial-level features for {len(fusion_df)} trials -> {out_path}")


def _reclassify_only_condition(sj_num, condition, run_dir, classifier, world_video_dir=None):
    """Re-classify existing crops with ResNet, then regenerate plots."""
    label = condition
    vision_dir = get_vision_out_dir(run_dir, sj_num, label)
    results_csv = os.path.join(vision_dir, f"sj{sj_num:02d}_{label}_vision_results.csv")

    if not os.path.exists(results_csv):
        print(f"    No existing vision_results.csv — cannot reclassify-only. Run full pipeline first.")
        return None

    stable_crop_dir = get_crop_dir(sj_num, label)
    crop_files = sorted(f for f in os.listdir(stable_crop_dir) if f.endswith(".png"))
    if not crop_files:
        print(f"    No crops in {stable_crop_dir}")
        return None

    print(f"    Loading {len(crop_files)} crops from stable storage...")
    crops_rgb = []
    for cf in crop_files:
        img = cv2.imread(os.path.join(stable_crop_dir, cf))
        if img is None:
            continue
        crops_rgb.append(img[:, :, ::-1].copy())

    print(f"    Reclassifying {len(crops_rgb)} crops with ResNet...")
    batch_results = classifier.classify_batch(crops_rgb)

    results_df = pd.read_csv(results_csv)
    for i, res in enumerate(batch_results):
        if i >= len(results_df):
            break
        results_df.loc[i, "gaze_target_category"] = res["label"]
        results_df.loc[i, "confidence"] = res["confidence"]
        for cat_label, score in res["all_scores"].items():
            results_df.loc[i, f"score_{cat_label}"] = score
    results_df.to_csv(results_csv, index=False)
    print(f"    Saved -> {results_csv}")
    print("    Category distribution:")
    print(results_df["gaze_target_category"].value_counts().to_string(header=False))

    crops_dir_local = os.path.join(vision_dir, "crops")
    if not os.path.exists(crops_dir_local):
        os.symlink(stable_crop_dir, crops_dir_local)

    eye_dir = get_eye_dir(data_root_from_config(), sj_num, label)
    video_path = get_world_video_path(sj_num, label, world_video_dir)
    _run_visualizations(sj_num, label, results_df, None,
                        run_dir, vision_dir, eye_dir, video_path)
    return results_df


def run(run_dir_override=None, reclassify_only=False):
    """Run the vision pipeline.

    Args:
        run_dir_override: Absolute path to the run directory.
            If None, uses the most recent run in _PROJECT_ROOT/runs/.
        reclassify_only: If True, skip crop generation; only re-classify
            existing crops with ResNet and regenerate plots.
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
    try:
        from device_utils import print_device_banner
        print_device_banner(prefix="  ")
    except ImportError:
        pass

    label_names = list(CATEGORIES.keys())
    print("Loading ResNet classifier...")
    classifier = _load_classifier(label_names)

    summary = []

    run_conditions = CONDITIONS or _conditions_from_run_dir(the_run_dir)
    run_subjects = _subjects_from_run_dir(the_run_dir)
    world_video_dir = _world_video_dir_from_run_dir(the_run_dir)
    print(f"Subjects:        {run_subjects}")
    print(f"Conditions:      {run_conditions}")
    print(f"World video dir: {world_video_dir or '(not set)'}")

    _pairs = [(sj, cond) for sj in run_subjects for cond in run_conditions]
    _total_pairs = len(_pairs)

    for _pair_idx, (sj_num, condition) in enumerate(_pairs):
            print(f"\n{'='*60}")
            print(f"  Vision Pipeline — sj{sj_num:02d} {condition}")
            print(f"{'='*60}")

            try:
                from pipeline_progress import write_progress
                write_progress(the_run_dir, f"sj{sj_num:02d}_{condition}",
                               "running", _pair_idx, _total_pairs,
                               f"Processing sj{sj_num:02d} {condition}...")
            except ImportError:
                pass

            if reclassify_only:
                results_df = _reclassify_only_condition(
                    sj_num, condition, the_run_dir, classifier,
                    world_video_dir=world_video_dir)
            else:
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

    try:
        from pipeline_progress import write_progress, clear_progress
        write_progress(the_run_dir, "complete", "done",
                       _total_pairs, _total_pairs, "Vision pipeline complete!")
        clear_progress(the_run_dir)
    except ImportError:
        pass

    print("\nVision pipeline complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gaze-contingent scene classification")
    parser.add_argument("--run-dir", default=None,
                        help="Absolute path to the run directory")
    parser.add_argument("--reclassify-only", action="store_true",
                        help="Re-classify existing crops with ResNet and regenerate plots only")
    args = parser.parse_args()
    run(run_dir_override=args.run_dir, reclassify_only=args.reclassify_only)
