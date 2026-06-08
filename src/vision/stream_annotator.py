"""Streamlit-based gaze crop annotation, training, and results tool.

Run:  streamlit run src/vision/stream_annotator.py

Six tabs:
  1. Generate Crops  — crop status grid, data availability
  2. Label           — annotation interface with labeler ID + flag support
  3. Statistics      — per-class distributions, coverage, inter-rater agreement
  4. Train           — ResNet-50 fine-tuning, live metrics, model versioning
  5. Evaluate        — test-set confusion matrices and model comparison
  6. Deploy          — classify all crops with best model
"""

import glob
import multiprocessing as _mp
import os
import subprocess
import sys
import threading
import time
from collections import deque

# If we were imported as a multiprocessing spawn worker (e.g. a stray
# DataLoader worker on Windows), bail out immediately. Without this guard
# Python re-runs the entire Streamlit script per worker, which crashes on
# missing files and corrupts Streamlit's session state.
if _mp.current_process().name != "MainProcess":
    sys.exit(0)

import numpy as np
import pandas as pd
import streamlit as st
import yaml

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.config import CATEGORIES, CATEGORY_COLORS, ET_FOLDER_MAP
from vision.label_store import (
    FLAG_LABEL,
    PROJECT_ROOT,
    append_label,
    available_subjects_conditions,
    cohens_kappa_matrix,
    crop_status_grid,
    crops_exist,
    data_root_from_config,
    get_crop_dir,
    get_crop_path,
    inter_rater_overlaps,
    label_counts,
    labeler_ids,
    load_flagged,
    load_labels,
    load_labels_for,
    load_trainable_labels,
    migrate_existing_labels,
    relabel,
    remove_last_label,
    scan_data_subjects,
    subject_condition_counts,
)
from vision.vision_main import generate_crops_for_condition
from evaluate import (
    evaluate_vision_models,
    relabel_crops_with_best,
    make_cm_figure,
    load_vision_data,
    generate_vision_table,
    RESULTS_DIR,
)

LABEL_NAMES = list(CATEGORIES.keys())
RUNS_ROOT = os.path.abspath(os.environ.get("PSY197B_RUNS_DIR") or os.path.join(PROJECT_ROOT, "runs"))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _find_venv_python() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe"),
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python3.11"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return sys.executable


VENV_PYTHON = _find_venv_python()


def _run_subprocess_with_status(cmd, label, timeout_s=1800):
    """Run a subprocess with live output in a Streamlit status widget."""
    with st.status(f"Running {label}...", expanded=True) as status:
        log_placeholder = st.empty()
        try:
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, cwd=PROJECT_ROOT,
            )
            log_lines: deque = deque()
            log_lock = threading.Lock()

            def _drain():
                for line in iter(process.stdout.readline, ""):
                    with log_lock:
                        log_lines.append(line)
                process.stdout.close()

            drain_thread = threading.Thread(target=_drain, daemon=True)
            drain_thread.start()

            start_time = time.time()
            last_render = 0.0

            while process.poll() is None:
                if time.time() - start_time > timeout_s:
                    process.kill()
                    st.error(f"{label} timed out ({timeout_s // 60} min limit)")
                    break
                now = time.time()
                if now - last_render > 2.0:
                    with log_lock:
                        tail = "".join(list(log_lines)[-150:])
                    if tail:
                        log_placeholder.code(tail, language="text")
                    last_render = now
                time.sleep(1.0)

            drain_thread.join(timeout=5.0)

            if process.returncode == 0:
                status.update(label=f"{label} completed!", state="complete")
            else:
                status.update(label=f"{label} failed (exit {process.returncode})", state="error")
                with log_lock:
                    tail = "".join(list(log_lines)[-200:])
                if tail:
                    log_placeholder.code(tail, language="text")
        except Exception as e:
            st.error(f"Error running {label}: {e}")
    st.cache_data.clear()

# ΓöÇΓöÇ Page config ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

st.set_page_config(
    page_title="Gaze Crop Tool",
    layout="wide",
)

# ΓöÇΓöÇ Session state defaults ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

_SS_DEFAULTS = {
    "labeler_id": "",
    "crop_idx": {},
    "last_pair": None,
    "train_history_resnet": [],
    "train_running": False,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ΓöÇΓöÇ Migration (run once per session) ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

@st.cache_data(show_spinner=False)
def _run_migration():
    return migrate_existing_labels(RUNS_ROOT)

migrated = _run_migration()
if migrated > 0:
    st.toast(f"Migrated {migrated} labels into central store.", icon="Γ£à")

# ΓöÇΓöÇ Cached helpers ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

@st.cache_data(ttl=30)
def _load_all_labels():
    return load_labels()


@st.cache_data(ttl=30)
def _load_trainable():
    return load_trainable_labels()


@st.cache_data(ttl=20)
def _get_all_pairs_with_crops():
    """All (sj_num, condition, n_crops) from data/ dirs + data/crops/."""
    all_conditions = list(ET_FOLDER_MAP.keys())
    subjects = scan_data_subjects()

    rows = {}
    for sj in subjects:
        for cond in all_conditions:
            crop_dir = get_crop_dir(sj, cond)
            n = 0
            if os.path.isdir(crop_dir):
                n = sum(1 for f in os.listdir(crop_dir) if f.lower().endswith(".png"))
            rows[(sj, cond)] = n

    # Include any pair that has crops but isn't in data/sj* (shouldn't happen often)
    for sj, cond in available_subjects_conditions():
        if (sj, cond) not in rows:
            crop_dir = get_crop_dir(sj, cond)
            rows[(sj, cond)] = sum(
                1 for f in os.listdir(crop_dir) if f.lower().endswith(".png")
            )

    return sorted((sj, cond, n) for (sj, cond), n in rows.items())


# ΓöÇΓöÇ Results tab helpers ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


@st.cache_data(ttl=30)
def _list_runs():
    if not os.path.isdir(RUNS_ROOT):
        return []
    return sorted(
        [d for d in os.listdir(RUNS_ROOT)
         if os.path.isdir(os.path.join(RUNS_ROOT, d))],
        reverse=True,
    )


@st.cache_data(ttl=60)
def _find_subjects_conditions(run_name):
    dd = os.path.join(RUNS_ROOT, run_name, "data")
    if not os.path.isdir(dd):
        return [], []
    subjects = set()
    conditions = set()
    suffixes = [
        "_fused_metadata.csv",
        "_features.csv",
        "_EEG_Prepro1-epo.fif",
        "_ET_Prepro1.csv",
    ]
    for f in os.listdir(dd):
        for sfx in suffixes:
            if f.endswith(sfx):
                parts = f.replace(sfx, "").split("_", 1)
                try:
                    sj = int(parts[0].replace("sj", ""))
                    cond = parts[1]
                    subjects.add(sj)
                    conditions.add(cond)
                except (ValueError, IndexError):
                    pass
                break
    return sorted(subjects), sorted(conditions)


def _vision_dir(run_name, sj, cond):
    return os.path.join(RUNS_ROOT, run_name, "vision", f"sj{sj:02d}_{cond}")


def _vision_plots_dir(run_name):
    return os.path.join(RUNS_ROOT, run_name, "plots", "vision")


@st.cache_data(ttl=60)
def _load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=120)
def _aggregate_vision_results(run_name, subjects_tuple, conds_tuple):
    frames = []
    for sj in subjects_tuple:
        for cond in conds_tuple:
            vr_path = os.path.join(
                _vision_dir(run_name, sj, cond),
                f"sj{sj:02d}_{cond}_vision_results.csv",
            )
            df = _load_csv(vr_path)
            if df is not None and len(df) > 0:
                d2 = df.copy()
                d2["subject"] = sj
                d2["condition"] = cond
                frames.append(d2)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


_CAT_COLORS = CATEGORY_COLORS

VISION_FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "vision_features")


@st.cache_data(ttl=60)
def _load_vision_comparison():
    """Load previously saved vision comparison results."""
    path = os.path.join(RESULTS_DIR, "vision_comparison.json")
    if os.path.exists(path):
        import json
        with open(path) as f:
            return json.load(f)
    return None


# ΓöÇΓöÇ Sidebar ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

with st.sidebar:
    st.title("Gaze Crop Tool")

    labeler_id_input = st.text_input(
        "Your name / labeler ID",
        value=st.session_state.labeler_id,
        placeholder="e.g. alice",
        help="Required for inter-rater reliability. Crops you label are tracked per labeler.",
    )
    st.session_state.labeler_id = labeler_id_input.strip()
    labeler_id = st.session_state.labeler_id

    if not labeler_id:
        st.warning("Enter a labeler ID to enable per-labeler tracking.")

    st.markdown("---")

    all_df = _load_all_labels()
    total_all = len(all_df)
    trainable_df = _load_trainable()
    flagged_df = load_flagged()

    col_a, col_b = st.columns(2)
    col_a.metric("Total labels", total_all)
    col_b.metric("Trainable", len(trainable_df))
    if len(flagged_df) > 0:
        st.warning(f"ΓÜæ {len(flagged_df)} flagged for review")

    st.markdown("---")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("`data/human_labels.csv`")

# ΓöÇΓöÇ Main tabs ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

t_gen, t_label, t_stats, t_train, t_eval, t_deploy = st.tabs([
    "Generate Crops", "Label", "Statistics", "Train", "Evaluate", "Deploy"
])


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# TAB 1 ΓÇö Generate Crops
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

with t_gen:
    st.header("Crop Generation")

    # ΓöÇΓöÇ Config editor ΓöÇΓöÇ
    _cfg_path = os.path.join(PROJECT_ROOT, "src", "run_config.yaml")

    with st.expander("Pipeline Config (run_config.yaml)", expanded=False):
        try:
            with open(_cfg_path) as _f:
                _cfg_raw = _f.read()
        except FileNotFoundError:
            _cfg_raw = ""
            st.warning(f"Config file not found: {_cfg_path}")

        if _cfg_raw:
            _cfg_edited = st.text_area(
                "Edit config (YAML)", value=_cfg_raw, height=300,
                help="Key fields for crop generation: data.subjects, data.world_video_dir, conditions, et.folder_map",
            )
            if st.button("Save Config"):
                try:
                    yaml.safe_load(_cfg_edited)  # validate
                    with open(_cfg_path, "w") as _f:
                        _f.write(_cfg_edited)
                    st.success("Config saved!")
                    st.cache_data.clear()
                except yaml.YAMLError as _e:
                    st.error(f"Invalid YAML: {_e}")

    # ΓöÇΓöÇ Status grid ΓöÇΓöÇ
    st.subheader("Status")

    try:
        status_df = crop_status_grid()
    except Exception as e:
        st.error(f"Could not scan data directory: {e}")
        status_df = pd.DataFrame()

    if status_df.empty:
        _scanned = data_root_from_config()
        st.info(
            f"No `sjNN` subject folders found under:\n\n`{_scanned}`\n\n"
            "Check that `data.root` in `run_config.yaml` points to the folder "
            "that directly contains `sj01/`, `sj02/`, ΓÇª subdirectories."
        )
    else:
        def _status_icon(val):
            return "Γ£à" if val else "Γ¥î"

        display_df = status_df.copy()
        display_df["has_video"] = display_df["has_video"].map(_status_icon)
        display_df["has_fixations"] = display_df["has_fixations"].map(_status_icon)
        display_df["has_gaze"] = display_df["has_gaze"].map(_status_icon)
        display_df["subject"] = display_df["subject_id"].apply(lambda x: f"sj{x:02d}")
        display_df = display_df.rename(columns={
            "condition": "Condition",
            "n_crops": "Crops",
            "has_video": "World Video",
            "has_fixations": "fixations.csv",
            "has_gaze": "gaze_positions.csv",
        })[["subject", "Condition", "Crops", "World Video", "fixations.csv", "gaze_positions.csv"]]

        st.dataframe(display_df, hide_index=True, use_container_width=True)

        total_crops = status_df["n_crops"].sum()
        labeled_total = len(trainable_df)
        st.markdown(f"**{total_crops}** total crops across all subjects.  "
                    f"**{labeled_total}** labeled ({labeled_total/max(total_crops,1)*100:.1f}%).")

    # ΓöÇΓöÇ Generate controls ΓöÇΓöÇ
    st.markdown("---")
    st.subheader("Generate Crops")

    if not status_df.empty:
        # Build selectable pairs: only those with video + fixations available
        _generable = status_df[status_df["has_video"] & status_df["has_fixations"]]
        _gen_options = [
            f"sj{int(r['subject_id']):02d} ΓÇö {r['condition']}  ({int(r['n_crops'])} crops)"
            for _, r in _generable.iterrows()
        ]
        _gen_selected = st.multiselect(
            "Subject / condition pairs to generate",
            options=_gen_options,
            default=_gen_options,
            help="Only pairs with world video + fixations.csv available are shown.",
        )

        _col1, _col2 = st.columns(2)
        _force_regen = _col1.checkbox("Force regenerate (overwrite existing crops)")
        _gen_button = _col2.button("Generate Crops", type="primary",
                                    disabled=len(_gen_selected) == 0)

        if _gen_button:
            # Read world_video_dir from config
            _wvd = None
            try:
                with open(_cfg_path) as _f:
                    _cfg = yaml.safe_load(_f)
                _wvd = _cfg.get("data", {}).get("world_video_dir")
            except Exception:
                pass

            # Parse selected pairs back to (sj_num, condition)
            _pairs = []
            for sel in _gen_selected:
                sj_str = sel.split(" ΓÇö ")[0]
                cond_str = sel.split(" ΓÇö ")[1].split("  (")[0]
                _pairs.append((int(sj_str[2:]), cond_str))

            with st.status(f"Generating crops for {len(_pairs)} pair(s)...", expanded=True) as _status:
                _prog = st.progress(0.0)
                _msg = st.empty()

                for _pi, (_sj, _cond) in enumerate(_pairs):
                    _pair_label = f"sj{_sj:02d}_{_cond}"
                    _msg.markdown(f"**{_pair_label}** ΓÇö starting...")

                    def _crop_cb(phase, cur, tot, text, _pl=_pair_label, _idx=_pi, _total=len(_pairs)):
                        _overall = (_idx + cur / max(tot, 1)) / _total
                        _prog.progress(min(_overall, 1.0))
                        _msg.markdown(f"**{_pl}** ΓÇö {phase}: {text}")

                    try:
                        _dir, _n = generate_crops_for_condition(
                            _sj, _cond,
                            world_video_dir=_wvd,
                            force=_force_regen,
                            progress_cb=_crop_cb,
                        )
                        _msg.markdown(f"**{_pair_label}** ΓÇö {_n} crops")
                    except Exception as _e:
                        st.error(f"Error generating crops for {_pair_label}: {_e}")

                _prog.progress(1.0)
                _status.update(label="Crop generation complete!", state="complete")

            st.cache_data.clear()
            st.rerun()
    else:
        st.info("No subject data found. Add data directories under `data/sj{NN}/`.")

    st.info(
        "**Required per subject/condition**: world video file + `fixations.csv` "
        "+ `gaze_positions.csv` in `data/sj{N}/eye/{Condition}/`."
    )


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# TAB 2 ΓÇö Label
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

with t_label:
    st.header("Annotate Gaze Crops")

    # Subject / condition selector
    all_pairs = _get_all_pairs_with_crops()
    pairs_with_crops = [(sj, cond, n) for sj, cond, n in all_pairs if n > 0]

    if not all_pairs:
        st.error(
            "No subjects found in `data/`. "
            "Make sure data is in `data/sj03/`, `data/sj04/`, etc."
        )
        st.stop()

    pair_labels = [
        f"sj{sj:02d}  {cond}  ({n} crops)" if n > 0 else f"sj{sj:02d}  {cond}  ΓÇö no crops"
        for sj, cond, n in all_pairs
    ]
    selected_idx = st.selectbox(
        "Subject / Condition",
        range(len(all_pairs)),
        format_func=lambda i: pair_labels[i],
    )
    sj_num, condition, n_crops_available = all_pairs[selected_idx]

    if n_crops_available == 0:
        st.warning(
            f"No crops yet for **sj{sj_num:02d} {condition}**. "
            "Go to the Generate Crops tab for instructions."
        )
        st.stop()

    # Load all PNGs + per-labeler labeled set
    _crop_dir = get_crop_dir(sj_num, condition)
    if not os.path.isdir(_crop_dir):
        st.warning(
            f"Crop directory missing for **sj{sj_num:02d} {condition}**: "
            f"`{_crop_dir}`. Generate crops first from the Generate Crops tab."
        )
        st.stop()
    all_pngs = sorted(
        f for f in os.listdir(_crop_dir)
        if f.lower().endswith(".png")
    )
    subject_labels_df = load_labels_for(sj_num, condition)

    if labeler_id:
        my_labels = subject_labels_df[subject_labels_df["labeler_id"] == labeler_id]
    else:
        my_labels = subject_labels_df

    labeled_set = set(my_labels["filename"].values)
    unlabeled = [f for f in all_pngs if f not in labeled_set]

    # Category color legend
    legend_parts = []
    for i, name in enumerate(LABEL_NAMES):
        color = CATEGORY_COLORS.get(name, "#888")
        legend_parts.append(
            f"<span style='background:{color}; color:#fff; "
            f"padding:2px 10px; border-radius:4px; margin-right:6px; "
            f"font-weight:600'>{i+1} {name}</span>"
        )
    st.markdown(" ".join(legend_parts), unsafe_allow_html=True)
    st.markdown("")

    # Per-category progress
    per_cat = (
        my_labels[~my_labels["is_flagged"]]["human_label"].value_counts()
        if not my_labels.empty else pd.Series(dtype=int)
    )
    target_per_cat = st.number_input(
        "Target per category", value=100, min_value=5, step=25,
        help="Suggested label count per class."
    )
    prog_cols = st.columns(len(LABEL_NAMES))
    for i, cat in enumerate(LABEL_NAMES):
        cnt = int(per_cat.get(cat, 0))
        color = CATEGORY_COLORS.get(cat, "#888")
        pct = min(cnt / target_per_cat, 1.0)
        with prog_cols[i]:
            st.markdown(
                f"<div style='font-size:11px; color:{color}; font-weight:600'>"
                f"{cat}<br/>{cnt}/{target_per_cat}</div>",
                unsafe_allow_html=True,
            )
            st.progress(pct)

    st.markdown("---")

    if not unlabeled:
        n_flagged = int(my_labels["is_flagged"].sum()) if not my_labels.empty else 0
        st.success(
            f"All {len(all_pngs)} crops labeled!"
            + (f" ({n_flagged} flagged for review)" if n_flagged else "")
        )
    else:
        # Reset crop index when switching subject/condition
        pair_key = (sj_num, condition, labeler_id)
        if st.session_state.last_pair != pair_key:
            st.session_state.crop_idx[pair_key] = 0
            st.session_state.last_pair = pair_key

        idx = st.session_state.crop_idx.get(pair_key, 0)
        if idx >= len(unlabeled):
            idx = 0
            st.session_state.crop_idx[pair_key] = 0

        fname = unlabeled[idx]
        parts = fname.replace(".png", "").split("_")
        fix_id = parts[0] if parts else fname
        ts_ns = parts[1] if len(parts) > 1 else ""

        img_path = get_crop_path(sj_num, condition, fname)

        col_img, col_ctrl = st.columns([2, 1])

        with col_img:
            if os.path.exists(img_path):
                st.image(img_path, width=448,
                         caption=f"sj{sj_num:02d} {condition} ΓÇö fixation {fix_id}")
            else:
                st.warning(f"Image not found: {img_path}")

        with col_ctrl:
            st.markdown(f"**{idx + 1}** of {len(unlabeled)} remaining  "
                        f"({len(labeled_set)} labeled by "
                        f"{'you' if labeler_id else 'anyone'})")
            st.markdown(f"Fix ID: `{fix_id}`")

            chosen = None
            btn_cols = st.columns(2)
            for i, name in enumerate(LABEL_NAMES):
                if btn_cols[i % 2].button(
                    f"{i+1}  {name}", key=f"btn_{name}", use_container_width=True
                ):
                    chosen = name

            st.markdown("---")
            col_skip, col_flag = st.columns(2)

            if col_skip.button("Skip", use_container_width=True):
                st.session_state.crop_idx[pair_key] = idx + 1
                st.rerun()

            if col_flag.button("ΓÜæ Flag / Ambiguous", use_container_width=True,
                               help="Mark as ambiguous and move on ΓÇö shown in review queue"):
                append_label(sj_num, condition, fix_id, ts_ns, fname,
                             FLAG_LABEL, labeler_id=labeler_id, is_flagged=True)
                st.session_state.crop_idx[pair_key] = idx + 1
                st.cache_data.clear()
                st.rerun()

        if chosen:
            if not labeler_id:
                st.toast("Tip: enter a labeler ID in the sidebar for inter-rater tracking.")
            append_label(sj_num, condition, fix_id, ts_ns, fname, chosen,
                         labeler_id=labeler_id, is_flagged=False)
            st.session_state.crop_idx[pair_key] = idx + 1
            st.cache_data.clear()
            st.rerun()

    # ΓöÇΓöÇ Undo + review ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    st.markdown("---")
    undo_col, _ = st.columns([1, 3])
    if undo_col.button("Γå⌐ Undo last label"):
        pair_key = (sj_num, condition, labeler_id)
        if remove_last_label(sj_num, condition, labeler_id=labeler_id):
            idx = st.session_state.crop_idx.get(pair_key, 1)
            st.session_state.crop_idx[pair_key] = max(0, idx - 1)
            st.cache_data.clear()
            st.rerun()

    with st.expander("Review labeled crops", expanded=False):
        review_df = subject_labels_df if subject_labels_df is not None else pd.DataFrame()
        if review_df.empty:
            st.info("No labels yet.")
        else:
            rc1, rc2 = st.columns([1, 2])
            filter_cat = rc1.selectbox("Filter category", ["all"] + LABEL_NAMES + ["flagged"])
            filter_labeler = rc2.selectbox(
                "Filter labeler", ["all"] + list(review_df["labeler_id"].dropna().unique())
            )
            show_df = review_df.copy()
            if filter_cat != "all":
                if filter_cat == "flagged":
                    show_df = show_df[show_df["is_flagged"]]
                else:
                    show_df = show_df[show_df["human_label"] == filter_cat]
            if filter_labeler != "all":
                show_df = show_df[show_df["labeler_id"] == filter_labeler]

            sample = show_df.tail(min(20, len(show_df)))
            cols = st.columns(5)
            for i, (_, row) in enumerate(sample.iterrows()):
                fpath = get_crop_path(sj_num, condition, row["filename"])
                if os.path.exists(fpath):
                    with cols[i % 5]:
                        color = CATEGORY_COLORS.get(row["human_label"], "#888")
                        flag_marker = " ΓÜæ" if row.get("is_flagged") else ""
                        st.image(fpath, width=120)
                        st.markdown(
                            f"<span style='color:{color}; font-weight:600'>"
                            f"{row['human_label']}{flag_marker}</span>",
                            unsafe_allow_html=True,
                        )


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# TAB 3 ΓÇö Statistics
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

with t_stats:
    st.header("Label Statistics & Quality")

    trainable = _load_trainable()
    all_labels_df = _load_all_labels()

    if trainable.empty:
        st.info("No trainable labels yet. Start labeling in the Label tab.")
    else:
        # ΓöÇΓöÇ Class distribution ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        st.subheader("Class distribution (trainable labels)")

        counts = (
            trainable["human_label"].value_counts()
            .reindex(LABEL_NAMES, fill_value=0)
            .reset_index()
        )
        counts.columns = ["category", "count"]

        bar_colors = [CATEGORY_COLORS.get(c, "#888") for c in counts["category"]]

        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=counts["category"], y=counts["count"],
                marker_color=bar_colors,
                text=counts["count"], textposition="outside",
            ))
            fig.update_layout(margin=dict(t=20, b=20), height=300,
                              yaxis_title="Count", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(counts.set_index("category")["count"])

        # ΓöÇΓöÇ Per-subject/condition coverage ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        st.subheader("Coverage per subject ├ù condition")

        sj_cond = subject_condition_counts()
        if not sj_cond.empty:
            # Build a pivot table
            try:
                pivot = sj_cond.pivot_table(
                    index="subject_id", columns="condition", values="count", fill_value=0
                )
                st.dataframe(pivot, use_container_width=True)
            except Exception:
                st.dataframe(sj_cond, hide_index=True, use_container_width=True)

        # ΓöÇΓöÇ Flagged crops ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        st.subheader("Flagged / ambiguous crops")

        flagged = load_flagged()
        if flagged.empty:
            st.success("No flagged crops.")
        else:
            st.markdown(f"**{len(flagged)}** crops flagged for review.")

            all_pairs_flag = _get_all_pairs_with_crops()
            pairs_with_flags = flagged[["subject_id", "condition"]].drop_duplicates()

            for _, pr in pairs_with_flags.iterrows():
                f_sj, f_cond = int(pr["subject_id"]), str(pr["condition"])
                sub_flagged = flagged[
                    (flagged["subject_id"] == f_sj) & (flagged["condition"] == f_cond)
                ]
                with st.expander(f"sj{f_sj:02d} {f_cond} ΓÇö {len(sub_flagged)} flagged"):
                    flag_cols = st.columns(5)
                    for i, (_, row) in enumerate(sub_flagged.head(20).iterrows()):
                        fpath = get_crop_path(f_sj, f_cond, row["filename"])
                        if os.path.exists(fpath):
                            with flag_cols[i % 5]:
                                st.image(fpath, width=100)
                                st.caption(
                                    f"by {row.get('labeler_id','?') or '?'}"
                                )
                                new_lbl = st.selectbox(
                                    "Re-label",
                                    ["(keep flagged)"] + LABEL_NAMES,
                                    key=f"relabel_{f_sj}_{f_cond}_{row['filename']}",
                                )
                                if new_lbl != "(keep flagged)":
                                    if st.button("Apply", key=f"apply_{row['filename']}"):
                                        relabel(f_sj, f_cond, row["filename"],
                                                new_lbl, labeler_id=str(row.get("labeler_id", "")))
                                        st.cache_data.clear()
                                        st.rerun()

        # ΓöÇΓöÇ Inter-rater agreement ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        st.subheader("Inter-rater reliability")

        known_labelers = labeler_ids()
        if len(known_labelers) < 2:
            st.info(
                "Need labels from ΓëÑ2 labelers with overlapping crops to compute agreement. "
                "Make sure labelers annotate some of the same crops (different labelers can "
                "label any crop since labeled_set is tracked per labeler ID)."
            )
        else:
            st.markdown(f"Labelers detected: {', '.join(f'**{l}**' for l in known_labelers)}")

            overlaps = inter_rater_overlaps()
            if overlaps.empty:
                st.info(
                    "No overlapping annotations yet. "
                    "Two labelers need to label the same crops for agreement metrics."
                )
            else:
                n_overlap = len(overlaps)
                pct_agree = overlaps["agree"].mean()
                st.markdown(f"**{n_overlap}** crop pairs annotated by 2+ labelers.  "
                            f"Overall agreement: **{pct_agree:.1%}**")

                kappa_df = cohens_kappa_matrix()
                if not kappa_df.empty:
                    st.markdown("**Cohen's Kappa per labeler pair:**")
                    st.dataframe(
                        kappa_df.style.background_gradient(
                            subset=["kappa"], cmap="RdYlGn", vmin=0, vmax=1
                        ),
                        hide_index=True,
                        use_container_width=True,
                    )
                    st.caption(
                        "╬║ < 0.4 = poor,  0.4ΓÇô0.6 = moderate,  "
                        "0.6ΓÇô0.8 = substantial,  > 0.8 = near-perfect"
                    )

                with st.expander("View disagreements"):
                    disagree = overlaps[~overlaps["agree"]]
                    if disagree.empty:
                        st.success("All overlapping labels agree!")
                    else:
                        st.dataframe(disagree, hide_index=True, use_container_width=True)

        # ΓöÇΓöÇ Download ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        st.markdown("---")
        if not all_labels_df.empty:
            st.download_button(
                "Γ¼ç Download all labels CSV",
                all_labels_df.to_csv(index=False),
                file_name="human_labels.csv",
                mime="text/csv",
            )


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# TAB 4 ΓÇö Train
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

with t_train:
    st.header("Train Vision Models")

    trainable_for_train = _load_trainable()

    if len(trainable_for_train) < 20:
        st.warning(
            f"Only {len(trainable_for_train)} trainable labels. "
            "Label at least 20 crops across categories before training."
        )
    else:
        st.markdown(f"**{len(trainable_for_train)}** trainable labels available.")

        # ΓöÇΓöÇ Configuration ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        cfg_col1, cfg_col2 = st.columns(2)

        with cfg_col1:
            st.subheader("Split Strategy")
            split_strategy = st.radio(
                "Split strategy",
                ["within_subject", "cross_subject"],
                format_func=lambda s: {
                    "within_subject": "Within-subject  (70/15/15 stratified random)",
                    "cross_subject": "Cross-subject  (leave-one-subject-out)",
                }[s],
                help=(
                    "Within-subject: random train/val/test from same sessions. "
                    "Cross-subject: hold out one subject entirely for test ΓÇö "
                    "measures generalisation to unseen people."
                ),
            )

            if split_strategy == "cross_subject":
                all_sjs = sorted(trainable_for_train["subject_id"].unique())
                if len(all_sjs) < 2:
                    st.warning("Cross-subject split requires labels from ΓëÑ2 subjects.")
                    split_strategy = "within_subject"
                else:
                    test_sj = st.selectbox(
                        "Hold-out test subject",
                        all_sjs,
                        index=len(all_sjs) - 1,
                        format_func=lambda s: f"sj{s:02d}",
                    )
            else:
                test_sj = None

        with cfg_col2:
            st.subheader("Model Settings")

            resnet_epochs = st.slider("ResNet epochs", 10, 60, 30, 5)
            resnet_batch = st.selectbox("Batch size", [16, 32, 64], index=1)

        st.markdown("---")
        os.makedirs(MODELS_DIR, exist_ok=True)

        # ── ResNet Training ──────────────────────────────────────────
        st.subheader("ResNet-50 Fine-Tuned")

            try:
                from torchvision import models as _tv
                from vision.resnet_head import train_from_label_store as train_from_label_store_rn
                _has_tv = True
            except ImportError:
                _has_tv = False
                train_from_label_store_rn = None

            if not _has_tv:
                st.error("torchvision not installed. Run `pip install torchvision Pillow`.")
            else:
                crops_ok = any(n > 0 for _, _, n in _get_all_pairs_with_crops())
                if not crops_ok:
                    st.warning("No crop PNGs found in `data/crops/`. Generate crops first.")
                else:
                    if st.button("Γû╢ Train ResNet-50", type="primary", key="btn_train_resnet"):
                        from vision.resnet_head import train_from_label_store as train_rn

                        progress_bar_rn = st.progress(0.0)
                        status_text_rn = st.empty()
                        chart_placeholder_rn = st.empty()
                        resnet_hist = []

                        def resnet_cb(epoch, n_epochs, metrics):
                            progress_bar_rn.progress(epoch / n_epochs)
                            status_text_rn.markdown(
                                f"Epoch **{epoch}/{n_epochs}** ΓÇö "
                                f"train loss: `{metrics['train_loss']}` ΓÇö "
                                f"val loss: `{metrics['val_loss']}` ΓÇö "
                                f"val acc: `{metrics['val_acc']}`"
                            )
                            resnet_hist.append({"epoch": epoch, **metrics})
                            if len(resnet_hist) > 1:
                                hist_df = pd.DataFrame(resnet_hist).set_index("epoch")
                                chart_placeholder_rn.line_chart(
                                    hist_df[["train_loss", "val_loss", "val_acc"]]
                                )

                        timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
                        rn_path = os.path.join(MODELS_DIR, f"resnet50_{timestamp}.pt")

                        with st.spinner("Training ResNet-50 (this may take several minutes)ΓÇª"):
                            stats_rn = train_from_label_store_rn(
                                out_path=rn_path,
                                test_size=0.15,
                                val_size=0.15,
                                n_epochs=resnet_epochs,
                                batch_size=resnet_batch,
                                progress_cb=resnet_cb,
                            )

                        progress_bar_rn.progress(1.0)
                        status_text_rn.empty()
                        if stats_rn:
                            st.success(
                                f"Γ£à ResNet-50 trained ΓÇö "
                                f"best val acc: **{stats_rn['best_val_acc']:.1%}**  "
                                f"test acc: **{stats_rn.get('test_acc', '?'):.1%}**  "
                                f"ΓåÆ `{os.path.basename(rn_path)}`"
                            )
                            st.session_state.train_history_resnet = resnet_hist
                        else:
                            st.error("Training failed ΓÇö not enough data or missing crops.")
                        st.cache_data.clear()

        # ── Saved models ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Saved models")

        _pt_files = sorted(
            f for f in os.listdir(MODELS_DIR)
            if f.endswith(".pt") and os.path.isfile(os.path.join(MODELS_DIR, f))
        ) if os.path.isdir(MODELS_DIR) else []
        if not _pt_files:
            st.info("No saved models yet.")
        else:
            rows = []
            for f in _pt_files:
                fp = os.path.join(MODELS_DIR, f)
                size_mb = os.path.getsize(fp) / 1e6
                rows.append({"File": f, "Size (MB)": f"{size_mb:.1f}"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# TAB 5 — Evaluate (ResNet-50 evaluation)
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

with t_eval:
    st.header("Model Evaluation")
    st.markdown(
        "Evaluate **ResNet-50** on your labeled crops using repeated stratified splits."
    )

    _prev_results = _load_vision_comparison()

    if _prev_results and "summary" in _prev_results:
        st.subheader("Previous Results")
        _, comp_df = generate_vision_table(_prev_results)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

        best = _prev_results["summary"].get("best_model")
        if best:
            best_f1 = _prev_results["summary"][best]["macro_f1_mean"]
            _model_display = {
                "resnet50": "ResNet-50",
            }
            st.success(
                f"Best model: **{_model_display.get(best, best)}** "
                f"(macro F1 = {best_f1:.3f})"
            )

        with st.expander("Confusion Matrices"):
            import matplotlib
            matplotlib.use("Agg")

            _cm_models = {
                "resnet50": "ResNet-50",
            }
            label_names = _prev_results.get("label_names", LABEL_NAMES)
            for key, display in _cm_models.items():
                s = _prev_results["summary"].get(key)
                if s and "confusion_matrix" in s:
                    fig = make_cm_figure(
                        s["confusion_matrix"], label_names, title=display
                    )
                    st.pyplot(fig)
                    import matplotlib.pyplot as plt
                    plt.close(fig)

    st.markdown("---")
    st.subheader("Run New Comparison")

    n_repeats = st.slider(
        "Number of repeated splits", 1, 10, 3,
        help="More repeats = more stable estimates, but slower."
    )

    if st.button("Run Comparison", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _eval_cb(step, total_steps, message):
            progress_bar.progress(step / total_steps)
            status_text.markdown(f"**Step {step}/{total_steps}**: {message}")

        with st.spinner("Running model comparison..."):
            result = evaluate_vision_models(
                n_repeats=n_repeats, progress_cb=_eval_cb
            )

        progress_bar.progress(1.0)
        status_text.empty()

        if result and "error" not in result:
            st.cache_data.clear()
            _, comp_df = generate_vision_table(result)
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

            best = result["summary"].get("best_model")
            if best:
                best_f1 = result["summary"][best]["macro_f1_mean"]
                _model_display = {
                    "resnet50": "ResNet-50",
                }
                st.success(
                    f"Best model: **{_model_display.get(best, best)}** "
                    f"(macro F1 = {best_f1:.3f})"
                )
            st.info("Results saved to `results/vision_comparison.json`.")
        elif result and result.get("error") == "insufficient_data":
            st.error(
                "Not enough labeled data. Label at least 20 crops "
                "in the Label tab before running evaluation."
            )
        else:
            st.error("Evaluation failed. Check console for details.")

    if not _prev_results:
        st.info(
            "No previous results found. Click **Run Comparison** above to "
            "evaluate ResNet-50 on your labeled crops."
        )


# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ
# TAB 6 ΓÇö Deploy (Classify all crops with best model)
# ΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉΓòÉ

with t_deploy:
    st.header("Deploy Best Model")
    st.markdown(
        "Classify **unlabeled** gaze crops with the selected model and merge "
        "with existing `vision_results` + **human labels** (human labels are "
        "never overwritten). Regenerates `vision_trial_features.csv` for fusion."
    )

    # ΓöÇΓöÇ Evaluation summary ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    _deploy_results = _load_vision_comparison()

    if _deploy_results and "summary" in _deploy_results:
        best = _deploy_results["summary"].get("best_model")
        if best:
            best_f1 = _deploy_results["summary"][best]["macro_f1_mean"]
            st.success(
                f"Evaluation: ResNet-50 macro F1 = {best_f1:.3f}"
            )
    else:
        st.info(
            "No evaluation results yet. Run one in the **Evaluate** tab first, "
            "or deploy directly below."
        )

    # ── Model check ──────────────────────────────────────────────
    _has_resnet = (
        os.path.isdir(MODELS_DIR) and (
            os.path.exists(os.path.join(MODELS_DIR, "resnet50.pt")) or
            any(f.startswith("resnet") and f.endswith(".pt")
                for f in os.listdir(MODELS_DIR) if os.path.isfile(os.path.join(MODELS_DIR, f)))
        )
    )
    deploy_model = "resnet50"

    if not _has_resnet:
        st.error("No ResNet-50 model found in `models/`. Train one first.")
    else:

        # ΓöÇΓöÇ Crop inventory ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
        st.markdown("---")
        st.subheader("Crop Inventory")

        from vision.label_store import CROPS_BASE
        _crop_dirs = []
        if os.path.isdir(CROPS_BASE):
            for d in sorted(os.listdir(CROPS_BASE)):
                dp = os.path.join(CROPS_BASE, d)
                if os.path.isdir(dp):
                    n = sum(1 for f in os.listdir(dp) if f.endswith(".png"))
                    _crop_dirs.append({"Directory": d, "Crops": n})

        if not _crop_dirs:
            st.warning(
                "No crops found in `data/crops/`. "
                "Generate crops in the **Generate Crops** tab first."
            )
        else:
            st.dataframe(
                pd.DataFrame(_crop_dirs), hide_index=True,
                use_container_width=True,
            )
            total_crops = sum(r["Crops"] for r in _crop_dirs)
            from vision.label_store import trainable_labeled_crop_keys
            _labeled_keys = trainable_labeled_crop_keys()
            _n_human = len(_labeled_keys)
            st.markdown(
                f"**Total: {total_crops} crops** across "
                f"{len(_crop_dirs)} subject/condition pairs · "
                f"**{_n_human}** human-labeled (skipped when box below is on)"
            )

            # ΓöÇΓöÇ Check existing output ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
            _existing_features = []
            if os.path.isdir(VISION_FEATURES_DIR):
                _existing_features = [
                    f for f in os.listdir(VISION_FEATURES_DIR)
                    if f.endswith("_vision_trial_features.csv")
                ]
            if _existing_features:
                st.info(
                    f"{len(_existing_features)} feature files already exist "
                    f"in `data/vision_features/`."
                )

            # ── Deploy button ─────────────────────────────────────────────
            st.markdown("---")
            _n_hand = len(load_trainable_labels())
            if _n_hand > 0:
                st.info(
                    f"**{_n_hand} hand-labeled crops** are stored in "
                    f"`data/human_labels.csv` and will **not** be overwritten "
                    f"— the model only classifies unlabeled crops."
                )
            only_unlabeled = st.checkbox(
                "Only classify crops without a human label",
                value=True,
                help="Skips rows in data/human_labels.csv (trainable labels). "
                     "For long runs, CLI is slightly faster: "
                     "python src/evaluate.py --deploy --deploy-model resnet50",
            )
            if st.button("Deploy Model", type="primary"):
                os.makedirs(VISION_FEATURES_DIR, exist_ok=True)

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def _deploy_cb(step, total, message):
                    progress_bar.progress(step / total)
                    status_text.markdown(
                        f"**{step}/{total}**: {message}"
                    )

                with st.spinner("Classifying unlabeled crops with ResNet-50..."):
                    deploy_result = relabel_crops_with_best(
                        deploy_model,
                        run_name=None,
                        progress_cb=_deploy_cb,
                        output_dir=VISION_FEATURES_DIR,
                        only_unlabeled=only_unlabeled,
                    )

                progress_bar.progress(1.0)
                status_text.empty()

                if deploy_result and "error" not in deploy_result:
                    skipped = deploy_result.get("total_skipped_human_labeled", 0)
                    st.success(
                        f"Done! Model classified **{deploy_result['total_relabeled']}** "
                        f"new crop(s)"
                        + (f" (skipped **{skipped}** human-labeled)."
                           if skipped else ".")
                        + " Features written to `data/vision_features/`."
                    )
                    if deploy_result.get("conditions"):
                        rows = []
                        for cs in deploy_result["conditions"]:
                            rows.append({
                                "Subject": f"sj{cs['sj_num']:02d}",
                                "Condition": cs["condition"],
                                "In results CSV": cs["n_crops"],
                                "New (model)": cs.get("n_model_classified", 0),
                                "Human": cs.get("n_human_labeled", 0),
                            })
                        st.dataframe(
                            pd.DataFrame(rows), hide_index=True,
                            use_container_width=True,
                        )
                    st.cache_data.clear()
                elif deploy_result and deploy_result.get("error"):
                    st.error(f"Deploy failed: {deploy_result['error']}")
                else:
                    st.error("Deploy failed. Check console for details.")
