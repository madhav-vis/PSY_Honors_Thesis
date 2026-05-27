"""
Mobile EEG + Eyetracking Pipeline Dashboard
Run:  streamlit run src/dashboard.py
"""

import gc
import json
import multiprocessing as _mp
import os
import subprocess
import sys
import threading
import time
from collections import deque

# Bail out fast if we were imported as a spawned multiprocessing worker
# (see stream_annotator.py for the rationale).
if _mp.current_process().name != "MainProcess":
    sys.exit(0)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
import yaml
import mne
from plotly.subplots import make_subplots

import et_viz
from pipeline_progress import read_progress, clear_progress

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "run_config.yaml")


def _find_venv_python() -> str:
    """Locate the project venv interpreter across OSes.

    Falls back to sys.executable so the dashboard still works when
    launched from an already-activated venv or a non-standard layout.
    """
    candidates = [
        os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe"),  # Windows
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python"),          # macOS/Linux
        os.path.join(PROJECT_ROOT, ".venv", "bin", "python3.11"),      # legacy
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return sys.executable


VENV_PYTHON = _find_venv_python()

st.set_page_config(
    page_title="Mobile EEG + Eyetracking Pipeline",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ──────────────────────────────────────────────────

@st.cache_data(ttl=30)
def list_runs():
    if not os.path.isdir(RUNS_ROOT):
        return []
    return sorted(
        [d for d in os.listdir(RUNS_ROOT)
         if os.path.isdir(os.path.join(RUNS_ROOT, d))],
        reverse=True,
    )


def run_dir(run_name):
    return os.path.join(RUNS_ROOT, run_name)


def data_dir(rn):
    return os.path.join(run_dir(rn), "data")


def plots_dir(rn):
    return os.path.join(run_dir(rn), "plots")


def vision_dir(rn, sj, cond):
    return os.path.join(run_dir(rn), "vision", f"sj{sj:02d}_{cond}")


@st.cache_data(ttl=60)
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=60)
def load_first_timestamp_s(path):
    if not os.path.exists(path):
        return None
    try:
        row = pd.read_csv(path, usecols=["timestamp [ns]"], nrows=1)
        if row.empty:
            return None
        return float(row["timestamp [ns]"].iloc[0]) / 1e9
    except Exception:
        return None


def _project_data_root():
    env = os.environ.get("PSY197B_DATA_DIR")
    if env:
        return env
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        root = cfg.get("data", {}).get("root", "data")
        return root if os.path.isabs(root) else os.path.join(PROJECT_ROOT, root)
    except Exception:
        return os.path.join(PROJECT_ROOT, "data")


def _et_folder_map():
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("et", {}).get("folder_map", {})
    except Exception:
        return {}


@st.cache_data(ttl=120, max_entries=8)
def _load_epochs_cached(epo_path):
    """Cache-friendly epoch loader — returns serializable dicts instead of MNE objects."""
    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
    meta = epochs.metadata
    result = {
        "data": epochs.get_data(),
        "times": epochs.times,
        "ch_names": list(epochs.ch_names),
        "metadata": meta.to_dict("list") if meta is not None else None,
        "events": epochs.events,
    }
    del epochs
    return result


def _cached_metadata(ep):
    """Reconstruct a DataFrame from the cached dict representation."""
    if ep["metadata"] is None:
        return None
    return pd.DataFrame(ep["metadata"])


@st.cache_data(ttl=120)
def _load_or_build_erp_cache(rn, sj, conds_tuple):
    """Load precomputed ERP summaries from disk, or build + persist them."""
    cache_path = os.path.join(data_dir(rn), f"dashboard_cache_sj{sj:02d}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("version") == 2:
            return cached
        os.remove(cache_path)

    cache = {"version": 2, "erp_by_cell": {}, "errors": [], "missing": []}
    for cond in conds_tuple:
        epo_path = os.path.join(data_dir(rn), f"sj{sj:02d}_{cond}_Features-epo.fif")
        if not os.path.exists(epo_path):
            cache["missing"].append(cond)
            continue
        try:
            ep = _load_epochs_cached(epo_path)
            data_ep = ep["data"]
            if len(data_ep) == 0:
                cache["errors"].append(f"{cond}: empty")
                continue
            ch = "Pz" if "Pz" in ep["ch_names"] else ep["ch_names"][0]
            ch_i = ep["ch_names"].index(ch)
            meta = _cached_metadata(ep)
            if meta is not None and "trialType" in meta.columns:
                raw = pd.to_numeric(meta["trialType"], errors="coerce").to_numpy(dtype=float)
            else:
                raw = ep["events"][:, 2].astype(float)
            go_idx = np.flatnonzero(raw == 10.0)
            nogo_idx = np.flatnonzero(raw == 20.0)
            if len(go_idx) == 0 and len(nogo_idx) == 0:
                cache["errors"].append(f"{cond}: no Go/NoGo trials")
                continue
            cell = _condition_grid_label(cond)
            go_traces = data_ep[go_idx, ch_i, :] * 1e6 if len(go_idx) else None
            nogo_traces = data_ep[nogo_idx, ch_i, :] * 1e6 if len(nogo_idx) else None
            cache["erp_by_cell"][cell] = {
                "times_ms": (ep["times"] * 1000.0).tolist(),
                "go": go_traces.mean(0).tolist() if go_traces is not None else None,
                "go_sem": (go_traces.std(0) / np.sqrt(len(go_idx))).tolist() if go_traces is not None else None,
                "nogo": nogo_traces.mean(0).tolist() if nogo_traces is not None else None,
                "nogo_sem": (nogo_traces.std(0) / np.sqrt(len(nogo_idx))).tolist() if nogo_traces is not None else None,
                "n_go": int(len(go_idx)),
                "n_nogo": int(len(nogo_idx)),
            }
        except Exception as e:
            cache["errors"].append(f"{cond}: {e}")

    try:
        with open(cache_path, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass
    return cache


# ── Cross-subject aggregation helpers ────────────────────────────


@st.cache_data(ttl=180)
def _load_or_build_all_erp_cache(rn, subjects_tuple, conds_tuple):
    """Grand-average ERP traces across `subjects_tuple` for each grid cell.

    Reads the per-subject `dashboard_cache_sj{NN}.json` caches (building them
    on the fly if missing) and averages the per-cell `go`/`nogo` arrays. n
    counts are summed across subjects. Times are taken from the first
    subject that contributes to the cell (all subjects share the epoch
    timeline because the pipeline enforces sfreq/tmin/tmax).
    """
    agg = {
        "erp_by_cell": {},
        "errors": [],
        "missing": [],
        "n_subjects_by_cell": {},
    }
    per_cell_acc = {}
    for sj in subjects_tuple:
        sub_cache = _load_or_build_erp_cache(rn, sj, conds_tuple)
        for msg in sub_cache.get("errors", []):
            agg["errors"].append(f"sj{sj:02d}: {msg}")
        for cell, item in sub_cache.get("erp_by_cell", {}).items():
            slot = per_cell_acc.setdefault(cell, {
                "times_ms": item["times_ms"],
                "go_stack": [],
                "nogo_stack": [],
                "n_go": 0,
                "n_nogo": 0,
                "n_subjects": 0,
                "go_sem_fallback": None,
                "nogo_sem_fallback": None,
            })
            if item.get("go") is not None:
                slot["go_stack"].append(np.asarray(item["go"], dtype=float))
                slot["n_go"] += int(item.get("n_go", 0))
                slot["go_sem_fallback"] = item.get("go_sem")
            if item.get("nogo") is not None:
                slot["nogo_stack"].append(np.asarray(item["nogo"], dtype=float))
                slot["n_nogo"] += int(item.get("n_nogo", 0))
                slot["nogo_sem_fallback"] = item.get("nogo_sem")
            slot["n_subjects"] += 1

    for cell, slot in per_cell_acc.items():
        go_arr = np.vstack(slot["go_stack"]) if slot["go_stack"] else None
        nogo_arr = np.vstack(slot["nogo_stack"]) if slot["nogo_stack"] else None
        n_go_sj = len(slot["go_stack"])
        n_nogo_sj = len(slot["nogo_stack"])

        # When multiple subjects contribute, use between-subject SEM.
        # When only 1 subject contributes, fall back to that subject's
        # within-subject trial-level SEM (already stored in per-subject cache).
        go_sem = None
        nogo_sem = None
        if go_arr is not None:
            if n_go_sj > 1:
                go_sem = (go_arr.std(0) / np.sqrt(n_go_sj)).tolist()
            else:
                go_sem = slot.get("go_sem_fallback")
        if nogo_arr is not None:
            if n_nogo_sj > 1:
                nogo_sem = (nogo_arr.std(0) / np.sqrt(n_nogo_sj)).tolist()
            else:
                nogo_sem = slot.get("nogo_sem_fallback")

        agg["erp_by_cell"][cell] = {
            "times_ms": slot["times_ms"],
            "go": go_arr.mean(0).tolist() if go_arr is not None else None,
            "go_sem": go_sem,
            "nogo": nogo_arr.mean(0).tolist() if nogo_arr is not None else None,
            "nogo_sem": nogo_sem,
            "n_go": slot["n_go"],
            "n_nogo": slot["n_nogo"],
        }
        agg["n_subjects_by_cell"][cell] = slot["n_subjects"]

    # Track conditions where NO subject produced a feature epoch file
    present_cells = set(agg["erp_by_cell"].keys())
    expected_cells = {_condition_grid_label(c) for c in conds_tuple}
    agg["missing"] = sorted(expected_cells - present_cells)
    return agg


@st.cache_data(ttl=120)
def _aggregate_behavior(rn, subjects_tuple, conds_tuple):
    """Pool per-subject *_features.csv rows and return rows for the
    Sit/Walk behavior plot. Each row is one (subject, condition) pair
    so the existing `_plot_sit_walk_behavior_matplotlib` can take
    mean +/- SEM across subjects."""
    rows = []
    for sj in subjects_tuple:
        for cond in conds_tuple:
            _mv, _att = _parse_condition_parts(cond)
            if _att != "attend":
                continue
            feat = load_csv(os.path.join(
                data_dir(rn), f"sj{sj:02d}_{cond}_features.csv"
            ))
            row = _build_movement_behavior_rows(feat, cond)
            if row is not None:
                rows.append(row)
    return rows


@st.cache_data(ttl=120)
def _aggregate_et_metrics(rn, subjects_tuple, conds_tuple):
    """For each condition, mean +/- SD across subjects of total distance and mean rate.

    Returns a DataFrame with columns: condition, n_subjects,
    total_distance_mean, total_distance_sd, mean_rate_mean, mean_rate_sd.
    Uses the same per-subject Euclidean computation as the single-subject
    view via `et_viz.compute_euclidean`.
    """
    et_map = _et_folder_map()
    data_root = _project_data_root()
    rows = []
    for cond in conds_tuple:
        totals = []
        rates = []
        for sj in subjects_tuple:
            csv_path = os.path.join(
                data_root, f"sj{sj:02d}", "eye",
                et_map.get(cond, ""), "gaze_positions.csv",
            )
            if not os.path.exists(csv_path):
                continue
            euc = et_viz.compute_euclidean(csv_path)
            if euc is None:
                continue
            totals.append(float(euc["total_distance"]))
            rates.append(float(euc["mean_rate"]))
        if not totals:
            continue
        rows.append({
            "condition": cond,
            "n_subjects": len(totals),
            "total_distance_mean": float(np.mean(totals)),
            "total_distance_sd": float(np.std(totals, ddof=1)) if len(totals) > 1 else 0.0,
            "mean_rate_mean": float(np.mean(rates)),
            "mean_rate_sd": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=120)
def find_subjects_conditions(rn):
    dd = data_dir(rn)
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


def file_exists_icon(path):
    return "yes" if os.path.exists(path) else "no"


def _parse_condition_parts(cond_label):
    low = str(cond_label).lower()
    movement = "walk" if "walk" in low else ("sit" if "sit" in low else "other")
    attention = (
        "attend" if "attend" in low and "unattend" not in low
        else ("unattend" if "unattend" in low else "other")
    )
    return movement, attention


def _condition_grid_label(cond_label):
    movement, attention = _parse_condition_parts(cond_label)
    if movement in {"walk", "sit"} and attention in {"attend", "unattend"}:
        return f"{attention.title()} {movement.title()}"
    return cond_label


def _go_nogo_indices_from_epochs(epochs):
    """Indices for Go (10) and NoGo (20); tolerates string/object trialType in metadata."""
    if epochs.metadata is not None and "trialType" in epochs.metadata.columns:
        raw = epochs.metadata["trialType"].to_numpy()
    else:
        raw = epochs.events[:, 2]
    codes = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    go_idx = np.flatnonzero(codes == 10.0)
    nogo_idx = np.flatnonzero(codes == 20.0)
    return go_idx, nogo_idx


def _go_nogo_indices_from_cached(ep):
    """Same as _go_nogo_indices_from_epochs but for cached dict."""
    meta = _cached_metadata(ep)
    if meta is not None and "trialType" in meta.columns:
        raw = meta["trialType"].to_numpy()
    else:
        raw = ep["events"][:, 2]
    codes = pd.to_numeric(pd.Series(raw), errors="coerce").to_numpy(dtype=float)
    return np.flatnonzero(codes == 10.0), np.flatnonzero(codes == 20.0)


def _build_movement_behavior_rows(features_df, cond_label):
    if features_df is None or "outcome" not in features_df.columns:
        return None
    movement, _ = _parse_condition_parts(cond_label)
    if movement not in {"sit", "walk"}:
        return None
    # Use straightforward trial-level rates:
    # Hits = HIT / total trials, Commission errors = COMMISSION_ERROR / total trials.
    # This avoids empty bars when trialType has dtype inconsistencies.
    outcome = features_df["outcome"].astype(str).str.upper()
    p_correct = float((outcome == "HIT").mean()) if len(outcome) else np.nan
    p_error = float((outcome == "COMMISSION_ERROR").mean()) if len(outcome) else np.nan
    return {
        "movement": movement.title(),
        "pCorrect": p_correct,
        "pError": p_error,
    }


def _plot_sit_walk_behavior_matplotlib(beh_df):
    """
    1x2 bar chart over Sit/Walk with mean ± SEM for pCorrect and pError.
    Left axis is intentionally high-range for accuracy; right starts at zero.
    """
    order = ["Sit", "Walk"]
    agg = (
        beh_df.groupby("movement")[["pCorrect", "pError"]]
        .agg(["mean", "sem"])
        .reindex(order)
    )
    means_correct = agg[("pCorrect", "mean")].to_numpy(dtype=float)
    sem_correct = np.nan_to_num(agg[("pCorrect", "sem")].to_numpy(dtype=float), nan=0.0)
    means_error = agg[("pError", "mean")].to_numpy(dtype=float)
    sem_error = np.nan_to_num(agg[("pError", "sem")].to_numpy(dtype=float), nan=0.0)

    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    bar_color = "#1f77b4"
    err_kw = {"capsize": 5, "elinewidth": 1.5}

    axes[0].bar(
        x,
        means_correct,
        yerr=sem_correct,
        color=bar_color,
        width=0.6,
        error_kw=err_kw,
    )
    axes[0].set_title("Hits (pCorrect)")
    axes[0].set_xticks(x, order)
    axes[0].set_xlabel("Movement")
    axes[0].set_ylabel("Proportion of trials")
    valid_correct = means_correct[np.isfinite(means_correct)]
    if valid_correct.size:
        # Keep a zoomed view but never clip valid bars (e.g., ~0.40 when
        # averaging Attend + Unattend within movement).
        y0 = max(0.0, float(valid_correct.min() - 0.1))
        axes[0].set_ylim(y0, 1.0)
    else:
        axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(
        x,
        means_error,
        yerr=sem_error,
        color=bar_color,
        width=0.6,
        error_kw=err_kw,
    )
    axes[1].set_title("Commission Errors (pError)")
    axes[1].set_xticks(x, order)
    axes[1].set_xlabel("Movement")
    axes[1].set_ylabel("Proportion of trials")
    valid_error = means_error[np.isfinite(means_error)]
    ymax = min(1.0, float(valid_error.max() + 0.1)) if valid_error.size else 0.25
    axes[1].set_ylim(0.0, max(0.1, ymax))
    axes[1].grid(axis="y", alpha=0.25)

    return fig


# ── Sidebar ──────────────────────────────────────────────────

st.sidebar.header("Run History")

if st.sidebar.button("Clear memory cache"):
    st.cache_data.clear()
    gc.collect()
    st.rerun()

runs = list_runs()
if not runs:
    st.sidebar.warning("No runs found in runs/")
    selected_run = None
else:
    selected_run = st.sidebar.selectbox("Run", runs)

subjects = []
conditions = []

if selected_run:
    subjects, conditions = find_subjects_conditions(selected_run)
    if not conditions:
        cfg_path = os.path.join(run_dir(selected_run),
                                 "run_config_snapshot.yaml")
        try:
            with open(cfg_path) as _f:
                snap = yaml.safe_load(_f)
            conds_raw = snap.get("conditions", [])
            if isinstance(conds_raw, list):
                conditions = [c["eeg_label"] for c in conds_raw
                              if isinstance(c, dict) and "eeg_label" in c]
            elif isinstance(conds_raw, dict):
                conditions = sorted(conds_raw.keys())
        except Exception:
            pass


# ── Main page header ─────────────────────────────────────────

st.title("Mobile EEG + Eyetracking Pipeline")


# ── Tabs ─────────────────────────────────────────────────────

tab_run, tab_overview, tab_et, tab_nogo, tab_eval = st.tabs([
    "Run Manager",
    "Overview",
    "Eye Tracking",
    "EEGNet",
    "Evaluation",
])


# ════════════════════════════════════════════════════════════
# TAB 1 — RUN MANAGER
# ════════════════════════════════════════════════════════════

with tab_run:
    col_yaml, col_launch = st.columns([1, 1])

    with col_yaml:
        st.subheader("Pipeline Configuration")
        try:
            with open(CONFIG_PATH, "r") as f:
                yaml_text = f.read()
        except FileNotFoundError:
            yaml_text = ""
            st.error(f"Config not found: {CONFIG_PATH}")

        edited_yaml = st.text_area(
            "run_config.yaml",
            value=yaml_text,
            height=450,
            key="yaml_editor",
        )
        if st.button("Save Config", type="primary"):
            try:
                yaml.safe_load(edited_yaml)
                with open(CONFIG_PATH, "w") as f:
                    f.write(edited_yaml)
                st.success("Config saved!")
                st.cache_data.clear()
            except yaml.YAMLError as e:
                st.error(f"Invalid YAML: {e}")

    with col_launch:
        st.subheader("Run Pipeline")

        run_main = st.button("Run EEG/ET Pipeline", type="primary",
                              disabled=st.session_state.get("pipeline_running", False))
        run_overview = st.button("Run Overview Only",
                                 help="Steps 1–4 only (EEG, ET, Fuse, Features). "
                                      "Skips DL Prep and Sanity Checks.",
                                 disabled=st.session_state.get("pipeline_running", False))

        if "pipeline_log" not in st.session_state:
            st.session_state.pipeline_log = ""
        if "pipeline_running" not in st.session_state:
            st.session_state.pipeline_running = False

        def _run_pipeline_with_progress(cmd, label, progress_run_dir, timeout_s):
            """Run a pipeline subprocess with live progress bar polling."""
            st.session_state.pipeline_running = True
            st.session_state.pipeline_log = ""

            with st.status(f"Running {label}...", expanded=True) as status:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                log_placeholder = st.empty()

                try:
                    process = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        bufsize=1,
                        cwd=PROJECT_ROOT,
                    )

                    # Drain stdout on a background thread. Windows pipe buffers
                    # are ~4 KB; if we don't read, the child blocks on write
                    # the instant the buffer fills, causing a silent deadlock.
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
                    last_log_render = 0.0

                    while process.poll() is None:
                        if time.time() - start_time > timeout_s:
                            process.kill()
                            st.error(f"{label} timed out ({timeout_s // 60} min limit)")
                            break

                        if progress_run_dir:
                            prog = read_progress(progress_run_dir)
                            if prog:
                                frac = prog["current"] / max(prog["total"], 1)
                                progress_bar.progress(min(frac, 1.0))
                                status_text.markdown(f"**{prog['step']}** — {prog['message']}")

                        # Refresh the live log every couple of seconds so the
                        # user can see progress without redraw thrash.
                        now = time.time()
                        if now - last_log_render > 2.0:
                            with log_lock:
                                tail = "".join(list(log_lines)[-200:])
                            if tail:
                                log_placeholder.code(tail, language="text")
                            last_log_render = now

                        time.sleep(1.0)

                    drain_thread.join(timeout=5.0)
                    with log_lock:
                        full_log = "".join(log_lines)
                    st.session_state.pipeline_log = full_log

                    if process.returncode == 0:
                        progress_bar.progress(1.0)
                        status.update(label=f"{label} completed!", state="complete")
                    else:
                        status.update(label=f"{label} failed (exit {process.returncode})", state="error")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    st.session_state.pipeline_running = False
                    if progress_run_dir:
                        clear_progress(progress_run_dir)
                    st.cache_data.clear()

        if run_main and not st.session_state.pipeline_running:
            cmd = [VENV_PYTHON, os.path.join(PROJECT_ROOT, "src", "main.py"),
                   "all"]
            # Run dir is created by config.py at import time — read it from config
            _eeg_run_dir = None
            try:
                with open(CONFIG_PATH) as _f:
                    _cfg = yaml.safe_load(_f)
                _run_name = _cfg.get("run_name", "")
                _date = _cfg.get("date", "auto")
                if _date == "auto":
                    import datetime
                    _date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
                _eeg_run_dir = os.path.join(RUNS_ROOT, f"{_date}_{_run_name}")
            except Exception:
                pass
            _run_pipeline_with_progress(cmd, "EEG/ET pipeline", _eeg_run_dir, 14400)

        if run_overview and not st.session_state.pipeline_running:
            cmd = [VENV_PYTHON, os.path.join(PROJECT_ROOT, "src", "main.py"),
                   "eeg", "et", "fuse", "features"]
            _ov_run_dir = None
            try:
                with open(CONFIG_PATH) as _f:
                    _cfg = yaml.safe_load(_f)
                _run_name = _cfg.get("run_name", "")
                _date = _cfg.get("date", "auto")
                if _date == "auto":
                    import datetime
                    _date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
                _ov_run_dir = os.path.join(RUNS_ROOT, f"{_date}_{_run_name}")
            except Exception:
                pass
            _run_pipeline_with_progress(cmd, "Overview pipeline", _ov_run_dir, 14400)

        if st.session_state.pipeline_log:
            with st.expander("Pipeline Output", expanded=True):
                st.code(st.session_state.pipeline_log, language="text")


# ════════════════════════════════════════════════════════════
# TAB 2 — OVERVIEW
# ════════════════════════════════════════════════════════════

with tab_overview:
    if not selected_run:
        st.info("Select a run from the sidebar.")
    elif not subjects:
        st.info("No subjects with preprocessed data in this run yet.")
    else:
        _ov_mode = st.radio(
            "View mode",
            ["All subjects (average)", "Single subject"],
            index=0,
            horizontal=True,
            key="overview_view_mode",
        )
        aggregate_mode = _ov_mode.startswith("All")
        sj_num = None
        if not aggregate_mode:
            sj_num = st.selectbox(
                "Subject", subjects,
                format_func=lambda x: f"sj{x:02d}",
                key="overview_subject",
            )
        if aggregate_mode:
            st.header(f"Overview — All subjects (n={len(subjects)})")
        else:
            st.header(f"Overview — sj{sj_num:02d}")

        # ── 2×2 ERP Grid (Go vs NoGo, Attend/Unattend × Sit/Walk) ────
        st.subheader("Go vs NoGo ERPs — Pz")

        if aggregate_mode:
            erp_cache = _load_or_build_all_erp_cache(
                selected_run, tuple(subjects), tuple(conditions))
            missing_feature_epo = [
                (cell, "(no subject has data for this cell)")
                for cell in erp_cache.get("missing", [])]
        else:
            erp_cache = _load_or_build_erp_cache(
                selected_run, sj_num, tuple(conditions))
            missing_feature_epo = [
                (c, f"sj{sj_num:02d}_{c}_Features-epo.fif")
                for c in erp_cache.get("missing", [])]
        erp_by_cell = erp_cache["erp_by_cell"]
        erp_load_errors = erp_cache.get("errors", [])

        cell_order = ["Attend Sit", "Unattend Sit", "Attend Walk", "Unattend Walk"]
        pos = {
            "Attend Sit": (1, 1), "Unattend Sit": (1, 2),
            "Attend Walk": (2, 1), "Unattend Walk": (2, 2),
        }

        if missing_feature_epo or erp_load_errors:
            with st.expander("ERP data issues", expanded=True):
                for c, p in missing_feature_epo:
                    st.warning(f"`{c}` — missing `{os.path.basename(p)}`")
                for msg in erp_load_errors:
                    st.warning(msg)

        _shared_y_range = None
        if any(c in erp_by_cell for c in cell_order):
            fig_grid = make_subplots(
                rows=2, cols=2,
                subplot_titles=cell_order,
                shared_xaxes=True, shared_yaxes=True,
                vertical_spacing=0.14, horizontal_spacing=0.10,
            )
            for cell, (r, c) in pos.items():
                item = erp_by_cell.get(cell)
                if not item:
                    continue
                t = item["times_ms"]
                if item["nogo"] is not None:
                    nogo_y = np.asarray(item["nogo"])
                    fig_grid.add_trace(go.Scatter(
                        x=t, y=nogo_y.tolist(),
                        mode="lines", name=f"NoGo (n={item['n_nogo']})",
                        line=dict(color="#e74c3c", width=2),
                        showlegend=(cell == "Attend Sit"),
                    ), row=r, col=c)
                    if aggregate_mode and item.get("nogo_sem") is not None:
                        sem = np.asarray(item["nogo_sem"])
                        fig_grid.add_trace(go.Scatter(
                            x=t + t[::-1],
                            y=(nogo_y + sem).tolist() + (nogo_y - sem)[::-1].tolist(),
                            fill="toself", fillcolor="rgba(231,76,60,0.12)",
                            line=dict(width=0), showlegend=False, hoverinfo="skip",
                        ), row=r, col=c)
                if item["go"] is not None:
                    go_y = np.asarray(item["go"])
                    fig_grid.add_trace(go.Scatter(
                        x=t, y=go_y.tolist(),
                        mode="lines", name=f"Go (n={item['n_go']})",
                        line=dict(color="#2980b9", width=2),
                        showlegend=(cell == "Attend Sit"),
                    ), row=r, col=c)
                    if aggregate_mode and item.get("go_sem") is not None:
                        sem = np.asarray(item["go_sem"])
                        fig_grid.add_trace(go.Scatter(
                            x=t + t[::-1],
                            y=(go_y + sem).tolist() + (go_y - sem)[::-1].tolist(),
                            fill="toself", fillcolor="rgba(41,128,185,0.12)",
                            line=dict(width=0), showlegend=False, hoverinfo="skip",
                        ), row=r, col=c)
                fig_grid.add_vline(x=0, line_color="gray", line_dash="dot",
                                   row=r, col=c)
                fig_grid.add_vrect(x0=250, x1=700, fillcolor="rgba(255,200,0,0.08)",
                                   line_width=0, row=r, col=c)
            fig_grid.update_xaxes(title_text="Time (ms)", row=2, col=1)
            fig_grid.update_xaxes(title_text="Time (ms)", row=2, col=2)
            fig_grid.update_yaxes(title_text="Amplitude (µV)", row=1, col=1)
            fig_grid.update_yaxes(title_text="Amplitude (µV)", row=2, col=1)
            _global_y_vals = []
            for _item in erp_by_cell.values():
                for _key in ("go", "nogo"):
                    if _item.get(_key) is not None:
                        _global_y_vals.extend(np.asarray(_item[_key]).tolist())
            if _global_y_vals:
                _y_pad = (max(_global_y_vals) - min(_global_y_vals)) * 0.1
                _shared_y_range = [min(_global_y_vals) - _y_pad, max(_global_y_vals) + _y_pad]
            else:
                _shared_y_range = None

            if _shared_y_range:
                fig_grid.update_yaxes(range=_shared_y_range)
            fig_grid.update_layout(
                height=620, template="plotly_white",
                legend=dict(orientation="h", y=-0.08),
            )
            st.plotly_chart(fig_grid, use_container_width=True)
            st.caption("Shaded band = P300 window (250–700 ms). Shaded ribbons = ± SEM (aggregate mode only). Dashed line = stimulus onset.")
        else:
            st.info("No ERP data found. Run the full pipeline first (EEG preprocess → fusion → extract_features).")

        # ── Per-component ERP grids (N1, P2, P300) ──────────────
        try:
            with open(CONFIG_PATH, "r") as _f:
                _yaml_cfg = yaml.safe_load(_f)
            _erp_components = _yaml_cfg.get("erp", {}).get("components", {})
        except Exception:
            _erp_components = {}

        for _comp_name, _comp_cfg in _erp_components.items():
            if _comp_name == "p300":
                continue
            _comp_channels = _comp_cfg.get("channels", [])
            _comp_window = _comp_cfg.get("window", [0, 1])
            _win_ms = (_comp_window[0] * 1000, _comp_window[1] * 1000)

            with st.expander(
                f"{_comp_name.upper()} ERP — {', '.join(_comp_channels)} "
                f"({_win_ms[0]:.0f}–{_win_ms[1]:.0f} ms)",
                expanded=False,
            ):
                _comp_fig = make_subplots(
                    rows=2, cols=2, subplot_titles=cell_order,
                    shared_xaxes=True, shared_yaxes=True,
                    vertical_spacing=0.14, horizontal_spacing=0.10,
                )
                _has_data = False
                for _cc_cell, (_cc_r, _cc_c) in pos.items():
                    for cond in conditions:
                        if _condition_grid_label(cond) != _cc_cell:
                            continue
                        epo_path = os.path.join(
                            data_dir(selected_run),
                            f"sj{sj_num:02d}_{cond}_Features-epo.fif"
                            if not aggregate_mode else "")
                        if aggregate_mode:
                            _comp_go_stack, _comp_nogo_stack = [], []
                            for _sj in subjects:
                                _ep_path = os.path.join(
                                    data_dir(selected_run),
                                    f"sj{_sj:02d}_{cond}_Features-epo.fif")
                                if not os.path.exists(_ep_path):
                                    continue
                                try:
                                    _ep = _load_epochs_cached(_ep_path)
                                except Exception:
                                    continue
                                _ch_picks = [c for c in _comp_channels
                                             if c in _ep["ch_names"]]
                                if not _ch_picks:
                                    continue
                                _ch_idxs = [_ep["ch_names"].index(c) for c in _ch_picks]
                                _d = _ep["data"][:, _ch_idxs, :].mean(axis=1) * 1e6
                                _meta = _cached_metadata(_ep)
                                if _meta is not None and "trialType" in _meta.columns:
                                    _raw_tt = pd.to_numeric(_meta["trialType"], errors="coerce").to_numpy(float)
                                else:
                                    _raw_tt = _ep["events"][:, 2].astype(float)
                                _gi = np.flatnonzero(_raw_tt == 10.0)
                                _ni = np.flatnonzero(_raw_tt == 20.0)
                                if len(_gi):
                                    _comp_go_stack.append(_d[_gi].mean(0))
                                if len(_ni):
                                    _comp_nogo_stack.append(_d[_ni].mean(0))
                            _times_ms = None
                            if _comp_go_stack or _comp_nogo_stack:
                                _ref_path = None
                                for _sj in subjects:
                                    _ref_path = os.path.join(
                                        data_dir(selected_run),
                                        f"sj{_sj:02d}_{cond}_Features-epo.fif")
                                    if os.path.exists(_ref_path):
                                        break
                                if _ref_path and os.path.exists(_ref_path):
                                    _ref_ep = _load_epochs_cached(_ref_path)
                                    _times_ms = (_ref_ep["times"] * 1000).tolist()
                            if _times_ms and _comp_nogo_stack:
                                _arr = np.vstack(_comp_nogo_stack)
                                _mu = _arr.mean(0)
                                _comp_fig.add_trace(go.Scatter(
                                    x=_times_ms, y=_mu.tolist(),
                                    mode="lines", name="NoGo",
                                    line=dict(color="#e74c3c", width=2),
                                    showlegend=(_cc_cell == "Attend Sit"),
                                ), row=_cc_r, col=_cc_c)
                                if len(_comp_nogo_stack) > 1:
                                    _sem = _arr.std(0) / np.sqrt(len(_comp_nogo_stack))
                                    _comp_fig.add_trace(go.Scatter(
                                        x=_times_ms + _times_ms[::-1],
                                        y=(_mu + _sem).tolist() + (_mu - _sem)[::-1].tolist(),
                                        fill="toself", fillcolor="rgba(231,76,60,0.12)",
                                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                                    ), row=_cc_r, col=_cc_c)
                                _has_data = True
                            if _times_ms and _comp_go_stack:
                                _arr = np.vstack(_comp_go_stack)
                                _mu = _arr.mean(0)
                                _comp_fig.add_trace(go.Scatter(
                                    x=_times_ms, y=_mu.tolist(),
                                    mode="lines", name="Go",
                                    line=dict(color="#2980b9", width=2),
                                    showlegend=(_cc_cell == "Attend Sit"),
                                ), row=_cc_r, col=_cc_c)
                                if len(_comp_go_stack) > 1:
                                    _sem = _arr.std(0) / np.sqrt(len(_comp_go_stack))
                                    _comp_fig.add_trace(go.Scatter(
                                        x=_times_ms + _times_ms[::-1],
                                        y=(_mu + _sem).tolist() + (_mu - _sem)[::-1].tolist(),
                                        fill="toself", fillcolor="rgba(41,128,185,0.12)",
                                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                                    ), row=_cc_r, col=_cc_c)
                                _has_data = True
                        else:
                            if not os.path.exists(epo_path):
                                continue
                            try:
                                _ep = _load_epochs_cached(epo_path)
                            except Exception:
                                continue
                            _ch_picks = [c for c in _comp_channels
                                         if c in _ep["ch_names"]]
                            if not _ch_picks:
                                continue
                            _ch_idxs = [_ep["ch_names"].index(c) for c in _ch_picks]
                            _d = _ep["data"][:, _ch_idxs, :].mean(axis=1) * 1e6
                            _times_ms = (_ep["times"] * 1000).tolist()
                            _meta = _cached_metadata(_ep)
                            if _meta is not None and "trialType" in _meta.columns:
                                _raw_tt = pd.to_numeric(_meta["trialType"], errors="coerce").to_numpy(float)
                            else:
                                _raw_tt = _ep["events"][:, 2].astype(float)
                            _gi = np.flatnonzero(_raw_tt == 10.0)
                            _ni = np.flatnonzero(_raw_tt == 20.0)
                            if len(_ni):
                                _nogo_d = _d[_ni]
                                _mu = _nogo_d.mean(0)
                                _comp_fig.add_trace(go.Scatter(
                                    x=_times_ms, y=_mu.tolist(),
                                    mode="lines", name="NoGo",
                                    line=dict(color="#e74c3c", width=2),
                                    showlegend=(_cc_cell == "Attend Sit"),
                                ), row=_cc_r, col=_cc_c)
                                _has_data = True
                            if len(_gi):
                                _go_d = _d[_gi]
                                _mu = _go_d.mean(0)
                                _comp_fig.add_trace(go.Scatter(
                                    x=_times_ms, y=_mu.tolist(),
                                    mode="lines", name="Go",
                                    line=dict(color="#2980b9", width=2),
                                    showlegend=(_cc_cell == "Attend Sit"),
                                ), row=_cc_r, col=_cc_c)
                                _has_data = True
                        break

                    _comp_fig.add_vline(x=0, line_color="gray", line_dash="dot",
                                        row=_cc_r, col=_cc_c)
                    _comp_fig.add_vrect(
                        x0=_win_ms[0], x1=_win_ms[1],
                        fillcolor="rgba(255,200,0,0.12)", line_width=0,
                        row=_cc_r, col=_cc_c)

                if _has_data:
                    _comp_fig.update_xaxes(title_text="Time (ms)", row=2, col=1)
                    _comp_fig.update_xaxes(title_text="Time (ms)", row=2, col=2)
                    _comp_fig.update_yaxes(title_text="Amplitude (uV)", row=1, col=1)
                    _comp_fig.update_yaxes(title_text="Amplitude (uV)", row=2, col=1)
                    if _shared_y_range:
                        _comp_fig.update_yaxes(range=_shared_y_range)
                    _comp_fig.update_layout(
                        height=620, template="plotly_white",
                        legend=dict(orientation="h", y=-0.08),
                    )
                    st.plotly_chart(_comp_fig, use_container_width=True)
                else:
                    st.info("No data available for this component.")

        # ── MNE Topographic Maps (250–700 ms) ──────────────────
        st.markdown("---")
        st.subheader("Scalp Topography (250–700 ms mean)")
        st.caption(
            "Mean ERP amplitude averaged over 250–700 ms, plotted per channel. "
            "Go and NoGo shown side-by-side for each condition."
        )

        _topo_tmin, _topo_tmax = 0.25, 0.70

        _topo_cols = st.columns(2)
        _topo_idx = 0
        _topo_conditions_to_show = conditions

        for _t_cond in _topo_conditions_to_show:
            _topo_epochs_loaded = []
            _topo_trial_types = []

            if aggregate_mode:
                for _sj in subjects:
                    _t_path = os.path.join(
                        data_dir(selected_run),
                        f"sj{_sj:02d}_{_t_cond}_EEG_Prepro1-epo.fif")
                    if not os.path.exists(_t_path):
                        continue
                    try:
                        _t_epo = mne.read_epochs(_t_path, preload=True, verbose=False)
                        _t_epo.pick_types(eeg=True)
                        _topo_epochs_loaded.append(_t_epo)
                    except Exception:
                        continue
            else:
                _t_path = os.path.join(
                    data_dir(selected_run),
                    f"sj{sj_num:02d}_{_t_cond}_EEG_Prepro1-epo.fif")
                if os.path.exists(_t_path):
                    try:
                        _t_epo = mne.read_epochs(_t_path, preload=True, verbose=False)
                        _t_epo.pick_types(eeg=True)
                        _topo_epochs_loaded.append(_t_epo)
                    except Exception:
                        pass

            if not _topo_epochs_loaded:
                continue

            # Subjects preprocessed under different montage code can have
            # slightly different channel sets (e.g. 31 vs 32 channels when Fz
            # is missing). Intersect to the common channels in a stable order
            # so np.vstack never sees a shape mismatch.
            _common_chs = set(_topo_epochs_loaded[0].ch_names)
            for _t_epo in _topo_epochs_loaded[1:]:
                _common_chs &= set(_t_epo.ch_names)
            _common_order = [
                c for c in _topo_epochs_loaded[0].ch_names if c in _common_chs
            ]
            if not _common_order:
                continue
            _dropped = sorted(
                {c for _t_epo in _topo_epochs_loaded for c in _t_epo.ch_names}
                - _common_chs
            )
            if _dropped:
                st.caption(
                    f"Topomap channel intersection across subjects dropped: "
                    f"{', '.join(_dropped)} (subjects preprocessed with "
                    f"different montages)."
                )
            _topo_epochs_loaded = [
                _t_epo.copy().pick(_common_order)
                for _t_epo in _topo_epochs_loaded
            ]

            _all_go_topo = []
            _all_nogo_topo = []
            _topo_info = _topo_epochs_loaded[0].info

            for _t_epo in _topo_epochs_loaded:
                _t_data = _t_epo.get_data() * 1e6
                _t_times = _t_epo.times
                _t_mask = (_t_times >= _topo_tmin) & (_t_times <= _topo_tmax)

                if _t_epo.metadata is not None and "trialType" in _t_epo.metadata.columns:
                    _tt = pd.to_numeric(
                        _t_epo.metadata["trialType"], errors="coerce"
                    ).to_numpy(dtype=float)
                else:
                    _tt = _t_epo.events[:, 2].astype(float)

                _t_go = np.flatnonzero(_tt == 10.0)
                _t_nogo = np.flatnonzero(_tt == 20.0)

                if len(_t_go):
                    _all_go_topo.append(
                        _t_data[_t_go][:, :, _t_mask].mean(axis=(0, 2)))
                if len(_t_nogo):
                    _all_nogo_topo.append(
                        _t_data[_t_nogo][:, :, _t_mask].mean(axis=(0, 2)))

            _cond_label = _condition_grid_label(_t_cond)
            _fig_topo, _ax_topo = plt.subplots(1, 2, figsize=(7, 3.5))

            if _all_go_topo:
                _go_avg = np.mean(np.vstack([t[np.newaxis, :] for t in _all_go_topo]), axis=0)
                mne.viz.plot_topomap(
                    _go_avg, _topo_info, axes=_ax_topo[0], show=False,
                    cmap="RdBu_r", vlim=(-8, 8))
                _n_go = sum(1 for _ in _all_go_topo)
                _ax_topo[0].set_title(f"Go", fontsize=12)
            else:
                _ax_topo[0].text(0.5, 0.5, "No Go trials",
                                 ha="center", va="center")
                _ax_topo[0].set_axis_off()

            if _all_nogo_topo:
                _nogo_avg = np.mean(np.vstack([t[np.newaxis, :] for t in _all_nogo_topo]), axis=0)
                mne.viz.plot_topomap(
                    _nogo_avg, _topo_info, axes=_ax_topo[1], show=False,
                    cmap="RdBu_r", vlim=(-8, 8))
                _ax_topo[1].set_title(f"NoGo", fontsize=12)
            else:
                _ax_topo[1].text(0.5, 0.5, "No NoGo trials",
                                 ha="center", va="center")
                _ax_topo[1].set_axis_off()

            _fig_topo.suptitle(_cond_label, fontsize=14)
            _fig_topo.tight_layout()
            with _topo_cols[_topo_idx % 2]:
                st.pyplot(_fig_topo, clear_figure=True)
            _topo_idx += 1
            plt.close(_fig_topo)

        if _topo_idx == 0:
            st.info("No EEG epoch files found for topomap rendering.")

        st.markdown("---")
        st.subheader("Behavior by Movement (Sit vs Walk)")
        if aggregate_mode:
            beh_rows = _aggregate_behavior(
                selected_run, tuple(subjects), tuple(conditions))
        else:
            beh_rows = []
            for _c in conditions:
                _mv, _att = _parse_condition_parts(_c)
                if _att != "attend":
                    continue
                _feat = load_csv(
                    os.path.join(data_dir(selected_run), f"sj{sj_num:02d}_{_c}_features.csv")
                )
                _row = _build_movement_behavior_rows(_feat, _c)
                if _row is not None:
                    beh_rows.append(_row)
        if beh_rows:
            _beh = pd.DataFrame(beh_rows)
            fig_beh = _plot_sit_walk_behavior_matplotlib(_beh)
            st.pyplot(fig_beh, clear_figure=True, width="stretch")
            if aggregate_mode:
                st.caption(
                    f"Attend-only conditions, pooled across {len(subjects)} subjects. "
                    "Error bars = SEM across subjects."
                )
            else:
                st.caption("Attend-only conditions: Hit/Commission Error rates over total trials.")
        else:
            st.info("No Attend-condition movement behavior data found (`*_features.csv`).")




# ════════════════════════════════════════════════════════════
# TAB 4 — EYE TRACKING
# ════════════════════════════════════════════════════════════

with tab_et:
    if not selected_run:
        st.info("Select a run from the sidebar.")
    elif not subjects:
        st.info("No subjects with data in this run yet.")
    else:
        _et_view = st.radio(
            "View mode",
            ["All subjects (average)", "Single subject"],
            index=0,
            horizontal=True,
            key="et_view_mode",
        )
        aggregate_mode = _et_view.startswith("All")
        sj_num = None
        if not aggregate_mode:
            sj_num = st.selectbox(
                "Subject", subjects,
                format_func=lambda x: f"sj{x:02d}",
                key="et_subject",
            )
        if aggregate_mode:
            st.header(f"Eye Tracking — All subjects (n={len(subjects)})")
            st.caption(
                "Aggregate mode shows cross-subject means; per-subject "
                "scanpaths and the optical-axis / gyro / pupil triptych "
                "are only available in Single subject view."
            )

            et_agg = _aggregate_et_metrics(
                selected_run, tuple(subjects), tuple(conditions))

            if et_agg is None or len(et_agg) == 0:
                st.info(
                    "No `gaze_positions.csv` files found for any subject "
                    "under the configured data root."
                )
            else:
                st.subheader("Total gaze distance (mean +/- SD across subjects)")
                fig_total = px.bar(
                    et_agg, x="condition", y="total_distance_mean",
                    error_y="total_distance_sd",
                    color="condition",
                    labels={
                        "condition": "Condition",
                        "total_distance_mean": "Total Distance (px)",
                    },
                )
                fig_total.update_layout(
                    showlegend=False, height=380, template="plotly_white",
                )
                st.plotly_chart(fig_total, width="stretch")

                st.subheader("Mean gaze rate (mean +/- SD across subjects)")
                fig_rate = px.bar(
                    et_agg, x="condition", y="mean_rate_mean",
                    error_y="mean_rate_sd",
                    color="condition",
                    labels={
                        "condition": "Condition",
                        "mean_rate_mean": "Mean Rate (px/s)",
                    },
                )
                fig_rate.update_layout(
                    showlegend=False, height=380, template="plotly_white",
                )
                st.plotly_chart(fig_rate, width="stretch")

                with st.expander("Per-subject contributions"):
                    st.dataframe(
                        et_agg.assign(
                            **{
                                "total_distance_mean": et_agg["total_distance_mean"].round(0),
                                "total_distance_sd": et_agg["total_distance_sd"].round(0),
                                "mean_rate_mean": et_agg["mean_rate_mean"].round(0),
                                "mean_rate_sd": et_agg["mean_rate_sd"].round(0),
                            }
                        ),
                        hide_index=True, width="stretch",
                    )
        else:
            st.header(f"Eye Tracking — sj{sj_num:02d}")

            et_map = _et_folder_map()
            data_root = _project_data_root()
            n_conds = len(conditions)

            # Build paths & load data for every condition up front
            _et_eye_dirs = {}
            _et_gaze = {}
            _et_fix = {}
            _et_euc = {}
            for _c in conditions:
                _dir = os.path.join(data_root, f"sj{sj_num:02d}", "eye",
                                    et_map.get(_c, ""))
                _et_eye_dirs[_c] = _dir
                _et_gaze[_c] = et_viz.load_gaze_for_viz(
                    os.path.join(_dir, "gaze_positions.csv"))
                _et_fix[_c] = et_viz.load_fixations(
                    os.path.join(_dir, "fixations.csv"))
                _et_euc[_c] = et_viz.compute_euclidean(
                    os.path.join(_dir, "gaze_positions.csv"))

            # ── 1. Euclidean Distance ─────────────────────────────
            euc_valid = {c: d for c, d in _et_euc.items() if d is not None}
            if euc_valid:
                st.subheader("Euclidean Distance")

                # Summary metrics
                _m_cols = st.columns(n_conds)
                for i, (cond, d) in enumerate(euc_valid.items()):
                    _m_cols[i].metric(
                        cond,
                        f"{d['total_distance']:,.0f} px",
                        help=f"Total gaze path length over {d['duration_s']:.0f}s",
                    )
                _m_cols2 = st.columns(n_conds)
                for i, (cond, d) in enumerate(euc_valid.items()):
                    _m_cols2[i].metric(
                        f"{cond} — rate",
                        f"{d['mean_rate']:,.0f} px/s",
                        help="Mean displacement per second",
                    )

                # Cumulative distance overlay (the key comparison plot)
                st.plotly_chart(
                    et_viz.fig_cumulative_distance(euc_valid),
                    width="stretch",
                )
                st.caption(
                    "Slope = rate of eye movement. "
                    "Steeper = more gaze displacement."
                )

                # Raw distance + rolling average side by side
                _euc_cols = st.columns(n_conds)
                for i, cond in enumerate(conditions):
                    if cond not in euc_valid:
                        continue
                    with _euc_cols[i]:
                        st.markdown(f"**{cond}**")
                        st.plotly_chart(
                            et_viz.fig_raw_distance(euc_valid[cond]),
                            width="stretch",
                        )
                        st.plotly_chart(
                            et_viz.fig_rolling_distance(euc_valid[cond]),
                            width="stretch",
                        )

            # ── 1b. Optical axis + IMU + pupil (aligned) ─────────
            st.markdown("---")
            st.subheader("Eye vs head vs pupil (same time axis)")
            st.caption(
                "Compare **optical-axis rotation speed** (eyes) with **gyro magnitude** "
                "(head) and **pupil size** (arousal / effort). "
                "High eye speed with low gyro often suggests scanning while the head is still."
            )
            _tri_cond = conditions[0] if len(conditions) == 1 else st.selectbox(
                "Condition (3-panel physiology)",
                conditions,
                key="triptych_cond",
            )
            _tri_dir = _et_eye_dirs.get(_tri_cond, "")
            _tri_series = (
                et_viz.build_axis_gyro_pupil_series(_tri_dir)
                if _tri_dir and os.path.isdir(_tri_dir)
                else None
            )
            if _tri_series is not None:
                _et_prepro_path = os.path.join(
                    data_dir(selected_run),
                    f"sj{sj_num:02d}_{_tri_cond}_ET_Prepro1.csv",
                )
                _et_prepro = load_csv(_et_prepro_path)
                _trigger_rel = None
                _x_for_plot = _tri_series["t_s"]
                _x_label = "Time (s) from first 3d eye sample"

                _vision_rel = None
                _vr_path = os.path.join(
                    vision_dir(selected_run, sj_num, _tri_cond),
                    f"sj{sj_num:02d}_{_tri_cond}_vision_results.csv",
                )
                _vision_results = load_csv(_vr_path)

                if (
                    _et_prepro is not None
                    and "trigger_time" in _et_prepro.columns
                    and _et_prepro["trigger_time"].notna().any()
                ):
                    _trigger_abs = (
                        _et_prepro["trigger_time"].dropna().astype(float).to_numpy()
                    )
                    _t0_trig = float(_trigger_abs.min())
                    _x_for_plot = _tri_series["t_abs_s"] - _t0_trig
                    _x_label = "Behavior-aligned time (s from first trial trigger)"
                    _trigger_rel = _trigger_abs - _t0_trig

                    # Vision timestamps are stored relative to gaze start.
                    # Convert to absolute with gaze start, then to trigger-relative.
                    if (
                        _vision_results is not None
                        and "timestamp_s" in _vision_results.columns
                        and _vision_results["timestamp_s"].notna().any()
                    ):
                        _gaze_t0 = load_first_timestamp_s(
                            os.path.join(_tri_dir, "gaze_positions.csv")
                        )
                        if _gaze_t0 is not None:
                            _vision_abs = _gaze_t0 + _vision_results["timestamp_s"].astype(float).to_numpy()
                            _vision_rel = _vision_abs - _t0_trig

                st.plotly_chart(
                    et_viz.fig_axis_gyro_pupil_triptych(
                        _tri_series,
                        title=f"sj{sj_num:02d} · {_tri_cond}",
                        x_s=_x_for_plot,
                        x_label=_x_label,
                        trigger_s=_trigger_rel,
                        vision_s=_vision_rel,
                    ),
                    width="stretch",
                )
                if _trigger_rel is not None:
                    st.caption(
                        "Dotted vertical lines = behavioral trial triggers. "
                        "Yellow dots on panel 1 = vision fixation timestamps."
                    )
                else:
                    st.caption(
                        "Behavior trigger table not found for this run/condition, "
                        "so this view uses raw session time."
                    )
            else:
                st.info(
                    f"No `3d_eye_states.csv` in `{_tri_dir or '(unknown)'}` — "
                    "needed for optical axes and pupil."
                )

            with st.expander("How this plot is computed (blinks, pipeline, …)", expanded=False):
                st.markdown(
                    """
**Panel 1 — Eye movement intensity (optical axis)**  
- Source: `3d_eye_states.csv` (Pupil Labs export).  
- Left and right optical-axis vectors are **row-normalized** to unit vectors **û**.  
- At each time sample we estimate **|dû/dt|** using `numpy.gradient` along the
  session clock (uneven spacing is allowed). That norm is the instantaneous
  **angular speed of the gaze direction** in space (deg/s).  
- The trace is the **mean of left and right** angular speeds.  
- This is **not** the same as screen-plane Euclidean distance from
  `gaze_positions.csv` (the plots above): axis motion is 3-D gaze direction;
  screen metrics mix projection and head movement.

**Panel 2 — Head movement intensity**  
- Source: `imu.csv` gyro columns (`gyro x/y/z [deg/s]`).  
- Plotted value: **√(gx² + gy² + gz²)** in deg/s.  
- IMU timestamps rarely match eye samples; we **linearly interpolate** gyro
  magnitude onto the eye time grid (same **t = 0** as the first row of
  `3d_eye_states.csv`).

**Panel 3 — Pupil (arousal / load)**  
- **Average of left and right** `pupil diameter [mm]` from the same file as
  panel 1. **Not** blink-rejected here: during blinks, diameters often go to
  junk values; interpret dips with the blink bands or with trial-level blink
  flags from preprocessing.

**Behavior + vision alignment (for long recordings)**  
- If `ET_Prepro1.csv` is available, all three traces are re-plotted on a
  **behavior-anchored axis**: seconds from the first trial trigger.  
- Dotted vertical lines mark every behavioral trigger (`trigger_time`).  
- Vision fixation timestamps (`vision_results.csv`) are converted from their
  gaze-relative clock to that same trigger-aligned axis and shown as yellow dots.

**Gray vertical bands — blink windows**  
- The dashboard infers blink-like periods from eyelid aperture in
  `3d_eye_states.csv` (when either `eyelid aperture left/right [mm]` drops
  below ~1 mm).  
- **Separate from this figure:** the main pipeline (`et_preprocess` /
  `et_timeseries`) computes epoch-level blink flags independently; those are the
  `has_blink` values shown in **Session Details**.

**Performance**  
- Long sessions are **decimated** (~25k points max) for responsiveness; totals
  and shapes are unchanged in the raw files.
                    """
                )

            # ── 1c. Sit vs walk physiology + behavior summary ───────────
            st.markdown("---")
            st.subheader("Sit vs Walk Summary")

            # Physiology summary from 3d_eye_states + imu.
            phys_rows = []
            for _c in conditions:
                _s = et_viz.build_axis_gyro_pupil_series(_et_eye_dirs.get(_c, ""))
                if _s is None:
                    continue
                _mv, _att = _parse_condition_parts(_c)
                phys_rows.append({
                    "Condition": _c,
                    "Movement": _mv.title(),
                    "Attention": _att.title(),
                    "Pupil (mm)": float(np.nanmedian(_s["pupil_mm"])),
                    "Eye speed (deg/s)": float(np.nanmedian(_s["omega_eye_deg_s"])),
                    "Gyro (deg/s)": float(np.nanmedian(_s["gyro_mag_deg_s"])),
                })
            if phys_rows:
                st.dataframe(pd.DataFrame(phys_rows), width="stretch", hide_index=True)
            else:
                st.info("No `3d_eye_states.csv` available for current conditions.")

            # Behavior summary bars (no error bars): pCorrect (Go HIT rate), pError (NoGo CE rate).
            beh_rows = []
            for _c in conditions:
                _feat = load_csv(
                    os.path.join(data_dir(selected_run), f"sj{sj_num:02d}_{_c}_features.csv")
                )
                _row = _build_movement_behavior_rows(_feat, _c)
                if _row is not None:
                    beh_rows.append(_row)
            if beh_rows:
                _beh = pd.DataFrame(beh_rows)
                _agg = (
                    _beh.groupby("movement", as_index=False)[["pCorrect", "pError"]]
                    .mean()
                    .rename(columns={"movement": "Movement"})
                )
                c1, c2 = st.columns(2)
                with c1:
                    fig_hit = px.bar(
                        _agg,
                        x="Movement",
                        y="pCorrect",
                        title="Hits (pCorrect)",
                        color_discrete_sequence=["#1f77b4"],
                        range_y=[0, 1],
                    )
                    fig_hit.update_layout(showlegend=False, height=330)
                    st.plotly_chart(fig_hit, width="stretch")
                with c2:
                    fig_ce = px.bar(
                        _agg,
                        x="Movement",
                        y="pError",
                        title="Commission Errors (pError)",
                        color_discrete_sequence=["#1f77b4"],
                        range_y=[0, 1],
                    )
                    fig_ce.update_layout(showlegend=False, height=330)
                    st.plotly_chart(fig_ce, width="stretch")

            # ── 2. Spatial Analysis (side by side) ────────────────
            if any(_et_gaze[c] is not None for c in conditions):
                st.markdown("---")
                st.subheader("Gaze Heatmap")
                _hm_cols = st.columns(n_conds)
                for i, cond in enumerate(conditions):
                    with _hm_cols[i]:
                        st.markdown(f"**{cond}**")
                        if _et_gaze[cond] is not None:
                            st.plotly_chart(
                                et_viz.fig_heatmap(_et_gaze[cond]),
                                width="stretch",
                            )

            if any(_et_fix[c] is not None for c in conditions):
                st.markdown("---")
                st.subheader("Scanpath")
                _sp_cols = st.columns(n_conds)
                for i, cond in enumerate(conditions):
                    with _sp_cols[i]:
                        st.markdown(f"**{cond}**")
                        if _et_fix[cond] is not None:
                            st.plotly_chart(
                                et_viz.fig_scanpath(_et_fix[cond]),
                                width="stretch",
                            )

                st.subheader("Fixation Map")
                _fm_cols = st.columns(n_conds)
                for i, cond in enumerate(conditions):
                    with _fm_cols[i]:
                        st.markdown(f"**{cond}**")
                        if _et_fix[cond] is not None:
                            st.plotly_chart(
                                et_viz.fig_fixation_map(_et_fix[cond]),
                                width="stretch",
                            )

            # ── 3. Space-Time Cube (side by side, collapsed) ──────
            if any(_et_gaze[c] is not None for c in conditions):
                st.markdown("---")
                with st.expander("Space-Time Cube (3D)", expanded=False):
                    _st_cols = st.columns(n_conds)
                    for i, cond in enumerate(conditions):
                        with _st_cols[i]:
                            st.markdown(f"**{cond}**")
                            if _et_gaze[cond] is not None:
                                st.plotly_chart(
                                    et_viz.fig_spacetime_cube(_et_gaze[cond]),
                                    width="stretch",
                                )

            # ── 4. Per-condition details ──────────────────────────
            st.markdown("---")
            st.subheader("Session Details")
            for _et_cond in conditions:
                with st.expander(_et_cond, expanded=False):
                    gaze_img = os.path.join(
                        plots_dir(selected_run),
                        f"sj{sj_num:02d}_L1_gaze_xy_pupil_{_et_cond}.png",
                    )
                    if os.path.exists(gaze_img):
                        st.image(gaze_img, width="stretch",
                                 caption="Full-Session Gaze Trace")

                    traj_img = os.path.join(
                        plots_dir(selected_run),
                        f"sj{sj_num:02d}_L_gaze_trajectories_{_et_cond}.png",
                    )
                    if os.path.exists(traj_img):
                        st.image(traj_img, width="stretch",
                                 caption="Gaze Trajectories by Outcome")

                    et_info_path = os.path.join(
                        data_dir(selected_run),
                        f"sj{sj_num:02d}_{_et_cond}_et_tensor_info.json",
                    )
                    if os.path.exists(et_info_path):
                        with open(et_info_path) as f:
                            et_info = json.load(f)
                        _c1, _c2, _c3 = st.columns(3)
                        shape = et_info.get("shape", [])
                        _c1.metric("Shape", f"{shape}")
                        _c2.metric("Channels",
                                   ", ".join(et_info.get("channel_names", [])))
                        n_failed = len(et_info.get("failed_trials", []))
                        _c3.metric("Failed Trials", n_failed)
                        st.caption(
                            f"Epochs with blinks: "
                            f"{sum(et_info.get('has_blink', []))}"
                        )

                    et_prepro = load_csv(os.path.join(
                        data_dir(selected_run),
                        f"sj{sj_num:02d}_{_et_cond}_ET_Prepro1.csv",
                    ))
                    if et_prepro is not None:
                        _c1, _c2, _c3 = st.columns(3)
                        _c1.metric("ET Trials", len(et_prepro))
                        _c2.metric("Mean Gaze Samples/Trial",
                                   f"{et_prepro['gaze_n_samples'].mean():.0f}")
                        _c3.metric("Mean Gaze X",
                                   f"{et_prepro['gaze_mean_x_px'].mean():.0f} px")
                        with st.expander("ET Prepro Table"):
                            st.dataframe(et_prepro, width="stretch",
                                         height=300)


# ════════════════════════════════════════════════════════════
# TAB 4 — EEGNET (No-Go ML Pipeline)
# ════════════════════════════════════════════════════════════

with tab_nogo:
    if not selected_run:
        st.info("Select a run from the sidebar.")
    elif not subjects:
        st.info("No subjects with data in this run yet.")
    else:
        sj_num = st.selectbox(
            "Subject", subjects,
            format_func=lambda x: f"sj{x:02d}",
            key="eegnet_subject",
        )
        st.header(f"EEGNet — sj{sj_num:02d}")
        st.caption(
            "No-go trial classification: correct rejection vs false alarm. "
            "Phase 6 = EEG-only (EEGNet), Phase 7 = EEG + CLIP gaze fusion."
        )

        # ── Run button ──
        col_run1, col_run2, col_run3 = st.columns(3)
        run_p6 = col_run1.button("Run Phase 6 (EEG-only)", type="primary")
        run_p7 = col_run2.button("Run Phase 7 (Fusion)")
        run_p67 = col_run3.button("Run Both (6 + 7)")

        if any([run_p6, run_p7, run_p67]):
            phases_to_run = []
            if run_p6:
                phases_to_run = ["6"]
            elif run_p7:
                phases_to_run = ["7"]
            else:
                phases_to_run = ["6", "7"]

            cmd = [
                VENV_PYTHON,
                os.path.join(PROJECT_ROOT, "src", "train.py"),
                "--phase", *phases_to_run,
                "--run", selected_run,
            ]
            with st.spinner(f"Running Phase {'+'.join(phases_to_run)}..."):
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True,
                        encoding="utf-8", errors="replace",
                        timeout=600, cwd=PROJECT_ROOT,
                    )
                    if result.returncode == 0:
                        st.success("Complete!")
                    else:
                        st.error(f"Failed (exit code {result.returncode})")
                    with st.expander("Output", expanded=result.returncode != 0):
                        st.code(result.stdout + result.stderr, language="text")
                    st.cache_data.clear()
                except subprocess.TimeoutExpired:
                    st.error("Timed out (10 min limit)")
                except Exception as e:
                    st.error(f"Error: {e}")

        # ── Load results ──
        nogo_results_path = os.path.join(
            run_dir(selected_run), "nogo_results.json")
        nogo_res = None
        if os.path.exists(nogo_results_path):
            with open(nogo_results_path) as _f:
                nogo_res = json.load(_f)

        ml_results_path = os.path.join(
            run_dir(selected_run), "ml_results.json")
        ml_res = None
        if os.path.exists(ml_results_path):
            with open(ml_results_path) as _f:
                ml_res = json.load(_f)

        if nogo_res is None and ml_res is None:
            st.info("No EEGNet results yet. "
                    "Click a Run button above to train.")
        else:
            p6 = nogo_res.get("phase6", {}) if nogo_res else {}
            p7 = nogo_res.get("phase7", {}) if nogo_res else {}

            # ── Phase 6: EEG-Only ──
            st.markdown("---")
            st.subheader("Model A — EEG-Only (EEGNet)")

            if p6 and "summary" in p6:
                s6 = p6["summary"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Balanced Accuracy",
                          f"{s6.get('balanced_accuracy_mean', 0):.3f} "
                          f"± {s6.get('balanced_accuracy_std', 0):.3f}")
                c2.metric("AUC-ROC",
                          f"{s6.get('auc_roc_mean', 0):.3f} "
                          f"± {s6.get('auc_roc_std', 0):.3f}")
                c3.metric("F1",
                          f"{s6.get('f1_mean', 0):.3f} "
                          f"± {s6.get('f1_std', 0):.3f}")
                c4.metric("No-Go Trials",
                          f"{p6.get('n_cr', '?')} CR / "
                          f"{p6.get('n_fa', '?')} FA")
                c5, c6, _ , _ = st.columns(4)
                c5.metric("Precision",
                          f"{s6.get('precision_mean', 0):.3f} "
                          f"± {s6.get('precision_std', 0):.3f}")
                c6.metric("Recall",
                          f"{s6.get('recall_mean', 0):.3f} "
                          f"± {s6.get('recall_std', 0):.3f}")

                folds6 = p6.get("fold_results", [])
                if folds6:
                    auc_vals = [f["auc_roc"] for f in folds6]
                    import plotly.graph_objects as _pgo
                    fig6 = _pgo.Figure()
                    fig6.add_trace(_pgo.Bar(
                        x=[f"Fold {f['fold']+1}" for f in folds6],
                        y=auc_vals,
                        marker_color="#3498db",
                        text=[f"{v:.3f}" for v in auc_vals],
                        textposition="auto",
                    ))
                    fig6.add_hline(y=0.5, line_dash="dash",
                                   line_color="red",
                                   annotation_text="Chance")
                    fig6.update_layout(
                        title="Phase 6 — Per-Fold AUC-ROC",
                        yaxis_title="AUC-ROC", yaxis_range=[0, 1],
                        height=350, template="plotly_white",
                    )
                    st.plotly_chart(fig6, width="stretch")

                # Confusion matrix
                cm = s6.get("confusion_matrix_sum")
                if cm:
                    cm_arr = np.array(cm)
                    fig_cm = px.imshow(
                        cm_arr,
                        labels=dict(x="Predicted", y="True", color="Count"),
                        x=["False Alarm", "Correct Rejection"],
                        y=["False Alarm", "Correct Rejection"],
                        color_continuous_scale="Blues", text_auto=True,
                    )
                    fig_cm.update_layout(title="Confusion Matrix (summed)",
                                         height=350)
                    st.plotly_chart(fig_cm, width="stretch")
            else:
                st.info("Phase 6 not yet run.")

            # ── Phase 7: Fusion ──
            st.markdown("---")
            st.subheader("Model B — EEG + CLIP Gaze Fusion")

            if p7 and "summary" in p7:
                s7 = p7["summary"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Balanced Accuracy",
                          f"{s7.get('balanced_accuracy_mean', 0):.3f} "
                          f"± {s7.get('balanced_accuracy_std', 0):.3f}")
                c2.metric("AUC-ROC",
                          f"{s7.get('auc_roc_mean', 0):.3f} "
                          f"± {s7.get('auc_roc_std', 0):.3f}")
                c3.metric("F1",
                          f"{s7.get('f1_mean', 0):.3f} "
                          f"± {s7.get('f1_std', 0):.3f}")
                c4, c5, _ = st.columns(3)
                c4.metric("Precision",
                          f"{s7.get('precision_mean', 0):.3f} "
                          f"± {s7.get('precision_std', 0):.3f}")
                c5.metric("Recall",
                          f"{s7.get('recall_mean', 0):.3f} "
                          f"± {s7.get('recall_std', 0):.3f}")

                # Side-by-side bar chart
                folds6 = p6.get("fold_results", []) if p6 else []
                folds7 = p7.get("fold_results", [])
                if folds6 and folds7:
                    n_f = min(len(folds6), len(folds7))
                    comp_rows = []
                    for i in range(n_f):
                        comp_rows.append({
                            "Fold": f"Fold {i+1}",
                            "AUC-ROC": folds6[i]["auc_roc"],
                            "Model": "A: EEG Only",
                        })
                        comp_rows.append({
                            "Fold": f"Fold {i+1}",
                            "AUC-ROC": folds7[i]["auc_roc"],
                            "Model": "B: EEG + Gaze",
                        })
                    fig_comp = px.bar(
                        pd.DataFrame(comp_rows),
                        x="Fold", y="AUC-ROC", color="Model",
                        barmode="group",
                        color_discrete_map={
                            "A: EEG Only": "#3498db",
                            "B: EEG + Gaze": "#e67e22",
                        },
                    )
                    fig_comp.add_hline(y=0.5, line_dash="dash",
                                       line_color="red")
                    fig_comp.update_layout(
                        title="Model A vs B — Per-Fold AUC-ROC",
                        yaxis_range=[0, 1], height=400,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_comp, width="stretch")

                # Wilcoxon comparison
                comp = p7.get("comparison", {})
                if comp and "p_value" in comp:
                    st.subheader("Statistical Comparison")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Wilcoxon p",
                              f"{comp['p_value']:.4f}")
                    c2.metric("Mean AUC Diff",
                              f"{comp.get('mean_difference', 0):.3f}")
                    c3.metric("Significant (p<.05)",
                              "Yes" if comp.get("significant") else "No")

                    if comp.get("significant"):
                        st.success(
                            "Gaze context significantly improves "
                            "inhibitory control prediction!")
                    else:
                        st.info(
                            "No significant difference. "
                            "More subjects may increase power.")
            elif p7 and p7.get("skipped"):
                st.warning(
                    f"Phase 7 skipped: {p7.get('reason', 'unknown')}. "
                    f"Run the CLIP vision pipeline first.")
            else:
                st.info("Phase 7 not yet run.")

            # ── ERP: CR vs FA ──
            st.markdown("---")
            st.subheader("ERP: Correct Rejection vs False Alarm")

            for cond in conditions:
                try:
                    epo_path = os.path.join(
                        data_dir(selected_run),
                        f"sj{sj_num:02d}_{cond}_Features-epo.fif")
                    if not os.path.exists(epo_path):
                        continue
                    ep = _load_epochs_cached(epo_path)
                    meta = _cached_metadata(ep)
                    if meta is None:
                        continue
                    if "outcome" not in meta.columns:
                        continue

                    cr_idx = meta[
                        meta["outcome"].str.upper() == "CORRECT_REJECTION"
                    ].index.tolist()
                    fa_idx = meta[
                        meta["outcome"].str.upper() == "COMMISSION_ERROR"
                    ].index.tolist()

                    if not cr_idx and not fa_idx:
                        continue

                    pz_i = None
                    for ch in ["Pz", "CPz", "Cz"]:
                        if ch in ep["ch_names"]:
                            pz_i = ep["ch_names"].index(ch)
                            break
                    if pz_i is None:
                        pz_i = 0

                    edata = ep["data"]
                    times = ep["times"] * 1000
                    import plotly.graph_objects as _pgo2

                    fig_erp = _pgo2.Figure()
                    if cr_idx:
                        fig_erp.add_trace(_pgo2.Scatter(
                            x=times,
                            y=edata[cr_idx, pz_i, :].mean(0) * 1e6,
                            name=f"CR (n={len(cr_idx)})",
                            line=dict(color="#3498db"),
                        ))
                    if fa_idx:
                        fig_erp.add_trace(_pgo2.Scatter(
                            x=times,
                            y=edata[fa_idx, pz_i, :].mean(0) * 1e6,
                            name=f"FA (n={len(fa_idx)})",
                            line=dict(color="#e74c3c"),
                        ))
                    fig_erp.add_vrect(x0=180, x1=350, fillcolor="yellow",
                                      opacity=0.1, line_width=0,
                                      annotation_text="N2/P3")
                    fig_erp.add_vline(x=0, line_dash="dash",
                                      line_color="gray")
                    fig_erp.update_layout(
                        title=f"{cond} — {ep['ch_names'][pz_i]}",
                        xaxis_title="Time (ms)",
                        yaxis_title="Amplitude (µV)",
                        height=350, template="plotly_white",
                    )
                    st.plotly_chart(fig_erp, width="stretch")
                except Exception:
                    pass

            # ── UMAP Embeddings ──
            models_dir = os.path.join(run_dir(selected_run), "models")
            emb_path = os.path.join(models_dir, "nogo_eeg_embeddings.npy")
            lab_path = os.path.join(models_dir,
                                     "nogo_eeg_embedding_labels.npy")

            if os.path.exists(emb_path) and os.path.exists(lab_path):
                st.markdown("---")
                st.subheader("Embedding Explorer (UMAP)")

                emb = np.load(emb_path)
                emb_labels = np.load(lab_path)

                try:
                    from umap import UMAP as _UMAP

                    @st.cache_data
                    def _compute_umap(_emb_bytes, n_pts):
                        _emb = np.frombuffer(_emb_bytes,
                                             dtype=np.float32).reshape(n_pts, -1)
                        return _UMAP(n_neighbors=15, min_dist=0.1,
                                     n_components=2,
                                     random_state=42).fit_transform(_emb)

                    coords = _compute_umap(emb.tobytes(), len(emb))

                    umap_df = pd.DataFrame({
                        "UMAP1": coords[:, 0],
                        "UMAP2": coords[:, 1],
                        "Label": ["CR" if l == 1 else "FA"
                                   for l in emb_labels],
                    })
                    fig_umap = px.scatter(
                        umap_df, x="UMAP1", y="UMAP2", color="Label",
                        color_discrete_map={"CR": "#3498db", "FA": "#e74c3c"},
                        opacity=0.7,
                    )
                    fig_umap.update_layout(
                        height=500, template="plotly_white",
                        title="EEG Embeddings (No-Go Trials)",
                    )
                    fig_umap.update_traces(marker=dict(size=6))
                    st.plotly_chart(fig_umap, width="stretch")
                except ImportError:
                    st.warning("Install umap-learn for embedding plots: "
                               "pip install umap-learn")

            # ── Filter Visualization ──
            filter_dir = os.path.join(run_dir(selected_run), "plots", "filters")
            filter_images = {
                "Spatial Filters (Topomap)": "spatial_filters_topomap.png",
                "Temporal Filters": "temporal_filters.png",
                "Input Saliency": "saliency_topomap_class1.png",
            }
            has_filters = any(
                os.path.exists(os.path.join(filter_dir, fn))
                for fn in filter_images.values()
            )
            if has_filters:
                st.markdown("---")
                st.subheader("Model Interpretability")
                for title, fname in filter_images.items():
                    fpath = os.path.join(filter_dir, fname)
                    if os.path.exists(fpath):
                        st.image(fpath, caption=title, use_container_width=True)

            # ── LOSO Results ──
            loso_path = os.path.join(run_dir(selected_run), "loso_results.json")
            if os.path.exists(loso_path):
                st.markdown("---")
                st.subheader("LOSO Cross-Validation (Phase 8)")
                with open(loso_path) as _f:
                    loso_data = json.load(_f)
                ls = loso_data.get("summary", {})
                c1, c2, c3 = st.columns(3)
                c1.metric("Balanced Accuracy",
                          f"{ls.get('balanced_accuracy_mean', 0):.3f} "
                          f"± {ls.get('balanced_accuracy_std', 0):.3f}")
                c2.metric("AUC-ROC",
                          f"{ls.get('auc_roc_mean', 0):.3f} "
                          f"± {ls.get('auc_roc_std', 0):.3f}")
                c3.metric("F1",
                          f"{ls.get('f1_mean', 0):.3f} "
                          f"± {ls.get('f1_std', 0):.3f}")

                loso_folds = loso_data.get("fold_results", [])
                if loso_folds:
                    import plotly.graph_objects as _pgo3
                    fig_loso = _pgo3.Figure()
                    fig_loso.add_trace(_pgo3.Bar(
                        x=[f"Subject {f.get('held_out_subject', i)}"
                           for i, f in enumerate(loso_folds)],
                        y=[f["auc_roc"] for f in loso_folds],
                        marker_color="#2ecc71",
                        text=[f"{f['auc_roc']:.3f}" for f in loso_folds],
                        textposition="auto",
                    ))
                    fig_loso.add_hline(y=0.5, line_dash="dash",
                                       line_color="red",
                                       annotation_text="Chance")
                    fig_loso.update_layout(
                        title="LOSO — Per-Subject AUC-ROC",
                        yaxis_title="AUC-ROC", yaxis_range=[0, 1],
                        height=350, template="plotly_white",
                    )
                    st.plotly_chart(fig_loso, width="stretch")

                mod = loso_data.get("moderator", {})
                if mod and "attend" in mod and "unattend" in mod:
                    st.markdown("**Moderator Analysis: Attend vs Unattend**")
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Attend AUC",
                               f"{mod['attend']['auc_roc_mean']:.3f} "
                               f"± {mod['attend']['auc_roc_std']:.3f}")
                    mc2.metric("Unattend AUC",
                               f"{mod['unattend']['auc_roc_mean']:.3f} "
                               f"± {mod['unattend']['auc_roc_std']:.3f}")

            # ── Cross-Condition Transfer Results ──
            xc_path = os.path.join(run_dir(selected_run),
                                   "cross_condition_results.json")
            if os.path.exists(xc_path):
                st.markdown("---")
                st.subheader("Cross-Condition Transfer (Phase 10)")
                with open(xc_path) as _f:
                    xc_data = json.load(_f)
                for direction, res in xc_data.items():
                    label = direction.replace("_", " → ").title()
                    st.markdown(f"**{label}**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Balanced Acc", f"{res.get('balanced_accuracy', 0):.3f}")
                    c2.metric("AUC-ROC", f"{res.get('auc_roc', 0):.3f}")
                    c3.metric("F1", f"{res.get('f1', 0):.3f}")

            # ── Gaze Comparison Results ──
            gc_path = os.path.join(run_dir(selected_run),
                                   "gaze_comparison_results.json")
            if os.path.exists(gc_path):
                st.markdown("---")
                st.subheader("Walking: EEG vs EEG+Gaze (Phase 9)")
                with open(gc_path) as _f:
                    gc_data = json.load(_f)
                for model_name in ["model_a_eeg", "model_b_fusion"]:
                    mr = gc_data.get(model_name, {})
                    if mr:
                        label = "Model A (EEG)" if "a_eeg" in model_name else "Model B (EEG+Gaze)"
                        st.metric(f"{label} AUC",
                                  f"{mr.get('auc_roc_mean', 0):.3f} "
                                  f"± {mr.get('auc_roc_std', 0):.3f}")
                comp = gc_data.get("comparison", {})
                if comp:
                    st.markdown(f"**Cohen's d:** {comp.get('cohens_d', 0):.3f}")
                    if "p_value" in comp:
                        st.markdown(
                            f"**Wilcoxon p:** {comp['p_value']:.4f} "
                            f"({'Significant' if comp.get('significant') else 'Not significant'})"
                        )


# ════════════════════════════════════════════════════════════
# ██  EVALUATION TAB
# ════════════════════════════════════════════════════════════

with tab_eval:
    st.header("Model Evaluation")
    st.caption("Vision pipeline comparison & EEG LOSO cross-validation")

    EVAL_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    # ── Cached results loader ───────────────────────────────
    def _load_cached_vision():
        p = os.path.join(EVAL_RESULTS_DIR, "vision_comparison.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    def _load_cached_eeg():
        p = os.path.join(EVAL_RESULTS_DIR, "eeg_loso_comparison.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    # ── Controls ────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1, 1, 1])
    with ctrl_c1:
        run_vision = st.checkbox("Vision Comparison", value=True)
    with ctrl_c2:
        run_eeg = st.checkbox("EEG LOSO Comparison", value=True)
    with ctrl_c3:
        n_repeats = st.number_input("CV repeats (vision)", min_value=1,
                                    max_value=20, value=5, step=1)

    run_eval = st.button("Run Evaluation", type="primary",
                         use_container_width=True)

    if run_eval:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
        import evaluate as eval_mod

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def _progress_cb(step, total, msg):
            progress_bar.progress(min(step / max(total, 1), 1.0))
            status_text.text(f"Step {step}/{total}: {msg}")

        with st.spinner("Running evaluation..."):
            res = eval_mod.run_full_evaluation(
                vision=run_vision, eeg=run_eeg,
                n_repeats=n_repeats, progress_cb=_progress_cb,
            )

        progress_bar.progress(1.0)
        status_text.text("Done!")

        if run_vision and "vision" in res:
            st.session_state["eval_vision"] = res["vision"]
        if run_eeg and "eeg" in res:
            st.session_state["eval_eeg"] = res["eeg"]

    # ── Load from cache / session ───────────────────────────
    vision_result = st.session_state.get("eval_vision") or _load_cached_vision()
    eeg_result = st.session_state.get("eval_eeg") or _load_cached_eeg()

    # ═══════════════════════════════════════════════════
    #  VISION RESULTS
    # ═══════════════════════════════════════════════════

    if vision_result and "summary" in vision_result and "error" not in vision_result:
        st.subheader("Vision Pipeline Comparison")

        summary = vision_result["summary"]
        label_names = vision_result.get("label_names", [])
        best_model = summary.get("best_model")

        model_display = {
            "clip_zeroshot": "Zero-shot CLIP",
            "clip_head": "Trained CLIP Head",
            "resnet50": "Fine-tuned ResNet-50",
        }

        if best_model:
            st.success(f"Best model (by Macro F1): **{model_display.get(best_model, best_model)}** "
                       f"— Macro F1 = {summary[best_model]['macro_f1_mean']:.3f} "
                       f"± {summary[best_model]['macro_f1_std']:.3f}")

        # Comparison table
        comp_rows = []
        for key in ["clip_zeroshot", "clip_head", "resnet50"]:
            s = summary.get(key)
            if not s:
                continue
            comp_rows.append({
                "Method": model_display.get(key, key),
                "Macro R": f"{s['macro_recall_mean']:.3f} ± {s['macro_recall_std']:.3f}",
                "Macro P": f"{s['macro_precision_mean']:.3f} ± {s['macro_precision_std']:.3f}",
                "Macro F1": f"{s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f}",
                "Weighted F1": f"{s['weighted_f1_mean']:.3f} ± {s['weighted_f1_std']:.3f}",
                "Accuracy": f"{s['accuracy_mean']:.3f} ± {s['accuracy_std']:.3f}",
            })
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True,
                         hide_index=True)

        # Label distribution
        label_dist = vision_result.get("label_dist")
        if label_dist:
            with st.expander("Label distribution"):
                dist_df = pd.DataFrame([
                    {"Category": k, "Count": v}
                    for k, v in label_dist.items()
                ])
                st.dataframe(dist_df, hide_index=True)

        # Confusion matrices side by side
        st.markdown("#### Confusion Matrices (aggregate)")
        cm_models = [k for k in ["clip_zeroshot", "clip_head", "resnet50"]
                     if k in summary and "confusion_matrix" in summary[k]]
        if cm_models and label_names:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
            import evaluate as eval_mod

            cols = st.columns(len(cm_models))
            for i, mk in enumerate(cm_models):
                with cols[i]:
                    fig = eval_mod.make_cm_figure(
                        summary[mk]["confusion_matrix"],
                        label_names,
                        title=model_display.get(mk, mk),
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

        # Relabel button
        if best_model:
            st.markdown("---")
            st.markdown("#### Relabel All Crops with Best Model")
            st.info(f"This will classify all gaze crops using "
                    f"**{model_display.get(best_model, best_model)}** and "
                    f"regenerate fusion features for the EEG pipeline.")

            if st.button("Relabel All Crops", type="secondary"):
                sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
                import evaluate as eval_mod

                relabel_bar = st.progress(0.0)
                relabel_status = st.empty()

                def _relabel_cb(step, total, msg):
                    relabel_bar.progress(min(step / max(total, 1), 1.0))
                    relabel_status.text(f"{msg} ({step}/{total})")

                with st.spinner("Relabeling crops..."):
                    relabel_res = eval_mod.relabel_crops_with_best(
                        best_model, progress_cb=_relabel_cb,
                    )

                relabel_bar.progress(1.0)
                if relabel_res.get("error"):
                    st.error(f"Relabeling failed: {relabel_res['error']}")
                else:
                    n_relabeled = relabel_res.get("n_crops_relabeled", 0)
                    st.success(f"Relabeled {n_relabeled} crops. "
                               f"Fusion CSV regenerated.")
                    cats = relabel_res.get("category_counts", {})
                    if cats:
                        st.dataframe(
                            pd.DataFrame([{"Category": k, "Count": v}
                                          for k, v in cats.items()]),
                            hide_index=True,
                        )

        # LaTeX export
        with st.expander("LaTeX source (vision)"):
            tex_path = os.path.join(EVAL_RESULTS_DIR,
                                    "table_vision_comparison.tex")
            if os.path.exists(tex_path):
                with open(tex_path) as f:
                    st.code(f.read(), language="latex")
            else:
                st.caption("Run evaluation to generate LaTeX.")

        # Download buttons
        dl_c1, dl_c2, dl_c3 = st.columns(3)
        csv_path = os.path.join(EVAL_RESULTS_DIR,
                                "table_vision_comparison.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                dl_c1.download_button("Download CSV", f.read(),
                                      "vision_comparison.csv", "text/csv")
        tex_path = os.path.join(EVAL_RESULTS_DIR,
                                "table_vision_comparison.tex")
        if os.path.exists(tex_path):
            with open(tex_path) as f:
                dl_c2.download_button("Download LaTeX", f.read(),
                                      "vision_comparison.tex", "text/plain")
        png_path = os.path.join(EVAL_RESULTS_DIR,
                                "cm_vision_comparison.png")
        if os.path.exists(png_path):
            with open(png_path, "rb") as f:
                dl_c3.download_button("Download CM (PNG)", f.read(),
                                      "cm_vision_comparison.png", "image/png")

    elif vision_result and vision_result.get("error"):
        st.warning(f"Vision evaluation error: {vision_result['error']}")

    # ═══════════════════════════════════════════════════
    #  EEG LOSO RESULTS
    # ═══════════════════════════════════════════════════

    if eeg_result and "summary" in eeg_result and "error" not in eeg_result:
        st.subheader("EEG LOSO Cross-Validation")

        eeg_summary = eeg_result["summary"]
        eeg_model_display = {
            "eeg_only": "EEGNet (EEG-only)",
            "multimodal": "MultimodalNet (EEG+ET)",
        }

        # Summary metrics as metric cards
        met_cols = st.columns(len(eeg_summary))
        for i, (mkey, s) in enumerate(eeg_summary.items()):
            with met_cols[i]:
                st.metric(
                    label=eeg_model_display.get(mkey, mkey),
                    value=f"F1 = {s['f1_mean']:.3f}",
                    delta=f"AUC {s['auc_roc_mean']:.3f} | "
                          f"Bal.Acc {s['balanced_accuracy_mean']:.3f}",
                )

        # Comparison table
        eeg_rows = []
        for key in ["eeg_only", "multimodal"]:
            s = eeg_summary.get(key)
            if not s:
                continue
            eeg_rows.append({
                "Method": eeg_model_display.get(key, key),
                "Bal. Acc": f"{s['balanced_accuracy_mean']:.3f} ± {s['balanced_accuracy_std']:.3f}",
                "AUC-ROC": f"{s['auc_roc_mean']:.3f} ± {s['auc_roc_std']:.3f}",
                "F1": f"{s['f1_mean']:.3f} ± {s['f1_std']:.3f}",
                "Precision": f"{s['precision_mean']:.3f} ± {s['precision_std']:.3f}",
                "Recall": f"{s['recall_mean']:.3f} ± {s['recall_std']:.3f}",
            })
        if eeg_rows:
            st.dataframe(pd.DataFrame(eeg_rows), use_container_width=True,
                         hide_index=True)

        # Per-fold table
        fold_data = eeg_result.get("fold_results", {})
        fold_rows = []
        for mkey in ["eeg_only", "multimodal"]:
            for fold in fold_data.get(mkey, []):
                fold_rows.append({
                    "Method": eeg_model_display.get(mkey, mkey),
                    "Held-out Subject": fold.get("held_out_subject", "?"),
                    "Bal. Acc": f"{fold['balanced_accuracy']:.3f}",
                    "AUC-ROC": f"{fold['auc_roc']:.3f}",
                    "F1": f"{fold['f1']:.3f}",
                    "Precision": f"{fold['precision']:.3f}",
                    "Recall": f"{fold['recall']:.3f}",
                })
        if fold_rows:
            with st.expander("Per-fold results"):
                st.dataframe(pd.DataFrame(fold_rows), hide_index=True,
                             use_container_width=True)

        # Per-fold bar chart
        if fold_rows:
            chart_data = []
            for mkey in ["eeg_only", "multimodal"]:
                for fold in fold_data.get(mkey, []):
                    chart_data.append({
                        "Model": eeg_model_display.get(mkey, mkey),
                        "Subject": f"sj{fold.get('held_out_subject', '?'):02d}",
                        "F1": fold["f1"],
                        "AUC-ROC": fold["auc_roc"],
                    })
            if chart_data:
                chart_df = pd.DataFrame(chart_data)
                fig_bar = px.bar(
                    chart_df, x="Subject", y="F1", color="Model",
                    barmode="group",
                    title="Per-fold F1 by held-out subject",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # Confusion matrices
        st.markdown("#### Confusion Matrices (aggregate)")
        cm_keys = [k for k in ["eeg_only", "multimodal"]
                   if k in eeg_summary and "confusion_matrix_sum" in eeg_summary[k]]
        if cm_keys:
            sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
            import evaluate as eval_mod

            cm_cols = st.columns(len(cm_keys))
            eeg_labels = ["CR", "FA"]
            for i, mk in enumerate(cm_keys):
                with cm_cols[i]:
                    fig = eval_mod.make_cm_figure(
                        eeg_summary[mk]["confusion_matrix_sum"],
                        eeg_labels,
                        title=eeg_model_display.get(mk, mk),
                    )
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

        # LaTeX export
        with st.expander("LaTeX source (EEG)"):
            tex_path = os.path.join(EVAL_RESULTS_DIR,
                                    "table_eeg_comparison.tex")
            if os.path.exists(tex_path):
                with open(tex_path) as f:
                    st.code(f.read(), language="latex")
            else:
                st.caption("Run evaluation to generate LaTeX.")

        # Download buttons
        dl_e1, dl_e2, dl_e3 = st.columns(3)
        csv_path = os.path.join(EVAL_RESULTS_DIR,
                                "table_eeg_comparison.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                dl_e1.download_button("Download CSV (EEG)", f.read(),
                                      "eeg_loso_comparison.csv", "text/csv")
        tex_path = os.path.join(EVAL_RESULTS_DIR,
                                "table_eeg_comparison.tex")
        if os.path.exists(tex_path):
            with open(tex_path) as f:
                dl_e2.download_button("Download LaTeX (EEG)", f.read(),
                                      "eeg_loso_comparison.tex", "text/plain")
        png_path = os.path.join(EVAL_RESULTS_DIR,
                                "cm_eeg_comparison.png")
        if os.path.exists(png_path):
            with open(png_path, "rb") as f:
                dl_e3.download_button("Download CM (PNG, EEG)", f.read(),
                                      "cm_eeg_comparison.png", "image/png")

    elif eeg_result and eeg_result.get("error"):
        st.warning(f"EEG LOSO error: {eeg_result['error']}")

    # ── No results yet ──────────────────────────────────────
    if not vision_result and not eeg_result:
        st.info("No evaluation results found. Click **Run Evaluation** above, "
                "or run `python src/evaluate.py` from the terminal.")
