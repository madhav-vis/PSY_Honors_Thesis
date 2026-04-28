"""
PSY197B — Mobile EEG + Eye Tracking Dashboard
Run:  streamlit run src/dashboard.py
"""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
import mne
from plotly.subplots import make_subplots

import et_viz

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "run_config.yaml")
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python3.11")

st.set_page_config(
    page_title="PSY197B Dashboard",
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


def vision_plots_dir(rn):
    return os.path.join(plots_dir(rn), "vision")


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
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        rel = cfg.get("data", {}).get("root", "data")
        return os.path.join(PROJECT_ROOT, rel)
    except Exception:
        return os.path.join(PROJECT_ROOT, "data")


def _et_folder_map():
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg.get("et", {}).get("folder_map", {})
    except Exception:
        return {}


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


def _build_movement_behavior_rows(features_df, cond_label):
    if features_df is None or "trialType" not in features_df.columns or "outcome" not in features_df.columns:
        return None
    movement, _ = _parse_condition_parts(cond_label)
    if movement not in {"sit", "walk"}:
        return None
    go = features_df[features_df["trialType"] == 10]
    nogo = features_df[features_df["trialType"] == 20]
    p_correct = float((go["outcome"].astype(str).str.upper() == "HIT").mean()) if len(go) else np.nan
    p_error = float((nogo["outcome"].astype(str).str.upper() == "COMMISSION_ERROR").mean()) if len(nogo) else np.nan
    return {
        "movement": movement.title(),
        "pCorrect": p_correct,
        "pError": p_error,
    }


# ── Sidebar ──────────────────────────────────────────────────

st.sidebar.header("Run History")

runs = list_runs()
if not runs:
    st.sidebar.warning("No runs found in runs/")
    selected_run = None
else:
    selected_run = st.sidebar.selectbox("Run", runs)

sj_num = None
subjects = []
conditions = []

if selected_run:
    subjects, conditions = find_subjects_conditions(selected_run)
    if subjects:
        sj_num = subjects[0] if len(subjects) == 1 else None
    if sj_num is None:
        import re
        m = re.search(r"sj(\d+)", selected_run)
        if m:
            sj_num = int(m.group(1))
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

st.title("PSY197B")
st.caption("Mobile EEG + Eye Tracking")

if selected_run and len(subjects) > 1:
    sj_num = st.selectbox("Subject", subjects,
                           format_func=lambda x: f"sj{x:02d}")


# ── Tabs ─────────────────────────────────────────────────────

tab_run, tab_overview, tab_eeg, tab_et, tab_vision, tab_fusion, tab_nogo = st.tabs([
    "Run Manager",
    "Overview",
    "EEG",
    "Eye Tracking",
    "Vision",
    "Fusion & DL",
    "EEGNet",
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

        col_b1, col_b2 = st.columns(2)
        run_main = col_b1.button("Run EEG/ET Pipeline", type="primary")
        run_vision = col_b2.button("Run Vision Pipeline")

        if "pipeline_log" not in st.session_state:
            st.session_state.pipeline_log = ""
        if "pipeline_running" not in st.session_state:
            st.session_state.pipeline_running = False

        if run_main and not st.session_state.pipeline_running:
            st.session_state.pipeline_running = True
            st.session_state.pipeline_log = ""
            cmd = [VENV_PYTHON, os.path.join(PROJECT_ROOT, "src", "main.py"),
                   "all"]
            with st.spinner("Running EEG/ET pipeline..."):
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=1800,
                        cwd=PROJECT_ROOT,
                    )
                    st.session_state.pipeline_log = result.stdout + result.stderr
                    if result.returncode == 0:
                        st.success("Pipeline completed!")
                    else:
                        st.error(f"Pipeline failed (exit code {result.returncode})")
                except subprocess.TimeoutExpired:
                    st.error("Pipeline timed out (30 min limit)")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    st.session_state.pipeline_running = False
                    st.cache_data.clear()

        if run_vision and not st.session_state.pipeline_running:
            st.session_state.pipeline_running = True
            st.session_state.pipeline_log = ""
            cmd = [VENV_PYTHON,
                   os.path.join(PROJECT_ROOT, "src", "vision", "vision_main.py"),
                   "--run-dir", run_dir(selected_run) if selected_run else ""]
            with st.spinner("Running vision pipeline..."):
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=3600,
                        cwd=PROJECT_ROOT,
                    )
                    st.session_state.pipeline_log = result.stdout + result.stderr
                    if result.returncode == 0:
                        st.success("Vision pipeline completed!")
                    else:
                        st.error(f"Vision pipeline failed (exit code {result.returncode})")
                except subprocess.TimeoutExpired:
                    st.error("Vision pipeline timed out (60 min limit)")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    st.session_state.pipeline_running = False
                    st.cache_data.clear()

        if st.session_state.pipeline_log:
            with st.expander("Pipeline Output", expanded=True):
                st.code(st.session_state.pipeline_log, language="text")


# ════════════════════════════════════════════════════════════
# TAB 2 — OVERVIEW
# ════════════════════════════════════════════════════════════

with tab_overview:
    if not selected_run:
        st.info("Select a run from the sidebar.")
    elif not sj_num:
        st.info("Select a subject.")
    else:
        st.header(f"Overview — sj{sj_num:02d}")

        # Condition comparison table
        comp_rows = []
        for cond in conditions:
            f_path = os.path.join(data_dir(selected_run),
                                  f"sj{sj_num:02d}_{cond}_features.csv")
            df = load_csv(f_path)
            if df is not None:
                row = {"Condition": cond, "N Trials": len(df)}
                if "trialType" in df.columns:
                    row["Go"] = int((df["trialType"] == 10).sum())
                    row["NoGo"] = int((df["trialType"] == 20).sum())
                if "rt" in df.columns:
                    row["Mean RT"] = f"{df['rt'].dropna().mean():.3f}"
                comp_rows.append(row)
        if comp_rows:
            st.subheader("Condition Comparison")
            st.dataframe(pd.DataFrame(comp_rows), width="stretch",
                         hide_index=True)

        for cond in conditions:
            st.markdown("---")
            st.subheader(f"{cond}")

            features = load_csv(os.path.join(
                data_dir(selected_run),
                f"sj{sj_num:02d}_{cond}_features.csv",
            ))
            if features is None:
                st.warning(f"No features.csv found for {cond}.")
                continue

            if "outcome" in features.columns:
                outcome_counts = features["outcome"].value_counts()
                fig_oc = px.bar(
                    x=outcome_counts.index, y=outcome_counts.values,
                    labels={"x": "Outcome", "y": "Count"},
                    color=outcome_counts.index,
                    color_discrete_map={
                        "HIT": "#2ecc71", "MISS": "#e67e22",
                        "CORRECT_REJECTION": "#3498db",
                        "COMMISSION_ERROR": "#e74c3c",
                    },
                )
                fig_oc.update_layout(showlegend=False, height=350,
                                     title="Outcome Distribution")
                st.plotly_chart(fig_oc, width="stretch")

            stat_cols = [c for c in features.columns if "rt" in c]
            if stat_cols:
                stats_df = features[stat_cols].describe().T[["mean", "std", "min", "max"]]
                stats_df.columns = ["Mean", "Std", "Min", "Max"]
                st.dataframe(stats_df.style.format("{:.3f}"),
                             width="stretch")


# ════════════════════════════════════════════════════════════
# TAB 3 — EEG
# ════════════════════════════════════════════════════════════

with tab_eeg:
    if not selected_run or not sj_num:
        st.info("Select a run and subject.")
    else:
        st.header(f"EEG — sj{sj_num:02d}")

        pd_run = plots_dir(selected_run)
        erp_cluster = os.path.join(pd_run,
                                   f"sj{sj_num:02d}_L1_ERPs_cluster.png")
        erp_pz = os.path.join(pd_run, f"sj{sj_num:02d}_L1_ERPs_Pz.png")
        if os.path.exists(erp_cluster):
            erp_img = erp_cluster
            erp_caption = "Go vs NoGo ERPs (mean over erp.target_channels)"
        elif os.path.exists(erp_pz):
            erp_img = erp_pz
            erp_caption = "Go vs NoGo ERPs at Pz (legacy plot)"
        else:
            erp_img = None
            erp_caption = None
        if erp_img:
            st.subheader(erp_caption)
            st.image(erp_img, width="stretch")
        else:
            st.info("No ERP plot found. Run sanity checks first.")

        # Condition-grid ERP (Attend/Unattend × Sit/Walk), no error bars.
        erp_by_cell = {}
        erp_load_errors = []
        missing_feature_epo = []
        for cond in conditions:
            epo_path = os.path.join(
                data_dir(selected_run),
                f"sj{sj_num:02d}_{cond}_Features-epo.fif",
            )
            if not os.path.exists(epo_path):
                missing_feature_epo.append((cond, epo_path))
                continue
            try:
                epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
                if len(epochs) == 0:
                    erp_load_errors.append(f"{cond}: empty Features-epo.fif")
                    continue
                ch = "Pz" if "Pz" in epochs.ch_names else epochs.ch_names[0]
                ch_i = epochs.ch_names.index(ch)
                data_ep = epochs.get_data()
                go_idx, nogo_idx = _go_nogo_indices_from_epochs(epochs)
                if len(go_idx) == 0 and len(nogo_idx) == 0:
                    erp_load_errors.append(
                        f"{cond}: no trials with trialType 10/20 after coercion"
                    )
                    continue
                erp_by_cell[_condition_grid_label(cond)] = {
                    "times_ms": epochs.times * 1000.0,
                    "go": data_ep[go_idx, ch_i, :].mean(axis=0) * 1e6 if len(go_idx) else None,
                    "nogo": data_ep[nogo_idx, ch_i, :].mean(axis=0) * 1e6 if len(nogo_idx) else None,
                    "n_go": int(len(go_idx)),
                    "n_nogo": int(len(nogo_idx)),
                }
            except Exception as exc:
                erp_load_errors.append(f"{cond}: {exc}")

        cell_order = ["Attend Sit", "Unattend Sit", "Attend Walk", "Unattend Walk"]
        if missing_feature_epo:
            lines = [
                f"`{c}` — expected `{os.path.basename(p)}`"
                for c, p in missing_feature_epo
            ]
            st.warning(
                "Some conditions have **no feature epochs file** (the interactive ERP grid "
                "skips them). Usually the pipeline stopped before fusion or feature extraction "
                "for that block. Pick a run where those files exist, or re-run from EEG "
                "preprocess / fusion / `extract_features`.\n\n"
                + "\n".join(lines)
            )
        if erp_load_errors:
            st.warning("Could not build ERP for some conditions:\n\n" + "\n".join(erp_load_errors))

        grid_status_rows = []
        for cond in conditions:
            epo_path = os.path.join(
                data_dir(selected_run),
                f"sj{sj_num:02d}_{cond}_Features-epo.fif",
            )
            cell = _condition_grid_label(cond)
            grid_status_rows.append({
                "Condition": cond,
                "Panel": cell,
                "Features-epo.fif": "yes" if os.path.exists(epo_path) else "no",
                "Plotted": "yes" if cell in erp_by_cell else "no",
            })
        if grid_status_rows:
            with st.expander(
                "ERP grid: `Features-epo.fif` status per condition",
                expanded=bool(missing_feature_epo),
            ):
                st.dataframe(
                    pd.DataFrame(grid_status_rows),
                    hide_index=True,
                    width="stretch",
                )

        if any(c in erp_by_cell for c in cell_order):
            st.markdown("---")
            st.subheader("Go vs NoGo ERPs by attention and movement")
            fig_grid = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=cell_order,
                shared_xaxes=True,
                shared_yaxes=True,
                vertical_spacing=0.14,
                horizontal_spacing=0.10,
            )
            pos = {
                "Attend Sit": (1, 1),
                "Unattend Sit": (1, 2),
                "Attend Walk": (2, 1),
                "Unattend Walk": (2, 2),
            }
            for cell, (r, c) in pos.items():
                item = erp_by_cell.get(cell)
                if not item:
                    continue
                if item["nogo"] is not None:
                    fig_grid.add_trace(
                        go.Scatter(
                            x=item["times_ms"],
                            y=item["nogo"],
                            mode="lines",
                            name="NoGo",
                            line=dict(color="#e74c3c", width=2),
                            showlegend=(cell == "Attend Sit"),
                        ),
                        row=r,
                        col=c,
                    )
                if item["go"] is not None:
                    fig_grid.add_trace(
                        go.Scatter(
                            x=item["times_ms"],
                            y=item["go"],
                            mode="lines",
                            name="Go",
                            line=dict(color="#2980b9", width=2),
                            showlegend=(cell == "Attend Sit"),
                        ),
                        row=r,
                        col=c,
                    )
                fig_grid.add_vline(x=0, line_color="gray", line_dash="dot", row=r, col=c)
            fig_grid.update_xaxes(title_text="Time (ms)", row=2, col=1)
            fig_grid.update_xaxes(title_text="Time (ms)", row=2, col=2)
            fig_grid.update_yaxes(title_text="Amplitude (uV)", row=1, col=1)
            fig_grid.update_yaxes(title_text="Amplitude (uV)", row=2, col=1)
            fig_grid.update_layout(height=620, template="plotly_white", legend=dict(orientation="h"))
            st.plotly_chart(fig_grid, width="stretch")


# ════════════════════════════════════════════════════════════
# TAB 4 — EYE TRACKING
# ════════════════════════════════════════════════════════════

with tab_et:
    if not selected_run or not sj_num:
        st.info("Select a run and subject.")
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
# TAB 5 — VISION
# ════════════════════════════════════════════════════════════

with tab_vision:
    if not selected_run or not sj_num:
        st.info("Select a run and subject.")
    else:
        st.header(f"Vision — sj{sj_num:02d}")

        vp_dir = vision_plots_dir(selected_run)

        for _v_cond in conditions:
            st.markdown("---")
            st.subheader(f"{_v_cond}")

            _v_prefix = f"sj{sj_num:02d}_{_v_cond}"

            v5_img = os.path.join(vp_dir, f"{_v_prefix}_V5_embedding_clusters.png")
            if os.path.exists(v5_img):
                st.image(v5_img, width="stretch",
                         caption="CLIP Embedding Clusters (UMAP)")

            v4_img = os.path.join(vp_dir, f"{_v_prefix}_V4_optimal_k.png")
            if os.path.exists(v4_img):
                st.image(v4_img, width="stretch",
                         caption="Optimal K Analysis")

            v6_img = os.path.join(vp_dir, f"{_v_prefix}_V6_cluster_timeline.png")
            if os.path.exists(v6_img):
                st.image(v6_img, width="stretch",
                         caption="Cluster Timeline")

            v1_img = os.path.join(vp_dir, f"{_v_prefix}_V1_labeled_frames.png")
            if os.path.exists(v1_img):
                st.image(v1_img, width="stretch",
                         caption="Labeled Frame Grid")

            v2_img = os.path.join(vp_dir, f"{_v_prefix}_V2_category_timeline.png")
            if os.path.exists(v2_img):
                st.image(v2_img, width="stretch",
                         caption="Category Timeline")

            v3_img = os.path.join(vp_dir, f"{_v_prefix}_V3_clip_vs_human.png")
            if os.path.exists(v3_img):
                st.image(v3_img, width="stretch",
                         caption="CLIP vs Human Labels (Accuracy)")

            vr_path = os.path.join(
                vision_dir(selected_run, sj_num, _v_cond),
                f"sj{sj_num:02d}_{_v_cond}_vision_results.csv",
            )
            vision_results = load_csv(vr_path)
            if vision_results is not None:
                _c1, _c2, _c3 = st.columns(3)
                _c1.metric("Total Fixations", len(vision_results))
                if "confidence" in vision_results.columns:
                    _c2.metric("Mean Confidence",
                               f"{vision_results['confidence'].mean():.3f}")
                if "cluster_id" in vision_results.columns:
                    _c3.metric("N Clusters",
                               vision_results["cluster_id"].nunique())

                if "gaze_target_category" in vision_results.columns:
                    cat_counts = vision_results["gaze_target_category"].value_counts()
                    _cat_colors = {
                        "sky": "#87CEEB", "ocean": "#1E90FF",
                        "water": "#4169E1", "people": "#FF6B6B",
                        "vegetation": "#2E8B57", "trail_ground": "#CD853F",
                        "other": "#A9A9A9",
                    }
                    fig_cat = px.bar(
                        x=cat_counts.index,
                        y=cat_counts.values,
                        labels={"x": "Category", "y": "Count"},
                        color=cat_counts.index,
                        color_discrete_map=_cat_colors,
                    )
                    fig_cat.update_layout(showlegend=False, height=350,
                                          title="Category Distribution (Fine-Tuned Head)")
                    st.plotly_chart(fig_cat, use_container_width=True)

                if "cluster_id" in vision_results.columns:
                    cluster_counts = vision_results["cluster_id"].value_counts().sort_index()
                    fig_cl = px.bar(
                        x=cluster_counts.index.astype(str),
                        y=cluster_counts.values,
                        labels={"x": "Cluster ID", "y": "Count"},
                        color=cluster_counts.index.astype(str),
                    )
                    fig_cl.update_layout(showlegend=False, height=300,
                                         title="Cluster Size Distribution")
                    st.plotly_chart(fig_cl, width="stretch")

                if "confidence" in vision_results.columns:
                    fig_conf = px.histogram(
                        vision_results, x="confidence", nbins=30,
                        labels={"confidence": "CLIP Confidence"},
                        color_discrete_sequence=["#3498db"],
                    )
                    fig_conf.add_vline(x=0.25, line_dash="dash",
                                       line_color="red",
                                       annotation_text="chance")
                    fig_conf.add_vline(x=0.45, line_dash="dash",
                                       line_color="green",
                                       annotation_text="reliable")
                    fig_conf.update_layout(height=300,
                                           title="Confidence Distribution")
                    st.plotly_chart(fig_conf, width="stretch")

                with st.expander("Full Fixation Table"):
                    st.dataframe(vision_results, width="stretch",
                                 height=400)

            crops_d = os.path.join(
                vision_dir(selected_run, sj_num, _v_cond), "crops"
            )
            if os.path.isdir(crops_d):
                crop_files = sorted(glob.glob(os.path.join(crops_d, "*.png")))
                if crop_files:
                    n_show = min(12, len(crop_files))
                    sample = crop_files[::max(1, len(crop_files) // n_show)][:n_show]
                    cols = st.columns(4)
                    for i, cf in enumerate(sample):
                        with cols[i % 4]:
                            st.image(cf, caption=os.path.basename(cf),
                                     width="stretch")


# ════════════════════════════════════════════════════════════
# TAB 6 — FUSION & DL
# ════════════════════════════════════════════════════════════

with tab_fusion:
    if not selected_run or not sj_num:
        st.info("Select a run and subject.")
    else:
        st.header(f"Fusion & DL — sj{sj_num:02d}")

        # Attend vs Unattend cluster entropy comparison
        if len(conditions) > 1:
            entropy_data = []
            for cond in conditions:
                vf = load_csv(os.path.join(
                    data_dir(selected_run),
                    f"sj{sj_num:02d}_{cond}_vision_trial_features.csv",
                ))
                if vf is not None and "cluster_entropy" in vf.columns:
                    valid = vf["cluster_entropy"].dropna()
                    for v in valid:
                        entropy_data.append({"Condition": cond,
                                             "Cluster Entropy": v})
            if entropy_data:
                st.subheader("Cluster Entropy: Attend vs Unattend")
                fig_ent = px.box(
                    pd.DataFrame(entropy_data),
                    x="Condition", y="Cluster Entropy",
                    color="Condition",
                )
                fig_ent.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig_ent, width="stretch")

        for _f_cond in conditions:
            st.markdown("---")
            st.subheader(f"{_f_cond}")

            fused = load_csv(os.path.join(
                data_dir(selected_run),
                f"sj{sj_num:02d}_{_f_cond}_fused_metadata.csv",
            ))
            vision_feats = load_csv(os.path.join(
                data_dir(selected_run),
                f"sj{sj_num:02d}_{_f_cond}_vision_trial_features.csv",
            ))

            if fused is not None:
                st.dataframe(fused.head(20), width="stretch",
                             height=350)
                st.caption(f"Shape: {fused.shape}")

            dl_dir = os.path.join(data_dir(selected_run), "dl_tensors")
            if os.path.isdir(dl_dir):
                tensor_info = []
                _f_prefix = f"sj{sj_num:02d}_{_f_cond}"
                for name in ["X_eeg_train", "X_eeg_val", "X_et_train",
                              "X_et_val", "y_train", "y_val"]:
                    npy_path = os.path.join(dl_dir, f"{_f_prefix}_{name}.npy")
                    if os.path.exists(npy_path):
                        arr = np.load(npy_path, mmap_mode="r")
                        tensor_info.append({
                            "Tensor": name,
                            "Shape": str(arr.shape),
                            "Dtype": str(arr.dtype),
                            "Size (MB)": f"{os.path.getsize(npy_path) / 1e6:.1f}",
                        })
                if tensor_info:
                    st.dataframe(pd.DataFrame(tensor_info),
                                 width="stretch", hide_index=True)

            if vision_feats is not None:
                with st.expander("Vision Trial Features"):
                    display_cols = [c for c in vision_feats.columns
                                    if c != "mean_embedding"]
                    st.dataframe(vision_feats[display_cols],
                                 width="stretch", height=350)


# ════════════════════════════════════════════════════════════
# TAB 7 — INHIBITORY CONTROL (No-Go ML Pipeline)
# ════════════════════════════════════════════════════════════

with tab_nogo:
    if not selected_run or not sj_num:
        st.info("Select a run and subject.")
    else:
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
                    import mne
                    epo_path = os.path.join(
                        data_dir(selected_run),
                        f"sj{sj_num:02d}_{cond}_Features-epo.fif")
                    if not os.path.exists(epo_path):
                        continue
                    epochs = mne.read_epochs(epo_path, preload=True,
                                             verbose=False)
                    if epochs.metadata is None:
                        continue
                    meta = epochs.metadata
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
                        if ch in epochs.ch_names:
                            pz_i = epochs.ch_names.index(ch)
                            break
                    if pz_i is None:
                        pz_i = 0

                    edata = epochs.get_data()
                    times = epochs.times * 1000
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
                        title=f"{cond} — {epochs.ch_names[pz_i]}",
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
