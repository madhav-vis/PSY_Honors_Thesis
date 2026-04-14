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


def attend_unattend_pairs(cond_list):
    """Pairs (attend_cond, unattend_cond) e.g. walk_attend + walk_unattend."""
    pairs = []
    attend_conds = [c for c in cond_list if c.endswith("_attend")]
    for a in sorted(attend_conds):
        prefix = a[: -len("_attend")]
        u = f"{prefix}_unattend"
        if u in cond_list:
            pairs.append((a, u))
    return pairs


@st.cache_data(ttl=120)
def load_gaze_session_trace(csv_path, max_points=8000):
    """Relative-time gaze x/y; subsample for interactive plots."""
    if not csv_path or not os.path.exists(csv_path):
        return None
    gaze = pd.read_csv(csv_path)
    if "timestamp [ns]" not in gaze.columns:
        return None
    gx_col = "gaze x [px]" if "gaze x [px]" in gaze.columns else None
    gy_col = "gaze y [px]" if "gaze y [px]" in gaze.columns else None
    if gx_col is None or gy_col is None:
        return None
    gaze = gaze.copy()
    gaze["timestamp_s"] = gaze["timestamp [ns]"] / 1e9
    t0 = gaze["timestamp_s"].iloc[0]
    gaze["rel_time_s"] = gaze["timestamp_s"] - t0
    n = len(gaze)
    if n > max_points:
        step = max(1, n // max_points)
        gaze = gaze.iloc[::step]
    return gaze[["rel_time_s", gx_col, gy_col]].rename(
        columns={gx_col: "gaze_x", gy_col: "gaze_y"}
    )


def find_subjects_conditions(rn):
    dd = data_dir(rn)
    if not os.path.isdir(dd):
        return [], []
    subjects = set()
    conditions = set()
    for f in os.listdir(dd):
        if f.endswith("_fused_metadata.csv"):
            parts = f.replace("_fused_metadata.csv", "").split("_", 1)
            sj = int(parts[0].replace("sj", ""))
            cond = parts[1]
            subjects.add(sj)
            conditions.add(cond)
    return sorted(subjects), sorted(conditions)


def file_exists_icon(path):
    return "yes" if os.path.exists(path) else "no"


# ── Sidebar ──────────────────────────────────────────────────

st.sidebar.title("PSY197B")
st.sidebar.caption("Mobile EEG + Eye Tracking")

runs = list_runs()
if not runs:
    st.sidebar.warning("No runs found in runs/")
    selected_run = None
else:
    selected_run = st.sidebar.selectbox("Run", runs)

sj_num = None
selected_cond = None
subjects = []
conditions = []

if selected_run:
    subjects, conditions = find_subjects_conditions(selected_run)
    if subjects:
        sj_num = st.sidebar.selectbox("Subject", subjects,
                                       format_func=lambda x: f"sj{x:02d}")
    if conditions:
        selected_cond = st.sidebar.selectbox("Condition", conditions)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Project root**  \n`{PROJECT_ROOT}`"
)


# ── Tabs ─────────────────────────────────────────────────────

tab_run, tab_overview, tab_eeg, tab_et, tab_vision, tab_fusion = st.tabs([
    "Run Manager",
    "Overview",
    "EEG",
    "Eye Tracking",
    "Vision",
    "Fusion & DL",
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

        step_options = {
            "all": "Full Pipeline (all steps)",
            "eeg": "01 — EEG Preprocessing",
            "et": "02 — ET Preprocessing",
            "fuse": "03 — EEG + ET Fusion",
            "features": "04 — Feature Extraction",
            "dl": "05 — DL Preparation",
            "checks": "06 — Sanity Checks",
        }
        selected_step = st.selectbox(
            "Step to run",
            list(step_options.keys()),
            format_func=lambda k: step_options[k],
        )

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
                   selected_step]
            with st.spinner(f"Running: {step_options[selected_step]}..."):
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
                   os.path.join(PROJECT_ROOT, "src", "vision", "vision_main.py")]
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

    # Run history
    st.markdown("---")
    st.subheader("Run History")
    if runs:
        history_rows = []
        for rn in runs:
            dd = data_dir(rn)
            pd_dir = plots_dir(rn)
            vd = os.path.join(run_dir(rn), "vision")
            n_data = len(glob.glob(os.path.join(dd, "*.csv"))) if os.path.isdir(dd) else 0
            n_plots = len(glob.glob(os.path.join(pd_dir, "*.png"))) if os.path.isdir(pd_dir) else 0
            n_fif = len(glob.glob(os.path.join(dd, "*.fif"))) if os.path.isdir(dd) else 0
            has_vision = "yes" if os.path.isdir(vd) and os.listdir(vd) else "no"
            has_tensors = "yes" if os.path.isdir(os.path.join(dd, "dl_tensors")) else "no"
            history_rows.append({
                "Run": rn,
                "CSVs": n_data,
                "Epochs (.fif)": n_fif,
                "Plots": n_plots,
                "Vision": has_vision,
                "DL Tensors": has_tensors,
            })
        st.dataframe(pd.DataFrame(history_rows), use_container_width=True,
                     hide_index=True)
    else:
        st.info("No runs yet. Configure and run the pipeline above.")


# ════════════════════════════════════════════════════════════
# TAB 2 — OVERVIEW
# ════════════════════════════════════════════════════════════

with tab_overview:
    if not selected_run:
        st.info("Select a run from the sidebar.")
    elif not sj_num or not selected_cond:
        st.info("Select a subject and condition from the sidebar.")
    else:
        st.header(f"Overview — sj{sj_num:02d} {selected_cond}")

        features = load_csv(os.path.join(
            data_dir(selected_run),
            f"sj{sj_num:02d}_{selected_cond}_features.csv",
        ))
        fused = load_csv(os.path.join(
            data_dir(selected_run),
            f"sj{sj_num:02d}_{selected_cond}_fused_metadata.csv",
        ))

        if features is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Trials", len(features))
            if "trialType" in features.columns:
                n_go = (features["trialType"] == 10).sum()
                n_nogo = (features["trialType"] == 20).sum()
                c2.metric("Go / NoGo", f"{n_go} / {n_nogo}")
            if "P300_cluster_uV" in features.columns:
                c3.metric("Mean P300 Cluster",
                          f"{features['P300_cluster_uV'].mean():.2f} uV")

            # Outcome distribution
            if "outcome" in features.columns:
                st.subheader("Outcome Distribution")
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
                fig_oc.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig_oc, use_container_width=True)

            # Quick stats table
            st.subheader("Quick Stats")
            stat_cols = [c for c in features.columns
                         if any(k in c for k in ["P300", "rt"])]
            if stat_cols:
                stats_df = features[stat_cols].describe().T[["mean", "std", "min", "max"]]
                stats_df.columns = ["Mean", "Std", "Min", "Max"]
                st.dataframe(stats_df.style.format("{:.3f}"),
                             use_container_width=True)
        else:
            st.warning("No features.csv found for this condition.")

        # Condition comparison
        if len(conditions) > 1:
            st.subheader("Condition Comparison")
            comp_rows = []
            for cond in conditions:
                f_path = os.path.join(data_dir(selected_run),
                                      f"sj{sj_num:02d}_{cond}_features.csv")
                df = load_csv(f_path)
                if df is not None:
                    row = {"Condition": cond, "N Trials": len(df)}
                    if "P300_cluster_uV" in df.columns:
                        row["P300 uV"] = f"{df['P300_cluster_uV'].mean():.2f}"
                    if "rt" in df.columns:
                        row["Mean RT"] = f"{df['rt'].dropna().mean():.3f}"
                    comp_rows.append(row)
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True,
                             hide_index=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — EEG
# ════════════════════════════════════════════════════════════

with tab_eeg:
    if not selected_run or not sj_num or not selected_cond:
        st.info("Select a run, subject, and condition from the sidebar.")
    else:
        st.header(f"EEG — sj{sj_num:02d} {selected_cond}")

        features = load_csv(os.path.join(
            data_dir(selected_run),
            f"sj{sj_num:02d}_{selected_cond}_features.csv",
        ))

        # Read P300 channels from the YAML config
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f)
            p300_channels = cfg.get("erp", {}).get("target_channels", [])
        except Exception:
            p300_channels = []
        channel_label = ", ".join(p300_channels) if p300_channels else "cluster"

        # ERP plot (pre-rendered)
        erp_img = os.path.join(plots_dir(selected_run),
                               f"sj{sj_num:02d}_L1_ERPs_Pz.png")
        if os.path.exists(erp_img):
            st.subheader("Go vs NoGo ERPs at Pz")
            st.image(erp_img, use_container_width=True)
        else:
            st.info("No ERP plot found. Run sanity checks first.")

        if features is not None and "trialType" in features.columns:
            if "P300_cluster_uV" in features.columns:
                st.subheader(f"P300 Cluster Amplitude ({channel_label})")

                c1, c2 = st.columns(2)
                go_vals = features.loc[features["trialType"] == 10,
                                       "P300_cluster_uV"]
                nogo_vals = features.loc[features["trialType"] == 20,
                                         "P300_cluster_uV"]
                c1.metric("Go mean", f"{go_vals.mean():.2f} uV")
                c2.metric("NoGo mean", f"{nogo_vals.mean():.2f} uV")

                fig_p3 = px.box(
                    features, x="trialType", y="P300_cluster_uV",
                    color="trialType",
                    labels={"trialType": "Trial Type",
                            "P300_cluster_uV": "P300 (uV)"},
                    color_discrete_map={10: "#3498db", 20: "#e74c3c"},
                    category_orders={"trialType": [10, 20]},
                )
                fig_p3.update_layout(
                    height=400, showlegend=False,
                    xaxis_ticktext=["Go", "NoGo"],
                    xaxis_tickvals=[10, 20],
                )
                st.plotly_chart(fig_p3, use_container_width=True)

                # P300 by outcome
                if "outcome" in features.columns:
                    st.subheader("P300 by Outcome")
                    fig_p3_oc = px.box(
                        features, x="outcome", y="P300_cluster_uV",
                        color="outcome",
                        labels={"outcome": "Outcome",
                                "P300_cluster_uV": "P300 (uV)"},
                        color_discrete_map={
                            "HIT": "#2ecc71", "MISS": "#e67e22",
                            "CORRECT_REJECTION": "#3498db",
                            "COMMISSION_ERROR": "#e74c3c",
                        },
                    )
                    fig_p3_oc.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig_p3_oc, use_container_width=True)

        elif features is None:
            st.warning("No features.csv found.")


# ════════════════════════════════════════════════════════════
# TAB 4 — EYE TRACKING
# ════════════════════════════════════════════════════════════

with tab_et:
    if not selected_run or not sj_num or not selected_cond:
        st.info("Select a run, subject, and condition from the sidebar.")
    else:
        st.header(f"Eye Tracking — sj{sj_num:02d} {selected_cond}")

        # Attend vs Unattend — same axes (session-relative time)
        et_map = _et_folder_map()
        data_root = _project_data_root()
        pairs = attend_unattend_pairs(conditions)
        gaze_compare_rows = []
        for a_cond, u_cond in pairs:
            a_path = os.path.join(
                data_root, f"sj{sj_num:02d}", "eye",
                et_map.get(a_cond, ""), "gaze_positions.csv",
            )
            u_path = os.path.join(
                data_root, f"sj{sj_num:02d}", "eye",
                et_map.get(u_cond, ""), "gaze_positions.csv",
            )
            ga = load_gaze_session_trace(a_path)
            gu = load_gaze_session_trace(u_path)
            if ga is not None and gu is not None:
                gaze_compare_rows.append((a_cond, u_cond, ga, gu))
        if gaze_compare_rows:
            st.subheader("Attend vs Unattend — Gaze X & Y (same time axis)")
            st.caption(
                "Each session starts at 0 s. Curves are subsampled for responsiveness."
            )
            for a_cond, u_cond, ga, gu in gaze_compare_rows:
                modality = a_cond[: -len("_attend")].replace("_", " ")
                st.markdown(f"**{modality}** — `{a_cond}` vs `{u_cond}`")
                fig_x = go.Figure()
                fig_x.add_trace(go.Scatter(
                    x=ga["rel_time_s"], y=ga["gaze_x"],
                    mode="lines", name=f"{a_cond} X",
                    line=dict(width=0.6, color="#3498db"),
                    opacity=0.85,
                ))
                fig_x.add_trace(go.Scatter(
                    x=gu["rel_time_s"], y=gu["gaze_x"],
                    mode="lines", name=f"{u_cond} X",
                    line=dict(width=0.6, color="#e74c3c"),
                    opacity=0.85,
                ))
                fig_x.update_layout(
                    height=320,
                    margin=dict(l=40, r=20, t=30, b=40),
                    xaxis_title="Time in session (s)",
                    yaxis_title="Gaze X (px)",
                    legend=dict(orientation="h", yanchor="bottom",
                                y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_x, use_container_width=True)

                fig_y = go.Figure()
                fig_y.add_trace(go.Scatter(
                    x=ga["rel_time_s"], y=ga["gaze_y"],
                    mode="lines", name=f"{a_cond} Y",
                    line=dict(width=0.6, color="#2980b9"),
                    opacity=0.85,
                ))
                fig_y.add_trace(go.Scatter(
                    x=gu["rel_time_s"], y=gu["gaze_y"],
                    mode="lines", name=f"{u_cond} Y",
                    line=dict(width=0.6, color="#c0392b"),
                    opacity=0.85,
                ))
                fig_y.update_layout(
                    height=320,
                    margin=dict(l=40, r=20, t=10, b=40),
                    xaxis_title="Time in session (s)",
                    yaxis_title="Gaze Y (px)",
                    legend=dict(orientation="h", yanchor="bottom",
                                y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_y, use_container_width=True)

        # Full-session gaze plot (pre-rendered)
        gaze_img = os.path.join(
            plots_dir(selected_run),
            f"sj{sj_num:02d}_L1_gaze_xy_pupil_{selected_cond}.png",
        )
        if os.path.exists(gaze_img):
            st.subheader("Full-Session Gaze Trace")
            st.image(gaze_img, use_container_width=True)

        # Gaze trajectory by outcome (pre-rendered)
        traj_img = os.path.join(
            plots_dir(selected_run),
            f"sj{sj_num:02d}_L_gaze_trajectories_{selected_cond}.png",
        )
        if os.path.exists(traj_img):
            st.subheader("Gaze Trajectories by Outcome")
            st.image(traj_img, use_container_width=True)

        # ET tensor info
        et_info_path = os.path.join(
            data_dir(selected_run),
            f"sj{sj_num:02d}_{selected_cond}_et_tensor_info.json",
        )
        if os.path.exists(et_info_path):
            st.subheader("ET Tensor Info")
            with open(et_info_path) as f:
                et_info = json.load(f)
            c1, c2, c3 = st.columns(3)
            shape = et_info.get("shape", [])
            c1.metric("Shape", f"{shape}")
            c2.metric("Channels", ", ".join(et_info.get("channel_names", [])))
            n_failed = len(et_info.get("failed_trials", []))
            c3.metric("Failed Trials", n_failed)

            n_blink = sum(et_info.get("has_blink", []))
            st.caption(f"Epochs with blinks: {n_blink}")

        # ET prepro summary
        et_prepro = load_csv(os.path.join(
            data_dir(selected_run),
            f"sj{sj_num:02d}_{selected_cond}_ET_Prepro1.csv",
        ))
        if et_prepro is not None:
            st.subheader("ET Preprocessing Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("ET Trials", len(et_prepro))
            c2.metric("Mean Gaze Samples/Trial",
                      f"{et_prepro['gaze_n_samples'].mean():.0f}")
            c3.metric("Mean Gaze X",
                      f"{et_prepro['gaze_mean_x_px'].mean():.0f} px")

            with st.expander("ET Prepro Table"):
                st.dataframe(et_prepro, use_container_width=True, height=300)


# ════════════════════════════════════════════════════════════
# TAB 5 — VISION
# ════════════════════════════════════════════════════════════

with tab_vision:
    if not selected_run or not sj_num or not selected_cond:
        st.info("Select a run, subject, and condition from the sidebar.")
    else:
        st.header(f"Vision — sj{sj_num:02d} {selected_cond}")

        vp_dir = vision_plots_dir(selected_run)
        prefix = f"sj{sj_num:02d}_{selected_cond}"

        # V5: Embedding clusters
        v5_img = os.path.join(vp_dir, f"{prefix}_V5_embedding_clusters.png")
        if os.path.exists(v5_img):
            st.subheader("CLIP Embedding Clusters (UMAP)")
            st.image(v5_img, use_container_width=True)

        # V4: Optimal K
        v4_img = os.path.join(vp_dir, f"{prefix}_V4_optimal_k.png")
        if os.path.exists(v4_img):
            st.subheader("Optimal K Analysis")
            st.image(v4_img, use_container_width=True)

        # V6: Cluster timeline
        v6_img = os.path.join(vp_dir, f"{prefix}_V6_cluster_timeline.png")
        if os.path.exists(v6_img):
            st.subheader("Cluster Timeline")
            st.image(v6_img, use_container_width=True)

        # V1: Labeled frames
        v1_img = os.path.join(vp_dir, f"{prefix}_V1_labeled_frames.png")
        if os.path.exists(v1_img):
            st.subheader("Labeled Frame Grid (Zero-Shot)")
            st.image(v1_img, use_container_width=True)

        # V2: Category timeline
        v2_img = os.path.join(vp_dir, f"{prefix}_V2_category_timeline.png")
        if os.path.exists(v2_img):
            st.subheader("Category Timeline (Zero-Shot)")
            st.image(v2_img, use_container_width=True)

        # Vision results table
        vr_path = os.path.join(
            vision_dir(selected_run, sj_num, selected_cond),
            f"sj{sj_num:02d}_{selected_cond}_vision_results.csv",
        )
        vision_results = load_csv(vr_path)
        if vision_results is not None:
            st.subheader("Fixation-Level Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Fixations", len(vision_results))
            if "confidence" in vision_results.columns:
                c2.metric("Mean Confidence",
                          f"{vision_results['confidence'].mean():.3f}")
            if "cluster_id" in vision_results.columns:
                n_clusters = vision_results["cluster_id"].nunique()
                c3.metric("N Clusters", n_clusters)

            # Cluster distribution bar chart
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
                st.plotly_chart(fig_cl, use_container_width=True)

            # Confidence histogram
            if "confidence" in vision_results.columns:
                fig_conf = px.histogram(
                    vision_results, x="confidence", nbins=30,
                    labels={"confidence": "CLIP Confidence"},
                    color_discrete_sequence=["#3498db"],
                )
                fig_conf.add_vline(x=0.25, line_dash="dash", line_color="red",
                                   annotation_text="chance")
                fig_conf.add_vline(x=0.45, line_dash="dash", line_color="green",
                                   annotation_text="reliable")
                fig_conf.update_layout(height=300,
                                       title="Confidence Distribution")
                st.plotly_chart(fig_conf, use_container_width=True)

            with st.expander("Full Fixation Table"):
                st.dataframe(vision_results, use_container_width=True,
                             height=400)

        # Sample crops
        crops_d = os.path.join(
            vision_dir(selected_run, sj_num, selected_cond), "crops"
        )
        if os.path.isdir(crops_d):
            crop_files = sorted(glob.glob(os.path.join(crops_d, "*.png")))
            if crop_files:
                st.subheader("Sample Gaze Crops")
                n_show = min(12, len(crop_files))
                sample = crop_files[::max(1, len(crop_files) // n_show)][:n_show]
                cols = st.columns(4)
                for i, cf in enumerate(sample):
                    with cols[i % 4]:
                        st.image(cf, caption=os.path.basename(cf),
                                 use_container_width=True)


# ════════════════════════════════════════════════════════════
# TAB 6 — FUSION & DL
# ════════════════════════════════════════════════════════════

with tab_fusion:
    if not selected_run or not sj_num or not selected_cond:
        st.info("Select a run, subject, and condition from the sidebar.")
    else:
        st.header(f"Fusion & DL — sj{sj_num:02d} {selected_cond}")

        fused = load_csv(os.path.join(
            data_dir(selected_run),
            f"sj{sj_num:02d}_{selected_cond}_fused_metadata.csv",
        ))
        features = load_csv(os.path.join(
            data_dir(selected_run),
            f"sj{sj_num:02d}_{selected_cond}_features.csv",
        ))
        vision_feats = load_csv(os.path.join(
            data_dir(selected_run),
            f"sj{sj_num:02d}_{selected_cond}_vision_trial_features.csv",
        ))

        # Fused metadata preview
        if fused is not None:
            st.subheader("Fused Metadata")
            st.dataframe(fused.head(20), use_container_width=True, height=350)
            st.caption(f"Shape: {fused.shape}")

        # P300 vs dominant cluster
        if (features is not None and vision_feats is not None
                and "dominant_cluster" in vision_feats.columns
                and "P300_cluster_uV" in features.columns):
            merged = features.merge(
                vision_feats[["trialIdx", "dominant_cluster", "emb_spread",
                              "cluster_entropy"]],
                on="trialIdx", how="left",
            )
            merged_valid = merged.dropna(subset=["dominant_cluster"])

            if len(merged_valid) > 0:
                st.subheader("P300 by Visual Scene Cluster")
                fig_p3c = px.box(
                    merged_valid,
                    x="dominant_cluster",
                    y="P300_cluster_uV",
                    color="dominant_cluster",
                    labels={"dominant_cluster": "Dominant Cluster",
                            "P300_cluster_uV": "P300 (µV)"},
                )
                fig_p3c.update_layout(showlegend=False, height=400)
                st.plotly_chart(fig_p3c, use_container_width=True)

                # emb_spread vs P300
                if "emb_spread" in merged_valid.columns:
                    st.subheader("Visual Diversity vs P300")
                    fig_scatter = px.scatter(
                        merged_valid, x="emb_spread", y="P300_cluster_uV",
                        color="outcome" if "outcome" in merged_valid.columns else None,
                        labels={"emb_spread": "Embedding Spread",
                                "P300_cluster_uV": "P300 (µV)"},
                        opacity=0.5,
                        color_discrete_map={
                            "HIT": "#2ecc71", "MISS": "#e67e22",
                            "CORRECT_REJECTION": "#3498db",
                            "COMMISSION_ERROR": "#e74c3c",
                        },
                        trendline="ols",
                    )
                    fig_scatter.update_layout(height=400)
                    st.plotly_chart(fig_scatter, use_container_width=True)

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
                st.plotly_chart(fig_ent, use_container_width=True)

        # DL tensor shapes
        dl_dir = os.path.join(data_dir(selected_run), "dl_tensors")
        if os.path.isdir(dl_dir):
            st.subheader("DL Tensor Summary")
            tensor_info = []
            prefix = f"sj{sj_num:02d}_{selected_cond}"
            for name in ["X_eeg_train", "X_eeg_val", "X_et_train", "X_et_val",
                          "y_train", "y_val"]:
                npy_path = os.path.join(dl_dir, f"{prefix}_{name}.npy")
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
                             use_container_width=True, hide_index=True)
            else:
                st.info("No DL tensors found for this condition.")

        # Vision trial features table
        if vision_feats is not None:
            with st.expander("Vision Trial Features Table"):
                display_cols = [c for c in vision_feats.columns
                                if c != "mean_embedding"]
                st.dataframe(vision_feats[display_cols],
                             use_container_width=True, height=350)
