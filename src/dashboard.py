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
import streamlit as st
import yaml

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
                         caption="Labeled Frame Grid (Zero-Shot)")

            v2_img = os.path.join(vp_dir, f"{_v_prefix}_V2_category_timeline.png")
            if os.path.exists(v2_img):
                st.image(v2_img, width="stretch",
                         caption="Category Timeline (Zero-Shot)")

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
