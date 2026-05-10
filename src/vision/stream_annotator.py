"""Streamlit-based gaze crop annotation + training tool.

Run:  streamlit run src/vision/stream_annotator.py

Five tabs:
  1. Generate Crops  — crop status grid, data availability
  2. Label           — annotation interface with labeler ID + flag support
  3. Statistics      — per-class distributions, coverage, inter-rater agreement
  4. Train           — CLIP linear head + ResNet-50, live metrics, model versioning
  5. Evaluate        — test-set confusion matrices and model comparison
"""

import os
import sys

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

LABEL_NAMES = list(CATEGORIES.keys())
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ── Page config ───────────────────────────────────────────────

st.set_page_config(
    page_title="Gaze Crop Tool",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────

_SS_DEFAULTS = {
    "labeler_id": "",
    "crop_idx": {},
    "last_pair": None,
    "train_history_clip": [],
    "train_history_resnet": [],
    "train_running": False,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Migration (run once per session) ─────────────────────────

@st.cache_data(show_spinner=False)
def _run_migration():
    return migrate_existing_labels(RUNS_ROOT)

migrated = _run_migration()
if migrated > 0:
    st.toast(f"Migrated {migrated} labels into central store.", icon="✅")

# ── Cached helpers ────────────────────────────────────────────

@st.cache_data(ttl=30)
def _load_all_labels():
    return load_labels()


@st.cache_data(ttl=30)
def _load_trainable():
    return load_trainable_labels()


@st.cache_data(ttl=60)
def _find_all_embeddings() -> dict:
    """Return dict: (sj_num, condition) → embeddings_base_path (most recent run)."""
    result = {}
    if not os.path.isdir(RUNS_ROOT):
        return result
    for run_name in sorted(os.listdir(RUNS_ROOT), reverse=True):
        vision_dir = os.path.join(RUNS_ROOT, run_name, "vision")
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


# ── Sidebar ───────────────────────────────────────────────────

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
        st.warning(f"⚑ {len(flagged_df)} flagged for review")

    st.markdown("---")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption("`data/human_labels.csv`")

# ── Main tabs ─────────────────────────────────────────────────

t_gen, t_label, t_stats, t_train, t_eval = st.tabs([
    "Generate Crops", "Label", "Statistics", "Train", "Evaluate"
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — Generate Crops
# ═══════════════════════════════════════════════════════════════

with t_gen:
    st.header("Crop Generation")

    # ── Config editor ──
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

    # ── Status grid ──
    st.subheader("Status")

    try:
        status_df = crop_status_grid()
    except Exception as e:
        st.error(f"Could not scan data directory: {e}")
        status_df = pd.DataFrame()

    if status_df.empty:
        st.info(
            "No subject data directories found under `data/`. "
            "Make sure your data is in `data/sj03/`, `data/sj04/`, etc."
        )
    else:
        def _status_icon(val):
            return "✅" if val else "❌"

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

    # ── Generate controls ──
    st.markdown("---")
    st.subheader("Generate Crops")

    if not status_df.empty:
        # Build selectable pairs: only those with video + fixations available
        _generable = status_df[status_df["has_video"] & status_df["has_fixations"]]
        _gen_options = [
            f"sj{int(r['subject_id']):02d} — {r['condition']}  ({int(r['n_crops'])} crops)"
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
                sj_str = sel.split(" — ")[0]
                cond_str = sel.split(" — ")[1].split("  (")[0]
                _pairs.append((int(sj_str[2:]), cond_str))

            with st.status(f"Generating crops for {len(_pairs)} pair(s)...", expanded=True) as _status:
                _prog = st.progress(0.0)
                _msg = st.empty()

                for _pi, (_sj, _cond) in enumerate(_pairs):
                    _pair_label = f"sj{_sj:02d}_{_cond}"
                    _msg.markdown(f"**{_pair_label}** — starting...")

                    def _crop_cb(phase, cur, tot, text, _pl=_pair_label, _idx=_pi, _total=len(_pairs)):
                        _overall = (_idx + cur / max(tot, 1)) / _total
                        _prog.progress(min(_overall, 1.0))
                        _msg.markdown(f"**{_pl}** — {phase}: {text}")

                    try:
                        _dir, _n = generate_crops_for_condition(
                            _sj, _cond,
                            world_video_dir=_wvd,
                            force=_force_regen,
                            progress_cb=_crop_cb,
                        )
                        _msg.markdown(f"**{_pair_label}** — {_n} crops")
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
        "+ `gaze_positions.csv` in `data/sj{N}/eye/{Condition}/`. "
        "EEG behavioral files are only needed for trial-level EEG fusion, "
        "not for crop generation or vision model training."
    )


# ═══════════════════════════════════════════════════════════════
# TAB 2 — Label
# ═══════════════════════════════════════════════════════════════

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
        f"sj{sj:02d}  {cond}  ({n} crops)" if n > 0 else f"sj{sj:02d}  {cond}  — no crops"
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
    all_pngs = sorted(
        f for f in os.listdir(get_crop_dir(sj_num, condition))
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
                         caption=f"sj{sj_num:02d} {condition} — fixation {fix_id}")
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

            if col_flag.button("⚑ Flag / Ambiguous", use_container_width=True,
                               help="Mark as ambiguous and move on — shown in review queue"):
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

    # ── Undo + review ───────────────────────────────────────────
    st.markdown("---")
    undo_col, _ = st.columns([1, 3])
    if undo_col.button("↩ Undo last label"):
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
                        flag_marker = " ⚑" if row.get("is_flagged") else ""
                        st.image(fpath, width=120)
                        st.markdown(
                            f"<span style='color:{color}; font-weight:600'>"
                            f"{row['human_label']}{flag_marker}</span>",
                            unsafe_allow_html=True,
                        )


# ═══════════════════════════════════════════════════════════════
# TAB 3 — Statistics
# ═══════════════════════════════════════════════════════════════

with t_stats:
    st.header("Label Statistics & Quality")

    trainable = _load_trainable()
    all_labels_df = _load_all_labels()

    if trainable.empty:
        st.info("No trainable labels yet. Start labeling in the Label tab.")
    else:
        # ── Class distribution ──────────────────────────────────
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

        # ── Per-subject/condition coverage ──────────────────────
        st.subheader("Coverage per subject × condition")

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

        # ── Flagged crops ────────────────────────────────────────
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
                with st.expander(f"sj{f_sj:02d} {f_cond} — {len(sub_flagged)} flagged"):
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

        # ── Inter-rater agreement ────────────────────────────────
        st.subheader("Inter-rater reliability")

        known_labelers = labeler_ids()
        if len(known_labelers) < 2:
            st.info(
                "Need labels from ≥2 labelers with overlapping crops to compute agreement. "
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
                        "κ < 0.4 = poor,  0.4–0.6 = moderate,  "
                        "0.6–0.8 = substantial,  > 0.8 = near-perfect"
                    )

                with st.expander("View disagreements"):
                    disagree = overlaps[~overlaps["agree"]]
                    if disagree.empty:
                        st.success("All overlapping labels agree!")
                    else:
                        st.dataframe(disagree, hide_index=True, use_container_width=True)

        # ── Download ─────────────────────────────────────────────
        st.markdown("---")
        if not all_labels_df.empty:
            st.download_button(
                "⬇ Download all labels CSV",
                all_labels_df.to_csv(index=False),
                file_name="human_labels.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════
# TAB 4 — Train
# ═══════════════════════════════════════════════════════════════

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

        # ── Configuration ────────────────────────────────────────
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
                    "Cross-subject: hold out one subject entirely for test — "
                    "measures generalisation to unseen people."
                ),
            )

            if split_strategy == "cross_subject":
                all_sjs = sorted(trainable_for_train["subject_id"].unique())
                if len(all_sjs) < 2:
                    st.warning("Cross-subject split requires labels from ≥2 subjects.")
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
            st.subheader("Models to Train")
            train_clip_flag = st.checkbox("CLIP + Linear Head", value=True)
            train_resnet_flag = st.checkbox(
                "ResNet-50 Fine-Tuned",
                value=False,
                help="Trains end-to-end on raw 224×224 crop PNGs. "
                     "Slower but can outperform CLIP head with enough labels.",
            )

            if train_clip_flag:
                clip_epochs = st.slider("CLIP head epochs", 100, 500, 200, 50)

            if train_resnet_flag:
                resnet_epochs = st.slider("ResNet epochs", 10, 60, 30, 5)
                resnet_batch = st.selectbox("Batch size", [16, 32, 64], index=1)

        st.markdown("---")
        os.makedirs(MODELS_DIR, exist_ok=True)

        # ── CLIP Head Training ───────────────────────────────────
        if train_clip_flag:
            st.subheader("CLIP Linear Head")

            emb_lookup = _find_all_embeddings()
            if not emb_lookup:
                st.warning(
                    "No CLIP embeddings found. Run the vision pipeline first "
                    "(phases 1–3 extract embeddings into `runs/`)."
                )
            else:
                n_emb_pairs = len(emb_lookup)
                st.info(
                    f"Embeddings available for {n_emb_pairs} subject/condition pair(s). "
                    "Training will pool all labeled crops that have matching embeddings."
                )

                if st.button("▶ Train CLIP Head", type="primary", key="btn_train_clip"):
                    from vision.train_head import train_with_holdout, save_head_versioned, save_head, _load_labeled_embeddings
                    import tempfile

                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    chart_placeholder = st.empty()

                    clip_history = []

                    def clip_cb(epoch, n_epochs, metrics):
                        progress_bar.progress(epoch / n_epochs)
                        status_text.markdown(
                            f"Epoch **{epoch}/{n_epochs}** — "
                            f"train loss: `{metrics['train_loss']}` — "
                            f"val acc: `{metrics.get('val_acc', '—')}`"
                        )
                        clip_history.append({"epoch": epoch, **metrics})
                        if len(clip_history) > 1:
                            hist_df = pd.DataFrame(clip_history).set_index("epoch")
                            chart_placeholder.line_chart(hist_df)

                    # Pool embeddings from all available pairs
                    X_all, y_all, sj_ids_all, fn_all = [], [], [], []
                    label_names = list(CATEGORIES.keys())
                    label_to_idx = {n: i for i, n in enumerate(label_names)}

                    for (sj, cond), emb_base in emb_lookup.items():
                        try:
                            subset = trainable_for_train[
                                (trainable_for_train["subject_id"] == sj)
                                & (trainable_for_train["condition"] == cond)
                            ]
                            if subset.empty:
                                continue
                            with tempfile.NamedTemporaryFile(
                                mode="w", suffix=".csv", delete=False
                            ) as tmp:
                                tmp_path = tmp.name
                                subset[["fixation_id", "human_label"]].to_csv(tmp_path, index=False)
                            X_sub, y_sub, _ = _load_labeled_embeddings(
                                tmp_path,
                                f"{emb_base}.npy",
                                f"{emb_base}_ids.csv",
                            )
                            os.unlink(tmp_path)
                            X_all.append(X_sub)
                            y_all.extend(y_sub)
                            sj_ids_all.extend([sj] * len(X_sub))
                            fnames_sub = subset[
                                subset["fixation_id"].isin(
                                    pd.read_csv(f"{emb_base}_ids.csv")["fixation_id"].astype(int).values
                                )
                            ]["filename"].tolist()
                            fn_all.extend(fnames_sub[:len(X_sub)])
                        except Exception as exc:
                            st.warning(f"Skipping sj{sj:02d} {cond}: {exc}")

                    if not X_all:
                        st.error("No embeddings matched any labels. Run the vision pipeline to extract embeddings.")
                    else:
                        X_pooled = np.vstack(X_all)
                        y_pooled = np.array(y_all)
                        sj_arr = np.array(sj_ids_all)

                        with st.spinner("Training…"):
                            model, stats, split = train_with_holdout(
                                X_pooled, y_pooled, label_names,
                                subject_ids=sj_arr if split_strategy == "cross_subject" else None,
                                strategy=split_strategy,
                                test_subject=test_sj,
                                n_epochs=clip_epochs,
                                progress_cb=clip_cb,
                            )

                        save_path = save_head_versioned(model, stats, MODELS_DIR, prefix="clip_head")
                        save_head(model, stats, os.path.join(MODELS_DIR, "clip_head.pt"))
                        progress_bar.progress(1.0)
                        status_text.empty()
                        st.success(
                            f"✅ CLIP head trained — "
                            f"val acc: **{stats['best_val_acc']:.1%}**  "
                            f"test acc: **{stats['test_acc']:.1%}**  "
                            f"→ `{os.path.basename(save_path)}`"
                        )
                        st.session_state.train_history_clip = clip_history
                        st.cache_data.clear()

        # ── ResNet Training ──────────────────────────────────────
        if train_resnet_flag:
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
                    if st.button("▶ Train ResNet-50", type="primary", key="btn_train_resnet"):
                        from vision.resnet_head import train_from_label_store as train_rn

                        progress_bar_rn = st.progress(0.0)
                        status_text_rn = st.empty()
                        chart_placeholder_rn = st.empty()
                        resnet_hist = []

                        def resnet_cb(epoch, n_epochs, metrics):
                            progress_bar_rn.progress(epoch / n_epochs)
                            status_text_rn.markdown(
                                f"Epoch **{epoch}/{n_epochs}** — "
                                f"train loss: `{metrics['train_loss']}` — "
                                f"val loss: `{metrics['val_loss']}` — "
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

                        with st.spinner("Training ResNet-50 (this may take several minutes)…"):
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
                                f"✅ ResNet-50 trained — "
                                f"best val acc: **{stats_rn['best_val_acc']:.1%}**  "
                                f"test acc: **{stats_rn.get('test_acc', '?'):.1%}**  "
                                f"→ `{os.path.basename(rn_path)}`"
                            )
                            st.session_state.train_history_resnet = resnet_hist
                        else:
                            st.error("Training failed — not enough data or missing crops.")
                        st.cache_data.clear()

        # ── Saved models ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("Saved models")

        from vision.train_head import list_saved_models
        saved = list_saved_models(MODELS_DIR)
        if not saved:
            st.info("No saved models yet.")
        else:
            rows = []
            for m in saved:
                rows.append({
                    "File": m["filename"],
                    "Type": m["model_type"],
                    "Saved": m["saved_at"][:16] if m["saved_at"] else "—",
                    "N samples": m["n_samples"],
                    "Val acc": f"{m['val_acc']:.3f}" if m["val_acc"] is not None else "—",
                    "Test acc": f"{m['test_acc']:.3f}" if m["test_acc"] is not None else "—",
                    "Split": m["split_strategy"],
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — Evaluate
# ═══════════════════════════════════════════════════════════════

with t_eval:
    st.header("Model Evaluation")

    from vision.train_head import list_saved_models, load_head
    saved_models = list_saved_models(MODELS_DIR)

    if not saved_models:
        st.info("No saved models yet. Train a model in the Train tab.")
    else:
        model_options = {m["filename"]: m for m in saved_models}
        selected_files = st.multiselect(
            "Select models to evaluate",
            list(model_options.keys()),
            default=[saved_models[0]["filename"]],
            format_func=lambda f: (
                f"{model_options[f]['model_type']}  |  "
                f"val={model_options[f]['val_acc']:.3f}  "
                f"test={model_options[f]['test_acc'] or '?'}  |  "
                f"{model_options[f]['saved_at'][:16]}"
            ),
        )

        if not selected_files:
            st.info("Select at least one model above.")
        else:
            # ── Summary comparison table ──────────────────────────
            st.subheader("Summary comparison")
            summary_rows = []
            for fname in selected_files:
                m = model_options[fname]
                summary_rows.append({
                    "Model": fname,
                    "Type": m["model_type"],
                    "Val acc": f"{m['val_acc']:.3f}" if m["val_acc"] else "—",
                    "Test acc": f"{m['test_acc']:.3f}" if m["test_acc"] else "—",
                    "Split": m["split_strategy"],
                    "N train": m["n_samples"],
                })
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

            # ── Per-model detail ──────────────────────────────────
            for fname in selected_files:
                m = model_options[fname]
                with st.expander(f"Details: {fname}", expanded=len(selected_files) == 1):
                    try:
                        ckpt = __import__("torch").load(
                            m["path"], map_location="cpu", weights_only=False
                        )
                        stats = ckpt.get("stats", {})
                        label_names = stats.get("label_names", LABEL_NAMES)
                        test_order = stats.get("test_label_order", label_names)

                        # Training history chart
                        history = stats.get("history", [])
                        if history:
                            hist_df = pd.DataFrame(history).set_index("epoch")
                            st.markdown("**Training history**")
                            st.line_chart(hist_df)

                        # Test classification report
                        test_report = stats.get("test_classification_report") or \
                                      stats.get("classification_report")
                        if test_report:
                            st.markdown("**Test-set classification report**")
                            report_rows = []
                            for cls_name, metrics in test_report.items():
                                if cls_name in ("accuracy", "macro avg", "weighted avg"):
                                    continue
                                if isinstance(metrics, dict):
                                    report_rows.append({
                                        "Class": cls_name,
                                        "Precision": f"{metrics.get('precision', 0):.3f}",
                                        "Recall": f"{metrics.get('recall', 0):.3f}",
                                        "F1": f"{metrics.get('f1-score', 0):.3f}",
                                        "Support": int(metrics.get("support", 0)),
                                    })
                            if report_rows:
                                report_df = pd.DataFrame(report_rows)
                                st.dataframe(report_df, hide_index=True, use_container_width=True)
                            macro = test_report.get("macro avg", {})
                            weighted = test_report.get("weighted avg", {})
                            if macro:
                                st.markdown(
                                    f"**Macro avg** — "
                                    f"precision: `{macro.get('precision',0):.3f}` "
                                    f"recall: `{macro.get('recall',0):.3f}` "
                                    f"F1: `{macro.get('f1-score',0):.3f}`"
                                )

                        # Confusion matrix
                        test_cm = stats.get("test_confusion_matrix")
                        if test_cm and test_order:
                            st.markdown("**Confusion matrix (test set)**")
                            try:
                                import plotly.figure_factory as ff
                                cm_arr = np.array(test_cm)
                                fig = ff.create_annotated_heatmap(
                                    z=cm_arr,
                                    x=test_order,
                                    y=test_order,
                                    colorscale="Blues",
                                    showscale=True,
                                )
                                fig.update_layout(
                                    xaxis_title="Predicted",
                                    yaxis_title="True",
                                    margin=dict(t=30, b=30),
                                    height=400,
                                )
                                fig.update_xaxes(side="bottom")
                                st.plotly_chart(fig, use_container_width=True)
                            except ImportError:
                                st.dataframe(
                                    pd.DataFrame(test_cm, index=test_order, columns=test_order),
                                    use_container_width=True,
                                )

                        # Show test sample crops if filenames were saved
                        test_fnames = ckpt.get("test_filenames", [])
                        test_true = ckpt.get("test_labels", [])
                        if test_fnames and test_true:
                            st.markdown("**Test set sample crops**")
                            show_n = min(10, len(test_fnames))
                            crop_cols = st.columns(5)
                            for i in range(show_n):
                                fn = test_fnames[i]
                                lbl = test_true[i] if i < len(test_true) else "?"
                                # Try to find which sj/cond this filename belongs to
                                found = False
                                for sj, cond, _ in _get_all_pairs_with_crops():
                                    fpath = get_crop_path(sj, cond, fn)
                                    if os.path.exists(fpath):
                                        with crop_cols[i % 5]:
                                            st.image(fpath, width=110)
                                            color = CATEGORY_COLORS.get(lbl, "#888")
                                            st.markdown(
                                                f"<span style='color:{color}; font-size:11px'>"
                                                f"{lbl}</span>",
                                                unsafe_allow_html=True,
                                            )
                                        found = True
                                        break

                    except Exception as exc:
                        st.error(f"Could not load checkpoint: {exc}")
