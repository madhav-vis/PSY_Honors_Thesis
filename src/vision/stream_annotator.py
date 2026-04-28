"""Streamlit-based crop annotator for training a CLIP classification head.

Run:  streamlit run src/vision/stream_annotator.py

Labels are written to the central store:  data/human_labels.csv
Crop images are read from stable storage:  data/crops/sj{sj:02d}_{condition}/
"""

import os
import sys

import pandas as pd
import streamlit as st

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.config import CATEGORIES, CATEGORY_COLORS
from vision.label_store import (
    PROJECT_ROOT,
    append_label,
    available_subjects_conditions,
    get_crop_dir,
    get_crop_path,
    label_counts,
    load_labels,
    load_labels_for,
    migrate_existing_labels,
    remove_last_label,
    subject_condition_counts,
)

LABEL_NAMES = list(CATEGORIES.keys())
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

st.set_page_config(
    page_title="Crop Annotator",
    page_icon="T",
    layout="wide",
)


# ── Helpers ───────────────────────────────────────────────────

def _find_embeddings(sj_num: int, condition: str):
    """Return embeddings base path (without extension) if both files exist, else None."""
    if not os.path.isdir(RUNS_ROOT):
        return None
    for run_name in sorted(os.listdir(RUNS_ROOT), reverse=True):
        sj_cond = f"sj{sj_num:02d}_{condition}"
        vision_dir = os.path.join(RUNS_ROOT, run_name, "vision", sj_cond)
        base = os.path.join(vision_dir, f"{sj_cond}_embeddings")
        if os.path.exists(f"{base}.npy") and os.path.exists(f"{base}_ids.csv"):
            return base
    return None


# ── Auto-migrate on first load ────────────────────────────────

@st.cache_data(show_spinner=False)
def _run_migration():
    return migrate_existing_labels(RUNS_ROOT)


migrated = _run_migration()
if migrated > 0:
    st.toast(f"Migrated {migrated} existing labels into central store.", icon="✅")


# ── Sidebar: subject + condition selector ─────────────────────

st.sidebar.title("Crop Annotator")
st.sidebar.caption("Labels → `data/human_labels.csv`")

pairs = available_subjects_conditions()

if not pairs:
    st.error(
        "No crops found in `data/crops/`. "
        "Run the vision pipeline first — it will mirror crops to stable storage."
    )
    st.stop()

pair_labels = [f"sj{sj:02d}  {cond}" for sj, cond in pairs]
selected_idx = st.sidebar.selectbox(
    "Subject / Condition",
    range(len(pairs)),
    format_func=lambda i: pair_labels[i],
)
sj_num, condition = pairs[selected_idx]

all_pngs = sorted(
    f for f in os.listdir(get_crop_dir(sj_num, condition))
    if f.lower().endswith(".png")
)

# ── Load labels for this subject/condition ────────────────────

subject_labels_df = load_labels_for(sj_num, condition)
labeled_set = set(subject_labels_df["filename"].values)
unlabeled = [f for f in all_pngs if f not in labeled_set]
n_labeled = len(labeled_set)

# ── Sidebar: per-category progress ───────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("This subject / condition")

per_cat = (
    subject_labels_df["human_label"].value_counts()
    if not subject_labels_df.empty
    else pd.Series(dtype=int)
)
for cat in LABEL_NAMES:
    cnt = int(per_cat.get(cat, 0))
    color = CATEGORY_COLORS.get(cat, "#888")
    st.sidebar.markdown(
        f"<span style='color:{color}; font-weight:600'>{cat}</span>: **{cnt}**",
        unsafe_allow_html=True,
    )

target_per_cat = st.sidebar.number_input(
    "Target labels per category", value=50, min_value=5, step=10,
)
target_total = target_per_cat * len(LABEL_NAMES)
pct = min(n_labeled / target_total, 1.0) if target_total > 0 else 0.0
st.sidebar.progress(pct, text=f"{n_labeled} / {target_total} labels")

# ── Sidebar: global cross-subject summary ────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("All subjects (global)")

all_labels = load_labels()
total_all = len(all_labels)
st.sidebar.metric("Total labels", total_all)

sj_cond_counts = subject_condition_counts()
if not sj_cond_counts.empty:
    st.sidebar.dataframe(
        sj_cond_counts.rename(
            columns={"subject_id": "sj", "condition": "cond", "count": "n"}
        ),
        hide_index=True,
        use_container_width=True,
        height=min(200, 35 + 35 * len(sj_cond_counts)),
    )

# ── Sidebar: train models ─────────────────────────────────────

st.sidebar.markdown("---")

emb_base = _find_embeddings(sj_num, condition)
has_embeddings = emb_base is not None

if has_embeddings and n_labeled >= 10:
    os.makedirs(MODELS_DIR, exist_ok=True)
    head_pt = os.path.join(MODELS_DIR, "clip_head.pt")

    if st.sidebar.button("Train CLIP head (this sj/cond)", type="primary"):
        with st.sidebar:
            with st.spinner("Training..."):
                from vision.train_head import train_from_label_store
                stats = train_from_label_store(
                    sj_num=sj_num,
                    condition=condition,
                    embeddings_base=emb_base,
                    out_model_path=head_pt,
                )
                if stats:
                    st.success(
                        f"Done — acc: {stats['train_acc']:.1%} "
                        f"({stats['n_samples']} samples)"
                    )
                else:
                    st.error("Not enough matched labels.")

    if total_all >= 50 and st.sidebar.button("Train CLIP head (all subjects pooled)"):
        with st.sidebar:
            with st.spinner("Training pooled..."):
                from vision.train_head import train_from_label_store
                stats = train_from_label_store(
                    sj_num=None,
                    condition=None,
                    embeddings_base=emb_base,
                    out_model_path=head_pt,
                )
                if stats:
                    st.success(
                        f"Pooled — acc: {stats['train_acc']:.1%} "
                        f"({stats['n_samples']} samples)"
                    )

    if os.path.exists(head_pt):
        st.sidebar.success("Head exists: `clip_head.pt`")

elif not has_embeddings:
    st.sidebar.info("Run vision pipeline to extract embeddings first.")
else:
    st.sidebar.info(f"Need ≥10 labels to train (have {n_labeled}).")

st.sidebar.markdown("---")
if not all_labels.empty and st.sidebar.button("Download all labels CSV"):
    st.sidebar.download_button(
        "Save CSV",
        all_labels.to_csv(index=False),
        file_name="human_labels.csv",
        mime="text/csv",
    )

# ── Main area ─────────────────────────────────────────────────

st.title("Gaze Crop Annotator")

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

if not unlabeled:
    st.success("All crops for this subject/condition have been labeled!")
    st.stop()

# Reset index when switching subject/condition
if (
    "crop_idx" not in st.session_state
    or st.session_state.get("last_pair") != (sj_num, condition)
):
    st.session_state.crop_idx = 0
    st.session_state.last_pair = (sj_num, condition)

idx = st.session_state.crop_idx
if idx >= len(unlabeled):
    st.success("All crops for this subject/condition have been labeled!")
    st.stop()

fname = unlabeled[idx]
parts = fname.replace(".png", "").split("_")
fix_id = parts[0] if parts else fname
ts_ns = parts[1] if len(parts) > 1 else ""

img_path = get_crop_path(sj_num, condition, fname)

col_img, col_ctrl = st.columns([2, 1])

with col_img:
    if os.path.exists(img_path):
        st.image(
            img_path, width=448,
            caption=f"sj{sj_num:02d} {condition} — fixation {fix_id}",
        )
    else:
        st.warning(f"Image not found: {img_path}")

with col_ctrl:
    st.markdown(f"**Crop {idx + 1}** of {len(unlabeled)} remaining")
    st.markdown(f"Fixation ID: `{fix_id}`  |  Subject: `sj{sj_num:02d}`")

    chosen = None
    btn_cols = st.columns(2)
    for i, name in enumerate(LABEL_NAMES):
        if btn_cols[i % 2].button(
            f"{i + 1}  {name}",
            key=f"btn_{name}",
            use_container_width=True,
        ):
            chosen = name

    st.markdown("---")
    if st.button("Skip", use_container_width=True):
        st.session_state.crop_idx += 1
        st.rerun()

if chosen:
    append_label(sj_num, condition, fix_id, ts_ns, fname, chosen)
    st.session_state.crop_idx += 1
    st.rerun()

# ── Review mode ───────────────────────────────────────────────

st.markdown("---")
with st.expander("Review labeled crops", expanded=False):
    if subject_labels_df.empty:
        st.info("No labels yet for this subject/condition.")
    else:
        filter_cat = st.selectbox("Filter category", ["all"] + LABEL_NAMES)
        show_df = (
            subject_labels_df if filter_cat == "all"
            else subject_labels_df[subject_labels_df["human_label"] == filter_cat]
        )
        sample = show_df.tail(min(20, len(show_df)))
        cols = st.columns(5)
        for i, (_, row) in enumerate(sample.iterrows()):
            fpath = get_crop_path(sj_num, condition, row["filename"])
            if os.path.exists(fpath):
                with cols[i % 5]:
                    color = CATEGORY_COLORS.get(row["human_label"], "#888")
                    st.image(fpath, width=120)
                    st.markdown(
                        f"<span style='color:{color}; font-weight:600'>"
                        f"{row['human_label']}</span>",
                        unsafe_allow_html=True,
                    )

        if st.button("Undo last label"):
            if remove_last_label(sj_num, condition):
                if st.session_state.crop_idx > 0:
                    st.session_state.crop_idx -= 1
                st.rerun()
