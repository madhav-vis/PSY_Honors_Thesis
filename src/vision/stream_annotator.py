"""Streamlit-based crop annotator for training a CLIP classification head.

Run:  streamlit run src/vision/stream_annotator.py
"""

import base64
import os
import random
import sys

import pandas as pd
import streamlit as st

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from vision.config import CATEGORIES, CATEGORY_COLORS

PROJECT_ROOT = os.path.dirname(_SRC_DIR)
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")

LABEL_NAMES = list(CATEGORIES.keys())
LABEL_KEYBINDS = {str(i + 1): name for i, name in enumerate(LABEL_NAMES)}

st.set_page_config(
    page_title="Crop Annotator",
    page_icon="T",
    layout="wide",
)


# ── Helpers ───────────────────────────────────────────────────

def _find_crops_dirs():
    """Scan runs/ for directories containing crop PNGs."""
    dirs = []
    if not os.path.isdir(RUNS_ROOT):
        return dirs
    for run_name in sorted(os.listdir(RUNS_ROOT), reverse=True):
        vision_root = os.path.join(RUNS_ROOT, run_name, "vision")
        if not os.path.isdir(vision_root):
            continue
        for sj_cond in sorted(os.listdir(vision_root)):
            crops_dir = os.path.join(vision_root, sj_cond, "crops")
            if os.path.isdir(crops_dir):
                pngs = [f for f in os.listdir(crops_dir)
                        if f.lower().endswith(".png")]
                if pngs:
                    dirs.append({
                        "label": f"{run_name} / {sj_cond}",
                        "path": crops_dir,
                        "n_crops": len(pngs),
                    })
    return dirs


def _load_or_init_labels(csv_path):
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=["fixation_id", "timestamp_ns",
                                  "filename", "human_label"])


def _save_labels(df, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)


def _img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── Sidebar: select crops directory ──────────────────────────

st.sidebar.title("Crop Annotator")
st.sidebar.caption("Label gaze crops for CLIP head training")

crops_dirs = _find_crops_dirs()
if not crops_dirs:
    st.error("No crop directories found under runs/. "
             "Run the vision pipeline first.")
    st.stop()

selected_dir = st.sidebar.selectbox(
    "Crops directory",
    range(len(crops_dirs)),
    format_func=lambda i: f"{crops_dirs[i]['label']}  ({crops_dirs[i]['n_crops']} crops)",
)
crops_info = crops_dirs[selected_dir]
crops_dir = crops_info["path"]

vision_dir = os.path.dirname(crops_dir)
sj_cond = os.path.basename(vision_dir)
csv_path = os.path.join(vision_dir, f"{sj_cond}_human_labels.csv")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Output CSV**  \n`{csv_path}`")

# ── Load state ───────────────────────────────────────────────

all_pngs = sorted(f for f in os.listdir(crops_dir)
                   if f.lower().endswith(".png"))

labels_df = _load_or_init_labels(csv_path)
labeled_set = set(labels_df["filename"].values)
unlabeled = [f for f in all_pngs if f not in labeled_set]

random.seed(42)
random.shuffle(unlabeled)

n_labeled = len(labeled_set)
n_total = len(all_pngs)

# ── Progress ─────────────────────────────────────────────────

st.sidebar.markdown("---")
st.sidebar.subheader("Progress")

per_cat = labels_df["human_label"].value_counts()
for cat in LABEL_NAMES:
    cnt = int(per_cat.get(cat, 0))
    color = CATEGORY_COLORS.get(cat, "#888")
    st.sidebar.markdown(
        f"<span style='color:{color}; font-weight:600'>{cat}</span>: "
        f"**{cnt}**",
        unsafe_allow_html=True,
    )

target_per_cat = st.sidebar.number_input(
    "Target labels per category", value=50, min_value=5, step=10,
)
target_total = target_per_cat * len(LABEL_NAMES)

pct = min(n_labeled / target_total, 1.0) if target_total > 0 else 0
st.sidebar.progress(pct, text=f"{n_labeled} / {target_total} labels")

cats_present = [cat for cat in LABEL_NAMES if per_cat.get(cat, 0) > 0]
cats_missing = [cat for cat in LABEL_NAMES if per_cat.get(cat, 0) == 0]
min_cat_count = int(min(per_cat.get(cat, 0) for cat in LABEL_NAMES))
cats_done = sum(1 for cat in LABEL_NAMES
                if per_cat.get(cat, 0) >= target_per_cat)

if cats_done == len(LABEL_NAMES):
    st.sidebar.success("All categories have enough labels!")
elif n_labeled >= 10 and len(cats_present) >= 2:
    note = ""
    if cats_missing:
        note = f" Missing: {', '.join(cats_missing)}."
    st.sidebar.info(
        f"You can train now ({len(cats_present)} categories, "
        f"smallest has {min_cat_count}).{note} "
        f"More labels = better accuracy."
    )

st.sidebar.markdown("---")

# Train head directly from annotator
emb_npy = os.path.join(vision_dir, f"{sj_cond}_embeddings.npy")
emb_ids = os.path.join(vision_dir, f"{sj_cond}_embeddings_ids.csv")
_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "models")
os.makedirs(_models_dir, exist_ok=True)
head_pt = os.path.join(_models_dir, "clip_head.pt")
has_embeddings = os.path.exists(emb_npy) and os.path.exists(emb_ids)

if has_embeddings and n_labeled >= 10:
    if st.sidebar.button("Train classification head", type="primary"):
        with st.sidebar:
            with st.spinner("Training linear head..."):
                from vision.train_head import train_from_files
                stats = train_from_files(
                    labels_csv=csv_path,
                    embeddings_npy=emb_npy,
                    embeddings_ids_csv=emb_ids,
                    out_model_path=head_pt,
                )
                if stats:
                    st.success(
                        f"Trained! Accuracy: {stats['train_acc']:.1%} "
                        f"({stats['n_samples']} samples)"
                    )
                else:
                    st.error("Not enough matched labels to train.")

    if os.path.exists(head_pt):
        st.sidebar.success(f"Trained head exists: `{os.path.basename(head_pt)}`")
elif not has_embeddings:
    st.sidebar.info("Run the vision pipeline first to extract embeddings.")
else:
    st.sidebar.info(f"Need at least 10 labels to train (have {n_labeled}).")

st.sidebar.markdown("---")
if st.sidebar.button("Download labels CSV"):
    st.sidebar.download_button(
        "Save CSV",
        labels_df.to_csv(index=False),
        file_name=f"{sj_cond}_human_labels.csv",
        mime="text/csv",
    )

# ── Main area ────────────────────────────────────────────────

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
    st.success("All crops in this directory have been labeled!")
    st.stop()

# Session-state index into the unlabeled list
if "crop_idx" not in st.session_state:
    st.session_state.crop_idx = 0

idx = st.session_state.crop_idx
if idx >= len(unlabeled):
    st.success("All crops in this directory have been labeled!")
    st.stop()

fname = unlabeled[idx]
parts = fname.replace(".png", "").split("_")
fix_id = parts[0] if parts else fname
ts_ns = parts[1] if len(parts) > 1 else ""

img_path = os.path.join(crops_dir, fname)

col_img, col_ctrl = st.columns([2, 1])

with col_img:
    st.image(img_path, width=448, caption=f"Fixation {fix_id}")

with col_ctrl:
    st.markdown(f"**Crop {idx + 1}** of {len(unlabeled)} remaining")
    st.markdown(f"Fixation ID: `{fix_id}`")

    chosen = None
    btn_cols = st.columns(2)
    for i, name in enumerate(LABEL_NAMES):
        color = CATEGORY_COLORS.get(name, "#888")
        col = btn_cols[i % 2]
        if col.button(
            f"{i+1}  {name}",
            key=f"btn_{name}",
            use_container_width=True,
        ):
            chosen = name

    st.markdown("---")
    if st.button("Skip", use_container_width=True):
        st.session_state.crop_idx += 1
        st.rerun()

if chosen:
    new_row = pd.DataFrame([{
        "fixation_id": fix_id,
        "timestamp_ns": ts_ns,
        "filename": fname,
        "human_label": chosen,
    }])
    labels_df = pd.concat([labels_df, new_row], ignore_index=True)
    _save_labels(labels_df, csv_path)
    st.session_state.crop_idx += 1
    st.rerun()

# ── Review mode ──────────────────────────────────────────────

st.markdown("---")
with st.expander("Review labeled crops", expanded=False):
    if labels_df.empty:
        st.info("No labels yet.")
    else:
        filter_cat = st.selectbox("Filter category", ["all"] + LABEL_NAMES)
        show_df = labels_df if filter_cat == "all" else labels_df[
            labels_df["human_label"] == filter_cat
        ]

        n_show = min(20, len(show_df))
        sample = show_df.tail(n_show)

        cols = st.columns(5)
        for i, (_, row) in enumerate(sample.iterrows()):
            fpath = os.path.join(crops_dir, row["filename"])
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
            if not labels_df.empty:
                removed = labels_df.iloc[-1]
                labels_df = labels_df.iloc[:-1]
                _save_labels(labels_df, csv_path)
                if st.session_state.crop_idx > 0:
                    st.session_state.crop_idx -= 1
                st.rerun()
