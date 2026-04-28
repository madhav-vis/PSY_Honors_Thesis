"""Central label store for human-annotated gaze crops.

All labels from all subjects/conditions live in one CSV:
  {PROJECT_ROOT}/data/human_labels.csv

Crop images are mirrored to a stable directory that survives pipeline re-runs:
  {PROJECT_ROOT}/data/crops/sj{sj:02d}_{condition}/

This decouples labels from run directories, which get wiped on re-runs.
"""

import os
import shutil
from datetime import datetime

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(_SRC_DIR)

LABELS_CSV = os.path.join(PROJECT_ROOT, "data", "human_labels.csv")
CROPS_BASE = os.path.join(PROJECT_ROOT, "data", "crops")

LABEL_COLUMNS = [
    "subject_id", "condition", "fixation_id", "timestamp_ns",
    "filename", "human_label", "labeled_at",
]


# ── Read / Write ──────────────────────────────────────────────

def load_labels() -> pd.DataFrame:
    """Load all labels. Returns empty DataFrame if file not found."""
    if os.path.exists(LABELS_CSV):
        df = pd.read_csv(LABELS_CSV)
        df["subject_id"] = df["subject_id"].astype(int)
        return df
    return pd.DataFrame(columns=LABEL_COLUMNS)


def save_labels(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)
    df.to_csv(LABELS_CSV, index=False)


def load_labels_for(sj_num: int, condition: str) -> pd.DataFrame:
    """Return labels filtered to one subject + condition."""
    df = load_labels()
    if df.empty:
        return df
    return df[
        (df["subject_id"] == int(sj_num)) & (df["condition"] == condition)
    ].reset_index(drop=True)


def append_label(sj_num: int, condition: str, fixation_id,
                 timestamp_ns, filename: str, human_label: str) -> None:
    """Append one label row (load → concat → save)."""
    df = load_labels()
    new_row = pd.DataFrame([{
        "subject_id": int(sj_num),
        "condition": condition,
        "fixation_id": int(fixation_id),
        "timestamp_ns": int(timestamp_ns) if timestamp_ns else 0,
        "filename": filename,
        "human_label": human_label,
        "labeled_at": datetime.now().isoformat(timespec="seconds"),
    }])
    save_labels(pd.concat([df, new_row], ignore_index=True))


def remove_last_label(sj_num: int, condition: str) -> bool:
    """Remove most recent label for a subject/condition. Returns True if removed."""
    df = load_labels()
    mask = (df["subject_id"] == int(sj_num)) & (df["condition"] == condition)
    indices = df[mask].index
    if len(indices) == 0:
        return False
    save_labels(df.drop(indices[-1]).reset_index(drop=True))
    return True


def label_counts() -> pd.DataFrame:
    """Per-category counts across all subjects. Returns DataFrame[category, count]."""
    df = load_labels()
    if df.empty:
        return pd.DataFrame(columns=["human_label", "count"])
    return (
        df["human_label"].value_counts()
        .rename_axis("human_label")
        .reset_index(name="count")
    )


def subject_condition_counts() -> pd.DataFrame:
    """Label counts grouped by subject + condition."""
    df = load_labels()
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "condition", "count"])
    return (
        df.groupby(["subject_id", "condition"])
        .size()
        .reset_index(name="count")
        .sort_values(["subject_id", "condition"])
    )


# ── Crop mirroring ────────────────────────────────────────────

def get_crop_dir(sj_num: int, condition: str) -> str:
    """Stable directory for a subject+condition's crop PNGs."""
    return os.path.join(CROPS_BASE, f"sj{sj_num:02d}_{condition}")


def get_crop_path(sj_num: int, condition: str, filename: str) -> str:
    return os.path.join(get_crop_dir(sj_num, condition), filename)


def mirror_crop(src_path: str, sj_num: int, condition: str,
                filename: str) -> str:
    """Copy one crop PNG to stable storage. Returns destination path."""
    dest_dir = get_crop_dir(sj_num, condition)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if not os.path.exists(dest):
        shutil.copy2(src_path, dest)
    return dest


def mirror_crops_dir(crops_dir: str, sj_num: int, condition: str) -> int:
    """Copy all PNGs from a run crops dir to stable storage. Returns count copied."""
    dest_dir = get_crop_dir(sj_num, condition)
    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    for fname in os.listdir(crops_dir):
        if fname.lower().endswith(".png"):
            dst = os.path.join(dest_dir, fname)
            if not os.path.exists(dst):
                shutil.copy2(os.path.join(crops_dir, fname), dst)
                n += 1
    return n


def available_subjects_conditions() -> list[tuple[int, str]]:
    """List (sj_num, condition) pairs that have crops in stable storage."""
    if not os.path.isdir(CROPS_BASE):
        return []
    pairs = []
    for name in sorted(os.listdir(CROPS_BASE)):
        path = os.path.join(CROPS_BASE, name)
        if not os.path.isdir(path):
            continue
        parts = name.split("_", 1)
        if len(parts) != 2 or not parts[0].startswith("sj"):
            continue
        try:
            sj_num = int(parts[0][2:])
        except ValueError:
            continue
        # Only include if there are actually PNGs
        if any(f.lower().endswith(".png") for f in os.listdir(path)):
            pairs.append((sj_num, parts[1]))
    return pairs


# ── Migration ─────────────────────────────────────────────────

def migrate_existing_labels(runs_root: str) -> int:
    """Scan runs/ for old per-subject label CSVs and merge into central store.

    Also mirrors crop PNGs to stable data/crops/ storage.
    Returns number of new labels migrated.
    """
    if not os.path.isdir(runs_root):
        return 0

    existing = load_labels()
    existing_keys: set[tuple] = set()
    if not existing.empty:
        existing_keys = set(
            zip(existing["subject_id"].astype(int),
                existing["condition"],
                existing["filename"])
        )

    new_rows = []
    for run_name in sorted(os.listdir(runs_root)):
        vision_root = os.path.join(runs_root, run_name, "vision")
        if not os.path.isdir(vision_root):
            continue
        for sj_cond in sorted(os.listdir(vision_root)):
            # Parse "sj04_walk_attend" → sj_num=4, condition="walk_attend"
            parts = sj_cond.split("_", 1)
            if len(parts) != 2 or not parts[0].startswith("sj"):
                continue
            try:
                sj_num = int(parts[0][2:])
            except ValueError:
                continue
            condition = parts[1]

            old_csv = os.path.join(
                vision_root, sj_cond, f"{sj_cond}_human_labels.csv"
            )
            if not os.path.exists(old_csv):
                continue

            mtime = datetime.fromtimestamp(
                os.path.getmtime(old_csv)
            ).isoformat(timespec="seconds")

            crops_dir = os.path.join(vision_root, sj_cond, "crops")
            if os.path.isdir(crops_dir):
                mirror_crops_dir(crops_dir, sj_num, condition)

            old_df = pd.read_csv(old_csv)
            for _, row in old_df.iterrows():
                fname = str(row.get("filename", ""))
                key = (sj_num, condition, fname)
                if key in existing_keys:
                    continue
                new_rows.append({
                    "subject_id": sj_num,
                    "condition": condition,
                    "fixation_id": row.get("fixation_id", ""),
                    "timestamp_ns": row.get("timestamp_ns", 0),
                    "filename": fname,
                    "human_label": str(row.get("human_label", "")),
                    "labeled_at": mtime,
                })
                existing_keys.add(key)

    if new_rows:
        combined = pd.concat(
            [existing, pd.DataFrame(new_rows)], ignore_index=True
        )
        save_labels(combined)

    return len(new_rows)
