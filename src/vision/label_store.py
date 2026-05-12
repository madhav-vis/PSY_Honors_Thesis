"""Central label store for human-annotated gaze crops.

All labels from all subjects/conditions live in one CSV:
  {PROJECT_ROOT}/data/human_labels.csv

Crop images are mirrored to a stable directory that survives pipeline re-runs:
  {PROJECT_ROOT}/data/crops/sj{sj:02d}_{condition}/

Schema (v2): subject_id, condition, fixation_id, timestamp_ns,
             filename, human_label, labeled_at, labeler_id, is_flagged
"""

import os
import shutil
from datetime import datetime

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_RUN_CONFIG_PATH = os.path.join(PROJECT_ROOT, "src", "run_config.yaml")

LABELS_CSV = os.path.join(PROJECT_ROOT, "data", "human_labels.csv")
CROPS_BASE = os.path.join(PROJECT_ROOT, "data", "crops")

LABEL_COLUMNS = [
    "subject_id", "condition", "fixation_id", "timestamp_ns",
    "filename", "human_label", "labeled_at", "labeler_id", "is_flagged",
]

# Crops with this label are excluded from model training
FLAG_LABEL = "flagged"


# ── Schema migration ──────────────────────────────────────────

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Add v2 columns with defaults for rows written by older code."""
    if df.empty:
        return df
    changed = False
    if "labeler_id" not in df.columns:
        df = df.copy()
        df["labeler_id"] = ""
        changed = True
    if "is_flagged" not in df.columns:
        if not changed:
            df = df.copy()
        df["is_flagged"] = False
    df["is_flagged"] = df["is_flagged"].fillna(False).astype(bool)
    df["labeler_id"] = df["labeler_id"].fillna("").astype(str)
    return df


# ── Read / Write ──────────────────────────────────────────────

def load_labels() -> pd.DataFrame:
    """Load all labels. Returns empty DataFrame if file not found."""
    if os.path.exists(LABELS_CSV):
        df = pd.read_csv(LABELS_CSV)
        df["subject_id"] = df["subject_id"].astype(int)
        return _ensure_schema(df)
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


def load_trainable_labels() -> pd.DataFrame:
    """Return all labels suitable for training (exclude flagged / empty labels)."""
    df = load_labels()
    if df.empty:
        return df
    return df[
        (~df["is_flagged"]) & (df["human_label"].notna()) & (df["human_label"] != "")
        & (df["human_label"] != FLAG_LABEL)
    ].reset_index(drop=True)


def load_flagged() -> pd.DataFrame:
    """Return all flagged / ambiguous labels awaiting review."""
    df = load_labels()
    if df.empty:
        return df
    return df[df["is_flagged"] | (df["human_label"] == FLAG_LABEL)].reset_index(drop=True)


def append_label(
    sj_num: int,
    condition: str,
    fixation_id,
    timestamp_ns,
    filename: str,
    human_label: str,
    labeler_id: str = "",
    is_flagged: bool = False,
) -> None:
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
        "labeler_id": str(labeler_id).strip(),
        "is_flagged": bool(is_flagged),
    }])
    save_labels(pd.concat([df, new_row], ignore_index=True))


def remove_last_label(sj_num: int, condition: str, labeler_id: str = "") -> bool:
    """Remove most recent label for a subject/condition (+optional labeler). Returns True if removed."""
    df = load_labels()
    mask = (df["subject_id"] == int(sj_num)) & (df["condition"] == condition)
    if labeler_id:
        mask &= df["labeler_id"] == labeler_id
    indices = df[mask].index
    if len(indices) == 0:
        return False
    save_labels(df.drop(indices[-1]).reset_index(drop=True))
    return True


def relabel(sj_num: int, condition: str, filename: str,
            new_label: str, labeler_id: str = "") -> bool:
    """Update label for a specific crop (most recent row matching filename + labeler)."""
    df = load_labels()
    mask = (
        (df["subject_id"] == int(sj_num))
        & (df["condition"] == condition)
        & (df["filename"] == filename)
    )
    if labeler_id:
        mask &= df["labeler_id"] == labeler_id
    indices = df[mask].index
    if len(indices) == 0:
        return False
    df.loc[indices[-1], "human_label"] = new_label
    df.loc[indices[-1], "is_flagged"] = False
    df.loc[indices[-1], "labeled_at"] = datetime.now().isoformat(timespec="seconds")
    save_labels(df)
    return True


# ── Counts / summaries ────────────────────────────────────────

def label_counts() -> pd.DataFrame:
    """Per-category counts across all subjects (trainable labels only)."""
    df = load_trainable_labels()
    if df.empty:
        return pd.DataFrame(columns=["human_label", "count"])
    return (
        df["human_label"].value_counts()
        .rename_axis("human_label")
        .reset_index(name="count")
    )


def subject_condition_counts() -> pd.DataFrame:
    """Label counts grouped by subject + condition (trainable labels only)."""
    df = load_trainable_labels()
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "condition", "count"])
    return (
        df.groupby(["subject_id", "condition"])
        .size()
        .reset_index(name="count")
        .sort_values(["subject_id", "condition"])
    )


def labeler_ids() -> list[str]:
    """Return sorted list of unique non-empty labeler IDs."""
    df = load_labels()
    if df.empty:
        return []
    ids = df["labeler_id"].dropna().astype(str).str.strip()
    return sorted(i for i in ids.unique() if i)


# ── Inter-rater reliability ───────────────────────────────────

def inter_rater_overlaps() -> pd.DataFrame:
    """Return crops labeled by 2+ distinct labelers, paired for comparison.

    Columns: subject_id, condition, filename, labeler_1, label_1, labeler_2, label_2, agree
    """
    df = load_labels()
    if df.empty:
        return pd.DataFrame()

    df = df[df["labeler_id"].str.strip() != ""].copy()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (sj, cond, fname), group in df.groupby(["subject_id", "condition", "filename"]):
        unique_labelers = group["labeler_id"].unique()
        if len(unique_labelers) < 2:
            continue
        sorted_group = group.sort_values("labeler_id")
        labeler_list = list(sorted_group.iterrows())
        for i in range(len(labeler_list)):
            for j in range(i + 1, len(labeler_list)):
                _, r1 = labeler_list[i]
                _, r2 = labeler_list[j]
                rows.append({
                    "subject_id": sj,
                    "condition": cond,
                    "filename": fname,
                    "labeler_1": r1["labeler_id"],
                    "label_1": r1["human_label"],
                    "labeler_2": r2["labeler_id"],
                    "label_2": r2["human_label"],
                    "agree": r1["human_label"] == r2["human_label"],
                })
    return pd.DataFrame(rows)


def cohens_kappa_matrix() -> pd.DataFrame:
    """Compute Cohen's Kappa between every pair of labelers with overlapping annotations.

    Returns DataFrame: labeler_1, labeler_2, kappa, n_overlap, pct_agree
    """
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        return pd.DataFrame(columns=["labeler_1", "labeler_2", "kappa", "n_overlap"])

    overlaps = inter_rater_overlaps()
    if overlaps.empty:
        return pd.DataFrame(columns=["labeler_1", "labeler_2", "kappa", "n_overlap", "pct_agree"])

    rows = []
    for (l1, l2), group in overlaps.groupby(["labeler_1", "labeler_2"]):
        if len(group) < 2:
            continue
        try:
            kappa = cohen_kappa_score(group["label_1"], group["label_2"])
        except Exception:
            kappa = float("nan")
        pct = group["agree"].mean()
        rows.append({
            "labeler_1": l1,
            "labeler_2": l2,
            "kappa": round(kappa, 3),
            "n_overlap": len(group),
            "pct_agree": round(pct, 3),
        })
    return pd.DataFrame(rows)


# ── Data directory scanning ───────────────────────────────────

def _load_run_config() -> dict:
    """Parse src/run_config.yaml; returns empty dict if missing or invalid."""
    if not os.path.isfile(_RUN_CONFIG_PATH):
        return {}
    try:
        import yaml
        with open(_RUN_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def data_root_from_config() -> str:
    """Return the absolute path that contains sj* folders, per run_config.yaml data.root."""
    cfg = _load_run_config()
    data_cfg = cfg.get("data", {})
    root = data_cfg.get("root", "data") if isinstance(data_cfg, dict) else "data"
    # os.path.join ignores PROJECT_ROOT when root is already absolute (e.g. C:/...)
    return os.path.normpath(os.path.join(PROJECT_ROOT, root))


def scan_data_subjects() -> list[int]:
    """Return sorted list of subject numbers found in sj* dirs under the configured data root."""
    data_root = data_root_from_config()
    subjects = []
    if not os.path.isdir(data_root):
        return subjects
    for name in sorted(os.listdir(data_root)):
        if name.startswith("sj") and os.path.isdir(os.path.join(data_root, name)):
            try:
                subjects.append(int(name[2:]))
            except ValueError:
                pass
    return subjects


def crop_status_grid() -> pd.DataFrame:
    """Return a DataFrame showing crop counts and data availability per subject × condition.

    Columns: subject_id, condition, n_crops, has_video, has_fixations, has_gaze
    """
    from vision.config import VISION_ET_FOLDER_MAP, get_world_video_path

    all_conditions = list(VISION_ET_FOLDER_MAP.keys())
    subjects = scan_data_subjects()
    data_root = data_root_from_config()

    cfg = _load_run_config()
    world_video_dir = None
    data_cfg = cfg.get("data", {})
    if isinstance(data_cfg, dict):
        world_video_dir = data_cfg.get("world_video_dir")

    rows = []
    for sj in subjects:
        for cond in all_conditions:
            crop_dir = get_crop_dir(sj, cond)
            n_crops = 0
            if os.path.isdir(crop_dir):
                n_crops = sum(1 for f in os.listdir(crop_dir) if f.lower().endswith(".png"))

            et_folder = VISION_ET_FOLDER_MAP[cond]
            et_dir = os.path.join(data_root, f"sj{sj:02d}", "eye", et_folder)
            has_fixations = os.path.exists(os.path.join(et_dir, "fixations.csv"))
            has_gaze = os.path.exists(os.path.join(et_dir, "gaze_positions.csv"))

            video_path = get_world_video_path(sj, cond, world_video_dir)
            has_video = video_path is not None and os.path.exists(video_path)

            rows.append({
                "subject_id": sj,
                "condition": cond,
                "n_crops": n_crops,
                "has_video": has_video,
                "has_fixations": has_fixations,
                "has_gaze": has_gaze,
            })

    return pd.DataFrame(rows)


# ── Crop mirroring ────────────────────────────────────────────

def get_crop_dir(sj_num: int, condition: str) -> str:
    """Stable directory for a subject+condition's crop PNGs."""
    return os.path.join(CROPS_BASE, f"sj{sj_num:02d}_{condition}")


def crops_exist(sj_num: int, condition: str) -> int:
    """Return count of .png files in the stable crop directory, or 0 if missing."""
    crop_dir = get_crop_dir(sj_num, condition)
    if not os.path.isdir(crop_dir):
        return 0
    return sum(1 for f in os.listdir(crop_dir) if f.lower().endswith(".png"))


def list_crop_files(sj_num: int, condition: str) -> list:
    """Return sorted list of crop filenames in the stable crop directory."""
    crop_dir = get_crop_dir(sj_num, condition)
    if not os.path.isdir(crop_dir):
        return []
    return sorted(f for f in os.listdir(crop_dir) if f.lower().endswith(".png"))


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
                    "labeler_id": "",
                    "is_flagged": False,
                })
                existing_keys.add(key)

    if new_rows:
        combined = pd.concat(
            [existing, pd.DataFrame(new_rows)], ignore_index=True
        )
        save_labels(combined)

    return len(new_rows)
