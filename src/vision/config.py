import os

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
_RUN_CONFIG_PATH = os.path.join(_SRC_DIR, "run_config.yaml")


def _load_vision_config() -> tuple[dict, list]:
    """Read et.folder_map and vision_conditions from run_config.yaml.

    Returns:
        (et_folder_map, vision_conditions) where vision_conditions is the
        subset of condition keys to use for crop generation / CLIP inference.
        Falls back to all walk conditions if vision_conditions is absent.
    """
    _default_map = {
        "walk_attend": "walk_attend",
        "walk_unattend": "walk_unattend",
        "sit_attend": "sit_attend",
        "sit_unattend": "sit_unattend",
    }
    try:
        with open(_RUN_CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        full_map = cfg.get("et", {}).get("folder_map")
        if not isinstance(full_map, dict) or not full_map:
            full_map = _default_map
        vis_conds = cfg.get("vision_conditions")
        if not isinstance(vis_conds, list) or not vis_conds:
            # Default: only walk conditions
            vis_conds = [k for k in full_map if "walk" in k]
        return full_map, vis_conds
    except Exception:
        return _default_map, ["walk_attend", "walk_unattend"]


_FULL_ET_FOLDER_MAP, VISION_CONDITIONS = _load_vision_config()

# Full map (all conditions) — used by EEG-side helpers like get_eye_dir
ET_FOLDER_MAP = _FULL_ET_FOLDER_MAP

# Filtered map — only conditions where vision crops should be generated
VISION_ET_FOLDER_MAP = {k: v for k, v in ET_FOLDER_MAP.items() if k in VISION_CONDITIONS}

CATEGORIES = {
    "sky": (
        "a wide angle fisheye view looking up at sky and clouds "
        "from a wearable camera"
    ),
    "ocean": (
        "ocean or sea water visible in the distance from a "
        "first person walking perspective"
    ),
    "water": (
        "a lake, pond, or calm body of water seen from a "
        "first person walking perspective"
    ),
    "people": (
        "a person or people seen from behind or ahead on a trail "
        "from a first person wearable camera view"
    ),
    "vegetation": (
        "trees, bushes, grass, or natural vegetation seen from "
        "a first person walking perspective on a trail"
    ),
    "trail_ground": (
        "a wide angle fisheye view looking down at a dirt trail, "
        "ground, rocks, roots, or feet while walking"
    ),
    "other": (
        "an unclear or unidentifiable object in a fisheye "
        "wearable camera image"
    ),
}

CATEGORY_COLORS = {
    "sky": "#87CEEB",
    "ocean": "#1E90FF",
    "water": "#4169E1",
    "people": "#FF6B6B",
    "vegetation": "#2E8B57",
    "trail_ground": "#CD853F",
    "other": "#A9A9A9",
}

# Map condition label → ordered list of candidate filename templates.
# get_world_video_path() returns the first candidate that exists on disk.
# More-specific names (walk_attend, sit_attend) are tried before generic ones
# so the right file is picked when both naming conventions are present.
WORLD_VIDEO_CANDIDATES = {
    "walk_attend":   [
        "sj{sj_num:02d}_walk_attend_world_video.mp4",   # preferred: explicit walk
        "sj{sj_num:02d}_attend_world_video.mp4",         # legacy fallback
    ],
    "walk_unattend": [
        "sj{sj_num:02d}_walk_unattend_world_video.mp4",
        "sj{sj_num:02d}_unattend_world_video.mp4",
    ],
    "sit_attend":    [
        "sj{sj_num:02d}_sit_attend_world_video.mp4",
        "sj{sj_num:02d}_attend_sit_world_video.mp4",    # legacy order variant
    ],
    "sit_unattend":  [
        "sj{sj_num:02d}_sit_unattend_world_video.mp4",
        "sj{sj_num:02d}_unattend_sit_world_video.mp4",
    ],
}

CROP_SIZE = 224
MIN_FIXATION_MS = 80
CLIP_MODEL = "ViT-B/32"


def get_eye_dir(data_root, sj_num, condition_label):
    """Build path to a subject's Pupil Labs export folder for one condition.

    Args:
        data_root: Absolute directory that contains ``sjNN/`` subject folders
            (i.e. the resolved ``data.root`` from ``run_config.yaml``).
    """
    folder = ET_FOLDER_MAP.get(condition_label)
    if folder is None:
        raise KeyError(f"Unknown condition label for eye folder: {condition_label}")
    return os.path.join(data_root, f"sj{sj_num:02d}", "eye", folder)


def get_world_video_path(sj_num, condition_label, world_video_dir):
    """Return absolute path to the world video for this subject/condition.

    Tries each candidate filename in WORLD_VIDEO_CANDIDATES order and returns
    the first one that exists on disk.  This lets both naming conventions
    (e.g. ``walk_attend_world_video.mp4`` and the legacy ``attend_world_video.mp4``)
    coexist without manual config changes.

    Args:
        world_video_dir: Directory containing world videos — read from
            run_config.yaml ``data.world_video_dir``.

    Returns:
        Absolute path string if a matching file is found, else None.
    """
    candidates = WORLD_VIDEO_CANDIDATES.get(condition_label)
    if not candidates or not world_video_dir:
        return None
    for template in candidates:
        path = os.path.join(world_video_dir, template.format(sj_num=sj_num))
        if os.path.exists(path):
            return path
    # Return the first candidate path (will trigger a clear missing-file message)
    return os.path.join(world_video_dir, candidates[0].format(sj_num=sj_num))


def get_vision_out_dir(run_dir, sj_num, condition_label):
    out = os.path.join(run_dir, "vision", f"sj{sj_num:02d}_{condition_label}")
    os.makedirs(out, exist_ok=True)
    return out
