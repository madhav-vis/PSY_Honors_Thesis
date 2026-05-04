import os

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

ET_FOLDER_MAP = {
    "walk_attend": "Attend_Walk",
    "walk_unattend": "Unattend_Walk",
    "sit_attend": "Attend_Sit",
    "sit_unattend": "Unattend_Sit",
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


def get_eye_dir(r_dir, sj_num, condition_label):
    folder = ET_FOLDER_MAP.get(condition_label)
    if folder is None:
        raise KeyError(f"Unknown condition label for eye folder: {condition_label}")
    return os.path.join(r_dir, "data", f"sj{sj_num:02d}", "eye", folder)


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
