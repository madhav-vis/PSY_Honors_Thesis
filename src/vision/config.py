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
}

WORLD_VIDEO_DIR = "/Users/madhav/PSY197B_worldvideos"

# Map condition label → world video filename
WORLD_VIDEO_MAP = {
    "walk_attend": "sj{sj_num:02d}_attend_world_video.mp4",
    "walk_unattend": "sj{sj_num:02d}_unattend_world_video.mp4",
}

CROP_SIZE = 224
MIN_FIXATION_MS = 80
CLIP_MODEL = "ViT-B/32"


def get_eye_dir(r_dir, sj_num, condition_label):
    folder = ET_FOLDER_MAP[condition_label]
    return os.path.join(r_dir, "data", f"sj{sj_num:02d}", "eye", folder)


def get_world_video_path(sj_num, condition_label):
    fname = WORLD_VIDEO_MAP[condition_label].format(sj_num=sj_num)
    return os.path.join(WORLD_VIDEO_DIR, fname)


def get_vision_out_dir(run_dir, sj_num, condition_label):
    out = os.path.join(run_dir, "vision", f"sj{sj_num:02d}_{condition_label}")
    os.makedirs(out, exist_ok=True)
    return out
