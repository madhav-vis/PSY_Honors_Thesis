"""Hand-labeling tool for gaze crops using an OpenCV GUI window."""

import os
import random

import cv2
import pandas as pd

from .config import CATEGORIES


_KEY_MAP = {
    ord("1"): "sky",
    ord("2"): "ocean",
    ord("3"): "water",
    ord("4"): "people",
    ord("5"): "vegetation",
    ord("6"): "trail_ground",
    ord("7"): "other",
}


def run_annotator(crops_dir, output_csv_path, categories=None, n_samples=100):
    """Present random crops for manual labeling via OpenCV window.

    Resumable: skips already-labeled crops if output_csv_path exists.
    """
    if categories is None:
        categories = CATEGORIES

    all_pngs = sorted(
        f for f in os.listdir(crops_dir) if f.lower().endswith(".png")
    )
    if not all_pngs:
        print("    No crop PNGs found — nothing to annotate")
        return

    already_labeled = set()
    if os.path.exists(output_csv_path):
        prev = pd.read_csv(output_csv_path)
        already_labeled = set(prev["filename"].values)
        print(f"    Resuming — {len(already_labeled)} already labeled")

    candidates = [f for f in all_pngs if f not in already_labeled]
    random.seed(42)
    random.shuffle(candidates)
    candidates = candidates[:n_samples]

    if not candidates:
        print("    All sampled crops already labeled")
        return

    label_names = list(categories.keys())
    shortcut_text = "  ".join(
        f"{i + 1}={label_names[i]}" for i in range(len(label_names))
    )
    shortcut_text += "  q=quit"

    records = []
    if os.path.exists(output_csv_path):
        records = pd.read_csv(output_csv_path).to_dict("records")

    try:
        for idx, fname in enumerate(candidates):
            img = cv2.imread(os.path.join(crops_dir, fname))
            if img is None:
                continue

            display = cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST)

            cv2.putText(
                display, shortcut_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )

            parts = fname.replace(".png", "").split("_")
            fix_id = parts[0] if parts else fname

            win_title = f"Fixation {fix_id} — {idx + 1}/{len(candidates)}"
            cv2.imshow(win_title, display)

            while True:
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    print(f"    Labeled {len(records) - len(already_labeled)} / {len(candidates)} crops")
                    return
                if key in _KEY_MAP:
                    ts_ns = parts[1] if len(parts) > 1 else ""
                    records.append(
                        {
                            "fixation_id": fix_id,
                            "timestamp_ns": ts_ns,
                            "filename": fname,
                            "human_label": _KEY_MAP[key],
                        }
                    )
                    pd.DataFrame(records).to_csv(output_csv_path, index=False)
                    break

            cv2.destroyAllWindows()

        print(f"    Labeled {len(records) - len(already_labeled)} / {len(candidates)} crops")
    finally:
        cv2.destroyAllWindows()
