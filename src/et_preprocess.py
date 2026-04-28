import os

import numpy as np
import pandas as pd

from config import (
    CONDITIONS,
    DATA_DIR,
    ET_FOLDER_MAP,
    OUTPUT_DATA_DIR,
    SUBJECTS,
    TMAX,
    TMIN,
)


def parse_annotations(annotations):
    annotations["trigger"] = annotations["label"].str.extract(r"Event_(\d+)").astype(float)
    annotations["timestamp_s"] = annotations["timestamp [ns]"] / 1e9

    triggers = annotations.dropna(subset=["trigger"]).copy()
    triggers["trigger"] = triggers["trigger"].astype(int)

    excluded = triggers[~triggers["trigger"].between(1, 200)]["label"].unique()
    if len(excluded):
        print(f"    Non-trial event codes excluded: {excluded}")

    triggers = triggers[triggers["trigger"].between(1, 200)].copy()
    triggers = triggers.rename(columns={"timestamp_s": "start_timestamp"}).reset_index(drop=True)
    return triggers


def preprocess_et_wide(sj_num, cond):
    """Produce one row per trial with mean x, y gaze position and pupil diameter."""
    label = cond["eeg_label"]
    sj_dir = os.path.join(DATA_DIR, f"sj{sj_num:02d}", "eye", ET_FOLDER_MAP[label])

    def load_et(fname):
        path = os.path.join(sj_dir, fname)
        return pd.read_csv(path) if os.path.exists(path) else None

    annotations = load_et("annotations.csv")
    gaze = load_et("gaze_positions.csv")
    eye3d = load_et("3d_eye_states.csv")

    if annotations is None or gaze is None:
        print("    Missing annotations or gaze — skipping")
        return None

    triggers = parse_annotations(annotations)
    print(f"    Found {len(triggers)} trial triggers (1–200)")

    gaze["timestamp_s"] = gaze["timestamp [ns]"] / 1e9

    has_pupil = "pupil diameter [mm]" in gaze.columns
    has_3d_pupil = (
        eye3d is not None
        and "timestamp [ns]" in eye3d.columns
        and (
            "pupil diameter left [mm]" in eye3d.columns
            or "pupil diameter right [mm]" in eye3d.columns
        )
    )
    if has_3d_pupil:
        eye3d["timestamp_s"] = eye3d["timestamp [ns]"] / 1e9

    epoch_records = []
    for _, row in triggers.iterrows():
        t0 = row["start_timestamp"]
        t_code = int(row["trigger"])

        gaze_win = gaze[gaze["timestamp_s"].between(t0 + TMIN, t0 + TMAX)]

        rec = {
            "trialIdx": t_code,
            "trigger_time": t0,
            "gaze_n_samples": len(gaze_win),
            "gaze_mean_x_px": gaze_win["gaze x [px]"].mean() if len(gaze_win) else np.nan,
            "gaze_mean_y_px": gaze_win["gaze y [px]"].mean() if len(gaze_win) else np.nan,
        }

        pupil_val = np.nan
        if has_pupil and len(gaze_win):
            pupil_val = gaze_win["pupil diameter [mm]"].mean()

        # Fallback for recordings where gaze_positions has no/poor pupil values:
        # use 3d_eye_states left/right pupil diameters in the same trial window.
        if (not np.isfinite(pupil_val)) and has_3d_pupil:
            eye3d_win = eye3d[eye3d["timestamp_s"].between(t0 + TMIN, t0 + TMAX)]
            if len(eye3d_win):
                pupils = []
                if "pupil diameter left [mm]" in eye3d_win.columns:
                    pupils.append(
                        eye3d_win["pupil diameter left [mm]"].to_numpy(dtype=float)
                    )
                if "pupil diameter right [mm]" in eye3d_win.columns:
                    pupils.append(
                        eye3d_win["pupil diameter right [mm]"].to_numpy(dtype=float)
                    )
                if pupils:
                    pupil_val = np.nanmean(np.vstack(pupils), axis=0).mean()

        rec["pupil_diameter_mm"] = pupil_val

        epoch_records.append(rec)

    et_epochs = pd.DataFrame(epoch_records)
    print(f"    Created {len(et_epochs)} ET epochs (wide)")

    out_path = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_ET_Prepro1.csv")
    et_epochs.to_csv(out_path, index=False)
    print(f"    Saved: {out_path}")
    return et_epochs


def run():
    for sj_num in SUBJECTS:
        print(f"\nProcessing ET — Subject {sj_num:02d}")
        for cond in CONDITIONS:
            print(f"  Condition: {cond['eeg_label']}")
            preprocess_et_wide(sj_num, cond)
    print("\nET preprocessing complete!")


if __name__ == "__main__":
    run()
