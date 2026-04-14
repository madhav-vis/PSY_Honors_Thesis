"""Extract raw ET time series per trial epoch, matching EEG epoch window.

Output: (n_channels × n_timepoints) matrix per trial, interpolated to
the EEG sampling rate so both modalities share the same timebase.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ET_SAMPLE_RATE = 200
EEG_SAMPLE_RATE = 250
EPOCH_TMIN = -0.2
EPOCH_TMAX = 1.0
N_TIMEPOINTS = int((EPOCH_TMAX - EPOCH_TMIN) * EEG_SAMPLE_RATE) + 1  # 301

GAZE_X_MAX = 1600.0
GAZE_Y_MAX = 1200.0

ET_CHANNELS = ["gaze_x", "gaze_y", "azimuth", "elevation"]


def load_et_data(eye_dir):
    """Load all ET data files from the Pupil Labs export folder.

    Returns dict with keys 'gaze', 'pupil', 'blinks'.
    """
    result = {"gaze": None, "pupil": None, "blinks": None}

    gaze_path = os.path.join(eye_dir, "gaze_positions.csv")
    if not os.path.exists(gaze_path):
        print(f"    Missing gaze file: {gaze_path}")
        return result

    gaze = pd.read_csv(gaze_path)
    gaze["timestamp_s"] = gaze["timestamp [ns]"] / 1e9
    rename_map = {
        "gaze x [px]": "gaze_x",
        "gaze y [px]": "gaze_y",
    }
    if "azimuth [deg]" in gaze.columns:
        rename_map["azimuth [deg]"] = "azimuth"
    if "elevation [deg]" in gaze.columns:
        rename_map["elevation [deg]"] = "elevation"

    gaze = gaze.rename(columns=rename_map)
    keep = ["timestamp_s"] + [v for v in rename_map.values() if v in gaze.columns]
    gaze = gaze[keep].sort_values("timestamp_s").reset_index(drop=True)

    duration_s = gaze["timestamp_s"].iloc[-1] - gaze["timestamp_s"].iloc[0]
    rate = len(gaze) / duration_s if duration_s > 0 else 0
    print(f"    Loaded gaze: {len(gaze)} samples at ~{rate:.0f} Hz")
    result["gaze"] = gaze

    # Z-score normalize azimuth and elevation across full session
    for col in ["azimuth", "elevation"]:
        if col in gaze.columns:
            mu = gaze[col].mean()
            sd = gaze[col].std() + 1e-8
            gaze[col] = (gaze[col] - mu) / sd

    # Pupil data
    pupil_path = os.path.join(eye_dir, "3d_eye_states.csv")
    if os.path.exists(pupil_path):
        pupil = pd.read_csv(pupil_path)
        pupil["timestamp_s"] = pupil["timestamp [ns]"] / 1e9
        left = pupil.get("pupil_diameter_left_mm")
        right = pupil.get("pupil_diameter_right_mm")
        if left is not None and right is not None:
            pupil["pupil_diameter"] = (left + right) / 2
        elif left is not None:
            pupil["pupil_diameter"] = left
        elif right is not None:
            pupil["pupil_diameter"] = right
        else:
            pupil = None
        if pupil is not None:
            pupil = pupil[["timestamp_s", "pupil_diameter"]].sort_values(
                "timestamp_s"
            ).reset_index(drop=True)
            print(f"    Loaded pupil: {len(pupil)} samples")
            result["pupil"] = pupil
    else:
        print("    No pupil data found")

    # Blinks
    blinks_path = os.path.join(eye_dir, "blinks.csv")
    if os.path.exists(blinks_path):
        blinks = pd.read_csv(blinks_path)
        if "start timestamp [ns]" in blinks.columns:
            blinks["start_timestamp_s"] = blinks["start timestamp [ns]"] / 1e9
            blinks["end_timestamp_s"] = blinks["end timestamp [ns]"] / 1e9
            result["blinks"] = blinks[["start_timestamp_s", "end_timestamp_s"]]
            print(f"    Loaded blinks: {len(blinks)} events")
    else:
        print("    No blink data found")

    return result


def extract_et_epoch(et_data, trigger_time_s, tmin=EPOCH_TMIN,
                     tmax=EPOCH_TMAX, target_rate=EEG_SAMPLE_RATE):
    """Extract and interpolate ET signals for one trial epoch.

    Returns array of shape (n_channels, N_TIMEPOINTS) or None.
    """
    gaze = et_data["gaze"]
    if gaze is None:
        return None, False

    t_start = trigger_time_s + tmin
    t_end = trigger_time_s + tmax

    window = gaze[(gaze["timestamp_s"] >= t_start) &
                  (gaze["timestamp_s"] <= t_end)]

    if len(window) < 10:
        return None, False

    t_src = window["timestamp_s"].values
    t_target = np.linspace(t_start, t_end, N_TIMEPOINTS)

    channels = []

    # Gaze x — normalized to [0, 1]
    gx = window["gaze_x"].values / GAZE_X_MAX
    f_gx = interp1d(t_src, gx, kind="linear", bounds_error=False,
                     fill_value="extrapolate")
    channels.append(f_gx(t_target))

    # Gaze y — normalized to [0, 1]
    gy = window["gaze_y"].values / GAZE_Y_MAX
    f_gy = interp1d(t_src, gy, kind="linear", bounds_error=False,
                     fill_value="extrapolate")
    channels.append(f_gy(t_target))

    # Azimuth and elevation (already z-scored across session)
    for col in ["azimuth", "elevation"]:
        if col in window.columns:
            vals = window[col].values
            f = interp1d(t_src, vals, kind="linear", bounds_error=False,
                         fill_value="extrapolate")
            channels.append(f(t_target))

    # Pupil
    has_blink = False
    pupil_df = et_data.get("pupil")
    if pupil_df is not None:
        p_win = pupil_df[(pupil_df["timestamp_s"] >= t_start) &
                         (pupil_df["timestamp_s"] <= t_end)]
        if len(p_win) >= 5:
            tp = p_win["timestamp_s"].values
            pv = p_win["pupil_diameter"].values
            mu = pv.mean()
            sd = pv.std() + 1e-8
            pv_z = (pv - mu) / sd
            f_p = interp1d(tp, pv_z, kind="linear", bounds_error=False,
                           fill_value="extrapolate")
            channels.append(f_p(t_target))

    # Blink artifact flagging
    blinks = et_data.get("blinks")
    if blinks is not None and not blinks.empty:
        for _, blink in blinks.iterrows():
            if blink["end_timestamp_s"] > t_start and blink["start_timestamp_s"] < t_end:
                has_blink = True
                break

    epoch = np.stack(channels, axis=0).astype(np.float32)

    # NaN → 0 safety net
    epoch = np.nan_to_num(epoch, nan=0.0)

    return epoch, has_blink


def extract_all_et_epochs(eye_dir, trigger_times_s):
    """Extract ET epochs for all trials.

    Parameters
    ----------
    eye_dir : str
        Path to Pupil Labs export folder.
    trigger_times_s : array-like of float
        Trigger timestamps in seconds (one per trial, in EEG epoch order).
        Typically from the aligned fused metadata's trigger_time column.

    Returns dict with X_et, channel_names, failed_trials, has_blink.
    """
    et_data = load_et_data(eye_dir)
    if et_data["gaze"] is None:
        raise FileNotFoundError(f"No gaze data in {eye_dir}")

    has_pupil = et_data["pupil"] is not None
    channel_names = list(ET_CHANNELS)
    if has_pupil:
        channel_names.append("pupil")
    n_channels = len(channel_names)

    trigger_times_s = np.asarray(trigger_times_s, dtype=np.float64)

    epochs_list = []
    failed_trials = []
    blink_flags = []

    for i, tt in enumerate(trigger_times_s):
        if np.isnan(tt):
            epochs_list.append(np.zeros((n_channels, N_TIMEPOINTS),
                                        dtype=np.float32))
            failed_trials.append(i)
            blink_flags.append(False)
            continue

        epoch, has_blink = extract_et_epoch(et_data, tt)

        if epoch is None:
            epochs_list.append(np.zeros((n_channels, N_TIMEPOINTS),
                                        dtype=np.float32))
            failed_trials.append(i)
            blink_flags.append(False)
        else:
            if epoch.shape[0] != n_channels:
                padded = np.zeros((n_channels, N_TIMEPOINTS), dtype=np.float32)
                padded[:epoch.shape[0]] = epoch
                epoch = padded
            epochs_list.append(epoch)
            blink_flags.append(has_blink)

    X_et = np.stack(epochs_list, axis=0)

    n_failed = len(failed_trials)
    n_blink = sum(blink_flags)
    print(f"    Extracted ET epochs: {len(trigger_times_s)} trials, "
          f"{n_channels} channels, {N_TIMEPOINTS} timepoints")
    print(f"    Failed epochs (tracking loss): {n_failed} trials")
    print(f"    Epochs with blinks: {n_blink}")

    return {
        "X_et": X_et,
        "channel_names": channel_names,
        "failed_trials": failed_trials,
        "has_blink": np.array(blink_flags, dtype=bool),
    }


def save_et_tensor(result_dict, out_path):
    """Save ET tensor and companion info JSON."""
    out_str = str(out_path)
    npy_path = f"{out_str}_et_tensor.npy"
    json_path = f"{out_str}_et_tensor_info.json"

    np.save(npy_path, result_dict["X_et"])

    info = {
        "channel_names": result_dict["channel_names"],
        "failed_trials": result_dict["failed_trials"],
        "has_blink": result_dict["has_blink"].tolist(),
        "shape": list(result_dict["X_et"].shape),
    }
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"    Saved ET tensor: shape {result_dict['X_et'].shape} "
          f"to {npy_path}")


def annotate_et_with_clusters(X_et, channel_names, results_df,
                               trial_trigger_times, tmin=EPOCH_TMIN,
                               tmax=EPOCH_TMAX):
    """Add cluster_id channel to ET tensor based on vision results.

    For each trial/timepoint, assigns the nearest fixation's cluster_id
    if within 100ms, else -1.

    Returns (X_et_annotated, updated_channel_names).
    """
    if "cluster_id" not in results_df.columns:
        print("    No cluster_id in vision results — skipping annotation")
        return X_et, channel_names

    fix_ts = results_df["timestamp_s"].values
    fix_cluster = results_df["cluster_id"].values.astype(np.int32)

    n_trials, n_ch, n_t = X_et.shape
    cluster_channel = np.full((n_trials, 1, n_t), -1, dtype=np.float32)

    t_offsets = np.linspace(tmin, tmax, n_t)

    for i in range(n_trials):
        abs_times = trial_trigger_times[i] + t_offsets
        for j, t in enumerate(abs_times):
            diffs = np.abs(fix_ts - t)
            nearest_idx = np.argmin(diffs)
            if diffs[nearest_idx] <= 0.1:  # 100 ms
                cluster_channel[i, 0, j] = fix_cluster[nearest_idx]

    X_annotated = np.concatenate([X_et, cluster_channel], axis=1)
    updated_names = list(channel_names) + ["cluster_id"]

    assigned = (cluster_channel[:, 0, :] >= 0).sum()
    total = n_trials * n_t
    print(f"    Cluster annotation: {assigned}/{total} timepoints assigned "
          f"({assigned/total:.1%})")

    return X_annotated, updated_names


if __name__ == "__main__":
    from config import CONDITIONS, DATA_DIR, ET_FOLDER_MAP, OUTPUT_DATA_DIR, SUBJECTS

    for sj_num in SUBJECTS:
        for cond in CONDITIONS:
            label = cond["eeg_label"]
            eye_dir = os.path.join(DATA_DIR, f"sj{sj_num:02d}", "eye",
                                   ET_FOLDER_MAP[label])
            meta_path = os.path.join(
                OUTPUT_DATA_DIR,
                f"sj{sj_num:02d}_{label}_fused_metadata.csv",
            )

            if not os.path.exists(meta_path):
                print(f"Skipping {label} — no fused metadata")
                continue

            meta = pd.read_csv(meta_path)
            trigger_times = meta["trigger_time"].values

            print(f"\n{'='*50}")
            print(f"  ET Timeseries — sj{sj_num:02d} {label}")
            print(f"{'='*50}")

            result = extract_all_et_epochs(eye_dir, trigger_times)
            save_et_tensor(result,
                           os.path.join(OUTPUT_DATA_DIR,
                                        f"sj{sj_num:02d}_{label}"))
