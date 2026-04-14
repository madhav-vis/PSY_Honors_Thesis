"""
Sanity checks — Layer 1 only (each modality works independently):
  - EEG:  Go vs NoGo ERP at Pz
  - ET:   x position, y position, pupil diameter traces
  - Trial count summary
"""

import os

import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    CONDITIONS,
    DATA_DIR,
    ET_FOLDER_MAP,
    OUTPUT_DATA_DIR,
    OUTPUT_PLOT_DIR,
    SUBJECTS,
)


def check_erp_go_nogo(sj_num, conditions=None):
    """Plot Go vs NoGo ERPs at Pz for each condition (2x2 layout)."""
    if conditions is None:
        conditions = CONDITIONS

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i_cond, cond in enumerate(conditions):
        label = cond["eeg_label"]
        trial_label = cond.get("trial_label", label)
        print(f"  Processing condition: {label}")

        epochs_file = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_EEG_Prepro1-epo.fif")
        if not os.path.exists(epochs_file):
            print(f"    Warning: File not found: {epochs_file}")
            axes[i_cond].text(0.5, 0.5, "File not found", ha="center", va="center")
            continue

        epochs = mne.read_epochs(epochs_file, preload=True, verbose=False)

        if epochs.metadata is not None and len(epochs.metadata) == len(epochs):
            trial_data = epochs.metadata.copy()
        else:
            trial_data_file = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_trialData.csv")
            if os.path.exists(trial_data_file):
                trial_data = pd.read_csv(trial_data_file)
            else:
                print(f"    Error: No trial data found for {label}")
                continue

        if "outcome" in trial_data.columns:
            false_alarm_mask = trial_data["outcome"] == "COMMISSION_ERROR"
            if false_alarm_mask.any():
                print(f"    Removing {false_alarm_mask.sum()} False-Alarm trials")
                keep_indices = ~false_alarm_mask
                trial_data = trial_data[keep_indices].reset_index(drop=True)
                epochs = epochs[keep_indices]

        if "trialType" not in trial_data.columns:
            print(f"    Error: trialType column not found")
            continue

        go_indices = np.where(trial_data["trialType"] == 10)[0]
        nogo_indices = np.where(trial_data["trialType"] == 20)[0]
        print(f"    Go trials: {len(go_indices)}, NoGo trials: {len(nogo_indices)}")

        chan_to_plot = ["Pz"]
        if chan_to_plot[0] not in epochs.ch_names:
            chan_to_plot = [epochs.ch_names[0]]

        epochs_picked = epochs.copy().pick(chan_to_plot)
        times_ms = epochs_picked.times * 1000
        data = epochs_picked.get_data() * 1e6

        ax = axes[i_cond]
        if len(nogo_indices) > 0:
            erp_nogo = np.mean(data[nogo_indices, :, :], axis=0).mean(axis=0)
            ax.plot(times_ms, erp_nogo, color="r", linewidth=3, label="NoGo")
        if len(go_indices) > 0:
            erp_go = np.mean(data[go_indices, :, :], axis=0).mean(axis=0)
            ax.plot(times_ms, erp_go, color="b", linewidth=3, label="Go")

        ax.set_ylim([-20, 20])
        ax.set_xlabel("Time (ms)", fontsize=16)
        ax.set_ylabel("Amplitude (µV)", fontsize=16)
        ax.set_title(trial_label, fontsize=18)
        ax.axvline(x=0, color="k", linestyle="--", linewidth=1)
        ax.legend(fontsize=14)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)

    for i in range(len(conditions), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"sj{sj_num:02d} Tone Locked ERPs Pz", fontsize=24, y=0.995)
    plt.tight_layout()
    out = os.path.join(OUTPUT_PLOT_DIR, f"sj{sj_num:02d}_L1_ERPs_Pz.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def check_trial_counts(sj_num, conditions=None):
    """Print trial counts per condition to catch silent drops."""
    if conditions is None:
        conditions = CONDITIONS

    print(f"\n  Trial counts — sj{sj_num:02d}")
    for cond in conditions:
        label = cond["eeg_label"]
        epo_path = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_EEG_Prepro1-epo.fif")
        if not os.path.exists(epo_path):
            print(f"    {label}: FILE NOT FOUND")
            continue
        epochs = mne.read_epochs(epo_path, preload=False, verbose=False)
        print(f"    {label}: {len(epochs)} epochs")


def check_gaze_xy_pupil(sj_num, conditions=None):
    """Plot x, y gaze position and pupil diameter over time for the full recording."""
    if conditions is None:
        conditions = CONDITIONS

    for cond in conditions:
        label = cond["eeg_label"]
        sj_dir = os.path.join(DATA_DIR, f"sj{sj_num:02d}", "eye", ET_FOLDER_MAP.get(label, label))
        gaze_path = os.path.join(sj_dir, "gaze_positions.csv")
        if not os.path.exists(gaze_path):
            print(f"    Gaze file not found for {label}")
            continue

        gaze = pd.read_csv(gaze_path)
        gaze["timestamp_s"] = gaze["timestamp [ns]"] / 1e9
        t0 = gaze["timestamp_s"].iloc[0]
        gaze["rel_time"] = gaze["timestamp_s"] - t0

        has_pupil = "pupil diameter [mm]" in gaze.columns
        n_rows = 3 if has_pupil else 2

        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4 * n_rows), sharex=True)

        axes[0].plot(gaze["rel_time"], gaze["gaze x [px]"], linewidth=0.3, color="steelblue")
        axes[0].set_ylabel("Gaze X (px)", fontsize=14)
        axes[0].set_title(f"sj{sj_num:02d} {label} — Gaze X Position", fontsize=16)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(gaze["rel_time"], gaze["gaze y [px]"], linewidth=0.3, color="coral")
        axes[1].set_ylabel("Gaze Y (px)", fontsize=14)
        axes[1].set_title(f"sj{sj_num:02d} {label} — Gaze Y Position", fontsize=16)
        axes[1].grid(True, alpha=0.3)

        if has_pupil:
            axes[2].plot(gaze["rel_time"], gaze["pupil diameter [mm]"], linewidth=0.3, color="mediumpurple")
            axes[2].set_ylabel("Pupil Diameter (mm)", fontsize=14)
            axes[2].set_title(f"sj{sj_num:02d} {label} — Pupil Diameter", fontsize=16)
            axes[2].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)", fontsize=14)
        plt.tight_layout()

        out = os.path.join(OUTPUT_PLOT_DIR, f"sj{sj_num:02d}_L1_gaze_xy_pupil_{label}.png")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {out}")


def plot_gaze_trajectories_by_outcome(et_tensor_dict, metadata, out_path,
                                       condition_label=""):
    """Plot gaze x/y trajectories and gaze density heatmaps split by outcome."""
    from et_timeseries import EPOCH_TMIN, EPOCH_TMAX, N_TIMEPOINTS, GAZE_X_MAX, GAZE_Y_MAX

    X_et = et_tensor_dict["X_et"]
    ch_names = et_tensor_dict["channel_names"]

    gx_idx = ch_names.index("gaze_x") if "gaze_x" in ch_names else None
    gy_idx = ch_names.index("gaze_y") if "gaze_y" in ch_names else None
    if gx_idx is None or gy_idx is None:
        print("    No gaze_x/gaze_y channels — skipping trajectory plot")
        return

    outcomes = ["HIT", "MISS", "CORRECT_REJECTION", "COMMISSION_ERROR"]
    times_ms = np.linspace(EPOCH_TMIN * 1000, EPOCH_TMAX * 1000, N_TIMEPOINTS)

    # Stim onset index for post-stim window
    stim_idx = np.argmin(np.abs(times_ms))
    post_500_idx = np.argmin(np.abs(times_ms - 500))

    available = []
    for oc in outcomes:
        mask = metadata["outcome"] == oc
        if mask.sum() >= 5:
            available.append((oc, mask))

    if not available:
        print("    No outcomes with >=5 trials — skipping trajectory plot")
        return

    n_cols = len(available)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    outcome_colors = {"HIT": "blue", "MISS": "orange",
                      "CORRECT_REJECTION": "green", "COMMISSION_ERROR": "red"}

    for col_i, (oc, mask) in enumerate(available):
        idx = np.where(mask.values)[0]
        gx_trials = X_et[idx, gx_idx, :] * GAZE_X_MAX  # back to pixels
        gy_trials = X_et[idx, gy_idx, :] * GAZE_Y_MAX

        # Row 1: time series
        ax_ts = axes[0, col_i]
        gx_mean = gx_trials.mean(axis=0)
        gy_mean = gy_trials.mean(axis=0)
        gx_sem = gx_trials.std(axis=0) / np.sqrt(len(idx))
        gy_sem = gy_trials.std(axis=0) / np.sqrt(len(idx))

        ax_ts.plot(times_ms, gx_mean, color="steelblue", linewidth=1.5,
                   label="gaze X")
        ax_ts.fill_between(times_ms, gx_mean - gx_sem, gx_mean + gx_sem,
                           color="steelblue", alpha=0.2)
        ax_ts.plot(times_ms, gy_mean, color="coral", linewidth=1.5,
                   label="gaze Y")
        ax_ts.fill_between(times_ms, gy_mean - gy_sem, gy_mean + gy_sem,
                           color="coral", alpha=0.2)
        ax_ts.axvline(x=0, color="k", linestyle="--", linewidth=1)
        ax_ts.set_title(f"{oc} (n={len(idx)})", fontsize=12)
        ax_ts.set_xlabel("Time (ms)")
        ax_ts.set_ylabel("Pixel position")
        ax_ts.legend(fontsize=8)
        ax_ts.grid(True, alpha=0.3)

        # Row 2: 2D gaze density heatmap (0–500ms post-stimulus)
        ax_hm = axes[1, col_i]
        gx_post = gx_trials[:, stim_idx:post_500_idx].flatten()
        gy_post = gy_trials[:, stim_idx:post_500_idx].flatten()

        valid = ((gx_post >= 0) & (gx_post <= GAZE_X_MAX) &
                 (gy_post >= 0) & (gy_post <= GAZE_Y_MAX))
        gx_post = gx_post[valid]
        gy_post = gy_post[valid]

        if len(gx_post) > 10:
            h, xedges, yedges = np.histogram2d(
                gx_post, gy_post, bins=50,
                range=[[0, GAZE_X_MAX], [0, GAZE_Y_MAX]],
            )
            h = h.T / len(idx)  # normalize by trial count
            ax_hm.imshow(h, extent=[0, GAZE_X_MAX, GAZE_Y_MAX, 0],
                         aspect="auto", cmap="hot_r", origin="upper")
        ax_hm.axhline(y=GAZE_Y_MAX / 2, color="cyan", linewidth=0.5,
                      linestyle="--", alpha=0.5)
        ax_hm.axvline(x=GAZE_X_MAX / 2, color="cyan", linewidth=0.5,
                      linestyle="--", alpha=0.5)
        ax_hm.set_xlim(0, GAZE_X_MAX)
        ax_hm.set_ylim(GAZE_Y_MAX, 0)
        ax_hm.set_title(f"{oc} — gaze density 0–500ms", fontsize=10)
        ax_hm.set_xlabel("X (px)")
        ax_hm.set_ylabel("Y (px)")

    fig.suptitle(
        f"Gaze Trajectories by Trial Outcome — {condition_label}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


def plot_pupil_by_outcome(et_tensor_dict, metadata, out_path,
                           condition_label=""):
    """Plot pupil diameter over time split by outcome."""
    from et_timeseries import EPOCH_TMIN, EPOCH_TMAX, N_TIMEPOINTS

    ch_names = et_tensor_dict["channel_names"]
    if "pupil" not in ch_names:
        print("    No pupil data — skipping pupil plot")
        return

    X_et = et_tensor_dict["X_et"]
    pupil_idx = ch_names.index("pupil")
    times_ms = np.linspace(EPOCH_TMIN * 1000, EPOCH_TMAX * 1000, N_TIMEPOINTS)

    outcomes = ["HIT", "MISS", "CORRECT_REJECTION", "COMMISSION_ERROR"]
    colors = {"HIT": "blue", "MISS": "orange",
              "CORRECT_REJECTION": "green", "COMMISSION_ERROR": "red"}

    fig, ax = plt.subplots(figsize=(12, 5))

    for oc in outcomes:
        mask = metadata["outcome"] == oc
        if mask.sum() < 5:
            continue
        idx = np.where(mask.values)[0]
        pupil_trials = X_et[idx, pupil_idx, :]
        mu = pupil_trials.mean(axis=0)
        sem = pupil_trials.std(axis=0) / np.sqrt(len(idx))
        ax.plot(times_ms, mu, color=colors[oc], linewidth=1.5, label=oc)
        ax.fill_between(times_ms, mu - sem, mu + sem,
                        color=colors[oc], alpha=0.15)

    ax.axvline(x=0, color="k", linestyle="--", linewidth=1)
    ax.axvspan(300, 500, color="lightgrey", alpha=0.3, label="P300 window")
    ax.set_xlabel("Time (ms)", fontsize=12)
    ax.set_ylabel("Pupil diameter (z-scored)", fontsize=12)
    ax.set_title(f"Pupil Diameter by Outcome — {condition_label}", fontsize=14)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path}")


def run():
    for sj_num in SUBJECTS:
        print(f"\n{'='*60}")
        print(f"  SANITY CHECKS — Subject {sj_num:02d}")
        print(f"{'='*60}")

        print("\n--- Layer 1: Modality Independence ---")
        check_erp_go_nogo(sj_num)
        check_trial_counts(sj_num)
        check_gaze_xy_pupil(sj_num)

    print("\nAll sanity checks complete!")


if __name__ == "__main__":
    run()
