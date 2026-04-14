import os

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch

from config import (
    CONDITIONS,
    OUTPUT_DATA_DIR,
    P300_WINDOW,
    SUBJECTS,
    TARGET_CHANNELS,
)


def _mean_amplitude(epochs, channels, window):
    """Mean amplitude across channels in a time window (microvolts)."""
    picks = [ch for ch in channels if ch in epochs.ch_names]
    if not picks:
        return np.full(len(epochs), np.nan)

    data = epochs.copy().pick(picks).get_data()  # (n_epochs, n_channels, n_times)
    times = epochs.times
    t_mask = (times >= window[0]) & (times <= window[1])
    return data[:, :, t_mask].mean(axis=(1, 2)) * 1e6


def _alpha_power(epochs, channels, tmin, tmax, freq_range=(8, 12)):
    """Mean alpha-band power across channels in a time window (µV²)."""
    picks = [ch for ch in channels if ch in epochs.ch_names]
    if not picks:
        return np.full(len(epochs), np.nan)

    data = epochs.copy().pick(picks).get_data()
    times = epochs.times
    t_mask = (times >= tmin) & (times <= tmax)
    data_win = data[:, :, t_mask]
    sfreq = epochs.info["sfreq"]

    n_per_seg = min(data_win.shape[2], int(sfreq * 0.5))
    if n_per_seg < 4:
        return np.full(len(epochs), np.nan)

    alpha_powers = np.zeros(len(epochs))
    for i in range(len(epochs)):
        powers = []
        for ch in range(data_win.shape[1]):
            freqs, psd = welch(data_win[i, ch, :], fs=sfreq,
                               nperseg=n_per_seg, noverlap=n_per_seg // 2)
            f_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            if f_mask.any():
                powers.append(psd[f_mask].mean())
        alpha_powers[i] = np.mean(powers) * 1e12 if powers else np.nan

    return alpha_powers


def extract_features(sj_num, cond):
    label = cond["eeg_label"]

    epo_path = os.path.join(OUTPUT_DATA_DIR,
                            f"sj{sj_num:02d}_{label}_EEG_ET_Fused-epo.fif")
    if not os.path.exists(epo_path):
        print(f"    Missing fused epochs: {epo_path}")
        return None

    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
    print(f"    Loaded {len(epochs)} fused epochs")

    # P300 — single channel (Pz) for backward compatibility
    p300_pz = _mean_amplitude(epochs, ["Pz"], P300_WINDOW)

    # P300 — cluster from run_config.yaml erp.target_channels
    available_p300 = [c for c in TARGET_CHANNELS if c in epochs.ch_names]
    if len(available_p300) < 3:
        print(f"    Warning: only {len(available_p300)} P300 cluster channels "
              f"available: {available_p300}")
    if not available_p300:
        raise ValueError(f"No P300 channels found. Available: {epochs.ch_names}")
    print(f"    P300 cluster: using {available_p300} ({len(available_p300)} channels)")
    p300_cluster = _mean_amplitude(epochs, available_p300, P300_WINDOW)

    # N200 — frontocentral cluster (negative values = expected N200)
    N200_CHANNELS = ["FCz", "Fz", "FC1", "FC2"]
    N200_TMIN, N200_TMAX = 0.180, 0.260
    available_n200 = [c for c in N200_CHANNELS if c in epochs.ch_names]
    if len(available_n200) < 3:
        print(f"    Warning: only {len(available_n200)} N200 cluster channels "
              f"available: {available_n200}")
    if available_n200:
        print(f"    N200 cluster: using {available_n200} "
              f"({len(available_n200)} channels)")
        n200_cluster = _mean_amplitude(epochs, available_n200,
                                       (N200_TMIN, N200_TMAX))
    else:
        print("    N200 cluster: no channels available — filling NaN")
        n200_cluster = np.full(len(epochs), np.nan)

    # Alpha power — pre-stimulus baseline window
    ALPHA_TMIN, ALPHA_TMAX = -0.200, 0.0
    ALPHA_CHANNELS_FRONTAL = ["Fz", "F3", "F4"]
    ALPHA_CHANNELS_PARIETAL = ["Pz", "P3", "P4"]
    ALPHA_CHANNELS_OCCIPITAL = ["O1", "Oz", "O2"]

    alpha_frontal = _alpha_power(epochs, ALPHA_CHANNELS_FRONTAL,
                                 ALPHA_TMIN, ALPHA_TMAX)
    alpha_parietal = _alpha_power(epochs, ALPHA_CHANNELS_PARIETAL,
                                  ALPHA_TMIN, ALPHA_TMAX)
    alpha_occipital = _alpha_power(epochs, ALPHA_CHANNELS_OCCIPITAL,
                                   ALPHA_TMIN, ALPHA_TMAX)
    print("    Alpha power extracted: frontal, parietal, occipital clusters")

    meta = (epochs.metadata.copy().reset_index(drop=True)
            if epochs.metadata is not None else pd.DataFrame())
    meta["P300_amplitude"] = p300_pz
    meta["P300_Pz_uV"] = p300_pz
    meta["P300_cluster_uV"] = p300_cluster
    meta["N200_cluster_uV"] = n200_cluster
    meta["alpha_frontal_uV2"] = alpha_frontal
    meta["alpha_parietal_uV2"] = alpha_parietal
    meta["alpha_occipital_uV2"] = alpha_occipital

    epochs.metadata = meta

    out_epo = os.path.join(OUTPUT_DATA_DIR,
                           f"sj{sj_num:02d}_{label}_Features-epo.fif")
    out_csv = os.path.join(OUTPUT_DATA_DIR,
                           f"sj{sj_num:02d}_{label}_features.csv")
    epochs.save(out_epo, overwrite=True)
    meta.to_csv(out_csv, index=False)
    print(f"    Saved: {out_epo}")
    print(f"    Saved: {out_csv}")

    return epochs


def run():
    for sj_num in SUBJECTS:
        print(f"\nExtracting features — Subject {sj_num:02d}")
        for cond in CONDITIONS:
            print(f"  Condition: {cond['eeg_label']}")
            extract_features(sj_num, cond)
    print("\nFeature extraction complete!")


if __name__ == "__main__":
    run()
