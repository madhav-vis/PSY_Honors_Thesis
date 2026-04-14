import os

import mne
import numpy as np

from config import (
    BAD_CHAN_Z_THRESH,
    BASELINE,
    CONDITIONS,
    DATA_DIR,
    FILTER_HIGH,
    FILTER_LOW,
    OUTPUT_DATA_DIR,
    OUTPUT_PLOT_DIR,
    REF_CHANNELS,
    SFREQ_TARGET,
    SUBJECTS,
    TMAX,
    TMIN,
    TRIGGER_LATENCY_OFFSET,
)


def preprocess_eeg(sj_num, cond):
    label = cond["eeg_label"]
    source_dir_eeg = os.path.join(DATA_DIR, f"sj{sj_num:02d}", "eeg")
    source_dir_trial = os.path.join(DATA_DIR, f"sj{sj_num:02d}", "beh")

    import pandas as pd

    trial_data_list = []
    for i_block in range(1, 6):
        filename = os.path.join(
            source_dir_trial,
            f"sj{sj_num:02d}_block{i_block}_{cond['trial_label']}.csv",
        )
        if os.path.exists(filename):
            trial_data_list.append(pd.read_csv(filename))
        else:
            print(f"    Warning: No file for block {i_block}: {filename}")

    if not trial_data_list:
        print(f"    Error: No trial data for {cond['trial_label']}")
        return None, None

    trial_data = pd.concat(trial_data_list, ignore_index=True)
    print(f"    Loaded {len(trial_data)} trials")

    eeg_file = os.path.join(source_dir_eeg, f"sj{sj_num:02d}_{label}.vhdr")
    if not os.path.exists(eeg_file):
        print(f"    Error: EEG file not found: {eeg_file}")
        return None, None

    print(f"    Loading EEG from: {eeg_file}")
    raw = mne.io.read_raw_brainvision(eeg_file, preload=True)

    if raw.info["sfreq"] != SFREQ_TARGET:
        print(f"    Downsampling from {raw.info['sfreq']:.1f} Hz to {SFREQ_TARGET} Hz")
        raw = raw.resample(SFREQ_TARGET)

    if all(ch in raw.ch_names for ch in REF_CHANNELS):
        print(f"    Re-referencing to average of {REF_CHANNELS}")
        raw.set_eeg_reference(REF_CHANNELS, ch_type="eeg")
    else:
        print("    Warning: reference channels not found, skipping re-reference")

    ch_to_remove = [
        ch
        for ch in raw.ch_names
        if any(
            x in ch.lower()
            for x in ["x_dir", "y_dir", "z_dir", "r_x", "r_y", "r_z", "l_x", "l_y", "l_z"]
        )
        or ch == "32"
    ]
    if ch_to_remove:
        print(f"    Removing channels: {ch_to_remove}")
        raw.drop_channels(ch_to_remove)

    print(f"    Filtering: {FILTER_LOW}–{FILTER_HIGH} Hz")
    raw.filter(FILTER_LOW, FILTER_HIGH, fir_design="firwin2")

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    if len(eeg_picks) > 0:
        data_eeg = raw.get_data(picks=eeg_picks)
        chan_std = np.std(data_eeg, axis=1)
        z_scores = (chan_std - np.mean(chan_std)) / (np.std(chan_std) + 1e-12)
        bad_chans = [raw.ch_names[eeg_picks[i]] for i, z in enumerate(z_scores) if z > BAD_CHAN_Z_THRESH]
        if bad_chans:
            print(f"    Marking bad channels (z>{BAD_CHAN_Z_THRESH}): {bad_chans}")
            raw.info["bads"].extend(bad_chans)
            raw.interpolate_bads(reset_bads=True)

    print("    ICA: fitting fastica (n_components=0.99 variance on EEG picks)...")
    ica = mne.preprocessing.ICA(n_components=0.99, method="fastica", random_state=97, max_iter="auto")
    ica.fit(raw, picks="eeg")
    ica_component_indices = list(range(ica.n_components_))
    if ica_component_indices:
        print(
            f"    ICA: fit done — {ica.n_components_} components "
            f"(indices {ica_component_indices[0]}…{ica_component_indices[-1]})"
        )
    else:
        print("    ICA: fit done — 0 components (unexpected)")

    eog_channels = [ch for ch in ["Fp1", "Fp2", "AF3", "AF4"] if ch in raw.ch_names]
    eog_indices = []
    if eog_channels:
        for eog_ch in eog_channels:
            try:
                inds, _ = ica.find_bads_eog(raw, ch_name=eog_ch, verbose=False)
                eog_indices.extend(inds)
            except Exception:
                pass
        eog_indices = list(set(eog_indices))
        if eog_indices:
            print(f"    ICA: EOG-based candidates — {len(eog_indices)} component(s): {sorted(eog_indices)}")
            ica.exclude = eog_indices
        else:
            print("    ICA: find_bads_eog found no components to exclude")
    else:
        print("    ICA: no Fp1/Fp2/AF3/AF4 in data — skipping automatic EOG detection")

    excluded = sorted(set(ica.exclude))
    print(f"    ICA: ica.exclude (will be projected out) = {excluded}")
    if not excluded:
        print("    ICA: apply(raw) — exclude empty, continuous data unchanged by ICA")

    ica_fif = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_ica.fif")
    ica.save(ica_fif, overwrite=True)
    print(f"    ICA: saved solution + exclude list → {ica_fif}")

    ica_plot_dir = os.path.join(OUTPUT_PLOT_DIR, "ica")
    os.makedirs(ica_plot_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        n_maps = min(ica.n_components_, 24)
        if n_maps < 1:
            raise ValueError("no ICA components to plot")
        picks = ica_component_indices[:n_maps]
        figs = ica.plot_components(inst=raw, picks=picks, show=False)
        if not isinstance(figs, (list, tuple)):
            figs = [figs]
        for fi, fig in enumerate(figs):
            comp_path = os.path.join(
                ica_plot_dir,
                f"sj{sj_num:02d}_{label}_ica_components_{fi}.png",
            )
            fig.savefig(comp_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        print(f"    ICA: saved component topomaps ({n_maps} maps) → {ica_plot_dir}")
    except Exception as exc:
        print(f"    ICA: component topomap export skipped — {exc}")

    if excluded:
        print(f"    ICA: apply(raw) — projecting out {len(excluded)} component(s)")
    raw = ica.apply(raw)
    print("    ICA: apply(raw) finished (continuous EEG updated for epoching)")

    events, event_id = mne.events_from_annotations(raw)

    print(f"    Adjusting trigger latencies (+{TRIGGER_LATENCY_OFFSET} samples for triggers <= 200)")
    events_adjusted = events.copy()
    events_adjusted[events_adjusted[:, 2] <= 200, 0] += TRIGGER_LATENCY_OFFSET

    valid_events = events_adjusted[events_adjusted[:, 2] <= 200]
    valid_event_ids = {str(code): code for code in np.unique(valid_events[:, 2])}

    epochs = mne.Epochs(
        raw,
        valid_events,
        event_id=valid_event_ids,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        verbose=False,
    )
    epochs.apply_baseline(baseline=BASELINE)
    print(f"    Created {len(epochs)} epochs")

    eeg_event_list = epochs.events[:, 2]

    missing_in_eeg = set(trial_data["trialIdx"]) - set(eeg_event_list)
    if missing_in_eeg:
        print(f"    Removing {len(missing_in_eeg)} trials with missing EEG triggers")
        trial_data = trial_data[trial_data["trialIdx"].isin(eeg_event_list)].reset_index(drop=True)

    trial_data = align_to_eeg_events(trial_data, eeg_event_list)
    if trial_data is None:
        print("    ABORT — EEG/trial sync failed")
        return None, None

    if len(trial_data) == len(epochs):
        epochs.metadata = trial_data.copy()
        print("    Added trial data as metadata")

    epo_path = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_EEG_Prepro1-epo.fif")
    trial_path = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_trialData.csv")
    epochs.save(epo_path, overwrite=True)
    trial_data.to_csv(trial_path, index=False)
    print(f"    Saved: {epo_path}")
    print(f"    Saved: {trial_path}")

    return epochs, trial_data


def align_to_eeg_events(df, eeg_event_list, idx_col="trialIdx"):
    import numpy as np

    beh_codes = df[idx_col].values
    eeg_codes = np.array(eeg_event_list)

    if len(beh_codes) == len(eeg_codes) and np.all(beh_codes == eeg_codes):
        print("    SYNC SUCCESS (exact match, all blocks)")
        return df.reset_index(drop=True)

    if len(beh_codes) > len(eeg_codes):
        print(f"    BEH has {len(beh_codes)} trials, EEG has {len(eeg_codes)} — trimming BEH")
        beh_ptr = 0
        aligned_rows = []
        for eeg_code in eeg_codes:
            while beh_ptr < len(beh_codes) and beh_codes[beh_ptr] != eeg_code:
                beh_ptr += 1
            if beh_ptr >= len(beh_codes):
                print("    SYNC FAIL — ran out of BEH trials")
                return None
            aligned_rows.append(beh_ptr)
            beh_ptr += 1

        aligned = df.iloc[aligned_rows].reset_index(drop=True)
        if np.all(aligned[idx_col].values == eeg_codes):
            print(f"    SYNC SUCCESS ({len(aligned)} trials aligned)")
            return aligned
        else:
            print("    SYNC FAIL — final check mismatch")
            return None

    if len(beh_codes) < len(eeg_codes):
        print(f"    BEH has fewer trials ({len(beh_codes)}) than EEG ({len(eeg_codes)})")
        print("    SYNC FAIL — cannot align")
        return None

    print("    SYNC FAIL — unknown alignment issue")
    return None


def run():
    for sj_num in SUBJECTS:
        print(f"\nProcessing Subject {sj_num}...")
        for cond in CONDITIONS:
            print(f"  Processing condition: {cond['eeg_label']}")
            preprocess_eeg(sj_num, cond)
    print("\nEEG preprocessing complete!")


if __name__ == "__main__":
    run()
