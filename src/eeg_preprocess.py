import os

import mne
import numpy as np
import pandas as pd

from config import (
    APPLY_ICA,
    BAD_CHAN_Z_THRESH,
    BASELINE,
    CONDITIONS,
    DATA_DIR,
    DETECT_BAD_CHANNELS,
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
    USE_GEDAI,
)

# ── Hard-coded trial removals (MATLAB parity for sj01–04) ───────────
# Values are **1-based** indices matching MATLAB's trialData(row,:)=[]
# statements.  Removed sequentially in descending order so earlier
# deletions don't shift later indices (same semantics as MATLAB).
_MANUAL_TRIAL_DROPS_1BASED = {
    (1, "sit_attend"):     [753],
    (1, "sit_unattend"):   [742, 939],
    (1, "walk_attend"):    [341, 648, 801, 974, 995],
    (1, "walk_unattend"):  [130, 133, 295],
    (2, "walk_attend"):    [888],
    (2, "walk_unattend"):  [434, 601, 872],
    (3, "walk_attend"):    [63, 197, 275, 303, 819, 825, 909, 934],
    (3, "walk_unattend"):  [72, 188, 193, 318, 384, 389, 417, 420,
                            472, 490, 502, 658, 726, 898, 900, 902],
    (4, "sit_unattend"):   [984],
}


def remove_trials_matlab_style(df, indices_1based):
    """Remove rows using MATLAB's sequential 1-based deletion semantics.

    Drops in descending order so each removal doesn't shift the indices
    of subsequent removals — identical to MATLAB's repeated
    trialData(row,:)=[] pattern.
    """
    if not indices_1based:
        return df
    for idx in sorted([i - 1 for i in indices_1based], reverse=True):
        if idx < len(df):
            df = df.drop(df.index[idx]).reset_index(drop=True)
    return df


def load_correct_montage_for_early_subjects(raw, sj_num):
    """Remap channel names/positions for sj01–04 using a reference cap file."""
    if sj_num > 4:
        return raw
    ref_path = os.path.join(
        DATA_DIR, "Dependencies",
        "EEG_32ch_Cap_Correct_Montage", "Test_32ch.vhdr",
    )
    if not os.path.exists(ref_path):
        print(f"    Warning: reference montage not found ({ref_path}) — skipping")
        return raw
    ref = mne.io.read_raw_brainvision(ref_path, preload=False, verbose=False)
    rename_map = dict(zip(raw.ch_names, ref.ch_names))
    raw.rename_channels(rename_map)
    if ref.get_montage() is not None:
        raw.set_montage(ref.get_montage(), on_missing="warn")
    print(f"    Montage corrected for sj{sj_num:02d} using {ref_path}")
    return raw


def preprocess_eeg(sj_num, cond):
    label = cond["eeg_label"]
    source_dir_eeg = os.path.join(DATA_DIR, f"sj{sj_num:02d}", "eeg")
    source_dir_trial = os.path.join(DATA_DIR, f"sj{sj_num:02d}", "beh")

    # ── Load behavioural trial data ──────────────────────────────────
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

    # ── Hard-coded trial removals (MATLAB parity, sj01–04) ──────────
    drop_key = (sj_num, label)
    if drop_key in _MANUAL_TRIAL_DROPS_1BASED:
        idxs = _MANUAL_TRIAL_DROPS_1BASED[drop_key]
        trial_data = remove_trials_matlab_style(trial_data, idxs)
        print(f"    Manual trial removal ({label}): dropped {len(idxs)} → "
              f"{len(trial_data)} trials remain")

    # ── Load and prepare continuous EEG ──────────────────────────────
    eeg_file = os.path.join(source_dir_eeg, f"sj{sj_num:02d}_{label}.vhdr")
    if not os.path.exists(eeg_file):
        print(f"    Error: EEG file not found: {eeg_file}")
        return None, None

    print(f"    Loading EEG from: {eeg_file}")
    raw = mne.io.read_raw_brainvision(eeg_file, preload=True)

    # FIX 1 — montage correction for early subjects
    raw = load_correct_montage_for_early_subjects(raw, sj_num)

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
            for x in ["x_dir", "y_dir", "z_dir", "r_x", "r_y", "r_z",
                       "l_x", "l_y", "l_z"]
        )
        or ch == "32"
    ]
    if ch_to_remove:
        print(f"    Removing channels: {ch_to_remove}")
        raw.drop_channels(ch_to_remove)

    print(f"    Filtering: {FILTER_LOW}–{FILTER_HIGH} Hz")
    raw.filter(FILTER_LOW, FILTER_HIGH, fir_design="firwin2")

    # ── Bad-channel detection (YAML: eeg.detect_bad_channels) ────────
    if DETECT_BAD_CHANNELS:
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) > 0:
            data_eeg = raw.get_data(picks=eeg_picks)
            chan_std = np.std(data_eeg, axis=1)
            z_scores = (chan_std - np.mean(chan_std)) / (np.std(chan_std) + 1e-12)
            bad_chans = [
                raw.ch_names[eeg_picks[i]]
                for i, z in enumerate(z_scores)
                if z > BAD_CHAN_Z_THRESH
            ]
            if bad_chans:
                print(f"    Marking bad channels (z>{BAD_CHAN_Z_THRESH}): {bad_chans}")
                raw.info["bads"].extend(bad_chans)
                raw.interpolate_bads(reset_bads=True)
    else:
        print("    Bad-channel detection: disabled (eeg.detect_bad_channels=false)")

    # ── Artifact removal: GEDAI or ICA ─────────────────────────────
    if USE_GEDAI:
        from gedai_preprocess import apply_gedai
        gedai_plot_dir = os.path.join(OUTPUT_PLOT_DIR, "gedai")
        raw, _ = apply_gedai(
            raw,
            output_plot_dir=gedai_plot_dir,
            label=f"sj{sj_num:02d}_{label}",
        )
        print("    GEDAI: finished")
    elif APPLY_ICA:
        print("    ICA: fitting fastica (n_components=0.99 variance on EEG picks)...")
        ica = mne.preprocessing.ICA(
            n_components=0.99, method="fastica",
            random_state=97, max_iter="auto",
        )
        ica.fit(raw, picks="eeg")
        ica_component_indices = list(range(ica.n_components_))
        if ica_component_indices:
            print(f"    ICA: fit done — {ica.n_components_} components "
                  f"(indices {ica_component_indices[0]}…{ica_component_indices[-1]})")
        else:
            print("    ICA: fit done — 0 components (unexpected)")

        eog_channels = [ch for ch in ["Fp1", "Fp2", "AF3", "AF4"]
                        if ch in raw.ch_names]
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
                print(f"    ICA: EOG-based candidates — "
                      f"{len(eog_indices)} component(s): {sorted(eog_indices)}")
                ica.exclude = eog_indices
            else:
                print("    ICA: find_bads_eog found no components to exclude")
        else:
            print("    ICA: no Fp1/Fp2/AF3/AF4 — skipping automatic EOG detection")

        excluded = sorted(set(ica.exclude))
        print(f"    ICA: ica.exclude = {excluded}")

        ica_fif = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_ica.fif")
        ica.save(ica_fif, overwrite=True)
        print(f"    ICA: saved solution → {ica_fif}")

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
            print(f"    ICA: saved topomaps ({n_maps} maps) → {ica_plot_dir}")
        except Exception as exc:
            print(f"    ICA: topomap export skipped — {exc}")

        if excluded:
            print(f"    ICA: projecting out {len(excluded)} component(s)")
        raw = ica.apply(raw)
        print("    ICA: apply(raw) finished")
    else:
        print("    Artifact removal: disabled (apply_ica=false, use_gedai=false)")

    # ── Epoching ─────────────────────────────────────────────────────
    events, event_id = mne.events_from_annotations(raw)

    print(f"    Adjusting trigger latencies "
          f"(+{TRIGGER_LATENCY_OFFSET} samples for triggers <= 200)")
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

    # ── Trial ↔ EEG alignment ───────────────────────────────────────
    eeg_event_list = epochs.events[:, 2]

    missing_in_eeg = set(trial_data["trialIdx"]) - set(eeg_event_list)
    if missing_in_eeg:
        print(f"    Removing {len(missing_in_eeg)} trials with missing EEG triggers")
        trial_data = trial_data[
            trial_data["trialIdx"].isin(eeg_event_list)
        ].reset_index(drop=True)

    trial_data = align_to_eeg_events(trial_data, eeg_event_list)
    if trial_data is None:
        print("    ABORT — EEG/trial sync failed")
        return None, None

    # ── Alignment diagnostics ────────────────────────────────────────
    print(f"    SYNC VERIFICATION:")
    print(f"      EEG epochs: {len(eeg_event_list)}")
    print(f"      Trial data rows: {len(trial_data)}")
    if len(trial_data) == len(eeg_event_list):
        mismatches = int(np.sum(
            trial_data["trialIdx"].values != eeg_event_list
        ))
        print(f"      Event code mismatches: {mismatches}")
        if mismatches > 0:
            print("      ERROR: Trial alignment has mismatched event codes!")
            for i in range(min(5, len(trial_data))):
                t = trial_data["trialIdx"].values[i]
                e = eeg_event_list[i]
                if t != e:
                    print(f"        Row {i}: trial={t}, eeg={e}")

    if len(trial_data) == len(epochs):
        epochs.metadata = trial_data.copy()
        print("    Added trial data as metadata")

    epo_path = os.path.join(
        OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_EEG_Prepro1-epo.fif")
    trial_path = os.path.join(
        OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_trialData.csv")
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


def clear_cached_epochs(sj_num, conditions):
    """Delete cached epoch files so preprocessing runs from scratch."""
    for cond in conditions:
        label = cond["eeg_label"]
        epo_file = os.path.join(
            OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_EEG_Prepro1-epo.fif")
        if os.path.exists(epo_file):
            os.remove(epo_file)
            print(f"    Deleted cached file: {epo_file}")


def run():
    for sj_num in SUBJECTS:
        print(f"\nProcessing Subject {sj_num}...")
        clear_cached_epochs(sj_num, CONDITIONS)
        for cond in CONDITIONS:
            print(f"  Processing condition: {cond['eeg_label']}")
            preprocess_eeg(sj_num, cond)
    print("\nEEG preprocessing complete!")


if __name__ == "__main__":
    run()
