import gc
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
    MANUAL_BAD_CHANNELS,
    GEDAI_STRENGTH,
    ICA_EOG_CHANNELS,
    ICA_EOG_H_FREQ,
    ICA_EOG_L_FREQ,
    ICA_EOG_MEASURE,
    ICA_EOG_THRESHOLD,
    ICA_EOG_VIRTUAL,
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
    # sj03 — MATLAB parity drops disabled (2026-04): caused EEG/beh row sync
    # issues for walk_unattend; re-enable when indices match current exports.
    # (3, "walk_attend"):    [63, 197, 275, 303, 819, 825, 909, 934],
    # (3, "walk_unattend"):  [72, 188, 193, 318, 384, 389, 417, 420,
    #                         472, 490, 502, 658, 726, 898, 900, 902],
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


def load_correct_montage(raw):
    """Remap channel names/positions for all subjects using a reference cap file.

    Keeps only the first 32 EEG channels; drops accelerometer/auxiliary channels.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = os.path.join(repo_root, "assets", "reference_montage", "Test_32ch.vhdr")
    if not os.path.exists(ref_path):
        print(f"    Warning: reference montage not found ({ref_path}) — skipping")
        return raw
    ref = mne.io.read_raw_brainvision(ref_path, preload=False, verbose=False)
    rename_map = dict(zip(raw.ch_names[:32], ref.ch_names[:32]))
    raw.rename_channels(rename_map)
    raw.pick(raw.ch_names[:32])
    if ref.get_montage() is not None:
        raw.set_montage(ref.get_montage(), on_missing="warn")
    print(f"    Montage corrected ({len(raw.ch_names)} channels kept)")
    return raw


def _preprocess_pre_gedai(sj_num, cond):
    """Phase A: load Raw EEG, montage, resample, reference, filter, bad-channel
    detection.  Returns a GEDAI-ready (raw, trial_data) pair, or (None, None)."""
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
        swapped_label = "_".join(reversed(label.split("_")))
        alt_file = os.path.join(source_dir_eeg, f"sj{sj_num:02d}_{swapped_label}.vhdr")
        if os.path.exists(alt_file):
            print(f"    Note: using swapped filename convention "
                  f"({label} -> {swapped_label})")
            eeg_file = alt_file
        else:
            print(f"    Error: EEG file not found: {eeg_file}")
            print(f"           (also tried: {alt_file})")
            return None, None

    print(f"    Loading EEG from: {eeg_file}")
    raw = mne.io.read_raw_brainvision(eeg_file, preload=True)

    raw = load_correct_montage(raw)

    if raw.info["sfreq"] != SFREQ_TARGET:
        print(f"    Downsampling from {raw.info['sfreq']:.1f} Hz to {SFREQ_TARGET} Hz")
        raw = raw.resample(SFREQ_TARGET)

    if USE_GEDAI:
        print("    Re-referencing to average (GEDAI leadfield requires avg ref)")
        raw.set_eeg_reference("average", ch_type="eeg")
    elif all(ch in raw.ch_names for ch in REF_CHANNELS):
        print(f"    Re-referencing to average of {REF_CHANNELS}")
        raw.set_eeg_reference(REF_CHANNELS, ch_type="eeg")
    else:
        print("    Warning: reference channels not found, skipping re-reference")

    print(f"    Filtering: {FILTER_LOW}–{FILTER_HIGH} Hz")
    raw.filter(FILTER_LOW, FILTER_HIGH, fir_design="firwin2")

    # ── Bad-channel detection (YAML: eeg.detect_bad_channels) ────────
    # ── Bad-channel detection + manual overrides (single interpolation pass) ──
    # Manual overrides are added first so they don't inflate the z-score
    # distribution (two simultaneously bad channels suppress each other's z).
    all_bad = set(raw.info.get("bads", []))

    manual_bads = MANUAL_BAD_CHANNELS.get(sj_num, [])
    manual_new = [ch for ch in manual_bads if ch in raw.ch_names]
    if manual_new:
        print(f"    Manual bad-channel override for sj{sj_num:02d}: {manual_new}")
        all_bad.update(manual_new)

    if DETECT_BAD_CHANNELS:
        eeg_picks = mne.pick_types(raw.info, eeg=True,
                                   exclude=list(all_bad))
        if len(eeg_picks) > 0:
            data_eeg = raw.get_data(picks=eeg_picks)
            chan_std = np.std(data_eeg, axis=1)
            z_scores = (chan_std - np.mean(chan_std)) / (np.std(chan_std) + 1e-12)
            auto_bad = [
                raw.ch_names[eeg_picks[i]]
                for i, z in enumerate(z_scores)
                if z > BAD_CHAN_Z_THRESH
            ]
            if auto_bad:
                print(f"    Auto bad channels (z>{BAD_CHAN_Z_THRESH}): {auto_bad}")
                all_bad.update(auto_bad)
    else:
        print("    Bad-channel detection: disabled (eeg.detect_bad_channels=false)")

    if all_bad:
        raw.info["bads"] = sorted(all_bad)
        raw.interpolate_bads(reset_bads=True)
        print(f"    Interpolated {len(all_bad)} bad channel(s): {sorted(all_bad)}")

    return raw, trial_data


def _raw_with_virtual_eog(raw, eog_channels):
    """Add VEOG = mean(frontal proxies) for more stable find_bads_eog."""
    if len(eog_channels) < 2:
        return raw, []
    raw_det = raw.copy()
    veog_data = raw_det.get_data(picks=eog_channels).mean(axis=0, keepdims=True)
    veog_info = mne.create_info(["VEOG"], raw_det.info["sfreq"], ["eog"])
    raw_det.add_channels(
        [mne.io.RawArray(veog_data, veog_info)],
        force_update_info=True,
    )
    return raw_det, ["VEOG"]


def find_ica_eog_excludes(ica, raw):
    """ICA components correlated with frontal/VEOG channels (configurable threshold)."""
    eog_channels = [ch for ch in ICA_EOG_CHANNELS if ch in raw.ch_names]
    if not eog_channels:
        print("    ICA: no EOG proxy channels in cap — skipping automatic EOG detection")
        return []

    raw_det = raw
    scan_channels = list(eog_channels)
    if ICA_EOG_VIRTUAL:
        try:
            raw_det, virtual = _raw_with_virtual_eog(raw, eog_channels)
            scan_channels.extend(virtual)
        except Exception as exc:
            print(f"    ICA: virtual VEOG skipped ({exc})")

    print(
        f"    ICA: find_bads_eog (measure={ICA_EOG_MEASURE}, "
        f"threshold={ICA_EOG_THRESHOLD}, channels={scan_channels})"
    )

    eog_indices = []
    hits_by_channel = {}
    for eog_ch in scan_channels:
        try:
            inds, _ = ica.find_bads_eog(
                raw_det,
                ch_name=eog_ch,
                threshold=ICA_EOG_THRESHOLD,
                measure=ICA_EOG_MEASURE,
                l_freq=ICA_EOG_L_FREQ,
                h_freq=ICA_EOG_H_FREQ,
                verbose=False,
            )
        except Exception as exc:
            print(f"    ICA: find_bads_eog({eog_ch}) failed — {exc}")
            continue
        if inds:
            inds = [int(i) for i in inds]
            hits_by_channel[eog_ch] = inds
            eog_indices.extend(inds)

    eog_indices = sorted(set(eog_indices))
    if hits_by_channel:
        for ch, inds in hits_by_channel.items():
            print(f"    ICA: EOG match via {ch} → component(s) {inds}")
    return eog_indices


def _postprocess_after_gedai(raw, trial_data, sj_num, cond):
    """Phase B: mastoid re-reference, epoch, trial alignment, save."""
    label = cond["eeg_label"]

    # ── Post-GEDAI re-reference to mastoids for analysis ────────────
    if USE_GEDAI and all(ch in raw.ch_names for ch in REF_CHANNELS):
        print(f"    Re-referencing to {REF_CHANNELS} (post-GEDAI, for analysis)")
        raw.set_eeg_reference(REF_CHANNELS, ch_type="eeg")

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
    del raw
    gc.collect()
    epochs.apply_baseline(baseline=BASELINE)
    print(f"    Created {len(epochs)} epochs")

    if "Pz" in epochs.ch_names:
        pz_data = epochs.get_data(picks=["Pz"]) * 1e6
        peak = pz_data.mean(axis=0).squeeze()
        print(f"    P300 sanity: Pz grand-avg peak={peak.max():.2f} µV "
              f"(at {epochs.times[peak.argmax()]*1000:.0f} ms)")

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


def _preprocess_subject_combined_gedai(sj_num, conditions):
    """Preprocess all conditions, apply GEDAI once on the combined data,
    then split back and epoch each condition independently."""
    from gedai_preprocess import apply_gedai

    pre_results = {}
    for cond in conditions:
        label = cond["eeg_label"]
        print(f"  Pre-GEDAI preprocessing: {label}")
        raw, trial_data = _preprocess_pre_gedai(sj_num, cond)
        if raw is None:
            print(f"    Skipping {label} (preprocessing failed)")
            continue
        pre_results[label] = (raw, trial_data)

    if not pre_results:
        print(f"  No conditions preprocessed for sj{sj_num:02d} — skipping")
        return

    # ── Concatenate all conditions ───────────────────────────────────
    boundaries = {}
    raws_to_concat = []
    cumulative = 0
    for cond in conditions:
        label = cond["eeg_label"]
        if label not in pre_results:
            continue
        raw = pre_results[label][0]
        n_samples = raw.n_times
        boundaries[label] = (cumulative, cumulative + n_samples)
        cumulative += n_samples
        raws_to_concat.append(raw)

    print(f"  Concatenating {len(raws_to_concat)} conditions "
          f"({cumulative} total samples) for combined GEDAI")
    combined_raw = mne.concatenate_raws(raws_to_concat)

    eeg_data = combined_raw.get_data(picks="eeg")
    print(f"  Pre-GEDAI combined data scale: mean={eeg_data.mean():.2e}, "
          f"std={eeg_data.std():.2e} V")
    del eeg_data

    gedai_plot_dir = os.path.join(OUTPUT_PLOT_DIR, "gedai")
    combined_raw, _ = apply_gedai(
        combined_raw,
        denoising_strength=GEDAI_STRENGTH,
        output_plot_dir=gedai_plot_dir,
        label=f"sj{sj_num:02d}_combined",
    )
    print("  Combined GEDAI: finished")

    # ── Split back and epoch each condition ──────────────────────────
    sfreq = combined_raw.info["sfreq"]
    for cond in conditions:
        label = cond["eeg_label"]
        if label not in boundaries:
            continue
        start, end = boundaries[label]
        tmin_crop = start / sfreq
        tmax_crop = (end - 1) / sfreq
        print(f"  Post-GEDAI processing: {label} "
              f"(samples {start}–{end}, t={tmin_crop:.1f}–{tmax_crop:.1f}s)")
        raw_cond = combined_raw.copy().crop(tmin=tmin_crop, tmax=tmax_crop)
        trial_data = pre_results[label][1]
        _postprocess_after_gedai(raw_cond, trial_data, sj_num, cond)
        del raw_cond
        gc.collect()

    del combined_raw
    gc.collect()


def preprocess_eeg(sj_num, cond):
    """Non-GEDAI preprocessing path (ICA or no artifact removal)."""
    label = cond["eeg_label"]
    raw, trial_data = _preprocess_pre_gedai(sj_num, cond)
    if raw is None:
        return None, None

    if APPLY_ICA:
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

        eog_indices = find_ica_eog_excludes(ica, raw)
        if eog_indices:
            ica.exclude = eog_indices
            print(f"    ICA: excluding {len(eog_indices)} component(s): {eog_indices}")
        else:
            print("    ICA: find_bads_eog found no components to exclude")
            ica.exclude = []

        excluded = sorted(set(int(i) for i in ica.exclude))
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

    return _postprocess_after_gedai(raw, trial_data, sj_num, cond)


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
        if USE_GEDAI:
            _preprocess_subject_combined_gedai(sj_num, CONDITIONS)
        else:
            for cond in CONDITIONS:
                print(f"  Processing condition: {cond['eeg_label']}")
                preprocess_eeg(sj_num, cond)
                gc.collect()
    print("\nEEG preprocessing complete!")


if __name__ == "__main__":
    run()
