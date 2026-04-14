import os

import mne
import pandas as pd

from config import CONDITIONS, OUTPUT_DATA_DIR, SUBJECTS
from eeg_preprocess import align_to_eeg_events


def fuse(sj_num, cond):
    label = cond["eeg_label"]

    eeg_path = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_EEG_Prepro1-epo.fif")
    et_path = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_ET_Prepro1.csv")

    if not os.path.exists(eeg_path):
        print(f"    Missing EEG: {eeg_path}")
        return None
    if not os.path.exists(et_path):
        print(f"    Missing ET:  {et_path}")
        return None

    epochs = mne.read_epochs(eeg_path, preload=False, verbose=False)
    et = pd.read_csv(et_path)
    eeg_event_list = epochs.events[:, 2]
    print(f"    EEG: {len(eeg_event_list)} epochs  |  ET: {len(et)} epochs")

    et_aligned = align_to_eeg_events(et, eeg_event_list, idx_col="trialIdx")
    if et_aligned is None:
        print("    Skipping — alignment failed")
        return None

    fused_meta = (
        epochs.metadata.copy().reset_index(drop=True)
        if epochs.metadata is not None
        else pd.DataFrame()
    )
    for col in [c for c in et_aligned.columns if c != "trialIdx"]:
        fused_meta[col] = et_aligned[col].values

    # Load vision trial features if available
    vision_path = os.path.join(
        OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_vision_trial_features.csv"
    )
    if os.path.exists(vision_path):
        vision_df = pd.read_csv(vision_path)

        scalar_cols = ["dominant_cluster", "cluster_entropy",
                       "n_fixations_in_window", "emb_spread"]
        scalar_cols += [c for c in vision_df.columns
                        if c.startswith("vis_cluster_")]

        available_scalar = [c for c in scalar_cols if c in vision_df.columns]
        if available_scalar:
            vision_scalar = vision_df[["trialIdx"] + available_scalar]
            vision_scalar = vision_scalar.set_index("trialIdx")
            for col in available_scalar:
                fused_meta[col] = fused_meta["trialIdx"].map(
                    vision_scalar[col]).values

        if "mean_embedding" in vision_df.columns:
            emb_map = vision_df.set_index("trialIdx")["mean_embedding"]
            fused_meta["mean_embedding"] = fused_meta["trialIdx"].map(
                emb_map).values

        n_vision_cols = len(available_scalar) + (
            1 if "mean_embedding" in vision_df.columns else 0)
        print(f"    Added {n_vision_cols} vision features")
    else:
        print(f"    No vision features found — run vision pipeline first")

    epochs.metadata = fused_meta

    out_epo = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_EEG_ET_Fused-epo.fif")
    out_meta = os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_fused_metadata.csv")
    epochs.save(out_epo, overwrite=True)
    fused_meta.to_csv(out_meta, index=False)
    print(f"    Saved: {out_epo}")
    print(f"    Saved: {out_meta}")

    return epochs


def run():
    for sj_num in SUBJECTS:
        print(f"\nFusing EEG+ET — Subject {sj_num:02d}")
        for cond in CONDITIONS:
            print(f"  Condition: {cond['eeg_label']}")
            fuse(sj_num, cond)
    print("\nEEG + ET fusion complete!")


if __name__ == "__main__":
    run()
