import os

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from config import CONDITIONS, DATA_DIR, ET_FOLDER_MAP, OUTPUT_DATA_DIR, SUBJECTS
from et_timeseries import extract_all_et_epochs, save_et_tensor


def _epochs_to_tensor(epochs):
    """Convert MNE epochs to numpy array (n_epochs, n_channels, n_times)."""
    return epochs.get_data().astype(np.float32)


def _normalize_epochs(X_train, X_val, X_test=None):
    """Channel-wise z-score normalization fit on training data.

    Computes a single mean and std per channel across all epochs and time
    points (global per-channel), so the ERP shape is preserved.
    """
    n_epochs, n_channels, n_times = X_train.shape

    scalers = []
    for ch in range(n_channels):
        # Flatten to (n_epochs * n_times,) so we get one mean/std per channel
        flat_train = X_train[:, ch, :].reshape(-1, 1)
        scaler = StandardScaler()
        X_train[:, ch, :] = scaler.fit_transform(flat_train).reshape(n_epochs, n_times)
        n_val = X_val.shape[0]
        X_val[:, ch, :] = scaler.transform(
            X_val[:, ch, :].reshape(-1, 1)
        ).reshape(n_val, n_times)
        if X_test is not None:
            n_test = X_test.shape[0]
            X_test[:, ch, :] = scaler.transform(
                X_test[:, ch, :].reshape(-1, 1)
            ).reshape(n_test, n_times)
        scalers.append(scaler)

    return X_train, X_val, X_test, scalers


def prepare_dl_data(sj_num, cond, test_size=0.2, random_state=42):
    label = cond["eeg_label"]

    epo_path = os.path.join(OUTPUT_DATA_DIR,
                            f"sj{sj_num:02d}_{label}_Features-epo.fif")
    if not os.path.exists(epo_path):
        print(f"    Missing feature epochs: {epo_path}")
        return None

    epochs = mne.read_epochs(epo_path, preload=True, verbose=False)
    print(f"    Loaded {len(epochs)} epochs")

    X_eeg = _epochs_to_tensor(epochs)
    meta = (epochs.metadata.copy().reset_index(drop=True)
            if epochs.metadata is not None else pd.DataFrame())

    if "trialType" in meta.columns:
        y = (meta["trialType"].values == 20).astype(int)  # 0=go, 1=nogo
    elif "trialIdx" in meta.columns:
        y = meta["trialIdx"].values
    else:
        y = epochs.events[:, 2]

    # ── Build ET time series tensor ──
    et_tensor_path = os.path.join(
        OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}_et_tensor.npy"
    )
    X_et = None
    et_result = None

    if os.path.exists(et_tensor_path):
        X_et = np.load(et_tensor_path)
        print(f"    Loaded existing ET tensor: {X_et.shape}")
    else:
        eye_dir = os.path.join(DATA_DIR, f"sj{sj_num:02d}", "eye",
                               ET_FOLDER_MAP[label])
        if "trigger_time" in meta.columns and os.path.isdir(eye_dir):
            trigger_times = meta["trigger_time"].values
            try:
                et_result = extract_all_et_epochs(eye_dir, trigger_times)
                X_et = et_result["X_et"]
                save_et_tensor(
                    et_result,
                    os.path.join(OUTPUT_DATA_DIR, f"sj{sj_num:02d}_{label}"),
                )
            except Exception as e:
                print(f"    ET tensor extraction failed for {label}: {e}")
                print("    Continuing with EEG-only tensors for this condition.")
        else:
            print("    Cannot build ET tensor — missing trigger_time or "
                  "eye dir")

    if X_et is not None:
        assert X_et.shape[0] == X_eeg.shape[0], (
            f"ET/EEG trial count mismatch: "
            f"ET={X_et.shape[0]}, EEG={X_eeg.shape[0]}"
        )
        print(f"    ET tensor: {X_et.shape}")

    # ── Split ──
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=random_state)
    train_idx, val_idx = next(splitter.split(X_eeg, y))

    X_eeg_train = X_eeg[train_idx].copy()
    X_eeg_val = X_eeg[val_idx].copy()
    y_train = y[train_idx]
    y_val = y[val_idx]
    meta_train = meta.iloc[train_idx].reset_index(drop=True)
    meta_val = meta.iloc[val_idx].reset_index(drop=True)

    X_eeg_train, X_eeg_val, _, scalers = _normalize_epochs(
        X_eeg_train, X_eeg_val
    )

    out_dir = os.path.join(OUTPUT_DATA_DIR, "dl_tensors")
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"sj{sj_num:02d}_{label}"

    np.save(os.path.join(out_dir, f"{prefix}_X_eeg_train.npy"), X_eeg_train)
    np.save(os.path.join(out_dir, f"{prefix}_X_eeg_val.npy"), X_eeg_val)
    np.save(os.path.join(out_dir, f"{prefix}_y_train.npy"), y_train)
    np.save(os.path.join(out_dir, f"{prefix}_y_val.npy"), y_val)
    meta_train.to_csv(os.path.join(out_dir, f"{prefix}_meta_train.csv"),
                      index=False)
    meta_val.to_csv(os.path.join(out_dir, f"{prefix}_meta_val.csv"),
                    index=False)

    # Also save old names for backward compatibility
    np.save(os.path.join(out_dir, f"{prefix}_X_train.npy"), X_eeg_train)
    np.save(os.path.join(out_dir, f"{prefix}_X_val.npy"), X_eeg_val)

    print(f"    EEG Train: {X_eeg_train.shape}  Val: {X_eeg_val.shape}")

    result = {
        "X_eeg_train": X_eeg_train,
        "X_eeg_val": X_eeg_val,
        "y_train": y_train,
        "y_val": y_val,
        "meta_train": meta_train,
        "meta_val": meta_val,
    }

    if X_et is not None:
        X_et_train = X_et[train_idx].copy()
        X_et_val = X_et[val_idx].copy()
        X_et_train, X_et_val, _, _ = _normalize_epochs(X_et_train, X_et_val)

        np.save(os.path.join(out_dir, f"{prefix}_X_et_train.npy"), X_et_train)
        np.save(os.path.join(out_dir, f"{prefix}_X_et_val.npy"), X_et_val)
        print(f"    ET  Train: {X_et_train.shape}  Val: {X_et_val.shape}")

        result["X_et_train"] = X_et_train
        result["X_et_val"] = X_et_val

    print(f"    Saved tensors to: {out_dir}")
    return result


def run():
    for sj_num in SUBJECTS:
        print(f"\nDL prep — Subject {sj_num:02d}")
        for cond in CONDITIONS:
            print(f"  Condition: {cond['eeg_label']}")
            prepare_dl_data(sj_num, cond)
    print("\nDL preparation complete!")


if __name__ == "__main__":
    run()
