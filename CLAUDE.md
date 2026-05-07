# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PSY197B is a mobile EEG + eye-tracking research pipeline for studying inhibitory control (Go/NoGo task) under different movement (sit/walk) and attention (attend/unattend) conditions. Data comes from Pupil Labs eye trackers and BrainVision EEG caps. The project preprocesses raw signals, fuses modalities, extracts neural/gaze features, and trains deep learning models (EEGNet, multimodal fusion) to classify trial outcomes.

## Commands

```bash
# Activate the venv (Python 3.11)
source .venv/bin/activate

# Run full preprocessing pipeline (from src/ directory)
cd src && python main.py

# Run individual pipeline steps
python src/main.py eeg          # EEG preprocessing only
python src/main.py et           # Eye-tracking preprocessing only
python src/main.py fuse         # Fuse EEG + ET
python src/main.py features     # Extract ERP/alpha features
python src/main.py dl           # Prepare DL tensors
python src/main.py checks       # Sanity check plots

# Training (runs against latest run dir by default)
python src/train.py                        # all 7 phases
python src/train.py --phase 6 7            # no-go EEGNet + fusion only
python src/train.py --run 2026-04-27_1546_sj03_all_cond_test  # specific run

# Vision pipeline (CLIP-based gaze scene classification)
python src/vision/vision_main.py --run-dir runs/<run_name>

# Dashboard
streamlit run src/dashboard.py

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_vision.txt  # adds CLIP, opencv, etc.
```

## Architecture

### Pipeline Flow (src/main.py orchestrates)

```
01 eeg_preprocess ──┐
                    ├─→ 03 fuse_eeg_et → 04 extract_features → 05 dl_prep
02 et_preprocess  ──┘
                        06 sanity_checks (standalone audit)
```

Steps 01 and 02 are independent and can run in parallel. Step 03 onward requires both.

### Two Config Systems

- **`src/run_config.yaml`** — active pipeline config. Read at import time by `src/config.py`, which exports all settings as module-level constants (e.g., `SUBJECTS`, `CONDITIONS`, `SFREQ_TARGET`). Every pipeline module imports from `config.py`.
- **`configs/config.yaml`** — model architecture and training hyperparameters (EEGNet, gaze encoder, fusion, training schedule). Used by `train.py` and the vision pipeline.

`config.py` snapshots `run_config.yaml` into each run directory for reproducibility.

### Run Directory Convention

Each pipeline execution creates `runs/<date>_<run_name>/` containing:
- `data/` — preprocessed epochs (.fif), metadata (.csv), dl_tensors/
- `plots/` — ERP plots, gaze traces, vision diagnostics
- `models/` — saved PyTorch model weights, embeddings
- `run_config_snapshot.yaml`, `ml_results.json`, `nogo_results.json`

### Data Layout (per subject)

```
data/sj{NN}/
  eeg/    — BrainVision .vhdr/.eeg/.vmrk files per condition
  beh/    — behavioral CSVs (5 blocks × condition)
  eye/    — Pupil Labs exports per condition folder
            (gaze_positions.csv, annotations.csv, 3d_eye_states.csv,
             fixations.csv, blinks.csv, world.mp4)
```

### Training Phases (src/train.py)

1. Scalar baselines (LogReg/SVM/LDA on ERP + gaze features)
2. EEGNet (per-condition + pooled)
3. MultimodalNet — dual-branch EEGNet + ET CNN, late fusion
4. Outcome prediction (correct vs error, HIT vs MISS)
5. Vision integration (CLIP gaze features as scalar inputs)
6. No-go EEGNet — CR vs FA with stratified k-fold
7. NoGoFusionNet — EEGNet + CLIP gaze sequence encoder (LSTM), Wilcoxon comparison vs Phase 6

### Key Model Classes (src/train.py)

- **EEGNet** — Lawhern et al. 2018 compact CNN; has `.embed()` for extracting pre-classifier features
- **MultimodalNet** — EEGNet branch + small CNN for ET, late-fusion classifier
- **GazeSequenceEncoder** — embeds CLIP fixation category sequences via bidirectional LSTM or 1D CNN
- **NoGoFusionNet** — EEGNet + GazeSequenceEncoder, can warm-start EEG branch from Phase 6 weights

### Vision Pipeline (src/vision/)

CLIP-based scene classification of gaze-contingent video crops. Extracts frames at fixation timestamps from world camera video, crops around gaze position, classifies with CLIP zero-shot + optional fine-tuned ResNet head. Outputs per-fixation category labels, CLIP embeddings, and cluster assignments that feed into the fusion pipeline.

### Eye-Tracking Time Series (src/et_timeseries.py)

Extracts raw ET signals (gaze x/y, azimuth, elevation, pupil) per trial epoch, interpolated to EEG sampling rate (250 Hz) for temporal alignment. Produces `(n_trials, n_channels, 301)` tensors matching the EEG epoch window.

### Artifact Removal

Controlled by `run_config.yaml` flags `use_gedai` and `apply_ica`:
- **GEDAI** (default) — `src/gedai_preprocess.py`, leadfield-aware multiresolution artifact removal using the `gedai` library
- **ICA** — MNE fastica with automatic EOG detection via Fp1/Fp2/AF3/AF4

### EEG Preprocessing Quirks

- Subjects 1–4 have a known montage mismatch; `load_correct_montage_for_early_subjects()` remaps channel names/positions from a reference cap file at `data/Dependencies/EEG_32ch_Cap_Correct_Montage/`.
- Hard-coded trial drops in `_MANUAL_TRIAL_DROPS_1BASED` maintain MATLAB parity for early subjects. sj03 drops are currently disabled due to sync issues.
- Trigger latency offset (default 60 samples) is applied to event codes ≤ 200 to correct stimulus timing.
- Trial alignment uses `align_to_eeg_events()` which handles BEH > EEG trial count mismatches via greedy matching.

### Dashboard (src/dashboard.py)

Streamlit app with tabs: Run Manager (edit config + launch pipelines), Overview (ERPs, behavior), Eye Tracking (heatmaps, scanpaths, euclidean distance, optical axis/gyro/pupil triptych), Vision (CLIP results), Fusion & DL (tensors, entropy), EEGNet (no-go classification results + UMAP embeddings).

## Conventions

- All pipeline modules expose a `run()` function called by `main.py`.
- Labels use `trialType=10` for Go, `trialType=20` for NoGo. Outcome column: HIT, MISS, CORRECT_REJECTION, COMMISSION_ERROR.
- MNE epoch files use the suffix `_EEG_Prepro1-epo.fif`; fused epochs use `_EEG_ET_Fused-epo.fif`; feature epochs use `_Features-epo.fif`.
- ET folder names map from snake_case condition labels (e.g., `walk_attend`) to PascalCase directory names (e.g., `Attend_Walk`) via `et.folder_map` in config.
- DL tensors are channel-wise z-scored (fit on train split only).
- The project uses MPS (Apple Silicon) when available, falling back to CUDA then CPU.
