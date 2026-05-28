# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PSY197B is a mobile EEG + eye-tracking research pipeline for studying inhibitory control (Go/NoGo task) under different movement (sit/walk) and attention (attend/unattend) conditions. Data comes from Pupil Labs eye trackers and BrainVision EEG caps. The project preprocesses raw signals, fuses modalities, extracts neural/gaze features, and trains deep learning models (EEGNet, multimodal fusion) to classify trial outcomes.

## Commands

```bash
# Activate the venv (Python 3.11) — Windows
.venv\Scripts\activate

# Run full preprocessing pipeline
python src/main.py

# Run individual pipeline steps
python src/main.py eeg          # EEG preprocessing only
python src/main.py et           # Eye-tracking preprocessing only
python src/main.py fuse         # Fuse EEG + ET
python src/main.py features     # Extract ERP/alpha features
python src/main.py dl           # Prepare DL tensors
python src/main.py checks       # Sanity check plots

# Training (runs against latest run dir by default)
python src/train.py                        # all phases 1–7
python src/train.py --phase 6             # No-go EEGNet only
python src/train.py --phase 6 7           # No-go EEGNet + fusion
python src/train.py --phase 8             # LOSO cross-validation
python src/train.py --phase 9             # Gaze comparison (walking)
python src/train.py --phase 10            # Cross-condition transfer
python src/train.py --run 2026-04-27_1546_sj03_all_cond_test  # specific run

# Vision pipeline (CLIP-based gaze scene classification)
python src/vision/vision_main.py --run-dir runs/<run_name>

# Train vision models from CLI (CLIP head + ResNet-50)
python src/vision/train_models.py                    # both models
python src/vision/train_models.py --clip-only
python src/vision/train_models.py --resnet-only --resnet-epochs 40
python src/vision/resnet_head.py --output models/resnet50.pt --versioned

# Unified evaluation (vision + EEG)
python src/evaluate.py                    # full evaluation
python src/evaluate.py --vision-only
python src/evaluate.py --eeg-only
python src/evaluate.py --n-repeats 3

# Dashboard
streamlit run src/dashboard.py
make run

# Vision annotator + results
streamlit run src/vision/stream_annotator.py
make annotator                            # runs on port 8502

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_vision.txt    # adds CLIP, opencv, torchvision, etc.
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
- **`configs/config.yaml`** — model architecture and training hyperparameters (EEGNet, gaze encoder, fusion, training schedule). Also controls LOSO run directories for phases 8–10.

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
8. LOSO cross-validation (primary research question)
9. Gaze comparison in walking conditions
10. Cross-condition transfer

### Key Model Classes (src/train.py)

- **EEGNet** — Lawhern et al. 2018 compact CNN; has `.embed()` for extracting pre-classifier features
- **MultimodalNet** — EEGNet branch + small CNN for ET, late-fusion classifier
- **GazeSequenceEncoder** — embeds CLIP fixation category sequences via bidirectional LSTM or 1D CNN
- **NoGoFusionNet** — EEGNet + GazeSequenceEncoder, can warm-start EEG branch from Phase 6 weights

### Vision Pipeline (src/vision/)

CLIP-based scene classification of gaze-contingent video crops. Extracts frames at fixation timestamps from world camera video, crops around gaze position, classifies with CLIP zero-shot + optional fine-tuned ResNet head. Outputs per-fixation category labels, CLIP embeddings, and cluster assignments that feed into the fusion pipeline.

**Canonical model paths** (used by the vision pipeline and annotator's Deploy tab):
- `models/clip_head.pt` — trained CLIP linear head
- `models/resnet50.pt` — fine-tuned ResNet-50

**Human label protection**: Human annotations live only in `data/human_labels.csv` (managed by `src/vision/label_store.py`). The vision pipeline never writes to this file — it writes to per-run `vision_results.csv` files. Re-running the pipeline or redeploying a model will not overwrite hand labels.

**Reclassification behavior**: When a trained head (CLIP or ResNet) is deployed, it overwrites `gaze_target_category` in the run's `vision_results.csv` only, not human labels. The `_build_fusion_csv` function also skips overwriting embedding-based trial features if they already exist.

### Vision Model Training (src/vision/train_models.py + resnet_head.py)

- `train_models.py` — top-level CLI that trains both CLIP head and ResNet-50 sequentially, saves timestamped copies alongside canonical paths.
- `resnet_head.py` — ResNet-50 end-to-end trainer on raw 224×224 PNG crops. Uses `StratifiedShuffleSplit` for train/val/test, weighted sampling and focal loss for class imbalance, saves a pre-test checkpoint (safe if test eval crashes).
- `class_balance.py` — shared helpers: `make_weighted_sampler`, `make_classification_loss`, `balanced_val_accuracy`.

### Evaluation (src/evaluate.py)

Standalone evaluation pipeline writing results to `results/` (configurable via `PSY197B_RESULTS_DIR`). Compares zero-shot CLIP vs trained CLIP head vs ResNet-50 on held-out crops; also runs LOSO EEG evaluation. Used by the Evaluate tab in the annotator dashboard.

### Eye-Tracking Time Series (src/et_timeseries.py)

Extracts raw ET signals (gaze x/y, azimuth, elevation, pupil) per trial epoch, interpolated to EEG sampling rate (250 Hz) for temporal alignment. Produces `(n_trials, n_channels, 301)` tensors matching the EEG epoch window.

### Artifact Removal

Controlled by `run_config.yaml` flags `use_gedai` and `apply_ica`:
- **GEDAI** (default) — `src/gedai_preprocess.py`, leadfield-aware multiresolution artifact removal using the `gedai` library
- **ICA** — MNE fastica with automatic EOG detection via Fp1/Fp2/AF3/AF4

### EEG Preprocessing Quirks

- All subjects have a montage correction applied by `load_correct_montage()`, which remaps channel names/positions from a reference cap file at `assets/reference_montage/` and keeps only the first 32 EEG channels (dropping accelerometer/auxiliary channels). sj20 has fewer auxiliary channels (missing leg accelerometers) but is handled uniformly.
- Hard-coded trial drops in `_MANUAL_TRIAL_DROPS_1BASED` maintain MATLAB parity for early subjects. sj03 drops are currently disabled due to sync issues.
- Trigger latency offset (default 36 samples) is applied to event codes ≤ 200 to correct stimulus timing.
- Trial alignment uses `align_to_eeg_events()` which handles BEH > EEG trial count mismatches via greedy matching.

### Dashboard (src/dashboard.py)

Streamlit app with tabs: Run Manager (edit config + launch EEG/ET pipeline), Overview (ERPs, behavior), Eye Tracking (heatmaps, scanpaths, euclidean distance, optical axis/gyro/pupil triptych), EEGNet (no-go classification results + UMAP embeddings). Each tab has its own view mode toggle (aggregate vs single subject) and subject selector.

### Vision Annotator (src/vision/stream_annotator.py)

Streamlit app for gaze crop annotation, model training, and vision pipeline results. Tabs: Generate Crops, Label, Statistics, Train, Evaluate, Results (CLIP results, categories, clusters).

## Conventions

- All pipeline modules expose a `run()` function called by `main.py`.
- Labels use `trialType=10` for Go, `trialType=20` for NoGo. Outcome column: HIT, MISS, CORRECT_REJECTION, COMMISSION_ERROR.
- MNE epoch files use the suffix `_EEG_Prepro1-epo.fif`; fused epochs use `_EEG_ET_Fused-epo.fif`; feature epochs use `_Features-epo.fif`.
- ET folder names map from snake_case condition labels (e.g., `walk_attend`) to PascalCase directory names (e.g., `Attend_Walk`) via `et.folder_map` in config.
- DL tensors are channel-wise z-scored (fit on train split only).
- The project uses CUDA when available, falling back to CPU (Windows; MPS is Apple Silicon only).
- `run_config.yaml` uses absolute Windows paths (e.g., `C:/Users/twbul/OneDrive/...`). Update `data.root` and `data.world_video_dir` when moving machines.
- Environment variables `PSY197B_RUNS_DIR` and `PSY197B_RESULTS_DIR` override default `runs/` and `results/` locations.
