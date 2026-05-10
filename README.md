# PSY197 — Mobile EEG + Eye-Tracking Pipeline
Code for UCSB Psychology Honors Thesis. Contents include a research pipeline for studying inhibitory control (Go/NoGo task) under movement (sit/walk) and attention (attend/unattend) conditions using mobile EEG and Pupil Labs eye tracking.

## Pipeline

```
EEG Preprocessing ──┐
                    ├── Fuse EEG + ET ── Extract Features ── DL Tensor Prep
ET Preprocessing  ──┘
```

## Quick Start

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Run full pipeline
cd src && python main.py

# Run individual steps
python src/main.py eeg       # EEG preprocessing
python src/main.py et        # Eye-tracking preprocessing
python src/main.py fuse      # Fuse modalities
python src/main.py features  # Extract ERP/alpha features
python src/main.py dl        # Prepare DL tensors

# Train models
python src/train.py                  # All phases
python src/train.py --phase 6 7      # Specific phases

# Vision pipeline (CLIP-based gaze classification)
pip install -r requirements_vision.txt
python src/vision/vision_main.py --run-dir runs/<run_name>

# Dashboard
streamlit run src/dashboard.py
```

## Training Phases

| Phase | Description |
|-------|-------------|
| 1 | Scalar baselines (LogReg / SVM / LDA) |
| 2 | EEGNet (per-condition + pooled) |
| 3 | MultimodalNet (EEG + ET late fusion) |
| 4 | Outcome prediction (correct vs error) |
| 5 | Vision integration (CLIP gaze features) |
| 6 | No-Go EEGNet (CR vs FA, stratified k-fold) |
| 7 | NoGoFusionNet (EEGNet + CLIP gaze LSTM) |

## Data Layout

```
data/sj{NN}/
  eeg/   — BrainVision .vhdr/.eeg/.vmrk per condition
  beh/   — Behavioral CSVs (5 blocks × condition)
  eye/   — Pupil Labs exports (gaze, fixations, blinks, world video)
```

## Key Components

- **EEGNet** — Compact CNN (Lawhern et al. 2018) for EEG classification
- **MultimodalNet** — Dual-branch EEG + ET with late fusion
- **NoGoFusionNet** — EEG + CLIP gaze sequence encoder (bidirectional LSTM)
- **Vision Pipeline** — CLIP zero-shot + ResNet scene classification of gaze-contingent video crops
- **GEDAI** — Leadfield-aware artifact removal (default over ICA)
- **Dashboard** — Streamlit app for exploring ERPs, gaze, CLIP results, and model outputs

## Configuration

- `src/run_config.yaml` — Pipeline settings (subjects, conditions, sampling rate, artifact method)
- `configs/config.yaml` — Model architecture and training hyperparameters

## Requirements

- Python 3.11
- MNE-Python, PyTorch, Streamlit
- Apple Silicon (MPS), CUDA, or CPU
