# PSY197B — Mobile EEG + Eye-Tracking Pipeline

UCSB Psychology Honors Thesis. Research pipeline for studying inhibitory control (Go/NoGo task) under movement (sit/walk) and attention (attend/unattend) conditions using mobile EEG and Pupil Labs eye tracking.

## Pipeline

```
EEG Preprocessing ──┐
                    ├── Fuse EEG + ET ── Extract Features ── DL Tensor Prep ── Train ── Evaluate
ET Preprocessing  ──┘
```

## Setup

```bash
# Mac
bash setup_mac.sh

# Windows
setup_windows.bat

# Or manually:
python3 -m venv .venv
source .venv/bin/activate          # Mac/Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
pip install -r requirements_vision.txt
```

## End-to-End Workflow

```bash
# 1. Preprocess all subjects (EEG + eye-tracking + fuse + features + DL tensors)
python src/main.py

# 2. Train all model phases (1-10)
python src/train.py

# 3. Evaluate models
python src/evaluate.py

# 4. Vision pipeline — classify gaze crops with ResNet-50 (optional, per run)
python src/vision/vision_main.py --run-dir runs/<run_name>
```

### Running Individual Steps

```bash
python src/main.py eeg       # EEG preprocessing only
python src/main.py et        # Eye-tracking preprocessing only
python src/main.py fuse      # Fuse EEG + ET
python src/main.py features  # Extract ERP/alpha features
python src/main.py dl        # Prepare DL tensors

python src/train.py --phase 6 7   # Specific training phases only
```

### Annotation Tool

```bash
streamlit run src/vision/stream_annotator.py --server.port 8502
# or: make annotator
```

Tabs: Generate Crops, Label, Statistics, Train (ResNet-50), Evaluate, Deploy.

## Training Phases

| Phase | Description |
|-------|-------------|
| 1 | Scalar baselines (LogReg / SVM / LDA) |
| 2 | EEGNet (per-condition + pooled) |
| 3 | RawGazeFusionNet (EEG + ET late fusion) |
| 4 | Outcome prediction (correct vs error) |
| 5 | Vision integration (gaze features as scalar inputs) |
| 6 | No-Go EEGNet (CR vs FA, stratified k-fold) |
| 7 | SemanticGazeFusionNet (EEGNet + gaze sequence LSTM) |
| 8 | LOSO cross-validation |
| 9 | Gaze comparison in walking conditions |
| 10 | Cross-condition transfer |

## Data Layout

```
data/sj{NN}/
  eeg/   — BrainVision .vhdr/.eeg/.vmrk per condition
  beh/   — Behavioral CSVs (5 blocks x condition)
  eye/   — Pupil Labs exports (gaze, fixations, blinks, world video)
```

## Key Components

- **EEGNet** — Compact CNN (Lawhern et al. 2018) for EEG classification
- **RawGazeFusionNet** — Dual-branch EEG + ET with late fusion
- **SemanticGazeFusionNet** — EEG + gaze sequence encoder (bidirectional LSTM)
- **Vision Pipeline** — ResNet-50 scene classification of gaze-contingent video crops
- **GEDAI** — Leadfield-aware artifact removal (default over ICA)

## Configuration

- `src/run_config.yaml` — Pipeline settings (subjects, conditions, sampling rate, artifact method)
- `configs/config.yaml` — Model architecture and training hyperparameters

Update `data.root` and `data.world_video_dir` in `src/run_config.yaml` when moving machines.

## Requirements

- Python 3.11
- MNE-Python, PyTorch
- CUDA, Apple Silicon (MPS), or CPU
