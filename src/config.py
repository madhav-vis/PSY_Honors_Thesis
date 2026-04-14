import os
import shutil
from datetime import datetime

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_config.yaml")

with open(CONFIG_PATH, "r") as f:
    _raw_yaml = f.read()
CFG = yaml.safe_load(_raw_yaml)

# Run identity
RUN_NAME = CFG.get("run_name", "default_run")
RUN_DATE = CFG.get("date", "auto")
if RUN_DATE == "auto":
    RUN_DATE = datetime.now().strftime("%Y-%m-%d_%H%M")

# Output directory: runs/<date>_<run_name>/
RUNS_ROOT = os.path.join(PROJECT_ROOT, "runs")
RUN_DIR = os.path.join(RUNS_ROOT, f"{RUN_DATE}_{RUN_NAME}")
OUTPUT_DATA_DIR = os.path.join(RUN_DIR, "data")
OUTPUT_PLOT_DIR = os.path.join(RUN_DIR, "plots")

for d in [OUTPUT_DATA_DIR, OUTPUT_PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# Snapshot the config into the run directory for reproducibility
_config_snapshot = os.path.join(RUN_DIR, "run_config_snapshot.yaml")
with open(_config_snapshot, "w") as f:
    f.write(f"# Snapshot taken at {datetime.now().isoformat()}\n")
    f.write(_raw_yaml)

# Data
DATA_DIR = os.path.join(PROJECT_ROOT, CFG["data"]["root"])
SUBJECTS = CFG["data"]["subjects"]

# Conditions
CONDITIONS = CFG["conditions"]

ET_FOLDER_MAP = CFG["et"]["folder_map"]

# EEG settings
_eeg = CFG["eeg"]
SFREQ_TARGET = _eeg["sfreq_target"]
FILTER_LOW = _eeg["filter_low"]
FILTER_HIGH = _eeg["filter_high"]
REF_CHANNELS = _eeg["reference_channels"]
BAD_CHAN_Z_THRESH = _eeg["bad_channel_z_thresh"]
DETECT_BAD_CHANNELS = _eeg.get("detect_bad_channels", True)
APPLY_ICA = _eeg.get("apply_ica", True)
TRIGGER_LATENCY_OFFSET = _eeg["trigger_latency_offset"]
TMIN = _eeg["tmin"]
TMAX = _eeg["tmax"]
BASELINE = tuple(_eeg["baseline"])

# ERP / analysis
_erp = CFG["erp"]
TARGET_CHANNELS = _erp["target_channels"]
ERP_CODES = set(_erp["codes"])

# Feature extraction
_feat = CFG["features"]
P300_WINDOW = tuple(_feat["p300_window"])
