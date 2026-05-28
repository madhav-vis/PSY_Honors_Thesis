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
# Override with PSY197B_RUNS_DIR to keep large .fif outputs off OneDrive.
RUNS_ROOT = os.environ.get("PSY197B_RUNS_DIR") or os.path.join(PROJECT_ROOT, "runs")
RUNS_ROOT = os.path.abspath(RUNS_ROOT)
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

# Data — env var > YAML value; YAML can be absolute or relative to PROJECT_ROOT
_data_root = os.environ.get("PSY197B_DATA_DIR") or CFG["data"]["root"]
DATA_DIR = _data_root if os.path.isabs(_data_root) else os.path.join(PROJECT_ROOT, _data_root)
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
ICA_EOG_THRESHOLD = float(_eeg.get("ica_eog_threshold", 2.5))
ICA_EOG_MEASURE = _eeg.get("ica_eog_measure", "zscore")
ICA_EOG_CHANNELS = _eeg.get(
    "ica_eog_channels", ["Fp1", "Fp2", "AF3", "AF4"])
ICA_EOG_L_FREQ = float(_eeg.get("ica_eog_l_freq", 1))
ICA_EOG_H_FREQ = float(_eeg.get("ica_eog_h_freq", 10))
ICA_EOG_VIRTUAL = _eeg.get("ica_eog_virtual", True)
USE_GEDAI = _eeg.get("use_gedai", False)
GEDAI_STRENGTH = _eeg.get("gedai_strength", "auto")
TRIGGER_LATENCY_OFFSET = _eeg["trigger_latency_offset"]
TMIN = _eeg["tmin"]
TMAX = _eeg["tmax"]
BASELINE = tuple(_eeg["baseline"])

# ERP / analysis
_erp = CFG["erp"]
TARGET_CHANNELS = _erp["target_channels"]
ERP_CODES = set(_erp["codes"])

ERP_COMPONENTS = {}
for _comp_name, _comp_cfg in _erp.get("components", {}).items():
    ERP_COMPONENTS[_comp_name] = {
        "channels": _comp_cfg["channels"],
        "window": tuple(_comp_cfg["window"]),
    }

# Feature extraction
_feat = CFG["features"]
P300_WINDOW = tuple(_feat["p300_window"])
