"""Progress file protocol for subprocess → Streamlit communication.

Pipeline subprocesses write JSON progress updates; the Streamlit dashboard
polls and renders them as progress bars.
"""

import json
import os
import time


def progress_file_path(run_dir):
    return os.path.join(run_dir, ".pipeline_progress.json")


def write_progress(run_dir, step, phase, current, total, message=""):
    path = progress_file_path(run_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "step": step,
        "phase": phase,
        "current": current,
        "total": total,
        "message": message,
        "timestamp": time.time(),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def read_progress(run_dir):
    path = progress_file_path(run_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def clear_progress(run_dir):
    path = progress_file_path(run_dir)
    if os.path.exists(path):
        os.unlink(path)
