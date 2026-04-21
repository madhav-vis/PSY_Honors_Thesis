"""
Main pipeline orchestrator.

Dependencies:
  01 eeg_preprocess  — independent
  02 et_preprocess   — independent
  03 fuse_eeg_et     — requires 01 + 02
  04 extract_features — requires 03
  05 dl_prep         — requires 04
  06 sanity_checks   — standalone audit (requires 04)
"""

import argparse
import io
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TeeStream:
    """Write to both a real stream and a StringIO buffer."""

    def __init__(self, real_stream):
        self.real = real_stream
        self.buf = io.StringIO()

    def write(self, data):
        self.real.write(data)
        self.buf.write(data)

    def flush(self):
        self.real.flush()

    def getvalue(self):
        return self.buf.getvalue()


def run_step(name, module_run):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    module_run()


def run_all():
    import eeg_preprocess
    import et_preprocess
    import fuse_eeg_et
    import extract_features
    import dl_prep
    import sanity_checks

    run_step("01 — EEG Preprocessing", eeg_preprocess.run)
    run_step("02 — ET Preprocessing", et_preprocess.run)
    run_step("03 — EEG + ET Fusion", fuse_eeg_et.run)
    run_step("04 — Feature Extraction", extract_features.run)
    run_step("05 — DL Preparation", dl_prep.run)
    run_step("06 — Sanity Checks", sanity_checks.run)

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")


def save_log(tee_stdout, tee_stderr, run_dir):
    """Write captured stdout+stderr to a timestamped log in the run directory."""
    os.makedirs(run_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(run_dir, f"pipeline_log_{timestamp}.txt")
    with open(log_path, "w") as f:
        f.write(f"Pipeline run: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n\n")
        f.write(tee_stdout.getvalue())
        stderr_text = tee_stderr.getvalue()
        if stderr_text.strip():
            f.write(f"\n{'='*60}\n")
            f.write("STDERR:\n")
            f.write(stderr_text)
    print(f"\nLog saved: {log_path}")


def main():
    steps = {
        "eeg": ("01 — EEG Preprocessing", "eeg_preprocess"),
        "et": ("02 — ET Preprocessing", "et_preprocess"),
        "fuse": ("03 — EEG + ET Fusion", "fuse_eeg_et"),
        "features": ("04 — Feature Extraction", "extract_features"),
        "dl": ("05 — DL Preparation", "dl_prep"),
        "checks": ("06 — Sanity Checks", "sanity_checks"),
    }

    parser = argparse.ArgumentParser(description="Run the EEG+ET processing pipeline.")
    parser.add_argument(
        "steps",
        nargs="*",
        default=["all"],
        help=f"Steps to run: {', '.join(steps.keys())}, or 'all' (default: all)",
    )
    args = parser.parse_args()

    # Tee stdout/stderr so output goes to terminal AND gets saved
    tee_stdout = TeeStream(sys.stdout)
    tee_stderr = TeeStream(sys.stderr)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    try:
        from config import (
            APPLY_ICA, DETECT_BAD_CHANNELS, RUN_DIR, RUN_DATE, RUN_NAME,
            SFREQ_TARGET, FILTER_LOW, FILTER_HIGH, TARGET_CHANNELS,
            USE_GEDAI,
        )
        print(f"\n{'='*60}")
        print(f"  RUN CONFIG")
        print(f"{'='*60}")
        print(f"  Run dir:              {RUN_DIR}")
        print(f"  Date:                 {RUN_DATE}")
        print(f"  Name:                 {RUN_NAME}")
        print(f"  apply_ica:            {APPLY_ICA}")
        print(f"  use_gedai:            {USE_GEDAI}")
        print(f"  detect_bad_channels:  {DETECT_BAD_CHANNELS}")
        print(f"  sfreq:                {SFREQ_TARGET} Hz")
        print(f"  filter:               {FILTER_LOW}–{FILTER_HIGH} Hz")
        print(f"  target_channels:      {TARGET_CHANNELS}")
        print(f"  steps:                {args.steps}")
        print(f"{'='*60}")

        if "all" in args.steps:
            run_all()
        else:
            for step_key in args.steps:
                if step_key not in steps:
                    print(f"Unknown step: {step_key}. "
                          f"Choose from: {', '.join(steps.keys())}")
                    sys.exit(1)
                name, module_name = steps[step_key]
                module = __import__(module_name)
                run_step(name, module.run)

            print(f"\n{'='*60}")
            print("  SELECTED STEPS COMPLETE")
            print(f"{'='*60}")
    finally:
        sys.stdout = tee_stdout.real
        sys.stderr = tee_stderr.real

        # Resolve run dir from config (imported lazily to avoid side effects)
        try:
            from config import RUN_DIR
            save_log(tee_stdout, tee_stderr, RUN_DIR)
        except Exception as e:
            print(f"Warning: could not save log — {e}")


if __name__ == "__main__":
    main()